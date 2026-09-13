"""Train CAPE SuperFly with autonomous validation and untouched held-out tests."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .assessment import assess
from .circuit import (
    CapeSuperFlyPolicy,
    cape_distill,
    materialize_cape_subgraph,
    save_cape_checkpoint,
)
from .curriculum import collect_corrective_rollouts, _rollout_summary
from .superfly import _make_env, collect_teacher_rollouts, evaluate


@torch.no_grad()
def readout_dataset(policy, rollouts):
    """Replay rollouts and expose only the states CAPE permits to drive motors."""
    features, targets = [], []
    for rollout in rollouts:
        state = policy.initial_state()
        for obs in rollout["observations"]:
            _, _, state = policy.step(torch.as_tensor(obs, dtype=torch.float32), state)
            motor_features = (
                policy.motor_features(state) if hasattr(policy, "motor_features") else state
            )
            features.append(motor_features[0].clone())
        native = policy.normalize_action(rollout["actions"])
        targets.append(torch.as_tensor(np.arctanh(native), dtype=torch.float64))
    if not features:
        raise ValueError("Readout fitting requires observations")
    x = torch.stack(features).double()
    return torch.cat([x, torch.ones((len(x), 1), dtype=torch.float64)], 1), torch.cat(targets)


@torch.no_grad()
def fit_readout(policy, x, y, ridge):
    """Solve mean-square pre-tanh motor prediction with L2 regularization."""
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    if x.ndim != 2 or y.shape != (len(x), policy.action_dim) or len(x) == 0:
        raise ValueError("Invalid readout dataset shapes")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Readout dataset must be finite")
    regularizer = torch.eye(x.shape[1], dtype=x.dtype) * (ridge * len(x))
    regularizer[-1, -1] = 0
    solution = torch.linalg.solve(x.T @ x + regularizer, x.T @ y)
    if not torch.isfinite(solution).all():
        raise ValueError("Non-finite readout solution")
    policy.motor.weight.copy_(solution[:-1].T.float())
    policy.motor.bias.copy_(solution[-1].float())
    return float(torch.mean((torch.tanh(x @ solution) - torch.tanh(y)) ** 2))


def validation_score(metrics):
    episodes = metrics["episodes"]
    passing = sum(
        e["completed_reference"]
        and e["duration_s"] >= 0.59
        and e["mean_tracking_error_cm"] <= 0.1
        for e in episodes
    )
    return (
        passing,
        float(np.mean([e["duration_s"] for e in episodes])),
        -float(np.mean([e["mean_tracking_error_cm"] for e in episodes])),
    )


def validation_protocol() -> dict:
    """Keep readout tuning, round promotion, and final testing on disjoint seeds."""
    protocol = {
        "ridge_selection_seeds": [30000, 30001, 30002],
        "round_promotion_seeds": [31000, 31001, 31002],
        "held_out_test_seeds": list(range(70000, 70010)),
    }
    validate_validation_protocol(protocol)
    return protocol


def _validate_seed_panel(name: str, seeds) -> list[int]:
    """Validate the exact seed panel semantics supported by evaluate(seed, episodes)."""
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError(f"{name} must be a non-empty seed sequence")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError(f"{name} must contain non-negative integer seeds")
    normalized = list(seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate seeds")
    expected = list(range(normalized[0], normalized[0] + len(normalized)))
    if normalized != expected:
        raise ValueError(
            f"{name} must be consecutive because evaluate() expands seed + episode index"
        )
    return normalized


def validate_validation_protocol(protocol: dict) -> dict:
    """Reject overlapping or ambiguous tuning/promotion/test seed definitions."""
    required = (
        "ridge_selection_seeds",
        "round_promotion_seeds",
        "held_out_test_seeds",
    )
    if not isinstance(protocol, dict) or set(protocol) != set(required):
        raise ValueError("validation protocol must define exactly tuning, promotion, and test panels")
    panels = {name: _validate_seed_panel(name, protocol[name]) for name in required}
    names = list(required)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if set(panels[first]).intersection(panels[second]):
                raise ValueError(f"validation seed panels overlap: {first} and {second}")
    return panels


def evaluate_seed_panel(policy, seeds, *, output: Path, video: bool, asset_cache: Path):
    """Evaluate exactly one validated consecutive seed panel."""
    panel = _validate_seed_panel("evaluation_seeds", seeds)
    metrics = evaluate(
        policy,
        episodes=len(panel),
        seed=panel[0],
        output=output,
        video=video,
        asset_cache=asset_cache,
    )
    observed = [episode.get("seed") for episode in metrics.get("episodes", [])]
    if observed != panel:
        raise RuntimeError(
            f"evaluation returned unexpected seeds: expected {panel}, observed {observed}"
        )
    return metrics


def teacher_mix_beta(round_index: int, *, start: float = 0.8, floor: float = 0.15) -> float:
    """Anneal teacher control without dropping abruptly onto a collapsed student."""
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if not (0.0 <= floor <= start <= 1.0):
        raise ValueError("teacher mixing must satisfy 0 <= floor <= start <= 1")
    return float(max(floor, start * (0.72 ** round_index)))


def training_rounds_complete(round_index: int, rounds: int) -> bool:
    """Stop only after the configured DAgger budget, never on a tiny validation panel."""
    if round_index < 0 or rounds < 0:
        raise ValueError("round indices must be non-negative")
    return round_index >= rounds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/flight-student"))
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    p.add_argument("--nodes", type=int, default=384)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--circuit-substeps", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if args.nodes < 32:
        p.error("nodes must be at least 32")
    if args.rounds < 0 or args.rounds > 20:
        p.error("rounds must be between 0 and 20")
    if args.circuit_substeps < 1 or args.circuit_substeps > 16:
        p.error("circuit-substeps must be between 1 and 16")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = validation_protocol()

    torch.manual_seed(args.seed)
    graph = materialize_cape_subgraph(
        args.cache,
        args.output / "graph.npz",
        n_nodes=args.nodes,
        min_synapses=5,
        role_fraction=0.55,
    )
    env, _ = _make_env(seed=args.seed, asset_cache=args.cache)
    try:
        policy = CapeSuperFlyPolicy(
            graph,
            int(np.prod(env.observation_space.shape)),
            env.action_space.low,
            env.action_space.high,
            circuit_substeps=args.circuit_substeps,
        )
    finally:
        env.close()

    demonstrations, _ = collect_teacher_rollouts(4, args.seed, args.cache)
    aggregate = list(demonstrations)
    losses = cape_distill(
        policy,
        aggregate,
        epochs=10,
        learning_rate=3e-4,
        bptt_steps=96,
        transient_steps=400,
        transient_boost=4.0,
    )
    manifest = {
        "algorithm": "cape_role_routed_transient_distillation_dagger_ridge",
        "seed": args.seed,
        "graph_nodes": graph.n_nodes,
        "graph_edges": graph.n_edges,
        "graph_kind": graph.metadata.get("kind"),
        "routing": policy.routing_summary(),
        "graph_routing": graph.metadata.get("routing"),
        "teacher_episodes": 4,
        "initial_losses": losses,
        "normalization": "fixed after initial expert demonstrations",
        "transient_weighting": {"steps": 400, "initial_boost": 4.0},
        "teacher_mix_schedule": "max(0.15, 0.8 * 0.72**round)",
        "training_round_policy": "run every configured DAgger round; promotion selects checkpoints only",
        "validation_protocol": protocol,
        "expert_actions_at_evaluation": False,
        "rl_steps": 0,
        "candidates": [],
        "promotions": [],
        "corrective_rounds": [],
    }
    best_score = None
    best_state = None

    for round_index in range(args.rounds + 1):
        x, y = readout_dataset(policy, aggregate)
        candidates = [None, 1e-7, 1e-5]
        round_state = copy.deepcopy(policy.state_dict())
        round_best, round_score, round_label = None, None, None
        for ridge in candidates:
            policy.load_state_dict(round_state)
            loss = fit_readout(policy, x, y, ridge) if ridge else None
            label = f"round-{round_index:02d}-ridge-{ridge}"
            metrics = evaluate_seed_panel(
                policy,
                protocol["ridge_selection_seeds"],
                output=args.output / "validation" / "ridge-selection" / label,
                video=False,
                asset_cache=args.cache,
            )
            score = validation_score(metrics)
            candidate = {
                "name": label,
                "ridge": ridge,
                "training_action_mse": loss,
                "selection_score": score,
            }
            manifest["candidates"].append(candidate)
            print(json.dumps(candidate), flush=True)
            if round_score is None or score > round_score:
                round_score = score
                round_best = copy.deepcopy(policy.state_dict())
                round_label = label

        policy.load_state_dict(round_best)
        promotion_metrics = evaluate_seed_panel(
            policy,
            protocol["round_promotion_seeds"],
            output=args.output / "validation" / "round-promotion" / f"round-{round_index:02d}",
            video=False,
            asset_cache=args.cache,
        )
        promotion_score = validation_score(promotion_metrics)
        promotion = {
            "round": round_index,
            "candidate": round_label,
            "promotion_score": promotion_score,
        }
        manifest["promotions"].append(promotion)
        print(json.dumps(promotion), flush=True)
        if best_score is None or promotion_score > best_score:
            best_score = promotion_score
            best_state = copy.deepcopy(policy.state_dict())
            manifest["selected_candidate"] = round_label
            manifest["selected_round"] = round_index
            save_cape_checkpoint(policy, graph, args.output / "student.pt", manifest)
        (args.output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if training_rounds_complete(round_index, args.rounds):
            break

        beta = teacher_mix_beta(round_index)
        corrective, _ = collect_corrective_rollouts(
            policy,
            episodes=2,
            seed=args.seed + 10_000 + round_index * 2,
            expert_cache=args.cache,
            beta=beta,
        )
        aggregate.extend(corrective)
        corrective_summary = {
            "round": round_index,
            "beta": beta,
            **_rollout_summary(corrective),
        }
        corrective_summary["recurrent_distillation_losses"] = cape_distill(
            policy,
            aggregate,
            epochs=1,
            learning_rate=1e-4,
            bptt_steps=96,
            reset_normalization=False,
            transient_steps=400,
            transient_boost=4.0,
        )
        manifest["corrective_rounds"].append(corrective_summary)
        print(json.dumps(corrective_summary), flush=True)
        (args.output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")

    policy.load_state_dict(best_state)
    save_cape_checkpoint(policy, graph, args.output / "student.pt", manifest)
    metrics = evaluate_seed_panel(
        policy,
        protocol["held_out_test_seeds"],
        output=args.output / "evaluation",
        video=True,
        asset_cache=args.cache,
    )
    result = assess(metrics)
    (args.output / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"held_out_assessment": result}), flush=True)


if __name__ == "__main__":
    main()
