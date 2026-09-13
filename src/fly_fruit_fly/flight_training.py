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


def teacher_mix_beta(round_index: int, *, start: float = 0.8, floor: float = 0.15) -> float:
    """Anneal teacher control without dropping abruptly onto a collapsed student."""
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if not (0.0 <= floor <= start <= 1.0):
        raise ValueError("teacher mixing must satisfy 0 <= floor <= start <= 1")
    return float(max(floor, start * (0.72 ** round_index)))


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
        "validation_seeds": [30000, 30001, 30002],
        "held_out_test_seeds": list(range(70000, 70010)),
        "expert_actions_at_evaluation": False,
        "rl_steps": 0,
        "candidates": [],
        "corrective_rounds": [],
    }
    best_score = None
    best_state = None

    for round_index in range(args.rounds + 1):
        x, y = readout_dataset(policy, aggregate)
        candidates = [None, 1e-7, 1e-5]
        round_state = copy.deepcopy(policy.state_dict())
        round_best, round_score = None, None
        for ridge in candidates:
            policy.load_state_dict(round_state)
            loss = fit_readout(policy, x, y, ridge) if ridge else None
            label = f"round-{round_index:02d}-ridge-{ridge}"
            metrics = evaluate(
                policy,
                episodes=3,
                seed=30000,
                output=args.output / "validation" / label,
                video=False,
                asset_cache=args.cache,
            )
            score = validation_score(metrics)
            candidate = {
                "name": label,
                "ridge": ridge,
                "training_action_mse": loss,
                "validation_score": score,
            }
            manifest["candidates"].append(candidate)
            print(json.dumps(candidate), flush=True)
            if round_score is None or score > round_score:
                round_score, round_best = score, copy.deepcopy(policy.state_dict())
            if best_score is None or score > best_score:
                best_score, best_state = score, copy.deepcopy(policy.state_dict())
                manifest["selected_candidate"] = label
                save_cape_checkpoint(policy, graph, args.output / "student.pt", manifest)
            (args.output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if best_score[0] == 3 or round_index == args.rounds:
            break

        policy.load_state_dict(round_best)
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
        # The previous trainer only refit its motor readout after DAgger. CAPE also
        # updates perception, recurrent dynamics and source gains on aggregated
        # expert labels while keeping observation normalization fixed.
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
    metrics = evaluate(
        policy,
        episodes=10,
        seed=70000,
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
