"""Corrective imitation curriculum for SuperFly.

Plain behavior cloning only sees expert states. A slightly imperfect student then
visits new states, where its errors compound. This module adds DAgger-style dataset
aggregation: roll the current student through Flybody, query the released expert on
every visited state, retrain on the aggregated corrective labels, and only then
optionally apply reinforcement learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .connectome import degree_preserving_shuffle, materialize_banc_subgraph, save_graph
from .superfly import (
    SuperFlyPolicy,
    _make_env,
    canonical_to_native,
    distill,
    evaluate,
    ppo_finetune,
    save_checkpoint,
    save_teacher_rollouts,
)


def dagger_betas(rounds: int, start: float = 0.8, end: float = 0.0) -> list[float]:
    """Teacher-action mixing coefficients, annealed toward student control."""
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("DAgger beta values must be in [0, 1]")
    if rounds == 0:
        return []
    if rounds == 1:
        return [float(start)]
    return [float(x) for x in np.linspace(start, end, rounds)]


def collect_corrective_rollouts(
    policy: SuperFlyPolicy,
    *,
    episodes: int,
    seed: int,
    expert_cache: Path,
    beta: float,
):
    """Collect expert labels on states visited by a teacher/student action mixture.

    `beta=1` executes the teacher. `beta=0` executes only the student. At every
    state the stored training target remains the expert's native 12-D action.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    from .expert import OfficialFlightPolicy

    env, wing_pattern = _make_env(seed=seed, asset_cache=expert_cache)
    teacher = OfficialFlightPolicy.from_cache(expert_cache)
    rollouts = []
    policy.eval()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            state = policy.initial_state()
            observations, targets, rewards = [], [], []
            student_distances = []
            terminated = truncated = False
            while not (terminated or truncated):
                canonical = teacher.predict(env.raw_observation)
                expert_action = canonical_to_native(
                    canonical, env.action_space.low, env.action_space.high
                )
                student_action, next_state = policy.predict(obs, state)
                executed_action = (
                    beta * expert_action + (1.0 - beta) * student_action
                ).astype(np.float32)
                observations.append(obs.copy())
                targets.append(expert_action.copy())
                student_distances.append(float(np.mean(np.abs(student_action - expert_action))))
                obs, reward, terminated, truncated, _ = env.step(executed_action)
                rewards.append(float(reward))
                state = next_state
            rollouts.append({
                "seed": seed + episode,
                "observations": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(targets, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "beta": float(beta),
                "executed_steps": len(rewards),
                "executed_return": float(np.sum(rewards)),
                "mean_student_teacher_l1": float(np.mean(student_distances)),
                "completed_reference": bool(truncated),
                "failed": bool(terminated),
            })
    finally:
        env.close()
    return rollouts, wing_pattern


def _rollout_summary(rollouts) -> dict:
    return {
        "episodes": len(rollouts),
        "steps": int(sum(len(r["actions"]) for r in rollouts)),
        "mean_executed_return": float(np.mean([np.sum(r["rewards"]) for r in rollouts])),
        "completion_rate": float(np.mean([bool(r.get("completed_reference")) for r in rollouts])),
        "mean_student_teacher_l1": float(
            np.mean([r.get("mean_student_teacher_l1", 0.0) for r in rollouts])
        ),
    }


def train_curriculum(args):
    """Train one measured- or shuffled-connectome SuperFly with corrective imitation."""
    torch.manual_seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)

    graph_path = args.output / "banc-v888-superfly.npz"
    graph = materialize_banc_subgraph(
        args.cache,
        graph_path,
        n_nodes=args.nodes,
        min_synapses=args.min_synapses,
    )
    if args.shuffled:
        graph = degree_preserving_shuffle(graph, seed=args.seed)
        graph_path = args.output / "banc-v888-shuffled-control.npz"
        save_graph(graph, graph_path)

    probe, wing_pattern = _make_env(seed=args.seed, asset_cache=args.expert_cache)
    try:
        policy = SuperFlyPolicy(
            graph,
            int(np.prod(probe.observation_space.shape)),
            probe.action_space.low,
            probe.action_space.high,
        )
    finally:
        probe.close()

    # Start from on-distribution expert demonstrations.
    from .superfly import collect_teacher_rollouts

    demonstrations, teacher_wing_pattern = collect_teacher_rollouts(
        args.teacher_episodes, args.seed, args.expert_cache
    )
    if teacher_wing_pattern.resolve() != wing_pattern.resolve():
        raise RuntimeError("teacher and student environments resolved different wing patterns")
    aggregate = list(demonstrations)
    initial_losses = distill(
        policy,
        aggregate,
        epochs=args.distill_epochs,
        learning_rate=args.distill_lr,
        bptt_steps=args.bptt_steps,
    )

    # Correct covariate shift by asking the teacher what to do on student-visited states.
    rounds = []
    for round_index, beta in enumerate(
        dagger_betas(args.dagger_rounds, args.dagger_start_beta, args.dagger_end_beta)
    ):
        corrective, corrective_wing_pattern = collect_corrective_rollouts(
            policy,
            episodes=args.dagger_episodes,
            seed=args.seed + 10_000 + round_index * args.dagger_episodes,
            expert_cache=args.expert_cache,
            beta=beta,
        )
        if corrective_wing_pattern.resolve() != wing_pattern.resolve():
            raise RuntimeError("DAgger and student environments resolved different wing patterns")
        aggregate.extend(corrective)
        round_losses = distill(
            policy,
            aggregate,
            epochs=args.dagger_epochs,
            learning_rate=args.distill_lr,
            bptt_steps=args.bptt_steps,
            reset_normalization=False,
        )
        rounds.append({
            "round": round_index + 1,
            "beta": beta,
            "corrective": _rollout_summary(corrective),
            "aggregate_episodes": len(aggregate),
            "aggregate_steps": int(sum(len(r["actions"]) for r in aggregate)),
            "distillation_losses": round_losses,
        })

    save_teacher_rollouts(aggregate, args.output / "corrective-dataset.npz")

    rl_history = ppo_finetune(
        policy,
        total_steps=args.rl_steps,
        seed=args.seed + 30_000,
        asset_cache=args.expert_cache,
        learning_rate=args.rl_lr,
    )
    wing_manifest = json.loads(wing_pattern.with_suffix(".json").read_text())
    training = {
        "algorithm": "expert_distillation_dagger_then_recurrent_clipped_actor_critic",
        "connectome": "BANC v888 measured neuron-to-neuron wiring",
        "graph_file": graph_path.name,
        "graph_nodes": graph.n_nodes,
        "graph_edges": graph.n_edges,
        "shuffled_control": bool(args.shuffled),
        "teacher": "official_flybody_expert",
        "teacher_action_space": "canonical_mean_scaled_to_native_12d_actuator_bounds",
        "teacher_episodes": args.teacher_episodes,
        "teacher_steps": int(sum(len(r["actions"]) for r in demonstrations)),
        "teacher_mean_return": float(np.mean([np.sum(r["rewards"]) for r in demonstrations])),
        "initial_distillation_losses": initial_losses,
        "dagger_rounds": rounds,
        "aggregate_episodes": len(aggregate),
        "aggregate_steps": int(sum(len(r["actions"]) for r in aggregate)),
        "rl_steps": args.rl_steps,
        "rl_history": rl_history,
        "wing_pattern": "official_fmech",
        "wing_pattern_sha256": wing_manifest["sha256"],
        "seed": args.seed,
    }
    save_checkpoint(policy, graph, args.output / "superfly.pt", training)
    (args.output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    metrics = evaluate(
        policy,
        episodes=args.eval_episodes,
        seed=args.seed + 50_000,
        output=args.output / "evaluation",
        video=args.video,
        asset_cache=args.expert_cache,
    )
    print(json.dumps({"training": training, "evaluation": metrics}, indent=2))
    return training, metrics


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    parser.add_argument("--expert-cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    parser.add_argument("--nodes", type=positive, default=512)
    parser.add_argument("--min-synapses", type=positive, default=5)
    parser.add_argument("--teacher-episodes", type=positive, default=2)
    parser.add_argument("--distill-epochs", type=positive, default=6)
    parser.add_argument("--distill-lr", type=float, default=3e-4)
    parser.add_argument("--bptt-steps", type=positive, default=64)
    parser.add_argument("--dagger-rounds", type=nonnegative, default=3)
    parser.add_argument("--dagger-episodes", type=positive, default=1)
    parser.add_argument("--dagger-epochs", type=positive, default=2)
    parser.add_argument("--dagger-start-beta", type=probability, default=0.8)
    parser.add_argument("--dagger-end-beta", type=probability, default=0.0)
    parser.add_argument("--rl-steps", type=nonnegative, default=4096)
    parser.add_argument("--rl-lr", type=float, default=5e-5)
    parser.add_argument("--eval-episodes", type=positive, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/superfly-curriculum"))
    args = parser.parse_args()
    train_curriculum(args)


if __name__ == "__main__":
    main()
