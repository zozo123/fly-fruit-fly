"""Train and evaluate SuperFly, a connectome-structured flight controller.

Learning is staged:
1. collect trajectories from the released Flybody expert with the official wing pattern;
2. distill its native 12-D actions into a BANC-structured recurrent student;
3. fine-tune the student in Flybody with a clipped actor-critic objective;
4. compare measured BANC wiring with a degree-preserving shuffled control.

The BANC graph constrains recurrent signal flow. Sensory mapping, neural dynamics,
and motor/value readouts are learned and are not asserted to be biological.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .connectome import (
    ConnectomeGraph,
    degree_preserving_shuffle,
    load_graph,
    materialize_banc_subgraph,
    save_graph,
)


class SuperFlyPolicy(nn.Module):
    """Sparse recurrent policy whose fixed recurrent graph follows BANC wiring."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        graph.validate()
        self.n_nodes = graph.n_nodes
        self.node_ids = graph.node_ids.copy()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(len(action_low))
        if self.action_dim != 12:
            raise ValueError("SuperFly targets Flybody's full 12-D action space")

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        if action_low.shape != (self.action_dim,) or action_high.shape != (self.action_dim,):
            raise ValueError("invalid action bounds")
        if not np.all(action_high > action_low):
            raise ValueError("every action dimension must have a non-zero range")

        indices = torch.tensor(np.vstack([graph.edge_dst, graph.edge_src]), dtype=torch.long)
        values = torch.tensor(graph.edge_weight, dtype=torch.float32)
        adjacency = torch.sparse_coo_tensor(
            indices, values, size=(self.n_nodes, self.n_nodes)
        ).coalesce()
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self.register_buffer("action_low", torch.tensor(action_low))
        self.register_buffer("action_high", torch.tensor(action_high))

        self.sensory = nn.Linear(self.obs_dim, self.n_nodes)
        self.node_bias = nn.Parameter(torch.zeros(self.n_nodes))
        self.leak_logits = nn.Parameter(torch.full((self.n_nodes,), -1.0))
        self.recurrent_gain = nn.Parameter(torch.tensor(0.0))
        self.motor = nn.Linear(self.n_nodes, self.action_dim)
        self.value = nn.Linear(self.n_nodes, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.5))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.sensory.weight, gain=0.35)
        nn.init.zeros_(self.sensory.bias)
        nn.init.orthogonal_(self.motor.weight, gain=0.01)
        nn.init.zeros_(self.motor.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def set_observation_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.obs_mean.device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=self.obs_std.device)
        if mean_t.shape != self.obs_mean.shape or std_t.shape != self.obs_std.shape:
            raise ValueError("observation normalization shape mismatch")
        self.obs_mean.copy_(mean_t)
        self.obs_std.copy_(torch.clamp(std_t, min=1e-4))

    def initial_state(self, batch_size: int = 1, *, device=None) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.n_nodes),
            dtype=torch.float32,
            device=device if device is not None else self.obs_mean.device,
        )

    def step(self, obs: torch.Tensor, state: torch.Tensor):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        normalized = torch.clamp((obs - self.obs_mean) / self.obs_std, -10.0, 10.0)
        sensory_drive = self.sensory(normalized)
        recurrent_drive = torch.sparse.mm(self.adjacency, state.T).T
        recurrent_gain = torch.sigmoid(self.recurrent_gain) * 1.8
        candidate = torch.tanh(
            sensory_drive + self.node_bias + recurrent_gain * recurrent_drive
        )
        leak = torch.sigmoid(self.leak_logits)
        next_state = (1.0 - leak) * state + leak * candidate
        mean_raw = self.motor(next_state)
        value = self.value(next_state).squeeze(-1)
        return mean_raw, value, next_state

    def distribution(self, obs: torch.Tensor, state: torch.Tensor):
        mean_raw, value, next_state = self.step(obs, state)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).expand_as(mean_raw)
        return Normal(mean_raw, std), value, next_state

    def action_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        normalized = torch.tanh(raw)
        midpoint = (self.action_high + self.action_low) * 0.5
        half_range = (self.action_high - self.action_low) * 0.5
        return midpoint + half_range * normalized

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        midpoint = ((self.action_high + self.action_low) * 0.5).detach().cpu().numpy()
        half_range = ((self.action_high - self.action_low) * 0.5).detach().cpu().numpy()
        return np.clip((np.asarray(action) - midpoint) / half_range, -0.999, 0.999)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, state: torch.Tensor | None = None):
        if state is None:
            state = self.initial_state()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.obs_mean.device)
        mean_raw, _, next_state = self.step(obs_t, state)
        action = self.action_from_raw(mean_raw)[0].cpu().numpy().astype(np.float32)
        return action, next_state


def canonical_to_native(canonical: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Match Flybody's upstream CanonicalSpecWrapper mapping into actuator bounds."""
    canonical = np.asarray(canonical, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if canonical.shape != low.shape or low.shape != high.shape:
        raise ValueError("canonical action and actuator bounds must share a shape")
    return (low + 0.5 * (np.clip(canonical, -1.0, 1.0) + 1.0) * (high - low)).astype(
        np.float32
    )


def _official_wing_pattern(cache_dir: Path) -> Path:
    from .expert import ensure_official_wing_pattern

    return ensure_official_wing_pattern(cache_dir)


def _make_env(*, seed: int, asset_cache: Path, render_mode=None):
    from .env import FlightEnv

    wing_pattern = _official_wing_pattern(asset_cache)
    return FlightEnv(
        seed=seed,
        render_mode=render_mode,
        control_mode="full",
        action_repeat=1,
        wpg_pattern_path=wing_pattern,
    ), wing_pattern


def collect_teacher_rollouts(episodes: int, seed: int, expert_cache: Path):
    """Collect expert state/action pairs in the same native action space as SuperFly."""
    from .expert import OfficialFlightPolicy

    env, wing_pattern = _make_env(seed=seed, asset_cache=expert_cache)
    teacher = OfficialFlightPolicy.from_cache(expert_cache)
    rollouts = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            observations, actions, rewards = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                canonical = teacher.predict(env.raw_observation)
                native = canonical_to_native(
                    canonical, env.action_space.low, env.action_space.high
                )
                observations.append(obs.copy())
                actions.append(native.copy())
                obs, reward, terminated, truncated, _ = env.step(native)
                rewards.append(float(reward))
            rollouts.append({
                "seed": seed + episode,
                "observations": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float32),
            })
    finally:
        env.close()
    return rollouts, wing_pattern


def save_teacher_rollouts(rollouts, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    manifest = []
    for i, rollout in enumerate(rollouts):
        arrays[f"obs_{i}"] = rollout["observations"]
        arrays[f"actions_{i}"] = rollout["actions"]
        arrays[f"rewards_{i}"] = rollout["rewards"]
        manifest.append({
            "index": i,
            "seed": rollout["seed"],
            "steps": len(rollout["actions"]),
            "return": float(np.sum(rollout["rewards"])),
        })
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(output, **arrays)


def _observation_stats(rollouts):
    observations = np.concatenate([r["observations"] for r in rollouts], axis=0)
    return observations.mean(axis=0), observations.std(axis=0) + 1e-4


def distill(
    policy: SuperFlyPolicy,
    rollouts,
    *,
    epochs: int = 4,
    learning_rate: float = 3e-4,
    bptt_steps: int = 64,
    reset_normalization: bool = True,
) -> list[float]:
    """Sequentially imitate the released expert with truncated BPTT."""
    if reset_normalization:
        mean, std = _observation_stats(rollouts)
        policy.set_observation_normalization(mean, std)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    losses = []
    policy.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for rollout in rollouts:
            obs = torch.as_tensor(rollout["observations"], dtype=torch.float32)
            target = np.stack([policy.normalize_action(a) for a in rollout["actions"]])
            target = torch.as_tensor(target, dtype=torch.float32)
            state = policy.initial_state()
            for start in range(0, len(obs), bptt_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.tensor(0.0)
                end = min(start + bptt_steps, len(obs))
                for t in range(start, end):
                    mean_raw, _, state = policy.step(obs[t], state)
                    loss = loss + torch.mean((torch.tanh(mean_raw[0]) - target[t]) ** 2)
                loss = loss / (end - start)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                state = state.detach()
                epoch_loss += float(loss.detach()) * (end - start)
                epoch_steps += end - start
        losses.append(epoch_loss / max(epoch_steps, 1))
    return losses


def _compute_gae(rewards, values, dones, next_value, gamma=0.999, gae_lambda=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_v = float(next_value)
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_v = float(values[t + 1])
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last = delta + gamma * gae_lambda * next_nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def _bootstrap_reward(reward, *, terminated, truncated, next_value, gamma):
    """Bootstrap time limits while keeping GAE from crossing episode resets."""
    return float(reward) + (gamma * float(next_value) if truncated and not terminated else 0.0)


def ppo_finetune(
    policy: SuperFlyPolicy,
    *,
    total_steps: int,
    seed: int,
    asset_cache: Path,
    rollout_steps: int = 512,
    epochs: int = 4,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    clip_ratio: float = 0.2,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
) -> list[dict]:
    """Fine-tune SuperFly with a PPO-style clipped recurrent actor-critic update."""
    if total_steps <= 0:
        return []
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, _ = _make_env(seed=seed, asset_cache=asset_cache)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    obs, _ = env.reset(seed=seed)
    state = policy.initial_state()
    history = []
    steps_done = 0
    policy.train()
    try:
        while steps_done < total_steps:
            n = min(rollout_steps, total_steps - steps_done)
            obs_buf, state_buf, raw_buf, logp_buf = [], [], [], []
            reward_buf, done_buf, value_buf = [], [], []
            for _ in range(n):
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                state_before = state.detach().clone()
                with torch.no_grad():
                    distribution, value, next_state = policy.distribution(obs_t, state_before)
                    raw = distribution.sample()
                    logp = distribution.log_prob(raw).sum(-1)
                    action = policy.action_from_raw(raw)[0].cpu().numpy()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                if truncated and not terminated:
                    with torch.no_grad():
                        _, terminal_value, _ = policy.step(
                            torch.as_tensor(next_obs, dtype=torch.float32), next_state
                        )
                    reward = _bootstrap_reward(
                        reward, terminated=terminated, truncated=truncated,
                        next_value=float(terminal_value[0]), gamma=gamma,
                    )
                done = bool(terminated or truncated)
                obs_buf.append(obs.copy())
                state_buf.append(state_before[0].cpu().numpy())
                raw_buf.append(raw[0].cpu().numpy())
                logp_buf.append(float(logp[0]))
                reward_buf.append(float(reward))
                done_buf.append(done)
                value_buf.append(float(value[0]))
                if done:
                    next_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    next_state = policy.initial_state()
                obs, state = next_obs, next_state.detach()

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, next_value_t, _ = policy.step(obs_t, state)
                next_value = float(next_value_t[0])
            rewards = np.asarray(reward_buf, dtype=np.float32)
            values = np.asarray(value_buf, dtype=np.float32)
            dones = np.asarray(done_buf, dtype=bool)
            advantages, returns = _compute_gae(
                rewards, values, dones, next_value, gamma=gamma, gae_lambda=gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_tensor = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32)
            state_tensor = torch.as_tensor(np.asarray(state_buf), dtype=torch.float32)
            raw_tensor = torch.as_tensor(np.asarray(raw_buf), dtype=torch.float32)
            old_logp = torch.as_tensor(logp_buf, dtype=torch.float32)
            adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)
            ret_tensor = torch.as_tensor(returns, dtype=torch.float32)
            losses = []
            for _ in range(epochs):
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
                    distribution, new_value, _ = policy.distribution(obs_tensor[idx], state_tensor[idx])
                    new_logp = distribution.log_prob(raw_tensor[idx]).sum(-1)
                    ratio = torch.exp(new_logp - old_logp[idx])
                    unclipped = ratio * adv_tensor[idx]
                    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_tensor[idx]
                    actor_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * torch.mean((new_value - ret_tensor[idx]) ** 2)
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = actor_loss + 0.5 * value_loss - 0.001 * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            steps_done += n
            history.append({
                "steps": steps_done,
                "mean_reward_per_step": float(rewards.mean()),
                "episodes_ended": int(dones.sum()),
                "loss": float(np.mean(losses)),
            })
    finally:
        env.close()
    return history


def evaluate(
    policy: SuperFlyPolicy,
    *,
    episodes: int,
    seed: int,
    output: Path,
    video: bool,
    asset_cache: Path,
):
    import imageio.v2 as imageio
    from .env import video_capture_stride

    output.mkdir(parents=True, exist_ok=True)
    env, wing_pattern = _make_env(
        seed=seed, asset_cache=asset_cache, render_mode="rgb_array" if video else None
    )
    writer = None
    capture_stride = realized_slowdown = None
    rows = []
    episodes_out = []
    try:
        if video:
            capture_stride, realized_slowdown = video_capture_stride(
                env.agent_dt, fps=env.metadata["render_fps"], slowdown=10
            )
            writer = imageio.get_writer(output / "superfly.mp4", fps=env.metadata["render_fps"])
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            state = policy.initial_state()
            total_reward = 0.0
            error_sum = 0.0
            control_steps = 0
            policy_steps = 0
            terminated = truncated = False
            if writer and episode == 0:
                writer.append_data(env.render())
            while not (terminated or truncated):
                action, state = policy.predict(obs, state)
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("non-finite simulator output")
                total_reward += reward
                error_sum += info["tracking_error_sum_cm"]
                control_steps += info["inner_control_steps"]
                policy_steps += 1
                rows.append({
                    "seed": seed + episode,
                    "step": policy_steps,
                    "time_s": info["time_s"],
                    "tracking_error_cm": info["tracking_error_cm"],
                    "height_cm": info["height_cm"],
                    "reward": reward,
                })
                if writer and episode == 0 and (
                    policy_steps % capture_stride == 0 or terminated or truncated
                ):
                    writer.append_data(env.render())
            episodes_out.append({
                "seed": seed + episode,
                "steps": policy_steps,
                "control_steps": control_steps,
                "return": total_reward,
                "duration_s": info["time_s"],
                "mean_tracking_error_cm": error_sum / max(control_steps, 1),
                "final_height_cm": info["height_cm"],
                "failed": bool(terminated),
                "completed_reference": bool(truncated),
            })
    finally:
        if writer:
            writer.close()
        env.close()

    with (output / "trajectory.csv").open("w", newline="") as handle:
        fieldnames = ["seed", "step", "time_s", "tracking_error_cm", "height_cm", "reward"]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    wing_manifest = json.loads(wing_pattern.with_suffix(".json").read_text())
    metrics = {
        "schema_version": 2,
        "controller": "superfly_connectome_student",
        "control_mode": "full",
        "action_repeat": 1,
        "policy_interval_s": env.agent_dt,
        "task": {
            "name": "flybody_straight_flight",
            "speed_cm_s": 20,
            "initial_height_cm": 1,
            "reference_duration_s": 0.6,
            "wing_pattern": "official_fmech",
        },
        "wing_pattern_provenance": wing_manifest,
        "video_slowdown": realized_slowdown if video else None,
        "video_capture_stride": capture_stride if video else None,
        "episodes": episodes_out,
        "completion_rate": float(np.mean([e["completed_reference"] for e in episodes_out])),
        "mean_return": float(np.mean([e["return"] for e in episodes_out])),
        "note": (
            "Connectome-structured learned controller. BANC supplies measured wiring; "
            "sensory mapping, neural dynamics and motor readout are learned engineering components."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def set_motor_hidden(policy: SuperFlyPolicy, hidden: int) -> None:
    """Add nonlinear motor decoding while retaining the connectome recurrent core."""
    if hidden <= 0:
        raise ValueError("Motor hidden width must be positive")
    policy.motor = nn.Sequential(nn.Linear(policy.n_nodes, hidden), nn.Tanh(),
                                 nn.Linear(hidden, policy.action_dim))
    policy.motor_hidden = hidden


def save_checkpoint(policy: SuperFlyPolicy, graph: ConnectomeGraph, output: Path, training: dict):
    graph.validate()
    _check_graph_adjacency(policy.adjacency, graph)
    if not np.array_equal(policy.node_ids, graph.node_ids):
        raise ValueError("Policy neuron IDs do not match the supplied connectome graph")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "graph_node_ids": torch.as_tensor(graph.node_ids.copy(), dtype=torch.int64),
        "motor_hidden": getattr(policy, "motor_hidden", 0),
        "state_dict": policy.state_dict(),
        "obs_dim": policy.obs_dim,
        "action_low": policy.action_low.cpu(),
        "action_high": policy.action_high.cpu(),
        "graph_metadata": graph.metadata,
        "training": training,
    }, output)


def _check_graph_adjacency(adjacency: torch.Tensor, graph: ConnectomeGraph) -> None:
    """Compare canonical sparse wiring before state_dict can overwrite it."""
    if not adjacency.is_sparse:
        raise ValueError("Checkpoint adjacency must be sparse")
    actual = adjacency.detach().cpu().coalesce()
    expected = torch.sparse_coo_tensor(
        torch.as_tensor(np.vstack([graph.edge_dst, graph.edge_src]), dtype=torch.long),
        torch.as_tensor(graph.edge_weight, dtype=torch.float32),
        size=(graph.n_nodes, graph.n_nodes),
    ).coalesce()
    if (actual.shape != expected.shape or
            not torch.equal(actual.indices(), expected.indices()) or
            not torch.equal(actual.values(), expected.values())):
        raise ValueError("Checkpoint wiring does not match the supplied connectome graph")


def load_checkpoint(checkpoint: Path, graph: ConnectomeGraph) -> SuperFlyPolicy:
    graph.validate()
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _check_graph_adjacency(data["state_dict"]["adjacency"], graph)
    if data.get("graph_metadata") != graph.metadata:
        raise ValueError("Checkpoint provenance does not match the supplied connectome graph")
    # Older checkpoints did not record IDs; their wiring and provenance are still checked.
    if "graph_node_ids" in data and not torch.equal(
        data["graph_node_ids"], torch.as_tensor(graph.node_ids, dtype=torch.int64)
    ):
        raise ValueError("Checkpoint neuron IDs do not match the supplied connectome graph")
    policy = SuperFlyPolicy(
        graph,
        int(data["obs_dim"]),
        np.asarray(data["action_low"], dtype=np.float32),
        np.asarray(data["action_high"], dtype=np.float32),
    )
    if data.get("motor_hidden", 0):
        set_motor_hidden(policy, int(data["motor_hidden"]))
    policy.load_state_dict(data["state_dict"])
    policy.eval()
    return policy


def train_command(args):
    torch.manual_seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)

    graph_path = args.output / "banc-v888-superfly.npz"
    graph = materialize_banc_subgraph(
        args.cache, graph_path, n_nodes=args.nodes, min_synapses=args.min_synapses
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

    rollouts, teacher_wing_pattern = collect_teacher_rollouts(
        args.teacher_episodes, args.seed, args.expert_cache
    )
    if teacher_wing_pattern.resolve() != wing_pattern.resolve():
        raise RuntimeError("teacher and student environments resolved different wing patterns")
    save_teacher_rollouts(rollouts, args.output / "teacher-rollouts.npz")
    imitation_losses = distill(
        policy,
        rollouts,
        epochs=args.distill_epochs,
        learning_rate=args.distill_lr,
        bptt_steps=args.bptt_steps,
    )
    rl_history = ppo_finetune(
        policy,
        total_steps=args.rl_steps,
        seed=args.seed + 1000,
        asset_cache=args.expert_cache,
        learning_rate=args.rl_lr,
    )
    training = {
        "algorithm": "expert_distillation_then_recurrent_clipped_actor_critic",
        "connectome": "BANC v888 measured neuron-to-neuron wiring",
        "graph_file": graph_path.name,
        "graph_nodes": graph.n_nodes,
        "graph_edges": graph.n_edges,
        "shuffled_control": bool(args.shuffled),
        "teacher": "official_flybody_expert",
        "teacher_action_space": "canonical_mean_scaled_to_native_12d_actuator_bounds",
        "teacher_episodes": args.teacher_episodes,
        "teacher_steps": int(sum(len(r["actions"]) for r in rollouts)),
        "teacher_mean_return": float(np.mean([np.sum(r["rewards"]) for r in rollouts])),
        "wing_pattern": "official_fmech",
        "wing_pattern_sha256": json.loads(wing_pattern.with_suffix(".json").read_text())["sha256"],
        "distillation_losses": imitation_losses,
        "rl_steps": args.rl_steps,
        "rl_history": rl_history,
        "seed": args.seed,
    }
    save_checkpoint(policy, graph, args.output / "superfly.pt", training)
    (args.output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    metrics = evaluate(
        policy,
        episodes=args.eval_episodes,
        seed=args.seed + 20_000,
        output=args.output / "evaluation",
        video=args.video,
        asset_cache=args.expert_cache,
    )
    print(json.dumps({"training": training, "evaluation": metrics}, indent=2))


def build_command(args):
    graph = materialize_banc_subgraph(
        args.cache, args.output, n_nodes=args.nodes, min_synapses=args.min_synapses
    )
    print(json.dumps(graph.metadata, indent=2))
    if args.shuffled_output:
        shuffled = degree_preserving_shuffle(graph, seed=args.seed)
        save_graph(shuffled, args.shuffled_output)


def evaluate_command(args):
    graph = load_graph(args.graph)
    policy = load_checkpoint(args.checkpoint, graph)
    metrics = evaluate(
        policy,
        episodes=args.episodes,
        seed=args.seed,
        output=args.output,
        video=args.video,
        asset_cache=args.expert_cache,
    )
    print(json.dumps(metrics, indent=2))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-connectome", help="materialize a reproducible BANC v888 subgraph")
    build.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    build.add_argument("--nodes", type=positive, default=512)
    build.add_argument("--min-synapses", type=positive, default=5)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--output", type=Path, default=Path("runs/connectome/banc-v888.npz"))
    build.add_argument("--shuffled-output", type=Path)
    build.set_defaults(func=build_command)

    train = sub.add_parser("train", help="distill the expert then reinforcement-fine-tune SuperFly")
    train.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    train.add_argument("--expert-cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    train.add_argument("--nodes", type=positive, default=512)
    train.add_argument("--min-synapses", type=positive, default=5)
    train.add_argument("--teacher-episodes", type=positive, default=4)
    train.add_argument("--distill-epochs", type=positive, default=6)
    train.add_argument("--distill-lr", type=float, default=3e-4)
    train.add_argument("--bptt-steps", type=positive, default=64)
    train.add_argument("--rl-steps", type=nonnegative, default=16_384)
    train.add_argument("--rl-lr", type=float, default=1e-4)
    train.add_argument("--eval-episodes", type=positive, default=10)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--shuffled", action="store_true", help="train the degree-preserving rewired control")
    train.add_argument("--video", action="store_true")
    train.add_argument("--output", type=Path, default=Path("runs/superfly"))
    train.set_defaults(func=train_command)

    evaluation = sub.add_parser("evaluate", help="evaluate a saved SuperFly checkpoint")
    evaluation.add_argument("--graph", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--expert-cache", type=Path, default=Path.home() / ".cache" / "fly-fruit-fly")
    evaluation.add_argument("--episodes", type=positive, default=10)
    evaluation.add_argument("--seed", type=int, default=20_000)
    evaluation.add_argument("--video", action="store_true")
    evaluation.add_argument("--output", type=Path, default=Path("runs/superfly-eval"))
    evaluation.set_defaults(func=evaluate_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
