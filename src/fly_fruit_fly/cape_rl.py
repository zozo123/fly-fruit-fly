"""Conservative PPO refinement for CAPE without changing the baseline trainer.

The default SuperFly PPO loop is intentionally generic. CAPE reaches PPO only after
expert distillation and adaptive DAgger, so destroying that imitation solution with
large exploratory updates is especially costly. This module provides an experimental
refinement lane that anchors every update to the pre-RL policy and starts from small
raw-action exploration. It can be injected into ``flight_training`` without changing
its checkpoint promotion or held-out evaluation logic.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.distributions import kl_divergence

from . import flight_training
from .superfly import _bootstrap_reward, _compute_gae, _make_env


def _validated_positive(name: str, value: float) -> float:
    """Return one finite positive scalar or raise a configuration error."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def set_exploration_std(policy, exploration_std: float) -> float:
    """Set a deliberately small isotropic raw-action standard deviation."""
    std = _validated_positive("exploration_std", exploration_std)
    log_std = math.log(std)
    if log_std < -5.0 or log_std > 1.0:
        raise ValueError("exploration_std must map inside SuperFly's log-std clamp")
    with torch.no_grad():
        policy.log_std.fill_(log_std)
    return std


def _guarded_optimizer_step(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    kl_fn: Callable[[], torch.Tensor],
    max_kl: float,
) -> tuple[bool, float]:
    """Apply one update transactionally and reject any step outside the KL trust region.

    The current policy is checked before mutating it. Parameters/buffers and optimizer
    state are then snapshotted, the proposed gradient step is applied, and KL is
    recomputed from the updated policy. If the candidate crosses ``max_kl`` both the
    model and optimizer are restored exactly, so an early-stop decision cannot leave
    a policy that already violated the anchor constraint.
    """
    max_kl = _validated_positive("max_kl", max_kl)
    with torch.no_grad():
        current_kl = float(kl_fn().detach())
    if not math.isfinite(current_kl):
        raise ValueError("anchor KL must remain finite")
    if current_kl > max_kl:
        return False, current_kl

    module_state = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        post_kl = float(kl_fn().detach())
    if not math.isfinite(post_kl) or post_kl > max_kl:
        module.load_state_dict(module_state)
        optimizer.load_state_dict(optimizer_state)
        return False, post_kl
    return True, post_kl


def anchored_ppo_finetune(
    policy,
    *,
    total_steps: int,
    seed: int,
    asset_cache: Path,
    rollout_steps: int = 512,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 5e-5,
    clip_ratio: float = 0.1,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    exploration_std: float = 0.05,
    kl_coef: float = 2.0,
    target_kl: float = 0.02,
) -> list[dict]:
    """Refine CAPE with PPO while anchoring every accepted step to imitation.

    The immutable ``anchor`` is a deep copy of the incoming imitation/DAgger policy.
    PPO learns from environment reward, but each minibatch pays an analytic Gaussian
    KL penalty against that fixed controller. A candidate optimizer step is accepted
    only if the *post-update* mean KL remains at or below ``2 * target_kl``; otherwise
    model and Adam state are restored. The outer trainer still performs independent
    promotion-panel rollback, so this is an additional safety layer rather than a
    replacement for autonomous validation.
    """
    if total_steps <= 0:
        return []
    if rollout_steps <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("rollout_steps, epochs, and batch_size must be positive")
    learning_rate = _validated_positive("learning_rate", learning_rate)
    clip_ratio = _validated_positive("clip_ratio", clip_ratio)
    kl_coef = _validated_positive("kl_coef", kl_coef)
    target_kl = _validated_positive("target_kl", target_kl)
    exploration_std = set_exploration_std(policy, exploration_std)
    max_anchor_kl = 2.0 * target_kl

    anchor = copy.deepcopy(policy)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, _ = _make_env(seed=seed, asset_cache=asset_cache)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    obs, _ = env.reset(seed=seed)
    state = policy.initial_state()
    history: list[dict] = []
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
                        reward,
                        terminated=terminated,
                        truncated=truncated,
                        next_value=float(terminal_value[0]),
                        gamma=gamma,
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

            losses: list[float] = []
            kls: list[float] = []
            rejected_update_kl = None
            stopped_for_kl = False
            for _ in range(epochs):
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
                    distribution, new_value, _ = policy.distribution(
                        obs_tensor[idx], state_tensor[idx]
                    )
                    with torch.no_grad():
                        anchor_distribution, _, _ = anchor.distribution(
                            obs_tensor[idx], state_tensor[idx]
                        )
                    new_logp = distribution.log_prob(raw_tensor[idx]).sum(-1)
                    ratio = torch.exp(new_logp - old_logp[idx])
                    unclipped = ratio * adv_tensor[idx]
                    clipped = (
                        torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_tensor[idx]
                    )
                    actor_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * torch.mean((new_value - ret_tensor[idx]) ** 2)
                    anchor_kl = kl_divergence(distribution, anchor_distribution).sum(-1).mean()
                    loss = actor_loss + 0.5 * value_loss + kl_coef * anchor_kl

                    def current_anchor_kl() -> torch.Tensor:
                        current_distribution, _, _ = policy.distribution(
                            obs_tensor[idx], state_tensor[idx]
                        )
                        return kl_divergence(
                            current_distribution, anchor_distribution
                        ).sum(-1).mean()

                    accepted, observed_kl = _guarded_optimizer_step(
                        policy,
                        optimizer,
                        loss,
                        kl_fn=current_anchor_kl,
                        max_kl=max_anchor_kl,
                    )
                    if not accepted:
                        rejected_update_kl = observed_kl
                        stopped_for_kl = True
                        break
                    losses.append(float(loss.detach()))
                    kls.append(observed_kl)
                if stopped_for_kl:
                    break

            steps_done += n
            history.append(
                {
                    "steps": steps_done,
                    "mean_reward_per_step": float(rewards.mean()),
                    "episodes_ended": int(dones.sum()),
                    "loss": float(np.mean(losses)) if losses else None,
                    "mean_anchor_kl": float(np.mean(kls)) if kls else 0.0,
                    "max_observed_anchor_kl": float(np.max(kls)) if kls else 0.0,
                    "rejected_update_kl": rejected_update_kl,
                    "stopped_for_kl": stopped_for_kl,
                    "exploration_std": exploration_std,
                    "target_kl": target_kl,
                    "max_allowed_anchor_kl": max_anchor_kl,
                    "kl_coef": kl_coef,
                }
            )
    finally:
        env.close()
    return history


def main() -> None:
    """Run the ordinary CAPE trainer with anchored PPO injected as one experiment."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exploration-std", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--kl-coef", type=float, default=2.0)
    known, remaining = parser.parse_known_args()

    def refinement(policy, **kwargs):
        """Inject anchored PPO while preserving the ordinary trainer interface."""
        return anchored_ppo_finetune(
            policy,
            exploration_std=known.exploration_std,
            target_kl=known.target_kl,
            kl_coef=known.kl_coef,
            **kwargs,
        )

    flight_training.ppo_finetune = refinement
    sys.argv = [sys.argv[0], *remaining]
    flight_training.main()


if __name__ == "__main__":
    main()
