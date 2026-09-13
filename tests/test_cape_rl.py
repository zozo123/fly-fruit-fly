import math

import pytest
import torch

from fly_fruit_fly.cape_rl import anchored_ppo_finetune, set_exploration_std


class DummyPolicy:
    def __init__(self):
        self.log_std = torch.nn.Parameter(torch.full((12,), -1.5))


def test_conservative_exploration_std_is_explicit_and_bounded():
    policy = DummyPolicy()
    assert set_exploration_std(policy, 0.05) == pytest.approx(0.05)
    assert torch.allclose(
        policy.log_std.detach(),
        torch.full((12,), math.log(0.05)),
    )
    with pytest.raises(ValueError, match="finite and positive"):
        set_exploration_std(policy, 0.0)
    with pytest.raises(ValueError, match="log-std clamp"):
        set_exploration_std(policy, 1e-4)


def test_zero_step_anchored_ppo_is_a_noop_without_environment_access():
    policy = DummyPolicy()
    assert anchored_ppo_finetune(
        policy,
        total_steps=0,
        seed=7,
        asset_cache=None,
    ) == []
    assert torch.all(policy.log_std.detach() == -1.5)
