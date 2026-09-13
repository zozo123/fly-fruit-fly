import copy
import math

import pytest
import torch

from fly_fruit_fly.cape_rl import (
    _guarded_optimizer_step,
    anchored_ppo_finetune,
    set_exploration_std,
)


class DummyPolicy:
    def __init__(self):
        self.log_std = torch.nn.Parameter(torch.full((12,), -1.5))


class ScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))


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


def test_kl_guard_rolls_back_model_and_optimizer_after_oversized_step():
    model = ScalarModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    loss = -model.value

    accepted, observed_kl = _guarded_optimizer_step(
        model,
        optimizer,
        loss,
        kl_fn=lambda: model.value.square(),
        max_kl=0.01,
    )

    assert not accepted
    assert observed_kl > 0.01
    torch.testing.assert_close(model.value.detach(), before_model["value"])
    assert optimizer.state_dict() == before_optimizer


def test_kl_guard_refuses_already_out_of_bounds_policy_without_mutation():
    model = ScalarModule()
    with torch.no_grad():
        model.value.fill_(0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    before = model.value.detach().clone()
    loss = -model.value

    accepted, observed_kl = _guarded_optimizer_step(
        model,
        optimizer,
        loss,
        kl_fn=lambda: model.value.square(),
        max_kl=0.01,
    )

    assert not accepted
    assert observed_kl == pytest.approx(0.04)
    torch.testing.assert_close(model.value.detach(), before)
    assert not optimizer.state
