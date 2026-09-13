import numpy as np
import pytest
import torch

from fly_fruit_fly.connectome import ConnectomeGraph
from fly_fruit_fly.superfly import SuperFlyPolicy, distill
from fly_fruit_fly.flight_training import (
    _validate_seed_panel,
    fit_readout,
    training_rounds_complete,
    validate_validation_protocol,
    validation_protocol,
    validation_score,
)


def policy():
    graph = ConnectomeGraph(np.arange(4), np.arange(4), np.roll(np.arange(4), 1),
                            np.ones(4, dtype=np.float32), {})
    return SuperFlyPolicy(graph, 3, -np.ones(12), np.ones(12))


def test_ridge_recovers_motor_mapping_without_changing_wiring():
    torch.manual_seed(4)
    model = policy()
    before = model.adjacency.clone().to_dense()
    x = torch.cat([torch.randn(100,4,dtype=torch.float64), torch.ones(100,1)],1)
    weights = torch.randn(5,12,dtype=torch.float64)*.1
    y = x @ weights
    assert fit_readout(model, x, y, 1e-9) < 1e-12
    torch.testing.assert_close(model.adjacency.to_dense(), before)
    with pytest.raises(ValueError):
        fit_readout(model, x, y, 0)


def test_corrective_distillation_keeps_normalization_fixed():
    model = policy()
    model.set_observation_normalization(np.array([1.,2.,3.]),np.array([2.,3.,4.]))
    before_mean, before_std = model.obs_mean.clone(), model.obs_std.clone()
    rollouts = [{'observations':np.zeros((4,3),dtype=np.float32),
                 'actions':np.zeros((4,12),dtype=np.float32)}]
    distill(model,rollouts,epochs=1,bptt_steps=2,reset_normalization=False)
    torch.testing.assert_close(model.obs_mean,before_mean)
    torch.testing.assert_close(model.obs_std,before_std)


def test_validation_prioritizes_real_completion_over_reward():
    def report(completed, duration, error):
        return {'episodes':[{'completed_reference':completed,'duration_s':duration,
                             'mean_tracking_error_cm':error}]}
    assert validation_score(report(True,.5988,.09)) > validation_score(report(False,.58,.01))


def test_validation_protocol_keeps_tuning_promotion_and_test_seeds_disjoint():
    protocol = validation_protocol()
    tuning = set(protocol['ridge_selection_seeds'])
    promotion = set(protocol['round_promotion_seeds'])
    held_out = set(protocol['held_out_test_seeds'])
    assert len(tuning) == 3
    assert len(promotion) == 3
    assert len(held_out) == 10
    assert tuning.isdisjoint(promotion)
    assert tuning.isdisjoint(held_out)
    assert promotion.isdisjoint(held_out)


def test_seed_panels_must_match_evaluate_consecutive_seed_semantics():
    assert _validate_seed_panel('panel', [10, 11, 12]) == [10, 11, 12]
    with pytest.raises(ValueError, match='consecutive'):
        _validate_seed_panel('panel', [10, 12])
    with pytest.raises(ValueError, match='duplicate'):
        _validate_seed_panel('panel', [10, 10])
    with pytest.raises(ValueError, match='non-negative integer'):
        _validate_seed_panel('panel', [10, -1])


def test_validation_protocol_rejects_any_cross_panel_overlap():
    protocol = validation_protocol()
    protocol['round_promotion_seeds'] = [30002, 30003, 30004]
    with pytest.raises(ValueError, match='overlap'):
        validate_validation_protocol(protocol)


def test_training_consumes_full_configured_dagger_round_budget():
    assert not training_rounds_complete(0, 8)
    assert not training_rounds_complete(7, 8)
    assert training_rounds_complete(8, 8)
    assert training_rounds_complete(9, 8)
    with pytest.raises(ValueError, match='non-negative'):
        training_rounds_complete(-1, 8)
