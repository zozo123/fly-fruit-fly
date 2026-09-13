import numpy as np
import pytest
import torch

from fly_fruit_fly.connectome import ConnectomeGraph
from fly_fruit_fly.superfly import SuperFlyPolicy, distill
from fly_fruit_fly.flight_training import (
    _episode_gate_progress,
    _validate_seed_panel,
    fit_readout,
    rl_training_seed,
    teacher_mix_beta,
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


def test_gate_progress_uses_limiting_duration_or_tracking_margin():
    duration_limited = {
        'completed_reference': False,
        'duration_s': .295,
        'mean_tracking_error_cm': .05,
    }
    error_limited = {
        'completed_reference': False,
        'duration_s': .59,
        'mean_tracking_error_cm': .2,
    }
    assert _episode_gate_progress(duration_limited) == pytest.approx(.5)
    assert _episode_gate_progress(error_limited) == pytest.approx(.5)


def test_validation_does_not_trade_large_tracking_error_for_duration():
    def report(duration, error):
        return {'episodes':[{'completed_reference':False,'duration_s':duration,
                             'mean_tracking_error_cm':error}]}
    balanced = report(.20, .20)
    long_but_bad = report(.30, .50)
    assert validation_score(balanced) > validation_score(long_but_bad)


def test_teacher_mix_freezes_when_student_teacher_divergence_rises():
    previous = {
        'beta': .2985984,
        'completion_rate': 1.0,
        'mean_student_teacher_l1': .074,
    }
    assert teacher_mix_beta(4, previous_summary=previous) == pytest.approx(.2985984)


def test_teacher_mix_recovers_after_corrective_rollout_collapse():
    previous = {
        'beta': .214990848,
        'completion_rate': 0.0,
        'mean_student_teacher_l1': .108,
    }
    assert teacher_mix_beta(5, previous_summary=previous) == pytest.approx(.2985984)


def test_validation_protocol_keeps_all_seed_domains_disjoint():
    protocol = validation_protocol()
    panels = {
        name: set(seeds)
        for name, seeds in protocol.items()
    }
    assert len(panels['ridge_selection_seeds']) == 3
    assert len(panels['round_promotion_seeds']) == 3
    assert len(panels['development_test_seeds']) == 10
    assert len(panels['confirmation_test_seeds']) == 10
    names = list(panels)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert panels[first].isdisjoint(panels[second])


def test_confirmation_panel_is_fresh_relative_to_consumed_development_panel():
    protocol = validation_protocol()
    assert protocol['development_test_seeds'] == list(range(70000, 70010))
    assert protocol['confirmation_test_seeds'] == list(range(71000, 71010))


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
    protocol['confirmation_test_seeds'] = [70009, 70010, 70011]
    with pytest.raises(ValueError, match='overlap'):
        validate_validation_protocol(protocol)


def test_rl_training_seed_uses_a_separate_namespace():
    protocol = validation_protocol()
    seed = rl_training_seed(1234, protocol)
    reserved = {value for panel in protocol.values() for value in panel}
    assert seed == 101234
    assert seed not in reserved


def test_rl_training_seed_rejects_collision_with_any_evaluation_panel():
    protocol = validation_protocol()
    protocol['confirmation_test_seeds'] = list(range(100000, 100010))
    with pytest.raises(ValueError, match='overlaps'):
        rl_training_seed(0, protocol)
