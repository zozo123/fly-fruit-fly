import numpy as np
import pytest
import torch

from fly_fruit_fly.circuit import (
    CAPE_ARCHITECTURE,
    CapeSuperFlyPolicy,
    _require_routed_path,
    balanced_role_nodes,
    load_cape_checkpoint,
    save_cape_checkpoint,
    transient_weight,
)
from fly_fruit_fly.connectome import ConnectomeGraph
from fly_fruit_fly.flight_training import teacher_mix_beta


def role_graph():
    node_ids = np.arange(100, 108, dtype=np.int64)
    return ConnectomeGraph(
        node_ids=node_ids,
        edge_src=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        edge_dst=np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int64),
        edge_weight=np.linspace(0.1, 0.8, 8, dtype=np.float32),
        metadata={
            "kind": "test-cape",
            "roles": {
                "sensory_input_node_ids": [100],
                "ascending_node_ids": [101],
                "descending_node_ids": [104],
                "motor_output_node_ids": [105],
                "flight_annotated_node_ids": [106],
            },
        },
    )


def make_policy(substeps=1):
    torch.manual_seed(7)
    return CapeSuperFlyPolicy(
        role_graph(), 4, -np.ones(12, dtype=np.float32), np.ones(12, dtype=np.float32),
        circuit_substeps=substeps,
    )


def test_balanced_role_nodes_represents_available_roles_deterministically():
    roles = {
        "sensory_input": {1, 2},
        "ascending": {3},
        "descending": {4},
        "motor_output": {5},
        "flight_annotated": {6},
    }
    scores = {1: 1.0, 2: 9.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0}
    first = balanced_role_nodes(roles, scores, 5)
    second = balanced_role_nodes(roles, scores, 5)
    assert first == second
    assert set(first) == {2, 3, 4, 5, 6}


def test_cape_requires_a_directed_path_from_inputs_to_outputs():
    _require_routed_path({"reachable_output_nodes": 1})
    with pytest.raises(RuntimeError, match="no directed path"):
        _require_routed_path({"reachable_output_nodes": 0})


def test_cape_policy_rejects_disconnected_role_routing():
    graph = ConnectomeGraph(
        node_ids=np.arange(100, 104, dtype=np.int64),
        edge_src=np.array([0, 2], dtype=np.int64),
        edge_dst=np.array([1, 3], dtype=np.int64),
        edge_weight=np.ones(2, dtype=np.float32),
        metadata={
            "kind": "test-cape-disconnected",
            "roles": {
                "sensory_input_node_ids": [100],
                "ascending_node_ids": [],
                "descending_node_ids": [],
                "motor_output_node_ids": [103],
                "flight_annotated_node_ids": [],
            },
        },
    )
    with pytest.raises(RuntimeError, match="no directed path"):
        CapeSuperFlyPolicy(
            graph,
            4,
            -np.ones(12, dtype=np.float32),
            np.ones(12, dtype=np.float32),
        )


def test_cape_routes_observations_only_into_input_roles_on_first_substep():
    policy = make_policy(substeps=1)
    with torch.no_grad():
        policy.node_bias.zero_()
    _, _, state = policy.step(torch.tensor([1.0, -0.5, 0.25, 2.0]), policy.initial_state())
    input_mask = policy.input_role_mask.bool()
    assert torch.count_nonzero(state[0][~input_mask]) == 0
    assert torch.count_nonzero(state[0][input_mask]) > 0


def test_cape_motor_features_only_expose_effector_roles():
    policy = make_policy()
    state = torch.ones((1, policy.n_nodes))
    features = policy.motor_features(state)
    output_mask = policy.output_role_mask.bool()
    assert torch.count_nonzero(features[0][~output_mask]) == 0
    assert torch.all(features[0][output_mask] == 1)


def test_cape_keeps_measured_adjacency_fixed_but_learns_neuron_dynamics():
    policy = make_policy(substeps=4)
    assert policy.adjacency.is_sparse
    assert not any(name == "adjacency" for name, _ in policy.named_parameters())
    assert any(name == "source_gain" for name, _ in policy.named_parameters())
    assert policy.routing_summary()["architecture"] == CAPE_ARCHITECTURE
    assert policy.routing_summary()["circuit_substeps"] == 4


def test_cape_checkpoint_roundtrip_preserves_routed_behavior(tmp_path):
    graph = role_graph()
    policy = make_policy(substeps=3)
    checkpoint = tmp_path / "cape.pt"
    save_cape_checkpoint(policy, graph, checkpoint, {"seed": 7})
    restored = load_cape_checkpoint(checkpoint, graph)
    obs = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    np.testing.assert_array_equal(policy.predict(obs)[0], restored.predict(obs)[0])
    assert restored.circuit_substeps == 3


def test_transient_weight_and_teacher_mix_decay_safely():
    assert transient_weight(0, transient_steps=400, boost=4.0) == pytest.approx(4.0)
    assert transient_weight(400, transient_steps=400, boost=4.0) == pytest.approx(1.0)
    betas = [teacher_mix_beta(i) for i in range(12)]
    assert betas[0] == pytest.approx(0.8)
    assert all(a >= b for a, b in zip(betas, betas[1:]))
    assert min(betas) == pytest.approx(0.15)
