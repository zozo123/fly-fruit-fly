import numpy as np
import torch

from fly_fruit_fly.connectome import (
    ConnectomeGraph,
    degree_preserving_shuffle,
    load_graph,
    save_graph,
)
from fly_fruit_fly.superfly import SuperFlyPolicy, _compute_gae, distill


def toy_graph():
    return ConnectomeGraph(
        node_ids=np.arange(8, dtype=np.int64) + 100,
        edge_src=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=np.int64),
        edge_dst=np.array([1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7, 1, 0, 2], dtype=np.int64),
        edge_weight=np.linspace(0.1, 1.6, 16, dtype=np.float32),
        metadata={"dataset": "toy"},
    )


def degrees(graph):
    out_degree = np.bincount(graph.edge_src, minlength=graph.n_nodes)
    in_degree = np.bincount(graph.edge_dst, minlength=graph.n_nodes)
    return in_degree, out_degree


def test_degree_preserving_shuffle_is_a_real_control():
    graph = toy_graph()
    graph.validate()
    shuffled = degree_preserving_shuffle(graph, seed=7, swaps=40)
    original_in, original_out = degrees(graph)
    shuffled_in, shuffled_out = degrees(shuffled)
    np.testing.assert_array_equal(original_in, shuffled_in)
    np.testing.assert_array_equal(original_out, shuffled_out)
    np.testing.assert_allclose(np.sort(graph.edge_weight), np.sort(shuffled.edge_weight))
    np.testing.assert_array_equal(graph.node_ids, shuffled.node_ids)
    assert set(zip(graph.edge_src, graph.edge_dst)) != set(zip(shuffled.edge_src, shuffled.edge_dst))


def test_graph_round_trip_preserves_provenance(tmp_path):
    graph = toy_graph()
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    loaded = load_graph(path)
    np.testing.assert_array_equal(loaded.node_ids, graph.node_ids)
    np.testing.assert_array_equal(loaded.edge_src, graph.edge_src)
    np.testing.assert_array_equal(loaded.edge_dst, graph.edge_dst)
    np.testing.assert_allclose(loaded.edge_weight, graph.edge_weight)
    assert loaded.metadata == graph.metadata


def test_policy_uses_fixed_sparse_graph_and_emits_finite_bounded_actions():
    graph = toy_graph()
    low = np.full(12, -1.0, dtype=np.float32)
    high = np.full(12, 1.0, dtype=np.float32)
    policy = SuperFlyPolicy(graph, obs_dim=10, action_low=low, action_high=high)
    assert policy.adjacency.is_sparse
    assert not any(name == "adjacency" for name, _ in policy.named_parameters())
    action, state = policy.predict(np.linspace(-1, 1, 10, dtype=np.float32))
    assert action.shape == (12,)
    assert state.shape == (1, graph.n_nodes)
    assert np.isfinite(action).all()
    assert np.all(action >= low) and np.all(action <= high)


def test_distillation_learns_a_simple_teacher_mapping():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    graph = toy_graph()
    low = np.full(12, -1.0, dtype=np.float32)
    high = np.full(12, 1.0, dtype=np.float32)
    policy = SuperFlyPolicy(graph, obs_dim=4, action_low=low, action_high=high)
    obs = rng.normal(size=(96, 4)).astype(np.float32)
    projection = rng.normal(scale=0.15, size=(4, 12)).astype(np.float32)
    actions = np.tanh(obs @ projection).astype(np.float32)
    rollouts = [{
        "seed": 3,
        "observations": obs,
        "actions": actions,
        "rewards": np.zeros(len(obs), dtype=np.float32),
    }]
    losses = distill(policy, rollouts, epochs=5, learning_rate=3e-3, bptt_steps=24)
    assert len(losses) == 5
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


def test_gae_is_finite_and_rewards_propagate_backwards():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.array([False, False, True])
    advantages, returns = _compute_gae(rewards, values, dones, next_value=0.0)
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()
    assert advantages[0] > 0
    assert advantages[1] > advantages[0]
    assert advantages[2] == 1.0
