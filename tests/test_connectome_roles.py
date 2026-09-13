import numpy as np
import pytest

from fly_fruit_fly.connectome import ConnectomeGraph, classify_banc_roles, graph_role_mask


def test_banc_roles_keep_peripheral_io_and_cns_relays_distinct():
    roots = np.arange(10, 16, dtype=np.int64)
    roles = classify_banc_roles(
        roots,
        flow=["afferent", "efferent", "intrinsic", "intrinsic", "intrinsic", "intrinsic"],
        super_class=["sensory", "motor", "ascending", "descending", "central", "central"],
        cell_function=["", "", "", "flight steering", "wing premotor", "unrelated"],
    )

    assert roles["sensory_input"] == {10}
    assert roles["motor_output"] == {11}
    assert roles["ascending"] == {12}
    assert roles["descending"] == {13}
    assert roles["flight_annotated"] == {13, 14}
    assert roles["selection_seed"] == {11, 13, 14}
    assert roles["sensory_input"].isdisjoint(roles["motor_output"])


def test_banc_role_columns_must_align_with_root_ids():
    with pytest.raises(ValueError, match="same length"):
        classify_banc_roles([1, 2], flow=["afferent"])


def test_graph_role_mask_uses_recorded_root_ids_not_node_positions():
    graph = ConnectomeGraph(
        node_ids=np.array([300, 100, 200], dtype=np.int64),
        edge_src=np.array([0, 1, 2], dtype=np.int64),
        edge_dst=np.array([1, 2, 0], dtype=np.int64),
        edge_weight=np.ones(3, dtype=np.float32),
        metadata={
            "roles": {
                "sensory_input_node_ids": [100, 300],
                "motor_output_node_ids": [200],
            }
        },
    )
    np.testing.assert_array_equal(graph_role_mask(graph, "sensory_input"), [True, True, False])
    np.testing.assert_array_equal(graph_role_mask(graph, "motor_output"), [False, False, True])


def test_legacy_graph_does_not_invent_role_provenance():
    graph = ConnectomeGraph(
        node_ids=np.array([1, 2], dtype=np.int64),
        edge_src=np.array([0], dtype=np.int64),
        edge_dst=np.array([1], dtype=np.int64),
        edge_weight=np.array([1.0], dtype=np.float32),
        metadata={"dataset": "legacy"},
    )
    with pytest.raises(ValueError, match="no recorded BANC role provenance"):
        graph_role_mask(graph, "sensory_input")
