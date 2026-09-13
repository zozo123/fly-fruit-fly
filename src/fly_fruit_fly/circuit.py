"""CAPE: Connectome-Aligned Perception and Effector routing for SuperFly.

CAPE keeps the measured BANC v888 neuron-to-neuron topology fixed while making the
engineering interfaces biologically explicit: observations enter sensory/ascending
neurons, recurrent state propagates through measured wiring, and motor commands are
read from motor/descending/flight-related neurons. Learned dynamics are engineering
components and are not claims about biological membrane or synapse dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .connectome import (
    ConnectomeGraph,
    _feather,
    _metadata_role_ids,
    ensure_banc_files,
    graph_role_mask,
    save_graph,
)
from .superfly import (
    SuperFlyPolicy,
    _check_graph_adjacency,
    _observation_stats,
    save_checkpoint,
)

CAPE_ARCHITECTURE = "cape_superfly_v1"
CAPE_ROLE_ORDER = (
    "sensory_input",
    "ascending",
    "descending",
    "motor_output",
    "flight_annotated",
)


def balanced_role_nodes(
    role_ids: dict[str, set[int]],
    scores: dict[int, float],
    budget: int,
) -> list[int]:
    """Round-robin high-score role seeds so one large class cannot crowd out I/O."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget == 0:
        return []
    ranked: dict[str, list[int]] = {}
    for role in CAPE_ROLE_ORDER:
        candidates = {int(node) for node in role_ids.get(role, set()) if int(node) in scores}
        ranked[role] = sorted(candidates, key=lambda node: (-scores[node], node))

    selected: list[int] = []
    used: set[int] = set()
    cursors = {role: 0 for role in CAPE_ROLE_ORDER}
    while len(selected) < budget:
        progressed = False
        for role in CAPE_ROLE_ORDER:
            candidates = ranked[role]
            cursor = cursors[role]
            while cursor < len(candidates) and candidates[cursor] in used:
                cursor += 1
            cursors[role] = cursor
            if cursor >= len(candidates):
                continue
            node = candidates[cursor]
            cursors[role] = cursor + 1
            selected.append(node)
            used.add(node)
            progressed = True
            if len(selected) >= budget:
                break
        if not progressed:
            break
    return selected


def _score_nodes(pre: np.ndarray, post: np.ndarray, count: np.ndarray):
    nodes = np.concatenate([pre, post])
    weights = np.concatenate([count, count])
    unique, inverse = np.unique(nodes, return_inverse=True)
    scores = np.bincount(inverse, weights=weights)
    return unique.astype(np.int64, copy=False), scores


def _routing_reachability(
    n_nodes: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    input_mask: np.ndarray,
    output_mask: np.ndarray,
) -> dict:
    """Summarize directed paths from routed input neurons to routed output neurons."""
    adjacency = [[] for _ in range(n_nodes)]
    for src, dst in zip(edge_src, edge_dst):
        adjacency[int(src)].append(int(dst))
    distance = np.full(n_nodes, -1, dtype=np.int64)
    queue = [int(i) for i in np.flatnonzero(input_mask)]
    for node in queue:
        distance[node] = 0
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for neighbor in adjacency[node]:
            if distance[neighbor] >= 0:
                continue
            distance[neighbor] = distance[node] + 1
            queue.append(neighbor)
    output_indices = np.flatnonzero(output_mask)
    reachable = output_indices[distance[output_indices] >= 0]
    hops = distance[reachable]
    return {
        "input_nodes": int(input_mask.sum()),
        "output_nodes": int(output_mask.sum()),
        "reachable_output_nodes": int(len(reachable)),
        "minimum_input_to_output_hops": int(hops.min()) if len(hops) else None,
        "maximum_input_to_output_hops": int(hops.max()) if len(hops) else None,
    }


def _require_routed_path(reachability: dict) -> None:
    """Reject CAPE graphs where biological inputs cannot reach any routed output."""
    if int(reachability.get("reachable_output_nodes", 0)) <= 0:
        raise RuntimeError(
            "CAPE graph has no directed path from routed biological inputs to outputs"
        )


def materialize_cape_subgraph(
    cache_dir: Path,
    output: Path,
    *,
    n_nodes: int = 384,
    min_synapses: int = 5,
    role_fraction: float = 0.55,
) -> ConnectomeGraph:
    """Build a role-balanced, closed-loop BANC subgraph for CAPE control.

    A fixed fraction of the node budget is reserved for high-connectivity neurons
    across sensory, ascending, descending, motor and flight-annotated roles. The
    remainder is filled by their strongest measured partners, then globally strong
    nodes. Every recurrent edge is still a real BANC v888 edge with its released
    input-normalized ``norm`` weight.
    """
    if n_nodes < 32:
        raise ValueError("CAPE requires at least 32 nodes")
    if min_synapses < 1:
        raise ValueError("min_synapses must be positive")
    if not 0.0 < role_fraction <= 1.0:
        raise ValueError("role_fraction must be in (0, 1]")

    edge_path, meta_path, source_manifest = ensure_banc_files(cache_dir)
    feather = _feather()
    table = feather.read_table(
        edge_path, columns=["pre", "post", "count", "norm"], memory_map=True
    )
    pre = table["pre"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    post = table["post"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    count = table["count"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    norm = table["norm"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    keep = np.isfinite(count) & np.isfinite(norm) & (count >= min_synapses) & (norm > 0)
    pre, post, count, norm = pre[keep], post[keep], count[keep], norm[keep]

    role_ids = _metadata_role_ids(meta_path)
    global_nodes, global_scores = _score_nodes(pre, post, count)
    score_map = {int(node): float(score) for node, score in zip(global_nodes, global_scores)}
    available_roles = sum(
        bool(set(map(int, role_ids.get(role, set()))).intersection(score_map))
        for role in CAPE_ROLE_ORDER
    )
    role_budget = min(
        n_nodes,
        max(available_roles, int(round(n_nodes * role_fraction))),
    )
    role_seeds = balanced_role_nodes(role_ids, score_map, role_budget)
    if not role_seeds:
        raise RuntimeError("BANC metadata produced no CAPE role seeds")

    selected: list[int] = list(role_seeds)
    used = set(selected)
    seed_array = np.asarray(role_seeds, dtype=np.int64)
    incident = np.isin(pre, seed_array) | np.isin(post, seed_array)
    if incident.any():
        partner_nodes, partner_scores = _score_nodes(pre[incident], post[incident], count[incident])
        partner_order = np.lexsort((partner_nodes, -partner_scores))
        for node in partner_nodes[partner_order]:
            node = int(node)
            if node in used:
                continue
            selected.append(node)
            used.add(node)
            if len(selected) >= n_nodes:
                break

    if len(selected) < n_nodes:
        global_order = np.lexsort((global_nodes, -global_scores))
        for node in global_nodes[global_order]:
            node = int(node)
            if node in used:
                continue
            selected.append(node)
            used.add(node)
            if len(selected) >= n_nodes:
                break

    selected_array = np.sort(np.asarray(selected[:n_nodes], dtype=np.int64))
    induced = np.isin(pre, selected_array) & np.isin(post, selected_array)
    ipre, ipost, iweight, icount = pre[induced], post[induced], norm[induced], count[induced]
    if ipre.size == 0:
        raise RuntimeError("selected CAPE nodes have no induced BANC edges")
    node_to_index = {int(node): i for i, node in enumerate(selected_array)}
    edge_src = np.fromiter((node_to_index[int(node)] for node in ipre), dtype=np.int64)
    edge_dst = np.fromiter((node_to_index[int(node)] for node in ipost), dtype=np.int64)
    edge_weight = iweight.astype(np.float32, copy=False)

    selected_set = set(map(int, selected_array))
    selected_roles = {
        role: sorted(selected_set.intersection(map(int, role_ids.get(role, set()))))
        for role in CAPE_ROLE_ORDER
    }
    input_ids = set(selected_roles["sensory_input"]) | set(selected_roles["ascending"])
    output_ids = (
        set(selected_roles["motor_output"])
        | set(selected_roles["descending"])
        | set(selected_roles["flight_annotated"])
    )
    if not input_ids or not output_ids:
        raise RuntimeError("CAPE graph must contain both biological input and output roles")
    input_mask = np.isin(selected_array, np.fromiter(input_ids, dtype=np.int64))
    output_mask = np.isin(selected_array, np.fromiter(output_ids, dtype=np.int64))
    reachability = _routing_reachability(
        len(selected_array), edge_src, edge_dst, input_mask, output_mask
    )
    _require_routed_path(reachability)

    metadata = {
        "kind": "banc_cape_closed_loop_structural_prior",
        "source": source_manifest,
        "selection": {
            "strategy": "balanced biological roles + strongest measured partners",
            "n_nodes_requested": n_nodes,
            "n_nodes": int(selected_array.size),
            "n_edges": int(edge_src.size),
            "min_synapses": min_synapses,
            "role_fraction": float(role_fraction),
            "role_seed_budget": int(role_budget),
            "role_seed_count": int(len(role_seeds)),
            "weight": "BANC input-normalized synaptic weight (norm)",
            "synapse_count_sum": float(icount.sum()),
        },
        "roles": {
            "sensory_input_node_ids": selected_roles["sensory_input"],
            "motor_output_node_ids": selected_roles["motor_output"],
            "ascending_node_ids": selected_roles["ascending"],
            "descending_node_ids": selected_roles["descending"],
            "flight_annotated_node_ids": selected_roles["flight_annotated"],
            "sensory_rule": "flow=afferent OR super_class=sensory",
            "motor_rule": "flow=efferent OR super_class=motor",
            "relay_rule": "super_class=ascending/descending",
            "cape_input_roles": ["sensory_input", "ascending"],
            "cape_output_roles": ["motor_output", "descending", "flight_annotated"],
        },
        "routing": reachability,
    }
    graph = ConnectomeGraph(selected_array, edge_src, edge_dst, edge_weight, metadata)
    graph.validate()
    save_graph(graph, output)
    return graph


class CapeSuperFlyPolicy(SuperFlyPolicy):
    """SuperFly with BANC-role-gated I/O and several fixed-graph propagation steps."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        circuit_substeps: int = 4,
        strict_role_routing: bool = True,
    ):
        super().__init__(graph, obs_dim, action_low, action_high)
        if circuit_substeps < 1 or circuit_substeps > 16:
            raise ValueError("circuit_substeps must be in [1, 16]")
        self.circuit_substeps = int(circuit_substeps)

        try:
            input_mask = graph_role_mask(graph, "sensory_input") | graph_role_mask(graph, "ascending")
            output_mask = (
                graph_role_mask(graph, "motor_output")
                | graph_role_mask(graph, "descending")
                | graph_role_mask(graph, "flight_annotated")
            )
        except ValueError:
            if strict_role_routing:
                raise
            input_mask = np.ones(graph.n_nodes, dtype=bool)
            output_mask = np.ones(graph.n_nodes, dtype=bool)
        if strict_role_routing and (not input_mask.any() or not output_mask.any()):
            raise ValueError("CAPE requires non-empty BANC input and output role masks")
        if strict_role_routing:
            reachability = _routing_reachability(
                graph.n_nodes,
                graph.edge_src,
                graph.edge_dst,
                input_mask,
                output_mask,
            )
            _require_routed_path(reachability)

        self.register_buffer(
            "input_role_mask", torch.as_tensor(input_mask, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "output_role_mask", torch.as_tensor(output_mask, dtype=torch.float32), persistent=False
        )
        # BANC v888 supplies unsigned structural weights. This bounded learned source
        # gain lets the engineering dynamics learn effective excitation/inhibition
        # without changing a single measured edge or inventing synapses.
        initial = np.arctanh(2.0 / 3.0)
        self.source_gain = torch.nn.Parameter(torch.full((self.n_nodes,), float(initial)))

    def motor_features(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.output_role_mask

    def routing_summary(self) -> dict:
        return {
            "architecture": CAPE_ARCHITECTURE,
            "circuit_substeps": self.circuit_substeps,
            "input_nodes": int(self.input_role_mask.sum().item()),
            "output_nodes": int(self.output_role_mask.sum().item()),
            "input_roles": ["sensory_input", "ascending"],
            "output_roles": ["motor_output", "descending", "flight_annotated"],
            "recurrent_topology": "fixed BANC v888 measured edges",
            "recurrent_source_gain": "learned bounded engineering dynamics",
        }

    def step(self, obs: torch.Tensor, state: torch.Tensor):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        normalized = torch.clamp((obs - self.obs_mean) / self.obs_std, -10.0, 10.0)
        sensory_drive = self.sensory(normalized) * self.input_role_mask
        recurrent_gain = torch.sigmoid(self.recurrent_gain) * 1.8
        source_gain = torch.tanh(self.source_gain) * 1.5
        leak = torch.sigmoid(self.leak_logits)
        next_state = state
        for _ in range(self.circuit_substeps):
            presynaptic = next_state * source_gain
            recurrent_drive = torch.sparse.mm(self.adjacency, presynaptic.T).T
            candidate = torch.tanh(
                sensory_drive + self.node_bias + recurrent_gain * recurrent_drive
            )
            next_state = (1.0 - leak) * next_state + leak * candidate
        mean_raw = self.motor(self.motor_features(next_state))
        value = self.value(next_state).squeeze(-1)
        return mean_raw, value, next_state


def transient_weight(step: int, *, transient_steps: int = 400, boost: float = 4.0) -> float:
    """Emphasize the early flight transient where previous students diverged."""
    if transient_steps <= 0:
        raise ValueError("transient_steps must be positive")
    if boost < 1.0:
        raise ValueError("boost must be at least 1")
    phase = max(0.0, 1.0 - float(step) / float(transient_steps))
    return 1.0 + (boost - 1.0) * phase


def cape_distill(
    policy: CapeSuperFlyPolicy,
    rollouts,
    *,
    epochs: int,
    learning_rate: float,
    bptt_steps: int = 96,
    reset_normalization: bool = True,
    transient_steps: int = 400,
    transient_boost: float = 4.0,
) -> list[float]:
    """Distill expert actions while explicitly weighting the unstable launch transient."""
    if epochs <= 0 or bptt_steps <= 0:
        raise ValueError("epochs and bptt_steps must be positive")
    if reset_normalization:
        mean, std = _observation_stats(rollouts)
        policy.set_observation_normalization(mean, std)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    losses: list[float] = []
    policy.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for rollout in rollouts:
            obs = torch.as_tensor(rollout["observations"], dtype=torch.float32)
            target = torch.as_tensor(
                policy.normalize_action(rollout["actions"]), dtype=torch.float32
            )
            state = policy.initial_state()
            for start in range(0, len(obs), bptt_steps):
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = torch.zeros((), dtype=torch.float32)
                weight_sum = 0.0
                end = min(start + bptt_steps, len(obs))
                for t in range(start, end):
                    mean_raw, _, state = policy.step(obs[t], state)
                    weight = transient_weight(
                        t, transient_steps=transient_steps, boost=transient_boost
                    )
                    weighted_loss = weighted_loss + weight * torch.mean(
                        (torch.tanh(mean_raw[0]) - target[t]) ** 2
                    )
                    weight_sum += weight
                loss = weighted_loss / weight_sum
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                state = state.detach()
                epoch_loss += float(loss.detach()) * (end - start)
                epoch_steps += end - start
        losses.append(epoch_loss / max(epoch_steps, 1))
    return losses


def save_cape_checkpoint(
    policy: CapeSuperFlyPolicy,
    graph: ConnectomeGraph,
    output: Path,
    training: dict,
) -> None:
    manifest = dict(training)
    manifest["policy_architecture"] = CAPE_ARCHITECTURE
    manifest["cape_routing"] = policy.routing_summary()
    save_checkpoint(policy, graph, output, manifest)


def load_cape_checkpoint(checkpoint: Path, graph: ConnectomeGraph) -> CapeSuperFlyPolicy:
    """Load a CAPE checkpoint without silently falling back to dense I/O routing."""
    graph.validate()
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _check_graph_adjacency(data["state_dict"]["adjacency"], graph)
    if data.get("graph_metadata") != graph.metadata:
        raise ValueError("Checkpoint provenance does not match the supplied connectome graph")
    if "graph_node_ids" in data and not torch.equal(
        data["graph_node_ids"], torch.as_tensor(graph.node_ids, dtype=torch.int64)
    ):
        raise ValueError("Checkpoint neuron IDs do not match the supplied connectome graph")
    training = data.get("training", {})
    if training.get("policy_architecture") != CAPE_ARCHITECTURE:
        raise ValueError("Checkpoint is not a CAPE SuperFly checkpoint")
    substeps = int(training.get("cape_routing", {}).get("circuit_substeps", 4))
    policy = CapeSuperFlyPolicy(
        graph,
        int(data["obs_dim"]),
        np.asarray(data["action_low"], dtype=np.float32),
        np.asarray(data["action_high"], dtype=np.float32),
        circuit_substeps=substeps,
    )
    policy.load_state_dict(data["state_dict"])
    policy.eval()
    return policy