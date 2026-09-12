"""BANC v888 connectome materialization for SuperFly.

The measured connectome is a structural prior, not a claim that the model recreates
biological neural dynamics. We keep source/version/checksum metadata with every
materialized graph and provide a degree-preserving shuffled control.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

import numpy as np


BANC_VERSION = "888"
BANC_PAPER_DOI = "10.1038/s41586-026-10735-w"
BANC_DATA_DOI = "10.7910/DVN/7WTH1N"
BANC_BASE = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
    "compiled_data/banc_888"
)
BANC_EDGELIST_URL = f"{BANC_BASE}/banc_888_edgelist_simple_v2.feather"
BANC_META_URL = f"{BANC_BASE}/banc_888_meta.feather"


@dataclass(frozen=True)
class ConnectomeGraph:
    node_ids: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_weight: np.ndarray
    metadata: dict

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.size)

    def validate(self) -> None:
        if self.node_ids.ndim != 1 or self.node_ids.size < 2:
            raise ValueError("connectome graph must contain at least two nodes")
        if len(set(map(int, self.node_ids))) != self.n_nodes:
            raise ValueError("connectome node IDs must be unique")
        if not (self.edge_src.shape == self.edge_dst.shape == self.edge_weight.shape):
            raise ValueError("connectome edge arrays must have identical shapes")
        if self.n_edges == 0:
            raise ValueError("connectome graph must contain at least one edge")
        if np.any(self.edge_src < 0) or np.any(self.edge_src >= self.n_nodes):
            raise ValueError("edge_src contains an invalid node index")
        if np.any(self.edge_dst < 0) or np.any(self.edge_dst >= self.n_nodes):
            raise ValueError("edge_dst contains an invalid node index")
        if not np.isfinite(self.edge_weight).all() or np.any(self.edge_weight <= 0):
            raise ValueError("edge weights must be finite and positive")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                shutil.copyfileobj(response, tmp)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError(f"empty download from {url}")
            tmp_path.replace(destination)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def ensure_banc_files(cache_dir: Path) -> tuple[Path, Path, dict]:
    """Download the paper-version BANC v888 metadata and neuron edgelist."""
    cache_dir = Path(cache_dir).expanduser().resolve() / "banc-v888"
    edge_path = cache_dir / "banc_888_edgelist_simple_v2.feather"
    meta_path = cache_dir / "banc_888_meta.feather"
    if not edge_path.is_file():
        _download(BANC_EDGELIST_URL, edge_path)
    if not meta_path.is_file():
        _download(BANC_META_URL, meta_path)
    manifest = {
        "dataset": "BANC",
        "materialization": BANC_VERSION,
        "paper_doi": BANC_PAPER_DOI,
        "data_doi": BANC_DATA_DOI,
        "edgelist": {
            "url": BANC_EDGELIST_URL,
            "sha256": _file_sha256(edge_path),
            "bytes": edge_path.stat().st_size,
        },
        "metadata": {
            "url": BANC_META_URL,
            "sha256": _file_sha256(meta_path),
            "bytes": meta_path.stat().st_size,
        },
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return edge_path, meta_path, manifest


def _read_feather(path: Path, columns: list[str]):
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise RuntimeError(
            "BANC materialization needs pyarrow. Install with: pip install -e '.[connectome]'"
        ) from exc
    return feather.read_table(path, columns=columns)


def _metadata_seed_ids(meta_path: Path) -> set[int]:
    """Return motor/descending/flight-related neurons when annotations permit it."""
    try:
        import pyarrow.feather as feather
        import pyarrow.compute as pc
    except ImportError as exc:
        raise RuntimeError(
            "BANC materialization needs pyarrow. Install with: pip install -e '.[connectome]'"
        ) from exc

    schema = feather.read_table(meta_path, memory_map=True).schema
    names = set(schema.names)
    root_name = "root_id" if "root_id" in names else "pt_root_id"
    requested = [root_name]
    for candidate in ("super_class", "cell_function", "cell_class", "class"):
        if candidate in names:
            requested.append(candidate)
    table = feather.read_table(meta_path, columns=requested)
    roots = np.asarray(table[root_name]).astype(np.int64, copy=False)
    selected = np.zeros(len(roots), dtype=bool)

    if "super_class" in requested:
        values = np.asarray(pc.fill_null(table["super_class"], "")).astype(str)
        selected |= np.isin(np.char.lower(values), ["descending", "motor", "efferent"])

    keywords = ("flight", "wing", "haltere", "neck_motor", "steering")
    for column in ("cell_function", "cell_class", "class"):
        if column not in requested:
            continue
        values = np.asarray(pc.fill_null(table[column], "")).astype(str)
        lowered = np.char.lower(values)
        for keyword in keywords:
            selected |= np.char.find(lowered, keyword) >= 0

    return set(map(int, roots[selected]))


def materialize_banc_subgraph(
    cache_dir: Path,
    output: Path,
    *,
    n_nodes: int = 512,
    min_synapses: int = 5,
) -> ConnectomeGraph:
    """Build a deterministic motor/descending-centered induced BANC subgraph.

    Node selection ranks annotated motor/descending/flight-related cells and their
    strongest partners by total measured synapse count. If annotations are sparse,
    the ranking falls back to weighted degree. The final graph contains only real
    BANC v888 neuron-to-neuron edges and uses the paper's input-normalized weights.
    """
    if n_nodes < 16:
        raise ValueError("n_nodes must be at least 16")
    if min_synapses < 1:
        raise ValueError("min_synapses must be positive")

    edge_path, meta_path, source_manifest = ensure_banc_files(cache_dir)
    table = _read_feather(edge_path, ["pre", "post", "count", "norm"])
    pre = np.asarray(table["pre"]).astype(np.int64, copy=False)
    post = np.asarray(table["post"]).astype(np.int64, copy=False)
    count = np.asarray(table["count"]).astype(np.float64, copy=False)
    norm = np.asarray(table["norm"]).astype(np.float64, copy=False)
    keep = np.isfinite(count) & np.isfinite(norm) & (count >= min_synapses) & (norm > 0)
    pre, post, count, norm = pre[keep], post[keep], count[keep], norm[keep]

    seed_ids = _metadata_seed_ids(meta_path)
    if seed_ids:
        seed_array = np.fromiter(seed_ids, dtype=np.int64)
        incident = np.isin(pre, seed_array) | np.isin(post, seed_array)
    else:
        incident = np.ones(pre.shape, dtype=bool)

    p = pre[incident]
    q = post[incident]
    c = count[incident]
    if p.size == 0:
        p, q, c = pre, post, count
    nodes = np.concatenate([p, q])
    weights = np.concatenate([c, c])
    unique, inverse = np.unique(nodes, return_inverse=True)
    scores = np.bincount(inverse, weights=weights)
    is_seed = np.fromiter((int(x) in seed_ids for x in unique), dtype=bool, count=len(unique))
    order = np.lexsort((unique, -scores, ~is_seed))
    selected = unique[order[: min(n_nodes, len(unique))]]

    if selected.size < n_nodes:
        all_nodes = np.concatenate([pre, post])
        all_weights = np.concatenate([count, count])
        global_unique, global_inverse = np.unique(all_nodes, return_inverse=True)
        global_scores = np.bincount(global_inverse, weights=all_weights)
        global_order = np.lexsort((global_unique, -global_scores))
        selected_set = set(map(int, selected))
        fill = [x for x in global_unique[global_order] if int(x) not in selected_set]
        selected = np.concatenate([selected, np.asarray(fill[: n_nodes - selected.size])])

    selected = np.sort(selected.astype(np.int64, copy=False))
    induced = np.isin(pre, selected) & np.isin(post, selected)
    ipre, ipost, iweight, icount = pre[induced], post[induced], norm[induced], count[induced]
    if ipre.size == 0:
        raise RuntimeError("selected BANC nodes have no induced edges")

    node_to_index = {int(node): i for i, node in enumerate(selected)}
    edge_src = np.fromiter((node_to_index[int(x)] for x in ipre), dtype=np.int64)
    edge_dst = np.fromiter((node_to_index[int(x)] for x in ipost), dtype=np.int64)
    edge_weight = iweight.astype(np.float32, copy=False)

    metadata = {
        "kind": "banc_connectome_structural_prior",
        "source": source_manifest,
        "selection": {
            "n_nodes_requested": n_nodes,
            "n_nodes": int(selected.size),
            "n_edges": int(edge_src.size),
            "min_synapses": min_synapses,
            "seed_rule": "BANC motor/descending/flight annotations + strongest measured partners",
            "weight": "BANC input-normalized synaptic weight (norm)",
            "synapse_count_sum": float(icount.sum()),
        },
    }
    graph = ConnectomeGraph(selected, edge_src, edge_dst, edge_weight, metadata)
    graph.validate()
    save_graph(graph, output)
    return graph


def save_graph(graph: ConnectomeGraph, path: Path) -> None:
    graph.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        node_ids=graph.node_ids.astype(np.int64),
        edge_src=graph.edge_src.astype(np.int64),
        edge_dst=graph.edge_dst.astype(np.int64),
        edge_weight=graph.edge_weight.astype(np.float32),
        metadata=np.asarray(json.dumps(graph.metadata, sort_keys=True)),
    )


def load_graph(path: Path) -> ConnectomeGraph:
    with np.load(Path(path), allow_pickle=False) as data:
        graph = ConnectomeGraph(
            node_ids=data["node_ids"].astype(np.int64),
            edge_src=data["edge_src"].astype(np.int64),
            edge_dst=data["edge_dst"].astype(np.int64),
            edge_weight=data["edge_weight"].astype(np.float32),
            metadata=json.loads(str(data["metadata"].item())),
        )
    graph.validate()
    return graph


def degree_preserving_shuffle(
    graph: ConnectomeGraph, *, seed: int = 0, swaps: int | None = None
) -> ConnectomeGraph:
    """Rewire directed edges while preserving every node's in/out edge degree."""
    graph.validate()
    rng = np.random.default_rng(seed)
    src = graph.edge_src.copy()
    dst = graph.edge_dst.copy()
    weight = graph.edge_weight.copy()
    target_swaps = swaps if swaps is not None else max(graph.n_edges * 5, 1)
    edges = set(zip(map(int, src), map(int, dst)))
    accepted = 0
    attempts = 0
    max_attempts = max(target_swaps * 20, 100)
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, graph.n_edges, size=2)
        if i == j:
            continue
        a, b = int(src[i]), int(dst[i])
        c, d = int(src[j]), int(dst[j])
        if a == d or c == b or b == d:
            continue
        e1, e2 = (a, d), (c, b)
        if e1 in edges or e2 in edges:
            continue
        edges.remove((a, b))
        edges.remove((c, d))
        dst[i], dst[j] = d, b
        edges.add(e1)
        edges.add(e2)
        accepted += 1

    metadata = dict(graph.metadata)
    metadata["control"] = {
        "kind": "degree_preserving_directed_shuffle",
        "seed": seed,
        "requested_swaps": target_swaps,
        "accepted_swaps": accepted,
        "attempts": attempts,
        "preserves": [
            "node_ids",
            "directed_in_degree",
            "directed_out_degree",
            "edge_weight_multiset",
        ],
    }
    shuffled = ConnectomeGraph(graph.node_ids.copy(), src, dst, weight, metadata)
    shuffled.validate()
    return shuffled
