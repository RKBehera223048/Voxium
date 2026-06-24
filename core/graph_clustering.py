"""
Voxium — Graph Clustering (Zero-GPU)
=======================================
Topology-based community detection using Leiden/Louvain algorithms.
Operates entirely on graph edge density — no embeddings, no GPU, no LLM.

DIRECTLY PORTED from graphify's cluster.py:
    - _partition(): Tries graspologic Leiden, falls back to NetworkX Louvain
    - cluster(): Oversized community splitting + low-cohesion re-splitting
    - cohesion_score(): Intra-community edge density ratio
    - score_all(): Cohesion for all communities

Adapted for conversational memory graphs with additional:
    - label_communities(): Auto-labels from highest-degree nodes
    - detect_bridge_nodes(): Cross-community bridges via betweenness centrality
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import sys
import logging
from typing import Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)


# =============================================================================
# Partition — Direct port from graphify/cluster.py lines 22-77
# =============================================================================

def _suppress_output():
    """Context manager to suppress stdout during library calls.

    graspologic's leiden() emits ANSI escape sequences that corrupt
    PowerShell 5.1's scroll buffer on Windows. Ported from graphify.
    """
    return contextlib.redirect_stdout(io.StringIO())


def _partition(G: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Run community detection. Returns {node_id: community_id}.

    Tries Leiden (graspologic) first — best quality.
    Falls back to Louvain (built into networkx) if graspologic is not installed.

    Ported from graphify/cluster.py _partition().
    """
    # Build a deterministic copy — sorted nodes and edges for reproducibility
    stable = nx.Graph()
    stable.add_nodes_from(sorted(G.nodes(), key=str))
    edge_rows = sorted(
        G.edges(data=True),
        key=lambda row: (
            str(row[0]),
            str(row[1]),
            json.dumps(row[2], sort_keys=True, ensure_ascii=False, default=str),
        ),
    )
    for src, tgt, attrs in edge_rows:
        stable.add_edge(src, tgt, **attrs)

    # Strategy 1: graspologic Leiden (best quality)
    try:
        from graspologic.partition import leiden

        lsig = inspect.signature(leiden).parameters
        kwargs: dict = {}
        if "random_seed" in lsig:
            kwargs["random_seed"] = 42
        if "trials" in lsig:
            kwargs["trials"] = 1
        if "resolution" in lsig:
            kwargs["resolution"] = resolution

        old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            with _suppress_output():
                result = leiden(stable, **kwargs)
        finally:
            sys.stderr = old_stderr

        logger.debug("Clustering: used graspologic Leiden")
        return result
    except ImportError:
        pass

    # Strategy 2: NetworkX Louvain (built-in fallback)
    kwargs = {"seed": 42, "threshold": 1e-4, "resolution": resolution}
    if "max_level" in inspect.signature(nx.community.louvain_communities).parameters:
        kwargs["max_level"] = 10

    communities = nx.community.louvain_communities(stable, **kwargs)
    logger.debug("Clustering: used NetworkX Louvain")
    return {node: cid for cid, nodes in enumerate(communities) for node in nodes}


# =============================================================================
# Cohesion — Direct port from graphify/cluster.py lines 209-221
# =============================================================================

def cohesion_score(G: nx.Graph, community_nodes: list[str]) -> float:
    """Ratio of actual intra-community edges to maximum possible.

    Ported verbatim from graphify/cluster.py cohesion_score().
    """
    n = len(community_nodes)
    if n <= 1:
        return 1.0
    subgraph = G.subgraph(community_nodes)
    actual = subgraph.number_of_edges()
    possible = n * (n - 1) / 2
    return actual / possible if possible > 0 else 0.0


def score_all(
    G: nx.Graph,
    communities: dict[int, list[str]],
) -> dict[int, float]:
    """Cohesion scores for all communities.

    Ported from graphify/cluster.py score_all().
    """
    return {cid: cohesion_score(G, nodes) for cid, nodes in communities.items()}


# =============================================================================
# Community Splitting — Ported from graphify/cluster.py lines 80-188
# =============================================================================

_MAX_COMMUNITY_FRACTION = 0.25   # communities > 25% of graph get split
_MIN_SPLIT_SIZE = 10             # only split if community has >= this many nodes
_COHESION_SPLIT_THRESHOLD = 0.05 # re-split communities with cohesion below this
_COHESION_SPLIT_MIN_SIZE = 50    # only cohesion-split if >= this many nodes


def _split_community(G: nx.Graph, nodes: list[str]) -> list[list[str]]:
    """Run a second partition pass on a community subgraph to split it.

    Ported from graphify/cluster.py _split_community().
    """
    subgraph = G.subgraph(nodes)
    if subgraph.number_of_edges() == 0:
        return [[n] for n in sorted(nodes)]
    try:
        sub_partition = _partition(subgraph)
        sub_communities: dict[int, list[str]] = {}
        for node, cid in sub_partition.items():
            sub_communities.setdefault(cid, []).append(node)
        if len(sub_communities) <= 1:
            return [sorted(nodes)]
        return [sorted(v) for v in sub_communities.values()]
    except Exception:
        return [sorted(nodes)]


# =============================================================================
# Main Clustering — Ported from graphify/cluster.py cluster()
# =============================================================================

def cluster_graph(
    G: nx.Graph,
    resolution: float = 1.0,
) -> dict[int, list[str]]:
    """Run Leiden/Louvain community detection. Returns {community_id: [node_ids]}.

    Community IDs are stable across runs: 0 = largest community after splitting.
    Oversized communities (>25% of graph nodes, min 10) are split by running
    a second pass on the subgraph.

    Ported from graphify/cluster.py cluster() with hub-exclusion logic
    removed (not needed for conversational memory graphs which don't have
    code-style super-hubs).
    """
    if G.number_of_nodes() == 0:
        return {}

    if G.is_directed():
        G = G.to_undirected()

    if G.number_of_edges() == 0:
        return {i: [n] for i, n in enumerate(sorted(G.nodes))}

    # Handle isolates separately (Leiden warns on them)
    isolates = [n for n in G.nodes() if G.degree(n) == 0]
    connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    connected = G.subgraph(connected_nodes)

    raw: dict[int, list[str]] = {}
    if connected.number_of_nodes() > 0:
        partition = _partition(connected, resolution=resolution)
        for node, cid in partition.items():
            raw.setdefault(cid, []).append(node)

    # Each isolate becomes its own single-node community
    next_cid = max(raw.keys(), default=-1) + 1
    for node in isolates:
        raw[next_cid] = [node]
        next_cid += 1

    # Split oversized communities
    max_size = max(_MIN_SPLIT_SIZE, int(G.number_of_nodes() * _MAX_COMMUNITY_FRACTION))
    final_communities: list[list[str]] = []
    for nodes in raw.values():
        if len(nodes) > max_size:
            final_communities.extend(_split_community(G, nodes))
        else:
            final_communities.append(nodes)

    # Second pass: re-split low-cohesion communities
    second_pass: list[list[str]] = []
    for nodes in final_communities:
        if (len(nodes) >= _COHESION_SPLIT_MIN_SIZE
                and cohesion_score(G, nodes) < _COHESION_SPLIT_THRESHOLD):
            splits = _split_community(G, nodes)
            second_pass.extend(splits if len(splits) > 1 else [nodes])
        else:
            second_pass.append(nodes)
    final_communities = second_pass

    # Re-index by size descending with deterministic tiebreak
    # (ported from graphify — ensures identical grouping always gets identical IDs)
    final_communities.sort(
        key=lambda nodes: (-len(nodes), tuple(sorted(map(str, nodes))))
    )
    return {i: sorted(nodes) for i, nodes in enumerate(final_communities)}


# =============================================================================
# Community Labeling — Conversational adaptation
# =============================================================================

def label_communities(
    G: nx.Graph,
    communities: dict[int, list[str]],
) -> dict[int, str]:
    """Auto-label each community by its highest-degree node's label.

    For conversational memory, the most-connected entity in a community
    is typically the central topic, making it a natural label.
    """
    labels: dict[int, str] = {}
    for cid, nodes in communities.items():
        if not nodes:
            labels[cid] = f"Topic {cid}"
            continue

        # Find the node with highest degree in this community
        best_node = max(nodes, key=lambda n: G.degree(n))
        label = G.nodes[best_node].get("label", best_node)
        labels[cid] = label

    return labels


# =============================================================================
# Bridge Node Detection — Ported from graphify/analyze.py
# =============================================================================

def detect_bridge_nodes(
    G: nx.Graph,
    communities: dict[int, list[str]],
    top_n: int = 5,
) -> list[dict]:
    """Find nodes that bridge multiple communities.

    Uses betweenness centrality to identify cross-community connectors.
    Ported from graphify's surprising_connections() pattern in analyze.py.
    """
    if G.number_of_edges() == 0 or not communities:
        return []

    # Build node → community map
    node_community = {n: cid for cid, nodes in communities.items() for n in nodes}

    # Compute betweenness centrality (approximate for large graphs)
    k = min(100, G.number_of_nodes()) if G.number_of_nodes() > 1000 else None
    betweenness = nx.betweenness_centrality(G, k=k, seed=42)

    # Find nodes that connect different communities
    bridges: list[dict] = []
    for node_id, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
        if score <= 0:
            continue

        node_cid = node_community.get(node_id)
        neighbor_comms = {
            node_community.get(n)
            for n in G.neighbors(node_id)
            if node_community.get(n) != node_cid
        }

        if not neighbor_comms:
            continue

        bridges.append({
            "id": node_id,
            "label": G.nodes[node_id].get("label", node_id),
            "betweenness": round(score, 4),
            "home_community": node_cid,
            "bridges_to": list(neighbor_comms - {None}),
            "degree": G.degree(node_id),
        })

        if len(bridges) >= top_n:
            break

    return bridges
