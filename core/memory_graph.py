"""
Voxium — Graph-RAG Memory Engine
===================================
Persistent, token-efficient memory layer using NetworkX.
Maintains a dynamic graph of entities and relationships derived
from the user's ongoing conversations.

Architectural patterns ported from graphify:
    - Graph construction: build.py's build_from_json() → node_link_data
    - Persistence: export.py's to_json() → JSON serialization
    - Clustering: cluster.py's Leiden/Louvain community detection
    - Analysis: analyze.py's god_nodes() and surprise detection

The graph is queried before sending context to the LLM, providing
a compact, relevant memory snapshot instead of raw conversation history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx
from networkx.readwrite import json_graph

from core.entity_extractor import extract_all, Entity, Relationship
from core.graph_clustering import (
    cluster_graph,
    score_all,
    label_communities,
    detect_bridge_nodes,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MemoryGraphConfig:
    """Configuration for the memory graph engine."""
    graph_path: str = ""
    context_max_tokens: int = 500
    auto_cluster: bool = True
    cluster_threshold: int = 10  # Recluster after this many new nodes
    persist_debounce_seconds: float = 2.0
    recency_decay_halflife: float = 3600.0  # 1 hour in seconds

    @classmethod
    def from_env(cls) -> MemoryGraphConfig:
        return cls(
            graph_path=os.getenv("GRAPH_DB_PATH", "data/db/graph.json"),
            context_max_tokens=int(os.getenv("GRAPH_CONTEXT_MAX_TOKENS", "500")),
            auto_cluster=os.getenv("GRAPH_AUTO_CLUSTER", "true").lower() == "true",
        )


# =============================================================================
# Memory Graph Engine
# =============================================================================

class MemoryGraph:
    """
    Persistent graph-based memory for conversational context.

    Maintains a NetworkX graph of entities and relationships extracted
    from conversation turns. Provides efficient context retrieval via
    BFS traversal from query-matched seed nodes.

    Thread-safe via asyncio.Lock.
    """

    def __init__(self, config: Optional[MemoryGraphConfig] = None):
        self._config = config or MemoryGraphConfig.from_env()
        self._graph = nx.Graph()
        self._communities: dict[int, list[str]] = {}
        self._community_labels: dict[int, str] = {}
        self._cohesion_scores: dict[int, float] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._nodes_since_last_cluster = 0
        self._persist_task: Optional[asyncio.Task] = None
        self._last_persist_time = 0.0

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def community_count(self) -> int:
        return len(self._communities)

    # ── Ingestion ───────────────────────────────────────────────────────

    async def ingest(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Extract entities and relationships from text and add to the graph.

        Args:
            text: Conversational text to process.
            metadata: Optional metadata (role, timestamp, etc.)

        Returns:
            Number of new entities added.
        """
        if not text or len(text.strip()) < 5:
            return 0

        entities, relationships = extract_all(text)
        if not entities:
            return 0

        now = time.time()
        new_count = 0
        meta = metadata or {}

        async with self._lock:
            # Add/update entity nodes
            for entity in entities:
                if self._graph.has_node(entity.id):
                    # Update existing node
                    node_data = self._graph.nodes[entity.id]
                    node_data["mention_count"] = node_data.get("mention_count", 0) + 1
                    node_data["last_seen"] = now
                    # Append source turn reference
                    sources = node_data.get("source_turns", [])
                    sources.append(now)
                    # Keep only last 20 turn references
                    node_data["source_turns"] = sources[-20:]
                else:
                    # Add new node — schema mirrors graphify's node format
                    self._graph.add_node(
                        entity.id,
                        label=entity.label,
                        type=entity.type,
                        source_file="conversation",  # graphify compat
                        confidence=entity.confidence,
                        mention_count=1,
                        first_seen=now,
                        last_seen=now,
                        source_turns=[now],
                    )
                    new_count += 1

            # Add/update relationship edges
            for rel in relationships:
                if not self._graph.has_node(rel.source_id):
                    continue
                if not self._graph.has_node(rel.target_id):
                    continue

                if self._graph.has_edge(rel.source_id, rel.target_id):
                    # Update existing edge weight
                    edge_data = self._graph.edges[rel.source_id, rel.target_id]
                    edge_data["weight"] = edge_data.get("weight", 1.0) + rel.weight
                    edge_data["last_seen"] = now
                else:
                    # Add new edge — schema mirrors graphify's edge format
                    self._graph.add_edge(
                        rel.source_id,
                        rel.target_id,
                        relation=rel.relation,
                        confidence=rel.confidence,
                        weight=rel.weight,
                        first_seen=now,
                        last_seen=now,
                        source_file="conversation",  # graphify compat
                    )

            self._dirty = True
            self._nodes_since_last_cluster += new_count

        # Auto-cluster if threshold reached
        if (self._config.auto_cluster
                and self._nodes_since_last_cluster >= self._config.cluster_threshold):
            await self.recluster()

        # Schedule debounced persist
        await self._schedule_persist()

        if new_count > 0:
            logger.debug(
                "Graph ingested: +%d nodes, %d total nodes, %d edges",
                new_count, self.node_count, self.edge_count,
            )

        return new_count

    # ── Context Retrieval ───────────────────────────────────────────────

    async def query_context(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Retrieve relevant memory context for a query.

        Uses BFS from seed nodes (matched by the query) and returns
        nodes/edges ranked by relevance score:
            score = mention_count × recency_decay × edge_weight

        This mirrors graphify's traversal approach but optimized for
        conversational context injection into LLM prompts.

        Args:
            query: The query text to find relevant context for.
            max_tokens: Maximum approximate token count for output.

        Returns:
            A compact string summarizing the relevant memory subgraph.
        """
        if self.node_count == 0:
            return ""

        max_tok = max_tokens or self._config.context_max_tokens
        query_lower = query.lower()
        query_words = set(query_lower.split())

        async with self._lock:
            # Step 1: Find seed nodes matching the query
            seed_nodes: list[tuple[str, float]] = []

            for node_id, data in self._graph.nodes(data=True):
                label = data.get("label", "").lower()
                score = 0.0

                # Exact label match
                if label in query_lower:
                    score += 10.0
                # Word overlap
                label_words = set(label.split())
                overlap = query_words & label_words
                if overlap:
                    score += len(overlap) * 3.0
                # Partial substring match
                elif any(w in label for w in query_words if len(w) >= 3):
                    score += 1.0

                if score > 0:
                    # Apply recency decay
                    recency = self._recency_weight(data.get("last_seen", 0))
                    mention_boost = min(data.get("mention_count", 1) / 5.0, 3.0)
                    score *= recency * mention_boost
                    seed_nodes.append((node_id, score))

            if not seed_nodes:
                # No direct match — use top god nodes as fallback
                return self._build_god_nodes_context(max_tok)

            # Sort seeds by score descending
            seed_nodes.sort(key=lambda x: x[1], reverse=True)

            # Step 2: BFS from top seeds to collect relevant subgraph
            visited: set[str] = set()
            context_nodes: list[tuple[str, dict, float]] = []
            context_edges: list[tuple[str, str, dict]] = []

            for seed_id, seed_score in seed_nodes[:5]:  # Top 5 seeds
                if seed_id in visited:
                    continue

                # BFS with depth limit
                queue = [(seed_id, 0, seed_score)]
                while queue:
                    node_id, depth, parent_score = queue.pop(0)
                    if node_id in visited or depth > 2:
                        continue
                    visited.add(node_id)

                    data = dict(self._graph.nodes[node_id])
                    node_score = parent_score * (0.5 ** depth)
                    context_nodes.append((node_id, data, node_score))

                    # Collect edges to this node
                    for neighbor in self._graph.neighbors(node_id):
                        edge_data = dict(self._graph.edges[node_id, neighbor])
                        context_edges.append((node_id, neighbor, edge_data))

                        if neighbor not in visited and depth < 2:
                            edge_weight = edge_data.get("weight", 1.0)
                            queue.append((neighbor, depth + 1, node_score * edge_weight))

            # Step 3: Build compact text output
            return self._format_context(context_nodes, context_edges, max_tok)

    def _recency_weight(self, timestamp: float) -> float:
        """Exponential decay based on time since last mention."""
        if timestamp <= 0:
            return 0.1
        age = time.time() - timestamp
        halflife = self._config.recency_decay_halflife
        return max(0.1, 2.0 ** (-age / halflife))

    def _format_context(
        self,
        nodes: list[tuple[str, dict, float]],
        edges: list[tuple[str, str, dict]],
        max_tokens: int,
    ) -> str:
        """Format subgraph as a compact text string for LLM context."""
        if not nodes:
            return ""

        # Sort by relevance score
        nodes.sort(key=lambda x: x[2], reverse=True)

        lines: list[str] = ["[Memory Graph Context]"]
        char_budget = max_tokens * 4  # ~4 chars per token approximation
        used = len(lines[0])

        # Key entities
        lines.append("Key entities:")
        used += 15
        for node_id, data, score in nodes:
            label = data.get("label", node_id)
            etype = data.get("type", "unknown")
            mentions = data.get("mention_count", 1)
            line = f"  - {label} ({etype}, mentioned {mentions}x)"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)

        # Key relationships
        seen_edges: set[tuple[str, str]] = set()
        rel_lines: list[str] = []
        for src_id, tgt_id, edata in edges:
            pair = (min(src_id, tgt_id), max(src_id, tgt_id))
            if pair in seen_edges:
                continue
            seen_edges.add(pair)

            src_label = self._graph.nodes[src_id].get("label", src_id) if self._graph.has_node(src_id) else src_id
            tgt_label = self._graph.nodes[tgt_id].get("label", tgt_id) if self._graph.has_node(tgt_id) else tgt_id
            relation = edata.get("relation", "related_to")
            line = f"  - {src_label} → {relation} → {tgt_label}"
            if used + len(line) > char_budget:
                break
            rel_lines.append(line)
            used += len(line)

        if rel_lines:
            lines.append("Relationships:")
            lines.extend(rel_lines)

        # Community context
        if self._communities and self._community_labels:
            node_ids = {n[0] for n in nodes}
            relevant_comms = set()
            node_to_comm = {
                n: cid for cid, members in self._communities.items()
                for n in members
            }
            for nid in node_ids:
                if nid in node_to_comm:
                    relevant_comms.add(node_to_comm[nid])

            if relevant_comms:
                comm_strs = [
                    self._community_labels.get(cid, f"Topic {cid}")
                    for cid in relevant_comms
                ]
                topic_line = f"Related topics: {', '.join(comm_strs)}"
                if used + len(topic_line) <= char_budget:
                    lines.append(topic_line)

        return "\n".join(lines)

    def _build_god_nodes_context(self, max_tokens: int) -> str:
        """Fallback context when no seed nodes match — return top entities."""
        if self.node_count == 0:
            return ""

        degree = dict(self._graph.degree())
        top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]

        if not top_nodes:
            return ""

        lines = ["[Memory Graph Context — Recent Topics]"]
        char_budget = max_tokens * 4
        used = len(lines[0])

        for node_id, deg in top_nodes:
            data = self._graph.nodes[node_id]
            label = data.get("label", node_id)
            mentions = data.get("mention_count", 1)
            line = f"  - {label} (mentioned {mentions}x, {deg} connections)"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)

        return "\n".join(lines)

    # ── Clustering ──────────────────────────────────────────────────────

    async def recluster(self) -> None:
        """Run community detection on the current graph.

        Uses graphify's Leiden/Louvain strategy from graph_clustering.py.
        """
        async with self._lock:
            if self._graph.number_of_nodes() < 3:
                return

            try:
                self._communities = cluster_graph(self._graph)
                self._cohesion_scores = score_all(self._graph, self._communities)
                self._community_labels = label_communities(
                    self._graph, self._communities,
                )
                self._nodes_since_last_cluster = 0
                self._dirty = True

                logger.info(
                    "Graph reclustered: %d communities from %d nodes",
                    len(self._communities),
                    self._graph.number_of_nodes(),
                )
            except Exception as e:
                logger.error("Clustering failed: %s", e)

    # ── Persistence ─────────────────────────────────────────────────────

    async def load(self) -> None:
        """Load graph from disk on startup."""
        graph_path = Path(self._config.graph_path)
        if not graph_path.exists():
            logger.info("No existing graph at %s — starting fresh", graph_path)
            return

        async with self._lock:
            try:
                data = json.loads(graph_path.read_text(encoding="utf-8"))

                # Reconstruct graph from node_link_data format
                # Handle both "links" (networkx default) and "edges" keys
                if "links" in data or "edges" in data:
                    try:
                        self._graph = json_graph.node_link_graph(
                            data, edges="links" if "links" in data else "edges",
                        )
                    except TypeError:
                        # Older networkx versions
                        self._graph = json_graph.node_link_graph(data)

                # Restore communities
                stored_communities = data.get("communities", {})
                if stored_communities:
                    self._communities = {
                        int(k): v for k, v in stored_communities.items()
                    }
                    self._cohesion_scores = score_all(
                        self._graph, self._communities,
                    )
                    self._community_labels = label_communities(
                        self._graph, self._communities,
                    )

                logger.info(
                    "Graph loaded: %d nodes, %d edges, %d communities from %s",
                    self._graph.number_of_nodes(),
                    self._graph.number_of_edges(),
                    len(self._communities),
                    graph_path,
                )
            except Exception as e:
                logger.error("Failed to load graph from %s: %s", graph_path, e)
                self._graph = nx.Graph()

    async def persist(self) -> None:
        """Save graph to disk."""
        if not self._dirty:
            return

        graph_path = Path(self._config.graph_path)

        async with self._lock:
            try:
                # Ensure directory exists
                graph_path.parent.mkdir(parents=True, exist_ok=True)

                # Serialize using networkx's node_link_data (matches graphify)
                try:
                    data = json_graph.node_link_data(self._graph, edges="links")
                except TypeError:
                    data = json_graph.node_link_data(self._graph)

                # Add community data
                data["communities"] = {
                    str(k): v for k, v in self._communities.items()
                }
                data["metadata"] = {
                    "node_count": self._graph.number_of_nodes(),
                    "edge_count": self._graph.number_of_edges(),
                    "community_count": len(self._communities),
                    "last_updated": time.time(),
                }

                graph_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                self._dirty = False
                self._last_persist_time = time.time()

                logger.debug("Graph persisted to %s", graph_path)
            except Exception as e:
                logger.error("Failed to persist graph to %s: %s", graph_path, e)

    async def _schedule_persist(self) -> None:
        """Schedule a debounced persist to avoid excessive disk writes."""
        if self._persist_task and not self._persist_task.done():
            # Already scheduled
            return

        async def _debounced_persist():
            await asyncio.sleep(self._config.persist_debounce_seconds)
            await self.persist()

        try:
            self._persist_task = asyncio.create_task(_debounced_persist())
        except RuntimeError:
            # No running event loop — persist synchronously
            await self.persist()

    # ── Query API ───────────────────────────────────────────────────────

    async def get_graph_data(self) -> dict:
        """Return full graph as node_link_data for API serialization.

        Matches graphify's to_json() output format.
        """
        async with self._lock:
            try:
                data = json_graph.node_link_data(self._graph, edges="links")
            except TypeError:
                data = json_graph.node_link_data(self._graph)

            # Annotate nodes with community assignments
            node_to_comm = {
                n: cid for cid, members in self._communities.items()
                for n in members
            }
            for node in data.get("nodes", []):
                node_id = node.get("id")
                cid = node_to_comm.get(node_id)
                node["community"] = cid
                if cid is not None:
                    node["community_name"] = self._community_labels.get(
                        cid, f"Topic {cid}",
                    )

            data["communities"] = {
                str(k): v for k, v in self._communities.items()
            }
            data["metadata"] = {
                "node_count": self._graph.number_of_nodes(),
                "edge_count": self._graph.number_of_edges(),
                "community_count": len(self._communities),
            }
            return data

    async def get_clusters(self) -> dict:
        """Return community assignments, labels, and cohesion scores."""
        async with self._lock:
            return {
                "communities": {
                    str(k): v for k, v in self._communities.items()
                },
                "labels": {
                    str(k): v for k, v in self._community_labels.items()
                },
                "cohesion": {
                    str(k): round(v, 4)
                    for k, v in self._cohesion_scores.items()
                },
                "bridge_nodes": detect_bridge_nodes(
                    self._graph, self._communities,
                ),
            }

    async def search_nodes(self, query: str, limit: int = 20) -> list[dict]:
        """Fuzzy search nodes by label."""
        if not query:
            return []

        query_lower = query.lower()
        results: list[tuple[float, dict]] = []

        async with self._lock:
            for node_id, data in self._graph.nodes(data=True):
                label = data.get("label", "").lower()
                score = 0.0

                if query_lower == label:
                    score = 10.0
                elif query_lower in label:
                    score = 5.0
                elif label in query_lower:
                    score = 3.0
                else:
                    # Word overlap
                    q_words = set(query_lower.split())
                    l_words = set(label.split())
                    overlap = q_words & l_words
                    if overlap:
                        score = len(overlap) * 2.0

                if score > 0:
                    results.append((score, {
                        "id": node_id,
                        "label": data.get("label", node_id),
                        "type": data.get("type", "unknown"),
                        "mention_count": data.get("mention_count", 0),
                        "degree": self._graph.degree(node_id),
                        "score": round(score, 2),
                    }))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]

    async def get_node_detail(self, node_id: str) -> Optional[dict]:
        """Get full details for a single node + its neighbors."""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return None

            data = dict(self._graph.nodes[node_id])
            data["id"] = node_id
            data["degree"] = self._graph.degree(node_id)

            # Get neighbors with edge info
            neighbors = []
            for neighbor_id in self._graph.neighbors(node_id):
                edge_data = dict(self._graph.edges[node_id, neighbor_id])
                neighbor_data = dict(self._graph.nodes[neighbor_id])
                neighbors.append({
                    "id": neighbor_id,
                    "label": neighbor_data.get("label", neighbor_id),
                    "type": neighbor_data.get("type", "unknown"),
                    "relation": edge_data.get("relation", "related"),
                    "weight": edge_data.get("weight", 1.0),
                })

            data["neighbors"] = neighbors

            # Community info
            node_to_comm = {
                n: cid for cid, members in self._communities.items()
                for n in members
            }
            cid = node_to_comm.get(node_id)
            if cid is not None:
                data["community"] = cid
                data["community_name"] = self._community_labels.get(
                    cid, f"Topic {cid}",
                )

            return data

    async def get_stats(self) -> dict:
        """Return graph statistics."""
        async with self._lock:
            return {
                "node_count": self._graph.number_of_nodes(),
                "edge_count": self._graph.number_of_edges(),
                "community_count": len(self._communities),
                "density": (
                    nx.density(self._graph)
                    if self._graph.number_of_nodes() > 1
                    else 0.0
                ),
                "connected_components": (
                    nx.number_connected_components(self._graph)
                    if self._graph.number_of_nodes() > 0
                    else 0
                ),
            }
