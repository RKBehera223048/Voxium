from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

# Assuming these will be available per Phase 3 plan
from memory.chromadb import VectorStore
from memory.graphify import MemoryGraph
from memory.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retrieval combining vector search with graph traversal.
    
    Flow:
        1. Vector search: find top-k semantically similar chunks.
        2. Entity lookup: find graph entities linked to those chunks via metadata.
        3. Graph traversal: BFS from those entities, collecting neighbors.
        4. Merge: combine chunk text + graph context into a ranked prompt.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        memory_graph: MemoryGraph,
        embedding_provider: EmbeddingProvider,
        vector_search_top_k: int = 5,
        graph_traversal_max_hops: int = 2,
        graph_traversal_max_nodes: int = 30,
    ):
        self.vector_store = vector_store
        self.memory_graph = memory_graph
        self.embedding_provider = embedding_provider
        self.vector_search_top_k = vector_search_top_k
        self.graph_traversal_max_hops = graph_traversal_max_hops
        self.graph_traversal_max_nodes = graph_traversal_max_nodes

    async def search(self, query: str, max_tokens: int = 800) -> str:
        """
        Executes the hybrid search and formats the results into a context string.
        """
        # 1. Embed query text
        loop = asyncio.get_running_loop()
        try:
            if hasattr(self.embedding_provider, 'embed_async'):
                query_embedding = await self.embedding_provider.embed_async(query)
            else:
                # If embed() is synchronous, run it in an executor
                query_embedding = await loop.run_in_executor(None, self.embedding_provider.embed, query)
        except Exception as e:
            logger.warning(f"HybridRetriever: query embedding failed: {e}")
            return ""

        # 2. Search ChromaDB
        vector_results: List[Dict[str, Any]] = []
        try:
            if hasattr(self.vector_store, 'search_async'):
                vector_results = await self.vector_store.search_async(
                    query_embedding, n_results=self.vector_search_top_k
                )
            else:
                vector_results = self.vector_store.search(
                    query_embedding, n_results=self.vector_search_top_k
                )
        except Exception as e:
            logger.warning(f"HybridRetriever: vector search failed: {e}")

        # 3. Extract linked entities from chunk metadata
        seed_entity_ids: Set[str] = set()
        for vr in vector_results:
            metadata = vr.get("metadata", {})
            entities_raw = metadata.get("entities", "[]")
            
            entities = []
            if isinstance(entities_raw, str):
                try:
                    entities = json.loads(entities_raw)
                except Exception:
                    pass
            elif isinstance(entities_raw, list):
                entities = entities_raw
                
            if isinstance(entities, list):
                for e in entities:
                    seed_entity_ids.add(str(e))
                    
        # Direct graph seed matching + BFS traversal under the graph lock
        # [M-2] Use the public lock-protected accessor instead of _graph
        query_lower = query.lower()
        query_words = set(query_lower.split())

        async with self.memory_graph.locked_graph() as graph:
            for node_id, data in graph.nodes(data=True):
                label = data.get("label", "").lower()
                if label and (label in query_lower or query_lower in label):
                    seed_entity_ids.add(node_id)
                elif label and (query_words & set(label.split())):
                    seed_entity_ids.add(node_id)

            # 4. BFS graph traversal from seeds
            visited: Set[str] = set()
            traversed_nodes: List[Tuple[str, dict, float]] = []
            traversed_edges: List[Tuple[str, str, dict]] = []

            # Score seeds to prioritize the BFS
            scored_seeds: List[Tuple[str, float]] = []
            for eid in seed_entity_ids:
                if not graph.has_node(eid):
                    continue
                    
                vscore = 0.0
                # Determine max score from vector results containing this entity
                for i, vr in enumerate(vector_results):
                    metadata = vr.get("metadata", {})
                    entities_raw = metadata.get("entities", "[]")
                    entities = []
                    if isinstance(entities_raw, str):
                        try:
                            entities = json.loads(entities_raw)
                        except Exception:
                            pass
                    elif isinstance(entities_raw, list):
                        entities = entities_raw
                        
                    if eid in entities:
                        # ChromaDB distances might be returned, but we rank by position if score is absent
                        score = vr.get("score", 1.0 / (i + 1))
                        vscore = max(vscore, score)

                degree_boost = min(graph.degree(eid) / 10.0, 2.0)
                scored_seeds.append((eid, vscore + degree_boost))

            scored_seeds.sort(key=lambda x: x[1], reverse=True)

            # Limit to top seeds to keep BFS bounded
            for seed_id, seed_score in scored_seeds[:10]:
                if seed_id in visited:
                    continue

                queue = [(seed_id, 0, seed_score)]
                while queue and len(visited) < self.graph_traversal_max_nodes:
                    node_id, depth, parent_score = queue.pop(0)
                    
                    if node_id in visited or depth > self.graph_traversal_max_hops:
                        continue
                    if not graph.has_node(node_id):
                        continue

                    visited.add(node_id)
                    data = graph.nodes[node_id]
                    decayed_score = parent_score * (0.8 ** depth)
                    traversed_nodes.append((node_id, data, decayed_score))

                    for neighbor in graph.neighbors(node_id):
                        if neighbor not in visited:
                            edge_data = graph.get_edge_data(node_id, neighbor)
                            if edge_data:
                                traversed_edges.append((node_id, neighbor, edge_data))
                            queue.append((neighbor, depth + 1, decayed_score))

            # 5. Merge + rank results (still under lock to read labels)
            traversed_nodes.sort(key=lambda x: x[2], reverse=True)

            # 6. Format context string
            context_parts: List[str] = []

            if vector_results:
                context_parts.append("Semantic Context:")
                for vr in vector_results:
                    text = vr.get("text", "")
                    if text:
                        context_parts.append(f"- {text}")

            if traversed_nodes:
                context_parts.append("\nGraph Entities:")
                for node_id, data, score in traversed_nodes[:15]:
                    label = data.get("label", node_id)
                    desc = data.get("description", "")
                    if desc:
                        context_parts.append(f"- {label}: {desc}")
                    else:
                        context_parts.append(f"- {label}")

            if traversed_edges:
                context_parts.append("\nRelationships:")
                unique_edges = set()
                for u, v, data in traversed_edges:
                    rel_type = data.get("relation", "connected to")
                    desc = data.get("description", "")
                    
                    # Retrieve readable labels if available
                    u_label = graph.nodes[u].get("label", u) if graph.has_node(u) else u
                    v_label = graph.nodes[v].get("label", v) if graph.has_node(v) else v
                    
                    edge_str = f"{u_label} --[{rel_type}]--> {v_label}"
                    if desc:
                        edge_str += f" ({desc})"
                    
                    if edge_str not in unique_edges:
                        unique_edges.add(edge_str)

                for edge_str in list(unique_edges)[:20]:
                    context_parts.append(f"- {edge_str}")

        # Truncate to max tokens (approx 4 chars per token)
        max_chars = max_tokens * 4
        full_context = "\n".join(context_parts)
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n... [Context truncated]"

        return full_context

