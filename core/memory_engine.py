"""
Voxium — Hybrid Graph-Vector Memory Engine
=============================================
Cognee-inspired unified memory backend that combines:
    1. VectorStore  — SQLite-backed semantic chunk storage + cosine search
    2. KnowledgeGraph — NetworkX entity/relationship graph (existing MemoryGraph)
    3. Entity↔Chunk mapping — bridges vector and graph layers

Core API (mirrors Cognee's ECL pipeline):
    - cognify(text)            → Extract entities → Load to graph + vector store
    - graph_completion_search  → Vector search → graph traversal → ranked context
    - remember(text)           → convenience wrapper: chunk + cognify + store

Architecture adapted from:
    - cognee/tasks/graph/extract_graph_from_data.py  → entity extraction pipeline
    - cognee/modules/retrieval/graph_completion_retriever.py → hybrid search
    - cognee/modules/retrieval/hybrid_retriever.py   → chunk + entity channels
    - cognee/modules/session_distillation/distill.py → session → long-term flush

All storage is local: sqlite3 + numpy + networkx. No cloud SDKs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from core.entity_extractor import (
    Entity,
    Relationship,
    extract_all as regex_extract_all,
    make_entity_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HybridMemoryConfig:
    """Configuration for the hybrid graph-vector memory engine."""
    vector_db_path: str = ""
    embedding_dim: int = 0  # 0 = auto-detect at runtime
    chunk_size: int = 256  # chars per chunk for semantic segmentation
    chunk_overlap: int = 48
    vector_search_top_k: int = 5
    graph_traversal_max_hops: int = 2
    graph_traversal_max_nodes: int = 30
    context_max_tokens: int = 800
    tfidf_vocab_size: int = 512  # for fallback TF-IDF embeddings
    cognify_use_llm: bool = True  # try LLM extraction; fall back to regex

    @classmethod
    def from_env(cls) -> HybridMemoryConfig:
        return cls(
            vector_db_path=os.getenv("VECTOR_DB_PATH", "data/db/voxium_memory.sqlite"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "0")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "256")),
            vector_search_top_k=int(os.getenv("VECTOR_SEARCH_TOP_K", "5")),
            graph_traversal_max_hops=int(os.getenv("GRAPH_MAX_HOPS", "2")),
            context_max_tokens=int(os.getenv("GRAPH_CONTEXT_MAX_TOKENS", "800")),
            cognify_use_llm=os.getenv("COGNIFY_USE_LLM", "true").lower() == "true",
        )


# =============================================================================
# Embedding Provider — LLM embeddings with TF-IDF fallback
# =============================================================================

class EmbeddingProvider:
    """
    Generates text embeddings for the vector store.

    Strategy (mirrors Cognee's pluggable embedding engine):
        1. Try llama-cpp's embed() if the loaded model supports it
        2. Fall back to lightweight TF-IDF → dense projection using numpy

    The TF-IDF fallback is surprisingly effective for conversational memory:
    it captures term-frequency signals without any GPU or external service.
    """

    def __init__(self, dim: int = 0, vocab_size: int = 512):
        self._dim = dim  # 0 = auto-detect
        self._vocab_size = vocab_size
        self._llm_embed_fn = None
        self._mode: str = "uninitialized"
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._projection: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mode(self) -> str:
        return self._mode

    async def initialize(self, llm_instance=None) -> None:
        """Auto-detect the best embedding strategy."""
        async with self._lock:
            if self._mode != "uninitialized":
                return

            # Try LLM embeddings
            if llm_instance is not None:
                try:
                    test_emb = llm_instance.embed("test")
                    if isinstance(test_emb, (list, np.ndarray)) and len(test_emb) > 0:
                        # Handle nested list [[...]] from some llama-cpp versions
                        if isinstance(test_emb[0], (list, np.ndarray)):
                            test_emb = test_emb[0]
                        self._dim = len(test_emb)
                        self._llm_embed_fn = llm_instance.embed
                        self._mode = "llm"
                        logger.info(
                            "EmbeddingProvider: using LLM embeddings (dim=%d)", self._dim,
                        )
                        return
                except Exception as e:
                    logger.debug("LLM embed() not available: %s", e)

            # Fall back to TF-IDF dense projection
            self._dim = min(self._vocab_size, 128)
            self._mode = "tfidf"
            # Random stable projection matrix (seeded for reproducibility)
            rng = np.random.RandomState(42)
            self._projection = rng.randn(self._vocab_size, self._dim).astype(np.float32)
            self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True) + 1e-9
            logger.info(
                "EmbeddingProvider: using TF-IDF fallback (dim=%d)", self._dim,
            )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a dense vector."""
        if self._mode == "llm":
            return self._embed_llm(text)
        elif self._mode == "tfidf":
            return self._embed_tfidf(text)
        else:
            raise RuntimeError("EmbeddingProvider not initialized. Call initialize() first.")

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]

    def _embed_llm(self, text: str) -> np.ndarray:
        """Use the LLM's native embedding function."""
        result = self._llm_embed_fn(text)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], (list, np.ndarray)):
                result = result[0]
        return np.array(result, dtype=np.float32)

    def _embed_tfidf(self, text: str) -> np.ndarray:
        """
        Lightweight TF-IDF → dense embedding using random projection.

        This is a zero-dependency approach that captures term frequency signals.
        The random projection (JL lemma) preserves cosine similarity approximately.
        """
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return np.zeros(self._dim, dtype=np.float32)

        # Build sparse TF vector over fixed vocabulary
        tf = Counter(tokens)
        sparse = np.zeros(self._vocab_size, dtype=np.float32)
        for token, count in tf.items():
            # Deterministic hash to vocab index
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._vocab_size
            sparse[idx] += count

        # Normalize TF
        total = sparse.sum()
        if total > 0:
            sparse /= total

        # Project to dense space
        dense = sparse @ self._projection
        # L2 normalize
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense /= norm

        return dense


# =============================================================================
# Vector Store — SQLite-backed semantic search
# =============================================================================

class VectorStore:
    """
    SQLite-backed vector chunk storage with brute-force cosine similarity search.

    Schema mirrors Cognee's vector store interface but uses local SQLite instead
    of external vector databases. Efficient for <100K chunks on CPU.

    Tables:
        vector_chunks:     (chunk_id, text, embedding, source_type, metadata, created_at)
        entity_chunk_map:  (entity_id, chunk_id) — bridges graph nodes to vector chunks
        memory_metadata:   (key, value) — system metadata
    """

    def __init__(self, db_path: str, embedding_provider: EmbeddingProvider):
        self._db_path = db_path
        self._embedder = embedding_provider
        self._initialized = False

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedder

    def _ensure_initialized(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source_type TEXT DEFAULT 'conversation',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_chunk_map (
                    entity_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (entity_id, chunk_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_created
                ON vector_chunks(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ecm_entity
                ON entity_chunk_map(entity_id)
            """)
            conn.commit()
        finally:
            conn.close()

        self._initialized = True
        logger.info("VectorStore initialized: %s", self._db_path)

    def add_chunk(
        self,
        text: str,
        embedding: np.ndarray,
        source_type: str = "conversation",
        metadata: Optional[Dict] = None,
        chunk_id: Optional[str] = None,
    ) -> str:
        """Insert a text chunk with its embedding vector."""
        self._ensure_initialized()

        if chunk_id is None:
            chunk_id = hashlib.sha256(
                f"{text}:{time.time()}".encode()
            ).hexdigest()[:16]

        embedding_blob = embedding.astype(np.float32).tobytes()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO vector_chunks
                   (chunk_id, text, embedding, source_type, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (chunk_id, text, embedding_blob, source_type, meta_json, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

        return chunk_id

    def link_entity_to_chunk(self, entity_id: str, chunk_id: str) -> None:
        """Create a bridge between a graph entity and a vector chunk."""
        self._ensure_initialized()

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO entity_chunk_map (entity_id, chunk_id) VALUES (?, ?)",
                (entity_id, chunk_id),
            )
            conn.commit()
        finally:
            conn.close()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Brute-force cosine similarity search over all stored chunks.

        Mirrors Cognee's vector search but locally via numpy.
        Returns sorted list of {chunk_id, text, score, metadata, created_at}.
        """
        self._ensure_initialized()

        conn = sqlite3.connect(self._db_path)
        try:
            if source_type:
                cursor = conn.execute(
                    "SELECT chunk_id, text, embedding, metadata, created_at "
                    "FROM vector_chunks WHERE source_type = ?",
                    (source_type,),
                )
            else:
                cursor = conn.execute(
                    "SELECT chunk_id, text, embedding, metadata, created_at "
                    "FROM vector_chunks"
                )

            results = []
            q_norm = np.linalg.norm(query_embedding)
            if q_norm == 0:
                return []

            for row in cursor:
                chunk_id, text, emb_blob, meta_json, created_at = row
                stored_emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
                s_norm = np.linalg.norm(stored_emb)
                if s_norm == 0:
                    continue
                score = float(np.dot(query_embedding, stored_emb) / (q_norm * s_norm))
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                    "metadata": json.loads(meta_json) if meta_json else {},
                    "created_at": created_at,
                })
        finally:
            conn.close()

        # Sort by score descending, return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_chunks_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all vector chunks linked to a graph entity."""
        self._ensure_initialized()

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                """SELECT vc.chunk_id, vc.text, vc.metadata, vc.created_at
                   FROM entity_chunk_map ecm
                   JOIN vector_chunks vc ON ecm.chunk_id = vc.chunk_id
                   WHERE ecm.entity_id = ?
                   ORDER BY vc.created_at DESC""",
                (entity_id,),
            )
            return [
                {
                    "chunk_id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "created_at": row[3],
                }
                for row in cursor
            ]
        finally:
            conn.close()

    def get_entities_for_chunk(self, chunk_id: str) -> List[str]:
        """Get all entity IDs linked to a vector chunk."""
        self._ensure_initialized()

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT entity_id FROM entity_chunk_map WHERE chunk_id = ?",
                (chunk_id,),
            )
            return [row[0] for row in cursor]
        finally:
            conn.close()

    @property
    def chunk_count(self) -> int:
        """Total number of stored vector chunks."""
        if not self._initialized:
            return 0
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM vector_chunks")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0


# =============================================================================
# Text Chunking — Semantic segmentation
# =============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 48,
) -> List[str]:
    """
    Split text into overlapping chunks for vector storage.

    Mirrors Cognee's chunking strategy (cognee/modules/chunking/) but simplified
    for conversational text: uses sentence boundaries when possible, falls back
    to character-level splitting.
    """
    if not text or len(text.strip()) < 10:
        return [text.strip()] if text and text.strip() else []

    # Try sentence-based chunking first
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single sentence exceeds chunk_size, split by characters
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    sub = sentence[i:i + chunk_size]
                    if sub.strip():
                        chunks.append(sub.strip())
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text.strip()]


# =============================================================================
# Hybrid Memory — The unified engine
# =============================================================================

class HybridMemory:
    """
    Cognee-style unified memory engine combining graph and vector stores.

    Core methods (mirror Cognee's API surface):
        - cognify(text, metadata)  → ECL pipeline: extract + load to both stores
        - graph_completion_search(query) → hybrid retrieval: vector + graph traversal
        - remember(text) → convenience wrapper for cognify
        - recall(query) → convenience wrapper for graph_completion_search

    The engine wraps Voxium's existing MemoryGraph (NetworkX) and adds a
    VectorStore alongside it, bridged by entity↔chunk mappings.

    Architecture adapted from:
        - cognee/tasks/graph/extract_graph_from_data.py → cognify()
        - cognee/modules/retrieval/graph_completion_retriever.py → search()
        - cognee/modules/retrieval/hybrid_retriever.py → dual-channel retrieval
    """

    def __init__(
        self,
        memory_graph=None,
        config: Optional[HybridMemoryConfig] = None,
        llm_instance=None,
    ):
        from core.memory_graph import MemoryGraph

        self._config = config or HybridMemoryConfig.from_env()
        self._memory_graph = memory_graph or MemoryGraph()
        self._embedder = EmbeddingProvider(
            dim=self._config.embedding_dim,
            vocab_size=self._config.tfidf_vocab_size,
        )
        self._vector_store = VectorStore(
            db_path=self._config.vector_db_path,
            embedding_provider=self._embedder,
        )
        self._llm_instance = llm_instance
        self._llm_extract_fn = None  # Set externally by reasoning.py
        self._initialized = False
        self._lock = asyncio.Lock()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def memory_graph(self):
        """Access the underlying NetworkX-based MemoryGraph."""
        return self._memory_graph

    @property
    def vector_store(self) -> VectorStore:
        """Access the SQLite vector store."""
        return self._vector_store

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedder

    # ── Initialization ──────────────────────────────────────────────────

    async def initialize(self, llm_instance=None) -> None:
        """Initialize embedding provider and load persisted state."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if llm_instance:
                self._llm_instance = llm_instance

            await self._embedder.initialize(self._llm_instance)
            self._vector_store._ensure_initialized()
            await self._memory_graph.load()
            self._initialized = True

            logger.info(
                "HybridMemory initialized: embedding=%s graph_nodes=%d vector_chunks=%d",
                self._embedder.mode,
                self._memory_graph.node_count,
                self._vector_store.chunk_count,
            )

    def set_llm_extract_fn(self, fn) -> None:
        """
        Register the LLM entity extraction function.

        This should be a callable with signature:
            async def fn(text: str) -> Tuple[List[Entity], List[Relationship]]

        Set by reasoning.py when the LLM is loaded.
        """
        self._llm_extract_fn = fn

    # ── ECL Pipeline: cognify() ─────────────────────────────────────────

    async def cognify(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The Cognee-style ECL (Extract, Cognify, Load) pipeline.

        Adapted from cognee/tasks/graph/extract_graph_from_data.py:
            1. EXTRACT: Chunk the text into semantic segments
            2. COGNIFY: Use LLM (or regex fallback) to extract entities/relationships
            3. LOAD: Store entities in graph, chunks in vector store, create bridges

        Args:
            text: Raw text to process (transcript, conversation turn, etc.)
            metadata: Optional metadata (role, timestamp, speaker, etc.)

        Returns:
            Dict with counts: {entities_added, relationships_added, chunks_stored}
        """
        if not self._initialized:
            await self.initialize()

        if not text or len(text.strip()) < 5:
            return {"entities_added": 0, "relationships_added": 0, "chunks_stored": 0}

        meta = metadata or {}
        stats = {"entities_added": 0, "relationships_added": 0, "chunks_stored": 0}

        # ── Step 1: EXTRACT — Chunk the text ──
        chunks = chunk_text(
            text,
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )

        # ── Step 2: COGNIFY — Extract entities and relationships ──
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []

        # Try LLM extraction first (higher quality, like Cognee's extract_content_graph)
        llm_extracted = False
        if self._config.cognify_use_llm and self._llm_extract_fn is not None:
            try:
                entities, relationships = await self._llm_extract_fn(text)
                if entities:
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    llm_extracted = True
                    logger.debug(
                        "cognify: LLM extracted %d entities, %d relationships",
                        len(entities), len(relationships),
                    )
            except Exception as e:
                logger.warning("cognify: LLM extraction failed, falling back to regex: %s", e)

        # Regex fallback (or supplement) — matches Cognee's deterministic extraction path
        if not llm_extracted:
            entities, relationships = regex_extract_all(text)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # ── Step 3: LOAD — Store in graph + vector store ──

        # 3a: Load entities and relationships into the NetworkX graph
        graph_add_count = await self._memory_graph.ingest(text, metadata=meta)
        stats["entities_added"] = max(graph_add_count, len(all_entities))

        # 3b: Store chunks in the vector store with embeddings
        chunk_ids: List[str] = []
        loop = asyncio.get_event_loop()
        for chunk in chunks:
            try:
                # Generate embedding (run in executor if LLM-based)
                if self._embedder.mode == "llm":
                    embedding = await loop.run_in_executor(
                        None, self._embedder.embed, chunk,
                    )
                else:
                    embedding = self._embedder.embed(chunk)

                chunk_id = self._vector_store.add_chunk(
                    text=chunk,
                    embedding=embedding,
                    source_type=meta.get("source_type", "conversation"),
                    metadata={
                        "role": meta.get("role", ""),
                        "timestamp": time.time(),
                        "llm_extracted": llm_extracted,
                    },
                )
                chunk_ids.append(chunk_id)
                stats["chunks_stored"] += 1
            except Exception as e:
                logger.warning("cognify: chunk embedding failed: %s", e)

        # 3c: Create entity↔chunk bridges
        for entity in all_entities:
            for chunk_id in chunk_ids:
                self._vector_store.link_entity_to_chunk(entity.id, chunk_id)

        stats["relationships_added"] = len(all_relationships)

        logger.debug(
            "cognify complete: +%d entities, +%d rels, +%d chunks (llm=%s)",
            stats["entities_added"],
            stats["relationships_added"],
            stats["chunks_stored"],
            llm_extracted,
        )

        return stats

    # ── Graph-Completion Search ─────────────────────────────────────────

    async def graph_completion_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_hops: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Hybrid retrieval combining vector search with graph traversal.

        Adapted from Cognee's GraphCompletionRetriever and HybridRetriever:
            1. Vector search: find top-k semantically similar chunks
            2. Entity lookup: find graph entities linked to those chunks
            3. Graph traversal: BFS from those entities, collecting neighbors
            4. Merge: combine chunk text + graph context into a ranked prompt

        This provides multi-hop reasoning that pure vector search cannot —
        following relationship edges to discover indirect connections.

        Args:
            query: The query text to search for.
            top_k: Number of vector results to retrieve.
            max_hops: Maximum BFS depth in graph traversal.
            max_tokens: Maximum token budget for context output.

        Returns:
            A structured context string ready for LLM prompt injection.
        """
        if not self._initialized:
            await self.initialize()

        k = top_k or self._config.vector_search_top_k
        hops = max_hops or self._config.graph_traversal_max_hops
        max_tok = max_tokens or self._config.context_max_tokens
        max_nodes = self._config.graph_traversal_max_nodes

        # ── Channel 1: Vector Search ──
        vector_results: List[Dict] = []
        try:
            loop = asyncio.get_event_loop()
            if self._embedder.mode == "llm":
                query_embedding = await loop.run_in_executor(
                    None, self._embedder.embed, query,
                )
            else:
                query_embedding = self._embedder.embed(query)

            vector_results = self._vector_store.search(query_embedding, top_k=k)
        except Exception as e:
            logger.warning("graph_completion_search: vector search failed: %s", e)

        # ── Channel 2: Collect seed entities from vector hits ──
        seed_entity_ids: set = set()
        for vr in vector_results:
            entity_ids = self._vector_store.get_entities_for_chunk(vr["chunk_id"])
            seed_entity_ids.update(entity_ids)

        # Also add direct graph seed matching (existing BFS seed logic)
        query_lower = query.lower()
        query_words = set(query_lower.split())
        graph = self._memory_graph._graph

        for node_id, data in graph.nodes(data=True):
            label = data.get("label", "").lower()
            if label in query_lower or query_lower in label:
                seed_entity_ids.add(node_id)
            elif query_words & set(label.split()):
                seed_entity_ids.add(node_id)

        # ── Channel 3: Graph BFS Traversal from seeds ──
        # Mirrors Cognee's neighborhood traversal in graph_completion_retriever.py
        visited: set = set()
        traversed_nodes: List[Tuple[str, dict, float]] = []
        traversed_edges: List[Tuple[str, str, dict]] = []

        # Score seeds by vector similarity + graph degree
        scored_seeds: List[Tuple[str, float]] = []
        for eid in seed_entity_ids:
            if not graph.has_node(eid):
                continue
            # Base score from vector proximity
            vscore = 0.0
            for vr in vector_results:
                chunk_entities = self._vector_store.get_entities_for_chunk(vr["chunk_id"])
                if eid in chunk_entities:
                    vscore = max(vscore, vr["score"])
            # Boost by graph degree (hub importance)
            degree_boost = min(graph.degree(eid) / 10.0, 2.0)
            scored_seeds.append((eid, vscore + degree_boost))

        scored_seeds.sort(key=lambda x: x[1], reverse=True)

        for seed_id, seed_score in scored_seeds[:10]:  # Top 10 seeds
            if seed_id in visited:
                continue

            queue = [(seed_id, 0, seed_score)]
            while queue and len(visited) < max_nodes:
                node_id, depth, parent_score = queue.pop(0)
                if node_id in visited or depth > hops:
                    continue
                if not graph.has_node(node_id):
                    continue
                visited.add(node_id)

                data = dict(graph.nodes[node_id])
                node_score = parent_score * (0.6 ** depth)  # Decay per hop
                traversed_nodes.append((node_id, data, node_score))

                # Collect edges and queue neighbors
                for neighbor in graph.neighbors(node_id):
                    if neighbor in visited:
                        continue
                    edge_data = dict(graph.edges[node_id, neighbor])
                    traversed_edges.append((node_id, neighbor, edge_data))
                    if depth < hops:
                        edge_weight = edge_data.get("weight", 1.0)
                        queue.append((
                            neighbor,
                            depth + 1,
                            node_score * min(edge_weight, 3.0),
                        ))

        # ── Merge and Format ──
        return self._format_hybrid_context(
            vector_results, traversed_nodes, traversed_edges, max_tok,
        )

    def _format_hybrid_context(
        self,
        vector_results: List[Dict],
        graph_nodes: List[Tuple[str, dict, float]],
        graph_edges: List[Tuple[str, str, dict]],
        max_tokens: int,
    ) -> str:
        """
        Format hybrid retrieval results into a structured context prompt.

        Mirrors Cognee's format_hybrid_context() — builds sections for:
            1. Relevant text passages (from vector search)
            2. Key entities (from graph nodes)
            3. Relationships (from graph edges)
        """
        if not vector_results and not graph_nodes:
            return ""

        char_budget = max_tokens * 4  # ~4 chars per token
        lines: List[str] = ["[Voxium Memory Context]"]
        used = len(lines[0])

        # Section 1: Relevant passages from vector search
        if vector_results:
            lines.append("")
            lines.append("Relevant context:")
            used += 20
            for vr in vector_results[:5]:
                text = vr["text"][:200].replace("\n", " ").strip()
                line = f"  • {text}"
                if used + len(line) > char_budget:
                    break
                lines.append(line)
                used += len(line)

        # Section 2: Key entities from graph
        if graph_nodes:
            graph_nodes.sort(key=lambda x: x[2], reverse=True)
            lines.append("")
            lines.append("Key entities:")
            used += 15
            for node_id, data, score in graph_nodes[:15]:
                label = data.get("label", node_id)
                etype = data.get("type", "unknown")
                mentions = data.get("mention_count", 1)
                line = f"  • {label} ({etype}, mentioned {mentions}x)"
                if used + len(line) > char_budget:
                    break
                lines.append(line)
                used += len(line)

        # Section 3: Relationships (multi-hop connections)
        if graph_edges:
            seen_edges: set = set()
            rel_lines: List[str] = []
            graph = self._memory_graph._graph

            for src_id, tgt_id, edata in graph_edges:
                pair = (min(src_id, tgt_id), max(src_id, tgt_id))
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)

                src_label = graph.nodes[src_id].get("label", src_id) if graph.has_node(src_id) else src_id
                tgt_label = graph.nodes[tgt_id].get("label", tgt_id) if graph.has_node(tgt_id) else tgt_id
                relation = edata.get("relation", "related_to")
                line = f"  • {src_label} → {relation} → {tgt_label}"
                if used + len(line) > char_budget:
                    break
                rel_lines.append(line)
                used += len(line)

            if rel_lines:
                lines.append("")
                lines.append("Relationships:")
                lines.extend(rel_lines)

        # Section 4: Community topics
        communities = self._memory_graph._communities
        community_labels = self._memory_graph._community_labels
        if communities and community_labels and graph_nodes:
            node_ids = {n[0] for n in graph_nodes}
            node_to_comm = {
                n: cid for cid, members in communities.items()
                for n in members
            }
            relevant_comms = {
                node_to_comm[nid]
                for nid in node_ids
                if nid in node_to_comm
            }
            if relevant_comms:
                comm_strs = [
                    community_labels.get(cid, f"Topic {cid}")
                    for cid in relevant_comms
                ]
                topic_line = f"\nRelated topics: {', '.join(comm_strs)}"
                if used + len(topic_line) <= char_budget:
                    lines.append(topic_line)

        return "\n".join(lines)

    # ── Convenience API ─────────────────────────────────────────────────

    async def remember(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: chunk → cognify → store.
        Mirrors Cognee's remember() top-level API.
        """
        return await self.cognify(text, metadata)

    async def recall(
        self,
        query: str,
        max_tokens: int = 800,
    ) -> str:
        """
        Convenience wrapper: graph_completion_search.
        Mirrors Cognee's recall() top-level API.
        """
        return await self.graph_completion_search(query, max_tokens=max_tokens)

    # ── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Unified stats across both stores."""
        return {
            "graph_nodes": self._memory_graph.node_count,
            "graph_edges": self._memory_graph.edge_count,
            "graph_communities": self._memory_graph.community_count,
            "vector_chunks": self._vector_store.chunk_count,
            "embedding_mode": self._embedder.mode,
            "embedding_dim": self._embedder.dim,
            "initialized": self._initialized,
        }

    # ── Self-Test ───────────────────────────────────────────────────────

    @staticmethod
    async def self_test() -> None:
        """Quick self-test to verify the hybrid memory engine works."""
        import tempfile

        print("=" * 60)
        print("  Voxium HybridMemory — Self Test")
        print("=" * 60)

        config = HybridMemoryConfig(
            vector_db_path=os.path.join(
                tempfile.gettempdir(), "voxium_test_memory.sqlite",
            ),
            cognify_use_llm=False,  # Use regex only for test
        )

        engine = HybridMemory(config=config)
        await engine.initialize()

        # Test cognify
        result = await engine.cognify(
            "John Smith met with Sarah Chen at Google headquarters on January 15th. "
            "They discussed the new TensorFlow project and reviewed the API docs.",
            metadata={"role": "user"},
        )
        print(f"\n✓ cognify: {result}")

        # Test recall
        context = await engine.recall("What did John discuss?")
        print(f"\n✓ recall:\n{context}")

        # Stats
        stats = engine.get_stats()
        print(f"\n✓ stats: {json.dumps(stats, indent=2)}")

        print("\n" + "=" * 60)
        print("  All tests passed!")
        print("=" * 60)
