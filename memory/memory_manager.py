from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.entity_extractor import (
    Entity,
    Relationship,
    extract_all as regex_extract_all,
)
from memory.chromadb import VectorStore
from memory.graphify import MemoryGraph
from memory.embeddings import EmbeddingProvider
from memory.retrieval import HybridRetriever

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HybridMemoryConfig:
    """Configuration for the hybrid graph-vector memory engine."""
    vector_db_path: str = ""
    embedding_dim: int = 0  # 0 = auto-detect at runtime
    chunk_size: int = 256
    chunk_overlap: int = 48
    vector_search_top_k: int = 5
    graph_traversal_max_hops: int = 2
    graph_traversal_max_nodes: int = 30
    context_max_tokens: int = 800
    tfidf_vocab_size: int = 512
    cognify_use_llm: bool = True

    @classmethod
    def from_env(cls) -> HybridMemoryConfig:
        return cls(
            vector_db_path=os.getenv("VECTOR_DB_PATH", "data/db/chromadb"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "0")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "256")),
            vector_search_top_k=int(os.getenv("VECTOR_SEARCH_TOP_K", "5")),
            graph_traversal_max_hops=int(os.getenv("GRAPH_MAX_HOPS", "2")),
            context_max_tokens=int(os.getenv("GRAPH_CONTEXT_MAX_TOKENS", "800")),
            cognify_use_llm=os.getenv("COGNIFY_USE_LLM", "true").lower() == "true",
        )

# =============================================================================
# Text Chunking
# =============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 48,
) -> List[str]:
    """
    Split text into overlapping chunks for vector storage.
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
# Hybrid Memory Manager
# =============================================================================

class HybridMemory:
    """
    Cognee-style unified memory engine combining graph and ChromaDB vector stores.
    
    Core API:
        - cognify(text, metadata) -> ECL pipeline: extract + load to both stores
        - graph_completion_search(query) -> hybrid retrieval via HybridRetriever
        - remember(text) -> convenience wrapper for cognify
        - on_turn(role, content, metadata) -> hooks into LangGraph StateManager
    """

    def __init__(
        self,
        config: Optional[HybridMemoryConfig] = None,
        llm_instance: Optional[Any] = None,
    ):
        self._config = config or HybridMemoryConfig.from_env()
        self._memory_graph = MemoryGraph()
        
        # We pass dim to EmbeddingProvider. It will handle the implementation internally.
        self._embedder = EmbeddingProvider(dim=self._config.embedding_dim)
        
        self._vector_store = VectorStore(
            db_path=self._config.vector_db_path,
        )
        
        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            memory_graph=self._memory_graph,
            embedding_provider=self._embedder,
            vector_search_top_k=self._config.vector_search_top_k,
            graph_traversal_max_hops=self._config.graph_traversal_max_hops,
            graph_traversal_max_nodes=self._config.graph_traversal_max_nodes,
        )
        
        self._llm_instance = llm_instance
        self._llm_extract_fn = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def memory_graph(self) -> MemoryGraph:
        """Access the underlying NetworkX-based MemoryGraph."""
        return self._memory_graph

    @property
    def vector_store(self) -> VectorStore:
        """Access the ChromaDB vector store."""
        return self._vector_store

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedder

    async def initialize(self, llm_instance: Optional[Any] = None) -> None:
        """Initialize embedding provider, graph, and vector stores."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if llm_instance:
                self._llm_instance = llm_instance

            # Initialize components if they expose an initialize method
            if hasattr(self._embedder, "initialize"):
                await self._embedder.initialize(self._llm_instance)
            if hasattr(self._memory_graph, "load"):
                await self._memory_graph.load()
            if hasattr(self._vector_store, "initialize"):
                await self._vector_store.initialize()

            self._initialized = True
            logger.info("HybridMemory initialized")

    def set_llm_extract_fn(self, fn: Any) -> None:
        """
        Register the LLM entity extraction function.
        Signature: async def fn(text: str) -> Tuple[List[Entity], List[Relationship]]
        """
        self._llm_extract_fn = fn

    async def cognify(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The ECL (Extract, Cognify, Load) pipeline.
        Extracts semantic chunks, identifies entities/relationships, and loads them
        into the Knowledge Graph and ChromaDB Vector Store.
        """
        if not self._initialized:
            await self.initialize()

        if not text or len(text.strip()) < 5:
            return {"entities_added": 0, "relationships_added": 0, "chunks_stored": 0}

        meta = metadata or {}
        stats = {"entities_added": 0, "relationships_added": 0, "chunks_stored": 0}

        # 1: EXTRACT - Chunk the text
        chunks = chunk_text(
            text,
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )

        # 2: COGNIFY - Extract entities and relationships
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        llm_extracted = False

        if self._config.cognify_use_llm and self._llm_extract_fn is not None:
            try:
                entities, relationships = await self._llm_extract_fn(text)
                if entities:
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    llm_extracted = True
                    logger.debug(
                        f"cognify: LLM extracted {len(entities)} entities, {len(relationships)} relationships"
                    )
            except Exception as e:
                logger.warning(f"cognify: LLM extraction failed, falling back to regex: {e}")

        if not llm_extracted:
            entities, relationships = regex_extract_all(text)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # 3: LOAD - Store in graph and vector store

        # 3a: Graph ingestion
        if hasattr(self._memory_graph, "ingest_async"):
            graph_add_count = await self._memory_graph.ingest_async(text, metadata=meta)
        else:
            graph_add_count = await self._memory_graph.ingest(text, metadata=meta)
            
        stats["entities_added"] = max(graph_add_count, len(all_entities))
        stats["relationships_added"] = len(all_relationships)

        # 3b: Vector store ingestion via ChromaDB
        loop = asyncio.get_running_loop()
        
        # Batch embedding
        embeddings = []
        try:
            if hasattr(self._embedder, "embed_batch_async"):
                embeddings = await self._embedder.embed_batch_async(chunks)
            else:
                embeddings = await loop.run_in_executor(None, self._embedder.embed_batch, chunks)
                
            entity_ids = [str(e.id) for e in all_entities]
            # Store entities as JSON string in metadata
            entity_json = json.dumps(entity_ids)

            metadatas = []
            for _ in chunks:
                chunk_meta = {
                    "role": meta.get("role", ""),
                    "timestamp": time.time(),
                    "source_type": meta.get("source_type", "conversation"),
                    "llm_extracted": llm_extracted,
                    "entities": entity_json,
                }
                metadatas.append(chunk_meta)

            if hasattr(self._vector_store, "add_chunks_async"):
                await self._vector_store.add_chunks_async(
                    texts=chunks, embeddings=embeddings, metadata=metadatas
                )
            else:
                await self._vector_store.add_chunks(
                    texts=chunks, embeddings=embeddings, metadatas=metadatas
                )
                
            stats["chunks_stored"] = len(chunks)
        except Exception as e:
            logger.warning(f"cognify: vector store insertion failed: {e}")

        logger.debug(
            f"cognify complete: +{stats['entities_added']} entities, "
            f"+{stats['relationships_added']} rels, +{stats['chunks_stored']} chunks"
        )
        return stats

    async def graph_completion_search(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Hybrid retrieval combining ChromaDB vector search with NetworkX graph traversal.
        Delegates logic to HybridRetriever.
        """
        if not self._initialized:
            await self.initialize()

        return await self._retriever.search(
            query=query,
            max_tokens=max_tokens or self._config.context_max_tokens,
        )

    async def remember(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper for cognify.
        """
        return await self.cognify(text, metadata)

    async def on_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Callback handler to register with StateManager.
        Captures conversation turns and automatically processes them into memory.
        """
        meta = metadata or {}
        meta["role"] = role
        meta["source_type"] = "conversation_turn"
        await self.cognify(content, metadata=meta)
