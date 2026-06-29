"""
Voxium — Session State Manager (Cognee-Enhanced)
====================================================
Thread-safe state tracking for the active document, conversation history,
current session speakers, orchestrator state, and hybrid graph-vector memory.

Provides a centralized place for all pipeline stages to read/write state
without race conditions (uses asyncio.Lock).

Cognee-Inspired Upgrades:
    - SessionMemory: Short-term ring buffer with entity accumulation
    - LongTermGraphMemory: Persistent hybrid graph-vector store
    - Async flush: SessionMemory → LongTermGraphMemory when heuristic fires
    - graph_completion_search: Multi-hop retrieval replacing BFS-only

Architecture adapted from:
    - cognee/modules/session_distillation/distill.py → session→long-term flush
    - cognee/modules/retrieval/graph_completion_retriever.py → hybrid search
    - cognee/modules/session_lifecycle/ → session state management
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from core.memory_graph import MemoryGraph
from core.memory_engine import HybridMemory, HybridMemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentState:
    """State of the active document being edited."""
    title: str = ""
    content: str = ""
    cursor_position: int = 0
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None


@dataclass
class SessionSpeaker:
    """A speaker identified in the current session."""
    speaker_id: str
    display_name: Optional[str] = None
    profile_id: Optional[str] = None
    segment_count: int = 0
    last_seen: float = field(default_factory=time.time)


# =============================================================================
# SessionMemory — Short-term conversational memory
# =============================================================================

class SessionMemory:
    """
    Short-term memory for the current conversation session.

    Adapted from Cognee's session lifecycle management:
        - cognee/modules/session_lifecycle/ → session state tracking
        - cognee/modules/session_distillation/ → flush heuristics

    Tracks conversation turns in a ring buffer and accumulates extracted
    entities during the session. When the flush heuristic fires, accumulated
    data is packaged and sent to LongTermGraphMemory for persistence.

    The flush heuristic considers:
        1. Entity count threshold (enough new facts to be worth persisting)
        2. Time since last flush (periodic persistence for long sessions)
        3. Turn count threshold (batch processing for efficiency)
    """

    # Flush thresholds
    ENTITY_THRESHOLD = 3       # Flush when this many new entities accumulate
    TURN_THRESHOLD = 5         # Flush every N turns
    TIME_THRESHOLD = 300.0     # Flush every 5 minutes regardless

    MAX_HISTORY_TURNS = 50     # Ring buffer size

    def __init__(self):
        self._turns: List[ConversationTurn] = []
        self._accumulated_text: List[str] = []
        self._accumulated_metadata: List[Dict] = []
        self._entity_count: int = 0
        self._turns_since_flush: int = 0
        self._last_flush_time: float = time.time()
        self._lock = asyncio.Lock()

    async def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Add a conversation turn to the session buffer."""
        async with self._lock:
            turn = ConversationTurn(
                role=role,
                content=content,
                metadata=metadata or {},
            )
            self._turns.append(turn)

            # Trim ring buffer
            if len(self._turns) > self.MAX_HISTORY_TURNS:
                self._turns = self._turns[-self.MAX_HISTORY_TURNS:]

            # Accumulate for flush
            self._accumulated_text.append(content)
            self._accumulated_metadata.append(metadata or {})
            self._turns_since_flush += 1

            return turn

    async def record_entities(self, count: int) -> None:
        """Record that entities were extracted from recent turns."""
        async with self._lock:
            self._entity_count += count

    async def should_flush(self) -> bool:
        """
        Determine if accumulated session data should be flushed to long-term memory.

        Adapted from Cognee's session distillation heuristics in
        cognee/modules/session_distillation/distill.py.
        """
        async with self._lock:
            if not self._accumulated_text:
                return False

            # Condition 1: Enough entities accumulated
            if self._entity_count >= self.ENTITY_THRESHOLD:
                return True

            # Condition 2: Enough turns since last flush
            if self._turns_since_flush >= self.TURN_THRESHOLD:
                return True

            # Condition 3: Time-based periodic flush
            if (time.time() - self._last_flush_time) >= self.TIME_THRESHOLD:
                return True

            return False

    async def get_flush_payload(self) -> Dict[str, Any]:
        """
        Package accumulated session data for flushing to long-term memory.

        Returns the accumulated text and metadata, then resets the counters.
        Mirrors Cognee's load_distillable_session_inputs() which packages
        QA turns and context entries for distillation.
        """
        async with self._lock:
            payload = {
                "text": "\n".join(self._accumulated_text),
                "turns": len(self._accumulated_text),
                "metadata": {
                    "source": "session_flush",
                    "entity_count": self._entity_count,
                    "flush_time": time.time(),
                },
            }

            # Reset accumulators
            self._accumulated_text.clear()
            self._accumulated_metadata.clear()
            self._entity_count = 0
            self._turns_since_flush = 0
            self._last_flush_time = time.time()

            return payload

    async def get_history(self, last_n: int = 10) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        async with self._lock:
            return list(self._turns[-last_n:])

    async def get_context_string(self, last_n: int = 5) -> str:
        """Get conversation history as a formatted string for LLM context."""
        history = await self.get_history(last_n)
        if not history:
            return ""
        parts = []
        for turn in history:
            parts.append(f"{turn.role}: {turn.content}")
        return "\n".join(parts)

    async def clear(self) -> None:
        """Reset all session state."""
        async with self._lock:
            self._turns.clear()
            self._accumulated_text.clear()
            self._accumulated_metadata.clear()
            self._entity_count = 0
            self._turns_since_flush = 0
            self._last_flush_time = time.time()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# =============================================================================
# LongTermGraphMemory — Persistent hybrid graph-vector memory
# =============================================================================

class LongTermGraphMemory:
    """
    Persistent long-term memory using the hybrid graph-vector engine.

    Adapted from Cognee's permanent memory layer:
        - cognee/modules/session_distillation/distill.py → persist distilled lessons
        - cognee/modules/retrieval/ → query across graph + vector stores

    Wraps HybridMemory and handles:
        1. Ingesting flushed session data via cognify()
        2. Querying via graph_completion_search() for multi-hop retrieval
        3. Lifecycle management (load, persist, stats)
    """

    def __init__(self, hybrid_memory: Optional[HybridMemory] = None):
        self._hybrid = hybrid_memory or HybridMemory()
        self._flush_count: int = 0
        self._total_cognified: int = 0

    @property
    def hybrid_memory(self) -> HybridMemory:
        return self._hybrid

    @property
    def memory_graph(self) -> MemoryGraph:
        """Access the underlying NetworkX MemoryGraph."""
        return self._hybrid.memory_graph

    async def initialize(self, llm_instance=None) -> None:
        """Initialize the hybrid memory engine."""
        await self._hybrid.initialize(llm_instance=llm_instance)

    def set_llm_extract_fn(self, fn) -> None:
        """Register the LLM entity extraction function from reasoning.py."""
        self._hybrid.set_llm_extract_fn(fn)

    async def ingest_flush(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a flushed session payload into long-term memory.

        This is the "Load" step of the ECL pipeline, adapted from Cognee's
        publish_distilled_lessons() in session_distillation/distill.py.

        The payload text goes through cognify() which:
            1. Chunks the text
            2. Extracts entities via LLM (or regex fallback)
            3. Stores in both graph and vector stores

        Args:
            payload: Dict from SessionMemory.get_flush_payload() with
                     'text', 'turns', 'metadata' keys.

        Returns:
            Dict with cognify results (entities_added, chunks_stored, etc.)
        """
        text = payload.get("text", "")
        if not text or len(text.strip()) < 5:
            return {"entities_added": 0, "chunks_stored": 0}

        metadata = payload.get("metadata", {})
        result = await self._hybrid.cognify(text, metadata=metadata)

        self._flush_count += 1
        self._total_cognified += result.get("entities_added", 0)

        logger.info(
            "LongTermMemory flush #%d: +%d entities, +%d chunks",
            self._flush_count,
            result.get("entities_added", 0),
            result.get("chunks_stored", 0),
        )

        return result

    async def query(
        self,
        text: str,
        max_tokens: int = 800,
    ) -> str:
        """
        Query long-term memory using hybrid graph-completion search.

        Delegates to HybridMemory.graph_completion_search() which:
            1. Vector searches for semantically similar chunks
            2. Finds linked graph entities
            3. BFS-traverses the graph for multi-hop context
            4. Merges and ranks results

        This replaces the old BFS-only query_context() approach.
        """
        return await self._hybrid.graph_completion_search(
            text, max_tokens=max_tokens,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get unified stats from the hybrid memory engine."""
        stats = self._hybrid.get_stats()
        stats["flush_count"] = self._flush_count
        stats["total_cognified"] = self._total_cognified
        return stats


# =============================================================================
# StateManager — Centralized orchestrator state (Cognee-enhanced)
# =============================================================================

class StateManager:
    """
    Centralized, thread-safe state manager for the Voxium orchestrator.

    Cognee-Enhanced Architecture:
        - SessionMemory: Short-term ring buffer with entity accumulation
        - LongTermGraphMemory: Persistent hybrid graph-vector store
        - Async flush: Session → LongTerm when heuristic fires

    Tracks:
        - Active document state
        - Conversation history (via SessionMemory)
        - Current session speakers (from diarization)
        - Recording/processing state
        - Hybrid graph-vector memory (via LongTermGraphMemory)
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._document = DocumentState()
        self._speakers: Dict[str, SessionSpeaker] = {}
        self._is_recording = False
        self._is_processing = False
        self._agent_name = "Voxium"
        self._listeners: List = []

        # Dual-layer memory (Cognee architecture)
        self._session_memory = SessionMemory()
        self._long_term = LongTermGraphMemory()

        # Legacy compatibility: keep _memory_graph reference
        self._memory_graph = self._long_term.memory_graph

        # Flush task tracking
        self._flush_task: Optional[asyncio.Task] = None

    # ── Initialization ──────────────────────────────────────────────────

    async def initialize_memory(self, reasoning_engine=None) -> None:
        """
        Initialize the hybrid memory system.

        Call this after the reasoning engine is loaded to enable LLM-based
        entity extraction in the cognify pipeline.
        """
        llm_instance = None
        if reasoning_engine is not None:
            llm_instance = reasoning_engine.get_llm_instance()

            # Register the LLM extraction function
            self._long_term.set_llm_extract_fn(
                reasoning_engine.extract_entities_json
            )

        await self._long_term.initialize(llm_instance=llm_instance)
        logger.info("StateManager hybrid memory initialized")

    # ── Document State ──────────────────────────────────────────────────

    async def get_document(self) -> DocumentState:
        async with self._lock:
            return self._document

    async def update_document(self, **kwargs) -> None:
        async with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._document, key):
                    setattr(self._document, key, value)

    # ── Conversation History ────────────────────────────────────────────

    async def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a conversation turn to session memory.

        Also triggers:
            1. Immediate graph ingestion (for real-time entity tracking)
            2. Async flush check → cognify to long-term memory if heuristic fires
        """
        # Add to session memory
        await self._session_memory.add_turn(role, content, metadata)

        # Immediate graph ingestion (keeps existing real-time behavior)
        try:
            new_count = await self._memory_graph.ingest(
                content, metadata={"role": role},
            )
            if new_count > 0:
                await self._session_memory.record_entities(new_count)
        except Exception as e:
            logger.warning("Graph ingestion failed: %s", e)

        # Check if we should flush session → long-term memory
        try:
            if await self._session_memory.should_flush():
                await self._schedule_flush()
        except Exception as e:
            logger.warning("Flush check failed: %s", e)

    async def get_history(self, last_n: int = 10) -> List[ConversationTurn]:
        return await self._session_memory.get_history(last_n)

    async def get_context_string(self, last_n: int = 5) -> str:
        """Get conversation history as a formatted string for LLM context."""
        return await self._session_memory.get_context_string(last_n)

    async def clear_history(self) -> None:
        await self._session_memory.clear()

    # ── Speaker Tracking ────────────────────────────────────────────────

    async def register_speaker(
        self,
        speaker_id: str,
        display_name: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> None:
        async with self._lock:
            if speaker_id in self._speakers:
                speaker = self._speakers[speaker_id]
                speaker.segment_count += 1
                speaker.last_seen = time.time()
                if display_name:
                    speaker.display_name = display_name
                if profile_id:
                    speaker.profile_id = profile_id
            else:
                self._speakers[speaker_id] = SessionSpeaker(
                    speaker_id=speaker_id,
                    display_name=display_name,
                    profile_id=profile_id,
                )

    async def get_speakers(self) -> Dict[str, SessionSpeaker]:
        async with self._lock:
            return dict(self._speakers)

    async def clear_speakers(self) -> None:
        async with self._lock:
            self._speakers.clear()

    # ── Recording State ─────────────────────────────────────────────────

    async def set_recording(self, is_recording: bool) -> None:
        async with self._lock:
            self._is_recording = is_recording

    async def set_processing(self, is_processing: bool) -> None:
        async with self._lock:
            self._is_processing = is_processing

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            # Build comprehensive state including hybrid memory stats
            state = {
                "is_recording": self._is_recording,
                "is_processing": self._is_processing,
                "agent_name": self._agent_name,
                "speaker_count": len(self._speakers),
                "history_length": self._session_memory.turn_count,
                "graph_nodes": self._memory_graph.node_count,
                "graph_edges": self._memory_graph.edge_count,
                "graph_communities": self._memory_graph.community_count,
            }

            # Add hybrid memory stats
            try:
                hybrid_stats = self._long_term.get_stats()
                state["vector_chunks"] = hybrid_stats.get("vector_chunks", 0)
                state["embedding_mode"] = hybrid_stats.get("embedding_mode", "uninitialized")
                state["memory_flush_count"] = hybrid_stats.get("flush_count", 0)
            except Exception:
                pass

            return state

    # ── Agent Name ──────────────────────────────────────────────────────

    async def set_agent_name(self, name: str) -> None:
        async with self._lock:
            self._agent_name = name

    async def get_agent_name(self) -> str:
        async with self._lock:
            return self._agent_name

    # ── Graph-RAG Memory (Hybrid) ───────────────────────────────────────

    @property
    def memory_graph(self) -> MemoryGraph:
        """Access the graph-RAG memory engine (backward compatible)."""
        return self._memory_graph

    @property
    def session_memory(self) -> SessionMemory:
        """Access the session memory layer."""
        return self._session_memory

    @property
    def long_term_memory(self) -> LongTermGraphMemory:
        """Access the long-term hybrid memory layer."""
        return self._long_term

    async def get_graph_context(
        self,
        query: str,
        max_tokens: int = 800,
    ) -> str:
        """
        Get relevant memory context for LLM prompts.

        Uses the hybrid graph_completion_search (vector + graph traversal)
        for multi-hop context retrieval. Falls back to the original
        BFS-only approach if hybrid memory isn't initialized.
        """
        # Try hybrid search first (Cognee-style multi-hop)
        try:
            if self._long_term.hybrid_memory._initialized:
                return await self._long_term.query(query, max_tokens=max_tokens)
        except Exception as e:
            logger.debug("Hybrid search failed, falling back to graph-only: %s", e)

        # Fallback: original BFS-only graph search
        return await self._memory_graph.query_context(
            query, max_tokens=max_tokens,
        )

    # ── Async Flush: Session → Long-Term ────────────────────────────────

    async def _schedule_flush(self) -> None:
        """
        Schedule an async flush of session memory to long-term storage.

        Adapted from Cognee's session distillation pattern where finished
        session data is distilled into the persistent knowledge graph.
        The flush is debounced to avoid excessive processing.
        """
        if self._flush_task and not self._flush_task.done():
            return  # Already scheduled

        async def _do_flush():
            """Execute the flush: package session data → cognify → persist."""
            try:
                # Small delay to batch rapid-fire turns
                await asyncio.sleep(1.0)

                payload = await self._session_memory.get_flush_payload()
                if payload.get("text"):
                    await self._long_term.ingest_flush(payload)

                    # Persist the graph to disk
                    await self._memory_graph.persist()

                    logger.debug(
                        "Session→LongTerm flush complete: %d turns processed",
                        payload.get("turns", 0),
                    )
            except Exception as e:
                logger.warning("Session flush failed: %s", e)

        try:
            self._flush_task = asyncio.create_task(_do_flush())
        except RuntimeError:
            # No running event loop — skip async flush
            pass
