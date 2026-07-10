"""
Voxium — Session State Manager (Refactored)
================================================
Thread-safe state tracking for the active document, conversation history,
current session speakers, and orchestrator state.

Phase 2 Refactoring:
    - Removed SessionMemory and LongTermGraphMemory classes (moved to memory/)
    - Simplified role: document state + speaker tracking + config state
    - Conversation history kept as simple ring buffer (no flush logic)
    - Memory integration deferred to memory/ module (Phase 3)

The StateManager provides a centralized, thread-safe store for all
pipeline stages to read/write state without race conditions.
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
# StateManager — Centralized orchestrator state (simplified)
# =============================================================================

class StateManager:
    """
    Centralized, thread-safe state manager for the Voxium orchestrator.

    Simplified Architecture (Phase 2):
        - Document state management
        - Conversation history as a simple ring buffer
        - Speaker tracking from diarization
        - Recording/processing flags
        - Agent name configuration

    Memory integration (SessionMemory, LongTermGraphMemory) has been
    extracted to the ``memory/`` module and will be reconnected in Phase 3.
    The StateManager now exposes a ``memory_bridge`` interface for external
    memory systems to hook into conversation events.
    """

    MAX_HISTORY_TURNS = 50  # Ring buffer size for conversation turns

    def __init__(self):
        self._lock = asyncio.Lock()
        self._document = DocumentState()
        self._speakers: Dict[str, SessionSpeaker] = {}
        self._is_recording = False
        self._is_processing = False
        self._agent_name = "Voxium"
        self._listeners: List = []

        # Conversation history (simple ring buffer, no flush logic)
        self._turns: List[ConversationTurn] = []

        # Memory bridge — external memory system can register callbacks
        self._on_turn_callbacks: List = []

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
        Add a conversation turn to the history ring buffer.

        Also notifies any registered memory bridge callbacks so external
        memory systems (Phase 3) can process the turn.
        """
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

        # Notify memory bridge callbacks (outside the lock to avoid deadlock)
        for callback in self._on_turn_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(role, content, metadata or {})
                else:
                    callback(role, content, metadata or {})
            except Exception as e:
                logger.warning("Memory bridge callback error: %s", e)

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

    async def clear_history(self) -> None:
        async with self._lock:
            self._turns.clear()

    async def get_turn_count(self) -> int:
        """Thread-safe turn count accessor."""
        async with self._lock:
            return len(self._turns)

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
            return {
                "is_recording": self._is_recording,
                "is_processing": self._is_processing,
                "agent_name": self._agent_name,
                "speaker_count": len(self._speakers),
                "history_length": len(self._turns),
            }

    # ── Agent Name ──────────────────────────────────────────────────────

    async def set_agent_name(self, name: str) -> None:
        async with self._lock:
            self._agent_name = name

    async def get_agent_name(self) -> str:
        async with self._lock:
            return self._agent_name

    # ── Memory Bridge ───────────────────────────────────────────────────

    def register_on_turn(self, callback) -> None:
        """
        Register a callback to be called on each new conversation turn.

        This is the bridge for the memory module (Phase 3) to hook into
        conversation events without tight coupling. The callback will
        receive (role: str, content: str, metadata: dict).

        Usage::

            from memory.memory_manager import HybridMemory

            memory = HybridMemory()
            state_manager.register_on_turn(memory.on_turn)
        """
        self._on_turn_callbacks.append(callback)

    def unregister_on_turn(self, callback) -> None:
        """Remove a previously registered turn callback."""
        self._on_turn_callbacks = [
            cb for cb in self._on_turn_callbacks if cb != callback
        ]

    # ── Legacy Compatibility ────────────────────────────────────────────
    # These properties provided backward compatibility during Phase 2.
    # They will be removed once Phase 3 memory migration is complete.

    async def initialize_memory(self, reasoning_engine=None) -> None:
        """
        Legacy stub — memory initialization moved to memory/ module.

        In Phase 3, the memory module will be initialized separately
        and connected via register_on_turn().
        """
        logger.info(
            "StateManager.initialize_memory() is a legacy stub. "
            "Memory initialization will be handled by the memory module in Phase 3."
        )

    async def get_graph_context(
        self,
        query: str,
        max_tokens: int = 800,
    ) -> str:
        """
        Legacy stub — graph context retrieval moved to memory/ module.

        Returns empty string until memory module is connected.
        """
        logger.debug(
            "StateManager.get_graph_context() is a legacy stub. "
            "Graph context retrieval will be handled by the memory module in Phase 3."
        )
        return ""
