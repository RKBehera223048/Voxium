"""
Voxium — Session State Manager
================================
Thread-safe state tracking for the active document, conversation history,
current session speakers, and orchestrator state.

Provides a centralized place for all pipeline stages to read/write state
without race conditions (uses asyncio.Lock).
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


class StateManager:
    """
    Centralized, thread-safe state manager for the Voxium orchestrator.

    Tracks:
        - Active document state
        - Conversation history (bounded ring buffer)
        - Current session speakers (from diarization)
        - Recording/processing state
    """

    MAX_HISTORY_TURNS = 50

    def __init__(self):
        self._lock = asyncio.Lock()
        self._document = DocumentState()
        self._history: List[ConversationTurn] = []
        self._speakers: Dict[str, SessionSpeaker] = {}
        self._is_recording = False
        self._is_processing = False
        self._agent_name = "Voxium"
        self._listeners: List = []

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
        async with self._lock:
            turn = ConversationTurn(
                role=role,
                content=content,
                metadata=metadata or {},
            )
            self._history.append(turn)
            # Trim to max size
            if len(self._history) > self.MAX_HISTORY_TURNS:
                self._history = self._history[-self.MAX_HISTORY_TURNS:]

    async def get_history(self, last_n: int = 10) -> List[ConversationTurn]:
        async with self._lock:
            return list(self._history[-last_n:])

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
            self._history.clear()

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
                "history_length": len(self._history),
            }

    # ── Agent Name ──────────────────────────────────────────────────────

    async def set_agent_name(self, name: str) -> None:
        async with self._lock:
            self._agent_name = name

    async def get_agent_name(self) -> str:
        async with self._lock:
            return self._agent_name
