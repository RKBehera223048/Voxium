"""
Voxium — Session Manager
===========================
Manages recording session lifecycle and maps sessions to LangGraph
checkpoint threads.

Responsibilities:
    - Create/resume/end recording sessions
    - Map SocketIO client IDs to LangGraph thread IDs
    - Track session-scoped state (speakers, config overrides)
    - Session timeout and cleanup
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Session timeout (30 minutes of inactivity)
SESSION_TIMEOUT_SECONDS = 1800


@dataclass
class Session:
    """A Voxium interaction session."""
    session_id: str
    thread_id: str                    # LangGraph checkpoint thread ID
    client_id: str                    # SocketIO sid
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True
    trigger: str = "dictation"
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    @property
    def is_expired(self) -> bool:
        return self.idle_seconds > SESSION_TIMEOUT_SECONDS


class SessionManager:
    """
    Manages recording and conversation sessions.

    Each SocketIO client gets a session that maps to a LangGraph thread_id.
    The thread_id is used for checkpoint-based conversation persistence.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._client_to_session: Dict[str, str] = {}  # client_id → session_id
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        client_id: str,
        trigger: str = "dictation",
        **metadata,
    ) -> Session:
        """
        Create a new session for a SocketIO client.

        If the client already has an active session, it is reused.
        """
        async with self._lock:
            # Reuse existing session if still active
            if client_id in self._client_to_session:
                existing_id = self._client_to_session[client_id]
                existing = self._sessions.get(existing_id)
                if existing and existing.is_active and not existing.is_expired:
                    existing.last_activity = time.time()
                    existing.trigger = trigger
                    return existing

            # Create new session
            session_id = str(uuid.uuid4())[:8]
            thread_id = f"voxium-{client_id}-{session_id}"

            session = Session(
                session_id=session_id,
                thread_id=thread_id,
                client_id=client_id,
                trigger=trigger,
                metadata=metadata,
            )

            self._sessions[session_id] = session
            self._client_to_session[client_id] = session_id

            logger.info(
                "Session created: %s (client=%s, thread=%s)",
                session_id, client_id, thread_id,
            )
            return session

    async def get_session(self, client_id: str) -> Optional[Session]:
        """Get the active session for a client."""
        async with self._lock:
            session_id = self._client_to_session.get(client_id)
            if session_id:
                session = self._sessions.get(session_id)
                if session and session.is_active:
                    return session
            return None

    async def touch(self, client_id: str) -> None:
        """Update last activity time for a client's session."""
        async with self._lock:
            session_id = self._client_to_session.get(client_id)
            if session_id and session_id in self._sessions:
                self._sessions[session_id].last_activity = time.time()

    async def end_session(self, client_id: str) -> Optional[Session]:
        """End the active session for a client."""
        async with self._lock:
            session_id = self._client_to_session.pop(client_id, None)
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.is_active = False
                logger.info(
                    "Session ended: %s (age=%.0fs)",
                    session_id, session.age_seconds,
                )
                return session
            return None

    async def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of cleaned up sessions."""
        async with self._lock:
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if session.is_expired
            ]
            for sid in expired_ids:
                session = self._sessions.pop(sid)
                # Remove client mapping
                for cid, mapped_sid in list(self._client_to_session.items()):
                    if mapped_sid == sid:
                        del self._client_to_session[cid]

            if expired_ids:
                logger.info("Cleaned up %d expired sessions", len(expired_ids))
            return len(expired_ids)

    async def get_all_sessions(self) -> List[Session]:
        """Get all active sessions."""
        async with self._lock:
            return [s for s in self._sessions.values() if s.is_active]

    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        active = sum(1 for s in self._sessions.values() if s.is_active)
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active,
            "connected_clients": len(self._client_to_session),
        }
