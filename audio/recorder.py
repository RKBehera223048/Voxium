"""
Voxium — Server-Side Audio Recorder
======================================
Manages the audio recording lifecycle on the server side.

Receives audio chunks from the WebSocket connection (browser mic via
AudioWorklet), accumulates them into a recording buffer, and provides
the complete recording to the audio pipeline when the session ends.

Handles:
    - WebSocket audio chunk accumulation
    - Format conversion (browser Float32 44.1kHz → 16kHz PCM)
    - Recording session lifecycle (start/stop/pause/resume)
    - Audio buffer size limits and overflow protection
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

from utils.audio_utils import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Browser AudioWorklet typically sends at 44.1kHz or 48kHz
BROWSER_SAMPLE_RATE = 44100
# Maximum recording duration (5 minutes)
MAX_RECORDING_SECONDS = 300
# Maximum buffer size in samples
MAX_BUFFER_SAMPLES = MAX_RECORDING_SECONDS * TARGET_SAMPLE_RATE


@dataclass
class RecordingSession:
    """State of a single recording session."""
    session_id: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    chunks_received: int = 0
    total_samples: int = 0
    is_active: bool = True
    trigger: str = "dictation"  # "dictation" | "voice_agent"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioRecorder:
    """
    Server-side audio buffer manager for WebSocket streaming.

    Receives Float32 PCM audio chunks from the browser's AudioWorklet,
    resamples to 16kHz, and accumulates into a buffer. When the recording
    session ends, the complete buffer is provided as a numpy array.

    Usage:
        recorder = AudioRecorder()

        # Start a session
        session = recorder.start_session("session-1", trigger="dictation")

        # Feed audio chunks from WebSocket
        recorder.add_chunk("session-1", audio_bytes)

        # End session and get complete audio
        audio = recorder.end_session("session-1")
    """

    def __init__(self):
        self._sessions: Dict[str, RecordingSession] = {}
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._lock = asyncio.Lock()

    async def start_session(
        self,
        session_id: str,
        trigger: str = "dictation",
        **metadata,
    ) -> RecordingSession:
        """
        Start a new recording session.

        Args:
            session_id: Unique session identifier (typically SocketIO sid).
            trigger: Recording trigger type ("dictation" or "voice_agent").
            **metadata: Additional session metadata.

        Returns:
            RecordingSession tracking object.
        """
        async with self._lock:
            # End any existing session for this client
            if session_id in self._sessions:
                logger.warning("Ending existing session %s before starting new one", session_id)
                self._finalize_session(session_id)

            session = RecordingSession(
                session_id=session_id,
                trigger=trigger,
                metadata=metadata,
            )
            self._sessions[session_id] = session
            self._buffers[session_id] = []

            logger.debug("Recording session started: %s (trigger=%s)", session_id, trigger)
            return session

    async def add_chunk(
        self,
        session_id: str,
        audio_data: bytes,
        sample_rate: int = BROWSER_SAMPLE_RATE,
    ) -> bool:
        """
        Add an audio chunk to the recording buffer.

        Args:
            session_id: Active session ID.
            audio_data: Raw audio bytes (Float32 PCM from browser).
            sample_rate: Sample rate of the incoming audio.

        Returns:
            True if chunk was accepted, False if session not found or buffer full.
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False

            session = self._sessions[session_id]
            if not session.is_active:
                return False

            # Convert bytes to float32 numpy array
            try:
                if isinstance(audio_data, (bytes, bytearray)):
                    samples = np.frombuffer(audio_data, dtype=np.float32)
                elif isinstance(audio_data, np.ndarray):
                    samples = audio_data.astype(np.float32)
                else:
                    logger.warning("Unexpected audio data type: %s", type(audio_data))
                    return False
            except Exception as e:
                logger.warning("Failed to parse audio chunk: %s", e)
                return False

            # Resample to 16kHz if needed
            if sample_rate != TARGET_SAMPLE_RATE:
                samples = self._resample(samples, sample_rate, TARGET_SAMPLE_RATE)

            # Check buffer overflow
            current_samples = sum(len(chunk) for chunk in self._buffers[session_id])
            if current_samples + len(samples) > MAX_BUFFER_SAMPLES:
                logger.warning(
                    "Recording buffer full for session %s (%.1fs)",
                    session_id, current_samples / TARGET_SAMPLE_RATE,
                )
                return False

            self._buffers[session_id].append(samples)
            session.chunks_received += 1
            session.total_samples += len(samples)

            return True

    async def end_session(self, session_id: str) -> Optional[np.ndarray]:
        """
        End a recording session and return the complete audio buffer.

        Args:
            session_id: Session to end.

        Returns:
            Complete audio as float32 numpy array at 16kHz, or None if
            session not found.
        """
        async with self._lock:
            if session_id not in self._sessions:
                return None

            session = self._sessions[session_id]
            session.is_active = False
            session.ended_at = time.time()

            buffer = self._buffers.get(session_id, [])
            if not buffer:
                self._cleanup_session(session_id)
                return None

            # Concatenate all chunks
            audio = np.concatenate(buffer)

            duration = len(audio) / TARGET_SAMPLE_RATE
            logger.info(
                "Recording session ended: %s (chunks=%d, duration=%.1fs)",
                session_id, session.chunks_received, duration,
            )

            self._cleanup_session(session_id)
            return audio

    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get info about an active session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            current_samples = sum(len(chunk) for chunk in self._buffers.get(session_id, []))
            return {
                "session_id": session.session_id,
                "is_active": session.is_active,
                "trigger": session.trigger,
                "chunks_received": session.chunks_received,
                "duration_seconds": current_samples / TARGET_SAMPLE_RATE,
                "started_at": session.started_at,
            }

    def _finalize_session(self, session_id: str) -> None:
        """Mark session as ended (called within lock)."""
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            session.ended_at = time.time()

    def _cleanup_session(self, session_id: str) -> None:
        """Remove session data (called within lock)."""
        self._sessions.pop(session_id, None)
        self._buffers.pop(session_id, None)

    @staticmethod
    def _resample(
        samples: np.ndarray,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if src_rate == dst_rate:
            return samples

        ratio = dst_rate / src_rate
        output_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, output_len)
        return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
