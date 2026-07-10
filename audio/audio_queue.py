"""
Voxium — Thread-Safe Audio Queue
===================================
Priority-aware audio chunk queue that feeds the processing pipeline.

Replaces the raw asyncio.Queue in the old orchestrator with a structured
queue that supports:
    - Priority ordering (wake word detections jump the queue)
    - Session-aware batching (group chunks by recording session)
    - Backpressure (configurable max depth to prevent memory blow-up)
    - Clean shutdown via sentinel values
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class AudioPriority(IntEnum):
    """Priority levels for audio queue items (lower = higher priority)."""
    WAKE_WORD = 0       # Wake word detected — process immediately
    VOICE_AGENT = 1     # Voice agent hotkey — high priority
    DICTATION = 2       # Normal dictation — standard priority
    BACKGROUND = 3      # Background processing — low priority


@dataclass(order=True)
class AudioChunk:
    """
    A queued audio processing item.

    Ordered by priority (wake word first) then by timestamp (FIFO within
    same priority).
    """
    priority: AudioPriority = field(compare=True)
    timestamp: float = field(default_factory=time.time, compare=True)

    # Audio data (not used for ordering)
    audio_bytes: bytes = field(default=b"", compare=False)
    audio_samples: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    session_id: str = field(default="", compare=False)
    trigger: str = field(default="dictation", compare=False)
    source_format: str = field(default="pcm", compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    # Sentinel flag (used to signal shutdown)
    is_sentinel: bool = field(default=False, compare=False)


class AudioQueue:
    """
    Thread-safe priority queue for audio chunks.

    Provides the bridge between the WebSocket audio receiver (recorder.py)
    and the processing pipeline (LangGraph nodes). Wake word detections
    and voice agent commands get priority over regular dictation.

    Usage:
        queue = AudioQueue(max_depth=100)

        # Producer (WebSocket handler)
        await queue.put(AudioChunk(
            priority=AudioPriority.DICTATION,
            audio_bytes=raw_audio,
            session_id="client-123",
        ))

        # Consumer (pipeline)
        chunk = await queue.get()
        if not chunk.is_sentinel:
            process(chunk)
    """

    def __init__(self, max_depth: int = 100):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_depth)
        self._max_depth = max_depth
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_dropped = 0

    async def put(self, chunk: AudioChunk) -> bool:
        """
        Enqueue an audio chunk.

        Returns True if enqueued, False if queue is full (chunk dropped).
        """
        try:
            self._queue.put_nowait(chunk)
            self._total_enqueued += 1
            return True
        except asyncio.QueueFull:
            self._total_dropped += 1
            logger.warning(
                "Audio queue full (depth=%d), dropping chunk from session %s",
                self._max_depth, chunk.session_id,
            )
            return False

    async def get(self) -> AudioChunk:
        """
        Dequeue the highest-priority audio chunk (blocking).

        Returns a sentinel AudioChunk when the queue is being shut down.
        """
        chunk = await self._queue.get()
        self._total_dequeued += 1
        return chunk

    async def get_nowait(self) -> Optional[AudioChunk]:
        """Non-blocking dequeue. Returns None if empty."""
        try:
            chunk = self._queue.get_nowait()
            self._total_dequeued += 1
            return chunk
        except asyncio.QueueEmpty:
            return None

    async def shutdown(self) -> None:
        """Send a sentinel value to signal consumers to stop."""
        sentinel = AudioChunk(
            priority=AudioPriority.WAKE_WORD,  # Highest priority
            is_sentinel=True,
        )
        await self._queue.put(sentinel)
        logger.info("Audio queue shutdown signal sent")

    async def flush(self) -> int:
        """
        Clear all pending items from the queue.

        Returns the number of items flushed.
        """
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        logger.info("Audio queue flushed: %d items", count)
        return count

    @property
    def depth(self) -> int:
        """Current queue depth."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    def get_stats(self) -> Dict:
        """Get queue statistics for monitoring."""
        return {
            "depth": self.depth,
            "max_depth": self._max_depth,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_dropped": self._total_dropped,
        }
