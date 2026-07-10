"""
Voxium — Event Bus
====================
Pub/sub event system for decoupling backend components from frontend
notification. Components emit events, and listeners (including the
WebSocket bridge) react without tight coupling.

Event Types:
    AUDIO_RECEIVED      — Audio chunk received from WebSocket
    VAD_COMPLETE        — VAD processing finished
    TRANSCRIPTION_DONE  — Speech-to-text complete
    ROUTE_RESOLVED      — Pipeline route determined
    AGENT_RESPONSE      — Agent LLM produced a response
    CLEANUP_DONE        — Dictation cleanup complete
    TTS_READY           — TTS audio synthesized and ready
    TOOL_EXECUTED       — Tool call finished
    WAKE_WORD_DETECTED  — Wake word trigger
    ERROR               — Pipeline error occurred
    SESSION_START       — Recording session started
    SESSION_END         — Recording session ended

Usage:
    from core.event_bus import event_bus

    # Subscribe
    event_bus.on("TRANSCRIPTION_DONE", lambda data: print(data["text"]))

    # Emit
    event_bus.emit("TRANSCRIPTION_DONE", {"text": "Hello world"})
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventBus:
    """
    Simple publish/subscribe event system.

    Thread-safe for the Flask-SocketIO eventlet model where all handlers
    run in the same green thread context.
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._socketio = None  # Set by app.py to bridge events to WebSocket

    def on(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        self._listeners[event_type].append(callback)

    def off(self, event_type: str, callback: Optional[Callable] = None) -> None:
        """Unsubscribe from an event type. If no callback, remove all."""
        if callback is None:
            self._listeners[event_type].clear()
        else:
            self._listeners[event_type] = [
                cb for cb in self._listeners[event_type] if cb != callback
            ]

    def emit(self, event_type: str, data: Any = None) -> None:
        """Emit an event to all registered listeners."""
        listeners = self._listeners.get(event_type, [])
        for callback in listeners:
            try:
                callback(data)
            except Exception as e:
                logger.error(
                    "Event listener error (event=%s): %s", event_type, e,
                    exc_info=True,
                )

        # Bridge to WebSocket if configured
        if self._socketio is not None:
            self._emit_to_websocket(event_type, data)

    def set_socketio(self, socketio) -> None:
        """
        Attach a Flask-SocketIO instance for automatic WebSocket bridging.

        When set, all emitted events are automatically forwarded to
        connected clients as SocketIO events.
        """
        self._socketio = socketio

    def _emit_to_websocket(self, event_type: str, data: Any) -> None:
        """Forward event to WebSocket clients."""
        # Map internal events to SocketIO event names
        ws_event_map = {
            "TRANSCRIPTION_DONE": "transcription_result",
            "TTS_READY": "tts_audio",
            "WAKE_WORD_DETECTED": "wake_word_detected",
            "ROUTE_RESOLVED": "processing",
            "ERROR": "error",
            "VAD_COMPLETE": "status",
            "AGENT_RESPONSE": "transcription_result",
            "CLEANUP_DONE": "transcription_result",
        }

        ws_event = ws_event_map.get(event_type)
        if ws_event:
            try:
                self._socketio.emit(ws_event, data)
            except Exception as e:
                logger.warning("WebSocket emit failed: %s", e)

    @property
    def listener_count(self) -> int:
        return sum(len(v) for v in self._listeners.values())

    def get_stats(self) -> Dict[str, int]:
        return {
            event: len(listeners)
            for event, listeners in self._listeners.items()
            if listeners
        }


# Global singleton
event_bus = EventBus()
