"""
Voxium — TTS Synthesis Node
==============================
LangGraph node that converts processed text to speech audio using
Piper TTS. Runs after agent or cleanup processing.

Reads: tts_text (or processed_text), tts_enabled
Writes: tts_audio
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Any

from voxgraph.state import VoxiumState

logger = logging.getLogger(__name__)

# Lazy-loaded TTS engine
_tts_engine = None


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        from audio.tts import PiperTTS
        _tts_engine = PiperTTS()
    return _tts_engine


async def tts_node(state: VoxiumState) -> Dict[str, Any]:
    """
    LangGraph node: Synthesize speech from text.

    Converts the processed response text into WAV audio using Piper TTS.
    Skips synthesis if TTS is disabled or no text is available.
    """
    tts_enabled = state.get("tts_enabled", os.getenv("TTS_ENABLED", "false").lower() == "true")

    if not tts_enabled:
        return {"tts_audio": b""}

    text = state.get("tts_text", "") or state.get("processed_text", "")
    if not text or not text.strip():
        return {"tts_audio": b""}

    # Only synthesize for agent responses (not cleanup)
    route = state.get("route", "skip")
    if route != "agent":
        return {"tts_audio": b""}

    start = time.perf_counter()
    tts = _get_tts()

    try:
        await tts.load()
        result = await tts.synthesize(text)
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        return {"tts_audio": b"", "error": f"TTS error: {e}"}

    elapsed = (time.perf_counter() - start) * 1000

    if result.success:
        logger.info(
            "TTS node: synthesized %d bytes (%.0fms audio, %.0fms compute)",
            len(result.audio_bytes), result.duration_ms, elapsed,
        )
        return {"tts_audio": result.audio_bytes}
    else:
        logger.warning("TTS synthesis failed: %s", result.error)
        return {"tts_audio": b""}
