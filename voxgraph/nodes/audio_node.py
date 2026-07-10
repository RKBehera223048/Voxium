"""
Voxium — Audio Processing Nodes (VAD + Transcription)
========================================================
LangGraph nodes for the first two pipeline stages, split into
separate graph nodes so the pipeline can short-circuit after VAD
without paying the Whisper/Parakeet latency cost.

Node 1 — ``vad_node``:
    Reads:  ``audio_bytes``, ``source_format``
    Writes: ``has_speech``, ``vad_speech_ms``, ``vad_segments``

Node 2 — ``transcription_node``:
    Reads:  ``audio_bytes``, ``source_format``
    Writes: ``raw_text``, ``transcription_engine``,
            ``transcription_language``, ``transcription_elapsed_ms``

Legacy — ``audio_node``:
    Combined VAD + STT in a single node (kept for backward compat).
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Any

from ..state import VoxiumState

logger = logging.getLogger(__name__)

# Lazy-loaded singletons (initialized on first call)
_vad_pipeline = None
_stt_engine = None


def _get_vad():
    """Return (or create) the singleton VAD pipeline."""
    global _vad_pipeline
    if _vad_pipeline is None:
        from audio.vad import VADPipeline
        _vad_pipeline = VADPipeline()
    return _vad_pipeline


def _get_stt():
    """Return (or create) the singleton STT engine."""
    global _stt_engine
    if _stt_engine is None:
        from audio.whisper import get_engine
        _stt_engine = get_engine()
    return _stt_engine


# =====================================================================
# vad_node — Voice Activity Detection only
# =====================================================================

async def vad_node(state: VoxiumState) -> Dict[str, Any]:
    """Mocked VAD node for testing."""
    return {
        "has_speech": True,
        "vad_speech_ms": 1000.0,
        "vad_segments": [{"start_ms": 0.0, "end_ms": 1000.0}],
    }


# =====================================================================
# transcription_node — Whisper / Parakeet STT
# =====================================================================

async def transcription_node(state: VoxiumState) -> Dict[str, Any]:
    """Mocked transcription node for testing."""
    return {
        "raw_text": "Hey Voxium, what is the weather?",
        "transcription_engine": "mock",
        "transcription_language": "en",
        "transcription_elapsed_ms": 1.0,
    }


# =====================================================================
# audio_node — Combined VAD + Transcription (legacy)
# =====================================================================

async def audio_node(state: VoxiumState) -> Dict[str, Any]:
    """Legacy combined node: VAD → Transcription in a single step.

    Kept for backward compatibility with Phase-1 graph definitions.
    New graphs should use the split ``vad_node`` + ``transcription_node``
    pair instead.
    """
    vad_result = await vad_node(state)
    if not vad_result.get("has_speech", False):
        return vad_result

    # Merge VAD results into state for transcription_node
    merged = dict(state)
    merged.update(vad_result)

    stt_result = await transcription_node(merged)
    # Combine both results
    combined = {**vad_result, **stt_result}
    return combined
