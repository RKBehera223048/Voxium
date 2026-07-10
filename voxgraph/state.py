"""
Voxium — LangGraph State Schema
==================================
Defines the VoxiumState TypedDict that flows through the entire
LangGraph pipeline. Every node reads from and writes to this state.

The state replaces the ad-hoc data passing in the old asyncio.Queue
orchestrator with a structured, typed contract between graph nodes.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, TypedDict

from dataclasses import dataclass, field


# =============================================================================
# Agent Action (shared with reasoning.py)
# =============================================================================

@dataclass
class AgentAction:
    """Parsed action from an agent command."""
    intent: str  # e.g., "calendar.create", "web.search", "document.edit"
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    confidence: float = 0.0


# =============================================================================
# VoxiumState — The central state schema for the LangGraph pipeline
# =============================================================================

class VoxiumState(TypedDict, total=False):
    """
    State that flows through every node of the Voxium LangGraph pipeline.

    This replaces the scattered local variables in the old orchestrator's
    _process_event() method with a single typed contract.

    Flow:
        audio_bytes → audio_node (VAD + STT)
        → route_node → agent_node / cleanup_node → tts_node → END

    Using total=False so nodes only need to set the fields they produce.
    """

    # ── Input ────────────────────────────────────────────────────────────
    audio_bytes: bytes                          # Raw audio input from frontend
    trigger: str                                # "dictation" | "voice_agent" | "wake_word"
    source_format: str                          # Audio format: "webm", "wav", etc.

    # ── VAD Stage ────────────────────────────────────────────────────────
    has_speech: bool                            # Whether VAD detected speech
    vad_speech_ms: float                        # Total speech duration in ms
    vad_segments: List[Dict[str, float]]        # [{start_ms, end_ms}, ...]

    # ── Transcription Stage ──────────────────────────────────────────────
    raw_text: str                               # Raw transcript text from Whisper
    transcription_language: str                 # Detected language code
    transcription_engine: str                   # Which STT engine was used
    transcription_elapsed_ms: float             # STT processing time

    # ── Routing Stage ────────────────────────────────────────────────────
    route: str                                  # "agent" | "cleanup" | "skip"
    agent_invoked: bool                         # Wake word / agent trigger detected?
    agent_name: str                             # Configured agent name

    # ── Processing Stage ─────────────────────────────────────────────────
    processed_text: str                         # Final output text (cleaned or agent response)
    tts_text: str                               # Text to synthesize (may differ from processed_text)
    actions: List[AgentAction]                  # Tool calls to execute
    tool_results: List[Dict[str, Any]]          # Results from tool execution

    # ── Memory Context ───────────────────────────────────────────────────
    memory_context: str                         # Retrieved memory context for agent

    # ── TTS Stage ────────────────────────────────────────────────────────
    tts_audio: bytes                            # Synthesized speech audio bytes
    tts_sample_rate: int                        # TTS audio sample rate
    tts_duration_ms: float                      # TTS audio duration

    # ── Messages (LangGraph convention) ──────────────────────────────────
    messages: list                              # LangChain message history

    # ── Metadata ─────────────────────────────────────────────────────────
    error: str                                  # Error message if any stage failed
    elapsed_ms: float                           # Total pipeline elapsed time
    session_id: str                             # Current session identifier

def create_initial_state(audio_bytes: bytes, source_format: str, trigger: str, agent_name: str, language: str, custom_dictionary: str) -> VoxiumState:
    return {
        "audio_bytes": audio_bytes,
        "source_format": source_format,
        "trigger": trigger,
        "agent_name": agent_name,
        "messages": [],
        "has_speech": False,
        "raw_text": "",
        "route": "skip",
        "processed_text": "",
        "tts_text": "",
        "tts_audio": b"",
        "error": "",
    }
