"""
Voxium — LangGraph Conditional Edges
=======================================
Edge functions that control the flow through the pipeline graph.
Each function reads the current state and returns the name of the
next node to execute.
"""

from __future__ import annotations

from typing import Literal

from voxgraph.state import VoxiumState


def should_transcribe(state: VoxiumState) -> Literal["transcribe", "skip"]:
    """
    After VAD: should we proceed to transcription?

    Routes to "skip" if VAD detected no speech (skip the whole pipeline).
    Routes to "transcribe" if speech was detected.
    """
    has_speech = state.get("has_speech", False)
    has_speech = state.get("has_speech", False)

    if not has_speech:
        return "skip"
    return "transcribe"


def resolve_route(state: VoxiumState) -> Literal["agent", "cleanup", "skip"]:
    """
    After route_node: which processing path to take?

    Returns the route determined by route_node:
        - "agent": Voice command → LLM agent processing
        - "cleanup": Dictation → LLM text cleanup
        - "skip": Return raw text as-is
    """
    route = state.get("route", "skip")
    if route == "agent":
        return "agent"
    elif route == "cleanup":
        return "cleanup"
    return "skip"


def has_tool_calls(state: VoxiumState) -> Literal["tools", "respond"]:
    """
    After agent_node: does the response contain tool calls?

    If the agent produced actions, route to tool_node for execution.
    Otherwise, proceed to TTS synthesis.
    """
    actions = state.get("actions", [])
    if actions:
        return "tools"
    return "respond"


def should_speak(state: VoxiumState) -> Literal["speak", "silent"]:
    """
    After processing: should TTS generate audio?

    Only generates TTS for agent responses when TTS is enabled.
    Cleanup results (dictation text) don't need TTS.
    """
    import os
    tts_enabled = state.get("tts_enabled", os.getenv("TTS_ENABLED", "false").lower() == "true")
    route = state.get("route", "skip")
    processed_text = state.get("processed_text", "")

    if tts_enabled and route == "agent" and processed_text:
        return "speak"
    return "silent"
