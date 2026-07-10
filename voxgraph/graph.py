"""
Voxium — Main LangGraph Pipeline Definition
================================================
Builds the declarative state graph that replaces the asyncio.Queue
event loop from the old orchestrator.

Pipeline topology::

    START → vad_node → [should_transcribe?]
        → "skip"       → END
        → "transcribe"  → transcription_node → route_node → [resolve_route?]
            → "agent"   → agent_node → [has_tool_calls?]
                → "tools"   → tool_node → agent_node  (loop back)
                → "respond"  → tts_node → END
            → "cleanup"  → cleanup_node → tts_node → END
            → "skip"     → END

Architecture notes:
    - The ``voxgraph/`` package sits alongside the project root so its
      name does NOT shadow the PyPI ``langgraph`` package.  This means
      ``from langgraph.graph import StateGraph, END`` resolves to the
      real library, while sibling modules are imported with explicit
      relative imports (``from .state import ...``).
    - The ``audio_node`` from Phase-1 combined VAD + transcription in a
      single node.  Phase-2 splits them into ``vad_node`` and
      ``transcription_node`` so the graph can short-circuit after VAD
      without paying the Whisper cost.

Usage::

    from voxgraph.graph import build_graph, invoke_pipeline

    graph = build_graph()
    result = await invoke_pipeline(graph, audio_bytes, trigger="dictation")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

# ── PyPI langgraph imports (no collision — our package is ``voxgraph``) ───
from langgraph.graph import StateGraph, END

# ── Local (relative) imports ─────────────────────────────────────────────
from .state import VoxiumState
from .edges import should_transcribe, resolve_route, has_tool_calls, should_speak
from .nodes.audio_node import vad_node, transcription_node
from .nodes.route_node import route_node
from .nodes.agent_node import agent_node
from .nodes.cleanup_node import cleanup_node
from .nodes.tts_node import tts_node
from .nodes.tool_node import tool_node
from .checkpoints import get_checkpointer

logger = logging.getLogger(__name__)


# =====================================================================
# Graph Builder
# =====================================================================

def build_graph(
    checkpointer: Any = None,
    *,
    with_checkpointer: bool = True,
) -> StateGraph:
    """Build and compile the Voxium processing graph.

    The compiled graph is a deterministic state machine: every node
    receives the full ``VoxiumState`` dict, performs its work, and
    returns a *partial* update dict that gets merged back into the
    state before the next node fires.

    Args:
        checkpointer: Optional pre-built LangGraph checkpointer.
            When *None* (the default) and *with_checkpointer* is
            ``True``, a SQLite-backed checkpointer is created
            automatically via :func:`get_checkpointer`.
        with_checkpointer: If ``False``, skip checkpoint persistence
            entirely (useful for unit tests and one-shot invocations).

    Returns:
        A compiled :class:`~langgraph.graph.state.CompiledStateGraph`
        ready for ``ainvoke`` / ``invoke``.
    """
    builder = StateGraph(VoxiumState)

    # ── Register Nodes ───────────────────────────────────────────────
    builder.add_node("vad", vad_node)
    builder.add_node("transcribe", transcription_node)
    builder.add_node("route", route_node)
    builder.add_node("agent", agent_node)
    builder.add_node("cleanup", cleanup_node)
    builder.add_node("tts", tts_node)
    builder.add_node("tools", tool_node)

    # ── Entry Point ──────────────────────────────────────────────────
    builder.set_entry_point("vad")

    # ── Edges ────────────────────────────────────────────────────────

    # 1. VAD → should we bother transcribing?
    builder.add_conditional_edges(
        "vad",
        should_transcribe,
        {
            "transcribe": "transcribe",
            "skip": END,
        },
    )

    # 2. Transcription → route resolution (always)
    builder.add_edge("transcribe", "route")

    # 3. Route → branch into agent / cleanup / skip
    builder.add_conditional_edges(
        "route",
        resolve_route,
        {
            "agent": "agent",
            "cleanup": "cleanup",
            "skip": END,
        },
    )

    # 4. Agent → check for tool calls or go to TTS
    builder.add_conditional_edges(
        "agent",
        has_tool_calls,
        {
            "tools": "tools",
            "respond": "tts",
        },
    )

    # 5. Tool execution → loop back to agent for next reasoning step
    builder.add_edge("tools", "agent")

    # 6. Cleanup → TTS (dictation results may also benefit from TTS)
    builder.add_edge("cleanup", "tts")

    # 7. TTS → check whether audio was actually produced
    builder.add_conditional_edges(
        "tts",
        should_speak,
        {
            "speak": END,
            "silent": END,
        },
    )

    # ── Compile ──────────────────────────────────────────────────────
    if with_checkpointer:
        cp = checkpointer or get_checkpointer()
        compiled = builder.compile(checkpointer=cp)
    else:
        compiled = builder.compile()

    logger.info("LangGraph pipeline compiled (%d nodes)", len(compiled.nodes))
    return compiled


# =====================================================================
# Convenience Invocation
# =====================================================================

async def invoke_pipeline(
    graph: Any,
    audio_bytes: bytes,
    trigger: str = "dictation",
    session_id: str = "default",
    source_format: str = "webm",
    *,
    agent_name: Optional[str] = None,
    language: Optional[str] = None,
    tts_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the full Voxium pipeline on a single audio clip.

    This is the primary entry point that replaces the old orchestrator's
    ``process_audio_sync()`` method.  It builds the initial
    :class:`VoxiumState`, invokes the graph asynchronously, and returns
    a normalized result dict suitable for JSON serialisation.

    Args:
        graph: Compiled LangGraph state graph (from :func:`build_graph`).
        audio_bytes: Raw audio data from the frontend.
        trigger: ``"dictation"`` | ``"voice_agent"`` | ``"wake_word"``.
        session_id: Session ID for checkpoint threading.
        source_format: Audio format hint (``"webm"``, ``"wav"``, …).
        agent_name: Override the agent name (default: ``$AGENT_NAME``).
        language: Override the language code (default: ``$PREFERRED_LANGUAGE``).
        tts_enabled: Override TTS toggle (default: from env).

    Returns:
        Dict with keys ``success``, ``route``, ``raw_text``,
        ``processed_text``, ``tts_audio``, ``has_speech``,
        ``agent_invoked``, ``tool_results``, ``elapsed_ms``, ``error``.
    """
    pipeline_start = time.perf_counter()

    # ── Build initial state ──────────────────────────────────────────
    initial_state: Dict[str, Any] = {
        "audio_bytes": audio_bytes,
        "trigger": trigger,
        "source_format": source_format,
        "session_id": session_id,
        "agent_name": agent_name or os.getenv("AGENT_NAME", "Voxium"),
    }

    if language is not None:
        initial_state["language"] = language
    if tts_enabled is not None:
        initial_state["tts_enabled"] = tts_enabled

    config = {"configurable": {"thread_id": session_id}}

    # ── Invoke ───────────────────────────────────────────────────────
    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.error("Pipeline invocation failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "error": str(exc),
            "elapsed_ms": round(elapsed, 1),
            "route": "skip",
            "raw_text": "",
            "processed_text": "",
            "tts_audio": b"",
            "has_speech": False,
            "agent_invoked": False,
            "tool_results": [],
        }

    elapsed = (time.perf_counter() - pipeline_start) * 1000

    return {
        "success": True,
        "route": result.get("route", "skip"),
        "raw_text": result.get("raw_text", ""),
        "processed_text": result.get("processed_text", ""),
        "tts_audio": result.get("tts_audio", b""),
        "has_speech": result.get("has_speech", False),
        "agent_invoked": result.get("agent_invoked", False),
        "tool_results": result.get("tool_results", []),
        "elapsed_ms": round(elapsed, 1),
        "error": result.get("error"),
    }
