"""
Voxium — Route Resolution Node
=================================
LangGraph node that determines the pipeline path based on the transcript
and trigger type. Direct port of the routing logic from orchestrator.py.

Reads: raw_text, trigger, agent_name
Writes: route, agent_invoked
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any

from voxgraph.state import VoxiumState

logger = logging.getLogger(__name__)


def route_node(state: VoxiumState) -> Dict[str, Any]:
    """
    LangGraph node: Resolve the processing route.

    Determines whether the transcript should go through:
        - AGENT: Voice command processing (wake word or voice_agent trigger)
        - CLEANUP: Dictation cleanup (polish raw text)
        - SKIP: Return raw text as-is

    This is a synchronous node (no I/O needed).
    """
    raw_text = state.get("raw_text", "")
    trigger = state.get("trigger", "dictation")
    agent_name = state.get("agent_name", os.getenv("AGENT_NAME", "Voxium"))

    # Import routing logic from intent_router (or core.prompts for now)
    from core.prompts import detect_agent_invocation, strip_agent_name

    # Check trigger type
    voice_agent_requested = trigger == "voice_agent"
    agent_invoked = detect_agent_invocation(raw_text, agent_name)

    # Check if agent and cleanup are reachable
    agent_enabled = os.getenv("AGENT_ENABLED", "true").lower() == "true"
    cleanup_enabled = os.getenv("CLEANUP_ENABLED", "true").lower() == "true"

    # Resolve route (port of orchestrator's resolve_route_kind)
    if voice_agent_requested:
        route = "agent" if agent_enabled else "skip"
    elif agent_enabled and agent_invoked:
        route = "agent"
    elif cleanup_enabled:
        route = "cleanup"
    else:
        route = "skip"

    logger.info(
        "Route node: route=%s (trigger=%s, agent_invoked=%s, text='%s')",
        route, trigger, agent_invoked, raw_text[:50],
    )

    return {
        "route": route,
        "agent_invoked": agent_invoked,
        "agent_name": agent_name,
    }
