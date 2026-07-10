"""
Voxium — Intent Router
========================
Extracted routing logic from the original orchestrator.py.

Determines the pipeline path (AGENT / CLEANUP / SKIP) based on:
    - Trigger type (dictation vs voice_agent hotkey)
    - Wake word detection in the transcript
    - Agent and cleanup availability

This module is used by both:
    - langgraph/nodes/route_node.py (new LangGraph pipeline)
    - core/orchestrator.py (legacy compatibility)
"""

from __future__ import annotations

import os
import re
import logging
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)


class RouteKind(str, Enum):
    """The three possible routing outcomes."""
    AGENT = "agent"
    CLEANUP = "cleanup"
    SKIP = "skip"


class TriggerType(str, Enum):
    """How the recording was initiated."""
    DICTATION = "dictation"
    VOICE_AGENT = "voice_agent"
    WAKE_WORD = "wake_word"


def resolve_route_kind(
    cleanup_reachable: bool,
    agent_reachable: bool,
    agent_invoked: bool,
    voice_agent_requested: bool,
) -> RouteKind:
    """
    Determine the routing path for a finished dictation.

    Port of OpenWhispr's resolveDictationRouteKind():
        if voiceAgentRequested → agent (if reachable) else skip
        if agentReachable && agentInvoked → agent
        if cleanupReachable → cleanup
        else → skip
    """
    if voice_agent_requested:
        return RouteKind.AGENT if agent_reachable else RouteKind.SKIP
    if agent_reachable and agent_invoked:
        return RouteKind.AGENT
    if cleanup_reachable:
        return RouteKind.CLEANUP
    return RouteKind.SKIP


def detect_agent_invocation(
    text: str,
    agent_name: str = "Voxium",
) -> bool:
    """
    Check if the transcript contains a wake word / agent invocation.

    Matches patterns like "Hey Voxium", "OK Voxium", "Voxium," at the
    start of the transcript.
    """
    if not text or not agent_name:
        return False

    # Normalize
    text_lower = text.lower().strip()
    name_lower = agent_name.lower()

    # Check for common wake word patterns
    patterns = [
        f"hey {name_lower}",
        f"ok {name_lower}",
        f"okay {name_lower}",
        f"hi {name_lower}",
        f"{name_lower},",       # "Voxium, do something"
        f"{name_lower} ",       # "Voxium do something"
    ]

    for pattern in patterns:
        if text_lower.startswith(pattern):
            return True

    return False


def strip_agent_name(text: str, agent_name: str = "Voxium") -> str:
    """
    Remove the wake word / agent name prefix from a command.

    "Hey Voxium, set a timer" → "set a timer"
    """
    if not text or not agent_name:
        return text

    name_lower = agent_name.lower()

    # Patterns to strip (order matters — most specific first)
    patterns = [
        rf"(?i)^(?:hey|ok|okay|hi)\s+{re.escape(name_lower)}[,.]?\s*",
        rf"(?i)^{re.escape(name_lower)}[,.]?\s*",
    ]

    for pattern in patterns:
        result = re.sub(pattern, "", text, count=1)
        if result != text:
            return result.strip()

    return text


def classify_intent(
    text: str,
    agent_name: str = "Voxium",
) -> dict:
    """
    Enhanced intent classification for a transcript.

    Returns a dict with routing information:
        - route: RouteKind value
        - agent_invoked: bool
        - command_text: str (with agent name stripped)
        - trigger_pattern: str (which pattern matched)
    """
    agent_invoked = detect_agent_invocation(text, agent_name)
    command_text = strip_agent_name(text, agent_name) if agent_invoked else text

    # Determine trigger pattern
    trigger_pattern = ""
    if agent_invoked:
        text_lower = text.lower().strip()
        name_lower = agent_name.lower()
        for prefix in ["hey", "ok", "okay", "hi"]:
            if text_lower.startswith(f"{prefix} {name_lower}"):
                trigger_pattern = f"{prefix} {name_lower}"
                break
        if not trigger_pattern:
            trigger_pattern = name_lower

    return {
        "agent_invoked": agent_invoked,
        "command_text": command_text,
        "trigger_pattern": trigger_pattern,
    }
