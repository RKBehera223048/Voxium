"""
Voxium — Prompt Engineering
============================
System prompts for the dual-pipeline architecture:
    - Cleanup prompt: Polishes raw dictation into clean, formatted text
    - Agent prompt: Processes voice commands for action execution

Ported from OpenWhispr's ReasoningService.ts prompt patterns and the
resolvePrompt configuration from src/config/prompts.
"""

from __future__ import annotations

import os
from typing import Optional, List


# =============================================================================
# Cleanup Prompt (Dictation Mode)
# =============================================================================

CLEANUP_SYSTEM_PROMPT = """You are a dictation cleanup assistant. Your job is to take raw speech-to-text output and clean it up into polished, well-formatted text.

Rules:
1. Fix punctuation, capitalization, and grammar errors introduced by the speech-to-text engine.
2. Remove filler words (um, uh, like, you know) and false starts.
3. Do NOT change the meaning, tone, or intent of the original text.
4. Do NOT add information that wasn't in the original dictation.
5. Do NOT summarize — preserve the full content.
6. Maintain paragraph breaks where the speaker clearly paused or changed topics.
7. Format lists, numbers, and dates naturally.
8. If the text appears to be code or technical content, preserve its structure.
{language_instruction}
{dictionary_instruction}

Respond ONLY with the cleaned-up text. No explanations, no preamble."""


def build_cleanup_prompt(
    raw_text: str,
    language: str = "en",
    custom_dictionary: Optional[List[str]] = None,
) -> str:
    """
    Build the system prompt for dictation cleanup.

    This mirrors OpenWhispr's resolvePrompt("dictationCleanup", ...) which
    builds context-aware prompts including language preference and custom
    dictionary terms.

    Args:
        raw_text: The raw transcription to clean up.
        language: Preferred language code.
        custom_dictionary: List of custom words/phrases to preserve exactly.

    Returns:
        Complete system prompt string.
    """
    language_instruction = ""
    if language and language != "auto":
        language_instruction = f"\nThe text is in {_language_name(language)}. Respond in the same language."

    dictionary_instruction = ""
    if custom_dictionary:
        words = ", ".join(custom_dictionary)
        dictionary_instruction = (
            f"\nThe following words/phrases should be preserved exactly as-is "
            f"(they are domain-specific terms): {words}"
        )

    return CLEANUP_SYSTEM_PROMPT.format(
        language_instruction=language_instruction,
        dictionary_instruction=dictionary_instruction,
    )


# =============================================================================
# Agent Prompt (Command Mode)
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are {agent_name}, a voice-controlled AI assistant. The user has given you a voice command that needs to be interpreted and acted upon.

Your capabilities:
1. Execute commands (calendar events, web automation, document editing)
2. Answer questions using your knowledge
3. Provide structured responses that can be parsed for action execution

Rules:
1. Parse the user's spoken command into a clear intent and action.
2. If the command is ambiguous, ask for clarification.
3. If the command references a tool or action you can perform, structure your response as an actionable instruction.
4. Remove the agent name ("{agent_name}") from the final output if present.
5. Be concise — voice responses should be brief and direct.
{language_instruction}
{dictionary_instruction}

Available actions:
- calendar: Create, read, update, delete calendar events
- web: Open URLs, search the web, automate browser tasks
- document: Format, highlight, delete, or edit text in the active document
- system: Control system settings, volume, brightness, etc.

If no specific action is needed, respond conversationally."""


def build_agent_prompt(
    raw_text: str,
    agent_name: str = "Voxium",
    language: str = "en",
    custom_dictionary: Optional[List[str]] = None,
    context: Optional[str] = None,
) -> str:
    """
    Build the system prompt for agent command processing.

    This mirrors OpenWhispr's resolvePrompt("dictationAgent", ...) which
    constructs the agent prompt with the user's configured agent name,
    language, and dictionary context.

    Args:
        raw_text: The raw voice command transcript.
        agent_name: The user's configured agent name (e.g., "Voxium").
        language: Preferred language code.
        custom_dictionary: List of custom words/phrases.
        context: Optional additional context (active document, conversation history).

    Returns:
        Complete system prompt string.
    """
    language_instruction = ""
    if language and language != "auto":
        language_instruction = f"\nRespond in {_language_name(language)}."

    dictionary_instruction = ""
    if custom_dictionary:
        words = ", ".join(custom_dictionary)
        dictionary_instruction = (
            f"\nThe following domain-specific terms may appear: {words}"
        )

    return AGENT_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        language_instruction=language_instruction,
        dictionary_instruction=dictionary_instruction,
    )


# =============================================================================
# Agent Name Detection
# =============================================================================

def detect_agent_invocation(text: str, agent_name: str) -> bool:
    """
    Detect if the user addressed their named agent in the transcript.

    Port of OpenWhispr's detectAgentName (src/config/agentDetection):
        - Matches "Hey {AgentName}" patterns at the beginning of text
        - Case-insensitive matching

    Args:
        text: Raw transcription text.
        agent_name: The user's configured agent name.

    Returns:
        True if the agent was invoked.
    """
    if not text or not agent_name:
        return False

    normalized = text.strip().lower()
    agent_lower = agent_name.strip().lower()

    # Match common wake word patterns
    wake_patterns = [
        f"hey {agent_lower}",
        f"hi {agent_lower}",
        f"ok {agent_lower}",
        f"okay {agent_lower}",
        f"{agent_lower}",  # Direct address at start
    ]

    for pattern in wake_patterns:
        if normalized.startswith(pattern):
            return True

    return False


def strip_agent_name(text: str, agent_name: str) -> str:
    """
    Remove the agent invocation prefix from the transcript.

    Port of OpenWhispr's ReasoningService.ts which removes the agent
    name from the final output (CLAUDE.md line 144).

    Args:
        text: Raw transcript possibly starting with "Hey AgentName".
        agent_name: The agent name to strip.

    Returns:
        Text with the agent invocation prefix removed.
    """
    if not text or not agent_name:
        return text

    normalized = text.strip().lower()
    agent_lower = agent_name.strip().lower()

    for prefix in [
        f"hey {agent_lower}, ",
        f"hey {agent_lower} ",
        f"hi {agent_lower}, ",
        f"hi {agent_lower} ",
        f"ok {agent_lower}, ",
        f"ok {agent_lower} ",
        f"okay {agent_lower}, ",
        f"okay {agent_lower} ",
    ]:
        if normalized.startswith(prefix):
            return text[len(prefix):]

    return text


# =============================================================================
# Helpers
# =============================================================================

def _language_name(code: str) -> str:
    """Map a language code to its full name."""
    language_map = {
        "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "pt": "Portuguese", "it": "Italian",
        "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
        "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
        "nl": "Dutch", "pl": "Polish", "sv": "Swedish",
        "da": "Danish", "fi": "Finnish", "no": "Norwegian",
        "tr": "Turkish", "uk": "Ukrainian", "cs": "Czech",
        "ro": "Romanian", "hu": "Hungarian", "el": "Greek",
        "he": "Hebrew", "th": "Thai", "vi": "Vietnamese",
        "id": "Indonesian", "ms": "Malay",
    }
    return language_map.get(code, code)


def get_custom_dictionary() -> List[str]:
    """Load custom dictionary from environment."""
    raw = os.getenv("CUSTOM_DICTIONARY", "")
    if not raw.strip():
        return []
    return [w.strip() for w in raw.split(",") if w.strip()]
