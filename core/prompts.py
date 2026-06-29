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


# =============================================================================
# Entity Extraction Prompt (Cognee-style ECL Pipeline)
# =============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extraction engine. Your task is to extract structured entities and relationships from conversational text.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no preamble, no code fences.

The JSON object MUST follow this exact schema:
{
  "entities": [
    {
      "name": "exact entity name",
      "type": "person|place|organization|concept|date|technical|event",
      "description": "one-sentence description of this entity in context"
    }
  ],
  "relationships": [
    {
      "source": "source entity name (must match an entity name above)",
      "target": "target entity name (must match an entity name above)",
      "relation": "verb or short phrase describing the relationship",
      "description": "one-sentence fact expressed by this relationship"
    }
  ]
}

Rules:
1. Extract ALL meaningful entities: people, places, organizations, concepts, dates, technical terms, events.
2. Entity names should be normalized (e.g., "John" and "John Smith" → use "John Smith" if full name is known).
3. Entity types must be one of: person, place, organization, concept, date, technical, event.
4. Relationships must reference entities that exist in your entities list.
5. Relationship "relation" should be a concise verb phrase (e.g., "works_at", "discussed", "scheduled_for").
6. Extract at least the co-occurrence relationships between entities mentioned in the same sentence.
7. If no entities are found, return {"entities": [], "relationships": []}.
8. Do NOT invent entities or relationships not supported by the text.

Example input: "Sarah told me she's moving to Berlin next month for her new job at Siemens."
Example output:
{"entities":[{"name":"Sarah","type":"person","description":"Person who is relocating"},{"name":"Berlin","type":"place","description":"City Sarah is moving to"},{"name":"Siemens","type":"organization","description":"Sarah's new employer"},{"name":"next month","type":"date","description":"When Sarah is moving"}],"relationships":[{"source":"Sarah","target":"Berlin","relation":"moving_to","description":"Sarah is relocating to Berlin"},{"source":"Sarah","target":"Siemens","relation":"works_at","description":"Sarah has a new job at Siemens"},{"source":"Sarah","target":"next month","relation":"scheduled_for","description":"The move is planned for next month"}]}

Now extract entities and relationships from the following text. Respond with ONLY the JSON object:"""


def build_entity_extraction_prompt(text: str) -> str:
    """
    Build the complete entity extraction prompt for the LLM.

    This prompt is designed to force local GGUF models to output reliable
    structured JSON for the Cognee-style ECL pipeline. The few-shot example
    and explicit schema definition help small models stay on-format.

    Adapted from Cognee's extract_content_graph() prompt pattern in
    cognee/infrastructure/llm/extraction.py.

    Args:
        text: The raw text to extract entities from.

    Returns:
        Complete system prompt string.
    """
    return ENTITY_EXTRACTION_SYSTEM_PROMPT


# =============================================================================
# Graph Context Prompt (for multi-hop retrieval results)
# =============================================================================

def build_graph_context_prompt(
    context: str,
    query: str,
) -> str:
    """
    Format multi-hop graph retrieval results into a context block for the LLM.

    Adapted from Cognee's graph_context_for_question.txt prompt template used
    in GraphCompletionRetriever.get_completion_from_context().

    Args:
        context: The structured context string from graph_completion_search().
        query: The original user query.

    Returns:
        Formatted context string for injection into the LLM prompt.
    """
    if not context:
        return query

    return (
        f"Use the following memory context to inform your response. "
        f"The context contains relevant facts, entities, and relationships "
        f"from prior conversations.\n\n"
        f"{context}\n\n"
        f"User query: {query}"
    )
