"""
Voxium — Agent Command Processing Node
=========================================
LangGraph node that runs the AGENT pipeline: interprets voice commands
via the local LLM and optionally requests tool execution.

Reads: raw_text, agent_name, memory_context, messages
Writes: processed_text, actions, messages
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from voxgraph.state import VoxiumState

logger = logging.getLogger(__name__)

# Lazy-loaded reasoning engine
_reasoning_engine = None


def _get_reasoning():
    global _reasoning_engine
    if _reasoning_engine is None:
        from llm.reasoning import LocalReasoningEngine
        from llm.inference import ChatLLM
        from llm.model_loader import load_model
        import os
        model_path = os.getenv("MODEL_PATH", "models/mistral.gguf")
        _reasoning_engine = LocalReasoningEngine(llm=ChatLLM(load_model(model_path)))
    return _reasoning_engine


async def agent_node(state: VoxiumState) -> Dict[str, Any]:
    """
    LangGraph node: Process a voice command through the local LLM.

    The transcript BYPASSES cleanup and goes straight to the reasoning
    engine for command interpretation. If the LLM requests tool calls,
    they are added to state.actions for the tool_node to execute.

    Memory context from the hybrid graph-vector store is injected into
    the prompt for multi-hop awareness of prior conversations.
    """
    start = time.perf_counter()

    raw_text = state.get("raw_text", "")
    agent_name = state.get("agent_name", os.getenv("AGENT_NAME", "Voxium"))
    memory_context = state.get("memory_context", "")

    if not raw_text:
        return {"processed_text": "", "error": "No text to process"}

    # Strip the agent name from the command
    from core.prompts import (
        strip_agent_name,
        build_agent_prompt,
        get_custom_dictionary,
        build_graph_context_prompt,
    )

    command_text = strip_agent_name(raw_text, agent_name)
    custom_dict = get_custom_dictionary()
    language = os.getenv("PREFERRED_LANGUAGE", "en")

    # Build conversation context from LangGraph messages
    messages = state.get("messages", [])
    context_parts = []
    for msg in messages[-6:]:  # Last 3 turns (user + assistant)
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        context_parts.append(f"{role}: {msg.content}")
    context = "\n".join(context_parts) if context_parts else ""

    # Build the agent system prompt
    system_prompt = build_agent_prompt(
        raw_text=command_text,
        agent_name=agent_name,
        language=language,
        custom_dictionary=custom_dict,
        context=context,
    )

    # Inject memory context if available
    user_message = command_text
    if memory_context:
        user_message = build_graph_context_prompt(memory_context, command_text)
    if context:
        user_message = f"Context:\n{context}\n\nCommand: {user_message}"

    # Run LLM inference
    reasoning = _get_reasoning()
    try:
        result = await reasoning.process_text(
            text=user_message,
            system_prompt=system_prompt,
            temperature=0.3,
        )
    except Exception as e:
        logger.error("Agent LLM inference failed: %s", e)
        return {
            "processed_text": raw_text,
            "error": f"Agent inference error: {e}",
        }

    elapsed = (time.perf_counter() - start) * 1000

    if not result.success:
        logger.warning("Agent processing failed: %s", result.error)
        return {
            "processed_text": raw_text,
            "error": result.error,
        }

    response_text = result.text

    logger.info(
        "Agent node: response='%s' (%.0fms, %d tokens)",
        response_text[:80], elapsed, result.tokens_used,
    )

    # Add to message history
    new_messages = [
        HumanMessage(content=command_text),
        AIMessage(content=response_text),
    ]

    return {
        "processed_text": response_text,
        "tts_text": response_text,
        "messages": new_messages,
        "elapsed_ms": elapsed,
    }
