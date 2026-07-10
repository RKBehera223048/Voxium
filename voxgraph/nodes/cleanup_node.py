"""
Voxium — Dictation Cleanup Node
==================================
LangGraph node for the CLEANUP pipeline: polishes raw speech-to-text
into clean, well-formatted text via the local LLM.

Reads: raw_text
Writes: processed_text, messages
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from voxgraph.state import VoxiumState

logger = logging.getLogger(__name__)

# Lazy-loaded reasoning engine (shared with agent_node)
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


async def cleanup_node(state: VoxiumState) -> Dict[str, Any]:
    """
    LangGraph node: Clean up raw dictation text.

    Runs the LLM with the cleanup system prompt to polish the raw
    transcription into properly formatted text. If cleanup fails,
    falls back to returning the raw text.
    """
    start = time.perf_counter()
    raw_text = state.get("raw_text", "")

    if not raw_text:
        return {"processed_text": ""}

    from core.prompts import build_cleanup_prompt, get_custom_dictionary

    custom_dict = get_custom_dictionary()
    language = os.getenv("PREFERRED_LANGUAGE", "en")

    system_prompt = build_cleanup_prompt(
        raw_text=raw_text,
        language=language,
        custom_dictionary=custom_dict,
    )

    reasoning = _get_reasoning()
    try:
        result = await reasoning.process_text(
            text=raw_text,
            system_prompt=system_prompt,
        )
    except Exception as e:
        logger.error("Cleanup LLM inference failed: %s", e)
        return {"processed_text": raw_text, "error": f"Cleanup error: {e}"}

    elapsed = (time.perf_counter() - start) * 1000

    if result.success and result.text:
        cleaned = result.text
        logger.info("Cleanup node: cleaned in %.0fms", elapsed)
    else:
        cleaned = raw_text
        logger.warning("Cleanup failed, returning raw: %s", result.error)

    return {
        "processed_text": cleaned,
        "messages": [
            HumanMessage(content=raw_text),
            AIMessage(content=cleaned),
        ],
        "elapsed_ms": elapsed,
    }
