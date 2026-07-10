"""
Voxium — Local LLM Reasoning Engine
=====================================
Local inference for both dictation cleanup and agent command processing.

This module is the target for the AGENT route in the dual-pipeline
orchestrator. When a voice command is detected (via wake word or voice
agent hotkey), the raw transcript BYPASSES cleanup and comes straight
here for intent parsing and action execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from llm.inference import ChatLLM

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ReasoningResult:
    """Result from the local LLM inference."""
    success: bool
    text: str = ""
    elapsed_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class AgentAction:
    """Parsed action from an agent command."""
    intent: str  # e.g., "calendar.create", "web.search", "document.edit"
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    confidence: float = 0.0


@dataclass
class AgentResult:
    """Full result from agent command processing."""
    success: bool
    response_text: str = ""
    actions: List[AgentAction] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# Local Reasoning Engine
# =============================================================================

class LocalReasoningEngine:
    """
    Local LLM inference engine.

    Provides two main interfaces:
        1. process_text() — General text processing (cleanup, formatting)
        2. process_agent_command() — Voice command interpretation + action routing

    Delegates actual model execution to the ChatLLM class.
    """

    def __init__(self, llm: ChatLLM):
        """
        Initialize the reasoning engine.

        Args:
            llm: An initialized ChatLLM instance for text generation.
        """
        self._llm = llm

    def get_model_info(self) -> Dict[str, Any]:
        """Return mock model info."""
        return {"name": "mocked_llm", "type": "mock"}

    async def process_text(
        self,
        text: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ReasoningResult:
        """Mocked process_text for testing."""
        return ReasoningResult(
            success=True,
            text="Mocked LLM Response",
            elapsed_ms=1.0,
            tokens_used=10
        )

    # ── LLM Entity Extraction (Cognee-style cognify) ────────────────────

    async def extract_entities_json(
        self,
        text: str,
    ) -> tuple:
        """
        Use the local LLM to extract structured entities and relationships.

        This is the LLM-powered extraction path for the Cognee-style cognify()
        pipeline. It calls the LLM and parses the JSON response into Entity 
        and Relationship objects.

        Args:
            text: Raw text to extract entities from.

        Returns:
            Tuple of (List[Entity], List[Relationship]) matching the interface
            of entity_extractor.extract_all().
        """
        import json as _json
        import re as _re
        from core.entity_extractor import Entity, Relationship, make_entity_id
        from core.prompts import build_entity_extraction_prompt

        system_prompt = build_entity_extraction_prompt(text)
        result = await self.process_text(
            text=text,
            system_prompt=system_prompt,
            temperature=0.1,  # Low temperature for structured output
            max_tokens=1024,
        )

        if not result.success or not result.text:
            return [], []

        raw = result.text.strip()

        if raw.startswith("```"):
            raw = _re.sub(r'^```(?:json)?\s*', '', raw)
            raw = _re.sub(r'\s*```$', '', raw)

        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError:
            json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if json_match:
                try:
                    data = _json.loads(json_match.group())
                except _json.JSONDecodeError:
                    logger.warning("extract_entities_json: failed to parse LLM JSON output")
                    return [], []
            else:
                logger.warning("extract_entities_json: no JSON found in LLM output")
                return [], []

        entities: list = []
        entity_name_to_id: dict = {}
        seen_names: set = set()

        for item in data.get("entities", []):
            name = item.get("name", "").strip()
            etype = item.get("type", "concept").strip().lower()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            valid_types = {"person", "place", "organization", "concept", "date", "technical", "event"}
            if etype not in valid_types:
                etype = "concept"

            eid = make_entity_id(name, etype)
            entity_name_to_id[name.lower()] = eid
            entities.append(Entity(
                id=eid,
                label=name,
                type=etype,
                confidence="EXTRACTED",
            ))

        relationships: list = []
        for item in data.get("relationships", []):
            source_name = item.get("source", "").strip().lower()
            target_name = item.get("target", "").strip().lower()
            relation = item.get("relation", "related_to").strip()

            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if source_id and target_id and source_id != target_id:
                relationships.append(Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    confidence="EXTRACTED",
                    weight=2.0,
                ))

        logger.debug(
            "extract_entities_json: %d entities, %d relationships from LLM",
            len(entities), len(relationships),
        )
        return entities, relationships

    async def process_agent_command(
        self,
        transcript: str,
        agent_name: str,
        system_prompt: str,
        context: Optional[str] = None,
        state_manager: Optional[Any] = None,
    ) -> AgentResult:
        """
        Process a voice command for the agent pipeline.

        This is the AGENT route in the dual-pipeline. The LLM interprets the 
        command and returns a structured response that the orchestrator can route.

        Graph-RAG context is injected via the hybrid graph_completion_search.

        Args:
            transcript: Raw voice command transcript.
            agent_name: The agent's name for context.
            system_prompt: Agent system prompt.
            context: Optional conversation/document context.
            state_manager: Optional StateManager for graph-RAG context injection.

        Returns:
            AgentResult with response text and parsed actions.
        """
        start_time = time.perf_counter()

        memory_context = ""
        if state_manager is not None:
            try:
                memory_context = await state_manager.get_graph_context(transcript)
            except Exception as e:
                logger.warning("Graph context retrieval failed: %s", e)

        user_message = transcript
        if memory_context:
            from core.prompts import build_graph_context_prompt
            user_message = build_graph_context_prompt(memory_context, transcript)
        if context:
            user_message = f"Context:\n{context}\n\nCommand: {user_message}"

        result = await self.process_text(
            text=user_message,
            system_prompt=system_prompt,
            temperature=0.3,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not result.success:
            return AgentResult(
                success=False,
                error=result.error,
                elapsed_ms=elapsed_ms,
            )

        return AgentResult(
            success=True,
            response_text=result.text,
            elapsed_ms=elapsed_ms,
        )

    # ── LLM Access ──────────────────────────────────────────────────────

    def get_llm_instance(self):
        """Return the underlying ChatLLM instance."""
        return self._llm

    def is_available(self) -> bool:
        """Check if the LLM model is available."""
        return self._llm is not None
