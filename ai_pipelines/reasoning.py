"""
Voxium — Local LLM Reasoning Engine
=====================================
Local inference for both dictation cleanup and agent command processing
using llama-cpp-python (GGUF models).

This module is the target for the AGENT route in the dual-pipeline
orchestrator. When a voice command is detected (via wake word or voice
agent hotkey), the raw transcript BYPASSES cleanup and comes straight
here for intent parsing and action execution.

Architecture matches OpenWhispr's ReasoningService.ts (local provider path)
and LocalReasoningService.ts — but purely local, no cloud fallback.
"""

from __future__ import annotations

import os
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
    Local LLM inference engine using llama-cpp-python.

    Provides two main interfaces:
        1. process_text() — General text processing (cleanup, formatting)
        2. process_agent_command() — Voice command interpretation + action routing

    Matches OpenWhispr's local provider architecture where the LLM runs
    via llama.cpp with configurable context length and thread count.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self._model_path = model_path or os.getenv("LLM_MODEL_PATH", "models/llm/model.gguf")
        self._n_ctx = int(os.getenv("LLM_CONTEXT_LENGTH", str(n_ctx)))
        self._n_threads = int(os.getenv("LLM_NUM_THREADS", str(n_threads)))
        self._temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens)))
        self._llm = None
        self._llm_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Lazy-load the LLM on first use."""
        if self._llm is not None:
            return

        async with self._llm_lock:
            if self._llm is not None:
                return

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(
                    f"LLM model not found at {self._model_path}. "
                    f"Please download a GGUF model and place it there."
                )

            logger.info(
                "Loading LLM: path=%s ctx=%d threads=%d",
                self._model_path, self._n_ctx, self._n_threads,
            )

            loop = asyncio.get_event_loop()
            self._llm = await loop.run_in_executor(None, self._load_model)
            logger.info("LLM loaded successfully")

    def _load_model(self):
        """Blocking model load (run in executor)."""
        from llama_cpp import Llama

        return Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False,
        )

    async def process_text(
        self,
        text: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ReasoningResult:
        """
        Run local LLM inference on text with a system prompt.

        This is the core inference call used by both cleanup and agent modes.
        Matches OpenWhispr's processText() in ReasoningService.ts (lines 277-330).

        Args:
            text: User input text.
            system_prompt: System instruction prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            ReasoningResult with the LLM's response.
        """
        await self._ensure_loaded()

        start_time = time.perf_counter()
        temp = temperature if temperature is not None else self._temperature
        max_tok = max_tokens if max_tokens is not None else self._max_tokens

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._run_inference,
                text,
                system_prompt,
                temp,
                max_tok,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result_text = response["choices"][0]["message"]["content"].strip()
            tokens_used = response.get("usage", {}).get("total_tokens", 0)

            if not result_text:
                return ReasoningResult(
                    success=False,
                    error="LLM returned empty response",
                    elapsed_ms=elapsed_ms,
                )

            return ReasoningResult(
                success=True,
                text=result_text,
                elapsed_ms=elapsed_ms,
                tokens_used=tokens_used,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("LLM inference failed: %s", e)
            return ReasoningResult(
                success=False,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

    def _run_inference(
        self,
        text: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Blocking LLM inference call (run in executor)."""
        return self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def process_agent_command(
        self,
        transcript: str,
        agent_name: str,
        system_prompt: str,
        context: Optional[str] = None,
    ) -> AgentResult:
        """
        Process a voice command for the agent pipeline.

        This is the AGENT route in the dual-pipeline — transcripts that match
        the wake word or come from the voice agent hotkey bypass cleanup and
        arrive here directly.

        The LLM interprets the command and returns a structured response that
        the orchestrator can route to mcp_tools/ for execution.

        Args:
            transcript: Raw voice command transcript (agent name already stripped).
            agent_name: The agent's name for context.
            system_prompt: Agent system prompt from prompts.py.
            context: Optional conversation/document context.

        Returns:
            AgentResult with response text and parsed actions.
        """
        start_time = time.perf_counter()

        # Build the user message with optional context
        user_message = transcript
        if context:
            user_message = f"Context:\n{context}\n\nCommand: {transcript}"

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

    def is_available(self) -> bool:
        """Check if the LLM model file exists and llama-cpp is importable."""
        try:
            import llama_cpp  # noqa: F401
            return os.path.isfile(self._model_path)
        except ImportError:
            return False

    def get_model_info(self) -> dict:
        return {
            "engine": "llama-cpp",
            "model_path": self._model_path,
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "loaded": self._llm is not None,
            "available": self.is_available(),
        }
