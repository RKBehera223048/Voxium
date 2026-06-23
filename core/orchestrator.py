"""
Voxium — Main Orchestrator
============================
The central event loop that receives audio, runs the VAD → transcription
pipeline, and routes results through either the CLEANUP or AGENT path.

Dual-Pipeline Intent Routing (ported from OpenWhispr):
======================================================
This is a direct port of OpenWhispr's dictationRouting.js (21 lines)
and audioManager.js resolveReasoningRoute (lines 27-72).

The routing logic:
    1. VOICE_AGENT trigger → always take AGENT route (no wake word needed)
    2. Wake word detected ("Hey {AgentName}") → AGENT route
    3. Cleanup model configured → CLEANUP route (polish the dictation)
    4. Nothing matches → SKIP (return raw transcript)

This replaces OpenWhispr's Electron IPC event system with a Python
asyncio.Queue-based event loop.
"""

from __future__ import annotations

import os
import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict

from ai_pipelines.ingestion import VADPipeline, VADResult
from ai_pipelines.transcription import (
    TranscriptionEngine,
    TranscriptionResult,
    get_engine,
)
from ai_pipelines.reasoning import LocalReasoningEngine, AgentResult
from core.prompts import (
    build_cleanup_prompt,
    build_agent_prompt,
    detect_agent_invocation,
    strip_agent_name,
    get_custom_dictionary,
)
from core.state_manager import StateManager

logger = logging.getLogger(__name__)


# =============================================================================
# Route Resolution — Port of OpenWhispr dictationRouting.js
# =============================================================================

class RouteKind(str, Enum):
    """
    The three possible routing outcomes.
    Direct port of dictationRouting.js return values (lines 11, 14, 16, 19).
    """
    AGENT = "agent"
    CLEANUP = "cleanup"
    SKIP = "skip"


class TriggerType(str, Enum):
    """
    How the recording was initiated.
    Maps to OpenWhispr's hotkey system:
        - DICTATION: Standard dictation hotkey
        - VOICE_AGENT: Dedicated voice agent hotkey (bypasses wake word)
        - WAKE_WORD: Agent invoked via "Hey {AgentName}" in transcript
    """
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

    EXACT port of OpenWhispr's resolveDictationRouteKind()
    from dictationRouting.js (lines 4-20):

        if voiceAgentRequested → agent (if reachable) else skip
        if agentReachable && agentInvoked → agent
        if cleanupReachable → cleanup
        else → skip

    A voice agent recording ALWAYS takes the agent route — it never
    falls back to cleanup. This is a deliberate design choice from
    OpenWhispr (CLAUDE.md line 538).
    """
    if voice_agent_requested:
        return RouteKind.AGENT if agent_reachable else RouteKind.SKIP
    if agent_reachable and agent_invoked:
        return RouteKind.AGENT
    if cleanup_reachable:
        return RouteKind.CLEANUP
    return RouteKind.SKIP


# =============================================================================
# Pipeline Event
# =============================================================================

@dataclass
class AudioEvent:
    """An audio processing event for the orchestrator queue."""
    audio_bytes: bytes
    trigger: TriggerType = TriggerType.DICTATION
    source_format: str = "webm"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete result from the orchestrator pipeline."""
    success: bool
    route: RouteKind = RouteKind.SKIP
    raw_text: str = ""
    processed_text: str = ""
    agent_result: Optional[AgentResult] = None
    vad_result: Optional[VADResult] = None
    transcription_result: Optional[TranscriptionResult] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# Orchestrator
# =============================================================================

class VoxiumOrchestrator:
    """
    Main event loop for the Voxium voice assistant.

    Replaces OpenWhispr's Electron IPC-based event system with a Python
    asyncio.Queue. The orchestrator:

        1. Receives audio events via the queue
        2. Runs VAD pre-filtering (ingestion.py)
        3. Transcribes speech (transcription.py)
        4. Resolves the routing (dictation cleanup vs. agent command)
        5. Processes through the appropriate pipeline
        6. Returns results via callback

    The full flow mirrors OpenWhispr's audioManager.js processAudio()
    (lines 591-736) but in async Python.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        on_result: Optional[Callable[[PipelineResult], Any]] = None,
    ):
        self.state = state_manager or StateManager()
        self.on_result = on_result

        # Pipelines (lazy-initialized)
        self._vad = VADPipeline()
        self._stt_engine: Optional[TranscriptionEngine] = None
        self._reasoning: Optional[LocalReasoningEngine] = None

        # Event queue
        self._queue: asyncio.Queue[Optional[AudioEvent]] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Configuration
        self._agent_enabled = os.getenv("AGENT_ENABLED", "true").lower() == "true"
        self._cleanup_enabled = os.getenv("CLEANUP_ENABLED", "true").lower() == "true"
        self._language = os.getenv("PREFERRED_LANGUAGE", "en")

    async def start(self) -> None:
        """Start the orchestrator event loop."""
        if self._running:
            return

        self._running = True

        # Initialize the STT engine
        self._stt_engine = get_engine()
        logger.info("STT engine: %s", self._stt_engine.get_model_info())

        # Initialize the reasoning engine (lazy — only loads model on first use)
        self._reasoning = LocalReasoningEngine()

        # Set agent name from environment
        agent_name = os.getenv("AGENT_NAME", "Voxium")
        await self.state.set_agent_name(agent_name)

        # Start the event loop
        self._task = asyncio.create_task(self._event_loop())
        logger.info("Orchestrator started (agent=%s)", agent_name)

    async def stop(self) -> None:
        """Stop the orchestrator event loop."""
        self._running = False
        # Send sentinel to unblock the queue
        await self._queue.put(None)
        if self._task:
            await self._task
            self._task = None
        logger.info("Orchestrator stopped")

    async def submit_audio(
        self,
        audio_bytes: bytes,
        trigger: TriggerType = TriggerType.DICTATION,
        source_format: str = "webm",
        **metadata,
    ) -> None:
        """
        Submit audio for processing.

        This is the main entry point called by the Flask routes
        when audio arrives from the frontend.
        """
        event = AudioEvent(
            audio_bytes=audio_bytes,
            trigger=trigger,
            source_format=source_format,
            metadata=metadata,
        )
        await self._queue.put(event)
        logger.debug("Audio event queued (trigger=%s, size=%d)", trigger.value, len(audio_bytes))

    async def process_audio_sync(
        self,
        audio_bytes: bytes,
        trigger: TriggerType = TriggerType.DICTATION,
        source_format: str = "webm",
    ) -> PipelineResult:
        """
        Process audio synchronously (for REST API calls).
        Bypasses the queue and runs the pipeline directly.
        """
        event = AudioEvent(
            audio_bytes=audio_bytes,
            trigger=trigger,
            source_format=source_format,
        )
        return await self._process_event(event)

    # ── Event Loop ──────────────────────────────────────────────────────

    async def _event_loop(self) -> None:
        """
        Main event loop — processes audio events from the queue.
        Mirrors OpenWhispr's recording → processing → routing flow.
        """
        logger.info("Event loop started")

        while self._running:
            try:
                event = await self._queue.get()

                # Sentinel value to stop the loop
                if event is None:
                    break

                result = await self._process_event(event)

                # Notify listener
                if self.on_result:
                    try:
                        if asyncio.iscoroutinefunction(self.on_result):
                            await self.on_result(result)
                        else:
                            self.on_result(result)
                    except Exception as e:
                        logger.error("Result callback error: %s", e)

            except Exception as e:
                logger.error("Event loop error: %s", e, exc_info=True)

        logger.info("Event loop stopped")

    async def _process_event(self, event: AudioEvent) -> PipelineResult:
        """
        Full pipeline for a single audio event.

        Port of OpenWhispr audioManager.js processAudio() (lines 591-736):
            1. Speech gate check
            2. Transcription (Whisper or Parakeet)
            3. Route resolution
            4. Cleanup OR Agent processing
        """
        pipeline_start = time.perf_counter()
        await self.state.set_processing(True)

        try:
            # ── Step 1: VAD Pre-Filter ──
            vad_result = await self._vad.detect_speech(
                event.audio_bytes, event.source_format
            )

            if not self._vad.should_transcribe(vad_result):
                elapsed = (time.perf_counter() - pipeline_start) * 1000
                logger.info(
                    "VAD rejected audio (speech_ms=%.0f)",
                    vad_result.total_speech_ms,
                )
                return PipelineResult(
                    success=True,
                    route=RouteKind.SKIP,
                    vad_result=vad_result,
                    elapsed_ms=elapsed,
                )

            # ── Step 2: Transcription ──
            # Get custom dictionary for prompt biasing
            custom_dict = get_custom_dictionary()
            initial_prompt = ", ".join(custom_dict) if custom_dict else None

            transcription = await self._stt_engine.transcribe(
                event.audio_bytes,
                language=self._language,
                source_format=event.source_format,
                initial_prompt=initial_prompt,
            )

            if not transcription.success or not transcription.text.strip():
                elapsed = (time.perf_counter() - pipeline_start) * 1000
                return PipelineResult(
                    success=True,
                    route=RouteKind.SKIP,
                    raw_text="",
                    vad_result=vad_result,
                    transcription_result=transcription,
                    elapsed_ms=elapsed,
                )

            raw_text = transcription.text.strip()

            # Store raw transcription in history
            await self.state.add_turn("user", raw_text, {
                "engine": transcription.engine,
                "trigger": event.trigger.value,
            })

            # ── Step 3: Route Resolution ──
            agent_name = await self.state.get_agent_name()
            voice_agent_requested = event.trigger == TriggerType.VOICE_AGENT
            agent_invoked = detect_agent_invocation(raw_text, agent_name)

            # Check if agent and cleanup are reachable
            agent_reachable = (
                self._agent_enabled
                and self._reasoning is not None
                and self._reasoning.is_available()
            )
            cleanup_reachable = (
                self._cleanup_enabled
                and self._reasoning is not None
                and self._reasoning.is_available()
            )

            route = resolve_route_kind(
                cleanup_reachable=cleanup_reachable,
                agent_reachable=agent_reachable,
                agent_invoked=agent_invoked,
                voice_agent_requested=voice_agent_requested,
            )

            logger.info(
                "Route resolved: %s (trigger=%s, agent_invoked=%s, text_preview='%s')",
                route.value,
                event.trigger.value,
                agent_invoked,
                raw_text[:50],
            )

            # ── Step 4: Execute Route ──
            if route == RouteKind.AGENT:
                result = await self._execute_agent_route(
                    raw_text, agent_name, custom_dict
                )
                elapsed = (time.perf_counter() - pipeline_start) * 1000
                return PipelineResult(
                    success=True,
                    route=RouteKind.AGENT,
                    raw_text=raw_text,
                    processed_text=result.response_text if result.success else raw_text,
                    agent_result=result,
                    vad_result=vad_result,
                    transcription_result=transcription,
                    elapsed_ms=elapsed,
                )

            elif route == RouteKind.CLEANUP:
                cleaned = await self._execute_cleanup_route(
                    raw_text, custom_dict
                )
                elapsed = (time.perf_counter() - pipeline_start) * 1000
                return PipelineResult(
                    success=True,
                    route=RouteKind.CLEANUP,
                    raw_text=raw_text,
                    processed_text=cleaned,
                    vad_result=vad_result,
                    transcription_result=transcription,
                    elapsed_ms=elapsed,
                )

            else:  # SKIP
                elapsed = (time.perf_counter() - pipeline_start) * 1000
                return PipelineResult(
                    success=True,
                    route=RouteKind.SKIP,
                    raw_text=raw_text,
                    processed_text=raw_text,
                    vad_result=vad_result,
                    transcription_result=transcription,
                    elapsed_ms=elapsed,
                )

        except Exception as e:
            elapsed = (time.perf_counter() - pipeline_start) * 1000
            logger.error("Pipeline error: %s", e, exc_info=True)
            return PipelineResult(
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
            )

        finally:
            await self.state.set_processing(False)

    # ── Route Executors ─────────────────────────────────────────────────

    async def _execute_agent_route(
        self,
        raw_text: str,
        agent_name: str,
        custom_dictionary: list,
    ) -> AgentResult:
        """
        Execute the AGENT route.

        The transcript BYPASSES cleanup and goes straight to the reasoning
        engine for command interpretation. This mirrors OpenWhispr's
        audioManager.js where voice agent recordings skip cleanup entirely
        (CLAUDE.md line 538).
        """
        # Strip the agent name from the command
        command_text = strip_agent_name(raw_text, agent_name)

        # Build the agent prompt
        context = await self.state.get_context_string(last_n=3)
        system_prompt = build_agent_prompt(
            raw_text=command_text,
            agent_name=agent_name,
            language=self._language,
            custom_dictionary=custom_dictionary,
            context=context,
        )

        # Run agent inference
        result = await self._reasoning.process_agent_command(
            transcript=command_text,
            agent_name=agent_name,
            system_prompt=system_prompt,
            context=context,
        )

        if result.success:
            # Store agent response in history
            await self.state.add_turn("assistant", result.response_text, {
                "route": "agent",
            })

        return result

    async def _execute_cleanup_route(
        self,
        raw_text: str,
        custom_dictionary: list,
    ) -> str:
        """
        Execute the CLEANUP route.

        Standard dictation cleanup — the LLM polishes the raw transcription
        into clean, well-formatted text. This mirrors OpenWhispr's
        processTranscription() path in audioManager.js.
        """
        system_prompt = build_cleanup_prompt(
            raw_text=raw_text,
            language=self._language,
            custom_dictionary=custom_dictionary,
        )

        result = await self._reasoning.process_text(
            text=raw_text,
            system_prompt=system_prompt,
        )

        if result.success:
            return result.text

        # If cleanup fails, fall back to raw text
        logger.warning("Cleanup failed, returning raw text: %s", result.error)
        return raw_text
