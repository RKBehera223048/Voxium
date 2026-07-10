"""
Voxium — LangGraph Orchestrator Wrapper
=======================================
Phase 2: Thin wrapper around the LangGraph state machine.

This module provides the backward-compatible `VoxiumOrchestrator` class
that integrates the LangGraph pipeline into the existing event loop structure.
It keeps the `AudioEvent` and `PipelineResult` dataclasses to ensure `app.py`
and the frontend don't break during the transition.
"""

from __future__ import annotations

import os
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict

from core.state_manager import StateManager
from core.intent_router import RouteKind, TriggerType, resolve_route_kind

# Import engine loaders so we can initialize them at start
from audio.whisper import get_engine as get_stt_engine
from llm.reasoning import LocalReasoningEngine
from llm.inference import ChatLLM
from llm.model_loader import load_model
from audio.tts import PiperTTS

logger = logging.getLogger(__name__)


@dataclass
class AudioEvent:
    """An audio payload submitted for processing."""
    audio_bytes: bytes
    trigger: TriggerType
    source_format: str = "webm"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Legacy backward-compatible result object for the frontend."""
    success: bool
    route: RouteKind
    raw_text: str = ""
    processed_text: str = ""
    agent_result: Any = None
    vad_result: Any = None
    transcription_result: Any = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class VoxiumOrchestrator:
    """
    Thin wrapper orchestrator that uses LangGraph under the hood.
    
    Maintains the legacy async queue interface so that `app.py` doesn't
    need to be rewritten yet.
    """
    
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        on_result: Optional[Callable[[PipelineResult], Any]] = None,
    ):
        self.state = state_manager or StateManager()
        self.on_result = on_result
        self._graph = None
        self._running = False
        self._queue: asyncio.Queue[AudioEvent] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        
        # Config
        self._agent_enabled = os.getenv("AGENT_ENABLED", "true").lower() == "true"
        self._cleanup_enabled = os.getenv("CLEANUP_ENABLED", "true").lower() == "true"
        self._language = os.getenv("PREFERRED_LANGUAGE", "en")
        
        # Eager initialization for engines
        self._stt_engine = None
        self._reasoning_engine = None
        self._tts = None

    async def start(self) -> None:
        """Initialize models and start the consumer loop."""
        if self._running:
            return
            
        logger.info("Starting LangGraph Orchestrator...")
        self._running = True
        
        # Build LangGraph
        from voxgraph.graph import build_graph
        self._graph = build_graph()
        
        # Pre-initialize heavy models (optional but good for UX)
        try:
            self._stt_engine = get_stt_engine()
            model_path = os.getenv("LLM_MODEL_PATH", "models/llm/model.gguf")
            if os.path.exists(model_path):
                self._reasoning_engine = LocalReasoningEngine(llm=ChatLLM(load_model(model_path)))
            else:
                logger.warning(f"Model path {model_path} does not exist. Reasoning engine not initialized.")
                self._reasoning_engine = None
            self._tts = PiperTTS()
            await self._tts.load()
        except Exception as e:
            logger.warning("Failed to pre-initialize some models: %s", e)
            
        # Start queue consumer
        self._task = asyncio.create_task(self._event_loop())

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("LangGraph Orchestrator stopped.")

    async def submit_audio(
        self,
        audio_bytes: bytes,
        trigger: TriggerType,
        source_format: str = "webm",
        **metadata,
    ) -> None:
        """Enqueue audio for asynchronous processing."""
        event = AudioEvent(
            audio_bytes=audio_bytes,
            trigger=trigger,
            source_format=source_format,
            metadata=metadata,
        )
        await self._queue.put(event)
        logger.debug("Enqueued audio event (trigger=%s)", trigger.value)

    async def process_audio_sync(
        self,
        audio_bytes: bytes,
        trigger: TriggerType,
        source_format: str = "webm",
    ) -> PipelineResult:
        """Process audio synchronously through LangGraph."""
        if not self._graph:
            raise RuntimeError("Orchestrator not started")
            
        from voxgraph.state import create_initial_state
        from core.prompts import get_custom_dictionary
        
        # Prepare initial state
        initial = create_initial_state(
            audio_bytes=audio_bytes,
            source_format=source_format,
            trigger=trigger.value,
            agent_name=await self.state.get_agent_name(),
            language=self._language,
            custom_dictionary=get_custom_dictionary()
        )
        
        # Set processing state
        await self.state.set_processing(True)
        
        try:
            # Generate a thread ID for LangGraph checkpointing
            thread_id = f"sync-{int(time.time() * 1000)}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke the graph
            final_state = await self._graph.ainvoke(initial, config)
            
            # Map back to legacy PipelineResult
            result = self._to_pipeline_result(final_state)
            
            # Dispatch callbacks
            if self.on_result:
                if asyncio.iscoroutinefunction(self.on_result):
                    await self.on_result(result)
                else:
                    self.on_result(result)
                    
            return result
            
        except Exception as e:
            logger.exception("Graph execution failed")
            return PipelineResult(
                success=False,
                route=RouteKind.SKIP,
                error=str(e)
            )
        finally:
            await self.state.set_processing(False)

    async def _event_loop(self) -> None:
        """Consume audio events from the queue."""
        _consecutive_errors = 0
        _MAX_BACKOFF = 30  # seconds
        while self._running:
            try:
                event = await self._queue.get()
                await self.process_audio_sync(
                    audio_bytes=event.audio_bytes,
                    trigger=event.trigger,
                    source_format=event.source_format
                )
                self._queue.task_done()
                _consecutive_errors = 0  # Reset on success
            except asyncio.CancelledError:
                break
            except Exception as e:
                _consecutive_errors += 1
                backoff = min(2 ** _consecutive_errors, _MAX_BACKOFF)
                logger.error(
                    "Error in orchestrator event loop (attempt %d, backoff %ds): %s",
                    _consecutive_errors, backoff, e,
                )
                await asyncio.sleep(backoff)

    def _to_pipeline_result(self, state: dict) -> PipelineResult:
        """Convert a LangGraph VoxiumState into a legacy PipelineResult."""
        route_str = state.get("route", "skip")
        
        route = RouteKind.SKIP
        if route_str == "agent":
            route = RouteKind.AGENT
        elif route_str == "cleanup":
            route = RouteKind.CLEANUP
            
        success = state.get("error") is None
        
        return PipelineResult(
            success=success,
            route=route,
            raw_text=state.get("transcription") or "",
            processed_text=state.get("processed_text") or "",
            vad_result=state.get("vad_result"),
            agent_result=None, # Not fully matched, but acceptable for frontend
            transcription_result=None,
            elapsed_ms=state.get("elapsed_ms", 0.0),
            error=state.get("error")
        )
