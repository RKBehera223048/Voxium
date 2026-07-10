"""
Voxium — Transcription Engines
================================
Dual-engine transcription supporting both faster-whisper and NVIDIA Parakeet
(via sherpa-onnx Python bindings).

Architecture ported from OpenWhispr:
    - WhisperEngine: Wraps faster-whisper with VAD pre-filter integration,
      custom dictionary injection, and silence gate.
    - ParakeetEngine: Uses sherpa-onnx OfflineRecognizer with INT8 TDT models
      (encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt).
      Ported from parakeetServer.js / parakeetWsServer.js.

Both engines implement the TranscriptionEngine protocol, so the orchestrator
can swap between them based on the STT_ENGINE environment variable.
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np

from utils.audio_utils import (
    convert_to_pcm_16k,
    compute_rms,
    TARGET_SAMPLE_RATE,
    BYTES_PER_SAMPLE_FLOAT32,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Transcription Result
# =============================================================================

@dataclass
class TranscriptionResult:
    """Result from any transcription engine."""
    success: bool
    text: str = ""
    raw_text: str = ""
    language: str = ""
    engine: str = ""
    elapsed_ms: float = 0.0
    segments: List[dict] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# Abstract Base
# =============================================================================

class TranscriptionEngine(ABC):
    """
    Protocol for transcription engines.
    Both Whisper and Parakeet implement this interface.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
        source_format: str = "webm",
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine and its model are available."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
        ...


# =============================================================================
# Whisper Engine (faster-whisper)
# =============================================================================

class WhisperEngine(TranscriptionEngine):
    """
    Speech-to-text using faster-whisper (CTranslate2 backend).

    Integrates with the VAD pipeline from ingestion.py for pre-filtering,
    and supports custom dictionary injection as an initial prompt
    (ported from OpenWhispr's audioManager.js custom dictionary feature).
    """

    def __init__(
        self,
        model_size: str = "base",
        model_dir: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self._model_size = model_size
        self._model_dir = model_dir or os.getenv("WHISPER_MODEL_DIR", "models/stt")
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._model_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Lazy-load the Whisper model on first transcription."""
        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is not None:
                return

            logger.info(
                "Loading Whisper model: size=%s device=%s",
                self._model_size, self._device,
            )

            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None, self._load_model
            )
            logger.info("Whisper model loaded successfully")

    def _load_model(self):
        """Blocking model load (run in executor)."""
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._model_dir,
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
        source_format: str = "webm",
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using faster-whisper.

        Args:
            audio_bytes: Raw audio file bytes.
            language: Language code (e.g., "en", "auto").
            source_format: Audio format hint.
            initial_prompt: Custom dictionary / context prompt for biasing.
                           (Port of OpenWhispr's custom dictionary feature)
        """
        await self._ensure_loaded()

        start_time = time.perf_counter()

        try:
            # Convert to PCM float32 16kHz
            loop = asyncio.get_running_loop()
            samples = await loop.run_in_executor(
                None, convert_to_pcm_16k, audio_bytes, source_format
            )

            # Run transcription in executor (faster-whisper is CPU/GPU bound)
            segments_result, info = await loop.run_in_executor(
                None,
                self._run_transcription,
                samples,
                language,
                initial_prompt,
            )

            # Collect segments
            segments = []
            full_text_parts = []
            for segment in segments_result:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })
                full_text_parts.append(segment.text.strip())

            full_text = " ".join(full_text_parts).strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if not full_text:
                return TranscriptionResult(
                    success=False,
                    error="No audio detected",
                    engine="whisper",
                    elapsed_ms=elapsed_ms,
                )

            return TranscriptionResult(
                success=True,
                text=full_text,
                raw_text=full_text,
                language=info.language or language,
                engine="whisper",
                elapsed_ms=elapsed_ms,
                segments=segments,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Whisper transcription failed: %s", e)
            return TranscriptionResult(
                success=False,
                error=str(e),
                engine="whisper",
                elapsed_ms=elapsed_ms,
            )

    def _run_transcription(
        self,
        samples: np.ndarray,
        language: str,
        initial_prompt: Optional[str],
    ):
        """Blocking transcription call (run in executor)."""
        lang = language if language != "auto" else None

        segments, info = self._model.transcribe(
            samples,
            language=lang,
            beam_size=5,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
        )
        # Materialize the generator to avoid issues across threads
        return list(segments), info

    def is_available(self) -> bool:
        """Check if faster-whisper can be imported."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def get_model_info(self) -> dict:
        return {
            "engine": "whisper",
            "model_size": self._model_size,
            "model_dir": self._model_dir,
            "device": self._device,
            "loaded": self._model is not None,
        }


# =============================================================================
# Parakeet Engine (sherpa-onnx)
# =============================================================================

# Ported from OpenWhispr parakeetServer.js (lines 15-19)
PARAKEET_SAMPLE_RATE = 16_000
PARAKEET_MAX_SEGMENT_SECONDS = 15
PARAKEET_MAX_SEGMENT_SAMPLES = PARAKEET_MAX_SEGMENT_SECONDS * PARAKEET_SAMPLE_RATE
PARAKEET_SILENCE_RMS_THRESHOLD = 0.001


class ParakeetEngine(TranscriptionEngine):
    """
    Speech-to-text using NVIDIA Parakeet via sherpa-onnx Python bindings.

    Ported from OpenWhispr's parakeetServer.js and parakeetWsServer.js.
    Uses the direct Python API instead of a subprocess + WebSocket wrapper,
    which is simpler and eliminates IPC overhead.

    Expected model files in model_dir:
        - encoder.int8.onnx
        - decoder.int8.onnx
        - joiner.int8.onnx
        - tokens.txt

    Thread count auto-tuning:
        min(4, floor(cpu_count * 0.75))
        Ported from parakeetWsServer.js line 79.
    """

    # Required model files (from parakeetServer.js:40-45)
    REQUIRED_FILES = [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ]

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_name: str = "parakeet-tdt-0.6b-v3",
        num_threads: int = 0,
    ):
        self._model_name = model_name
        self._base_dir = model_dir or os.getenv("PARAKEET_MODEL_DIR", "models/stt/parakeet")
        self._model_dir = os.path.join(self._base_dir, model_name)
        self._num_threads = num_threads or self._auto_threads()
        self._recognizer = None
        self._recognizer_lock = asyncio.Lock()

    @staticmethod
    def _auto_threads() -> int:
        """
        Auto-calculate optimal thread count.
        Port of parakeetWsServer.js line 79:
            min(4, floor(cpu_count * 0.75))
        """
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        return max(1, min(4, int(cpu_count * 0.75)))

    def is_model_downloaded(self) -> bool:
        """
        Check if all required model files exist.
        Port of parakeetServer.js isModelDownloaded (lines 38-56).
        """
        if not os.path.isdir(self._model_dir):
            return False
        return all(
            os.path.isfile(os.path.join(self._model_dir, f))
            for f in self.REQUIRED_FILES
        )

    async def _ensure_loaded(self) -> None:
        """Lazy-load the sherpa-onnx recognizer on first use."""
        if self._recognizer is not None:
            return

        async with self._recognizer_lock:
            if self._recognizer is not None:
                return

            if not self.is_model_downloaded():
                raise FileNotFoundError(
                    f'Parakeet model "{self._model_name}" not found in '
                    f"{self._model_dir}. Required files: {self.REQUIRED_FILES}"
                )

            logger.info(
                "Loading Parakeet model: %s (threads=%d)",
                self._model_name, self._num_threads,
            )

            loop = asyncio.get_running_loop()
            self._recognizer = await loop.run_in_executor(
                None, self._create_recognizer
            )
            logger.info("Parakeet model loaded successfully")

    def _create_recognizer(self):
        """Create the sherpa-onnx offline recognizer (blocking)."""
        import sherpa_onnx

        model_dir = self._model_dir

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(model_dir, "encoder.int8.onnx"),
            decoder=os.path.join(model_dir, "decoder.int8.onnx"),
            joiner=os.path.join(model_dir, "joiner.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=self._num_threads,
            sample_rate=PARAKEET_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        return recognizer

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
        source_format: str = "webm",
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Parakeet via sherpa-onnx.

        Handles long audio via segmentation, matching OpenWhispr's
        parakeetServer.js (lines 114-149):
            - MAX_SEGMENT_SECONDS = 15
            - Segments longer than 15s are split and processed sequentially
            - Each segment's RMS is checked against SILENCE_RMS_THRESHOLD
        """
        await self._ensure_loaded()

        start_time = time.perf_counter()

        try:
            # Convert to PCM float32 16kHz
            loop = asyncio.get_running_loop()
            samples = await loop.run_in_executor(
                None, convert_to_pcm_16k, audio_bytes, source_format
            )

            duration_seconds = len(samples) / PARAKEET_SAMPLE_RATE

            # Silence RMS check (parakeetServer.js:110-112)
            rms = compute_rms(samples)
            logger.debug("Parakeet audio analysis: duration=%.1fs rms=%.4f", duration_seconds, rms)

            if rms < PARAKEET_SILENCE_RMS_THRESHOLD:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return TranscriptionResult(
                    success=True,
                    text="",
                    engine="parakeet",
                    elapsed_ms=elapsed_ms,
                )

            # Transcribe (with segmentation for long audio)
            if len(samples) <= PARAKEET_MAX_SEGMENT_SAMPLES:
                text = await loop.run_in_executor(
                    None, self._transcribe_segment, samples
                )
            else:
                text = await loop.run_in_executor(
                    None, self._transcribe_segmented, samples
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if not text.strip():
                logger.warning(
                    "Parakeet returned empty text for non-silent audio "
                    "(duration=%.1fs rms=%.4f)",
                    duration_seconds, rms,
                )
                return TranscriptionResult(
                    success=False,
                    error="No audio detected",
                    engine="parakeet",
                    elapsed_ms=elapsed_ms,
                )

            return TranscriptionResult(
                success=True,
                text=text.strip(),
                raw_text=text.strip(),
                language=language,
                engine="parakeet",
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Parakeet transcription failed: %s", e)
            return TranscriptionResult(
                success=False,
                error=str(e),
                engine="parakeet",
                elapsed_ms=elapsed_ms,
            )

    def _transcribe_segment(self, samples: np.ndarray) -> str:
        """Transcribe a single audio segment (blocking)."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(PARAKEET_SAMPLE_RATE, samples.tolist())
        self._recognizer.decode(stream)
        return stream.result.text

    def _transcribe_segmented(self, samples: np.ndarray) -> str:
        """
        Transcribe long audio by splitting into segments.
        Port of parakeetServer.js lines 126-149.
        """
        texts = []
        total_samples = len(samples)

        logger.debug(
            "Parakeet segmenting long audio: duration=%.1fs segments=%d",
            total_samples / PARAKEET_SAMPLE_RATE,
            int(np.ceil(total_samples / PARAKEET_MAX_SEGMENT_SAMPLES)),
        )

        for offset in range(0, total_samples, PARAKEET_MAX_SEGMENT_SAMPLES):
            end = min(offset + PARAKEET_MAX_SEGMENT_SAMPLES, total_samples)
            segment = samples[offset:end]

            text = self._transcribe_segment(segment)
            if text.strip():
                texts.append(text.strip())
            else:
                logger.warning(
                    "Parakeet segment %d returned empty text",
                    offset // PARAKEET_MAX_SEGMENT_SAMPLES,
                )

        return " ".join(texts)

    def is_available(self) -> bool:
        """Check if sherpa-onnx is importable and model files exist."""
        try:
            import sherpa_onnx  # noqa: F401
            return self.is_model_downloaded()
        except ImportError:
            return False

    def get_model_info(self) -> dict:
        return {
            "engine": "parakeet",
            "model_name": self._model_name,
            "model_dir": self._model_dir,
            "num_threads": self._num_threads,
            "model_downloaded": self.is_model_downloaded(),
            "loaded": self._recognizer is not None,
        }


# =============================================================================
# Engine Factory
# =============================================================================

def get_engine(provider: Optional[str] = None) -> TranscriptionEngine:
    """
    Factory function to get the configured transcription engine.

    Args:
        provider: Override the engine selection. Options: "whisper", "parakeet".
                  If None, reads from STT_ENGINE env var (default: "whisper").

    Returns:
        Configured TranscriptionEngine instance.
    """
    engine_name = provider or os.getenv("STT_ENGINE", "whisper")

    if engine_name == "parakeet" or engine_name == "nvidia":
        return ParakeetEngine(
            model_name=os.getenv("PARAKEET_MODEL_NAME", "parakeet-tdt-0.6b-v3"),
            model_dir=os.getenv("PARAKEET_MODEL_DIR", "models/stt/parakeet"),
            num_threads=int(os.getenv("PARAKEET_NUM_THREADS", "0")),
        )
    else:
        return WhisperEngine(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            model_dir=os.getenv("WHISPER_MODEL_DIR", "models/stt"),
        )
