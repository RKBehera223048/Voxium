"""
Voxium — VAD (Voice Activity Detection) Ingestion Pipeline
============================================================
Two-tier speech pre-filtering that prevents Whisper/Parakeet hallucinations
on silence or low-speech recordings.

Architecture (ported from OpenWhispr):
    Tier 1: Fast RMS gate (localSpeechGate.js) — drops pure silence in <1ms
    Tier 2: Silero VAD frame analysis — neural network speech detection on
             512-sample windows, matching OpenWhispr's liveSpeakerIdentifier.js
             window size

The pipeline runs BEFORE any transcription engine, saving compute and
eliminating the "Thank you for watching" / phantom text hallucinations
that Whisper produces on near-silent input.
"""

from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from utils.audio_utils import (
    convert_to_pcm_16k,
    compute_rms,
    compute_peak,
    analyze_audio_gate,
    SpeechGateDecision,
    GateReason,
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VADConfig:
    """
    Voice Activity Detection configuration.
    Defaults match OpenWhispr's calibrated thresholds.
    """
    # Silero VAD model thresholds
    silero_threshold: float = 0.5
    min_speech_duration_ms: int = 300
    min_silence_duration_ms: int = 100

    # Silero VAD window size in samples (matches liveSpeakerIdentifier.js:15)
    window_size_samples: int = 512

    # RMS gate thresholds (from localSpeechGate.js)
    silence_rms_threshold: float = 0.002
    parakeet_silence_rms_threshold: float = 0.001  # Parakeet uses tighter threshold

    # Minimum speech requirements to proceed to transcription
    min_speech_windows: int = 1
    min_total_speech_ms: int = 300

    # Sample rate
    sample_rate: int = TARGET_SAMPLE_RATE

    @classmethod
    def from_env(cls) -> VADConfig:
        """Load configuration from environment variables."""
        return cls(
            silero_threshold=float(os.getenv("VAD_SILERO_THRESHOLD", "0.5")),
            min_speech_duration_ms=int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "300")),
            min_silence_duration_ms=int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "100")),
            silence_rms_threshold=float(os.getenv("VAD_SILENCE_RMS_THRESHOLD", "0.002")),
        )


# =============================================================================
# VAD Result
# =============================================================================

@dataclass
class SpeechSegment:
    """A contiguous segment of detected speech."""
    start_sample: int
    end_sample: int

    @property
    def start_seconds(self) -> float:
        return self.start_sample / TARGET_SAMPLE_RATE

    @property
    def end_seconds(self) -> float:
        return self.end_sample / TARGET_SAMPLE_RATE

    @property
    def duration_ms(self) -> float:
        return (self.end_sample - self.start_sample) / TARGET_SAMPLE_RATE * 1000


@dataclass
class VADResult:
    """Complete result from the VAD pipeline."""
    has_speech: bool
    speech_segments: List[SpeechSegment] = field(default_factory=list)
    total_speech_ms: float = 0.0
    total_silence_ms: float = 0.0
    gate_decision: Optional[SpeechGateDecision] = None
    silero_used: bool = False


# =============================================================================
# VAD Pipeline
# =============================================================================

class VADPipeline:
    """
    Two-tier Voice Activity Detection pipeline.

    Tier 1 (Fast RMS Gate):
        Port of OpenWhispr's localSpeechGate.js — runs in <1ms on any audio.
        Catches pure silence and obviously non-speech recordings.

    Tier 2 (Silero VAD):
        Neural network-based speech detection using Silero VAD model.
        Processes 512-sample windows (32ms at 16kHz), matching the window
        size used in OpenWhispr's liveSpeakerIdentifier.js.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self._config = config or VADConfig.from_env()
        self._silero_model = None
        self._silero_utils = None
        self._model_lock = asyncio.Lock()

    async def _ensure_silero_loaded(self) -> None:
        """
        Lazy-load Silero VAD model on first use.
        The model is cached by torch.hub in ~/.cache/torch/hub/
        """
        if self._silero_model is not None:
            return

        async with self._model_lock:
            # Double-check after acquiring lock
            if self._silero_model is not None:
                return

            logger.info("Loading Silero VAD model (first-time download may take a moment)...")

            # Run the blocking torch.hub.load in a thread to keep async
            loop = asyncio.get_event_loop()
            model, utils = await loop.run_in_executor(
                None, self._load_silero_model
            )
            self._silero_model = model
            self._silero_utils = utils
            logger.info("Silero VAD model loaded successfully")

    @staticmethod
    def _load_silero_model():
        """Load Silero VAD from torch.hub (blocking call)."""
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        return model, utils

    async def detect_speech(
        self,
        audio_bytes: bytes,
        source_format: str = "webm",
    ) -> VADResult:
        """
        Full VAD pipeline: RMS gate → Silero VAD → speech segment extraction.

        Args:
            audio_bytes: Raw audio file bytes (any format supported by pydub).
            source_format: Audio format hint (webm, wav, ogg, mp3, etc.).

        Returns:
            VADResult with speech detection results.
        """
        # ── Step 1: Convert to PCM float32 16kHz ──
        try:
            samples = await asyncio.get_event_loop().run_in_executor(
                None, convert_to_pcm_16k, audio_bytes, source_format
            )
        except Exception as e:
            logger.error("Audio conversion failed: %s", e)
            # If conversion fails, don't gate — let the transcription engine handle it
            return VADResult(has_speech=True)

        total_duration_ms = len(samples) / self._config.sample_rate * 1000

        # ── Step 2: Tier 1 — Fast RMS Gate ──
        gate_decision = analyze_audio_gate(samples, sample_rate=self._config.sample_rate)

        if gate_decision.skip:
            logger.info(
                "Tier 1 RMS gate rejected audio: reason=%s peak_rms=%.4f",
                gate_decision.reason.value,
                gate_decision.peak_rms,
            )
            return VADResult(
                has_speech=False,
                total_silence_ms=total_duration_ms,
                gate_decision=gate_decision,
                silero_used=False,
            )

        # ── Step 3: Tier 2 — Silero VAD ──
        await self._ensure_silero_loaded()

        speech_segments = await asyncio.get_event_loop().run_in_executor(
            None, self._run_silero_vad, samples
        )

        total_speech_ms = sum(seg.duration_ms for seg in speech_segments)
        total_silence_ms = total_duration_ms - total_speech_ms

        has_speech = (
            len(speech_segments) >= self._config.min_speech_windows
            and total_speech_ms >= self._config.min_total_speech_ms
        )

        if not has_speech:
            logger.info(
                "Tier 2 Silero VAD rejected audio: segments=%d total_speech_ms=%.0f",
                len(speech_segments),
                total_speech_ms,
            )

        return VADResult(
            has_speech=has_speech,
            speech_segments=speech_segments,
            total_speech_ms=total_speech_ms,
            total_silence_ms=total_silence_ms,
            gate_decision=gate_decision,
            silero_used=True,
        )

    def _run_silero_vad(self, samples: np.ndarray) -> List[SpeechSegment]:
        """
        Run Silero VAD on audio samples.
        Processes in 512-sample windows matching OpenWhispr's
        liveSpeakerIdentifier.js VAD_WINDOW_SIZE (line 15).
        """
        import torch

        model = self._silero_model
        model.reset_states()

        window_size = self._config.window_size_samples
        threshold = self._config.silero_threshold

        # Track speech/silence state (mirrors liveSpeakerIdentifier.js logic)
        speech_active = False
        segment_start = 0
        silence_windows = 0
        # ~24 silence windows to end speech (from liveSpeakerIdentifier.js:28)
        silence_windows_to_end = 24
        segments: List[SpeechSegment] = []

        for offset in range(0, len(samples) - window_size + 1, window_size):
            window = samples[offset : offset + window_size]
            tensor = torch.from_numpy(window).float()

            # Get speech probability from Silero
            with torch.no_grad():
                probability = model(tensor, self._config.sample_rate).item()

            if speech_active:
                if probability >= threshold * 0.5:
                    # Still in speech (use lower threshold to avoid cutting mid-word)
                    silence_windows = 0
                else:
                    silence_windows += 1
                    if silence_windows >= silence_windows_to_end:
                        # End of speech segment
                        segments.append(SpeechSegment(
                            start_sample=segment_start,
                            end_sample=offset + window_size,
                        ))
                        speech_active = False
                        silence_windows = 0
            else:
                if probability >= threshold:
                    # Start of speech
                    speech_active = True
                    segment_start = offset
                    silence_windows = 0

        # Handle speech that extends to the end of the audio
        if speech_active:
            segments.append(SpeechSegment(
                start_sample=segment_start,
                end_sample=len(samples),
            ))

        return segments

    def extract_speech_frames(
        self,
        samples: np.ndarray,
        vad_result: VADResult,
    ) -> np.ndarray:
        """
        Extract only the speech frames from audio, dropping silence.

        This produces a shorter audio buffer containing only voiced segments,
        ideal for feeding to transcription engines to avoid hallucinations
        on silence gaps.

        Args:
            samples: Original float32 audio samples.
            vad_result: VAD result with speech segment boundaries.

        Returns:
            np.ndarray of concatenated speech frames.
        """
        if not vad_result.speech_segments:
            return np.array([], dtype=np.float32)

        speech_chunks = [
            samples[seg.start_sample : seg.end_sample]
            for seg in vad_result.speech_segments
        ]
        return np.concatenate(speech_chunks)

    def should_transcribe(self, vad_result: VADResult) -> bool:
        """
        Decision gate: should we send this audio to the transcription engine?

        Requires at least 300ms of speech in at least 1 speech window.
        This prevents wasting compute on silence, background noise,
        or sub-word artifacts.
        """
        return vad_result.has_speech

    def reset(self) -> None:
        """Reset the Silero model state for a new session."""
        if self._silero_model is not None:
            self._silero_model.reset_states()
