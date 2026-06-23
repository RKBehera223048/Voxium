"""
Voxium — Audio Utilities
=========================
PCM conversion, sample rate adjustment, and the fast RMS-based speech gate
ported from OpenWhispr's localSpeechGate.js.

The SpeechGateState class provides a lightweight, sub-millisecond pre-filter
that drops obviously silent or insufficient-speech recordings BEFORE they reach
the heavier Silero VAD or Whisper/Parakeet transcription engines.
"""

from __future__ import annotations

import io
import struct
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — ported from OpenWhispr localSpeechGate.js (lines 1-4)
# =============================================================================

SILENCE_RMS_THRESHOLD = 0.002
SPEECH_WINDOW_RMS_THRESHOLD = 0.003
SPEECH_WINDOW_PEAK_THRESHOLD = 0.02
STRONG_SPEECH_RMS_THRESHOLD = 0.006

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
BYTES_PER_SAMPLE_INT16 = 2
BYTES_PER_SAMPLE_FLOAT32 = 4


# =============================================================================
# Audio Format Conversion
# =============================================================================

def convert_to_pcm_16k(
    audio_bytes: bytes,
    source_format: str = "webm",
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Convert any audio format (webm, ogg, mp3, wav, etc.) to 16 kHz mono PCM
    float32 samples using pydub + ffmpeg.

    Returns:
        np.ndarray of float32 samples in [-1.0, 1.0] range at 16 kHz mono.

    Raises:
        RuntimeError: If ffmpeg is not available or conversion fails.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is required for audio conversion. "
            "Install it with: pip install pydub"
        )

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
    except Exception:
        # Fallback: let pydub/ffmpeg auto-detect the format
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Normalize to 16 kHz mono
    audio = audio.set_frame_rate(target_sample_rate).set_channels(TARGET_CHANNELS)

    # Extract raw PCM int16 samples and convert to float32 [-1.0, 1.0]
    raw_data = audio.raw_data
    int16_samples = np.frombuffer(raw_data, dtype=np.int16)
    float32_samples = int16_samples.astype(np.float32) / 32768.0

    return float32_samples


def pcm_int16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw PCM int16 bytes to float32 numpy array in [-1.0, 1.0]."""
    int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_samples.astype(np.float32) / 32768.0


def float32_to_pcm_int16(samples: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw PCM int16 bytes."""
    clamped = np.clip(samples, -1.0, 1.0)
    int16_samples = (clamped * 32767).astype(np.int16)
    return int16_samples.tobytes()


def create_wav_header(
    data_size: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Create a WAV file header.
    Ported from OpenWhispr diarization.js _createWavHeader (lines 550-571).
    """
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    byte_rate = sample_rate * block_align

    header = struct.pack(
        "<4sI4s"  # RIFF header
        "4sIHHIIHH"  # fmt chunk
        "4sI",  # data chunk header
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def samples_to_wav_bytes(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> bytes:
    """Convert float32 samples to a complete WAV file in memory."""
    pcm_data = float32_to_pcm_int16(samples)
    header = create_wav_header(len(pcm_data), sample_rate)
    return header + pcm_data


# =============================================================================
# RMS and Peak Computation
# =============================================================================

def compute_rms(samples: np.ndarray) -> float:
    """
    Compute Root Mean Square energy of audio samples.

    Ported from OpenWhispr parakeetServer.js computeFloat32RMS
    and audioManager.js silence detection interval.
    """
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def compute_peak(samples: np.ndarray) -> float:
    """Compute peak absolute amplitude of audio samples."""
    if len(samples) == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


# =============================================================================
# SpeechGateState — ported from OpenWhispr localSpeechGate.js
# =============================================================================

class GateReason(str, Enum):
    """Reason for the speech gate decision."""
    SILENCE = "silence"
    INSUFFICIENT_SPEECH = "insufficient_speech"
    SPEECH_DETECTED = "speech_detected"
    UNAVAILABLE = "unavailable"


@dataclass
class SpeechGateDecision:
    """Result of the speech gate evaluation."""
    skip: bool
    reason: GateReason
    peak_rms: float = 0.0
    peak_amplitude: float = 0.0
    window_count: int = 0
    speech_window_count: int = 0
    max_consecutive_speech_windows: int = 0


@dataclass
class SpeechGateState:
    """
    Windowed accumulator for fast speech/silence classification.

    Direct port of OpenWhispr's createLocalSpeechGateState /
    recordLocalSpeechWindow / getLocalSpeechGateDecision from
    localSpeechGate.js (lines 6-63).

    Usage:
        gate = SpeechGateState()
        for window in audio_windows:
            rms = compute_rms(window)
            peak = compute_peak(window)
            gate.record_window(rms, peak)
        decision = gate.get_decision()
        if decision.skip:
            print(f"Skipping: {decision.reason}")
    """
    peak_rms: float = 0.0
    peak_amplitude: float = 0.0
    window_count: int = 0
    speech_window_count: int = 0
    consecutive_speech_windows: int = 0
    max_consecutive_speech_windows: int = 0

    def record_window(self, rms: float, peak: float) -> None:
        """
        Record a single analysis window.
        Port of recordLocalSpeechWindow (localSpeechGate.js:15-37).
        """
        self.window_count += 1
        self.peak_rms = max(self.peak_rms, rms)
        self.peak_amplitude = max(self.peak_amplitude, peak)

        is_speech_window = (
            rms >= SPEECH_WINDOW_RMS_THRESHOLD
            and peak >= SPEECH_WINDOW_PEAK_THRESHOLD
        )

        if not is_speech_window:
            self.consecutive_speech_windows = 0
            return

        self.speech_window_count += 1
        self.consecutive_speech_windows += 1
        self.max_consecutive_speech_windows = max(
            self.max_consecutive_speech_windows,
            self.consecutive_speech_windows,
        )

    def get_decision(self) -> SpeechGateDecision:
        """
        Evaluate accumulated windows and decide whether to skip transcription.
        Port of getLocalSpeechGateDecision (localSpeechGate.js:39-63).
        """
        metrics = dict(
            peak_rms=self.peak_rms,
            peak_amplitude=self.peak_amplitude,
            window_count=self.window_count,
            speech_window_count=self.speech_window_count,
            max_consecutive_speech_windows=self.max_consecutive_speech_windows,
        )

        if self.window_count == 0:
            return SpeechGateDecision(skip=False, reason=GateReason.UNAVAILABLE, **metrics)

        if self.peak_rms < SILENCE_RMS_THRESHOLD:
            return SpeechGateDecision(skip=True, reason=GateReason.SILENCE, **metrics)

        has_speech = (
            self.speech_window_count >= 1
            or self.peak_rms >= STRONG_SPEECH_RMS_THRESHOLD
        )

        if not has_speech:
            return SpeechGateDecision(
                skip=True, reason=GateReason.INSUFFICIENT_SPEECH, **metrics
            )

        return SpeechGateDecision(
            skip=False, reason=GateReason.SPEECH_DETECTED, **metrics
        )

    def reset(self) -> None:
        """Reset accumulator for a new recording session."""
        self.peak_rms = 0.0
        self.peak_amplitude = 0.0
        self.window_count = 0
        self.speech_window_count = 0
        self.consecutive_speech_windows = 0
        self.max_consecutive_speech_windows = 0


def analyze_audio_gate(
    samples: np.ndarray,
    window_duration_ms: int = 100,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> SpeechGateDecision:
    """
    Convenience function: run the full speech gate analysis on an audio buffer.

    This mirrors OpenWhispr's audioManager.js where the gate runs on 100ms
    intervals during recording (lines 411-423).

    Args:
        samples: float32 audio samples.
        window_duration_ms: Analysis window size in milliseconds.
        sample_rate: Sample rate of the audio.

    Returns:
        SpeechGateDecision with skip/reason.
    """
    gate = SpeechGateState()
    window_size = int(sample_rate * window_duration_ms / 1000)

    for start in range(0, len(samples), window_size):
        window = samples[start : start + window_size]
        if len(window) < window_size // 2:
            break  # Skip tiny trailing windows
        rms = compute_rms(window)
        peak = compute_peak(window)
        gate.record_window(rms, peak)

    decision = gate.get_decision()
    logger.debug(
        "Speech gate result: skip=%s reason=%s peak_rms=%.4f windows=%d speech_windows=%d",
        decision.skip,
        decision.reason.value,
        decision.peak_rms,
        decision.window_count,
        decision.speech_window_count,
    )
    return decision
