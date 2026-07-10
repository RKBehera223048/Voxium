"""
Voxium — Text-to-Speech Engine (Piper TTS)
=============================================
Offline text-to-speech using Piper TTS — a fast, lightweight C++ engine
with ONNX voice models.

Features:
    - Offline-only: all models run locally
    - Low latency: < 300ms for first audio chunk
    - Streaming synthesis: chunks text into sentences for real-time playback
    - Multiple voices: supports all Piper ONNX voice models
    - CPU-only: no GPU required (~200MB RAM)

Models are stored in models/tts/ and downloaded from:
    https://github.com/rhasspy/piper/blob/master/VOICES.md

Default voice: en_US-lessac-medium (~75MB, good quality)
"""

from __future__ import annotations

import os
import io
import wave
import asyncio
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, AsyncIterator, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Piper default sample rate
PIPER_SAMPLE_RATE = 22050


@dataclass
class VoiceInfo:
    """Information about an available TTS voice."""
    name: str
    language: str
    quality: str  # "low", "medium", "high"
    model_path: str
    config_path: str
    sample_rate: int = PIPER_SAMPLE_RATE


@dataclass
class SynthesisResult:
    """Result from TTS synthesis."""
    success: bool
    audio_bytes: bytes = b""
    sample_rate: int = PIPER_SAMPLE_RATE
    duration_ms: float = 0.0
    format: str = "wav"
    error: Optional[str] = None


@dataclass
class TTSConfig:
    """Configuration for the TTS engine."""
    model_dir: str = "models/tts"
    default_voice: str = "en_US-lessac-medium"
    # Speed adjustment (1.0 = normal, < 1.0 = slower, > 1.0 = faster)
    speed: float = 1.0
    # Sentence splitting for streaming
    chunk_by_sentence: bool = True
    # Maximum characters per chunk for streaming
    max_chunk_chars: int = 200

    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Load configuration from environment variables."""
        return cls(
            model_dir=os.getenv("TTS_MODEL_DIR", "models/tts"),
            default_voice=os.getenv("TTS_DEFAULT_VOICE", "en_US-lessac-medium"),
            speed=float(os.getenv("TTS_SPEED", "1.0")),
        )


class PiperTTS:
    """
    Piper TTS engine wrapper for offline text-to-speech.

    Uses the piper-tts Python package which wraps the Piper C++ binary
    for fast ONNX-based speech synthesis.

    Usage:
        tts = PiperTTS()
        await tts.load()

        # Full synthesis
        result = await tts.synthesize("Hello, how can I help you?")

        # Streaming synthesis (for real-time playback)
        async for chunk in tts.synthesize_streaming("Long response text..."):
            play_audio(chunk)
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self._config = config or TTSConfig.from_env()
        self._voice = None
        self._voice_lock = asyncio.Lock()
        self._loaded = False
        self._available_voices: Dict[str, VoiceInfo] = {}

    async def load(self, voice_name: Optional[str] = None) -> None:
        """Load a Piper voice model (lazy, thread-safe)."""
        voice_name = voice_name or self._config.default_voice

        if self._loaded and self._voice is not None:
            return

        async with self._voice_lock:
            if self._loaded and self._voice is not None:
                return

            logger.info("Loading Piper TTS voice: %s", voice_name)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_voice, voice_name)

    def _load_voice(self, voice_name: str) -> None:
        """Blocking voice model load (run in executor)."""
        try:
            from piper import PiperVoice

            model_dir = Path(self._config.model_dir)
            model_path = model_dir / f"{voice_name}.onnx"
            config_path = model_dir / f"{voice_name}.onnx.json"

            if not model_path.exists():
                # Try to find any .onnx file in the model directory
                onnx_files = list(model_dir.glob("*.onnx"))
                if onnx_files:
                    model_path = onnx_files[0]
                    config_path = Path(str(model_path) + ".json")
                    logger.info("Using found voice model: %s", model_path.name)
                else:
                    logger.warning(
                        "TTS model not found at %s. "
                        "Download a voice from https://github.com/rhasspy/piper/blob/master/VOICES.md "
                        "and place the .onnx and .onnx.json files in %s",
                        model_path, model_dir,
                    )
                    self._voice = None
                    self._loaded = False
                    return

            self._voice = PiperVoice.load(
                str(model_path),
                config_path=str(config_path) if config_path.exists() else None,
            )
            self._loaded = True
            logger.info("Piper TTS loaded: %s", model_path.name)

        except ImportError:
            logger.warning(
                "piper-tts not installed. TTS disabled. "
                "Install with: pip install piper-tts"
            )
            self._voice = None
            self._loaded = False
        except Exception as e:
            logger.error("Failed to load Piper TTS: %s", e)
            self._voice = None
            self._loaded = False

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize.
            voice: Optional voice name override.

        Returns:
            SynthesisResult with WAV audio bytes.
        """
        if not text or not text.strip():
            return SynthesisResult(success=False, error="Empty text")

        # Load voice if needed
        if not self._loaded:
            await self.load(voice)

        if self._voice is None:
            return SynthesisResult(
                success=False,
                error="TTS engine not available. Ensure piper-tts is installed and models are in models/tts/.",
            )

        try:
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None, self._synthesize_blocking, text
            )

            if not audio_bytes:
                return SynthesisResult(success=False, error="Synthesis produced no audio")

            # Calculate duration from WAV data
            duration_ms = len(audio_bytes) / (PIPER_SAMPLE_RATE * 2) * 1000  # 16-bit = 2 bytes/sample

            return SynthesisResult(
                success=True,
                audio_bytes=audio_bytes,
                sample_rate=PIPER_SAMPLE_RATE,
                duration_ms=duration_ms,
                format="wav",
            )

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return SynthesisResult(success=False, error=str(e))

    def _synthesize_blocking(self, text: str) -> bytes:
        """Blocking synthesis call (run in executor)."""
        # Synthesize to WAV bytes in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setframerate(self._voice.config.sample_rate)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setnchannels(1)  # Mono

            self._voice.synthesize(
                text,
                wav_file,
                length_scale=1.0 / self._config.speed if self._config.speed != 1.0 else None,
            )

        return wav_buffer.getvalue()

    async def synthesize_streaming(
        self,
        text: str,
    ) -> AsyncIterator[bytes]:
        """
        Stream TTS synthesis sentence-by-sentence for real-time playback.

        Splits text into sentence-sized chunks and synthesizes each one
        independently, yielding WAV bytes as they're ready. This allows
        the first audio chunk to play while later sentences are still
        being synthesized.

        Args:
            text: Full text to synthesize.

        Yields:
            WAV bytes for each sentence chunk.
        """
        if not self._loaded:
            await self.load()

        if self._voice is None:
            return

        # Split text into sentences
        chunks = self._split_into_chunks(text)

        for chunk in chunks:
            if not chunk.strip():
                continue

            result = await self.synthesize(chunk)
            if result.success:
                yield result.audio_bytes

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into sentence-sized chunks for streaming synthesis."""
        import re

        if not self._config.chunk_by_sentence:
            # Split by character count
            chunks = []
            while text:
                chunks.append(text[:self._config.max_chunk_chars])
                text = text[self._config.max_chunk_chars:]
            return chunks

        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Merge very short sentences
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > self._config.max_chunk_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks or [text]

    def list_voices(self) -> List[VoiceInfo]:
        """List available voice models in the model directory."""
        model_dir = Path(self._config.model_dir)
        voices = []

        if not model_dir.exists():
            return voices

        for onnx_file in sorted(model_dir.glob("*.onnx")):
            name = onnx_file.stem
            # Parse voice name: lang_REGION-speaker-quality
            parts = name.split("-")
            language = parts[0] if parts else "unknown"
            quality = parts[-1] if len(parts) > 2 else "medium"

            voices.append(VoiceInfo(
                name=name,
                language=language,
                quality=quality,
                model_path=str(onnx_file),
                config_path=str(onnx_file) + ".json",
            ))

        return voices

    def is_available(self) -> bool:
        """Check if TTS engine is loaded and ready."""
        return self._voice is not None and self._loaded

    def get_info(self) -> Dict:
        """Get engine info for status reporting."""
        return {
            "engine": "piper",
            "loaded": self._loaded,
            "available": self.is_available(),
            "model_dir": self._config.model_dir,
            "default_voice": self._config.default_voice,
            "speed": self._config.speed,
            "voices": [v.name for v in self.list_voices()],
        }
