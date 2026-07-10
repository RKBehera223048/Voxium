"""
Voxium — Wake Word Detection (OpenWakeWord)
=============================================
Lightweight, offline wake word detection using OpenWakeWord.

Replaces the previous approach of transcribing ALL audio then regex-matching
"Hey {AgentName}" in the transcript. This dedicated detector runs in < 10ms
per audio window, only triggering the full Whisper pipeline when the wake
word is actually spoken.

Uses OpenWakeWord models (ONNX) stored in models/wakeword/.
Default wake word: "hey voxium" (trained from "hey jarvis" base model).
"""

from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Target sample rate for OpenWakeWord (must be 16kHz)
WAKEWORD_SAMPLE_RATE = 16_000


@dataclass
class WakeWordResult:
    """Result from a wake word detection check."""
    detected: bool
    keyword: str = ""
    confidence: float = 0.0
    timestamp_ms: float = 0.0


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    # Detection threshold (0.0 - 1.0)
    threshold: float = 0.5
    # Custom model directory
    model_dir: str = "models/wakeword"
    # Wake word keywords to listen for
    keywords: List[str] = None
    # Agent name (used for keyword selection)
    agent_name: str = "Voxium"

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = ["hey jarvis"]  # Base model, closest to "hey voxium"

    @classmethod
    def from_env(cls) -> "WakeWordConfig":
        """Load configuration from environment variables."""
        return cls(
            threshold=float(os.getenv("WAKEWORD_THRESHOLD", "0.5")),
            model_dir=os.getenv("WAKEWORD_MODEL_DIR", "models/wakeword"),
            agent_name=os.getenv("AGENT_NAME", "Voxium"),
        )


class WakeWordDetector:
    """
    OpenWakeWord-based wake word detection engine.

    Provides real-time keyword detection on 16kHz PCM audio streams.
    Runs alongside VAD on the same audio buffer, consuming 512-sample
    windows with < 10ms latency per window.

    Usage:
        detector = WakeWordDetector()
        await detector.load()

        # Process audio windows (512 or 1280 samples at 16kHz)
        result = detector.detect(audio_chunk)
        if result.detected:
            print(f"Wake word '{result.keyword}' detected!")
    """

    def __init__(self, config: Optional[WakeWordConfig] = None):
        self._config = config or WakeWordConfig.from_env()
        self._model = None
        self._model_lock = asyncio.Lock()
        self._loaded = False

    async def load(self) -> None:
        """Load the OpenWakeWord model (lazy, thread-safe)."""
        if self._loaded:
            return

        async with self._model_lock:
            if self._loaded:
                return

            logger.info("Loading OpenWakeWord model...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            self._loaded = True
            logger.info(
                "OpenWakeWord loaded (keywords=%s, threshold=%.2f)",
                self._config.keywords,
                self._config.threshold,
            )

    def _load_model(self) -> None:
        """Blocking model load (run in executor)."""
        try:
            import openwakeword
            from openwakeword.model import Model

            # Ensure base models are downloaded
            openwakeword.utils.download_models()

            # Build model path list for custom models
            custom_model_paths = []
            model_dir = Path(self._config.model_dir)
            if model_dir.exists():
                for onnx_file in model_dir.glob("*.onnx"):
                    custom_model_paths.append(str(onnx_file))
                    logger.info("Found custom wake word model: %s", onnx_file.name)

            # Initialize model with configured keywords
            self._model = Model(
                wakeword_models=custom_model_paths if custom_model_paths else self._config.keywords,
                inference_framework="onnx",
            )

        except ImportError:
            logger.warning(
                "openwakeword not installed. Wake word detection disabled. "
                "Install with: pip install openwakeword"
            )
            self._model = None
        except Exception as e:
            logger.error("Failed to load OpenWakeWord: %s", e)
            self._model = None

    def detect(self, audio_chunk: np.ndarray) -> WakeWordResult:
        """
        Run wake word detection on an audio chunk.

        Args:
            audio_chunk: Float32 numpy array of audio samples at 16kHz.
                         Optimal sizes: 1280 samples (80ms) or 512 samples (32ms).

        Returns:
            WakeWordResult with detection status and confidence.
        """
        if self._model is None:
            return WakeWordResult(detected=False)

        # Convert float32 [-1, 1] to int16 range if needed
        if audio_chunk.dtype == np.float32:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk.astype(np.int16)

        # Run prediction
        prediction = self._model.predict(audio_int16)

        # Check each keyword against threshold
        for keyword, score in prediction.items():
            if score >= self._config.threshold:
                logger.info(
                    "Wake word detected: '%s' (confidence=%.3f)",
                    keyword, score,
                )
                # Reset model state to avoid repeated detections
                self._model.reset()
                return WakeWordResult(
                    detected=True,
                    keyword=keyword,
                    confidence=score,
                )

        return WakeWordResult(detected=False)

    async def detect_async(self, audio_chunk: np.ndarray) -> WakeWordResult:
        """Async wrapper for detect() — runs in executor."""
        if self._model is None:
            return WakeWordResult(detected=False)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect, audio_chunk)

    def reset(self) -> None:
        """Reset the model state for a new detection session."""
        if self._model is not None:
            self._model.reset()

    def is_available(self) -> bool:
        """Check if wake word detection is available."""
        return self._model is not None and self._loaded

    def get_info(self) -> Dict:
        """Get model info for status reporting."""
        return {
            "engine": "openwakeword",
            "loaded": self._loaded,
            "available": self.is_available(),
            "keywords": self._config.keywords,
            "threshold": self._config.threshold,
        }
