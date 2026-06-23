"""
Voxium — Text-to-Speech Synthesis (Stub)
==========================================
Placeholder for TTS generation and chunking.
To be implemented with a local TTS engine (e.g., Coqui TTS, Piper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result from TTS synthesis."""
    success: bool
    audio_bytes: bytes = b""
    sample_rate: int = 22050
    duration_ms: float = 0.0
    error: Optional[str] = None


class SynthesisEngine:
    """Local text-to-speech engine (stub — to be implemented)."""

    async def synthesize(self, text: str, voice: str = "default") -> SynthesisResult:
        """Convert text to speech audio."""
        logger.warning("TTS synthesis not yet implemented")
        return SynthesisResult(success=False, error="TTS not implemented")

    def is_available(self) -> bool:
        return False
