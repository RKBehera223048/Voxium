"""
Voxium — Speaker Diarization Pipeline
========================================
On-device speaker labeling using pyannote.audio for segmentation/embedding,
with cross-session speaker recognition via SQLite-persisted profiles.

Ported from three OpenWhispr modules:
    1. diarization.js — Offline diarization with pyannote segmentation
       (which OpenWhispr runs via sherpa-onnx CLI wrapping pyannote models)
    2. speakerEmbeddings.js — Speaker embedding extraction with cosine
       similarity, centroid computation, and duration guards
    3. liveSpeakerIdentifier.js — Real-time speaker identification with
       VAD-gated extraction, transient cluster management, and stored
       profile matching

Since pyannote.audio IS the native Python implementation of the models
that OpenWhispr wraps through sherpa-onnx, we use it directly — which
gives us a simpler, more Pythonic API.
"""

from __future__ import annotations

import os
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

from memory.sqlite import (
    VoxiumDB as SpeakerProfileDB,
    SpeakerProfile,
    cosine_similarity,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — ported from OpenWhispr
# =============================================================================

# From speakerEmbeddings.js (lines 7-12)
SAMPLE_RATE = 16_000
MIN_SEGMENT_SECONDS = 1.5
MIN_SEGMENT_SAMPLES = int(SAMPLE_RATE * MIN_SEGMENT_SECONDS)
MAX_EMBEDDING_SECONDS = 8
MAX_EMBEDDING_SAMPLES = SAMPLE_RATE * MAX_EMBEDDING_SECONDS

# From liveSpeakerIdentifier.js (lines 29-30)
MATCH_THRESHOLD = 0.65
MATCH_MARGIN = 0.03

# From liveSpeakerIdentifier.js (lines 17-23)
LIVE_IDENTIFICATION_MIN_SECONDS = 1.6
LIVE_IDENTIFICATION_MIN_SAMPLES = int(SAMPLE_RATE * LIVE_IDENTIFICATION_MIN_SECONDS)
LIVE_IDENTIFICATION_INTERVAL_SECONDS = 1.0

# From liveSpeakerIdentifier.js (lines 26-28)
SPEECH_THRESHOLD = 0.15
SILENCE_THRESHOLD = 0.08
SILENCE_WINDOWS_TO_END = 24

# From .env defaults
DEFAULT_MAX_SPEAKERS = 10


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class DiarizedSegment:
    """A single speaker-labeled audio segment."""
    start: float  # seconds
    end: float    # seconds
    speaker: str  # e.g., "speaker_0", "speaker_1"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EnrichedSegment:
    """A transcript segment enriched with speaker labels."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    speaker_name: Optional[str] = None
    source: str = "mic"  # "mic" or "system"


@dataclass
class SpeakerIdentification:
    """Result from live speaker identification."""
    speaker_id: str
    display_name: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0


# =============================================================================
# Speaker Embedding Manager
# =============================================================================

class SpeakerEmbeddingManager:
    """
    Speaker embedding extraction and comparison.

    Port of OpenWhispr's speakerEmbeddings.js:
        - Extract 512-dim embeddings from audio segments
        - Cosine similarity matching
        - Centroid computation for multi-sample speaker profiles
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._embedding_model = None
        self._model_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Lazy-load the pyannote embedding model."""
        if self._embedding_model is not None:
            return

        async with self._model_lock:
            if self._embedding_model is not None:
                return

            hf_token = os.getenv("HF_AUTH_TOKEN", "")
            if not hf_token:
                raise RuntimeError(
                    "HF_AUTH_TOKEN is required for pyannote speaker embedding model. "
                    "Set it in .env and accept the model license on HuggingFace Hub."
                )

            logger.info("Loading pyannote speaker embedding model...")

            loop = asyncio.get_event_loop()
            self._embedding_model = await loop.run_in_executor(
                None, self._load_model, hf_token
            )
            logger.info("Speaker embedding model loaded")

    def _load_model(self, hf_token: str):
        """Load pyannote embedding model (blocking)."""
        from pyannote.audio import Model, Inference
        import torch

        model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token,
        )
        inference = Inference(model, window="whole", device=self._device)
        return inference

    async def extract_embedding(
        self,
        waveform: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Optional[np.ndarray]:
        """
        Extract a speaker embedding from an audio segment.

        Port of speakerEmbeddings.js extractEmbeddingFromSamples (lines 72-79):
            - Minimum segment: 1.5 seconds
            - Maximum segment: 8 seconds (uses the last 8 seconds)

        Args:
            waveform: float32 audio samples.
            sample_rate: Sample rate of the audio.

        Returns:
            Float32 embedding array, or None if segment is too short.
        """
        if len(waveform) < MIN_SEGMENT_SAMPLES:
            return None

        # Cap to MAX_EMBEDDING_SECONDS from the end (speakerEmbeddings.js:74-78)
        if len(waveform) > MAX_EMBEDDING_SAMPLES:
            waveform = waveform[-MAX_EMBEDDING_SAMPLES:]

        await self._ensure_loaded()

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._run_embedding, waveform, sample_rate
        )
        return embedding

    def _run_embedding(
        self, waveform: np.ndarray, sample_rate: int
    ) -> Optional[np.ndarray]:
        """Run embedding extraction (blocking)."""
        import torch

        # pyannote expects shape (1, num_samples) or {"waveform": ..., "sample_rate": ...}
        tensor = torch.from_numpy(waveform).float().unsqueeze(0)

        embedding = self._embedding_model(
            {"waveform": tensor, "sample_rate": sample_rate}
        )
        return np.array(embedding).flatten().astype(np.float32)

    @staticmethod
    def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute the centroid (mean) of a list of embeddings.
        Port of speakerEmbeddings.js computeCentroid (lines 129-142).
        """
        if not embeddings:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for emb in embeddings:
            centroid += emb
        centroid /= len(embeddings)
        return centroid


# =============================================================================
# Diarization Pipeline
# =============================================================================

class DiarizationPipeline:
    """
    Full offline speaker diarization using pyannote.audio.

    Port of OpenWhispr's diarization.js which runs pyannote segmentation
    + embedding models via sherpa-onnx. Since pyannote.audio IS the
    native Python implementation, we use it directly.

    Pipeline:
        1. Load pyannote diarization pipeline
        2. Run diarization on audio file → speaker-labeled segments
        3. Merge with transcript segments (port of diarization.js
           mergeWithTranscript, lines 436-511)
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cpu",
        max_speakers: int = DEFAULT_MAX_SPEAKERS,
    ):
        self._hf_token = hf_token or os.getenv("HF_AUTH_TOKEN", "")
        self._device = device or os.getenv("DIARIZATION_DEVICE", "cpu")
        self._max_speakers = int(os.getenv("SPEAKER_MAX_COUNT", str(max_speakers)))
        self._pipeline = None
        self._pipeline_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Lazy-load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            return

        async with self._pipeline_lock:
            if self._pipeline is not None:
                return

            if not self._hf_token:
                raise RuntimeError(
                    "HF_AUTH_TOKEN required for pyannote diarization. "
                    "Set it in .env and accept the model license on HuggingFace."
                )

            logger.info("Loading pyannote diarization pipeline...")
            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None, self._load_pipeline
            )
            logger.info("Diarization pipeline loaded")

    def _load_pipeline(self):
        """Load pyannote pipeline (blocking)."""
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._hf_token,
        )
        pipeline.to(self._device)
        return pipeline

    async def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> List[DiarizedSegment]:
        """
        Run offline speaker diarization on an audio file.

        Port of diarization.js diarize() (lines 306-400).

        Args:
            audio_path: Path to a WAV audio file (16kHz mono recommended).
            num_speakers: Expected number of speakers (None for auto-detect).

        Returns:
            List of speaker-labeled segments.
        """
        await self._ensure_loaded()

        start_time = time.perf_counter()
        logger.info("Starting diarization: %s", audio_path)

        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, self._run_diarization, audio_path, num_speakers
        )

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Diarization complete: %d segments in %.0fms",
            len(segments), elapsed,
        )
        return segments

    def _run_diarization(
        self,
        audio_path: str,
        num_speakers: Optional[int],
    ) -> List[DiarizedSegment]:
        """Run diarization (blocking)."""
        kwargs = {}
        if num_speakers is not None and num_speakers > 0:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["max_speakers"] = self._max_speakers

        diarization = self._pipeline(audio_path, **kwargs)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))

        return segments

    def merge_with_transcript(
        self,
        transcript_segments: List[dict],
        diarization_segments: List[DiarizedSegment],
    ) -> List[EnrichedSegment]:
        """
        Merge transcription segments with diarization labels.

        Port of diarization.js mergeWithTranscript (lines 436-511).

        For each transcript segment, finds the diarization segment with
        maximum overlap and assigns that speaker label. If no overlap,
        uses the nearest diarization segment by midpoint distance.

        Args:
            transcript_segments: List of dicts with 'start', 'end', 'text' keys.
            diarization_segments: Speaker-labeled segments from diarize().

        Returns:
            Enriched segments with speaker labels.
        """
        if not transcript_segments:
            return []

        if not diarization_segments:
            return [
                EnrichedSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                )
                for seg in transcript_segments
            ]

        # Build speaker renumbering map (diarization.js:444-450)
        speaker_set = sorted(set(seg.speaker for seg in diarization_segments))
        speaker_map = {sp: f"speaker_{i}" for i, sp in enumerate(speaker_set)}

        enriched = []
        for i, seg in enumerate(transcript_segments):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", seg_start + 2.5)
            midpoint = seg_start + (seg_end - seg_start) / 2

            best_speaker = None
            best_overlap = 0.0
            best_distance = float("inf")

            for d_seg in diarization_segments:
                # Overlap calculation (diarization.js:482-486)
                overlap = min(seg_end, d_seg.end) - max(seg_start, d_seg.start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d_seg.speaker

                # Distance fallback (diarization.js:488-499)
                if midpoint < d_seg.start:
                    distance = d_seg.start - midpoint
                elif midpoint > d_seg.end:
                    distance = midpoint - d_seg.end
                else:
                    distance = 0.0

                if best_speaker is None and distance < best_distance:
                    best_distance = distance
                    best_speaker = d_seg.speaker

            enriched.append(EnrichedSegment(
                text=seg.get("text", ""),
                start=seg_start,
                end=seg_end,
                speaker=speaker_map.get(best_speaker, best_speaker) if best_speaker else None,
            ))

        return enriched


# =============================================================================
# Live Speaker Identifier
# =============================================================================

class LiveSpeakerIdentifier:
    """
    Real-time speaker identification with transient cluster management.

    Port of OpenWhispr's liveSpeakerIdentifier.js (798 lines).

    Key features:
        - VAD-gated speaker embedding extraction
        - Transient in-session cluster management
        - Cross-session matching against stored SQLite profiles
        - Running centroid updates for improving accuracy over time
        - Max speaker capping with forced merge to nearest cluster

    Matching thresholds (from liveSpeakerIdentifier.js:29-30):
        MATCH_THRESHOLD = 0.65
        MATCH_MARGIN = 0.03
    """

    def __init__(
        self,
        speaker_db: SpeakerProfileDB,
        embedding_manager: SpeakerEmbeddingManager,
        max_speakers: int = DEFAULT_MAX_SPEAKERS,
    ):
        self._db = speaker_db
        self._embeddings = embedding_manager
        self._max_speakers = int(os.getenv("SPEAKER_MAX_COUNT", str(max_speakers)))

        # Transient per-session state (mirrors liveSpeakerIdentifier.js:127-154)
        self._transient_embeddings: Dict[str, np.ndarray] = {}
        self._transient_counts: Dict[str, int] = {}
        self._transient_display_names: Dict[str, str] = {}
        self._transient_profile_ids: Dict[str, str] = {}
        self._next_live_index = 0

        self._running = False

    async def start(self) -> bool:
        """Start the live speaker identifier for a new session."""
        self._reset_session_state()
        self._running = True
        logger.info("Live speaker identifier started (max_speakers=%d)", self._max_speakers)
        return True

    async def stop(self) -> Dict[str, dict]:
        """
        Stop and return the transient state.
        Port of liveSpeakerIdentifier.js stop() (lines 215-233).
        """
        self._running = False
        state = self.get_transient_state()
        self._reset_session_state()
        return state

    def get_transient_state(self) -> Dict[str, dict]:
        """
        Get current transient speaker state.
        Port of liveSpeakerIdentifier.js getTransientState() (lines 165-178).
        """
        state = {}
        for speaker_id, embedding in self._transient_embeddings.items():
            state[speaker_id] = {
                "embedding": embedding.tolist(),
                "display_name": self._transient_display_names.get(speaker_id),
                "profile_id": self._transient_profile_ids.get(speaker_id),
            }
        return state

    async def identify_speaker(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Optional[SpeakerIdentification]:
        """
        Identify the speaker in an audio segment.

        Runs the full identification pipeline:
            1. Extract embedding from audio
            2. Match against transient session clusters
            3. Match against stored SQLite profiles
            4. Assign new speaker ID or merge to nearest

        Port of liveSpeakerIdentifier.js _finalizeSpeechSegment() (lines 583-624).
        """
        if not self._running:
            return None

        # Extract embedding
        embedding = await self._embeddings.extract_embedding(audio_samples, sample_rate)
        if embedding is None:
            return None

        # Resolve speaker
        speaker_id, display_name = await self._resolve_speaker(embedding, update_centroid=True)

        if not speaker_id:
            return None

        return SpeakerIdentification(
            speaker_id=speaker_id,
            display_name=display_name,
        )

    async def _resolve_speaker(
        self,
        embedding: np.ndarray,
        update_centroid: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve which speaker an embedding belongs to.

        Port of liveSpeakerIdentifier.js _resolveSpeakerForEmbedding()
        (lines 690-733).

        Priority:
            1. Match against transient session clusters
            2. Match against stored SQLite profiles
            3. Create new speaker or force-merge to nearest
        """
        # 1. Check transient clusters
        transient_match = self._find_transient_match(embedding)
        if transient_match:
            if update_centroid:
                self._update_centroid(transient_match, embedding)
            return transient_match, self._transient_display_names.get(transient_match)

        # 2. Check stored profiles
        stored_match = await self._db.find_matching_profile(
            embedding,
            threshold=MATCH_THRESHOLD,
            margin=MATCH_MARGIN,
        )
        if stored_match:
            # Find or create transient entry for this stored profile
            transient_id = self._find_transient_for_profile(stored_match.id)
            if not transient_id:
                transient_id = self._assign_or_force_cluster(embedding)

            self._transient_profile_ids[transient_id] = stored_match.id
            self._transient_display_names[transient_id] = stored_match.display_name

            # Update stored profile centroid
            await self._db.save_profile(
                stored_match.id,
                stored_match.display_name,
                embedding,
            )

            return transient_id, stored_match.display_name

        # 3. New speaker
        speaker_id = self._assign_or_force_cluster(embedding)
        return speaker_id, self._transient_display_names.get(speaker_id)

    def _find_transient_match(self, embedding: np.ndarray) -> Optional[str]:
        """
        Find matching transient speaker.
        Port of liveSpeakerIdentifier.js _findTransientMatch() (lines 626-646).
        """
        best_id = None
        best_sim = 0.0
        second_best_sim = 0.0

        for speaker_id, centroid in self._transient_embeddings.items():
            sim = cosine_similarity(embedding, centroid)
            if sim > best_sim:
                second_best_sim = best_sim
                best_sim = sim
                best_id = speaker_id
            elif sim > second_best_sim:
                second_best_sim = sim

        if best_sim >= MATCH_THRESHOLD and (best_sim - second_best_sim) >= MATCH_MARGIN:
            return best_id
        return None

    def _find_transient_for_profile(self, profile_id: str) -> Optional[str]:
        """
        Find transient speaker ID mapped to a stored profile.
        Port of liveSpeakerIdentifier.js _findTransientSpeakerForProfile()
        (lines 759-767).
        """
        for speaker_id, pid in self._transient_profile_ids.items():
            if pid == profile_id:
                return speaker_id
        return None

    def _assign_or_force_cluster(self, embedding: np.ndarray) -> str:
        """
        Assign a new speaker ID or force-merge to nearest cluster if at capacity.
        Port of liveSpeakerIdentifier.js _assignOrForceCluster() (lines 735-744).
        """
        if len(self._transient_embeddings) >= self._max_speakers:
            nearest = self._find_nearest_transient(embedding)
            if nearest:
                self._update_centroid(nearest, embedding)
                return nearest

        return self._assign_speaker_id(embedding)

    def _find_nearest_transient(self, embedding: np.ndarray) -> Optional[str]:
        """Find the nearest transient cluster by cosine similarity."""
        best_id = None
        best_sim = -float("inf")

        for speaker_id, centroid in self._transient_embeddings.items():
            sim = cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = speaker_id

        return best_id

    def _assign_speaker_id(self, embedding: np.ndarray) -> str:
        """
        Create a new transient speaker.
        Port of liveSpeakerIdentifier.js _assignSpeakerId() (lines 769-774).
        """
        speaker_id = f"speaker_{self._next_live_index}"
        self._next_live_index += 1
        self._transient_embeddings[speaker_id] = embedding.copy()
        self._transient_counts[speaker_id] = 1
        return speaker_id

    def _update_centroid(self, speaker_id: str, embedding: np.ndarray) -> None:
        """
        Running weighted centroid update.
        Port of liveSpeakerIdentifier.js _updateCentroid() (lines 777-792):

            centroid[i] = (centroid[i] * count + embedding[i]) / (count + 1)
        """
        centroid = self._transient_embeddings.get(speaker_id)
        if centroid is None:
            return

        count = self._transient_counts.get(speaker_id, 1)
        updated = (centroid * count + embedding) / (count + 1)

        self._transient_embeddings[speaker_id] = updated
        self._transient_counts[speaker_id] = count + 1

    def map_speaker(
        self,
        live_id: str,
        profile_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> bool:
        """
        Map a live speaker to a stored profile and/or display name.
        Port of liveSpeakerIdentifier.js mapSpeaker() (lines 345-363).
        """
        if live_id not in self._transient_embeddings:
            return False

        if profile_id:
            self._transient_profile_ids[live_id] = profile_id
        if display_name:
            self._transient_display_names[live_id] = display_name

        return True

    async def persist_session_speakers(self) -> int:
        """
        Save all transient session speakers to the SQLite database.

        This is called at the end of a session to persist new speaker
        embeddings for cross-session recognition. This is the key feature
        that OpenWhispr's transient-only approach doesn't provide.

        Returns:
            Number of profiles saved.
        """
        saved = 0
        for speaker_id, embedding in self._transient_embeddings.items():
            display_name = self._transient_display_names.get(speaker_id, speaker_id)
            profile_id = self._transient_profile_ids.get(speaker_id)

            await self._db.save_profile(
                speaker_id=profile_id,  # Use stored ID if mapped, else creates new
                display_name=display_name,
                embedding=embedding,
            )
            saved += 1

        logger.info("Persisted %d speaker profiles to database", saved)
        return saved

    def _reset_session_state(self) -> None:
        """Reset all transient state for a new session."""
        self._transient_embeddings.clear()
        self._transient_counts.clear()
        self._transient_display_names.clear()
        self._transient_profile_ids.clear()
        self._next_live_index = 0
