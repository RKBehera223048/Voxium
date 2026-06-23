"""
Voxium — Speaker Profile Database
====================================
SQLite persistence for speaker voice embeddings, enabling cross-session
speaker recognition.

Ported from OpenWhispr's architecture where speaker embeddings are extracted
via ONNX Runtime (speakerEmbeddings.js) and matched against stored profiles
(liveSpeakerIdentifier.js _findStoredProfileMatch, lines 648-688).

This module provides the storage layer that OpenWhispr doesn't fully
expose — it keeps embeddings in transient memory. Voxium persists them
to SQLite so the system remembers voices across restarts.
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # Matches speakerEmbeddings.js:8


@dataclass
class SpeakerProfile:
    """A stored speaker profile with voice embedding."""
    id: str
    display_name: str
    embedding: np.ndarray  # float32 array of EMBEDDING_DIM dimensions
    embedding_count: int = 1
    created_at: str = ""
    last_seen_at: str = ""


class SpeakerProfileDB:
    """
    SQLite-backed speaker profile storage.

    Schema mirrors the transient state from OpenWhispr's
    liveSpeakerIdentifier.js, but persisted to disk:
        - id: Unique speaker identifier
        - display_name: User-assigned name for the speaker
        - embedding: Voice embedding (serialized float32 blob)
        - embedding_count: Number of embeddings used to compute the centroid
        - timestamps for creation and last seen

    Centroid Update:
        When a known speaker is heard again, their embedding is updated
        using a running weighted average (ported from
        liveSpeakerIdentifier.js _updateCentroid, lines 777-792):

            centroid[i] = (centroid[i] * count + new_embedding[i]) / (count + 1)
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.getenv(
            "DATABASE_PATH", "data/db/voxium.sqlite"
        )
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Create the database and tables if they don't exist."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            import aiosqlite

            # Ensure directory exists
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS speaker_profiles (
                        id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_count INTEGER DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_speaker_last_seen
                    ON speaker_profiles(last_seen_at)
                """)
                await db.commit()

            self._initialized = True
            logger.info("Speaker profile database initialized: %s", self._db_path)

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize a float32 numpy array to bytes for SQLite BLOB storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        """Deserialize a BLOB back to a float32 numpy array."""
        return np.frombuffer(blob, dtype=np.float32).copy()

    async def save_profile(
        self,
        speaker_id: Optional[str],
        display_name: str,
        embedding: np.ndarray,
    ) -> str:
        """
        Save or update a speaker profile.

        If the speaker_id already exists, updates the centroid embedding
        and bumps the embedding_count. Otherwise creates a new profile.

        Args:
            speaker_id: Existing speaker ID, or None to generate a new one.
            display_name: Human-readable name for the speaker.
            embedding: Float32 voice embedding (EMBEDDING_DIM dimensions).

        Returns:
            The speaker profile ID.
        """
        await self._ensure_initialized()
        import aiosqlite

        if speaker_id is None:
            speaker_id = str(uuid.uuid4())

        embedding_blob = self._serialize_embedding(embedding)

        async with aiosqlite.connect(self._db_path) as db:
            # Check if profile exists
            cursor = await db.execute(
                "SELECT embedding, embedding_count FROM speaker_profiles WHERE id = ?",
                (speaker_id,),
            )
            row = await cursor.fetchone()

            if row is not None:
                # Update existing — compute running centroid
                existing_embedding = self._deserialize_embedding(row[0])
                count = row[1]

                # Weighted centroid update (liveSpeakerIdentifier.js:786-788)
                updated = np.empty_like(existing_embedding)
                for i in range(len(existing_embedding)):
                    updated[i] = (existing_embedding[i] * count + embedding[i]) / (count + 1)

                await db.execute(
                    """UPDATE speaker_profiles
                       SET display_name = ?,
                           embedding = ?,
                           embedding_count = ?,
                           last_seen_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    (display_name, self._serialize_embedding(updated), count + 1, speaker_id),
                )
            else:
                # Insert new profile
                await db.execute(
                    """INSERT INTO speaker_profiles
                       (id, display_name, embedding, embedding_count)
                       VALUES (?, ?, ?, 1)""",
                    (speaker_id, display_name, embedding_blob),
                )

            await db.commit()

        logger.debug("Speaker profile saved: id=%s name=%s", speaker_id, display_name)
        return speaker_id

    async def get_all_profiles(self) -> List[SpeakerProfile]:
        """
        Retrieve all stored speaker profiles with deserialized embeddings.

        This is called by the LiveSpeakerIdentifier to match incoming
        audio against known speakers (mirrors liveSpeakerIdentifier.js
        _findStoredProfileMatch, lines 648-688).
        """
        await self._ensure_initialized()
        import aiosqlite

        profiles = []
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM speaker_profiles ORDER BY last_seen_at DESC"
            )
            rows = await cursor.fetchall()

            for row in rows:
                profiles.append(SpeakerProfile(
                    id=row["id"],
                    display_name=row["display_name"],
                    embedding=self._deserialize_embedding(row["embedding"]),
                    embedding_count=row["embedding_count"],
                    created_at=row["created_at"],
                    last_seen_at=row["last_seen_at"],
                ))

        return profiles

    async def find_matching_profile(
        self,
        embedding: np.ndarray,
        threshold: float = 0.65,
        margin: float = 0.03,
    ) -> Optional[SpeakerProfile]:
        """
        Find the best matching stored profile for an embedding.

        Uses the same matching logic as OpenWhispr's
        liveSpeakerIdentifier.js _findStoredProfileMatch (lines 648-688):
            - Compute cosine similarity against all stored profiles
            - Best match must be above threshold (0.65)
            - Must have sufficient margin (0.03) over second-best match

        Args:
            embedding: The query embedding to match.
            threshold: Minimum cosine similarity for a match.
            margin: Minimum gap between best and second-best match.

        Returns:
            Best matching SpeakerProfile, or None if no match found.
        """
        profiles = await self.get_all_profiles()
        if not profiles:
            return None

        best_profile = None
        best_similarity = 0.0
        second_best_similarity = 0.0

        for profile in profiles:
            similarity = cosine_similarity(embedding, profile.embedding)

            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_profile = profile
            elif similarity > second_best_similarity:
                second_best_similarity = similarity

        if best_similarity >= threshold and (best_similarity - second_best_similarity) >= margin:
            return best_profile

        return None

    async def update_last_seen(self, speaker_id: str) -> None:
        """Update the last_seen_at timestamp for a speaker."""
        await self._ensure_initialized()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE speaker_profiles SET last_seen_at = CURRENT_TIMESTAMP WHERE id = ?",
                (speaker_id,),
            )
            await db.commit()

    async def delete_profile(self, speaker_id: str) -> bool:
        """Delete a speaker profile."""
        await self._ensure_initialized()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM speaker_profiles WHERE id = ?",
                (speaker_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_profile_count(self) -> int:
        """Get the number of stored profiles."""
        await self._ensure_initialized()
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM speaker_profiles")
            row = await cursor.fetchone()
            return row[0]


# =============================================================================
# Cosine Similarity — Port of speakerEmbeddings.js:144-155
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Direct port of OpenWhispr speakerEmbeddings.js cosineSimilarity
    (lines 144-155).
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(dot / denom)
