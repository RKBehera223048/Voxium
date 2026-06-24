"""
Voxium — Entity & Relationship Extractor
==========================================
Pure-CPU extraction of entities and relationships from conversational text.
No LLM calls, no GPU, no external API — just regex heuristics and Python stdlib.

Inspired by graphify's deterministic AST extraction approach (extract.py)
but adapted for natural-language conversation rather than source code.

Entity IDs use deterministic SHA-256 hashes matching graphify's _make_id() pattern.
"""

from __future__ import annotations

import hashlib
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class Entity:
    """An extracted entity from conversational text."""
    id: str
    label: str
    type: str  # person | place | concept | action | date | technical | quoted
    span_start: int = 0
    span_end: int = 0
    confidence: str = "EXTRACTED"  # EXTRACTED | INFERRED — matches graphify schema

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "confidence": self.confidence,
        }


@dataclass
class Relationship:
    """An extracted relationship between two entities."""
    source_id: str
    target_id: str
    relation: str  # co_occurs_with | action_verb | references
    confidence: str = "INFERRED"  # EXTRACTED | INFERRED — matches graphify schema
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "weight": self.weight,
        }


# =============================================================================
# ID Generation — mirrors graphify's _make_id()
# =============================================================================

def make_entity_id(label: str, entity_type: str) -> str:
    """
    Generate a deterministic entity ID from label + type.

    Uses SHA-256 truncated to 12 hex chars, matching graphify's approach
    for stable, collision-resistant IDs.
    """
    normalized = label.strip().lower()
    raw = f"{normalized}:{entity_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Regex Patterns
# =============================================================================

# Named entities: Capitalized word sequences (e.g., "John Smith", "New York")
_NAMED_ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# Single capitalized words — detected during sentence-level analysis below
# (lookbehinds removed; mid-sentence proper noun detection handled in extract_entities)

# Standalone proper nouns — found via sentence analysis, not regex


# Dates: various formats
_DATE_PATTERNS = [
    # ISO dates: 2024-01-15
    re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b"),
    # US dates: 01/15/2024, 1/15/24
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),
    # Relative dates
    re.compile(
        r"\b(today|tomorrow|yesterday|"
        r"next\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year)|"
        r"last\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year)|"
        r"this\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year))\b",
        re.IGNORECASE,
    ),
    # Month day patterns: January 15, Jan 15th
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
        re.IGNORECASE,
    ),
]

# Quoted strings: "exact phrase" or 'exact phrase'
_QUOTED_RE = re.compile(r"""["']([^"']{3,60})["']""")

# Technical terms: CamelCase, snake_case identifiers, dot.separated
_CAMEL_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
_SNAKE_CASE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b")
_DOT_SEPARATED_RE = re.compile(r"\b([a-zA-Z]\w+(?:\.\w+){1,})\b")

# Acronyms: 2+ uppercase letters
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")

# Action verbs for relationship extraction
_ACTION_VERBS = re.compile(
    r"\b(?:called|asked|told|sent|emailed|messaged|contacted|"
    r"mentioned|discussed|met|visited|scheduled|created|"
    r"updated|deleted|built|deployed|fixed|reviewed|"
    r"assigned|delegated|reported|presented|shared)\b",
    re.IGNORECASE,
)

# Common stop words that should NOT be entities
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that", "this",
    "these", "those", "i", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "what", "which", "who", "whom",
    "also", "still", "already", "even", "really", "quite", "rather",
    # Common sentence starters that aren't proper nouns
    "However", "Therefore", "Moreover", "Furthermore", "Additionally",
    "Meanwhile", "Nevertheless", "Nonetheless", "Otherwise", "Instead",
    "Indeed", "Perhaps", "Actually", "Basically", "Certainly",
    "Please", "Thanks", "Sorry", "Hello", "Hey", "Well", "Yes", "No",
    "OK", "Okay", "Sure", "Right", "Let", "Make", "Take", "Get",
    "Now", "Then", "Here", "There", "Today", "Also", "Just", "Like",
})

# Common words that look like proper nouns at sentence start
_SENTENCE_STARTER_NOISE = frozenset({
    "The", "This", "That", "These", "Those", "There", "Here",
    "What", "Which", "Where", "When", "Why", "How", "Who",
    "Some", "Any", "Each", "Every", "All", "Both", "Many",
    "Such", "Other", "Another", "First", "Last", "Next",
    "Our", "Your", "Their", "Its", "His", "Her", "My",
    "But", "And", "For", "Nor", "Yet", "Not",
})


# =============================================================================
# Entity Extraction
# =============================================================================

def extract_entities(text: str) -> List[Entity]:
    """
    Extract entities from conversational text using regex heuristics.

    Returns a deduplicated list of Entity objects. Pure CPU, no LLM.

    Entity types:
        - person: Capitalized multi-word names
        - date: Date/time references
        - quoted: Quoted exact phrases
        - technical: CamelCase, snake_case, dot.separated identifiers
        - concept: Acronyms, standalone proper nouns
    """
    if not text or len(text.strip()) < 3:
        return []

    entities: List[Entity] = []
    seen_labels: set[str] = set()

    def _add(label: str, etype: str, start: int = 0, end: int = 0,
             confidence: str = "EXTRACTED") -> None:
        """Add entity if not a stop word and not already seen."""
        normalized = label.strip()
        if not normalized or len(normalized) < 2:
            return
        if normalized.lower() in _STOP_WORDS:
            return
        if normalized in _SENTENCE_STARTER_NOISE:
            return

        key = normalized.lower()
        if key in seen_labels:
            return
        seen_labels.add(key)

        entities.append(Entity(
            id=make_entity_id(normalized, etype),
            label=normalized,
            type=etype,
            span_start=start,
            span_end=end,
            confidence=confidence,
        ))

    # 1. Named entities (multi-word capitalized sequences)
    for match in _NAMED_ENTITY_RE.finditer(text):
        _add(match.group(1), "person", match.start(), match.end())

    # 2. Dates
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            _add(match.group(1), "date", match.start(), match.end())

    # 3. Quoted strings
    for match in _QUOTED_RE.finditer(text):
        _add(match.group(1), "quoted", match.start(), match.end())

    # 4. Technical terms
    for match in _CAMEL_CASE_RE.finditer(text):
        _add(match.group(1), "technical", match.start(), match.end())

    for match in _SNAKE_CASE_RE.finditer(text):
        _add(match.group(1), "technical", match.start(), match.end())

    for match in _DOT_SEPARATED_RE.finditer(text):
        _add(match.group(1), "technical", match.start(), match.end())

    # 5. Acronyms (2-6 uppercase letters)
    for match in _ACRONYM_RE.finditer(text):
        word = match.group(1)
        # Skip common non-entity acronyms
        if word not in {"I", "A", "OK", "AM", "PM", "US", "IT", "OR", "AN", "AS",
                        "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "ME",
                        "MY", "NO", "OF", "ON", "SO", "TO", "UP", "WE"}:
            _add(word, "concept", match.start(), match.end())

    # 6. Standalone proper nouns (single capitalized words in mid-sentence)
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Skip the first word of each sentence
        words = sentence.split()
        for word in words[1:]:
            # Clean trailing punctuation
            clean = re.sub(r'[,;:!?.\'"]+$', '', word)
            if (clean and len(clean) >= 3 and clean[0].isupper()
                    and clean[1:].islower() and clean not in _SENTENCE_STARTER_NOISE):
                _add(clean, "concept", confidence="INFERRED")

    return entities


# =============================================================================
# Relationship Extraction
# =============================================================================

def extract_relationships(
    text: str,
    entities: List[Entity],
) -> List[Relationship]:
    """
    Extract relationships between entities in the text.

    Two strategies:
        1. Co-occurrence: Entities in the same sentence get a `co_occurs_with` edge
        2. Verb-mediated: "X called Y" patterns → directed action edges

    Returns a list of Relationship objects.
    """
    if len(entities) < 2:
        return []

    relationships: List[Relationship] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    # Split into sentences for co-occurrence analysis
    sentences = re.split(r'[.!?\n]+', text)

    # Build a label→entity map for lookup
    label_to_entity: dict[str, Entity] = {}
    for e in entities:
        label_to_entity[e.label.lower()] = e

    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue

        # Find which entities appear in this sentence
        present: List[Entity] = []
        for entity in entities:
            if entity.label.lower() in sentence_lower:
                present.append(entity)

        # Co-occurrence edges (all pairs in the same sentence)
        for i, e1 in enumerate(present):
            for e2 in present[i + 1:]:
                pair_key = (
                    min(e1.id, e2.id),
                    max(e1.id, e2.id),
                    "co_occurs_with",
                )
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    relationships.append(Relationship(
                        source_id=e1.id,
                        target_id=e2.id,
                        relation="co_occurs_with",
                        confidence="EXTRACTED",
                        weight=1.0,
                    ))

        # Verb-mediated relationships
        # Look for patterns: <Entity> <verb> <Entity>
        for verb_match in _ACTION_VERBS.finditer(sentence):
            verb = verb_match.group(0).lower()
            before = sentence[:verb_match.start()].strip()
            after = sentence[verb_match.end():].strip()

            # Find entity in the "before" part (closest to the verb)
            source_entity: Optional[Entity] = None
            for entity in entities:
                if entity.label.lower() in before.lower():
                    source_entity = entity

            # Find entity in the "after" part (closest to the verb)
            target_entity: Optional[Entity] = None
            for entity in entities:
                if entity.label.lower() in after.lower():
                    target_entity = entity
                    break  # Take the first one after the verb

            if source_entity and target_entity and source_entity.id != target_entity.id:
                pair_key = (source_entity.id, target_entity.id, verb)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    relationships.append(Relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        relation=verb,
                        confidence="INFERRED",
                        weight=1.5,
                    ))

    return relationships


# =============================================================================
# Convenience
# =============================================================================

def extract_all(text: str) -> Tuple[List[Entity], List[Relationship]]:
    """
    Extract both entities and relationships from text in one call.

    Returns:
        (entities, relationships) tuple.
    """
    entities = extract_entities(text)
    relationships = extract_relationships(text, entities)
    return entities, relationships
