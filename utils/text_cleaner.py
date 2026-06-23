"""
Voxium — Text Cleaner Utilities
=================================
Helper functions for stripping punctuation, normalizing whitespace,
and formatting output text.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_filler_words(text: str) -> str:
    """Remove common filler words from transcribed text."""
    fillers = [
        r"\bum\b", r"\buh\b", r"\blike\b(?=\s*,)",
        r"\byou know\b", r"\bI mean\b", r"\bso\b(?=\s*,)",
        r"\bactually\b(?=\s*,)", r"\bbasically\b(?=\s*,)",
    ]
    result = text
    for pattern in fillers:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return normalize_whitespace(result)


def capitalize_sentences(text: str) -> str:
    """Capitalize the first letter of each sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s[0].upper() + s[1:] if s else s for s in sentences)


def format_numbers(text: str) -> str:
    """Convert spelled-out numbers to digits where appropriate."""
    number_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10",
    }
    result = text
    for word, digit in number_map.items():
        result = re.sub(rf"\b{word}\b", digit, result, flags=re.IGNORECASE)
    return result


def clean_transcription(text: str) -> str:
    """Apply all text cleaning transformations."""
    text = normalize_whitespace(text)
    text = strip_filler_words(text)
    text = capitalize_sentences(text)
    return text
