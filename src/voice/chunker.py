"""Sentence chunker — splits LLM text into speakable chunks for TTS.

Fish Audio rejects empty/whitespace-only chunks, so we filter them.
Splits on sentence boundaries and newlines for natural speech.
"""
from __future__ import annotations

import re
from typing import List

# Split on: . ! ? ; newlines, but keep the delimiter
_SPLIT_RE = re.compile(r'(?<=[.!?;])\s+|\n+')


def speakable(text: str) -> bool:
    """Check if text has actual content to speak (not just whitespace/tags)."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Skip pure punctuation or emotion tags
    if all(c in ' \t\n.,!?;:...—–-<>' for c in stripped):
        return False
    return True


def chunk_text(text: str) -> List[str]:
    """Split text into speakable sentence-level chunks."""
    if not text:
        return []
    parts = _SPLIT_RE.split(text)
    return [p.strip() for p in parts if speakable(p)]
