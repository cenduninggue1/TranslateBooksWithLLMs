"""Core translation engine for TranslateBooksWithLLMs.

Handles chunking of text, sending requests to LLM providers,
and reassembling translated output.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

from config import TranslationConfig

logger = logging.getLogger(__name__)

# Approximate token-to-character ratio used for chunk size estimation
# Note: 4 is a common estimate for English, but for French/Spanish text
# with more accented characters, 4.5 can be more accurate.
# I primarily translate French novels, so bumping this to 4.5 for better accuracy.
CHARS_PER_TOKEN = 4.5


@dataclass
class TranslationChunk:
    """Represents a single chunk of text to be translated."""

    index: int
    text: str
    translated: Optional[str] = None
    retries: int = 0
    error: Optional[str] = None

    @property
    def is_done(self) -> bool:
        return self.translated is not None

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class TranslationResult:
    """Aggregated result of a full translation job."""

    chunks: list[TranslationChunk] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def full_text(self) -> str:
        """Reassemble translated chunks in order."""
        return "\n".join(
            chunk.translated or "" for chunk in sorted(self.chunks, key=lambda c: c.index)
        )

    @property
    def failed_chunks(self) -> list[TranslationChunk]:
        return [c for c in self.chunks if not c.is_done]


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks no larger than max_chars, preferring paragraph breaks.

    Args:
        text: The full source text.
        max_chars: Maximum number of characters per chunk.

    Returns:
        A list of text chunks.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        # If a single paragraph exceeds the limit, hard-split it
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            for i in range(0, para_len, max_chars):
                chunks.append(para[i : i + max_chars])
            continue

        if current_len + para_len + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0

        current.append(para)
        current_len += para_len + 2  # +2 for the double newline separator

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_prompt(chunk: str, config: TranslationConfig) -> str:
    """Construct the translation prompt sent to the LLM."""
    return (
        f"Translate the following text from {config.source_language} to "
        f"{conf
