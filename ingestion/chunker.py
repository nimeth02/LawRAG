"""Text -> chunks. Paragraph-aware, falls back to sentences, then hard splits; with overlap."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_BLANK_LINE = re.compile(r"\n\s*\n")


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    pieces: list[str] = []
    cur = ""
    for sentence in _SENTENCE_END.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > size:  # a single run-on sentence: hard split
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(sentence[i : i + size] for i in range(0, len(sentence), size))
            continue
        if cur and len(cur) + 1 + len(sentence) > size:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` characters, trimmed to start on a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else tail


def chunk_text(text: str, *, size: int = 1200, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of at most `size` chars (plus overlap prefix).

    Paragraphs are never split unless one alone exceeds `size`; then it is split on
    sentence boundaries, and as a last resort on raw character counts.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    text = text.strip()
    if not text:
        return []

    units: list[str] = []
    for para in _BLANK_LINE.split(text):
        para = " ".join(para.split())  # collapse internal newlines/spaces within a paragraph
        if not para:
            continue
        units.extend(_split_long_paragraph(para, size) if len(para) > size else [para])

    chunks: list[str] = []
    cur = ""
    for unit in units:
        if cur and len(cur) + 2 + len(unit) > size:
            chunks.append(cur)
            cur = unit
        else:
            cur = f"{cur}\n\n{unit}" if cur else unit
    if cur:
        chunks.append(cur)

    if overlap and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for prev, cur_chunk in zip(chunks, chunks[1:]):
            tail = _tail(prev, overlap)
            with_overlap.append(f"{tail} {cur_chunk}" if tail else cur_chunk)
        chunks = with_overlap
    return chunks
