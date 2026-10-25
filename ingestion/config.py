"""Settings for the ingestion layer, read from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass
class IngestSettings:
    chunk_size: int = 1200  # characters per chunk
    chunk_overlap: int = 150  # characters carried over from the previous chunk
    max_upload_mb: float = 10.0
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf")
    graph_extract: bool = True  # run the LLM entity/relation extractor per chunk (needs a graph store wired)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "IngestSettings":
        load_dotenv(dotenv_path, override=False)
        return cls(
            chunk_size=_int("INGEST_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("INGEST_CHUNK_OVERLAP", cls.chunk_overlap),
            max_upload_mb=_float("INGEST_MAX_UPLOAD_MB", cls.max_upload_mb),
            graph_extract=(os.getenv("INGEST_GRAPH_EXTRACT") or "true").strip().lower() not in ("0", "false", "no", "off"),
        )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
