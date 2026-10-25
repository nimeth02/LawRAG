"""Settings for the RAG layer, read from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class RAGSettings:
    chroma_path: str = "data/chroma"
    collection: str = "lankalaw_kb"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    top_k: int = 5
    min_score: float = 0.0  # cosine similarity below this is dropped
    max_context_chars: int = 12_000  # budget for one search tool result
    max_steps: int = 6  # model calls per question; after that the agent must answer
    reflect: bool = True  # grade results after each search round and plan the next move (extra model call)
    graph_enabled: bool = True  # on-premises knowledge graph (Kùzu) beside the vector store
    graph_path: str = "data/graph/lankalaw.kuzu"
    graph_expand_k: int = 3  # extra passages pulled in per search through graph links

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "RAGSettings":
        load_dotenv(dotenv_path, override=False)
        return cls(
            chroma_path=os.getenv("CHROMA_PATH") or cls.chroma_path,
            collection=os.getenv("RAG_COLLECTION") or cls.collection,
            embedding_model=os.getenv("EMBEDDING_MODEL") or cls.embedding_model,
            embedding_dim=_int("EMBEDDING_DIM", cls.embedding_dim),
            top_k=_int("RAG_TOP_K", cls.top_k),
            min_score=_float("RAG_MIN_SCORE", cls.min_score),
            max_context_chars=_int("RAG_MAX_CONTEXT_CHARS", cls.max_context_chars),
            max_steps=_int("RAG_MAX_STEPS", cls.max_steps),
            reflect=_bool("RAG_REFLECT", cls.reflect),
            graph_enabled=_bool("GRAPH_ENABLED", cls.graph_enabled),
            graph_path=os.getenv("GRAPH_PATH") or cls.graph_path,
            graph_expand_k=_int("GRAPH_EXPAND_K", cls.graph_expand_k),
        )
