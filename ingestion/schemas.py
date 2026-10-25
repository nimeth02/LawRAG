"""Domain objects for ingestion. Plain dataclasses, no framework types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestResult:
    doc_id: str
    source: str
    chunks: int
    chars: int
    replaced: bool  # True when an earlier version of the same source was overwritten
    entities: int = 0  # knowledge-graph entities extracted (0 when the graph is off)
    relations: int = 0
    graph_errors: int = 0  # chunks whose extraction failed (their text is still in the graph, unlinked)


@dataclass
class IndexedDocument:
    doc_id: str
    source: str
    chunks: int
    ingested_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # user metadata only
