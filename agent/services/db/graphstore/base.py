"""GraphStore protocol. A knowledge graph of chunks, the entities they mention and the
relations between entities. Any backend implementing it can be plugged into the agent."""
from __future__ import annotations

from typing import Protocol

from .types import ChunkFacts, ChunkRecord, EntityContext, EntityInfo, GraphHit


class GraphStore(Protocol):
    def upsert(self, facts: ChunkFacts) -> None:
        """Write one chunk (with its text), its entities and relations (idempotent)."""
        ...

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """Chunks by id, in the order given; unknown ids are skipped."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove a document's chunks, their mentions and relations; prune orphan entities. Returns chunks removed."""
        ...

    def neighbours(self, chunk_ids: list[str], *, limit: int = 5) -> list[GraphHit]:
        """Chunks connected to the given ones (shared entities, adjacent passages), best first."""
        ...

    def search_entities(self, text: str, *, limit: int = 10) -> list[EntityInfo]:
        """Entities whose label matches the text, most mentioned first."""
        ...

    def entity_context(self, keys: list[str], *, limit_chunks: int = 10) -> EntityContext:
        """Relations involving the entities and the chunks that mention them."""
        ...

    def top_entities(self, *, limit: int = 20) -> list[EntityInfo]: ...

    def stats(self) -> dict[str, int]:
        """{"chunks", "entities", "mentions", "relations"}"""
        ...
