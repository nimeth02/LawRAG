"""VectorStore protocol. Any backend that implements it can be plugged into the retriever."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # type-only: avoids an import cycle with agent.rag at runtime
    from agent.rag.schemas import Document, RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None: ...

    def search(self, query_embedding: list[float], k: int, where: dict[str, Any] | None = None) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...

    def get(self, ids: list[str], query_embedding: list[float] | None = None) -> list[RetrievedChunk]:
        """Fetch chunks by id (order preserved, missing ids skipped). With a query embedding the
        score is the cosine similarity to it, otherwise 0.0."""
        ...

    def all_metadata(self) -> list[tuple[str, dict[str, Any]]]:
        """(chunk_id, metadata) for every stored chunk. Used to list documents."""
        ...

    def delete_where(self, where: dict[str, Any]) -> int:
        """Delete chunks matching a metadata filter; return how many were removed."""
        ...
