"""Query -> ranked chunks.

`Retriever`              embed the query, search the vector store, apply the relevance cutoff.
`GraphExpandingRetriever` the same, then follow knowledge-graph links from the top hits (shared
                          entities, adjacent passages) and pull in a few connected chunks that
                          similarity alone would have missed. Their text comes from the graph
                          store itself; they carry `via` so the UI can say how they got there.
"""
from __future__ import annotations

from typing import Any

from agent.services.db.graphstore import ChunkRecord, GraphStore
from agent.services.db.vectorstore import VectorStore

from .embeddings import Embedder
from .schemas import RetrievedChunk


def chunk_from_graph(rec: ChunkRecord, via: str) -> RetrievedChunk:
    """A graph-stored chunk as a RetrievedChunk. Score is 0.0: it was found by links, not similarity."""
    return RetrievedChunk(
        id=rec.chunk_id,
        text=rec.text,
        metadata={"source": rec.source, "doc_id": rec.doc_id, "chunk_index": rec.chunk_index},
        score=0.0,
        via=via,
    )


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, *, top_k: int = 5, min_score: float = 0.0) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str, *, k: int | None = None, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        vec = self.embedder.embed_query(query)
        return self._search(vec, k or self.top_k, where)

    def _search(self, vec: list[float], k: int, where: dict[str, Any] | None) -> list[RetrievedChunk]:
        chunks = self.store.search(vec, k, where=where)
        return [c for c in chunks if c.score >= self.min_score]


class GraphExpandingRetriever(Retriever):
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        graph: GraphStore,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        expand_k: int = 3,
        seed_k: int = 3,
    ) -> None:
        super().__init__(embedder, store, top_k=top_k, min_score=min_score)
        self.graph = graph
        self.expand_k = expand_k
        self.seed_k = seed_k

    def retrieve(self, query: str, *, k: int | None = None, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        vec = self.embedder.embed_query(query)
        seeds = self._search(vec, k or self.top_k, where)
        if not seeds or self.expand_k <= 0:
            return seeds
        seen = {c.id for c in seeds}
        hits = [h for h in self.graph.neighbours([c.id for c in seeds[: self.seed_k]], limit=self.expand_k + len(seen)) if h.chunk_id not in seen]
        hits = hits[: self.expand_k]
        if not hits:
            return seeds
        via = {h.chunk_id: "graph: " + ", ".join(h.via) for h in hits}
        extra = [chunk_from_graph(r, via[r.chunk_id]) for r in self.graph.get_chunks([h.chunk_id for h in hits])]
        return seeds + extra
