"""ChromaDB-backed VectorStore (persisted to disk, or in-memory for tests)."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

if TYPE_CHECKING:  # type-only: avoids an import cycle with agent.rag at runtime
    from agent.rag.schemas import Document, RetrievedChunk


class ChromaVectorStore:
    """Local persistent ChromaDB collection using cosine distance."""

    def __init__(self, path: str | Path, collection: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    @classmethod
    def ephemeral(cls, collection: str | None = None) -> "ChromaVectorStore":
        """In-memory store for tests. The ephemeral client is process-wide, so each call gets
        its own collection name unless one is given, keeping tests isolated."""
        obj = cls.__new__(cls)
        obj._client = chromadb.EphemeralClient()
        name = collection or f"ephemeral_{uuid.uuid4().hex[:12]}"
        obj._col = obj._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        return obj

    def upsert(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if not docs:
            return
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        self._col.upsert(
            ids=[d.id for d in docs],
            embeddings=embeddings,
            documents=[d.text for d in docs],
            metadatas=[d.metadata or {"_": ""} for d in docs],  # chroma rejects empty metadata dicts
        )

    def search(self, query_embedding: list[float], k: int, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        from agent.rag.schemas import RetrievedChunk  # local import: see TYPE_CHECKING note above

        total = self.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        chunks = []
        for i, text, meta, dist in zip(ids, docs, metas, dists):
            meta = {k: v for k, v in (meta or {}).items() if k != "_"}
            chunks.append(RetrievedChunk(id=i, text=text, metadata=meta, score=round(1.0 - float(dist), 4)))
        return chunks

    def count(self) -> int:
        return self._col.count()

    def all_metadata(self) -> list[tuple[str, dict[str, Any]]]:
        if self.count() == 0:
            return []
        res = self._col.get(include=["metadatas"])
        return [(i, {k: v for k, v in (m or {}).items() if k != "_"}) for i, m in zip(res["ids"], res["metadatas"])]

    def delete_where(self, where: dict[str, Any]) -> int:
        before = self.count()
        if before == 0:
            return 0
        self._col.delete(where=where)
        return before - self.count()
