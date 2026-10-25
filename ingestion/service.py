"""Ingestion use-cases: text or file in, chunks embedded and stored, documents listed/deleted.

Shares the Embedder and VectorStore with the RAG pipeline so what is ingested here is
immediately searchable there. When a GraphStore is attached, every chunk is also written to
the knowledge graph with the entities and relations an extractor finds in it.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from agent.rag.embeddings import Embedder
from agent.rag.schemas import Document
from agent.services.db.graphstore import ChunkFacts, GraphStore
from agent.services.db.vectorstore import VectorStore
from agent.services.llm import LLMError

from .chunker import chunk_text
from .config import IngestSettings
from .graph_extractor import GraphExtractor
from .loaders import EmptyDocumentError, UnsupportedFileError, extract_text, normalise_text
from .schemas import IndexedDocument, IngestResult

log = logging.getLogger("ingestion")

RESERVED_KEYS = {"doc_id", "source", "chunk_index", "chunk_count", "ingested_at"}
Scalar = str | int | float | bool


class FileTooLargeError(ValueError):
    pass


def doc_id_for(source: str) -> str:
    """Stable id per source label, so re-ingesting the same source replaces it."""
    return hashlib.sha1(source.strip().lower().encode("utf-8")).hexdigest()[:16]


def clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Scalar]:
    """Keep only scalar values the vector store accepts; drop reserved and empty keys."""
    out: dict[str, Scalar] = {}
    for k, v in (metadata or {}).items():
        key = str(k).strip()
        if not key or key in RESERVED_KEYS or v is None or v == "":
            continue
        if isinstance(v, bool | int | float):
            out[key] = v
        else:
            out[key] = str(v).strip()
    return out


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: IngestSettings | None = None,
        *,
        graph: GraphStore | None = None,
        extractor: GraphExtractor | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings or IngestSettings()
        self.graph = graph
        self.extractor = extractor

    # ------------------------------------------------------------------ ingest
    def ingest_text(self, text: str, *, source: str, metadata: dict[str, Any] | None = None) -> IngestResult:
        source = source.strip()
        if not source:
            raise ValueError("source is required")
        text = normalise_text(text)
        if not text:
            raise EmptyDocumentError("text is empty")

        chunks = chunk_text(text, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap)
        doc_id = doc_id_for(source)
        replaced = self.store.delete_where({"doc_id": doc_id}) > 0
        if self.graph is not None:
            self.graph.delete_document(doc_id)

        base = {
            **clean_metadata(metadata),
            "source": source,
            "doc_id": doc_id,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunk_count": len(chunks),
        }
        docs = [
            Document(id=f"{doc_id}-{i:04d}", text=chunk, metadata={**base, "chunk_index": i})
            for i, chunk in enumerate(chunks)
        ]
        vectors = self.embedder.embed_documents([d.text for d in docs])
        self.store.upsert(docs, vectors)
        ents, rels, errs = self._index_graph(docs)
        log.info(
            "ingested '%s' -> %d chunks (%d chars)%s; graph: %d entities, %d relations, %d failures",
            source, len(docs), len(text), " [replaced]" if replaced else "", ents, rels, errs,
        )
        return IngestResult(
            doc_id=doc_id, source=source, chunks=len(docs), chars=len(text), replaced=replaced,
            entities=ents, relations=rels, graph_errors=errs,
        )

    def ingest_file(
        self,
        filename: str,
        data: bytes,
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        if len(data) > self.settings.max_upload_bytes:
            raise FileTooLargeError(f"{filename} is {len(data) / 1e6:.1f} MB; limit is {self.settings.max_upload_mb} MB")
        text = extract_text(filename, data)  # raises UnsupportedFileError / EmptyDocumentError
        meta = {**(metadata or {}), "filename": filename}
        return self.ingest_text(text, source=source or filename, metadata=meta)

    def index_prechunked(self, docs: list[Document]) -> int:
        """Index passages that are already chunk-sized (e.g. the seed file). Each doc is its own document."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prepared = []
        for d in docs:
            meta = clean_metadata(d.metadata)
            meta.update(
                source=str(d.metadata.get("source") or d.id),
                doc_id=d.id,
                chunk_index=0,
                chunk_count=1,
                ingested_at=now,
            )
            prepared.append(Document(id=d.id, text=normalise_text(d.text), metadata=meta))  # same id: re-seeding overwrites
        for d in prepared:
            self.store.delete_where({"doc_id": d.metadata["doc_id"]})
            if self.graph is not None:
                self.graph.delete_document(str(d.metadata["doc_id"]))
        vectors = self.embedder.embed_documents([d.text for d in prepared])
        self.store.upsert(prepared, vectors)
        self._index_graph(prepared)
        return len(prepared)

    def _index_graph(self, docs: list[Document]) -> tuple[int, int, int]:
        """Write chunks (+ extracted entities/relations) to the graph. Returns (entities, relations, failures)."""
        if self.graph is None:
            return 0, 0, 0
        ents = rels = errs = 0
        for d in docs:
            facts = ChunkFacts(
                chunk_id=d.id,
                doc_id=str(d.metadata.get("doc_id") or d.id),
                source=str(d.metadata.get("source") or d.id),
                chunk_index=int(d.metadata.get("chunk_index") or 0),
                text=d.text,
            )
            if self.extractor is not None and self.settings.graph_extract:
                try:
                    found = self.extractor.extract(d.text, source=facts.source)
                    facts.entities, facts.relations = found.entities, found.relations
                except LLMError as exc:  # best effort: keep the chunk node, lose only its links
                    errs += 1
                    log.warning("graph extraction failed for %s: %s", d.id, exc)
            self.graph.upsert(facts)
            ents += len(facts.entities)
            rels += len(facts.relations)
        return ents, rels, errs

    # ------------------------------------------------------------------ manage
    def list_documents(self) -> list[IndexedDocument]:
        groups: dict[str, IndexedDocument] = {}
        for chunk_id, meta in self.store.all_metadata():
            meta = meta or {}
            doc_id = str(meta.get("doc_id") or chunk_id)
            doc = groups.get(doc_id)
            if doc is None:
                doc = IndexedDocument(
                    doc_id=doc_id,
                    source=str(meta.get("source") or doc_id),
                    chunks=0,
                    ingested_at=meta.get("ingested_at"),
                    metadata={k: v for k, v in meta.items() if k not in RESERVED_KEYS},
                )
                groups[doc_id] = doc
            doc.chunks += 1
        return sorted(groups.values(), key=lambda d: (d.ingested_at or "", d.source), reverse=True)

    def delete_document(self, doc_id: str) -> int:
        removed = self.store.delete_where({"doc_id": doc_id})
        if self.graph is not None:
            self.graph.delete_document(doc_id)
        return removed


__all__ = [
    "EmptyDocumentError",
    "FileTooLargeError",
    "IngestionService",
    "UnsupportedFileError",
    "clean_metadata",
    "doc_id_for",
]
