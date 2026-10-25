"""Offline tests for the ingestion layer and its API. Fake embedder, real in-memory Chroma."""
from __future__ import annotations

import hashlib
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import (  # noqa: E402
    EmptyDocumentError,
    IngestionService,
    IngestSettings,
    UnsupportedFileError,
    chunk_text,
    doc_id_for,
    extract_text,
)
from agent.rag import AgenticRAG, Document, Retriever  # noqa: E402
from agent.services.db.vectorstore import ChromaVectorStore  # noqa: E402
from agent.services.llm import FallbackLLM, ModelRef  # noqa: E402
from agent.services.llm.providers.base import Provider  # noqa: E402
from agent.services.llm.types import ProviderResult, Usage  # noqa: E402


class HashEmbedder:
    """Deterministic bag-of-hashed-words embedding; related texts land near each other."""

    dim = 32

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class StubProvider(Provider):
    name = "fake"

    def complete(self, model, messages, **kw):
        return ProviderResult(text="ok [1]", tool_calls=[], finish_reason="STOP", usage=Usage())

    async def acomplete(self, model, messages, **kw):
        return self.complete(model, messages)


# --------------------------------------------------------------------- chunker
def test_chunker_keeps_short_paragraphs_whole():
    text = "Para one is short.\n\nPara two is also short."
    assert chunk_text(text, size=200, overlap=0) == [text]


def test_chunker_splits_at_paragraphs_and_overlaps():
    paras = [f"Paragraph {i} " + "word " * 30 for i in range(6)]
    chunks = chunk_text("\n\n".join(paras), size=400, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 40 + 1 for c in chunks)
    # each later chunk starts with the tail of the previous one
    assert chunks[1].split(" ")[0] in chunks[0]


def test_chunker_hard_splits_runaway_text():
    chunks = chunk_text("x" * 1000, size=300, overlap=0)
    assert len(chunks) == 4 and all(len(c) <= 300 for c in chunks)


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
    assert chunk_text("   ", size=10, overlap=0) == []


# --------------------------------------------------------------------- loaders
def test_extract_text_plain_and_markdown():
    assert extract_text("a.txt", "hello\r\n\r\n\r\n\r\nworld".encode()) == "hello\n\nworld"
    assert extract_text("notes.MD", "# Title\n\nbody".encode("utf-8-sig")) == "# Title\n\nbody"


def test_extract_text_pdf():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(EmptyDocumentError):  # blank page has no text
        extract_text("blank.pdf", buf.getvalue())


def test_extract_text_rejects_unknown_and_empty():
    with pytest.raises(UnsupportedFileError):
        extract_text("sheet.xlsx", b"...")
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"   \n  ")


# --------------------------------------------------------------------- service
@pytest.fixture
def svc():
    store = ChromaVectorStore.ephemeral()
    return IngestionService(HashEmbedder(), store, IngestSettings(chunk_size=120, chunk_overlap=20))


def test_ingest_text_chunks_and_lists(svc):
    text = "\n\n".join(f"Section {i}: " + "detention rules " * 6 for i in range(5))
    r = svc.ingest_text(text, source="PTA s.9", metadata={"act": "PTA", "year": 1979, "empty": "", "doc_id": "hack"})
    assert r.chunks > 1 and r.doc_id == doc_id_for("PTA s.9") and not r.replaced
    docs = svc.list_documents()
    assert len(docs) == 1
    d = docs[0]
    assert d.source == "PTA s.9" and d.chunks == r.chunks
    assert d.metadata == {"act": "PTA", "year": 1979}  # reserved + empty keys dropped
    assert svc.store.count() == r.chunks


def test_reingest_same_source_replaces(svc):
    svc.ingest_text("one two three " * 40, source="RTI s.25")
    r2 = svc.ingest_text("short now", source="rti s.25 ")  # case/space-insensitive same source
    assert r2.replaced and r2.chunks == 1
    assert svc.store.count() == 1 and len(svc.list_documents()) == 1


def test_delete_document(svc):
    r = svc.ingest_text("alpha beta " * 50, source="X")
    svc.ingest_text("gamma", source="Y")
    assert svc.delete_document(r.doc_id) == r.chunks
    assert [d.source for d in svc.list_documents()] == ["Y"]
    assert svc.delete_document("nope") == 0


def test_ingest_file_uses_filename_and_limits(svc):
    r = svc.ingest_file("act.txt", b"The Act says things.", metadata={"year": 2020})
    assert r.source == "act.txt" and svc.list_documents()[0].metadata == {"year": 2020, "filename": "act.txt"}
    svc.settings.max_upload_mb = 0.00001
    from ingestion import FileTooLargeError

    with pytest.raises(FileTooLargeError):
        svc.ingest_file("big.txt", b"x" * 1000)


def test_ingested_text_is_searchable_by_rag(svc):
    svc.ingest_text("Fourteen working days to answer an RTI request.", source="RTI")
    svc.ingest_text("Twelve months maximum detention under the PTA.", source="PTA")
    retriever = Retriever(svc.embedder, svc.store, top_k=1, min_score=0.0)
    assert retriever.retrieve("RTI request working days")[0].source == "RTI"


def test_index_prechunked_sets_doc_metadata(svc):
    n = svc.index_prechunked([Document("c1", "Article 12 equality", {"source": "Const 12", "topic": "rights"})])
    assert n == 1
    d = svc.list_documents()[0]
    assert d.doc_id == "c1" and d.source == "Const 12" and d.chunks == 1 and d.metadata == {"topic": "rights"}


# ------------------------------------------------------------------------- api
@pytest.fixture
def client(svc):
    from app.main import create_app

    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": StubProvider()}, sleep=lambda s: None)
    rag = AgenticRAG(Retriever(svc.embedder, svc.store, top_k=3), llm)
    with TestClient(create_app(rag=rag, ingestor=svc)) as c:
        yield c


def test_api_text_then_list_then_delete(client):
    r = client.post("/api/ingest/text", json={"source": "Companies Act s.4", "text": "A private company " * 30, "metadata": {"year": 2007}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks"] >= 1 and body["replaced"] is False

    docs = client.get("/api/ingest/documents").json()
    assert docs["documents"][0]["source"] == "Companies Act s.4" and docs["total_chunks"] == body["chunks"]

    assert client.get("/api/health").json()["documents"] == body["chunks"]

    d = client.delete(f"/api/ingest/documents/{body['doc_id']}")
    assert d.status_code == 200 and d.json()["deleted_chunks"] == body["chunks"]
    assert client.delete(f"/api/ingest/documents/{body['doc_id']}").status_code == 404


def test_api_file_upload(client):
    files = [("files", ("notes.md", b"# PDPA\n\nPenalties up to ten million rupees.", "text/markdown"))]
    r = client.post("/api/ingest/files", files=files, data={"source": "PDPA note", "metadata": '{"act": "PDPA"}'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["errors"] == [] and body["results"][0]["source"] == "PDPA note"
    doc = client.get("/api/ingest/documents").json()["documents"][0]
    assert doc["metadata"]["act"] == "PDPA" and doc["metadata"]["filename"] == "notes.md"


def test_api_file_errors(client):
    bad = [("files", ("data.xlsx", b"zzz", "application/octet-stream"))]
    assert client.post("/api/ingest/files", files=bad).status_code == 400
    mixed = [("files", ("ok.txt", b"fine text", "text/plain")), ("files", ("bad.xlsx", b"zzz", "application/octet-stream"))]
    r = client.post("/api/ingest/files", files=mixed)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1 and r.json()["errors"][0]["filename"] == "bad.xlsx"
    assert client.post("/api/ingest/files", files=[("files", ("a.txt", b"x", "text/plain"))], data={"metadata": "[1,2]"}).status_code == 400


def test_api_text_validation(client):
    assert client.post("/api/ingest/text", json={"source": "", "text": "x"}).status_code == 422
    assert client.post("/api/ingest/text", json={"source": "s", "text": "   "}).status_code == 400


def test_api_settings(client):
    s = client.get("/api/ingest/settings").json()
    assert s["chunk_size"] == 120 and ".pdf" in s["allowed_extensions"]
