"""HTTP surface for ingestion. Thin: validate, call the service, map errors."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ingestion import (
    EmptyDocumentError,
    FileTooLargeError,
    IngestionService,
    UnsupportedFileError,
    supported_extensions,
)
from agent.services.llm import LLMError

from ..deps import get_ingestor
from agent.rag import Document

from ..schemas import (
    DeleteResponse,
    DocumentsResponse,
    GraphStatsOut,
    IndexedDocumentOut,
    IngestBatchResponse,
    IngestFileError,
    IngestRecordsRequest,
    IngestRecordsResponse,
    IngestResultOut,
    IngestSettingsOut,
    IngestTextRequest,
)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.get("/settings", response_model=IngestSettingsOut)
def settings(ingestor: IngestionService = Depends(get_ingestor)) -> IngestSettingsOut:
    s = ingestor.settings
    return IngestSettingsOut(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        max_upload_mb=s.max_upload_mb,
        allowed_extensions=list(supported_extensions()),
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(ingestor: IngestionService = Depends(get_ingestor)) -> DocumentsResponse:
    docs = ingestor.list_documents()
    return DocumentsResponse(documents=[IndexedDocumentOut.from_doc(d) for d in docs], total_chunks=sum(d.chunks for d in docs))


@router.post("/text", response_model=IngestResultOut)
def ingest_text(req: IngestTextRequest, ingestor: IngestionService = Depends(get_ingestor)) -> IngestResultOut:
    try:
        result = ingestor.ingest_text(req.text, source=req.source, metadata=req.metadata)
    except (EmptyDocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
    return IngestResultOut.from_result(result)


@router.post("/records", response_model=IngestRecordsResponse)
def ingest_records(req: IngestRecordsRequest, ingestor: IngestionService = Depends(get_ingestor)) -> IngestRecordsResponse:
    """Bulk-load pre-chunked passages (seed format). Used by scripts/seed_kb.py while the server
    is running, because the embedded graph database allows only one writer process."""
    docs = [Document(id=r.id, text=r.text, metadata=dict(r.metadata)) for r in req.records]
    try:
        n = ingestor.index_prechunked(docs)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
    graph = ingestor.graph
    return IngestRecordsResponse(
        indexed=n,
        total_chunks=ingestor.store.count(),
        graph=GraphStatsOut(**graph.stats()) if graph is not None else None,
    )


@router.post("/files", response_model=IngestBatchResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    source: str | None = Form(None, description="Override the source label (only sensible for a single file)"),
    metadata: str | None = Form(None, description="JSON object of extra metadata applied to every file"),
    ingestor: IngestionService = Depends(get_ingestor),
) -> IngestBatchResponse:
    try:
        meta = json.loads(metadata) if metadata else {}
        if not isinstance(meta, dict):
            raise ValueError("metadata must be a JSON object")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad metadata: {exc}") from exc

    results: list[IngestResultOut] = []
    errors: list[IngestFileError] = []
    for f in files:
        name = f.filename or "upload"
        data = await f.read()
        try:
            label = source.strip() if source and len(files) == 1 else None
            results.append(IngestResultOut.from_result(ingestor.ingest_file(name, data, source=label, metadata=meta)))
        except FileTooLargeError as exc:
            if len(files) == 1:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            errors.append(IngestFileError(filename=name, error=str(exc)))
        except (UnsupportedFileError, EmptyDocumentError, ValueError) as exc:
            if len(files) == 1:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            errors.append(IngestFileError(filename=name, error=str(exc)))
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
    return IngestBatchResponse(results=results, errors=errors)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, ingestor: IngestionService = Depends(get_ingestor)) -> DeleteResponse:
    deleted = ingestor.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"no document with id {doc_id}")
    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)
