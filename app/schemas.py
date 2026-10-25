"""HTTP request/response models. Kept separate from the domain dataclasses."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ingestion import IndexedDocument, IngestResult
from agent.rag import RAGAnswer, RetrievedChunk
from agent.rag.schemas import AgentStep
from agent.services.db.graphstore import ChunkRecord, EntityInfo

# ------------------------------------------------------------------------ Q&A


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(None, ge=1, le=20)


class SourceOut(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]
    source: str
    via: str | None = None  # None = similarity search; else how the knowledge graph pulled it in

    @classmethod
    def from_chunk(cls, c: RetrievedChunk) -> "SourceOut":
        return cls(id=c.id, text=c.text, score=c.score, metadata=c.metadata, source=c.source, via=c.via)


class AgentStepOut(BaseModel):
    n: int
    kind: str  # "search" | "reflect" | "answer"
    call_id: str | None
    tool: str | None  # search: which tool ("search_knowledge_base" | "explore_knowledge_graph")
    pending: bool
    provider: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    text: str | None
    query: str | None
    citations: list[int]
    new_passages: int
    note: str | None
    decision: str | None  # reflect: "search" | "answer"
    parts: list[dict[str, Any]]  # reflect: [{part, status, citations}]
    next_queries: list[str]  # reflect: proposed searches

    @classmethod
    def from_step(cls, s: AgentStep) -> "AgentStepOut":
        return cls(**{k: getattr(s, k) for k in cls.model_fields})


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    provider: str | None
    model: str | None
    fell_back: bool
    input_tokens: int
    output_tokens: int
    queries: list[str]  # searches the agent ran, in order
    steps: int  # model calls
    trace: list[AgentStepOut]  # step-by-step: searches (query, passages returned) and the answer

    @classmethod
    def from_answer(cls, a: RAGAnswer) -> "AskResponse":
        return cls(
            question=a.question,
            answer=a.answer,
            sources=[SourceOut.from_chunk(c) for c in a.sources],
            provider=a.provider,
            model=a.model,
            fell_back=a.fell_back,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            queries=a.queries,
            steps=a.steps,
            trace=[AgentStepOut.from_step(s) for s in a.trace],
        )


class SearchResponse(BaseModel):
    question: str
    sources: list[SourceOut]


# Server-sent events emitted by POST /api/ask/stream (one `event:` line + one JSON `data:` line each).
class TokenEvent(BaseModel):
    n: int  # model call in progress
    text: str  # delta


class ModelEvent(BaseModel):
    n: int
    steps: list[AgentStepOut]  # pending search steps, or the answer step


class ToolEvent(BaseModel):
    n: int
    step: AgentStepOut  # completed search
    sources: list[SourceOut]  # all sources retrieved so far, in citation order


class ReflectEvent(BaseModel):
    n: int  # model call whose results were graded
    step: AgentStepOut  # kind == "reflect"


class ErrorEvent(BaseModel):
    status: int
    detail: str


class GraphStatsOut(BaseModel):
    chunks: int
    entities: int
    mentions: int
    relations: int


class HealthResponse(BaseModel):
    status: str
    documents: int
    chain: list[str]
    cooldowns: dict[str, float]
    max_steps: int
    graph: GraphStatsOut | None  # None when the knowledge graph is disabled


# ----------------------------------------------------------------- knowledge graph


class EntityOut(BaseModel):
    key: str
    label: str
    type: str
    mentions: int

    @classmethod
    def from_info(cls, e: EntityInfo) -> "EntityOut":
        return cls(key=e.key, label=e.label, type=e.type, mentions=e.mentions)


class TripleOut(BaseModel):
    subject: str
    predicate: str
    object: str
    chunk_id: str
    doc_id: str


class GraphPassageOut(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str

    @classmethod
    def from_record(cls, r: ChunkRecord) -> "GraphPassageOut":
        return cls(chunk_id=r.chunk_id, doc_id=r.doc_id, source=r.source, chunk_index=r.chunk_index, text=r.text)


class EntitiesResponse(BaseModel):
    query: str | None
    entities: list[EntityOut]
    stats: GraphStatsOut


class EntityContextResponse(BaseModel):
    entities: list[EntityOut]
    triples: list[TripleOut]
    passages: list[GraphPassageOut]


# ------------------------------------------------------------------- ingestion

Scalar = str | int | float | bool


class IngestTextRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=300, description="Label shown as the citation, e.g. 'RTI Act No. 12 of 2016, s. 25'")
    text: str = Field(..., min_length=1, max_length=2_000_000)
    metadata: dict[str, Scalar] = Field(default_factory=dict)


class IngestResultOut(BaseModel):
    doc_id: str
    source: str
    chunks: int
    chars: int
    replaced: bool
    entities: int  # graph entities extracted (0 when the graph is off)
    relations: int
    graph_errors: int

    @classmethod
    def from_result(cls, r: IngestResult) -> "IngestResultOut":
        return cls(
            doc_id=r.doc_id, source=r.source, chunks=r.chunks, chars=r.chars, replaced=r.replaced,
            entities=r.entities, relations=r.relations, graph_errors=r.graph_errors,
        )


class IngestRecord(BaseModel):
    id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=200_000)
    metadata: dict[str, Scalar] = Field(default_factory=dict)


class IngestRecordsRequest(BaseModel):
    """Pre-chunked passages (the seed-file format). Each record becomes its own one-chunk document."""

    records: list[IngestRecord] = Field(..., min_length=1, max_length=5000)


class IngestRecordsResponse(BaseModel):
    indexed: int
    total_chunks: int
    graph: GraphStatsOut | None


class IngestFileError(BaseModel):
    filename: str
    error: str


class IngestBatchResponse(BaseModel):
    results: list[IngestResultOut]
    errors: list[IngestFileError]


class IndexedDocumentOut(BaseModel):
    doc_id: str
    source: str
    chunks: int
    ingested_at: str | None
    metadata: dict[str, Any]

    @classmethod
    def from_doc(cls, d: IndexedDocument) -> "IndexedDocumentOut":
        return cls(doc_id=d.doc_id, source=d.source, chunks=d.chunks, ingested_at=d.ingested_at, metadata=d.metadata)


class DocumentsResponse(BaseModel):
    documents: list[IndexedDocumentOut]
    total_chunks: int


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int


class IngestSettingsOut(BaseModel):
    chunk_size: int
    chunk_overlap: int
    max_upload_mb: float
    allowed_extensions: list[str]
