"""Domain objects passed between RAG layers. No framework types here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of text stored in the vector store (already chunk-sized)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity, higher is better
    via: str | None = None  # None = vector search; otherwise how the graph pulled it in, e.g. "graph: detention, PTA"

    @property
    def source(self) -> str:
        return str(self.metadata.get("source") or self.metadata.get("title") or self.id)


@dataclass
class Reflection:
    """Verdict of the reflect step that runs after each round of search results."""

    after_call: int  # model call whose results were reviewed
    parts: list[dict[str, Any]]  # [{"part": str, "status": answered|partial|unanswered, "citations": [int]}]
    decision: str  # "search" | "answer"
    queries: list[str]  # proposed next searches (decision == "search")
    rationale: str
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_messages_seen: int = 0  # how many tool results existed when this was made (dedup key)


@dataclass
class AgentStep:
    """One thing the agent did, in order: a search it ran, a reflection, or the answer it gave."""

    n: int  # 1-based model call this step belongs to (parallel searches share an n; a reflect step follows call n)
    kind: str  # "search" | "reflect" | "answer"
    call_id: str | None = None  # tool call id (search only); lets a live UI match results to the step
    tool: str | None = None  # search only: "search_knowledge_base" | "explore_knowledge_graph"
    pending: bool = False  # search issued, result not in yet (streaming only)
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    text: str | None = None  # model text on this call (reasoning before a search, or the answer)
    query: str | None = None  # search only
    citations: list[int] = field(default_factory=list)  # passage numbers the search returned
    new_passages: int = 0  # how many of those were first seen in this search
    note: str | None = None  # "no results", "search failed: ...", "search budget exhausted"
    decision: str | None = None  # reflect only: "search" | "answer"
    parts: list[dict[str, Any]] = field(default_factory=list)  # reflect only: per-part verdicts
    next_queries: list[str] = field(default_factory=list)  # reflect only: proposed searches


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    provider: str | None = None
    model: str | None = None
    fell_back: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    queries: list[str] = field(default_factory=list)  # searches the agent ran, in order
    steps: int = 0  # model calls it took
    trace: list[AgentStep] = field(default_factory=list)  # what it did, step by step


@dataclass
class RAGEvent:
    """One item of `AgenticRAG.stream`.

    event == "token":  `n` (model call in progress) and `text` (delta the model just produced)
    event == "model":  model call `n` finished; `steps` are its search steps (pending) or its answer step
    event == "tool":   one search finished; `step` is complete and `sources` is the full source list so far
    event == "reflect": the reflect step graded the results; `step` carries parts, decision and next queries
    event == "done":   `answer` is the final RAGAnswer (same object `ask()` returns)
    """

    event: str
    n: int | None = None
    text: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    step: AgentStep | None = None
    sources: list[RetrievedChunk] = field(default_factory=list)
    answer: RAGAnswer | None = None
