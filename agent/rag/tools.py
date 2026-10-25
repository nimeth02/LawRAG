"""Tools the agent can call, plus the per-request log that turns tool results into citations.

Citation numbers are assigned on first retrieval and stay stable for the whole run, so a
passage found by two different tool calls is always `[n]` for the same n, and `RAGAnswer.sources`
lines up with the numbers in the answer text.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from langchain.tools import ToolRuntime, tool

from agent.services.db.graphstore import GraphStore
from agent.services.llm import LLMError

from .retriever import Retriever, chunk_from_graph
from .schemas import Reflection, RetrievedChunk

SEARCH_TOOL = "search_knowledge_base"
GRAPH_TOOL = "explore_knowledge_graph"

NO_RESULTS = "No relevant passages found. Try different wording, a narrower topic, or the exact act or section name."
NO_ENTITY = "No matching entity in the knowledge graph. Try another name, or search_knowledge_base for free text."


@dataclass
class SearchRecord:
    """What one tool call did (keyed by tool_call_id in the log)."""

    query: str
    tool: str = SEARCH_TOOL
    citations: list[int] = field(default_factory=list)
    new_passages: int = 0
    error: str | None = None


@dataclass
class RetrievalLog:
    """Request-scoped state handed to tools through `ToolRuntime.context`."""

    k: int | None = None
    where: dict[str, Any] | None = None
    chunks: dict[str, RetrievedChunk] = field(default_factory=dict)  # insertion order == citation order
    searches: dict[str, SearchRecord] = field(default_factory=dict)  # tool_call_id -> record, in call order
    reflections: list[Reflection] = field(default_factory=list)  # one per reflect step, in order
    error: LLMError | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, call_id: str, query: str, results: list[RetrievedChunk], *, tool: str = SEARCH_TOOL) -> list[tuple[int, RetrievedChunk]]:
        rec = SearchRecord(query=query, tool=tool)
        numbered: list[tuple[int, RetrievedChunk]] = []
        with self._lock:  # parallel tool calls run on worker threads
            for c in results:
                if c.id not in self.chunks:
                    self.chunks[c.id] = c
                    rec.new_passages += 1
                n = self.number(c.id)
                rec.citations.append(n)
                numbered.append((n, c))
            self.searches[call_id] = rec
        return numbered

    def record_error(self, call_id: str, query: str, exc: LLMError, *, tool: str = SEARCH_TOOL) -> None:
        self.error = exc
        self.searches[call_id] = SearchRecord(query=query, tool=tool, error=str(exc))

    def number(self, chunk_id: str) -> int:
        return list(self.chunks).index(chunk_id) + 1

    @property
    def queries(self) -> list[str]:
        return [s.query for s in self.searches.values()]

    @property
    def sources(self) -> list[RetrievedChunk]:
        return list(self.chunks.values())


def format_passages(numbered: list[tuple[int, RetrievedChunk]], max_chars: int) -> str:
    """Render `[n] Source: ...` blocks within a character budget (the first block is always kept)."""
    blocks: list[str] = []
    used = 0
    for n, c in numbered:
        block = f"[{n}] Source: {c.source}\n{c.text.strip()}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def make_search_tool(retriever: Retriever, *, max_result_chars: int = 12_000):
    """Build the `search_knowledge_base` tool bound to a retriever."""

    @tool(SEARCH_TOOL)
    def search_knowledge_base(query: str, runtime: ToolRuntime[RetrievalLog]) -> str:
        """Search the Sri Lankan law knowledge base for passages relevant to `query`.

        Returns passages ranked by similarity, each prefixed with a citation number like
        "[3] Source: ...". Cite with those numbers. If the passages do not answer the question,
        call again with different wording, a narrower sub-question, or the exact act/section name.
        """
        log = runtime.context
        call_id = runtime.tool_call_id or f"call-{len(log.searches)}"
        try:
            results = retriever.retrieve(query, k=log.k, where=log.where)
        except LLMError as exc:  # embedding provider failed: remember it, let the agent continue
            log.record_error(call_id, query, exc)
            return f"Search failed: {exc}"
        numbered = log.record(call_id, query, results)
        return format_passages(numbered, max_result_chars) if numbered else NO_RESULTS

    return search_knowledge_base


def make_graph_tool(graph: GraphStore, *, max_result_chars: int = 12_000, max_chunks: int = 5):
    """Build the `explore_knowledge_graph` tool: entity lookup -> relations + passages that mention it."""

    @tool(GRAPH_TOOL)
    def explore_knowledge_graph(entity: str, runtime: ToolRuntime[RetrievalLog]) -> str:
        """Look up an act, section, institution, person or legal concept in the knowledge graph.

        Returns the relations it has with other entities (e.g. AMENDS, REPEALS, DEFINES,
        TIME_LIMIT, APPLIES_TO) and the passages that mention it, numbered for citation.
        Use it for questions about how things connect: what amended or repealed an act, what a
        term is defined as, which provisions refer to a section, what a body must do. For free-text
        questions use search_knowledge_base instead.
        """
        log = runtime.context
        call_id = runtime.tool_call_id or f"call-{len(log.searches)}"
        ents = graph.search_entities(entity, limit=5)
        if not ents:
            log.record(call_id, entity, [], tool=GRAPH_TOOL)
            return NO_ENTITY
        limit = max(log.k or max_chunks, 1)
        ctx = graph.entity_context([e.key for e in ents], limit_chunks=limit)
        labels = ", ".join(e.label for e in ents[:3])
        chunks = [chunk_from_graph(r, f"graph: {labels}") for r in graph.get_chunks(ctx.chunk_ids[:limit])]
        numbered = log.record(call_id, entity, chunks, tool=GRAPH_TOOL)
        cite = {c.id: n for n, c in numbered}

        lines = ["Entities: " + "; ".join(f"{e.label} ({e.type}, {e.mentions} passage(s))" for e in ents)]
        if ctx.triples:
            lines.append("Relations:")
            for t in ctx.triples[:40]:
                ref = f" [{cite[t.chunk_id]}]" if t.chunk_id in cite else ""
                lines.append(f"- {t.subject} —{t.predicate}→ {t.object}{ref}")
        else:
            lines.append("Relations: none recorded.")
        if numbered:
            lines.append("")
            lines.append("Passages:")
            lines.append(format_passages(numbered, max_result_chars))
        return "\n".join(lines)

    return explore_knowledge_graph


__all__ = [
    "GRAPH_TOOL",
    "NO_ENTITY",
    "NO_RESULTS",
    "SEARCH_TOOL",
    "RetrievalLog",
    "SearchRecord",
    "format_passages",
    "make_graph_tool",
    "make_search_tool",
]
