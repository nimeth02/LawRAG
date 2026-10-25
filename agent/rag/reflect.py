"""Reflect step: after each round of search results, grade coverage and decide what to do next.

Implemented as LangChain agent middleware:
  * `before_model`   - when the last message is a tool result (and the budget is not spent),
                       make a separate structured model call that lists the parts of the
                       question, marks each answered / partial / unanswered with citations,
                       and decides "search" (with new queries) or "answer". The verdict is
                       stored on the request-scoped RetrievalLog so it shows up as a step.
  * `wrap_model_call` - inject that verdict into the next model request as guidance; on
                       "answer" the tools are withdrawn so the loop ends deterministically.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.services.llm import FallbackLLM, LLMError, Message

from .schemas import Reflection
from .tools import RetrievalLog, format_passages

log = logging.getLogger("agent.rag.reflect")

STATUSES = ("answered", "partial", "unanswered")

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string", "description": "One distinct thing the question asks"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "citations": {"type": "array", "items": {"type": "integer"}, "description": "Passage numbers that answer it"},
                },
                "required": ["part", "status", "citations"],
            },
        },
        "decision": {"type": "string", "enum": ["search", "answer"]},
        "queries": {"type": "array", "items": {"type": "string"}, "description": "New searches to run if decision is search"},
        "rationale": {"type": "string", "description": "One or two sentences"},
    },
    "required": ["parts", "decision", "queries", "rationale"],
}

REFLECT_SYSTEM = """You are the planning step of a legal research agent for Sri Lankan law.
You see the user's question, the searches run so far and the passages they returned.
Your job: judge coverage and decide the next move. Output JSON only, matching the schema.

Rules:
- Split the question into its distinct parts (one part for a simple question).
- For each part: "answered" if the passages state it, "partial" if only some of it, else
  "unanswered". List the passage numbers that support it.
- Decide "search" if a part is unanswered or partial AND a differently worded, narrower or
  more specific search (act name, section number, key legal term) could plausibly find it.
  Propose up to {max_queries} new queries that differ from those already run.
- Decide "answer" if every part is answered, or the missing parts have already been
  searched for with different wordings without success. Do not keep searching for what the
  knowledge base clearly does not contain."""


def _text(m: Any) -> str:
    t = getattr(m, "text", "")
    return t if isinstance(t, str) else str(t or "")


def reflection_note(ref: Reflection) -> str:
    """What the main model is told before its next call."""
    lines = [f"[Reflection after {ref.after_call} model call(s)] {ref.rationale}".rstrip()]
    for p in ref.parts:
        cites = "".join(f"[{c}]" for c in p.get("citations", []))
        lines.append(f"- {p.get('part', '')}: {p.get('status', '')} {cites}".rstrip())
    if ref.decision == "search" and ref.queries:
        qs = "; ".join(f'"{q}"' for q in ref.queries)
        lines.append(
            f"Decision: SEARCH now. In this turn call search_knowledge_base once per query: {qs}. "
            "If a query names a specific act, section or legal term, explore_knowledge_graph (when available) may find its relations."
        )
    else:
        lines.append("Decision: ANSWER now from the passages retrieved. Do not search again.")
    return "\n".join(lines)


class ReflectAfterRetrieval(AgentMiddleware[Any, RetrievalLog]):
    def __init__(self, llm: FallbackLLM, max_steps: int, *, max_queries: int = 3, max_context_chars: int = 12_000) -> None:
        super().__init__()
        self.llm = llm
        self.max_steps = max_steps
        self.max_queries = max_queries
        self.max_context_chars = max_context_chars

    # ----------------------------------------------------------------- hooks
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[-1], ToolMessage):
            return None  # nothing new to review
        calls = sum(isinstance(m, AIMessage) for m in msgs)
        if calls >= self.max_steps:
            return None  # ForceFinalAnswer will end the run; a verdict would be wasted
        run: RetrievalLog = runtime.context
        tool_count = sum(isinstance(m, ToolMessage) for m in msgs)
        if run.reflections and run.reflections[-1].tool_messages_seen == tool_count:
            return None  # already reflected on this round (node re-run)
        question = next((_text(m) for m in msgs if isinstance(m, HumanMessage)), "")
        ref = self.reflect(question, run, after_call=calls, tool_count=tool_count)
        if ref is not None:
            run.reflections.append(ref)
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def _apply(self, request: ModelRequest) -> ModelRequest:
        run: RetrievalLog = request.runtime.context
        if not run.reflections:
            return request
        ref = run.reflections[-1]
        if ref.tool_messages_seen != sum(isinstance(m, ToolMessage) for m in request.messages):
            return request  # stale verdict (belongs to an earlier round)
        messages = [*request.messages, HumanMessage(reflection_note(ref))]
        if ref.decision == "answer":
            return request.override(tools=[], messages=messages)
        return request.override(messages=messages)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:  # type: ignore[override]
        return await handler(self._apply(request))

    # ------------------------------------------------------------- the call
    def build_prompt(self, question: str, run: RetrievalLog) -> list[Message]:
        searches = "\n".join(
            f'{i}. {s.tool}("{s.query}") -> ' + (f"search failed: {s.error}" if s.error else f"passages {s.citations}" if s.citations else "no results")
            for i, s in enumerate(run.searches.values(), start=1)
        )
        numbered = [(i, c) for i, c in enumerate(run.sources, start=1)]
        passages = format_passages(numbered, self.max_context_chars) if numbered else "(none)"
        user = (
            f"Question: {question}\n\n"
            f"Searches run so far:\n{searches or '(none)'}\n\n"
            f"Passages retrieved so far (numbered as the agent cites them):\n{passages}\n\n"
            "Grade coverage and decide: search (with new queries) or answer."
        )
        return [Message.system(REFLECT_SYSTEM.format(max_queries=self.max_queries)), Message.user(user)]

    def reflect(self, question: str, run: RetrievalLog, *, after_call: int, tool_count: int) -> Reflection | None:
        try:
            resp = self.llm.invoke(self.build_prompt(question, run), json_schema=REFLECTION_SCHEMA, temperature=0.0)
        except LLMError as exc:
            log.warning("reflect step skipped: %s", exc)
            return None
        try:
            data = json.loads(resp.text)
            parts = [
                {
                    "part": str(p.get("part", "")).strip(),
                    "status": p["status"] if p.get("status") in STATUSES else "unanswered",
                    "citations": [int(c) for c in p.get("citations", []) if str(c).lstrip("-").isdigit()],
                }
                for p in data.get("parts", [])
                if isinstance(p, dict)
            ]
            decision = "search" if data.get("decision") == "search" else "answer"
            seen = {s.query.strip().lower() for s in run.searches.values()}
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            queries = [q for q in queries if q.lower() not in seen][: self.max_queries]
            if decision == "search" and not queries:
                decision = "answer"  # nothing new to try: searching again would just repeat
            return Reflection(
                after_call=after_call,
                parts=parts,
                decision=decision,
                queries=queries if decision == "search" else [],
                rationale=str(data.get("rationale", "")).strip(),
                provider=resp.provider,
                model=resp.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                tool_messages_seen=tool_count,
            )
        except (ValueError, TypeError, AttributeError) as exc:
            log.warning("reflect step produced unusable output (%s): %.200s", exc, resp.text)
            return None


__all__ = ["REFLECTION_SCHEMA", "ReflectAfterRetrieval", "reflection_note"]
