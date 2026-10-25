"""HTTP surface for the agentic RAG. Thin: validate, call the agent, map errors."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.rag import AgenticRAG, RAGEvent
from agent.services.llm import AllModelsFailedError, BadRequestError, LLMError

from ..deps import get_rag
from ..schemas import (
    AgentStepOut,
    AskRequest,
    AskResponse,
    ErrorEvent,
    GraphStatsOut,
    HealthResponse,
    ModelEvent,
    ReflectEvent,
    SearchResponse,
    SourceOut,
    TokenEvent,
    ToolEvent,
)

router = APIRouter(prefix="/api", tags=["rag"])
log = logging.getLogger("app.rag")


def _http_error(exc: LLMError) -> tuple[int, str]:
    if isinstance(exc, AllModelsFailedError):
        return 503, str(exc)
    if isinstance(exc, BadRequestError):
        return 500, f"bad LLM request: {exc}"
    return 502, str(exc)


@router.get("/health", response_model=HealthResponse)
def health(rag: AgenticRAG = Depends(get_rag)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents=rag.document_count,
        chain=[r.key for r in rag.llm.chain],
        cooldowns=rag.llm.cooldowns(),
        max_steps=rag.max_steps,
        graph=GraphStatsOut(**rag.graph_store.stats()) if rag.graph_store is not None else None,
    )


@router.post("/search", response_model=SearchResponse)
def search(req: AskRequest, rag: AgenticRAG = Depends(get_rag)) -> SearchResponse:
    """Retrieval only, no agent. Useful to inspect what the search tool would return."""
    try:
        chunks = rag.search(req.question, k=req.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
    return SearchResponse(question=req.question, sources=[SourceOut.from_chunk(c) for c in chunks])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, rag: AgenticRAG = Depends(get_rag)) -> AskResponse:
    try:
        answer = rag.ask(req.question, k=req.top_k)
    except LLMError as exc:
        status, detail = _http_error(exc)
        raise HTTPException(status_code=status, detail=detail) from exc
    return AskResponse.from_answer(answer)


def _sse(event: str, data: BaseModel) -> str:
    return f"event: {event}\ndata: {json.dumps(data.model_dump(), ensure_ascii=False)}\n\n"


def _to_sse(ev: RAGEvent) -> str | None:
    if ev.event == "token":
        return _sse("token", TokenEvent(n=ev.n or 0, text=ev.text or ""))
    if ev.event == "model":
        return _sse("model", ModelEvent(n=ev.n or 0, steps=[AgentStepOut.from_step(s) for s in ev.steps]))
    if ev.event == "tool" and ev.step is not None:
        return _sse("tool", ToolEvent(n=ev.n or 0, step=AgentStepOut.from_step(ev.step), sources=[SourceOut.from_chunk(c) for c in ev.sources]))
    if ev.event == "reflect" and ev.step is not None:
        return _sse("reflect", ReflectEvent(n=ev.n or 0, step=AgentStepOut.from_step(ev.step)))
    if ev.event == "done" and ev.answer is not None:
        return _sse("done", AskResponse.from_answer(ev.answer))
    return None


@router.post("/ask/stream")
def ask_stream(req: AskRequest, rag: AgenticRAG = Depends(get_rag)) -> StreamingResponse:
    """Same as /ask, streamed as server-sent events: token, model, tool, reflect, done (or error).

    `token` carries answer/reasoning text as the model produces it; `model` marks a finished
    model call with the searches it issued (or its answer step); `tool` a finished search with
    its passages; `done` the full AskResponse. The stream always ends with `done` or `error`.
    """

    def events() -> Iterator[str]:
        try:
            for ev in rag.stream(req.question, k=req.top_k):
                frame = _to_sse(ev)
                if frame:
                    yield frame
        except LLMError as exc:
            status, detail = _http_error(exc)
            yield _sse("error", ErrorEvent(status=status, detail=detail))
        except Exception as exc:  # noqa: BLE001 - the stream is already open, so report instead of 500
            log.exception("agent stream failed")
            yield _sse("error", ErrorEvent(status=500, detail=f"{type(exc).__name__}: {exc}"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
