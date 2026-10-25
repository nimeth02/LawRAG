"""Agentic RAG: a LangChain agent (LangGraph loop) that decides when and what to retrieve.

Flow per question:  model -> search_knowledge_base(query) [one or many in parallel] -> model -> ... -> answer.
The model states what it is looking for, searches (several queries in one turn for multi-part
questions), reads the passages, and may search again, up to `max_steps` model calls; after
that it gets no tools and must answer with what it has.

`stream()` yields live events (tokens, finished model calls, finished searches, final answer);
`ask()` is the same run collapsed to its final answer.

The LLM is our own `FallbackLLM` (Gemini -> Groq chain) wrapped as a LangChain chat model,
so provider fallback, retries and cooldowns keep working inside the agent loop.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage

from agent.services.db.graphstore import GraphStore, KuzuGraphStore
from agent.services.db.vectorstore import ChromaVectorStore, VectorStore
from agent.services.llm import FallbackLLM, LLMSettings
from agent.services.llm.langchain_adapter import FallbackChatModel

from .config import RAGSettings
from .embeddings import Embedder, GeminiEmbedder
from .prompts import OUT_OF_SEARCHES, SYSTEM_PROMPT
from .reflect import ReflectAfterRetrieval
from .retriever import GraphExpandingRetriever, Retriever
from .schemas import AgentStep, RAGAnswer, RAGEvent, Reflection, RetrievedChunk
from .tools import RetrievalLog, SearchRecord, make_graph_tool, make_search_tool

log = logging.getLogger("agent.rag")

NO_CONTEXT_ANSWER = "I could not find anything relevant to that question in the knowledge base."


class ForceFinalAnswer(AgentMiddleware[Any, RetrievalLog]):
    """Once the model has been called `max_steps` times, strip the tools so the next call must answer."""

    def __init__(self, max_steps: int) -> None:
        super().__init__()
        self.max_steps = max_steps

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        calls = sum(isinstance(m, AIMessage) for m in request.messages)
        if calls < self.max_steps or not request.tools:
            return request
        return request.override(tools=[], messages=[*request.messages, HumanMessage(OUT_OF_SEARCHES)])

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._prepare(request))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:  # type: ignore[override]
        return await handler(self._prepare(request))


class AgenticRAG:
    """Question in, cited answer out. The only object the API layer talks to for Q&A."""

    def __init__(
        self,
        retriever: Retriever,
        llm: FallbackLLM,
        *,
        max_steps: int = 6,
        max_context_chars: int = 12_000,
        temperature: float = 0.0,
        system_prompt: str = SYSTEM_PROMPT,
        reflect: bool = True,
        graph: GraphStore | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.reflect = reflect
        self.graph_store = graph
        self.model = FallbackChatModel(llm=llm, temperature=temperature)
        self.search_tool = make_search_tool(retriever, max_result_chars=max_context_chars)
        self.tools = [self.search_tool]
        if graph is not None:
            self.tools.append(make_graph_tool(graph, max_result_chars=max_context_chars))
        middleware: list[AgentMiddleware] = []
        if reflect:
            middleware.append(ReflectAfterRetrieval(llm, max_steps, max_context_chars=max_context_chars))
        middleware.append(ForceFinalAnswer(max_steps))
        self.graph = create_agent(
            self.model,
            self.tools,
            system_prompt=system_prompt,
            middleware=middleware,
            context_schema=RetrievalLog,
            name="lankalaw-rag",
        )

    # ------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        llm_settings: LLMSettings,
        rag: RAGSettings,
        embedder: Embedder,
        store: VectorStore,
        graph: GraphStore | None = None,
        llm: FallbackLLM | None = None,
    ) -> "AgenticRAG":
        """Assemble from already-constructed embedder/store/graph so other services can share them."""
        if graph is not None:
            retriever: Retriever = GraphExpandingRetriever(
                embedder, store, graph, top_k=rag.top_k, min_score=rag.min_score, expand_k=rag.graph_expand_k
            )
        else:
            retriever = Retriever(embedder, store, top_k=rag.top_k, min_score=rag.min_score)
        agent = cls(
            retriever,
            llm or FallbackLLM.from_settings(llm_settings),
            max_steps=rag.max_steps,
            max_context_chars=rag.max_context_chars,
            reflect=rag.reflect,
            graph=graph,
        )
        log.info(
            "agentic rag ready: %d chunks in '%s', max %d model calls, reflect=%s, graph=%s",
            store.count(), rag.collection, rag.max_steps, rag.reflect, graph.stats() if graph else "off",
        )
        return agent

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "AgenticRAG":
        llm_settings = LLMSettings.from_env(dotenv_path)
        rag = RAGSettings.from_env(dotenv_path)
        if not llm_settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for embeddings")
        embedder = GeminiEmbedder(llm_settings.gemini_api_key, rag.embedding_model, rag.embedding_dim)
        store = ChromaVectorStore(rag.chroma_path, rag.collection)
        graph = KuzuGraphStore(rag.graph_path) if rag.graph_enabled else None
        return cls.build(llm_settings, rag, embedder, store, graph)

    # ------------------------------------------------------------------- use
    def search(self, question: str, *, k: int | None = None, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        """Plain retrieval, no agent. Useful to inspect what the search tool would return."""
        return self.retriever.retrieve(question, k=k, where=where)

    def ask(self, question: str, *, k: int | None = None, where: dict[str, Any] | None = None) -> RAGAnswer:
        for ev in self.stream(question, k=k, where=where):
            if ev.event == "done" and ev.answer is not None:
                return ev.answer
        raise RuntimeError("agent run produced no answer")  # pragma: no cover - stream() always ends with done

    def stream(self, question: str, *, k: int | None = None, where: dict[str, Any] | None = None) -> Iterator[RAGEvent]:
        """Run the agent and yield `RAGEvent`s as it goes. The last event is always "done"."""
        run = RetrievalLog(k=k, where=where)
        messages: list[BaseMessage] = [HumanMessage(question)]
        n = 0  # model calls completed so far
        call_n: dict[str, int] = {}  # tool_call_id -> model call that issued it
        call_query: dict[str, str] = {}
        reflected = 0  # reflections already emitted

        for mode, payload in self.graph.stream(
            {"messages": [HumanMessage(question)]},
            context=run,
            config={"recursion_limit": 3 * self.max_steps + 4},  # model + tool + reflect node per step, plus slack
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, _meta = payload
                if isinstance(chunk, AIMessageChunk) and chunk.text:
                    yield RAGEvent(event="token", n=n + 1, text=chunk.text)
                continue

            while reflected < len(run.reflections):  # reflect ran inside a before_model node
                ref = run.reflections[reflected]
                reflected += 1
                yield RAGEvent(event="reflect", n=ref.after_call, step=self._reflect_step(ref))

            for update in payload.values():  # updates: {node_name: {"messages": [...]}}
                new = (update or {}).get("messages") if isinstance(update, dict) else None
                for m in new or []:
                    messages.append(m)
                    if isinstance(m, AIMessage):
                        n += 1
                        steps = self._steps_for_call(n, m, run, pending=True)
                        for tc in m.tool_calls:
                            call_n[tc["id"]] = n
                            call_query[tc["id"]] = self._query_of(tc)
                        yield RAGEvent(event="model", n=n, steps=steps)
                    elif isinstance(m, ToolMessage):
                        cid = m.tool_call_id
                        step = self._search_step(
                            call_n.get(cid, n), cid, call_query.get(cid), run.searches.get(cid), text=None, meta={}, tool=m.name
                        )
                        yield RAGEvent(event="tool", n=call_n.get(cid, n), step=step, sources=run.sources)

        yield RAGEvent(event="done", answer=self._finish(question, messages, run))

    # ------------------------------------------------------------- assembling
    def _finish(self, question: str, messages: list[BaseMessage], run: RetrievalLog) -> RAGAnswer:
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        final = ai_messages[-1] if ai_messages else None

        if run.error is not None and not run.chunks:
            raise run.error  # every search failed at the embedding layer: surface it, don't fake an answer

        answer = (final.text if final else "").strip()
        if not run.chunks:
            answer = NO_CONTEXT_ANSWER
        meta = final.response_metadata if final else {}
        trace: list[AgentStep] = []
        for i, m in enumerate(ai_messages, start=1):
            trace.extend(self._steps_for_call(i, m, run, final_text=answer if m is final else None))
            trace.extend(self._reflect_step(r) for r in run.reflections if r.after_call == i)
        return RAGAnswer(
            question=question,
            answer=answer,
            sources=run.sources,
            provider=meta.get("provider"),
            model=meta.get("model"),
            fell_back=any(m.response_metadata.get("fell_back") for m in ai_messages),
            input_tokens=sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai_messages),
            output_tokens=sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai_messages),
            queries=run.queries,
            steps=len(ai_messages),
            trace=trace,
        )

    @staticmethod
    def _reflect_step(ref: Reflection) -> AgentStep:
        return AgentStep(
            n=ref.after_call,
            kind="reflect",
            text=ref.rationale or None,
            decision=ref.decision,
            parts=ref.parts,
            next_queries=list(ref.queries),
            provider=ref.provider,
            model=ref.model,
            input_tokens=ref.input_tokens,
            output_tokens=ref.output_tokens,
        )

    @staticmethod
    def _call_meta(m: AIMessage) -> dict[str, Any]:
        meta = m.response_metadata or {}
        usage = m.usage_metadata or {}
        return dict(
            provider=meta.get("provider"),
            model=meta.get("model"),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    @staticmethod
    def _query_of(tc: dict[str, Any]) -> str:
        args = tc.get("args") or {}
        return str(args.get("query") or args.get("entity") or next(iter(args.values()), "") or "")

    @staticmethod
    def _search_step(
        n: int,
        call_id: str | None,
        query: str | None,
        rec: SearchRecord | None,
        *,
        text: str | None,
        meta: dict[str, Any],
        pending: bool = False,
        tool: str | None = None,
    ) -> AgentStep:
        if pending:
            note = None
        elif rec is None:
            note = "tool call not executed"
        elif rec.error:
            note = f"search failed: {rec.error}"
        elif not rec.citations:
            note = "no results"
        else:
            note = None
        return AgentStep(
            n=n,
            kind="search",
            call_id=call_id,
            tool=(rec.tool if rec else None) or tool,
            pending=pending,
            text=text,
            query=query or (rec.query if rec else None),
            citations=list(rec.citations) if rec else [],
            new_passages=rec.new_passages if rec else 0,
            note=note,
            **meta,
        )

    def _steps_for_call(self, n: int, m: AIMessage, run: RetrievalLog, *, pending: bool = False, final_text: str | None = None) -> list[AgentStep]:
        """Steps for one model call: one per tool call it issued, or one answer step."""
        meta = self._call_meta(m)
        text = (m.text or "").strip() or None
        if m.tool_calls:
            steps = []
            for tc in m.tool_calls:
                cid = tc.get("id") or ""
                rec = None if pending else run.searches.get(cid)
                steps.append(
                    self._search_step(
                        n, cid, self._query_of(tc), rec, text=text, meta=meta, pending=pending and cid not in run.searches, tool=tc.get("name")
                    )
                )
                text = None  # attach the model's reasoning to the first search only
            return steps
        return [
            AgentStep(
                kind="answer",
                n=n,
                text=final_text if final_text is not None else text,
                # ForceFinalAnswer withdraws the tools on call max_steps+1 (the nudge it adds is
                # request-only and never reaches the stored messages), so detect it by count.
                note="search budget exhausted" if n > self.max_steps else None,
                **meta,
            )
        ]

    @property
    def document_count(self) -> int:
        return self.retriever.store.count()
