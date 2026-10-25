"""Offline tests for the agentic RAG loop and its API, using fakes for the embedder and LLM.
The vector store is a real in-memory ChromaDB and the agent loop is the real LangChain graph."""
from __future__ import annotations

import json
import math
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag import NO_CONTEXT_ANSWER, AgenticRAG, Document, ChromaVectorStore, Retriever  # noqa: E402
from agent.rag.prompts import OUT_OF_SEARCHES  # noqa: E402
from agent.rag.tools import NO_RESULTS  # noqa: E402
from agent.services.llm import FallbackLLM, ModelRef  # noqa: E402
from agent.services.llm.providers.base import Provider  # noqa: E402
from agent.services.llm.types import ProviderResult, ToolCall, Usage  # noqa: E402


# ----------------------------------------------------------------------- fakes
class WordEmbedder:
    """Deterministic bag-of-words embedding over a fixed vocabulary: good enough to rank."""

    VOCAB = ["detention", "pta", "months", "rti", "information", "days", "equality", "article", "company", "shares"]
    dim = len(VOCAB) + 1  # last slot = "unknown words only", orthogonal to every document

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        v = [float(t.count(w)) for w in self.VOCAB]
        if not any(v):
            return [0.0] * len(self.VOCAB) + [1.0]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] + [0.0]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ScriptedProvider(Provider):
    """Plays an agent: issues the scripted searches one per turn, then answers.

    `searches=None` means "search once with the user's question verbatim". Reflection calls
    (json_schema set) are answered with a verdict consistent with the script and recorded in
    `reflections`, not `calls`.
    """

    name = "fake"

    def __init__(self, searches: list[str] | None = None, answer: str = "Answer based on [1].", always_search: bool = False):
        self.searches = searches
        self.answer = answer
        self.always_search = always_search
        self.calls: list[list] = []  # messages of every agent model call
        self.reflections: list[list] = []  # messages of every reflect call

    def _remaining(self, done: int) -> list[str]:
        if self.always_search:
            return [f"PTA detention search {done}"]
        wanted = self.searches if self.searches is not None else []
        return [q for q in wanted[done:] if q]

    def complete(self, model, messages, tools=None, json_schema=None, **kw):
        if json_schema is not None:  # reflect step
            self.reflections.append(messages)
            # the reflect prompt lists one line per search run so far
            done = sum(1 for line in messages[-1].content.splitlines() if line[:2].rstrip(".").isdigit() and '"' in line)
            nxt = self._remaining(done)
            verdict = {
                "parts": [{"part": "the question", "status": "answered" if not nxt else "partial", "citations": [1]}],
                "decision": "search" if nxt else "answer",
                "queries": nxt[:1],
                "rationale": "scripted",
            }
            return ProviderResult(text=json.dumps(verdict), tool_calls=[], finish_reason="STOP", usage=Usage(3, 2, 5))
        self.calls.append(messages)
        done = sum(m.role == "tool" for m in messages)
        if tools:
            wanted = self.searches if self.searches is not None else [messages[-1].content if done == 0 else None]
            query = wanted[done] if done < len(wanted) and wanted[done] else None
            if self.always_search:
                query = f"PTA detention search {done}"  # hits the PTA passage every time
            if query:
                call = ToolCall(id=f"call-{done}", name="search_knowledge_base", arguments={"query": query})
                return ProviderResult(text="", tool_calls=[call], finish_reason="TOOL_CALLS", usage=Usage(10, 5, 15))
        return ProviderResult(text=self.answer, tool_calls=[], finish_reason="STOP", usage=Usage(10, 5, 15))

    async def acomplete(self, model, messages, **kw):
        return self.complete(model, messages, **kw)


DOCS = [
    Document("pta", "PTA detention may last up to twelve months after the 2022 amendment.", {"source": "PTA 2022"}),
    Document("rti", "RTI requests: a public authority must give information within fourteen days.", {"source": "RTI Act"}),
    Document("eq", "Article 12 guarantees equality before the law.", {"source": "Constitution Art 12"}),
    Document("co", "A private company may not have more than fifty shares holders.", {"source": "Companies Act"}),
]


def make_rag(provider: ScriptedProvider, **kw) -> AgenticRAG:
    store = ChromaVectorStore.ephemeral("test_kb")
    emb = WordEmbedder()
    store.upsert(DOCS, emb.embed_documents([d.text for d in DOCS]))
    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": provider}, sleep=lambda s: None)
    kw.setdefault("max_steps", 4)
    rag = AgenticRAG(Retriever(emb, store, top_k=2, min_score=0.1), llm, **kw)
    rag._provider = provider  # for assertions
    return rag


@pytest.fixture
def rag():
    return make_rag(ScriptedProvider())


# ---------------------------------------------------------------------- layers
def test_retriever_ranks_relevant_chunk_first(rag):
    chunks = rag.search("how many months of PTA detention")
    assert chunks and chunks[0].id == "pta"
    assert all(c.score >= 0.1 for c in chunks)


def test_min_score_filters_unrelated(rag):
    assert rag.search("zebra giraffe") == []


def test_agent_searches_then_answers_with_citations(rag):
    ans = rag.ask("How long is PTA detention?")
    calls = rag._provider.calls
    assert len(calls) == 2 and ans.steps == 2  # one search round, then the answer
    assert calls[0][0].role == "system" and "search_knowledge_base" in calls[0][0].content
    tool_msg = next(m for m in calls[1] if m.role == "tool")
    assert tool_msg.name == "search_knowledge_base" and "[1] Source: PTA 2022" in tool_msg.content
    assert ans.answer == "Answer based on [1]."
    assert ans.queries == ["How long is PTA detention?"]
    assert ans.sources[0].source == "PTA 2022"
    assert ans.model == "m" and ans.provider == "fake"
    assert ans.input_tokens == 20 and ans.output_tokens == 10  # summed over both agent model calls
    kinds = [(s.n, s.kind) for s in ans.trace]
    assert kinds == [(1, "search"), (1, "reflect"), (2, "answer")]
    search, reflect, final = ans.trace
    assert search.query == "How long is PTA detention?" and search.citations == [1] and search.new_passages == 1
    assert search.note is None and search.model == "m" and search.input_tokens == 10
    assert reflect.decision == "answer" and reflect.parts[0]["status"] == "answered" and reflect.text == "scripted"
    assert final.text == "Answer based on [1]." and final.note is None
    # the verdict was injected into the final call and the tools were withdrawn
    last_call = rag._provider.calls[-1]
    assert last_call[-1].role == "user" and "Decision: ANSWER now" in last_call[-1].content
    assert len(rag._provider.reflections) == 1 and "Searches run so far" in rag._provider.reflections[0][-1].content


def test_no_hits_yields_no_context_answer(rag):
    ans = rag.ask("zebra giraffe")
    assert ans.answer == NO_CONTEXT_ANSWER and ans.sources == []
    tool_msg = next(m for m in rag._provider.calls[1] if m.role == "tool")
    assert tool_msg.content == NO_RESULTS
    assert ans.trace[0].note == "no results" and ans.trace[0].citations == []


def test_citation_numbers_are_stable_across_searches():
    # third query is a reworded repeat of the first (an exact repeat would be vetoed by the reflect step)
    rag = make_rag(ScriptedProvider(searches=["PTA detention months", "RTI information days", "months of PTA detention"], answer="See [1] and [2]."))
    ans = rag.ask("Compare PTA detention and RTI deadlines")
    assert ans.queries == ["PTA detention months", "RTI information days", "months of PTA detention"]
    assert [s.id for s in ans.sources][:2] == ["pta", "rti"]  # first-seen order, no duplicates
    assert len({s.id for s in ans.sources}) == len(ans.sources)
    third_result = [m for m in rag._provider.calls[3] if m.role == "tool"][-1].content
    assert third_result.startswith("[1] Source: PTA 2022")  # same number the second time around
    assert ans.steps == 4
    searches = [s for s in ans.trace if s.kind == "search"]
    assert [s.citations[0] for s in searches] == [1, 2, 1]
    assert [s.new_passages for s in searches][-1] == 0  # repeat search found nothing new
    reflects = [s for s in ans.trace if s.kind == "reflect"]
    assert [r.decision for r in reflects] == ["search", "search", "answer"]
    assert reflects[0].next_queries == ["RTI information days"]  # proposed query reached the trace
    # and the proposal was handed to the model before its next call
    assert 'Decision: SEARCH now' in rag._provider.calls[1][-1].content and '"RTI information days"' in rag._provider.calls[1][-1].content


def test_search_budget_forces_an_answer():
    rag = make_rag(ScriptedProvider(always_search=True, answer="forced"), max_steps=2)
    ans = rag.ask("anything")
    calls = rag._provider.calls
    assert ans.steps == 3 and len(ans.queries) == 2  # 2 searches allowed, 3rd call must answer
    last = calls[-1]
    assert last[-1].role == "user" and last[-1].content == OUT_OF_SEARCHES
    assert ans.answer == "forced"
    assert ans.trace[-1].kind == "answer" and ans.trace[-1].note == "search budget exhausted"
    assert len(rag._provider.reflections) == 1  # no reflect call once the budget is spent


def test_reflect_vetoes_exact_repeat_searches():
    rag = make_rag(ScriptedProvider(searches=["PTA detention months", "PTA detention months"], answer="done"))
    ans = rag.ask("PTA?")
    assert ans.queries == ["PTA detention months"]  # repeat proposal dropped -> decision became "answer"
    assert [s.decision for s in ans.trace if s.kind == "reflect"] == ["answer"] and ans.steps == 2


def test_reflect_can_be_disabled():
    rag = make_rag(ScriptedProvider(), reflect=False)
    ans = rag.ask("How long is PTA detention?")
    assert [s.kind for s in ans.trace] == ["search", "answer"] and rag._provider.reflections == []


def test_reflect_failure_does_not_break_the_run():
    class Broken(ScriptedProvider):
        def complete(self, model, messages, tools=None, json_schema=None, **kw):
            if json_schema is not None:
                return ProviderResult(text="not json at all", tool_calls=[], finish_reason="STOP", usage=Usage())
            return super().complete(model, messages, tools=tools, **kw)

    rag = make_rag(Broken())
    ans = rag.ask("How long is PTA detention?")
    assert ans.answer == "Answer based on [1]." and [s.kind for s in ans.trace] == ["search", "answer"]


def test_tool_result_respects_context_budget():
    rag = make_rag(ScriptedProvider(searches=["information days rti"]), max_context_chars=60)
    rag.ask("RTI?")
    tool_msg = next(m for m in rag._provider.calls[1] if m.role == "tool")
    assert tool_msg.content.count("Source:") == 1  # only the first block fits (first is always kept)


def test_embedding_failure_surfaces_as_llm_error():
    from agent.services.llm import ProviderServerError

    rag = make_rag(ScriptedProvider())

    def boom(text):
        raise ProviderServerError("embedding down", provider="gemini", model="emb")

    rag.retriever.embedder.embed_query = boom
    with pytest.raises(ProviderServerError):
        rag.ask("How long is PTA detention?")


def test_adapter_round_trips_tool_call_provider_data():
    """Gemini 3 thought signatures must survive the LangChain message round trip."""
    from agent.services.llm.langchain_adapter import to_ai_message, to_internal
    from agent.services.llm.types import LLMResponse

    tc = ToolCall(id="c1", name="search_knowledge_base", arguments={"query": "x"}, provider_data={"thought_signature": b"sig"})
    resp = LLMResponse(text="", tool_calls=[tc], provider="gemini", model="g", finish_reason="STOP", usage=Usage(1, 1, 2), attempts=[])
    ai = to_ai_message(resp)
    back = to_internal([ai])[0]
    assert back.role == "assistant" and back.tool_calls[0].id == "c1"
    assert back.tool_calls[0].provider_data == {"thought_signature": b"sig"}


# ------------------------------------------------------------------------- api
@pytest.fixture
def client(rag):
    from app.main import create_app

    app = create_app(rag=rag)  # injected: no .env, no network
    app.state.ingestor = None
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == 4 and body["chain"] == ["fake:m"] and body["max_steps"] == 4


def test_ask_endpoint(client):
    r = client.post("/api/ask", json={"question": "RTI information within how many days?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Answer")
    assert body["sources"][0]["source"] == "RTI Act"
    assert body["provider"] == "fake" and body["steps"] == 2
    assert body["queries"] == ["RTI information within how many days?"]
    assert [s["kind"] for s in body["trace"]] == ["search", "reflect", "answer"]
    assert body["trace"][0]["citations"][0] == 1 and body["trace"][0]["query"] == body["queries"][0]


def test_search_endpoint(client):
    r = client.post("/api/search", json={"question": "article equality", "top_k": 1})
    assert r.status_code == 200 and [s["id"] for s in r.json()["sources"]] == ["eq"]


def test_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "top_k": 99}).status_code == 422


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "LankaLaw" in r.text
