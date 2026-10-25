"""Streaming: provider -> fallback chain -> LangChain adapter -> agent events -> SSE endpoint."""
from __future__ import annotations

import json
import os
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.services.llm import FallbackLLM, Message, ModelRef, ProviderServerError, ToolSpec  # noqa: E402
from agent.services.llm.providers.base import Provider  # noqa: E402
from agent.services.llm.providers.groq import GroqProvider  # noqa: E402
from agent.services.llm.types import ProviderResult, StreamChunk, ToolCall, Usage  # noqa: E402
from tests.test_agentic_rag import ScriptedProvider, make_rag  # noqa: E402


# ------------------------------------------------------------- provider layer
class TokenProvider(Provider):
    """Streams 'a','b','c' then a result; `fail_after` tokens raises mid-stream."""

    name = "fake"

    def __init__(self, fail_after: int | None = None, fail_first: bool = False):
        self.fail_after = fail_after
        self.fail_first = fail_first

    def complete(self, model, messages, **kw):
        return ProviderResult(text="abc", tool_calls=[], finish_reason="STOP", usage=Usage(1, 1, 2))

    async def acomplete(self, model, messages, **kw):
        return self.complete(model, messages, **kw)

    def stream(self, model, messages, **kw):
        if self.fail_first:
            raise ProviderServerError("down", provider="fake", model=model)
        for i, ch in enumerate("abc"):
            if self.fail_after is not None and i >= self.fail_after:
                raise ProviderServerError("cut", provider="fake", model=model)
            yield StreamChunk(text=ch)
        yield StreamChunk(result=self.complete(model, messages))


def test_base_provider_stream_default_is_single_chunk():
    p = ScriptedProvider(answer="hello")
    chunks = list(p.stream("m", [Message.user("hi")]))
    assert [c.text for c in chunks] == ["hello"] and chunks[0].result.text == "hello"


def test_fallback_stream_yields_tokens_then_response():
    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": TokenProvider()}, sleep=lambda s: None)
    events = list(llm.stream([Message.user("hi")]))
    assert "".join(e.text for e in events) == "abc"
    assert events[-1].response is not None and events[-1].response.text == "abc" and not events[-1].response.fell_back


def test_fallback_stream_falls_back_before_first_token():
    class Two(Provider):
        name = "fake"

        def __init__(self):
            self.good = TokenProvider()
            self.bad = TokenProvider(fail_first=True)

        def complete(self, model, messages, **kw):
            return self.good.complete(model, messages)

        async def acomplete(self, model, messages, **kw):
            return self.complete(model, messages)

        def stream(self, model, messages, **kw):
            return (self.bad if model == "m1" else self.good).stream(model, messages, **kw)

    llm = FallbackLLM([ModelRef("fake", "m1"), ModelRef("fake", "m2")], {"fake": Two()}, max_retries_per_model=0, sleep=lambda s: None)
    events = list(llm.stream([Message.user("hi")]))
    assert "".join(e.text for e in events) == "abc"
    assert events[-1].response.model == "m2" and events[-1].response.fell_back


def test_fallback_stream_does_not_restart_after_tokens_were_sent():
    llm = FallbackLLM([ModelRef("fake", "m1"), ModelRef("fake", "m2")], {"fake": TokenProvider(fail_after=1)}, sleep=lambda s: None)
    got = []
    with pytest.raises(ProviderServerError):
        for e in llm.stream([Message.user("hi")]):
            got.append(e.text)
    assert got == ["a"]  # the caller saw one token, so no silent retry on another model


def test_groq_sse_stream_parses_text_and_tool_calls():
    sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":"Let me "},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"look."},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search_knowledge_base","arguments":"{\\"que"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ry\\": \\"PTA\\"}"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"x_groq":{"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}}',
            "data: [DONE]",
            "",
        ]
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    p = GroqProvider("key", base_url="https://groq.test/openai/v1")
    p._client = httpx.Client(transport=httpx.MockTransport(handler))
    chunks = list(p.stream("gpt", [Message.user("q")], tools=[ToolSpec(name="search_knowledge_base", description="d")]))
    assert seen["body"]["stream"] is True
    assert "".join(c.text for c in chunks) == "Let me look."
    res = chunks[-1].result
    assert res.text == "Let me look." and res.finish_reason == "tool_calls" and res.usage.input_tokens == 7
    assert res.tool_calls == [ToolCall(id="call_1", name="search_knowledge_base", arguments={"query": "PTA"})]


def test_groq_sse_stream_http_error_is_classified():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    p = GroqProvider("key", base_url="https://groq.test/openai/v1")
    p._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderServerError):
        list(p.stream("gpt", [Message.user("q")]))


# ------------------------------------------------------------ adapter layer
def test_langchain_adapter_stream_reassembles_tool_calls():
    from agent.services.llm.langchain_adapter import FallbackChatModel

    class SearchingProvider(TokenProvider):
        def complete(self, model, messages, **kw):
            tc = ToolCall(id="c1", name="search_knowledge_base", arguments={"query": "x"}, provider_data={"thought_signature": b"s"})
            return ProviderResult(text="abc", tool_calls=[tc], finish_reason="TOOL_CALLS", usage=Usage(1, 1, 2))

    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": SearchingProvider()}, sleep=lambda s: None)
    model = FallbackChatModel(llm=llm).bind_tools([{"type": "function", "function": {"name": "search_knowledge_base", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}])
    chunks = list(model.stream("hi"))
    assert [c.text for c in chunks][:3] == ["a", "b", "c"]
    merged = chunks[0]
    for c in chunks[1:]:
        merged = merged + c
    assert merged.text == "abc"
    assert merged.tool_calls == [{"name": "search_knowledge_base", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}]
    assert merged.additional_kwargs["tool_call_provider_data"] == {"c1": {"thought_signature": b"s"}}
    assert merged.usage_metadata["input_tokens"] == 1 and merged.response_metadata["model"] == "m"


# -------------------------------------------------------------- agent events
def test_agent_stream_event_sequence():
    rag = make_rag(ScriptedProvider(searches=["PTA detention months", "RTI information days"], answer="See [1][2]."))
    events = list(rag.stream("Compare PTA detention and RTI deadlines"))
    kinds = [e.event for e in events]
    assert kinds[-1] == "done"
    assert kinds.count("model") == 3 and kinds.count("tool") == 2 and kinds.count("reflect") == 2
    # reflect comes after the tool results it grades and before the next model call
    assert kinds.index("reflect") > kinds.index("tool") and kinds.index("reflect") < [i for i, k in enumerate(kinds) if k == "model"][1]
    first_reflect = next(e for e in events if e.event == "reflect")
    assert first_reflect.n == 1 and first_reflect.step.kind == "reflect" and first_reflect.step.decision == "search"
    # a search model-call is announced as pending, then completed by a tool event carrying sources so far
    first_model = next(e for e in events if e.event == "model")
    assert first_model.n == 1 and first_model.steps[0].kind == "search" and first_model.steps[0].pending
    first_tool = next(e for e in events if e.event == "tool")
    assert first_tool.n == 1 and not first_tool.step.pending and first_tool.step.citations[0] == 1
    assert [s.id for s in first_tool.sources][0] == "pta"
    # tokens of the final call stream before its model event, tagged with the in-progress call number
    tokens = [e for e in events if e.event == "token"]
    assert tokens and tokens[-1].n == 3 and "".join(t.text for t in tokens if t.n == 3) == "See [1][2]."
    answer = events[-1].answer
    assert answer.steps == 3 and answer.answer == "See [1][2]."
    assert [s.kind for s in answer.trace] == ["search", "reflect", "search", "reflect", "answer"]
    assert all(not s.pending for s in answer.trace)


def test_ask_equals_last_stream_event():
    rag = make_rag(ScriptedProvider())
    a = rag.ask("How long is PTA detention?")
    assert a.answer == "Answer based on [1]." and a.trace[0].call_id == "call-0"


# ---------------------------------------------------------------- SSE route
def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for frame in text.strip().split("\n\n"):
        ev, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        out.append((ev, json.loads(data)))
    return out


@pytest.fixture
def client():
    from app.main import create_app

    rag = make_rag(ScriptedProvider(searches=["PTA detention months"], answer="Answer [1]."))
    app = create_app(rag=rag)
    app.state.ingestor = None
    with TestClient(app) as c:
        yield c


def test_stream_endpoint_emits_events_and_final_answer(client):
    r = client.post("/api/ask/stream", json={"question": "PTA?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] in ("token", "model") and names[-1] == "done"
    assert "tool" in names and "reflect" in names
    reflect = next(d for e, d in events if e == "reflect")
    assert reflect["step"]["kind"] == "reflect" and reflect["step"]["decision"] == "answer"
    tool = next(d for e, d in events if e == "tool")
    assert tool["step"]["kind"] == "search" and tool["step"]["citations"][0] == 1 and tool["sources"][0]["id"] == "pta"
    done = events[-1][1]
    assert done["answer"] == "Answer [1]." and done["steps"] == 2 and done["trace"][-1]["kind"] == "answer"


def test_stream_endpoint_reports_errors_in_band(client):
    def boom(text):
        raise ProviderServerError("embedding down", provider="gemini", model="emb")

    client.app.state.rag.retriever.embedder.embed_query = boom
    events = _parse_sse(client.post("/api/ask/stream", json={"question": "PTA?"}).text)
    assert events[-1][0] == "error" and events[-1][1]["status"] == 502 and "embedding down" in events[-1][1]["detail"]
