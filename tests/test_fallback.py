"""Unit tests for the fallback chain using fake providers (no network)."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.services.llm import (  # noqa: E402
    AllModelsFailedError,
    AuthenticationError,
    BadRequestError,
    FallbackLLM,
    LLMSettings,
    Message,
    ModelRef,
    ModelUnavailableError,
    RateLimitError,
)
from agent.services.llm.errors import classify_http  # noqa: E402
from agent.services.llm.providers.base import Provider  # noqa: E402
from agent.services.llm.types import ProviderResult, Usage  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeProvider(Provider):
    """`script` maps model -> list of outcomes consumed per call. An outcome is either a
    string (returned as text) or an exception instance (raised). Unknown models raise
    ModelUnavailableError, like a 404 from a real provider."""

    def __init__(self, name: str, script: dict[str, list]) -> None:
        self.name = name
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def _next(self, model: str) -> ProviderResult:
        self.calls.append(model)
        outcomes = self.script.get(model)
        if not outcomes:
            raise ModelUnavailableError("model not found", provider=self.name, model=model, status=404)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return ProviderResult(text=out, tool_calls=[], finish_reason="STOP", usage=Usage())

    def complete(self, model, messages, **kw):
        return self._next(model)

    async def acomplete(self, model, messages, **kw):
        return self._next(model)


def build(gemini_script, groq_script, **kw):
    clock = FakeClock()
    gem = FakeProvider("gemini", gemini_script)
    grq = FakeProvider("groq", groq_script)
    chain = [ModelRef("gemini", "g1"), ModelRef("gemini", "g2"), ModelRef("groq", "q1")]
    llm = FallbackLLM(chain, {"gemini": gem, "groq": grq}, clock=clock, sleep=clock.sleep, **kw)
    return llm, gem, grq, clock


MSG = [Message.user("hello")]


def test_primary_succeeds_no_fallback():
    llm, gem, grq, _ = build({"g1": ["ok"]}, {})
    r = llm.invoke(MSG)
    assert r.text == "ok" and r.model == "g1" and not r.fell_back
    assert gem.calls == ["g1"] and grq.calls == []


def test_rate_limit_retries_same_model_then_moves_on():
    rl = RateLimitError("429", provider="gemini", model="g1", status=429)
    llm, gem, grq, clock = build({"g1": [rl, rl], "g2": ["from g2"]}, {}, max_retries_per_model=1)
    r = llm.invoke(MSG)
    assert r.text == "from g2" and r.fell_back
    assert gem.calls == ["g1", "g1", "g2"]  # one retry on g1, then g2
    assert [a.error_kind for a in r.attempts[:-1]] == ["rate_limit", "rate_limit"]
    assert "gemini:g1" in llm.cooldowns()  # g1 parked


def test_retry_after_header_is_honoured():
    rl = RateLimitError("429", provider="gemini", model="g1", status=429, retry_after_s=3.0)
    llm, gem, _, clock = build({"g1": [rl, "ok"]}, {}, max_retries_per_model=1)
    t0 = clock.t
    llm.invoke(MSG)
    assert clock.t - t0 == pytest.approx(3.0)


def test_unknown_model_skips_immediately_without_retry():
    llm, gem, grq, _ = build({"g2": ["g2 ok"]}, {}, max_retries_per_model=3)
    r = llm.invoke(MSG)
    assert r.model == "g2" and gem.calls == ["g1", "g2"]


def test_auth_error_skips_whole_provider():
    auth = AuthenticationError("bad key", provider="gemini", model="g1", status=401)
    llm, gem, grq, _ = build({"g1": [auth], "g2": ["should not be reached"]}, {"q1": ["groq ok"]})
    r = llm.invoke(MSG)
    assert r.provider == "groq" and r.text == "groq ok"
    assert gem.calls == ["g1"]  # g2 never tried
    assert set(llm.cooldowns()) == {"gemini:g1", "gemini:g2"}


def test_bad_request_raises_immediately():
    bad = BadRequestError("schema invalid", provider="gemini", model="g1", status=400)
    llm, gem, grq, _ = build({"g1": [bad], "g2": ["nope"]}, {"q1": ["nope"]})
    with pytest.raises(BadRequestError):
        llm.invoke(MSG)
    assert gem.calls == ["g1"] and grq.calls == []


def test_all_fail_raises_with_attempt_log():
    llm, gem, grq, _ = build({}, {}, max_retries_per_model=0)
    with pytest.raises(AllModelsFailedError) as ei:
        llm.invoke(MSG)
    assert [a.model for a in ei.value.attempts] == ["g1", "g2", "q1"]
    assert "gemini:g1" in str(ei.value)


def test_cooldown_reorders_chain_then_expires():
    rl = RateLimitError("429", provider="gemini", model="g1", status=429)
    llm, gem, grq, clock = build({"g1": [rl, "g1 back"], "g2": ["g2 a", "g2 b"]}, {}, max_retries_per_model=0)
    assert llm.invoke(MSG).model == "g2"  # g1 failed -> g2
    assert llm.ordered_chain()[0].key == "gemini:g2"  # g1 moved to back
    assert llm.invoke(MSG).model == "g2"  # still cooling: g2 first
    clock.t += 61
    assert llm.invoke(MSG).model == "g1"  # cooldown over: g1 primary again
    assert llm.cooldowns() == {}


def test_all_models_cooling_still_tries_them():
    rl = RateLimitError("429", provider="gemini", model="g1", status=429)
    llm, gem, grq, _ = build({"g1": [rl, "g1 ok"], "g2": [rl], "q1": [rl]}, {}, max_retries_per_model=0)
    with pytest.raises(AllModelsFailedError):
        llm.invoke(MSG)
    assert llm.invoke(MSG).text == "g1 ok"  # cooling models are deferred, not dropped


def test_async_path_falls_back():
    llm, gem, grq, _ = build({}, {"q1": ["async groq"]}, max_retries_per_model=0)
    r = asyncio.run(llm.ainvoke(MSG))
    assert r.provider == "groq" and r.text == "async groq"
    assert [a.model for a in r.attempts] == ["g1", "g2", "q1"]


# ------------------------------------------------------------- classification
@pytest.mark.parametrize(
    "status,msg,kind",
    [
        (429, "Resource has been exhausted", "rate_limit"),
        (401, "Invalid API key", "auth"),
        (403, "API key not valid. Please pass a valid API key.", "auth"),
        (403, "Quota exceeded for quota metric", "rate_limit"),
        (404, "models/foo is not found for API version v1beta", "model_unavailable"),
        (400, "The model `llama3-8b` has been decommissioned", "model_unavailable"),
        (400, "Invalid JSON payload received. Unknown name 'foo'", "bad_request"),
        (503, "The model is overloaded. Please try again later.", "server_error"),
        (500, "Internal error", "server_error"),
    ],
)
def test_classify_http(status, msg, kind):
    assert classify_http(status, msg, provider="p", model="m").kind == kind


# ----------------------------------------------------------------- settings
def test_chain_order_primary_then_cross_provider():
    s = LLMSettings(
        provider="gemini",
        gemini_api_key="x",
        gemini_model="a",
        gemini_fallback_models=["b", "a"],
        groq_api_key="y",
        groq_model="q",
        groq_fallback_models=["q2"],
    )
    assert [r.key for r in s.chain()] == ["gemini:a", "gemini:b", "groq:q", "groq:q2"]
    s.provider = "groq"
    assert s.chain()[0].key == "groq:q"
    s.groq_api_key = None
    s.provider = "gemini"
    assert [r.provider for r in s.chain()] == ["gemini", "gemini"]
