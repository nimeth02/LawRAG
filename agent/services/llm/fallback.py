"""Model fallback chain.

    settings = LLMSettings.from_env()
    llm = FallbackLLM.from_settings(settings)
    resp = llm.invoke([Message.user("Hi")])

Behaviour, per call:
  1. Walk the chain (primary provider's models, then the other provider's models).
  2. Models in cooldown (recently failed) are moved to the back, never dropped.
  3. For transient errors (429 / 5xx / timeout / empty) retry the SAME model up to
     `max_retries_per_model` times with exponential backoff, honouring Retry-After.
  4. For model-level errors (unknown model, blocked content) move on immediately.
  5. For provider-level errors (bad API key) skip every model of that provider.
  6. For our own mistakes (400 bad request) raise at once: no model will fix that.
  7. If everything fails raise AllModelsFailedError with the full attempt log.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable, Iterator
from typing import Any

from .config import LLMSettings, ModelRef
from .errors import AllModelsFailedError, EmptyResponseError, LLMError
from .providers import GeminiProvider, GroqProvider, Provider
from .types import Attempt, LLMResponse, Message, ProviderResult, StreamEvent, ToolSpec

log = logging.getLogger("agent.services.llm")


class FallbackLLM:
    def __init__(
        self,
        chain: list[ModelRef],
        providers: dict[str, Provider],
        *,
        max_retries_per_model: int = 1,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 8.0,
        timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not chain:
            raise ValueError("fallback chain is empty")
        missing = {r.provider for r in chain} - set(providers)
        if missing:
            raise ValueError(f"chain references providers with no implementation: {sorted(missing)}")
        self.chain = list(chain)
        self.providers = providers
        self.max_retries_per_model = max(0, max_retries_per_model)
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.timeout_s = timeout_s
        self._clock = clock
        self._sleep = sleep
        self._cooldown_until: dict[str, float] = {}

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_settings(cls, settings: LLMSettings) -> "FallbackLLM":
        providers: dict[str, Provider] = {}
        if settings.gemini_api_key:
            providers["gemini"] = GeminiProvider(settings.gemini_api_key, settings.gemini_base_url, settings.timeout_s)
        if settings.groq_api_key:
            providers["groq"] = GroqProvider(settings.groq_api_key, settings.groq_base_url, settings.timeout_s)
        return cls(
            settings.chain(),
            providers,
            max_retries_per_model=settings.max_retries_per_model,
            backoff_base_s=settings.backoff_base_s,
            backoff_max_s=settings.backoff_max_s,
            timeout_s=settings.timeout_s,
        )

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "FallbackLLM":
        return cls.from_settings(LLMSettings.from_env(dotenv_path))

    # ------------------------------------------------------------- cooldown mgmt
    def _in_cooldown(self, ref: ModelRef) -> bool:
        return self._cooldown_until.get(ref.key, 0.0) > self._clock()

    def _set_cooldown(self, ref: ModelRef, seconds: float) -> None:
        if seconds > 0:
            until = self._clock() + seconds
            self._cooldown_until[ref.key] = max(self._cooldown_until.get(ref.key, 0.0), until)

    def _clear_cooldown(self, ref: ModelRef) -> None:
        self._cooldown_until.pop(ref.key, None)

    def ordered_chain(self) -> list[ModelRef]:
        """Healthy models first (chain order), then cooling-down ones (chain order)."""
        healthy = [r for r in self.chain if not self._in_cooldown(r)]
        cooling = [r for r in self.chain if self._in_cooldown(r)]
        return healthy + cooling

    def cooldowns(self) -> dict[str, float]:
        """Remaining cooldown seconds per model key (only those still cooling)."""
        now = self._clock()
        return {k: round(t - now, 1) for k, t in self._cooldown_until.items() if t > now}

    def _backoff(self, attempt_index: int, err: LLMError) -> float:
        if err.retry_after_s:
            return min(err.retry_after_s, self.backoff_max_s * 4)
        delay = self.backoff_base_s * (2**attempt_index)
        return min(delay, self.backoff_max_s) * (0.8 + 0.4 * random.random())

    # -------------------------------------------------------------------- core
    def _on_failure(self, ref: ModelRef, err: LLMError, attempts: list[Attempt], latency_ms: int) -> None:
        attempts.append(
            Attempt(
                provider=ref.provider,
                model=ref.model,
                ok=False,
                latency_ms=latency_ms,
                error_kind=err.kind,
                error=str(err.args[0]),
            )
        )
        log.warning("llm attempt failed %s -> %s: %s", ref.key, err.kind, err.args[0])

    def _handle_error(self, ref: ModelRef, err: LLMError, attempt_index: int, dead_providers: set[str]) -> str:
        """Decide what to do after a failure: 'retry', 'next', or raise."""
        if err.skip_provider:
            dead_providers.add(ref.provider)
            for r in self.chain:
                if r.provider == ref.provider:
                    self._set_cooldown(r, err.cooldown_s)
            return "next"
        if not err.try_next_model:
            raise err
        if err.retry_same_model and attempt_index < self.max_retries_per_model:
            return "retry"
        self._set_cooldown(ref, err.cooldown_s)
        return "next"

    def _finish(self, ref: ModelRef, result: ProviderResult, attempts: list[Attempt], latency_ms: int) -> LLMResponse:
        attempts.append(Attempt(provider=ref.provider, model=ref.model, ok=True, latency_ms=latency_ms))
        self._clear_cooldown(ref)
        if len(attempts) > 1:
            log.info("llm answered by fallback %s after %d failed attempt(s)", ref.key, len(attempts) - 1)
        return LLMResponse(
            text=result.text,
            tool_calls=result.tool_calls,
            provider=ref.provider,
            model=ref.model,
            finish_reason=result.finish_reason,
            usage=result.usage,
            attempts=attempts,
            raw=result.raw,
        )

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        attempts: list[Attempt] = []
        dead_providers: set[str] = set()
        kwargs = dict(
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
            timeout_s=self.timeout_s,
        )

        for ref in self.ordered_chain():
            if ref.provider in dead_providers:
                continue
            provider = self.providers[ref.provider]
            for i in range(1 + self.max_retries_per_model):
                t0 = self._clock()
                try:
                    result = provider.complete(ref.model, messages, **kwargs)
                except LLMError as err:
                    self._on_failure(ref, err, attempts, int((self._clock() - t0) * 1000))
                    action = self._handle_error(ref, err, i, dead_providers)
                    if action == "retry":
                        self._sleep(self._backoff(i, err))
                        continue
                    break
                else:
                    return self._finish(ref, result, attempts, int((self._clock() - t0) * 1000))

        raise AllModelsFailedError(attempts)

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        attempts: list[Attempt] = []
        dead_providers: set[str] = set()
        kwargs = dict(
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
            timeout_s=self.timeout_s,
        )

        for ref in self.ordered_chain():
            if ref.provider in dead_providers:
                continue
            provider = self.providers[ref.provider]
            for i in range(1 + self.max_retries_per_model):
                t0 = self._clock()
                try:
                    result = await provider.acomplete(ref.model, messages, **kwargs)
                except LLMError as err:
                    self._on_failure(ref, err, attempts, int((self._clock() - t0) * 1000))
                    action = self._handle_error(ref, err, i, dead_providers)
                    if action == "retry":
                        await asyncio.sleep(self._backoff(i, err))
                        continue
                    break
                else:
                    return self._finish(ref, result, attempts, int((self._clock() - t0) * 1000))

        raise AllModelsFailedError(attempts)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Like `invoke`, but yields text deltas as they arrive and a final `StreamEvent.response`.

        Fallback semantics are the same as `invoke` *until the first token is emitted*. After
        that, a failure is raised to the caller: silently restarting on another model would
        duplicate text the caller has already shown.
        """
        attempts: list[Attempt] = []
        dead_providers: set[str] = set()
        kwargs = dict(
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
            timeout_s=self.timeout_s,
        )

        for ref in self.ordered_chain():
            if ref.provider in dead_providers:
                continue
            provider = self.providers[ref.provider]
            for i in range(1 + self.max_retries_per_model):
                t0 = self._clock()
                emitted = False
                try:
                    for chunk in provider.stream(ref.model, messages, **kwargs):
                        if chunk.text:
                            emitted = True
                            yield StreamEvent(text=chunk.text)
                        if chunk.result is not None:
                            yield StreamEvent(response=self._finish(ref, chunk.result, attempts, int((self._clock() - t0) * 1000)))
                            return
                    raise EmptyResponseError("stream ended without a result", provider=ref.provider, model=ref.model)
                except LLMError as err:
                    self._on_failure(ref, err, attempts, int((self._clock() - t0) * 1000))
                    if emitted:
                        raise
                    action = self._handle_error(ref, err, i, dead_providers)
                    if action == "retry":
                        self._sleep(self._backoff(i, err))
                        continue
                    break

        raise AllModelsFailedError(attempts)

    # ------------------------------------------------------------- convenience
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        msgs = [Message.system(system)] if system else []
        msgs.append(Message.user(prompt))
        return self.invoke(msgs, **kwargs).text
