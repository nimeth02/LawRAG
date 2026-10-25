from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from ..types import Message, ProviderResult, StreamChunk, ToolSpec


class Provider(ABC):
    """One LLM backend. Implementations must raise only `agent.services.llm.errors.LLMError`
    subclasses so the fallback chain can decide what to do."""

    name: str

    @abstractmethod
    def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ProviderResult: ...

    @abstractmethod
    async def acomplete(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ProviderResult: ...

    def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[StreamChunk]:
        """Yield text deltas as they arrive, then one final chunk carrying the full result.

        Default: no token streaming; the whole completion arrives as a single chunk. Providers
        that support streaming override this.
        """
        result = self.complete(
            model,
            messages,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
            timeout_s=timeout_s,
        )
        yield StreamChunk(text=result.text, result=result)
