"""Provider-agnostic message, tool and response types for the LLM layer."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque, provider-specific data that must be echoed back when this call is replayed in a
    # later turn (e.g. Gemini 3 `thought_signature`). Other providers ignore it.
    provider_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant messages only
    tool_call_id: str | None = None  # tool messages only
    name: str | None = None  # tool messages only: the function that produced the result

    @staticmethod
    def system(content: str) -> "Message":
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> "Message":
        return Message(role="user", content=content)

    @staticmethod
    def assistant(content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return Message(role="assistant", content=content, tool_calls=list(tool_calls or []))

    @staticmethod
    def tool(tool_call_id: str, name: str, result: Any) -> "Message":
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolSpec:
    """A function the model may call. `parameters` is a JSON schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ProviderResult:
    """What a single provider call returns on success."""

    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage
    raw: Any = None


@dataclass
class StreamChunk:
    """One item of a provider stream: a text delta, and on the last item the full result."""

    text: str = ""
    result: ProviderResult | None = None


@dataclass
class Attempt:
    """One call to one model, successful or not. Returned for observability."""

    provider: str
    model: str
    ok: bool
    latency_ms: int
    error_kind: str | None = None
    error: str | None = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    provider: str
    model: str
    finish_reason: str | None
    usage: Usage
    attempts: list[Attempt]
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def fell_back(self) -> bool:
        """True when the model that answered was not the first one tried."""
        return len(self.attempts) > 1

    def as_message(self) -> Message:
        """Turn the response into an assistant message to append to the conversation."""
        return Message.assistant(self.text, self.tool_calls)


@dataclass
class StreamEvent:
    """One item of `FallbackLLM.stream`: a text delta, and on the last item the full response."""

    text: str = ""
    response: LLMResponse | None = None
