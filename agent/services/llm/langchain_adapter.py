"""Expose `FallbackLLM` as a LangChain `BaseChatModel`.

LangChain agents (`langchain.agents.create_agent`) need a chat model that supports
`bind_tools`. This adapter keeps our provider chain, retries and cooldowns in charge of
every call while LangGraph runs the agent loop on top.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any, Sequence

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from .fallback import FallbackLLM
from .types import LLMResponse, Message, ToolCall, ToolSpec

PROVIDER_DATA_KEY = "tool_call_provider_data"


def _text(msg: BaseMessage) -> str:
    """Plain text of a message whose content may be a string or a list of content blocks."""
    text = msg.text
    return text if isinstance(text, str) else str(text or "")


def to_internal(messages: Sequence[BaseMessage]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append(Message.system(_text(m)))
        elif isinstance(m, HumanMessage):
            out.append(Message.user(_text(m)))
        elif isinstance(m, AIMessage):
            provider_data = m.additional_kwargs.get(PROVIDER_DATA_KEY) or {}
            calls = [
                ToolCall(
                    id=tc.get("id") or uuid.uuid4().hex,
                    name=tc["name"],
                    arguments=dict(tc.get("args") or {}),
                    provider_data=dict(provider_data.get(tc.get("id") or "", {})),
                )
                for tc in m.tool_calls
            ]
            out.append(Message.assistant(_text(m), calls))
        elif isinstance(m, ToolMessage):
            out.append(Message.tool(m.tool_call_id, m.name or "tool", _text(m)))
        else:  # ChatMessage / FunctionMessage: best effort by role name
            role = getattr(m, "role", None) or m.type
            out.append(Message(role="user" if role not in ("system", "assistant", "tool") else role, content=_text(m)))
    return out


def to_ai_message(resp: LLMResponse) -> AIMessage:
    calls = [(tc.id or uuid.uuid4().hex, tc) for tc in resp.tool_calls]
    return AIMessage(
        content=resp.text or "",
        tool_calls=[{"id": cid, "name": tc.name, "args": tc.arguments, "type": "tool_call"} for cid, tc in calls],
        # LangChain tool-call dicts have no slot for provider payloads (Gemini thought signatures),
        # so they ride along on the message and are re-attached in `to_internal`.
        additional_kwargs={PROVIDER_DATA_KEY: {cid: tc.provider_data for cid, tc in calls if tc.provider_data}},
        response_metadata={
            "provider": resp.provider,
            "model": resp.model,
            "finish_reason": resp.finish_reason,
            "fell_back": resp.fell_back,
            "attempts": len(resp.attempts),
        },
        usage_metadata=UsageMetadata(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.total_tokens or (resp.usage.input_tokens + resp.usage.output_tokens),
        ),
    )


def _tool_specs(tools: Any) -> list[ToolSpec] | None:
    if not tools:
        return None
    specs: list[ToolSpec] = []
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else convert_to_openai_tool(t)["function"]
        specs.append(
            ToolSpec(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return specs


class FallbackChatModel(BaseChatModel):
    """LangChain chat model backed by `FallbackLLM` (Gemini -> Groq chain with cooldowns)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: FallbackLLM
    temperature: float | None = 0.0
    max_output_tokens: int | None = None

    @property
    def _llm_type(self) -> str:
        return "fallback-llm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"chain": [r.key for r in self.llm.chain], "temperature": self.temperature}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return super().bind(tools=formatted, **kwargs)  # tool_choice is not supported by the provider layer

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resp = self.llm.invoke(
            to_internal(messages),
            tools=_tool_specs(kwargs.get("tools")),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(resp))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Token streaming. LangChain picks this automatically when a streaming callback is
        attached (e.g. `graph.stream(stream_mode="messages")`); plain `invoke` uses `_generate`."""
        events = self.llm.stream(
            to_internal(messages),
            tools=_tool_specs(kwargs.get("tools")),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        for ev in events:
            if ev.response is None:
                if ev.text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=ev.text))
                continue
            # Final chunk: tool calls, usage and metadata. Text already went out as deltas, so
            # content is empty here and LangChain's chunk merging reassembles the full message.
            final = to_ai_message(ev.response)
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i, "type": "tool_call_chunk"}
                        for i, tc in enumerate(final.tool_calls)
                    ],
                    additional_kwargs=final.additional_kwargs,
                    response_metadata=final.response_metadata,
                    usage_metadata=final.usage_metadata,
                )
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resp = await self.llm.ainvoke(
            to_internal(messages),
            tools=_tool_specs(kwargs.get("tools")),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(resp))])


__all__ = ["FallbackChatModel", "to_ai_message", "to_internal"]
