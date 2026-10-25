"""Groq provider via its OpenAI-compatible chat completions endpoint (plain httpx)."""
from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from ..errors import (
    ContentBlockedError,
    EmptyResponseError,
    LLMError,
    ProviderServerError,
    ProviderTimeoutError,
    classify_http,
)
from ..types import Message, ProviderResult, StreamChunk, ToolCall, ToolSpec, Usage
from .base import Provider


class GroqProvider(Provider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)
        self._aclient = httpx.AsyncClient(timeout=timeout_s)

    # ------------------------------------------------------------------ request
    @staticmethod
    def _payload(
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        temperature: float | None,
        max_output_tokens: int | None,
        json_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                msgs.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                msgs.append(entry)
            elif m.role == "tool":
                msgs.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})

        body: dict[str, Any] = {"model": model, "messages": msgs}
        if temperature is not None:
            body["temperature"] = temperature
        if max_output_tokens is not None:
            body["max_completion_tokens"] = max_output_tokens
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
            body["tool_choice"] = "auto"
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        return body

    # ----------------------------------------------------------------- response
    @staticmethod
    def _tool_call(tc_id: str | None, name: str, raw_args: str) -> ToolCall:
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        return ToolCall(id=tc_id or f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args)

    def _assemble(
        self,
        text: str,
        tool_calls: list[ToolCall],
        finish: str | None,
        u: dict[str, Any],
        model: str,
        raw: Any,
    ) -> ProviderResult:
        if not text and not tool_calls:
            if finish == "content_filter":
                raise ContentBlockedError("completion blocked: content_filter", provider=self.name, model=model)
            raise EmptyResponseError(f"empty completion (finish_reason={finish})", provider=self.name, model=model)
        usage = Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        return ProviderResult(text=text, tool_calls=tool_calls, finish_reason=finish, usage=usage, raw=raw)

    def _parse(self, data: dict[str, Any], model: str) -> ProviderResult:
        choices = data.get("choices") or []
        if not choices:
            raise EmptyResponseError("no choices in response", provider=self.name, model=model)
        choice = choices[0]
        msg = choice.get("message") or {}
        tool_calls = [
            self._tool_call(tc.get("id"), (tc.get("function") or {}).get("name", ""), (tc.get("function") or {}).get("arguments") or "{}")
            for tc in msg.get("tool_calls") or []
        ]
        return self._assemble(msg.get("content") or "", tool_calls, choice.get("finish_reason"), data.get("usage") or {}, model, data)

    def _iter_sse(self, lines: Any, model: str):
        """Parse an OpenAI-style chat completions SSE stream into StreamChunks."""
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}  # tool_call index -> accumulating {id, name, arguments}
        finish: str | None = None
        usage: dict[str, Any] = {}
        last: Any = None
        for line in lines:
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            last = evt
            if evt.get("error"):
                err = evt["error"]
                raise ProviderServerError(str(err.get("message") if isinstance(err, dict) else err), provider=self.name, model=model)
            # Groq reports usage on the last chunk under x_groq; OpenAI-style servers under usage.
            usage = evt.get("usage") or (evt.get("x_groq") or {}).get("usage") or usage
            for choice in evt.get("choices") or []:
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                    yield StreamChunk(text=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", len(calls)), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
        if last is None:
            raise EmptyResponseError("empty stream", provider=self.name, model=model)
        tool_calls = [self._tool_call(c["id"], c["name"], c["arguments"]) for _, c in sorted(calls.items())]
        yield StreamChunk(result=self._assemble("".join(text_parts), tool_calls, finish, usage, model, last))

    # ------------------------------------------------------------------- errors
    def _raise_for_status(self, resp: httpx.Response, model: str) -> None:
        if resp.status_code < 400:
            return
        try:
            err = resp.json().get("error") or {}
            message = err.get("message") or resp.text
            code = err.get("code")
            if code and code not in message:
                message = f"{message} ({code})"
        except (ValueError, AttributeError):
            message = resp.text
        retry_after = resp.headers.get("retry-after")
        try:
            retry_after_s = float(retry_after) if retry_after else None
        except ValueError:
            retry_after_s = None
        raise classify_http(resp.status_code, message, provider=self.name, model=model, retry_after_s=retry_after_s)

    def _translate(self, exc: Exception, model: str) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(str(exc) or "timeout", provider=self.name, model=model)
        if isinstance(exc, httpx.HTTPError):
            return ProviderServerError(f"network error: {exc}", provider=self.name, model=model)
        return LLMError(f"{type(exc).__name__}: {exc}", provider=self.name, model=model)

    # ---------------------------------------------------------------- interface
    def complete(
        self,
        model,
        messages,
        *,
        tools=None,
        temperature=None,
        max_output_tokens=None,
        json_schema=None,
        timeout_s=None,
    ):
        body = self._payload(model, messages, tools, temperature, max_output_tokens, json_schema)
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=body,
                timeout=timeout_s or self._timeout_s,
            )
            self._raise_for_status(resp, model)
            return self._parse(resp.json(), model)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, model) from exc

    async def acomplete(
        self,
        model,
        messages,
        *,
        tools=None,
        temperature=None,
        max_output_tokens=None,
        json_schema=None,
        timeout_s=None,
    ):
        body = self._payload(model, messages, tools, temperature, max_output_tokens, json_schema)
        try:
            resp = await self._aclient.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=body,
                timeout=timeout_s or self._timeout_s,
            )
            self._raise_for_status(resp, model)
            return self._parse(resp.json(), model)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, model) from exc

    def stream(
        self,
        model,
        messages,
        *,
        tools=None,
        temperature=None,
        max_output_tokens=None,
        json_schema=None,
        timeout_s=None,
    ):
        body = self._payload(model, messages, tools, temperature, max_output_tokens, json_schema)
        body["stream"] = True
        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=body,
                timeout=timeout_s or self._timeout_s,
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._raise_for_status(resp, model)
                yield from self._iter_sse(resp.iter_lines(), model)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, model) from exc
