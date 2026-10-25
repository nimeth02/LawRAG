"""Gemini provider built on the official google-genai SDK."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
from google import genai
from google.genai import errors as gerr
from google.genai import types as gt

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

_BLOCKING_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def _split_base_url(url: str | None) -> tuple[str | None, str | None]:
    """'https://host/v1beta' -> ('https://host', 'v1beta'). The SDK wants them separate."""
    if not url:
        return None, None
    url = url.rstrip("/")
    m = re.search(r"/(v1(?:alpha|beta)?)$", url)
    if m:
        return url[: m.start()] or None, m.group(1)
    return url, None


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, base_url: str | None = None, timeout_s: float = 60.0) -> None:
        base, version = _split_base_url(base_url)
        http = gt.HttpOptions(
            base_url=base,
            api_version=version,
            timeout=int(timeout_s * 1000),
            # The fallback chain owns retries; stop the SDK from retrying underneath us.
            retry_options=gt.HttpRetryOptions(attempts=1),
        )
        self._client = genai.Client(api_key=api_key, http_options=http)

    # ------------------------------------------------------------------ request
    def _build(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        temperature: float | None,
        max_output_tokens: int | None,
        json_schema: dict[str, Any] | None,
        timeout_s: float | None,
    ) -> tuple[list[gt.Content], gt.GenerateContentConfig]:
        system_parts: list[str] = []
        contents: list[gt.Content] = []
        call_names: dict[str, str] = {}  # tool_call_id -> function name

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                contents.append(gt.Content(role="user", parts=[gt.Part.from_text(text=m.content)]))
            elif m.role == "assistant":
                parts: list[gt.Part] = []
                if m.content:
                    parts.append(gt.Part.from_text(text=m.content))
                for tc in m.tool_calls:
                    call_names[tc.id] = tc.name
                    part = gt.Part.from_function_call(name=tc.name, args=tc.arguments)
                    # Gemini 3 rejects replayed function calls without the signature it issued.
                    sig = tc.provider_data.get("thought_signature")
                    if sig:
                        part.thought_signature = sig
                    parts.append(part)
                if parts:
                    contents.append(gt.Content(role="model", parts=parts))
            elif m.role == "tool":
                name = m.name or call_names.get(m.tool_call_id or "", "tool")
                try:
                    payload = json.loads(m.content)
                    if not isinstance(payload, dict):
                        payload = {"result": payload}
                except (json.JSONDecodeError, TypeError):
                    payload = {"result": m.content}
                part = gt.Part.from_function_response(name=name, response=payload)
                # Gemini wants all results for parallel calls in ONE turn: merge consecutive tool msgs.
                last = contents[-1] if contents else None
                if last is not None and last.role == "user" and last.parts and last.parts[0].function_response:
                    last.parts.append(part)
                else:
                    contents.append(gt.Content(role="user", parts=[part]))

        # The chain drives tool execution itself; never let the SDK auto-run functions.
        cfg: dict[str, Any] = {"automatic_function_calling": gt.AutomaticFunctionCallingConfig(disable=True)}
        if system_parts:
            cfg["system_instruction"] = "\n\n".join(system_parts)
        if temperature is not None:
            cfg["temperature"] = temperature
        if max_output_tokens is not None:
            cfg["max_output_tokens"] = max_output_tokens
        if tools:
            cfg["tools"] = [
                gt.Tool(
                    function_declarations=[
                        gt.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.parameters
                        )
                        for t in tools
                    ]
                )
            ]
        if json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = json_schema
        if timeout_s is not None:
            cfg["http_options"] = gt.HttpOptions(timeout=int(timeout_s * 1000))
        return contents, gt.GenerateContentConfig(**cfg)

    # ----------------------------------------------------------------- response
    @staticmethod
    def _tool_call(part: gt.Part) -> ToolCall:
        fc = part.function_call
        sig = getattr(part, "thought_signature", None)
        return ToolCall(
            id=fc.id or f"call_{uuid.uuid4().hex[:12]}",
            name=fc.name or "",
            arguments=dict(fc.args or {}),
            provider_data={"thought_signature": sig} if sig else {},
        )

    def _assemble(
        self,
        text: str,
        tool_calls: list[ToolCall],
        finish: str | None,
        um: gt.GenerateContentResponseUsageMetadata | None,
        model: str,
        raw: Any,
    ) -> ProviderResult:
        if not text and not tool_calls:
            if finish in _BLOCKING_FINISH:
                raise ContentBlockedError(f"completion blocked: {finish}", provider=self.name, model=model)
            raise EmptyResponseError(f"empty completion (finish_reason={finish})", provider=self.name, model=model)
        usage = Usage(
            input_tokens=(um.prompt_token_count or 0) if um else 0,
            output_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0,
            total_tokens=(um.total_token_count or 0) if um else 0,
        )
        return ProviderResult(text=text, tool_calls=tool_calls, finish_reason=finish, usage=usage, raw=raw)

    def _check_blocked(self, resp: gt.GenerateContentResponse, model: str) -> None:
        fb = resp.prompt_feedback
        if fb and fb.block_reason:
            raise ContentBlockedError(f"prompt blocked: {fb.block_reason.name}", provider=self.name, model=model)

    def _parse(self, resp: gt.GenerateContentResponse, model: str) -> ProviderResult:
        cands = resp.candidates or []
        if not cands:
            self._check_blocked(resp, model)
            raise ContentBlockedError("prompt blocked: no candidates", provider=self.name, model=model)

        cand = cands[0]
        finish = cand.finish_reason.name if cand.finish_reason else None
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in (cand.content.parts if cand.content and cand.content.parts else []):
            if getattr(part, "thought", False):
                continue
            if part.function_call:
                tool_calls.append(self._tool_call(part))
            elif part.text:
                text_parts.append(part.text)
        return self._assemble("".join(text_parts), tool_calls, finish, resp.usage_metadata, model, resp)

    def _iter_stream(self, chunks: Any, model: str):
        """Shared body of `stream`: yield text deltas, then the assembled result."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish: str | None = None
        um = None
        last = None
        for chunk in chunks:
            last = chunk
            if chunk.usage_metadata:
                um = chunk.usage_metadata
            cands = chunk.candidates or []
            if not cands:
                self._check_blocked(chunk, model)
                continue
            cand = cands[0]
            if cand.finish_reason:
                finish = cand.finish_reason.name
            for part in (cand.content.parts if cand.content and cand.content.parts else []):
                if getattr(part, "thought", False):
                    continue
                if part.function_call:
                    tool_calls.append(self._tool_call(part))
                elif part.text:
                    text_parts.append(part.text)
                    yield StreamChunk(text=part.text)
        if last is None:
            raise EmptyResponseError("empty stream", provider=self.name, model=model)
        yield StreamChunk(result=self._assemble("".join(text_parts), tool_calls, finish, um, model, last))

    # ------------------------------------------------------------------- errors
    def _translate(self, exc: Exception, model: str) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, gerr.APIError):
            return classify_http(exc.code, exc.message or str(exc), provider=self.name, model=model)
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
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
        contents, config = self._build(messages, tools, temperature, max_output_tokens, json_schema, timeout_s)
        try:
            resp = self._client.models.generate_content(model=model, contents=contents, config=config)
            return self._parse(resp, model)
        except Exception as exc:  # noqa: BLE001 - normalised below
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
        contents, config = self._build(messages, tools, temperature, max_output_tokens, json_schema, timeout_s)
        try:
            resp = await self._client.aio.models.generate_content(model=model, contents=contents, config=config)
            return self._parse(resp, model)
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
        contents, config = self._build(messages, tools, temperature, max_output_tokens, json_schema, timeout_s)
        try:
            chunks = self._client.models.generate_content_stream(model=model, contents=contents, config=config)
            yield from self._iter_stream(chunks, model)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, model) from exc
