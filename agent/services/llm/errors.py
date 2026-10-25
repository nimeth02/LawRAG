"""Error taxonomy for the LLM layer.

Every provider error is normalised into one of these classes. The class attributes
tell the fallback chain what to do:

  retry_same_model  - transient: back off and retry the same model first
  try_next_model    - worth moving to the next model in the chain
  skip_provider     - the whole provider is unusable (bad key): skip all its models
  cooldown_s        - how long to avoid this model after the failure
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Attempt


class LLMError(Exception):
    kind: str = "unknown"
    retry_same_model: bool = False
    try_next_model: bool = True
    skip_provider: bool = False
    cooldown_s: float = 30.0

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status = status
        self.retry_after_s = retry_after_s

    def __str__(self) -> str:
        where = f" {self.provider}:{self.model}" if self.provider else ""
        code = f" [{self.status}]" if self.status else ""
        return f"{self.kind}{code}{where}: {self.args[0]}"


class RateLimitError(LLMError):
    kind = "rate_limit"
    retry_same_model = True
    cooldown_s = 60.0


class ProviderServerError(LLMError):
    kind = "server_error"
    retry_same_model = True
    cooldown_s = 30.0


class ProviderTimeoutError(LLMError):
    kind = "timeout"
    retry_same_model = True
    cooldown_s = 15.0


class ModelUnavailableError(LLMError):
    """Model id unknown, decommissioned, or not enabled for this key."""

    kind = "model_unavailable"
    cooldown_s = 600.0


class AuthenticationError(LLMError):
    kind = "auth"
    skip_provider = True
    cooldown_s = 3600.0


class BadRequestError(LLMError):
    """Our request was malformed. Another model would fail the same way, so no fallback."""

    kind = "bad_request"
    try_next_model = False
    cooldown_s = 0.0


class ContentBlockedError(LLMError):
    """Safety filter blocked the prompt or the completion."""

    kind = "content_blocked"
    cooldown_s = 0.0


class EmptyResponseError(LLMError):
    kind = "empty_response"
    retry_same_model = True
    cooldown_s = 0.0


class AllModelsFailedError(LLMError):
    kind = "all_models_failed"
    try_next_model = False

    def __init__(self, attempts: list["Attempt"]) -> None:
        self.attempts = attempts
        lines = [f"  - {a.provider}:{a.model} -> {a.error_kind}: {a.error}" for a in attempts]
        super().__init__("every model in the chain failed:\n" + "\n".join(lines))

    def __str__(self) -> str:
        return self.args[0]


_MODEL_GONE = re.compile(
    r"(model|models/)[^\n]*?(not found|does not exist|decommissioned|deprecated|"
    r"not supported|is not available|unsupported|no longer)",
    re.IGNORECASE,
)
_KEY_BAD = re.compile(r"(api[ _-]?key|invalid.*credential|unauthenticated|permission[_ ]denied)", re.IGNORECASE)
_QUOTA = re.compile(r"(quota|resource[_ ]exhausted|rate ?limit|too many requests)", re.IGNORECASE)


def classify_http(
    status: int | None,
    message: str,
    *,
    provider: str,
    model: str,
    retry_after_s: float | None = None,
) -> LLMError:
    """Map an HTTP status + provider message onto the taxonomy above."""
    kw = {"provider": provider, "model": model, "status": status, "retry_after_s": retry_after_s}
    msg = message or f"HTTP {status}"

    if status == 429:
        return RateLimitError(msg, **kw)
    if status == 401:
        return AuthenticationError(msg, **kw)
    if status == 403:
        if _QUOTA.search(msg):
            return RateLimitError(msg, **kw)
        if _KEY_BAD.search(msg):
            return AuthenticationError(msg, **kw)
        return ModelUnavailableError(msg, **kw)
    if status == 404:
        return ModelUnavailableError(msg, **kw)
    if status == 400:
        if _MODEL_GONE.search(msg):
            return ModelUnavailableError(msg, **kw)
        if _KEY_BAD.search(msg):
            return AuthenticationError(msg, **kw)
        if _QUOTA.search(msg):
            return RateLimitError(msg, **kw)
        return BadRequestError(msg, **kw)
    if status == 408:
        return ProviderTimeoutError(msg, **kw)
    if status is not None and status >= 500:
        return ProviderServerError(msg, **kw)
    if _QUOTA.search(msg):
        return RateLimitError(msg, **kw)
    if _MODEL_GONE.search(msg):
        return ModelUnavailableError(msg, **kw)
    return LLMError(msg, **kw)
