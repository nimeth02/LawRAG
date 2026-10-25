"""Settings for the LLM layer, read from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

SUPPORTED_PROVIDERS = ("gemini", "groq")


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def __str__(self) -> str:
        return self.key


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass
class LLMSettings:
    provider: str = "gemini"

    gemini_api_key: str | None = None
    gemini_base_url: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_models: list[str] = field(default_factory=list)

    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_models: list[str] = field(default_factory=list)

    timeout_s: float = 60.0
    max_retries_per_model: int = 1  # extra attempts on the SAME model for transient errors
    backoff_base_s: float = 1.0
    backoff_max_s: float = 8.0

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "LLMSettings":
        load_dotenv(dotenv_path, override=False)
        provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER={provider!r}; expected one of {SUPPORTED_PROVIDERS}")
        return cls(
            provider=provider,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_base_url=os.getenv("GEMINI_BASE_URL") or None,
            gemini_model=os.getenv("GEMINI_MODEL") or cls.gemini_model,
            gemini_fallback_models=_csv(os.getenv("GEMINI_FALLBACK_MODELS")),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_base_url=os.getenv("GROQ_BASE_URL") or cls.groq_base_url,
            groq_model=os.getenv("GROQ_MODEL") or cls.groq_model,
            groq_fallback_models=_csv(os.getenv("GROQ_FALLBACK_MODELS")),
            timeout_s=_float("LLM_TIMEOUT_S", cls.timeout_s),
            max_retries_per_model=_int("LLM_MAX_RETRIES_PER_MODEL", cls.max_retries_per_model),
            backoff_base_s=_float("LLM_BACKOFF_BASE_S", cls.backoff_base_s),
            backoff_max_s=_float("LLM_BACKOFF_MAX_S", cls.backoff_max_s),
        )

    def models_for(self, provider: str) -> list[str]:
        if provider == "gemini":
            return [self.gemini_model, *self.gemini_fallback_models]
        if provider == "groq":
            return [self.groq_model, *self.groq_fallback_models]
        raise ValueError(provider)

    def has_key(self, provider: str) -> bool:
        return bool({"gemini": self.gemini_api_key, "groq": self.groq_api_key}[provider])

    def chain(self) -> list[ModelRef]:
        """Ordered list of models to try: primary provider first, then the others."""
        order = [self.provider, *[p for p in SUPPORTED_PROVIDERS if p != self.provider]]
        refs: list[ModelRef] = []
        seen: set[str] = set()
        for p in order:
            if not self.has_key(p):
                continue
            for m in self.models_for(p):
                ref = ModelRef(p, m)
                if ref.key not in seen:
                    seen.add(ref.key)
                    refs.append(ref)
        if not refs:
            raise ValueError("no LLM provider has an API key configured (GEMINI_API_KEY / GROQ_API_KEY)")
        return refs
