"""Text -> vector. One protocol, one Gemini implementation.

Swap in a local model later (e.g. sentence-transformers) by implementing `Embedder`.
"""
from __future__ import annotations

from typing import Protocol

from google import genai
from google.genai import errors as gerr
from google.genai import types as gt

from agent.services.llm.errors import LLMError, classify_http


class EmbeddingError(LLMError):
    kind = "embedding"


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Gemini embedding model with asymmetric task types (document vs query)."""

    BATCH = 100  # API limit per call

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = 768) -> None:
        self.model = model
        self.dim = dim
        self._client = genai.Client(api_key=api_key)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            try:
                resp = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=gt.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
                )
            except gerr.APIError as exc:
                raise classify_http(exc.code, exc.message or str(exc), provider="gemini", model=self.model) from exc
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"{type(exc).__name__}: {exc}", provider="gemini", model=self.model) from exc
            out.extend(list(e.values) for e in resp.embeddings or [])
        if len(out) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(out)}", provider="gemini", model=self.model)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
