"""LLM layer: provider-agnostic chat with model + provider fallback."""
from .config import LLMSettings, ModelRef
from .errors import (
    AllModelsFailedError,
    AuthenticationError,
    BadRequestError,
    ContentBlockedError,
    EmptyResponseError,
    LLMError,
    ModelUnavailableError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitError,
)
from .fallback import FallbackLLM
from .providers import GeminiProvider, GroqProvider, Provider
from .types import Attempt, LLMResponse, Message, StreamChunk, StreamEvent, ToolCall, ToolSpec, Usage

__all__ = [
    "AllModelsFailedError",
    "Attempt",
    "AuthenticationError",
    "BadRequestError",
    "ContentBlockedError",
    "EmptyResponseError",
    "FallbackLLM",
    "GeminiProvider",
    "GroqProvider",
    "LLMError",
    "LLMResponse",
    "LLMSettings",
    "Message",
    "ModelRef",
    "ModelUnavailableError",
    "Provider",
    "ProviderServerError",
    "ProviderTimeoutError",
    "RateLimitError",
    "StreamChunk",
    "StreamEvent",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
