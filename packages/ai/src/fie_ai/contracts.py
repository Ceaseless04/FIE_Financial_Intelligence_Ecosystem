"""Provider-neutral request/response contracts.

Application and agent code speaks only this vocabulary. Nothing above the
provider layer imports ``anthropic``, ``httpx``, or any vendor type — that is
what makes Claude swappable for Ollama or vLLM per environment.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from fie_observability.metrics import TokenUsage
from fie_schemas.base import FrozenModel


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(FrozenModel):
    """One conversational turn."""

    role: Role
    content: str = Field(min_length=1)


class ThinkingMode(StrEnum):
    """How much internal reasoning the model should do.

    ``ADAPTIVE`` lets the model decide per request and is the platform default.
    Providers that cannot reason map this to their nearest equivalent rather
    than failing.
    """

    ADAPTIVE = "adaptive"
    DISABLED = "disabled"


class Effort(StrEnum):
    """Thoroughness/cost dial, passed through to providers that support it."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    #: The model declined on safety grounds. Retrying the same request produces
    #: the same refusal — route to a fallback provider instead.
    REFUSAL = "refusal"


class CompletionRequest(FrozenModel):
    """A provider-neutral completion request."""

    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    #: ``None`` uses the provider's configured default model.
    model: str | None = None
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    thinking: ThinkingMode = ThinkingMode.ADAPTIVE
    effort: Effort | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    #: JSON Schema for structured output. Providers enforce natively where they
    #: can; otherwise the response is validated against it after the fact.
    response_schema: dict[str, Any] | None = None
    #: Advisory only. Providers that reject sampling parameters drop this
    #: rather than erroring — see ClaudeProvider.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Request prompt caching of the system prompt where the provider supports it.
    cache_system_prompt: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_conversation(self) -> CompletionRequest:
        if self.messages[0].role is not Role.USER:
            raise ValueError("conversation must start with a user message")
        if self.response_schema is not None and not isinstance(self.response_schema, dict):
            raise ValueError("response_schema must be a JSON Schema object")
        return self

    @property
    def requires_streaming(self) -> bool:
        """Large outputs must stream or the HTTP request times out."""
        return self.max_tokens > 16_000


class CompletionResponse(FrozenModel):
    """A provider-neutral completion response."""

    text: str
    model: str
    provider: str
    stop_reason: StopReason = StopReason.END_TURN
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0)
    #: Populated only when ``stop_reason`` is ``REFUSAL``.
    refusal_category: str | None = None
    #: Which provider actually served the request, if a fallback was used.
    served_by: str | None = None

    model_config = FrozenModel.model_config | {"protected_namespaces": ()}

    @property
    def is_refusal(self) -> bool:
        return self.stop_reason is StopReason.REFUSAL

    @property
    def is_truncated(self) -> bool:
        """Output hit the token ceiling — the text is incomplete."""
        return self.stop_reason is StopReason.MAX_TOKENS


class ProviderCapabilities(FrozenModel):
    """What a provider can do, so callers degrade instead of erroring."""

    supports_streaming: bool = True
    supports_structured_output: bool = False
    supports_thinking: bool = False
    supports_effort: bool = False
    #: False for models that reject temperature/top_p/top_k outright.
    supports_sampling_params: bool = True
    supports_token_counting: bool = False
    supports_prompt_caching: bool = False
    max_context_tokens: int = 128_000


__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "Effort",
    "Message",
    "ProviderCapabilities",
    "Role",
    "StopReason",
    "ThinkingMode",
    "TokenUsage",
]
