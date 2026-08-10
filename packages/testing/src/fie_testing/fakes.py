"""High-fidelity fakes.

The testing architecture forbids relying exclusively on mocked tests. These
fakes are therefore built to be *behaviourally* faithful rather than
convenient: ``FakeAnthropicClient`` reproduces the SDK's response object shape
(including refusals, cache-token fields, and thinking blocks) so the real
translation logic in ``ClaudeProvider`` is exercised, not bypassed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from fie_ai.base import AIProvider
from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    ProviderCapabilities,
    StopReason,
    TokenUsage,
)
from fie_common.resilience import ResiliencePolicy, RetryPolicy


class FakeAIProvider(AIProvider):
    """Scriptable provider returning queued responses or raising queued errors."""

    name: ClassVar[str] = "fake"

    def __init__(
        self,
        responses: Sequence[CompletionResponse | Exception] | None = None,
        *,
        model: str = "fake-model-1",
        capabilities: ProviderCapabilities | None = None,
        provider_name: str | None = None,
        resilience: ResiliencePolicy | None = None,
    ) -> None:
        # Single attempt by default: a scripted double must consume exactly one
        # queued item per call, otherwise a queued error is silently retried and
        # the next response returned instead. Pass `resilience` explicitly to
        # exercise retry behaviour.
        super().__init__(
            resilience=resilience
            or ResiliencePolicy(
                timeout_seconds=30.0, retry=RetryPolicy(max_attempts=1), breaker=None
            )
        )
        if provider_name is not None:
            # ClassVar assignment on the instance keeps registry lookups working
            # when several fakes coexist in one test.
            self.name = provider_name  # type: ignore[misc]
        self._model = model
        self._capabilities = capabilities or ProviderCapabilities(
            supports_streaming=True,
            supports_structured_output=True,
            supports_thinking=True,
            supports_effort=True,
            supports_sampling_params=False,
            supports_token_counting=True,
            supports_prompt_caching=True,
        )
        self._queue: list[CompletionResponse | Exception] = list(responses or [])
        self.calls: list[CompletionRequest] = []
        self.ping_count = 0
        self.closed = False

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def default_model(self) -> str:
        return self._model

    def enqueue(self, *items: CompletionResponse | Exception) -> FakeAIProvider:
        """Append responses or errors to the script."""
        self._queue.extend(items)
        return self

    def enqueue_text(self, text: str, **kwargs: Any) -> FakeAIProvider:
        """Convenience: queue a plain successful text response."""
        return self.enqueue(self.make_response(text, **kwargs))

    def make_response(
        self,
        text: str,
        *,
        stop_reason: StopReason = StopReason.END_TURN,
        input_tokens: int = 10,
        output_tokens: int = 20,
        refusal_category: str | None = None,
    ) -> CompletionResponse:
        return CompletionResponse(
            text=text,
            model=self._model,
            provider=self.name,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            refusal_category=refusal_category,
        )

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        if not self._queue:
            return self.make_response(f"fake response to: {request.messages[-1].content}")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def _stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        response = await self._complete(request)
        for word in response.text.split(" "):
            yield word + " "

    async def _ping(self) -> None:
        self.ping_count += 1

    async def aclose(self) -> None:
        self.closed = True

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_request(self) -> CompletionRequest | None:
        return self.calls[-1] if self.calls else None


# ---------------------------------------------------------------------------
# Anthropic SDK response doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeThinkingBlock:
    """A thinking block. Its text is empty unless summarized display is set."""

    thinking: str = ""
    type: str = "thinking"


@dataclass
class FakeStopDetails:
    category: str | None = None
    explanation: str | None = None
    type: str = "refusal"


@dataclass
class FakeMessage:
    """Mirrors the shape of an Anthropic ``Message``."""

    content: list[Any] = field(default_factory=list)
    model: str = "claude-opus-5"
    stop_reason: str = "end_turn"
    stop_details: FakeStopDetails | None = None
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeTokenCount:
    input_tokens: int = 0


class _FakeStream:
    """Async context manager mirroring ``client.messages.stream(...)``."""

    def __init__(self, message: FakeMessage) -> None:
        self._message = message

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get_final_message(self) -> FakeMessage:
        return self._message

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def generator() -> AsyncIterator[str]:
            for block in self._message.content:
                if getattr(block, "type", None) == "text":
                    for word in block.text.split(" "):
                        yield word + " "

        return generator()


class FakeMessagesAPI:
    """Stands in for ``client.messages``."""

    def __init__(self, owner: FakeAnthropicClient) -> None:
        self._owner = owner

    async def create(self, **payload: Any) -> FakeMessage:
        self._owner.requests.append(payload)
        return self._owner._next_message()

    def stream(self, **payload: Any) -> _FakeStream:
        self._owner.requests.append(payload)
        return _FakeStream(self._owner._next_message())

    async def count_tokens(self, **payload: Any) -> FakeTokenCount:
        self._owner.token_count_requests.append(payload)
        return FakeTokenCount(input_tokens=self._owner.token_count)


class FakeAnthropicClient:
    """Drop-in double for ``anthropic.AsyncAnthropic``.

    Records every request payload so tests can assert on what the provider
    actually sent — that sampling parameters were dropped, that the system
    prompt carried a cache breakpoint, that thinking was adaptive.
    """

    def __init__(
        self,
        messages: Sequence[FakeMessage | Exception] | None = None,
        *,
        token_count: int = 42,
    ) -> None:
        self._queue: list[FakeMessage | Exception] = list(messages or [])
        self.requests: list[dict[str, Any]] = []
        self.token_count_requests: list[dict[str, Any]] = []
        self.token_count = token_count
        self.closed = False
        self.messages = FakeMessagesAPI(self)

    def _next_message(self) -> FakeMessage:
        if not self._queue:
            return FakeMessage(content=[FakeTextBlock(text="ok")], usage=FakeUsage(5, 7))
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def enqueue(self, *items: FakeMessage | Exception) -> FakeAnthropicClient:
        self._queue.extend(items)
        return self

    async def close(self) -> None:
        self.closed = True

    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1] if self.requests else {}


def make_text_message(
    text: str,
    *,
    model: str = "claude-opus-5",
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_read: int = 0,
    cache_write: int = 0,
    with_thinking: bool = False,
) -> FakeMessage:
    """Build a successful message, optionally preceded by a thinking block."""
    content: list[Any] = []
    if with_thinking:
        content.append(FakeThinkingBlock())
    content.append(FakeTextBlock(text=text))
    return FakeMessage(
        content=content,
        model=model,
        stop_reason="end_turn",
        usage=FakeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def make_refusal_message(*, category: str = "cyber", model: str = "claude-opus-5") -> FakeMessage:
    """Build a refusal: HTTP 200, empty content, ``stop_reason='refusal'``."""
    return FakeMessage(
        content=[],
        model=model,
        stop_reason="refusal",
        stop_details=FakeStopDetails(category=category, explanation="declined by policy"),
        usage=FakeUsage(input_tokens=8, output_tokens=0),
    )


__all__ = [
    "FakeAIProvider",
    "FakeAnthropicClient",
    "FakeMessage",
    "FakeStopDetails",
    "FakeTextBlock",
    "FakeThinkingBlock",
    "FakeTokenCount",
    "FakeUsage",
    "make_refusal_message",
    "make_text_message",
]
