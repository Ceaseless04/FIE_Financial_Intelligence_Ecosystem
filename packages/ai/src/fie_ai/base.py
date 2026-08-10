"""The ``AIProvider`` abstraction every agent and service codes against.

Concrete providers implement three small hooks (``_complete``, ``_stream``,
``_ping``). The template methods here add the cross-cutting concerns exactly
once — timeout, retry, circuit breaking, tracing, and token/latency metrics —
so no provider can forget them and no caller has to remember them.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    ProviderCapabilities,
    StopReason,
)
from fie_common.errors import AIProviderError, FIEError
from fie_common.resilience import CircuitBreaker, ResiliencePolicy, RetryPolicy
from fie_observability.logging import get_logger
from fie_observability.metrics import PlatformMetrics, get_metrics
from fie_observability.tracing import traced
from fie_schemas.health import ComponentHealth, HealthStatus

logger = get_logger(__name__)


class AIProvider(ABC):
    """Base class for every AI provider integration."""

    #: Stable provider identifier used in metrics, logs, and routing config.
    name: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        resilience: ResiliencePolicy | None = None,
        metrics: PlatformMetrics | None = None,
    ) -> None:
        self._resilience = resilience or ResiliencePolicy(
            timeout_seconds=120.0,
            retry=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.5),
            breaker=CircuitBreaker(f"ai:{self.name}"),
        )
        self._metrics = metrics or get_metrics()

    # -- provider contract ---------------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """What this provider supports."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Model used when a request does not name one."""

    @abstractmethod
    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform the actual provider call. No retry or instrumentation here."""

    @abstractmethod
    async def _ping(self) -> None:
        """Cheap liveness probe. Must not consume model tokens."""

    def _stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Return an async iterator of text deltas.

        Declared as a plain function returning an ``AsyncIterator`` rather than
        an ``async def`` generator: an ``async def`` body would need an
        unreachable ``yield`` to make it a generator. Overrides are free to be
        ``async def`` generators — calling one also returns an ``AsyncIterator``,
        so the signatures are compatible.
        """
        raise NotImplementedError(f"{self.name} does not support streaming")

    # -- template methods ----------------------------------------------------

    def resolve_model(self, request: CompletionRequest) -> str:
        return request.model or self.default_model

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run a completion with the full resilience and observability stack."""
        model = self.resolve_model(request)
        started = time.perf_counter()
        status = "error"
        response: CompletionResponse | None = None

        with traced(
            f"ai.complete.{self.name}",
            attributes={
                "ai.provider": self.name,
                "ai.model": model,
                "ai.max_tokens": request.max_tokens,
                "ai.thinking": str(request.thinking),
            },
        ) as span:
            try:
                response = await self._resilience.execute(
                    lambda: self._complete(request),
                    description=f"{self.name}.complete",
                )
                # A refusal is a successful HTTP call with a declined body. It is
                # recorded distinctly so routing and dashboards can tell it apart
                # from a transport failure.
                status = "refusal" if response.is_refusal else "success"
                span.set_attribute("ai.stop_reason", str(response.stop_reason))
                span.set_attribute("ai.tokens.input", response.usage.input_tokens)
                span.set_attribute("ai.tokens.output", response.usage.output_tokens)
                return response
            except FIEError as error:
                status = error.code
                raise
            finally:
                duration_ms = (time.perf_counter() - started) * 1000.0
                self._metrics.record_ai_request(
                    provider=self.name,
                    model=model,
                    status=status,
                    duration_ms=duration_ms,
                    usage=response.usage if response else None,
                )
                logger.info(
                    "ai_completion",
                    provider=self.name,
                    model=model,
                    status=status,
                    duration_ms=round(duration_ms, 2),
                    **(response.usage.to_dict() if response else {}),
                )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Stream text deltas for ``request``."""
        if not self.capabilities.supports_streaming:
            raise AIProviderError(
                f"{self.name} does not support streaming",
                details={"provider": self.name},
            )
        async for chunk in self._stream(request):
            yield chunk

    async def count_tokens(self, request: CompletionRequest) -> int:
        """Count input tokens for ``request``.

        The default is a deliberately crude character-based approximation for
        providers with no counting endpoint. Never use it for billing — Phase 7
        cost attribution must read real usage off the response.
        """
        text = (request.system or "") + "".join(message.content for message in request.messages)
        return max(1, len(text) // 4)

    async def health_check(self) -> ComponentHealth:
        """Probe the provider without consuming tokens."""
        started = time.perf_counter()
        try:
            await self._ping()
        except Exception as error:
            return ComponentHealth(
                name=f"ai:{self.name}",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                message=str(error),
                required=False,
            )
        return ComponentHealth(
            name=f"ai:{self.name}",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            required=False,
        )

    async def aclose(self) -> None:
        """Release provider resources. Override where a client must be closed."""
        return None

    async def __aenter__(self) -> AIProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.default_model!r})"


def refusal_response(
    *, provider: str, model: str, category: str | None, message: str
) -> CompletionResponse:
    """Build the canonical refusal response shape."""
    return CompletionResponse(
        text=message,
        model=model,
        provider=provider,
        stop_reason=StopReason.REFUSAL,
        refusal_category=category,
    )


__all__ = ["AIProvider", "refusal_response"]
