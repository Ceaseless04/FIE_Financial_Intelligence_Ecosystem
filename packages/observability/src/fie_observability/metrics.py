"""Platform metrics, including the AI telemetry FinOps consumes.

Scope boundary: this module *records* token counts, latencies, and call
outcomes. It deliberately does not price them — cost calculation is FinOps
domain logic (Phase 7) and must live in a deterministic service, not in
shared infrastructure and not in an LLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import metrics as otel_metrics

_METER_NAME = "fie.platform"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single model call.

    Cache fields are tracked separately because cached input is billed at a
    different rate than uncached input; collapsing them would make any
    downstream cost attribution wrong.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class PlatformMetrics:
    """Lazily-created OTel instruments shared across the platform."""

    meter: otel_metrics.Meter = field(default_factory=lambda: otel_metrics.get_meter(_METER_NAME))

    def __post_init__(self) -> None:
        self.ai_requests = self.meter.create_counter(
            "fie.ai.requests",
            unit="1",
            description="AI provider requests by provider, model, and outcome",
        )
        self.ai_tokens = self.meter.create_counter(
            "fie.ai.tokens",
            unit="1",
            description="AI tokens by provider, model, and token kind",
        )
        self.ai_latency = self.meter.create_histogram(
            "fie.ai.request.duration",
            unit="ms",
            description="AI provider request latency",
        )
        self.agent_executions = self.meter.create_counter(
            "fie.agent.executions",
            unit="1",
            description="Agent executions by agent name and outcome",
        )
        self.agent_latency = self.meter.create_histogram(
            "fie.agent.duration",
            unit="ms",
            description="Agent execution wall time",
        )
        self.events_published = self.meter.create_counter(
            "fie.events.published",
            unit="1",
            description="Domain events published by type",
        )
        self.events_consumed = self.meter.create_counter(
            "fie.events.consumed",
            unit="1",
            description="Domain events consumed by type and outcome",
        )
        self.db_queries = self.meter.create_histogram(
            "fie.db.query.duration",
            unit="ms",
            description="Database query latency by backend and operation",
        )

    def record_ai_request(
        self,
        *,
        provider: str,
        model: str,
        status: str,
        duration_ms: float,
        usage: TokenUsage | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one AI provider call: outcome, latency, and token breakdown."""
        attributes: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "status": status,
        }
        if extra_attributes:
            attributes.update(extra_attributes)

        self.ai_requests.add(1, attributes)
        self.ai_latency.record(duration_ms, attributes)

        if usage is not None:
            for kind, value in (
                ("input", usage.input_tokens),
                ("output", usage.output_tokens),
                ("cache_write", usage.cache_creation_input_tokens),
                ("cache_read", usage.cache_read_input_tokens),
            ):
                if value:
                    self.ai_tokens.add(value, {**attributes, "token_kind": kind})

    def record_agent_execution(self, *, agent: str, status: str, duration_ms: float) -> None:
        attributes = {"agent": agent, "status": status}
        self.agent_executions.add(1, attributes)
        self.agent_latency.record(duration_ms, attributes)

    def record_event_published(self, *, event_type: str, stream: str) -> None:
        self.events_published.add(1, {"event_type": event_type, "stream": stream})

    def record_event_consumed(self, *, event_type: str, stream: str, status: str) -> None:
        self.events_consumed.add(1, {"event_type": event_type, "stream": stream, "status": status})

    def record_db_query(self, *, backend: str, operation: str, duration_ms: float) -> None:
        self.db_queries.record(duration_ms, {"backend": backend, "operation": operation})


_metrics: PlatformMetrics | None = None


def get_metrics() -> PlatformMetrics:
    """Process-wide metrics singleton."""
    global _metrics
    if _metrics is None:
        _metrics = PlatformMetrics()
    return _metrics


def reset_metrics() -> None:
    """Drop the singleton so tests can rebind a fresh meter provider."""
    global _metrics
    _metrics = None


__all__ = ["PlatformMetrics", "TokenUsage", "get_metrics", "reset_metrics"]
