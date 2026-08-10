"""Unit tests for token accounting and platform metrics."""

from __future__ import annotations

import pytest

from fie_observability.metrics import (
    PlatformMetrics,
    TokenUsage,
    get_metrics,
    reset_metrics,
)

pytestmark = pytest.mark.unit


class TestTokenUsage:
    def test_defaults_to_zero(self) -> None:
        usage = TokenUsage()

        assert usage.total_tokens == 0

    def test_total_includes_cache_token_kinds(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )

        assert usage.total_tokens == 200

    def test_cache_kinds_are_tracked_separately(self) -> None:
        # Cached input is billed at a different rate than uncached input;
        # collapsing them would make cost attribution wrong.
        usage = TokenUsage(cache_read_input_tokens=1000, cache_creation_input_tokens=500)

        assert usage.cache_read_input_tokens == 1000
        assert usage.cache_creation_input_tokens == 500
        assert usage.input_tokens == 0

    def test_addition_sums_every_field(self) -> None:
        first = TokenUsage(input_tokens=10, output_tokens=5, cache_read_input_tokens=1)
        second = TokenUsage(input_tokens=20, output_tokens=7, cache_creation_input_tokens=2)

        combined = first + second

        assert combined.input_tokens == 30
        assert combined.output_tokens == 12
        assert combined.cache_read_input_tokens == 1
        assert combined.cache_creation_input_tokens == 2

    def test_addition_does_not_mutate_operands(self) -> None:
        first = TokenUsage(input_tokens=10)
        second = TokenUsage(input_tokens=20)
        first + second

        assert first.input_tokens == 10
        assert second.input_tokens == 20

    def test_to_dict_exposes_all_kinds_plus_total(self) -> None:
        usage = TokenUsage(input_tokens=1, output_tokens=2)

        assert usage.to_dict() == {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 3,
        }

    def test_is_immutable(self) -> None:
        usage = TokenUsage(input_tokens=1)

        with pytest.raises(AttributeError):
            usage.input_tokens = 2  # type: ignore[misc]


class TestPlatformMetrics:
    @pytest.fixture(autouse=True)
    def _reset(self) -> None:
        reset_metrics()

    def test_singleton_is_reused(self) -> None:
        assert get_metrics() is get_metrics()

    def test_reset_produces_a_new_instance(self) -> None:
        first = get_metrics()
        reset_metrics()

        assert get_metrics() is not first

    def test_instruments_are_created(self) -> None:
        metrics = PlatformMetrics()

        for name in (
            "ai_requests",
            "ai_tokens",
            "ai_latency",
            "agent_executions",
            "agent_latency",
            "events_published",
            "events_consumed",
            "db_queries",
        ):
            assert getattr(metrics, name) is not None

    def test_record_ai_request_accepts_a_full_usage_breakdown(self) -> None:
        metrics = PlatformMetrics()

        metrics.record_ai_request(
            provider="claude",
            model="claude-opus-5",
            status="success",
            duration_ms=120.5,
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=10,
                cache_creation_input_tokens=5,
            ),
        )

    def test_record_ai_request_without_usage(self) -> None:
        PlatformMetrics().record_ai_request(
            provider="claude", model="claude-opus-5", status="error", duration_ms=5.0
        )

    def test_record_ai_request_accepts_extra_attributes(self) -> None:
        PlatformMetrics().record_ai_request(
            provider="ollama",
            model="llama3.1:8b",
            status="success",
            duration_ms=1.0,
            extra_attributes={"agent": "filing"},
        )

    def test_record_agent_execution(self) -> None:
        PlatformMetrics().record_agent_execution(
            agent="valuation", status="success", duration_ms=42.0
        )

    def test_record_event_metrics(self) -> None:
        metrics = PlatformMetrics()

        metrics.record_event_published(
            event_type="marketmind.company.ingested", stream="fie.events.marketmind"
        )
        metrics.record_event_consumed(
            event_type="marketmind.company.ingested",
            stream="fie.events.marketmind",
            status="success",
        )

    def test_record_db_query(self) -> None:
        PlatformMetrics().record_db_query(backend="postgres", operation="select", duration_ms=3.2)
