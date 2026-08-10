"""Unit tests for tracing configuration and the ``traced`` helper."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from fie_common.config import CoreSettings, Environment
from fie_common.utils import REDACTED
from fie_observability.tracing import (
    configure_tracing,
    get_tracer,
    reset_tracing,
    traced,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def _memory_exporter() -> Iterator[InMemorySpanExporter]:
    """Configure tracing once for the module.

    OpenTelemetry permits exactly one global tracer provider per process — a
    second ``set_tracer_provider`` is ignored with a warning. That matches real
    usage (configure once at service startup), so the provider is installed
    once here and the exporter is cleared between tests for isolation.
    """
    reset_tracing()
    memory_exporter = InMemorySpanExporter()
    provider = configure_tracing(
        CoreSettings(service_name="test-service", environment=Environment.TEST),
        exporter=memory_exporter,
        force=True,
    )
    assert provider is not None
    yield memory_exporter
    memory_exporter.clear()
    reset_tracing()


@pytest.fixture
def exporter(_memory_exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    """Per-test view of the module's exporter, cleared before each test."""
    _memory_exporter.clear()
    yield _memory_exporter
    _memory_exporter.clear()


class TestConfiguration:
    def test_provider_carries_service_resource_attributes(
        self, exporter: InMemorySpanExporter
    ) -> None:
        with traced("op"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.resource.attributes["service.name"] == "test-service"
        assert span.resource.attributes["deployment.environment"] == "test"

    def test_configure_is_idempotent_without_force(self) -> None:
        reset_tracing()
        configure_tracing(exporter=InMemorySpanExporter())

        assert configure_tracing(exporter=InMemorySpanExporter()) is None
        reset_tracing()

    def test_configure_without_an_exporter_is_allowed(self) -> None:
        reset_tracing()

        assert configure_tracing(force=True) is not None
        reset_tracing()

    def test_get_tracer_returns_a_tracer(self) -> None:
        assert get_tracer("test") is not None


class TestTraced:
    def test_records_a_span_with_the_given_name(self, exporter: InMemorySpanExporter) -> None:
        with traced("ai.complete.claude"):
            pass

        assert [span.name for span in exporter.get_finished_spans()] == ["ai.complete.claude"]

    def test_successful_block_sets_ok_status(self, exporter: InMemorySpanExporter) -> None:
        with traced("op"):
            pass

        assert exporter.get_finished_spans()[0].status.status_code is StatusCode.OK

    def test_exception_is_recorded_and_reraised(self, exporter: InMemorySpanExporter) -> None:
        with pytest.raises(ValueError, match="boom"):
            with traced("op"):
                raise ValueError("boom")

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code is StatusCode.ERROR
        assert any(event.name == "exception" for event in span.events)

    def test_attributes_are_attached(self, exporter: InMemorySpanExporter) -> None:
        with traced("op", attributes={"ai.provider": "claude", "ai.max_tokens": 4096}):
            pass

        attributes = exporter.get_finished_spans()[0].attributes or {}
        assert attributes["ai.provider"] == "claude"
        assert attributes["ai.max_tokens"] == 4096

    def test_sensitive_attributes_are_redacted(self, exporter: InMemorySpanExporter) -> None:
        # Span attributes reach the same backends as logs and leak just as easily.
        with traced("op", attributes={"api_key": "sk-ant-secret", "model": "claude-opus-5"}):
            pass

        attributes = exporter.get_finished_spans()[0].attributes or {}
        assert attributes["api_key"] == REDACTED
        assert attributes["model"] == "claude-opus-5"

    def test_span_is_yielded_for_further_annotation(self, exporter: InMemorySpanExporter) -> None:
        with traced("op") as span:
            span.set_attribute("ai.stop_reason", "end_turn")

        attributes = exporter.get_finished_spans()[0].attributes or {}
        assert attributes["ai.stop_reason"] == "end_turn"

    def test_nested_spans_are_both_recorded(self, exporter: InMemorySpanExporter) -> None:
        with traced("outer"):
            with traced("inner"):
                pass

        assert {span.name for span in exporter.get_finished_spans()} == {"outer", "inner"}
