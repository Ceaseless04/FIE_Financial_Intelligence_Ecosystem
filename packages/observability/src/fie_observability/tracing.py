"""OpenTelemetry tracing setup and helpers.

When no tracer provider is configured the OTel API returns no-op objects, so
instrumented code is safe to run in tests and local development without a
collector.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from fie_common.config import CoreSettings
from fie_common.utils import redact_mapping

_configured = False


def configure_tracing(
    settings: CoreSettings | None = None,
    *,
    exporter: SpanExporter | None = None,
    force: bool = False,
) -> TracerProvider | None:
    """Install a tracer provider identifying this service.

    ``exporter`` is injected rather than read from configuration so tests can
    capture spans in memory and production can choose OTLP without this module
    depending on the exporter package.
    """
    global _configured
    if _configured and not force:
        return None

    settings = settings or CoreSettings()
    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.service_version,
            "deployment.environment": str(settings.environment),
        }
    )
    provider = TracerProvider(resource=resource)
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _configured = True
    return provider


def reset_tracing() -> None:
    """Drop configured state so tests can install a fresh provider."""
    global _configured
    _configured = False


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for ``name``."""
    return trace.get_tracer(name)


@contextmanager
def traced(
    name: str,
    *,
    tracer_name: str = "fie",
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Run a block inside a span, recording exceptions and setting status.

    Attributes are redacted before being attached — span attributes reach the
    same backends as logs and are just as capable of leaking a credential.
    """
    tracer = get_tracer(tracer_name)
    safe_attributes = redact_mapping(dict(attributes)) if attributes else {}
    with tracer.start_as_current_span(name, attributes=safe_attributes) as span:
        try:
            yield span
        except BaseException as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


__all__ = ["configure_tracing", "get_tracer", "reset_tracing", "traced"]
