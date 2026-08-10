"""fie_observability — structured logging, tracing, and metrics."""

from fie_observability.context import (
    RequestContext,
    bind_context,
    current_context,
    request_context,
    reset_context,
    set_context,
)
from fie_observability.logging import configure_logging, get_logger, reset_logging
from fie_observability.metrics import (
    PlatformMetrics,
    TokenUsage,
    get_metrics,
    reset_metrics,
)
from fie_observability.tracing import (
    configure_tracing,
    get_tracer,
    reset_tracing,
    traced,
)

__version__ = "0.1.0"

__all__ = [
    "PlatformMetrics",
    "RequestContext",
    "TokenUsage",
    "__version__",
    "bind_context",
    "configure_logging",
    "configure_tracing",
    "current_context",
    "get_logger",
    "get_metrics",
    "get_tracer",
    "request_context",
    "reset_context",
    "reset_logging",
    "reset_metrics",
    "reset_tracing",
    "set_context",
    "traced",
]
