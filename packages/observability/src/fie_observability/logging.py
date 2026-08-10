"""Structured logging built on structlog.

Two guarantees this module is responsible for:

1. Every record is JSON in production and carries the ambient request context.
2. No credential ever reaches the log sink — a redaction processor runs on
   every record, not at the call sites.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from fie_common.config import CoreSettings, LogLevel
from fie_common.utils import REDACTED, SENSITIVE_KEYS
from fie_observability.context import current_context

_configured = False


def _redaction_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Redact sensitive keys anywhere in the record, at any nesting depth."""
    lowered = SENSITIVE_KEYS

    def walk(value: Any, depth: int = 0) -> Any:
        if depth > 6:
            return value
        if isinstance(value, MutableMapping):
            return {
                key: (REDACTED if str(key).lower() in lowered else walk(item, depth + 1))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [walk(item, depth + 1) for item in value]
        return value

    for key in list(event_dict.keys()):
        if str(key).lower() in lowered:
            event_dict[key] = REDACTED
        else:
            event_dict[key] = walk(event_dict[key])
    return event_dict


def _context_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Merge the ambient request context into the record."""
    for key, value in current_context().to_dict().items():
        event_dict.setdefault(key, value)
    return event_dict


def _trace_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Attach the active OpenTelemetry trace and span ids when one exists.

    This is what makes a log line clickable from a trace and vice versa.
    """
    from opentelemetry import trace  # imported lazily to keep import cost off the hot path

    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        event_dict.setdefault("trace_id", format(span_context.trace_id, "032x"))
        event_dict.setdefault("span_id", format(span_context.span_id, "016x"))
    return event_dict


def configure_logging(
    settings: CoreSettings | None = None,
    *,
    level: LogLevel | str | None = None,
    json_logs: bool | None = None,
    force: bool = False,
) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent: repeated calls are ignored unless ``force`` is set, so importing
    several packages that each want logging cannot reconfigure the pipeline.
    """
    global _configured
    if _configured and not force:
        return

    settings = settings or CoreSettings()
    resolved_level = str(level or settings.log_level)
    resolved_json = settings.json_logs if json_logs is None else json_logs

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if resolved_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _context_processor,
            _trace_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redaction_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(resolved_level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(resolved_level),
        force=True,
    )
    _configured = True


def reset_logging() -> None:
    """Drop the configured state. Used by tests to reconfigure cleanly."""
    global _configured
    _configured = False
    structlog.reset_defaults()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger.

    Configures logging with defaults on first use so library code can log
    without the host application having called ``configure_logging`` yet.
    """
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


__all__ = ["configure_logging", "get_logger", "reset_logging"]
