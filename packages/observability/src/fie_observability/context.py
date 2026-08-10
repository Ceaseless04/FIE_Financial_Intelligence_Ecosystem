"""Ambient request context propagated through logs, traces, and events.

Every log line, span, and published event carries the same correlation id, so
a single research request can be followed across Atlas -> MarketMind ->
Sentinel -> CFO.ai in one query.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any

from fie_common.utils import new_id


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Identifiers attached to everything emitted while handling one request."""

    correlation_id: str
    request_id: str | None = None
    tenant_id: str | None = None
    principal_id: str | None = None
    service_name: str | None = None
    #: Which of the six products originated the work (atlas, marketmind, ...).
    source_app: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Non-null fields, ready to merge into a log record or span."""
        return {
            key: value
            for key, value in {
                "correlation_id": self.correlation_id,
                "request_id": self.request_id,
                "tenant_id": self.tenant_id,
                "principal_id": self.principal_id,
                "service_name": self.service_name,
                "source_app": self.source_app,
            }.items()
            if value is not None
        }


_EMPTY = RequestContext(correlation_id="")

_context: ContextVar[RequestContext] = ContextVar("fie_request_context", default=_EMPTY)


def current_context() -> RequestContext:
    """The active context, or an empty one when running outside a request."""
    return _context.get()


def set_context(context: RequestContext) -> Token[RequestContext]:
    """Replace the active context, returning a token for restoration."""
    return _context.set(context)


def reset_context(token: Token[RequestContext]) -> None:
    """Restore the context captured by ``set_context``."""
    _context.reset(token)


def bind_context(**fields: Any) -> Token[RequestContext]:
    """Merge ``fields`` into the active context.

    A correlation id is generated when one is not already present, so an entry
    point never has to remember to create one.
    """
    current = _context.get()
    correlation_id = fields.pop("correlation_id", None) or current.correlation_id or new_id("corr")
    updated = replace(current, correlation_id=correlation_id, **fields)
    return _context.set(updated)


@contextmanager
def request_context(**fields: Any) -> Iterator[RequestContext]:
    """Scope a request context to a block, restoring the previous one on exit."""
    token = bind_context(**fields)
    try:
        yield _context.get()
    finally:
        _context.reset(token)


__all__ = [
    "RequestContext",
    "bind_context",
    "current_context",
    "request_context",
    "reset_context",
    "set_context",
]
