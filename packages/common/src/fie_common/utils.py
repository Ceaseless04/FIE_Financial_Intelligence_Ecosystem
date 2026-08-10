"""Small, dependency-free helpers shared across the platform."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar

T = TypeVar("T")

#: Keys whose values are redacted from logs, traces, and error details.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "api_key",
        "apikey",
        "anthropic_api_key",
        "private_key",
        "client_secret",
        "session_key",
        "credential",
        "credentials",
        "dsn",
        "connection_string",
        "database_url",
    }
)

REDACTED = "***REDACTED***"


def utc_now() -> datetime:
    """Timezone-aware current time in UTC.

    Always use this instead of ``datetime.now()``: naive timestamps silently
    corrupt time-series comparisons across services in different regions.
    """
    return datetime.now(UTC)


def new_id(prefix: str = "") -> str:
    """Generate a unique identifier, optionally namespaced by ``prefix``."""
    raw = uuid.uuid4().hex
    return f"{prefix}_{raw}" if prefix else raw


def redact_secret(value: str | None, *, keep: int = 4) -> str:
    """Mask a secret, leaving the last ``keep`` characters for correlation."""
    if not value:
        return REDACTED
    if keep <= 0 or len(value) <= keep:
        return REDACTED
    return f"{REDACTED}{value[-keep:]}"


def redact_mapping(
    data: Mapping[str, Any],
    *,
    sensitive_keys: Iterable[str] = SENSITIVE_KEYS,
    max_depth: int = 6,
) -> dict[str, Any]:
    """Recursively redact sensitive values from a mapping.

    Applied to every structured log record and error payload so credentials
    cannot leak into log aggregation or an API error response.
    """
    lowered = {key.lower() for key in sensitive_keys}

    def _walk(value: Any, depth: int) -> Any:
        if depth > max_depth:
            return value
        if isinstance(value, Mapping):
            return {
                key: (REDACTED if str(key).lower() in lowered else _walk(item, depth + 1))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_walk(item, depth + 1) for item in value]
        return value

    result = _walk(data, 0)
    assert isinstance(result, dict)
    return result


def stable_json(value: Any) -> str:
    """Serialize to JSON deterministically.

    Sorted keys and fixed separators matter for two reasons: prompt caching is
    a byte-prefix match, so unstable key ordering silently destroys the cache
    hit rate; and idempotency keys derived from payloads must be reproducible.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    """Stable SHA-256 of any JSON-serializable value."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield ``items`` in lists of at most ``size``."""
    if size < 1:
        raise ValueError("size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def coalesce(*values: T | None) -> T | None:
    """Return the first non-``None`` argument."""
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "REDACTED",
    "SENSITIVE_KEYS",
    "chunked",
    "coalesce",
    "content_hash",
    "new_id",
    "redact_mapping",
    "redact_secret",
    "stable_json",
    "utc_now",
]
