"""Integration-test infrastructure probes.

Integration tests must exercise real Postgres, Redis, and Neo4j — not mocks.
When that infrastructure is not running, they skip with an explicit reason
rather than failing, so a developer without Docker still gets a green unit
suite while CI (which does start service containers) runs the full set.

Set ``FIE_REQUIRE_INFRA=1`` to turn skips into failures. CI sets this so a
silently-unavailable service can never be mistaken for a passing gate.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

import pytest

REQUIRE_INFRA_ENV = "FIE_REQUIRE_INFRA"


def infra_required() -> bool:
    """Whether missing infrastructure should fail rather than skip."""
    return os.getenv(REQUIRE_INFRA_ENV, "").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=32)
def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Probe a TCP endpoint once per process.

    Cached because service availability does not change during a test run — CI
    starts its service containers before pytest — and an uncached probe costs a
    full connect timeout for every skipped test.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (TimeoutError, OSError):
        return False


def reset_probe_cache() -> None:
    """Clear cached availability. For tests that start a service mid-run."""
    _port_open.cache_clear()


@dataclass(frozen=True)
class ServiceEndpoint:
    name: str
    host: str
    port: int

    @property
    def is_available(self) -> bool:
        return _port_open(self.host, self.port)

    @property
    def skip_reason(self) -> str:
        return (
            f"{self.name} is not reachable at {self.host}:{self.port} — "
            f"start it with `docker compose -f docker-compose.dev.yml up -d {self.name}`"
        )


def _endpoint_from_env(
    name: str, host_var: str, port_var: str, default_port: int
) -> ServiceEndpoint:
    return ServiceEndpoint(
        name=name,
        host=os.getenv(host_var, "localhost"),
        port=int(os.getenv(port_var, str(default_port))),
    )


def postgres_endpoint() -> ServiceEndpoint:
    return _endpoint_from_env("postgres", "FIE_POSTGRES_HOST", "FIE_POSTGRES_PORT", 5432)


def redis_endpoint() -> ServiceEndpoint:
    return _endpoint_from_env("redis", "FIE_REDIS_HOST", "FIE_REDIS_PORT", 6379)


def neo4j_endpoint() -> ServiceEndpoint:
    uri = os.getenv("FIE_NEO4J_URI", "bolt://localhost:7687")
    parsed = urlparse(uri)
    return ServiceEndpoint(
        name="neo4j", host=parsed.hostname or "localhost", port=parsed.port or 7687
    )


def ollama_endpoint() -> ServiceEndpoint:
    base = os.getenv("FIE_OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urlparse(base)
    return ServiceEndpoint(
        name="ollama", host=parsed.hostname or "localhost", port=parsed.port or 11434
    )


def require_service(endpoint: ServiceEndpoint) -> None:
    """Skip (or fail, under ``FIE_REQUIRE_INFRA``) when a service is down."""
    if endpoint.is_available:
        return
    if infra_required():
        pytest.fail(f"{REQUIRE_INFRA_ENV} is set but {endpoint.skip_reason}")
    pytest.skip(endpoint.skip_reason)


def has_anthropic_key() -> bool:
    """Whether a live Claude API key is configured for provider tests."""
    return bool(os.getenv("FIE_CLAUDE_API_KEY", "").strip())


def require_anthropic_key() -> None:
    if has_anthropic_key():
        return
    pytest.skip("FIE_CLAUDE_API_KEY is not set — skipping live Claude provider test")


__all__ = [
    "REQUIRE_INFRA_ENV",
    "ServiceEndpoint",
    "has_anthropic_key",
    "infra_required",
    "neo4j_endpoint",
    "ollama_endpoint",
    "postgres_endpoint",
    "redis_endpoint",
    "require_anthropic_key",
    "require_service",
    "reset_probe_cache",
]
