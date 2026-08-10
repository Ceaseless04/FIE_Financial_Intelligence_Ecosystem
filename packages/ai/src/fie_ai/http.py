"""Shared HTTP client construction for HTTP-based providers.

``httpx.AsyncClient()`` eagerly builds an ``ssl.SSLContext``, which loads and
parses the system CA bundle — roughly 570ms per client on a cold cache, paid
even for plain-HTTP base URLs. Providers are normally long-lived singletons so
this is invisible in steady state, but it is paid again on every reconnect,
per-tenant client, or test-suite construction.

Caching one context per verification setting reduces client construction to
about 1ms without changing verification behaviour.
"""

from __future__ import annotations

import ssl
from functools import lru_cache
from typing import Any

import httpx


@lru_cache(maxsize=4)
def _shared_ssl_context(verify: bool) -> ssl.SSLContext:
    """Process-wide SSL context.

    ``ssl.SSLContext`` is safe to share across clients and connections; httpx
    builds an equivalent one per client by default.
    """
    if not verify:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    return ssl.create_default_context()


def build_async_client(
    *,
    base_url: str,
    timeout: float,
    headers: dict[str, str] | None = None,
    verify: bool = True,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Construct an ``httpx.AsyncClient`` backed by the shared SSL context."""
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        headers=headers or {},
        verify=_shared_ssl_context(verify),
        **kwargs,
    )


__all__ = ["build_async_client"]
