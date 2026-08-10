"""Redis access: caching, counters, and distributed locks."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from fie_common.errors import ConflictError, DatabaseError
from fie_common.utils import new_id
from fie_database.config import RedisSettings
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth, HealthStatus

logger = get_logger(__name__)

# Releases a lock only if this caller still owns it. Without the ownership
# check, a caller whose lock already expired would delete the next holder's.
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class RedisClient:
    """Thin async Redis wrapper with JSON helpers and a safe lock."""

    def __init__(self, settings: RedisSettings | None = None, *, client: Any | None = None) -> None:
        self._settings = settings or RedisSettings()
        self._client = client or aioredis.from_url(
            self._settings.dsn,
            max_connections=self._settings.max_connections,
            socket_timeout=self._settings.socket_timeout_seconds,
            decode_responses=True,
        )

    @property
    def client(self) -> Any:
        """The underlying client, for operations this wrapper does not cover."""
        return self._client

    async def get_json(self, key: str) -> Any | None:
        """Read and decode a JSON value, returning ``None`` when absent."""
        try:
            raw = await self._client.get(key)
        except RedisError as error:
            raise DatabaseError(f"redis GET failed for {key!r}: {error}") from error
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # A corrupt cache entry must not break the caller — treat it as a miss.
            logger.warning("redis_corrupt_cache_entry", key=key)
            return None

    async def set_json(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        """Write a JSON value with a TTL.

        A TTL is always applied: unbounded cache entries are how a Redis
        instance silently becomes a database.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._settings.default_ttl_seconds
        try:
            await self._client.set(key, json.dumps(value, default=str), ex=ttl)
        except RedisError as error:
            raise DatabaseError(f"redis SET failed for {key!r}: {error}") from error

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        try:
            deleted: int = await self._client.delete(*keys)
            return deleted
        except RedisError as error:
            raise DatabaseError(f"redis DEL failed: {error}") from error

    async def exists(self, key: str) -> bool:
        try:
            return bool(await self._client.exists(key))
        except RedisError as error:
            raise DatabaseError(f"redis EXISTS failed for {key!r}: {error}") from error

    async def increment(self, key: str, *, amount: int = 1, ttl_seconds: int | None = None) -> int:
        try:
            value: int = await self._client.incrby(key, amount)
            if ttl_seconds is not None and value == amount:
                await self._client.expire(key, ttl_seconds)
            return value
        except RedisError as error:
            raise DatabaseError(f"redis INCRBY failed for {key!r}: {error}") from error

    async def set_if_absent(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        """SET NX with a TTL. The primitive behind idempotency keys."""
        try:
            return bool(await self._client.set(key, value, nx=True, ex=ttl_seconds))
        except RedisError as error:
            raise DatabaseError(f"redis SET NX failed for {key!r}: {error}") from error

    @asynccontextmanager
    async def lock(
        self, key: str, *, ttl_seconds: int = 30, owner: str | None = None
    ) -> AsyncIterator[str]:
        """Acquire a distributed lock for the duration of a block.

        The lock always carries a TTL so a crashed holder cannot deadlock the
        system, and release is ownership-checked via a Lua script.

        Raises:
            ConflictError: if the lock is already held.
        """
        token = owner or new_id("lock")
        lock_key = f"lock:{key}"
        acquired = await self.set_if_absent(lock_key, token, ttl_seconds=ttl_seconds)
        if not acquired:
            raise ConflictError(f"could not acquire lock {key!r}", details={"lock_key": lock_key})
        try:
            yield token
        finally:
            try:
                await self._client.eval(_RELEASE_SCRIPT, 1, lock_key, token)
            except RedisError as error:  # pragma: no cover — best-effort release
                logger.warning("redis_lock_release_failed", key=lock_key, error=str(error))

    async def health_check(self) -> ComponentHealth:
        started = time.perf_counter()
        try:
            await self._client.ping()
        except Exception as error:  # noqa: BLE001 — a health probe reports
            # failure rather than raising; see AIProvider.health_check.
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                message=str(error),
                required=True,
            )
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            required=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["RedisClient"]
