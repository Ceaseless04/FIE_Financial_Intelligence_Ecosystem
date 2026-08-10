"""Neo4j access — the substrate for the MarketMind knowledge graph (Phase 2).

Phase 1 provides connection management, transaction scoping, and health
checks only. Entity and relationship semantics belong to MarketMind.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from fie_common.errors import DatabaseError, ValidationError
from fie_database.config import Neo4jSettings
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth, HealthStatus

logger = get_logger(__name__)


class Neo4jClient:
    """Async Neo4j driver wrapper with read/write transaction helpers."""

    def __init__(
        self, settings: Neo4jSettings | None = None, *, driver: AsyncDriver | None = None
    ) -> None:
        self._settings = settings or Neo4jSettings()
        self._driver = driver or AsyncGraphDatabase.driver(
            self._settings.uri,
            auth=self._settings.auth,
            max_connection_pool_size=self._settings.max_connection_pool_size,
            connection_timeout=self._settings.connection_timeout_seconds,
            max_transaction_retry_time=self._settings.max_transaction_retry_seconds,
        )

    @property
    def driver(self) -> AsyncDriver:
        return self._driver

    @asynccontextmanager
    async def session(self, *, database: str | None = None) -> AsyncIterator[AsyncSession]:
        session = self._driver.session(database=database or self._settings.database)
        try:
            yield session
        finally:
            await session.close()

    async def execute_read(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read query and materialize the result rows."""
        self._reject_writes(query)
        return await self._execute(query, parameters, write=False)

    async def execute_write(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a write query inside a managed transaction."""
        return await self._execute(query, parameters, write=True)

    async def _execute(
        self, query: str, parameters: Mapping[str, Any] | None, *, write: bool
    ) -> list[dict[str, Any]]:
        params = dict(parameters or {})

        async def work(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(query, params)
            return [record.data() async for record in result]

        try:
            async with self.session() as session:
                runner = session.execute_write if write else session.execute_read
                rows: list[dict[str, Any]] = await runner(work)
                return rows
        except ServiceUnavailable as error:
            raise DatabaseError(f"Neo4j is unavailable: {error}") from error
        except Neo4jError as error:
            raise DatabaseError(
                f"Neo4j query failed: {error}", details={"code": getattr(error, "code", None)}
            ) from error

    @staticmethod
    def _reject_writes(query: str) -> None:
        """Guard the read path against mutating clauses.

        Read queries route to follower replicas in a cluster; a mutation sent
        down that path fails at runtime in production but succeeds against a
        single-node dev instance, so it is caught here instead.
        """
        mutating = ("CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP")
        normalized = " ".join(query.upper().split())
        if any(keyword in normalized for keyword in mutating):
            raise ValidationError(
                "mutating clause found in a read query; use execute_write instead",
                details={"query_preview": query[:120]},
            )

    async def health_check(self) -> ComponentHealth:
        started = time.perf_counter()
        try:
            await self._driver.verify_connectivity()
        except Exception as error:
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                message=str(error),
                required=False,
            )
        return ComponentHealth(
            name="neo4j",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            required=False,
        )

    async def aclose(self) -> None:
        await self._driver.close()


__all__ = ["Neo4jClient"]
