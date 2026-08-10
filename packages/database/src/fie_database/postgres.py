"""PostgreSQL access: declarative base, engine lifecycle, and session scoping."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, String, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fie_common.errors import DatabaseError
from fie_common.utils import new_id, utc_now
from fie_database.config import PostgresSettings
from fie_observability.logging import get_logger
from fie_schemas.health import ComponentHealth, HealthStatus

logger = get_logger(__name__)

# Explicit constraint naming makes Alembic autogenerate produce stable,
# reviewable migration names instead of database-assigned identifiers.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every product's models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Audit timestamps applied to persisted rows."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class UUIDPrimaryKeyMixin:
    """String primary key generated application-side."""

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)


class PostgresDatabase:
    """Owns the async engine and hands out scoped sessions."""

    def __init__(
        self, settings: PostgresSettings | None = None, *, engine: AsyncEngine | None = None
    ) -> None:
        self._settings = settings or PostgresSettings()
        self._engine = engine or self._create_engine()
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    def _create_engine(self) -> AsyncEngine:
        engine = create_async_engine(
            self._settings.dsn,
            pool_size=self._settings.pool_size,
            max_overflow=self._settings.max_overflow,
            pool_timeout=self._settings.pool_timeout_seconds,
            pool_recycle=self._settings.pool_recycle_seconds,
            pool_pre_ping=True,
            echo=self._settings.echo,
        )
        if self._settings.statement_timeout_ms > 0:
            self._install_statement_timeout(engine)
        return engine

    def _install_statement_timeout(self, engine: AsyncEngine) -> None:
        """Cap query runtime server-side.

        A single unbounded analytical query holding a connection is how a
        research workload starves the transactional path.
        """
        timeout_ms = self._settings.statement_timeout_ms

        @event.listens_for(engine.sync_engine, "connect")
        def _set_timeout(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(f"SET statement_timeout = {timeout_ms}")
            finally:
                cursor.close()

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Scope a session to a block, committing on success.

        Any exception rolls the transaction back before propagating, so a
        partially-applied write can never be committed by a later handler.
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as error:
            await session.rollback()
            raise DatabaseError(f"database operation failed: {error}") from error
        except BaseException:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def health_check(self) -> ComponentHealth:
        started = time.perf_counter()
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as error:
            return ComponentHealth(
                name="postgres",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                message=str(error),
                required=True,
            )
        return ComponentHealth(
            name="postgres",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            required=True,
        )

    async def create_all(self) -> None:
        """Create tables from metadata. Tests and local bootstrap only.

        Production schema changes go through Alembic so they are reviewable and
        reversible.
        """
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        """Drop all tables. Test teardown only."""
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def aclose(self) -> None:
        await self._engine.dispose()


__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "PostgresDatabase",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
]
