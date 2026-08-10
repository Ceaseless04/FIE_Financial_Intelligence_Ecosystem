"""Integration tests against real PostgreSQL, Redis, and Neo4j.

These deliberately do not use mocks. They skip with an explicit reason when the
infrastructure is not running locally, and CI sets ``FIE_REQUIRE_INFRA=1`` so a
silently-unavailable service fails the gate instead of quietly skipping.

Start the stack with:
    docker compose -f docker-compose.dev.yml up -d postgres redis neo4j
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import String, select, text
from sqlalchemy.orm import Mapped, mapped_column

from fie_common.errors import ConflictError, DatabaseError, ValidationError
from fie_database.config import Neo4jSettings, PostgresSettings, RedisSettings
from fie_database.neo4j_client import Neo4jClient
from fie_database.postgres import Base, PostgresDatabase
from fie_database.redis_client import RedisClient
from fie_testing.infra import (
    neo4j_endpoint,
    postgres_endpoint,
    redis_endpoint,
    require_service,
)

pytestmark = [pytest.mark.integration]


class SampleRow(Base):
    """Throwaway table used only to prove the ORM path works end to end."""

    __tablename__ = "fie_integration_sample"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(255))


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


@pytest.fixture
async def postgres() -> AsyncIterator[PostgresDatabase]:
    require_service(postgres_endpoint())
    database = PostgresDatabase(PostgresSettings(statement_timeout_ms=5_000))
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[SampleRow.__table__])
    yield database
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all, tables=[SampleRow.__table__])
    await database.aclose()


class TestPostgresIntegration:
    async def test_health_check_reports_healthy(self, postgres: PostgresDatabase) -> None:
        health = await postgres.health_check()

        assert health.status.is_serving is True
        assert health.latency_ms is not None

    async def test_session_commits_on_success(self, postgres: PostgresDatabase) -> None:
        async with postgres.session() as session:
            session.add(SampleRow(id="row-1", label="committed"))

        async with postgres.session() as session:
            found = await session.get(SampleRow, "row-1")
            assert found is not None
            assert found.label == "committed"

    async def test_session_rolls_back_on_error(self, postgres: PostgresDatabase) -> None:
        # A partially-applied write must never survive a failed handler.
        with pytest.raises(RuntimeError):
            async with postgres.session() as session:
                session.add(SampleRow(id="row-2", label="should not persist"))
                await session.flush()
                raise RuntimeError("handler failed")

        async with postgres.session() as session:
            assert await session.get(SampleRow, "row-2") is None

    async def test_sqlalchemy_errors_are_translated(self, postgres: PostgresDatabase) -> None:
        with pytest.raises(DatabaseError):
            async with postgres.session() as session:
                await session.execute(text("SELECT * FROM table_that_does_not_exist"))

    async def test_duplicate_primary_key_is_rejected(self, postgres: PostgresDatabase) -> None:
        async with postgres.session() as session:
            session.add(SampleRow(id="row-dup", label="first"))

        with pytest.raises(DatabaseError):
            async with postgres.session() as session:
                session.add(SampleRow(id="row-dup", label="second"))

    async def test_queries_run_through_the_orm(self, postgres: PostgresDatabase) -> None:
        async with postgres.session() as session:
            session.add_all([SampleRow(id=f"row-q{i}", label="queryable") for i in range(3)])

        async with postgres.session() as session:
            result = await session.execute(select(SampleRow).where(SampleRow.label == "queryable"))
            assert len(result.scalars().all()) == 3

    async def test_constraint_naming_convention_is_applied(
        self, postgres: PostgresDatabase
    ) -> None:
        # Stable constraint names are what make Alembic autogenerate reviewable.
        assert SampleRow.__table__.primary_key.name == "pk_fie_integration_sample"


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.fixture
async def redis() -> AsyncIterator[RedisClient]:
    require_service(redis_endpoint())
    client = RedisClient(RedisSettings())
    await client.client.flushdb()
    yield client
    await client.client.flushdb()
    await client.aclose()


class TestRedisIntegration:
    async def test_health_check_reports_healthy(self, redis: RedisClient) -> None:
        assert (await redis.health_check()).status.is_serving is True

    async def test_json_round_trip(self, redis: RedisClient) -> None:
        await redis.set_json("company:ACME", {"ticker": "ACME", "employees": 1200})

        assert await redis.get_json("company:ACME") == {
            "ticker": "ACME",
            "employees": 1200,
        }

    async def test_ttl_is_applied(self, redis: RedisClient) -> None:
        await redis.set_json("expiring", {"v": 1}, ttl_seconds=60)

        assert await redis.client.ttl("expiring") > 0

    async def test_set_if_absent_is_atomic(self, redis: RedisClient) -> None:
        assert await redis.set_if_absent("once", "a", ttl_seconds=60) is True
        assert await redis.set_if_absent("once", "b", ttl_seconds=60) is False

    async def test_increment_and_expire(self, redis: RedisClient) -> None:
        assert await redis.increment("counter", ttl_seconds=60) == 1
        assert await redis.increment("counter", amount=4) == 5

    async def test_distributed_lock_is_exclusive(self, redis: RedisClient) -> None:
        async with redis.lock("ingest:ACME", ttl_seconds=30):
            with pytest.raises(ConflictError):
                async with redis.lock("ingest:ACME", ttl_seconds=30):
                    pass

    async def test_lock_release_is_ownership_checked(self, redis: RedisClient) -> None:
        async with redis.lock("shared", ttl_seconds=30, owner="owner-a"):
            await redis.client.set("lock:shared", "owner-b")

        assert await redis.client.get("lock:shared") == "owner-b"


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


@pytest.fixture
async def neo4j() -> AsyncIterator[Neo4jClient]:
    require_service(neo4j_endpoint())
    client = Neo4jClient(Neo4jSettings())
    await client.execute_write("MATCH (n:FIEIntegrationTest) DETACH DELETE n")
    yield client
    await client.execute_write("MATCH (n:FIEIntegrationTest) DETACH DELETE n")
    await client.aclose()


class TestNeo4jIntegration:
    async def test_health_check_reports_healthy(self, neo4j: Neo4jClient) -> None:
        assert (await neo4j.health_check()).status.is_serving is True

    async def test_write_then_read(self, neo4j: Neo4jClient) -> None:
        await neo4j.execute_write(
            "CREATE (n:FIEIntegrationTest {name: $name, value: $value})",
            {"name": "node-1", "value": 42},
        )

        rows = await neo4j.execute_read(
            "MATCH (n:FIEIntegrationTest {name: $name}) RETURN n.value AS value",
            {"name": "node-1"},
        )

        assert rows == [{"value": 42}]

    async def test_relationships_traverse(self, neo4j: Neo4jClient) -> None:
        await neo4j.execute_write(
            """
            CREATE (a:FIEIntegrationTest {name: 'supplier'})
            CREATE (b:FIEIntegrationTest {name: 'customer'})
            CREATE (a)-[:SUPPLIES]->(b)
            """
        )

        rows = await neo4j.execute_read(
            """
            MATCH (a:FIEIntegrationTest)-[:SUPPLIES]->(b:FIEIntegrationTest)
            RETURN a.name AS supplier, b.name AS customer
            """
        )

        assert rows == [{"supplier": "supplier", "customer": "customer"}]

    async def test_the_read_path_rejects_mutations(self, neo4j: Neo4jClient) -> None:
        with pytest.raises(ValidationError, match="mutating clause"):
            await neo4j.execute_read("CREATE (n:FIEIntegrationTest)")

    async def test_invalid_cypher_is_translated(self, neo4j: Neo4jClient) -> None:
        with pytest.raises(DatabaseError):
            await neo4j.execute_read("THIS IS NOT CYPHER")

    async def test_parameters_are_bound_not_interpolated(self, neo4j: Neo4jClient) -> None:
        # Parameter binding is what keeps user-supplied entity names from
        # becoming Cypher injection.
        hostile = "x' OR 1=1 //"
        await neo4j.execute_write("CREATE (n:FIEIntegrationTest {name: $name})", {"name": hostile})

        rows = await neo4j.execute_read(
            "MATCH (n:FIEIntegrationTest {name: $name}) RETURN count(n) AS total",
            {"name": hostile},
        )

        assert rows == [{"total": 1}]
