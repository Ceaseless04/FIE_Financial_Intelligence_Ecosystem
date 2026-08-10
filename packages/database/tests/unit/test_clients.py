"""Unit tests for the Redis and Neo4j client wrappers.

The Redis tests run against a functional in-memory double that implements real
key/value, expiry, and Lua-release semantics — the same logic then runs against
real Redis in the integration suite.
"""

from __future__ import annotations

import pytest
from redis.exceptions import RedisError

from fie_common.errors import ConflictError, DatabaseError, ValidationError
from fie_database.neo4j_client import Neo4jClient
from fie_database.redis_client import RedisClient
from fie_testing.fake_redis import FakeRedis

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def redis_client(fake_redis: FakeRedis) -> RedisClient:
    return RedisClient(client=fake_redis)


class TestRedisJsonHelpers:
    async def test_set_then_get_round_trips(self, redis_client: RedisClient) -> None:
        await redis_client.set_json("company:ACME", {"ticker": "ACME", "sector": "tech"})

        assert await redis_client.get_json("company:ACME") == {
            "ticker": "ACME",
            "sector": "tech",
        }

    async def test_missing_key_returns_none(self, redis_client: RedisClient) -> None:
        assert await redis_client.get_json("absent") is None

    async def test_a_ttl_is_always_applied(
        self, redis_client: RedisClient, fake_redis: FakeRedis
    ) -> None:
        # Unbounded cache entries turn Redis into an unmanaged database.
        await redis_client.set_json("k", {"v": 1}, ttl_seconds=60)

        assert await fake_redis.ttl("k") > 0

    async def test_entries_expire(self, redis_client: RedisClient, fake_redis: FakeRedis) -> None:
        await redis_client.set_json("k", {"v": 1}, ttl_seconds=10)
        fake_redis.advance(11_000)

        assert await redis_client.get_json("k") is None

    async def test_a_corrupt_entry_is_treated_as_a_miss(
        self, redis_client: RedisClient, fake_redis: FakeRedis
    ) -> None:
        # A poisoned cache entry must not break the caller.
        await fake_redis.set("k", "{not json")

        assert await redis_client.get_json("k") is None

    async def test_non_serializable_values_fall_back_to_str(
        self, redis_client: RedisClient
    ) -> None:
        from datetime import datetime

        await redis_client.set_json("k", {"when": datetime(2024, 1, 1)})

        assert await redis_client.get_json("k") is not None


class TestRedisPrimitives:
    async def test_delete_reports_how_many_keys_were_removed(
        self, redis_client: RedisClient
    ) -> None:
        await redis_client.set_json("a", 1)
        await redis_client.set_json("b", 2)

        assert await redis_client.delete("a", "b", "missing") == 2

    async def test_delete_with_no_keys_is_a_no_op(self, redis_client: RedisClient) -> None:
        assert await redis_client.delete() == 0

    async def test_exists(self, redis_client: RedisClient) -> None:
        await redis_client.set_json("k", 1)

        assert await redis_client.exists("k") is True
        assert await redis_client.exists("nope") is False

    async def test_increment_counts_up(self, redis_client: RedisClient) -> None:
        assert await redis_client.increment("hits") == 1
        assert await redis_client.increment("hits", amount=5) == 6

    async def test_increment_sets_a_ttl_on_first_write(
        self, redis_client: RedisClient, fake_redis: FakeRedis
    ) -> None:
        await redis_client.increment("hits", ttl_seconds=60)

        assert await fake_redis.ttl("hits") > 0

    async def test_set_if_absent_is_exclusive(self, redis_client: RedisClient) -> None:
        assert await redis_client.set_if_absent("k", "first", ttl_seconds=60) is True
        assert await redis_client.set_if_absent("k", "second", ttl_seconds=60) is False


class TestRedisLock:
    async def test_lock_is_acquired_and_released(self, redis_client: RedisClient) -> None:
        async with redis_client.lock("ingest:ACME") as token:
            assert token

        # Released, so it can be taken again.
        async with redis_client.lock("ingest:ACME"):
            pass

    async def test_a_held_lock_blocks_a_second_holder(self, redis_client: RedisClient) -> None:
        async with redis_client.lock("ingest:ACME"):
            with pytest.raises(ConflictError, match="could not acquire lock"):
                async with redis_client.lock("ingest:ACME"):
                    pass

    async def test_lock_is_released_even_when_the_block_raises(
        self, redis_client: RedisClient
    ) -> None:
        with pytest.raises(RuntimeError):
            async with redis_client.lock("ingest:ACME"):
                raise RuntimeError("handler failed")

        async with redis_client.lock("ingest:ACME"):
            pass

    async def test_lock_carries_a_ttl_so_a_crash_cannot_deadlock(
        self, redis_client: RedisClient, fake_redis: FakeRedis
    ) -> None:
        await redis_client.set_if_absent("lock:stuck", "dead-owner", ttl_seconds=30)
        fake_redis.advance(31_000)

        async with redis_client.lock("stuck"):
            pass

    async def test_release_is_ownership_checked(
        self, redis_client: RedisClient, fake_redis: FakeRedis
    ) -> None:
        # If an expired holder released blindly it would delete the *next*
        # holder's lock. The Lua script compares the owner token first.
        async with redis_client.lock("shared", owner="owner-a"):
            await fake_redis.set("lock:shared", "owner-b")

        assert await fake_redis.get("lock:shared") == "owner-b"


class TestRedisErrorTranslation:
    async def test_backend_errors_become_database_errors(self) -> None:
        class BrokenRedis(FakeRedis):
            async def get(self, key: str) -> str | None:
                raise RedisError("connection lost")

        client = RedisClient(client=BrokenRedis())

        with pytest.raises(DatabaseError, match="redis GET failed"):
            await client.get_json("k")

    async def test_set_errors_become_database_errors(self) -> None:
        class BrokenRedis(FakeRedis):
            async def set(self, *args: object, **kwargs: object) -> bool:
                raise RedisError("write failed")

        client = RedisClient(client=BrokenRedis())

        with pytest.raises(DatabaseError, match="redis SET failed"):
            await client.set_json("k", 1)


class TestRedisHealth:
    async def test_reports_healthy_when_reachable(self, redis_client: RedisClient) -> None:
        health = await redis_client.health_check()

        assert health.name == "redis"
        assert health.status.is_serving is True
        assert health.required is True

    async def test_reports_unhealthy_without_raising(self) -> None:
        class DeadRedis(FakeRedis):
            async def ping(self) -> bool:
                raise RedisError("down")

        health = await RedisClient(client=DeadRedis()).health_check()

        assert health.status.is_serving is False
        assert "down" in (health.message or "")

    async def test_aclose_closes_the_client(
        self, redis_client: RedisClient, fake_redis: FakeRedis
    ) -> None:
        await redis_client.aclose()

        assert fake_redis.closed is True


class TestNeo4jReadGuard:
    """The read path must reject mutating Cypher.

    Read queries route to follower replicas in a cluster, where a mutation
    fails at runtime — but succeeds against a single-node dev instance, so the
    bug reaches production undetected without this guard.
    """

    @pytest.fixture
    def client(self) -> Neo4jClient:
        return Neo4jClient(driver=object())  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "query",
        [
            "CREATE (c:Company {ticker: $ticker})",
            "MATCH (c:Company) SET c.name = $name",
            "MERGE (c:Company {ticker: $ticker})",
            "MATCH (c:Company) DELETE c",
            "MATCH (c:Company) REMOVE c.stale",
            "DROP INDEX company_ticker",
        ],
    )
    def test_mutating_clauses_are_rejected_on_the_read_path(
        self, client: Neo4jClient, query: str
    ) -> None:
        with pytest.raises(ValidationError, match="mutating clause"):
            client._reject_writes(query)

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (c:Company) RETURN c",
            "MATCH (c:Company)-[:SUPPLIES]->(s) RETURN s.name",
            "MATCH (c:Company) WHERE c.ticker = $ticker RETURN c LIMIT 10",
        ],
    )
    def test_read_queries_pass_the_guard(self, client: Neo4jClient, query: str) -> None:
        client._reject_writes(query)

    def test_the_guard_is_case_insensitive(self, client: Neo4jClient) -> None:
        with pytest.raises(ValidationError):
            client._reject_writes("create (c:Company)")
