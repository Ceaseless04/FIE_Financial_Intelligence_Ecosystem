"""Unit tests for the Redis Streams event bus and idempotent handling.

Run against the functional in-memory Redis double so the dead-letter and
reclaim paths — which need controllable delivery counts and idle time — are
deterministic. The same logic runs against real Redis in the integration suite.
"""

from __future__ import annotations

import pytest
from redis.exceptions import RedisError

from fie_common.errors import EventBusError
from fie_database.redis_client import RedisClient
from fie_events.bus import Subscription, dead_letter_stream
from fie_events.config import EventSettings
from fie_events.idempotency import IdempotencyStore, IdempotentHandler
from fie_events.redis_bus import RedisStreamEventBus
from fie_events.schemas import DomainEvent
from fie_testing.fake_redis import FakeRedis

pytestmark = pytest.mark.unit

STREAM = "fie.events.marketmind"
GROUP = "atlas-consumers"


def event(event_type: str = "marketmind.company.ingested", **overrides: object) -> DomainEvent:
    fields: dict[str, object] = {
        "event_type": event_type,
        "source_app": "marketmind",
        "payload": {"ticker": "ACME"},
        "correlation_id": "corr_1",
    }
    fields.update(overrides)
    return DomainEvent(**fields)  # type: ignore[arg-type]


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def settings() -> EventSettings:
    return EventSettings(block_ms=100, max_delivery_attempts=3, reclaim_idle_ms=1000)


@pytest.fixture
def bus(fake_redis: FakeRedis, settings: EventSettings) -> RedisStreamEventBus:
    return RedisStreamEventBus(RedisClient(client=fake_redis), settings=settings)


async def subscribe(bus: RedisStreamEventBus, handler: object, **overrides: object) -> Subscription:
    subscription = Subscription(
        stream=STREAM,
        group=GROUP,
        consumer="consumer-1",
        handler=handler,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )
    await bus.subscribe(subscription)
    return subscription


class TestPublishing:
    async def test_publishes_to_the_product_stream(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        message_id = await bus.publish(event())

        assert message_id
        assert len(fake_redis.stream_entries(STREAM)) == 1

    async def test_the_payload_is_the_wire_format(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        published = event()
        await bus.publish(published)

        _message_id, fields = fake_redis.stream_entries(STREAM)[0]
        assert fields["event_type"] == published.event_type
        assert fields["event_id"] == published.event_id

    async def test_publish_many(self, bus: RedisStreamEventBus, fake_redis: FakeRedis) -> None:
        ids = await bus.publish_many([event(), event()])

        assert len(ids) == 2
        assert len(fake_redis.stream_entries(STREAM)) == 2

    async def test_backend_failures_become_event_bus_errors(self, settings: EventSettings) -> None:
        class BrokenRedis(FakeRedis):
            async def xadd(self, *args: object, **kwargs: object) -> str:
                raise RedisError("stream unavailable")

        bus = RedisStreamEventBus(RedisClient(client=BrokenRedis()), settings=settings)

        with pytest.raises(EventBusError, match="failed to publish"):
            await bus.publish(event())


class TestSubscription:
    async def test_creates_the_consumer_group(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        await subscribe(bus, _collector([]))

        assert fake_redis.pending_count(STREAM, GROUP) == 0  # group exists, nothing pending

    async def test_subscribing_twice_tolerates_an_existing_group(
        self, bus: RedisStreamEventBus
    ) -> None:
        await subscribe(bus, _collector([]))
        await subscribe(bus, _collector([]))

    async def test_group_creation_failures_are_surfaced(self, settings: EventSettings) -> None:
        class BrokenRedis(FakeRedis):
            async def xgroup_create(self, *args: object, **kwargs: object) -> bool:
                raise RedisError("cannot create group")

        bus = RedisStreamEventBus(RedisClient(client=BrokenRedis()), settings=settings)

        with pytest.raises(EventBusError, match="failed to create consumer group"):
            await subscribe(bus, _collector([]))

    async def test_run_without_subscriptions_is_an_error(self, bus: RedisStreamEventBus) -> None:
        with pytest.raises(EventBusError, match="no subscriptions"):
            await bus.run()


def _collector(sink: list[DomainEvent]):  # type: ignore[no-untyped-def]
    async def handler(received: DomainEvent) -> None:
        sink.append(received)

    return handler


class TestConsumption:
    async def test_delivers_and_acknowledges(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        received: list[DomainEvent] = []
        subscription = await subscribe(bus, _collector(received))
        await bus.publish(event())

        handled = await bus._consume_new(subscription)

        assert handled == 1
        assert len(received) == 1
        assert fake_redis.pending_count(STREAM, GROUP) == 0, "a handled event must be acked"

    async def test_filters_by_event_type_and_acknowledges_the_rest(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        received: list[DomainEvent] = []
        subscription = await subscribe(
            bus, _collector(received), event_types=("marketmind.company",)
        )
        await bus.publish(event("marketmind.company.ingested"))
        await bus.publish(event("marketmind.news.linked"))

        await bus._consume_new(subscription)

        assert len(received) == 1
        assert fake_redis.pending_count(STREAM, GROUP) == 0, "filtered events are still acked"

    async def test_request_context_is_bound_from_the_event(self, bus: RedisStreamEventBus) -> None:
        from fie_observability.context import current_context

        seen: list[str] = []

        async def handler(_received: DomainEvent) -> None:
            seen.append(current_context().correlation_id)

        subscription = await subscribe(bus, handler)
        await bus.publish(event(correlation_id="corr_traced"))

        await bus._consume_new(subscription)

        assert seen == ["corr_traced"]

    async def test_a_failing_handler_leaves_the_message_unacknowledged(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        # Redelivery is the retry mechanism, so the message must stay pending.
        async def failing(_received: DomainEvent) -> None:
            raise RuntimeError("handler failed")

        subscription = await subscribe(bus, failing)
        await bus.publish(event())

        await bus._consume_new(subscription)

        assert fake_redis.pending_count(STREAM, GROUP) == 1

    async def test_an_undecodable_message_is_dead_lettered_immediately(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        # A malformed message can never succeed, so burning the retry budget on
        # it only delays the queue.
        received: list[DomainEvent] = []
        subscription = await subscribe(bus, _collector(received))
        await fake_redis.xadd(STREAM, {"event_type": "broken"})

        await bus._consume_new(subscription)

        assert len(fake_redis.stream_entries(dead_letter_stream(STREAM))) == 1
        assert fake_redis.pending_count(STREAM, GROUP) == 0
        assert received == []


class TestReclaimAndDeadLetter:
    async def test_a_stale_pending_message_is_reclaimed_and_retried(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis, settings: EventSettings
    ) -> None:
        attempts: list[int] = []

        async def flaky(_received: DomainEvent) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient failure")

        subscription = await subscribe(bus, flaky)
        await bus.publish(event())
        await bus._consume_new(subscription)
        assert fake_redis.pending_count(STREAM, GROUP) == 1

        # The consumer "crashes"; the message goes idle and another claims it.
        fake_redis.advance(settings.reclaim_idle_ms + 1)
        reclaimed = await bus._reclaim_stale(subscription)

        assert reclaimed == 1
        assert len(attempts) == 2
        assert fake_redis.pending_count(STREAM, GROUP) == 0

    async def test_a_poison_message_is_dead_lettered_after_the_attempt_budget(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis, settings: EventSettings
    ) -> None:
        # One permanently failing event must not block the partition forever.
        async def always_failing(_received: DomainEvent) -> None:
            raise RuntimeError("poison")

        subscription = await subscribe(bus, always_failing)
        await bus.publish(event())
        await bus._consume_new(subscription)

        message_id = fake_redis.stream_entries(STREAM)[0][0]
        fake_redis.set_delivery_count(STREAM, GROUP, message_id, settings.max_delivery_attempts + 1)
        fake_redis.advance(settings.reclaim_idle_ms + 1)

        await bus._reclaim_stale(subscription)

        dlq = fake_redis.stream_entries(dead_letter_stream(STREAM))
        assert len(dlq) == 1
        assert dlq[0][1]["dlq_reason"] == "max_attempts"
        assert fake_redis.pending_count(STREAM, GROUP) == 0, "dead-lettered events are acked"

    async def test_the_dead_letter_entry_preserves_the_original_payload(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis, settings: EventSettings
    ) -> None:
        async def always_failing(_received: DomainEvent) -> None:
            raise RuntimeError("poison")

        published = event()
        subscription = await subscribe(bus, always_failing)
        await bus.publish(published)
        await bus._consume_new(subscription)

        message_id = fake_redis.stream_entries(STREAM)[0][0]
        fake_redis.set_delivery_count(STREAM, GROUP, message_id, settings.max_delivery_attempts + 1)
        fake_redis.advance(settings.reclaim_idle_ms + 1)
        await bus._reclaim_stale(subscription)

        dlq_fields = fake_redis.stream_entries(dead_letter_stream(STREAM))[0][1]
        assert dlq_fields["event_id"] == published.event_id
        assert dlq_fields["dlq_group"] == GROUP

    async def test_dead_letter_depth_is_reportable(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        await fake_redis.xadd(dead_letter_stream(STREAM), {"a": "1"})

        assert await bus.dead_letter_depth(STREAM) == 1

    async def test_a_message_below_the_idle_threshold_is_not_reclaimed(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        async def failing(_received: DomainEvent) -> None:
            raise RuntimeError("failed")

        subscription = await subscribe(bus, failing)
        await bus.publish(event())
        await bus._consume_new(subscription)

        assert await bus._reclaim_stale(subscription) == 0


class TestIdempotency:
    async def test_an_event_is_handled_once_per_group(
        self, bus: RedisStreamEventBus, fake_redis: FakeRedis
    ) -> None:
        # At-least-once delivery means duplicates happen; double-processing a
        # financial event is not an acceptable outcome.
        received: list[DomainEvent] = []
        subscription = await subscribe(bus, _collector(received))
        published = event()
        await bus.publish(published)

        await bus._consume_new(subscription)
        # Simulate a redelivery of the same event.
        await fake_redis.xadd(STREAM, published.to_wire())
        await bus._consume_new(subscription)

        assert len(received) == 1, "the duplicate must be suppressed"

    async def test_different_groups_each_get_the_event(self, fake_redis: FakeRedis) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis))
        published = event()

        assert await store.claim("group-a", published.event_id) is True
        assert await store.claim("group-b", published.event_id) is True

    async def test_a_claim_blocks_a_second_claim(self, fake_redis: FakeRedis) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis))

        assert await store.claim("g", "evt_1") is True
        assert await store.claim("g", "evt_1") is False

    async def test_release_allows_a_retry(self, fake_redis: FakeRedis) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis))
        await store.claim("g", "evt_1")
        await store.release("g", "evt_1")

        assert await store.claim("g", "evt_1") is True

    async def test_was_processed_reports_claim_state(self, fake_redis: FakeRedis) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis))

        assert await store.was_processed("g", "evt_1") is False
        await store.claim("g", "evt_1")
        assert await store.was_processed("g", "evt_1") is True

    async def test_a_failed_handler_releases_its_claim_so_retry_can_work(
        self, fake_redis: FakeRedis
    ) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis))
        attempts: list[int] = []

        async def flaky(_received: DomainEvent) -> None:
            attempts.append(1)
            raise RuntimeError("failed")

        handler = IdempotentHandler(flaky, store=store, group="g")
        published = event()

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await handler(published)

        assert len(attempts) == 2, "a transient failure must remain retryable"

    async def test_a_successful_handler_keeps_its_claim(self, fake_redis: FakeRedis) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis))
        attempts: list[int] = []

        async def handler_fn(_received: DomainEvent) -> None:
            attempts.append(1)

        handler = IdempotentHandler(handler_fn, store=store, group="g")
        published = event()

        await handler(published)
        await handler(published)

        assert len(attempts) == 1

    async def test_claims_expire_so_the_store_does_not_grow_forever(
        self, fake_redis: FakeRedis
    ) -> None:
        store = IdempotencyStore(RedisClient(client=fake_redis), retention_seconds=60)
        await store.claim("g", "evt_1")

        fake_redis.advance(61_000)

        assert await store.claim("g", "evt_1") is True
