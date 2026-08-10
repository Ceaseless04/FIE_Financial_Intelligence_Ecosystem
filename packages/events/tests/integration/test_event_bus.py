"""Integration tests for the event bus against real Redis Streams.

Covers the guarantees that only real Redis can prove: consumer-group
semantics, pending-entry tracking, cross-consumer reclaim, and dead-lettering.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from fie_database.config import RedisSettings
from fie_database.redis_client import RedisClient
from fie_events.bus import Subscription, dead_letter_stream, stream_for
from fie_events.config import EventSettings
from fie_events.redis_bus import RedisStreamEventBus
from fie_events.schemas import DomainEvent
from fie_testing.infra import redis_endpoint, require_service

pytestmark = [pytest.mark.integration]

STREAM = stream_for("marketmind.company.ingested")
GROUP = "integration-consumers"


def event(event_type: str = "marketmind.company.ingested", **overrides: object) -> DomainEvent:
    fields: dict[str, object] = {
        "event_type": event_type,
        "source_app": "marketmind",
        "payload": {"ticker": "ACME"},
        "correlation_id": "corr_integration",
    }
    fields.update(overrides)
    return DomainEvent(**fields)  # type: ignore[arg-type]


@pytest.fixture
async def bus() -> AsyncIterator[RedisStreamEventBus]:
    require_service(redis_endpoint())
    redis = RedisClient(RedisSettings())
    await redis.client.delete(STREAM, dead_letter_stream(STREAM))
    # Clear idempotency keys left by an earlier run.
    async for key in redis.client.scan_iter(match="idem:*"):
        await redis.client.delete(key)

    instance = RedisStreamEventBus(
        redis,
        settings=EventSettings(
            block_ms=100, max_delivery_attempts=2, reclaim_idle_ms=200, batch_size=16
        ),
    )
    yield instance
    await redis.client.delete(STREAM, dead_letter_stream(STREAM))
    await redis.aclose()


class TestPublishConsume:
    async def test_published_events_are_delivered(self, bus: RedisStreamEventBus) -> None:
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        await bus.subscribe(subscription)
        published = await bus.publish(event())

        assert published
        await bus._consume_new(subscription)

        assert len(received) == 1
        assert received[0].payload == {"ticker": "ACME"}

    async def test_ordering_is_preserved_within_a_stream(self, bus: RedisStreamEventBus) -> None:
        received: list[str] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(str(received_event.payload["seq"]))

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        await bus.subscribe(subscription)
        for index in range(5):
            await bus.publish(event(payload={"seq": index}))

        await bus._consume_new(subscription)

        assert received == ["0", "1", "2", "3", "4"]

    async def test_correlation_id_survives_the_round_trip(self, bus: RedisStreamEventBus) -> None:
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        await bus.subscribe(subscription)
        await bus.publish(event(correlation_id="corr_traced"))
        await bus._consume_new(subscription)

        assert received[0].correlation_id == "corr_traced"

    async def test_a_handled_event_is_acknowledged(self, bus: RedisStreamEventBus) -> None:
        async def handler(_event: DomainEvent) -> None:
            return None

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        await bus.subscribe(subscription)
        await bus.publish(event())
        await bus._consume_new(subscription)

        pending = await bus.client.xpending(STREAM, GROUP)
        assert pending["pending"] == 0


class TestFailureHandling:
    async def test_a_failed_handler_leaves_the_message_pending(
        self, bus: RedisStreamEventBus
    ) -> None:
        async def failing(_event: DomainEvent) -> None:
            raise RuntimeError("handler failed")

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=failing)
        await bus.subscribe(subscription)
        await bus.publish(event())
        await bus._consume_new(subscription)

        pending = await bus.client.xpending(STREAM, GROUP)
        assert pending["pending"] == 1

    async def test_another_consumer_reclaims_stranded_work(self, bus: RedisStreamEventBus) -> None:
        # This is the crashed-consumer path: work must not be stranded.
        attempts: list[str] = []

        async def flaky(_event: DomainEvent) -> None:
            attempts.append("x")
            if len(attempts) == 1:
                raise RuntimeError("first consumer died")

        first = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=flaky)
        await bus.subscribe(first)
        await bus.publish(event())
        await bus._consume_new(first)

        await asyncio.sleep(0.3)  # exceed reclaim_idle_ms

        second = Subscription(stream=STREAM, group=GROUP, consumer="c2", handler=flaky)
        reclaimed = await bus._reclaim_stale(second)

        assert reclaimed == 1
        assert len(attempts) == 2

    async def test_a_poison_message_is_dead_lettered(self, bus: RedisStreamEventBus) -> None:
        async def always_failing(_event: DomainEvent) -> None:
            raise RuntimeError("poison")

        subscription = Subscription(
            stream=STREAM, group=GROUP, consumer="c1", handler=always_failing
        )
        await bus.subscribe(subscription)
        await bus.publish(event())

        # Exhaust the delivery budget (max_delivery_attempts=2).
        await bus._consume_new(subscription)
        for _ in range(4):
            await asyncio.sleep(0.25)
            await bus._reclaim_stale(subscription)

        assert await bus.dead_letter_depth(STREAM) == 1
        pending = await bus.client.xpending(STREAM, GROUP)
        assert pending["pending"] == 0, "a dead-lettered event must be acknowledged"

    async def test_an_undecodable_message_is_dead_lettered(self, bus: RedisStreamEventBus) -> None:
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        await bus.subscribe(subscription)
        await bus.client.xadd(STREAM, {"event_type": "garbage"})

        await bus._consume_new(subscription)

        assert await bus.dead_letter_depth(STREAM) == 1
        assert received == []


class TestIdempotentDelivery:
    async def test_a_redelivered_event_is_processed_once(self, bus: RedisStreamEventBus) -> None:
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        subscription = Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        await bus.subscribe(subscription)

        published = event()
        await bus.publish(published)
        await bus._consume_new(subscription)
        # Publish the identical event again, as a broker redelivery would.
        await bus.client.xadd(STREAM, published.to_wire())
        await bus._consume_new(subscription)

        assert len(received) == 1

    async def test_independent_groups_each_receive_the_event(
        self, bus: RedisStreamEventBus
    ) -> None:
        first_received: list[DomainEvent] = []
        second_received: list[DomainEvent] = []

        async def first_handler(received_event: DomainEvent) -> None:
            first_received.append(received_event)

        async def second_handler(received_event: DomainEvent) -> None:
            second_received.append(received_event)

        first = Subscription(
            stream=STREAM, group="group-atlas", consumer="c1", handler=first_handler
        )
        second = Subscription(
            stream=STREAM, group="group-sentinel", consumer="c1", handler=second_handler
        )
        await bus.subscribe(first)
        await bus.subscribe(second)
        await bus.publish(event())

        await bus._consume_new(first)
        await bus._consume_new(second)

        assert len(first_received) == 1
        assert len(second_received) == 1


class TestRunLoop:
    async def test_run_consumes_until_stopped(self, bus: RedisStreamEventBus) -> None:
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        await bus.subscribe(
            Subscription(stream=STREAM, group=GROUP, consumer="c1", handler=handler)
        )
        await bus.publish(event())

        stop = asyncio.Event()
        task = asyncio.create_task(bus.run(stop_event=stop))
        await asyncio.sleep(0.5)
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)

        assert len(received) == 1
