"""Redis Streams event bus.

Consumer groups give at-least-once delivery with per-consumer acknowledgement.
Three failure modes are handled explicitly:

* **Handler failure** — the message is left unacknowledged and redelivered.
* **Consumer crash** — pending messages idle past ``reclaim_idle_ms`` are
  claimed by another consumer, so work is never stranded.
* **Poison message** — once ``max_delivery_attempts`` is exceeded the event is
  moved to a dead-letter stream and acknowledged, so one bad event cannot block
  the partition forever.
"""

from __future__ import annotations

import asyncio
from typing import Any

from redis.exceptions import RedisError, ResponseError

from fie_common.errors import EventBusError, ValidationError
from fie_common.resilience import with_timeout
from fie_database.redis_client import RedisClient
from fie_events.bus import EventBus, Subscription, dead_letter_stream, stream_for
from fie_events.config import EventSettings
from fie_events.idempotency import IdempotencyStore, IdempotentHandler
from fie_events.schemas import DomainEvent
from fie_observability.context import request_context
from fie_observability.logging import get_logger
from fie_observability.metrics import PlatformMetrics, get_metrics
from fie_observability.tracing import traced

logger = get_logger(__name__)


class RedisStreamEventBus(EventBus):
    """Production event bus backed by Redis Streams."""

    def __init__(
        self,
        redis: RedisClient,
        *,
        settings: EventSettings | None = None,
        idempotency: IdempotencyStore | None = None,
        metrics: PlatformMetrics | None = None,
    ) -> None:
        self._redis = redis
        self._settings = settings or EventSettings()
        self._idempotency = idempotency or IdempotencyStore(
            redis, retention_seconds=self._settings.idempotency_retention_seconds
        )
        self._metrics = metrics or get_metrics()
        self._subscriptions: list[Subscription] = []

    @property
    def client(self) -> Any:
        return self._redis.client

    # -- publishing ----------------------------------------------------------

    async def publish(self, event: DomainEvent) -> str:
        """Append an event to its product stream."""
        stream = stream_for(event.event_type)
        try:
            message_id: str = await self.client.xadd(
                stream,
                event.to_wire(),
                maxlen=self._settings.max_stream_length,
                approximate=True,
            )
        except RedisError as error:
            raise EventBusError(
                f"failed to publish {event.event_type}: {error}",
                details={"stream": stream, "event_id": event.event_id},
            ) from error

        self._metrics.record_event_published(event_type=event.event_type, stream=stream)
        logger.info(
            "event_published",
            event_type=event.event_type,
            event_id=event.event_id,
            stream=stream,
            message_id=message_id,
        )
        return message_id

    async def publish_many(self, events: list[DomainEvent]) -> list[str]:
        return [await self.publish(event) for event in events]

    # -- subscribing ---------------------------------------------------------

    async def subscribe(self, subscription: Subscription) -> None:
        """Register a handler and ensure its consumer group exists."""
        await self._ensure_group(subscription.stream, subscription.group)
        self._subscriptions.append(subscription)
        logger.info(
            "event_subscription_registered",
            stream=subscription.stream,
            group=subscription.group,
            consumer=subscription.consumer,
        )

    async def _ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise EventBusError(
                    f"failed to create consumer group {group!r} on {stream!r}: {error}"
                ) from error
        except RedisError as error:
            raise EventBusError(f"failed to create consumer group: {error}") from error

    # -- consuming -----------------------------------------------------------

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        """Consume registered subscriptions until ``stop_event`` is set."""
        if not self._subscriptions:
            raise EventBusError("no subscriptions registered")
        stop = stop_event or asyncio.Event()
        while not stop.is_set():
            for subscription in self._subscriptions:
                if stop.is_set():
                    break
                await self._reclaim_stale(subscription)
                await self._consume_new(subscription)

    async def _consume_new(self, subscription: Subscription) -> int:
        """Read and dispatch undelivered messages. Returns the count handled."""
        try:
            response = await self.client.xreadgroup(
                subscription.group,
                subscription.consumer,
                {subscription.stream: ">"},
                count=self._settings.batch_size,
                block=self._settings.block_ms,
            )
        except RedisError as error:
            raise EventBusError(f"failed to read from {subscription.stream!r}: {error}") from error

        handled = 0
        for _stream, messages in response or []:
            for message_id, fields in messages:
                await self._dispatch(subscription, message_id, fields)
                handled += 1
        return handled

    async def _reclaim_stale(self, subscription: Subscription) -> int:
        """Re-drive or dead-letter messages a dead consumer left pending."""
        try:
            pending = await self.client.xpending_range(
                subscription.stream,
                subscription.group,
                min="-",
                max="+",
                count=self._settings.batch_size,
                idle=self._settings.reclaim_idle_ms,
            )
        except RedisError as error:
            raise EventBusError(f"failed to inspect pending messages: {error}") from error

        reclaimed = 0
        for entry in pending or []:
            message_id = entry["message_id"]
            deliveries = int(entry.get("times_delivered", 1))

            if deliveries > self._settings.max_delivery_attempts:
                await self._dead_letter_by_id(subscription, message_id, reason="max_attempts")
                continue

            claimed = await self.client.xclaim(
                subscription.stream,
                subscription.group,
                subscription.consumer,
                min_idle_time=self._settings.reclaim_idle_ms,
                message_ids=[message_id],
            )
            for claimed_id, fields in claimed or []:
                await self._dispatch(subscription, claimed_id, fields)
                reclaimed += 1
        return reclaimed

    async def _dispatch(
        self, subscription: Subscription, message_id: str, fields: dict[str, Any]
    ) -> None:
        """Decode, filter, and hand one message to its handler."""
        try:
            event = DomainEvent.from_wire(fields)
        except ValidationError as error:
            # An undecodable message can never succeed; dead-letter immediately
            # rather than burning the full retry budget on it.
            logger.error("event_decode_failed", stream=subscription.stream, message_id=message_id)
            await self._dead_letter_raw(subscription, message_id, fields, reason=str(error))
            return

        if not subscription.accepts(event):
            await self._ack(subscription, message_id)
            return

        handler = IdempotentHandler(
            subscription.handler, store=self._idempotency, group=subscription.group
        )

        with request_context(
            correlation_id=event.correlation_id,
            tenant_id=event.tenant_id,
            source_app=event.source_app,
        ):
            with traced(
                f"event.handle.{event.event_type}",
                attributes={
                    "event.type": event.event_type,
                    "event.id": event.event_id,
                    "event.stream": subscription.stream,
                    "event.group": subscription.group,
                },
            ):
                try:
                    await with_timeout(
                        lambda: handler(event),
                        seconds=self._settings.handler_timeout_seconds,
                        description=f"handler for {event.event_type}",
                    )
                except Exception as error:
                    self._metrics.record_event_consumed(
                        event_type=event.event_type,
                        stream=subscription.stream,
                        status="error",
                    )
                    logger.warning(
                        "event_handler_failed",
                        event_type=event.event_type,
                        event_id=event.event_id,
                        group=subscription.group,
                        error=str(error),
                    )
                    # Deliberately not acknowledged: redelivery is the retry.
                    return

        await self._ack(subscription, message_id)
        self._metrics.record_event_consumed(
            event_type=event.event_type, stream=subscription.stream, status="success"
        )

    async def _ack(self, subscription: Subscription, message_id: str) -> None:
        try:
            await self.client.xack(subscription.stream, subscription.group, message_id)
        except RedisError as error:
            raise EventBusError(f"failed to acknowledge {message_id}: {error}") from error

    async def _dead_letter_raw(
        self,
        subscription: Subscription,
        message_id: str,
        fields: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        """Move a message to the dead-letter stream and acknowledge it."""
        dlq = dead_letter_stream(subscription.stream)
        payload = {str(key): str(value) for key, value in fields.items()}
        payload["dlq_reason"] = reason
        payload["dlq_group"] = subscription.group
        payload["dlq_original_id"] = message_id
        try:
            await self.client.xadd(dlq, payload, maxlen=self._settings.max_stream_length)
        except RedisError as error:
            raise EventBusError(f"failed to dead-letter {message_id}: {error}") from error

        await self._ack(subscription, message_id)
        logger.error(
            "event_dead_lettered",
            stream=subscription.stream,
            dlq=dlq,
            message_id=message_id,
            reason=reason,
        )

    async def _dead_letter_by_id(
        self, subscription: Subscription, message_id: str, *, reason: str
    ) -> None:
        entries = await self.client.xrange(subscription.stream, min=message_id, max=message_id)
        fields = entries[0][1] if entries else {}
        await self._dead_letter_raw(subscription, message_id, fields, reason=reason)

    async def dead_letter_depth(self, stream: str) -> int:
        """Number of messages parked in a stream's dead-letter queue."""
        try:
            depth: int = await self.client.xlen(dead_letter_stream(stream))
            return depth
        except RedisError as error:
            raise EventBusError(f"failed to read DLQ depth: {error}") from error

    async def aclose(self) -> None:
        self._subscriptions.clear()


__all__ = ["RedisStreamEventBus"]
