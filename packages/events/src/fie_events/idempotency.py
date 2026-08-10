"""Idempotent event processing.

At-least-once delivery means a handler will occasionally see the same event
twice — after a consumer crash, a redelivery, or a retry. For financial
workloads that is not a tolerable duplicate: re-processing a "filing ingested"
event twice can double-count a position.

The guard is a Redis ``SET NX`` on ``(consumer_group, event_id)``. Claiming
happens *before* the handler runs, and the claim is released if the handler
fails, so a transient failure still gets retried while a successful one is
never repeated.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fie_common.utils import utc_now
from fie_database.redis_client import RedisClient
from fie_events.schemas import DomainEvent
from fie_observability.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RETENTION_SECONDS = 24 * 60 * 60


@dataclass
class IdempotencyStore:
    """Tracks which ``(group, event_id)`` pairs have been processed."""

    redis: RedisClient
    retention_seconds: int = DEFAULT_RETENTION_SECONDS
    key_prefix: str = "idem"

    def key(self, group: str, event_id: str) -> str:
        return f"{self.key_prefix}:{group}:{event_id}"

    async def claim(self, group: str, event_id: str) -> bool:
        """Attempt to claim an event for processing.

        Returns:
            True if this caller won the claim and should process the event;
            False if it was already claimed and should be skipped.
        """
        return await self.redis.set_if_absent(
            self.key(group, event_id),
            utc_now().isoformat(),
            ttl_seconds=self.retention_seconds,
        )

    async def release(self, group: str, event_id: str) -> None:
        """Release a claim so a failed event can be retried."""
        await self.redis.delete(self.key(group, event_id))

    async def was_processed(self, group: str, event_id: str) -> bool:
        return await self.redis.exists(self.key(group, event_id))


class IdempotentHandler:
    """Wraps a handler so each event is processed at most once per group."""

    def __init__(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        *,
        store: IdempotencyStore,
        group: str,
    ) -> None:
        self._handler = handler
        self._store = store
        self._group = group

    async def __call__(self, event: DomainEvent) -> None:
        if not await self._store.claim(self._group, event.event_id):
            logger.info(
                "event_skipped_duplicate",
                event_id=event.event_id,
                event_type=event.event_type,
                group=self._group,
            )
            return
        try:
            await self._handler(event)
        except BaseException:
            # Release so a redelivery can retry; a permanently failing event is
            # dead-lettered by the bus after its retry budget is exhausted.
            await self._store.release(self._group, event.event_id)
            raise


__all__ = [
    "DEFAULT_RETENTION_SECONDS",
    "IdempotencyStore",
    "IdempotentHandler",
]
