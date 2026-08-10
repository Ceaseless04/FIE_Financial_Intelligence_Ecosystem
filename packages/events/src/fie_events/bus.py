"""Event bus abstraction plus an in-memory implementation for tests.

Delivery is at-least-once. Handlers must therefore be idempotent — see
:mod:`fie_events.idempotency`, which the Redis bus applies automatically.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from fie_events.schemas import DomainEvent
from fie_observability.logging import get_logger

logger = get_logger(__name__)

EventHandler = Callable[[DomainEvent], Awaitable[None]]


def stream_for(event_type: str) -> str:
    """Map an event type to its stream.

    Streams are per-product rather than per-event-type: a consumer that cares
    about several of a product's events reads one stream and filters, instead
    of coordinating consumer groups across many streams.
    """
    return f"fie.events.{event_type.split('.', 1)[0]}"


def dead_letter_stream(stream: str) -> str:
    return f"{stream}.dlq"


@dataclass
class Subscription:
    """A consumer's interest in a stream."""

    stream: str
    group: str
    consumer: str
    handler: EventHandler
    #: Event-type prefixes this handler accepts; empty means all.
    event_types: Sequence[str] = field(default_factory=tuple)

    def accepts(self, event: DomainEvent) -> bool:
        if not self.event_types:
            return True
        return any(event.event_type.startswith(prefix) for prefix in self.event_types)


class EventBus(ABC):
    """Publish/subscribe contract every implementation satisfies."""

    @abstractmethod
    async def publish(self, event: DomainEvent) -> str:
        """Publish an event, returning the broker-assigned message id."""

    @abstractmethod
    async def subscribe(self, subscription: Subscription) -> None:
        """Register a handler. Delivery begins when ``run`` is called."""

    @abstractmethod
    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        """Consume until ``stop_event`` is set."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release broker resources."""


class InMemoryEventBus(EventBus):
    """In-process bus for unit tests and single-process development.

    Dispatch is synchronous within ``publish``, which makes assertions in tests
    deterministic — no polling for eventual delivery.
    """

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self.published: list[DomainEvent] = []
        self.dead_lettered: list[tuple[DomainEvent, str]] = []
        self._counter = 0

    async def publish(self, event: DomainEvent) -> str:
        self._counter += 1
        message_id = f"{self._counter}-0"
        self.published.append(event)

        target = stream_for(event.event_type)
        for subscription in self._subscriptions:
            if subscription.stream != target or not subscription.accepts(event):
                continue
            try:
                await subscription.handler(event)
            except Exception as error:  # noqa: BLE001 — one handler's failure
                # must not break the publisher or the other subscribers.
                self.dead_lettered.append((event, str(error)))
                logger.warning(
                    "event_handler_failed",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    group=subscription.group,
                    error=str(error),
                )
        return message_id

    async def subscribe(self, subscription: Subscription) -> None:
        self._subscriptions.append(subscription)

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        """No-op: delivery already happened inside ``publish``."""
        if stop_event is not None:
            await stop_event.wait()

    async def aclose(self) -> None:
        self._subscriptions.clear()

    def events_of_type(self, event_type: str) -> list[DomainEvent]:
        """Published events matching a type prefix. Test helper."""
        return [event for event in self.published if event.event_type.startswith(event_type)]

    def clear(self) -> None:
        self.published.clear()
        self.dead_lettered.clear()

    @property
    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for event in self.published:
            counts[event.event_type] += 1
        return dict(counts)


__all__ = [
    "EventBus",
    "EventHandler",
    "InMemoryEventBus",
    "Subscription",
    "dead_letter_stream",
    "stream_for",
]
