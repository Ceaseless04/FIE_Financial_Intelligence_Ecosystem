"""fie_events — domain events, the Redis Streams bus, and idempotent handling."""

from fie_events.bus import (
    EventBus,
    EventHandler,
    InMemoryEventBus,
    Subscription,
    dead_letter_stream,
    stream_for,
)
from fie_events.config import EventSettings
from fie_events.idempotency import IdempotencyStore, IdempotentHandler
from fie_events.redis_bus import RedisStreamEventBus
from fie_events.schemas import DomainEvent

__version__ = "0.1.0"

__all__ = [
    "DomainEvent",
    "EventBus",
    "EventHandler",
    "EventSettings",
    "IdempotencyStore",
    "IdempotentHandler",
    "InMemoryEventBus",
    "RedisStreamEventBus",
    "Subscription",
    "__version__",
    "dead_letter_stream",
    "stream_for",
]
