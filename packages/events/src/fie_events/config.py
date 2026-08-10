"""Event bus configuration (``FIE_EVENTS_*``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from fie_common.config import FIEBaseSettings


class EventSettings(FIEBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FIE_EVENTS_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    #: Messages read per XREADGROUP call.
    batch_size: int = Field(default=32, ge=1, le=1000)
    #: How long a read blocks waiting for new messages.
    block_ms: int = Field(default=2000, ge=100, le=60_000)
    #: Deliveries attempted before an event is dead-lettered.
    max_delivery_attempts: int = Field(default=5, ge=1, le=50)
    #: A pending message idle this long is reclaimed from its consumer. The
    #: default is deliberately conservative — reclaiming aggressively steals
    #: work from consumers that are merely slow. The floor only rules out
    #: pathological values; low-latency deployments may legitimately tune down.
    reclaim_idle_ms: int = Field(default=60_000, ge=100)
    #: Approximate cap on stream length, trimmed on publish.
    max_stream_length: int = Field(default=100_000, ge=1000)
    #: Suppress duplicate handling of the same event within this window.
    idempotency_retention_seconds: int = Field(default=86_400, ge=60)
    #: Handler wall-clock budget; exceeding it counts as a failed delivery.
    handler_timeout_seconds: float = Field(default=60.0, gt=0)


__all__ = ["EventSettings"]
