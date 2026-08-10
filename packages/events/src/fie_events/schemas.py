"""Domain event envelope shared by every producer and consumer."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from fie_common.errors import ValidationError
from fie_common.utils import new_id, utc_now
from fie_schemas.base import FrozenModel


class DomainEvent(FrozenModel):
    """An immutable fact that already happened.

    ``event_type`` is namespaced ``product.entity.action`` (for example
    ``marketmind.company.ingested``) so consumers can subscribe to a product,
    an entity, or a single action by prefix.
    """

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    event_type: str = Field(min_length=3)
    source_app: str = Field(min_length=1)
    occurred_at: datetime = Field(default_factory=utc_now)
    #: Ties the event to the request that produced it, across services.
    correlation_id: str | None = None
    tenant_id: str | None = None
    #: Schema version of ``payload``; consumers must tolerate older versions.
    version: int = Field(default=1, ge=1)
    #: Set when this event was emitted while handling another one.
    causation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def _validate_event_type(cls, value: str) -> str:
        parts = value.split(".")
        if len(parts) < 2 or not all(part and part.replace("_", "").isalnum() for part in parts):
            raise ValueError("event_type must be dot-delimited, e.g. 'marketmind.company.ingested'")
        return value

    @property
    def product(self) -> str:
        """Leading segment of ``event_type`` — the owning product."""
        return self.event_type.split(".", 1)[0]

    def to_wire(self) -> dict[str, str]:
        """Flatten to the string field map a Redis stream entry requires."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_app": self.source_app,
            "occurred_at": self.occurred_at.isoformat(),
            "correlation_id": self.correlation_id or "",
            "causation_id": self.causation_id or "",
            "tenant_id": self.tenant_id or "",
            "version": str(self.version),
            "payload": json.dumps(self.payload, default=str),
        }

    @classmethod
    def from_wire(cls, fields: dict[str, Any]) -> DomainEvent:
        """Rebuild an event from a Redis stream entry.

        Raises:
            ValidationError: if the entry is malformed. Callers route these to
                the dead-letter stream rather than crashing the consumer loop.
        """
        try:
            raw_payload = fields.get("payload") or "{}"
            return cls(
                event_id=fields["event_id"],
                event_type=fields["event_type"],
                source_app=fields["source_app"],
                occurred_at=datetime.fromisoformat(fields["occurred_at"]),
                correlation_id=fields.get("correlation_id") or None,
                causation_id=fields.get("causation_id") or None,
                tenant_id=fields.get("tenant_id") or None,
                version=int(fields.get("version", 1)),
                payload=json.loads(raw_payload),
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValidationError(
                f"malformed event on the wire: {error}",
                details={"fields": sorted(fields)},
            ) from error

    def caused(self, event_type: str, payload: dict[str, Any], *, source_app: str) -> DomainEvent:
        """Derive a follow-on event that inherits this one's correlation chain."""
        return DomainEvent(
            event_type=event_type,
            source_app=source_app,
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
            tenant_id=self.tenant_id,
            payload=payload,
        )


__all__ = ["DomainEvent"]
