"""Base model configuration shared by every schema in the ecosystem."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fie_common.utils import new_id, utc_now


class FIEModel(BaseModel):
    """Strict base model.

    ``extra="forbid"`` is deliberate: an unexpected field in a payload crossing
    a service boundary is a contract violation, and silently dropping it is how
    integrations rot. ``validate_assignment`` keeps invariants true after
    construction, not just at parse time.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
        ser_json_timedelta="float",
    )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to primitives suitable for JSON transport or event payloads."""
        return self.model_dump(mode="json", exclude_none=True)


class FrozenModel(FIEModel):
    """Immutable variant, for value objects and event payloads."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=False,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class IdentifiedModel(FIEModel):
    """Model carrying a generated unique identifier."""

    id: str = Field(default_factory=lambda: new_id(), description="Unique identifier")


class TimestampedModel(FIEModel):
    """Model carrying creation and update timestamps in UTC."""

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        """Advance ``updated_at`` to now."""
        self.updated_at = utc_now()


class IdentifiedTimestampedModel(IdentifiedModel, TimestampedModel):
    """Common persistence shape: identity plus audit timestamps."""


__all__ = [
    "FIEModel",
    "FrozenModel",
    "IdentifiedModel",
    "IdentifiedTimestampedModel",
    "TimestampedModel",
]
