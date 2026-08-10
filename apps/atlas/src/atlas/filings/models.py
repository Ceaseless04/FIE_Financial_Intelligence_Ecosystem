"""Filings — the documents Atlas reads figures out of."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from atlas.domain.periods import FiscalPeriod
from fie_common.utils import content_hash, new_id, utc_now
from fie_schemas.base import FIEModel
from fie_schemas.provenance import SourceLocator, SourceReference, SourceType


class FilingType(StrEnum):
    """SEC form types Atlas understands."""

    ANNUAL_10K = "10-K"
    QUARTERLY_10Q = "10-Q"
    CURRENT_8K = "8-K"
    FOREIGN_ANNUAL_20F = "20-F"
    PROXY_DEF14A = "DEF 14A"
    EARNINGS_RELEASE = "earnings_release"

    @property
    def carries_full_statements(self) -> bool:
        """Whether this form normally contains complete financial statements.

        An 8-K may attach an earnings release with figures, but it is not
        required to, so Atlas does not treat a missing statement in one as an
        extraction failure.
        """
        return self in (
            FilingType.ANNUAL_10K,
            FilingType.QUARTERLY_10Q,
            FilingType.FOREIGN_ANNUAL_20F,
            FilingType.EARNINGS_RELEASE,
        )


class Filing(FIEModel):
    """A source document, retained so every extracted figure stays citable."""

    id: str = Field(default_factory=lambda: new_id("fil"))
    #: MarketMind entity id for the filer. Atlas does not resolve identity
    #: itself — that is the knowledge graph's job, and duplicating it here
    #: would let the two disagree about who a company is.
    entity_id: str = Field(min_length=1)
    type: FilingType
    period: FiscalPeriod
    title: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=1)
    accession_number: str | None = None
    filed_at: date | None = None
    uri: str | None = None
    ingested_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("filing content must not be blank")
        return value

    @property
    def content_hash(self) -> str:
        """Stable hash used to recognise a redelivered filing."""
        return content_hash(self.content)

    def to_source_reference(
        self, locator: SourceLocator | None = None, excerpt: str | None = None
    ) -> SourceReference:
        """A citable pointer to this filing, optionally to an exact span."""
        return SourceReference(
            source_id=self.id,
            source_type=SourceType.SEC_FILING,
            title=self.title,
            uri=self.uri,
            published_at=(
                datetime.combine(self.filed_at, datetime.min.time()) if self.filed_at else None
            ),
            locator=locator,
            excerpt=excerpt[:1000] if excerpt else None,
        )


__all__ = ["Filing", "FilingType"]
