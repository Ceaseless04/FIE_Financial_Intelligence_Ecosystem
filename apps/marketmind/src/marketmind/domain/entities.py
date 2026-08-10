"""Knowledge-graph entity types.

MarketMind's scope is deliberately narrow: entities, relationships, knowledge,
retrieval, and graph reasoning. It records *what is true about the world* — that
a company exists, who runs it, who supplies it. It does not evaluate, score, or
recommend. A field like "investment rating" would belong to Atlas or Venture,
never here.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from fie_common.utils import new_id, utc_now
from fie_schemas.base import FIEModel, FrozenModel
from fie_schemas.provenance import Provenance, SourceReference


class EntityType(StrEnum):
    """Node labels in the graph."""

    COMPANY = "Company"
    EXECUTIVE = "Executive"
    INDUSTRY = "Industry"
    PRODUCT = "Product"
    GEOGRAPHY = "Geography"
    MACRO_INDICATOR = "MacroIndicator"
    NEWS_EVENT = "NewsEvent"


class IdentifierType(StrEnum):
    """Authoritative identifiers, in descending order of trust.

    Resolution treats a match on any of these as decisive: two records sharing
    a CIK are the same company regardless of how differently they spell the
    name.
    """

    CIK = "cik"
    TICKER = "ticker"
    LEI = "lei"
    ISIN = "isin"
    CUSIP = "cusip"
    DOMAIN = "domain"


_IDENTIFIER_PATTERNS: dict[IdentifierType, re.Pattern[str]] = {
    IdentifierType.CIK: re.compile(r"^\d{1,10}$"),
    IdentifierType.TICKER: re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$"),
    IdentifierType.LEI: re.compile(r"^[A-Z0-9]{20}$"),
    IdentifierType.ISIN: re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$"),
    IdentifierType.CUSIP: re.compile(r"^[A-Z0-9]{9}$"),
    IdentifierType.DOMAIN: re.compile(r"^[a-z0-9.\-]+\.[a-z]{2,}$"),
}


class Identifier(FrozenModel):
    """An authoritative identifier attached to an entity."""

    type: IdentifierType
    value: str = Field(min_length=1)

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> Identifier:
        normalized = self.value.strip()
        if self.type in (IdentifierType.DOMAIN,):
            normalized = normalized.lower()
        elif self.type is IdentifierType.CIK:
            # CIKs are commonly zero-padded to 10 digits in some feeds and not
            # in others; strip padding so the two forms compare equal.
            normalized = normalized.lstrip("0") or "0"
        else:
            normalized = normalized.upper()

        pattern = _IDENTIFIER_PATTERNS[self.type]
        if not pattern.match(normalized):
            raise ValueError(f"{self.value!r} is not a valid {self.type}")

        object.__setattr__(self, "value", normalized)
        return self

    @property
    def key(self) -> str:
        """Stable composite key, used for graph uniqueness constraints."""
        return f"{self.type}:{self.value}"


class Entity(FIEModel):
    """A node in the knowledge graph.

    Every entity carries provenance. An entity without a source is an entity
    nobody can defend in a research report.
    """

    id: str = Field(default_factory=lambda: new_id("ent"))
    type: EntityType
    name: str = Field(min_length=1, max_length=512)
    #: Normalized form used for blocking and matching. Set by the resolver.
    canonical_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    identifiers: list[Identifier] = Field(default_factory=list)
    description: str | None = Field(default=None, max_length=4000)
    attributes: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    #: Set once the entity has been resolved and written to the graph.
    resolved: bool = False
    created_at: Any = Field(default_factory=utc_now)
    updated_at: Any = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity name must not be blank")
        return value.strip()

    @field_validator("aliases")
    @classmethod
    def _dedupe_aliases(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for alias in value:
            cleaned = alias.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        return list(seen)

    @model_validator(mode="after")
    def _reject_evaluative_attributes(self) -> Entity:
        """Keep judgement out of the knowledge layer.

        MarketMind must not make investment recommendations. Attributes like
        "rating" or "price_target" are how that boundary erodes in practice —
        someone stores a score on a node "just for now" and three products
        start reading it as fact. Rejecting the attribute name is a cheap,
        durable guard.
        """
        forbidden = {
            "rating",
            "recommendation",
            "price_target",
            "buy_sell_hold",
            "investment_score",
            "fair_value",
            "valuation",
            "portfolio_weight",
        }
        offending = forbidden & {key.lower() for key in self.attributes}
        if offending:
            raise ValueError(
                "MarketMind stores knowledge, not judgements; "
                f"disallowed attribute(s): {sorted(offending)}. "
                "Valuation and recommendations belong to Atlas or Venture."
            )
        return self

    def identifier_of(self, identifier_type: IdentifierType) -> str | None:
        for identifier in self.identifiers:
            if identifier.type is identifier_type:
                return identifier.value
        return None

    @property
    def identifier_keys(self) -> set[str]:
        return {identifier.key for identifier in self.identifiers}

    @property
    def sources(self) -> list[SourceReference]:
        return list(self.provenance.sources)

    def merged_with(self, other: Entity) -> Entity:
        """Combine a duplicate into this entity.

        Union of aliases, identifiers, and sources; attributes from ``other``
        fill gaps but never overwrite, so the first-observed value wins and
        ingestion order cannot silently rewrite established facts.
        """
        if other.type is not self.type:
            raise ValueError("cannot merge entities of different types")

        aliases = list(dict.fromkeys([*self.aliases, other.name, *other.aliases]))
        aliases = [alias for alias in aliases if alias != self.name]

        identifiers = list(self.identifiers)
        existing_keys = self.identifier_keys
        for identifier in other.identifiers:
            if identifier.key not in existing_keys:
                identifiers.append(identifier)
                existing_keys.add(identifier.key)

        attributes = {**other.attributes, **self.attributes}

        sources = list(self.provenance.sources)
        seen_source_ids = {source.source_id for source in sources}
        for source in other.provenance.sources:
            if source.source_id not in seen_source_ids:
                sources.append(source)
                seen_source_ids.add(source.source_id)

        return self.model_copy(
            update={
                "aliases": aliases,
                "identifiers": identifiers,
                "attributes": attributes,
                "description": self.description or other.description,
                "provenance": self.provenance.model_copy(update={"sources": sources}),
                "updated_at": utc_now(),
            }
        )


class Company(FIEModel):
    """Typed view over a ``Company`` entity's attributes.

    The graph stores attributes loosely so ingestion never fails on an
    unexpected field; this gives callers a validated read model.
    """

    entity_id: str
    name: str
    ticker: str | None = None
    cik: str | None = None
    sector: str | None = None
    headquarters: str | None = None
    founded: date | None = None
    employee_count: int | None = Field(default=None, ge=0)

    @classmethod
    def from_entity(cls, entity: Entity) -> Company:
        if entity.type is not EntityType.COMPANY:
            raise ValueError(f"expected a Company entity, got {entity.type}")
        attributes = entity.attributes
        return cls(
            entity_id=entity.id,
            name=entity.name,
            ticker=entity.identifier_of(IdentifierType.TICKER),
            cik=entity.identifier_of(IdentifierType.CIK),
            sector=attributes.get("sector"),
            headquarters=attributes.get("headquarters"),
            founded=attributes.get("founded"),
            employee_count=attributes.get("employee_count"),
        )


__all__ = [
    "Company",
    "Entity",
    "EntityType",
    "Identifier",
    "IdentifierType",
]
