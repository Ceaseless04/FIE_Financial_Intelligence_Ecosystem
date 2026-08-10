"""Knowledge-graph relationship types.

Relationships are directed and typed. Each carries provenance and an optional
validity window, because most facts in this domain are true *for a period* — an
executive leads a company until they don't, a supplier relationship ends. A
graph that only stores the present tense silently answers historical questions
wrong.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from fie_common.utils import new_id, utc_now
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance
from marketmind.domain.entities import EntityType


class RelationshipType(StrEnum):
    """Edge types in the graph."""

    # Corporate structure and people
    EXECUTIVE_OF = "EXECUTIVE_OF"
    BOARD_MEMBER_OF = "BOARD_MEMBER_OF"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"

    # Commercial
    SUPPLIES = "SUPPLIES"
    CUSTOMER_OF = "CUSTOMER_OF"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"

    # Classification
    OPERATES_IN = "OPERATES_IN"
    PRODUCES = "PRODUCES"
    HEADQUARTERED_IN = "HEADQUARTERED_IN"

    # Financial (structural facts, not evaluations)
    ACQUIRED = "ACQUIRED"
    INVESTED_IN = "INVESTED_IN"

    # Context
    MENTIONED_IN = "MENTIONED_IN"
    EXPOSED_TO = "EXPOSED_TO"


#: Which entity types each relationship may connect. Rejecting a nonsensical
#: edge at write time is what keeps the graph queryable — an extractor that
#: hallucinates `Industry SUPPLIES Executive` corrupts every downstream
#: traversal, and it is far cheaper to catch here than to clean up later.
RELATIONSHIP_SCHEMA: dict[RelationshipType, tuple[set[EntityType], set[EntityType]]] = {
    RelationshipType.EXECUTIVE_OF: ({EntityType.EXECUTIVE}, {EntityType.COMPANY}),
    RelationshipType.BOARD_MEMBER_OF: ({EntityType.EXECUTIVE}, {EntityType.COMPANY}),
    RelationshipType.SUBSIDIARY_OF: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.SUPPLIES: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.CUSTOMER_OF: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.COMPETES_WITH: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.PARTNERS_WITH: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.OPERATES_IN: ({EntityType.COMPANY}, {EntityType.INDUSTRY}),
    RelationshipType.PRODUCES: ({EntityType.COMPANY}, {EntityType.PRODUCT}),
    RelationshipType.HEADQUARTERED_IN: ({EntityType.COMPANY}, {EntityType.GEOGRAPHY}),
    RelationshipType.ACQUIRED: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.INVESTED_IN: ({EntityType.COMPANY}, {EntityType.COMPANY}),
    RelationshipType.MENTIONED_IN: (
        {EntityType.COMPANY, EntityType.EXECUTIVE, EntityType.PRODUCT},
        {EntityType.NEWS_EVENT},
    ),
    RelationshipType.EXPOSED_TO: (
        {EntityType.COMPANY, EntityType.INDUSTRY},
        {EntityType.MACRO_INDICATOR, EntityType.GEOGRAPHY},
    ),
}

#: Relationships where direction carries no meaning. Stored one way and
#: traversed both, so "who competes with X" cannot depend on ingestion order.
SYMMETRIC_RELATIONSHIPS: frozenset[RelationshipType] = frozenset(
    {RelationshipType.COMPETES_WITH, RelationshipType.PARTNERS_WITH}
)


class Relationship(FIEModel):
    """A directed, typed, sourced edge between two entities."""

    id: str = Field(default_factory=lambda: new_id("rel"))
    type: RelationshipType
    source_entity_id: str = Field(min_length=1)
    target_entity_id: str = Field(min_length=1)
    #: Entity types of the endpoints, kept so the edge can be schema-checked
    #: without a round trip to the graph.
    source_entity_type: EntityType
    target_entity_type: EntityType
    #: When the relationship began and ended. ``None`` end means "still true as
    #: far as we know", which is different from "true forever".
    valid_from: date | None = None
    valid_to: date | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    created_at: Any = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate(self) -> Relationship:
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("a relationship cannot connect an entity to itself")

        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must not precede valid_from")

        allowed_sources, allowed_targets = RELATIONSHIP_SCHEMA[self.type]
        if self.source_entity_type not in allowed_sources:
            raise ValueError(
                f"{self.type} cannot originate from a {self.source_entity_type}; "
                f"expected one of {sorted(allowed_sources)}"
            )
        if self.target_entity_type not in allowed_targets:
            raise ValueError(
                f"{self.type} cannot point at a {self.target_entity_type}; "
                f"expected one of {sorted(allowed_targets)}"
            )
        return self

    @property
    def is_symmetric(self) -> bool:
        return self.type in SYMMETRIC_RELATIONSHIPS

    @property
    def is_current(self) -> bool:
        """Whether the relationship is believed to hold today."""
        return self.valid_to is None

    @property
    def dedupe_key(self) -> str:
        """Stable key for idempotent upserts.

        Symmetric relationships sort their endpoints so that ingesting
        ``A COMPETES_WITH B`` and ``B COMPETES_WITH A`` produces one edge
        rather than two.
        """
        first, second = self.canonical_endpoints
        return f"{self.type}:{first}:{second}"

    @property
    def canonical_endpoints(self) -> tuple[str, str]:
        """The endpoints in the order the edge should be stored.

        A matching dedupe key is not enough on its own: a graph MERGE pattern is
        directed, so ``(a)-[:COMPETES_WITH]->(b)`` does not match an existing
        ``(b)-[:COMPETES_WITH]->(a)`` however the properties compare, and both
        get written. Symmetric edges therefore have to agree on a stored
        direction as well as a key. Endpoints are only swapped when both ends
        are the same entity type, which every symmetric type requires anyway.
        """
        if self.is_symmetric and self.source_entity_type is self.target_entity_type:
            first, second = sorted([self.source_entity_id, self.target_entity_id])
            return first, second
        return self.source_entity_id, self.target_entity_id


__all__ = [
    "RELATIONSHIP_SCHEMA",
    "SYMMETRIC_RELATIONSHIPS",
    "Relationship",
    "RelationshipType",
]
