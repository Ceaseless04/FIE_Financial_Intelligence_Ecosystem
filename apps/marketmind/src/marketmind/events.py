"""Domain events MarketMind publishes.

MarketMind is the ecosystem's knowledge substrate: Atlas cites its entities in
research reports, Sentinel walks its relationships to trace exposure, and
Venture reads its company graph. Those products cache entity ids, so the graph
changing underneath them is an integration concern, not an internal detail.

``marketmind.entity.merged`` is the event that exists specifically for that
problem. When resolution decides two nodes were always the same company, one id
stops existing. A consumer holding the retired id must be told what replaced it
— otherwise it silently reads an empty neighbourhood and reports "no
relationships found" for a company with fifty.

Payloads are typed here rather than assembled as loose dicts at the call site,
so a field rename shows up as a failing build in the producer instead of a
missing key in a consumer.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from fie_events.schemas import DomainEvent
from fie_schemas.base import FrozenModel

SOURCE_APP = "marketmind"

#: A source document finished ingestion: chunked, embedded, and stored.
DOCUMENT_INGESTED = "marketmind.document.ingested"
#: A document arrived that had already been ingested; no work was done.
DOCUMENT_DUPLICATE = "marketmind.document.duplicate"
#: A new entity node was written to the graph.
ENTITY_CREATED = "marketmind.entity.created"
#: An existing entity gained aliases, identifiers, or sources.
ENTITY_UPDATED = "marketmind.entity.updated"
#: Two entities were resolved to one. The retired id is no longer resolvable.
ENTITY_MERGED = "marketmind.entity.merged"
#: A relationship was written between two entities.
RELATIONSHIP_CREATED = "marketmind.relationship.created"
#: An ingestion run finished; carries the totals for the run.
GRAPH_UPDATED = "marketmind.graph.updated"

#: Every type this app publishes. Consumers subscribe by prefix, but an
#: explicit list keeps the contract greppable from the other five apps.
PUBLISHED_EVENT_TYPES: tuple[str, ...] = (
    DOCUMENT_INGESTED,
    DOCUMENT_DUPLICATE,
    ENTITY_CREATED,
    ENTITY_UPDATED,
    ENTITY_MERGED,
    RELATIONSHIP_CREATED,
    GRAPH_UPDATED,
)


class DocumentIngestedPayload(FrozenModel):
    """A document is now retrievable and citable."""

    document_id: str
    document_type: str
    title: str
    content_hash: str
    chunk_count: int = Field(ge=0)
    embedded_chunk_count: int = Field(ge=0)
    entity_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    uri: str | None = None


class DocumentDuplicatePayload(FrozenModel):
    """A redelivered document that matched an existing content hash."""

    document_id: str
    existing_document_id: str
    content_hash: str


class EntityChangedPayload(FrozenModel):
    """An entity was created or updated."""

    entity_id: str
    entity_type: str
    name: str
    canonical_name: str
    #: ``type:value`` identifier keys, so a consumer can match on CIK or ticker
    #: without another round trip to the graph.
    identifier_keys: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)


class EntityMergedPayload(FrozenModel):
    """Two entities became one.

    ``merged_entity_id`` no longer resolves. Consumers holding it must repoint
    to ``surviving_entity_id``.
    """

    surviving_entity_id: str
    merged_entity_id: str
    entity_type: str
    name: str
    #: Why the resolver merged them — ``identifier_match``, ``fuzzy_name_match``,
    #: and so on. Carried so a surprising merge is explainable downstream.
    match_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class RelationshipCreatedPayload(FrozenModel):
    """A relationship was written to the graph."""

    relationship_id: str
    relationship_type: str
    source_entity_id: str
    target_entity_id: str
    source_entity_type: str
    target_entity_type: str
    source_document_ids: list[str] = Field(default_factory=list)


class GraphUpdatedPayload(FrozenModel):
    """Totals for one ingestion run."""

    document_id: str
    entities_created: int = Field(ge=0)
    entities_merged: int = Field(ge=0)
    relationships_created: int = Field(ge=0)
    rejected_count: int = Field(ge=0)


def build_event(
    event_type: str,
    payload: FrozenModel,
    *,
    correlation_id: str | None = None,
    causation_id: str | None = None,
    tenant_id: str | None = None,
) -> DomainEvent:
    """Wrap a typed payload in the shared envelope."""
    body: dict[str, Any] = payload.model_dump(mode="json")
    return DomainEvent(
        event_type=event_type,
        source_app=SOURCE_APP,
        correlation_id=correlation_id,
        causation_id=causation_id,
        tenant_id=tenant_id,
        payload=body,
    )


__all__ = [
    "DOCUMENT_DUPLICATE",
    "DOCUMENT_INGESTED",
    "ENTITY_CREATED",
    "ENTITY_MERGED",
    "ENTITY_UPDATED",
    "GRAPH_UPDATED",
    "PUBLISHED_EVENT_TYPES",
    "RELATIONSHIP_CREATED",
    "SOURCE_APP",
    "DocumentDuplicatePayload",
    "DocumentIngestedPayload",
    "EntityChangedPayload",
    "EntityMergedPayload",
    "GraphUpdatedPayload",
    "RelationshipCreatedPayload",
    "build_event",
]
