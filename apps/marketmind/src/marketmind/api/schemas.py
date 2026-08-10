"""Request and response contracts for the MarketMind API.

These are separate from the domain models on purpose. A domain ``Entity``
carries provenance objects, internal flags, and merge history that no HTTP
client needs; exposing it directly would make every internal refactor a
breaking API change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from fie_schemas.base import FIEModel, FrozenModel
from fie_schemas.provenance import Provenance
from marketmind.domain.documents import DocumentType
from marketmind.domain.entities import Entity, EntityType
from marketmind.graph.repository import GraphEdge, GraphNode


class EntityView(FrozenModel):
    """An entity as the API returns it."""

    id: str
    type: EntityType
    name: str
    canonical_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    #: Documents supporting this entity. The API always returns them: an
    #: unsourced entity is not something a downstream product should cite.
    source_document_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_entity(cls, entity: Entity) -> EntityView:
        return cls(
            id=entity.id,
            type=entity.type,
            name=entity.name,
            canonical_name=entity.canonical_name,
            aliases=list(entity.aliases),
            identifiers={
                str(identifier.type): identifier.value for identifier in entity.identifiers
            },
            description=entity.description,
            attributes=dict(entity.attributes),
            source_document_ids=[source.source_id for source in entity.provenance.sources],
        )

    @classmethod
    def from_node(cls, node: dict[str, Any], entity_type: EntityType) -> EntityView:
        """Build directly from a graph node, skipping domain rehydration.

        Search returns nodes of mixed provenance quality; refusing to render one
        because it fails a domain invariant would make search fail on data the
        graph already contains.
        """
        reserved = {
            "id",
            "name",
            "canonical_name",
            "description",
            "aliases",
            "identifier_keys",
            "source_ids",
            "created_at",
            "updated_at",
        }
        identifiers: dict[str, str] = {}
        for key in node.get("identifier_keys") or []:
            identifier_type, _, value = str(key).partition(":")
            identifiers[identifier_type] = value
        return cls(
            id=str(node.get("id") or ""),
            type=entity_type,
            name=str(node.get("name") or ""),
            canonical_name=str(node.get("canonical_name") or ""),
            aliases=[str(alias) for alias in node.get("aliases") or []],
            identifiers=identifiers,
            description=node.get("description"),
            attributes={k: v for k, v in node.items() if k not in reserved},
            source_document_ids=[str(s) for s in node.get("source_ids") or []],
        )


class RelationshipView(FrozenModel):
    """One relationship of an entity, from that entity's point of view."""

    type: str
    relationship_id: str | None = None
    direction: str
    other_entity_id: str
    other_entity_type: str
    other_entity_name: str
    valid_from: str | None = None
    valid_to: str | None = None
    source_document_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_edge(cls, edge: GraphEdge) -> RelationshipView:
        return cls(
            type=edge.type,
            relationship_id=edge.relationship_id,
            direction="outgoing" if edge.is_outgoing else "incoming",
            other_entity_id=edge.other_id,
            other_entity_type=edge.other_label,
            other_entity_name=edge.other_name,
            valid_from=str(edge.valid_from) if edge.valid_from else None,
            valid_to=str(edge.valid_to) if edge.valid_to else None,
            source_document_ids=list(edge.source_ids),
        )


class GraphNodeView(FrozenModel):
    """A node in a visualization payload."""

    id: str
    label: str
    name: str
    #: Hops from the seed entity. Frontends use it for radial layout.
    distance: int = 0
    source_document_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_node(cls, node: GraphNode) -> GraphNodeView:
        return cls(
            id=node.id,
            label=node.label,
            name=node.name,
            distance=node.distance,
            source_document_ids=list(node.source_ids),
        )


class GraphEdgeView(FrozenModel):
    """An edge in a visualization payload, in source -> target form."""

    id: str | None = None
    type: str
    source: str
    target: str
    source_document_ids: list[str] = Field(default_factory=list)


class GraphView(FrozenModel):
    """Nodes and edges ready for a force-directed rendering.

    Edges are normalised to explicit ``source``/``target`` ids rather than the
    traversal's relative "incoming/outgoing", because a renderer has no notion
    of which node the caller started from.
    """

    seed_entity_id: str
    depth: int
    nodes: list[GraphNodeView] = Field(default_factory=list)
    edges: list[GraphEdgeView] = Field(default_factory=list)
    #: True when the traversal hit its node cap and the view is partial.
    truncated: bool = False


class PathStep(FrozenModel):
    id: str
    name: str
    label: str


class PathView(FrozenModel):
    """The shortest connection between two entities."""

    source_entity_id: str
    target_entity_id: str
    length: int
    nodes: list[PathStep] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)


class GraphStatsView(FrozenModel):
    """Graph size, for dashboards and smoke checks."""

    total_nodes: int
    total_relationships: int
    nodes_by_label: dict[str, int] = Field(default_factory=dict)
    relationships_by_type: dict[str, int] = Field(default_factory=dict)


class IngestDocumentRequest(FIEModel):
    """Submit a document for ingestion."""

    type: DocumentType
    title: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=1)
    uri: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestDocumentResponse(FrozenModel):
    """Outcome of an ingestion request."""

    document_id: str
    duplicate: bool
    chunk_count: int
    embedded_chunk_count: int
    entities_created: int
    entities_merged: int
    relationships_created: int
    #: Extraction output that failed validation. Returned rather than hidden so
    #: a caller can see that a "successful" ingestion dropped half its edges.
    rejected: list[str] = Field(default_factory=list)
    is_retrievable: bool = True


class AskRequest(FIEModel):
    """A natural-language question against the knowledge graph."""

    question: str = Field(min_length=3, max_length=2000)
    #: Optional entity ids to seed traversal, skipping vector-based anchoring.
    entity_ids: list[str] = Field(default_factory=list, max_length=20)


class CitationView(FrozenModel):
    """A source the answer actually rests on."""

    document_id: str
    title: str | None = None
    uri: str | None = None


class AnswerResponse(FrozenModel):
    """A cited answer.

    ``grounded`` is not decoration: it is false when the model cited a source
    that was not in the supplied context. A client showing answers must treat
    an ungrounded one differently from a grounded one.
    """

    question: str
    answer: str
    grounded: bool
    sufficient_context: bool
    citations: list[CitationView] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    provenance: Provenance


class ContextPassageView(FrozenModel):
    """One retrieved passage, with the span it came from."""

    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    section: str | None = None
    similarity: float


class ContextResponse(FrozenModel):
    """What retrieval assembled, without asking a model to summarise it."""

    question: str
    passages: list[ContextPassageView] = Field(default_factory=list)
    graph_facts: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)


__all__ = [
    "AnswerResponse",
    "AskRequest",
    "CitationView",
    "ContextPassageView",
    "ContextResponse",
    "EntityView",
    "GraphEdgeView",
    "GraphNodeView",
    "GraphStatsView",
    "GraphView",
    "IngestDocumentRequest",
    "IngestDocumentResponse",
    "PathStep",
    "PathView",
    "RelationshipView",
]
