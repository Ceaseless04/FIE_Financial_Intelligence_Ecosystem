"""Graph repository — the only module that writes to Neo4j.

Writes are idempotent: re-ingesting the same filing produces the same graph, so
a redelivered document is harmless. That property is what lets the event
consumer be at-least-once without special-casing duplicates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from fie_common.errors import NotFoundError
from fie_common.utils import utc_now
from fie_database.neo4j_client import Neo4jClient
from fie_observability.logging import get_logger
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from marketmind.domain.entities import Entity, EntityType, Identifier, IdentifierType
from marketmind.domain.relationships import Relationship, RelationshipType
from marketmind.graph import queries

logger = get_logger(__name__)

#: Node properties that are structural rather than domain attributes.
_RESERVED_NODE_PROPERTIES = frozenset(
    {
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
)


def entity_from_node(node: dict[str, Any], entity_type: EntityType) -> Entity:
    """Rehydrate a stored node into a domain entity.

    The inverse of :meth:`GraphRepository._entity_parameters`, used to turn
    stored nodes back into resolution candidates. Provenance is reconstructed as
    ``KNOWLEDGE_GRAPH`` sources: the node records *which* documents supported it,
    but not the spans, so claiming a filing citation here would overstate what
    was actually retained.
    """
    identifiers: list[Identifier] = []
    for key in node.get("identifier_keys") or []:
        identifier_type, _, value = str(key).partition(":")
        try:
            identifiers.append(Identifier(type=IdentifierType(identifier_type), value=value))
        except ValueError:
            # A key written by an older schema version must not break reads.
            logger.warning("unparsable_identifier_key", key=str(key))

    # A node with no recorded documents still has evidence — the graph record
    # itself. Citing that is accurate; inventing a document id would not be.
    source_ids = [str(source_id) for source_id in node.get("source_ids") or []] or [str(node["id"])]
    sources = [
        SourceReference(source_id=source_id, source_type=SourceType.KNOWLEDGE_GRAPH)
        for source_id in source_ids
    ]

    attributes = {key: value for key, value in node.items() if key not in _RESERVED_NODE_PROPERTIES}

    return Entity(
        id=str(node["id"]),
        type=entity_type,
        name=str(node.get("name") or node["id"]),
        canonical_name=str(node.get("canonical_name") or ""),
        aliases=[str(alias) for alias in node.get("aliases") or []],
        identifiers=identifiers,
        description=node.get("description"),
        attributes=attributes,
        provenance=Provenance.fact(*sources),
        # An entity read back from the graph has, by definition, been resolved.
        resolved=True,
    )


class GraphNode(FIEModel):
    """A node as returned from a traversal."""

    id: str
    label: str
    name: str
    canonical_name: str = ""
    source_ids: list[str] = Field(default_factory=list)
    distance: int = 0


class GraphEdge(FIEModel):
    """An edge as returned from a traversal."""

    type: str
    relationship_id: str | None = None
    other_id: str
    other_label: str
    other_name: str
    is_outgoing: bool = True
    source_ids: list[str] = Field(default_factory=list)
    valid_from: Any = None
    valid_to: Any = None


class Neighbourhood(FIEModel):
    """A seed entity's k-hop neighbourhood, ready for context assembly."""

    seed_entity_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    depth: int = 1

    @property
    def source_ids(self) -> list[str]:
        """Every document that contributed to this neighbourhood."""
        seen: dict[str, None] = {}
        for node in self.nodes:
            for source_id in node.source_ids:
                seen.setdefault(source_id, None)
        return list(seen)


@dataclass
class GraphRepository:
    """Reads and writes the knowledge graph."""

    client: Neo4jClient
    #: APOC ships with the dev image but not every deployment; the portable
    #: upsert avoids it at some cost in write speed.
    use_apoc: bool = False

    async def apply_schema(self) -> int:
        """Create constraints and indexes. Idempotent and safe to re-run."""
        from marketmind.graph.schema import all_schema_statements

        applied = 0
        for statement in all_schema_statements():
            try:
                await self.client.execute_write(statement)
                applied += 1
            except Exception as error:  # noqa: BLE001 — one unsupported index
                # must not stop the rest of the schema from being applied.
                logger.warning(
                    "graph_schema_statement_failed",
                    statement=statement[:120],
                    error=str(error),
                )
        logger.info("graph_schema_applied", statements=applied)
        return applied

    # -- writes --------------------------------------------------------------

    async def upsert_entity(self, entity: Entity) -> str:
        """Write an entity, merging into any existing node with the same id."""
        query = (
            queries.upsert_entity_query(str(entity.type))
            if self.use_apoc
            else queries.upsert_entity_query_portable(str(entity.type))
        )
        parameters = self._entity_parameters(entity)
        rows = await self.client.execute_write(query, parameters)
        return str(rows[0]["id"]) if rows else entity.id

    async def upsert_entities(self, entities: Sequence[Entity]) -> list[str]:
        return [await self.upsert_entity(entity) for entity in entities]

    async def upsert_relationship(self, relationship: Relationship) -> str:
        """Write a relationship between two entities that must already exist.

        Raises:
            NotFoundError: if either endpoint is missing. The query's leading
                ``MATCH`` clauses simply return no rows in that case, so without
                this check a dangling edge would be reported as written and
                counted in the ingestion totals.
        """
        query = queries.upsert_relationship_query(
            str(relationship.type),
            str(relationship.source_entity_type),
            str(relationship.target_entity_type),
        )
        rows = await self.client.execute_write(query, self._relationship_parameters(relationship))
        if not rows:
            raise NotFoundError(
                "cannot write a relationship whose endpoints are not both in the graph",
                details={
                    "relationship_type": str(relationship.type),
                    "source_entity_id": relationship.source_entity_id,
                    "target_entity_id": relationship.target_entity_id,
                },
            )
        return str(rows[0]["id"])

    async def upsert_relationships(self, relationships: Sequence[Relationship]) -> list[str]:
        written: list[str] = []
        for relationship in relationships:
            try:
                written.append(await self.upsert_relationship(relationship))
            except Exception as error:  # noqa: BLE001 — a dangling endpoint is a
                # data problem, not a reason to abandon the batch.
                logger.warning(
                    "relationship_write_failed",
                    relationship_type=str(relationship.type),
                    error=str(error),
                )
        return written

    @staticmethod
    def _entity_parameters(entity: Entity) -> dict[str, Any]:
        # Neo4j properties must be primitives or arrays of primitives, so nested
        # attribute values are dropped rather than silently corrupting the write.
        attributes = {
            key: value
            for key, value in entity.attributes.items()
            if isinstance(value, (str, int, float, bool))
        }
        return {
            "id": entity.id,
            "name": entity.name,
            "canonical_name": entity.canonical_name,
            "description": entity.description,
            "aliases": list(entity.aliases),
            "identifier_keys": sorted(entity.identifier_keys),
            "source_ids": [source.source_id for source in entity.provenance.sources],
            "attributes": attributes,
            "now": utc_now().isoformat(),
        }

    @staticmethod
    def _relationship_parameters(relationship: Relationship) -> dict[str, Any]:
        attributes = {
            key: value
            for key, value in relationship.attributes.items()
            if isinstance(value, (str, int, float, bool))
        }
        # Symmetric edges are stored in a canonical direction; see
        # Relationship.canonical_endpoints for why the dedupe key alone is not
        # sufficient to keep MERGE from writing both directions.
        source_id, target_id = relationship.canonical_endpoints
        return {
            "id": relationship.id,
            "dedupe_key": relationship.dedupe_key,
            "source_id": source_id,
            "target_id": target_id,
            "valid_from": relationship.valid_from.isoformat() if relationship.valid_from else None,
            "valid_to": relationship.valid_to.isoformat() if relationship.valid_to else None,
            "source_ids": [source.source_id for source in relationship.provenance.sources],
            "attributes": attributes,
            "now": utc_now().isoformat(),
        }

    # -- reads ---------------------------------------------------------------

    async def find_by_identifier(
        self, entity_type: EntityType, identifier_key: str
    ) -> dict[str, Any] | None:
        rows = await self.client.execute_read(
            queries.find_entity_by_identifier_query(str(entity_type)),
            {"identifier_key": identifier_key},
        )
        return dict(rows[0]["n"]) if rows else None

    async def find_by_id(
        self, entity_id: str, *, entity_type: EntityType | None = None
    ) -> dict[str, Any] | None:
        """Fetch a node by id, returning its properties plus its ``label``.

        Ids are indexed per label, so a typed lookup is one index probe while an
        untyped one probes each label in turn. The type is therefore an
        optimisation the caller may supply, never a correctness requirement.
        """
        candidates = [entity_type] if entity_type is not None else list(EntityType)
        for candidate in candidates:
            rows = await self.client.execute_read(
                queries.find_entity_by_id_query(str(candidate)),
                {"entity_id": entity_id},
            )
            if rows:
                node = dict(rows[0]["n"])
                node["label"] = rows[0]["label"]
                return node
        return None

    async def find_by_canonical_name(
        self, entity_type: EntityType, canonical_name: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        rows = await self.client.execute_read(
            queries.find_entities_by_canonical_name_query(str(entity_type)),
            {"canonical_name": canonical_name, "limit": limit},
        )
        return [dict(row["n"]) for row in rows]

    async def search(
        self, term: str, *, entity_type: EntityType | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Substring search over names. Used by the API search endpoint."""
        rows = await self.client.execute_read(
            queries.search_entities_query(str(entity_type) if entity_type else None),
            {"term": term.strip().lower(), "limit": limit},
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            node = dict(row["n"])
            node["label"] = row["label"]
            results.append(node)
        return results

    async def neighbourhood(
        self,
        entity_id: str,
        *,
        depth: int = 1,
        relationship_types: Sequence[RelationshipType] | None = None,
        limit: int = 50,
    ) -> Neighbourhood:
        """Bounded k-hop expansion, the retrieval half of GraphRAG."""
        query = queries.neighbourhood_query(
            depth,
            [str(rel_type) for rel_type in relationship_types] if relationship_types else None,
        )
        rows = await self.client.execute_read(query, {"entity_id": entity_id, "limit": limit})

        nodes = [
            GraphNode(
                id=row["id"],
                label=row["label"],
                name=row["name"] or "",
                canonical_name=row.get("canonical_name") or "",
                source_ids=list(row.get("source_ids") or []),
                distance=int(row.get("distance") or 1),
            )
            for row in rows
            if row.get("id")
        ]
        return Neighbourhood(seed_entity_id=entity_id, nodes=nodes, depth=depth)

    async def relationships_of(
        self, entity_id: str, *, direction: str = "both", limit: int = 100
    ) -> list[GraphEdge]:
        rows = await self.client.execute_read(
            queries.entity_relationships_query(direction),
            {"entity_id": entity_id, "limit": limit},
        )
        return [
            GraphEdge(
                type=row["type"],
                relationship_id=row.get("relationship_id"),
                other_id=row["other_id"],
                other_label=row["other_label"],
                other_name=row["other_name"] or "",
                is_outgoing=bool(row.get("is_outgoing", True)),
                source_ids=list(row.get("source_ids") or []),
                valid_from=row.get("valid_from"),
                valid_to=row.get("valid_to"),
            )
            for row in rows
        ]

    async def shortest_path(
        self, source_id: str, target_id: str, *, max_depth: int = 4
    ) -> dict[str, Any] | None:
        """How two entities connect — the basis of exposure explanations."""
        rows = await self.client.execute_read(
            queries.shortest_path_query(max_depth),
            {"source_id": source_id, "target_id": target_id},
        )
        return dict(rows[0]) if rows else None

    async def stats(self) -> dict[str, Any]:
        node_rows = await self.client.execute_read(queries.GRAPH_STATS_QUERY)
        edge_rows = await self.client.execute_read(queries.RELATIONSHIP_STATS_QUERY)
        return {
            "nodes_by_label": {
                row["label"]: row["node_count"] for row in node_rows if row.get("label")
            },
            "relationships_by_type": {row["type"]: row["relationship_count"] for row in edge_rows},
            "total_nodes": sum(row["node_count"] for row in node_rows),
            "total_relationships": sum(row["relationship_count"] for row in edge_rows),
        }

    async def delete_all(self) -> None:
        """Wipe the graph. Test fixtures and local resets only."""
        await self.client.execute_write(queries.DELETE_ALL_QUERY)


__all__ = [
    "GraphEdge",
    "GraphNode",
    "GraphRepository",
    "Neighbourhood",
    "entity_from_node",
]
