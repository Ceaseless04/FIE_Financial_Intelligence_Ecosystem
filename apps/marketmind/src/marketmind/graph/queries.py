"""Cypher query construction.

Every value reaches Neo4j as a bound parameter. The only thing ever
interpolated into a query string is a label or relationship type — Cypher
cannot parameterize those — and both are validated against the domain enums
first, so an attacker-controlled string can never become a clause.

Queries are built here rather than inline in the repository so the injection
guard has exactly one place to live and can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fie_common.errors import ValidationError
from marketmind.domain.entities import EntityType
from marketmind.domain.relationships import RelationshipType

#: Labels and relationship types are validated against these, never escaped.
_VALID_LABELS = {str(entity_type) for entity_type in EntityType}
_VALID_RELATIONSHIP_TYPES = {str(rel_type) for rel_type in RelationshipType}


def validate_label(label: str) -> str:
    """Confirm a node label is one the domain defines.

    Cypher has no parameter form for labels, so this allowlist is the entire
    defence against label injection.
    """
    if label not in _VALID_LABELS:
        raise ValidationError(
            f"unknown node label {label!r}",
            details={"allowed": sorted(_VALID_LABELS)},
        )
    return label


def validate_relationship_type(relationship_type: str) -> str:
    """Confirm a relationship type is one the domain defines."""
    if relationship_type not in _VALID_RELATIONSHIP_TYPES:
        raise ValidationError(
            f"unknown relationship type {relationship_type!r}",
            details={"allowed": sorted(_VALID_RELATIONSHIP_TYPES)},
        )
    return relationship_type


@dataclass(frozen=True)
class CypherQuery:
    """A query and its bound parameters."""

    text: str
    parameters: dict[str, Any]


def upsert_entity_query(label: str) -> str:
    """Idempotent entity write, keyed on the application-assigned id.

    ``ON CREATE`` versus ``ON MATCH`` matters: ``created_at`` must survive
    re-ingestion, and alias/identifier lists are unioned rather than replaced so
    a document that mentions a company by only one of its names cannot erase
    the others.
    """
    validate_label(label)
    return f"""
    MERGE (n:{label} {{id: $id}})
    ON CREATE SET
        n.created_at = $now,
        n.name = $name,
        n.canonical_name = $canonical_name
    ON MATCH SET
        n.name = coalesce(n.name, $name),
        n.canonical_name = $canonical_name
    SET n.updated_at = $now,
        n.description = coalesce($description, n.description),
        n.aliases = apoc.coll.toSet(coalesce(n.aliases, []) + $aliases),
        n.identifier_keys = apoc.coll.toSet(
            coalesce(n.identifier_keys, []) + $identifier_keys
        ),
        n.source_ids = apoc.coll.toSet(coalesce(n.source_ids, []) + $source_ids),
        n += $attributes
    RETURN n.id AS id
    """


def upsert_entity_query_portable(label: str) -> str:
    """Same upsert without APOC, for deployments that do not install it.

    List de-duplication is done with a Cypher comprehension instead. Slower on
    long lists, but the community edition does not always ship APOC and a graph
    that cannot be written is worse than one written slowly.
    """
    validate_label(label)
    return f"""
    MERGE (n:{label} {{id: $id}})
    ON CREATE SET
        n.created_at = $now,
        n.name = $name,
        n.canonical_name = $canonical_name
    ON MATCH SET
        n.name = coalesce(n.name, $name),
        n.canonical_name = $canonical_name
    SET n.updated_at = $now,
        n.description = coalesce($description, n.description),
        n.aliases = [
            x IN coalesce(n.aliases, []) + $aliases
            WHERE NOT x IN [] | x
        ],
        n.identifier_keys = coalesce(n.identifier_keys, []) + [
            k IN $identifier_keys WHERE NOT k IN coalesce(n.identifier_keys, []) | k
        ],
        n.source_ids = coalesce(n.source_ids, []) + [
            s IN $source_ids WHERE NOT s IN coalesce(n.source_ids, []) | s
        ],
        n += $attributes
    RETURN n.id AS id
    """


def upsert_relationship_query(relationship_type: str, source_label: str, target_label: str) -> str:
    """Idempotent relationship write, keyed on a stable dedupe key."""
    validate_relationship_type(relationship_type)
    validate_label(source_label)
    validate_label(target_label)
    return f"""
    MATCH (source:{source_label} {{id: $source_id}})
    MATCH (target:{target_label} {{id: $target_id}})
    MERGE (source)-[r:{relationship_type} {{dedupe_key: $dedupe_key}}]->(target)
    ON CREATE SET r.id = $id, r.created_at = $now
    SET r.updated_at = $now,
        r.valid_from = $valid_from,
        r.valid_to = $valid_to,
        r.source_ids = coalesce(r.source_ids, []) + [
            s IN $source_ids WHERE NOT s IN coalesce(r.source_ids, []) | s
        ],
        r += $attributes
    RETURN r.id AS id
    """


def find_entity_by_identifier_query(label: str) -> str:
    validate_label(label)
    return f"""
    MATCH (n:{label})
    WHERE $identifier_key IN n.identifier_keys
    RETURN n
    LIMIT 1
    """


def find_entity_by_id_query(label: str) -> str:
    """Fetch one node by id, returning its label alongside its properties."""
    validate_label(label)
    return f"""
    MATCH (n:{label} {{id: $entity_id}})
    RETURN n, labels(n)[0] AS label
    LIMIT 1
    """


def find_entities_by_canonical_name_query(label: str) -> str:
    validate_label(label)
    return f"""
    MATCH (n:{label} {{canonical_name: $canonical_name}})
    RETURN n
    LIMIT $limit
    """


def search_entities_query(label: str | None) -> str:
    """Prefix/substring search over canonical names."""
    if label is None:
        match_clause = "MATCH (n)"
    else:
        validate_label(label)
        match_clause = f"MATCH (n:{label})"
    return f"""
    {match_clause}
    WHERE n.canonical_name CONTAINS $term
       OR toLower(n.name) CONTAINS $term
    RETURN n, labels(n)[0] AS label
    ORDER BY size(n.canonical_name) ASC
    LIMIT $limit
    """


def neighbourhood_query(depth: int, relationship_types: list[str] | None) -> str:
    """Bounded k-hop expansion around a seed entity.

    Depth is validated and interpolated because Cypher does not allow a
    parameter inside a variable-length pattern. It is bounded hard: an
    unbounded traversal on a dense financial graph will not return.
    """
    if not 1 <= depth <= 3:
        raise ValidationError(
            "traversal depth must be between 1 and 3",
            details={"requested_depth": depth},
        )

    if relationship_types:
        for relationship_type in relationship_types:
            validate_relationship_type(relationship_type)
        type_filter = ":" + "|".join(relationship_types)
    else:
        type_filter = ""

    return f"""
    MATCH path = (seed {{id: $entity_id}})-[r{type_filter}*1..{depth}]-(neighbour)
    WITH seed, neighbour, relationships(path) AS rels, length(path) AS distance
    RETURN
        neighbour.id AS id,
        labels(neighbour)[0] AS label,
        neighbour.name AS name,
        neighbour.canonical_name AS canonical_name,
        neighbour.source_ids AS source_ids,
        distance,
        [rel IN rels | {{
            type: type(rel),
            id: rel.id,
            source_ids: rel.source_ids,
            valid_from: rel.valid_from,
            valid_to: rel.valid_to
        }}] AS path_relationships
    ORDER BY distance ASC, neighbour.name ASC
    LIMIT $limit
    """


def entity_relationships_query(direction: str) -> str:
    """Direct relationships of one entity.

    ``direction`` is validated against a fixed set rather than interpolated
    freely — it is the one place a caller-supplied string shapes the pattern.
    """
    patterns = {
        "outgoing": "(n {id: $entity_id})-[r]->(other)",
        "incoming": "(n {id: $entity_id})<-[r]-(other)",
        "both": "(n {id: $entity_id})-[r]-(other)",
    }
    if direction not in patterns:
        raise ValidationError(
            f"unknown direction {direction!r}",
            details={"allowed": sorted(patterns)},
        )
    return f"""
    MATCH {patterns[direction]}
    RETURN
        type(r) AS type,
        r.id AS relationship_id,
        r.source_ids AS source_ids,
        r.valid_from AS valid_from,
        r.valid_to AS valid_to,
        other.id AS other_id,
        labels(other)[0] AS other_label,
        other.name AS other_name,
        startNode(r).id = $entity_id AS is_outgoing
    ORDER BY type(r), other.name
    LIMIT $limit
    """


def shortest_path_query(max_depth: int) -> str:
    """Shortest connection between two entities, for explaining exposure."""
    if not 1 <= max_depth <= 6:
        raise ValidationError(
            "shortest path depth must be between 1 and 6",
            details={"requested_depth": max_depth},
        )
    return f"""
    MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
    MATCH path = shortestPath((source)-[*1..{max_depth}]-(target))
    RETURN
        [node IN nodes(path) | {{
            id: node.id, name: node.name, label: labels(node)[0]
        }}] AS nodes,
        [rel IN relationships(path) | {{
            type: type(rel), id: rel.id
        }}] AS relationships,
        length(path) AS length
    """


GRAPH_STATS_QUERY = """
MATCH (n)
WITH labels(n)[0] AS label, count(*) AS node_count
RETURN label, node_count
ORDER BY node_count DESC
"""

RELATIONSHIP_STATS_QUERY = """
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS relationship_count
ORDER BY relationship_count DESC
"""

DELETE_ALL_QUERY = "MATCH (n) DETACH DELETE n"


__all__ = [
    "DELETE_ALL_QUERY",
    "GRAPH_STATS_QUERY",
    "RELATIONSHIP_STATS_QUERY",
    "CypherQuery",
    "entity_relationships_query",
    "find_entities_by_canonical_name_query",
    "find_entity_by_id_query",
    "find_entity_by_identifier_query",
    "neighbourhood_query",
    "search_entities_query",
    "shortest_path_query",
    "upsert_entity_query",
    "upsert_entity_query_portable",
    "upsert_relationship_query",
    "validate_label",
    "validate_relationship_type",
]
