"""Graph schema: constraints and indexes.

Constraints are applied at startup and are idempotent (``IF NOT EXISTS``). They
matter more than usual here because the uniqueness constraint on ``id`` is what
makes ``MERGE`` safe under concurrent ingestion — without it, two workers
processing the same company simultaneously create two nodes.
"""

from __future__ import annotations

from fie_observability.logging import get_logger
from marketmind.domain.entities import EntityType

logger = get_logger(__name__)


def constraint_statements() -> list[str]:
    """Uniqueness constraints, one per node label."""
    statements = [
        f"CREATE CONSTRAINT {str(entity_type).lower()}_id_unique IF NOT EXISTS "
        f"FOR (n:{entity_type}) REQUIRE n.id IS UNIQUE"
        for entity_type in EntityType
    ]
    return statements


def index_statements() -> list[str]:
    """Indexes supporting the query patterns the API actually issues."""
    statements: list[str] = []
    for entity_type in EntityType:
        label = str(entity_type)
        lowered = label.lower()
        # Resolution and search both look up by canonical name.
        statements.append(
            f"CREATE INDEX {lowered}_canonical_name IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.canonical_name)"
        )
        # Identifier lookup is the decisive path in entity resolution.
        statements.append(
            f"CREATE INDEX {lowered}_identifier_keys IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.identifier_keys)"
        )
    return statements


def fulltext_statements() -> list[str]:
    """Full-text index over names and aliases, for the search endpoint."""
    labels = "|".join(str(entity_type) for entity_type in EntityType)
    return [
        "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
        f"FOR (n:{labels}) ON EACH [n.name, n.canonical_name, n.description]"
    ]


def all_schema_statements() -> list[str]:
    return [*constraint_statements(), *index_statements(), *fulltext_statements()]


__all__ = [
    "all_schema_statements",
    "constraint_statements",
    "fulltext_statements",
    "index_statements",
]
