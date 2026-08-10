"""fie_database — PostgreSQL, Redis, and Neo4j infrastructure.

Connection management, transaction scoping, and health checks only. Schemas
and queries belong to the product packages that own the data.
"""

from fie_database.config import Neo4jSettings, PostgresSettings, RedisSettings
from fie_database.neo4j_client import Neo4jClient
from fie_database.postgres import (
    NAMING_CONVENTION,
    Base,
    PostgresDatabase,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from fie_database.redis_client import RedisClient

__version__ = "0.1.0"

__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "Neo4jClient",
    "Neo4jSettings",
    "PostgresDatabase",
    "PostgresSettings",
    "RedisClient",
    "RedisSettings",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "__version__",
]
