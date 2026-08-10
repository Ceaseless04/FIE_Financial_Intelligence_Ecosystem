"""Database connection settings.

DSNs are assembled from discrete parts so credentials never appear in a
committed connection string, and passwords are URL-quoted so a generated
password containing ``@`` or ``/`` cannot silently corrupt the DSN.
"""

from __future__ import annotations

from urllib.parse import quote

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from fie_common.config import (
    Environment,
    FIEBaseSettings,
    SecretStr,
    require_production_secret,
    secret_value,
)


class PostgresSettings(FIEBaseSettings):
    """PostgreSQL configuration (``FIE_POSTGRES_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_POSTGRES_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "fie"
    password: SecretStr = SecretStr("fie-local-development")
    database: str = "fie"
    #: Full DSN override; takes precedence over the discrete parts above.
    url: str | None = None

    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout_seconds: float = Field(default=30.0, gt=0)
    pool_recycle_seconds: int = Field(default=1800, ge=-1)
    statement_timeout_ms: int = Field(default=30_000, ge=0)
    echo: bool = False

    @property
    def dsn(self) -> str:
        """Async SQLAlchemy DSN."""
        if self.url:
            return self.url
        password = quote(secret_value(self.password) or "", safe="")
        user = quote(self.user, safe="")
        return f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_dsn(self) -> str:
        """Synchronous DSN, used by Alembic migrations."""
        return self.dsn.replace("postgresql+asyncpg://", "postgresql+psycopg://")

    def validate_for(self, environment: Environment) -> None:
        require_production_secret(
            self.password, name="FIE_POSTGRES_PASSWORD", environment=environment, min_length=16
        )


class RedisSettings(FIEBaseSettings):
    """Redis configuration (``FIE_REDIS_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_REDIS_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: SecretStr | None = None
    use_tls: bool = False
    url: str | None = None
    max_connections: int = Field(default=50, ge=1)
    socket_timeout_seconds: float = Field(default=5.0, gt=0)
    default_ttl_seconds: int = Field(default=3600, ge=1)

    @property
    def dsn(self) -> str:
        if self.url:
            return self.url
        scheme = "rediss" if self.use_tls else "redis"
        password = secret_value(self.password)
        credentials = f":{quote(password, safe='')}@" if password else ""
        return f"{scheme}://{credentials}{self.host}:{self.port}/{self.db}"

    def validate_for(self, environment: Environment) -> None:
        if environment.is_production:
            require_production_secret(
                self.password,
                name="FIE_REDIS_PASSWORD",
                environment=environment,
                min_length=16,
            )


class Neo4jSettings(FIEBaseSettings):
    """Neo4j configuration (``FIE_NEO4J_*``). Backs the MarketMind graph."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_NEO4J_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("fie-local-development")
    database: str = "neo4j"
    max_connection_pool_size: int = Field(default=50, ge=1)
    connection_timeout_seconds: float = Field(default=30.0, gt=0)
    max_transaction_retry_seconds: float = Field(default=30.0, gt=0)

    @property
    def auth(self) -> tuple[str, str]:
        return (self.user, secret_value(self.password) or "")

    def validate_for(self, environment: Environment) -> None:
        require_production_secret(
            self.password, name="FIE_NEO4J_PASSWORD", environment=environment, min_length=16
        )


__all__ = ["Neo4jSettings", "PostgresSettings", "RedisSettings"]
