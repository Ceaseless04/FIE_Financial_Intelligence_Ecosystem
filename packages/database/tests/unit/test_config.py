"""Unit tests for database connection settings and DSN assembly."""

from __future__ import annotations

import pytest

from fie_common.config import Environment, SecretStr
from fie_common.errors import ConfigurationError
from fie_database.config import Neo4jSettings, PostgresSettings, RedisSettings

pytestmark = pytest.mark.unit


class TestPostgresSettings:
    def test_assembles_an_async_dsn_from_parts(self) -> None:
        settings = PostgresSettings(
            host="db.internal",
            port=5432,
            user="fie",
            password=SecretStr("s3cret"),
            database="atlas",
        )

        assert settings.dsn == "postgresql+asyncpg://fie:s3cret@db.internal:5432/atlas"

    def test_url_overrides_the_discrete_parts(self) -> None:
        settings = PostgresSettings(host="ignored", url="postgresql+asyncpg://u:p@override:5432/db")

        assert settings.dsn == "postgresql+asyncpg://u:p@override:5432/db"

    def test_special_characters_in_the_password_are_url_quoted(self) -> None:
        # An unquoted '@' or '/' silently corrupts the DSN and produces a
        # confusing connection failure far from the cause.
        settings = PostgresSettings(password=SecretStr("p@ss/w:rd?"), user="fie")

        assert "p%40ss%2Fw%3Ard%3F" in settings.dsn
        assert settings.dsn.count("@") == 1

    def test_special_characters_in_the_user_are_url_quoted(self) -> None:
        settings = PostgresSettings(user="fie@corp", password=SecretStr("x"))

        assert "fie%40corp" in settings.dsn

    def test_sync_dsn_is_used_by_alembic(self) -> None:
        settings = PostgresSettings(password=SecretStr("x"))

        assert settings.sync_dsn.startswith("postgresql+psycopg://")

    def test_production_requires_a_strong_password(self) -> None:
        with pytest.raises(ConfigurationError):
            PostgresSettings().validate_for(Environment.PRODUCTION)

    def test_development_default_password_is_allowed(self) -> None:
        PostgresSettings().validate_for(Environment.DEVELOPMENT)

    def test_strong_production_password_passes(self) -> None:
        PostgresSettings(password=SecretStr("A7f" + "x" * 30)).validate_for(Environment.PRODUCTION)

    def test_pool_configuration_is_exposed(self) -> None:
        settings = PostgresSettings(pool_size=25, max_overflow=5)

        assert settings.pool_size == 25
        assert settings.max_overflow == 5

    def test_reads_from_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIE_POSTGRES_HOST", "env-host")
        monkeypatch.setenv("FIE_POSTGRES_DATABASE", "env-db")

        settings = PostgresSettings()

        assert settings.host == "env-host"
        assert settings.database == "env-db"


class TestRedisSettings:
    def test_assembles_a_redis_dsn(self) -> None:
        settings = RedisSettings(host="cache.internal", port=6379, db=2)

        assert settings.dsn == "redis://cache.internal:6379/2"

    def test_tls_uses_the_rediss_scheme(self) -> None:
        settings = RedisSettings(use_tls=True)

        assert settings.dsn.startswith("rediss://")

    def test_password_is_included_and_quoted(self) -> None:
        settings = RedisSettings(password=SecretStr("p@ss"))

        assert "p%40ss@" in settings.dsn

    def test_url_override_wins(self) -> None:
        settings = RedisSettings(url="redis://override:6379/0")

        assert settings.dsn == "redis://override:6379/0"

    def test_db_index_is_bounded(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RedisSettings(db=16)

    def test_production_requires_a_password(self) -> None:
        with pytest.raises(ConfigurationError):
            RedisSettings(password=None).validate_for(Environment.PRODUCTION)

    def test_development_needs_no_password(self) -> None:
        RedisSettings(password=None).validate_for(Environment.DEVELOPMENT)


class TestNeo4jSettings:
    def test_auth_tuple_is_built_from_user_and_password(self) -> None:
        settings = Neo4jSettings(user="neo4j", password=SecretStr("graph-secret"))

        assert settings.auth == ("neo4j", "graph-secret")

    def test_production_requires_a_strong_password(self) -> None:
        with pytest.raises(ConfigurationError):
            Neo4jSettings().validate_for(Environment.PRODUCTION)

    def test_strong_production_password_passes(self) -> None:
        Neo4jSettings(password=SecretStr("K9m" + "z" * 30)).validate_for(Environment.PRODUCTION)

    def test_default_uri_targets_bolt(self) -> None:
        assert Neo4jSettings().uri.startswith("bolt://")
