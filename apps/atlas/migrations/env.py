"""Alembic environment for Atlas.

Atlas keeps its own migration history rather than sharing the platform's. Two
reasons, both operational: each application must be independently deployable,
and several products share one database instance in development, so a single
``alembic_version`` table would make them contend for the head revision. The
version table is namespaced and autogenerate is scoped to the ``atlas`` schema,
so a migration generated here can never propose dropping MarketMind's tables —
it cannot see them.
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any

import sqlalchemy as sa
from alembic import context
from sqlalchemy import engine_from_config, pool

from atlas.storage import models as atlas_models
from fie_database.config import PostgresSettings
from fie_database.postgres import Base

SCHEMA = atlas_models.SCHEMA
VERSION_TABLE = "alembic_version_atlas"

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = PostgresSettings()
config.set_main_option("sqlalchemy.url", settings.sync_dsn)

target_metadata = Base.metadata


def include_object(
    obj: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """Restrict autogenerate to Atlas's own schema."""
    if type_ == "table":
        return bool(getattr(obj, "schema", None) == SCHEMA)
    if type_ == "index":
        parent = getattr(obj, "table", None)
        return bool(getattr(parent, "schema", None) == SCHEMA)
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — used for review and audit."""
    context.configure(
        url=settings.sync_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # The version table lives in Atlas's own schema, and Alembic creates it
        # before running the migration that would have created that schema.
        # Making the schema here breaks the cycle; it is idempotent, so a re-run
        # and the migration's own CREATE both remain safe.
        connection.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
            version_table_schema=SCHEMA,
            include_schemas=True,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
