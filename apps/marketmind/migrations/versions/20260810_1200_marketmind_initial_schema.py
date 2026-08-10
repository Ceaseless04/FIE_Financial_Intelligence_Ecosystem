"""MarketMind initial schema: documents, chunks with embeddings, entity mentions.

Revision ID: a1c3f2b7d8e4
Revises:
Create Date: 2026-08-10 12:00:00+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "a1c3f2b7d8e4"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "marketmind"

#: Fixed by the embedding model in use (nomic-embed-text). Changing models
#: means a new migration and a re-embed — pgvector fixes the column width, and
#: vectors from two different models are not comparable in any case.
EMBEDDING_DIMENSIONS = 768


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    # The dev image (pgvector/pgvector:pg16) ships the extension and the init
    # script enables it; this makes a fresh production database self-sufficient.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("uri", sa.String(length=2048), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_documents_type_published",
        "documents",
        ["type", "published_at"],
        schema=SCHEMA,
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=512), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            [f"{SCHEMA}.documents.id"],
            name="fk_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint("document_id", "index", name="uq_chunks_document_index"),
        schema=SCHEMA,
    )
    op.create_index("ix_chunks_document", "chunks", ["document_id"], schema=SCHEMA)

    # HNSW rather than IVFFlat: IVFFlat needs a populated table to train its
    # lists and gives poor recall until then, which is exactly the state a fresh
    # migration leaves the table in. HNSW builds incrementally and needs no
    # training pass.
    #
    # The operator class must match the query: retrieval uses cosine distance
    # (`<=>`), so a vector_l2_ops index here would simply never be used.
    op.execute(
        f"""
        CREATE INDEX ix_chunks_embedding_hnsw
        ON {SCHEMA}.chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    # Retrieval always filters on the embedding model before ranking, so the
    # planner needs this to avoid scanning vectors it cannot compare.
    op.create_index(
        "ix_chunks_embedding_model", "chunks", ["embedding_model"], schema=SCHEMA
    )

    op.create_table(
        "entity_mentions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("surface_form", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            [f"{SCHEMA}.chunks.id"],
            name="fk_entity_mentions_chunk_id_chunks",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_entity_mentions"),
        sa.UniqueConstraint(
            "entity_id", "chunk_id", name="uq_entity_mentions_entity_chunk"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_entity_mentions_entity", "entity_mentions", ["entity_id"], schema=SCHEMA
    )


def downgrade() -> None:
    op.drop_index("ix_entity_mentions_entity", table_name="entity_mentions", schema=SCHEMA)
    op.drop_table("entity_mentions", schema=SCHEMA)

    op.drop_index("ix_chunks_embedding_model", table_name="chunks", schema=SCHEMA)
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_document", table_name="chunks", schema=SCHEMA)
    op.drop_table("chunks", schema=SCHEMA)

    op.drop_index("ix_documents_type_published", table_name="documents", schema=SCHEMA)
    op.drop_table("documents", schema=SCHEMA)

    # Deliberately asymmetric: the tables go, the schema stays. Alembic's own
    # version table lives in this schema, so dropping it here would delete the
    # bookkeeping mid-downgrade and leave the database unmigratable. The
    # `vector` extension stays for the same reason in reverse — other products
    # may already have columns that depend on it.
