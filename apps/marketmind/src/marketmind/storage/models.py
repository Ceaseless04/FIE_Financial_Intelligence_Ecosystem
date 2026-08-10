"""Relational storage for documents, chunks, and embeddings.

The graph holds structure; Postgres holds the text those structures were
derived from, plus the vectors used to find it. Splitting them this way keeps
Neo4j small and fast — a graph database storing multi-megabyte filing bodies
degrades every traversal that touches those nodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fie_common.utils import new_id, utc_now
from fie_database.postgres import Base

#: Width of the embedding column. Must match the configured embedding model —
#: pgvector enforces it on write, so a model swap needs a migration.
EMBEDDING_DIMENSIONS = 768

SCHEMA = "marketmind"


class DocumentRow(Base):
    """A source document retained for citation resolution."""

    __tablename__ = "documents"
    __table_args__ = (
        # Re-ingestion of an identical document is recognised by content hash
        # even when the feed assigns it a new id.
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        Index("ix_documents_type_published", "type", "published_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("doc"))
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    uri: Mapped[str | None] = mapped_column(String(2048))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRow(Base):
    """A chunk of a document, with its embedding and exact source offsets."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "index", name="uq_chunks_document_index"),
        Index("ix_chunks_document", "document_id"),
        # Retrieval filters on the embedding model before ranking, so this index
        # carries real query weight rather than being defensive.
        Index("ix_chunks_embedding_model", "embedding_model"),
        # The HNSW index over `embedding` is created in the migration: SQLAlchemy
        # cannot express pgvector's operator classes or build parameters, and
        # getting `vector_cosine_ops` wrong silently produces an index the
        # planner never uses.
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("chk"))
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.documents.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(512))
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    #: Which model produced the embedding; a mixed-model table silently
    #: destroys similarity comparisons, so retrieval filters on it.
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


class EntityMentionRow(Base):
    """Links a graph entity back to the chunk that mentioned it.

    The graph stores which documents support an entity; this stores *where* in
    those documents, which is what turns a source id into a quotable citation.
    """

    __tablename__ = "entity_mentions"
    __table_args__ = (
        UniqueConstraint("entity_id", "chunk_id", name="uq_entity_mentions_entity_chunk"),
        Index("ix_entity_mentions_entity", "entity_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("men"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.chunks.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    surface_form: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "SCHEMA",
    "ChunkRow",
    "DocumentRow",
    "EntityMentionRow",
]
