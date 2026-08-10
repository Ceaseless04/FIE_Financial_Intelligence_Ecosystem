"""Document, chunk, and vector persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from fie_ai.embeddings import Vector
from fie_common.errors import ValidationError
from fie_observability.logging import get_logger
from fie_schemas.base import FIEModel
from marketmind.domain.documents import Chunk, Document, DocumentType
from marketmind.storage.models import ChunkRow, DocumentRow, EntityMentionRow

logger = get_logger(__name__)


class ChunkHit(FIEModel):
    """A chunk retrieved by similarity search."""

    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    section: str | None = None
    #: Cosine distance from the query vector; lower is closer.
    distance: float = 0.0

    @property
    def similarity(self) -> float:
        """Cosine similarity in [0, 1], easier to reason about than distance."""
        return max(0.0, min(1.0, 1.0 - self.distance))


@dataclass
class DocumentRepository:
    """Persists documents and chunks, and runs vector similarity search."""

    session: AsyncSession

    async def upsert_document(self, document: Document) -> tuple[str, bool]:
        """Store a document, returning its id and whether it was newly created.

        Deduplicates on content hash: feeds redeliver the same filing routinely,
        and re-chunking plus re-embedding it is pure waste.
        """
        existing = await self.session.scalar(
            select(DocumentRow).where(DocumentRow.content_hash == document.content_hash)
        )
        if existing is not None:
            return existing.id, False

        row = DocumentRow(
            id=document.id,
            type=str(document.type),
            title=document.title,
            content=document.content,
            content_hash=document.content_hash,
            uri=document.uri,
            published_at=document.published_at,
            ingested_at=document.ingested_at,
            doc_metadata=document.metadata,
        )
        self.session.add(row)
        await self.session.flush()
        return row.id, True

    async def get_document(self, document_id: str) -> Document | None:
        row = await self.session.get(DocumentRow, document_id)
        return self._to_document(row) if row else None

    async def store_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Vector] | None = None,
        *,
        embedding_model: str | None = None,
    ) -> int:
        """Persist chunks with their embeddings.

        Raises:
            ValidationError: if the embedding count does not match the chunks.
        """
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValidationError(
                "embedding count does not match chunk count",
                details={"chunks": len(chunks), "embeddings": len(embeddings)},
            )

        for position, chunk in enumerate(chunks):
            statement = (
                insert(ChunkRow)
                .values(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    index=chunk.index,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    section=chunk.section,
                    embedding=embeddings[position] if embeddings else None,
                    embedding_model=embedding_model if embeddings else None,
                )
                # Re-ingestion overwrites in place rather than failing, so a
                # changed chunker or a new embedding model can be rolled out.
                .on_conflict_do_update(
                    constraint="uq_chunks_document_index",
                    set_={
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "section": chunk.section,
                        "embedding": embeddings[position] if embeddings else None,
                        "embedding_model": embedding_model if embeddings else None,
                    },
                )
            )
            await self.session.execute(statement)

        return len(chunks)

    async def get_chunks(self, document_id: str) -> list[Chunk]:
        rows = await self.session.scalars(
            select(ChunkRow).where(ChunkRow.document_id == document_id).order_by(ChunkRow.index)
        )
        return [self._to_chunk(row) for row in rows]

    async def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = await self.session.scalars(select(ChunkRow).where(ChunkRow.id.in_(list(chunk_ids))))
        return [self._to_chunk(row) for row in rows]

    async def search_similar(
        self,
        query_vector: Vector,
        *,
        limit: int = 10,
        embedding_model: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[ChunkHit]:
        """Nearest chunks by cosine distance.

        Filtering on ``embedding_model`` is not optional in practice: comparing
        vectors from two different models produces confident nonsense, and a
        model rollout leaves both in the table at once.
        """
        distance = ChunkRow.embedding.cosine_distance(query_vector)
        statement = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.text,
                ChunkRow.start_char,
                ChunkRow.end_char,
                ChunkRow.section,
                distance.label("distance"),
            )
            .where(ChunkRow.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        if embedding_model is not None:
            statement = statement.where(ChunkRow.embedding_model == embedding_model)
        if document_ids:
            statement = statement.where(ChunkRow.document_id.in_(list(document_ids)))

        rows = await self.session.execute(statement)
        return [
            ChunkHit(
                chunk_id=row.id,
                document_id=row.document_id,
                text=row.text,
                start_char=row.start_char,
                end_char=row.end_char,
                section=row.section,
                distance=float(row.distance),
            )
            for row in rows
        ]

    async def record_mentions(
        self, entity_id: str, entity_type: str, surface_form: str, chunk_ids: Sequence[str]
    ) -> int:
        """Link an entity to the chunks that mention it."""
        written = 0
        for chunk_id in chunk_ids:
            statement = (
                insert(EntityMentionRow)
                .values(
                    entity_id=entity_id,
                    chunk_id=chunk_id,
                    entity_type=entity_type,
                    surface_form=surface_form,
                )
                .on_conflict_do_nothing(constraint="uq_entity_mentions_entity_chunk")
            )
            await self.session.execute(statement)
            written += 1
        return written

    async def entity_ids_for_chunks(self, chunk_ids: Sequence[str]) -> list[str]:
        """Entities mentioned in these chunks — the graph seeds for retrieval.

        This is the join that turns a vector hit into a graph anchor: the
        passage was found by similarity, and the entities it mentions are where
        traversal starts.
        """
        if not chunk_ids:
            return []
        rows = await self.session.execute(
            select(EntityMentionRow.entity_id)
            .where(EntityMentionRow.chunk_id.in_(list(chunk_ids)))
            .distinct()
        )
        return [str(row[0]) for row in rows]

    async def chunks_mentioning(self, entity_id: str, *, limit: int = 20) -> list[Chunk]:
        """Chunks that mention an entity — the evidence behind a graph node."""
        statement = (
            select(ChunkRow)
            .join(EntityMentionRow, EntityMentionRow.chunk_id == ChunkRow.id)
            .where(EntityMentionRow.entity_id == entity_id)
            .limit(limit)
        )
        rows = await self.session.scalars(statement)
        return [self._to_chunk(row) for row in rows]

    async def delete_document(self, document_id: str) -> None:
        await self.session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))

    @staticmethod
    def _to_document(row: DocumentRow) -> Document:
        return Document(
            id=row.id,
            type=DocumentType(row.type),
            title=row.title,
            content=row.content,
            uri=row.uri,
            published_at=row.published_at,
            ingested_at=row.ingested_at,
            metadata=dict(row.doc_metadata or {}),
        )

    @staticmethod
    def _to_chunk(row: ChunkRow) -> Chunk:
        return Chunk(
            id=row.id,
            document_id=row.document_id,
            index=row.index,
            text=row.text,
            start_char=row.start_char,
            end_char=row.end_char,
            section=row.section,
        )


__all__ = ["ChunkHit", "DocumentRepository"]
