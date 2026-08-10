"""Document, chunk, and vector persistence against a real PostgreSQL + pgvector.

Cosine distance, `ON CONFLICT` upserts, cascade deletes, and the embedding-model
filter are all database semantics. Asserting them against an in-memory stand-in
would prove only that the stand-in agrees with itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fixtures.documents import acme_10k, acme_news, northwind_10k
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fie_common.errors import ValidationError
from fie_database.config import PostgresSettings
from fie_database.postgres import PostgresDatabase
from fie_testing import DeterministicEmbeddingProvider, postgres_endpoint, require_service
from marketmind.domain.documents import Document
from marketmind.ingestion.chunking import ChunkingConfig, chunk_document
from marketmind.storage.models import SCHEMA, ChunkRow, DocumentRow, EntityMentionRow
from marketmind.storage.repository import DocumentRepository

pytestmark = pytest.mark.integration

CHUNKING = ChunkingConfig(max_chars=600, overlap_chars=60)


@pytest.fixture(scope="module", autouse=True)
def _requires_postgres() -> None:
    require_service(postgres_endpoint())


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A schema-created session, rolled back and dropped after each test."""
    database = PostgresDatabase(PostgresSettings())
    async with database.engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(
            lambda sync_connection: DocumentRow.metadata.create_all(
                sync_connection,
                tables=[DocumentRow.__table__, ChunkRow.__table__, EntityMentionRow.__table__],
            )
        )

    factory = database.session_factory
    async with factory() as active:
        try:
            yield active
        finally:
            await active.rollback()

    async with database.engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await database.aclose()


@pytest.fixture
def embedder() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(dimensions=768)


async def store(
    session: AsyncSession, document: Document, embedder: DeterministicEmbeddingProvider
) -> tuple[str, list]:
    repository = DocumentRepository(session)
    document_id, _created = await repository.upsert_document(document)
    chunks = chunk_document(document, CHUNKING)
    vectors = await embedder.embed_texts([chunk.text for chunk in chunks])
    await repository.store_chunks(chunks, vectors, embedding_model=embedder.model)
    await session.flush()
    return document_id, chunks


class TestDocumentPersistence:
    async def test_a_document_round_trips(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        document = acme_10k()
        repository = DocumentRepository(session)

        document_id, created = await repository.upsert_document(document)
        await session.flush()
        loaded = await repository.get_document(document_id)

        assert created
        assert loaded is not None
        assert loaded.title == document.title
        assert loaded.content_hash == document.content_hash
        assert loaded.metadata["ticker"] == "ACME"

    async def test_content_hash_deduplicates_a_redelivery(self, session: AsyncSession) -> None:
        repository = DocumentRepository(session)
        original = acme_news()
        reissued = Document(type=original.type, title=original.title, content=original.content)

        first_id, first_created = await repository.upsert_document(original)
        await session.flush()
        second_id, second_created = await repository.upsert_document(reissued)

        assert first_created
        assert not second_created
        assert second_id == first_id

    async def test_missing_document_returns_none(self, session: AsyncSession) -> None:
        assert await DocumentRepository(session).get_document("doc_missing") is None


class TestChunkPersistence:
    async def test_chunks_persist_with_their_offsets(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        document = acme_10k()
        _document_id, chunks = await store(session, document, embedder)

        loaded = await DocumentRepository(session).get_chunks(document.id)

        assert len(loaded) == len(chunks)
        assert [c.index for c in loaded] == list(range(len(chunks)))
        for chunk in loaded:
            assert document.content[chunk.start_char : chunk.end_char] == chunk.text

    async def test_re_ingestion_updates_in_place(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        """A re-chunk or a new embedding model must not duplicate rows."""
        document = acme_10k()
        await store(session, document, embedder)
        before = len(await DocumentRepository(session).get_chunks(document.id))

        await store(session, document, embedder)
        after = await DocumentRepository(session).get_chunks(document.id)

        assert len(after) == before

    async def test_mismatched_embedding_count_is_refused(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        document = acme_10k()
        chunks = chunk_document(document, CHUNKING)
        assert len(chunks) > 1, "fixture must produce several chunks for this test"
        repository = DocumentRepository(session)
        await repository.upsert_document(document)

        with pytest.raises(ValidationError, match="embedding count"):
            await repository.store_chunks(chunks, [[0.0] * 768], embedding_model=embedder.model)

    async def test_deleting_a_document_cascades_to_its_chunks(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        document = acme_10k()
        await store(session, document, embedder)
        repository = DocumentRepository(session)

        await repository.delete_document(document.id)
        await session.flush()

        assert await repository.get_chunks(document.id) == []


class TestVectorSearch:
    async def test_search_finds_the_semantically_closest_chunk(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        """Deterministic embeddings still put shared tokens closer together."""
        document = acme_10k()
        await store(session, document, embedder)
        repository = DocumentRepository(session)

        query = await embedder.embed_text(
            "Northwind Components precision bearings supply disruption"
        )
        hits = await repository.search_similar(query, limit=3, embedding_model=embedder.model)

        assert hits
        assert "Northwind" in hits[0].text
        assert hits[0].similarity > 0

    async def test_results_are_ordered_by_distance(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        await store(session, acme_10k(), embedder)
        query = await embedder.embed_text("competition industrial automation market")

        hits = await DocumentRepository(session).search_similar(
            query, limit=5, embedding_model=embedder.model
        )

        distances = [hit.distance for hit in hits]
        assert distances == sorted(distances)

    async def test_the_embedding_model_filter_excludes_incomparable_vectors(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        """Vectors from two models are not comparable; mixing them is nonsense."""
        await store(session, acme_10k(), embedder)
        query = await embedder.embed_text("robotics")

        matching = await DocumentRepository(session).search_similar(
            query, limit=5, embedding_model=embedder.model
        )
        mismatched = await DocumentRepository(session).search_similar(
            query, limit=5, embedding_model="some-other-model"
        )

        assert matching
        assert mismatched == []

    async def test_search_can_be_scoped_to_documents(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        acme = acme_10k()
        northwind = northwind_10k()
        await store(session, acme, embedder)
        await store(session, northwind, embedder)
        query = await embedder.embed_text("precision bearings")

        hits = await DocumentRepository(session).search_similar(
            query, limit=10, embedding_model=embedder.model, document_ids=[northwind.id]
        )

        assert hits
        assert {hit.document_id for hit in hits} == {northwind.id}

    async def test_a_hit_resolves_back_to_the_source_span(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        """The property that makes a citation verifiable rather than decorative."""
        document = acme_10k()
        await store(session, document, embedder)
        query = await embedder.embed_text("Sofia Marchetti Chief Executive Officer")

        hits = await DocumentRepository(session).search_similar(
            query, limit=1, embedding_model=embedder.model
        )

        hit = hits[0]
        assert document.content[hit.start_char : hit.end_char] == hit.text

    async def test_similarity_is_bounded_to_the_unit_interval(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        await store(session, acme_10k(), embedder)
        query = await embedder.embed_text("anything at all")

        hits = await DocumentRepository(session).search_similar(
            query, limit=5, embedding_model=embedder.model
        )

        assert all(0.0 <= hit.similarity <= 1.0 for hit in hits)


class TestMentions:
    async def test_mentions_link_an_entity_to_its_chunks(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        document = acme_10k()
        _document_id, chunks = await store(session, document, embedder)
        repository = DocumentRepository(session)

        await repository.record_mentions(
            "ent-acme", "Company", "Acme Robotics Corporation", [chunks[0].id, chunks[1].id]
        )
        await session.flush()

        found = await repository.chunks_mentioning("ent-acme")
        assert {chunk.id for chunk in found} == {chunks[0].id, chunks[1].id}

    async def test_recording_the_same_mention_twice_is_harmless(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        """At-least-once delivery means handlers re-run; they must be idempotent."""
        document = acme_news()
        _document_id, chunks = await store(session, document, embedder)
        repository = DocumentRepository(session)

        await repository.record_mentions("ent-acme", "Company", "Acme", [chunks[0].id])
        await repository.record_mentions("ent-acme", "Company", "Acme", [chunks[0].id])
        await session.flush()

        assert len(await repository.chunks_mentioning("ent-acme")) == 1

    async def test_entity_ids_for_chunks_powers_graph_anchoring(
        self, session: AsyncSession, embedder: DeterministicEmbeddingProvider
    ) -> None:
        document = acme_10k()
        _document_id, chunks = await store(session, document, embedder)
        repository = DocumentRepository(session)
        await repository.record_mentions("ent-acme", "Company", "Acme", [chunks[0].id])
        await repository.record_mentions("ent-sofia", "Executive", "Sofia", [chunks[0].id])
        await session.flush()

        anchors = await repository.entity_ids_for_chunks([chunks[0].id])

        assert set(anchors) == {"ent-acme", "ent-sofia"}

    async def test_no_chunks_means_no_query(self, session: AsyncSession) -> None:
        assert await DocumentRepository(session).entity_ids_for_chunks([]) == []
