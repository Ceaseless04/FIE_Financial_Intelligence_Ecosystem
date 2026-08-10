"""End-to-end ingestion and GraphRAG retrieval across both real stores.

Postgres and Neo4j are real; only the language model is a scripted double,
because a small local model gives no stable extraction to assert on. What is
being verified here is the wiring the fakes cannot show: that a chunk stored in
Postgres, an entity written to Neo4j, and a mention row joining them actually
compose into a retrievable, citable answer.

This is the Phase 2 gate — graph ingestion and retrieval must pass before merge.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from conftest import make_json_response, make_router
from fixtures.documents import acme_10k, acme_news, northwind_10k
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fie_database.config import Neo4jSettings, PostgresSettings
from fie_database.neo4j_client import Neo4jClient
from fie_database.postgres import PostgresDatabase
from fie_events.bus import InMemoryEventBus
from fie_testing import (
    DeterministicEmbeddingProvider,
    neo4j_endpoint,
    postgres_endpoint,
    require_service,
)
from marketmind import events as event_types
from marketmind.graph.repository import GraphRepository
from marketmind.ingestion.chunking import ChunkingConfig, chunk_document
from marketmind.ingestion.extraction import ExtractionService
from marketmind.ingestion.pipeline import IngestionPipeline
from marketmind.resolution.resolver import EntityResolver
from marketmind.retrieval.graphrag import GraphRAGConfig, GraphRAGService
from marketmind.storage.models import SCHEMA, ChunkRow, DocumentRow, EntityMentionRow
from marketmind.storage.repository import DocumentRepository

pytestmark = pytest.mark.integration

CHUNKING = ChunkingConfig(max_chars=600, overlap_chars=60)


@pytest.fixture(scope="module", autouse=True)
def _requires_infrastructure() -> None:
    require_service(postgres_endpoint())
    require_service(neo4j_endpoint())


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
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
async def graph() -> AsyncIterator[GraphRepository]:
    client = Neo4jClient(Neo4jSettings())
    repository = GraphRepository(client)
    await repository.delete_all()
    await repository.apply_schema()
    try:
        yield repository
    finally:
        await repository.delete_all()
        await client.aclose()


def acme_extraction(chunk_id: str) -> dict[str, Any]:
    """Structure a model would plausibly return for the Acme 10-K."""
    return {
        "entities": [
            {
                "name": "Acme Robotics Corporation",
                "type": "Company",
                "ticker": "ACME",
                "chunk_id": chunk_id,
            },
            {"name": "Sofia Marchetti", "type": "Executive", "chunk_id": chunk_id},
            {
                "name": "Northwind Components Ltd",
                "type": "Company",
                "chunk_id": chunk_id,
            },
        ],
        "relationships": [
            {
                "source_name": "Sofia Marchetti",
                "source_type": "Executive",
                "target_name": "Acme Robotics Corporation",
                "target_type": "Company",
                "type": "EXECUTIVE_OF",
                "chunk_id": chunk_id,
            },
            {
                "source_name": "Northwind Components Ltd",
                "source_type": "Company",
                "target_name": "Acme Robotics Corporation",
                "target_type": "Company",
                "type": "SUPPLIES",
                "chunk_id": chunk_id,
            },
        ],
    }


def build_pipeline(
    session: AsyncSession,
    graph: GraphRepository,
    embedder: DeterministicEmbeddingProvider,
    responses: list[Any],
    bus: InMemoryEventBus,
) -> IngestionPipeline:
    return IngestionPipeline(
        documents=DocumentRepository(session),
        graph=graph,
        embeddings=embedder,
        extraction=ExtractionService(make_router(responses), max_chunks_per_call=50),
        resolver=EntityResolver(),
        bus=bus,
        chunking=CHUNKING,
        embedding_batch_size=8,
    )


@pytest.fixture
def embedder() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(dimensions=768)


class TestEndToEndIngestion:
    async def test_a_filing_becomes_a_queryable_cited_graph(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        document = acme_10k()
        chunks = chunk_document(document, CHUNKING)
        bus = InMemoryEventBus()
        pipeline = build_pipeline(
            session,
            graph,
            embedder,
            [make_json_response(acme_extraction(chunks[0].id))],
            bus,
        )

        report = await pipeline.ingest(document, correlation_id="corr-e2e")
        await session.flush()

        assert not report.duplicate
        assert report.chunk_count == len(chunks)
        assert report.embedded_chunk_count == report.chunk_count
        assert report.entities_created == 3
        assert report.relationships_created == 2
        assert report.rejected == []

        stats = await graph.stats()
        assert stats["total_nodes"] == 3
        assert stats["total_relationships"] == 2
        assert stats["relationships_by_type"]["SUPPLIES"] == 1

        published = set(bus.counts_by_type)
        assert event_types.DOCUMENT_INGESTED in published
        assert event_types.GRAPH_UPDATED in published

    async def test_ingesting_twice_is_idempotent_across_both_stores(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        """Redelivery is normal; it must not double the graph or the chunks."""
        document = acme_10k()
        chunks = chunk_document(document, CHUNKING)
        response = make_json_response(acme_extraction(chunks[0].id))
        bus = InMemoryEventBus()
        pipeline = build_pipeline(session, graph, embedder, [response, response], bus)

        await pipeline.ingest(document)
        await session.flush()
        stats_before = await graph.stats()
        chunks_before = len(await DocumentRepository(session).get_chunks(document.id))

        second = await pipeline.ingest(acme_10k())
        await session.flush()

        assert second.duplicate
        assert await graph.stats() == stats_before
        assert len(await DocumentRepository(session).get_chunks(document.id)) == chunks_before

    async def test_a_second_filing_merges_into_the_existing_company(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        acme = acme_10k()
        northwind = northwind_10k()
        acme_chunks = chunk_document(acme, CHUNKING)
        northwind_chunks = chunk_document(northwind, CHUNKING)
        bus = InMemoryEventBus()

        pipeline = build_pipeline(
            session,
            graph,
            embedder,
            [
                make_json_response(acme_extraction(acme_chunks[0].id)),
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Northwind Components Ltd",
                                "type": "Company",
                                "ticker": "NWC",
                                "chunk_id": northwind_chunks[0].id,
                            }
                        ],
                        "relationships": [],
                    }
                ),
            ],
            bus,
        )

        await pipeline.ingest(acme)
        second = await pipeline.ingest(northwind)
        await session.flush()

        assert second.entities_created == 0
        assert second.entities_merged == 1

        stats = await graph.stats()
        assert stats["total_nodes"] == 3

        merged = bus.events_of_type(event_types.ENTITY_MERGED)
        assert merged
        surviving = merged[0].payload["surviving_entity_id"]
        node = await graph.find_by_id(surviving)
        assert node is not None
        # The merged node carries both filings and the ticker only the second
        # document supplied.
        assert {acme.id, northwind.id} <= set(node["source_ids"])
        assert "ticker:NWC" in node["identifier_keys"]

    async def test_mentions_survive_re_ingestion(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        """Stable chunk ids are what keep mention rows pointing at real chunks."""
        document = acme_news()
        chunks = chunk_document(document, CHUNKING)
        bus = InMemoryEventBus()
        pipeline = build_pipeline(
            session,
            graph,
            embedder,
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Acme Robotics Corporation",
                                "type": "Company",
                                "chunk_id": chunks[0].id,
                            }
                        ],
                        "relationships": [],
                    }
                )
            ],
            bus,
        )

        await pipeline.ingest(document)
        await session.flush()

        repository = DocumentRepository(session)
        stored_chunks = await repository.get_chunks(document.id)
        anchors = await repository.entity_ids_for_chunks([c.id for c in stored_chunks])

        assert anchors
        evidence = await repository.chunks_mentioning(anchors[0])
        assert evidence
        assert document.content[evidence[0].start_char : evidence[0].end_char] == evidence[0].text


class TestGraphRAG:
    async def test_retrieval_anchors_on_entities_and_expands_the_graph(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        document = acme_10k()
        chunks = chunk_document(document, CHUNKING)
        bus = InMemoryEventBus()
        pipeline = build_pipeline(
            session,
            graph,
            embedder,
            [make_json_response(acme_extraction(chunks[0].id))],
            bus,
        )
        await pipeline.ingest(document)
        await session.flush()

        service = GraphRAGService(
            documents=DocumentRepository(session),
            graph=graph,
            embeddings=embedder,
            router=make_router([]),
            config=GraphRAGConfig(top_k_chunks=4, minimum_similarity=0.0),
        )

        context = await service.retrieve("Who supplies Acme Robotics with bearings?")

        assert context.chunks, "vector search should seed the retrieval"
        assert context.entity_ids, "chunks should anchor onto graph entities"
        assert context.graph_facts, "traversal should contribute structure"
        assert any("supplies" in fact.lower() for fact in context.graph_facts)

    async def test_an_answer_is_verified_against_the_supplied_context(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        document = acme_10k()
        chunks = chunk_document(document, CHUNKING)
        bus = InMemoryEventBus()
        await build_pipeline(
            session,
            graph,
            embedder,
            [make_json_response(acme_extraction(chunks[0].id))],
            bus,
        ).ingest(document)
        await session.flush()

        answer_payload = {
            "answer": f"Northwind Components supplies Acme with bearings [{document.id}].",
            "cited_source_ids": [document.id, "doc-that-does-not-exist"],
            "inferences": ["A Northwind outage would constrain Acme's output."],
            "sufficient_context": True,
        }
        service = GraphRAGService(
            documents=DocumentRepository(session),
            graph=graph,
            embeddings=embedder,
            router=make_router([make_json_response(answer_payload)]),
            config=GraphRAGConfig(top_k_chunks=4, minimum_similarity=0.0),
        )

        result = await service.answer("Who supplies Acme Robotics?")

        # The invented id is stripped; the real one survives and resolves.
        assert result.cited_source_ids == [document.id]
        assert not result.grounded
        assert [s.source_id for s in result.provenance.sources] == [document.id]
        assert result.provenance.is_model_authored
        assert result.inferences

    async def test_an_unindexed_question_refuses_rather_than_inventing(
        self,
        session: AsyncSession,
        graph: GraphRepository,
        embedder: DeterministicEmbeddingProvider,
    ) -> None:
        from fie_common.errors import NotFoundError

        service = GraphRAGService(
            documents=DocumentRepository(session),
            graph=graph,
            embeddings=embedder,
            router=make_router([]),
            config=GraphRAGConfig(minimum_similarity=0.0),
        )

        with pytest.raises(NotFoundError):
            await service.answer("What is the capital structure of an unindexed company?")
