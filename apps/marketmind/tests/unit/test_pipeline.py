"""Ingestion pipeline orchestration.

The stores here are in-memory but behaviourally faithful — content-hash
deduplication, idempotent upserts, real mention rows — because the properties
under test are exactly those behaviours. A store that records calls without
enforcing its own rules would let every one of these tests pass against broken
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from conftest import make_json_response, make_router
from fixtures.documents import acme_10k, acme_news

from fie_common.errors import EventBusError, ValidationError
from fie_events.bus import InMemoryEventBus
from fie_events.schemas import DomainEvent
from fie_testing import DeterministicEmbeddingProvider
from marketmind import events as event_types
from marketmind.domain.documents import Document
from marketmind.domain.entities import Entity, EntityType
from marketmind.ingestion.chunking import ChunkingConfig
from marketmind.ingestion.extraction import ExtractionService
from marketmind.ingestion.pipeline import IngestionPipeline
from marketmind.resolution.resolver import EntityResolver

pytestmark = pytest.mark.unit


@dataclass
class InMemoryDocuments:
    """Faithful stand-in for DocumentRepository: dedupes, stores, links."""

    by_id: dict[str, Document] = field(default_factory=dict)
    by_hash: dict[str, str] = field(default_factory=dict)
    chunks: dict[str, list[Any]] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    mentions: list[tuple[str, str, str, tuple[str, ...]]] = field(default_factory=list)

    async def upsert_document(self, document: Document) -> tuple[str, bool]:
        existing = self.by_hash.get(document.content_hash)
        if existing is not None:
            return existing, False
        self.by_id[document.id] = document
        self.by_hash[document.content_hash] = document.id
        return document.id, True

    async def get_document(self, document_id: str) -> Document | None:
        return self.by_id.get(document_id)

    async def store_chunks(
        self,
        chunks: Any,
        vectors: Any = None,
        *,
        embedding_model: str | None = None,
    ) -> int:
        chunk_list = list(chunks)
        if vectors is not None and len(list(vectors)) != len(chunk_list):
            raise ValidationError("embedding count does not match chunk count")
        for position, chunk in enumerate(chunk_list):
            self.chunks.setdefault(chunk.document_id, []).append(chunk)
            if vectors is not None:
                self.embeddings[chunk.id] = list(vectors)[position]
        return len(chunk_list)

    async def record_mentions(
        self, entity_id: str, entity_type: str, surface_form: str, chunk_ids: Any
    ) -> int:
        self.mentions.append((entity_id, entity_type, surface_form, tuple(chunk_ids)))
        return len(list(chunk_ids))

    async def entity_ids_for_chunks(self, chunk_ids: Any) -> list[str]:
        wanted = set(chunk_ids)
        return [entity_id for entity_id, _type, _form, ids in self.mentions if wanted & set(ids)]


@dataclass
class InMemoryGraph:
    """Faithful stand-in for GraphRepository: idempotent, id-keyed."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    fail_relationships: bool = False

    async def upsert_entity(self, entity: Entity) -> str:
        existing = self.nodes.get(entity.id, {})
        self.nodes[entity.id] = {
            "id": entity.id,
            "label": str(entity.type),
            "name": entity.name,
            "canonical_name": entity.canonical_name,
            "aliases": sorted({*existing.get("aliases", []), *entity.aliases}),
            "identifier_keys": sorted(
                {*existing.get("identifier_keys", []), *entity.identifier_keys}
            ),
            "source_ids": sorted(
                {
                    *existing.get("source_ids", []),
                    *(source.source_id for source in entity.provenance.sources),
                }
            ),
        }
        return entity.id

    async def upsert_relationship(self, relationship: Any) -> str:
        if self.fail_relationships:
            raise RuntimeError("neo4j write failed")
        self.edges[relationship.dedupe_key] = {
            "id": relationship.id,
            "type": str(relationship.type),
            "source": relationship.source_entity_id,
            "target": relationship.target_entity_id,
        }
        return str(relationship.id)

    async def upsert_relationships(self, relationships: Any) -> list[str]:
        written: list[str] = []
        for relationship in relationships:
            try:
                written.append(await self.upsert_relationship(relationship))
            except RuntimeError:
                continue
        return written

    async def find_by_identifier(
        self, entity_type: EntityType, identifier_key: str
    ) -> dict[str, Any] | None:
        for node in self.nodes.values():
            if node["label"] == str(entity_type) and identifier_key in node["identifier_keys"]:
                return node
        return None

    async def find_by_canonical_name(
        self, entity_type: EntityType, canonical_name: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes.values()
            if node["label"] == str(entity_type) and node["canonical_name"] == canonical_name
        ][:limit]


class ExplodingBus(InMemoryEventBus):
    async def publish(self, event: DomainEvent) -> str:
        raise EventBusError("redis is unreachable")


def acme_extraction(chunk_id: str) -> dict[str, Any]:
    return {
        "entities": [
            {
                "name": "Acme Robotics Corporation",
                "type": "Company",
                "ticker": "ACME",
                "chunk_id": chunk_id,
            },
            {"name": "Sofia Marchetti", "type": "Executive", "chunk_id": chunk_id},
        ],
        "relationships": [
            {
                "source_name": "Sofia Marchetti",
                "source_type": "Executive",
                "target_name": "Acme Robotics Corporation",
                "target_type": "Company",
                "type": "EXECUTIVE_OF",
                "chunk_id": chunk_id,
            }
        ],
    }


def build_pipeline(
    responses: list[Any],
    *,
    documents: InMemoryDocuments | None = None,
    graph: InMemoryGraph | None = None,
    bus: Any = None,
    max_extraction_chunks: int = 60,
) -> tuple[IngestionPipeline, InMemoryDocuments, InMemoryGraph, Any]:
    store = documents or InMemoryDocuments()
    knowledge = graph or InMemoryGraph()
    event_bus = bus if bus is not None else InMemoryEventBus()
    pipeline = IngestionPipeline(
        documents=store,  # type: ignore[arg-type]
        graph=knowledge,  # type: ignore[arg-type]
        embeddings=DeterministicEmbeddingProvider(dimensions=64),
        extraction=ExtractionService(make_router(responses), max_chunks_per_call=50),
        resolver=EntityResolver(),
        bus=event_bus,
        chunking=ChunkingConfig(max_chars=1200, overlap_chars=100),
        embedding_batch_size=4,
        max_extraction_chunks=max_extraction_chunks,
    )
    return pipeline, store, knowledge, event_bus


class TestHappyPath:
    async def test_ingestion_stores_chunks_entities_and_edges(self) -> None:
        document = acme_10k()
        pipeline, store, graph, _bus = build_pipeline([])
        # The chunk ids are only known after chunking, so the scripted response
        # is built from the store's own output.
        pipeline.extraction = ExtractionService(
            make_router([make_json_response(acme_extraction("PLACEHOLDER"))]),
            max_chunks_per_call=50,
        )

        report = await pipeline.ingest(document)

        assert not report.duplicate
        assert report.chunk_count > 0
        assert report.embedded_chunk_count == report.chunk_count
        assert report.is_retrievable
        assert len(store.chunks[document.id]) == report.chunk_count
        # The placeholder chunk id is rejected, which is itself the guarantee:
        # an uncited extraction never reaches the graph.
        assert report.rejection_count > 0
        assert graph.nodes == {}

    async def test_extraction_with_real_chunk_ids_populates_the_graph(self) -> None:
        document = acme_10k()
        store = InMemoryDocuments()
        graph = InMemoryGraph()

        # Chunk first so the scripted extraction can cite a real chunk id.
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        pipeline, _, _, _bus = build_pipeline(
            [make_json_response(acme_extraction(chunks[0].id))],
            documents=store,
            graph=graph,
        )

        report = await pipeline.ingest(document)

        assert report.entities_created == 2
        assert report.relationships_created == 1
        assert report.rejected == []
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        company = next(n for n in graph.nodes.values() if n["label"] == "Company")
        assert "ticker:ACME" in company["identifier_keys"]
        assert document.id in company["source_ids"]

    async def test_mentions_link_entities_back_to_their_chunks(self) -> None:
        document = acme_10k()
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        pipeline, store, _, _ = build_pipeline([make_json_response(acme_extraction(chunks[0].id))])

        await pipeline.ingest(document)

        assert store.mentions
        for _entity_id, _type, _form, chunk_ids in store.mentions:
            assert chunk_ids
            assert all(chunk_id.startswith("chk") for chunk_id in chunk_ids)


class TestDeduplication:
    async def test_a_redelivered_document_does_no_work(self) -> None:
        document = acme_10k()
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        pipeline, store, _graph, bus = build_pipeline(
            [make_json_response(acme_extraction(chunks[0].id))]
        )

        first = await pipeline.ingest(document)
        chunk_count_after_first = len(store.chunks[document.id])
        second = await pipeline.ingest(acme_10k())

        assert not first.duplicate
        assert second.duplicate
        assert second.document_id == first.document_id
        assert len(store.chunks[document.id]) == chunk_count_after_first
        assert bus.events_of_type(event_types.DOCUMENT_DUPLICATE)

    async def test_duplicate_detection_is_content_based_not_id_based(self) -> None:
        """A feed reassigning an id must not create a second copy."""
        pipeline, _store, _, _ = build_pipeline([])
        original = acme_news()
        reissued = Document(type=original.type, title=original.title, content=original.content)
        assert reissued.id != original.id

        await pipeline.ingest(original)
        report = await pipeline.ingest(reissued)

        assert report.duplicate
        assert report.document_id == original.id


class TestResolutionEffects:
    async def test_edges_between_merged_entities_are_dropped_with_a_reason(self) -> None:
        """Merging two names into one turns an edge between them into a loop."""
        document = acme_news()
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        chunk_id = chunks[0].id
        pipeline, _, graph, _ = build_pipeline(
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Acme Robotics Corporation",
                                "type": "Company",
                                "chunk_id": chunk_id,
                            },
                            {
                                "name": "Acme Robotics Corp",
                                "type": "Company",
                                "chunk_id": chunk_id,
                            },
                        ],
                        "relationships": [
                            {
                                "source_name": "Acme Robotics Corporation",
                                "source_type": "Company",
                                "target_name": "Acme Robotics Corp",
                                "target_type": "Company",
                                "type": "COMPETES_WITH",
                                "chunk_id": chunk_id,
                            }
                        ],
                    }
                )
            ]
        )

        report = await pipeline.ingest(document)

        assert len(graph.nodes) == 1
        assert graph.edges == {}
        assert report.relationships_created == 0
        assert any("after resolution" in reason for reason in report.rejected)

    async def test_a_second_document_merges_into_the_existing_node(self) -> None:
        from marketmind.ingestion.chunking import chunk_document

        first_document = acme_10k()
        second_document = acme_news()
        first_chunks = chunk_document(
            first_document, ChunkingConfig(max_chars=1200, overlap_chars=100)
        )
        second_chunks = chunk_document(
            second_document, ChunkingConfig(max_chars=1200, overlap_chars=100)
        )

        pipeline, _, graph, bus = build_pipeline(
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Acme Robotics Corporation",
                                "type": "Company",
                                "chunk_id": first_chunks[0].id,
                            }
                        ],
                        "relationships": [],
                    }
                ),
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Acme Robotics Corp",
                                "type": "Company",
                                "chunk_id": second_chunks[0].id,
                            }
                        ],
                        "relationships": [],
                    }
                ),
            ]
        )

        first_report = await pipeline.ingest(first_document)
        second_report = await pipeline.ingest(second_document)

        assert first_report.entities_created == 1
        assert second_report.entities_created == 0
        assert second_report.entities_merged == 1
        assert len(graph.nodes) == 1

        merged_events = bus.events_of_type(event_types.ENTITY_MERGED)
        assert merged_events
        payload = merged_events[0].payload
        assert payload["surviving_entity_id"] != payload["merged_entity_id"]
        assert payload["match_reason"]

        node = next(iter(graph.nodes.values()))
        assert {first_document.id, second_document.id} <= set(node["source_ids"])


class TestEventsAndFailures:
    async def test_ingestion_publishes_the_expected_event_types(self) -> None:
        document = acme_10k()
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        pipeline, _, _, bus = build_pipeline([make_json_response(acme_extraction(chunks[0].id))])

        await pipeline.ingest(document, correlation_id="corr-xyz")

        published = set(bus.counts_by_type)
        assert event_types.ENTITY_CREATED in published
        assert event_types.RELATIONSHIP_CREATED in published
        assert event_types.DOCUMENT_INGESTED in published
        assert event_types.GRAPH_UPDATED in published
        assert all(event.correlation_id == "corr-xyz" for event in bus.published)

    async def test_a_bus_outage_does_not_fail_a_durable_ingestion(self) -> None:
        """The data is already written; losing a notification is the lesser harm."""
        document = acme_10k()
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        pipeline, _, graph, _ = build_pipeline(
            [make_json_response(acme_extraction(chunks[0].id))], bus=ExplodingBus()
        )

        report = await pipeline.ingest(document)

        assert report.entities_created == 2
        assert len(graph.nodes) == 2

    async def test_a_failed_relationship_write_does_not_abandon_the_batch(self) -> None:
        document = acme_10k()
        from marketmind.ingestion.chunking import chunk_document

        chunks = chunk_document(document, ChunkingConfig(max_chars=1200, overlap_chars=100))
        graph = InMemoryGraph(fail_relationships=True)
        pipeline, _, _, _ = build_pipeline(
            [make_json_response(acme_extraction(chunks[0].id))], graph=graph
        )

        report = await pipeline.ingest(document)

        assert report.entities_created == 2
        assert report.relationships_created == 0

    async def test_broken_chunk_offsets_fail_the_unit_of_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unverifiable citations must roll the document back, not be stored."""
        import marketmind.ingestion.pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "verify_chunk_offsets", lambda *_: False)
        pipeline, _, graph, _ = build_pipeline([])

        with pytest.raises(ValidationError, match="unverifiable"):
            await pipeline.ingest(acme_10k())

        assert graph.nodes == {}

    async def test_extraction_is_capped_per_document(self) -> None:
        """A 300-page filing must not become hundreds of model calls."""
        document = acme_10k()
        pipeline, _, _, _ = build_pipeline([], max_extraction_chunks=2)
        captured: list[int] = []

        original = pipeline.extraction.extract

        async def counting_extract(doc: Document, chunks: Any) -> Any:
            captured.append(len(list(chunks)))
            return await original(doc, [])

        pipeline.extraction.extract = counting_extract  # type: ignore[method-assign]

        await pipeline.ingest(document)

        assert captured == [2]

    async def test_no_bus_configured_is_not_an_error(self) -> None:
        document = acme_news()
        pipeline, _, _, _ = build_pipeline([], bus=None)
        pipeline.bus = None

        report = await pipeline.ingest(document)

        assert report.chunk_count > 0
