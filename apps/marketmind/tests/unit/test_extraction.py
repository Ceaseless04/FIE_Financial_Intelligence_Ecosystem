"""Extraction guardrails.

AI output is never asserted by exact text. What is asserted is structure: the
schema parsed, the citations resolved to chunks that were actually supplied, and
edges whose endpoint types are nonsensical never reached the graph.
"""

from __future__ import annotations

import pytest
from conftest import make_json_response, make_router
from fixtures.documents import acme_10k

from fie_ai.contracts import CompletionResponse, StopReason
from fie_common.errors import AIProviderError
from marketmind.domain.entities import EntityType, IdentifierType
from marketmind.domain.relationships import RelationshipType
from marketmind.ingestion.chunking import ChunkingConfig, chunk_document
from marketmind.ingestion.extraction import EXTRACTION_SYSTEM_PROMPT, ExtractionService

pytestmark = pytest.mark.unit


@pytest.fixture
def document_and_chunks() -> tuple[object, list[object]]:
    document = acme_10k()
    chunks = chunk_document(document, ChunkingConfig(max_chars=600, overlap_chars=60))
    return document, chunks


class TestExtractionPrompt:
    def test_prompt_forbids_evaluation(self) -> None:
        """The scope boundary is stated to the model, not only enforced after."""
        lowered = EXTRACTION_SYSTEM_PROMPT.lower()
        for forbidden in ("rating", "recommendation", "price target"):
            assert forbidden in lowered

    def test_prompt_requires_citation(self) -> None:
        assert "chunk_id" in EXTRACTION_SYSTEM_PROMPT
        assert "never invent" in EXTRACTION_SYSTEM_PROMPT.lower()


class TestExtractionOutcome:
    async def test_valid_extraction_becomes_domain_objects(
        self, document_and_chunks: tuple
    ) -> None:
        document, chunks = document_and_chunks
        chunk_id = chunks[0].id
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Acme Robotics Corporation",
                                "type": "Company",
                                "ticker": "ACME",
                                "chunk_id": chunk_id,
                            },
                            {
                                "name": "Sofia Marchetti",
                                "type": "Executive",
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
                            }
                        ],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        assert len(outcome.entities) == 2
        assert len(outcome.relationships) == 1
        assert outcome.rejected == []

        company = next(e for e in outcome.entities if e.type is EntityType.COMPANY)
        assert company.identifier_of(IdentifierType.TICKER) == "ACME"

        relationship = outcome.relationships[0]
        assert relationship.type is RelationshipType.EXECUTIVE_OF
        assert relationship.source_entity_id != relationship.target_entity_id

    async def test_every_entity_carries_a_citable_source(self, document_and_chunks: tuple) -> None:
        document, chunks = document_and_chunks
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [{"name": "Acme", "type": "Company", "chunk_id": chunks[0].id}],
                        "relationships": [],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        provenance = outcome.entities[0].provenance
        assert provenance.sources
        locator = provenance.sources[0].locator
        assert locator is not None
        assert locator.start_char == chunks[0].start_char
        assert locator.end_char == chunks[0].end_char

    async def test_fabricated_citations_are_rejected(self, document_and_chunks: tuple) -> None:
        """A cited chunk that was never supplied is a hallucination, not data."""
        document, chunks = document_and_chunks
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Invented Holdings",
                                "type": "Company",
                                "chunk_id": "chk_does_not_exist",
                            }
                        ],
                        "relationships": [],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        assert outcome.entities == []
        assert outcome.rejection_count == 1
        assert "unknown chunk" in outcome.rejected[0]

    async def test_schema_violating_relationship_is_dropped(
        self, document_and_chunks: tuple
    ) -> None:
        """An `Industry SUPPLIES Executive` edge corrupts every traversal."""
        document, chunks = document_and_chunks
        chunk_id = chunks[0].id
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Industrial Automation",
                                "type": "Industry",
                                "chunk_id": chunk_id,
                            },
                            {
                                "name": "Sofia Marchetti",
                                "type": "Executive",
                                "chunk_id": chunk_id,
                            },
                        ],
                        "relationships": [
                            {
                                "source_name": "Industrial Automation",
                                "source_type": "Industry",
                                "target_name": "Sofia Marchetti",
                                "target_type": "Executive",
                                "type": "SUPPLIES",
                                "chunk_id": chunk_id,
                            }
                        ],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        assert outcome.relationships == []
        assert any("not valid between" in reason for reason in outcome.rejected)

    async def test_relationship_to_an_unextracted_entity_is_dropped(
        self, document_and_chunks: tuple
    ) -> None:
        document, chunks = document_and_chunks
        chunk_id = chunks[0].id
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [{"name": "Acme", "type": "Company", "chunk_id": chunk_id}],
                        "relationships": [
                            {
                                "source_name": "Acme",
                                "source_type": "Company",
                                "target_name": "Ghost Corp",
                                "target_type": "Company",
                                "type": "SUPPLIES",
                                "chunk_id": chunk_id,
                            }
                        ],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        assert outcome.relationships == []
        assert any("was not extracted" in reason for reason in outcome.rejected)

    async def test_duplicate_entities_within_a_batch_collapse(
        self, document_and_chunks: tuple
    ) -> None:
        document, chunks = document_and_chunks
        chunk_id = chunks[0].id
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [
                            {"name": "Acme", "type": "Company", "chunk_id": chunk_id},
                            {"name": "acme", "type": "Company", "chunk_id": chunk_id},
                        ],
                        "relationships": [],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        assert len(outcome.entities) == 1

    async def test_malformed_batch_does_not_lose_the_document(
        self, document_and_chunks: tuple
    ) -> None:
        """One unparsable response must not discard the other batches."""
        document, chunks = document_and_chunks
        batch = chunks[:4]
        good = make_json_response(
            {
                "entities": [{"name": "Acme", "type": "Company", "chunk_id": batch[2].id}],
                "relationships": [],
            }
        )
        garbage = CompletionResponse(
            text="not json at all",
            model="fake-model-1",
            provider="fake",
            stop_reason=StopReason.END_TURN,
        )
        # Four chunks at two per call: the first batch fails, the second must
        # still be extracted rather than abandoning the document.
        router = make_router([garbage, good])

        service = ExtractionService(router, max_chunks_per_call=2)
        outcome = await service.extract(document, batch)

        assert len(outcome.entities) == 1
        assert outcome.rejection_count == 1

    async def test_refusal_is_recorded_not_swallowed(self, document_and_chunks: tuple) -> None:
        document, chunks = document_and_chunks
        refusal = CompletionResponse(
            text="",
            model="fake-model-1",
            provider="fake",
            stop_reason=StopReason.REFUSAL,
            refusal_category="policy",
        )
        outcome = await ExtractionService(make_router([refusal])).extract(document, chunks[:1])

        assert outcome.entities == []
        assert outcome.rejection_count == 1

    async def test_no_chunks_means_no_model_call(self) -> None:
        router = make_router([])
        outcome = await ExtractionService(router).extract(acme_10k(), [])
        assert outcome.entities == []
        assert outcome.relationships == []

    async def test_provider_failure_propagates(self, document_and_chunks: tuple) -> None:
        """A transport failure is not the same as a rejected extraction."""
        document, chunks = document_and_chunks
        router = make_router([AIProviderError("upstream is down")])
        with pytest.raises(AIProviderError):
            await ExtractionService(router).extract(document, chunks[:1])

    async def test_malformed_ticker_does_not_fail_the_entity(
        self, document_and_chunks: tuple
    ) -> None:
        document, chunks = document_and_chunks
        router = make_router(
            [
                make_json_response(
                    {
                        "entities": [
                            {
                                "name": "Acme",
                                "type": "Company",
                                "ticker": "not a ticker!",
                                "chunk_id": chunks[0].id,
                            }
                        ],
                        "relationships": [],
                    }
                )
            ]
        )

        outcome = await ExtractionService(router).extract(document, chunks[:1])

        assert len(outcome.entities) == 1
        assert outcome.entities[0].identifiers == []
