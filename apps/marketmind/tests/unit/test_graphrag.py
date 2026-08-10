"""GraphRAG: context assembly and citation verification.

The property under test is not answer quality — a fake model has none — but that
a fabricated citation never reaches a caller, and that an empty context never
becomes an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from conftest import make_json_response, make_router

from fie_ai.contracts import CompletionResponse, StopReason
from fie_common.errors import AIProviderResponseError, NotFoundError
from fie_testing import DeterministicEmbeddingProvider
from marketmind.domain.documents import Document, DocumentType
from marketmind.graph.repository import GraphEdge, GraphNode, Neighbourhood
from marketmind.retrieval.graphrag import (
    SYNTHESIS_SYSTEM_PROMPT,
    GraphRAGConfig,
    GraphRAGService,
    RetrievedContext,
)
from marketmind.storage.repository import ChunkHit

pytestmark = pytest.mark.unit


@dataclass
class StubDocuments:
    """Minimal stand-in for DocumentRepository's read surface."""

    hits: list[ChunkHit] = field(default_factory=list)
    documents: dict[str, Document] = field(default_factory=dict)
    entity_ids: list[str] = field(default_factory=list)

    async def search_similar(self, _vector: list[float], **_kwargs: Any) -> list[ChunkHit]:
        return list(self.hits)

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def entity_ids_for_chunks(self, _chunk_ids: list[str]) -> list[str]:
        return list(self.entity_ids)


@dataclass
class StubGraph:
    """Minimal stand-in for GraphRepository's traversal surface."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    async def neighbourhood(self, entity_id: str, **_kwargs: Any) -> Neighbourhood:
        return Neighbourhood(seed_entity_id=entity_id, nodes=list(self.nodes))

    async def relationships_of(self, _entity_id: str, **_kwargs: Any) -> list[GraphEdge]:
        return list(self.edges)


def make_hit(document_id: str, text: str, *, distance: float = 0.1) -> ChunkHit:
    return ChunkHit(
        chunk_id=f"chk-{document_id}",
        document_id=document_id,
        text=text,
        start_char=0,
        end_char=len(text),
        distance=distance,
    )


def make_document(document_id: str) -> Document:
    return Document(
        id=document_id,
        type=DocumentType.SEC_FILING,
        title=f"Filing {document_id}",
        content="Acme Robotics Corporation reported revenue growth.",
        uri=f"https://example.invalid/{document_id}",
    )


def build_service(
    *,
    hits: list[ChunkHit],
    responses: list[CompletionResponse | Exception],
    documents: dict[str, Document] | None = None,
    nodes: list[GraphNode] | None = None,
    edges: list[GraphEdge] | None = None,
    config: GraphRAGConfig | None = None,
) -> GraphRAGService:
    return GraphRAGService(
        documents=StubDocuments(hits=hits, documents=documents or {}),  # type: ignore[arg-type]
        graph=StubGraph(nodes=nodes or [], edges=edges or []),  # type: ignore[arg-type]
        embeddings=DeterministicEmbeddingProvider(dimensions=64),
        router=make_router(responses),
        config=config or GraphRAGConfig(),
    )


class TestSynthesisPrompt:
    def test_prompt_forbids_investment_advice(self) -> None:
        lowered = SYNTHESIS_SYSTEM_PROMPT.lower()
        assert "investment advice" in lowered
        assert "price target" in lowered

    def test_prompt_requires_citations_from_context_only(self) -> None:
        lowered = SYNTHESIS_SYSTEM_PROMPT.lower()
        assert "only the supplied context" in lowered
        assert "never invent" in lowered


class TestContextRendering:
    def test_source_ids_are_deduplicated_and_ordered(self) -> None:
        context = RetrievedContext(
            question="q",
            chunks=[make_hit("doc-1", "a"), make_hit("doc-2", "b"), make_hit("doc-1", "c")],
        )
        assert context.source_ids == ["doc-1", "doc-2"]

    def test_rendering_tags_each_passage_with_its_source(self) -> None:
        rendered = RetrievedContext(
            question="q", chunks=[make_hit("doc-1", "Revenue rose.")]
        ).render()
        assert 'source_id="doc-1"' in rendered
        assert "Revenue rose." in rendered

    def test_graph_facts_are_rendered_for_the_model(self) -> None:
        rendered = RetrievedContext(question="q", graph_facts=["Acme supplies Zenith"]).render()
        assert "Acme supplies Zenith" in rendered

    def test_context_with_neither_chunks_nor_facts_is_empty(self) -> None:
        assert RetrievedContext(question="q").is_empty


class TestRetrieval:
    async def test_low_similarity_chunks_are_dropped(self) -> None:
        service = build_service(
            hits=[
                make_hit("doc-1", "close", distance=0.1),
                make_hit("doc-2", "far", distance=0.99),
            ],
            responses=[],
            config=GraphRAGConfig(minimum_similarity=0.5),
        )

        context = await service.retrieve("what does Acme make?")

        assert [hit.document_id for hit in context.chunks] == ["doc-1"]

    async def test_supplied_entity_ids_seed_the_traversal(self) -> None:
        service = build_service(
            hits=[make_hit("doc-1", "text")],
            responses=[],
            nodes=[GraphNode(id="ent-2", label="Company", name="Zenith Automation")],
            edges=[
                GraphEdge(
                    type="COMPETES_WITH",
                    other_id="ent-2",
                    other_label="Company",
                    other_name="Zenith Automation",
                )
            ],
        )

        context = await service.retrieve("who competes with Acme?", entity_ids=["ent-1"])

        assert "ent-1" in context.entity_ids
        assert context.graph_facts
        assert "Zenith Automation" in context.graph_facts[0]

    async def test_edge_direction_is_rendered_correctly(self) -> None:
        service = build_service(
            hits=[make_hit("doc-1", "text")],
            responses=[],
            nodes=[GraphNode(id="ent-1", label="Company", name="Acme Robotics")],
            edges=[
                GraphEdge(
                    type="SUPPLIES",
                    other_id="ent-2",
                    other_label="Company",
                    other_name="Northwind Components",
                    is_outgoing=False,
                )
            ],
        )

        context = await service.retrieve("who supplies Acme?", entity_ids=["ent-1"])

        assert context.graph_facts == ["Northwind Components supplies Acme Robotics"]


class TestAnswerVerification:
    async def test_a_grounded_answer_returns_its_citations(self) -> None:
        service = build_service(
            hits=[make_hit("doc-1", "Acme reported revenue of $412.6 million.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[
                make_json_response(
                    {
                        "answer": "Acme reported revenue of $412.6 million [doc-1].",
                        "cited_source_ids": ["doc-1"],
                        "inferences": [],
                        "sufficient_context": True,
                    }
                )
            ],
        )

        result = await service.answer("what was Acme's revenue?")

        assert result.grounded
        assert result.is_answerable
        assert result.cited_source_ids == ["doc-1"]
        assert result.provenance.is_model_authored
        assert [s.source_id for s in result.provenance.sources] == ["doc-1"]

    async def test_a_fabricated_citation_is_stripped_and_flagged(self) -> None:
        """A caller must never receive a reference it cannot follow."""
        service = build_service(
            hits=[make_hit("doc-1", "Acme reported revenue growth.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[
                make_json_response(
                    {
                        "answer": "Revenue grew [doc-1], and margins expanded [doc-99].",
                        "cited_source_ids": ["doc-1", "doc-99"],
                        "inferences": [],
                        "sufficient_context": True,
                    }
                )
            ],
        )

        result = await service.answer("how did Acme perform?")

        assert result.cited_source_ids == ["doc-1"]
        assert not result.grounded
        assert not result.is_answerable

    async def test_inferences_are_kept_separate_from_the_answer(self) -> None:
        service = build_service(
            hits=[make_hit("doc-1", "Acme depends on Northwind for bearings.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[
                make_json_response(
                    {
                        "answer": "Acme sources bearings from Northwind [doc-1].",
                        "cited_source_ids": ["doc-1"],
                        "inferences": ["A Northwind disruption would affect Acme output."],
                        "sufficient_context": True,
                    }
                )
            ],
        )

        result = await service.answer("who supplies Acme?")

        assert result.inferences == ["A Northwind disruption would affect Acme output."]
        assert "would affect" not in result.answer

    async def test_insufficient_context_is_reported_not_hidden(self) -> None:
        service = build_service(
            hits=[make_hit("doc-1", "Unrelated commentary.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[
                make_json_response(
                    {
                        "answer": "The context does not say.",
                        "cited_source_ids": [],
                        "inferences": [],
                        "sufficient_context": False,
                    }
                )
            ],
        )

        result = await service.answer("what is Acme's dividend policy?")

        assert not result.sufficient_context
        assert not result.is_answerable

    async def test_empty_context_refuses_to_answer(self) -> None:
        """Answering from nothing is how a RAG system starts inventing facts."""
        service = build_service(hits=[], responses=[])

        with pytest.raises(NotFoundError, match="no relevant context"):
            await service.answer("what does Acme make?")

    async def test_a_refusal_surfaces_rather_than_becoming_an_answer(self) -> None:
        refusal = CompletionResponse(
            text="",
            model="fake-model-1",
            provider="fake",
            stop_reason=StopReason.REFUSAL,
            refusal_category="policy",
        )
        service = build_service(
            hits=[make_hit("doc-1", "Some text.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[refusal],
        )

        with pytest.raises(AIProviderResponseError):
            await service.answer("what does Acme make?")

    async def test_unparsable_output_does_not_become_an_answer(self) -> None:
        garbage = CompletionResponse(
            text="I think probably around 400 million?",
            model="fake-model-1",
            provider="fake",
            stop_reason=StopReason.END_TURN,
        )
        service = build_service(
            hits=[make_hit("doc-1", "Some text.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[garbage],
        )

        with pytest.raises(AIProviderResponseError):
            await service.answer("what was revenue?")

    async def test_citations_resolve_to_real_documents(self) -> None:
        """A citation whose document has been deleted is not returned as a source."""
        service = build_service(
            hits=[make_hit("doc-1", "Text."), make_hit("doc-2", "More text.")],
            documents={"doc-1": make_document("doc-1")},
            responses=[
                make_json_response(
                    {
                        "answer": "Both say things [doc-1][doc-2].",
                        "cited_source_ids": ["doc-1", "doc-2"],
                        "inferences": [],
                        "sufficient_context": True,
                    }
                )
            ],
        )

        result = await service.answer("what do the filings say?")

        assert result.cited_source_ids == ["doc-1", "doc-2"]
        assert [s.source_id for s in result.provenance.sources] == ["doc-1"]
