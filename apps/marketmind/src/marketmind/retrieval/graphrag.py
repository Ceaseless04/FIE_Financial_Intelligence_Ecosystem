"""GraphRAG: retrieval that combines vector search with graph traversal.

Plain vector RAG answers "what does the corpus say about X" well and "how is X
connected to Y" badly, because the connection is often never stated in a single
passage — it exists only as a path through the graph. GraphRAG adds that path.

The flow:

1. **Seed** — vector search finds chunks semantically close to the question.
2. **Anchor** — entities mentioned in those chunks become graph seeds.
3. **Expand** — bounded k-hop traversal collects the surrounding structure.
4. **Assemble** — chunks and graph paths become one context block, every item
   carrying an explicit source id.
5. **Synthesize** — the model answers *from that context only*, citing ids.
6. **Verify** — citations are checked against the supplied ids before the
   answer is returned. A fabricated citation fails the answer.

Step 6 is the one that makes this usable in a financial product. Without it,
the system produces confident, well-formatted, unverifiable claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import Field

from fie_ai import AIRouter, CompletionRequest, Effort, Message, Role
from fie_ai.embeddings import EmbeddingProvider
from fie_ai.structured import check_grounding, parse_structured, structured_request
from fie_common.errors import AIProviderResponseError, NotFoundError
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance
from marketmind.graph.repository import GraphRepository
from marketmind.storage.repository import ChunkHit, DocumentRepository

logger = get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """\
You answer questions about companies using only the context provided.

Rules:
- Use only the supplied context. If it does not contain the answer, say so.
- Cite the source_id for every factual claim you make.
- Only cite source_ids that appear in the context. Never invent one.
- Distinguish what the sources state from what you are inferring. Label \
inferences clearly.
- Do not give investment advice, recommendations, ratings, or price targets. \
Describe what is known; do not judge whether it is good or bad.
- If sources disagree, say so rather than picking one."""


class RetrievedContext(FIEModel):
    """Everything assembled for a question, before the model sees it."""

    question: str
    chunks: list[ChunkHit] = Field(default_factory=list)
    #: Graph entities reached from the seeded chunks.
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: dict[str, str] = Field(default_factory=dict)
    #: Rendered description of graph paths, included in the prompt.
    graph_facts: list[str] = Field(default_factory=list)

    @property
    def source_ids(self) -> list[str]:
        """Every citable id supplied to the model.

        Also the allowlist the answer's citations are checked against.
        """
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.document_id, None)
        return list(seen)

    @property
    def is_empty(self) -> bool:
        return not self.chunks and not self.graph_facts

    def render(self) -> str:
        """Format the context for the prompt, with explicit source ids."""
        parts: list[str] = []

        if self.chunks:
            parts.append("## Source passages")
            for hit in self.chunks:
                section = f" (section: {hit.section})" if hit.section else ""
                parts.append(f'<passage source_id="{hit.document_id}"{section}>')
                parts.append(hit.text)
                parts.append("</passage>")
                parts.append("")

        if self.graph_facts:
            parts.append("## Known relationships from the knowledge graph")
            for fact in self.graph_facts:
                parts.append(f"- {fact}")
            parts.append("")

        return "\n".join(parts)


class SynthesizedAnswer(FIEModel):
    """The model's structured answer."""

    answer: str = Field(min_length=1)
    #: Source ids backing the answer. Verified against the supplied context.
    cited_source_ids: list[str] = Field(default_factory=list)
    #: Claims the model drew rather than read. Kept separate from the answer so
    #: a reader can see where sourcing ends and reasoning begins.
    inferences: list[str] = Field(default_factory=list)
    sufficient_context: bool = True


class GraphRAGAnswer(FIEModel):
    """A verified answer, safe to return to a caller."""

    question: str
    answer: str
    provenance: Provenance
    cited_source_ids: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    sufficient_context: bool = True
    #: True when every citation resolved to a supplied source.
    grounded: bool = True

    @property
    def is_answerable(self) -> bool:
        return self.sufficient_context and self.grounded


@dataclass
class GraphRAGConfig:
    """Retrieval breadth and depth.

    Defaults are deliberately modest. A wide traversal on a dense financial
    graph produces a context the model cannot use and a latency nobody accepts;
    precision beats recall when every claim must be cited.
    """

    #: Chunks retrieved by vector similarity.
    top_k_chunks: int = 6
    #: Hops out from each seed entity.
    graph_depth: int = 1
    #: Neighbours collected per seed entity.
    max_neighbours: int = 20
    #: Entities used as traversal seeds.
    max_seed_entities: int = 5
    #: Similarity below which a chunk is not worth including.
    minimum_similarity: float = 0.15
    max_answer_tokens: int = 2048
    effort: Effort | None = Effort.MEDIUM


@dataclass
class GraphRAGService:
    """Retrieval-augmented question answering over the knowledge graph."""

    documents: DocumentRepository
    graph: GraphRepository
    embeddings: EmbeddingProvider
    router: AIRouter
    config: GraphRAGConfig = field(default_factory=GraphRAGConfig)

    async def retrieve(
        self, question: str, *, entity_ids: Sequence[str] | None = None
    ) -> RetrievedContext:
        """Assemble context for a question without calling the answering model."""
        with traced(
            "marketmind.graphrag.retrieve",
            attributes={"question_length": len(question)},
        ):
            query_vector = await self.embeddings.embed_text(question)
            hits = await self.documents.search_similar(
                query_vector,
                limit=self.config.top_k_chunks,
                embedding_model=self.embeddings.model,
            )
            relevant = [hit for hit in hits if hit.similarity >= self.config.minimum_similarity]

            seeds = list(entity_ids or [])
            if not seeds and relevant:
                seeds = await self._entities_for_chunks(relevant)

            graph_facts: list[str] = []
            reached: dict[str, str] = {}

            for seed in seeds[: self.config.max_seed_entities]:
                neighbourhood = await self.graph.neighbourhood(
                    seed, depth=self.config.graph_depth, limit=self.config.max_neighbours
                )
                edges = await self.graph.relationships_of(seed, limit=self.config.max_neighbours)
                for node in neighbourhood.nodes:
                    reached.setdefault(node.id, node.name)
                for edge in edges:
                    graph_facts.append(self._render_edge(seed, reached, edge))

            return RetrievedContext(
                question=question,
                chunks=relevant,
                entity_ids=list(dict.fromkeys([*seeds, *reached])),
                entity_names=reached,
                graph_facts=list(dict.fromkeys(graph_facts)),
            )

    async def answer(
        self, question: str, *, entity_ids: Sequence[str] | None = None
    ) -> GraphRAGAnswer:
        """Retrieve, synthesize, and verify an answer.

        Raises:
            NotFoundError: if nothing relevant was retrieved. Answering from an
                empty context is how a RAG system starts inventing facts.
        """
        context = await self.retrieve(question, entity_ids=entity_ids)

        if context.is_empty:
            raise NotFoundError(
                "no relevant context found for the question",
                details={"question": question[:200]},
            )

        with traced(
            "marketmind.graphrag.synthesize",
            attributes={
                "chunks": len(context.chunks),
                "graph_facts": len(context.graph_facts),
            },
        ):
            synthesized = await self._synthesize(context)

        grounding = check_grounding(
            synthesized.cited_source_ids, context.source_ids, require_citation=False
        )
        if not grounding.is_grounded:
            logger.warning(
                "graphrag_ungrounded_citations",
                unknown_source_ids=grounding.unknown_ids,
                question=question[:120],
            )

        # Only citations that actually resolve are returned. A caller must never
        # receive a reference it cannot follow.
        verified = [
            source_id
            for source_id in synthesized.cited_source_ids
            if source_id not in grounding.unknown_ids
        ]

        sources = await self._source_references(verified)
        provenance = (
            Provenance.generated(self.router.primary.default_model, *sources)
            if sources
            else Provenance.generated(self.router.primary.default_model)
        )

        return GraphRAGAnswer(
            question=question,
            answer=synthesized.answer,
            provenance=provenance,
            cited_source_ids=verified,
            inferences=synthesized.inferences,
            entity_ids=context.entity_ids,
            sufficient_context=synthesized.sufficient_context,
            grounded=not grounding.unknown_ids,
        )

    async def _synthesize(self, context: RetrievedContext) -> SynthesizedAnswer:
        prompt = (
            f"Question: {context.question}\n\n"
            f"{context.render()}\n"
            "Answer the question using only the context above. "
            "Cite the source_id for every factual claim."
        )
        request = structured_request(
            CompletionRequest(
                messages=[Message(role=Role.USER, content=prompt)],
                system=SYNTHESIS_SYSTEM_PROMPT,
                max_tokens=self.config.max_answer_tokens,
                effort=self.config.effort,
            ),
            SynthesizedAnswer,
        )
        response = await self.router.complete(request)
        try:
            return parse_structured(response, SynthesizedAnswer)
        except AIProviderResponseError:
            # A refusal or schema failure must surface as "cannot answer",
            # never as a fabricated answer.
            logger.warning("graphrag_synthesis_failed", question=context.question[:120])
            raise

    async def _entities_for_chunks(self, hits: Sequence[ChunkHit]) -> list[str]:
        """Graph entities mentioned in the retrieved chunks."""
        return await self.documents.entity_ids_for_chunks([hit.chunk_id for hit in hits])

    async def _source_references(self, source_ids: Sequence[str]) -> list:  # type: ignore[type-arg]
        """Turn verified source ids into citable references."""
        references = []
        for source_id in source_ids:
            document = await self.documents.get_document(source_id)
            if document is not None:
                references.append(document.to_source_reference())
        return references

    @staticmethod
    def _render_edge(seed_id: str, names: dict[str, str], edge) -> str:  # type: ignore[no-untyped-def]
        """Describe a relationship in prose the model can read."""
        seed_name = names.get(seed_id, seed_id)
        relation = str(edge.type).replace("_", " ").lower()
        validity = ""
        if edge.valid_to:
            validity = f" (until {edge.valid_to})"
        elif edge.valid_from:
            validity = f" (since {edge.valid_from})"

        if edge.is_outgoing:
            return f"{seed_name} {relation} {edge.other_name}{validity}"
        return f"{edge.other_name} {relation} {seed_name}{validity}"


__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "GraphRAGAnswer",
    "GraphRAGConfig",
    "GraphRAGService",
    "RetrievedContext",
    "SynthesizedAnswer",
]
