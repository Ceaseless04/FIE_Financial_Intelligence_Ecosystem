"""Entity and relationship extraction from text.

This is the one part of MarketMind where a language model is the right tool:
recognising that "Tim Cook, who leads Apple" states an EXECUTIVE_OF
relationship is genuinely ambiguous natural-language work, unlike identifier
parsing or identity resolution.

Three guardrails apply because the output enters a shared knowledge graph that
four other products read:

1. **Structured output** — the model fills a schema; the response is validated
   deterministically rather than parsed out of prose.
2. **Grounding** — every extracted item must cite a chunk it came from, and
   every cited chunk must be one that was actually supplied. Fabricated
   citations are rejected, not stored.
3. **Schema validation** — relationships whose endpoint types are nonsensical
   are dropped before they reach the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import Field

from fie_ai import (
    AIRouter,
    CompletionRequest,
    Effort,
    Message,
    Role,
)
from fie_ai.structured import check_grounding, parse_structured, structured_request
from fie_common.errors import AIProviderResponseError
from fie_observability.logging import get_logger
from fie_schemas.base import FIEModel
from fie_schemas.provenance import Provenance
from marketmind.domain.documents import Chunk, Document
from marketmind.domain.entities import Entity, EntityType, Identifier, IdentifierType
from marketmind.domain.relationships import (
    RELATIONSHIP_SCHEMA,
    Relationship,
    RelationshipType,
)

logger = get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You extract entities and relationships from financial documents to populate a \
knowledge graph.

Extract only what the text states. Do not infer, estimate, or use outside \
knowledge about these companies. If the text does not say it, it is not there.

Every item you extract must cite the chunk_id it came from. Only cite chunk ids \
that appear in the supplied text. Never invent a chunk id.

Do not produce evaluations, ratings, recommendations, price targets, or \
opinions about whether something is good or bad. This graph records what is \
true, not what it is worth.

Entity types: Company, Executive, Industry, Product, Geography, MacroIndicator.
Relationship types: EXECUTIVE_OF, BOARD_MEMBER_OF, SUBSIDIARY_OF, SUPPLIES, \
CUSTOMER_OF, COMPETES_WITH, PARTNERS_WITH, OPERATES_IN, PRODUCES, \
HEADQUARTERED_IN, ACQUIRED, INVESTED_IN, EXPOSED_TO.

Use the exact name as written in the document. Do not normalize or expand \
abbreviations."""


class ExtractedEntity(FIEModel):
    """An entity as returned by the model, before validation and resolution."""

    name: str = Field(min_length=1, max_length=512)
    type: EntityType
    ticker: str | None = None
    description: str | None = Field(default=None, max_length=1000)
    #: Chunk the entity was found in. Checked against the supplied chunks.
    chunk_id: str = Field(min_length=1)


class ExtractedRelationship(FIEModel):
    """A relationship as returned by the model, before validation."""

    source_name: str = Field(min_length=1, max_length=512)
    source_type: EntityType
    target_name: str = Field(min_length=1, max_length=512)
    target_type: EntityType
    type: RelationshipType
    chunk_id: str = Field(min_length=1)


class ExtractionResult(FIEModel):
    """The model's structured response."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


@dataclass
class ExtractionOutcome:
    """Validated extraction, with everything that was rejected and why.

    Rejections are returned rather than silently dropped: a run where half the
    relationships failed schema validation is a prompt problem, and it should
    be visible in metrics instead of showing up as a sparse graph.
    """

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    #: Entity name -> generated entity id, for wiring relationships to entities.
    _name_index: dict[tuple[str, EntityType], str] = field(default_factory=dict)

    @property
    def rejection_count(self) -> int:
        return len(self.rejected)


class ExtractionService:
    """Extracts graph structure from document chunks using an AI provider."""

    def __init__(
        self,
        router: AIRouter,
        *,
        max_chunks_per_call: int = 6,
        effort: Effort | None = Effort.LOW,
        max_tokens: int = 4096,
    ) -> None:
        self._router = router
        self._max_chunks_per_call = max_chunks_per_call
        self._effort = effort
        self._max_tokens = max_tokens

    async def extract(self, document: Document, chunks: Sequence[Chunk]) -> ExtractionOutcome:
        """Extract entities and relationships from a document's chunks."""
        outcome = ExtractionOutcome()
        if not chunks:
            return outcome

        for batch_start in range(0, len(chunks), self._max_chunks_per_call):
            batch = list(chunks[batch_start : batch_start + self._max_chunks_per_call])
            try:
                result = await self._extract_batch(document, batch)
            except AIProviderResponseError as error:
                # One malformed batch must not lose the whole document.
                logger.warning(
                    "extraction_batch_failed",
                    document_id=document.id,
                    chunk_count=len(batch),
                    error=str(error),
                )
                outcome.rejected.append(f"batch at chunk {batch_start}: {error}")
                continue

            self._collect(document, batch, result, outcome)

        logger.info(
            "extraction_completed",
            document_id=document.id,
            entities=len(outcome.entities),
            relationships=len(outcome.relationships),
            rejected=outcome.rejection_count,
        )
        return outcome

    async def _extract_batch(self, document: Document, chunks: Sequence[Chunk]) -> ExtractionResult:
        prompt = self._build_prompt(document, chunks)
        request = structured_request(
            CompletionRequest(
                messages=[Message(role=Role.USER, content=prompt)],
                system=EXTRACTION_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
                effort=self._effort,
            ),
            ExtractionResult,
        )
        response = await self._router.complete(request)
        return parse_structured(response, ExtractionResult)

    @staticmethod
    def _build_prompt(document: Document, chunks: Sequence[Chunk]) -> str:
        """Render chunks with explicit ids so citations are checkable."""
        parts = [
            f"Document type: {document.type}",
            f"Document title: {document.title}",
            "",
            "Extract entities and relationships from the following chunks. "
            "Cite the chunk_id each item came from.",
            "",
        ]
        for chunk in chunks:
            parts.append(f'<chunk id="{chunk.id}">')
            parts.append(chunk.text)
            parts.append("</chunk>")
            parts.append("")
        return "\n".join(parts)

    def _collect(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        result: ExtractionResult,
        outcome: ExtractionOutcome,
    ) -> None:
        """Validate the model's output and convert it into domain objects."""
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        allowed_ids = list(chunks_by_id)

        grounding = check_grounding(
            [item.chunk_id for item in result.entities]
            + [item.chunk_id for item in result.relationships],
            allowed_ids,
            require_citation=False,
        )
        if grounding.unknown_ids:
            logger.warning(
                "extraction_fabricated_citations",
                document_id=document.id,
                unknown_chunk_ids=grounding.unknown_ids,
            )

        for extracted in result.entities:
            chunk = chunks_by_id.get(extracted.chunk_id)
            if chunk is None:
                outcome.rejected.append(
                    f"entity {extracted.name!r} cited unknown chunk {extracted.chunk_id!r}"
                )
                continue

            key = (extracted.name.strip().lower(), extracted.type)
            if key in outcome._name_index:
                continue

            entity = Entity(
                type=extracted.type,
                name=extracted.name,
                description=extracted.description,
                identifiers=self._identifiers_for(extracted),
                provenance=Provenance.fact(chunk.to_source_reference(document)),
            )
            outcome.entities.append(entity)
            outcome._name_index[key] = entity.id

        for edge in result.relationships:
            chunk = chunks_by_id.get(edge.chunk_id)
            if chunk is None:
                outcome.rejected.append(
                    f"relationship {edge.type} cited unknown chunk {edge.chunk_id!r}"
                )
                continue

            source_id = outcome._name_index.get(
                (edge.source_name.strip().lower(), edge.source_type)
            )
            target_id = outcome._name_index.get(
                (edge.target_name.strip().lower(), edge.target_type)
            )
            if source_id is None or target_id is None:
                outcome.rejected.append(
                    f"relationship {edge.source_name!r} -{edge.type}-> "
                    f"{edge.target_name!r} references an entity that was not extracted"
                )
                continue

            allowed_sources, allowed_targets = RELATIONSHIP_SCHEMA[edge.type]
            if edge.source_type not in allowed_sources or edge.target_type not in allowed_targets:
                # A hallucinated edge type corrupts every traversal that touches
                # it, so it is dropped here rather than written and cleaned later.
                outcome.rejected.append(
                    f"relationship {edge.type} is not valid between "
                    f"{edge.source_type} and {edge.target_type}"
                )
                continue

            try:
                relationship = Relationship(
                    type=edge.type,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    source_entity_type=edge.source_type,
                    target_entity_type=edge.target_type,
                    provenance=Provenance.fact(chunk.to_source_reference(document)),
                )
            except ValueError as error:
                outcome.rejected.append(f"relationship rejected: {error}")
                continue

            outcome.relationships.append(relationship)

    @staticmethod
    def _identifiers_for(extracted: ExtractedEntity) -> list[Identifier]:
        if extracted.type is not EntityType.COMPANY or not extracted.ticker:
            return []
        try:
            return [Identifier(type=IdentifierType.TICKER, value=extracted.ticker)]
        except ValueError:
            # A malformed ticker is not worth failing the entity over.
            return []


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ExtractionOutcome",
    "ExtractionResult",
    "ExtractionService",
]
