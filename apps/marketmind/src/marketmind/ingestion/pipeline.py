"""The ingestion pipeline: document in, resolved graph structure out.

Order of operations, and why:

1. **Deduplicate** on content hash. Feeds redeliver constantly; re-embedding a
   filing that has not changed is the single most expensive way to do nothing.
2. **Chunk**, then **verify offsets**. The verification is a hard failure, not a
   warning: if a chunk's offsets no longer resolve to its text, every citation
   derived from this document points somewhere other than it claims.
3. **Embed and store** the chunks, so the text is retrievable and citable.
4. **Extract** entities and relationships with a model — the one step where a
   model is the right tool.
5. **Resolve** the extracted entities against what the graph already holds.
   Deterministic, auditable, and never delegated to a model.
6. **Write** the graph, then record which chunks mentioned which entity.
7. **Publish** what changed.

Transactionality: Postgres writes and Neo4j writes cannot share a transaction.
The pipeline writes Postgres first and expects to run inside a caller-managed
session scope, so a graph failure propagates and the caller's rollback undoes
the document. Redelivery then replays the whole thing, which is safe because
every graph write is idempotent. The reverse order would leave orphan nodes
citing a document that does not exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fie_ai.embeddings import EmbeddingProvider, Vector
from fie_common.errors import ValidationError
from fie_events.bus import EventBus
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from marketmind.domain.documents import Chunk, Document
from marketmind.domain.entities import Entity
from marketmind.domain.relationships import Relationship
from marketmind.events import (
    DOCUMENT_DUPLICATE,
    DOCUMENT_INGESTED,
    ENTITY_CREATED,
    ENTITY_MERGED,
    GRAPH_UPDATED,
    RELATIONSHIP_CREATED,
    DocumentDuplicatePayload,
    DocumentIngestedPayload,
    EntityChangedPayload,
    EntityMergedPayload,
    GraphUpdatedPayload,
    RelationshipCreatedPayload,
    build_event,
)
from marketmind.graph.repository import GraphRepository, entity_from_node
from marketmind.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    verify_chunk_offsets,
)
from marketmind.ingestion.extraction import ExtractionService
from marketmind.resolution.resolver import (
    EntityIndex,
    EntityResolver,
    MatchDecision,
    canonicalize,
)
from marketmind.storage.repository import DocumentRepository

logger = get_logger(__name__)


@dataclass
class IngestionReport:
    """What one ingestion run did.

    Returned rather than logged-and-forgotten so callers — the API, the event
    consumer, a backfill script — can assert on the outcome and surface partial
    failures instead of reporting success for a run that rejected half its
    extractions.
    """

    document_id: str
    duplicate: bool = False
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    relationships_created: int = 0
    #: Extraction output that failed validation, with the reason.
    rejected: list[str] = field(default_factory=list)
    #: False when embedding was skipped or unavailable; the document is stored
    #: and citable, but it will not surface in vector retrieval.
    embeddings_written: bool = False

    @property
    def is_retrievable(self) -> bool:
        """Whether this document can actually be found by a question."""
        return self.embedded_chunk_count > 0

    @property
    def rejection_count(self) -> int:
        return len(self.rejected)


@dataclass
class IngestionPipeline:
    """Orchestrates document ingestion into the knowledge graph.

    Configuration is passed as plain values rather than a settings object: the
    settings module builds on these components, so depending on it here would
    close an import cycle and make the pipeline untestable without an
    environment.
    """

    documents: DocumentRepository
    graph: GraphRepository
    embeddings: EmbeddingProvider
    extraction: ExtractionService
    resolver: EntityResolver = field(default_factory=EntityResolver)
    bus: EventBus | None = None
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    #: Texts per embedding request.
    embedding_batch_size: int = 32
    #: Upper bound on chunks sent for extraction. A full 10-K would otherwise
    #: turn one ingestion into hundreds of model calls.
    max_extraction_chunks: int = 60

    async def ingest(
        self, document: Document, *, correlation_id: str | None = None
    ) -> IngestionReport:
        """Ingest one document end to end.

        Raises:
            ValidationError: if the document has no usable content, or if its
                chunk offsets fail verification.
        """
        with traced(
            "marketmind.ingest",
            attributes={"document_id": document.id, "document_type": str(document.type)},
        ):
            document_id, created = await self.documents.upsert_document(document)

            if not created:
                logger.info(
                    "document_already_ingested",
                    document_id=document_id,
                    content_hash=document.content_hash,
                )
                await self._publish(
                    DOCUMENT_DUPLICATE,
                    DocumentDuplicatePayload(
                        document_id=document.id,
                        existing_document_id=document_id,
                        content_hash=document.content_hash,
                    ),
                    correlation_id=correlation_id,
                )
                return IngestionReport(document_id=document_id, duplicate=True)

            report = IngestionReport(document_id=document_id)

            chunks = chunk_document(document, self.chunking)
            if not verify_chunk_offsets(document, chunks):
                # Every citation from this document would be wrong. Fail the
                # unit of work so the caller rolls back rather than storing
                # references nobody can verify.
                raise ValidationError(
                    "chunk offsets do not resolve back to the document; "
                    "citations derived from these chunks would be unverifiable",
                    details={"document_id": document.id, "chunk_count": len(chunks)},
                )
            report.chunk_count = len(chunks)

            vectors = await self._embed_chunks(chunks)
            await self.documents.store_chunks(
                chunks, vectors, embedding_model=self.embeddings.model
            )
            report.embedded_chunk_count = len(vectors)
            report.embeddings_written = True

            await self._build_graph(document, chunks, report, correlation_id=correlation_id)

            await self._publish(
                DOCUMENT_INGESTED,
                DocumentIngestedPayload(
                    document_id=document_id,
                    document_type=str(document.type),
                    title=document.title,
                    content_hash=document.content_hash,
                    chunk_count=report.chunk_count,
                    embedded_chunk_count=report.embedded_chunk_count,
                    entity_count=report.entities_created + report.entities_merged,
                    relationship_count=report.relationships_created,
                    uri=document.uri,
                ),
                correlation_id=correlation_id,
            )
            await self._publish(
                GRAPH_UPDATED,
                GraphUpdatedPayload(
                    document_id=document_id,
                    entities_created=report.entities_created,
                    entities_merged=report.entities_merged,
                    relationships_created=report.relationships_created,
                    rejected_count=report.rejection_count,
                ),
                correlation_id=correlation_id,
            )

            logger.info(
                "document_ingested",
                document_id=document_id,
                chunks=report.chunk_count,
                entities_created=report.entities_created,
                entities_merged=report.entities_merged,
                relationships=report.relationships_created,
                rejected=report.rejection_count,
            )
            return report

    async def ingest_all(
        self, documents: Sequence[Document], *, correlation_id: str | None = None
    ) -> list[IngestionReport]:
        """Ingest a batch sequentially.

        Sequential on purpose: entities resolve against the graph as it stands,
        so concurrent ingestion of two documents mentioning the same company
        would race to create the node twice.
        """
        return [
            await self.ingest(document, correlation_id=correlation_id) for document in documents
        ]

    # -- steps ---------------------------------------------------------------

    async def _embed_chunks(self, chunks: Sequence[Chunk]) -> list[Vector]:
        """Embed chunks in batches, preserving order."""
        vectors: list[Vector] = []
        texts = [chunk.text for chunk in chunks]
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            vectors.extend(await self.embeddings.embed_texts(batch))
        return vectors

    async def _build_graph(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        report: IngestionReport,
        *,
        correlation_id: str | None,
    ) -> None:
        """Extract, resolve, and write graph structure for a document."""
        extraction_chunks = list(chunks[: self.max_extraction_chunks])
        if len(chunks) > self.max_extraction_chunks:
            logger.info(
                "extraction_truncated",
                document_id=document.id,
                total_chunks=len(chunks),
                extracted_chunks=len(extraction_chunks),
            )

        outcome = await self.extraction.extract(document, extraction_chunks)
        report.rejected.extend(outcome.rejected)
        if not outcome.entities:
            return

        index = await self._candidate_index(outcome.entities)
        known_before = {entity.id for entity in index.entities}

        resolved, decisions = self.resolver.resolve_batch(outcome.entities, index)
        remap = {
            extracted.id: settled.id
            for extracted, settled in zip(outcome.entities, resolved, strict=True)
        }

        for entity in resolved:
            await self.graph.upsert_entity(entity)

        await self._record_mentions(resolved, outcome.entities, chunks)

        relationships = self._remap_relationships(outcome.relationships, remap, report)
        written = await self.graph.upsert_relationships(relationships)
        written_ids = set(written)
        report.relationships_created = len(written)

        for extracted, settled, decision in zip(outcome.entities, resolved, decisions, strict=True):
            await self._publish_entity_change(
                extracted,
                settled,
                decision,
                known_before=known_before,
                report=report,
                correlation_id=correlation_id,
            )

        for relationship in relationships:
            if relationship.id not in written_ids:
                continue
            await self._publish(
                RELATIONSHIP_CREATED,
                RelationshipCreatedPayload(
                    relationship_id=relationship.id,
                    relationship_type=str(relationship.type),
                    source_entity_id=relationship.source_entity_id,
                    target_entity_id=relationship.target_entity_id,
                    source_entity_type=str(relationship.source_entity_type),
                    target_entity_type=str(relationship.target_entity_type),
                    source_document_ids=[
                        source.source_id for source in relationship.provenance.sources
                    ],
                ),
                correlation_id=correlation_id,
            )

    async def _candidate_index(self, entities: Sequence[Entity]) -> EntityIndex:
        """Load the entities that could plausibly match this batch.

        Blocking happens in the database, not in memory: a resolver that loads
        the whole graph to compare names stops working at the size where the
        graph becomes interesting.
        """
        index = EntityIndex(config=self.resolver.config)
        seen: set[str] = set()

        for entity in entities:
            candidates: list[dict[str, object]] = []

            for identifier in entity.identifiers:
                node = await self.graph.find_by_identifier(entity.type, identifier.key)
                if node is not None:
                    candidates.append(node)

            canonical = entity.canonical_name or canonicalize(entity.name, entity.type)
            if canonical:
                candidates.extend(await self.graph.find_by_canonical_name(entity.type, canonical))

            for node in candidates:
                node_id = str(node.get("id") or "")
                if not node_id or node_id in seen:
                    continue
                seen.add(node_id)
                try:
                    index.add(entity_from_node(dict(node), entity.type))
                except ValueError as error:
                    # A node that no longer satisfies the domain rules must not
                    # break ingestion of everything else.
                    logger.warning("candidate_node_rejected", node_id=node_id, error=str(error))

        return index

    async def _record_mentions(
        self,
        resolved: Sequence[Entity],
        extracted: Sequence[Entity],
        chunks: Sequence[Chunk],
    ) -> None:
        """Link each resolved entity to the chunks that mentioned it.

        This is what turns a graph node into a quotable citation: the node knows
        which documents support it, the mention rows know which spans.
        """
        for original, settled in zip(extracted, resolved, strict=True):
            chunk_ids = self._chunks_for_entity(original, chunks)
            if not chunk_ids:
                continue
            await self.documents.record_mentions(
                settled.id, str(settled.type), original.name, chunk_ids
            )

    @staticmethod
    def _chunks_for_entity(entity: Entity, chunks: Sequence[Chunk]) -> list[str]:
        """Chunks whose span matches the entity's cited source locator."""
        spans = {
            (source.locator.start_char, source.locator.end_char)
            for source in entity.provenance.sources
            if source.locator is not None
        }
        if not spans:
            return []
        return [chunk.id for chunk in chunks if (chunk.start_char, chunk.end_char) in spans]

    def _remap_relationships(
        self,
        relationships: Sequence[Relationship],
        remap: dict[str, str],
        report: IngestionReport,
    ) -> list[Relationship]:
        """Repoint relationships at resolved entity ids.

        Resolution can merge two extracted entities into one, which turns an
        edge between them into a self-loop. Rebuilding through the constructor
        re-runs the domain validators, so that case is rejected here rather than
        written and puzzled over later.
        """
        remapped: list[Relationship] = []
        for relationship in relationships:
            source_id = remap.get(relationship.source_entity_id, relationship.source_entity_id)
            target_id = remap.get(relationship.target_entity_id, relationship.target_entity_id)
            try:
                remapped.append(
                    Relationship(
                        id=relationship.id,
                        type=relationship.type,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        source_entity_type=relationship.source_entity_type,
                        target_entity_type=relationship.target_entity_type,
                        valid_from=relationship.valid_from,
                        valid_to=relationship.valid_to,
                        attributes=dict(relationship.attributes),
                        provenance=relationship.provenance,
                    )
                )
            except ValueError as error:
                report.rejected.append(
                    f"relationship {relationship.type} dropped after resolution: {error}"
                )
        return remapped

    # -- events --------------------------------------------------------------

    async def _publish_entity_change(
        self,
        extracted: Entity,
        settled: Entity,
        decision: MatchDecision,
        *,
        known_before: set[str],
        report: IngestionReport,
        correlation_id: str | None,
    ) -> None:
        """Emit the right event for what resolution decided."""
        if decision.matched and decision.matched_entity_id:
            report.entities_merged += 1
            await self._publish(
                ENTITY_MERGED,
                EntityMergedPayload(
                    surviving_entity_id=settled.id,
                    merged_entity_id=extracted.id,
                    entity_type=str(settled.type),
                    name=settled.name,
                    match_reason=str(decision.reason),
                    confidence=decision.confidence,
                    evidence=decision.evidence,
                ),
                correlation_id=correlation_id,
            )
            return

        if settled.id in known_before:
            return

        report.entities_created += 1
        await self._publish(
            ENTITY_CREATED,
            EntityChangedPayload(
                entity_id=settled.id,
                entity_type=str(settled.type),
                name=settled.name,
                canonical_name=settled.canonical_name,
                identifier_keys=sorted(settled.identifier_keys),
                source_document_ids=[source.source_id for source in settled.provenance.sources],
            ),
            correlation_id=correlation_id,
        )

    async def _publish(
        self,
        event_type: str,
        payload: DocumentIngestedPayload
        | DocumentDuplicatePayload
        | EntityChangedPayload
        | EntityMergedPayload
        | RelationshipCreatedPayload
        | GraphUpdatedPayload,
        *,
        correlation_id: str | None,
    ) -> None:
        """Publish, treating the bus as best-effort.

        Publication happens after the work, so a crash between the graph write
        and the publish loses the notification. Consumers are idempotent and
        re-ingestion replays the events, so this is at-least-once with a small
        loss window rather than exactly-once. Closing that window needs a
        transactional outbox, which belongs with the rest of the cross-service
        delivery guarantees in Phase 8.
        """
        if self.bus is None:
            return
        event = build_event(event_type, payload, correlation_id=correlation_id)
        try:
            await self.bus.publish(event)
        except Exception as error:  # noqa: BLE001 — a bus outage must not fail
            # an ingestion whose data is already durably written.
            logger.warning(
                "event_publish_failed",
                event_type=event_type,
                error=str(error),
            )


__all__ = ["IngestionPipeline", "IngestionReport"]
