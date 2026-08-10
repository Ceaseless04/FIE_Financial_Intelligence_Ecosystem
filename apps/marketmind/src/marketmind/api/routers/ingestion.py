"""Document ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from fie_observability.context import current_context
from marketmind.api.dependencies import (
    PERMISSION_WRITE_INGESTION,
    ContainerDep,
    SessionDep,
    require_permission,
)
from marketmind.api.schemas import IngestDocumentRequest, IngestDocumentResponse
from marketmind.domain.documents import Document
from marketmind.ingestion.metadata import extract_metadata

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["ingestion"],
    dependencies=[Depends(require_permission(PERMISSION_WRITE_INGESTION))],
)


@router.post(
    "",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into the knowledge graph",
)
async def ingest_document(
    request: IngestDocumentRequest, container: ContainerDep, session: SessionDep
) -> IngestDocumentResponse:
    """Chunk, embed, extract, resolve, and write one document.

    Synchronous by design at this stage: the caller learns whether the document
    was a duplicate, how much structure came out of it, and what was rejected.
    Moving ingestion onto a queue is a Phase 8 concern, and doing it before the
    pipeline's failure modes are visible would just hide them behind a job id.

    Re-submitting an identical document is safe — it is recognised by content
    hash and returns ``duplicate: true`` without redoing the work.
    """
    document = Document(
        type=request.type,
        title=request.title,
        content=request.content,
        uri=request.uri,
        published_at=request.published_at,
        metadata=dict(request.metadata),
    )
    # Deterministic enrichment before the pipeline runs: tickers, CIKs, and form
    # type have exact syntax, so a regex answers them faster and more reliably
    # than a model would. Feed-supplied values still win — that precedence is
    # applied inside extract_metadata.
    document = document.model_copy(
        update={"metadata": {**document.metadata, **extract_metadata(document).to_dict()}}
    )

    pipeline = container.pipeline(session)
    report = await pipeline.ingest(
        document, correlation_id=current_context().correlation_id or None
    )

    return IngestDocumentResponse(
        document_id=report.document_id,
        duplicate=report.duplicate,
        chunk_count=report.chunk_count,
        embedded_chunk_count=report.embedded_chunk_count,
        entities_created=report.entities_created,
        entities_merged=report.entities_merged,
        relationships_created=report.relationships_created,
        rejected=list(report.rejected),
        is_retrievable=report.is_retrievable,
    )


__all__ = ["router"]
