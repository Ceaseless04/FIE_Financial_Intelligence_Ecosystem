"""GraphRAG question answering and raw context retrieval."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from marketmind.api.dependencies import (
    PERMISSION_READ_RETRIEVAL,
    ContainerDep,
    SessionDep,
    require_permission,
)
from marketmind.api.schemas import (
    AnswerResponse,
    AskRequest,
    CitationView,
    ContextPassageView,
    ContextResponse,
)

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["retrieval"],
    dependencies=[Depends(require_permission(PERMISSION_READ_RETRIEVAL))],
)


@router.post("/context", response_model=ContextResponse, summary="Retrieve context only")
async def retrieve_context(
    request: AskRequest, container: ContainerDep, session: SessionDep
) -> ContextResponse:
    """Return what retrieval found, without asking a model to summarise it.

    Useful in its own right — another product may want the passages rather than
    prose — and it is the endpoint to reach for when an answer looks wrong,
    because it shows exactly what the model was given.
    """
    service = container.graphrag(session)
    context = await service.retrieve(request.question, entity_ids=request.entity_ids or None)
    return ContextResponse(
        question=context.question,
        passages=[
            ContextPassageView(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                start_char=hit.start_char,
                end_char=hit.end_char,
                section=hit.section,
                similarity=round(hit.similarity, 4),
            )
            for hit in context.chunks
        ],
        graph_facts=list(context.graph_facts),
        entity_ids=list(context.entity_ids),
    )


@router.post("/answer", response_model=AnswerResponse, summary="Answer a question with citations")
async def answer(
    request: AskRequest, container: ContainerDep, session: SessionDep
) -> AnswerResponse:
    """Answer a question from the graph and the documents behind it.

    Every citation returned has been verified against the context the model was
    actually given. A model-invented source id is stripped and ``grounded``
    turns false, so a client can tell a fully-sourced answer from one that
    reached beyond its evidence.

    Raises:
        NotFoundError: if retrieval found nothing relevant. Answering from an
            empty context is how a RAG system starts inventing facts.
    """
    service = container.graphrag(session)
    result = await service.answer(request.question, entity_ids=request.entity_ids or None)

    citations: list[CitationView] = []
    for source in result.provenance.sources:
        citations.append(
            CitationView(
                document_id=source.source_id,
                title=source.title,
                uri=source.uri,
            )
        )

    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        sufficient_context=result.sufficient_context,
        citations=citations,
        inferences=list(result.inferences),
        entity_ids=list(result.entity_ids),
        provenance=result.provenance,
    )


__all__ = ["router"]
