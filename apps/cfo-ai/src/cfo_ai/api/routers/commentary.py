"""Management commentary generation and retrieval.

The only router that reaches a language model, and it does so last. Every figure
and every direction is decided before the model is called, and the prose that
comes back is checked against both.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from cfo_ai.api.dependencies import (
    PERMISSION_READ_COMMENTARY,
    PERMISSION_WRITE_COMMENTARY,
    ContainerDep,
    SessionDep,
    require_permission,
)
from cfo_ai.api.schemas import CommentaryResponse, GenerateCommentaryRequest
from fie_common.errors import NotFoundError

router = APIRouter(prefix="/api/v1/commentary", tags=["commentary"])


@router.post(
    "",
    response_model=CommentaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate verified management commentary",
    dependencies=[Depends(require_permission(PERMISSION_WRITE_COMMENTARY))],
)
async def generate_commentary(
    request: GenerateCommentaryRequest, container: ContainerDep, session: SessionDep
) -> CommentaryResponse:
    """Write commentary on a period's variances, and verify it twice.

    Returns 201 even when the commentary fails verification. A draft that quotes
    a real number and reverses its meaning is stored and returned with
    ``publishable: false`` and the contradiction explained — discarding it would
    hide a systematic model problem, and a client that checks the flag can never
    mistake it for analysis.

    Raises:
        NotFoundError: if the plans or the chart of accounts are missing.
        ValidationError: if there are no variances to write about.
    """
    outcome = await container.reporting(session).report(
        request.entity_id,
        request.period.to_period(),
        company_name=request.company_name,
        budget_version=request.budget_version,
    )
    assert outcome.commentary is not None and outcome.commentary_id is not None
    return CommentaryResponse.from_commentary(outcome.commentary_id, outcome.commentary)


@router.get(
    "/{commentary_id}",
    response_model=CommentaryResponse,
    summary="Fetch stored commentary",
    dependencies=[Depends(require_permission(PERMISSION_READ_COMMENTARY))],
)
async def get_commentary(
    commentary_id: str, container: ContainerDep, session: SessionDep
) -> CommentaryResponse:
    """Retrieve commentary and its verification verdicts.

    Raises:
        NotFoundError: if no such commentary is stored.
    """
    stored = await container.commentaries(session).get(commentary_id)
    if stored is None:
        raise NotFoundError("no such commentary", details={"commentary_id": commentary_id})
    return CommentaryResponse.from_commentary(*stored)


@router.get(
    "/entities/{entity_id}",
    response_model=list[CommentaryResponse],
    summary="List commentary for a company",
    dependencies=[Depends(require_permission(PERMISSION_READ_COMMENTARY))],
)
async def list_commentary(
    entity_id: str,
    container: ContainerDep,
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=100),
    publishable_only: bool = Query(
        default=False,
        description=(
            "Return only commentary that passed both grounding checks. Off by "
            "default: a reversed commentary is evidence about the model, and "
            "hiding it by default would make that evidence hard to find."
        ),
    ),
) -> list[CommentaryResponse]:
    """Commentary for one entity, newest first."""
    stored = await container.commentaries(session).list_for_entity(
        entity_id, limit=limit, publishable_only=publishable_only
    )
    return [CommentaryResponse.from_commentary(item_id, item) for item_id, item in stored]


__all__ = ["router"]
