"""Research report generation and retrieval.

This is the only router that reaches a language model, and it does so last. By
the time the model is called, every figure the report may quote has already been
computed deterministically; the model's job is to explain them. Before the
response is returned, every number in the prose has been checked back against
those figures.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from atlas.api.dependencies import (
    PERMISSION_READ_RESEARCH,
    PERMISSION_WRITE_RESEARCH,
    ContainerDep,
    SessionDep,
    require_permission,
)
from atlas.api.schemas import ReportRequestBody, ReportResponse
from atlas.domain.money import Rate, to_decimal
from atlas.research.pipeline import ValuationInputs
from fie_common.errors import NotFoundError

router = APIRouter(prefix="/api/v1/research", tags=["research"])


@router.post(
    "/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a verified research note",
    dependencies=[Depends(require_permission(PERMISSION_WRITE_RESEARCH))],
)
async def generate_report(
    request: ReportRequestBody, container: ContainerDep, session: SessionDep
) -> ReportResponse:
    """Write a research note from a company's filed figures.

    Returns 201 even when the note fails verification. A report quoting a figure
    Atlas never computed is stored and returned with ``publishable: false`` and
    the offending figures named — discarding it would hide a systematic prompt
    or model problem behind an empty response, and a client that checks the flag
    can never mistake it for analysis.

    Raises:
        NotFoundError: if no statements are on file for the entity.
        ValidationError: if the filed statements support no computable metric. A
            narrative with no deterministic inputs would be entirely
            model-authored, which is the outcome this design exists to prevent.
    """
    valuation_inputs = None
    if request.valuation is not None:
        valuation_inputs = ValuationInputs(
            discount_rate=Rate.of(to_decimal(request.valuation.discount_rate)),
            terminal_growth=Rate.of(to_decimal(request.valuation.terminal_growth)),
            forecast_growth=Rate.of(to_decimal(request.valuation.forecast_growth)),
            projection_years=request.valuation.projection_years,
        )

    outcome = await container.research(session).produce(
        request.entity_id,
        company_name=request.company_name,
        fiscal_year=request.fiscal_year,
        valuation_inputs=valuation_inputs,
        include_graph_context=request.include_graph_context,
    )
    return ReportResponse.from_report(outcome.report_id, outcome.report)


@router.get(
    "/reports/{report_id}",
    response_model=ReportResponse,
    summary="Fetch a stored research note",
    dependencies=[Depends(require_permission(PERMISSION_READ_RESEARCH))],
)
async def get_report(
    report_id: str, container: ContainerDep, session: SessionDep
) -> ReportResponse:
    """Retrieve a report and its verification verdict.

    Raises:
        NotFoundError: if no such report is stored.
    """
    stored = await container.report_store(session).get(report_id)
    if stored is None:
        raise NotFoundError("no such report", details={"report_id": report_id})
    return ReportResponse.from_report(*stored)


@router.get(
    "/entities/{entity_id}/reports",
    response_model=list[ReportResponse],
    summary="List research notes for a company",
    dependencies=[Depends(require_permission(PERMISSION_READ_RESEARCH))],
)
async def list_reports(
    entity_id: str,
    container: ContainerDep,
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=100),
    publishable_only: bool = Query(
        default=False,
        description=(
            "Return only reports that passed both grounding checks. Off by "
            "default: an ungrounded report is evidence about the model, and "
            "hiding it by default would make that evidence hard to find."
        ),
    ),
) -> list[ReportResponse]:
    """Reports for one entity, newest first."""
    stored = await container.report_store(session).list_for_entity(
        entity_id, limit=limit, publishable_only=publishable_only
    )
    return [ReportResponse.from_report(report_id, report) for report_id, report in stored]


__all__ = ["router"]
