"""Deterministic analysis and valuation.

No handler in this module reaches an AI provider, and no computation behind one
can: :mod:`atlas.analysis` has no path to :mod:`fie_ai`, and a test asserts it.
Everything served here is arithmetic over filed figures, reproducible from the
inputs each metric carries.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from atlas.analysis.ratios import analyze
from atlas.api.dependencies import (
    PERMISSION_READ_ANALYSIS,
    PERMISSION_READ_VALUATION,
    ContainerDep,
    SessionDep,
    require_permission,
)
from atlas.api.schemas import AnalysisResponse, ValuationRequest, ValuationResponse
from atlas.domain.money import Rate, to_decimal
from atlas.domain.periods import PeriodKind
from atlas.research.pipeline import ValuationInputs
from fie_common.errors import NotFoundError, ValidationError

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.get(
    "/entities/{entity_id}/analysis",
    response_model=AnalysisResponse,
    summary="Deterministic ratios for a period",
    dependencies=[Depends(require_permission(PERMISSION_READ_ANALYSIS))],
)
async def get_analysis(
    entity_id: str,
    container: ContainerDep,
    session: SessionDep,
    fiscal_year: int | None = Query(default=None, ge=1900, le=2200),
    kind: PeriodKind = Query(default=PeriodKind.ANNUAL),
    quarter: int | None = Query(default=None, ge=1, le=4),
) -> AnalysisResponse:
    """Margins, returns, liquidity, leverage, and coverage for a filed period.

    Metrics the filing did not support are returned separately, each with the
    reason. That list is an answer in its own right: a thin filing and a
    complete one produce different analyses, and only reporting what computed
    would make them look identical.

    Raises:
        NotFoundError: if no statements are on file for the period.
    """
    repository = container.statements(session)
    statements = (
        await repository.latest(entity_id)
        if fiscal_year is None
        else await repository.get_for_period(
            entity_id, fiscal_year=fiscal_year, kind=kind, quarter=quarter
        )
    )
    if statements is None:
        raise NotFoundError(
            "no financial statements are on file for this entity and period",
            details={"entity_id": entity_id, "fiscal_year": fiscal_year},
        )
    return AnalysisResponse.from_analysis(analyze(statements))


@router.post(
    "/valuation/dcf",
    response_model=ValuationResponse,
    summary="Discounted cash flow from filed figures",
    dependencies=[Depends(require_permission(PERMISSION_READ_VALUATION))],
)
async def discounted_cash_flow(
    request: ValuationRequest, container: ContainerDep, session: SessionDep
) -> ValuationResponse:
    """Value a company from its filed free cash flow and your assumptions.

    The assumptions are yours because they are judgements, not measurements.
    They are echoed back with the result, and the provenance on the response is
    an ESTIMATE carrying them — which is what makes the number arguable rather
    than authoritative.

    Raises:
        NotFoundError: if no statements are on file for the entity.
        ValidationError: if the filing does not support a valuation — no free
            cash flow disclosed, or a negative one. Atlas will not invent a base
            cash flow to force a result.
    """
    pipeline = container.research(session)
    statements = await pipeline.load_statements(request.entity_id, request.fiscal_year)

    valuation = pipeline.value(
        request.entity_id,
        statements,
        ValuationInputs(
            discount_rate=Rate.of(to_decimal(request.discount_rate)),
            terminal_growth=Rate.of(to_decimal(request.terminal_growth)),
            forecast_growth=Rate.of(to_decimal(request.forecast_growth)),
            projection_years=request.projection_years,
        ),
    )
    if valuation is None:
        raise ValidationError(
            "the filed statements do not support a valuation: free cash flow was "
            "not disclosed or is not positive",
            details={"entity_id": request.entity_id, "period": statements.period.label},
        )

    metrics = valuation.to_metrics(statements.all_sources())
    return ValuationResponse.from_valuation(valuation, metrics[0].provenance)


__all__ = ["router"]
