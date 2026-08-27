"""Variance reporting.

No handler here reaches an AI provider, and no computation behind one can:
:mod:`cfo_ai.analysis` imports nothing from :mod:`fie_ai`, and a test asserts
it. Everything served here is arithmetic over stored plans, with the direction
decided by the chart of accounts.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from cfo_ai.analysis.compare import roll_up
from cfo_ai.api.dependencies import (
    PERMISSION_READ_VARIANCE,
    ContainerDep,
    SessionDep,
    require_permission,
)
from cfo_ai.api.schemas import PeriodRequest, VarianceResponse
from cfo_ai.domain.plan import PlanKind

router = APIRouter(prefix="/api/v1", tags=["variance"])


@router.post(
    "/variance",
    response_model=VarianceResponse,
    summary="Budget versus actual for a period",
    dependencies=[Depends(require_permission(PERMISSION_READ_VARIANCE))],
)
async def compute_variance(
    period: PeriodRequest,
    container: ContainerDep,
    session: SessionDep,
    entity_id: str = Query(min_length=1, max_length=64),
    budget_version: str | None = Query(default=None, max_length=64),
    include_rollups: bool = Query(default=True),
) -> VarianceResponse:
    """Compare the stored budget against the stored actuals.

    Every line comes back with its direction stated: the sign, the verdict, and
    a sentence carrying both. Favourable, unfavourable, and not-assessed are
    separate lists rather than one a client has to bucket, because bucketing
    would mean re-deriving the direction rule this service exists to decide
    once.

    Computed on demand rather than stored. A variance is a pure function of the
    two plans and the chart; a cached one would go stale the moment a line is
    restated, with nothing to mark it as old.

    Raises:
        NotFoundError: if either plan or the chart of accounts is missing. A
            variance without the chart would have no direction, which is worse
            than no variance at all.
    """
    fiscal_period = period.to_period()
    report = await container.reporting(session).variance(
        entity_id, fiscal_period, budget_version=budget_version
    )

    rollups = []
    if include_rollups:
        tree = await container.charts(session).cost_centres(entity_id)
        if tree is not None:
            plans = container.plans(session)
            budget = await plans.find(
                entity_id, kind=PlanKind.BUDGET, period=fiscal_period, version=budget_version
            )
            actual = await plans.find(entity_id, kind=PlanKind.ACTUAL, period=fiscal_period)
            if budget is not None and actual is not None:
                rollups = roll_up(report, tree, (budget, actual))

    return VarianceResponse.from_report(report, rollups)


__all__ = ["router"]
