"""The chart of accounts, the organisation, and the plans themselves."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from cfo_ai.api.dependencies import (
    PERMISSION_READ_PLAN,
    PERMISSION_WRITE_PLAN,
    ContainerDep,
    SessionDep,
    require_permission,
)
from cfo_ai.api.schemas import (
    AccountView,
    PlanView,
    PublishChartRequest,
    PublishOrganisationRequest,
    SubmitPlanRequest,
    SubmitPlanResponse,
)
from cfo_ai.domain.plan import PlanKind
from fie_common.errors import NotFoundError, ValidationError

router = APIRouter(prefix="/api/v1", tags=["plans"])


@router.put(
    "/entities/{entity_id}/chart",
    response_model=list[AccountView],
    summary="Publish a chart of accounts",
    dependencies=[Depends(require_permission(PERMISSION_WRITE_PLAN))],
)
async def publish_chart(
    entity_id: str,
    request: PublishChartRequest,
    container: ContainerDep,
    session: SessionDep,
) -> list[AccountView]:
    """Replace a company's chart of accounts.

    A PUT rather than a POST because a chart is published whole. A partial
    update would leave an account nobody meant to keep deciding whether a
    variance on it is good news.

    The response echoes each account's direction rules, so a client never has to
    re-derive "is higher better here" for itself.
    """
    if request.entity_id != entity_id:
        raise ValidationError(
            "the entity in the path and the body must match",
            details={"path": entity_id, "body": request.entity_id},
        )

    accounts = {item.code: item.to_account() for item in request.accounts}
    await container.planning(session).publish_chart(entity_id, dict(accounts))
    return [AccountView.from_account(account) for account in accounts.values()]


@router.get(
    "/entities/{entity_id}/chart",
    response_model=list[AccountView],
    summary="Fetch a chart of accounts",
    dependencies=[Depends(require_permission(PERMISSION_READ_PLAN))],
)
async def get_chart(
    entity_id: str, container: ContainerDep, session: SessionDep
) -> list[AccountView]:
    """The stored chart, with each account's direction rules.

    Raises:
        NotFoundError: if no chart is published for this entity.
    """
    accounts = await container.charts(session).accounts(entity_id)
    if not accounts:
        raise NotFoundError(
            "no chart of accounts is published for this entity",
            details={"entity_id": entity_id},
        )
    return [AccountView.from_account(account) for account in accounts.values()]


@router.put(
    "/entities/{entity_id}/organisation",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Publish a cost centre hierarchy",
    dependencies=[Depends(require_permission(PERMISSION_WRITE_PLAN))],
)
async def publish_organisation(
    entity_id: str,
    request: PublishOrganisationRequest,
    container: ContainerDep,
    session: SessionDep,
) -> None:
    """Replace a company's cost centre hierarchy.

    A dangling parent or a cycle is refused at the edge with a 422: a cycle
    would make every roll-up built from the tree non-terminating, and a dangling
    parent would make a centre's spend vanish from the totals.
    """
    if request.entity_id != entity_id:
        raise ValidationError(
            "the entity in the path and the body must match",
            details={"path": entity_id, "body": request.entity_id},
        )
    await container.charts(session).replace_cost_centres(entity_id, request.to_tree())


@router.post(
    "/plans",
    response_model=SubmitPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a budget, reforecast, or actuals",
    dependencies=[Depends(require_permission(PERMISSION_WRITE_PLAN))],
)
async def submit_plan(
    request: SubmitPlanRequest, container: ContainerDep, session: SessionDep
) -> SubmitPlanResponse:
    """Store a plan.

    Amounts are strings; a JSON number is refused as bad input rather than
    silently rounded. Two lines for one account, centre, and period are refused
    outright, because a duplicate is a double count into every total built from
    it and nothing downstream would notice.

    ``unassigned_cost_centres`` in the response names any centre the published
    hierarchy does not contain. That spend rolls up nowhere, so it is surfaced
    rather than absorbed — the plan is still stored, because a ledger export
    routinely carries a centre created after the org chart was last published.
    """
    if len(request.lines) > container.settings.max_plan_lines:
        raise ValidationError(
            "the plan exceeds the configured line limit",
            details={
                "lines": len(request.lines),
                "limit": container.settings.max_plan_lines,
            },
        )

    stored = await container.planning(session).store_plan(request.to_plan())
    return SubmitPlanResponse.from_stored(stored)


@router.get(
    "/plans/{plan_id}",
    response_model=PlanView,
    summary="Fetch a stored plan",
    dependencies=[Depends(require_permission(PERMISSION_READ_PLAN))],
)
async def get_plan(plan_id: str, container: ContainerDep, session: SessionDep) -> PlanView:
    """Retrieve one plan and its lines.

    Raises:
        NotFoundError: if no such plan is stored.
    """
    plan = await container.plans(session).get(plan_id)
    if plan is None:
        raise NotFoundError("no such plan", details={"plan_id": plan_id})
    return PlanView.from_plan(plan)


@router.get(
    "/entities/{entity_id}/plans",
    response_model=list[PlanView],
    summary="List plans for a company",
    dependencies=[Depends(require_permission(PERMISSION_READ_PLAN))],
)
async def list_plans(
    entity_id: str,
    container: ContainerDep,
    session: SessionDep,
    kind: PlanKind | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[PlanView]:
    """Plans on record, most recent period first.

    Versions coexist: an approved budget and a later board revision are separate
    plans, and neither overwrote the other.
    """
    plans = await container.plans(session).list_for_entity(entity_id, kind=kind, limit=limit)
    return [PlanView.from_plan(plan) for plan in plans]


__all__ = ["router"]
