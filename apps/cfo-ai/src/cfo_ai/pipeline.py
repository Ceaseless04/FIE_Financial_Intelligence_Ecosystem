"""The two workflows CFO.ai runs, in the order that makes them trustworthy.

Planning: a plan is stored, and the cost centres it references are checked
against the published hierarchy. Anything unrecognised is announced rather than
absorbed, because spend in a centre the org chart does not know about is
invisible to every roll-up built from it.

Reporting: budget and actuals are loaded, variance is computed deterministically
with the direction decided by the chart of accounts, cash and forecasts are
added as explicit estimates — and only then is a model asked what it all means.
The commentary that comes back is checked against both the figures and their
directions before anyone sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from cfo_ai.analysis.cash import CashPosition
from cfo_ai.analysis.compare import ComparisonSettings, compare, unassigned_spend
from cfo_ai.analysis.forecast import Forecast
from cfo_ai.analysis.variance import VarianceReport
from cfo_ai.domain.plan import Plan, PlanKind
from cfo_ai.events import (
    CHART_UPDATED,
    COMMENTARY_PUBLISHED,
    COMMENTARY_WITHHELD,
    PLAN_STORED,
    RUNWAY_ASSESSED,
    UNPLANNED_SPEND_DETECTED,
    VARIANCE_COMPUTED,
    ChartUpdatedPayload,
    CommentaryPayload,
    PlanStoredPayload,
    RunwayAssessedPayload,
    UnplannedSpendPayload,
    VarianceComputedPayload,
    build_event,
)
from cfo_ai.research.commentary import Commentary, CommentaryRequest, CommentaryService
from cfo_ai.storage.repository import ChartRepository, CommentaryRepository, PlanRepository
from fie_common.errors import NotFoundError, ValidationError
from fie_events.bus import EventBus
from fie_finance.periods import FiscalPeriod
from fie_observability.logging import get_logger
from fie_observability.tracing import traced
from fie_schemas.base import FrozenModel

logger = get_logger(__name__)


class PlanStored(FrozenModel):
    """What storing a plan produced."""

    plan_id: str
    entity_id: str
    line_count: int = Field(ge=0)
    total: str
    #: Centres the hierarchy does not contain. Their spend rolls up nowhere.
    unassigned_cost_centres: list[str] = Field(default_factory=list)


class ReportOutcome(FrozenModel):
    """A variance report and, when asked for, the commentary on it."""

    entity_id: str
    period_label: str
    variances: VarianceReport
    commentary_id: str | None = None
    commentary: Commentary | None = None


@dataclass
class PlanningPipeline:
    """Stores charts and plans, and says what the org chart cannot account for."""

    charts: ChartRepository
    plans: PlanRepository
    bus: EventBus | None = None

    async def publish_chart(self, entity_id: str, accounts: dict[str, object]) -> int:
        """Replace a company's chart of accounts."""
        from cfo_ai.domain.accounts import Account

        typed = [account for account in accounts.values() if isinstance(account, Account)]
        count = await self.charts.replace_accounts(entity_id, typed)
        await self._publish(
            CHART_UPDATED,
            ChartUpdatedPayload(
                entity_id=entity_id,
                account_count=count,
                account_types={account.code: str(account.type) for account in typed},
            ),
        )
        return count

    async def store_plan(self, plan: Plan) -> PlanStored:
        """Store a plan and report what the hierarchy cannot place."""
        with traced(
            "cfo_ai.plan.store",
            attributes={"entity_id": plan.entity_id, "kind": str(plan.kind)},
        ):
            plan_id = await self.plans.save(plan)

            tree = await self.charts.cost_centres(plan.entity_id)
            unassigned = unassigned_spend(plan, tree) if tree is not None else []

            await self._publish(
                PLAN_STORED,
                PlanStoredPayload(
                    plan_id=plan_id,
                    entity_id=plan.entity_id,
                    kind=str(plan.kind),
                    period_label=plan.period.label,
                    version=plan.version,
                    currency=plan.currency,
                    line_count=len(plan.lines),
                    total=str(plan.total().amount),
                    unassigned_cost_centres=unassigned,
                ),
            )

            if unassigned:
                logger.warning(
                    "plan_references_unknown_cost_centres",
                    entity_id=plan.entity_id,
                    centres=unassigned,
                )

            return PlanStored(
                plan_id=plan_id,
                entity_id=plan.entity_id,
                line_count=len(plan.lines),
                total=str(plan.total().amount),
                unassigned_cost_centres=unassigned,
            )

    async def _publish(self, event_type: str, payload: FrozenModel) -> None:
        if self.bus is None:
            return
        try:
            await self.bus.publish(build_event(event_type, payload))
        except Exception as error:  # noqa: BLE001 — an event must not fail the work
            logger.warning("event_publish_failed", event_type=event_type, error=str(error))


@dataclass
class ReportingPipeline:
    """Computes variance, then optionally has it explained and verifies the prose."""

    charts: ChartRepository
    plans: PlanRepository
    commentaries: CommentaryRepository
    service: CommentaryService
    settings: ComparisonSettings | None = None
    bus: EventBus | None = None

    async def variance(
        self,
        entity_id: str,
        period: FiscalPeriod,
        *,
        budget_version: str | None = None,
    ) -> VarianceReport:
        """Compare the stored budget against the stored actuals.

        Raises:
            NotFoundError: if either plan, or the chart of accounts, is missing.
                A variance computed without the chart would have no direction,
                which is worse than no variance at all.
        """
        budget = await self.plans.find(
            entity_id, kind=PlanKind.BUDGET, period=period, version=budget_version
        )
        if budget is None:
            raise NotFoundError(
                "no budget is on file for this entity and period",
                details={"entity_id": entity_id, "period": period.label},
            )

        actual = await self.plans.find(entity_id, kind=PlanKind.ACTUAL, period=period)
        if actual is None:
            raise NotFoundError(
                "no actuals are on file for this entity and period",
                details={"entity_id": entity_id, "period": period.label},
            )

        accounts = await self.charts.accounts(entity_id)
        if not accounts:
            raise NotFoundError(
                "no chart of accounts is published for this entity, so a variance "
                "would have no direction",
                details={"entity_id": entity_id},
            )

        report = compare(budget, actual, accounts, settings=self.settings)

        await self._publish(
            VARIANCE_COMPUTED,
            VarianceComputedPayload(
                entity_id=entity_id,
                period_label=period.label,
                currency=report.currency,
                net_operating_variance=str(report.net_operating_variance().amount),
                material_count=len(report.material),
                unfavourable_count=len(report.unfavourable),
                favourable_count=len(report.favourable),
                not_assessed_count=len(report.not_assessed),
                largest_variance_accounts=[
                    variance.account_code for variance in report.material[:5]
                ],
            ),
        )

        if report.unmatched_actual_keys:
            await self._publish(
                UNPLANNED_SPEND_DETECTED,
                UnplannedSpendPayload(
                    entity_id=entity_id,
                    period_label=period.label,
                    keys=list(report.unmatched_actual_keys),
                ),
            )

        return report

    async def report(
        self,
        entity_id: str,
        period: FiscalPeriod,
        *,
        company_name: str | None = None,
        budget_version: str | None = None,
        cash: CashPosition | None = None,
        forecasts: list[Forecast] | None = None,
        with_commentary: bool = True,
    ) -> ReportOutcome:
        """Produce a variance report and, optionally, verified commentary on it.

        Raises:
            NotFoundError: if the plans or chart are missing.
            ValidationError: if the report has no variances to write about.
        """
        report = await self.variance(entity_id, period, budget_version=budget_version)

        if cash is not None:
            await self._publish(
                RUNWAY_ASSESSED,
                RunwayAssessedPayload(
                    entity_id=entity_id,
                    period_label=period.label,
                    posture=str(cash.runway.posture),
                    months=str(cash.runway.months) if cash.runway.months is not None else None,
                    net_burn_per_month=str(cash.burn.net_burn_per_month.amount),
                    cash_on_hand=str(cash.runway.cash_on_hand.amount),
                    currency=cash.burn.net_burn_per_month.currency,
                    unavailable_reason=cash.runway.unavailable_reason,
                ),
            )

        if not with_commentary:
            return ReportOutcome(entity_id=entity_id, period_label=period.label, variances=report)

        if not report.variances:
            raise ValidationError(
                "there are no variances to write commentary about",
                details={"entity_id": entity_id, "period": period.label},
            )

        accounts = await self.charts.accounts(entity_id)
        commentary = await self.service.generate(
            CommentaryRequest(
                entity_id=entity_id,
                company_name=company_name or entity_id,
                variances=report,
                accounts=accounts,
                cash=cash,
                forecasts=forecasts or [],
            )
        )
        commentary_id = await self.commentaries.save(commentary)

        # Two event types rather than one with a flag: a commentary that
        # reverses a direction is the more persuasive failure, and a consumer
        # must opt into it deliberately.
        await self._publish(
            COMMENTARY_PUBLISHED if commentary.is_publishable else COMMENTARY_WITHHELD,
            CommentaryPayload(
                commentary_id=commentary_id,
                entity_id=entity_id,
                period_label=commentary.period_label,
                numerically_grounded=commentary.numerically_grounded,
                directionally_grounded=commentary.directionally_grounded,
                unsupported_figures=list(commentary.unsupported_figures),
                direction_contradictions=list(commentary.direction_contradictions),
                regeneration_count=commentary.regeneration_count,
            ),
        )

        return ReportOutcome(
            entity_id=entity_id,
            period_label=period.label,
            variances=report,
            commentary_id=commentary_id,
            commentary=commentary,
        )

    async def _publish(self, event_type: str, payload: FrozenModel) -> None:
        if self.bus is None:
            return
        try:
            await self.bus.publish(build_event(event_type, payload))
        except Exception as error:  # noqa: BLE001 — an event must not fail the work
            logger.warning("event_publish_failed", event_type=event_type, error=str(error))


__all__ = ["PlanStored", "PlanningPipeline", "ReportOutcome", "ReportingPipeline"]
