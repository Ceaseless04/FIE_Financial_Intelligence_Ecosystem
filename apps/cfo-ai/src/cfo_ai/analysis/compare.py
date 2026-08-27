"""Turning two plans into a variance report, and rolling it up the org chart.

Matching is by exact key — account, cost centre, period. Anything unmatched on
either side is reported rather than dropped, because both directions are
findings: an actual with no budget is unplanned spend, and a budget with no
actual is either a line nobody has booked against yet or an accrual somebody
forgot. Quietly discarding them makes the totals reconcile and the report lie.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cfo_ai.analysis.variance import (
    DEFAULT_MATERIAL_AMOUNT,
    DEFAULT_MATERIAL_PERCENT,
    Variance,
    VarianceReport,
)
from cfo_ai.domain.accounts import Account
from cfo_ai.domain.organisation import CostCentreTree
from cfo_ai.domain.plan import Plan, PlanKind
from fie_common.errors import ValidationError
from fie_finance.money import Money
from fie_schemas.base import FrozenModel


@dataclass(frozen=True)
class ComparisonSettings:
    """Thresholds a reader's attention is worth."""

    material_percent: Decimal = DEFAULT_MATERIAL_PERCENT
    material_amount: Decimal = DEFAULT_MATERIAL_AMOUNT


def compare(
    budget: Plan,
    actual: Plan,
    accounts: dict[str, Account],
    *,
    settings: ComparisonSettings | None = None,
) -> VarianceReport:
    """Compare a budget against actuals.

    Args:
        accounts: Chart of accounts by code. Required, not optional: a variance
            has no meaning without the account type that gives it a direction,
            so a missing account is an error rather than a line rendered with a
            neutral verdict.

    Raises:
        ValidationError: if the plans describe different entities, periods, or
            currencies, if either is the wrong kind, or if a line references an
            account that is not in the chart.
    """
    settings = settings or ComparisonSettings()

    if budget.kind is PlanKind.ACTUAL:
        raise ValidationError("the first argument must be a budget or reforecast, not actuals")
    if actual.kind is not PlanKind.ACTUAL:
        raise ValidationError("the second argument must be actuals")
    if budget.entity_id != actual.entity_id:
        raise ValidationError(
            "cannot compare plans for different entities",
            details={"budget": budget.entity_id, "actual": actual.entity_id},
        )
    if budget.period.key != actual.period.key:
        raise ValidationError(
            "cannot compare plans for different periods",
            details={"budget": budget.period.label, "actual": actual.period.label},
        )
    if budget.currency != actual.currency:
        raise ValidationError(
            "cannot compare plans in different currencies; convert explicitly "
            "with a stated exchange rate",
            details={"budget": budget.currency, "actual": actual.currency},
        )

    normalised = {code.strip().upper(): account for code, account in accounts.items()}
    budget_lines = budget.by_key
    actual_lines = actual.by_key

    variances: list[Variance] = []
    for key in sorted(set(budget_lines) | set(actual_lines)):
        account_code = key[0]
        account = normalised.get(account_code)
        if account is None:
            raise ValidationError(
                "a plan line references an account that is not in the chart of "
                "accounts, so its variance has no direction",
                details={"account_code": account_code},
            )

        budget_line = budget_lines.get(key)
        actual_line = actual_lines.get(key)
        period = (budget_line or actual_line).period  # type: ignore[union-attr]

        variances.append(
            Variance.between(
                account,
                key[1],
                period,
                budget_line.amount if budget_line else Money.zero(budget.currency),
                actual_line.amount if actual_line else Money.zero(actual.currency),
                material_percent=settings.material_percent,
                material_amount=settings.material_amount,
            )
        )

    return VarianceReport(
        entity_id=budget.entity_id,
        period=budget.period,
        currency=budget.currency,
        variances=variances,
        unmatched_budget_keys=[
            "/".join(key) for key in sorted(set(budget_lines) - set(actual_lines))
        ],
        unmatched_actual_keys=[
            "/".join(key) for key in sorted(set(actual_lines) - set(budget_lines))
        ],
    )


class CentreRollup(FrozenModel):
    """One cost centre's totals, including everything beneath it."""

    cost_centre_id: str
    cost_centre_name: str
    depth: int
    budget: Money
    actual: Money
    variance: Money
    #: Centres directly beneath this one, for rendering a tree.
    child_ids: list[str]


def roll_up(
    report: VarianceReport, tree: CostCentreTree, plans: tuple[Plan, Plan]
) -> list[CentreRollup]:
    """Total each cost centre including its descendants.

    Every centre's figure is computed from its own leaves rather than by adding
    up previously-computed child totals. Summing summaries is how a rounding
    difference at depth four becomes a reconciliation error at the top, and it
    means a parent's number can be checked directly against the lines beneath
    it rather than against another derived figure.
    """
    budget, actual = plans
    by_id = tree.by_id
    rollups: list[CentreRollup] = []

    for centre in tree.centres:
        included = {centre.id, *tree.descendants_of(centre.id)}
        centre_budget = Money.zero(report.currency)
        centre_actual = Money.zero(report.currency)

        for line in budget.lines:
            if line.cost_centre_id in included:
                centre_budget = centre_budget + line.amount
        for line in actual.lines:
            if line.cost_centre_id in included:
                centre_actual = centre_actual + line.amount

        rollups.append(
            CentreRollup(
                cost_centre_id=centre.id,
                cost_centre_name=by_id[centre.id].name,
                depth=tree.depth_of(centre.id),
                budget=centre_budget,
                actual=centre_actual,
                variance=centre_actual - centre_budget,
                child_ids=[child.id for child in tree.children_of(centre.id)],
            )
        )

    return sorted(rollups, key=lambda rollup: (rollup.depth, rollup.cost_centre_id))


def unassigned_spend(plan: Plan, tree: CostCentreTree) -> list[str]:
    """Cost centres a plan references that the hierarchy does not contain.

    Returned rather than raised. A ledger export routinely carries a centre that
    was created after the org chart was last published, and refusing the whole
    plan over it would block the analysis entirely — but its spend would vanish
    from every roll-up, so it has to be visible.
    """
    known = set(tree.by_id)
    seen: dict[str, None] = {}
    for line in plan.lines:
        if line.cost_centre_id not in known:
            seen.setdefault(line.cost_centre_id, None)
    return list(seen)


__all__ = [
    "CentreRollup",
    "ComparisonSettings",
    "compare",
    "roll_up",
    "unassigned_spend",
]
