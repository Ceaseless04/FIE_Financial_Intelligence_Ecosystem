"""Budget versus actual, and what it means.

The arithmetic is a subtraction. Everything difficult about this module is the
sentence that follows it.

**Sign convention, stated once because assuming it is how this goes wrong.**
Every amount is a positive magnitude in its own natural direction: a $400k
revenue plan is ``+400000`` and a $120k marketing plan is ``+120000``. Costs are
not carried as negatives. So ``actual - budget`` is positive whenever a line
came in *higher* than planned, whatever kind of line it is — and higher is good
news for revenue and bad news for marketing. That asymmetry is the whole point:
the number cannot tell you which, and the account can.

**Favourability is therefore never an argument.** It is derived from
:class:`~cfo_ai.domain.accounts.AccountType` inside the constructor, so there is
no path through this code that computes a difference and then decides
separately what it means. A caller cannot pass in the wrong verdict because a
caller cannot pass in a verdict at all.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import Field

from cfo_ai.domain.accounts import Account, AccountType
from fie_finance.metrics import Metric, Unit, derived
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import SourceReference

#: Identifier of this calculation, versioned. Changing the formula changes
#: historical results, and a reader needs to know which one produced a number.
COMPUTATION = "cfo_ai.variance.v1"

#: A line has to move by at least this fraction of plan before it is worth a
#: reader's attention, *and* clear the absolute floor below.
DEFAULT_MATERIAL_PERCENT = Decimal("5")

#: Absolute floor. Without it, 100% over on a $12 line ranks alongside a
#: $2m overspend, and a variance report that cries wolf gets skimmed.
DEFAULT_MATERIAL_AMOUNT = Decimal("10000")

_PERCENT_EXPONENT = Decimal("0.01")


class Favourability(StrEnum):
    """What a variance means for the business."""

    #: Better than plan: revenue above, or cost below.
    FAVOURABLE = "favourable"
    #: Worse than plan.
    UNFAVOURABLE = "unfavourable"
    #: Exactly on plan.
    ON_PLAN = "on_plan"
    #: The account carries no value judgement — see ``AccountType.has_direction``.
    NOT_ASSESSED = "not_assessed"

    @property
    def is_judgement(self) -> bool:
        """Whether this verdict actually asserts good or bad."""
        return self in (Favourability.FAVOURABLE, Favourability.UNFAVOURABLE)


def _favourability(account_type: AccountType, delta: Money) -> Favourability:
    """The one place a direction is decided.

    Private, and called only from :meth:`Variance.between`, so the verdict
    cannot be constructed independently of the numbers it describes.
    """
    if not account_type.has_direction:
        return Favourability.NOT_ASSESSED
    if delta.is_zero:
        return Favourability.ON_PLAN
    over_plan = not delta.is_negative
    if account_type.is_inflow:
        return Favourability.FAVOURABLE if over_plan else Favourability.UNFAVOURABLE
    # A cost above plan is an overspend, whatever its size.
    return Favourability.UNFAVOURABLE if over_plan else Favourability.FAVOURABLE


class Variance(FrozenModel):
    """One line's plan, outcome, and the difference between them."""

    account_code: str
    account_name: str
    account_type: AccountType
    cost_centre_id: str
    period: FiscalPeriod
    budget: Money
    actual: Money
    #: ``actual - budget``. Positive means the line came in higher than planned,
    #: which is favourable or not depending on the account.
    amount: Money
    #: Percentage of plan, or ``None`` when the plan was zero. Never infinity:
    #: a division by zero rendered as a number is a number nobody can act on.
    percent: Decimal | None = None
    favourability: Favourability
    #: Whether this line clears both materiality thresholds.
    is_material: bool = False

    @classmethod
    def between(
        cls,
        account: Account,
        cost_centre_id: str,
        period: FiscalPeriod,
        budget: Money,
        actual: Money,
        *,
        material_percent: Decimal = DEFAULT_MATERIAL_PERCENT,
        material_amount: Decimal = DEFAULT_MATERIAL_AMOUNT,
    ) -> Variance:
        """Compare a plan against an outcome.

        Raises:
            ValueError: if the two amounts are in different currencies. Money
                refuses the subtraction, which is what makes this safe rather
                than plausible.
        """
        delta = actual - budget  # raises on a currency mismatch

        percent: Decimal | None = None
        if not budget.is_zero:
            percent = (delta.amount / budget.amount.copy_abs() * Decimal(100)).quantize(
                _PERCENT_EXPONENT, rounding=ROUND_HALF_UP
            )

        material = delta.amount.copy_abs() >= material_amount and (
            percent is None or percent.copy_abs() >= material_percent
        )
        # A spend against no plan at all has no percentage, so it is judged on
        # size alone rather than being silently dropped from the report.
        if budget.is_zero:
            material = delta.amount.copy_abs() >= material_amount

        return cls(
            account_code=account.code,
            account_name=account.name,
            account_type=account.type,
            cost_centre_id=cost_centre_id,
            period=period,
            budget=budget,
            actual=actual,
            amount=delta,
            percent=percent,
            favourability=_favourability(account.type, delta),
            is_material=material,
        )

    @property
    def is_overspend(self) -> bool:
        """A cost line above plan. Named separately from unfavourable because a
        revenue miss is also unfavourable and is not an overspend."""
        return self.account_type.is_cost and not self.amount.is_negative and not self.amount.is_zero

    @property
    def absolute_amount(self) -> Money:
        return abs(self.amount)

    def to_metric(self, sources: list[SourceReference] | None = None) -> Metric:
        """Render as a provenance-bound metric.

        DERIVED, so it cannot be attributed to a model, and carrying the inputs
        so the subtraction is reproducible by anyone who doubts it.
        """
        return derived(
            f"variance.{self.account_code}.{self.cost_centre_id}",
            self.amount.amount,
            Unit.MONEY,
            computation=COMPUTATION,
            currency=self.amount.currency,
            sources=sources,
            inputs={
                "budget": str(self.budget.amount),
                "actual": str(self.actual.amount),
                "account_type": str(self.account_type),
                "favourability": str(self.favourability),
            },
        )

    def describe(self) -> str:
        """One line a human can read, with the direction spelled out.

        Used in the prompt so the model is told what the variance means rather
        than left to infer it from a sign — inferring it is the mistake this
        whole module exists to prevent.
        """
        movement = "over" if not self.amount.is_negative else "under"
        share = f" ({self.percent}% {movement} plan)" if self.percent is not None else ""
        verdict = (
            f" — {self.favourability}"
            if self.favourability.is_judgement
            else " — direction not assessed for this account type"
        )
        return (
            f"{self.account_name} [{self.account_code}] in {self.cost_centre_id}: "
            f"plan {self.budget}, actual {self.actual}, "
            f"{self.absolute_amount} {movement}{share}{verdict}"
        )


class VarianceReport(FrozenModel):
    """Every line's variance for one entity and period."""

    entity_id: str = Field(min_length=1)
    period: FiscalPeriod
    currency: str = Field(min_length=3, max_length=3)
    variances: list[Variance] = Field(default_factory=list)
    #: Lines present in one plan and missing from the other. Reported rather
    #: than skipped: an actual with no budget is unplanned spend, and a budget
    #: with no actual may be a line nobody booked against yet or a forgotten
    #: accrual. Both are findings.
    unmatched_budget_keys: list[str] = Field(default_factory=list)
    unmatched_actual_keys: list[str] = Field(default_factory=list)

    @property
    def material(self) -> list[Variance]:
        """Material lines, worst money first."""
        return sorted(
            (variance for variance in self.variances if variance.is_material),
            key=lambda variance: variance.absolute_amount.amount,
            reverse=True,
        )

    @property
    def unfavourable(self) -> list[Variance]:
        return [
            variance
            for variance in self.variances
            if variance.favourability is Favourability.UNFAVOURABLE
        ]

    @property
    def favourable(self) -> list[Variance]:
        return [
            variance
            for variance in self.variances
            if variance.favourability is Favourability.FAVOURABLE
        ]

    @property
    def not_assessed(self) -> list[Variance]:
        """Lines whose direction the system declines to judge."""
        return [
            variance
            for variance in self.variances
            if variance.favourability is Favourability.NOT_ASSESSED
        ]

    def for_account(self, account_code: str) -> list[Variance]:
        wanted = account_code.strip().upper()
        return [variance for variance in self.variances if variance.account_code == wanted]

    def net_operating_variance(self) -> Money:
        """Net effect on the operating result.

        Revenue variances add; cost variances subtract. Capital expenditure and
        cash movements are excluded — spending cash on a building is not an
        operating expense, and folding it in overstates the miss.
        """
        total = Money.zero(self.currency)
        for variance in self.variances:
            if not variance.account_type.affects_profit:
                continue
            if variance.account_type.is_cost:
                total = total - variance.amount
            else:
                total = total + variance.amount
        return total

    def to_metrics(self, sources: list[SourceReference] | None = None) -> list[Metric]:
        metrics = [variance.to_metric(sources) for variance in self.variances]
        metrics.append(
            derived(
                "net_operating_variance",
                self.net_operating_variance().amount,
                Unit.MONEY,
                computation=COMPUTATION,
                currency=self.currency,
                sources=sources,
                inputs={"lines": str(len(self.variances))},
            )
        )
        return metrics


__all__ = [
    "COMPUTATION",
    "DEFAULT_MATERIAL_AMOUNT",
    "DEFAULT_MATERIAL_PERCENT",
    "Favourability",
    "Variance",
    "VarianceReport",
]
