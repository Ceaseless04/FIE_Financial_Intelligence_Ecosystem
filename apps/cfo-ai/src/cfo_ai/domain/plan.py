"""Budgets and actuals.

Both are the same shape — an amount against an account, a cost centre, and a
period — so they share one line type. What differs is provenance: an actual is a
FACT booked in a ledger, while a budget is a plan somebody decided on. Keeping
them in one structure means variance is a subtraction over matching keys rather
than a reconciliation between two dialects.

Two integrity rules are enforced at construction, both of them the kind of thing
that is invisible until a total is wrong:

1. **One line per key.** Two rows for the same account, centre, and period is
   not extra detail; it is a double count, and it inflates the total silently.
2. **Every line's period belongs to the plan's period.** A December actual
   posted into a November plan moves a miss into the wrong month.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from fie_common.utils import new_id, utc_now
from fie_finance.money import Money, sum_money
from fie_finance.periods import FiscalPeriod
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import Provenance, SourceReference


class PlanKind(StrEnum):
    """What a set of lines represents."""

    #: The approved plan for a period.
    BUDGET = "budget"
    #: What was actually booked.
    ACTUAL = "actual"
    #: A revised expectation issued mid-year, replacing the budget for
    #: comparison purposes but never overwriting it.
    REFORECAST = "reforecast"


class PlanLine(FrozenModel):
    """One amount, against one account, in one cost centre, for one period."""

    account_code: str = Field(min_length=1, max_length=32)
    cost_centre_id: str = Field(min_length=1, max_length=64)
    period: FiscalPeriod
    amount: Money
    #: Free-form note from the planner or the ledger export.
    memo: str | None = Field(default=None, max_length=500)

    @field_validator("account_code")
    @classmethod
    def _normalise_code(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity of this line: account, centre, period.

        Two lines sharing a key are the same line, which is what makes a
        duplicate a double count rather than additional detail.
        """
        return (self.account_code, self.cost_centre_id, self.period.key)


class Plan(FrozenModel):
    """A complete set of lines for one entity and period.

    Provenance is required. A budget nobody approved and an actuals file nobody
    can trace back to a ledger export are both figures without an owner, and the
    variance report built on them inherits that.
    """

    id: str = Field(default_factory=lambda: new_id("pln"))
    entity_id: str = Field(min_length=1, max_length=64)
    kind: PlanKind
    period: FiscalPeriod
    currency: str = Field(default="USD", min_length=3, max_length=3)
    #: A budget can be revised; the label distinguishes "FY26 approved" from
    #: "FY26 board revision 2" without either overwriting the other.
    version: str = Field(default="v1", min_length=1, max_length=64)
    lines: list[PlanLine] = Field(min_length=1)
    provenance: Provenance
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate(self) -> Plan:
        seen: dict[tuple[str, str, str], None] = {}
        for line in self.lines:
            if line.key in seen:
                raise ValueError(
                    "duplicate plan line for account "
                    f"{line.account_code!r} / cost centre {line.cost_centre_id!r} / "
                    f"period {line.period.label}; two rows for one key double-count "
                    "into every total built from them"
                )
            seen[line.key] = None

            if line.amount.currency != self.currency:
                raise ValueError(
                    f"line {line.account_code!r} is in {line.amount.currency} but the "
                    f"plan is in {self.currency}; convert explicitly with a stated rate"
                )

            if not self._period_contains(line.period):
                raise ValueError(
                    f"line {line.account_code!r} covers {line.period.label}, which is "
                    f"outside the plan period {self.period.label}"
                )
        return self

    def _period_contains(self, period: FiscalPeriod) -> bool:
        """Whether a line's period sits inside the plan's.

        A plan for a fiscal year legitimately holds monthly or quarterly lines,
        so this is containment by date rather than equality of kind.
        """
        if period.key == self.period.key:
            return True
        if period.fiscal_year != self.period.fiscal_year:
            return False
        start = self.period.start_date
        if start is not None and period.end_date < start:
            return False
        return period.end_date <= self.period.end_date

    # -- lookups -------------------------------------------------------------

    @property
    def by_key(self) -> dict[tuple[str, str, str], PlanLine]:
        return {line.key: line for line in self.lines}

    def lines_for_account(self, account_code: str) -> list[PlanLine]:
        wanted = account_code.strip().upper()
        return [line for line in self.lines if line.account_code == wanted]

    def lines_for_centre(self, cost_centre_id: str) -> list[PlanLine]:
        return [line for line in self.lines if line.cost_centre_id == cost_centre_id]

    def total(self) -> Money:
        """Sum of every line. Currency is guaranteed uniform by validation."""
        return sum_money([line.amount for line in self.lines], currency=self.currency)

    def total_for_account(self, account_code: str) -> Money:
        return sum_money(
            [line.amount for line in self.lines_for_account(account_code)],
            currency=self.currency,
        )

    def total_for_centre(self, cost_centre_id: str) -> Money:
        return sum_money(
            [line.amount for line in self.lines_for_centre(cost_centre_id)],
            currency=self.currency,
        )

    @property
    def account_codes(self) -> list[str]:
        seen: dict[str, None] = {}
        for line in self.lines:
            seen.setdefault(line.account_code, None)
        return list(seen)

    @property
    def cost_centre_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for line in self.lines:
            seen.setdefault(line.cost_centre_id, None)
        return list(seen)

    @property
    def sources(self) -> list[SourceReference]:
        return list(self.provenance.sources)


def plan_from_lines(
    *,
    entity_id: str,
    kind: PlanKind,
    period: FiscalPeriod,
    lines: Iterable[PlanLine],
    provenance: Provenance,
    currency: str = "USD",
    version: str = "v1",
) -> Plan:
    """Build a plan, keeping the constructor's validation in one place."""
    return Plan(
        entity_id=entity_id,
        kind=kind,
        period=period,
        currency=currency,
        version=version,
        lines=list(lines),
        provenance=provenance,
    )


def line(
    account_code: str,
    cost_centre_id: str,
    period: FiscalPeriod,
    amount: Decimal | int | str,
    *,
    currency: str = "USD",
    memo: str | None = None,
) -> PlanLine:
    """Convenience constructor. ``amount`` never accepts a float."""
    return PlanLine(
        account_code=account_code,
        cost_centre_id=cost_centre_id,
        period=period,
        amount=Money.of(amount, currency),
        memo=memo,
    )


__all__ = ["Plan", "PlanKind", "PlanLine", "line", "plan_from_lines"]
