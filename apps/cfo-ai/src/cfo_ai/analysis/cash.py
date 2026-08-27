"""Burn and runway.

Runway is the number a CFO is asked for most often and the one most easily
rendered as a lie. Three cases have to be distinguished, and collapsing any two
of them produces a figure that reads as a measurement:

1. **Burning cash.** Runway is a finite number of months, and it is an estimate
   resting on the burn continuing at the observed rate.
2. **Cash-flow positive.** There is no runway to report. Not "infinite", not a
   very large number — the question does not apply, and a dashboard that shows
   ``∞`` next to a currency symbol invites someone to read it as a forecast.
3. **Exactly break-even.** Also not a number of months, and separated from case
   two because "we are not burning" and "we are building cash" are different
   facts about a company.

The distinction is carried in the type, not in a comment, so a caller has to
handle it.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import Field

from fie_common.errors import ValidationError
from fie_finance.metrics import Metric, Unit, estimated, unavailable
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import SourceReference

COMPUTATION = "cfo_ai.runway.v1"

_MONTHS_EXPONENT = Decimal("0.1")


class CashPosture(StrEnum):
    """Which of the three cases a company is in."""

    BURNING = "burning"
    GENERATING = "generating"
    BREAK_EVEN = "break_even"

    @property
    def has_runway(self) -> bool:
        return self is CashPosture.BURNING


class BurnRate(FrozenModel):
    """Average net cash movement per month over an observed window."""

    #: Positive means cash left the business.
    net_burn_per_month: Money
    months_observed: int = Field(ge=1)
    opening_cash: Money
    closing_cash: Money
    posture: CashPosture

    @classmethod
    def from_balances(cls, opening_cash: Money, closing_cash: Money, months: int) -> BurnRate:
        """Derive burn from two cash balances and the time between them.

        Raises:
            ValidationError: on a non-positive window, or on balances in
                different currencies.
        """
        if months < 1:
            raise ValidationError("a burn rate needs at least one month of observation")

        movement = closing_cash - opening_cash  # raises on a currency mismatch
        burn = (-movement).divided_by(months)

        if burn.is_zero:
            posture = CashPosture.BREAK_EVEN
        elif burn.is_negative:
            posture = CashPosture.GENERATING
        else:
            posture = CashPosture.BURNING

        return cls(
            net_burn_per_month=burn,
            months_observed=months,
            opening_cash=opening_cash,
            closing_cash=closing_cash,
            posture=posture,
        )


class Runway(FrozenModel):
    """How long the cash lasts, when that question has an answer."""

    posture: CashPosture
    #: ``None`` whenever the company is not burning. Never a sentinel, never
    #: infinity, never a very large number standing in for "not applicable".
    months: Decimal | None = None
    cash_on_hand: Money
    net_burn_per_month: Money
    #: Present when there is no runway to report, explaining why.
    unavailable_reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.months is not None

    def to_metric(self, sources: list[SourceReference] | None = None) -> Metric:
        """An ESTIMATE when it exists, an explicit unavailable when it does not.

        Runway is never DERIVED. It rests on the burn continuing, which is an
        assumption about the future and not a fact about the past.
        """
        if self.months is None:
            return unavailable(
                "runway_months",
                Unit.YEARS,
                reason=self.unavailable_reason or "the company is not burning cash",
                computation=COMPUTATION,
            )
        return estimated(
            "runway_months",
            self.months,
            Unit.YEARS,
            assumptions={
                "burn": f"continues at {self.net_burn_per_month} per month",
                "cash": f"no financing event; opening balance {self.cash_on_hand}",
            },
            sources=sources,
            inputs={
                "cash_on_hand": str(self.cash_on_hand.amount),
                "net_burn_per_month": str(self.net_burn_per_month.amount),
            },
        )

    def describe(self) -> str:
        """A sentence that cannot be misread as a measurement."""
        if self.months is None:
            return f"No runway figure: {self.unavailable_reason}. Cash on hand {self.cash_on_hand}."
        return (
            f"Approximately {self.months} months of runway at the current burn of "
            f"{self.net_burn_per_month} per month, on {self.cash_on_hand} of cash. "
            "This is an estimate that assumes the burn continues unchanged."
        )


def runway(cash_on_hand: Money, burn: BurnRate) -> Runway:
    """Months of cash remaining, or an explicit absence of one.

    Raises:
        ValidationError: if the cash balance is negative — an overdrawn account
            is a fact the caller must handle rather than something to divide.
    """
    if cash_on_hand.is_negative:
        raise ValidationError(
            "cannot compute runway from a negative cash balance",
            details={"cash_on_hand": str(cash_on_hand.amount)},
        )

    if burn.posture is CashPosture.GENERATING:
        return Runway(
            posture=burn.posture,
            months=None,
            cash_on_hand=cash_on_hand,
            net_burn_per_month=burn.net_burn_per_month,
            unavailable_reason="the business generated cash over the observed window",
        )
    if burn.posture is CashPosture.BREAK_EVEN:
        return Runway(
            posture=burn.posture,
            months=None,
            cash_on_hand=cash_on_hand,
            net_burn_per_month=burn.net_burn_per_month,
            unavailable_reason="net cash movement over the observed window was zero",
        )

    months = cash_on_hand.ratio_to(burn.net_burn_per_month).quantize(
        _MONTHS_EXPONENT, rounding=ROUND_HALF_UP
    )
    return Runway(
        posture=burn.posture,
        months=months,
        cash_on_hand=cash_on_hand,
        net_burn_per_month=burn.net_burn_per_month,
    )


class CashPosition(FrozenModel):
    """A period's cash picture, ready to hand to a narrative."""

    entity_id: str
    period: FiscalPeriod
    burn: BurnRate
    runway: Runway

    def to_metrics(self, sources: list[SourceReference] | None = None) -> list[Metric]:
        return [
            estimated(
                "net_burn_per_month",
                self.burn.net_burn_per_month.amount,
                Unit.MONEY,
                assumptions={
                    "window": f"{self.burn.months_observed} months",
                    "basis": "change in cash balance over the window, averaged",
                },
                currency=self.burn.net_burn_per_month.currency,
                sources=sources,
            ),
            self.runway.to_metric(sources),
        ]


__all__ = [
    "COMPUTATION",
    "BurnRate",
    "CashPosition",
    "CashPosture",
    "Runway",
    "runway",
]
