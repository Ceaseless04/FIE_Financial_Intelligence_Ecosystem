"""Growth and trend analysis.

The recurring failure mode this module exists to prevent is comparing periods
that are not comparable. A quarter measured against a full year, or Q3 against
Q2 in a seasonal business, produces a growth rate that is arithmetically correct
and analytically worthless. :meth:`FiscalPeriod.is_comparable_to` is checked
before any rate is computed, and a mismatch returns an unavailable metric naming
the reason instead of a number.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise

from atlas.analysis.results import Metric, Unit, derived, unavailable
from atlas.domain.statements import FinancialStatements
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod

_VERSION = "v1"
_PRECISION = Decimal("0.000001")


def period_over_period_growth(
    name: str,
    current: Money,
    prior: Money,
    *,
    current_period: FiscalPeriod,
    prior_period: FiscalPeriod,
) -> Metric:
    """Percentage change between two comparable periods.

    A negative prior value makes the percentage meaningless — growth from
    -100 to 50 is not "150% growth" in any sense a reader would accept — so it
    is reported as unavailable rather than rendered.
    """
    computation = f"atlas.growth.{name}.{_VERSION}"

    if not current_period.is_comparable_to(prior_period):
        return unavailable(
            f"{name}_growth",
            Unit.PERCENT,
            reason=(
                f"{current_period.label} and {prior_period.label} are not comparable "
                "periods; a growth rate between them would be misleading"
            ),
            computation=computation,
        )
    if current_period.end_date <= prior_period.end_date:
        return unavailable(
            f"{name}_growth",
            Unit.PERCENT,
            reason="the current period must end after the prior period",
            computation=computation,
        )
    if prior.is_zero:
        return unavailable(
            f"{name}_growth",
            Unit.PERCENT,
            reason="the prior period value is zero, so growth is undefined",
            computation=computation,
        )
    if prior.is_negative:
        return unavailable(
            f"{name}_growth",
            Unit.PERCENT,
            reason=(
                "the prior period value is negative, so a percentage change cannot be interpreted"
            ),
            computation=computation,
        )

    change = (current - prior).ratio_to(prior) * Decimal(100)
    return derived(
        f"{name}_growth",
        change.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
        Unit.PERCENT,
        computation=computation,
        inputs={
            "current": current,
            "prior": prior,
            "current_period": current_period.label,
            "prior_period": prior_period.label,
        },
    )


def compound_annual_growth_rate(
    name: str, first: Money, last: Money, *, years: Decimal | int
) -> Metric:
    """CAGR across a span of years.

    Both endpoints must be positive: the root of a negative number is not real,
    and the conventional workarounds silently change what is being measured.
    """
    computation = f"atlas.growth.{name}_cagr.{_VERSION}"
    span = Decimal(str(years))

    if span <= 0:
        return unavailable(
            f"{name}_cagr",
            Unit.PERCENT,
            reason="a compound growth rate needs a positive number of years",
            computation=computation,
        )
    if first.is_negative or first.is_zero or last.is_negative or last.is_zero:
        return unavailable(
            f"{name}_cagr",
            Unit.PERCENT,
            reason=("a compound growth rate requires positive values at both ends of the span"),
            computation=computation,
        )

    # (last / first) ** (1 / years) - 1, via Decimal's ln/exp so the whole
    # calculation stays in decimal arithmetic rather than round-tripping
    # through binary floating point.
    ratio = last.ratio_to(first)
    rate = (ratio.ln() / span).exp() - Decimal(1)

    return derived(
        f"{name}_cagr",
        (rate * Decimal(100)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
        Unit.PERCENT,
        computation=computation,
        inputs={"first": first, "last": last, "years": span},
    )


def revenue_growth(current: FinancialStatements, prior: FinancialStatements) -> Metric:
    """Revenue growth between two periods."""
    computation = f"atlas.growth.revenue.{_VERSION}"
    if current.income_statement is None or prior.income_statement is None:
        return unavailable(
            "revenue_growth",
            Unit.PERCENT,
            reason="an income statement is required for both periods",
            computation=computation,
        )
    return period_over_period_growth(
        "revenue",
        current.income_statement.revenue,
        prior.income_statement.revenue,
        current_period=current.period,
        prior_period=prior.period,
    )


def net_income_growth(current: FinancialStatements, prior: FinancialStatements) -> Metric:
    computation = f"atlas.growth.net_income.{_VERSION}"
    if current.income_statement is None or prior.income_statement is None:
        return unavailable(
            "net_income_growth",
            Unit.PERCENT,
            reason="an income statement is required for both periods",
            computation=computation,
        )
    return period_over_period_growth(
        "net_income",
        current.income_statement.net_income,
        prior.income_statement.net_income,
        current_period=current.period,
        prior_period=prior.period,
    )


def free_cash_flow(statements: FinancialStatements) -> Metric:
    """Operating cash flow less capital expenditures.

    Named and versioned because "free cash flow" has several accepted
    definitions. Which one produced a number matters when the figure is compared
    against one computed elsewhere, so the definition travels with the result.
    """
    computation = f"atlas.growth.free_cash_flow.{_VERSION}"
    cash_flow = statements.cash_flow_statement
    if cash_flow is None:
        return unavailable(
            "free_cash_flow",
            Unit.MONEY,
            reason="no cash flow statement was extracted for this period",
            computation=computation,
        )
    if cash_flow.capital_expenditures is None:
        return unavailable(
            "free_cash_flow",
            Unit.MONEY,
            reason="capital expenditures were not disclosed",
            computation=computation,
        )

    # Filings present capex as a negative investing outflow or a positive
    # spending figure depending on the statement; the definition subtracts the
    # magnitude either way.
    result = cash_flow.operating_cash_flow - abs(cash_flow.capital_expenditures)
    return derived(
        "free_cash_flow",
        result.amount,
        Unit.MONEY,
        computation=computation,
        sources=cash_flow.sources,
        inputs={
            "operating_cash_flow": cash_flow.operating_cash_flow,
            "capital_expenditures": cash_flow.capital_expenditures,
            "definition": "operating cash flow less capital expenditures",
        },
        currency=result.currency,
    )


def trend(values: list[tuple[FiscalPeriod, Money]]) -> list[Metric]:
    """Period-over-period growth across an ordered series.

    Returns one fewer metric than there are values, since the first period has
    nothing to be compared against.
    """
    ordered = sorted(values, key=lambda item: item[0].end_date)
    metrics: list[Metric] = []
    for (prior_period, prior_value), (current_period, current_value) in pairwise(ordered):
        metrics.append(
            period_over_period_growth(
                f"{current_period.label.replace(' ', '_').lower()}",
                current_value,
                prior_value,
                current_period=current_period,
                prior_period=prior_period,
            )
        )
    return metrics


__all__ = [
    "compound_annual_growth_rate",
    "free_cash_flow",
    "net_income_growth",
    "period_over_period_growth",
    "revenue_growth",
    "trend",
]
