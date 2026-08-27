"""Financial ratio analysis.

Pure functions over reported figures. No model call is reachable from this
module, which is the point: a margin is arithmetic, and arithmetic delegated to
a language model is arithmetic nobody can reproduce or defend.

Every ratio returns a :class:`~fie_finance.metrics.Metric` carrying the
inputs it used, so a reviewer can recompute it by hand from the report itself.
A ratio whose inputs the filing did not disclose comes back marked unavailable
rather than as a zero that reads like a measurement.
"""

from __future__ import annotations

from decimal import Decimal

from atlas.domain.statements import BalanceSheet, FinancialStatements, IncomeStatement
from fie_finance.metrics import AnalysisResult, Metric, Unit, derived, unavailable
from fie_finance.money import Money
from fie_schemas.provenance import SourceReference

_VERSION = "v1"


def _percent(numerator: Money, denominator: Money) -> Decimal:
    """Ratio expressed in percentage points."""
    return numerator.ratio_to(denominator) * Decimal(100)


def _ratio_metric(
    name: str,
    numerator: Money | None,
    denominator: Money | None,
    *,
    unit: Unit,
    computation: str,
    numerator_label: str,
    denominator_label: str,
    sources: list[SourceReference],
) -> Metric:
    """Compute a two-input ratio, reporting precisely why it could not be."""
    if numerator is None:
        return unavailable(
            name, unit, reason=f"{numerator_label} was not disclosed", computation=computation
        )
    if denominator is None:
        return unavailable(
            name, unit, reason=f"{denominator_label} was not disclosed", computation=computation
        )
    if denominator.is_zero:
        # Dividing by a zero denominator is not a very large ratio; it is an
        # undefined one, and reporting it as infinite would be a fabrication.
        return unavailable(
            name,
            unit,
            reason=f"{denominator_label} is zero, so the ratio is undefined",
            computation=computation,
        )

    value = (
        _percent(numerator, denominator)
        if unit is Unit.PERCENT
        else numerator.ratio_to(denominator)
    )
    return derived(
        name,
        value,
        unit,
        computation=computation,
        sources=sources,
        inputs={numerator_label: numerator, denominator_label: denominator},
    )


# -- profitability -----------------------------------------------------------


def gross_margin(income: IncomeStatement) -> Metric:
    """Gross profit as a percentage of revenue."""
    gross = income.gross_profit
    if gross is None and income.cost_of_revenue is not None:
        gross = income.revenue - income.cost_of_revenue
    return _ratio_metric(
        "gross_margin",
        gross,
        income.revenue,
        unit=Unit.PERCENT,
        computation=f"atlas.ratios.gross_margin.{_VERSION}",
        numerator_label="gross_profit",
        denominator_label="revenue",
        sources=income.sources,
    )


def operating_margin(income: IncomeStatement) -> Metric:
    return _ratio_metric(
        "operating_margin",
        income.operating_income,
        income.revenue,
        unit=Unit.PERCENT,
        computation=f"atlas.ratios.operating_margin.{_VERSION}",
        numerator_label="operating_income",
        denominator_label="revenue",
        sources=income.sources,
    )


def net_margin(income: IncomeStatement) -> Metric:
    return _ratio_metric(
        "net_margin",
        income.net_income,
        income.revenue,
        unit=Unit.PERCENT,
        computation=f"atlas.ratios.net_margin.{_VERSION}",
        numerator_label="net_income",
        denominator_label="revenue",
        sources=income.sources,
    )


def return_on_equity(income: IncomeStatement, balance: BalanceSheet) -> Metric:
    """Net income over shareholders' equity.

    Uses period-end equity rather than an average: an average needs the prior
    balance sheet, and quietly substituting the ending value would make the
    number incomparable to one computed properly elsewhere. The averaged form
    belongs behind its own name.
    """
    return _ratio_metric(
        "return_on_equity",
        income.net_income,
        balance.shareholders_equity,
        unit=Unit.PERCENT,
        computation=f"atlas.ratios.return_on_equity.{_VERSION}",
        numerator_label="net_income",
        denominator_label="shareholders_equity",
        sources=[*income.sources, *balance.sources],
    )


def return_on_assets(income: IncomeStatement, balance: BalanceSheet) -> Metric:
    return _ratio_metric(
        "return_on_assets",
        income.net_income,
        balance.total_assets,
        unit=Unit.PERCENT,
        computation=f"atlas.ratios.return_on_assets.{_VERSION}",
        numerator_label="net_income",
        denominator_label="total_assets",
        sources=[*income.sources, *balance.sources],
    )


# -- liquidity ---------------------------------------------------------------


def current_ratio(balance: BalanceSheet) -> Metric:
    return _ratio_metric(
        "current_ratio",
        balance.current_assets,
        balance.current_liabilities,
        unit=Unit.TIMES,
        computation=f"atlas.ratios.current_ratio.{_VERSION}",
        numerator_label="current_assets",
        denominator_label="current_liabilities",
        sources=balance.sources,
    )


def quick_ratio(balance: BalanceSheet) -> Metric:
    """Current ratio excluding inventory."""
    computation = f"atlas.ratios.quick_ratio.{_VERSION}"
    if balance.current_assets is None:
        return unavailable(
            "quick_ratio",
            Unit.TIMES,
            reason="current assets were not disclosed",
            computation=computation,
        )
    liquid = (
        balance.current_assets - balance.inventory
        if balance.inventory is not None
        else balance.current_assets
    )
    return _ratio_metric(
        "quick_ratio",
        liquid,
        balance.current_liabilities,
        unit=Unit.TIMES,
        computation=computation,
        numerator_label="current_assets_less_inventory",
        denominator_label="current_liabilities",
        sources=balance.sources,
    )


# -- leverage ----------------------------------------------------------------


def debt_to_equity(balance: BalanceSheet) -> Metric:
    """Interest-bearing debt over equity.

    Uses ``total_debt`` rather than total liabilities: including payables and
    deferred revenue inflates the figure and makes it incomparable to how the
    ratio is normally quoted.
    """
    return _ratio_metric(
        "debt_to_equity",
        balance.total_debt,
        balance.shareholders_equity,
        unit=Unit.TIMES,
        computation=f"atlas.ratios.debt_to_equity.{_VERSION}",
        numerator_label="total_debt",
        denominator_label="shareholders_equity",
        sources=balance.sources,
    )


def interest_coverage(income: IncomeStatement) -> Metric:
    """Operating income over interest expense."""
    computation = f"atlas.ratios.interest_coverage.{_VERSION}"
    expense = income.interest_expense
    if expense is not None:
        # Filings report interest expense as a positive cost or a negative
        # entry depending on presentation; the ratio wants the magnitude.
        expense = abs(expense)
    return _ratio_metric(
        "interest_coverage",
        income.operating_income,
        expense,
        unit=Unit.TIMES,
        computation=computation,
        numerator_label="operating_income",
        denominator_label="interest_expense",
        sources=income.sources,
    )


# -- per share ---------------------------------------------------------------


def diluted_eps(income: IncomeStatement) -> Metric:
    """Net income per diluted share."""
    computation = f"atlas.ratios.diluted_eps.{_VERSION}"
    if income.diluted_shares is None:
        return unavailable(
            "diluted_eps",
            Unit.MONEY,
            reason="diluted share count was not disclosed",
            computation=computation,
        )
    per_share = income.net_income.divided_by(income.diluted_shares.count)
    return derived(
        "diluted_eps",
        per_share.amount,
        Unit.MONEY,
        computation=computation,
        sources=income.sources,
        inputs={"net_income": income.net_income, "diluted_shares": income.diluted_shares.count},
        currency=per_share.currency,
    )


# -- composition -------------------------------------------------------------


def analyze(statements: FinancialStatements) -> AnalysisResult:
    """Compute every ratio the supplied statements can support.

    Metrics that require a statement which is absent come back marked
    unavailable with the reason, so a caller can tell "the filing did not
    disclose this" from "Atlas did not try".
    """
    metrics: list[Metric] = []
    income = statements.income_statement
    balance = statements.balance_sheet

    if income is not None:
        metrics.extend(
            [
                gross_margin(income),
                operating_margin(income),
                net_margin(income),
                interest_coverage(income),
                diluted_eps(income),
            ]
        )
    else:
        for name, unit in (
            ("gross_margin", Unit.PERCENT),
            ("operating_margin", Unit.PERCENT),
            ("net_margin", Unit.PERCENT),
            ("interest_coverage", Unit.TIMES),
            ("diluted_eps", Unit.MONEY),
        ):
            metrics.append(
                unavailable(
                    name,
                    unit,
                    reason="no income statement was extracted for this period",
                    computation=f"atlas.ratios.{name}.{_VERSION}",
                )
            )

    if balance is not None:
        metrics.extend([current_ratio(balance), quick_ratio(balance), debt_to_equity(balance)])
    else:
        for name in ("current_ratio", "quick_ratio", "debt_to_equity"):
            metrics.append(
                unavailable(
                    name,
                    Unit.TIMES,
                    reason="no balance sheet was extracted for this period",
                    computation=f"atlas.ratios.{name}.{_VERSION}",
                )
            )

    if income is not None and balance is not None:
        metrics.extend([return_on_equity(income, balance), return_on_assets(income, balance)])
    else:
        for name in ("return_on_equity", "return_on_assets"):
            metrics.append(
                unavailable(
                    name,
                    Unit.PERCENT,
                    reason="both an income statement and a balance sheet are required",
                    computation=f"atlas.ratios.{name}.{_VERSION}",
                )
            )

    return AnalysisResult(
        entity_id=statements.entity_id,
        period_label=statements.period.label,
        metrics=metrics,
    )


__all__ = [
    "analyze",
    "current_ratio",
    "debt_to_equity",
    "diluted_eps",
    "gross_margin",
    "interest_coverage",
    "net_margin",
    "operating_margin",
    "quick_ratio",
    "return_on_assets",
    "return_on_equity",
]
