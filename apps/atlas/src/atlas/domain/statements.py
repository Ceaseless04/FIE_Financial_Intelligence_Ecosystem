"""Financial statements.

Every figure here is a FACT read from a filing, never a computed value — the
derived quantities live in :mod:`atlas.analysis`. Keeping the boundary at the
type level is what lets a report say which numbers came from the document and
which came from Atlas.

The statements validate themselves against the accounting identities they are
required to satisfy. A balance sheet where assets do not equal liabilities plus
equity is not a balance sheet with a small problem; it means the extraction is
wrong, and every ratio computed from it will be wrong in a way that looks
entirely reasonable.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from fie_finance.money import Money, Shares
from fie_finance.periods import FiscalPeriod, PeriodKind
from fie_schemas.base import FrozenModel
from fie_schemas.provenance import Provenance, SourceReference

#: How far the accounting identity may be off before the statement is rejected.
#: Filings are reported in rounded units, so summing rounded components leaves a
#: genuine residue; anything larger is an extraction error, not rounding.
IDENTITY_TOLERANCE = Decimal("0.02")


def _within_tolerance(left: Money, right: Money, *, relative_to: Money | None = None) -> bool:
    """Whether two amounts agree once reporting-rounding is allowed for."""
    difference = abs((left - right).amount)
    if difference == 0:
        return True
    scale = abs((relative_to or left).amount)
    if scale == 0:
        return difference <= IDENTITY_TOLERANCE
    return (difference / scale) <= IDENTITY_TOLERANCE


class StatementBase(FrozenModel):
    """Shared identity and provenance for every statement."""

    period: FiscalPeriod
    currency: str = Field(default="USD", min_length=3, max_length=3)
    #: Where these figures were read from. A statement without a source cannot
    #: appear in a report, because nothing could be cited for it.
    provenance: Provenance

    @property
    def sources(self) -> list[SourceReference]:
        return list(self.provenance.sources)


class IncomeStatement(StatementBase):
    """Performance over a period.

    Fields are optional because filings differ in what they break out, and a
    partially populated statement is more useful than a rejected one — the
    analysis layer states which inputs it required and reports what was missing.
    """

    revenue: Money
    cost_of_revenue: Money | None = None
    gross_profit: Money | None = None
    operating_expenses: Money | None = None
    operating_income: Money | None = None
    interest_expense: Money | None = None
    pretax_income: Money | None = None
    income_tax_expense: Money | None = None
    net_income: Money
    #: Weighted average diluted shares, as reported.
    diluted_shares: Shares | None = None

    @model_validator(mode="after")
    def _validate(self) -> IncomeStatement:
        if not self.period.kind.is_flow:
            raise ValueError(
                "an income statement covers a span; an instant period describes "
                "a balance sheet position"
            )

        # Gross profit must reconcile when both components are present. This
        # catches a transposed figure at extraction time rather than letting it
        # surface as an implausible margin three layers later.
        if (
            self.gross_profit is not None
            and self.cost_of_revenue is not None
            and not _within_tolerance(
                self.revenue - self.cost_of_revenue, self.gross_profit, relative_to=self.revenue
            )
        ):
            raise ValueError(
                "gross profit does not reconcile: revenue minus cost of revenue "
                f"is {self.revenue - self.cost_of_revenue}, statement reports {self.gross_profit}"
            )

        if (
            self.pretax_income is not None
            and self.income_tax_expense is not None
            and not _within_tolerance(
                self.pretax_income - self.income_tax_expense,
                self.net_income,
                relative_to=self.pretax_income,
            )
        ):
            raise ValueError("net income does not reconcile with pretax income less tax expense")
        return self


class BalanceSheet(StatementBase):
    """Position at an instant.

    The fundamental identity — assets equal liabilities plus equity — is checked
    on construction. It is the single most valuable extraction check available,
    because it is true of every balance sheet ever published.
    """

    total_assets: Money
    current_assets: Money | None = None
    cash_and_equivalents: Money | None = None
    inventory: Money | None = None
    total_liabilities: Money
    current_liabilities: Money | None = None
    #: Interest-bearing debt, used for enterprise value and leverage.
    total_debt: Money | None = None
    shareholders_equity: Money

    @model_validator(mode="after")
    def _validate(self) -> BalanceSheet:
        if self.period.kind is not PeriodKind.INSTANT:
            raise ValueError(
                "a balance sheet is a position at an instant, not a flow over a period"
            )

        identity = self.total_liabilities + self.shareholders_equity
        if not _within_tolerance(identity, self.total_assets, relative_to=self.total_assets):
            raise ValueError(
                "balance sheet does not balance: liabilities plus equity is "
                f"{identity}, total assets is {self.total_assets}"
            )

        if self.current_assets is not None and self.current_assets > self.total_assets:
            raise ValueError("current assets cannot exceed total assets")
        if (
            self.current_liabilities is not None
            and self.current_liabilities > self.total_liabilities
        ):
            raise ValueError("current liabilities cannot exceed total liabilities")
        return self

    @property
    def net_debt(self) -> Money | None:
        """Debt net of cash. ``None`` when either component was not reported."""
        if self.total_debt is None or self.cash_and_equivalents is None:
            return None
        return self.total_debt - self.cash_and_equivalents


class CashFlowStatement(StatementBase):
    """Cash movement over a period.

    Free cash flow is deliberately *not* a field: it is a derived quantity with
    more than one accepted definition, so it belongs in the analysis layer where
    the definition used can be named and cited.
    """

    operating_cash_flow: Money
    capital_expenditures: Money | None = None
    investing_cash_flow: Money | None = None
    financing_cash_flow: Money | None = None
    net_change_in_cash: Money | None = None

    @model_validator(mode="after")
    def _validate(self) -> CashFlowStatement:
        if not self.period.kind.is_flow:
            raise ValueError("a cash flow statement covers a span, not an instant")

        components = [self.investing_cash_flow, self.financing_cash_flow]
        if self.net_change_in_cash is not None and all(c is not None for c in components):
            total = self.operating_cash_flow
            for component in components:
                assert component is not None  # narrowed by the guard above
                total = total + component
            if not _within_tolerance(
                total, self.net_change_in_cash, relative_to=self.net_change_in_cash
            ):
                raise ValueError(
                    "cash flow statement does not reconcile: operating, investing, "
                    f"and financing sum to {total}, reported change is {self.net_change_in_cash}"
                )
        return self


class FinancialStatements(FrozenModel):
    """One period's statements, as extracted from a single filing.

    Grouped because the analysis layer needs figures from more than one — return
    on equity takes net income from the income statement and equity from the
    balance sheet — and pairing them at the point of extraction keeps a caller
    from silently combining periods.
    """

    entity_id: str = Field(min_length=1)
    period: FiscalPeriod
    currency: str = Field(default="USD", min_length=3, max_length=3)
    income_statement: IncomeStatement | None = None
    balance_sheet: BalanceSheet | None = None
    cash_flow_statement: CashFlowStatement | None = None

    @model_validator(mode="after")
    def _validate(self) -> FinancialStatements:
        present = [
            self.income_statement,
            self.balance_sheet,
            self.cash_flow_statement,
        ]
        if not any(statement is not None for statement in present):
            raise ValueError("financial statements must contain at least one statement")

        for statement in present:
            if statement is not None and statement.currency != self.currency:
                raise ValueError("every statement in a set must share the set's reporting currency")

        # The income and cash flow statements must describe the same span; the
        # balance sheet is the instant at its end.
        for statement in (self.income_statement, self.cash_flow_statement):
            if statement is not None and statement.period.end_date != self.period.end_date:
                raise ValueError("income and cash flow statements must cover the set's period")
        if (
            self.balance_sheet is not None
            and self.balance_sheet.period.end_date != self.period.end_date
        ):
            raise ValueError("the balance sheet must be as of the end of the set's period")
        return self

    @property
    def fiscal_year(self) -> int:
        return self.period.fiscal_year

    @property
    def source_ids(self) -> list[str]:
        """Every document these figures were read from."""
        seen: dict[str, None] = {}
        for statement in (self.income_statement, self.balance_sheet, self.cash_flow_statement):
            if statement is None:
                continue
            for source in statement.sources:
                seen.setdefault(source.source_id, None)
        return list(seen)

    def all_sources(self) -> list[SourceReference]:
        """Deduplicated citations across the set."""
        seen: dict[str, SourceReference] = {}
        for statement in (self.income_statement, self.balance_sheet, self.cash_flow_statement):
            if statement is None:
                continue
            for source in statement.sources:
                seen.setdefault(source.source_id, source)
        return list(seen.values())


__all__ = [
    "IDENTITY_TOLERANCE",
    "BalanceSheet",
    "CashFlowStatement",
    "FinancialStatements",
    "IncomeStatement",
    "StatementBase",
]
