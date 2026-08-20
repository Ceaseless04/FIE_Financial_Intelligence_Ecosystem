"""Persistence for filings, statements, and reports.

The mapping in this module is deliberately explicit rather than automatic. A
generic object mapper would happily round-trip a monetary figure through a float
somewhere in the middle; writing each conversion out means every place a
``Decimal`` becomes a column and back is visible and testable.

Reading is where the care matters most. A row's ``NUMERIC`` comes back as a
:class:`~decimal.Decimal`, and it is handed straight to
:class:`~fie_finance.money.Money`, which would reject anything else. So the
statements rebuilt here are re-validated against the accounting identities on
construction — a row that was corrupted in the database fails to load rather
than flowing into an analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atlas.analysis.results import Metric
from atlas.domain.statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from atlas.filings.models import Filing, FilingType
from atlas.research.reports import ReportSection, ResearchReport
from atlas.storage.models import (
    BalanceSheetRow,
    CashFlowStatementRow,
    FilingRow,
    IncomeStatementRow,
    ResearchReportRow,
    StatementSetRow,
)
from fie_finance.money import Money, Shares
from fie_finance.periods import FiscalPeriod, PeriodKind, period_key
from fie_observability.logging import get_logger
from fie_schemas.provenance import Provenance

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def _amount(money: Money | None) -> Decimal | None:
    return money.amount if money is not None else None


def _money(value: Decimal | None, currency: str) -> Money | None:
    """Rebuild a monetary value from a NUMERIC column.

    ``value`` is a Decimal or None — never a float, because the column is
    NUMERIC. Money would refuse a float anyway, which is the point: a schema
    change to DOUBLE PRECISION would fail loudly here rather than quietly
    degrading every figure in the system.
    """
    return Money(amount=value, currency=currency) if value is not None else None


def _period_values(period: FiscalPeriod) -> dict[str, Any]:
    return {
        "period_kind": str(period.kind),
        "fiscal_year": period.fiscal_year,
        "fiscal_quarter": period.fiscal_quarter,
        "period_start": period.start_date,
        "period_end": period.end_date,
    }


def _period(
    kind: str, fiscal_year: int, fiscal_quarter: int | None, start: date | None, end: date
) -> FiscalPeriod:
    return FiscalPeriod(
        kind=PeriodKind(kind),
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        start_date=start,
        end_date=end,
    )


def _period_of(row: Any) -> FiscalPeriod:
    return _period(
        row.period_kind, row.fiscal_year, row.fiscal_quarter, row.period_start, row.period_end
    )


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------


@dataclass
class FilingRepository:
    """Stores source documents so extracted figures stay citable."""

    session: AsyncSession

    async def upsert(self, filing: Filing) -> tuple[str, bool]:
        """Store a filing, returning its id and whether it was newly created.

        Deduplicates on content hash. Feeds redeliver the same 10-K routinely,
        and re-extracting one costs a model call for a result already held.
        """
        existing = await self.session.scalar(
            select(FilingRow).where(FilingRow.content_hash == filing.content_hash)
        )
        if existing is not None:
            return existing.id, False

        row = FilingRow(
            id=filing.id,
            entity_id=filing.entity_id,
            type=str(filing.type),
            title=filing.title,
            content=filing.content,
            content_hash=filing.content_hash,
            accession_number=filing.accession_number,
            filed_at=filing.filed_at,
            uri=filing.uri,
            ingested_at=filing.ingested_at,
            filing_metadata=dict(filing.metadata),
            **_period_values(filing.period),
        )
        self.session.add(row)
        await self.session.flush()
        return row.id, True

    async def get(self, filing_id: str) -> Filing | None:
        row = await self.session.get(FilingRow, filing_id)
        return self._to_filing(row) if row is not None else None

    async def list_for_entity(self, entity_id: str, *, limit: int = 20) -> list[Filing]:
        rows = await self.session.scalars(
            select(FilingRow)
            .where(FilingRow.entity_id == entity_id)
            .order_by(FilingRow.period_end.desc())
            .limit(limit)
        )
        return [self._to_filing(row) for row in rows]

    async def delete(self, filing_id: str) -> None:
        await self.session.execute(delete(FilingRow).where(FilingRow.id == filing_id))

    @staticmethod
    def _to_filing(row: FilingRow) -> Filing:
        return Filing(
            id=row.id,
            entity_id=row.entity_id,
            type=FilingType(row.type),
            period=_period_of(row),
            title=row.title,
            content=row.content,
            accession_number=row.accession_number,
            filed_at=row.filed_at,
            uri=row.uri,
            ingested_at=row.ingested_at,
            metadata=dict(row.filing_metadata or {}),
        )


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass
class StatementRepository:
    """Stores validated statement sets and reads them back as domain models."""

    session: AsyncSession

    async def save(
        self,
        statements: FinancialStatements,
        *,
        filing_id: str,
        rejected: Sequence[str] = (),
    ) -> str:
        """Persist a statement set, replacing any existing set for the period.

        Replacement rather than a second row: a company has one set of figures
        for a fiscal quarter, and letting two versions coexist means every
        downstream query has to decide which is real. A re-extraction — a better
        prompt, a corrected filing — supersedes what came before.
        """
        period = statements.period
        existing = await self.session.scalar(
            select(StatementSetRow).where(
                StatementSetRow.entity_id == statements.entity_id,
                StatementSetRow.period_key == period.key,
            )
        )
        if existing is not None:
            # The children cascade, so removing the set removes its statements
            # rather than orphaning three rows nobody can reach.
            await self.session.delete(existing)
            await self.session.flush()

        row = StatementSetRow(
            entity_id=statements.entity_id,
            period_key=period.key,
            filing_id=filing_id,
            currency=statements.currency,
            rejected=list(rejected),
            **_period_values(period),
        )
        if statements.income_statement is not None:
            row.income_statement = self._income_row(statements.income_statement)
        if statements.balance_sheet is not None:
            row.balance_sheet = self._balance_row(statements.balance_sheet)
        if statements.cash_flow_statement is not None:
            row.cash_flow_statement = self._cash_flow_row(statements.cash_flow_statement)

        self.session.add(row)
        await self.session.flush()
        logger.info(
            "statements_stored",
            entity_id=statements.entity_id,
            period=period.label,
            statement_set_id=row.id,
        )
        return row.id

    async def get_for_period(
        self, entity_id: str, *, fiscal_year: int, kind: PeriodKind, quarter: int | None = None
    ) -> FinancialStatements | None:
        row = await self.session.scalar(
            self._with_statements().where(
                StatementSetRow.entity_id == entity_id,
                StatementSetRow.period_key == period_key(kind, fiscal_year, quarter),
            )
        )
        return self._to_statements(row) if row is not None else None

    async def latest(self, entity_id: str) -> FinancialStatements | None:
        """The most recent period on file for a company."""
        row = await self.session.scalar(
            self._with_statements()
            .where(StatementSetRow.entity_id == entity_id)
            .order_by(StatementSetRow.period_end.desc())
            .limit(1)
        )
        return self._to_statements(row) if row is not None else None

    async def history(self, entity_id: str, *, limit: int = 8) -> list[FinancialStatements]:
        """Periods newest first, for growth and trend analysis."""
        rows = await self.session.scalars(
            self._with_statements()
            .where(StatementSetRow.entity_id == entity_id)
            .order_by(StatementSetRow.period_end.desc())
            .limit(limit)
        )
        return [self._to_statements(row) for row in rows]

    @staticmethod
    def _with_statements() -> Any:
        # Eager-loaded: the caller always wants the statements, and lazy loading
        # them would emit three queries per set inside an async session that
        # cannot lazy-load at all.
        return select(StatementSetRow).options(
            selectinload(StatementSetRow.income_statement),
            selectinload(StatementSetRow.balance_sheet),
            selectinload(StatementSetRow.cash_flow_statement),
        )

    # -- writing -------------------------------------------------------------

    @staticmethod
    def _income_row(statement: IncomeStatement) -> IncomeStatementRow:
        return IncomeStatementRow(
            currency=statement.currency,
            provenance=statement.provenance.model_dump(mode="json"),
            revenue=statement.revenue.amount,
            net_income=statement.net_income.amount,
            cost_of_revenue=_amount(statement.cost_of_revenue),
            gross_profit=_amount(statement.gross_profit),
            operating_expenses=_amount(statement.operating_expenses),
            operating_income=_amount(statement.operating_income),
            interest_expense=_amount(statement.interest_expense),
            pretax_income=_amount(statement.pretax_income),
            income_tax_expense=_amount(statement.income_tax_expense),
            diluted_shares=(
                statement.diluted_shares.count if statement.diluted_shares is not None else None
            ),
            **_period_values(statement.period),
        )

    @staticmethod
    def _balance_row(statement: BalanceSheet) -> BalanceSheetRow:
        return BalanceSheetRow(
            currency=statement.currency,
            provenance=statement.provenance.model_dump(mode="json"),
            total_assets=statement.total_assets.amount,
            total_liabilities=statement.total_liabilities.amount,
            shareholders_equity=statement.shareholders_equity.amount,
            current_assets=_amount(statement.current_assets),
            cash_and_equivalents=_amount(statement.cash_and_equivalents),
            inventory=_amount(statement.inventory),
            current_liabilities=_amount(statement.current_liabilities),
            total_debt=_amount(statement.total_debt),
            **_period_values(statement.period),
        )

    @staticmethod
    def _cash_flow_row(statement: CashFlowStatement) -> CashFlowStatementRow:
        return CashFlowStatementRow(
            currency=statement.currency,
            provenance=statement.provenance.model_dump(mode="json"),
            operating_cash_flow=statement.operating_cash_flow.amount,
            capital_expenditures=_amount(statement.capital_expenditures),
            investing_cash_flow=_amount(statement.investing_cash_flow),
            financing_cash_flow=_amount(statement.financing_cash_flow),
            net_change_in_cash=_amount(statement.net_change_in_cash),
            **_period_values(statement.period),
        )

    # -- reading -------------------------------------------------------------

    @classmethod
    def _to_statements(cls, row: StatementSetRow) -> FinancialStatements:
        return FinancialStatements(
            entity_id=row.entity_id,
            period=_period_of(row),
            currency=row.currency,
            income_statement=(
                cls._to_income(row.income_statement) if row.income_statement else None
            ),
            balance_sheet=(cls._to_balance(row.balance_sheet) if row.balance_sheet else None),
            cash_flow_statement=(
                cls._to_cash_flow(row.cash_flow_statement) if row.cash_flow_statement else None
            ),
        )

    @staticmethod
    def _to_income(row: IncomeStatementRow) -> IncomeStatement:
        currency = row.currency
        return IncomeStatement(
            period=_period_of(row),
            currency=currency,
            provenance=Provenance.model_validate(row.provenance),
            revenue=Money(amount=row.revenue, currency=currency),
            net_income=Money(amount=row.net_income, currency=currency),
            cost_of_revenue=_money(row.cost_of_revenue, currency),
            gross_profit=_money(row.gross_profit, currency),
            operating_expenses=_money(row.operating_expenses, currency),
            operating_income=_money(row.operating_income, currency),
            interest_expense=_money(row.interest_expense, currency),
            pretax_income=_money(row.pretax_income, currency),
            income_tax_expense=_money(row.income_tax_expense, currency),
            diluted_shares=(
                Shares.of(row.diluted_shares) if row.diluted_shares is not None else None
            ),
        )

    @staticmethod
    def _to_balance(row: BalanceSheetRow) -> BalanceSheet:
        currency = row.currency
        return BalanceSheet(
            period=_period_of(row),
            currency=currency,
            provenance=Provenance.model_validate(row.provenance),
            total_assets=Money(amount=row.total_assets, currency=currency),
            total_liabilities=Money(amount=row.total_liabilities, currency=currency),
            shareholders_equity=Money(amount=row.shareholders_equity, currency=currency),
            current_assets=_money(row.current_assets, currency),
            cash_and_equivalents=_money(row.cash_and_equivalents, currency),
            inventory=_money(row.inventory, currency),
            current_liabilities=_money(row.current_liabilities, currency),
            total_debt=_money(row.total_debt, currency),
        )

    @staticmethod
    def _to_cash_flow(row: CashFlowStatementRow) -> CashFlowStatement:
        currency = row.currency
        return CashFlowStatement(
            period=_period_of(row),
            currency=currency,
            provenance=Provenance.model_validate(row.provenance),
            operating_cash_flow=Money(amount=row.operating_cash_flow, currency=currency),
            capital_expenditures=_money(row.capital_expenditures, currency),
            investing_cash_flow=_money(row.investing_cash_flow, currency),
            financing_cash_flow=_money(row.financing_cash_flow, currency),
            net_change_in_cash=_money(row.net_change_in_cash, currency),
        )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class ReportRepository:
    """Stores research reports, grounded or not."""

    session: AsyncSession

    async def save(self, report: ResearchReport) -> str:
        row = ResearchReportRow(
            entity_id=report.entity_id,
            period_label=report.period_label,
            summary=report.summary,
            sections=[section.model_dump(mode="json") for section in report.sections],
            open_questions=list(report.open_questions),
            metrics=[metric.model_dump(mode="json") for metric in report.metrics],
            cited_source_ids=list(report.cited_source_ids),
            provenance=report.provenance.model_dump(mode="json"),
            numerically_grounded=report.numerically_grounded,
            unsupported_figures=list(report.unsupported_figures),
            citations_grounded=report.citations_grounded,
            regeneration_count=report.regeneration_count,
        )
        self.session.add(row)
        await self.session.flush()
        logger.info(
            "report_stored",
            report_id=row.id,
            entity_id=report.entity_id,
            publishable=report.is_publishable,
        )
        return row.id

    async def get(self, report_id: str) -> tuple[str, ResearchReport] | None:
        row = await self.session.get(ResearchReportRow, report_id)
        return (row.id, self._to_report(row)) if row is not None else None

    async def list_for_entity(
        self, entity_id: str, *, limit: int = 20, publishable_only: bool = False
    ) -> list[tuple[str, ResearchReport]]:
        statement = (
            select(ResearchReportRow)
            .where(ResearchReportRow.entity_id == entity_id)
            .order_by(ResearchReportRow.created_at.desc())
            .limit(limit)
        )
        if publishable_only:
            statement = statement.where(
                ResearchReportRow.numerically_grounded.is_(True),
                ResearchReportRow.citations_grounded.is_(True),
            )
        rows = await self.session.scalars(statement)
        return [(row.id, self._to_report(row)) for row in rows]

    @staticmethod
    def _to_report(row: ResearchReportRow) -> ResearchReport:
        return ResearchReport(
            entity_id=row.entity_id,
            period_label=row.period_label,
            summary=row.summary,
            sections=[ReportSection.model_validate(section) for section in row.sections],
            open_questions=list(row.open_questions),
            # Metric values were written as JSON strings and are parsed back
            # into Decimal by the model's own validator, which rejects floats.
            metrics=[Metric.model_validate(metric) for metric in row.metrics],
            cited_source_ids=list(row.cited_source_ids),
            provenance=Provenance.model_validate(row.provenance),
            numerically_grounded=row.numerically_grounded,
            unsupported_figures=list(row.unsupported_figures),
            citations_grounded=row.citations_grounded,
            regeneration_count=row.regeneration_count,
        )


__all__ = ["FilingRepository", "ReportRepository", "StatementRepository"]
