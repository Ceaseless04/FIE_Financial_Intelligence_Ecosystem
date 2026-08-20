"""Persistence against a real PostgreSQL.

The property under test is the one an in-memory stand-in cannot check: that a
figure survives the round trip through the database *exactly*. NUMERIC columns,
Decimal round-tripping, JSONB serialisation of Decimal, and the unique
constraint's treatment of NULL are all database semantics. Asserting them
against a fake would prove only that the fake agrees with itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal

import pytest
from atlas_fixtures import (
    annual_period,
    balance_sheet,
    cash_flow_statement,
    filing,
    income_statement,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from atlas.analysis.ratios import analyze
from atlas.domain.statements import FinancialStatements
from atlas.research.reports import ReportSection, ResearchReport
from atlas.storage.models import (
    SCHEMA,
    BalanceSheetRow,
    CashFlowStatementRow,
    FilingRow,
    IncomeStatementRow,
    ResearchReportRow,
    StatementSetRow,
)
from atlas.storage.repository import FilingRepository, ReportRepository, StatementRepository
from fie_database.config import PostgresSettings
from fie_database.postgres import PostgresDatabase
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod, PeriodKind
from fie_schemas.provenance import Provenance
from fie_testing import postgres_endpoint, require_service

pytestmark = pytest.mark.integration

TABLES = [
    FilingRow.__table__,
    StatementSetRow.__table__,
    IncomeStatementRow.__table__,
    BalanceSheetRow.__table__,
    CashFlowStatementRow.__table__,
    ResearchReportRow.__table__,
]


@pytest.fixture(scope="module", autouse=True)
def _requires_postgres() -> None:
    require_service(postgres_endpoint())


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A schema-created session, rolled back and dropped after each test."""
    database = PostgresDatabase(PostgresSettings())
    async with database.engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        await connection.run_sync(
            lambda sync_connection: FilingRow.metadata.create_all(sync_connection, tables=TABLES)
        )

    factory = database.session_factory
    async with factory() as active:
        try:
            yield active
        finally:
            await active.rollback()

    async with database.engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await database.aclose()


def statements_for(year: int = 2025, **overrides: object) -> FinancialStatements:
    return FinancialStatements(
        entity_id="ent-acme",
        period=annual_period(year),
        income_statement=overrides.get("income_statement", income_statement(year=year)),
        balance_sheet=overrides.get("balance_sheet", balance_sheet(year=year)),
        cash_flow_statement=overrides.get("cash_flow_statement", cash_flow_statement(year=year)),
    )


async def store_filing_and_statements(
    session: AsyncSession, *, year: int = 2025
) -> tuple[str, str]:
    filing_id, _created = await FilingRepository(session).upsert(filing(year=year))
    statement_set_id = await StatementRepository(session).save(
        statements_for(year), filing_id=filing_id
    )
    await session.flush()
    return filing_id, statement_set_id


class TestFilingPersistence:
    async def test_a_filing_round_trips(self, session: AsyncSession) -> None:
        repository = FilingRepository(session)
        original = filing()

        filing_id, created = await repository.upsert(original)
        await session.flush()
        stored = await repository.get(filing_id)

        assert created
        assert stored is not None
        assert stored.content == original.content
        assert stored.content_hash == original.content_hash
        assert stored.period.label == "FY2025"
        assert stored.filed_at == date(2026, 2, 14)

    async def test_a_redelivered_filing_is_recognised_by_content(
        self, session: AsyncSession
    ) -> None:
        """Feeds redeliver the same 10-K, and re-extracting one costs a model call."""
        repository = FilingRepository(session)
        first_id, first_created = await repository.upsert(filing())
        await session.flush()

        # Same content, different id — as a feed would redeliver it.
        second_id, second_created = await repository.upsert(filing(filing_id="fil-different"))

        assert first_created
        assert not second_created
        assert second_id == first_id

    async def test_deleting_a_filing_removes_its_statements(self, session: AsyncSession) -> None:
        """The cascade is declared in the database, so only the database can prove it."""
        filing_id, _set_id = await store_filing_and_statements(session)

        await FilingRepository(session).delete(filing_id)
        await session.flush()

        assert await StatementRepository(session).latest("ent-acme") is None


class TestStatementPersistence:
    async def test_every_figure_survives_the_round_trip_exactly(
        self, session: AsyncSession
    ) -> None:
        """The point of NUMERIC. A float column would lose the last digits here."""
        await store_filing_and_statements(session)

        stored = await StatementRepository(session).latest("ent-acme")

        assert stored is not None
        assert stored.income_statement is not None
        assert stored.income_statement.revenue == Money.of("412600000")
        assert stored.income_statement.net_income.amount == Decimal("96548400")
        assert stored.balance_sheet is not None
        assert stored.balance_sheet.total_assets.amount == Decimal("800000000")
        assert stored.cash_flow_statement is not None
        assert stored.cash_flow_statement.capital_expenditures == Money.of("-45000000")

    async def test_a_fractional_amount_keeps_its_precision(self, session: AsyncSession) -> None:
        """A figure a float would round is the one worth asserting on."""
        precise = income_statement(revenue="412600000.1234", cost_of_revenue="247560000.0001")
        filing_id, _created = await FilingRepository(session).upsert(filing())
        await StatementRepository(session).save(
            statements_for(income_statement=precise), filing_id=filing_id
        )
        await session.flush()

        stored = await StatementRepository(session).latest("ent-acme")

        assert stored is not None
        assert stored.income_statement is not None
        assert stored.income_statement.revenue.amount == Decimal("412600000.1234")

    async def test_an_undisclosed_line_item_stays_undisclosed(self, session: AsyncSession) -> None:
        """Never zero. A zero reads as a measurement; None reads as "not disclosed"."""
        sparse = income_statement().model_copy(update={"inventory": None})
        filing_id, _created = await FilingRepository(session).upsert(filing())
        await StatementRepository(session).save(
            statements_for(
                income_statement=sparse,
                balance_sheet=balance_sheet().model_copy(update={"inventory": None}),
            ),
            filing_id=filing_id,
        )
        await session.flush()

        stored = await StatementRepository(session).latest("ent-acme")

        assert stored is not None
        assert stored.balance_sheet is not None
        assert stored.balance_sheet.inventory is None

    async def test_a_balance_sheet_keeps_its_instant_period(self, session: AsyncSession) -> None:
        """It is stored beside a span period and must not acquire one."""
        await store_filing_and_statements(session)

        stored = await StatementRepository(session).latest("ent-acme")

        assert stored is not None
        assert stored.balance_sheet is not None
        assert stored.balance_sheet.period.kind is PeriodKind.INSTANT
        assert stored.income_statement is not None
        assert stored.income_statement.period.kind is PeriodKind.ANNUAL

    async def test_provenance_survives_so_a_figure_stays_citable(
        self, session: AsyncSession
    ) -> None:
        await store_filing_and_statements(session)

        stored = await StatementRepository(session).latest("ent-acme")

        assert stored is not None
        assert stored.source_ids == ["fil-acme-2025"]

    async def test_re_extracting_a_period_replaces_it(self, session: AsyncSession) -> None:
        """Two rival versions of one quarter would make every later query ambiguous."""
        repository = StatementRepository(session)
        filing_id, _created = await FilingRepository(session).upsert(filing())
        await repository.save(statements_for(), filing_id=filing_id)
        await session.flush()

        revised = statements_for(income_statement=income_statement(revenue="500000000"))
        await repository.save(revised, filing_id=filing_id)
        await session.flush()

        history = await repository.history("ent-acme")
        assert len(history) == 1
        assert history[0].income_statement is not None
        assert history[0].income_statement.revenue.amount == Decimal("500000000")

    async def test_two_annual_sets_for_one_company_are_refused(self, session: AsyncSession) -> None:
        """The NULL-in-a-unique-constraint trap, asserted where it actually bites.

        `fiscal_quarter` is NULL for an annual period, and PostgreSQL treats
        every NULL as distinct — so a constraint over the period columns would
        have permitted this silently.
        """
        filing_id, _created = await FilingRepository(session).upsert(filing())
        session.add(
            StatementSetRow(
                entity_id="ent-acme",
                period_key="annual:2025:0",
                filing_id=filing_id,
                currency="USD",
                rejected=[],
                period_kind="annual",
                fiscal_year=2025,
                fiscal_quarter=None,
                period_start=date(2025, 1, 1),
                period_end=date(2025, 12, 31),
            )
        )
        session.add(
            StatementSetRow(
                entity_id="ent-acme",
                period_key="annual:2025:0",
                filing_id=filing_id,
                currency="USD",
                rejected=[],
                period_kind="annual",
                fiscal_year=2025,
                fiscal_quarter=None,
                period_start=date(2025, 1, 1),
                period_end=date(2025, 12, 31),
            )
        )

        with pytest.raises(Exception, match="uq_statement_sets_entity_period"):
            await session.flush()

    async def test_a_quarter_and_a_year_coexist(self, session: AsyncSession) -> None:
        """The constraint must separate periods, not collapse them."""
        repository = StatementRepository(session)
        filing_id, _created = await FilingRepository(session).upsert(filing())

        await repository.save(statements_for(), filing_id=filing_id)
        quarter = FiscalPeriod.quarterly(2025, 3, date(2025, 9, 30), date(2025, 7, 1))
        await repository.save(
            FinancialStatements(
                entity_id="ent-acme",
                period=quarter,
                income_statement=income_statement().model_copy(update={"period": quarter}),
            ),
            filing_id=filing_id,
        )
        await session.flush()

        assert len(await repository.history("ent-acme")) == 2

    async def test_history_is_newest_first(self, session: AsyncSession) -> None:
        repository = StatementRepository(session)
        for year in (2023, 2025, 2024):
            filing_id, _created = await FilingRepository(session).upsert(filing(year=year))
            await repository.save(statements_for(year), filing_id=filing_id)
        await session.flush()

        history = await repository.history("ent-acme")

        assert [statements.fiscal_year for statements in history] == [2025, 2024, 2023]

    async def test_a_stored_set_is_re_validated_on_read(self, session: AsyncSession) -> None:
        """A row corrupted in the database must fail to load, not flow into analysis."""
        _filing_id, set_id = await store_filing_and_statements(session)
        # The schema name is a module constant, not input; the value is bound.
        await session.execute(
            text(
                f"UPDATE {SCHEMA}.balance_sheets SET total_assets = 1 "  # noqa: S608
                "WHERE statement_set_id = :set_id"
            ),
            {"set_id": set_id},
        )
        session.expire_all()

        with pytest.raises(Exception, match="does not balance"):
            await StatementRepository(session).latest("ent-acme")

    async def test_stored_statements_analyse_identically(self, session: AsyncSession) -> None:
        """The end-to-end guarantee: persistence changes no computed figure."""
        original = statements_for()
        await store_filing_and_statements(session)

        stored = await StatementRepository(session).latest("ent-acme")
        assert stored is not None

        before = {metric.name: metric.value for metric in analyze(original).available}
        after = {metric.name: metric.value for metric in analyze(stored).available}
        assert before == after


class TestReportPersistence:
    def _report(self, *, grounded: bool = True) -> ResearchReport:
        analysis = analyze(statements_for())
        return ResearchReport(
            entity_id="ent-acme",
            period_label="FY2025",
            summary="Net margin was 23.4% on revenue of $412.6 million.",
            sections=[ReportSection(heading="Margins", body="Gross margin held.")],
            open_questions=["What drove the tax rate?"],
            metrics=analysis.available,
            cited_source_ids=["fil-acme-2025"],
            provenance=Provenance.generated("fake-model-1"),
            numerically_grounded=grounded,
            unsupported_figures=[] if grounded else ["31.2%"],
            citations_grounded=True,
            regeneration_count=0 if grounded else 1,
        )

    async def test_a_report_round_trips_with_its_metrics(self, session: AsyncSession) -> None:
        """Metric values are JSONB, and JSON has only floats — so this is the check."""
        repository = ReportRepository(session)
        original = self._report()

        report_id = await repository.save(original)
        await session.flush()
        session.expire_all()
        stored = await repository.get(report_id)

        assert stored is not None
        _id, report = stored
        assert report.summary == original.summary
        assert [metric.value for metric in report.metrics] == [
            metric.value for metric in original.metrics
        ]
        assert all(isinstance(metric.value, Decimal) for metric in report.metrics)

    async def test_a_computed_metric_is_never_attributed_to_the_model(
        self, session: AsyncSession
    ) -> None:
        """The architectural rule, checked after a database round trip."""
        repository = ReportRepository(session)
        report_id = await repository.save(self._report())
        await session.flush()
        session.expire_all()

        stored = await repository.get(report_id)

        assert stored is not None
        assert all(metric.provenance.model is None for metric in stored[1].metrics)

    async def test_an_ungrounded_report_is_stored_not_discarded(
        self, session: AsyncSession
    ) -> None:
        """A model that invents figures is visible in this table or nowhere."""
        repository = ReportRepository(session)
        report_id = await repository.save(self._report(grounded=False))
        await session.flush()

        stored = await repository.get(report_id)

        assert stored is not None
        assert not stored[1].is_publishable
        assert stored[1].unsupported_figures == ["31.2%"]

    async def test_publishable_only_filters_in_the_database(self, session: AsyncSession) -> None:
        """ "Did we ever serve an ungrounded report" must be a WHERE clause."""
        repository = ReportRepository(session)
        await repository.save(self._report())
        await repository.save(self._report(grounded=False))
        await session.flush()

        every = await repository.list_for_entity("ent-acme")
        publishable = await repository.list_for_entity("ent-acme", publishable_only=True)

        assert len(every) == 2
        assert len(publishable) == 1
        assert publishable[0][1].is_publishable
