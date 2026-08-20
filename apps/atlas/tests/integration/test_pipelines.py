"""Filing in, verified research note out — against a real database.

This is the test that exercises the ordering the whole product rests on: figures
are computed and stored before a model is asked what they mean, and the prose
that comes back is checked against those stored figures. The AI provider is
scripted (a real one would make the assertions non-deterministic), but
everything the figures pass through — extraction, validation, NUMERIC storage,
re-reading, analysis, grounding — is the real thing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from atlas_fixtures import extraction_payload, filing, make_json_response, make_router
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from atlas.events import (
    ANALYSIS_COMPLETED,
    FILING_DUPLICATE,
    FILING_INGESTED,
    REPORT_PUBLISHED,
    REPORT_WITHHELD,
    STATEMENTS_EXTRACTED,
    STATEMENTS_REJECTED,
    VALUATION_PRODUCED,
)
from atlas.filings.extraction import StatementExtractor
from atlas.filings.pipeline import FilingPipeline
from atlas.research.pipeline import ResearchPipeline, ValuationInputs
from atlas.research.reports import ReportService
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
from fie_common.errors import NotFoundError
from fie_events.bus import InMemoryEventBus
from fie_finance.money import Rate
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

VALUATION = ValuationInputs(
    discount_rate=Rate.of(Decimal("0.10")),
    terminal_growth=Rate.of(Decimal("0.025")),
    forecast_growth=Rate.of(Decimal("0.05")),
    projection_years=5,
)


@pytest.fixture(scope="module", autouse=True)
def _requires_postgres() -> None:
    require_service(postgres_endpoint())


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    from fie_database.config import PostgresSettings
    from fie_database.postgres import PostgresDatabase

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


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus()


def ingestion_pipeline(
    session: AsyncSession, responses: list, bus: InMemoryEventBus | None = None
) -> FilingPipeline:
    return FilingPipeline(
        filings=FilingRepository(session),
        statements=StatementRepository(session),
        extractor=StatementExtractor(make_router(responses)),
        bus=bus,
    )


def research_pipeline(
    session: AsyncSession, responses: list, bus: InMemoryEventBus | None = None
) -> ResearchPipeline:
    return ResearchPipeline(
        statements=StatementRepository(session),
        reports=ReportRepository(session),
        service=ReportService(make_router(responses)),
        bus=bus,
    )


def draft(summary: str, **overrides: object) -> dict[str, object]:
    return {
        "summary": summary,
        "sections": overrides.get("sections", []),
        "cited_source_ids": overrides.get("cited_source_ids", []),
        "open_questions": overrides.get("open_questions", []),
    }


class TestFilingIngestion:
    async def test_a_filing_becomes_stored_validated_statements(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        pipeline = ingestion_pipeline(session, [make_json_response(extraction_payload())], bus)

        result = await pipeline.ingest(filing())
        await session.flush()

        assert result.newly_ingested
        assert result.has_statements
        assert result.has_income_statement and result.has_balance_sheet
        assert result.rejected == []

        stored = await StatementRepository(session).latest("ent-acme")
        assert stored is not None
        assert stored.income_statement is not None
        assert stored.income_statement.revenue.amount == Decimal("412600000")

    async def test_a_reporting_scale_is_applied_by_atlas(self, session: AsyncSession) -> None:
        """ "In millions" is the most consequential line in a filing.

        The model reports the printed figures and states the scale; Atlas
        multiplies. A model doing both can be wrong by six orders of magnitude
        with nothing looking unusual.
        """
        in_millions = extraction_payload(
            scale="millions",
            income_statement={
                "revenue": "412.6",
                "cost_of_revenue": "247.56",
                "gross_profit": "165.04",
                "net_income": "96.5484",
                "pretax_income": "119.78",
                "income_tax_expense": "23.2316",
            },
            balance_sheet=None,
            cash_flow_statement=None,
        )
        pipeline = ingestion_pipeline(session, [make_json_response(in_millions)])

        await pipeline.ingest(filing())
        await session.flush()

        stored = await StatementRepository(session).latest("ent-acme")
        assert stored is not None
        assert stored.income_statement is not None
        assert stored.income_statement.revenue.amount == Decimal("412600000.000")

    async def test_a_balance_sheet_that_does_not_balance_is_refused(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        """The extraction check that matters, exercised through the real pipeline."""
        broken = extraction_payload(
            balance_sheet={
                "total_assets": "800000000",
                "total_liabilities": "480000000",
                "shareholders_equity": "200000000",
            }
        )
        pipeline = ingestion_pipeline(session, [make_json_response(broken)], bus)

        result = await pipeline.ingest(filing())
        await session.flush()

        assert result.has_statements  # income and cash flow survived
        assert any("balance sheet rejected" in reason for reason in result.rejected)

        stored = await StatementRepository(session).latest("ent-acme")
        assert stored is not None
        assert stored.balance_sheet is None

    async def test_the_filing_survives_a_failed_extraction(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        """Losing a citable document because a table could not be read is worse."""
        nothing = extraction_payload(
            income_statement=None, balance_sheet=None, cash_flow_statement=None
        )
        pipeline = ingestion_pipeline(session, [make_json_response(nothing)], bus)

        result = await pipeline.ingest(filing())
        await session.flush()

        assert not result.has_statements
        assert await FilingRepository(session).get(result.filing_id) is not None
        assert bus.counts_by_type[STATEMENTS_REJECTED] == 1

    async def test_a_redelivered_filing_is_not_re_extracted(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        """The scripted provider has one response; a second call would exhaust it."""
        pipeline = ingestion_pipeline(session, [make_json_response(extraction_payload())], bus)
        await pipeline.ingest(filing())
        await session.flush()

        again = await pipeline.ingest(filing(filing_id="fil-redelivered"))

        assert not again.newly_ingested
        assert bus.counts_by_type[FILING_DUPLICATE] == 1

    async def test_ingestion_announces_what_happened(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        pipeline = ingestion_pipeline(session, [make_json_response(extraction_payload())], bus)

        await pipeline.ingest(filing())
        await session.flush()

        counts = bus.counts_by_type
        assert counts[FILING_INGESTED] == 1
        assert counts[STATEMENTS_EXTRACTED] == 1
        payload = bus.events_of_type(STATEMENTS_EXTRACTED)[0].payload
        assert payload["entity_id"] == "ent-acme"
        assert payload["has_balance_sheet"] is True


class TestResearchPipeline:
    async def _ingest(self, session: AsyncSession) -> None:
        await ingestion_pipeline(session, [make_json_response(extraction_payload())]).ingest(
            filing()
        )
        await session.flush()

    async def test_a_faithful_note_is_published(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        await self._ingest(session)
        pipeline = research_pipeline(
            session,
            [make_json_response(draft("Net margin was 23.4% on revenue of $412.6 million."))],
            bus,
        )

        outcome = await pipeline.produce("ent-acme", company_name="Acme Robotics")
        await session.flush()

        assert outcome.report.is_publishable
        assert outcome.report.numerically_grounded
        assert bus.counts_by_type[REPORT_PUBLISHED] == 1
        assert REPORT_WITHHELD not in bus.counts_by_type

    async def test_an_invented_figure_is_caught_after_a_full_round_trip(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        """The end-to-end guarantee: grounding is checked against *stored* figures.

        The margin quoted here is plausible and is not one Atlas computed. It
        survives extraction, storage, and re-reading, and is still refused.
        """
        await self._ingest(session)
        invented = make_json_response(draft("Operating margin reached 31.2%."))
        pipeline = research_pipeline(session, [invented, invented], bus)

        outcome = await pipeline.produce("ent-acme")
        await session.flush()

        assert not outcome.report.is_publishable
        assert "31.2%" in outcome.report.unsupported_figures
        assert bus.counts_by_type[REPORT_WITHHELD] == 1

        # Stored anyway — a systematic model problem has to stay visible.
        stored = await ReportRepository(session).get(outcome.report_id)
        assert stored is not None
        assert not stored[1].is_publishable

    async def test_a_valuation_travels_as_an_estimate_with_its_assumptions(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        await self._ingest(session)
        pipeline = research_pipeline(session, [make_json_response(draft("Margins held."))], bus)

        outcome = await pipeline.produce("ent-acme", valuation_inputs=VALUATION)
        await session.flush()

        assert outcome.valuation is not None
        assert "discount_rate" in outcome.valuation.assumptions
        payload = bus.events_of_type(VALUATION_PRODUCED)[0].payload
        assert payload["assumptions"]["discount_rate"] == "0.10"
        assert payload["currency"] == "USD"

    async def test_the_analysis_event_names_what_was_not_disclosed(
        self, session: AsyncSession, bus: InMemoryEventBus
    ) -> None:
        """A thin filing and a complete one must not look identical downstream."""
        sparse = extraction_payload(balance_sheet=None, cash_flow_statement=None)
        await ingestion_pipeline(session, [make_json_response(sparse)]).ingest(filing())
        await session.flush()

        pipeline = research_pipeline(session, [make_json_response(draft("Revenue grew."))], bus)
        await pipeline.produce("ent-acme")
        await session.flush()

        payload = bus.events_of_type(ANALYSIS_COMPLETED)[0].payload
        assert "current_ratio" in payload["unavailable_names"]
        assert "net_margin" in payload["metric_names"]

    async def test_a_company_with_no_filings_is_a_404_not_an_empty_report(
        self, session: AsyncSession
    ) -> None:
        pipeline = research_pipeline(session, [])

        with pytest.raises(NotFoundError):
            await pipeline.produce("ent-unknown")

    async def test_a_report_is_written_from_the_period_asked_for(
        self, session: AsyncSession
    ) -> None:
        for year in (2024, 2025):
            await ingestion_pipeline(session, [make_json_response(extraction_payload())]).ingest(
                filing(year=year, content=f"Annual report for {year}. Revenue 412.6")
            )
        await session.flush()

        pipeline = research_pipeline(session, [make_json_response(draft("Steady."))])
        outcome = await pipeline.produce("ent-acme", fiscal_year=2024)

        assert outcome.report.period_label == "FY2024"
