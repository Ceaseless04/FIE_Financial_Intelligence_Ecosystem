"""Persistence against a real PostgreSQL.

What an in-memory stand-in cannot check: that a figure survives the round trip
exactly, that the database refuses a duplicate line the domain also refuses, and
that a plan rebuilt from rows re-runs every invariant rather than trusting what
it was handed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from cfo_fixtures import CHART, actuals, budget, fact, tree
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from cfo_ai.analysis.compare import compare
from cfo_ai.domain.plan import Plan, PlanKind, line
from cfo_ai.research.commentary import Commentary, CommentarySection
from cfo_ai.storage.models import (
    SCHEMA,
    AccountRow,
    CommentaryRow,
    CostCentreRow,
    PlanLineRow,
    PlanRow,
)
from cfo_ai.storage.repository import ChartRepository, CommentaryRepository, PlanRepository
from fie_database.config import PostgresSettings
from fie_database.postgres import PostgresDatabase
from fie_finance.money import Money
from fie_schemas.provenance import Provenance
from fie_testing import postgres_endpoint, require_service

pytestmark = pytest.mark.integration

TABLES = [
    AccountRow.__table__,
    CostCentreRow.__table__,
    PlanRow.__table__,
    PlanLineRow.__table__,
    CommentaryRow.__table__,
]


@pytest.fixture(scope="module", autouse=True)
def _requires_postgres() -> None:
    require_service(postgres_endpoint())


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    database = PostgresDatabase(PostgresSettings())
    async with database.engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        await connection.run_sync(lambda sync: AccountRow.metadata.create_all(sync, tables=TABLES))

    factory = database.session_factory
    async with factory() as active:
        try:
            yield active
        finally:
            await active.rollback()

    async with database.engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await database.aclose()


class TestChart:
    async def test_a_chart_round_trips_with_its_types(self, session: AsyncSession) -> None:
        """The account type decides every variance direction, so it is the one
        column that cannot be lost."""
        repository = ChartRepository(session)
        await repository.replace_accounts("ent-northwind", list(CHART.values()))
        await session.flush()

        stored = await repository.accounts("ent-northwind")

        assert set(stored) == set(CHART)
        assert stored["4000"].type is CHART["4000"].type
        assert stored["6100"].aliases == CHART["6100"].aliases

    async def test_replacing_a_chart_removes_what_was_there(self, session: AsyncSession) -> None:
        """A chart is published whole; a partial update leaves an account nobody
        meant to keep deciding a direction."""
        repository = ChartRepository(session)
        await repository.replace_accounts("ent-northwind", list(CHART.values()))
        await session.flush()

        await repository.replace_accounts("ent-northwind", [CHART["4000"]])
        await session.flush()

        assert list(await repository.accounts("ent-northwind")) == ["4000"]

    async def test_two_companies_keep_separate_charts(self, session: AsyncSession) -> None:
        repository = ChartRepository(session)
        await repository.replace_accounts("ent-a", list(CHART.values()))
        await repository.replace_accounts("ent-b", [CHART["4000"]])
        await session.flush()

        assert len(await repository.accounts("ent-a")) == 5
        assert len(await repository.accounts("ent-b")) == 1

    async def test_a_hierarchy_round_trips_and_is_revalidated(self, session: AsyncSession) -> None:
        repository = ChartRepository(session)
        await repository.replace_cost_centres("ent-northwind", tree())
        await session.flush()

        stored = await repository.cost_centres("ent-northwind")

        assert stored is not None
        assert stored.ancestors_of("cc-field") == ["cc-sales", "cc-co"]

    async def test_a_company_with_no_hierarchy_returns_none(self, session: AsyncSession) -> None:
        assert await ChartRepository(session).cost_centres("ent-nobody") is None


class TestPlans:
    async def test_every_amount_survives_exactly(self, session: AsyncSession) -> None:
        """The point of NUMERIC. A float column would lose the last digits."""
        repository = PlanRepository(session)
        plan_id = await repository.save(budget())
        await session.flush()
        session.expire_all()

        stored = await repository.get(plan_id)

        assert stored is not None
        assert stored.total() == Money.of("7600000")
        assert stored.total_for_account("4000") == Money.of("4000000")

    async def test_a_fractional_amount_keeps_its_precision(self, session: AsyncSession) -> None:
        period = budget().period
        precise = Plan(
            entity_id="ent-northwind",
            kind=PlanKind.BUDGET,
            period=period,
            lines=[line("6100", "cc-sales", period, "123456.7891")],
            provenance=fact(),
        )
        repository = PlanRepository(session)
        plan_id = await repository.save(precise)
        await session.flush()
        session.expire_all()

        stored = await repository.get(plan_id)

        assert stored is not None
        assert stored.lines[0].amount.amount == Decimal("123456.7891")

    async def test_re_saving_a_version_supersedes_it(self, session: AsyncSession) -> None:
        repository = PlanRepository(session)
        await repository.save(budget())
        await session.flush()
        await repository.save(budget(revenue="5000000"))
        await session.flush()

        plans = await repository.list_for_entity("ent-northwind", kind=PlanKind.BUDGET)

        assert len(plans) == 1
        assert plans[0].total_for_account("4000") == Money.of("5000000")

    async def test_a_board_revision_does_not_overwrite_the_approved_plan(
        self, session: AsyncSession
    ) -> None:
        """Version is part of the identity, so "FY26 approved" and "FY26 board
        revision 2" coexist rather than one silently replacing the other."""
        repository = PlanRepository(session)
        approved = budget()
        revision = budget(revenue="3000000").model_copy(update={"version": "v2"})

        await repository.save(approved)
        await repository.save(revision)
        await session.flush()

        plans = await repository.list_for_entity("ent-northwind", kind=PlanKind.BUDGET)
        assert {plan.version for plan in plans} == {"v1", "v2"}

    async def test_a_budget_and_actuals_for_one_period_coexist(self, session: AsyncSession) -> None:
        repository = PlanRepository(session)
        await repository.save(budget())
        await repository.save(actuals())
        await session.flush()

        assert await repository.find("ent-northwind", kind=PlanKind.BUDGET, period=budget().period)
        assert await repository.find("ent-northwind", kind=PlanKind.ACTUAL, period=actuals().period)

    async def test_the_database_refuses_a_duplicate_line(self, session: AsyncSession) -> None:
        """The domain refuses it at construction; the database refuses it again,
        because a second row for one key double-counts into every total."""
        plan = budget()
        plan_id = await PlanRepository(session).save(plan)
        await session.flush()

        original = plan.lines[0]
        session.add(
            PlanLineRow(
                plan_id=plan_id,
                account_code=original.account_code,
                cost_centre_id=original.cost_centre_id,
                period_key=original.period.key,
                amount=Decimal("1"),
                currency="USD",
                period_kind=str(original.period.kind),
                fiscal_year=original.period.fiscal_year,
                fiscal_quarter=original.period.fiscal_quarter,
                period_start=original.period.start_date,
                period_end=original.period.end_date,
            )
        )

        with pytest.raises(Exception, match="uq_plan_lines_key"):
            await session.flush()

    async def test_deleting_a_plan_removes_its_lines(self, session: AsyncSession) -> None:
        repository = PlanRepository(session)
        plan_id = await repository.save(budget())
        await session.flush()

        await repository.delete(plan_id)
        await session.flush()

        assert await repository.get(plan_id) is None

    async def test_provenance_survives_so_a_plan_keeps_its_owner(
        self, session: AsyncSession
    ) -> None:
        repository = PlanRepository(session)
        plan_id = await repository.save(budget())
        await session.flush()
        session.expire_all()

        stored = await repository.get(plan_id)

        assert stored is not None
        assert stored.provenance.sources[0].source_id == "bud-fy2026"

    async def test_stored_plans_compare_identically(self, session: AsyncSession) -> None:
        """The end-to-end guarantee: persistence changes no variance."""
        repository = PlanRepository(session)
        budget_id = await repository.save(budget())
        actual_id = await repository.save(actuals())
        await session.flush()
        session.expire_all()

        stored_budget = await repository.get(budget_id)
        stored_actual = await repository.get(actual_id)
        assert stored_budget is not None and stored_actual is not None

        before = compare(budget(), actuals(), CHART)
        after = compare(stored_budget, stored_actual, CHART)

        assert before.net_operating_variance() == after.net_operating_variance()
        assert [v.favourability for v in before.variances] == [
            v.favourability for v in after.variances
        ]

    async def test_a_corrupted_row_fails_to_load(self, session: AsyncSession) -> None:
        """A line moved into another fiscal year must not flow into a variance.

        The fiscal year is corrupted rather than the end date, because the end
        date alone does not break containment: a line whose period key still
        matches the plan is inside it by definition, and only a different year
        puts it outside.
        """
        repository = PlanRepository(session)
        plan_id = await repository.save(budget())
        await session.flush()

        # The schema name is a module constant, not input; the value is bound.
        await session.execute(
            text(
                f"UPDATE {SCHEMA}.plan_lines SET fiscal_year = :bad, period_key = :key "  # noqa: S608
                "WHERE plan_id = :plan_id"
            ),
            {"bad": 2099, "key": "annual:2099:0", "plan_id": plan_id},
        )
        session.expire_all()

        with pytest.raises(Exception, match="outside the plan period"):
            await repository.get(plan_id)


class TestCommentary:
    def _commentary(self, *, grounded: bool = True) -> Commentary:
        report = compare(budget(), actuals(), CHART)
        return Commentary(
            entity_id="ent-northwind",
            period_label="FY2026",
            summary="Revenue came in below plan.",
            sections=[CommentarySection(heading="Marketing", body="Above plan.")],
            open_questions=["What drove the marketing overspend?"],
            metrics=report.to_metrics(),
            provenance=Provenance.generated("fake-model-1"),
            numerically_grounded=True,
            directionally_grounded=grounded,
            direction_contradictions=[] if grounded else ["6100 was described as good news"],
        )

    async def test_a_commentary_round_trips_with_its_metrics(self, session: AsyncSession) -> None:
        """Metric values are JSONB, and JSON has only floats — so this is the check."""
        repository = CommentaryRepository(session)
        original = self._commentary()

        commentary_id = await repository.save(original)
        await session.flush()
        session.expire_all()
        stored = await repository.get(commentary_id)

        assert stored is not None
        assert [metric.value for metric in stored[1].metrics] == [
            metric.value for metric in original.metrics
        ]
        assert all(isinstance(metric.value, Decimal) for metric in stored[1].metrics)

    async def test_a_computed_metric_is_never_attributed_to_the_model(
        self, session: AsyncSession
    ) -> None:
        repository = CommentaryRepository(session)
        commentary_id = await repository.save(self._commentary())
        await session.flush()
        session.expire_all()

        stored = await repository.get(commentary_id)

        assert stored is not None
        assert stored[1].provenance.model == "fake-model-1"
        assert all(metric.provenance.model is None for metric in stored[1].metrics)

    async def test_a_reversed_commentary_is_stored_not_discarded(
        self, session: AsyncSession
    ) -> None:
        """A model that inverts directions systematically is visible in this
        table or nowhere."""
        repository = CommentaryRepository(session)
        commentary_id = await repository.save(self._commentary(grounded=False))
        await session.flush()

        stored = await repository.get(commentary_id)

        assert stored is not None
        assert not stored[1].is_publishable
        assert stored[1].direction_contradictions

    async def test_publishable_only_filters_in_the_database(self, session: AsyncSession) -> None:
        """ "Did we ever publish a commentary that reversed a variance" has to be
        a WHERE clause."""
        repository = CommentaryRepository(session)
        await repository.save(self._commentary())
        await repository.save(self._commentary(grounded=False))
        await session.flush()

        every = await repository.list_for_entity("ent-northwind")
        publishable = await repository.list_for_entity("ent-northwind", publishable_only=True)

        assert len(every) == 2
        assert len(publishable) == 1
