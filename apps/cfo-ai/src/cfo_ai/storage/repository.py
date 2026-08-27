"""Persistence for the chart, the org, plans, and commentary.

Reading is where the care goes. A ``NUMERIC`` column comes back as a
:class:`~decimal.Decimal` and is handed straight to
:class:`~fie_finance.money.Money`, which would refuse anything else — so a plan
rebuilt here re-runs every domain invariant on construction. A duplicated line
that somehow reached the database fails to load rather than quietly
double-counting into a variance report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cfo_ai.domain.accounts import Account, AccountType
from cfo_ai.domain.organisation import CostCentre, CostCentreTree
from cfo_ai.domain.plan import Plan, PlanKind, PlanLine
from cfo_ai.research.commentary import Commentary, CommentarySection
from cfo_ai.storage.models import (
    AccountRow,
    CommentaryRow,
    CostCentreRow,
    PlanLineRow,
    PlanRow,
)
from fie_finance.metrics import Metric
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod, PeriodKind
from fie_observability.logging import get_logger
from fie_schemas.provenance import Provenance

logger = get_logger(__name__)


def _period_values(period: FiscalPeriod) -> dict[str, Any]:
    return {
        "period_key": period.key,
        "period_kind": str(period.kind),
        "fiscal_year": period.fiscal_year,
        "fiscal_quarter": period.fiscal_quarter,
        "period_start": period.start_date,
        "period_end": period.end_date,
    }


def _period_of(row: Any) -> FiscalPeriod:
    return FiscalPeriod(
        kind=PeriodKind(row.period_kind),
        fiscal_year=row.fiscal_year,
        fiscal_quarter=row.fiscal_quarter,
        start_date=row.period_start,
        end_date=row.period_end,
    )


@dataclass
class ChartRepository:
    """The chart of accounts and the cost centre hierarchy."""

    session: AsyncSession

    async def replace_accounts(self, entity_id: str, accounts: Sequence[Account]) -> int:
        """Store a company's chart, replacing whatever was there.

        Replacement rather than merge: a chart of accounts is published as a
        whole, and a partial update would leave an account nobody meant to keep
        deciding the direction of a variance.
        """
        await self.session.execute(delete(AccountRow).where(AccountRow.entity_id == entity_id))
        for account in accounts:
            self.session.add(
                AccountRow(
                    entity_id=entity_id,
                    code=account.code,
                    name=account.name,
                    type=str(account.type),
                    parent_code=account.parent_code,
                    aliases=list(account.aliases),
                )
            )
        await self.session.flush()
        logger.info("chart_stored", entity_id=entity_id, accounts=len(accounts))
        return len(accounts)

    async def accounts(self, entity_id: str) -> dict[str, Account]:
        """The chart, keyed by code."""
        rows = await self.session.scalars(
            select(AccountRow).where(AccountRow.entity_id == entity_id).order_by(AccountRow.code)
        )
        return {
            row.code: Account(
                code=row.code,
                name=row.name,
                type=AccountType(row.type),
                parent_code=row.parent_code,
                aliases=list(row.aliases or []),
            )
            for row in rows
        }

    async def replace_cost_centres(self, entity_id: str, tree: CostCentreTree) -> int:
        """Store the hierarchy, replacing whatever was there.

        The tree validated itself on construction, so what reaches the database
        is already acyclic with every parent resolvable.
        """
        await self.session.execute(
            delete(CostCentreRow).where(CostCentreRow.entity_id == entity_id)
        )
        for centre in tree.centres:
            self.session.add(
                CostCentreRow(
                    entity_id=entity_id,
                    centre_id=centre.id,
                    name=centre.name,
                    parent_centre_id=centre.parent_id,
                    owner=centre.owner,
                )
            )
        await self.session.flush()
        return len(tree.centres)

    async def cost_centres(self, entity_id: str) -> CostCentreTree | None:
        """The hierarchy, re-validated on the way out.

        Returns ``None`` when the company has none. A malformed one raises
        rather than loading: a cycle that reached the database would make every
        roll-up built from it non-terminating.
        """
        rows = list(
            await self.session.scalars(
                select(CostCentreRow).where(CostCentreRow.entity_id == entity_id)
            )
        )
        if not rows:
            return None
        return CostCentreTree(
            centres=[
                CostCentre(
                    id=row.centre_id,
                    name=row.name,
                    parent_id=row.parent_centre_id,
                    owner=row.owner,
                )
                for row in rows
            ]
        )


@dataclass
class PlanRepository:
    """Budgets, reforecasts, and actuals."""

    session: AsyncSession

    async def save(self, plan: Plan) -> str:
        """Store a plan, replacing any existing one with the same identity.

        Identity is entity, kind, period, and version — so a board revision
        stored as ``v2`` never overwrites the approved ``v1``, and re-uploading
        the same version supersedes it rather than accumulating duplicates.
        """
        existing = await self.session.scalar(
            select(PlanRow).where(
                PlanRow.entity_id == plan.entity_id,
                PlanRow.kind == str(plan.kind),
                PlanRow.period_key == plan.period.key,
                PlanRow.version == plan.version,
            )
        )
        if existing is not None:
            await self.session.delete(existing)
            await self.session.flush()

        row = PlanRow(
            id=plan.id,
            entity_id=plan.entity_id,
            kind=str(plan.kind),
            version=plan.version,
            currency=plan.currency,
            provenance=plan.provenance.model_dump(mode="json"),
            created_at=plan.created_at,
            **_period_values(plan.period),
        )
        for line in plan.lines:
            row.lines.append(
                PlanLineRow(
                    account_code=line.account_code,
                    cost_centre_id=line.cost_centre_id,
                    amount=line.amount.amount,
                    currency=line.amount.currency,
                    memo=line.memo,
                    **_period_values(line.period),
                )
            )

        self.session.add(row)
        await self.session.flush()
        logger.info(
            "plan_stored",
            plan_id=row.id,
            entity_id=plan.entity_id,
            kind=str(plan.kind),
            lines=len(plan.lines),
        )
        return row.id

    async def get(self, plan_id: str) -> Plan | None:
        row = await self.session.scalar(
            select(PlanRow).options(selectinload(PlanRow.lines)).where(PlanRow.id == plan_id)
        )
        return self._to_plan(row) if row is not None else None

    async def find(
        self,
        entity_id: str,
        *,
        kind: PlanKind,
        period: FiscalPeriod,
        version: str | None = None,
    ) -> Plan | None:
        """One plan by identity, or the most recent version of it."""
        statement = (
            select(PlanRow)
            .options(selectinload(PlanRow.lines))
            .where(
                PlanRow.entity_id == entity_id,
                PlanRow.kind == str(kind),
                PlanRow.period_key == period.key,
            )
        )
        if version is not None:
            statement = statement.where(PlanRow.version == version)
        else:
            statement = statement.order_by(PlanRow.created_at.desc())

        row = await self.session.scalar(statement.limit(1))
        return self._to_plan(row) if row is not None else None

    async def list_for_entity(
        self, entity_id: str, *, kind: PlanKind | None = None, limit: int = 20
    ) -> list[Plan]:
        statement = (
            select(PlanRow)
            .options(selectinload(PlanRow.lines))
            .where(PlanRow.entity_id == entity_id)
            .order_by(PlanRow.period_end.desc(), PlanRow.created_at.desc())
            .limit(limit)
        )
        if kind is not None:
            statement = statement.where(PlanRow.kind == str(kind))
        rows = await self.session.scalars(statement)
        return [self._to_plan(row) for row in rows]

    async def delete(self, plan_id: str) -> None:
        await self.session.execute(delete(PlanRow).where(PlanRow.id == plan_id))

    @staticmethod
    def _to_plan(row: PlanRow) -> Plan:
        """Rebuild a plan, re-running every domain invariant on the way."""
        return Plan(
            id=row.id,
            entity_id=row.entity_id,
            kind=PlanKind(row.kind),
            period=_period_of(row),
            currency=row.currency,
            version=row.version,
            lines=[
                PlanLine(
                    account_code=line.account_code,
                    cost_centre_id=line.cost_centre_id,
                    period=_period_of(line),
                    amount=Money(amount=line.amount, currency=line.currency),
                    memo=line.memo,
                )
                for line in sorted(
                    row.lines, key=lambda line: (line.account_code, line.cost_centre_id)
                )
            ],
            provenance=Provenance.model_validate(row.provenance),
            created_at=row.created_at,
        )


@dataclass
class CommentaryRepository:
    """Generated commentary, grounded or not."""

    session: AsyncSession

    async def save(self, commentary: Commentary) -> str:
        row = CommentaryRow(
            entity_id=commentary.entity_id,
            period_label=commentary.period_label,
            summary=commentary.summary,
            sections=[section.model_dump(mode="json") for section in commentary.sections],
            open_questions=list(commentary.open_questions),
            metrics=[metric.model_dump(mode="json") for metric in commentary.metrics],
            provenance=commentary.provenance.model_dump(mode="json"),
            numerically_grounded=commentary.numerically_grounded,
            unsupported_figures=list(commentary.unsupported_figures),
            directionally_grounded=commentary.directionally_grounded,
            direction_contradictions=list(commentary.direction_contradictions),
            regeneration_count=commentary.regeneration_count,
        )
        self.session.add(row)
        await self.session.flush()
        logger.info(
            "commentary_stored",
            commentary_id=row.id,
            entity_id=commentary.entity_id,
            publishable=commentary.is_publishable,
        )
        return row.id

    async def get(self, commentary_id: str) -> tuple[str, Commentary] | None:
        row = await self.session.get(CommentaryRow, commentary_id)
        return (row.id, self._to_commentary(row)) if row is not None else None

    async def list_for_entity(
        self, entity_id: str, *, limit: int = 20, publishable_only: bool = False
    ) -> list[tuple[str, Commentary]]:
        statement = (
            select(CommentaryRow)
            .where(CommentaryRow.entity_id == entity_id)
            .order_by(CommentaryRow.created_at.desc())
            .limit(limit)
        )
        if publishable_only:
            statement = statement.where(
                CommentaryRow.numerically_grounded.is_(True),
                CommentaryRow.directionally_grounded.is_(True),
            )
        rows = await self.session.scalars(statement)
        return [(row.id, self._to_commentary(row)) for row in rows]

    @staticmethod
    def _to_commentary(row: CommentaryRow) -> Commentary:
        return Commentary(
            entity_id=row.entity_id,
            period_label=row.period_label,
            summary=row.summary,
            sections=[CommentarySection.model_validate(item) for item in row.sections],
            open_questions=list(row.open_questions),
            # Metric values were written as JSON strings; the model's own
            # validator parses them back into Decimal and refuses floats.
            metrics=[Metric.model_validate(item) for item in row.metrics],
            provenance=Provenance.model_validate(row.provenance),
            numerically_grounded=row.numerically_grounded,
            unsupported_figures=list(row.unsupported_figures),
            directionally_grounded=row.directionally_grounded,
            direction_contradictions=list(row.direction_contradictions),
            regeneration_count=row.regeneration_count,
        )


__all__ = ["ChartRepository", "CommentaryRepository", "PlanRepository"]
