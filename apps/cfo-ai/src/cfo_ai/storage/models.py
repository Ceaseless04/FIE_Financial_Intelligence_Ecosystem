"""Relational storage for charts of accounts, org structure, plans, and commentary.

Money is ``NUMERIC(28, 4)`` for the reason it is everywhere in this ecosystem:
``DOUBLE PRECISION`` would be smaller and faster and would silently reintroduce
the error :mod:`fie_finance.money` refuses at construction.

**Variance reports are deliberately not stored.** A variance is a pure function
of a budget, an actuals set, and the chart of accounts — all three of which are
here. Storing the result would create a second source of truth that goes stale
the moment a line is restated, and a stale variance is worse than a recomputed
one because nothing marks it as old. Commentary *is* stored, because it is
model-authored and cannot be reproduced.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fie_common.utils import new_id, utc_now
from fie_database.postgres import Base

SCHEMA = "cfo_ai"

#: Matches the precision used across the ecosystem's money columns.
MONEY_PRECISION = 28
MONEY_SCALE = 4


class AccountRow(Base):
    """One line in a company's chart of accounts.

    ``type`` is the load-bearing column: it decides whether a variance on this
    account is favourable, and nothing downstream can recover it if it is wrong.
    """

    __tablename__ = "accounts"
    __table_args__ = (
        UniqueConstraint("entity_id", "code", name="uq_accounts_entity_code"),
        Index("ix_accounts_entity", "entity_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("acc"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_code: Mapped[str | None] = mapped_column(String(32))
    #: Alternative names a narrative might use. Read by direction grounding to
    #: tie a sentence to the line it is talking about.
    aliases: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class CostCentreRow(Base):
    """An organisational unit that owns budget."""

    __tablename__ = "cost_centres"
    __table_args__ = (
        UniqueConstraint("entity_id", "centre_id", name="uq_cost_centres_entity_centre"),
        Index("ix_cost_centres_entity", "entity_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ccr"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The company's own identifier for the centre, unique within the entity.
    centre_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    parent_centre_id: Mapped[str | None] = mapped_column(String(64))
    owner: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class PlanRow(Base):
    """A budget, a reforecast, or a set of actuals."""

    __tablename__ = "plans"
    __table_args__ = (
        # A company has one approved FY26 budget and one FY26 actuals set, but
        # may legitimately hold several revisions. The version is part of the
        # key so a board revision does not overwrite what was approved.
        UniqueConstraint(
            "entity_id", "kind", "period_key", "version", name="uq_plans_entity_period_version"
        ),
        Index("ix_plans_entity_period", "entity_id", "period_end"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("pln"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    period_key: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    period_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)

    lines: Mapped[list[PlanLineRow]] = relationship(
        back_populates="plan", cascade="all, delete-orphan"
    )


class PlanLineRow(Base):
    """One amount against one account, centre, and period."""

    __tablename__ = "plan_lines"
    __table_args__ = (
        # The domain refuses a duplicate key at construction; the database
        # refuses it again, because a second row for one key is a double count
        # into every total built from it and nothing downstream would notice.
        UniqueConstraint(
            "plan_id",
            "account_code",
            "cost_centre_id",
            "period_key",
            name="uq_plan_lines_key",
        ),
        Index("ix_plan_lines_plan", "plan_id"),
        Index("ix_plan_lines_account", "account_code"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("pll"))
    plan_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.plans.id", ondelete="CASCADE"), nullable=False
    )
    account_code: Mapped[str] = mapped_column(String(32), nullable=False)
    cost_centre_id: Mapped[str] = mapped_column(String(64), nullable=False)
    period_key: Mapped[str] = mapped_column(String(64), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    memo: Mapped[str | None] = mapped_column(String(500))

    period_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)

    plan: Mapped[PlanRow] = relationship(back_populates="lines")


class CommentaryRow(Base):
    """Generated management commentary and both grounding verdicts.

    The two verdicts are real columns rather than derived on read, because the
    question an operator asks of this table is "did we ever publish a
    commentary that reversed a variance" — and that has to be answerable with a
    WHERE clause.
    """

    __tablename__ = "commentaries"
    __table_args__ = (
        Index("ix_commentaries_entity_created", "entity_id", "created_at"),
        Index(
            "ix_commentaries_grounding",
            "numerically_grounded",
            "directionally_grounded",
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cmt"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    period_label: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    open_questions: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    #: The deterministic figures the narrative was written from. Decimals
    #: serialize to JSON strings, so the values survive without becoming floats.
    metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    numerically_grounded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    unsupported_figures: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    directionally_grounded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: Explanations of any sentence that reversed the meaning of a line.
    direction_contradictions: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    regeneration_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


__all__ = [
    "MONEY_PRECISION",
    "MONEY_SCALE",
    "SCHEMA",
    "AccountRow",
    "CommentaryRow",
    "CostCentreRow",
    "PlanLineRow",
    "PlanRow",
]
