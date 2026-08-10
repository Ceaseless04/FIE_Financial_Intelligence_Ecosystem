"""Relational storage for filings, extracted statements, and research reports.

Two decisions here carry the whole phase's Decimal discipline into the database.

**Money is ``NUMERIC``, never ``float`` and never a JSON number.** PostgreSQL's
``NUMERIC`` is arbitrary-precision decimal and SQLAlchemy round-trips it to
:class:`~decimal.Decimal` exactly. ``DOUBLE PRECISION`` would silently reintroduce
the binary-float error that :mod:`atlas.domain.money` refuses at construction —
storing a validated Decimal into a float column and reading it back is a lossy
conversion no amount of care further up the stack can undo.

**Line items get their own columns rather than a JSON blob.** Statements are the
thing Atlas is asked questions about ("which of these companies grew revenue"),
and a figure buried in a document is a figure nobody can query. The narrative
parts of a report — its sections, its metrics, its provenance — *are* stored as
JSONB, because they are a record to be reproduced rather than aggregated. That
is safe for the same reason: Pydantic serializes ``Decimal`` to a JSON *string*,
so a metric's value survives the round trip without becoming a float.
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

SCHEMA = "atlas"

#: Precision for every monetary column. 24 integer digits is far beyond any
#: real balance sheet — the largest figure a filing has ever carried is ~1e13 —
#: and the four decimal places hold sub-cent amounts that appear in per-share
#: figures and in statements reported in units.
MONEY_PRECISION = 28
MONEY_SCALE = 4


def money_column(**kwargs: Any) -> Mapped[Decimal | None]:
    """A nullable monetary column.

    Nullable because filings differ in what they break out, and a missing line
    item must stay distinguishable from a reported zero. Substituting zero for
    "not disclosed" is the single most dangerous default available here: it
    computes, it looks plausible, and it is wrong.
    """
    return mapped_column(Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=True, **kwargs)


class PeriodColumns:
    """The fiscal period a row describes.

    Denormalised onto every statement table rather than joined from the parent:
    a balance sheet is an *instant* while the income statement covering the same
    filing is a *span*, so they genuinely do not share a period row.
    """

    period_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)


class FilingRow(Base):
    """A source document, retained so every extracted figure stays citable."""

    __tablename__ = "filings"
    __table_args__ = (
        # A redelivered filing is recognised by content hash even when the feed
        # assigns it a new accession number.
        UniqueConstraint("content_hash", name="uq_filings_content_hash"),
        Index("ix_filings_entity_period", "entity_id", "fiscal_year", "period_end"),
        Index("ix_filings_type_filed", "type", "filed_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("fil"))
    #: MarketMind's entity id. Atlas never resolves identity itself; duplicating
    #: that here would let the two products disagree about who a company is.
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    accession_number: Mapped[str | None] = mapped_column(String(64))
    filed_at: Mapped[date | None] = mapped_column(Date)
    uri: Mapped[str | None] = mapped_column(String(2048))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    filing_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )

    period_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)


class StatementSetRow(Base):
    """One period's validated statements, extracted from one filing.

    The set exists as its own row because the analysis layer reads across
    statements — return on equity takes net income from one and equity from
    another — and pairing them at extraction is what stops a caller silently
    combining two different periods.
    """

    __tablename__ = "statement_sets"
    __table_args__ = (
        # One set per company per period. Re-extracting the same filing replaces
        # what was there rather than accumulating rival versions of a quarter.
        #
        # The constraint is over `period_key`, not over the period columns: a
        # composite constraint including the nullable `fiscal_quarter` would not
        # bite for annual periods, because PostgreSQL treats each NULL as
        # distinct. See FiscalPeriod.key.
        UniqueConstraint("entity_id", "period_key", name="uq_statement_sets_entity_period"),
        Index("ix_statement_sets_entity", "entity_id"),
        Index("ix_statement_sets_filing", "filing_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("sts"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    period_key: Mapped[str] = mapped_column(String(64), nullable=False)
    filing_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.filings.id", ondelete="CASCADE"), nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    #: What the extractor refused, kept alongside what it accepted. A statement
    #: set with three rejections is a different object from a clean one, and a
    #: caller that cannot see the difference cannot judge the figures.
    rejected: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)

    period_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)

    income_statement: Mapped[IncomeStatementRow | None] = relationship(
        back_populates="statement_set", cascade="all, delete-orphan", uselist=False
    )
    balance_sheet: Mapped[BalanceSheetRow | None] = relationship(
        back_populates="statement_set", cascade="all, delete-orphan", uselist=False
    )
    cash_flow_statement: Mapped[CashFlowStatementRow | None] = relationship(
        back_populates="statement_set", cascade="all, delete-orphan", uselist=False
    )


class IncomeStatementRow(Base, PeriodColumns):
    """Performance over a span."""

    __tablename__ = "income_statements"
    __table_args__ = (
        UniqueConstraint("statement_set_id", name="uq_income_statements_set"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("inc"))
    statement_set_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.statement_sets.id", ondelete="CASCADE"), nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    #: Revenue and net income are the two figures the domain model requires, so
    #: they are the two the column definition requires.
    revenue: Mapped[Decimal] = mapped_column(Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False)
    net_income: Mapped[Decimal] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False
    )
    cost_of_revenue: Mapped[Decimal | None] = money_column()
    gross_profit: Mapped[Decimal | None] = money_column()
    operating_expenses: Mapped[Decimal | None] = money_column()
    operating_income: Mapped[Decimal | None] = money_column()
    interest_expense: Mapped[Decimal | None] = money_column()
    pretax_income: Mapped[Decimal | None] = money_column()
    income_tax_expense: Mapped[Decimal | None] = money_column()
    diluted_shares: Mapped[Decimal | None] = mapped_column(Numeric(MONEY_PRECISION, MONEY_SCALE))

    statement_set: Mapped[StatementSetRow] = relationship(back_populates="income_statement")


class BalanceSheetRow(Base, PeriodColumns):
    """Position at an instant."""

    __tablename__ = "balance_sheets"
    __table_args__ = (
        UniqueConstraint("statement_set_id", name="uq_balance_sheets_set"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("bal"))
    statement_set_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.statement_sets.id", ondelete="CASCADE"), nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    total_assets: Mapped[Decimal] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False
    )
    total_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False
    )
    shareholders_equity: Mapped[Decimal] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False
    )
    current_assets: Mapped[Decimal | None] = money_column()
    cash_and_equivalents: Mapped[Decimal | None] = money_column()
    inventory: Mapped[Decimal | None] = money_column()
    current_liabilities: Mapped[Decimal | None] = money_column()
    total_debt: Mapped[Decimal | None] = money_column()

    statement_set: Mapped[StatementSetRow] = relationship(back_populates="balance_sheet")


class CashFlowStatementRow(Base, PeriodColumns):
    """Cash movement over a span."""

    __tablename__ = "cash_flow_statements"
    __table_args__ = (
        UniqueConstraint("statement_set_id", name="uq_cash_flow_statements_set"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cfs"))
    statement_set_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.statement_sets.id", ondelete="CASCADE"), nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    operating_cash_flow: Mapped[Decimal] = mapped_column(
        Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=False
    )
    capital_expenditures: Mapped[Decimal | None] = money_column()
    investing_cash_flow: Mapped[Decimal | None] = money_column()
    financing_cash_flow: Mapped[Decimal | None] = money_column()
    net_change_in_cash: Mapped[Decimal | None] = money_column()

    statement_set: Mapped[StatementSetRow] = relationship(back_populates="cash_flow_statement")


class ResearchReportRow(Base):
    """A generated research note and the verdict of its verification.

    ``numerically_grounded`` and ``citations_grounded`` are stored as real
    columns rather than derived on read, because the question an operator asks
    of this table is "did we ever serve an ungrounded report" — and that has to
    be answerable with a ``WHERE`` clause, not by rehydrating every row.

    Ungrounded reports are stored, not discarded. A model that invents figures
    systematically is visible in this table and invisible if the failures are
    thrown away.
    """

    __tablename__ = "research_reports"
    __table_args__ = (
        Index("ix_research_reports_entity_created", "entity_id", "created_at"),
        # Partial-index territory in a larger deployment; a plain index is
        # correct here and the predicate is what matters.
        Index("ix_research_reports_grounding", "numerically_grounded", "citations_grounded"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("rpt"))
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    period_label: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    open_questions: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    #: The deterministic figures the narrative was written from. Decimals
    #: serialize to JSON strings, so the values survive without becoming floats.
    metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    cited_source_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    numerically_grounded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    unsupported_figures: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    citations_grounded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    regeneration_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


__all__ = [
    "MONEY_PRECISION",
    "MONEY_SCALE",
    "SCHEMA",
    "BalanceSheetRow",
    "CashFlowStatementRow",
    "FilingRow",
    "IncomeStatementRow",
    "ResearchReportRow",
    "StatementSetRow",
]
