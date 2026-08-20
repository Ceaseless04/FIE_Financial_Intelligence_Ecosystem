"""Atlas initial schema: filings, statement sets, statements, research reports.

Every monetary column is NUMERIC(28, 4). DOUBLE PRECISION would be smaller and
faster and would silently reintroduce the binary-float error that the domain
layer refuses at construction — a valuation is not allowed to depend on which
column type someone picked.

Revision ID: b2d4a6c8e013
Revises:
Create Date: 2026-08-10 16:00:00+00:00
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b2d4a6c8e013"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "atlas"

#: Matches atlas.storage.models.MONEY_PRECISION / MONEY_SCALE.
MONEY_PRECISION = 28
MONEY_SCALE = 4


def _money(name: str, *, nullable: bool = True) -> sa.Column[Any]:
    return sa.Column(name, sa.Numeric(MONEY_PRECISION, MONEY_SCALE), nullable=nullable)


def _period_columns() -> list[sa.Column[Any]]:
    return [
        sa.Column("period_kind", sa.String(length=32), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("fiscal_quarter", sa.Integer(), nullable=True),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=False),
    ]


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    op.create_table(
        "filings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("accession_number", sa.String(length=64), nullable=True),
        sa.Column("filed_at", sa.Date(), nullable=True),
        sa.Column("uri", sa.String(length=2048), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        *_period_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_filings"),
        sa.UniqueConstraint("content_hash", name="uq_filings_content_hash"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_filings_entity_period",
        "filings",
        ["entity_id", "fiscal_year", "period_end"],
        schema=SCHEMA,
    )
    op.create_index("ix_filings_type_filed", "filings", ["type", "filed_at"], schema=SCHEMA)

    op.create_table(
        "statement_sets",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        # A composite constraint including the nullable fiscal_quarter would not
        # constrain annual periods at all: PostgreSQL treats every NULL as
        # distinct, so two FY2025 sets for one company would both be accepted.
        sa.Column("period_key", sa.String(length=64), nullable=False),
        sa.Column("filing_id", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rejected", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        *_period_columns(),
        sa.ForeignKeyConstraint(
            ["filing_id"],
            [f"{SCHEMA}.filings.id"],
            name="fk_statement_sets_filing_id_filings",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_statement_sets"),
        sa.UniqueConstraint("entity_id", "period_key", name="uq_statement_sets_entity_period"),
        schema=SCHEMA,
    )
    op.create_index("ix_statement_sets_entity", "statement_sets", ["entity_id"], schema=SCHEMA)
    op.create_index("ix_statement_sets_filing", "statement_sets", ["filing_id"], schema=SCHEMA)

    op.create_table(
        "income_statements",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("statement_set_id", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        _money("revenue", nullable=False),
        _money("net_income", nullable=False),
        _money("cost_of_revenue"),
        _money("gross_profit"),
        _money("operating_expenses"),
        _money("operating_income"),
        _money("interest_expense"),
        _money("pretax_income"),
        _money("income_tax_expense"),
        _money("diluted_shares"),
        *_period_columns(),
        sa.ForeignKeyConstraint(
            ["statement_set_id"],
            [f"{SCHEMA}.statement_sets.id"],
            name="fk_income_statements_statement_set_id_statement_sets",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_income_statements"),
        sa.UniqueConstraint("statement_set_id", name="uq_income_statements_set"),
        schema=SCHEMA,
    )

    op.create_table(
        "balance_sheets",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("statement_set_id", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        _money("total_assets", nullable=False),
        _money("total_liabilities", nullable=False),
        _money("shareholders_equity", nullable=False),
        _money("current_assets"),
        _money("cash_and_equivalents"),
        _money("inventory"),
        _money("current_liabilities"),
        _money("total_debt"),
        *_period_columns(),
        sa.ForeignKeyConstraint(
            ["statement_set_id"],
            [f"{SCHEMA}.statement_sets.id"],
            name="fk_balance_sheets_statement_set_id_statement_sets",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_balance_sheets"),
        sa.UniqueConstraint("statement_set_id", name="uq_balance_sheets_set"),
        schema=SCHEMA,
    )

    op.create_table(
        "cash_flow_statements",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("statement_set_id", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        _money("operating_cash_flow", nullable=False),
        _money("capital_expenditures"),
        _money("investing_cash_flow"),
        _money("financing_cash_flow"),
        _money("net_change_in_cash"),
        *_period_columns(),
        sa.ForeignKeyConstraint(
            ["statement_set_id"],
            [f"{SCHEMA}.statement_sets.id"],
            name="fk_cash_flow_statements_statement_set_id_statement_sets",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cash_flow_statements"),
        sa.UniqueConstraint("statement_set_id", name="uq_cash_flow_statements_set"),
        schema=SCHEMA,
    )

    op.create_table(
        "research_reports",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("period_label", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("sections", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("open_questions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("cited_source_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        # Stored, not derived on read: "did we ever serve an ungrounded report"
        # has to be answerable with a WHERE clause.
        sa.Column("numerically_grounded", sa.Boolean(), nullable=False),
        sa.Column("unsupported_figures", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("citations_grounded", sa.Boolean(), nullable=False),
        sa.Column("regeneration_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_research_reports"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_research_reports_entity_created",
        "research_reports",
        ["entity_id", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_research_reports_grounding",
        "research_reports",
        ["numerically_grounded", "citations_grounded"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("ix_research_reports_grounding", table_name="research_reports", schema=SCHEMA)
    op.drop_index(
        "ix_research_reports_entity_created", table_name="research_reports", schema=SCHEMA
    )
    op.drop_table("research_reports", schema=SCHEMA)

    op.drop_table("cash_flow_statements", schema=SCHEMA)
    op.drop_table("balance_sheets", schema=SCHEMA)
    op.drop_table("income_statements", schema=SCHEMA)

    op.drop_index("ix_statement_sets_filing", table_name="statement_sets", schema=SCHEMA)
    op.drop_index("ix_statement_sets_entity", table_name="statement_sets", schema=SCHEMA)
    op.drop_table("statement_sets", schema=SCHEMA)

    op.drop_index("ix_filings_type_filed", table_name="filings", schema=SCHEMA)
    op.drop_index("ix_filings_entity_period", table_name="filings", schema=SCHEMA)
    op.drop_table("filings", schema=SCHEMA)

    # Deliberately asymmetric: the tables go, the schema stays. Alembic's own
    # version table lives in this schema, so dropping it here would delete the
    # bookkeeping mid-downgrade and leave the database unmigratable.
