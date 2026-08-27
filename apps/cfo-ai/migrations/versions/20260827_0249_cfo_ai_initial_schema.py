"""CFO.ai initial schema: chart of accounts, org structure, plans, commentary.

Every monetary column is NUMERIC(28, 4). DOUBLE PRECISION would be smaller and
faster and would silently reintroduce the binary-float error the domain layer
refuses at construction.

No variance table. A variance is a pure function of a budget, an actuals set,
and the chart of accounts, all three of which are stored here; a fourth copy
would go stale the moment a line is restated, and nothing would mark it as old.

Revision ID: 366f65f86f30
Revises: 
Create Date: 2026-08-27 02:49:59.570541+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision: str = '366f65f86f30'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Self-sufficient against a fresh database. env.py also creates the schema,
    # because Alembic writes its version table before any migration runs, but a
    # migration that depends on that is a migration nobody can replay from SQL.
    op.execute("CREATE SCHEMA IF NOT EXISTS cfo_ai")

    op.create_table('accounts',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('entity_id', sa.String(length=64), nullable=False),
    sa.Column('code', sa.String(length=32), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('type', sa.String(length=32), nullable=False),
    sa.Column('parent_code', sa.String(length=32), nullable=True),
    sa.Column('aliases', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_accounts')),
    sa.UniqueConstraint('entity_id', 'code', name='uq_accounts_entity_code'),
    schema='cfo_ai'
    )
    op.create_index('ix_accounts_entity', 'accounts', ['entity_id'], unique=False, schema='cfo_ai')
    op.create_table('commentaries',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('entity_id', sa.String(length=64), nullable=False),
    sa.Column('period_label', sa.String(length=64), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('sections', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('open_questions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('provenance', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('numerically_grounded', sa.Boolean(), nullable=False),
    sa.Column('unsupported_figures', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('directionally_grounded', sa.Boolean(), nullable=False),
    sa.Column('direction_contradictions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('regeneration_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_commentaries')),
    schema='cfo_ai'
    )
    op.create_index('ix_commentaries_entity_created', 'commentaries', ['entity_id', 'created_at'], unique=False, schema='cfo_ai')
    op.create_index('ix_commentaries_grounding', 'commentaries', ['numerically_grounded', 'directionally_grounded'], unique=False, schema='cfo_ai')
    op.create_table('cost_centres',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('entity_id', sa.String(length=64), nullable=False),
    sa.Column('centre_id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('parent_centre_id', sa.String(length=64), nullable=True),
    sa.Column('owner', sa.String(length=200), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_cost_centres')),
    sa.UniqueConstraint('entity_id', 'centre_id', name='uq_cost_centres_entity_centre'),
    schema='cfo_ai'
    )
    op.create_index('ix_cost_centres_entity', 'cost_centres', ['entity_id'], unique=False, schema='cfo_ai')
    op.create_table('plans',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('entity_id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('period_key', sa.String(length=64), nullable=False),
    sa.Column('version', sa.String(length=64), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('provenance', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('period_kind', sa.String(length=32), nullable=False),
    sa.Column('fiscal_year', sa.Integer(), nullable=False),
    sa.Column('fiscal_quarter', sa.Integer(), nullable=True),
    sa.Column('period_start', sa.Date(), nullable=True),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_plans')),
    sa.UniqueConstraint('entity_id', 'kind', 'period_key', 'version', name='uq_plans_entity_period_version'),
    schema='cfo_ai'
    )
    op.create_index('ix_plans_entity_period', 'plans', ['entity_id', 'period_end'], unique=False, schema='cfo_ai')
    op.create_table('plan_lines',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('plan_id', sa.String(length=64), nullable=False),
    sa.Column('account_code', sa.String(length=32), nullable=False),
    sa.Column('cost_centre_id', sa.String(length=64), nullable=False),
    sa.Column('period_key', sa.String(length=64), nullable=False),
    sa.Column('amount', sa.Numeric(precision=28, scale=4), nullable=False),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('memo', sa.String(length=500), nullable=True),
    sa.Column('period_kind', sa.String(length=32), nullable=False),
    sa.Column('fiscal_year', sa.Integer(), nullable=False),
    sa.Column('fiscal_quarter', sa.Integer(), nullable=True),
    sa.Column('period_start', sa.Date(), nullable=True),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.ForeignKeyConstraint(['plan_id'], ['cfo_ai.plans.id'], name=op.f('fk_plan_lines_plan_id_plans'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_plan_lines')),
    sa.UniqueConstraint('plan_id', 'account_code', 'cost_centre_id', 'period_key', name='uq_plan_lines_key'),
    schema='cfo_ai'
    )
    op.create_index('ix_plan_lines_account', 'plan_lines', ['account_code'], unique=False, schema='cfo_ai')
    op.create_index('ix_plan_lines_plan', 'plan_lines', ['plan_id'], unique=False, schema='cfo_ai')


def downgrade() -> None:
    op.drop_index('ix_plan_lines_plan', table_name='plan_lines', schema='cfo_ai')
    op.drop_index('ix_plan_lines_account', table_name='plan_lines', schema='cfo_ai')
    op.drop_table('plan_lines', schema='cfo_ai')
    op.drop_index('ix_plans_entity_period', table_name='plans', schema='cfo_ai')
    op.drop_table('plans', schema='cfo_ai')
    op.drop_index('ix_cost_centres_entity', table_name='cost_centres', schema='cfo_ai')
    op.drop_table('cost_centres', schema='cfo_ai')
    op.drop_index('ix_commentaries_grounding', table_name='commentaries', schema='cfo_ai')
    op.drop_index('ix_commentaries_entity_created', table_name='commentaries', schema='cfo_ai')
    op.drop_table('commentaries', schema='cfo_ai')
    op.drop_index('ix_accounts_entity', table_name='accounts', schema='cfo_ai')
    op.drop_table('accounts', schema='cfo_ai')

    # Deliberately asymmetric: the tables go, the schema stays. Alembic's own
    # version table lives in this schema, so dropping it here would delete the
    # bookkeeping mid-downgrade and leave the database unmigratable.
