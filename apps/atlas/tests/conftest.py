"""Shared fixtures for the Atlas suites."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from atlas.domain.money import Money, Shares
from atlas.domain.periods import FiscalPeriod
from atlas.domain.statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from fie_ai.contracts import CompletionResponse, StopReason, TokenUsage
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from fie_testing import FakeAIProvider

sys.path.insert(0, str(Path(__file__).parent))

FILING_ID = "fil-acme-2025"


def source(source_id: str = FILING_ID) -> SourceReference:
    return SourceReference(
        source_id=source_id, source_type=SourceType.SEC_FILING, title="Acme 10-K FY2025"
    )


def fact(source_id: str = FILING_ID) -> Provenance:
    return Provenance.fact(source(source_id))


def annual_period(year: int = 2025) -> FiscalPeriod:
    return FiscalPeriod.annual(year, date(year, 12, 31), date(year, 1, 1))


def instant_period(year: int = 2025) -> FiscalPeriod:
    return FiscalPeriod.instant(year, date(year, 12, 31))


def income_statement(
    *,
    year: int = 2025,
    revenue: str = "412600000",
    cost_of_revenue: str = "247560000",
    net_income: str = "96548400",
    operating_income: str = "123780000",
    shares: str = "100000000",
) -> IncomeStatement:
    """A self-consistent income statement.

    The defaults reconcile: gross profit equals revenue less cost of revenue,
    and net income equals pretax income less tax. Tests that want a
    contradiction introduce it deliberately.
    """
    revenue_money = Money.of(revenue)
    cost_money = Money.of(cost_of_revenue)
    return IncomeStatement(
        period=annual_period(year),
        provenance=fact(),
        revenue=revenue_money,
        cost_of_revenue=cost_money,
        gross_profit=revenue_money - cost_money,
        operating_income=Money.of(operating_income),
        interest_expense=Money.of("4000000"),
        pretax_income=Money.of("119780000"),
        income_tax_expense=Money.of("23231600"),
        net_income=Money.of(net_income),
        diluted_shares=Shares.of(shares),
    )


def balance_sheet(
    *,
    year: int = 2025,
    total_assets: str = "800000000",
    total_liabilities: str = "480000000",
    equity: str = "320000000",
) -> BalanceSheet:
    """A balancing balance sheet."""
    return BalanceSheet(
        period=instant_period(year),
        provenance=fact(),
        total_assets=Money.of(total_assets),
        current_assets=Money.of("300000000"),
        cash_and_equivalents=Money.of("120000000"),
        inventory=Money.of("90000000"),
        total_liabilities=Money.of(total_liabilities),
        current_liabilities=Money.of("150000000"),
        total_debt=Money.of("200000000"),
        shareholders_equity=Money.of(equity),
    )


def cash_flow_statement(*, year: int = 2025) -> CashFlowStatement:
    return CashFlowStatement(
        period=annual_period(year),
        provenance=fact(),
        operating_cash_flow=Money.of("140000000"),
        capital_expenditures=Money.of("-45000000"),
        investing_cash_flow=Money.of("-60000000"),
        financing_cash_flow=Money.of("-30000000"),
        net_change_in_cash=Money.of("50000000"),
    )


def statements(*, year: int = 2025, **overrides: Any) -> FinancialStatements:
    return FinancialStatements(
        entity_id="ent-acme",
        period=annual_period(year),
        income_statement=overrides.get("income_statement", income_statement(year=year)),
        balance_sheet=overrides.get("balance_sheet", balance_sheet(year=year)),
        cash_flow_statement=overrides.get("cash_flow_statement", cash_flow_statement(year=year)),
    )


def make_json_response(payload: dict[str, Any]) -> CompletionResponse:
    return CompletionResponse(
        text=json.dumps(payload),
        model="fake-model-1",
        provider="fake",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )


def make_router(responses: Sequence[CompletionResponse | Exception]) -> AIRouter:
    registry = ProviderRegistry()
    registry.register(FakeAIProvider(list(responses)), default=True)
    return AIRouter(registry, primary="fake")


@pytest.fixture
def acme_statements() -> FinancialStatements:
    return statements()
