"""Builders shared across the Atlas suites.

Named for the app rather than something generic. Test helper modules are
imported by bare module name — the app's ``tests`` directory is on ``sys.path``
— and ``sys.modules`` is process-wide, so two apps that both offered a module
called ``conftest`` or ``fixtures`` would silently resolve to whichever loaded
first. That failure only appears once a second app exists, and then it appears
as every test in one app erroring at setup.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

from atlas.domain.statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from atlas.filings.models import Filing, FilingType
from fie_ai.contracts import CompletionResponse, StopReason, TokenUsage
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_finance.money import Money, Shares
from fie_finance.periods import FiscalPeriod
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from fie_testing import FakeAIProvider

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


def filing(
    *,
    entity_id: str = "ent-acme",
    year: int = 2025,
    filing_type: FilingType = FilingType.ANNUAL_10K,
    content: str = "Item 8. Financial Statements. Revenue was 412.6 for the year.",
    filing_id: str | None = None,
) -> Filing:
    return Filing(
        **({"id": filing_id} if filing_id else {}),
        entity_id=entity_id,
        type=filing_type,
        period=annual_period(year),
        title=f"Acme Robotics Corporation Form 10-K FY{year}",
        content=content,
        accession_number=f"0000123456-{year}-000001",
        filed_at=date(year + 1, 2, 14),
        uri=f"https://sec.example.invalid/acme/{year}/10-K",
    )


def extraction_payload(**overrides: Any) -> dict[str, Any]:
    """A transcription the extractor will accept.

    Figures are strings, as the prompt demands, and the scale is stated
    separately so Atlas applies it rather than the model. The defaults reconcile
    against every accounting identity the domain models check.
    """
    payload: dict[str, Any] = {
        "currency": "USD",
        "scale": "units",
        "income_statement": {
            "revenue": "412600000",
            "cost_of_revenue": "247560000",
            "gross_profit": "165040000",
            "operating_income": "123780000",
            "interest_expense": "4000000",
            "pretax_income": "119780000",
            "income_tax_expense": "23231600",
            "net_income": "96548400",
            "diluted_shares": "100000000",
        },
        "balance_sheet": {
            "total_assets": "800000000",
            "current_assets": "300000000",
            "cash_and_equivalents": "120000000",
            "inventory": "90000000",
            "total_liabilities": "480000000",
            "current_liabilities": "150000000",
            "total_debt": "200000000",
            "shareholders_equity": "320000000",
        },
        "cash_flow_statement": {
            "operating_cash_flow": "140000000",
            "capital_expenditures": "-45000000",
            "investing_cash_flow": "-60000000",
            "financing_cash_flow": "-30000000",
            "net_change_in_cash": "50000000",
        },
    }
    payload.update(overrides)
    return payload


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
