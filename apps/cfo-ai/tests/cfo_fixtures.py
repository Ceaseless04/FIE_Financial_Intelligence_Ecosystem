"""Builders shared across the CFO.ai suites.

Named for the app rather than something generic: helper modules are imported by
bare module name and ``sys.modules`` is process-wide, so two apps offering a
module called ``conftest`` or ``fixtures`` would resolve to whichever loaded
first. Phase 3 lost an afternoon to exactly that.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

from cfo_ai.domain.accounts import Account, AccountType
from cfo_ai.domain.organisation import CostCentre, CostCentreTree
from cfo_ai.domain.plan import Plan, PlanKind, line
from fie_ai.contracts import CompletionResponse, StopReason, TokenUsage
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_finance.periods import FiscalPeriod
from fie_schemas.provenance import Provenance, SourceReference, SourceType
from fie_testing import FakeAIProvider

SOURCE_ID = "bud-fy2026"

REVENUE = Account(
    code="4000", name="Revenue", type=AccountType.REVENUE, aliases=["sales", "top line"]
)
COGS = Account(code="5000", name="Cost of Revenue", type=AccountType.COST_OF_REVENUE)
MARKETING = Account(
    code="6100", name="Marketing", type=AccountType.OPERATING_EXPENSE, aliases=["marketing spend"]
)
SALARIES = Account(code="6200", name="Salaries", type=AccountType.OPERATING_EXPENSE)
HEADCOUNT = Account(code="9000", name="Headcount", type=AccountType.HEADCOUNT)

CHART: dict[str, Account] = {
    account.code: account for account in (REVENUE, COGS, MARKETING, SALARIES, HEADCOUNT)
}


def source(source_id: str = SOURCE_ID) -> SourceReference:
    return SourceReference(
        source_id=source_id, source_type=SourceType.INTERNAL_DOCUMENT, title="FY2026 board budget"
    )


def fact(source_id: str = SOURCE_ID) -> Provenance:
    return Provenance.fact(source(source_id))


def annual(year: int = 2026) -> FiscalPeriod:
    return FiscalPeriod.annual(year, date(year, 12, 31), date(year, 1, 1))


def quarter(year: int = 2026, number: int = 1) -> FiscalPeriod:
    ends = {1: date(year, 3, 31), 2: date(year, 6, 30), 3: date(year, 9, 30), 4: date(year, 12, 31)}
    starts = {1: date(year, 1, 1), 2: date(year, 4, 1), 3: date(year, 7, 1), 4: date(year, 10, 1)}
    return FiscalPeriod.quarterly(year, number, ends[number], starts[number])


def tree() -> CostCentreTree:
    """Company > (Engineering, Sales > Field Sales)."""
    return CostCentreTree(
        centres=[
            CostCentre(id="cc-co", name="Company"),
            CostCentre(id="cc-eng", name="Engineering", parent_id="cc-co", owner="VP Eng"),
            CostCentre(id="cc-sales", name="Sales", parent_id="cc-co", owner="CRO"),
            CostCentre(id="cc-field", name="Field Sales", parent_id="cc-sales"),
        ]
    )


def budget(*, year: int = 2026, **amounts: str) -> Plan:
    """A plan whose defaults are round numbers, so variances are checkable by eye."""
    period = annual(year)
    values = {
        "revenue": "4000000",
        "cogs": "1200000",
        "marketing": "600000",
        "salaries": "1800000",
    }
    values.update(amounts)
    return Plan(
        entity_id="ent-northwind",
        kind=PlanKind.BUDGET,
        period=period,
        lines=[
            line("4000", "cc-sales", period, values["revenue"]),
            line("5000", "cc-eng", period, values["cogs"]),
            line("6100", "cc-sales", period, values["marketing"]),
            line("6200", "cc-eng", period, values["salaries"]),
        ],
        provenance=fact(),
    )


def actuals(*, year: int = 2026, **amounts: str) -> Plan:
    period = annual(year)
    values = {
        "revenue": "3600000",
        "cogs": "1150000",
        "marketing": "700000",
        "salaries": "1750000",
    }
    values.update(amounts)
    return Plan(
        entity_id="ent-northwind",
        kind=PlanKind.ACTUAL,
        period=period,
        lines=[
            line("4000", "cc-sales", period, values["revenue"]),
            line("5000", "cc-eng", period, values["cogs"]),
            line("6100", "cc-sales", period, values["marketing"]),
            line("6200", "cc-eng", period, values["salaries"]),
        ],
        provenance=fact("act-fy2026"),
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


__all__ = [
    "CHART",
    "COGS",
    "HEADCOUNT",
    "MARKETING",
    "REVENUE",
    "SALARIES",
    "SOURCE_ID",
    "actuals",
    "annual",
    "budget",
    "fact",
    "make_json_response",
    "make_router",
    "quarter",
    "source",
    "tree",
]
