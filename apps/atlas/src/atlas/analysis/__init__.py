"""Deterministic financial analysis.

No module in this package imports an AI provider, and none may. Every figure
Atlas reports is computed here, in plain Python, from figures read out of a
filing — which is what makes the output reproducible, auditable, and defensible.
"""

from atlas.analysis.comparables import (
    ComparableValuation,
    MarketData,
    PeerMultiple,
    value_from_peers,
)
from atlas.analysis.dcf import (
    DCFAssumptions,
    DCFModel,
    DCFValuation,
    project_cash_flows,
    sensitivity_grid,
    weighted_average_cost_of_capital,
)
from atlas.analysis.growth import (
    compound_annual_growth_rate,
    free_cash_flow,
    net_income_growth,
    revenue_growth,
)
from atlas.analysis.ratios import analyze
from fie_finance.metrics import AnalysisResult, Metric, Unit, derived, estimated, unavailable

__all__ = [
    "AnalysisResult",
    "ComparableValuation",
    "DCFAssumptions",
    "DCFModel",
    "DCFValuation",
    "MarketData",
    "Metric",
    "PeerMultiple",
    "Unit",
    "analyze",
    "compound_annual_growth_rate",
    "derived",
    "estimated",
    "free_cash_flow",
    "net_income_growth",
    "project_cash_flows",
    "revenue_growth",
    "sensitivity_grid",
    "unavailable",
    "value_from_peers",
    "weighted_average_cost_of_capital",
]
