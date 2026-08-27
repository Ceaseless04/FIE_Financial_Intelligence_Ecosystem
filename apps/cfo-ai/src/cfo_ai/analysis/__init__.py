"""Deterministic FP&A: variance, roll-ups, forecasting, cash, scenarios.

No module in this package imports an AI provider, and a test asserts it. Every
figure a CFO.ai narrative can quote is produced here first.
"""

from cfo_ai.analysis.cash import BurnRate, CashPosition, CashPosture, Runway, runway
from cfo_ai.analysis.compare import CentreRollup, ComparisonSettings, compare, roll_up
from cfo_ai.analysis.forecast import Forecast, ForecastMethod, Observation, forecast
from cfo_ai.analysis.scenarios import Driver, DriverScope, Scenario, ScenarioResult, apply_scenario
from cfo_ai.analysis.variance import Favourability, Variance, VarianceReport

__all__ = [
    "BurnRate",
    "CashPosition",
    "CashPosture",
    "CentreRollup",
    "ComparisonSettings",
    "Driver",
    "DriverScope",
    "Favourability",
    "Forecast",
    "ForecastMethod",
    "Observation",
    "Runway",
    "Scenario",
    "ScenarioResult",
    "Variance",
    "VarianceReport",
    "apply_scenario",
    "compare",
    "forecast",
    "roll_up",
    "runway",
]
