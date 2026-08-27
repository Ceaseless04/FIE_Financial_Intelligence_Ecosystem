"""Forecasting, cash, and scenarios — the parts that describe the future.

Everything here is an estimate, and the tests are mostly about that word. A
projection that travels without its assumptions, or a runway rendered as a
number when the question does not apply, is a guess wearing a measurement's
clothes.

The arithmetic is checked against values worked out independently rather than
against what the code happens to return.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from cfo_fixtures import budget
from pydantic import ValidationError as PydanticValidationError

from cfo_ai.analysis.cash import BurnRate, CashPosture, runway
from cfo_ai.analysis.forecast import ForecastMethod as Method
from cfo_ai.analysis.forecast import Observation, forecast
from cfo_ai.analysis.scenarios import (
    Driver,
    DriverScope,
    Scenario,
    apply_scenario,
    compare_scenarios,
)
from fie_common.errors import ValidationError
from fie_finance.money import Money
from fie_finance.periods import FiscalPeriod
from fie_schemas.provenance import AssertionKind

pytestmark = pytest.mark.unit


def history(*amounts: str, start_year: int = 2020) -> list[Observation]:
    return [
        Observation(
            period=FiscalPeriod.annual(start_year + index, date(start_year + index, 12, 31)),
            amount=Money.of(amount),
        )
        for index, amount in enumerate(amounts)
    ]


class TestForecastArithmetic:
    def test_a_run_rate_repeats_the_last_period(self) -> None:
        result = forecast("4000", history("100", "120", "150"), periods=2)
        assert [point.amount for point in result.points] == [Money.of("150")] * 2

    def test_a_moving_average_is_the_mean_of_the_window(self) -> None:
        """Hand: (100 + 110 + 120 + 130) / 4 = 115."""
        result = forecast("4000", history("100", "110", "120", "130"), method=Method.MOVING_AVERAGE)
        assert result.points[0].amount.amount == Decimal("115.000000")

    def test_a_window_shortens_the_average(self) -> None:
        """Hand: (120 + 130) / 2 = 125."""
        result = forecast(
            "4000", history("100", "110", "120", "130"), method=Method.MOVING_AVERAGE, window=2
        )
        assert result.points[0].amount.amount == Decimal("125.000000")

    def test_a_linear_trend_extends_the_line(self) -> None:
        """Hand: a perfect line through 100,110,120,130 has slope 10, so the next
        two points are 140 and 150."""
        result = forecast(
            "4000", history("100", "110", "120", "130"), periods=2, method=Method.LINEAR_TREND
        )

        assert result.assumptions["slope_per_period"] == "10.00"
        assert [point.amount.amount for point in result.points] == [
            Decimal("140.00"),
            Decimal("150.00"),
        ]

    def test_a_declining_trend_projects_downward(self) -> None:
        """Hand: slope -5 through 100,95,90 gives 85 next."""
        result = forecast("4000", history("100", "95", "90"), periods=1, method=Method.LINEAR_TREND)
        assert result.points[0].amount.amount == Decimal("85.00")

    def test_history_arrives_in_any_order(self) -> None:
        scrambled = list(reversed(history("100", "110", "120")))
        assert forecast("4000", scrambled).points[0].amount == Money.of("120")

    def test_the_total_sums_the_projection(self) -> None:
        result = forecast("4000", history("100"), periods=3)
        assert result.total == Money.of("300")


class TestForecastRefusals:
    def test_too_little_history_for_the_method_is_refused(self) -> None:
        """A trend through two points is a line through two points, not a trend."""
        with pytest.raises(ValidationError, match="at least 3 periods"):
            forecast("4000", history("100", "110"), method=Method.LINEAR_TREND)

    def test_a_seasonal_forecast_needs_a_full_year(self) -> None:
        with pytest.raises(ValidationError, match="at least 12 periods"):
            forecast("4000", history("100", "110", "120"), method=Method.SEASONAL_NAIVE)

    def test_empty_history_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs history"):
            forecast("4000", [])

    def test_duplicate_periods_are_refused(self) -> None:
        """They would double-count into every projection built from them."""
        duplicated = history("100", "110")
        duplicated.append(duplicated[0])
        with pytest.raises(ValidationError, match="two observations for one period"):
            forecast("4000", duplicated)

    def test_mixed_currencies_are_refused(self) -> None:
        mixed = history("100")
        mixed.append(
            Observation(
                period=FiscalPeriod.annual(2030, date(2030, 12, 31)), amount=Money.of("100", "EUR")
            )
        )
        with pytest.raises(ValidationError, match="mixed currencies"):
            forecast("4000", mixed)

    def test_a_zero_horizon_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one period"):
            forecast("4000", history("100"), periods=0)


class TestForecastProvenance:
    def test_a_projection_is_an_estimate_not_a_derived_fact(self) -> None:
        metrics = forecast("4000", history("100", "110", "120")).to_metrics()
        assert all(metric.provenance.kind is AssertionKind.ESTIMATE for metric in metrics)

    def test_the_method_travels_with_the_number(self) -> None:
        metric = forecast("4000", history("100", "110", "120")).to_metrics()[0]
        assert "run rate" in metric.provenance.assumptions["method"]

    def test_the_assumptions_say_what_the_method_implies(self) -> None:
        """So a reader knows a run rate assumes nothing changes."""
        result = forecast("4000", history("100", "110", "120"))
        assert "no growth" in result.assumptions["implies"]


class TestBurnAndRunway:
    def test_burn_is_the_average_monthly_drawdown(self) -> None:
        """Hand: (1,500,000 - 1,200,000) / 3 = 100,000 per month."""
        burn = BurnRate.from_balances(Money.of("1500000"), Money.of("1200000"), 3)

        assert burn.net_burn_per_month.amount == Decimal("100000.000000")
        assert burn.posture is CashPosture.BURNING

    def test_runway_is_cash_over_burn(self) -> None:
        """Hand: 1,200,000 / 100,000 = 12.0 months."""
        burn = BurnRate.from_balances(Money.of("1500000"), Money.of("1200000"), 3)
        assert runway(Money.of("1200000"), burn).months == Decimal("12.0")

    def test_a_cash_generating_business_has_no_runway_figure(self) -> None:
        """Not infinity, and not a very large number standing in for it."""
        burn = BurnRate.from_balances(Money.of("1000000"), Money.of("1300000"), 3)
        result = runway(Money.of("1300000"), burn)

        assert result.months is None
        assert result.posture is CashPosture.GENERATING
        assert "generated cash" in (result.unavailable_reason or "")

    def test_break_even_is_distinguished_from_generating(self) -> None:
        """ "We are not burning" and "we are building cash" are different facts."""
        burn = BurnRate.from_balances(Money.of("1000000"), Money.of("1000000"), 3)
        result = runway(Money.of("1000000"), burn)

        assert result.posture is CashPosture.BREAK_EVEN
        assert result.months is None

    def test_an_absent_runway_is_an_unavailable_metric_not_a_zero(self) -> None:
        burn = BurnRate.from_balances(Money.of("1000000"), Money.of("1300000"), 3)
        metric = runway(Money.of("1300000"), burn).to_metric()

        assert not metric.is_available
        assert metric.unavailable_reason

    def test_a_runway_figure_is_an_estimate(self) -> None:
        burn = BurnRate.from_balances(Money.of("1500000"), Money.of("1200000"), 3)
        metric = runway(Money.of("1200000"), burn).to_metric()

        assert metric.provenance.kind is AssertionKind.ESTIMATE
        assert "continues" in metric.provenance.assumptions["burn"]

    def test_the_description_cannot_be_misread_as_a_measurement(self) -> None:
        burn = BurnRate.from_balances(Money.of("1500000"), Money.of("1200000"), 3)
        assert "estimate" in runway(Money.of("1200000"), burn).describe()

    def test_a_negative_cash_balance_is_refused(self) -> None:
        burn = BurnRate.from_balances(Money.of("100"), Money.of("50"), 1)
        with pytest.raises(ValidationError, match="negative cash balance"):
            runway(Money.of("-1"), burn)

    def test_a_zero_length_window_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one month"):
            BurnRate.from_balances(Money.of("100"), Money.of("50"), 0)


class TestScenarios:
    def test_a_multiplier_scales_the_targeted_account(self) -> None:
        """Hand: revenue 4,000,000 x 0.8 = 3,200,000, so the total falls 800,000."""
        scenario = Scenario(
            name="downside",
            drivers=[
                Driver(
                    scope=DriverScope.ACCOUNT,
                    target="4000",
                    multiplier=Decimal("0.8"),
                    rationale="pipeline coverage below plan",
                )
            ],
        )

        result = apply_scenario(budget(), scenario)

        assert result.change == Money.of("-800000.00")
        assert result.changed_line_count == 1

    def test_a_cost_centre_driver_hits_every_line_in_it(self) -> None:
        scenario = Scenario(
            name="eng-freeze",
            drivers=[
                Driver(
                    scope=DriverScope.COST_CENTRE,
                    target="cc-eng",
                    multiplier=Decimal("0.9"),
                    rationale="hiring freeze",
                )
            ],
        )

        result = apply_scenario(budget(), scenario)

        # cc-eng holds cogs 1,200,000 and salaries 1,800,000 -> 3,000,000 x 0.9.
        assert result.change == Money.of("-300000.00")
        assert result.changed_line_count == 2

    def test_the_baseline_is_never_mutated(self) -> None:
        original = budget()
        before = original.total()

        apply_scenario(
            original,
            Scenario(
                name="x",
                drivers=[Driver(scope=DriverScope.ALL, multiplier=Decimal("0"), rationale="test")],
            ),
        )

        assert original.total() == before

    def test_a_scenario_plan_is_a_reforecast_carrying_an_estimate(self) -> None:
        """Stored as a budget it would eventually be compared against actuals as
        though someone had approved it."""
        result = apply_scenario(
            budget(),
            Scenario(
                name="x",
                drivers=[
                    Driver(scope=DriverScope.ALL, multiplier=Decimal("1.1"), rationale="upside")
                ],
            ),
        )

        assert result.plan.kind.value == "reforecast"
        assert result.plan.provenance.kind is AssertionKind.ESTIMATE
        assert result.plan.provenance.assumptions

    def test_a_driver_naming_an_unknown_target_is_refused(self) -> None:
        """A mistyped cost centre would otherwise look like a scenario with no impact."""
        scenario = Scenario(
            name="typo",
            drivers=[
                Driver(
                    scope=DriverScope.COST_CENTRE,
                    target="cc-enginering",
                    multiplier=Decimal("0.5"),
                    rationale="typo",
                )
            ],
        )

        with pytest.raises(ValidationError, match="cost centre that is not in the baseline"):
            apply_scenario(budget(), scenario)

    def test_a_driver_must_state_a_change(self) -> None:
        with pytest.raises(Exception, match="multiplier or a delta"):
            Driver(scope=DriverScope.ALL, rationale="nothing")

    def test_a_driver_cannot_state_both(self) -> None:
        """Applying both requires an order nobody specified."""
        with pytest.raises(Exception, match="not both"):
            Driver(
                scope=DriverScope.ALL,
                multiplier=Decimal("0.9"),
                delta=Decimal("-100"),
                rationale="both",
            )

    def test_a_driver_must_explain_itself(self) -> None:
        with pytest.raises(PydanticValidationError):
            Driver(scope=DriverScope.ALL, multiplier=Decimal("0.9"), rationale="")

    def test_comparison_orders_by_total_without_picking_a_winner(self) -> None:
        """Which scenario is best depends on risk appetite, which is not in this
        data."""
        cheap = apply_scenario(
            budget(),
            Scenario(
                name="cheap",
                drivers=[Driver(scope=DriverScope.ALL, multiplier=Decimal("0.5"), rationale="cut")],
            ),
        )
        dear = apply_scenario(
            budget(),
            Scenario(
                name="dear",
                drivers=[
                    Driver(scope=DriverScope.ALL, multiplier=Decimal("1.5"), rationale="grow")
                ],
            ),
        )

        ordered = compare_scenarios([dear, cheap])
        assert [result.scenario_name for result in ordered] == ["cheap", "dear"]
