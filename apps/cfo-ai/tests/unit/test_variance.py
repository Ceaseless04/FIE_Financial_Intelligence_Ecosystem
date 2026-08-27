"""Variance, and the direction that is the whole point.

The arithmetic in this module is a subtraction, and none of these tests are
really about the subtraction. They are about the sentence that follows it: the
same number means opposite things on different accounts, and a system that gets
that wrong produces a report which reads perfectly and says the reverse of the
truth.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from cfo_fixtures import CHART, HEADCOUNT, MARKETING, REVENUE, annual, tree

from cfo_ai.analysis.compare import compare, roll_up, unassigned_spend
from cfo_ai.analysis.variance import Favourability, Variance
from cfo_ai.domain.plan import Plan, PlanKind, line
from fie_common.errors import ValidationError
from fie_finance.money import Money
from fie_schemas.provenance import AssertionKind

pytestmark = pytest.mark.unit

PERIOD = annual()


def variance_of(account, budget: str, actual: str, **kwargs) -> Variance:
    return Variance.between(
        account, "cc-sales", PERIOD, Money.of(budget), Money.of(actual), **kwargs
    )


class TestDirection:
    def test_the_same_shortfall_is_bad_on_revenue_and_good_on_cost(self) -> None:
        """The load-bearing test of this phase.

        Identical inputs, identical subtraction, identical percentage — and
        opposite verdicts, decided entirely by the account.
        """
        revenue = variance_of(REVENUE, "400000", "380000")
        marketing = variance_of(MARKETING, "400000", "380000")

        assert revenue.amount == marketing.amount == Money.of("-20000")
        assert revenue.percent == marketing.percent == Decimal("-5.00")
        assert revenue.favourability is Favourability.UNFAVOURABLE
        assert marketing.favourability is Favourability.FAVOURABLE

    def test_overspend_and_overachievement_also_diverge(self) -> None:
        assert variance_of(REVENUE, "400000", "420000").favourability is Favourability.FAVOURABLE
        assert (
            variance_of(MARKETING, "400000", "420000").favourability is Favourability.UNFAVOURABLE
        )

    def test_exactly_on_plan_is_neither(self) -> None:
        assert variance_of(REVENUE, "400000", "400000").favourability is Favourability.ON_PLAN

    def test_headcount_gets_no_verdict(self) -> None:
        """Hiring behind plan is a saving to a CFO and a capacity problem to the
        director waiting for those people. The system does not know which."""
        under = variance_of(HEADCOUNT, "50", "44")

        assert under.favourability is Favourability.NOT_ASSESSED
        assert not under.favourability.is_judgement
        assert under.amount == Money.of("-6")  # still computed and reported

    def test_overspend_is_narrower_than_unfavourable(self) -> None:
        """A revenue miss is unfavourable and is not an overspend."""
        revenue_miss = variance_of(REVENUE, "400000", "380000")
        cost_overrun = variance_of(MARKETING, "400000", "420000")

        assert revenue_miss.favourability is Favourability.UNFAVOURABLE
        assert not revenue_miss.is_overspend
        assert cost_overrun.is_overspend

    def test_a_verdict_cannot_be_supplied_by_a_caller(self) -> None:
        """`between` derives it; there is no parameter to get wrong."""
        import inspect

        parameters = inspect.signature(Variance.between).parameters
        assert "favourability" not in parameters


class TestPercentAndMateriality:
    def test_a_zero_plan_yields_no_percentage(self) -> None:
        """Not infinity, and not a very large number standing in for it."""
        unplanned = variance_of(MARKETING, "0", "50000")

        assert unplanned.percent is None
        assert unplanned.favourability is Favourability.UNFAVOURABLE

    def test_unplanned_spend_is_still_material_on_size(self) -> None:
        """Judged on amount alone, rather than dropped for want of a percentage."""
        assert variance_of(MARKETING, "0", "50000").is_material

    def test_a_large_percentage_of_a_tiny_line_is_not_material(self) -> None:
        """Otherwise 100% over on a $12 line ranks beside a $2m overspend."""
        assert not variance_of(MARKETING, "100", "200").is_material

    def test_a_small_percentage_of_a_huge_line_is_not_material_either(self) -> None:
        assert not variance_of(REVENUE, "10000000", "9900000").is_material

    def test_both_thresholds_together_make_a_line_material(self) -> None:
        assert variance_of(REVENUE, "400000", "300000").is_material

    def test_thresholds_are_configurable(self) -> None:
        strict = variance_of(
            REVENUE,
            "400000",
            "390000",
            material_percent=Decimal("1"),
            material_amount=Decimal("1000"),
        )
        assert strict.is_material


class TestCurrencySafety:
    def test_comparing_across_currencies_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot combine"):
            Variance.between(
                REVENUE, "cc-sales", PERIOD, Money.of("100", "USD"), Money.of("100", "EUR")
            )


class TestProvenance:
    def test_a_variance_metric_is_derived_and_names_no_model(self) -> None:
        metric = variance_of(REVENUE, "400000", "380000").to_metric()

        assert metric.provenance.kind is AssertionKind.DERIVED
        assert metric.provenance.model is None
        assert metric.provenance.computation == "cfo_ai.variance.v1"

    def test_the_metric_carries_the_inputs_and_the_verdict(self) -> None:
        metric = variance_of(MARKETING, "600000", "700000").to_metric()

        assert metric.inputs["budget"] == "600000"
        assert metric.inputs["actual"] == "700000"
        assert metric.inputs["favourability"] == "unfavourable"

    def test_describe_spells_the_direction_out(self) -> None:
        """The prompt says what a variance means rather than leaving a model to
        infer it from a sign — inferring it is the mistake being prevented."""
        text = variance_of(MARKETING, "600000", "700000").describe()

        assert "over" in text
        assert "unfavourable" in text

    def test_describe_withholds_a_verdict_it_does_not_have(self) -> None:
        assert "not assessed" in variance_of(HEADCOUNT, "50", "44").describe()


class TestComparison:
    def test_a_report_separates_the_two_directions(self, fy26_budget, fy26_actuals) -> None:
        report = compare(fy26_budget, fy26_actuals, CHART)

        names = {variance.account_code: variance.favourability for variance in report.variances}
        # Revenue 4.0m planned, 3.6m actual -> miss. Marketing 600k -> 700k -> overspend.
        assert names["4000"] is Favourability.UNFAVOURABLE
        assert names["6100"] is Favourability.UNFAVOURABLE
        # COGS and salaries came in under plan.
        assert names["5000"] is Favourability.FAVOURABLE
        assert names["6200"] is Favourability.FAVOURABLE

    def test_the_net_operating_variance_nets_the_right_way(self, fy26_budget, fy26_actuals) -> None:
        """Revenue adds, costs subtract. Worked by hand:
        revenue -400,000; cogs -50,000 (a saving, so +50,000);
        marketing +100,000 (a cost, so -100,000); salaries -50,000 (so +50,000).
        Net: -400,000 + 50,000 - 100,000 + 50,000 = -400,000.
        """
        report = compare(fy26_budget, fy26_actuals, CHART)
        assert report.net_operating_variance() == Money.of("-400000")

    def test_material_lines_come_back_worst_money_first(self, fy26_budget, fy26_actuals) -> None:
        report = compare(fy26_budget, fy26_actuals, CHART)
        amounts = [variance.absolute_amount.amount for variance in report.material]

        assert amounts == sorted(amounts, reverse=True)
        assert report.material[0].account_code == "4000"

    def test_an_account_missing_from_the_chart_is_an_error(self, fy26_budget, fy26_actuals) -> None:
        """A variance with no account has no direction, so it is not renderable."""
        partial = {code: account for code, account in CHART.items() if code != "6100"}

        with pytest.raises(ValidationError, match="not in the chart"):
            compare(fy26_budget, fy26_actuals, partial)

    def test_unmatched_lines_are_reported_not_dropped(self, fy26_budget) -> None:
        """An actual with no budget is unplanned spend; a budget with no actual
        may be a forgotten accrual. Both are findings."""
        period = annual()
        sparse = Plan(
            entity_id="ent-northwind",
            kind=PlanKind.ACTUAL,
            period=period,
            lines=[
                line("4000", "cc-sales", period, "3600000"),
                line("6100", "cc-eng", period, "25000"),  # a centre the budget never used
            ],
            provenance=fy26_budget.provenance,
        )

        report = compare(fy26_budget, sparse, CHART)

        assert report.unmatched_budget_keys  # budgeted lines with no actual
        assert any("cc-eng" in key for key in report.unmatched_actual_keys)

    def test_different_periods_cannot_be_compared(self, fy26_budget) -> None:
        with pytest.raises(ValidationError, match="different periods"):
            compare(fy26_budget, __import__("cfo_fixtures").actuals(year=2025), CHART)

    def test_actuals_cannot_stand_in_for_a_budget(self, fy26_actuals) -> None:
        with pytest.raises(ValidationError, match="must be a budget"):
            compare(fy26_actuals, fy26_actuals, CHART)

    def test_a_budget_cannot_stand_in_for_actuals(self, fy26_budget) -> None:
        with pytest.raises(ValidationError, match="must be actuals"):
            compare(fy26_budget, fy26_budget, CHART)


class TestRollup:
    def test_a_parent_totals_everything_beneath_it(self, fy26_budget, fy26_actuals) -> None:
        """Sales owns revenue (4.0m) and marketing (600k) directly; Field Sales
        has nothing. The company root owns all four lines."""
        report = compare(fy26_budget, fy26_actuals, CHART)
        rollups = {
            r.cost_centre_id: r for r in roll_up(report, tree(), (fy26_budget, fy26_actuals))
        }

        assert rollups["cc-sales"].budget == Money.of("4600000")
        assert rollups["cc-co"].budget == Money.of("7600000")
        assert rollups["cc-field"].budget == Money.zero()

    def test_roots_come_before_their_children(self, fy26_budget, fy26_actuals) -> None:
        report = compare(fy26_budget, fy26_actuals, CHART)
        rollups = roll_up(report, tree(), (fy26_budget, fy26_actuals))

        depths = [rollup.depth for rollup in rollups]
        assert depths == sorted(depths)
        assert rollups[0].cost_centre_id == "cc-co"

    def test_a_centre_outside_the_hierarchy_is_surfaced(self, fy26_budget) -> None:
        """Its spend would otherwise vanish from every roll-up silently."""
        period = annual()
        stray = Plan(
            entity_id="ent-northwind",
            kind=PlanKind.BUDGET,
            period=period,
            lines=[line("6100", "cc-newteam", period, "10000")],
            provenance=fy26_budget.provenance,
        )

        assert unassigned_spend(stray, tree()) == ["cc-newteam"]

    def test_a_known_hierarchy_leaves_nothing_unassigned(self, fy26_budget) -> None:
        assert unassigned_spend(fy26_budget, tree()) == []
