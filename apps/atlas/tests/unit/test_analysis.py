"""The deterministic analysis layer.

Two things are asserted throughout: the arithmetic is right, and no computed
figure can be attributed to a language model. The second is the architectural
constraint — a DERIVED provenance that names a model will not construct, so
these tests fail loudly if a calculation is ever routed through a provider.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from atlas_fixtures import balance_sheet, income_statement, statements

from atlas.analysis import ratios
from atlas.analysis.comparables import (
    MarketData,
    PeerMultiple,
    enterprise_value_to_ebitda,
    price_to_earnings,
    value_from_peers,
)
from atlas.analysis.dcf import (
    DCFAssumptions,
    DCFModel,
    project_cash_flows,
    sensitivity_grid,
    weighted_average_cost_of_capital,
)
from atlas.analysis.growth import (
    compound_annual_growth_rate,
    free_cash_flow,
    period_over_period_growth,
    revenue_growth,
)
from atlas.analysis.results import Unit
from fie_common.errors import ValidationError
from fie_finance.money import Money, Rate, Shares
from fie_finance.periods import FiscalPeriod
from fie_schemas.provenance import AssertionKind

pytestmark = pytest.mark.unit


class TestArchitecturalConstraint:
    def test_no_analysis_module_can_reach_an_ai_provider(self) -> None:
        """The rule, checked against the source rather than trusted."""
        import pkgutil
        from pathlib import Path

        import atlas.analysis as package

        offenders: list[str] = []
        for module in pkgutil.iter_modules(package.__path__):
            source = Path(package.__path__[0]) / f"{module.name}.py"
            text = source.read_text(encoding="utf-8")
            if "AIRouter" in text or "CompletionRequest" in text or "fie_ai" in text:
                offenders.append(module.name)
        assert offenders == [], f"analysis modules must not import an AI provider: {offenders}"

    def test_a_computed_metric_cannot_name_a_model(self) -> None:
        metric = ratios.net_margin(income_statement())
        assert metric.provenance.kind is AssertionKind.DERIVED
        assert metric.provenance.model is None
        assert metric.provenance.computation

    def test_a_valuation_is_an_estimate_with_stated_assumptions(self) -> None:
        valuation = DCFModel(
            DCFAssumptions(
                discount_rate=Rate.from_percent("10"),
                terminal_growth_rate=Rate.from_percent("2"),
                projection_years=3,
            )
        ).value("ent-acme", [Money.of("100"), Money.of("110"), Money.of("121")])
        metric = valuation.to_metrics()[0]

        assert metric.provenance.kind is AssertionKind.ESTIMATE
        assert metric.provenance.model is None
        assert "discount_rate" in metric.provenance.assumptions


class TestRatios:
    def test_margins_are_computed_correctly(self) -> None:
        income = income_statement()

        # 165,040,000 / 412,600,000 = 40.0%
        assert ratios.gross_margin(income).value == Decimal("40.0000")
        # 96,548,400 / 412,600,000 = 23.4%
        assert ratios.net_margin(income).value == Decimal("23.4000")
        assert ratios.net_margin(income).unit is Unit.PERCENT

    def test_gross_margin_is_derived_when_not_reported(self) -> None:
        income = income_statement().model_copy(update={"gross_profit": None})
        assert ratios.gross_margin(income).value == Decimal("40.0000")

    def test_return_on_equity_uses_both_statements(self) -> None:
        # 96,548,400 / 320,000,000 = 30.1714%
        metric = ratios.return_on_equity(income_statement(), balance_sheet())
        assert metric.value == Decimal("30.1714")

    def test_ratios_carry_their_inputs_for_recomputation(self) -> None:
        metric = ratios.net_margin(income_statement())
        assert "net_income" in metric.inputs
        assert "revenue" in metric.inputs

    def test_a_missing_input_yields_an_unavailable_metric_not_a_zero(self) -> None:
        """A zero reads as a measurement; "not disclosed" is the truth."""
        sheet = balance_sheet().model_copy(update={"total_debt": None})
        metric = ratios.debt_to_equity(sheet)

        assert not metric.is_available
        assert "not disclosed" in (metric.unavailable_reason or "")

    def test_a_zero_denominator_is_undefined_not_infinite(self) -> None:
        sheet = balance_sheet().model_copy(update={"current_liabilities": Money.zero()})
        metric = ratios.current_ratio(sheet)
        assert not metric.is_available
        assert "undefined" in (metric.unavailable_reason or "")

    def test_interest_coverage_handles_either_sign_convention(self) -> None:
        positive = income_statement()
        negative = income_statement().model_copy(update={"interest_expense": Money.of("-4000000")})
        assert ratios.interest_coverage(positive).value == ratios.interest_coverage(negative).value

    def test_analyze_reports_what_it_could_not_compute(self) -> None:
        without_balance = statements(balance_sheet=None)
        result = ratios.analyze(without_balance)

        assert result.get("net_margin") is not None
        roe = result.get("return_on_equity")
        assert roe is not None and not roe.is_available
        assert any("balance sheet" in (m.unavailable_reason or "") for m in result.unavailable)


class TestDCF:
    def test_matches_a_hand_worked_example(self) -> None:
        """Verified independently: see the docstring arithmetic."""
        flows = [Money.of(x) for x in ("100", "110", "121", "133.1", "146.41")]
        valuation = DCFModel(
            DCFAssumptions(
                discount_rate=Rate.from_percent("10"),
                terminal_growth_rate=Rate.from_percent("2"),
                projection_years=5,
            )
        ).value("ent-acme", flows, net_debt=Money.of("200"), diluted_shares=Shares.of("100"))

        assert valuation.present_value_of_forecast.amount == Decimal("454.53")
        assert valuation.terminal_value.amount == Decimal("1866.73")
        assert valuation.enterprise_value.amount == Decimal("1613.59")
        assert valuation.equity_value is not None
        assert valuation.equity_value.amount == Decimal("1413.59")
        assert valuation.value_per_share is not None
        assert valuation.value_per_share.amount == Decimal("14.1359")

    def test_terminal_growth_at_or_above_the_discount_rate_is_refused(self) -> None:
        """The classic way a DCF produces an unbounded, meaningless number."""
        for growth in ("10", "12", "9.8"):
            with pytest.raises(ValidationError, match="exceed the terminal growth rate"):
                DCFAssumptions(
                    discount_rate=Rate.from_percent("10"),
                    terminal_growth_rate=Rate.from_percent(growth),
                )

    def test_a_negative_discount_rate_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="discount rate must be positive"):
            DCFAssumptions(
                discount_rate=Rate.from_percent("-5"),
                terminal_growth_rate=Rate.from_percent("-10"),
            )

    def test_cash_flows_must_match_the_horizon(self) -> None:
        model = DCFModel(
            DCFAssumptions(
                discount_rate=Rate.from_percent("10"),
                terminal_growth_rate=Rate.from_percent("2"),
                projection_years=5,
            )
        )
        with pytest.raises(ValidationError, match="must match the projection horizon"):
            model.value("ent-acme", [Money.of("100")])

    def test_mixed_currencies_are_refused(self) -> None:
        model = DCFModel(
            DCFAssumptions(
                discount_rate=Rate.from_percent("10"),
                terminal_growth_rate=Rate.from_percent("2"),
                projection_years=2,
            )
        )
        with pytest.raises(ValidationError, match="one currency"):
            model.value("ent-acme", [Money.of("100", "USD"), Money.of("100", "EUR")])

    def test_terminal_dominance_is_surfaced(self) -> None:
        """A valuation that is mostly perpetuity assumption must say so."""
        flows = [Money.of("100")] * 3
        valuation = DCFModel(
            DCFAssumptions(
                discount_rate=Rate.from_percent("8"),
                terminal_growth_rate=Rate.from_percent("6"),
                projection_years=3,
            )
        ).value("ent-acme", flows)
        assert valuation.is_terminal_dominated

    def test_projection_is_reproducible(self) -> None:
        flows = project_cash_flows(Money.of("100"), Rate.from_percent("10"), 3)
        assert [f.amount for f in flows] == [
            Decimal("110.0"),
            Decimal("121.00"),
            Decimal("133.100"),
        ]

    def test_wacc_applies_the_debt_tax_shield(self) -> None:
        # 0.8*0.12 + 0.2*0.06*(1-0.25) = 0.096 + 0.009 = 0.105
        wacc = weighted_average_cost_of_capital(
            equity_value=Money.of("800"),
            debt_value=Money.of("200"),
            cost_of_equity=Rate.from_percent("12"),
            cost_of_debt=Rate.from_percent("6"),
            tax_rate=Rate.from_percent("25"),
        )
        assert wacc.value == Decimal("0.105000")

    def test_sensitivity_marks_invalid_combinations_rather_than_hiding_them(self) -> None:
        grid = sensitivity_grid(
            "ent-acme",
            [Money.of("100")] * 3,
            discount_rates=[Rate.from_percent("8"), Rate.from_percent("10")],
            terminal_growth_rates=[Rate.from_percent("2"), Rate.from_percent("9")],
            projection_years=3,
        )
        assert grid[str(Rate.from_percent("8"))][str(Rate.from_percent("9"))] == "n/a"
        assert grid[str(Rate.from_percent("10"))][str(Rate.from_percent("2"))] != "n/a"


class TestGrowth:
    def test_growth_between_comparable_periods(self) -> None:
        metric = revenue_growth(statements(year=2025), statements(year=2024))
        assert metric.is_available
        assert metric.value == Decimal("0.0000")

    def test_incomparable_periods_refuse_to_produce_a_rate(self) -> None:
        """A quarter against a year is the classic misleading growth number."""
        annual = FiscalPeriod.annual(2025, date(2025, 12, 31))
        quarter = FiscalPeriod.quarterly(2024, 4, date(2024, 12, 31))

        metric = period_over_period_growth(
            "revenue",
            Money.of("400"),
            Money.of("100"),
            current_period=annual,
            prior_period=quarter,
        )
        assert not metric.is_available
        assert "not comparable" in (metric.unavailable_reason or "")

    def test_growth_from_a_negative_base_is_refused(self) -> None:
        current = FiscalPeriod.annual(2025, date(2025, 12, 31))
        prior = FiscalPeriod.annual(2024, date(2024, 12, 31))
        metric = period_over_period_growth(
            "net_income",
            Money.of("50"),
            Money.of("-100"),
            current_period=current,
            prior_period=prior,
        )
        assert not metric.is_available
        assert "negative" in (metric.unavailable_reason or "")

    def test_cagr_is_correct(self) -> None:
        # 100 -> 200 over 5 years is 14.8698%
        metric = compound_annual_growth_rate("revenue", Money.of("100"), Money.of("200"), years=5)
        assert metric.value == Decimal("14.8698")

    def test_cagr_requires_positive_endpoints(self) -> None:
        metric = compound_annual_growth_rate("revenue", Money.of("-100"), Money.of("200"), years=5)
        assert not metric.is_available

    def test_free_cash_flow_names_its_definition(self) -> None:
        metric = free_cash_flow(statements())
        assert metric.value == Decimal("95000000")
        assert "definition" in metric.inputs


class TestComparables:
    def test_price_to_earnings(self) -> None:
        market = MarketData(share_price=Money.of("50"), diluted_shares=Shares.of("100000000"))
        metric = price_to_earnings(market, Money.of("96548400"))
        # 5,000,000,000 / 96,548,400 = 51.78749726..., which rounds to 51.7875
        assert metric.value == Decimal("51.7875")

    def test_a_loss_making_company_has_no_pe(self) -> None:
        """A negative P/E is not a cheap stock; printing one invites misreading."""
        market = MarketData(share_price=Money.of("50"), diluted_shares=Shares.of("100000000"))
        metric = price_to_earnings(market, Money.of("-10000000"))
        assert not metric.is_available
        assert "does not apply" in (metric.unavailable_reason or "")

    def test_ev_to_ebitda_needs_net_debt(self) -> None:
        market = MarketData(share_price=Money.of("50"), diluted_shares=Shares.of("100000000"))
        metric = enterprise_value_to_ebitda(market, Money.of("200000000"))
        assert not metric.is_available

    def test_peer_valuation_uses_the_median(self) -> None:
        """One peer at 200x must not drag the whole comparison."""
        peers = [
            PeerMultiple(entity_id="a", name="A", multiple=Decimal("10"), basis="ebitda"),
            PeerMultiple(entity_id="b", name="B", multiple=Decimal("12"), basis="ebitda"),
            PeerMultiple(entity_id="c", name="C", multiple=Decimal("200"), basis="ebitda"),
        ]
        valuation = value_from_peers(
            "ent-acme", peers=peers, metric_value=Money.of("100000000"), basis="ebitda"
        )
        assert valuation.median_multiple == Decimal("12")
        assert valuation.implied_value.amount == Decimal("1200000000.00")

    def test_mixing_bases_is_refused(self) -> None:
        peers = [
            PeerMultiple(entity_id="a", name="A", multiple=Decimal("10"), basis="ebitda"),
            PeerMultiple(entity_id="b", name="B", multiple=Decimal("20"), basis="earnings"),
        ]
        with pytest.raises(ValidationError, match="same basis"):
            value_from_peers("ent-acme", peers=peers, metric_value=Money.of("100"), basis="ebitda")

    def test_an_empty_peer_set_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one peer"):
            value_from_peers("ent-acme", peers=[], metric_value=Money.of("100"), basis="ebitda")

    def test_a_thin_peer_set_is_flagged(self) -> None:
        peers = [PeerMultiple(entity_id="a", name="A", multiple=Decimal("10"), basis="ebitda")]
        valuation = value_from_peers(
            "ent-acme", peers=peers, metric_value=Money.of("100"), basis="ebitda"
        )
        assert valuation.peer_set_is_thin
