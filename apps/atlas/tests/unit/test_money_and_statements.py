"""Money precision and the accounting identities.

These are the load-bearing invariants of the whole product. A float that slips
into a valuation, or a balance sheet that does not balance, produces output that
is wrong in a way no downstream check can recover.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from atlas_fixtures import annual_period, balance_sheet, fact, income_statement, instant_period
from pydantic import ValidationError as PydanticValidationError

from atlas.domain.money import Money, Rate, Shares, sum_money, to_decimal
from atlas.domain.periods import FiscalPeriod, PeriodKind
from atlas.domain.statements import BalanceSheet, CashFlowStatement, FinancialStatements

pytestmark = pytest.mark.unit


class TestDecimalDiscipline:
    def test_float_is_refused(self) -> None:
        """0.1 + 0.2 != 0.3 in binary, and a DCF compounds that over ten years."""
        with pytest.raises(TypeError, match="float is not accepted"):
            to_decimal(0.1)  # type: ignore[arg-type]
        with pytest.raises(PydanticValidationError):
            Money(amount=1234.56, currency="USD")  # type: ignore[arg-type]

    def test_booleans_are_refused(self) -> None:
        with pytest.raises(TypeError):
            to_decimal(True)  # type: ignore[arg-type]

    def test_strings_and_ints_and_decimals_are_accepted(self) -> None:
        assert Money.of("1234.56").amount == Decimal("1234.56")
        assert Money.of(1234).amount == Decimal(1234)
        assert Money.of(Decimal("1234.56")).amount == Decimal("1234.56")

    def test_addition_is_exact(self) -> None:
        """The canonical float failure, which must not occur here."""
        total = Money.of("0.1") + Money.of("0.2")
        assert total.amount == Decimal("0.3")

    def test_infinities_and_nan_are_refused(self) -> None:
        for value in ("Infinity", "-Infinity", "NaN"):
            with pytest.raises(ValueError, match="finite"):
                to_decimal(value)

    def test_malformed_numbers_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not a valid decimal"):
            to_decimal("about a million")


class TestCurrencySafety:
    def test_mixing_currencies_raises(self) -> None:
        """A total that silently mixes currencies is worse than an error."""
        with pytest.raises(ValueError, match="cannot combine"):
            Money.of("100", "USD") + Money.of("100", "EUR")

    def test_comparison_across_currencies_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot combine"):
            _ = Money.of("100", "USD") < Money.of("100", "EUR")

    def test_currency_is_normalised_and_validated(self) -> None:
        assert Money.of("1", "usd").currency == "USD"
        with pytest.raises(PydanticValidationError):
            Money.of("1", "US")

    def test_sum_requires_an_explicit_currency_when_empty(self) -> None:
        assert sum_money([], currency="EUR").currency == "EUR"


class TestMoneyArithmetic:
    def test_division_by_zero_raises_rather_than_returning_infinity(self) -> None:
        with pytest.raises(ZeroDivisionError):
            Money.of("100").divided_by(0)
        with pytest.raises(ZeroDivisionError):
            Money.of("100").ratio_to(Money.zero())

    def test_scale_helpers_convert_reporting_units(self) -> None:
        assert Money.millions("412.6").amount == Decimal("412600000.0")
        assert Money.thousands("1234").amount == Decimal("1234000")

    def test_rounding_is_half_up_as_a_reader_expects(self) -> None:
        assert Money.of("2.675").rounded().amount == Decimal("2.68")
        assert Money.of("2.665").rounded().amount == Decimal("2.67")

    def test_shares_must_be_positive(self) -> None:
        with pytest.raises(PydanticValidationError, match="must be positive"):
            Shares.of(0)

    def test_rate_converts_percentage_points(self) -> None:
        assert Rate.from_percent("8.5").value == Decimal("0.085")
        assert Rate.from_percent("8.5").as_percent == Decimal("8.50")


class TestPeriods:
    def test_a_quarter_must_say_which_quarter(self) -> None:
        with pytest.raises(PydanticValidationError, match="which quarter"):
            FiscalPeriod(kind=PeriodKind.QUARTERLY, fiscal_year=2025, end_date=date(2025, 3, 31))

    def test_an_instant_has_no_span(self) -> None:
        with pytest.raises(PydanticValidationError, match="no span"):
            FiscalPeriod(
                kind=PeriodKind.INSTANT,
                fiscal_year=2025,
                start_date=date(2025, 1, 1),
                end_date=date(2025, 12, 31),
            )

    def test_a_quarter_is_not_comparable_to_a_year(self) -> None:
        """The most common source of a confidently wrong growth rate."""
        annual = FiscalPeriod.annual(2025, date(2025, 12, 31))
        quarter = FiscalPeriod.quarterly(2025, 4, date(2025, 12, 31))
        assert not annual.is_comparable_to(quarter)

    def test_only_the_same_quarter_is_comparable(self) -> None:
        q3_2025 = FiscalPeriod.quarterly(2025, 3, date(2025, 9, 30))
        q3_2024 = FiscalPeriod.quarterly(2024, 3, date(2024, 9, 30))
        q2_2025 = FiscalPeriod.quarterly(2025, 2, date(2025, 6, 30))
        assert q3_2025.is_comparable_to(q3_2024)
        assert not q3_2025.is_comparable_to(q2_2025)

    def test_labels_read_naturally(self) -> None:
        assert FiscalPeriod.annual(2025, date(2025, 12, 31)).label == "FY2025"
        assert FiscalPeriod.quarterly(2025, 3, date(2025, 9, 30)).label == "Q3 FY2025"

    def test_only_flows_accumulate(self) -> None:
        assert PeriodKind.ANNUAL.is_flow
        assert not PeriodKind.INSTANT.is_flow


class TestAccountingIdentities:
    def test_a_balance_sheet_must_balance(self) -> None:
        """True of every balance sheet ever published, so a violation is an error."""
        with pytest.raises(PydanticValidationError, match="does not balance"):
            balance_sheet(
                total_assets="800000000", total_liabilities="480000000", equity="200000000"
            )

    def test_reporting_rounding_is_tolerated(self) -> None:
        """Summing rounded components leaves a residue; that is not an error."""
        sheet = balance_sheet(
            total_assets="800000000", total_liabilities="480000000", equity="320000001"
        )
        assert sheet.total_assets.amount == Decimal("800000000")

    def test_gross_profit_must_reconcile(self) -> None:
        with pytest.raises(PydanticValidationError, match="gross profit does not reconcile"):
            income_statement(revenue="412600000", cost_of_revenue="247560000").model_copy(
                update={"gross_profit": Money.of("999999999")}
            ).model_validate(
                income_statement(revenue="412600000", cost_of_revenue="247560000").model_dump()
                | {"gross_profit": {"amount": Decimal("999999999"), "currency": "USD"}}
            )

    def test_net_income_must_reconcile_with_tax(self) -> None:
        with pytest.raises(PydanticValidationError, match="net income does not reconcile"):
            income_statement(net_income="1")

    def test_a_balance_sheet_rejects_a_span_period(self) -> None:
        with pytest.raises(PydanticValidationError, match="position at an instant"):
            BalanceSheet(
                period=annual_period(),
                provenance=fact(),
                total_assets=Money.of("100"),
                total_liabilities=Money.of("60"),
                shareholders_equity=Money.of("40"),
            )

    def test_an_income_statement_rejects_an_instant_period(self) -> None:
        with pytest.raises(PydanticValidationError, match="covers a span"):
            income_statement().model_copy(update={"period": instant_period()}).model_validate(
                income_statement().model_dump() | {"period": instant_period().model_dump()}
            )

    def test_current_assets_cannot_exceed_total(self) -> None:
        with pytest.raises(PydanticValidationError, match="current assets cannot exceed"):
            BalanceSheet(
                period=instant_period(),
                provenance=fact(),
                total_assets=Money.of("100"),
                current_assets=Money.of("150"),
                total_liabilities=Money.of("60"),
                shareholders_equity=Money.of("40"),
            )

    def test_cash_flow_must_reconcile(self) -> None:
        with pytest.raises(PydanticValidationError, match="does not reconcile"):
            CashFlowStatement(
                period=annual_period(),
                provenance=fact(),
                operating_cash_flow=Money.of("140000000"),
                investing_cash_flow=Money.of("-60000000"),
                financing_cash_flow=Money.of("-30000000"),
                net_change_in_cash=Money.of("999999999"),
            )

    def test_net_debt_needs_both_components(self) -> None:
        sheet = balance_sheet()
        assert sheet.net_debt == Money.of("80000000")


class TestStatementSet:
    def test_periods_must_agree(self) -> None:
        with pytest.raises(PydanticValidationError, match="must cover the set's period"):
            FinancialStatements(
                entity_id="ent-acme",
                period=annual_period(2025),
                income_statement=income_statement(year=2024),
            )

    def test_currencies_must_agree(self) -> None:
        foreign = income_statement().model_dump() | {"currency": "EUR"}
        with pytest.raises(PydanticValidationError):
            FinancialStatements(
                entity_id="ent-acme",
                period=annual_period(),
                currency="USD",
                income_statement=foreign,  # type: ignore[arg-type]
            )

    def test_at_least_one_statement_is_required(self) -> None:
        with pytest.raises(PydanticValidationError, match="at least one statement"):
            FinancialStatements(entity_id="ent-acme", period=annual_period())

    def test_sources_are_collected_without_duplicates(self, acme_statements) -> None:
        assert acme_statements.source_ids == ["fil-acme-2025"]
        assert len(acme_statements.all_sources()) == 1

    def test_every_statement_requires_provenance(self) -> None:
        """A figure nobody can cite cannot appear in a research report."""
        with pytest.raises(PydanticValidationError):
            BalanceSheet(  # type: ignore[call-arg]
                period=instant_period(),
                total_assets=Money.of("100"),
                total_liabilities=Money.of("60"),
                shareholders_equity=Money.of("40"),
            )
