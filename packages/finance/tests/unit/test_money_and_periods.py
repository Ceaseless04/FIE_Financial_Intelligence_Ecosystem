"""Money precision and fiscal period comparability.

These are the load-bearing invariants of every financial product built on this
platform. A float that slips into a valuation, a budget that silently mixes
currencies, or a quarter compared against a year all produce output that is
wrong in a way no downstream check can recover.

They live with the package rather than with any one application, because the
package is what the applications now share.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from fie_finance.money import Money, Rate, Shares, sum_money, to_decimal
from fie_finance.periods import FiscalPeriod, PeriodKind, period_key

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


class TestPeriodIdentity:
    """`period_key` is storage's idea of which period a row describes."""

    def test_an_annual_period_carries_an_explicit_zero(self) -> None:
        """PostgreSQL treats every NULL as distinct, so a nullable quarter in a
        unique constraint would not constrain annual periods at all."""
        assert FiscalPeriod.annual(2025, date(2025, 12, 31)).key == "annual:2025:0"

    def test_a_quarter_is_distinct_from_its_year(self) -> None:
        annual = FiscalPeriod.annual(2025, date(2025, 12, 31))
        quarter = FiscalPeriod.quarterly(2025, 4, date(2025, 12, 31))
        assert annual.key != quarter.key

    def test_lookup_by_parts_matches_lookup_by_object(self) -> None:
        """One definition, so a query and a write cannot disagree about a row."""
        period = FiscalPeriod.quarterly(2025, 3, date(2025, 9, 30))
        assert period_key(PeriodKind.QUARTERLY, 2025, 3) == period.key
