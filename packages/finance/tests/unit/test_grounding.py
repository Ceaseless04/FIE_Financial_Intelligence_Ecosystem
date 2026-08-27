"""Numeric grounding — rounding is permitted, invention is not.

A written figure is matched by rounding each computed value to the number of
significant digits the author actually wrote. "$1.6 billion" therefore matches a
computed 1,613,590,000 and "$1.9 billion" matches nothing at any tolerance,
which is both stricter and more explainable than a percentage band.

These live with the package because every product that writes prose about
numbers needs them, not only the one that needed them first.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from fie_finance.grounding import (
    NumericKind,
    check_numeric_grounding,
    extract_numeric_claims,
)

pytestmark = pytest.mark.unit

COMPUTED = {
    Decimal("412600000"),
    Decimal("1613590000"),
    Decimal("23.4"),
    Decimal("12.4"),
}


class TestClaimExtraction:
    @pytest.mark.parametrize(
        ("text", "expected_value", "expected_kind"),
        [
            ("$412.6 million", Decimal("412600000"), NumericKind.MONEY),
            ("1,234.56", Decimal("1234.56"), NumericKind.PLAIN),
            ("23.4%", Decimal("23.4"), NumericKind.PERCENT),
            ("12.4x", Decimal("12.4"), NumericKind.MULTIPLE),
            ("$1.6 billion", Decimal("1600000000"), NumericKind.MONEY),
            ("(1234)", Decimal("-1234"), NumericKind.PLAIN),
        ],
    )
    def test_written_figures_are_parsed(
        self, text: str, expected_value: Decimal, expected_kind: NumericKind
    ) -> None:
        claims = extract_numeric_claims(text)
        assert claims[0].value == expected_value
        assert claims[0].kind is expected_kind

    def test_years_are_not_treated_as_financial_claims(self) -> None:
        claims = extract_numeric_claims("Between 2023 and 2025 revenue grew.")
        assert all(claim.kind is NumericKind.YEAR for claim in claims)
        assert not any(claim.requires_grounding for claim in claims)


class TestNumericGrounding:
    def test_a_faithful_narrative_is_grounded(self) -> None:
        text = (
            "Revenue reached $412.6 million and the net margin was 23.4%. "
            "The enterprise value estimate is $1.6 billion, or 12.4x EBITDA."
        )
        report = check_numeric_grounding(text, COMPUTED)
        assert report.is_grounded
        assert report.total_claims == 4

    def test_an_invented_figure_is_caught(self) -> None:
        """The failure this module exists for: a plausible number nobody computed."""
        text = "Revenue reached $412.6 million and operating margin improved to 31.2%."
        report = check_numeric_grounding(text, COMPUTED)

        assert not report.is_grounded
        assert [figure.text for figure in report.unsupported] == ["31.2%"]

    @pytest.mark.parametrize("written", ["$1.6 billion", "$1.61 billion", "$1.614 billion"])
    def test_rounding_a_computed_value_is_permitted(self, written: str) -> None:
        assert check_numeric_grounding(written, {Decimal("1613590000")}).is_grounded

    @pytest.mark.parametrize("written", ["$1.9 billion", "$1.7 billion", "$16 billion"])
    def test_a_figure_that_is_not_a_rounding_is_refused(self, written: str) -> None:
        assert not check_numeric_grounding(written, {Decimal("1613590000")}).is_grounded

    def test_a_reported_loss_may_be_written_as_a_magnitude(self) -> None:
        report = check_numeric_grounding("a loss of $12.4 million", {Decimal("-12400000")})
        assert report.is_grounded

    def test_grounding_ratio_reports_partial_failure(self) -> None:
        text = "Revenue was $412.6 million, margin 23.4%, and cash flow $88 million."
        report = check_numeric_grounding(text, COMPUTED)
        assert report.grounding_ratio == Decimal("0.6667")

    def test_prose_with_no_figures_is_trivially_grounded(self) -> None:
        report = check_numeric_grounding("Margins improved across the segment.", COMPUTED)
        assert report.is_grounded and report.total_claims == 0
