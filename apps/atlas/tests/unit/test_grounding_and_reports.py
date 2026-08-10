"""Numeric grounding, and the report layer that depends on it.

The property under test is the one the whole product rests on: a language model
may explain Atlas's figures, and may not invent one. AI behaviour is never
asserted by matching text — what is asserted is that an invented number is
caught, and that a report carrying one is not marked publishable.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from atlas.analysis.ratios import analyze
from atlas.research.grounding import (
    NumericKind,
    check_numeric_grounding,
    extract_numeric_claims,
)
from atlas.research.reports import ReportRequest, ReportService
from conftest import make_json_response, make_router, source, statements
from fie_ai.contracts import CompletionResponse, StopReason
from fie_common.errors import AIProviderResponseError, ValidationError
from fie_schemas.provenance import AssertionKind

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


def _draft(summary: str, **overrides: object) -> dict[str, object]:
    return {
        "summary": summary,
        "sections": overrides.get("sections", []),
        "cited_source_ids": overrides.get("cited_source_ids", ["fil-acme-2025"]),
        "open_questions": overrides.get("open_questions", []),
    }


def _request() -> ReportRequest:
    data = statements()
    return ReportRequest(
        entity_id="ent-acme",
        company_name="Acme Robotics Corporation",
        statements=data,
        analysis=analyze(data),
        sources=[source()],
    )


class TestReportService:
    async def test_a_faithful_report_is_publishable(self) -> None:
        service = ReportService(
            make_router(
                [
                    make_json_response(
                        _draft("Net margin was 23.4% on revenue of $412.6 million [fil-acme-2025].")
                    )
                ]
            )
        )

        report = await service.generate(_request())

        assert report.numerically_grounded
        assert report.citations_grounded
        assert report.is_publishable
        assert report.cited_source_ids == ["fil-acme-2025"]
        assert report.provenance.kind is AssertionKind.GENERATED

    async def test_the_report_carries_the_computed_metrics(self) -> None:
        service = ReportService(make_router([make_json_response(_draft("Margins held."))]))
        report = await service.generate(_request())

        names = {metric.name for metric in report.metrics}
        assert "net_margin" in names
        assert all(metric.provenance.model is None for metric in report.metrics)

    async def test_an_invented_figure_triggers_one_regeneration(self) -> None:
        """The retry names the offending figure, so it is corrective."""
        service = ReportService(
            make_router(
                [
                    make_json_response(_draft("Operating margin reached 31.2%.")),
                    make_json_response(_draft("Net margin was 23.4%.")),
                ]
            )
        )

        report = await service.generate(_request())

        assert report.regeneration_count == 1
        assert report.numerically_grounded
        assert report.is_publishable

    async def test_a_persistently_ungrounded_report_is_returned_but_not_publishable(
        self,
    ) -> None:
        """Discarding it would hide a systematic prompt or model problem."""
        invented = make_json_response(_draft("Free cash flow was $88 million."))
        service = ReportService(make_router([invented, invented]))

        report = await service.generate(_request())

        assert not report.numerically_grounded
        assert not report.is_publishable
        assert "$88 million" in report.unsupported_figures

    async def test_a_fabricated_citation_is_stripped(self) -> None:
        service = ReportService(
            make_router(
                [
                    make_json_response(
                        _draft(
                            "Net margin was 23.4%.",
                            cited_source_ids=["fil-acme-2025", "fil-does-not-exist"],
                        )
                    )
                ]
            )
        )

        report = await service.generate(_request())

        assert report.cited_source_ids == ["fil-acme-2025"]
        assert not report.citations_grounded
        assert not report.is_publishable

    async def test_figures_in_sections_are_checked_not_only_the_summary(self) -> None:
        service = ReportService(
            make_router(
                [
                    make_json_response(
                        _draft(
                            "Results were solid.",
                            sections=[{"heading": "Margins", "body": "Gross margin hit 55.9%."}],
                        )
                    ),
                ]
            ),
            max_regenerations=0,
        )

        report = await service.generate(_request())

        assert not report.numerically_grounded
        assert "55.9%" in report.unsupported_figures

    async def test_a_report_cannot_be_written_without_computed_figures(self) -> None:
        """A narrative with no deterministic inputs would be entirely model-authored."""
        data = statements()
        empty = analyze(data).model_copy(update={"metrics": []})
        request = ReportRequest(
            entity_id="ent-acme",
            company_name="Acme",
            statements=data,
            analysis=empty,
            sources=[source()],
        )

        with pytest.raises(ValidationError, match="no computed figures"):
            await ReportService(make_router([])).generate(request)

    async def test_a_refusal_surfaces_rather_than_becoming_an_empty_report(self) -> None:
        refusal = CompletionResponse(
            text="",
            model="fake-model-1",
            provider="fake",
            stop_reason=StopReason.REFUSAL,
            refusal_category="policy",
        )
        with pytest.raises(AIProviderResponseError):
            await ReportService(make_router([refusal])).generate(_request())

    async def test_the_prompt_states_the_rule_and_supplies_the_figures(self) -> None:
        service = ReportService(make_router([make_json_response(_draft("Fine."))]))
        prompt = service._build_prompt(_request())

        assert "only numbers you may use" in prompt.lower()
        assert "net_margin" in prompt

    async def test_the_prompt_names_what_the_filing_did_not_disclose(self) -> None:
        """So the model writes "not disclosed" instead of estimating a figure."""
        data = statements(balance_sheet=None)
        request = ReportRequest(
            entity_id="ent-acme",
            company_name="Acme",
            statements=data,
            analysis=analyze(data),
            sources=[source()],
        )
        service = ReportService(make_router([make_json_response(_draft("Fine."))]))

        prompt = service._build_prompt(request)

        assert "Not available" in prompt
        assert "balance sheet" in prompt
