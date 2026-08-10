"""Unit tests for structured output and grounding checks.

This is the mechanism that makes AI behaviour testable without asserting on
exact model text: the model fills a schema, validation is deterministic, and
citations are verified against the sources actually supplied.
"""

from __future__ import annotations

import pytest
from pydantic import Field

from fie_ai.contracts import CompletionResponse, StopReason
from fie_ai.structured import (
    GroundingReport,
    check_grounding,
    extract_json,
    parse_structured,
    structured_request,
)
from fie_common.errors import AIProviderResponseError
from fie_schemas.base import FIEModel
from fie_testing.factories import make_completion_request

pytestmark = pytest.mark.unit


class FilingSummary(FIEModel):
    ticker: str
    fiscal_year: int
    revenue_usd: float
    source_ids: list[str] = Field(default_factory=list)


def response(text: str, **kwargs: object) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        model="claude-opus-5",
        provider="claude",
        **kwargs,  # type: ignore[arg-type]
    )


class TestExtractJson:
    def test_parses_bare_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_parses_a_fenced_block(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parses_an_unlabelled_fence(self) -> None:
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_recovers_json_after_a_preamble(self) -> None:
        # Models add a sentence even under schema constraints.
        assert extract_json('Here is the result:\n{"a": 1}') == {"a": 1}

    def test_parses_a_top_level_array(self) -> None:
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_recovers_an_array_after_a_preamble(self) -> None:
        assert extract_json("Results: [1, 2]") == [1, 2]

    def test_handles_nested_objects(self) -> None:
        assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}

    def test_empty_output_is_an_error(self) -> None:
        with pytest.raises(AIProviderResponseError, match="empty output"):
            extract_json("   ")

    def test_unparseable_output_is_an_error(self) -> None:
        with pytest.raises(AIProviderResponseError, match="did not contain valid JSON"):
            extract_json("there is no json here at all")

    def test_error_includes_a_preview_for_debugging(self) -> None:
        with pytest.raises(AIProviderResponseError) as exc_info:
            extract_json("not json")

        assert "not json" in exc_info.value.details["output_preview"]


class TestParseStructured:
    def test_validates_into_the_target_model(self) -> None:
        raw = '{"ticker": "ACME", "fiscal_year": 2024, "revenue_usd": 1200000.0}'

        summary = parse_structured(response(raw), FilingSummary)

        assert summary.ticker == "ACME"
        assert summary.revenue_usd == 1_200_000.0

    def test_schema_violations_are_reported_with_field_detail(self) -> None:
        raw = '{"ticker": "ACME", "fiscal_year": "not-a-year", "revenue_usd": 1.0}'

        with pytest.raises(AIProviderResponseError, match="failed schema validation") as exc_info:
            parse_structured(response(raw), FilingSummary)

        assert exc_info.value.details["errors"]

    def test_missing_required_fields_are_rejected(self) -> None:
        with pytest.raises(AIProviderResponseError):
            parse_structured(response('{"ticker": "ACME"}'), FilingSummary)

    def test_a_refusal_is_reported_as_such(self) -> None:
        refusal = response("", stop_reason=StopReason.REFUSAL, refusal_category="cyber")

        with pytest.raises(AIProviderResponseError, match="refused") as exc_info:
            parse_structured(refusal, FilingSummary)

        assert exc_info.value.details["refusal_category"] == "cyber"

    def test_truncated_output_is_rejected_before_parsing(self) -> None:
        # Truncated JSON can parse as valid-but-wrong; rejecting on the stop
        # reason avoids silently accepting a half-built financial record.
        truncated = response('{"ticker": "ACME"', stop_reason=StopReason.MAX_TOKENS)

        with pytest.raises(AIProviderResponseError, match="truncated"):
            parse_structured(truncated, FilingSummary)

    def test_fenced_structured_output_is_accepted(self) -> None:
        raw = '```json\n{"ticker": "ACME", "fiscal_year": 2024, "revenue_usd": 1.0}\n```'

        assert parse_structured(response(raw), FilingSummary).ticker == "ACME"


class TestStructuredRequest:
    def test_attaches_the_models_json_schema(self) -> None:
        request = structured_request(make_completion_request(), FilingSummary)

        assert request.response_schema is not None
        assert "ticker" in request.response_schema["properties"]

    def test_the_original_request_is_unchanged(self) -> None:
        original = make_completion_request()
        structured_request(original, FilingSummary)

        assert original.response_schema is None

    def test_strict_models_forbid_additional_properties(self) -> None:
        request = structured_request(make_completion_request(), FilingSummary)

        assert request.response_schema is not None
        assert request.response_schema.get("additionalProperties") is False


class TestGrounding:
    def test_citations_resolving_to_supplied_sources_are_grounded(self) -> None:
        report = check_grounding(["src_1", "src_2"], ["src_1", "src_2", "src_3"])

        assert report.is_grounded is True
        assert report.unknown_ids == []

    def test_a_fabricated_citation_is_detected(self) -> None:
        # The most dangerous hallucination in a research product: a citation
        # that looks real but points at nothing.
        report = check_grounding(["src_1", "src_fabricated"], ["src_1"])

        assert report.is_grounded is False
        assert report.unknown_ids == ["src_fabricated"]

    def test_missing_citations_fail_when_citation_is_required(self) -> None:
        report = check_grounding([], ["src_1"])

        assert report.uncited is True
        assert report.is_grounded is False

    def test_citation_can_be_optional(self) -> None:
        report = check_grounding([], ["src_1"], require_citation=False)

        assert report.uncited is False
        assert report.is_grounded is True

    def test_duplicate_citations_are_collapsed(self) -> None:
        report = check_grounding(["src_1", "src_1"], ["src_1"])

        assert report.cited_ids == ["src_1"]

    def test_report_is_serializable_for_test_output(self) -> None:
        report = check_grounding(["src_1"], ["src_1"])

        assert isinstance(report, GroundingReport)
        assert report.to_json_dict()["cited_ids"] == ["src_1"]

    def test_no_allowed_sources_makes_every_citation_unknown(self) -> None:
        report = check_grounding(["src_1"], [])

        assert report.unknown_ids == ["src_1"]
        assert report.is_grounded is False
