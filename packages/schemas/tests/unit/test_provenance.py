"""Unit tests for provenance — the grounding enforcement mechanism.

These tests are the executable form of the spec's cross-product constraints:
facts must be cited, derived values must name a deterministic computation,
estimates must state their assumptions, and model output must be labelled as
generated rather than presented as fact.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fie_schemas.provenance import (
    AssertionKind,
    Attributed,
    Provenance,
    SourceLocator,
    SourceReference,
    SourceType,
)

pytestmark = pytest.mark.unit


def make_source(source_id: str = "sec_0001") -> SourceReference:
    return SourceReference(
        source_id=source_id,
        source_type=SourceType.SEC_FILING,
        title="Form 10-K",
        excerpt="Total revenue was $1,000,000.",
    )


class TestSourceLocator:
    def test_accepts_a_valid_character_range(self) -> None:
        locator = SourceLocator(start_char=10, end_char=50)

        assert locator.end_char == 50

    def test_rejects_an_inverted_range(self) -> None:
        with pytest.raises(ValidationError, match="end_char must be >= start_char"):
            SourceLocator(start_char=50, end_char=10)

    def test_rejects_a_zero_page_number(self) -> None:
        with pytest.raises(ValidationError):
            SourceLocator(page=0)

    def test_all_fields_are_optional(self) -> None:
        assert SourceLocator().page is None


class TestSourceReference:
    def test_requires_a_non_empty_source_id(self) -> None:
        with pytest.raises(ValidationError):
            SourceReference(source_id="", source_type=SourceType.NEWS_ARTICLE)

    def test_retrieved_at_defaults_to_now(self) -> None:
        assert make_source().retrieved_at is not None

    def test_is_immutable(self) -> None:
        source = make_source()

        with pytest.raises(ValidationError):
            source.source_id = "changed"  # type: ignore[misc]


class TestFactProvenance:
    def test_a_fact_requires_at_least_one_source(self) -> None:
        with pytest.raises(ValidationError, match="FACT requires at least one source"):
            Provenance(kind=AssertionKind.FACT)

    def test_a_fact_with_a_source_is_valid(self) -> None:
        provenance = Provenance.fact(make_source())

        assert provenance.kind is AssertionKind.FACT
        assert provenance.is_verifiable is True
        assert provenance.is_model_authored is False

    def test_a_fact_cannot_name_a_model(self) -> None:
        # Model-authored content is GENERATED. Labelling it a FACT is exactly
        # the failure mode the constraint exists to prevent.
        with pytest.raises(ValidationError, match="FACT cannot name a model"):
            Provenance(kind=AssertionKind.FACT, sources=[make_source()], model="claude-opus-5")

    def test_multiple_sources_are_supported(self) -> None:
        provenance = Provenance.fact(make_source("a"), make_source("b"))

        assert len(provenance.sources) == 2


class TestDerivedProvenance:
    def test_a_derived_value_must_name_its_computation(self) -> None:
        with pytest.raises(ValidationError, match="must name the deterministic computation"):
            Provenance(kind=AssertionKind.DERIVED)

    def test_a_derived_value_with_a_computation_is_valid(self) -> None:
        provenance = Provenance.derived("dcf_valuation_v1", make_source())

        assert provenance.computation == "dcf_valuation_v1"
        assert provenance.is_verifiable is True

    def test_a_derived_value_cannot_come_from_a_model(self) -> None:
        # This is the architectural rule made executable: a calculation is
        # produced by a deterministic service, never by an LLM.
        with pytest.raises(ValidationError, match="deterministic service, not a model"):
            Provenance(
                kind=AssertionKind.DERIVED,
                computation="dcf_valuation_v1",
                model="claude-opus-5",
            )

    def test_a_derived_value_is_verifiable_without_sources(self) -> None:
        assert Provenance.derived("ratio_calculator_v2").is_verifiable is True


class TestEstimateProvenance:
    def test_an_estimate_must_state_its_assumptions(self) -> None:
        with pytest.raises(ValidationError, match="must state the assumptions"):
            Provenance(kind=AssertionKind.ESTIMATE)

    def test_an_estimate_with_assumptions_is_valid(self) -> None:
        provenance = Provenance.estimate(
            {"discount_rate": 0.12, "terminal_growth": 0.02, "horizon_years": 5}
        )

        assert provenance.assumptions["discount_rate"] == 0.12

    def test_assumptions_are_displayable(self) -> None:
        # Venture must "display assumptions", so they must survive serialization.
        provenance = Provenance.estimate({"tam_method": "top_down", "penetration": 0.03})

        assert provenance.to_json_dict()["assumptions"]["tam_method"] == "top_down"

    def test_an_estimate_may_also_cite_sources(self) -> None:
        provenance = Provenance.estimate({"growth": 0.1}, make_source())

        assert provenance.is_verifiable is True


class TestGeneratedProvenance:
    def test_generated_content_must_name_its_model(self) -> None:
        with pytest.raises(ValidationError, match="must name the model"):
            Provenance(kind=AssertionKind.GENERATED)

    def test_generated_content_is_flagged_as_model_authored(self) -> None:
        provenance = Provenance.generated("claude-opus-5")

        assert provenance.is_model_authored is True

    def test_generated_content_without_sources_is_not_verifiable(self) -> None:
        assert Provenance.generated("claude-opus-5").is_verifiable is False

    def test_generated_content_grounded_in_sources_is_verifiable(self) -> None:
        provenance = Provenance.generated("claude-opus-5", make_source())

        assert provenance.is_verifiable is True
        assert provenance.is_model_authored is True


class TestConfidence:
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_valid_confidence_range(self, confidence: float) -> None:
        assert Provenance.fact(make_source(), confidence=confidence).confidence == confidence

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_out_of_range_confidence_is_rejected(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            Provenance.fact(make_source(), confidence=confidence)

    def test_confidence_is_optional(self) -> None:
        assert Provenance.fact(make_source()).confidence is None


class TestAttributed:
    def test_binds_a_value_to_its_provenance(self) -> None:
        attributed = Attributed[float](value=1_000_000.0, provenance=Provenance.fact(make_source()))

        assert attributed.value == 1_000_000.0
        assert attributed.is_model_authored is False

    def test_model_authored_values_are_identifiable(self) -> None:
        attributed = Attributed[str](
            value="The company faces margin pressure.",
            provenance=Provenance.generated("claude-opus-5"),
            label="thesis",
        )

        assert attributed.is_model_authored is True

    def test_serialization_preserves_attribution(self) -> None:
        # Attribution must survive the trip into a report or across a service
        # boundary; a value that arrives without it cannot be labelled.
        attributed = Attributed[float](
            value=42.0, provenance=Provenance.derived("ratio_calculator_v1")
        )

        payload = attributed.to_json_dict()

        assert payload["value"] == 42.0
        assert payload["provenance"]["kind"] == "derived"
        assert payload["provenance"]["computation"] == "ratio_calculator_v1"

    def test_is_immutable(self) -> None:
        attributed = Attributed[int](value=1, provenance=Provenance.fact(make_source()))

        with pytest.raises(ValidationError):
            attributed.value = 2  # type: ignore[misc]
