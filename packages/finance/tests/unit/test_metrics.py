"""What a Metric refuses to be.

The rule these tests protect is the ecosystem's central one: a language model
may explain a computed figure and may never be the source of one. That is not a
convention here — it is a constructor that fails.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from fie_common.errors import ValidationError
from fie_finance.metrics import AnalysisResult, Unit, derived, estimated, unavailable
from fie_schemas.provenance import AssertionKind

pytestmark = pytest.mark.unit


class TestDerived:
    def test_a_computed_figure_cannot_name_a_model(self) -> None:
        """The whole architecture in one assertion."""
        metric = derived("net_margin", "23.4", Unit.PERCENT, computation="test.margin.v1")
        assert metric.provenance.kind is AssertionKind.DERIVED
        assert metric.provenance.model is None

    def test_the_computation_is_recorded_so_a_number_is_reproducible(self) -> None:
        metric = derived(
            "gross_margin",
            "40.0",
            Unit.PERCENT,
            computation="test.margin.v1",
            inputs={"revenue": "100", "cost": "60"},
        )
        assert metric.provenance.computation == "test.margin.v1"
        assert metric.inputs == {"revenue": "100", "cost": "60"}

    def test_a_float_is_refused(self) -> None:
        with pytest.raises(Exception, match="float is not accepted"):
            derived("x", 0.1, Unit.RATIO, computation="test.v1")  # type: ignore[arg-type]


class TestEstimated:
    def test_an_estimate_must_state_its_assumptions(self) -> None:
        """A projection nobody can argue with is one nobody should act on."""
        with pytest.raises(ValidationError, match="assumptions"):
            estimated("runway", "18", Unit.DAYS, assumptions={})

    def test_assumptions_travel_with_the_number(self) -> None:
        metric = estimated(
            "runway_months", "18", Unit.YEARS, assumptions={"burn": "constant at Q4 rate"}
        )
        assert metric.provenance.kind is AssertionKind.ESTIMATE
        assert metric.provenance.assumptions == {"burn": "constant at Q4 rate"}


class TestUnavailable:
    def test_a_missing_input_never_becomes_zero(self) -> None:
        """A zero reads as a measurement; "not disclosed" is a different claim."""
        metric = unavailable(
            "current_ratio", Unit.RATIO, reason="no balance sheet", computation="test.v1"
        )
        assert not metric.is_available
        assert metric.unavailable_reason == "no balance sheet"

    def test_unavailable_metrics_are_kept_separate_in_a_result(self) -> None:
        result = AnalysisResult(
            entity_id="ent-1",
            period_label="FY2025",
            metrics=[
                derived("a", "1", Unit.RATIO, computation="t.v1"),
                unavailable("b", Unit.RATIO, reason="missing", computation="t.v1"),
            ],
        )
        assert [m.name for m in result.available] == ["a"]
        assert [m.name for m in result.unavailable] == ["b"]
        assert result.numeric_values() == {Decimal(1)}
