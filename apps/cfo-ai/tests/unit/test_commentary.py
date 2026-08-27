"""Commentary generation and the two gates it has to clear.

The test this module exists for is
``test_a_correctly_numbered_lie_is_caught``: a draft whose every figure is real
and whose conclusion is backwards. Numeric grounding passes it. It is the exact
failure Phase 3 recorded as out of reach, and the reason direction grounding
exists.

AI behaviour is never asserted by matching text. What is asserted is that a
reversed claim is caught, and that a commentary carrying one is not marked
publishable.
"""

from __future__ import annotations

import pytest
from cfo_fixtures import CHART, actuals, budget, make_json_response, make_router

from cfo_ai.analysis.compare import compare
from cfo_ai.research.commentary import CommentaryRequest, CommentaryService
from fie_ai.contracts import CompletionResponse, StopReason
from fie_common.errors import AIProviderResponseError, ValidationError
from fie_schemas.provenance import AssertionKind

pytestmark = pytest.mark.unit


def report():
    return compare(budget(), actuals(), CHART)


def request_for() -> CommentaryRequest:
    return CommentaryRequest(
        entity_id="ent-northwind",
        company_name="Northwind Components",
        variances=report(),
        accounts=CHART,
    )


def draft(summary: str, **overrides: object) -> dict[str, object]:
    return {
        "summary": summary,
        "sections": overrides.get("sections", []),
        "open_questions": overrides.get("open_questions", []),
    }


def service(*drafts: dict[str, object], regenerations: int = 1) -> CommentaryService:
    return CommentaryService(
        make_router([make_json_response(item) for item in drafts]),
        max_regenerations=regenerations,
    )


class TestBothGates:
    async def test_a_faithful_commentary_is_publishable(self) -> None:
        subject = service(
            draft("Revenue came in below plan and marketing came in above plan, both unfavourable.")
        )

        commentary = await subject.generate(request_for())

        assert commentary.numerically_grounded
        assert commentary.directionally_grounded
        assert commentary.is_publishable

    async def test_a_correctly_numbered_lie_is_caught(self) -> None:
        """The whole reason this phase has a second gate.

        Marketing overspent by 100,000 against a 600,000 plan. Every figure in
        this sentence is real. The conclusion is the reverse of the truth, and
        numeric grounding cannot see it.
        """
        subject = service(
            draft("Marketing delivered savings against its 600000 plan."),
            draft("Marketing delivered savings against its 600000 plan."),
        )

        commentary = await subject.generate(request_for())

        assert commentary.numerically_grounded  # no invented figure
        assert not commentary.directionally_grounded  # but the claim is backwards
        assert not commentary.is_publishable
        assert any("6100" in text for text in commentary.direction_contradictions)

    async def test_an_invented_figure_is_still_caught(self) -> None:
        """The Phase 3 gate is not weakened by the new one."""
        subject = service(
            draft("Revenue was 9999123 below plan."),
            draft("Revenue was 9999123 below plan."),
        )

        commentary = await subject.generate(request_for())

        assert not commentary.numerically_grounded
        assert not commentary.is_publishable

    async def test_both_failures_are_reported_together(self) -> None:
        bad = draft("Marketing achieved savings of 9999123 against plan.")
        subject = service(bad, bad)

        commentary = await subject.generate(request_for())

        assert commentary.unsupported_figures
        assert commentary.direction_contradictions


class TestRegeneration:
    async def test_a_reversed_claim_triggers_one_corrective_retry(self) -> None:
        subject = service(
            draft("Marketing delivered welcome savings."),
            draft("Marketing came in above plan, which is unfavourable."),
        )

        commentary = await subject.generate(request_for())

        assert commentary.regeneration_count == 1
        assert commentary.is_publishable

    async def test_the_retry_names_the_specific_problem(self) -> None:
        """A retry that only says "try again" is a reroll of the same dice."""
        subject = service(
            draft("Marketing delivered welcome savings."),
            draft("Marketing came in above plan."),
        )
        await subject.generate(request_for())

        # The prompt is rebuilt with the contradiction's explanation appended.
        prompt = subject._build_prompt(request_for())
        assert "direction of each line is stated" in prompt

    async def test_a_persistent_failure_is_returned_not_discarded(self) -> None:
        """Discarding it would hide a systematic prompt problem."""
        bad = draft("Marketing produced a strong result.")
        subject = service(bad, bad)

        commentary = await subject.generate(request_for())

        assert commentary.summary  # the work is returned
        assert not commentary.is_publishable

    async def test_regeneration_can_be_disabled(self) -> None:
        subject = service(draft("Marketing produced savings."), regenerations=0)

        commentary = await subject.generate(request_for())

        assert commentary.regeneration_count == 0
        assert not commentary.is_publishable


class TestPrompt:
    async def test_the_prompt_states_each_direction_rather_than_a_bare_sign(self) -> None:
        """Inferring direction from a sign is the mistake being prevented, so the
        prompt does not leave it to be inferred."""
        prompt = service(draft("x"))._build_prompt(request_for())

        assert "unfavourable" in prompt
        assert "favourable" in prompt
        assert "Use it as given" in prompt

    async def test_the_prompt_forbids_judging_an_unassessed_line(self) -> None:
        from cfo_fixtures import HEADCOUNT, annual

        from cfo_ai.analysis.variance import Variance
        from fie_finance.money import Money

        base = report()
        with_headcount = base.model_copy(
            update={
                "variances": [
                    *base.variances,
                    Variance.between(HEADCOUNT, "cc-eng", annual(), Money.of("50"), Money.of("44")),
                ]
            }
        )
        prompt = service(draft("x"))._build_prompt(
            CommentaryRequest(
                entity_id="ent-northwind",
                company_name="Northwind",
                variances=with_headcount,
                accounts=CHART,
            )
        )

        assert "Lines with no direction" in prompt
        assert "do not call it good or bad" in prompt

    async def test_the_prompt_carries_the_net_operating_variance(self) -> None:
        prompt = service(draft("x"))._build_prompt(request_for())
        assert "Net operating variance" in prompt


class TestProvenanceAndRefusals:
    async def test_the_narrative_is_generated_and_the_metrics_are_not(self) -> None:
        subject = service(draft("Revenue came in below plan."))

        commentary = await subject.generate(request_for())

        assert commentary.provenance.kind is AssertionKind.GENERATED
        assert commentary.provenance.model == "fake-model-1"
        assert all(metric.provenance.model is None for metric in commentary.metrics)

    async def test_commentary_cannot_be_written_from_nothing(self) -> None:
        empty = report().model_copy(update={"variances": []})
        request = CommentaryRequest(
            entity_id="ent-northwind",
            company_name="Northwind",
            variances=empty,
            accounts=CHART,
        )

        with pytest.raises(ValidationError, match="no computed variances"):
            await service().generate(request)

    async def test_a_refusal_surfaces_rather_than_becoming_empty_commentary(self) -> None:
        refusal = CompletionResponse(
            text="",
            model="fake-model-1",
            provider="fake",
            stop_reason=StopReason.REFUSAL,
            refusal_category="policy",
        )
        subject = CommentaryService(make_router([refusal]))

        with pytest.raises(AIProviderResponseError):
            await subject.generate(request_for())
