"""Direction grounding — the check Phase 3 said it could not do.

The failure under test is one every number-checking mechanism is blind to: a
sentence in which every figure is correct and the claim about them is backwards.
"Marketing delivered real savings" against a $100k overspend contains no wrong
number at all.
"""

from __future__ import annotations

import pytest
from cfo_fixtures import CHART, HEADCOUNT, MARKETING, REVENUE, annual

from cfo_ai.analysis.variance import Variance
from cfo_ai.research.direction import (
    Assertion,
    ClaimKind,
    check_direction_grounding,
    extract_direction_claims,
)
from fie_finance.money import Money

pytestmark = pytest.mark.unit

PERIOD = annual()
ACCOUNTS = list(CHART.values())

#: Marketing overspent: plan 600k, actual 700k. Above plan, and unfavourable.
MARKETING_OVERSPEND = Variance.between(
    MARKETING, "cc-sales", PERIOD, Money.of("600000"), Money.of("700000")
)
#: Revenue missed: plan 4.0m, actual 3.6m. Below plan, and unfavourable.
REVENUE_MISS = Variance.between(
    REVENUE, "cc-sales", PERIOD, Money.of("4000000"), Money.of("3600000")
)
#: Marketing underspent: plan 600k, actual 500k. Below plan, and favourable.
MARKETING_SAVING = Variance.between(
    MARKETING, "cc-sales", PERIOD, Money.of("600000"), Money.of("500000")
)

BOTH = [MARKETING_OVERSPEND, REVENUE_MISS]


def check(text: str, variances=BOTH):
    return check_direction_grounding(text, variances, ACCOUNTS)


class TestTruthfulNarrative:
    def test_a_correct_narrative_passes(self) -> None:
        report = check(
            "Marketing came in above plan for the year. "
            "Revenue was below plan and represents a shortfall."
        )
        assert report.is_grounded
        assert report.checked_claims == 3

    def test_a_correct_saving_is_allowed_to_be_called_a_saving(self) -> None:
        """The check must not simply distrust praise."""
        report = check("Marketing delivered real savings this year.", [MARKETING_SAVING])
        assert report.is_grounded

    def test_prose_with_no_directional_claims_is_trivially_grounded(self) -> None:
        report = check("The finance team completed the year-end close on schedule.")
        assert report.is_grounded
        assert report.checked_claims == 0


class TestEvaluativeContradictions:
    def test_praising_an_overspend_is_caught(self) -> None:
        """Every figure could be right and the sentence still reverses the truth."""
        report = check("Marketing delivered a strong performance and real savings.")

        assert not report.is_grounded
        assert report.contradictions[0].kind is ClaimKind.EVALUATIVE
        assert report.contradictions[0].account_code == "6100"

    def test_praising_a_revenue_miss_is_caught(self) -> None:
        report = check("Revenue was ahead of plan, an encouraging result.")

        assert not report.is_grounded
        assert "unfavourable" in report.contradictions[0].computed

    def test_the_explanation_names_the_account_and_the_computed_verdict(self) -> None:
        report = check("Marketing produced welcome savings.")
        explanation = report.contradictions[0].explanation

        assert "6100" in explanation
        assert "unfavourable" in explanation

    def test_one_sentence_asserting_one_thing_is_one_finding(self) -> None:
        """ "A strong performance with real savings" matches four phrases and
        makes one claim; four findings would make the totals meaningless."""
        report = check("Marketing delivered a strong performance and real savings.")

        assert report.checked_claims == 1
        assert len(report.contradictions) == 1


class TestPositionalContradictions:
    def test_claiming_an_overspend_came_in_under_is_caught(self) -> None:
        report = check("Marketing came in under budget for the year.")

        assert not report.is_grounded
        assert report.contradictions[0].kind is ClaimKind.POSITIONAL
        assert report.contradictions[0].computed == "above plan"

    def test_claiming_a_miss_was_above_plan_is_caught(self) -> None:
        report = check("Revenue finished above plan.")

        assert not report.is_grounded
        assert report.contradictions[0].computed == "below plan"

    def test_position_and_evaluation_are_checked_separately(self) -> None:
        """ "Under budget, which is favourable" is two claims about a cost that
        overspent, and both are wrong."""
        report = check("Marketing was under budget, which is favourable.")

        kinds = {contradiction.kind for contradiction in report.contradictions}
        assert kinds == {ClaimKind.POSITIONAL, ClaimKind.EVALUATIVE}

    def test_a_correct_position_with_a_wrong_verdict_is_still_caught(self) -> None:
        """The sentence gets the sign right and the meaning backwards — which is
        exactly what happens when a writer forgets which way a cost runs."""
        report = check("Marketing came in above plan, a pleasing improvement.")

        assert len(report.contradictions) == 1
        assert report.contradictions[0].kind is ClaimKind.EVALUATIVE


class TestVocabularyDiscipline:
    def test_under_and_over_are_positional_not_evaluative(self) -> None:
        """ "Under budget" is good for a cost and bad for revenue. A checker that
        read the word itself as praise would manufacture contradictions."""
        claims, _ = extract_direction_claims("Marketing was under budget.", ACCOUNTS)

        assert [claim.kind for claim in claims] == [ClaimKind.POSITIONAL]

    def test_a_correct_underspend_is_not_flagged_for_the_word_under(self) -> None:
        report = check("Marketing came in under budget.", [MARKETING_SAVING])
        assert report.is_grounded

    def test_exceeded_is_positional_so_revenue_beating_plan_reads_correctly(self) -> None:
        beat = Variance.between(
            REVENUE, "cc-sales", PERIOD, Money.of("4000000"), Money.of("4400000")
        )
        report = check("Revenue exceeded plan for the year.", [beat])
        assert report.is_grounded


class TestNegation:
    def test_a_negated_position_flips(self) -> None:
        """ "Did not come in under budget" asserts the line was above it."""
        report = check("Marketing did not come in under budget.")
        assert report.is_grounded

    def test_a_negated_position_can_still_contradict(self) -> None:
        report = check("Marketing did not come in above plan.")
        assert not report.is_grounded

    def test_a_negated_evaluation_is_dropped_rather_than_flipped(self) -> None:
        """ "Not a strong quarter" does not assert the quarter was bad enough to
        contradict a favourable variance; treating it as the opposite invents a
        finding."""
        report = check("Marketing was not a strong performance.", [MARKETING_SAVING])
        assert report.is_grounded


class TestAttribution:
    def test_a_sentence_naming_no_account_is_counted_not_guessed(self) -> None:
        """Guessing which line "it came in ahead" refers to would invent the very
        thing this module checks."""
        report = check("The quarter came in ahead of plan overall.")

        assert report.checked_claims == 0
        assert report.unattributed_claims == 1
        assert report.is_grounded

    def test_a_sentence_naming_two_accounts_is_skipped(self) -> None:
        report = check("Revenue and marketing both came in above plan.")

        assert report.checked_claims == 0
        assert report.unattributed_claims == 1

    def test_an_account_with_no_computed_variance_is_unattributed(self) -> None:
        """The narrative may be discussing something outside this report, and
        inventing a contradiction is as damaging as missing one."""
        report = check("Salaries came in under budget.", [MARKETING_OVERSPEND])

        assert report.checked_claims == 0
        assert report.unattributed_claims == 1

    def test_an_account_alias_is_recognised(self) -> None:
        """The chart calls it Revenue; the narrative calls it sales."""
        report = check("Sales finished above plan.")

        assert report.checked_claims == 1
        assert not report.is_grounded

    def test_an_account_code_is_recognised(self) -> None:
        report = check("Account 6100 came in under budget.")
        assert report.checked_claims == 1


class TestUnjudgedAccounts:
    def test_no_verdict_is_contradicted_where_none_was_given(self) -> None:
        """Headcount carries no favourability, so an evaluative claim about it
        has nothing to contradict — a separate concern from asserting the wrong
        verdict."""
        headcount = Variance.between(HEADCOUNT, "cc-eng", PERIOD, Money.of("50"), Money.of("44"))
        report = check("Headcount was a disappointing result.", [headcount])

        assert report.is_grounded

    def test_a_positional_claim_about_headcount_is_still_checked(self) -> None:
        """The sign is a fact even where the verdict is withheld."""
        headcount = Variance.between(HEADCOUNT, "cc-eng", PERIOD, Money.of("50"), Money.of("44"))
        report = check("Headcount finished above plan.", [headcount])

        assert not report.is_grounded
        assert report.contradictions[0].kind is ClaimKind.POSITIONAL


class TestOnPlan:
    def test_a_line_exactly_on_plan_contradicts_neither_direction(self) -> None:
        exact = Variance.between(
            MARKETING, "cc-sales", PERIOD, Money.of("600000"), Money.of("600000")
        )
        assert check("Marketing came in above plan.", [exact]).is_grounded
        assert check("Marketing came in under budget.", [exact]).is_grounded


class TestReportSummary:
    def test_the_summary_reads_for_a_human(self) -> None:
        clean = check("Marketing came in above plan.")
        dirty = check("Marketing produced real savings.")

        assert "none contradicted" in clean.summary
        assert "contradict" in dirty.summary

    def test_claims_carry_the_phrase_that_triggered_them(self) -> None:
        claims, _ = extract_direction_claims("Marketing came in above plan.", ACCOUNTS)

        assert claims[0].phrase in claims[0].sentence.lower()
        assert claims[0].assertion is Assertion.ABOVE_PLAN


class TestPhraseMatching:
    def test_unfavourable_is_not_read_as_favourable(self) -> None:
        """Regression: "favourable" is a substring of "unfavourable", so a
        plain substring search read an accurate negative verdict as praise and
        manufactured a contradiction out of correct prose. Phrases match on word
        boundaries."""
        report = check("Marketing came in above plan, which is unfavourable.")
        assert report.is_grounded

    def test_the_matching_still_catches_the_real_claim(self) -> None:
        """The boundary fix must not blunt the check it protects."""
        report = check("Marketing came in above plan, which is favourable.")
        assert not report.is_grounded

    def test_saving_does_not_match_inside_a_longer_word(self) -> None:
        report = check("Marketing oversaving is not a word.", [MARKETING_OVERSPEND])
        assert report.checked_claims == 0
