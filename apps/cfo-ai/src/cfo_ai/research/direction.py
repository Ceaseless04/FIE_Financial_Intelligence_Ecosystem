"""Direction grounding — checking what the prose *claims*, not just its figures.

Phase 3 shipped numeric grounding and recorded what it could not do:

    Numeric grounding checks figures, not claims. "Margins improved" against a
    falling margin is not caught by this mechanism.

This module closes that gap for variances, where the claim is checkable because
the deterministic layer already decided the answer. A sentence saying marketing
came in "ahead of plan" against a computed overspend is wrong in a way no
figure-matching check can see: every number in it may be correct.

Two different kinds of claim are extracted, and they are checked against two
different things, because conflating them is the mistake this module exists to
avoid:

**Positional** claims assert a *sign* — "above plan", "under budget", "came in
below". Those are checked against the sign of the variance. They carry no
judgement: a cost under budget and revenue under budget are both "under", and
only one of them is good news.

**Evaluative** claims assert *good or bad* — "a strong quarter", "a shortfall",
"savings", "concerning". Those are checked against the computed favourability,
which is the account type's business and not the sentence's.

Deliberately excluded from the evaluative vocabulary: "exceeded", "over", and
"under". They look evaluative and are purely positional — "exceeded budget" is
bad and "exceeded plan" on revenue is good, and a checker that treated the word
itself as praise would manufacture contradictions.

**What this does not catch**, stated plainly because a check whose limits are
unclear invites more trust than it has earned:

- Sentences naming no account, or naming several. A claim that cannot be tied to
  exactly one computed variance is counted and skipped, never guessed at.
- Claims about anything other than variance direction — causes, forecasts,
  recommendations, comparisons between periods.
- Irony, hedging, and conditionals. "Marketing would have been ahead of plan"
  reads as a claim here.
- Any wording outside the vocabulary below.

It is a keyword check over sentences, and it is worth having because the failure
it catches — a fluent, correctly-numbered sentence asserting the reverse of the
truth — is the one a reader is least equipped to notice.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field

from cfo_ai.analysis.variance import Favourability, Variance
from cfo_ai.domain.accounts import Account
from fie_schemas.base import FrozenModel

#: Sentence boundaries. Crude on purpose — abbreviations split a sentence in
#: two, which costs a claim rather than inventing one.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Words that negate a claim appearing shortly after them.
_NEGATIONS = ("not", "no", "never", "hardly", "failed to", "far from", "rather than")

#: How many characters before a phrase a negation still reaches.
_NEGATION_WINDOW = 40

#: Claims that the line came in *higher* than plan. Sign only, no judgement.
_ABOVE_PLAN = (
    "above plan",
    "above budget",
    "above the plan",
    "above the budget",
    "over plan",
    "over budget",
    "overspent",
    "overspend",
    "overran",
    "overrun",
    "higher than plan",
    "higher than budget",
    "came in above",
    "came in over",
    "exceeded plan",
    "exceeded budget",
    "exceeded the plan",
    "exceeded the budget",
)

#: Claims that the line came in *lower* than plan.
_BELOW_PLAN = (
    "below plan",
    "below budget",
    "below the plan",
    "below the budget",
    "under plan",
    "under budget",
    "under the plan",
    "under the budget",
    "underspent",
    "underspend",
    "lower than plan",
    "lower than budget",
    "came in below",
    "came in under",
    "short of plan",
    "short of budget",
)

#: Claims that the outcome was *good*.
_GOOD = (
    "favourable",
    "favorable",
    "ahead of plan",
    "ahead of budget",
    "beat plan",
    "beat budget",
    "outperformed",
    "overachieved",
    "strong performance",
    "a strong",
    "saving",
    "savings",
    "improvement",
    "improved",
    "better than expected",
    "better than plan",
    "upside",
    "encouraging",
)

#: Claims that the outcome was *bad*.
_BAD = (
    "unfavourable",
    "unfavorable",
    "behind plan",
    "behind budget",
    "missed plan",
    "missed budget",
    "shortfall",
    "underperformed",
    "disappointing",
    "concerning",
    "worse than expected",
    "worse than plan",
    "weak performance",
    "a weak",
    "downside",
    "deterioration",
    "deteriorated",
)


class ClaimKind(StrEnum):
    """What a sentence asserts about a line."""

    #: Which side of plan the line landed on.
    POSITIONAL = "positional"
    #: Whether that was good or bad.
    EVALUATIVE = "evaluative"


class Assertion(StrEnum):
    """The claim itself."""

    ABOVE_PLAN = "above_plan"
    BELOW_PLAN = "below_plan"
    GOOD = "good"
    BAD = "bad"


class DirectionClaim(FrozenModel):
    """One checkable directional statement found in the prose."""

    sentence: str
    account_code: str
    kind: ClaimKind
    assertion: Assertion
    #: The words that triggered it, so a finding can be explained rather than
    #: merely reported.
    phrase: str
    negated: bool = False


class DirectionContradiction(FrozenModel):
    """A claim the computed variance refutes."""

    sentence: str
    account_code: str
    account_name: str
    kind: ClaimKind
    asserted: Assertion
    #: What the deterministic layer actually found.
    computed: str
    phrase: str
    explanation: str


class DirectionGroundingReport(FrozenModel):
    """The verdict on a narrative's directional claims."""

    #: Claims found and checked.
    checked_claims: int = Field(default=0, ge=0)
    #: Sentences that carried a directional phrase but named no single account,
    #: so nothing could be checked. Reported rather than hidden: a narrative
    #: whose claims are all unattributable has not been verified.
    unattributed_claims: int = Field(default=0, ge=0)
    contradictions: list[DirectionContradiction] = Field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when no checked claim contradicts a computed variance."""
        return not self.contradictions

    @property
    def summary(self) -> str:
        if self.is_grounded:
            return f"{self.checked_claims} directional claims checked, none contradicted"
        return (
            f"{len(self.contradictions)} of {self.checked_claims} directional claims "
            "contradict the computed variances"
        )


def _is_negated(sentence: str, position: int) -> bool:
    """Whether a negation shortly precedes a phrase."""
    window = sentence[max(0, position - _NEGATION_WINDOW) : position].lower()
    return any(re.search(rf"\b{re.escape(word)}\b", window) for word in _NEGATIONS)


def _accounts_in(sentence: str, accounts: list[Account]) -> list[Account]:
    """Accounts a sentence names, by code or by any of their search terms."""
    lowered = sentence.lower()
    found: list[Account] = []
    for account in accounts:
        if account.code.lower() in lowered:
            found.append(account)
            continue
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in account.search_terms):
            found.append(account)
    return found


def _claims_in(sentence: str, account_code: str) -> list[tuple[ClaimKind, Assertion, str, bool]]:
    """Every directional phrase in one sentence."""
    lowered = sentence.lower()
    found: list[tuple[ClaimKind, Assertion, str, bool]] = []

    for phrases, kind, assertion in (
        (_ABOVE_PLAN, ClaimKind.POSITIONAL, Assertion.ABOVE_PLAN),
        (_BELOW_PLAN, ClaimKind.POSITIONAL, Assertion.BELOW_PLAN),
        (_GOOD, ClaimKind.EVALUATIVE, Assertion.GOOD),
        (_BAD, ClaimKind.EVALUATIVE, Assertion.BAD),
    ):
        for phrase in phrases:
            # Word boundaries, not a substring search: "unfavourable" contains
            # "favourable", so a bare `find` reads a correct negative verdict as
            # praise and manufactures a contradiction out of accurate prose.
            match = re.search(rf"\b{re.escape(phrase)}\b", lowered)
            if match is not None:
                found.append((kind, assertion, phrase, _is_negated(sentence, match.start())))
    return found


def extract_direction_claims(
    narrative: str, accounts: list[Account]
) -> tuple[list[DirectionClaim], int]:
    """Find directional claims that can be tied to exactly one account.

    Returns the claims and a count of directional sentences that named no single
    account. A sentence mentioning two accounts is skipped rather than assigned
    to one of them — guessing which line "it came in ahead" refers to would
    invent the very thing this module checks.
    """
    claims: list[DirectionClaim] = []
    unattributed = 0

    for sentence in _SENTENCE_SPLIT.split(narrative):
        stripped = sentence.strip()
        if not stripped:
            continue

        raw_claims = _claims_in(stripped, "")
        if not raw_claims:
            continue

        named = _accounts_in(stripped, accounts)
        if len(named) != 1:
            unattributed += 1
            continue

        account = named[0]
        # One sentence asserting the same thing twice is one claim. "a strong
        # quarter with real savings" matches four phrases and makes two
        # assertions; counting the phrases would report a single misleading
        # sentence as four findings and make the totals meaningless. The longest
        # match wins, being the most specific.
        best: dict[tuple[ClaimKind, Assertion], tuple[ClaimKind, Assertion, str, bool]] = {}
        for candidate in raw_claims:
            key = (candidate[0], candidate[1])
            if key not in best or len(candidate[2]) > len(best[key][2]):
                best[key] = candidate

        for kind, assertion, phrase, negated in best.values():
            claims.append(
                DirectionClaim(
                    sentence=stripped,
                    account_code=account.code,
                    kind=kind,
                    assertion=assertion,
                    phrase=phrase,
                    negated=negated,
                )
            )

    return claims, unattributed


def check_direction_grounding(
    narrative: str, variances: list[Variance], accounts: list[Account]
) -> DirectionGroundingReport:
    """Verify a narrative's directional claims against the computed variances.

    A claim about an account with no computed variance is counted as
    unattributed rather than treated as a contradiction: the narrative may be
    discussing something outside this report, and inventing a contradiction is
    as damaging as missing one.
    """
    claims, unattributed = extract_direction_claims(narrative, accounts)
    by_code = {account.code: account for account in accounts}
    variance_by_code: dict[str, Variance] = {}
    for variance in variances:
        # First occurrence wins; a per-centre breakdown of one account shares a
        # direction often enough that checking against the first is safe, and
        # where it does not, the claim is about a line this cannot resolve.
        variance_by_code.setdefault(variance.account_code, variance)

    contradictions: list[DirectionContradiction] = []
    checked = 0

    for claim in claims:
        matched = variance_by_code.get(claim.account_code)
        if matched is None:
            unattributed += 1
            continue

        checked += 1
        asserted = _resolve(claim)
        if asserted is None:
            continue

        contradiction = _contradiction_for(claim, asserted, matched, by_code)
        if contradiction is not None:
            contradictions.append(contradiction)

    return DirectionGroundingReport(
        checked_claims=checked,
        unattributed_claims=unattributed,
        contradictions=contradictions,
    )


def _resolve(claim: DirectionClaim) -> Assertion | None:
    """Apply negation, flipping the claim to its opposite.

    A negated evaluative claim is dropped rather than flipped: "not a strong
    quarter" does not assert that the quarter was bad enough to contradict a
    favourable variance, and treating it as the opposite invents a finding.
    """
    if not claim.negated:
        return claim.assertion
    if claim.kind is ClaimKind.POSITIONAL:
        return (
            Assertion.BELOW_PLAN
            if claim.assertion is Assertion.ABOVE_PLAN
            else Assertion.ABOVE_PLAN
        )
    return None


def _contradiction_for(
    claim: DirectionClaim,
    asserted: Assertion,
    variance: Variance,
    accounts: dict[str, Account],
) -> DirectionContradiction | None:
    """Compare one resolved claim against the computed variance."""
    name = accounts[claim.account_code].name if claim.account_code in accounts else ""

    if claim.kind is ClaimKind.POSITIONAL:
        if variance.amount.is_zero:
            return None  # exactly on plan contradicts neither direction usefully
        actually_above = not variance.amount.is_negative
        claimed_above = asserted is Assertion.ABOVE_PLAN
        if actually_above == claimed_above:
            return None
        return DirectionContradiction(
            sentence=claim.sentence,
            account_code=claim.account_code,
            account_name=name,
            kind=claim.kind,
            asserted=asserted,
            computed="above plan" if actually_above else "below plan",
            phrase=claim.phrase,
            explanation=(
                f"the narrative says {claim.account_code} was "
                f"{'above' if claimed_above else 'below'} plan, but the computed "
                f"variance is {variance.amount} against a plan of {variance.budget}"
            ),
        )

    # Evaluative: check against the account's own idea of which way is up.
    if not variance.favourability.is_judgement:
        # The deterministic layer declined to judge, so there is nothing to
        # contradict — and a narrative asserting a verdict here is a separate
        # concern from asserting the *wrong* one.
        return None

    claimed_good = asserted is Assertion.GOOD
    computed_good = variance.favourability is Favourability.FAVOURABLE
    if claimed_good == computed_good:
        return None

    return DirectionContradiction(
        sentence=claim.sentence,
        account_code=claim.account_code,
        account_name=name,
        kind=claim.kind,
        asserted=asserted,
        computed=str(variance.favourability),
        phrase=claim.phrase,
        explanation=(
            f"the narrative describes {claim.account_code} as "
            f"{'good' if claimed_good else 'bad'} news, but the computed variance "
            f"is {variance.favourability} for a {variance.account_type} account"
        ),
    )


__all__ = [
    "Assertion",
    "ClaimKind",
    "DirectionClaim",
    "DirectionContradiction",
    "DirectionGroundingReport",
    "check_direction_grounding",
    "extract_direction_claims",
]
