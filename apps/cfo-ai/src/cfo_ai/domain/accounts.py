"""The chart of accounts, and the one thing it decides.

A variance is two numbers and a direction. The two numbers are arithmetic; the
direction is the part that goes wrong.

Revenue five percent under plan is bad news. Marketing spend five percent under
plan is good news. The subtraction is identical — the same sign, the same
magnitude — and only the *account* says which reading is correct. A report that
gets this backwards is not slightly wrong: it says the opposite of the truth,
in fluent prose, with the right numbers in it.

So favourability is never computed at a call site. It is a property of the
account type, and :func:`Variance` cannot be constructed without one. There is
no code path that produces a variance and then decides what it means.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, field_validator

from fie_schemas.base import FrozenModel

#: Account codes as a general ledger writes them: digits, optionally grouped.
_CODE_PATTERN = re.compile(r"^[0-9][0-9A-Z_.-]{0,31}$")


class AccountType(StrEnum):
    """What kind of line this is, and therefore which way is up."""

    #: Money coming in. More than plan is favourable.
    REVENUE = "revenue"
    #: Direct cost of delivering revenue. Less than plan is favourable.
    COST_OF_REVENUE = "cost_of_revenue"
    #: Operating expense — salaries, marketing, rent. Less is favourable.
    OPERATING_EXPENSE = "operating_expense"
    #: Capital expenditure. Less is favourable against a plan.
    CAPITAL_EXPENDITURE = "capital_expenditure"
    #: A computed subtotal such as gross profit or EBITDA. More is favourable.
    MARGIN = "margin"
    #: Cash balance. More is favourable.
    CASH = "cash"
    #: Headcount. Deliberately *not* directional — see ``has_direction``.
    HEADCOUNT = "headcount"

    @property
    def is_inflow(self) -> bool:
        """Whether a larger number is the better outcome.

        Only meaningful when :attr:`has_direction` is true.
        """
        return self in (AccountType.REVENUE, AccountType.MARGIN, AccountType.CASH)

    @property
    def is_cost(self) -> bool:
        """Whether this line consumes money."""
        return self in (
            AccountType.COST_OF_REVENUE,
            AccountType.OPERATING_EXPENSE,
            AccountType.CAPITAL_EXPENDITURE,
        )

    @property
    def has_direction(self) -> bool:
        """Whether over-plan and under-plan carry a value judgement at all.

        Headcount does not. Hiring behind plan is a saving to a CFO and a
        capacity problem to the engineering director who is waiting for those
        people, and a system that picks one of those readings and calls it
        "favourable" is asserting something it cannot know. The variance is
        still computed and still reported — only the verdict is withheld.
        """
        return self is not AccountType.HEADCOUNT

    @property
    def affects_profit(self) -> bool:
        """Whether the line belongs in an operating result.

        Capital expenditure and cash balances do not: spending cash on a
        building is not an expense, and rolling it into an operating variance
        overstates the miss.
        """
        return self in (
            AccountType.REVENUE,
            AccountType.COST_OF_REVENUE,
            AccountType.OPERATING_EXPENSE,
            AccountType.MARGIN,
        )


class Account(FrozenModel):
    """A single line in the chart of accounts."""

    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=200)
    type: AccountType
    #: Optional parent code, for accounts that roll into a subtotal.
    parent_code: str | None = Field(default=None, max_length=32)
    #: Alternative names this account is known by in a narrative — "sales" for
    #: "Revenue", "marketing" for "Marketing Expense". Used when checking that
    #: prose describes the right line; see cfo_ai.research.direction.
    aliases: list[str] = Field(default_factory=list)

    @field_validator("code", "parent_code")
    @classmethod
    def _validate_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        upper = value.strip().upper()
        if not _CODE_PATTERN.match(upper):
            raise ValueError(
                f"{value!r} is not a general ledger account code; codes start with a "
                "digit and contain digits, letters, dots, dashes, or underscores"
            )
        return upper

    @field_validator("aliases")
    @classmethod
    def _normalise_aliases(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for alias in value:
            cleaned = alias.strip().lower()
            if cleaned:
                seen.setdefault(cleaned, None)
        return list(seen)

    @property
    def search_terms(self) -> list[str]:
        """Every name this account might appear under, lowercased."""
        terms: dict[str, None] = {self.name.strip().lower(): None}
        for alias in self.aliases:
            terms.setdefault(alias, None)
        return list(terms)

    def is_own_parent(self) -> bool:
        return self.parent_code is not None and self.parent_code == self.code


__all__ = ["Account", "AccountType"]
