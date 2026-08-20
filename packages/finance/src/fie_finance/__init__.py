"""Money and fiscal period primitives shared by every financial product.

**This package contains no financial semantics.** There is no interest, no
valuation, no accounting, no budgeting — nothing that encodes what a number
*means*. It provides two value types and the rules that keep them honest:

- :class:`Money`, which refuses a float at construction and refuses to add two
  currencies together, because both conversions are silent and both produce a
  number that looks right.
- :class:`FiscalPeriod`, which knows a quarter is not comparable to a year,
  because that comparison is the most common way an analysis produces a
  confidently wrong growth rate.

That distinction is the same one :mod:`fie_schemas.provenance` draws: it is an
attribution *mechanism* rather than domain logic, and lives in the platform for
the same reason this does. Domain logic — valuation, ratio analysis, budgeting,
risk scoring — stays in the applications that own it. Atlas keeps its DCF and
its accounting identities; CFO.ai keeps its variance rules; neither belongs
here.

The alternative was a copy of ``Money`` in each of the six products. Four copies
by Phase 7, and a correction to rounding or currency handling in one of them
reaching none of the others — which is precisely the "two products quietly
disagree" failure the rest of the architecture is built to prevent.
"""

from fie_finance.money import (
    CENTS,
    Money,
    Rate,
    Shares,
    decimal_field,
    sum_money,
    to_decimal,
)
from fie_finance.periods import FiscalPeriod, PeriodKind, period_key

__all__ = [
    "CENTS",
    "FiscalPeriod",
    "Money",
    "PeriodKind",
    "Rate",
    "Shares",
    "decimal_field",
    "period_key",
    "sum_money",
    "to_decimal",
]
