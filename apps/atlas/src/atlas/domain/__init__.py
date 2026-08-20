"""Atlas's financial domain: reported statements.

Money and fiscal periods moved to :mod:`fie_finance` when CFO.ai needed them —
they are value primitives with no financial semantics, and four private copies
of ``Money`` would have meant four subtly different roundings. They are
re-exported here so a reader of this package still finds the whole vocabulary in
one place. What stays in Atlas is what Atlas actually owns: the statements, and
the accounting identities they must satisfy.
"""

from atlas.domain.statements import (
    IDENTITY_TOLERANCE,
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from fie_finance.money import CENTS, Money, Rate, Shares, sum_money, to_decimal
from fie_finance.periods import FiscalPeriod, PeriodKind

__all__ = [
    "CENTS",
    "IDENTITY_TOLERANCE",
    "BalanceSheet",
    "CashFlowStatement",
    "FinancialStatements",
    "FiscalPeriod",
    "IncomeStatement",
    "Money",
    "PeriodKind",
    "Rate",
    "Shares",
    "sum_money",
    "to_decimal",
]
