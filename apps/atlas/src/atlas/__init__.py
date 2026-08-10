"""Atlas — autonomous financial intelligence.

Atlas reads filings, computes financial analysis, and writes research. The
division of labour inside it is the ecosystem's central rule made concrete:

- **Deterministic Python** computes every number. Ratios, growth, discounted
  cash flow, and multiples live in :mod:`atlas.analysis`, which no model call
  can reach. The results carry the inputs they were computed from, so a reader
  can reproduce them by hand.
- **Claude** reads those results and explains them. It orchestrates the work,
  interprets what the figures mean, and writes the narrative — and it is never
  asked what a number is.

The boundary is enforced by construction. A computed value is built with
``Provenance.derived``, which the shared validator forbids from naming a model;
a projection is built with ``Provenance.estimate``, which cannot exist without
its assumptions. Before a report is returned, every figure in its prose is
checked against the figures Atlas actually computed.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
