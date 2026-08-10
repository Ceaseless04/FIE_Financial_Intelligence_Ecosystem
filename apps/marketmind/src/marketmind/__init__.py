"""MarketMind — the financial knowledge graph.

MarketMind answers *what is true and how things connect*: which companies
exist, who runs them, who supplies whom, which industries they operate in, and
what documents support each of those claims.

It deliberately does not answer *what something is worth*. Valuation belongs to
Atlas, portfolio construction to CFO.ai, risk scoring to Sentinel, and
investment judgement to Venture Intelligence. Those products read this graph;
none of their logic lives here. The boundary is enforced in code — see
:class:`marketmind.domain.entities.Entity`, which rejects evaluative attributes
outright rather than trusting convention to hold.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
