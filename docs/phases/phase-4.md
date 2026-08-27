# Phase 4 — CFO.ai (Financial Planning & Analysis)

**Status: complete.** Phase 5 (Sentinel) may begin.

## Objective

Budgets, actuals, variance, forecasting, and scenarios over a company's own
management figures. Where Atlas reads what a company *filed*, CFO.ai works on
what it *plans* and what it books internally.

## The rule this phase exists to enforce

A variance is two numbers and a direction. The two numbers are a subtraction.
The direction is the part that goes wrong.

Revenue five percent under plan is bad news. Marketing five percent under plan
is good news. Identical inputs, identical subtraction, identical percentage —
and only the *account* says which reading is correct. A report that gets this
backwards is not slightly wrong: it says the opposite of the truth, fluently,
with the right numbers in it.

| | Plan | Actual | Δ | % | Verdict |
|---|---|---|---|---|---|
| Revenue | 400,000 | 380,000 | −20,000 | −5.00% | **unfavourable** |
| Marketing | 400,000 | 380,000 | −20,000 | −5.00% | **favourable** |

So favourability is derived from `AccountType` inside `Variance.between`, and
**there is no parameter for a caller to pass**. A test asserts the signature has
none. There is no code path that computes a difference and then decides
separately what it means.

### One account type refuses to have an opinion

`HEADCOUNT.has_direction` is `False`. Hiring behind plan is a saving to a CFO
and a capacity problem to the engineering director waiting for those people, and
a system that picks one of those readings and calls it "favourable" is asserting
something it cannot know. The variance is still computed and still reported —
only the verdict is withheld, and the prompt tells the model to do the same.

## Direction grounding — the phase's signature check

Phase 3 shipped numeric grounding and recorded what it could not do:

> Numeric grounding checks figures, not claims. "Margins improved" against a
> falling margin is not caught by this mechanism.

This phase closes that gap where the claim is checkable, because the
deterministic layer already decided the answer:

> **"Marketing delivered savings against its 600000 plan."**

The plan really is 600,000. Every figure is real. Numeric grounding **passes
it** — and it says the reverse of the truth about a 100,000 overspend. That is
the sentence a reader is least equipped to doubt, because the numbers check out.

Two kinds of claim are extracted and checked against two different things:

| Kind | Example | Checked against |
|---|---|---|
| **Positional** | "above plan", "under budget" | the *sign* of the variance |
| **Evaluative** | "a strong quarter", "savings" | the computed *favourability* |

Keeping them apart is the point. A cost under budget and revenue under budget
are both "under", and only one is good news. So `exceeded`, `over`, and `under`
are **deliberately excluded** from the evaluative vocabulary — they look
evaluative and are purely positional, and a checker treating the word itself as
praise would manufacture contradictions out of accurate prose.

Commentary must clear **both** gates to be publishable. A draft that quotes a
real number and draws the opposite conclusion is not a lesser failure than one
that invents a number; it is a more persuasive one.

## Delivered

| Area | Module | What it does |
|---|---|---|
| Domain | `domain/accounts.py` | The chart of accounts, and the one thing it decides |
| Domain | `domain/organisation.py` | A cost centre hierarchy validated as an actual tree |
| Domain | `domain/plan.py` | Budgets and actuals with one line per key and period containment |
| Analysis | `analysis/variance.py` | Variance, favourability, materiality on two thresholds |
| Analysis | `analysis/compare.py` | Matching, unmatched-line reporting, roll-ups from leaves |
| Analysis | `analysis/forecast.py` | Run rate, moving average, least-squares trend, seasonal naive |
| Analysis | `analysis/cash.py` | Burn, and runway with three distinct outcomes |
| Analysis | `analysis/scenarios.py` | Driver-based scenarios that record what they changed |
| Research | `research/direction.py` | Direction grounding |
| Research | `research/commentary.py` | Narrative generation behind both gates |
| Pipeline | `pipeline.py` | Planning and reporting workflows, with events |
| Storage | `storage/` | NUMERIC-typed plan lines, charts, org, commentary |
| API | `api/` | Chart, organisation, plans, variance, commentary, health |
| Platform | `packages/finance` | `Money`, `FiscalPeriod`, `Metric`, numeric grounding |

## Gate results

Verified on 2026-08-27 against live Docker infrastructure (PostgreSQL 16, Redis
7, Neo4j 5, Ollama):

```
1333 passed, 9 skipped
Coverage:          92.24%   (gate: 90%)
ruff check:        All checks passed
ruff format:       230 files
mypy --strict:     no issues in 153 source files
alembic (cfo_ai):  upgrade -> downgrade -> upgrade -> check, all clean
image:             builds, runs as uid 1001, pip removed, healthy readiness
trivy:             0 CRITICAL/HIGH with a fix available
```

CFO.ai contributes 188 of those tests: 137 unit, 19 integration against real
PostgreSQL, and 32 API contract tests.

All three Alembic histories were confirmed coexisting in one database:

```
 atlas       | alembic_version_atlas
 cfo_ai      | alembic_version_cfo_ai
 marketmind  | alembic_version_marketmind
```

### Phase gate

> "Financial calculations must always be deterministic and independently
> validated rather than trusting an LLM's arithmetic."

Validated against hand-worked values rather than the code's own output: a
perfect line through 100/110/120/130 has slope 10 and projects 140 then 150; the
average of those four is 115; 1.2m of cash against 100k monthly burn is 12.0
months; and the FY26 fixture's net operating variance is −400,000 (−400,000
revenue, +50,000 COGS saving, −100,000 marketing overspend, +50,000 salary
saving), worked out longhand in the test's own docstring.

## Decisions worth carrying forward

**Favourability is never an argument.** Derived inside the constructor from the
account type. A caller cannot pass the wrong verdict because a caller cannot
pass a verdict.

**A zero budget yields no percentage.** `None`, not infinity, and not a very
large number standing in for it. The line is judged material on size alone
rather than dropped from the report.

**Runway has three outcomes, not one with edge cases.** Burning gives a number
of months; generating cash and break-even give *none*, with a stated reason.
"We are not burning" and "we are building cash" are different facts about a
company, and neither is a number.

**Variance reports are not stored.** A variance is a pure function of a budget,
an actuals set, and the chart of accounts — all three of which are stored.
Persisting the result would create a second source of truth that goes stale the
moment a line is restated, and nothing would mark it as old.

**A plan's identity includes its version.** "FY26 approved" and "FY26 board
revision 2" coexist; re-uploading the same version supersedes it.

**Roll-ups are computed from leaves, never by summing child totals.** Summing
summaries is how a rounding difference at depth four becomes a reconciliation
error at the top.

**Unmatched lines are findings, not noise.** An actual with no budget is
unplanned spend; a budget with no actual may be a forgotten accrual. Both are
returned, and unplanned spend gets its own event type.

**Scenarios refuse a driver naming a target the baseline lacks.** A mistyped
cost centre otherwise looks exactly like a scenario with no impact.

## Issues found and fixed during the phase

1. **`"favourable"` is a substring of `"unfavourable"`.** The phrase matcher
   used a plain substring search, so an accurate negative verdict — "above plan,
   which is unfavourable" — was read as praise and flagged as a contradiction.
   The check was manufacturing findings from correct writing. Fixed with word
   boundaries, plus a regression test.
2. **Overlapping phrases inflated the counts.** "A strong performance and real
   savings" matched four phrases and reported four findings for one claim,
   making the totals meaningless. Claims now collapse per assertion.
3. **A domain rejection inside a handler produced a 500, not a 422.** Submitting
   a plan with two lines for one key was refused by the domain from inside the
   handler — telling a client its own duplicate row was our outage. The request
   schema now delegates to the domain at the edge, the same fix Phase 3 applied
   to fiscal periods.
4. **The "hierarchy must have a root" check was dead code.** If every parent
   resolves and no path repeats, walking upward always terminates at a
   parentless centre — a rootless hierarchy is always a cyclic one, and the
   cycle check already rejects it. Removed, with the reasoning recorded.
5. **Atlas's authorization test was checking zero routes.** This version of
   FastAPI keeps an included router as a single wrapper object in `app.routes`
   rather than splicing its routes in, so the loop matched nothing and the test
   passed while verifying nothing. Both apps now flatten through the wrappers
   and assert a minimum route count, so the same silence cannot return.
6. **mypy caught a variable shadowed across two loops**, where
   `warn_unreachable` correctly flagged the `None` guard as dead.

Items 1, 2, and 5 are the same species: a check that looks like it is working.
Item 5 in particular had been green since Phase 3.

## Events published

Consumers subscribe to `fie.events.cfo-ai`.

| Event | Payload highlights |
|---|---|
| `cfo.plan.stored` | Plan id, totals, and cost centres the org chart cannot place |
| `cfo.chart.updated` | Account types, because they decide direction downstream |
| `cfo.variance.computed` | Net operating variance, counts by direction including not-assessed |
| `cfo.variance.unplanned_spend` | Keys with actuals and no budget |
| `cfo.forecast.produced` | Method, horizon, and the assumptions it rests on |
| `cfo.cash.runway_assessed` | Months **or** an explicit reason there are none |
| `cfo.scenario.modelled` | Baseline, scenario, change, and the drivers |
| `cfo.commentary.published` | Both grounding verdicts |
| `cfo.commentary.withheld` | The same, for commentary that failed one |

## Known limitations

Stated rather than hidden:

- **Direction grounding is a keyword check.** It catches explicit directional
  language tied to exactly one named account. Sentences naming no account or
  several are counted and skipped, never guessed at. Claims about causes,
  forecasts, or recommendations are out of scope, and irony and conditionals
  read as claims. It is worth having because the failure it catches is the one a
  reader is least equipped to notice — not because it is complete.
- **Event publication is not transactional**, as in Phases 2 and 3.
- **No currency conversion.** A plan is single-currency and mixing is refused
  rather than converted, because a conversion needs a rate and a date that
  nobody has supplied.
- **Forecasting is univariate.** Each account is projected from its own history;
  there is no driver model linking headcount to salary cost.
- **Seasonal naive needs twelve periods** and assumes monthly data. A quarterly
  history will not satisfy it, and it says so rather than guessing.
- **No approval workflow.** Plans are versioned but there is no notion of who
  approved which version; that is an access-control and audit feature rather
  than an analytical one.

## Scope boundary

CFO.ai plans and analyses a company's own management figures. It does not value
securities, make investment recommendations, or score third-party risk — those
belong to Atlas, Venture, and Sentinel respectively. The commentary prompt
additionally forbids recommending headcount actions and attributing a variance
to a named individual.

**Recorded honestly:** unlike MarketMind's and FinOps's, this boundary is
derived from the product split rather than quoted from an explicit "must not"
list in the specification. If one exists for CFO.ai, it should be reconciled
against this section.

## Phase 5 entry criteria

Satisfied:

- [x] Variance, roll-up, forecasting, cash, and scenario tests pass
- [x] Direction is decided by the chart of accounts and cannot be passed in
- [x] No analysis module can reach an AI provider (asserted by test)
- [x] Storage and migration verified against real PostgreSQL, alongside two
      other histories
- [x] API contract tests cover authentication, authorization, and validation —
      and the routing-table test now enumerates real routes
- [x] Coverage ≥ 90%
- [x] Lint, format, and strict type checking clean
- [x] Container builds, runs unprivileged, and passes Trivy

## What Phase 5 will need from Phase 4

Sentinel builds on: the `cfo.*` event stream — particularly
`cfo.variance.unplanned_spend`, which is a control finding as much as a
financial one — the cost centre hierarchy for attributing exposure to an owner,
and the direction-grounding mechanism, which generalises to any narrative making
a directional claim about a computed figure.

What Sentinel must add: risk detection, scoring, and propagation across the
MarketMind graph.
