# Phase 3 — Atlas (Autonomous Financial Research)

**Status: complete.** Phase 4 (CFO.ai) may begin.

## Objective

Turn filings into research a reader can check: extract financial statements,
compute every figure deterministically, value companies from stated assumptions,
and let a language model explain those figures without ever producing one.

## The rule this phase exists to enforce

> Claude should be the primary reasoning/agent model, but not the source of
> truth for financial calculations.

Phase 2 established provenance as a type. Phase 3 is where it does work. Three
mechanisms make the rule structural rather than aspirational:

**A computed number cannot be attributed to a model.** Every figure is built
through `analysis.results.derived`, which constructs `Provenance.derived(...)`.
The Phase 1 validator refuses a DERIVED assertion that names a model — so a
metric with a model attached does not fail review, it fails to construct. A
projection uses `Provenance.estimate`, which cannot exist without stated
assumptions, because a DCF is a judgement about a discount rate wearing a
number's clothes.

**No analysis module can reach an AI provider.** `atlas.analysis` contains every
calculation and imports nothing from `fie_ai`. A test walks the package and
fails if any module there mentions `AIRouter`, `CompletionRequest`, or `fie_ai`
— so the boundary cannot erode by someone adding one convenient import.

**The model transcribes; it never calculates.** Extraction returns every figure
as a string exactly as printed, plus the scale the statement is presented in.
Atlas applies the multiplier. "In thousands" at the top of a table is the single
most consequential piece of context in a filing, and a model that both reads it
and applies it can be wrong by three orders of magnitude with nothing looking
unusual.

## Numeric grounding — the phase's signature check

Phase 2 verified that a generated answer's *citations* resolved. Phase 3 applies
the same idea to the thing that actually matters in a research report: the
numbers.

Before any report is returned, every figure in its prose is matched against what
Atlas computed. Matching rounds each computed value to the number of significant
digits the model actually wrote, then compares exactly.

| Written | Against computed 1,613,590,000 | Verdict |
|---|---|---|
| `$1.6 billion` | rounds to 1.6bn | accepted |
| `$1.61 billion` | rounds to 1.61bn | accepted |
| `$1.614 billion` | rounds to 1.614bn | accepted |
| `$1.9 billion` | nothing rounds there | **rejected** |
| `$16 billion` | nothing rounds there | **rejected** |

**Rounding is permitted; invention is not.** This is stricter and far more
explainable than a percentage tolerance, which either accepts a wrong number or
rejects a legitimately rounded one depending on magnitude.

Magnitudes are compared rather than signed values, because prose carries the
sign in words: a reported net loss of −12,400,000 is written "a loss of $12.4
million". Bare four-digit integers in 1900–2100 are exempt as calendar years —
the only exemption, because each one is a hole in the check.

An ungrounded draft is regenerated **once**, with the offending figures named so
the retry is corrective rather than a reroll. If it fails again the report is
still returned, with `publishable: false` and the unsupported figures listed.
Discarding it would hide a systematic prompt or model problem behind an empty
response.

## Delivered

| Area | Module | What it does |
|---|---|---|
| Domain | `domain/money.py` | Decimal-only `Money`, `Rate`, `Shares`; float rejected at construction; currencies never mix silently |
| Domain | `domain/periods.py` | Period kinds, comparability rules, and a stable `key` for storage |
| Domain | `domain/statements.py` | Income, balance sheet, and cash flow statements that validate themselves against the accounting identities |
| Analysis | `analysis/results.py` | `Metric`, and the `derived` / `estimated` / `unavailable` constructors that bind a figure to how it was produced |
| Analysis | `analysis/ratios.py` | Margins, returns, liquidity, leverage, coverage, diluted EPS |
| Analysis | `analysis/dcf.py` | Gordon-growth DCF with an enforced discount-rate spread, WACC, sensitivity grid |
| Analysis | `analysis/growth.py` | Period-over-period growth, CAGR, free cash flow, trends |
| Analysis | `analysis/comparables.py` | Peer multiples on the **median**, mixed-basis rejection |
| Filings | `filings/extraction.py` | Transcription behind structured output; validation *is* construction of the domain models |
| Filings | `filings/pipeline.py` | Filing → store → transcribe → validate → persist → announce |
| Research | `research/grounding.py` | Significant-digit numeric grounding |
| Research | `research/reports.py` | Report generation, one corrective regeneration, citation verification |
| Research | `research/pipeline.py` | The full ordering: facts → figures → estimate → context → model → verify |
| Storage | `storage/models.py`, `repository.py` | NUMERIC-typed statement tables, filings, and reports |
| Clients | `clients/marketmind.py` | Company context that degrades to nothing rather than failing a report |
| API | `api/` | Filings, statements, analysis, valuation, research, health |

Also delivered: the `atlas` Alembic history with its own namespaced version
table, `Dockerfile.atlas`, dev and production compose services, the `ATLAS_*`
configuration block, and CI wiring for all of it.

## Gate results

Verified on 2026-08-10 against live Docker infrastructure (PostgreSQL 16, Redis
7, Neo4j 5, Ollama):

```
1138 passed, 9 skipped
Coverage:          92.78%   (gate: 90%)
ruff check:        All checks passed
ruff format:       183 files formatted
mypy --strict:     no issues in 121 source files
alembic (atlas):   upgrade -> downgrade -> upgrade -> check, all clean
```

Atlas contributes 193 of those tests: 128 unit, 30 integration against real
PostgreSQL, and 35 API contract tests. The 9 skips are the live-provider tests,
unchanged from Phase 2.

Both Alembic histories were verified coexisting in one database:

```
 table_schema |         table_name
--------------+----------------------------
 atlas        | alembic_version_atlas
 marketmind   | alembic_version_marketmind
```

That is what the namespaced version tables exist for, and what a shared
`alembic_version` table would have broken.

### Phase gate

> "Financial calculations must always be deterministic and independently
> validated rather than trusting an LLM's arithmetic."

Validated two ways. The DCF was checked against a hand-worked example — present
value of the forecast 454.53, terminal value 1,866.73, enterprise value 1,613.59
— and WACC against `0.8 × 0.12 + 0.2 × 0.06 × 0.75 = 0.105`. And
`test_pipelines.py` runs the whole path against a real database: a filing is
transcribed, validated against the accounting identities, stored as NUMERIC,
read back, analysed, and only then handed to a model — whose prose is checked
against the figures that came out of the database.

## Decisions worth carrying forward

**Money is NUMERIC in the database, never a float and never a JSON number.**
`DOUBLE PRECISION` would be smaller and faster and would silently reintroduce
the binary-float error the domain layer refuses at construction. Every figure
also crosses the HTTP boundary as a *string*: JSON has one numeric type and it
is a double, so a response emitting `412600000.5` as a JSON number hands the
client a float and undoes the whole discipline at the last step.

**A missing line item is `null`, never `0`.** A zero reads as a measurement.
"Not disclosed" is a different claim about the company, it is preserved through
storage and the API, and the report prompt is given the list explicitly so the
model writes "not disclosed" instead of estimating.

**Unavailable metrics are returned, not filtered out.** A thin filing and a
complete one produce different analyses; reporting only what computed makes them
look identical.

**Assumptions are supplied, never inferred.** The valuation endpoint requires a
discount rate, terminal growth, and forecast growth. A service that picks them
silently turns an arguable estimate into an apparent measurement. They are
echoed back with the result and carried in the ESTIMATE provenance.

**Atlas will not invent a base cash flow to force a valuation.** A filing that
does not disclose capital expenditure, or that has negative free cash flow, gets
a 422 — not a number.

**`report.published` and `report.withheld` are different event types.** Not one
event with a boolean. A downstream consumer has to opt into the ungrounded case
deliberately instead of missing a flag.

**MarketMind failures cost context, never a report.** Graph context makes a note
better; it is never what a figure is computed from. Every failure — including a
404 on the context endpoint — degrades to no context. Identity is the exception:
Atlas does not fall back to matching on name when MarketMind cannot confirm an
entity, because duplicating resolution is how two products end up disagreeing
about who a company is.

**A stored statement set is re-validated on read.** The accounting identities run
again when rows are rehydrated, so a row corrupted in the database fails to load
rather than flowing into an analysis.

## Issues found and fixed during the phase

Each was caught by running something rather than reading it:

1. **A JSON float in a money field produced a 500, not a 422.** Pydantic converts
   only `ValueError` and `AssertionError` into a validation error; the `TypeError`
   that `to_decimal` raises propagated untouched. A client posting
   `{"amount": 1234.56}` got an opaque server error instead of being told its
   request was malformed. `decimal_field` now re-raises as `ValueError` — the
   value is still refused, because the conversion is lossy, but as bad input.
2. **The error handler could not serialize its own body.** A field validator that
   raises `ValueError` puts the raised *exception object* into Pydantic's error
   `ctx`. Passing that into a JSON response raised inside the 422 handler and
   turned every such request into a 500 — the failure mode where a client's
   mistake looks like our outage. `ErrorDetail` now coerces its details to
   JSON-safe values. This was latent in MarketMind too; nothing there raised a
   `ValueError` from a request validator yet.
3. **An invalid fiscal period escaped request validation.** `PeriodRequest`
   accepted a quarterly period with no quarter, and the domain model rejected it
   later inside the handler — a 500 for what is plainly a bad request. The request
   schema now delegates to `FiscalPeriod` rather than restating its rules, so the
   two cannot drift.
4. **The unique constraint on statement sets did not constrain.** PostgreSQL
   treats every NULL as distinct, and `fiscal_quarter` is NULL for an annual
   period — so a constraint over `(entity_id, fiscal_year, kind, fiscal_quarter)`
   would have silently permitted two FY2025 sets for one company. Replaced with a
   NOT NULL `period_key` (`annual:2025:0`). The integration test asserts the
   duplicate is refused.
5. **Every Atlas test errored at setup when the full suite ran.** Both apps'
   `tests/conftest.py` offered helpers imported as `from conftest import ...`,
   and `sys.modules` is process-wide — so the first app collected won and the
   other app's imports resolved against the wrong module. Invisible while
   MarketMind was the only app. Helpers now live in `atlas_fixtures.py` and
   `marketmind_fixtures.py`; the convention is documented in both conftests
   because Phase 4 would have hit it again.
6. **Grounding was sign-sensitive in isolation.** `check_numeric_grounding`
   rejected "a loss of $12.4 million" against a computed −12,400,000 unless the
   caller had gone through `allowed_values_from`, which adds the negations. It
   now compares magnitudes directly.
7. **A hand-checked expectation was wrong in the last digit.** The P/E test
   asserted 51.7876 against a true value of 51.78749726. The implementation was
   right and the estimate was not — which is the reason for checking against
   worked examples rather than against the code's own output.

## Events published

Consumers subscribe to `fie.events.atlas`.

| Event | Payload highlights |
|---|---|
| `atlas.filing.ingested` | Filing id, entity, period, content hash |
| `atlas.filing.duplicate` | The redelivered id and the existing one |
| `atlas.statements.extracted` | Statement set id, which statements survived, what was rejected |
| `atlas.statements.rejected` | Reasons, and whether figures contradicted each other |
| `atlas.analysis.completed` | Metrics computed **and** metrics the filing did not support |
| `atlas.valuation.produced` | Enterprise/equity value as strings, assumptions, terminal dominance |
| `atlas.report.published` | Report id, citations, grounding verdict |
| `atlas.report.withheld` | The same, for a report that failed verification |

## Known limitations

Stated rather than hidden — each is a deliberate deferral:

- **Event publication is not transactional**, as in Phase 2. Same reasoning,
  same Phase 8 resolution.
- **Comparable-company valuation has no market data source.** `MarketData` is
  supplied by the caller; Atlas does not fetch prices. Wiring a market data feed
  is a Phase 8 integration concern, and inventing one now would mean a valuation
  resting on numbers nobody can trace.
- **Extraction reads the first 60,000 characters** of a filing by default.
  Statements sit in the financial section, but a filing that buries them later
  will be missed. The truncation is stated in the prompt and logged.
- **Growth analysis needs two stored periods** and does not fetch a prior filing
  automatically; the caller ingests both.
- **One filing, one period.** A 10-K carries comparative prior-year columns that
  Atlas does not currently extract separately.
- **Numeric grounding checks figures, not claims.** "Margins improved" against a
  falling margin is not caught by this mechanism. Detecting directional
  misstatement is a different check, and pretending this one covers it would be
  worse than saying so.

## Phase 4 entry criteria

Satisfied:

- [x] Filing extraction and analysis tests pass against real PostgreSQL
- [x] Every financial calculation is deterministic and independently validated
- [x] No analysis module can reach an AI provider (asserted by test)
- [x] API contract tests cover authentication, authorization, and validation
- [x] Coverage ≥ 90% (92.78%)
- [x] Lint, format, and strict type checking clean
- [x] Migrations apply, reverse, and re-apply against a real database, alongside
      MarketMind's
- [x] Money never becomes a float — at construction, in storage, or on the wire

## What Phase 4 will need from Phase 3

CFO.ai builds on: the `Money`/`Rate`/`Shares` primitives and their float
refusal, the `Metric` provenance constructors, the DCF and ratio machinery, and
the numeric grounding check — a budget narrative has exactly the same failure
mode as a research note.

What CFO.ai must add, and Atlas deliberately does not have: corporate budgeting,
internal forecasting against a plan, and scenario modelling over management
figures rather than filed ones.
