# Eval extraction: requirements for discodon

**Status:** Requirements — 2026-08-19
**For:** discodon, next release
**Why now:** the eval packages
([`family-convergence.md` §4.2](family-convergence.md#42-evals--3tears-eval-contractsrungenanalysis-new-from-discodon))
are extracted *from* discodon, and **there are no eval consumers yet**. The
contracts are unbound, so discodon's in-production shapes get to define them
rather than be migrated onto shapes nobody has run. This asks discodon to put
the package boundary inside its own tree, before any consumer binds, so the
eventual lift is a move rather than a rewrite.

**These are properties, not file edits.** A requirement written as "change these
named files" rots at the rate the tree moves, and discodon moves at roughly 65
commits a day. Each item below says what must be true, why, and how to prove it.

## What this is worth

`discodon/eval/` is 28,286 lines across 29 modules. Measured by what each module
imports from `discodon.*` outside `discodon.eval`:

| Tier | Lines | Coupling |
|---|---|---|
| None | 3,103 | imports nothing from the host |
| Three small types | 14,260 | `DiscodonBaseModel`, `LLMUsage`, `CognitiveStyleHints` |
| Genuinely host-resident | 10,923 | `service`, `runner`, `persona_factory`, `storage`, `cassette_proxy` |

The three types total 411 lines, and eval uses exactly one name from the first.
So 60% of the subsystem is a small compatibility surface away from lifting, and
the bottom tier is *correctly* coupled — it stays in discodon behind ports and
is not in scope here.

**Scope for this release** is the `3tears-eval-contracts` candidate set, ~6,900
lines: `identity.py`, `models.py`, `errors.py`, `result_condition.py`, `dsl.py`,
`versioned_inputs.py`, `budget.py`, `metering.py`, `usage_capture.py`. Nothing
moves out of discodon. No package is published. The deliverable is that those
modules stop reaching into the host.

## R1 — the eval boundary is asserted, not remembered

**Property.** No module in the scope set imports from `discodon.*` outside
`discodon.eval`. Where it needs something from the host, it takes it as an
argument or an injected port.

**Verify.** A test in discodon's own suite that walks those modules' ASTs and
fails on any `discodon.` import outside `discodon.eval`, naming the module and
the import. A test rather than a lint rule, so the boundary is enforced by the
suite that already gates merges, and so a new import is a red build rather than
a review comment somebody has to notice.

Every requirement below is a specific instance of this one. R1 is what makes
them stay true after they are met.

## R2 — eval owns its model base

**Property.** Models in the scope set inherit a base defined inside
`discodon/eval/`, not `discodon.models.base`.

It must reproduce the current configuration exactly — `extra="forbid"`,
`validate_assignment=True`, `str_strip_whitespace=True`,
`ser_json_inf_nan="constants"`, `populate_by_name=True` — **and** the
`to_json`/`from_json` pair including the lone-surrogate fallback that maps
U+D800–U+DFFF to U+FFFD. That fallback is not decoration: it exists because
externally-ingested data reaches these models, and an extraction that drops it
surfaces as a `UnicodeEncodeError` in whichever consumer first stores a scraped
string.

**Verify.** A round-trip test over a model carrying a lone surrogate, asserting
it serialises rather than raising; and an unknown-field test asserting the
`forbid` stance still holds.

**One decision to record, not to change yet.** `extra="forbid"` is a
wire-compatibility stance, and 3tears has already paid for that stance once: three
days of total refusal in production from a field no caller populated, diagnosed
only after the receiver was made to say why it was refusing. Stored eval
documents are lower risk than a live wire type — today the same version writes
and reads them. That stops being true with the **second** consumer, not with the
extraction. Keep `forbid` now; record it as a decision the first cross-repo
consumer has to revisit, so it is a choice on the record rather than a default
nobody examined.

## R3 — eval owns its usage record

**Property.** The scope set does not import `discodon.llm.usage`. Eval defines
the usage record it accounts in; the host adapts its `LLMUsage` into it at the
boundary.

**What must survive the adaptation.** `reasoning_tokens` is `int | None` where
`None` means *the provider did not report a split* and `0` means *the provider
reported zero*. Collapsing those fabricates an observation, and per-role cost
attribution depends on being able to say "not measured". The same distinction
applies to `cost_usd: float | None`.

**Verify.** A test that a host usage record with unreported reasoning tokens
crosses the boundary as `None` and not `0`, and that a reported zero crosses as
`0`.

## R4 — the cognitive-hints type crosses; the collector does not

**Property.** `models.py` holds cognitive hints as a type eval owns.
`collect_cognitive_hints`, which walks host `Tool` objects, stays host-side, and
the host passes the collected mapping in.

**Why.** The type is data and belongs with the snapshot that stores it; the
collector is a walk over the host's tool registry and cannot be anything else.
Today the snapshot model imports both, which is what makes a 3,188-line module
depend on the tool subsystem.

**Verify.** Covered by R1 once the collector call moves to the caller.

## R5 — external-call spend is reported by the caller, in provider-agnostic units

**Property.** Eval holds no provider-specific parameter-to-cost table and
interprets no provider-specific parameter. `usage_capture.py` does not import
`SEARCH_CREDITS_BY_DEPTH`, does not know what a `search_depth` is, and does not
derive a volume from one. **The thing that made the call reports what it
consumed**, in dimensions that do not name a provider:

- `calls` — how many provider calls were made;
- `provider_units` + the **name of that unit** — the provider's own weighted
  metering, whatever it is: Tavily bills credits and charges 2 for `advanced`,
  another provider may bill requests, queries, or nothing at all;
- `money` + `currency` — where the provider reports an actual charge.

Eval prices the units it is given at an operator-declared rate per
`(provider, unit)`, and never re-derives the volume.

**Why the direction matters more than the table.** Today the dependency points
the wrong way: eval reaches *into* a tool module for a constant, then
re-computes the call's cost itself from a parameter it also has to understand.
That is why a `web_search` call at `basic` gets billed at the research tool's
`advanced` rate unless something intervenes — the code that knows what the call
actually was is not the code doing the arithmetic. Reporting outward removes the
whole class: a caller cannot mis-bill a call it is describing.

**It also generalises past search, which is the point.** discodon meters image
generation, Wolfram and YouTube on the same budget mixin. An eval package whose
external-spend vocabulary is "credits by depth" can only ever account for one
provider of one kind of call.

`threetears.search.contracts.Spend` is already exactly this shape — `calls`,
`provider_units`, `money`/`currency`, wall-clock, bytes, with SR-E4 stating in
terms that the unit is provider-defined and Tavily `advanced` = 2 credits. Do
**not** import it: eval accounts for every metered external call, not only
search, so it owns its own vocabulary and search's `Spend` is one compatible
instance of it. If the two are worth unifying later, that is a shared-contracts
decision, not a dependency edge from eval to search.

**Keep the property the current rate card has.** `ExternalCallPricing` is
resolved once at launch and carried to every cell, so an operator editing a
hot-reloadable rate mid-run cannot leave one run measured in two currencies.
That must survive; it is the reason the rate is a per-run card rather than a
lookup at read time.

**Verify.** A test that prices a run's external calls from a supplied
`(unit, rate)` pairing with no import of any tool module, and a test that two
callers reporting different units on the same run are not silently summed.

## R6 — do NOT converge the eval cost cap onto `BudgetPort`

**Property.** `EvalRunCostCap` keeps its current semantics: `check()` takes no
estimate and reports accumulated-against-ceiling between cells.

**Why this is stated as a requirement rather than left alone.** 3tears' plan of
record (D27) says to wire the eval cost cap onto
`threetears.search.contracts.BudgetPort`, and a 3tears document records
discodon's `check()` as a defect for taking no estimate. **That reading is
wrong, and 3tears is changing it.** Search's port wraps exactly one provider
call — `check(estimate)` before it, `record(spend)` after it, both below the
retry boundary — so it can refuse a spend that has not happened yet. discodon's
cap fires *between eval cells*, where the cell's spend is already booked and the
cell contains an unknown number of LLM calls; there is no honest pre-estimate to
give. Two different problems, two correct shapes.

This is written down because an agent that reads the 3tears doc and finds the
mismatch will "fix" it, and the fix would break a working production control.
The eval budget contract will be shaped from what discodon runs.

## R7 — configuration arrives as values

**Property.** Scope-set modules take configuration as constructor or function
arguments. `budget.py` and `metering.py` do not import `discodon.config`; the
host resolves configuration and passes the values in.

**Verify.** Covered by R1.

## Explicitly not in scope

`service.py`, `runner.py`, `persona_factory.py`, `storage.py` and
`cassette_proxy.py` stay coupled to the host and are not being asked to move.
That coupling is correct — they are the host-resident half of the eval system,
and in the extracted world they are what implements the ports rather than what
crosses them.

Nothing here publishes a package, moves code out of discodon, or asks discodon
to depend on a 3tears package. Those come after this lands and are 3tears' work,
not discodon's.

## What 3tears owes back

- The eval budget contract, shaped from `EvalRunCostCap` rather than from
  `BudgetPort` (R6), and D27 amended to say so.
- ~~A unit **label** beside a weighted-unit count.~~ **Delivered 2026-08-19.**
  `Spend` now carries `provider_unit` as `"<provider>:<unit>"`, composed from
  the provider's own `metered_unit` declaration, and `Spend.__add__` refuses a
  mismatch the way it already refused mixed currencies. So the vocabulary R5
  asks discodon to adopt is not aspirational — the search side of it exists and
  is enforced by the adapter conformance suite.
- The package cut itself, once R1 holds — at which point the lift is mechanical.
