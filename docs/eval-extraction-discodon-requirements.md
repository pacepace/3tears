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

## R5 — provider credit metering is injected, not imported

**Property.** `usage_capture.py` does not import `SEARCH_CREDITS_BY_DEPTH` from
`discodon.tools.web_search_tool`. The credits-per-call table arrives as
configuration, alongside the operator-declared dollars-per-credit it is already
paired with.

**Why this one is different from the rest.** It is not a shim to be removed —
it is a convergence point. In the extracted world that table is the *search
adapter's* property, and `3tears-search` already owns the Tavily adapter.
Injecting it now means the eventual resolution against the search leaf is a
wiring change rather than a code change. The rationale already written beside
that constant — that the credit cost is a property of the call, and that eval's
run-level rate card resolves off the research tool's depth — is the same
argument for injecting it.

**Verify.** A test that constructs the rate card from a supplied table, with no
import of the tool module.

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
- The credit-table seam on the search side, so R5's injected value eventually
  resolves against `3tears-search` rather than a local constant.
- The package cut itself, once R1 holds — at which point the lift is mechanical.
