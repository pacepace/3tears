# Eval extraction: requirements for discodon

**Status:** Requirements — 2026-08-19; R8, R9 and the evidence section added 2026-08-20; R10 the same day; R11 and per-requirement status added 2026-08-20 (evening)
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

**How it works in discodon** (`tests/unit/eval/test_extraction_import_boundary.py`).
The AST walk is as specified. What holds it in place is **two registers, not one**:

- `_EVAL_HOST_ALLOWLIST` — **debt, and it may only shrink.** Each entry carries a
  reason and, as a typed field, the issue that retires it. Free-text scanning for
  an issue number is wrong: a reason legitimately mentions other issues and the
  scan then adopts the wrong owner.
- `_EVAL_HOST_ADAPTERS` — **the declared host-adapter seam, permanent by design.**
  These modules exist to be host-coupled; an extraction deletes them rather than
  porting them.

Filing a seam as debt records a retirement that must never happen; filing debt as
a seam launders it into architecture. Neither is detectable afterwards, so the
split is structural.

Both registers are **size-pinned, counted in `(file, host module)` pairs** — an
entry-counting ceiling lets a new module join an existing file's tuple silently.
Growing a ceiling is a two-line, diff-visible act.

**What this test does not catch:** a module added to the package and left
unclassified. Rename and deletion of a declared module fail; a new unclassified
file is simply absent from every register, and absence is not a failure. Closing
it requires asserting the classified set is total — write that assertion first,
while the module set is small (discodon is retrofitting it across ~35 files,
#2409).

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

**Each contribution names the provider it came from**, and that is a different
field from where a dollar figure came from: `price_source` today says
`"openrouter"` or `"tavily:configured_rate"` — the provenance of a *price* — and
answers nothing about who was called. Eval prices the units it is given at an
operator-declared rate per `(provider, unit)`, and never re-derives the volume.

**This is a change, not a relabelling.** Today every external call lands in one
`external` role with no provider identity on the row, and the run carries a
single `ExternalCallPricing` whose own docstring calls it "the rate card for a
credit-metered role (search)". Tavily, SearXNG and Wolfram are not
distinguishable in that structure, and `add_unpriced_calls` exists because
callers the one card cannot price still have to be counted. A self-hosted
SearXNG call is a real call with genuinely zero money — representable, and
different from an unpriced Tavily call, which is a real cost nobody declared a
rate for. One bucket cannot say which is which.

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

**Keep the property the current rate card has — as a table, not a card.**
`ExternalCallPricing` is resolved once at launch and carried to every cell, so
an operator editing a hot-reloadable rate mid-run cannot leave one run measured
in two currencies. That property must survive, and it is the reason the rate is
resolved per run rather than looked up at read time. What changes is its arity:
the run resolves a **rate table** once at launch, one entry per
`(provider, unit)`, instead of one card for one provider. A provider with no
declared rate stays counted and unpriced, exactly as today — that arm is right
and should not be lost in the widening.

**Units from different providers are never summed into one number.** A
cross-provider total is either per-provider, or money-only where every provider
involved had a declared rate. Credits and Wolfram calls are not one quantity,
and a figure that adds them is not a smaller truth but a fabricated one. This is
the same defect `Spend.__add__` already guards for currency and does not yet
guard for units — see what 3tears owes back, below.

**Verify.** Three tests: one that prices a run's external calls from a supplied
rate table with no import of any tool module; one that a run calling two
providers with different units prices each at its own rate and does not sum the
units; and one that a zero-money provider (a self-hosted search) and a provider
with no declared rate produce distinguishable rows rather than both reading as
free.

## R6 — do NOT converge the eval cost cap onto `BudgetPort`

**Property.** `EvalRunCostCap` keeps its current semantics: `check()` takes no
estimate and reports accumulated-against-ceiling between cells.

**Why this is stated as a requirement rather than left alone.** Several places
in 3tears instructed exactly the opposite, and an agent reading the repo would
have found them: `search-spec.md` §7 Phase 5 steps 1, 4 and 6;
`convergence-sequencing.md` Phase 1, Phase 4 and "Why this order"; and
`family-convergence.md` §5, which asked discodon to reshape the refusal
*"now, while discodon is sole owner of its evals"* so the later wiring would be
a one-line change. One of them also filed discodon's `check()` as a defect for
taking no estimate. **All are withdrawn as of 2026-08-19**, and if you find
another, it is stale rather than a competing decision — the withdrawal is the
ruling.

*(An earlier draft of this requirement attributed the instruction to D27. That
was imprecise: D27 rules that a replayed result reports both the recorded and
the execution spend, and says nothing about which port a cost cap implements.
The wiring instruction lived in the sequencing steps listed above, which cited
D27 for the adjacent execution-spend rule.)*

**Why the instruction was wrong.** Search's port wraps exactly one provider
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

## R8 — the lever vocabulary is host-declared, not enumerated in eval

**Property.** Scope-set modules ship the *classification* and the *machinery* —
`lever` / `apparatus` / `label`, the same/differs/unknown algebra, the confound
scan — and not the list of inputs. The host registers its own inputs, each with
a name, a role, a reader and its confounds prose. `versioned_inputs.py` holds no
name that presumes a host object model.

**Why R1 does not already cover this.** R1 is an AST test for `discodon.*`
imports. Every entry in the current vocabulary reads a field on eval's *own*
`EvalRun` — `run.prompt_overrides`, `run.persona_snapshot.entity_name` — so no
import exists to catch. **R1 goes green while Discodon's object model ships
inside the shared contract as vocabulary.** Import coupling and concept coupling
are different failures, and only one of them has a test.

Measured, 2026-08-20: the module declares 19 inputs (3 levers, 15 apparatus, 1
label). Six name Discodon concepts outright —

| Input | Role |
|---|---|
| `prompt_overrides` | lever |
| `inherited_research_model` | apparatus |
| `persona_id` | apparatus |
| `persona_graph_version_hash` | apparatus |
| `graph_version_hash` | apparatus |
| `persona_entity_name` | label |

— and a seventh, `tool_config_overrides`, is generic in name and Discodon's
config shape in fact. A consumer with no personas inherits six entries that are
permanently null and one whose reader cannot be satisfied, and nothing in the
proposed verification notices.

**This is R5 applied to a second axis, not a new philosophy.** R5 already ruled
that the thing that made the call reports what it consumed, in units eval does
not interpret. R8 is the same move for configuration: the host declares what it
swept, in names eval does not interpret. In both cases eval keeps the
arithmetic and gives up the taxonomy.

**Keep `models`.** The test is not "is this product-specific" but *would a
second LLM product have this concept under a different name, or not have it at
all?* Every consumer has models, a judge, a simulator, a cost ceiling — those
stay. Every consumer has *something* that shapes generation and can be swept,
and none of them share its shape: **the slot is shared, the shape is the
host's.**

**What must not be lost in the generalisation.** `versioned_inputs.py`'s own
docstring is the load-bearing part — *"The prose is the point, not decoration. A
bare dimension name is a label; the reason is what lets a reader judge whether
it matters."* An eval system whose inputs are opaque hashes degrades to "component
3 moved" and cannot write a readable analysis. So a registration carries the
reason, not only the name, and the `confounds` prose is **required** on
apparatus registrations rather than optional. That requirement, written for a
single-product system, is what makes a multi-product one legible.

**The rule binds every engine-owned name on a host-facing field, not just the input
registry.** Two cases found while applying it. A campaign field named `intended_k`
means iterations-per-cell to discodon and the candidate window to scriob (which calls
the repeat count `trials`), so one engine-owned name meant two things — it is
`intended_repetitions`. And a closed engine-owned enum has the same failure: a caveat
classified as `apparatus | sampling | instrument | scope` forces a domain with a
legitimate fifth kind to jam it into `scope` until the field means nothing, so the
engine kinds stay open to host-registered additions. **The slot is shared, the
vocabulary is the host's** — the same sentence, one level out from levers.

**Verify.** Construct a run from a host that registers nothing beyond the shared
core and assert the bisect and confound surfaces return a well-formed empty
answer rather than raising or reporting fabricated agreement; and a test that
the shared default registry contains no name outside the shared core.

## R9 — a swept component is identified by its content, not by a name in a host registry

**Property.** The extracted notion of a swept component is content-addressed. A
registration supplies the bytes that entered the run (or their hash), never a
key to be resolved against a store eval cannot see.

**Why.** `compute_variant_key` already does this correctly for most of what it
covers: `backstory`, `style`, `cognitive_style`, `example_statements`, `traits`,
`goals` and `directives` are hashed **as content**. Exactly one component is
hashed as a *name* — `prompt_overrides`, a `{slot: preset_name}` map that only
means anything against Discodon's prompt registry. That single entry is the
whole of what a second consumer cannot supply, and it is also the only mechanism
the system offers for *varying* anything at run scope.

So the asymmetry runs the opposite way to the intuition. Content-hashing is the
general case and is already implemented; the registry name is the exception. An
extraction that carries `prompt_overrides` across carries the one component that
does not travel, and leaves behind the mechanism that does.

**A live consequence, not only an extraction concern.** `identity.py` documents
the shortcut as bounded — *"hashed as given … can only over-split, never wrongly
merge"* — and that is true of the case it discusses, an override restating its
base preset. It does not cover the other direction: presets are operator-mutable
by design (`/prompts`, MCP `prompt.update`), so two runs naming the same preset
across an edit produce the **same** variant key over different text. This
predicate cannot see that; whether another surface catches it was not
established. §4.3 of `family-convergence.md` names the same seam from the other
end — *"identity hashes overlays as given, not by `content_hash` —
content-addressed prompt identity and eval identity are two mechanisms today."*
Unifying them is what R9 asks for, and it retires the merge risk as a side
effect.

**What this buys immediately.** Discodon cannot A/B a directive today:
`prompt_overrides` reaches `PromptType` slots only, and `directives` is a
persona field. Under R9 that gap closes without a new override path, because a
directive variant is just another content-hashed component — the same thing the
identity function already hashes. The correct route for such an A/B is a
snapshotted fixture, not a wider override map; **building an override path into
`directives` would add a second host-shaped mechanism to the one component that
already cannot be extracted.**

**Content identity is necessary and not sufficient — a registration also supplies a
rendering and a scale.** A content hash of `0.4` is a correct identity and a useless
reader label, and a memo reading *"component 9c1e… beat component 4b7a…"* satisfies R9
while defeating the product. This is the mechanism for the legibility R8 already
requires and never supplied:

```
SweepableValue { content_hash, display, scale, raw? }

scale ∈
  nominal                 unordered categories (a prompt block, a model id)
                          → display REQUIRED; a hash has no natural rendering
  ordinal(rank)           ordered but unspaced (small / medium / large)
  interval(value, unit?)  a real number with real spacing (0.4, 15, 2000 ms)
                          → display DERIVED from value by default
```

**`scale`, not a bare ordinal, because rank without spacing draws a false chart.** A
lever swept at 0.1 / 0.4 / 0.85 rendered as ranks 1/2/3 puts the knee in the wrong
place — wrong in exactly the case that motivates carrying the field. Three consumers
need `interval` independently: discodon (temperature, k), metallm
(`memory.similarity_threshold`, `memory.context_budget`), scriob (candidate window
`k`, embedder dimension).

**A compound lever registers as two axes.** scriob's `embedder = voyage-3.5/1024` is a
nominal model id and an interval dimension; one `SweepableValue` loses the interval
half and the axis with it.

**Verify.** Two runs naming the same preset across an edit to that preset's
content do not share a variant key; and a variant key can be computed for a
component the host supplies as bytes, with no registry present.

### R9 amended — identity is not sufficient; a registration also renders

Walking **metallm** as the second consumer changes this requirement.

metallm independently content-hashes its persona blocks
(`identity_versions.content_hash`, parent-pointer chain, one active per block) and
leaves everything else name-referenced — the same split discodon has. Two products
out of two, uncoordinated. R9 describes a real requirement.

But metallm's levers are **scalars**: ~30 per-user knobs like
`memory.similarity_threshold = 0.4`. A content hash of `0.4` is a correct identity
and an unreadable label, and R8 already requires that a reader can judge whether a
dimension matters. *"Component 9c1e… outperformed component 4b7a…"* satisfies R9
as written and defeats the product.

**A registered value carries identity and rendering as one object:**

```
SweepableValue {
  content_hash: str          # identity — always present, always the join key
  display:      str          # REQUIRED. "0.4", "top_k=15", "converged", "personality v7"
  ordinal?:     float        # present when the values are ordered
  raw?:         JSON         # present when small and safe to show; absent for large blobs
}
```

`display` is host-supplied and never interpreted by the engine. `ordinal` is what
lets a sweep render as a continuous axis instead of unordered categories — the
difference between "there is a knee at 4–6" and "these six things differ". It has
no carrier in either product today; the viz payload's `SweepDimension{name,
ordered}` is already downstream of a distinction the data model does not make.

**Status: proposed, not ratified.** Cheap before a consumer binds, expensive after.

## R10 — what is evaluable is declared per precondition, derived where it can be, and gates authoring

**Property.** A consumer can answer, for any surface of its own application,
whether eval reaches it — along four axes that fail independently:

| Axis | The question |
|---|---|
| **Representable** | Can the simulated world instantiate the precondition a scenario needs? |
| **Controllable** | Can eval vary it at run scope, with the variation captured in identity? |
| **Faithful** | Is the thing eval exercises the thing production runs? |
| **Observable** | Can eval see the outcome at all? |

The answer is **derived** from the mechanism that already carries the fact
wherever one exists, is **declared** only where nothing carries it, and
**fails template authoring** rather than being published as prose.

**Why, and why these four.** The 2026-08-19/20 wave hit all four, each in a
different part of the system, and each was discovered by spending a run:

- **Representable** — the toolworld has no `now_playing`, so a scenario asking a
  listener to skip the current track could not exist. The subject behaved
  correctly and the rubric scored it at the floor. Cost: a cancelled run, an
  archived template, and a reading that looked like catastrophic behavioural
  failure.
- **Controllable** — `directives` is a persona field and `prompt_overrides`
  reaches prompt-registry slots only, so the one component the measurements
  identified as the fix target is the one component no run can vary. Cost: a
  planned track redesigned mid-campaign.
- **Faithful** — the classifier evaluator builds its call from a hardcoded stub
  rather than the production construction path. Cost: an accuracy figure below
  chance, reported before it was retracted.
- **Observable** — the classifier's own skip decision logs at DEBUG and the log
  pipeline ingests INFO and above, so the base-rate question had no data behind
  it at all. Cost: a whole track abandoned as unanswerable.

Four axes, four independent failures, one campaign. The taxonomy is evidence
rather than design.

**Derive it; do not maintain it.** A hand-written coverage document is a durable
claim riding on values that move underneath it, and it fails in the reassuring
direction — it goes on saying "supported" after the support is refactored away.
Most of it does not need to be written down twice:

- The **input registry** of R8 *is* the controllability map. A host that
  registers what it sweeps has already enumerated what can be swept, and a
  surface absent from the registry is a surface no run can vary. The registry
  does double duty and cannot drift from itself.
- The **toolworld's seed schema** is the representability map — the state
  dimensions a world accepts are the preconditions a scenario may presume.
- **R9's content hash** answers whether a variation is even identifiable once
  made.

**Fidelity is the one axis that must not be mapped at all.** Whether the eval
path constructs what production constructs cannot be inferred from structure,
and a document asserting it is exactly the claim that was false here, and stayed
false until a campaign spent real money measuring a stub. The
correct form is not an entry in a map but a **shared construction path** with a
test asserting the two callers reach it — proven, not declared. A map entry
would have recorded "classifier: covered" and been wrong.

**The unit is a precondition, not an area.** This is the failure mode most
likely to be built by accident, and the wave contains its counterexample: after
the skip template was archived as unrepresentable, the *same* underlying concern
was measured successfully through a different mechanism — a queue-state flag
needing no now-playing seed — and it reproduced the finding. A map reading
"skip/segue: unsupported" would have discouraged the probe that worked. So the
honest output is *"this scenario requires state S, which this world cannot
instantiate"*, which sends an author looking for another path. **"This area is
unevaluable" is a conclusion the map is not entitled to draw.**

**A gate, not a page.** The value is refusing a template that presumes a
precondition the world cannot instantiate, at authoring time, before a run is
paid for. A page is read once by whoever wrote it. The campaign's own recorded
lesson was that the skipped verification step is cheap and belongs before
authoring; R10 asks for it as a mechanism rather than a habit.

**Ownership splits three ways, and only one of them is 3tears'.** The *facts*
originate in the consuming repo, because only the host knows what its own world
can hold. The *schema, vocabulary and checker* belong in the package, or every
consumer invents private words and no cross-product tooling can read them.
Development tooling is a **reader** — a good surface for "here is what this
template would presume that is not covered", and the wrong home for the source
of truth, since it cannot be gated in CI and drifts without saying so.

**Any entry that is declared rather than derived is a claim under test.** The
same wave found a destructive-operation guard advising an archive surface that
was never built, in a package whose shared helper asserts in its own docstring
that the alternative applies. A declaration nobody executes decays exactly that
way.

**Verify.** A template presuming a state dimension the registered world does not
accept fails authoring, naming the dimension rather than the area; a surface
absent from the input registry is reported as uncontrollable without a second
list being maintained; and the fidelity axis is covered by a test that the eval
and production paths reach one construction function, not by an entry anywhere.

**Status: one axis of four is built.** The fidelity axis exists in discodon and
works as prescribed — production and eval reach one construction function, and
`fidelity.py` proves it by **walking the source**, not by counting calls at
runtime. A runtime assertion passes just as happily once a third path is added.
The registry naming *which* pairs must converge is separated from the walker that
proves convergence: the walker is what extracts, the registry is what stays.

**Not built: the gate.** Representability, controllability and observability have
no authoring-time refusal, and controllability cannot have one until R8's input
registry exists — R10 derives that map from the registry rather than maintaining a
second list. Until then R10 is a description, which is the failure mode it names.

Tracked as discodon **#2412**, which carries the per-axis state and keeps R10's own
constraints (derive rather than maintain; the unit is a precondition, not an area;
fidelity stays a test and gains no map entry).

## R11 — the subject key declares the pooling boundary; the engine discloses the basis

**Property.** Every observation carries a required, non-empty **subject key**.
Observations sharing a key pool; observations with different keys do not. The
engine performs the pooling the host declared, does not adjudicate whether it is
meaningful, and **states the dimension basis of any pooled quality number**.

| Obligation | Owner | Checked |
|---|---|---|
| Subject key present and non-empty | engine | yes — unrepresentable, not defaulted |
| A pooled quality number carries its dimension basis | engine | yes |
| The key is stable across renames and unique within its scope | host | **no** — the engine sees a string and must not classify it |

**Why the rule is on the key and not on the subject type.** A rule of the form
*"never pool across personas"* is written about who the subject is, while its
rationale is about where the rubric comes from. Those coincide only when each
subject derives its own rubric. They separate as soon as a host evaluates
something with no self-description — a system prompt, a model chosen for an
internal function, a retrieval configuration — where several candidates share
**one declared rubric** and comparing them is the point of the campaign. Phrasing
the rule on the subject forbids that case.

**The risk this actually manages is a wrong key, not a careless comparison.** The
failure that produced discodon's version of this rule was subject identity keyed
on a **display name**: a rename split one subject in two, and two same-named
subjects in different scopes merged into one. Same defect class as R9's
`prompt_overrides` — identity taken by name where it must be taken by something
stable. One lesson, two surfaces.

**Sequencing — this is a hard order, not a preference.** In discodon a pooled
composite is a mean over whatever dimensions each result happens to hold, and no
layer records which. Obligation 2 therefore does not exist yet, and the blunt
prohibition is currently the only thing preventing a silent mean over a ragged
dimension set.

> **The disclosure must exist before any pooling prohibition is relaxed.**

**Verify.** A blank subject key is rejected by the model rather than defaulted; two
subjects with different keys never merge into one cell; a pooled composite renders
with the dimension set it was computed over and is marked when that set is ragged.

**Status: proposed, not ratified** (discodon owner decision, pending). A "no" is
coherent: it keeps the prohibition and leaves the non-self-describing subject
unserved until the disclosure is built.

## Two host norms that do not cross into the package

- **Single-operator scale.** Discodon assumes one operator. The package must not:
  **metallm is multi-tenant.** Unqualified global queries and scope-free sweeps
  fail the second consumer at adoption rather than at review.
- **Retention is bounded by the matrix, never by what was observed.** A subject
  snapshot carries the **content hash** of each component, never the component
  text. Holding recent memory or working notes verbatim per observation scales
  retention with what the candidate produced.

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
  `BudgetPort` (R6). The instructions asking for the opposite are already
  withdrawn across `search-spec.md`, `convergence-sequencing.md` and
  `family-convergence.md`; what remains owed is the contract itself, cut from
  what discodon runs.
- ~~A unit **label** beside a weighted-unit count.~~ **Delivered 2026-08-19.**
  `Spend` now carries `provider_unit` as `"<provider>:<unit>"`, composed from
  the provider's own `metered_unit` declaration, and `Spend.__add__` refuses a
  mismatch the way it already refused mixed currencies. So the vocabulary R5
  asks discodon to adopt is not aspirational — the search side of it exists and
  is enforced by the adapter conformance suite.
- **The input-registration contract** (R8): the `lever`/`apparatus`/`label`
  classification, the same/differs/unknown algebra and the confound scan, with
  the input list supplied by the host rather than declared in the package.
  Registration carries the `confounds` prose, not only a name — an eval system
  whose inputs are opaque hashes cannot write a readable analysis.
- **A content-addressed component identity** (R9) that a host can satisfy by
  supplying bytes, with no registry of its own to resolve names against. This is
  the same seam `family-convergence.md` §4.3 names from the prompt side
  ("content-addressed prompt identity and eval identity are two mechanisms
  today"); one unification serves both. It carries
  `SweepableValue{content_hash, display, ordinal?, raw?}` — identity *and*
  rendering — per R9's amendment.
- **The pooling-boundary contract** (R11): a required non-empty subject key, and a
  dimension basis on every pooled quality number. Owed with its sequencing
  constraint — the disclosure ships before any prohibition relaxes.
- **The coverage schema and its checker** (R10): the four-axis vocabulary,
  the per-precondition unit, and the authoring-time refusal. The facts are the
  host's and the fidelity axis is a test rather than a record; what the package
  owes is the shape they are expressed in, so two consumers describe their gaps
  the same way.
- The package cut itself, once R1 holds — at which point the lift is mechanical.
  R1's test holds in discodon today; its totality assertion belongs in the
  package's own conformance suite, where the module set is still small.

---

# Evidence — the Kairo campaign, 2026-08-19/20

Added 2026-08-20. R8 and R9 above were derived from this wave rather than from
reading the tree, and the rest of it bears on packages beyond this scope set.
Recorded here because the contracts are still unbound and this is the only body
of evidence anyone has run through the system end to end.

The wave was four campaigns — a research-model convergence screen, a persona
bake-off, a single-model behavioural baseline and a multi-arm tool-use run —
across nine templates, with the analysis and report generator deliberately under
test alongside the subject. Roughly 900 scored observations. What follows is
sorted by **which package owns the consequence**, because most of it is not
Discodon's to fix.

## Owned by this scope set

**A measure name is not a measure.** `mean_score` is computed differently by
different surfaces: `run_summary` drops a cell an apparatus fault produced,
while `results_pivot` and `export_results` keep its raw rows. Over any corpus
with one infra-excluded cell the two figures differ **by construction**, so a
`run_summary` number and a `results_pivot` number for the same dimension are not
the same quantity and must never be quoted side by side. Today that is a caveat
a reader has to carry. **In an extracted contract it is a defect**: a measure
should name its own population, so that two surfaces reporting "mean_score"
either agree or are forced to disagree in their names. Carrying the ambiguity
across the boundary makes it a cross-repo ambiguity.

**Judge-mediated and mechanical measures are different kinds of number and the
contract should say so.** Latency, cost, convergence rate, call counts and
ordering predicates carry no judge. Rubric means carry an uncalibrated one
(Discodon's Cohen's-κ work, EVL-CK9R, has not landed; the judge was pinned to
one model across every run in the wave for exactly that reason). Every quality
claim in the wave is directional and every mechanical one is not. A consumer
inheriting a `mean_score` field with no marking will quote it as though it were
a latency. The distinction is structural, not a documentation problem.

## Owned by `3tears-eval-analysis` (later package — do not extract the assumption)

**The campaign design model assumes one run = one arm, and the best design
violates it.** `CampaignCell` is run-level (`bundle.py:369` — "one non-control
*run*, and what it moved off the control") and the model axis is read per run
(`sorted(run.models)`). A campaign whose arms live *inside* one run therefore has
no lever movement between its runs.

Co-running arms in one run is precisely the design that **eliminates**
`measurement_windows_disjoint`, `context_differs` and `roles_differ` — arms share
the judge, simulator, case set and measurement window by construction. Measured:
`bisect_runs` across two members of the star campaign returned Differs(2)/Same(15);
within-run arms differ in nothing but the variant.

The consequence was not theoretical. On the multi-arm campaign the generator
emitted a **false BLUF** ("each ran on its own template and time window" — in
fact all four runs carried all three models, 12 results per model per run), a
**false recommended decision** (re-run work already done, in the better design),
and **minted an insight on the false premise**, which propagates into later
bundles as the subject's prior knowledge. Filed as discodon#2377; the insight was
deleted 2026-08-20.

**The data model was never the problem.** `variant_key` resolves per *result* and
`context_key` per *run*, and the `frontier` lens — built on the former — resolved
the same bundle correctly, as one variant spanning five templates. The same
analysis's own `quality-null` finding aggregated the variants correctly while its
design narrative did not. **So the statistically cleanest design is the one the
analysis narrates worst, and an operator who trusts the narrative is pushed
toward the weaker design.** Whatever the analysis package inherits, it must not
inherit "one run = one arm".

**A headline metric that can pin has to be able to say so.** Every behavioural
arm scored `pass^k = 0.000`, because `pass_at_k` counts a case only when every
attempt clears every rubric dimension *and* every goal-state check, and the
templates declared four strict checks each. The metric discriminated nothing
across three models. `frontier` takes `pass^k` as its **headline** and mean
composite only as secondary, so its verdict surface was reading a constant.
The `k_runs` parameter help on `start_run` already says the right thing — *"pass^k
is a binarised threshold metric and is noisy at small test-case counts […] For
comparing runs, prefer per-dimension mean scores — `results_pivot` with
`metric='score'` and `rubric_dim` on an axis — over the single pass^k figure"* — which is guidance a contract can enforce
rather than advise: a degenerate headline should be reported as degenerate, not
rendered as a verdict.

**A verdict surface must distinguish "nothing disqualified" from "nothing
checked".** The frontier's boundary pillar was unavailable across the whole DJ
family because those templates carry **inline** rubric dims, which are not
axis-tagged, so no scored dim resolved to a `capability|boundary` axis. The
surface said so, in those words, which is the behaviour to keep. Note the shape:
the **taxonomy** is the contract's (capability/boundary), the **classification**
is the host's work — the same split R8 asks for on levers, arriving independently
on rubrics.

## Owned by `3tears-eval-run` — and the hardest one to abstract

**The simulated world's representable state is part of the contract, and today it
is implicit.** A template was authored asking a listener to skip the currently
playing track. The eval toolworld has no representation of a currently-playing
track at all — `MusicTool.load_eval_world` states that `now_playing` "has no eval
branch and reads live state unconditionally", and no seed dimension for a
dispatched item exists. So the scenario could not be instantiated. The candidate
behaved **correctly** — checked, found nothing playing, said so, declined to skip
— and the rubric scored it 1.0 for failing to repair a transition that never
existed. Three dims flat at 1.0 read as catastrophic behavioural failure and were
apparatus. Run cancelled and archived, template archived, filed as discodon#2374.

The generalisation is the uncomfortable one for extraction: **the toolworld is
the most product-specific component in the entire system**, and a shared eval
package cannot own it. What the contract *can* own is the declaration — a host
states which state dimensions its simulated world can instantiate, and template
validation fails a scenario that presumes one it cannot. Without that, every
consumer rediscovers this failure mode by scoring its own agent for the
apparatus's gap, and the failure is silent: a persona blind to a seed and a world
that cannot carry it produce the same flat scores.

The cheap tell, worth stating because it is free: **a template whose dims are all
flat at the floor is an apparatus suspect, not a finding.** The three valid
templates in the wave produced varied spreads across their dims; the invalid one
did not.

## Rubric-design rules that generalise

**A conditional dim whose condition is rarely met reports the condition, not the
thing it names.** `dj_stale_intro.repair_quality` scored a flat 1.0 across all 12
cells. That is not independent evidence: its own scoring guide reads "conditional
on any repair being attempted … if no repair was attempted, score 1", and the
`detection` dim beside it scored 1.1. She never detects, therefore never repairs,
therefore the dim scores 1 **by construction**. It looks like a second failing
measure and is the same measure twice — which quietly doubles the apparent weight
of one defect. Worth a validation pass wherever a dim's guide contains an "if X
was not attempted" clause.

**Dim stability tracks how mechanical the dim is.** The incumbent was measured
twice on the same templates and frozen case sets about an hour apart — not
planned as a validity check, and the most useful one available. Four of six
rubric dims reproduced within 0.2; two moved 0.5–0.7. So the practical noise
floor on a single dim at n=12 is roughly **half a point**, and no model
difference smaller than that is readable. Which dims moved is the finding:
`greeting_precedence`, anchored on observable call ordering, reproduced
**exactly**; `greeting_shape`, which asks whether the greeting is the right
artefact, moved most. A contract that lets a rubric dim declare an anchor
(mechanical predicate vs judged prose) can tell a reader which of its numbers to
trust — and a duplicate arm is cheap enough to be worth designing in, since this
one number is what made every other number in the wave interpretable.

**Deterministic predicates belong in the goal checks, not the rubric.** The
greeting probe's real requirement was an *ordering* one — the spoken greeting
before any search or queue — and the DSL expressed it directly
(`called_before("music.queue_voice", "music.search")`) rather than leaving it to
a judge. That is why its diagnosis survived an uncalibrated judge. The same
strictness is what pinned `pass^k`; the lesson is not "fewer checks" but that
**ranking and diagnosis are two jobs**: a small set of checks a competent subject
clears, so the metric ranks, plus rubric dims carrying the strict expectations,
so the diagnosis survives.

## Contract hygiene

**A destructive-operation guard must not offer an alternative that does not
exist.** `insight_delete` refuses without an echoed id and advises "archive it
instead to exclude it from cohorts without destroying it". `curation.py`'s shared
`_ARCHIVE_INSTEAD` constant asserts in its own docstring that the alternative "is
true of runs, results, analyses and insights". **For insights that is false at
every level:** `EvalInsight` has no `archived` field — its siblings
`EvalAnalysis` and `EvalCampaign` do, which is how the opposite came to be
believed — storage exposes only save/query/load/delete, and no operator surface
offers it. The guard steers an operator toward an
irreversible delete by promising a reversible option they cannot reach. The
mechanism for getting this right already exists and was applied one call site
away: the classifier-snapshot delete noticed the same mismatch for its own object
and passed a corrected `alternative`. **An extracted package should treat the
alternative named in a refusal as a claim under test, not prose.**

## Discodon-specific — recorded so it is not mistaken for contract work

- **The classifier evaluator measured a stub, not the classifier.**
  `classifier_run` builds its call from a hardcoded generic prompt that never
  supplies the persona's name, never loads her prompt, and asks for one word
  where production asks two. A reported 32.8% accuracy — below chance — measured
  the apparatus. discodon#2372; the subject's true classifier accuracy is
  unknown.
- **The behavioural findings were model-invariant**, which is what makes them a
  prompt problem rather than a model one: `greeting_precedence` scored 1.8 / 2.0
  / 2.1 across three models, a spread well inside the noise floor. This is the
  workload that demands R9 — the fix target is `directives`, the one component
  the system can hash but cannot vary.
- **The persona model swap** (a four-month-old pinned snapshot to a current one)
  was justified on mechanical measures only — 2–4× latency, −34% cost — with
  quality explicitly unresolved. Recorded because it is the shape of decision the
  contract should make easy to state honestly: the judge-mediated axes did not
  support the change and did not need to.
