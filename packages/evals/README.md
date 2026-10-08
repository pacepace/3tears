# 3tears-evals

Evaluate an LLM-backed product the way you would run an experiment: declare the levers you can
change, run trials of each variant against a corpus of cases, grade each trial with code checks and
model judges, and read an analysis report that says which variant is better, by how much, at what
cost — and when the evidence cannot tell.

**Status: being extracted.** This package was cut from a production app's in-tree eval engine and is
being reshaped into a library any app can adopt. Its public API will change without notice until the
first release that says otherwise.

| Subpackage | What it holds |
|---|---|
| `threetears.evals.contracts` | the data models, the host contract an app implements, identity and scoring rules |
| `threetears.evals.run` | the trial loop, judges, simulated users, budgets and metering |
| `threetears.evals.gen` | case and rubric generation |
| `threetears.evals.analysis` | the analysis bundle, report generation and charts |
| `threetears.evals.storage` | the storage adapters the engine ships: the in-memory reference store |
| `threetears.evals.testing` | conformance kits an app runs in its own test suite: the store kit |
| `threetears.evals.quick` | the batteries: `run_eval` in one call, and the `python -m threetears.evals` command line |
| `threetears.evals.ops` | typed operations over a host, and one job contract for long work |
| `threetears.evals.actions` | the action catalogue every transport mounts: `evals` and `evals_admin` |
| `threetears.evals.transports.fastmcp` | the catalogue as FastMCP tools (extra `fastmcp`) |

Import from those roots and from `threetears.evals.contracts.host`, never from a module below
them. Every engine type a public signature hands you — a protocol you implement, a value you
receive, an exception you catch, a literal you annotate with — is exported from one of those roots.

Two example hosts live in this repository (not in the wheel), written as reference code that
imports nothing but the public roots and itself. `tests/fixtures/courierhost/` is the least a
product writes when it builds its own host, in one module; `tests/fixtures/toyhost/` exercises every shape of the host contract,
with a map of which file holds which step. Both run on `InMemoryDocumentStore`
(`threetears.evals.storage`), the engine's in-memory reference `DocumentStore`: scoped, with
conditional writes, and the shape to compare your own adapter against.

## Guides

- [Designing a classifier eval set](docs/designing-classifier-evals.md): the labels, the kinds of case a set
  needs (boundaries, lookalikes, contrast pairs, context), how many, and how to read the results. Start here
  if you have not built an eval before.

## Rung zero: one call

A function to test, cases to test it on, and code that grades an answer are enough:

```python
from threetears.evals.quick import run_eval

async def classify(case: dict) -> str: ...

def correct(case: dict, label: str) -> bool:
    return label == case["expected"]

summary = await run_eval(cases, classify, [correct], scope_id="dev", k=2)
print(summary.render())
```

`run_eval` builds the rest — a kind over the function, a host with one measure per scorer, the
in-memory store — launches one run through the engine's own launch path, and returns its
`EvalSummary`. A candidate that raises fails its cell; a scorer that raises excludes it. Pass
`host=callable_host(scorers)` to keep the store and compare several candidates' runs, or your own host:
it must declare a measure per scorer and a contract for the callable kind — `CALLABLE_KIND_CONTRACT`, or
a `KindContract(CALLABLE_KIND, seats=...)` seating only apparatus of your own that the runs read, never the
judge, the simulator or the spend ceiling (`CALLABLE_UNSEATED`) — or `run_eval` refuses it, since without
one every such run's blank judge and simulator read as unrecoverable and no two of them compare.
`examples/rung_zero.py` is the whole thing in one file.

**The command line** works in a host you name as `module:factory` — a zero-argument callable
returning an `EvalHost`, or a `LaunchHost` for `run`:

```
python -m threetears.evals run    --host myapp.evals:build_host --scope dev --template T --subject S [--model M ...]
                                  [--k N] [--max-cost-usd DOLLARS] [--judge-model MODEL] [--simulator-model MODEL]
                                  [--n-variations N] [--variation-model MODEL] [--apparatus-settings JSON]
python -m threetears.evals ls     --host myapp.evals:build_host --scope dev
python -m threetears.evals report CAMPAIGN --host myapp.evals:build_host --scope dev [--format markdown|html|json] [--out PATH]
python -m threetears.evals bundle CAMPAIGN --host myapp.evals:build_host --scope dev
python -m threetears.evals spend  --host myapp.evals:build_host --scope dev [--purpose P] [--launch-group ID] [--template ID]
```

`run` launches, waits and prints each run's summary. Each `--model` is one arm and one run; with no
`--model` the kind runs one arm on its own default model, and a kind with no default refuses the launch.
`--k` is the repeats per case (the launch default when omitted); `--max-cost-usd` caps each run at or
below the host's ceiling (a launch may only lower that ceiling; a value above it is refused); `--judge-model` and `--simulator-model` pin the judge and the simulated user
(omitted, the kind's defaults apply); `--n-variations` and `--variation-model` generate that many cases first (priced against
the host's out-of-run cap, outside the runs' caps); `--apparatus-settings` sets host-declared apparatus
values as a JSON object — each as `start_run`'s argument of the same name. `report` prints the campaign's
report (below) — its analysis, or, when it has none, a code-only report of its evidence; `bundle` prints the
analysis bundle a generation would read, as JSON. Neither calls a model. `spend` prints what the engine
spent outside any run in the scope — case generations, rubric proposals and analysis generations — narrowed by
its flags.

Exit codes: `0` done; `1` a launched run did not complete; `2` refused (a host that cannot be loaded, a
template that is not there, a launch the engine refuses, a malformed command line); `3` failed on an error
nothing anticipated — a host factory, launcher or host command raising — with its traceback on stderr.
They are `EXIT_OK`, `EXIT_RUN_DID_NOT_COMPLETE`, `EXIT_REFUSED` and `EXIT_FAILED` in `threetears.evals.quick`.

Mount the same commands under your own CLI with `run_cli(argv, host_factory=build_host, prog="myapp
evals")`; your users then never name the host.

## Adopting it: the host, the scope and the kind

**Your app is one value, the host.** Build an `EvalHost` (in `threetears.evals.contracts.host`) and
pass it to every entrypoint. It holds your `HostProfile` — the levers you sweep, the measures you
record, your bars and your world — the `EvalStorage` you read and write through, a factory for the
completion clients the engine's own judge, simulator and analysis roles call, your
`failure_describer` (how a raised provider call reads: the only thing that can say a call was refused
for the calling account, which stops a run rather than excluding its cells; `withhold_failure_detail`
is the honest one for an app with no error types of its own), and your tracing, executor and
cell-timeout choices. None of the last four has a default: each is named where the host is built. Nothing is ambient: there is no installed or default host, so two
hosts in one process never meet. An app that starts runs builds a `LaunchHost` (in
`threetears.evals.run`) around its `EvalHost`: the launch settings, a registry of the kinds it can
launch, and a job timeout; it builds its job manager over the host's own storage. Hand its
`eval_host` to the analysis side.

**A kind's launcher resolves only what is the kind's.** `start_run` refuses a launch argument the
kind declares it cannot honour before the launcher runs, then hands the launcher a `LaunchRequest`.
The launcher captures the subject the request names, freezes the cases, builds its judge
(`build_judge_service`) if it has one, and returns `launch_run(host, request, KindWiring(...))`. The
engine stamps everything the request and the template already say — scope, model, repeats, cassette
mode, overlays, spec, world seed, tool bound, ceilings — and refuses a wiring that contradicts them.

**Generating cases at launch.** A launch with `n_variations` > 0 asks for that many new cases from the
template's variation axes, generated once for every arm: call `generate_variations` inside
`request.launch_group.resolve_once(...)`, passing `blocking_executor=host.blocking_executor` (its case reads
and writes run there; its model calls stay on the loop), and hand its counts on as
`KindWiring(variation_counts=...)`. An
`llm` axis is written by the model the launch names as `variation_model` — required then, refused when
nothing would call it — which the launcher asks of the host's client factory in the `variation` role
(`clients("variation", request.variation_model)`), never the simulator's. The run records the model the
client resolved to on `variation_counts.variation_model`; it enters no identity, because the cases it
wrote are already hashed through `test_case_ids`. Generation runs before any run exists, so its calls
are outside every run's cost cap and metered-call ceiling — which is one reason the engine prices every
arm before any launcher runs (below). The launcher hands `generate_variations` the request's
`budget=request.generation_budget`: every `llm` axis's call is priced on the writer's client
(`price_ceiling`, the host's answer) against `LaunchSettings.max_out_of_run_cost_usd` before the first
is made, and each is ledgered as an `OutOfRunSpend` document (`EvalStorage.query_out_of_run_spend`)
under the launch's group, written on the host's blocking executor (an `OutOfRunBudget` names its
`blocking_executor`, as an `EvalHost` does). `propose_draft` takes a budget the same way. A battery prices every
template's arms and every template's writer calls (on the host's `variation` client, against the budget
each launch will be held to) before any template launches, so it pays for no template's cases until all
have been priced; its caps are per launch, as a launch's are (`start_universal_battery(max_cost_usd=...)`
names the per-run cap). What was spent out of run is read back by `scope_out_of_run_spend` — the
`scope_out_of_run_spend` action, and the CLI's `spend`.

**Every arm is priced before any launcher runs, by one rule.** The engine asks the kind what each arm
will run (`LaunchableKind.plan_arm` → `ArmPlan`: for an arm over stored cases, how many of the template's
stored cases it plays, for a generating arm at most `n_variations`; the model; the judges it will be scored
by, resolved with `plan_judge`; and its simulator) and, under an enforced cap, prices it with the host's
`LaunchHost.launch_pricer` (`ArmQuote.case_source` says which; `threetears.evals.ops.history_launch_pricer`
bounds it by the upper end of the band of runs launched the same way — template, model, cassette mode, the
model each scored dim was judged by, the simulator that ran, resolved apparatus settings) and refuses an arm
predicted above its run's cap, or one nothing can predict — no pricer, no plan, or no history to bound —
whose cap the run would merely inherit. An unpriceable arm under a cap the launch named runs under it.
Every arm is planned before any is priced, so `plan_arm` is where a kind makes its request-level refusals
(no model and no default — `require_candidate_model`, the tail's own refusal — a judge or simulator it
needs): the operator hears those before any "cannot be priced". The launch tail holds each launcher to its
plan (no more cases, no other model, judges or simulator), and refuses an arm that was never priced under an
enforced cap — a host composing its own launch through `launch_as_group` prices its arms with `price_arms`.
A battery prices each template's arms once, in its pre-flight, prepares every template before starting any,
and launches each template as it priced it. `launch_estimate` (`quote_launch` in `run`) runs the same steps
read-only and reports each arm's price and the launch's verdict, word for word; hand its result to a cost
pivot as `predicted_cost`, and each prediction sits only in the cell of its model and template. Once the launch
ran, pass its run ids as `launched_run_ids` too, and each predicted cell says how many of its observations came
from other runs — the history the prediction was drawn from among them. A host therefore prices no arm itself: a wrapper that priced assembled runs would
be a second rule, and a second pricing of the same arm.

**Setting the rig at launch.** `apparatus_settings` sets host-declared apparatus values — who sits in an
adjudicator's seat, say — so one template can be run at two of them and compared. A kind lists the ones
its launcher reads, each with the value its rig takes when a launch sets none
(`LaunchableKind.apparatus_settings={"adjudicator_seat": "model:default"}`, each a non-engine `apparatus`
declaration of the host). The dispatch resolves a launch's settings against those defaults, so the
launcher reads every one off `request.apparatus_settings` with no default of its own, and the run records
the rig as set up (`EvalRun.apparatus_settings`), a component of its measurement context — a launch naming
a default and one leaving it out are one condition.

**A host with no metered tools** sets `LaunchSettings.max_metered_calls=None`: its runs record a
ceiling of `0` with origin `none_declared`, and a metered call that happens anyway is refused and counted.

**Tenancy is one opaque `scope_id`.** Every stored document carries a non-empty `scope_id`, and the
engine never interprets it, defaults it or branches on it: it is your tenant, project or
environment, whatever you partition by. All of it goes through one `DocumentStore` you implement,
keyed by `(scope_id, doc_type, id)`. Every port call names its scope, and there is no scope-free
read: a caller that needs several scopes is told which by you and asks each. A campaign and the runs
it compares live in one scope. Your adapter strips whatever it injects (an etag, a timestamp) before
handing a document back, because every stored model reads strictly.

**Prove your store with the conformance kit.** `threetears.evals.testing.STORE_CONFORMANCE_CASES`
states every rule of the port — scoping, the strip on read, `exclude`/`keep` projection, ordering,
etag conflicts and the re-read that recovers one, merge and delete — as a case you hand a fresh,
empty store. Every case is mandatory: in particular a store must implement conditional writes,
because several writers share a run document as it finishes. Parametrise your test runner over it:

```python
@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_my_store_conforms(case: StoreConformanceCase, tmp_path: Path) -> None:
    case.run(MyDocumentStore(tmp_path / "evals.sqlite"))
```

The engine reads and writes through `EvalStorage`, built over that one store. Its consumers name the
area they touch rather than the whole of it — `RunStore`, `ResultStore`, `RunRecordStore`,
`DefinitionStore`, `CassetteStore` and `JobStore` (in `threetears.evals.contracts`) — so a function
typed `DefinitionStore` cannot reach a run, and a test of one hands it only that area.

**A kind is what you are evaluating.** You implement the candidate-kind seam: `prepare` builds a
candidate for one cell from the run's subject, its variant configuration and the seeded world, and
`invoke` runs it on a test case and returns a `CandidateOutput`. What a kind adds to the engine's
own fields is two Pydantic models you name once, on the profile's `kinds`, in a `KindContract`. Its
**overlays** are the knobs a launch may turn: validated by field before any run exists, frozen onto
the run, and read back as levers whose levels enter the variant key. Its **spec** is what a template
of that kind declares (`kind_spec`), such as a label set or a table setup: refused by field at
authoring, then validated again and frozen onto each run. You register neither anywhere else: the
profile adds the contract's levers to its `sweepables`, and the engine resolves every run's level of
them — with its `candidate_model` and its `candidate_kind` — into the variant key. Your profile's
`host_sweepables` (the shared core extended with your own levers) and its `variant_levers` reader
cover only the levers you declare beyond those; a host with none wires no reader. A run is one arm
with one `candidate_model`; a launch naming several models starts one run per model, and runs of
different kinds are always different variants.

**A world, through the cell's session.** A host whose subject lives in a stateful world declares it
on the profile (`WorldRegistry`), and each cell's `prepare` is handed a `WorldSession` over it as
`world` (`None` on a host with no world). The kind seeds through it — the seed walk, the dimensions'
own `seed` handles, then each attached carrier's `settle` handle before the first turn — announces
each turn (`at_turn`, which applies any ambient perturbation the template's seed scheduled), and fires
or observes its triggered dimensions (`fire`, `observe`). Once `invoke` returns, the runner reads every
attached dimension back through its `read` handle and stores it as the cell's end state; what fired is
stored on the result as `world_events`. The profile's registry is the declaration and its handles are
the world conformance proves; if your world is real per-cell state, declare
`WorldRegistry(..., binds_per_cell=True)` and have each cell's `prepare` call
`world.bind(<that cell's handle table>)` before `seed` — every call the session makes then lands in that
cell's world, and a session that seeds unbound is refused. A goal check reads the end state as `state.<dimension>`, the
calls the kind recorded on its `CallLedger`, and what fired as `fired("<dimension>")`; grade them with
`grade_goal_checks`, and a stored run re-grades from all three with `recheck_goal_states`. The runner's
session is `commissioned`; a host grading a cell it witnessed through a session of its own constructs
`WorldSession(registry, provenance="witnessed")`, so `fired_armed(...)` is not established there — no
seed armed the session — and `record_witnessed_cell` refuses any world event claiming `armed=True` or
`caused_by="rig"`.

**What the judge reads, the kind renders.** A judged kind returns `JudgeEvidence` with every
non-empty output: the subject as the judge should see it, the case material, and the artifact (for a
conversation, the transcript as your kind writes it). The engine places those strings and reads none
of them, so hidden information and per-player visibility are your kind's rules. The evidence is
stored on the cell's `EvalTrace`, and a re-judge sends exactly what the first judge read. A
conversing kind's template carries a `ConversationSpec`: its simulated actors, who speaks next, and
the turn limit. A document or classifier template carries none. The kind runs it with
`drive_conversation(driver, candidate_turn, post_user_turn, llm=..., sink=sink)`, handing over its cell's
sink: before every paid call the loop asks the run's cost cap whether the simulator's spend so far
reaches it (`CellSink.cost_cap_reached`), and stops `budget_stopped` when it does — the runner excludes
that cell and ends the run `budget_stopped`, so one conversation (thousands of simulator calls at the
schema's maxima) cannot run far past the cap. Fold the driver's calls with `fold_usage`: the stored
`simulator` rows are one per actor and purpose (`RoleUsage.actor_id`, `RoleUsage.purpose`).

**Background work, payloads and spend.** Work a candidate hands off and gets back turns later is
recorded as `async_deliveries`, one `AsyncDelivery` each: who asked, when it was acknowledged and
delivered, on what model, whether a harness supplied it, and what it spent — its tokens, model calls,
`cost_usd` with its own `price_source`, and any paid non-LLM calls (`external_spend`) — their calls, the
provider's own units, and `money` where the provider itself billed a figure. The engine folds
that spend into the result's `inner_agent` and `external` usage, for work still in flight when the
cell ended as well as work that delivered; a substituted entry reports no spend. Anything else your
kind wants kept with a result goes in `kind_payload`, which the engine stores and never reads;
register a measure for whatever should be compared. Each completion your client returns names its
own `price_source`; the engine stores what it is told and never assumes a provider. Your kind reports
its own calls' spend only as usage rows (`CandidateTelemetry.usage`), each call's dollars as your
client priced them; the engine derives a result's `cost_usd` from those rows and its background work's.
**Unpriced is a state, never zero**: a call your client could not price (a local model, say), or
background work's paid calls that report no `money` and that a run with declared rates has no rate
for, leaves the result's `cost_usd` as `None`. A reported `money` wins over the run's rate. Every cost aggregate leaves such a result out of its dollars and counts it beside them
(`n_cost_usd`, `n_unpriced`), and a capped run stops on its first unpriced result, because a cap cannot
enforce a ceiling on spend it cannot count. An uncapped run carries on and counts them.

**Cassettes record and replay through seams the kind supplies.** Launch a run with
`cassette_mode="capture"` to run its tools live and record what they answered into that run's own
corpus, or `"replay"` with `cassette_corpus_id` naming a capture run to be served that corpus instead.
The engine builds the run's lane and hands each cell's `prepare` a `cassettes` handle (`None` with
cassettes off) already bound to the corpus, template and case, so one kind instance serves every
cell. Your kind calls `cassettes.wire(seams)` once with its candidate's `CassetteSeams`: an
`ActionSeam` for the synchronous tools to wrap, and a `DeliverySeam` for each asynchronous tool, keyed
by that tool's name. A wrapped tool is a `ToolLike`: a `name`, `can_dispatch(action)` and an async
`act(action, parameters)`; a tool that is a plain function is adapted to that shape. A candidate whose
turn loop calls its tools as plain blocking calls declares a `SyncActionSeam` instead (`arm_sync_tools`),
its tools are `SyncToolLike` (`act_sync(action, parameters)`), and it calls the wrapped tools'
`act_sync`. Both seams record and replay through one implementation, so a corpus captured on either
replays on either. A delivery seam reports each piece of background work where it starts
(`recorder.started(request)`) and settles the ticket where it ends; under replay it takes
`replay.next(request)` instead of starting live work, and reports what it was served with
`substituted=True`. Every recording is keyed by what was asked and by which time it was asked, so a
repeated dice roll replays both rolls in order and two scouts are each paired with their own report;
an ask the capture never made, or made fewer times, stops the cell as the rig's failure rather than
running live. A kind that does not wire the handle it was given, or a candidate with no seams, is
refused rather than run live under a replay.

**A broken rig costs one cell, never the run.** A replay miss, a corrupt recording or any other
fault of the measuring rig rather than of the world under test is an `ApparatusError`; your kind's
tool boundary re-raises it instead of turning it into an ordinary tool failure. Out of `prepare` or
`invoke`, the engine records that cell excluded (termination `apparatus_failed`) with whatever spend
the kind had reported through its sink, and goes on. A run whose every cell the rig excluded measured
nothing, and ends `failed`. A cell cut off by its deadline or by its run's cancel is recorded the
same way, from its sink; when the cut lands while the cell is being judged, the evidence and the
scores that came back are kept and the unfinished dims are the cell's judge errors, which a re-judge
can repair. A re-judge re-asks only dims that errored: a judge's "can't tell" is an answer.

**Stored data is disposable, and reads are strict.** Every stored model refuses an unknown field, a
missing required one, and a document written under any schema version other than this build's
`EVAL_SCHEMA_VERSION`. There is no migration and no tolerant reader: across a schema change, drop
the eval documents and regenerate them. Identity keys carry their own `IDENTITY_VERSION`, so keys
from different predicates never silently pool.

## Reading an analysis: the report

A generated analysis is read through one document. `analysis_report(storage, analysis_id, scope_id)`
returns a `Report`: an ordered list of blocks — `text` (what the analysis's author wrote, with a role),
`table` (evidence, arms, decision surface), `chart` and `disclosure` (what code must add) — each linked
to the findings it belongs to or rests on. Serialize it three ways:

```python
from threetears.evals.analysis import analysis_report, report_html, report_markdown

report = analysis_report(host.storage, analysis_id, scope_id)
report.to_canonical_json()   # validated by the published schema, report/schema.json (see below)
report_markdown(report)      # for an agent, or to paste as a memo
report_html(report)          # a page that reads without a script
```

**The campaign's report** is `campaign_report(host, campaign_id, scope_id)` — the one answer the CLI's
`report` and the `report_read` action both give: the campaign's newest analysis that is not archived, or,
when it has none, a **code-only report** of its evidence (`build_code_only_report`). That one has
`basis="code_only"` and no author's words — no headline, no findings, no text block, which the published
schema and the model both refuse — and holds the arm table (every arm unresolved, since nothing decided), the decision surface,
the contrasts the evidence tested against the control, a distribution chart per measure and judged
dimension, and every disclosure the evidence carries, opening with a statement that no analysis was
generated and what one would add. `Report.basis` says which a report is; `REPORT_VERSION` is 2.

**What the schema checks, and what only the model does.** `schema.json` holds the report's shape and every
cross-field rule JSON Schema can state: a code-only report names no analysis or model and holds no headline,
finding or text block; an analysis report names both; a report with no findings links no block to one; a
finding's own words name their finding; a chart block carries exactly one of an intent and an error, the
intent of its own type. Three rules compare a value with a sibling's, which JSON Schema cannot: a block's
finding positions are below `finding_count`, a table's `total_rows` is at least the rows it shows, and a row
keys only its table's columns. Those only `Report.model_validate` holds, so a host that validates against the
schema alone accepts exactly those three malformations as well.

A chart block carries the chart's **intent** (`ChartIntent`, from `threetears.evals.analysis.viz`), never
a charting library's spec: its type from eval's eight, the rows it draws, what each field encodes (identity,
length, position, interval with what it varies over, level, class, ordinal), its axes with their units and
zero baselines, its order, the colour *slots* it uses and what it must disclose — plus its values as drawn,
which the HTML shows as a table. How a chart looks is the host's: a renderer reads the intent and the
host's palette — `StyleProfile.chart_palette`, a renderer-neutral `ChartPalette` (the eight numbered series
slots, slots 1-4 validated; a sequential ramp; background, ink, muted, grid, rule, context and on-fill),
every colour resolved `#rrggbb`. The presentation rules are checked on the intent (`check_intent`), so they hold for any
renderer, and the core ships no charting library.

### How far a judged score can be leaned on: evidence tiers

Every judged reading — each `judged_measures` arm, each judged reading on the decision surface, each
judged evidence row of a finding — carries an `evidence_tier` that code decides from what the judge's
reliability was measured to be (`threetears.evals.contracts.evidence_tiers`, owner ruling 2026-10-06):

| Tier | When |
|---|---|
| `calibrated` | the judge agrees with people: `judge_agreement` (person ratings only) at least `CALIBRATION_MIN_AGREEMENT` (0.6) over at least `CALIBRATION_MIN_RESULTS` (20) distinct results |
| `separation` | the judge agrees with itself: `judge_self_agreement` at least `SEPARATION_MIN_AGREEMENT` (0.8) over at least `SEPARATION_MIN_RESULTS` (20) distinct results |
| `incidental` | both measured over enough results, and both missed |
| `undetermined` | too little evidence to decide — never filed as incidental |

Agreement is one statistic computed by one rule for both — quadratic-weighted kappa on 1-5, kappa on pass/fail,
per rater (each person; each round of repeats) and pooled by result — every distinct result weighs 1, split across
the raters that measured it — so the figure weighs what the floor counts, distinct results, never pairs: neither a
small rater nor many raters re-measuring a few shared results (five annotators on the same three anchors; one result
repeated thirty times) can carry it, or the floor, over the bar; and a repeat that answers "can't tell" where the judge had scored is a disagreement, never set aside. A
judge is a served model and a judge config, so a tier measured under one prompt never sets another's. The bundle
lists each judge's tier per dimension with both criteria
(`judge_evidence_tiers`); a finding stands on the weakest tier among its rows (`FindingResolution.evidence_tier`:
`mechanical`, `calibrated`, `separation`, `undetermined`, `incidental` or `none`), which every report states
beside the finding; a code-only report also states each judge's tier with the numbers behind it. Tiers are
flagged, never a reason to drop a reading.

Self-agreement is measured by **repeating** a finished run's judge scores: `repeat_judge_scores` (operation
`judge_repeat`; `estimate_judge_repeat` / `judge_repeat_estimate` price it without a call) asks the same judge
the same question again from the evidence its first judge read, under the apparatus the run recorded, and
records each answer beside the score it repeats (`EvalResult.judge_repeats`) without changing the scores. Every
call it can make — parse retries included — is priced and admitted against the host's out-of-run cap before
the first is sent, and each is ledgered under purpose `judge` with the run's id.

### Drawing charts: the Vega-Lite adapter

The package's own renderer is an optional adapter, `threetears.evals.vega`. Install the extra for its
rasteriser (`vl-convert-python`):

```bash
pip install "3tears-evals[vega]"
```

```python
from pathlib import Path

from threetears.evals.analysis import finding_chart_intent
from threetears.evals.vega import VegaRenderer

# The host's declared palette, bound once; a host declaring none draws in the packaged "dark" palette.
renderer = VegaRenderer.for_style(host.profile.style, theme="dark", font_dir=Path("/srv/fonts"))
intent = finding_chart_intent(host.storage, analysis_id, scope_id, "0")  # or a chart block's `intent`
chart = renderer.draw(intent)   # a colourless Vega-Lite spec, chart.spec, for a browser to embed...
renderer.config()               # ...with this config beside it
renderer.png(chart)             # or rasterised, for a surface that cannot run a browser
```

Drawing a spec needs nothing past the core; only `png` and `svg` need the extra. Nothing in the core
imports the adapter.

A host bringing its own renderer implements `ChartRenderer` (`draw(intent)`, and `drawn_data(drawing)`
reading its drawing back) and runs the one conformance check every renderer passes —
`assert_renderer_conforms(renderer, intents)`, from `threetears.evals.analysis.viz`: what it draws agrees
with the intent's marks (`data`), per identity, and the intent's values-as-drawn table agrees with those marks
(`table_disagreements`, policy rule 12) — so a drawing that passes agrees with the table beside it. A table
column spelled from a drawn number (a delta's `+72.7%`) or carried only by the table is the builder's to
spell, and is compared with nothing drawn.

## Driving it from an agent: operations, actions and MCP

Every surface calls the same **operations** (`threetears.evals.ops`): one function per thing an operator
does, over an `OpsHost` — the `LaunchHost`, plus `AnalysisGeneration` (the prompt, output cap and budget a
background generation runs under) — returning a typed model. Long work is a **job**: `run_launch` and
`analysis_generate` return `JobsStarted`, and `job_poll` / `job_cancel` take any job id either returned. A
job id names the durable record its work writes, so it is still answerable after a restart. A job is
answered only in the caller's scope: another scope's generation reads `lost` on poll and is refused on cancel.

Every spend operation is bounded in dollars before it spends. A launch's runs are held to the host's per-run
ceiling, which a launch's `max_cost_usd` may only lower — one above it is refused on every surface. An analysis
generation is held to the host's out-of-run cap (`LaunchSettings.max_out_of_run_cost_usd`): its first call is
priced before the job starts (`analysis_estimate` prices it without spending), its one repair round-trip
before that is sent, and each call is ledgered under purpose `analysis`, so `scope_out_of_run_spend` reads it. A judge
repeat (`judge_repeat`) is held to the same cap, every call priced before the first is sent, and ledgered under
purpose `judge`.
A host's own `spend` action carries no such obligation: the class is a label a tool cut splits on, metered only
as far as the host's handler meters it.

The **action catalogue** (`threetears.evals.actions`) declares each operation once for an agent: a `noun_verb`
name, a permission class (`read`, `spend`, `write`, `destructive`), flat described parameters, a result and
its rendering. A host adds its own actions and cuts tools by class:

```python
from fastmcp import FastMCP
from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.ops import OpsHost
from threetears.evals.transports.fastmcp import mount_fastmcp

server = FastMCP("myapp")
mount_fastmcp(
    server,
    eval_catalogue(my_actions),               # the engine's actions, then the host's
    host=OpsHost(launch=launch_host, generation=my_generation),
    caller=lambda: Caller(scope_id=current_scope(), identity=current_user()),
    tools=standard_tools("evals"),            # `evals` (read, spend, write) and `evals_admin` (destructive)
)
```

An agent calls `action='help'` for the actions grouped by workflow and `action='help', topic=<action>` for one
action's parameters and an example. A parameter the action does not declare is refused, naming the ones it
accepts. `read_only_tools(prefix)` mounts a tool an agent can only read through. The FastMCP transport needs
`3tears-evals[fastmcp]`; the core does not.
