# Adopting the engine: the host, the scope and the kind

This guide is for the developer wiring 3tears-evals into an app for real: your own storage, your own
launch path, runs you can compare over weeks. If you only want to grade a function once, `run_eval` (the
README's [Rung zero](../README.md#rung-zero-one-call)) builds all of this for you. Read
[Concepts](concepts.md) first: this guide uses its terms (host, kind, lever, apparatus, scope, cell)
without stopping to define them.

## The shape of it, in plain words

You write four things, and the engine does the rest:

1. **A host** — one value that holds your vocabulary (what you can change, what you can measure), your
   store, and a few services. Every engine function takes it; nothing is global.
2. **A store** — one small key-value port (`DocumentStore`), proven correct by a conformance kit you run
   in your own tests.
3. **A kind** — the adapter that turns a case into a call to your production code: `prepare` builds a
   candidate, `invoke` runs it.
4. **A launcher** per kind — the code that, when someone launches a run, freezes the cases and hands the
   engine the wiring.

Everything past those — the trial loop, the stored documents, judging, pricing, the analysis — is the
engine's. The sections after "Launching" cover what you need only if your subject has a stateful world,
converses, does background work, or calls tools you want to record and replay.

## Read the reference hosts, in this order

Two example hosts live in this repository (not in the wheel), written as reference code that imports
nothing but the public roots and itself:

1. **[`tests/fixtures/courierhost/`](../tests/fixtures/courierhost/__init__.py)** — the least a product
   writes when it builds its own host, in one module (a courier route planner). Its docstring lists what a
   product supplies, in the order it is written. Start here.
2. **[`tests/fixtures/toyhost/README.md`](../tests/fixtures/toyhost/README.md)** — a host that exercises
   every shape of the host contract (invoice field extraction), with a map of which file holds which step.
   Read it once you have the courier host's outline and want to see a particular piece in full.

Both run on `InMemoryDocumentStore` (`threetears.evals.storage`), the engine's in-memory reference
`DocumentStore`: scoped, with conditional writes, and the shape to compare your own adapter against.

Import from the public roots listed in the README and from `threetears.evals.contracts.host`, never from a
module below them. Every engine type a public signature hands you — a protocol you implement, a value you
receive, an exception you catch, a literal you annotate with — is exported from one of those roots.

## The host

**Your app is one value, the host.** Build an `EvalHost` (in `threetears.evals.contracts.host`) and
pass it to every entrypoint. It holds:

- your `HostProfile` — the levers you sweep, the measures you record, your bars and your world;
- the `EvalStorage` you read and write through;
- a factory for the completion clients the engine's own judge, simulator and analysis roles call;
- your `failure_describer` — how a raised provider call reads. It is the only thing that can say a call
  was refused for the calling account, which stops a run rather than excluding its cells;
  `withhold_failure_detail` is the honest one for an app with no error types of its own;
- your tracing, executor and cell-timeout choices. None of these last four has a default: each is named
  where the host is built.

Nothing is ambient: there is no installed or default host, so two hosts in one process never meet.

An app that starts runs builds a `LaunchHost` (in `threetears.evals.run`) around its `EvalHost`: the
launch settings, a registry of the kinds it can launch, and a job timeout; it builds its job manager over
the host's own storage. Hand its `eval_host` to the analysis side.

## Tenancy: the scope

**Tenancy is one opaque `scope_id`.** Every stored document carries a non-empty `scope_id`, and the
engine never interprets it, defaults it or branches on it: it is your tenant, project or environment,
whatever you partition by. All of it goes through one `DocumentStore` you implement, keyed by
`(scope_id, doc_type, id)`. Every port call names its scope, and there is no scope-free read: a caller that
needs several scopes is told which by you and asks each. A campaign and the runs it compares live in one
scope. Your adapter strips whatever it injects (an etag, a timestamp) before handing a document back,
because every stored model reads strictly.

## The store, and the conformance kit that proves it

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

**Stored data is disposable, and reads are strict.** Every stored model refuses an unknown field, a
missing required one, and a document written under any schema version other than this build's
`EVAL_SCHEMA_VERSION`. There is no migration and no tolerant reader: across a schema change, drop
the eval documents and regenerate them. Identity keys carry their own `IDENTITY_VERSION`, so keys
from different predicates never silently pool.

## The kind: what you are evaluating

**A kind is what you are evaluating.** You implement the candidate-kind seam: `prepare` builds a
candidate for one cell from the run's subject, its variant configuration and the seeded world, and
`invoke` runs it on a test case and returns a `CandidateOutput`.

What a kind adds to the engine's own fields is two Pydantic models you name once, on the profile's
`kinds`, in a `KindContract`:

- Its **overlays** are the knobs a launch may turn: validated by field before any run exists, frozen onto
  the run, and read back as levers whose levels enter the variant key.
- Its **spec** is what a template of that kind declares (`kind_spec`), such as a label set or a table
  setup: refused by field at authoring, then validated again and frozen onto each run.

You register neither anywhere else: the profile adds the contract's levers to its `sweepables`, and the
engine resolves every run's level of them — with its `candidate_model` and its `candidate_kind` — into the
variant key. Your profile's `host_sweepables` (the shared core extended with your own levers) and its
`variant_levers` reader cover only the levers you declare beyond those; a host with none wires no reader.

A run is one arm with one `candidate_model`; a launch naming several models starts one run per model, and
runs of different kinds are always different variants. A reader of yours that answers for a run its lever
does not apply to (a run of another kind) returns `SweepableValue.not_this_kind(kind)` — the level a kind
contract's own levers sit at there — rather than a value of its own displayed alike: that level is what a
report recognises, by its content hash, as a lever the arm did not run, and never prints.

## Launching

**A kind's launcher resolves only what is the kind's.** `start_run` refuses a launch argument the
kind declares it cannot honour before the launcher runs, then hands the launcher a `LaunchRequest`.
The launcher captures the subject the request names, freezes the cases, builds its judge
(`build_judge_service`) if it has one, and returns `launch_run(host, request, KindWiring(...))`. The
engine stamps everything the request and the template already say — scope, model, repeats, cassette
mode, overlays, spec, world seed, tool bound, ceilings — and refuses a wiring that contradicts them.

Before any launcher runs, the engine asks each kind what each arm will run (`LaunchableKind.plan_arm`) and
prices it. `plan_arm` is therefore where a kind makes its request-level refusals (no model and no default —
`require_candidate_model` — a judge or simulator it needs), and the launch tail then holds each launcher to
its plan. The full rule is in [Cost and budgets: every arm is priced before any launcher
runs](cost-and-budgets.md#every-arm-is-priced-before-any-launcher-runs).

### Generating cases at launch

A launch with `n_variations` > 0 asks for that many new cases from the template's variation axes,
generated once for every arm:

- Call `generate_variations` inside `request.launch_group.resolve_once(...)`, passing
  `blocking_executor=host.blocking_executor` (its case reads and writes run there; its model calls stay
  on the loop), and hand its counts on as `KindWiring(variation_counts=...)`.
- An `llm` axis is written by the model the launch names as `variation_model` — required then, refused
  when nothing would call it — which the launcher asks of the host's client factory in the `variation`
  role (`clients("variation", request.variation_model)`), never the simulator's.
- The run records the model the client resolved to on `variation_counts.variation_model`; it enters no
  identity, because the cases it wrote are already hashed through `test_case_ids`.
- The launcher hands `generate_variations` the request's `budget=request.generation_budget`.
  `propose_draft` takes a budget the same way.

Generation runs before any run exists, so its calls are outside every run's cost cap and metered-call
ceiling. How they are priced and ledgered instead is in [Cost and budgets: spend outside any
run](cost-and-budgets.md#spend-outside-any-run).

### Setting the rig at launch

**`apparatus_settings`** sets host-declared apparatus values — who sits in an adjudicator's seat, say —
so one template can be run at two of them and compared. A kind lists the ones its launcher reads, each
with the value its rig takes when a launch sets none
(`LaunchableKind.apparatus_settings={"adjudicator_seat": "model:default"}`, each a non-engine `apparatus`
declaration of the host). The dispatch resolves a launch's settings against those defaults, so the
launcher reads every one off `request.apparatus_settings` with no default of its own, and the run records
the rig as set up (`EvalRun.apparatus_settings`), a component of its measurement context — a launch naming
a default and one leaving it out are one condition.

## A world, through the cell's session

Skip this section if your subject is stateless (a classifier, an extractor). A world is for a subject that
acts on something — a ticket queue it can close tickets in, a game table — whose state you want seeded
before each cell and checked after it.

Before building a host for one, try the quick path: `run_eval(..., world=World(...), seed=..., goal_checks=[...])`
declares a small world, seeds each case's starting state through this same session, hands the candidate
tools that act on it and grades the end state with the engine's goal-state checks (`examples/world.py`).
What follows is the host-side contract that path is built on.

**A world, through the cell's session.** A host whose subject lives in a stateful world declares it
on the profile (`WorldRegistry`), and each cell's `prepare` is handed a `WorldSession` over it as
`world` (`None` on a host with no world).

- **Seeding.** The kind seeds through it — the seed walk, the dimensions' own `seed` handles, then each
  attached carrier's `settle` handle before the first turn.
- **Turns.** It announces each turn (`at_turn`, which applies any ambient perturbation the template's seed
  scheduled), and fires or observes its triggered dimensions (`fire`, `observe`).
- **Reading back.** Once `invoke` returns, the runner reads every attached dimension back through its
  `read` handle and stores it as the cell's end state; what fired is stored on the result as
  `world_events`.
- **Per-cell worlds.** The profile's registry is the declaration and its handles are the world
  conformance proves; if your world is real per-cell state, declare
  `WorldRegistry(..., binds_per_cell=True)` and have each cell's `prepare` call
  `world.bind(<that cell's handle table>)` before `seed` — every call the session makes then lands in
  that cell's world, and a session that seeds unbound is refused.
- **Grading.** A goal check reads the end state as `state.<dimension>`, the calls the kind recorded on its
  `CallLedger`, and what fired as `fired("<dimension>")`; grade them with `grade_goal_checks`, and a
  stored run re-grades from all three with `recheck_goal_states`.
- **Witnessed cells.** The runner's session is `commissioned`; a host grading a cell it witnessed through
  a session of its own constructs `WorldSession(registry, provenance="witnessed")`, so `fired_armed(...)`
  is not established there — no seed armed the session — and `record_witnessed_cell` refuses any world
  event claiming `armed=True` or `caused_by="rig"`.

## What the judge reads, and conversations

**What the judge reads, the kind renders.** A judged kind returns `JudgeEvidence` with every
non-empty output: the subject as the judge should see it, the case material, and the artifact (for a
conversation, the transcript as your kind writes it). The engine places those strings and reads none
of them, so hidden information and per-player visibility are your kind's rules. The evidence is
stored on the cell's `EvalTrace`, and a re-judge sends exactly what the first judge read.

**Conversations.** A conversing kind's template carries a `ConversationSpec`: its simulated actors, who
speaks next, and the turn limit. A document or classifier template carries none. The kind runs it with
`drive_conversation(driver, candidate_turn, post_user_turn, llm=..., sink=sink)`, handing over its cell's
sink: before every paid call the loop asks the run's cost cap whether the simulator's spend so far
reaches it (`CellSink.cost_cap_reached`), and stops `budget_stopped` when it does — the runner excludes
that cell and ends the run `budget_stopped`, so one conversation (thousands of simulator calls at the
schema's maxima) cannot run far past the cap. Fold the driver's calls with `fold_usage`: the stored
`simulator` rows are one per actor and purpose (`RoleUsage.actor_id`, `RoleUsage.purpose`).

## Background work, payloads and spend

**Background work** a candidate hands off and gets back turns later is recorded as `async_deliveries`,
one `AsyncDelivery` each: who asked, when it was acknowledged and delivered, on what model, whether a
harness supplied it, and what it spent — its tokens, model calls, `cost_usd` with its own
`price_source`, and any paid non-LLM calls (`external_spend`) — their calls, the provider's own units, and
`money` where the provider itself billed a figure. The engine folds that spend into the result's
`inner_agent` and `external` usage, for work still in flight when the cell ended as well as work that
delivered; a substituted entry reports no spend.

**Payloads.** Anything else your kind wants kept with a result goes in `kind_payload`, which the engine
stores and never reads; register a measure for whatever should be compared.

**Spend.** Your kind reports its own calls' spend only as usage rows (`CandidateTelemetry.usage`). How
the engine turns those rows into a result's `cost_usd`, and what happens when a call cannot be priced, is
in [Cost and budgets: how a result's cost is counted](cost-and-budgets.md#how-a-results-cost-is-counted).

## Cassettes: recording and replaying tools

A cassette lets two arms face exactly the same tool answers: one run records what its tools said, and
later runs are served that recording instead of calling the tools live. You only need this if your
candidate calls tools whose answers vary or cost money.

**Cassettes record and replay through seams the kind supplies.** Launch a run with
`cassette_mode="capture"` to run its tools live and record what they answered into that run's own
corpus, or `"replay"` with `cassette_corpus_id` naming a capture run to be served that corpus instead.
The engine builds the run's lane and hands each cell's `prepare` a `cassettes` handle (`None` with
cassettes off) already bound to the corpus, template and case, so one kind instance serves every
cell.

- **Wiring.** Your kind calls `cassettes.wire(seams)` once with its candidate's `CassetteSeams`: an
  `ActionSeam` for the synchronous tools to wrap, and a `DeliverySeam` for each asynchronous tool, keyed
  by that tool's name.
- **Tool shape.** A wrapped tool is a `ToolLike`: a `name`, `can_dispatch(action)` and an async
  `act(action, parameters)`; a tool that is a plain function is adapted to that shape.
- **Blocking tools.** A candidate whose turn loop calls its tools as plain blocking calls declares a
  `SyncActionSeam` instead (`arm_sync_tools`), its tools are `SyncToolLike`
  (`act_sync(action, parameters)`), and it calls the wrapped tools' `act_sync`. Both seams record and replay through one
  implementation, so a corpus captured on either replays on either.
- **Background work.** A delivery seam reports each piece of background work where it starts
  (`recorder.started(request)`) and settles the ticket where it ends; under replay it takes
  `replay.next(request)` instead of starting live work, and reports what it was served with
  `substituted=True`.
- **Matching.** Every recording is keyed by what was asked and by which time it was asked, so a repeated
  dice roll replays both rolls in order and two scouts are each paired with their own report; an ask the
  capture never made, or made fewer times, stops the cell as the rig's failure rather than running live.
- **No silent live runs.** A kind that does not wire the handle it was given, or a candidate with no
  seams, is refused rather than run live under a replay.

## Rig failures: a broken rig costs one cell, never the run

**A broken rig costs one cell, never the run.** A replay miss, a corrupt recording or any other
fault of the measuring rig rather than of the world under test is an `ApparatusError`; your kind's
tool boundary re-raises it instead of turning it into an ordinary tool failure. Out of `prepare` or
`invoke`, the engine records that cell excluded (termination `apparatus_failed`) with whatever spend
the kind had reported through its sink, and goes on. A run whose every cell the rig excluded measured
nothing, and ends `failed`.

A cell cut off by its deadline or by its run's cancel is recorded the same way, from its sink; when the
cut lands while the cell is being judged, the evidence and the scores that came back are kept and the
unfinished dims are the cell's judge errors, which a re-judge can repair. A re-judge re-asks only dims
that errored: a judge's "can't tell" is an answer.
