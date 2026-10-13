# Adopting the engine: the host, the scope and the kind

**For** the developer wiring 3tears-evals into an app for real: your own storage, your own launch path, runs
you can compare over weeks. **Answers:** what you write (a host, a store, a kind, a launcher) and what the engine
does with each. If you only want to grade a function, and keep its runs in a file, `run_eval` (as in the
[tutorial](tutorial.md#8-keep-your-runs)) builds all of this for you. Read
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

Quick's pieces build some of these as ordinary contract objects you mix with your own (all in
`threetears.evals.quick`): `@measure(...)` declares a `MetricDescriptor` on the function that computes it, so you
register its `.descriptor` and grade with the function; `callable_kind(candidate, scorers)` is the kind over a
plain async function, declared under `callable_kind_contracts()`; and a quick `World`'s `registry` and
`bindings(state)` are a simple world's declaration and its handles. The toy host's kind computes its measures
this way.

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
`DocumentStore`: scoped, with conditional writes, and the shape to compare your own adapter against. Its
durable sibling, `SqliteDocumentStore`, keeps every document in one SQLite file through the standard library;
a host with no database of its own can use it as its store.

Import only from a public root (`threetears.evals.PUBLIC_ROOTS` lists them, including
`threetears.evals.kernel.host`, `threetears.evals.analysis.viz` and `threetears.evals.vega`), and only the names
its `__all__` declares; never from a module below one. Every engine type a public signature hands you — a protocol you implement, a value you
receive, an exception you catch, a literal you annotate with — is exported from one of those roots.
The stored documents and the ports you implement (`DocumentStore`, `CompletionClient`, `TraceSink`) are in
`threetears.evals.schema`; the behaviour over them (identity, scoring, `EvalStorage`, the candidate kind) is in
`threetears.evals.kernel`, with the host contract under `threetears.evals.kernel.host`.

## The host

**Your app is one value, the host.** Build an `EvalHost` (in `threetears.evals.kernel.host`) and
pass it to every entrypoint. It holds:

- your `HostProfile` — the levers you sweep, the measures you record, your bars and your world. Every measure
  you declare (`MetricDescriptor`) carries a `reader_name`, the few words a table header, chart title or axis
  label prints in place of its key ("Output speed", not `candidate_output_tokens_per_s`). `MeasureRegistry`
  refuses a measure without one, and one whose `reader_name` matches another measure's (yours or the engine's,
  ignoring case), so the mistake surfaces at startup, not on a chart. The key stays what the store files the
  number under and what a reader cites. Your bars (`BarRegistry`, each a `Bar`, with any per-behavior
  `PassThreshold`) can be built in code or kept as data: `load_bar_registry(mapping)` reads
  `{"bars": [...], "pass_thresholds": [...]}`, each entry its type's fields. It refuses a field it does not
  know and a value of the wrong type rather than ignoring or coercing them, because a misspelt `rationale`
  dropped silently would register a standard other than the one you wrote;
- the `EvalStorage` you read and write through;
- a factory for the completion clients the engine's own judge, simulator and analysis roles call. Every
  completion a client returns is read by `CompletionResult`'s attribute names, and the usage ledger reads the
  token and cost attributes with a default, so a renamed attribute would degrade every usage row to
  "unreported" in silence. Check your completion type in your own suite with
  `threetears.evals.testing.check_completion_conformance(an_instance)`, which fails naming the attribute.
  `served_model` must be the model the provider's RESPONSE named, never the request: it is how a candidate
  launched on a floating alias records which model answered it. Read it off the provider's response (the
  `model` its reply body names), and report `None` when the response names none rather than copying the
  request in, which would make two different models compare equal. A `None` is never refused: the judge's
  scores and the usage rows keep the id requested, and the run's summary (`run_get`), `run_summary`, the
  analysis bundle and the campaign report each say once that which model answered is unconfirmed, counting
  the results and scores it covers ([Reading reports](reading-reports.md#which-model-answered)). Your
  candidate kind's own usage rows (`CandidateTelemetry.usage`, below) carry it the same way: fold each
  completion with `RoleUsageLedger.add_llm_result`, which reads it. `temperature` must be what the request
  carried: the judge asks for 0 unless a config says otherwise, and a client whose model refuses a temperature
  sends none and reports `None`, which the score records as `model_default`. A judge completion with no
  `temperature` attribute at all is your code's defect, not a judging outcome: the engine raises
  `CompletionProtocolDefect` (a `TypeError`, in `threetears.evals.schema`) and the run fails with that message,
  as it does for a kind's `CandidateKindDefect`, rather than finishing with an error on every dimension. The
  error carries the calls already billed and their cost (`calls`, `cost_usd`), and the cell it surfaced in is
  recorded excluded (`apparatus_failed`) with what it was billed, so the run's totals and its cost cap count it, as
  a cell failed by a `CandidateKindDefect` is. A judged witnessed cell is not recorded, but its billed judge calls
  count against the run's judge ceiling. A re-judge, a judge repeat, a second judge and a witnessed cell's judging raise it the same way;
- your `failure_describer` — how a raised provider call reads. It is the only thing that can say a call
  was refused for the calling account, which stops a run rather than excluding its cells;
  `withhold_failure_detail` is the honest one for an app with no error types of its own;
- your tracing, executor and cell-timeout choices. None of these last three has a default: each is named
  where the host is built.

Nothing is ambient: there is no installed or default host, so two hosts in one process never meet.

An app that starts runs builds a `LaunchHost` (in `threetears.evals.run`) around its `EvalHost`: the
launch settings, a registry of the kinds it can launch, a job timeout, `world_placements` (required exactly
when the profile declares a world: what each assembled run did with every dimension) and a `launch_pricer`
(without one, an arm under an enforced cap is refused unless the launch names its own cap; see
[Cost and budgets](cost-and-budgets.md#every-arm-is-priced-before-any-launcher-runs)); it builds its job manager over
the host's own storage. Hand its `eval_host` to the analysis side.

Each kind in the registry (`LaunchableKind`) carries `resolve_subject`: whether the host holds the subject a launch
of that kind names — `None`, or an `UnknownSubject` saying how a caller finds one it does. It is the kind's, not the
host's, because one subject id means a live entity to one kind, a fixed prompt key to another and a free label to a
third; `launchable_judge_kind` brings its own, which admits any non-blank label. It is asked at admission, once the
template's kind is known, so `launch_estimate` and a launch refuse an unknown subject alike, before anything is priced.

## Tenancy: the scope

**Tenancy is one opaque `scope_id`.** Every stored document carries a non-empty `scope_id`, and the
engine never interprets it, defaults it or branches on it: it is your tenant, project or environment,
whatever you partition by. (The one default is the quick path's `"quick"`, used only in the in-memory store a
`run_eval` or `compare` call builds and drops; pass a `host=` or `store=` and `scope_id=` is required.) All of it goes through one `DocumentStore` you implement, keyed by
`(scope_id, doc_type, id)`. Every port call names its scope (an upsert takes it from the document it
writes), and there is no scope-free read: a caller that
needs several scopes is told which by you and asks each. A campaign and the runs it compares live in one
scope. Your adapter strips whatever it injects (an etag, a timestamp) before handing a document back,
because every stored model reads strictly.

## The store, and the conformance kit that proves it

**Prove your store with the conformance kit.** `threetears.evals.testing.STORE_CONFORMANCE_CASES`
states every rule of the port — scoping, the strip on read, `exclude`/`keep` projection, ordering,
etag conflicts and the re-read that recovers one, create-only writes, merge and delete — as a case you
hand a fresh, empty store. Every case is mandatory: in particular a store must implement both write
preconditions — `if_match` (the stored document still carries this etag), because several writers share a
run document as it finishes, and `create_only` (nothing is stored under this scope and id, of any type),
because two processes racing to start one campaign's analysis generation both create its one generation
slot and exactly one may land. A store that forwards writes to a backend passes both through. A store
may also offer `StoreClock` (`now()`, the store's own time), which job leases are then stamped in; offer it
only when the time is the store's, not the calling process's. Parametrise your test runner over it:

```python
@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_my_store_conforms(case: StoreConformanceCase, tmp_path: Path) -> None:
    case.run(MyDocumentStore(tmp_path / "evals.sqlite"))
```

**The store is called off the event loop.** The port is synchronous and the engine is async: every store call
the engine makes runs on the host's `blocking_executor`, never on the loop, and `EvalStorage` refuses a call made
on a thread running an event loop with `StoreCalledOnEventLoop` (in `threetears.evals.schema`). Your own async
code follows the same rule. A synchronous function that takes the storage or the host — most operations in
`threetears.evals.ops` (`runs_list`, `report_read`, `scope_history`), and the reads in `threetears.evals.run` and
`threetears.evals.analysis` — blocks: from async code, offload it with `await run_blocking(host.blocking_executor,
fn, ...)`, as the action catalogue does for every synchronous operation. An async operation (`run_launch`,
`job_poll`) offloads its own.

**If your database driver is async**, write your adapter against it as an `AsyncDocumentStore` (in
`threetears.evals.schema`: the same port, every method awaited and `iter_by_doc_type` an async iterator;
`AsyncStoreClock` for its clock) and hand the
engine `bridge_async_store(store, loop)` (in `threetears.evals.storage`), built with the loop the store's
connections belong to — in an async app, the running loop at startup. Each call the engine makes runs as a task on
that loop, in the calling context (a context variable your row-level security reads arrives set), while the
executor thread waits. Keep `blocking_executor` a thread pool. Prove the adapter through the bridge, with its loop
on a thread of its own, since a conformance case calls the store synchronously:

```python
@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_my_async_store_conforms(case: StoreConformanceCase) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        store = asyncio.run_coroutine_threadsafe(open_my_async_store(), loop).result()
        case.run(bridge_async_store(store, loop))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
```

The engine reads and writes through `EvalStorage`, built over that one store. Its consumers name the
area they touch rather than the whole of it — `RunStore`, `ResultStore`, `RunRecordStore`,
`DefinitionStore`, `CassetteStore` and `JobStore` (in `threetears.evals.kernel`) — so a function
typed `DefinitionStore` cannot reach a run, and a test of one hands it only that area.

**Watch the read tier's size.** The lenses that read a whole scope — `pivot` and `export_results` (and the
`scope_pivot` / `scope_export` operations over them) — project and aggregate every result in Python on each
call, which suits a scope of dozens-to-hundreds of cells. Each call logs one line on the
`threetears.evals.analysis.reads` logger, at INFO:

```
eval.read_tier lens=pivot scope_id=acme runs_in=12 results_in=480 records_out=2880 cells_out=24 elapsed_ms=41.7
```

`runs_in` and `results_in` are what the call read, `records_out` the score records it projected, `cells_out`
the pivot's cells (`refused` when the pivot was refused; absent on an export) and `elapsed_ms` its wall time.
Once `records_out` passes `READ_TIER_ROW_BUDGET` (50,000, in `threetears.evals.analysis`) the line is logged
at WARNING and ends `over READ_TIER_ROW_BUDGET=50000`, so you can alert on the level alone. Crossing it means
the scope has outgrown the in-Python projection, the signal to push it down into your store; it fires
before a page is slow, and you can alert on `elapsed_ms` instead.

## Stored data: what is kept

**Every stored type has facts.** `DOC_TYPE_FACTS` (in `threetears.evals.schema`) is the one list of the
types the engine stores, and says three things about each:

- **origin**: `input` (what runs read as given, whoever wrote it: templates, judge configs, rubric dims, case
  sets, test cases, campaigns), `record` (what happened: runs, results, traces, calibration ratings, the
  out-of-run spend ledger, sweeps, analysis attempts, cassettes, generation slots, launches, second judge passes,
  judge temperature comparisons, host action jobs) or `derived` (computed from other stored
  documents: analyses, insights, judge profiles).
- **lifecycle**: `kept` (read by every later release, below) or `regenerable` (refused after a format change,
  and made again: an analysis, insight or judge profile is regenerated, a cassette re-captured).
- **suppresses**: for a tombstone, the type it suppresses (`rubric_dim_tombstone` holds a `rubric_dim`'s
  slot, `judge_config_tombstone` a `judge_config`'s).

`EVAL_DOC_TYPES` (what the operator wipe sweeps) and `CORE_DOC_TYPES` (what is kept) are derived from it.

**The evidence core is kept.** Every release reads it, from core v14 (the first public release's baseline) on. The core is
what the engine cannot regenerate and an audit needs: test cases, runs, results, traces, calibration ratings,
the out-of-run spend ledger, case sets, and the templates, judge configs and rubric dims (with their
tombstones) those documents name by id; and the campaigns people declared and gathered runs into, and the
records of how each sweep ended, what each analysis generation spent, and which generation holds a campaign's
slot (a later build that could not read a slot would block the campaign's generations, or start a second beside
a live one), and the records of every other job: a launch (the runs it started), a second judge's pass (its report),
a judge temperature comparison (every answer, stored nowhere else) and a host action's job (its result).
`CORE_DOC_TYPES` lists them.

**The first public release is the baseline.** It writes core v14.0 (major `CORE_BASELINE_VERSION`, equal to
`CORE_SCHEMA_VERSION`, and minor `CORE_SCHEMA_MINOR`, 0), regenerable v9 (`REGENERABLE_SCHEMA_VERSION`) and
cassette v1 (`CASSETTE_SCHEMA_VERSION`), and registers no upgrade steps (`CORE_UPGRADERS` is empty): there is
nothing older for it to read.

**A core version is a major and a minor, and your store needs no migration for either.** Every core document
carries `schema_version`, the major, and `schema_minor`, the minor within it. Every change to a core shape is one
or the other, an added optional field included:

- **A minor is additive only**: a new optional field with a default, a new optional sub-object, a new value of an
  open vocabulary, a new doc type. Its ledger entry says, in words, that an older build that ignores the change
  reads the document correctly and its writes leave it true (`MINOR_ADDITIVITY_SENTENCE`); when that is false, the
  change is a major. **A minor rolls with no stop, in either direction.** A build reads any newer minor of its own
  major: every field it does not declare, at every depth, is kept on the model, out of every attribute, dump and
  digest, and written back where it was the next time the engine writes that document, stamped with the newer
  minor. So while a fleet is half upgraded, an older process reads what a newer one wrote as if the new fields
  were not there, and keeps them on every part of the document it carries over (a part it replaces with one it
  built afresh, such as a renewed lease, takes them with it); two builds can print different numbers from one
  store for that window, which each analysis bundle's package version discloses. **The tolerance follows the
  stored bytes, which are trusted:** a host that writes core documents straight into the store, with a
  `schema_minor` above the build's, gets it too, so write the minor your build writes. **The roll has a floor:**
  the tolerance arrived with the minor, in `v0.66.0-evaladopt.22`. A build before it (`.21` included) declares no
  `schema_minor` and refuses every field it does not declare, so it refuses every core document `.22` or a later
  build writes, though the major is 14 for both. Roll back no further than `.22`.
- **A major is everything else**, and ships an upgrade step (a `CoreUpgrader` appended to `CORE_UPGRADERS`),
  tested against frozen stored documents from every core major since the baseline. Reading an older core
  document upgrades it in memory, one major at a time, before it is validated as strictly as a document written
  today; the stored bytes stay as they are until the engine next writes that document, at the current version. A
  newer major is refused, so every process sharing the store moves to it together.

So one scope can hold documents at several core versions. Neither a minor nor a step changes a field the store
filters, sorts or projects on (`CORE_ADDRESSING_FIELDS`, which the minor is not one of), so an index your adapter
keeps on those fields stays valid. Two things an older build cannot do for a new doc type a minor adds: its
operator wipe does not sweep it, and its deletes (`run_delete` and the cascades it runs) cannot reach it, so a
document of the new type can be left orphaned for the upgraded build. Wipe and delete from the newest build
while a fleet is mixed.

**Some vocabularies are open.** A usage role, a simulator purpose, a ceiling's or a role model's origin, a
reasoning effort, an out-of-run purpose and a stored refusal kind are strings, with the values this build knows
registered beside each (`USAGE_ROLES`, `COST_CAP_ORIGINS`, …). A newer minor can add a value; this build reads it
from that minor's document and names it wherever it reports one (a role's spend on a line of its own, an origin
by name), and refuses it anywhere else. Only what is stored is open: a parameter you pass in (an action's filter,
a writer's argument) takes a value this build knows (`KnownUsageRole`, …) and refuses any other. Every other
vocabulary on a core document (a run's status, a cell's
termination, a scale, a provenance, …) is closed: a new value is a major. `schema/versioning.py` lists each.

**Everything else is regenerated.** Analyses, insights and judge profiles are built over the core. When one of
their shapes changes, `REGENERABLE_SCHEMA_VERSION` is bumped and a document written under another version is
refused: make it again over the kept runs (generate the analysis, profile the judge). After a bump,
`analyses_undescribable` (operation and action) lists the scope's analyses holding an arm this build can no longer
describe. A cassette is not touched by that bump: its format has its own `CASSETTE_SCHEMA_VERSION`, bumped only
when the cassette itself changes, because re-capturing a corpus calls your third-party tools again. A cassette
written under another cassette version is refused, by name, and you re-capture it (a new capture run, so a new
corpus); a replay that asks for a refused recording stops its cell with a `CassetteCorrupt` naming both versions.

**Each version space moves on its own.** Core documents (v14.0), regenerable documents (v9), cassettes
(`CASSETTE_SCHEMA_VERSION`, 1), the bundle shape (the bundle's `schema_version`, 52), the cell model
(`CELL_MODEL_VERSION`, 12), the report (`REPORT_VERSION`, 9) and the identity keys (`IDENTITY_VERSION`, 24) carry
separate versions, each at its first public release's value.
**A judge's identity moves with its wording, and no version number moves with it.** Each result records what asked
each dim (`EvalResult.judge_config_ids`): `builtin:<sha12>`, a fingerprint of the engine's own judge prompt, or
`<config id>@<sha12>`, your `JudgeConfig` and a fingerprint of the reply-format suffix the engine appends to its
prompt. A release that rewords either changes the fingerprint, so every result judged before it is another judge
for agreement, judge profiles, drift and evidence tiers, and a `calibrated` tier reads `undetermined` until you
calibrate the new judge. Nothing stored is refused or rewritten. Such a release's notes (`MIGRATIONS.toml`, printed
by `scripts/release_notes.py`) carry `JUDGE_WORDING_CHANGE_SENTENCE` (in `threetears.evals.schema`) and name
the wording that moved, so read them before you upgrade a scope whose readings rest on calibration. A run's own `judge_config_ids` (the
config set it launched with, part of its context key) holds bare config ids and no built-in dim, so a wording change
does not split runs' context: only the results' recorded judges tell the two judges apart.
An upgrade never recomputes an identity key, so keys from different predicates never silently pool. A kept
key is stored beside its pre-image and the identity version that computed it, so a read under a later
`IDENTITY_VERSION` needs no upgrader: a run keeps `variant_levers` beside the keys its results carry, and a
campaign's declared control keeps `control_levers` and `control_identity_version` beside `control`, recorded when
the control is addressed from a run (`control_from_run_id`, `set_campaign_control`) and never taken from a caller.
Under another version the control matches its stored key, which the cells stamped under its own version keep, and
the key recomputed from that map, which the arm's new runs carry, so it never goes stale while its old cells are in
the campaign. Its arm is one set of cells: the new runs' cells where there are any, with the old cells named beside
it as the control under the earlier version (`RealizedDesign.earlier_control`), disclosed, in no contrast, and no contestant on the frontier; the
control's numbers are the new cells only. The recomputation follows a change to how a map is digested only: a lever,
kind or overlay field your host renames, or a lever added to every map, gives new runs a different map, and the
control then matches only the cells from before it. A control key typed by hand
records no map; when no cell carries it under another version it reads as `stale`
(`RealizedDesign.control_excluded`), and the remedy is to address it again from a run.

**Reads stay strict everywhere the contract cannot vouch for the unknown.** Every stored model refuses an
unknown field and a missing required one, at any depth, with one exception: a core document of this build's major
at a newer minor, read as above. A core document written by a newer major is refused, never guessed at, so when
two builds share one store across a major, upgrade the older one. Only the stored read is tolerant: a document a
caller hands an authoring surface is validated strictly, and `schema_minor` is a server field it ignores. A core
document written before the first public release (below core v14) is refused
too: its read fails validation on `schema_version`, with a message that names it a pre-release document and
the baseline it is below. Pre-release documents are never read, so delete them (the operator wipe) rather than
keep them beside released ones.
A refused run stays as stored, and the boot reclaim (`sweep_abandoned_runs`) counts it `unreadable` rather than
settling it. `runs_unreadable` (operation and action) names each stored run of the scope the current model refuses,
of any status, with the fields it refused and why. The ones stored pending or running are marked stranded: they are
the reclaim's `unreadable` count for that scope, and nothing settles them. `run_delete` removes any of them from its
stored row, with the results and campaign memberships a readable run's delete removes. A stranded one is refused
while its stored lease has not lapsed, and when its lease cannot be read either: then whether a process still holds it
cannot be told, so deploy the build that wrote it, which can settle or delete it. Or deploy a build that reads them. Every other read of such a run refuses with `UnreadableRunError` (in
`threetears.evals.kernel`, status 500, remedy naming `runs_unreadable`), except the `runs_list` inventory, which lists
the runs that read and counts the rest; map it in your own routes like any other engine refusal.

### Partition by facts: one store per kind of document

You can keep each kind of document in a store suited to it, such as the evidence in your database and
everything the engine can remake in a cheaper store. `RoutedDocumentStore` (in `threetears.evals.storage`) is
one `DocumentStore` over several. A `DocTypeRoute` keys on one fact, `origin` or `lifecycle`, and names a store
for every value of it, plus one store for the types the registry does not hold (your own documents):

```python
from threetears.evals.kernel import EvalStorage
from threetears.evals.storage import DocTypeRoute, RoutedDocumentStore, SqliteDocumentStore

stores = {"evidence": MyPostgresStore(dsn), "scratch": SqliteDocumentStore("regenerable.sqlite")}
route = DocTypeRoute(by="lifecycle", to={"kept": "evidence", "regenerable": "scratch"}, unregistered="evidence")

ROUTING = "evals-routing.production"  # yours to name; never a scope you keep documents in

# Once, with the engine stopped: move what the stores already hold to where the route puts it.
RoutedDocumentStore.relocate(
    stores, route, scope_ids=["acme", "globex"], routing_scope=ROUTING, host_doc_types=["host_note"]
)

storage = EvalStorage(RoutedDocumentStore(stores, route, routing_scope=ROUTING), host_doc_types=["host_note"])
```

A route keys on facts, never on type names, so a type a later release adds is placed by what it is. Every
mistake a route can hold is refused when the router is built: a fact value left unmapped, a store that is
not among `stores`, one store passed under two names, a tombstone kept apart from the type it suppresses.

**Typed calls go to one store; id-only calls ask every store.** A write, `get_many`, `by_doc_type` and
`iter_by_doc_type` go to the store the type is kept in, and a conditional write's `if_match` or `create_only` reaches that store
unchanged. `get`, `get_with_etag`, `merge_fields` and `delete` carry only an id, so they ask every store. A store
that does not hold the id answers "not here", as the port says. One id lives in one store: a read that finds an
id in two stores, or in a store its type is not kept in, raises `MisplacedDocument` rather than picking one. A
`create_only` write asks every other store for the id first and is refused with `StoreConflict` if one holds
it, as a single store would refuse it. Only the owning store's create is atomic, so a write racing that check
can leave the id in two stores, and the next id-only read reports it as `MisplacedDocument`.

**Changing the route is an operation.** Each store records the placements it was written under, in the routing
scope you name (`routing_scope`, on the router and on `relocate` alike, with no default). Never keep documents in
it, and your adapter must be able to hold it. If you run several partitions in one database, such as
environments or tenants behind row-level security over tables whose key is global, run one router per partition
and give each its own routing scope: one shared scope would hold every partition's record under one key, which
only the first partition could write. A router built with another routing scope than `relocate` was given finds
no record and refuses to start.
The router refuses to start (`RouteChanged`) when a type would be read from a store it was not written to, when
two stores are passed under each other's names, and when no store has a record yet. `RoutedDocumentStore.relocate`
moves every document of the scopes and host types you name to its new store, then writes the records. Run it
once when you adopt the router and again whenever you change the route, with the engine stopped. If it is
interrupted, run it again. A release that adds a type needs no relocation unless your route places the new type
somewhere other than your `unregistered` store.

**Ask whether the stores hold anything before you relocate.** A host's start that adopts tables, rather than
relocating on every start, asks first: `RoutedDocumentStore.holds_documents(stores, scope_ids, host_doc_types=...)`
answers whether any store holds any document of any registered type, or of your own types, in the scopes you
name, and stops at the first one it finds. It reaches what `relocate` reaches and nothing more: the scopes and
host types you name. It does not count the router's placement record, which lives in the routing scope (never a
document scope) under a type no registry holds, so stores holding only a record answer `False`.

```python
if RoutedDocumentStore.holds_documents(stores, ["acme", "globex"], host_doc_types=["host_note"]):
    ...  # documents are there: relocate them, with the engine stopped, before the router starts
```

`RouteChanged` and `MisplacedDocument` carry `fault` and `remedy` apart (their `str()` reads both). A host that
offers its own relocation tool states its own remedy beside the router's `fault` instead of rewording the message.

**A store's clock passes through when every store has one.** When each of your stores offers `StoreClock`
(its own `now()`), so does the router, answering from the store your runs are kept in, so every process stamps
job leases on one clock. If any store has none, neither has the router, and each process reads its own clock.

Prove the router as you prove any store: the conformance kit runs over a `RoutedDocumentStore` of your stores.


**A record your host writes itself is held to what the engine writes.** A run or result the engine did not
launch (a witnessed session, a run your host assembles) is validated as strictly as one it did:

- a run that names a `judge_model` names the `judge_temperature` its unconfigured dims were requested at, and a
  run that records `goal_check_proofs` records the `goal_check_proof_rules` they were derived under;
- every `RubricScore` states its `axis` (`capability` or `boundary`). A score your own reviewers gave, not a judge
  call, leaves `judge_temperature` and `served_model` as `None` and records no judge identity for its dim: it reads
  as an unknown judge, never as the same as a recorded one;
- a judge identity in `EvalResult.judge_config_ids` is `builtin:<sha12>` or `<config id>@<sha12>`, as the judge
  service writes it; a bare config id is refused.

## Stability tiers

**Every public name has a tier, and the tier is what a release promises about it.** From 0.66.0, the first tagged
release, on:

- **`host-contract`** is what a host writes against: the host and its profile, the registries, the kind seam,
  launching, the stored documents a kind builds, the store port with its conformance kits and the engine's own
  stores, the refusals, the version constants, the `Known*` vocabularies and their value tuples (`USAGE_ROLES`, …),
  the quick path and what it returns, and every name this guide, the [tutorial](tutorial.md), the README or an
  example imports. Such a name moves only with
  a `MIGRATIONS.toml` record and a deprecation cycle: the release that moves it exports the new name beside the old
  one, which keeps working for that minor release and is named in its notes, and the next minor release removes the
  old name, recorded as any removal is.
- **`closure`** is every other public name: public because a public signature reaches it (a field's type, a value
  a function returns, an exception it documents), so you can type your code without reaching below a root. Such a
  name may move in any release, recorded in `MIGRATIONS.toml` and printed in that release's notes, never shimmed.

`packages/evals/api/surface.json` records each name's tier beside its kind and signature, and `HOST_CONTRACT` in
`packages/evals/scripts/public_surface.py` declares the first tier, with the reason for each group of names. A name
a topic guide imports and the list does not hold is `closure`. If your host binds to one, ask for it to be
promoted: promoting a name costs nothing, and demoting one is a break.

## Upgrading: names that move

**Every name that leaves a public root is written down, and only a host-contract name is kept for compatibility.**
When a release moves a public name to another root, renames it, or removes it, `packages/evals/MIGRATIONS.toml`
records it under that release: the old name, its new home or `removed`, and why. A `closure` name has no shim at the
old name; a `host-contract` one keeps working for a minor release first (above). The record starts
at 0.66.0, the package's first release, which moved nothing. Pre-release tags are not releases and are not in it.
The file is
in the repository, not the installed package: read it at the tag you are moving to, on GitHub
(`https://github.com/pacepace/3tears/blob/<tag>/packages/evals/MIGRATIONS.toml`) or in a checkout. In a checkout,
the release notes are printed from it, so they hold the full old-to-new map:

```bash
uv run python packages/evals/scripts/release_notes.py v0.66.0
```

Each line reads `old → new`, grouped under the reason for the move. Changes that are not a public name, such as
a renamed field, a moved submodule or a new report version, are listed after the map. To port, find each old name
you import in that list and import it from its new root.

**A name has one home.** It is exported from the root that holds its defining module. A second root exports it
only when that root needs it, and `packages/evals/api/reexports.toml` says why. For example, `ops` re-exports the
analysis results the action catalogue names, because `actions` may reach `analysis` only through `ops`. Import a
name from its home, which is the root the reference lists it under first.

**How the package holds this (for contributors).** `packages/evals/api/surface.json` is a checked-in snapshot of
every public root's `__all__`, one entry per `root.name` with its kind, signature and tier. The same walk that renders
the [reference](reference.md) writes it. `tests/test_api_surface.py` compares the installed package with the
snapshot:

- **A name added, or a kind, signature or tier changed.** The test fails until you regenerate the snapshot, so the
  change shows up as a diff in review. A signature change needs no record in `MIGRATIONS.toml`. When it breaks
  callers, add it to the newest section's `notes`. A tier changes only by editing `HOST_CONTRACT`, and every name
  the README, this guide, the tutorial or an example imports must be on it (`tests/test_api_tiers.py`).
- **A name moved, renamed or removed.** The test fails until `MIGRATIONS.toml`'s first section (the release being
  prepared) records it. The generator refuses to write a snapshot that drops an unrecorded name. A name the last
  final release did not export never reached a porter, so it needs only a regeneration, and a record of it is
  refused.
- **A section that names the tag before it (`since`).** The test reads both surfaces from git and holds the
  section to exactly the names that left a root between that tag and this version's. Before the version is
  tagged, it compares against the working tree. `since` is always a final release tag (`v<x>.<y>.<z>`). A tag
  from before the package existed (`v0.65.0`, for 0.66.0) has an empty surface. A checkout without the tags fails
  this check rather than passing it.
- **A name exported from two roots.** The test fails unless it is one object and `api/reexports.toml` declares
  it. Two different objects under one name always fail: rename one.

```bash
uv run python packages/evals/scripts/generate_api_snapshot.py          # regenerate (refused while a removal is unrecorded)
uv run python packages/evals/scripts/generate_api_snapshot.py --check  # report every finding, write nothing
```

**When a version is tagged**, rename the `unreleased` section of `MIGRATIONS.toml` to the version, without the
leading `v`. Then paste the output of `release_notes.py v<version>` into the release description. The next release
starts a new `unreleased` section at the top, with `since` set to the tag just cut. A pre-release tag (any suffix
after `v<x>.<y>.<z>`) gets no section, and the record never follows one.

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

The contract also takes `seats`, the parts of the rig the kind's runs fill (`judge`, `simulator`, or an
apparatus dimension; the default, `None`, holds the kind to every dimension), and `prefix`, the namespace of
its levers (`<prefix>.<field>`; the kind's name when unset).

An overlay field takes markers beside its type, in `Annotated[...]`: `Ordinal()` ranks a `Literal` or `Enum`
in declaration order, `Interval(unit=...)` names a number's unit, `ActsOn(measure)` names the measure the knob
is supposed to move, and `ResolvesInto(lever)` names the host lever the knob is written into. A map field's
entries are distinct knobs, so `ActsOn` is refused there; `MemberActsOn({"max_search_calls": "search_calls"})`
names the measure per entry instead, and every entry it does not list reads `unchecked`.

`ResolvesInto` is for a knob whose effect your host also records, resolved, as a lever of its own: a
`reasoning_effort` overlay and an `llm_parameters` lever hashing the model parameters it resolved into, say.
Without it every sweep of the knob reports two moved levers, each confounded by the other. With
`Annotated[Literal["low", "high"], ResolvesInto("llm_parameters")]`, the surface folds into the knob wherever the
compared runs show every level of the knob carrying one level of the surface: it gets no coverage row or confound
of its own, and the arm is named by the knob (the surface stays in the key, listed in the variant index's
`folded`). Where runs at one knob level carried different surfaces, something else wrote into it, and it stays a
lever and a confound. A run whose surface reader returned `None` folds nothing. On a map field the marker rides on the map's own
lever, and the surface is compared with the swept entries removed.

Only arms can test a fold: every run of one arm resolves the same surface, so only a knob level held by two or
more arms (differing in, say, the model) can show the surface moving on its own. A plain two-arm `low` / `high`
sweep still folds, but every comparison that does carries an `unverified_fold:llm_parameters` confound, which the
analysis generator and the code-only report both state.

A residual reader checks that sweep. `ResolvesInto("llm_parameters", read_residual=reader)` takes a
`reader(run, results, removed)` that returns the surface with the knob's contribution taken out (`removed` is the
knob's lever name), as JSON-safe content, or `None` where the run cannot say: the model parameters without their
`reasoning_effort` entry, say. Where every knob level was run by one arm, the engine compares the residuals across
the runs. Equal, the fold is checked and carries no mark; unequal, something else wrote into the surface and it
stays a lever and a confound; any `None` leaves it `unverified_fold`. A host's own fixed lever takes the same reader
as `Sweepable(resolves_into=..., read_residual=...)`. With no reader, nothing changes.

The lever named must be a fixed `lever` your host declares, not an open family, and not one another knob already
names (`RegistrationError`); a field marked twice is refused (`KindContractError`). The marker moves no variant key.
A host's own fixed lever declares the same thing as `Sweepable(resolves_into=...)`. One surface takes one knob: if
several knobs hash into one surface, give each its own resolved lever.

You register neither anywhere else: the profile adds the contract's levers to its `sweepables`, and the
engine resolves every run's level of them — with its `candidate_model` and its `candidate_kind` — into the
variant key. Your profile's `host_sweepables` (the shared core extended with your own levers) and its
`variant_levers` reader cover only the levers you declare beyond those; a host with none wires no reader.

**How many cells your kind runs at once is yours to declare.** A run not measuring latency executes its cells
side by side, and one kind instance may then be driving several of them: whether that is safe — the clients it
shares across cells, a pool, a world it does not bind per cell — only your kind's code knows. So the kind
declares it, as `max_concurrent_cells` beside `judged_artifact`: `1` is serial, a larger int is that many, and
`None` is no limit of its own. **A kind that declares nothing runs one cell at a time.** Its launcher hands the
same value to the launch (`KindWiring(max_concurrent_cells=kind.max_concurrent_cells)`), which runs the run at
the narrowest of that, the host's `LaunchSettings.max_concurrent_cells` (a ceiling, 4 by default) and the matrix.
The runner holds every cell's kind to it, refusing a kind that declares less than the run executes at before
anything is spent.

**Your kind's paid calls answer to the run's cost cap as they are made.** The cell's sink keeps the cell's spend
by source and holds each priced call against the cap while it is in flight, so cells running side by side cannot
together pass it. Before a call whose worst case your client can price, reserve it — `held =
sink.reserve("candidate", ceiling)`, then `held.settle(cost)` when it returns and `held.release()` if it raises —
or wrap the client in `CappedCompletion(client, sink=sink, source="candidate")`, which does both for every call,
pricing each by the client's `price_ceiling`. A call that cannot be priced is reserved with no ceiling, which
only asks whether the cap is already reached. Spend you learn of any other way, report as it lands
(`sink.add_spend(source, cost)`); it adds, and never replaces what another source reported. A reservation the
cap refuses raises `CostCapReached`: a one-shot kind catches it and returns what it has, and inside
`drive_conversation` it simply propagates out of your candidate turn. The full rule is in [Cost and budgets:
caps](cost-and-budgets.md#caps).

A run is one arm with one `candidate_model`; a launch naming several models starts one run per model, and
runs of different kinds are always different variants. A reader of yours that answers for a run its lever
does not apply to (a run of another kind) returns `SweepableValue.not_this_kind(kind)` — the level a kind
contract's own levers sit at there — rather than a value of its own displayed alike: that level is what a
report recognises, by its content hash, as a lever the arm did not run, and never prints.

## Launching

**A kind's launcher resolves only what is the kind's.** `start_run` refuses a launch argument the
kind declares it cannot honour before the launcher runs, then hands the launcher a `LaunchRequest`.
The launcher captures the subject the request names (admission has already asked the kind's `resolve_subject`
whether the host holds it, so a launcher never meets one it does not), freezes the cases, builds its judge
(`build_judge_service`) if it has one, and returns `launch_run(host, request, KindWiring(...))`. The
engine stamps everything the request and the template already say — scope, model, repeats, cassette
mode, overlays, spec, world seed, tool bound, ceilings — and refuses a wiring that contradicts them.

Before any launcher runs, the engine asks each kind what each arm will run (`LaunchableKind.plan_arm`) and
prices it. `plan_arm` is therefore where a kind makes its request-level refusals (no model and no default —
`require_candidate_model` — a judge or simulator it needs), and the launch tail then holds each launcher to
its plan. The full rule is in [Cost and budgets: every arm is priced before any launcher
runs](cost-and-budgets.md#every-arm-is-priced-before-any-launcher-runs).

**Price a floor under your subject kind's judge and simulator calls.** A judged subject kind — a persona, a
classifier — registers `price_subject_call_floor` (from `threetears.evals.run`) as its `price_largest_call`:

```python
LaunchableKind(
    launch=your_persona_launcher,
    resolve_subject=your_persona_subject,
    plan_arm=plan_persona_arm,  # plan_judge(eval_host, request.template, pin, request.judge_config_ids, ...)
    price_largest_call=lambda request: price_subject_call_floor(
        eval_host, request, judged_artifact=JudgedArtifact.TRANSCRIPT  # or DOCUMENT: what your judge reads
    ),
)
```

It prices the smallest judge call every judged case makes, and the smallest utterance your simulated user is
bound to make, with the transcript and evidence left out, on the clients your cells call through. The estimate
and the launch then refuse an arm whose cap cannot afford even that, before anything is spent. That catches a
judge or simulator client whose output cap and retries alone put one call above the cap. Pass the `eval_host`
your launcher builds the judge on and the `judged_artifact` it hands `build_judge_service`. The figure is a
floor, not the largest call (`CallCeiling.lower_bound`): a cap above it is admitted, and a call its transcript
makes larger is still held to the cap as the run reserves it. A kind with no judge and no simulated user has
nothing to price here, and returns `None`.

**A transport launches in two halves.** `run_launch` admits the launch in its call (`admit_launch`: every refusal
above, every arm planned and priced, the runs' room reserved) and runs the rest as one job
(`launch_job_work`): each arm's launcher — where your launcher builds its rig and generates cases — then the runs,
waited for to their end. So your launcher runs inside a job and may call a model; `plan_arm` and your pricer run
in the call and must not. A script awaits both halves in a row through `start_run`.

### Your own actions are jobs too

A host adds its own actions to the catalogue (`eval_catalogue(my_actions)`). One whose work calls a model — a
rubric proposer, a drafter of a classifier's cases — is a `spend` action and keeps the engine's rule: **its call
makes no completion request.** It prices and refuses in the call, then starts the work as a job with
`start_action_job`, which writes an `EvalActionJob` record carrying the job's lease and, once the work returns,
its typed result as JSON — what `job_poll` returns. **Name its wall-clock ceiling** (`budget_s`): no action job
has a default, so a forgotten one is a `TypeError` where you start it rather than an hour nobody chose. Declare
it `long_running=True` with result `JobsStarted`. It
needs no `render`: long work reads back as the jobs it started, with `render_jobs_started` (exported from
`threetears.evals.actions`), which is the default, and which an action returning anything else may not use.

An action whose work is **one paid model call** takes `start_paid_call` (in `threetears.evals.ops`), which keeps
the whole rule for it: it builds the role's client, admits the call against your out-of-run budget in the action's
call (refusing, with nothing started or spent, a call the cap cannot pay for), starts `make` as the job, and
releases the client on every path that does not reach `make`. Work of any other shape builds the same steps on
`start_action_job`:

```python
async def _propose(host: OpsHost, caller: Caller, params: ProposeParams) -> JobsStarted:
    budget = my_budget(host, caller.scope_id)
    call = plan_draft(axis=params.axis, system_prompt=..., subject_feed=..., catalog_feed=..., reset=...)

    async def draft(client: BoundCompletionClient, admitted: AdmittedCall) -> RubricProposal:
        async with client:                                    # the job's work owns the client from here
            proposed = await propose_draft(client, admitted, budget=budget, axis=params.axis, subject_id=...)
        return proposed.proposal

    return await start_paid_call(host, caller.scope_id, action="rubric_propose", role="proposer", model=None,
                                 purpose="proposer", budget=budget, call=call, make=draft, label=params.topic,
                                 budget_s=600.0)                  # required: how long a draft may take

async def _propose_happy_path(stage: ConformanceStage) -> Mapping[str, Any]:
    return {"topic": "tone"}                             # what reaches the work, with anything it must arrange

Action(name="rubric_propose", permission="spend", result=JobsStarted, long_running=True,
       handler=_propose, happy_path=_propose_happy_path, ...)
```

Every `spend` action declares a `happy_path`, and one without is refused where it is declared. **Prove it with the
jobs conformance kit**: build your `OpsHost` over a fresh store with your client factory wrapped by the kit's
guard, name a sample launch your host runs end to end and judges through that factory, and parametrise over
the cases. Every action's call must make no completion request; every spend action's happy path must return
`JobsStarted`, and its job, polled to its end, must make at least one — a happy path that never reaches the work
is refused as vacuous. Route every client your actions use through the factory you hand the guard; a client
built outside it is not watched.

```python
from threetears.evals.testing import CompletionGuard, JobsConformanceHost, jobs_conformance_cases

def build(guard: CompletionGuard) -> JobsConformanceHost:
    host = my_ops_host(store=InMemoryDocumentStore(), clients=guard.wrap(my_clients))
    return JobsConformanceHost(host=host, caller=Caller(scope_id="dev", identity="kit"),
                               sample=ConformanceSample(template_id="t", subject_id="s", models=("m",),
                                                        second_judge_model="judge/other"))

@pytest.mark.parametrize("case", jobs_conformance_cases(eval_catalogue(my_actions)), ids=lambda case: case.name)
async def test_no_call_waits_on_a_model(case: JobsConformanceCase) -> None:
    await case.check(build)
```

### Hearing about a regression when a run completes

`scope_history` flags a regression only when someone reads it. To hear about one as soon as a run
completes, build a `RegressionWatch` (from `threetears.evals.ops`) over your host, a sink of your own and
the measures to track, and hand it to the `LaunchHost` as `on_run_end`:

```python
watch = RegressionWatch(host=eval_host, sink=page_the_owner, measures=("composite", "cost_usd"),
                        min_absolute_change=0.2)
launch_host = LaunchHost(eval_host=eval_host, ..., on_run_end=watch)
```

When a run's `completed` status is stored, the watch reads each measure's history with the same test and
thresholds `scope_history` uses. For each of the run's contestants whose step into this run reads
`regressed`, it awaits your sink with one `RegressionAlert`. The alert names the contestant, the measure
and the two runs of the step, and carries the history's flag whole: the verdict, its test and its
thresholds. It also states whether code or a judge graded the measure, and the judges' evidence tier.

A judged measure fires only when every judge behind the step's scores is `calibrated` (see
[Judges and calibration](judges-and-calibration.md)). A run that ended any other way than `completed` is
not a point on the history, so it is never checked. The engine ships no delivery: with no watch, nothing
is checked. A sink that raises is logged, the next alert is still delivered, and the run's status never
changes.

### Generating cases at launch

A launch with `n_variations` > 0 asks for that many new cases from the template's variation axes,
generated once for every arm:

- Call `generate_variations` inside `request.launch_group.resolve_once(...)`, passing
  `blocking_executor=host.blocking_executor` (its case reads and writes run there; its model calls stay
  on the loop), and hand its counts on as `KindWiring(variation_counts=...)`.
- An `llm` axis is written by the model the launch names as `variation_model` — required then, refused
  when nothing would call it. The launch's call builds the writer from the host's client factory in the
  `variation` role (never the simulator's), plans and admits its calls there, and hands them to the launcher
  as `request.generation`. The launcher builds no writer of its own.
- The run records the model the client resolved to on `variation_counts.variation_model`; it enters no
  identity, because the cases it wrote are already hashed through `test_case_ids`.
- The launcher hands `generate_variations` the request's `generation=request.generation`, which makes exactly
  the calls the launch admitted. A rubric proposal is planned (`plan_draft`), admitted in your call and made in
  your job (`propose_draft(client, admitted, ...)`) the same way.

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

## The engine's prompts: seeded into your registry

The engine steers three LLM calls of its own with a prompt: the analysis generator (`ANALYSIS_GEN_PROMPT`,
one string), the capability proposer (`PROPOSER_PROMPT`) and the boundary proposer (`BOUNDARY_GEN_PROMPT`),
both sectioned templates (`SeedSection`'s zones are [provisional](#provisional-mechanisms)). Each is store-master. Your prompt registry holds the copy that runs, an operator
tunes it with no deploy, and the engine ships only the default a fresh deployment starts from.

`SEED_PROMPTS` (from `threetears.evals.kernel`) is that list, one `SeedPrompt` per prompt. Each names its id
(`prompt_type`), which registry seeds it (`kind`: `template` or `text`), and where the default lives:
`seed_ref` is the `(module, constant)` to import it from. Seed an empty slot from each entry and never overwrite a
stored one:

```python
for entry in SEED_PROMPTS:
    module, constant = entry.seed_ref
    default = getattr(importlib.import_module(module), constant)
    registry.seed_if_empty(entry.prompt_type, entry.kind, default)  # your registry's own call
```

Key the slots by the exported constants rather than by the strings. Declare your own prompts with the
same `SeedPrompt` type and seed from both lists. The engine's ids start with `eval_`, so your own can use
any other name. A `SeedPrompt` refuses an unknown `kind` or a blank field when it is constructed.

**See a stored copy the package has moved past, before you pay for it.** A slot seeded from an older default
keeps being sent after an upgrade changes the default, and if the change is one the package's own validation
depends on, every call it steers can fail. Record `seed_digest(default)` when you seed a slot, and on start-up
read each slot with `prompt_drift`:

```python
drift = prompt_drift(entry, shipped=default, stored=stored_copy, seeded_digest=recorded_digest,
                     reset=f"call prompt_reset with prompt_type={entry.prompt_type!r}",  # your own reset
                     customized=operator_owns_it)
if drift.stale:
    log.warning("%s: %s", drift.prompt_type, drift.remedy)
```

A stored copy reads `current` when it is the shipped default, `customized` when it changed since you seeded it
or you say it is deliberately yours (`customized=True`), and `stale` otherwise: untouched since seeding, or with
no record of its seeding, and not the default. Pass a template's stored copy as
`SeedTemplate.from_dict(payload)` and a text prompt's as its string. The digest names content, not the revision
number. A copy you tune is never reported stale. A provenance stamp does the same job as a recorded digest: if your registry marks a slot as
the system's seed until someone edits it, pass `customized=True` for an edited slot and `seeded_digest=None`, and an
untouched slot the package has moved past reads `stale`.

**A proposer prompt states the draft shape it was written for, and the engine checks it.** A tuned copy is
never stale, but it can still teach a draft shape the engine no longer reads. So each shipped proposer
default ends with its output-contract line, `eval-output-contract: rubric_proposal <fingerprint>`
(`gen.RUBRIC_PROPOSAL_CONTRACT.line`). The fingerprint is the digest of the draft's schema as the call sends
it, its field documentation aside, and of a revision for the rules the schema cannot state. `plan_draft`
reads the rendered prompt you hand it, before anything is priced, and raises `PromptContractError` (a
`ConflictError`, status 409) when the prompt carries another fingerprint (`outdated`) or none
(`undeclared`, which is every copy seeded before the line existed). The error names the prompt
(`prompt_type`) and the remedy: reset the slot to the shipped default, or keep an edit by rewriting its
output instructions for the current shape and making its line read the current one. The registry is yours,
so the engine cannot name your reset: pass it as `reset=` (an action or a command your operator can run), and
the remedy names it. `prompt_drift` takes the same `reset=`. An edited copy that carries the current line passes:
the line is the editor's statement that the edit was written for this shape, and the strict response
format and the draft's validation still stand behind it. The analysis generator's prompt states no shape
(its schema is sent and its grammar appended by code), so it binds no contract.

`prompt_drift` reads the same thing before any call. Each reading carries `contract` (`current`,
`outdated`, `undeclared`, or `unbound` for a prompt with no contract), `expected_contract` (the line the
shipped default carries) and `refused`, which is true when a call the copy steers is refused, whether the
copy is stale or customized. Show `refused` readings with their `remedy` wherever you list your prompts,
so an operator sees a copy that will be refused before a call is.

```python
if drift.stale or drift.refused:
    log.warning("%s: %s", drift.prompt_type, drift.remedy)
```

After an upgrade that moves a contract's fingerprint, every stored copy of that prompt is refused until it
is reset or its line updated. Check `prompt_drift` on start-up, so the reset happens before the first call
is refused.

### Launching the analysis reporter

The analysis reporter (`REPORTER_KIND`) evaluates the analysis writer itself. Its cases are frozen campaign
bundles (see [Agents and MCP](agents-and-mcp.md#reporter-runs-evaluating-the-analysis-writer-itself)). A
candidate is a preset of the `ANALYSIS_GEN_PROMPT` prompt on a generator model, or `AS_RECORDED_MODEL`,
which replays the memo each campaign got.

The kind is declared by the engine, so your host supplies only two things:

- **Your prompt registry**, as a `ReporterPrompts`: `resolve(preset)` returns that preset's text, or the
  system default for `None`. It returns `None` for a preset you do not hold, never the default in its
  place. A registry that fell back would bill the incumbent prompt under the candidate's name.
- **Your model resolution**, as a `ReporterModelResolver`: `(role, named)` returns the provider model,
  or the role's default when `named` is `None`. The roles are `generator` and `judge`. Raise
  `ValidationFailedError` for a name you do not know.

Declare `REPORTER_KIND_CONTRACT` (from `threetears.evals.analysis`) on your profile's `kinds`. Its one
overlay is `prompt_preset`, and it seats the judge and the spend ceiling. If the profile refuses it for
leaving one of your own apparatus dimensions unseated, build it with
`reporter_kind_contract(host_seats=[...])`. On a reporter template, your `refuse_undeliverable_template`
hook calls `refuse_undeliverable_reporter_template`.

Then register the kind's launch entry (from `threetears.evals.ops`):

```python
LaunchableKind(
    launch=your_reporter_launcher,
    resolve_subject=your_reporter_subject,  # admits the one subject id your reporter launches label
    unhonoured_launch_arguments=REPORTER_UNHONOURED_LAUNCH_ARGUMENTS,
    plan_arm=lambda request: plan_reporter_arm(eval_host, request, prompts=load_prompts(), resolve_model=resolve),
    price_largest_call=lambda request: price_largest_reporter_call(
        eval_host, request, prompts=load_prompts(), resolve_model=resolve
    ),
)
```

`price_largest_reporter_call` prices each case's first generator request before anything is spent, built as
the cell builds it, on the client your factory builds in the `analysis` role for the arm's model. The
estimate and the launch then refuse an arm whose largest generator call is reserved above its cap. Without
it, that call is refused only when the run reaches it, after the cases before it were paid for. Pass the
`eval_host` your launcher builds the generator client on, so the price is that client's. The memo's judge
call stays unpriced, since the memo it judges does not exist yet.

`plan_reporter_arm` and `admit_reporter_launch` make every refusal before the arm is priced, in this order:

1. a template asking for apparatus a memo writer cannot honour, or with no rubric;
2. a preset you do not hold, or no default;
3. a model named by alias (a name you resolve to another model), since the run records the name and an
   alias's target moves;
4. a template with no live case, or a campaign-and-memo pair holding two live cases. When the launch
   names a case set, its cases run instead, and each must carry a reporter case;
5. the replay over a case with no recorded memo.

Your launcher calls `admit_reporter_launch` again, off the event loop. It builds `ReporterKind` (prompt id
`ANALYSIS_GEN_PROMPT`) or `AsRecordedReporterKind` from what that returns, and wires its client, subject,
cell ceiling (`reporter_cell_timeout_s`) and judge as for any kind.

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
  scheduled), and fires or observes its triggered dimensions (`fire`, `observe`; triggered dimensions are
  [provisional](#provisional-mechanisms)).
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
sink: every simulator call is reserved against the run's cost cap at the client's own ceiling before it is
made (a client with no `price_ceiling` has the cap asked instead), and the loop asks before every decision
whether the cap is reached (`CellSink.cost_cap_reached`), stopping `budget_stopped` when it is — the runner
excludes that cell and ends the run `budget_stopped`, so one conversation (thousands of simulator calls at the
schema's maxima) cannot run past the cap. Your `candidate_turn` holds the candidate's own calls the same way;
when the cap has no room it raises `CostCapReached` (a refused `sink.reserve` raises it for you), and the
conversation ends `budget_stopped` — not `candidate_error` — with the rounds already answered kept. Fold the driver's calls with `fold_usage`: the stored
`simulator` rows are one per actor and purpose (`RoleUsage.actor_id`, `RoleUsage.purpose`).

**A round can be the world's.** `ConversationSpec.world_rounds` names rounds whose stimulus is a triggered
world dimension firing rather than an actor's line. Before that answer, the loop fires the dimension through
the cell's world session (pass `world=`), and the case's seed must arm it. Your `candidate_turn` receives one
turn spoken by `WORLD_SPEAKER` naming what fired, and nothing is posted. A template whose world supplies every
round declares no actors at all, and its conversation makes no simulator call.

## Background work, payloads and spend

**Background work** a candidate hands off and gets back turns later is recorded as `async_deliveries`,
one `AsyncDelivery` each: who asked, when it was acknowledged and delivered, on what model, whether a
harness supplied it, and what it spent — its tokens, model calls, `cost_usd` with its own
`price_source`, and any paid non-LLM calls (`external_spend`) — their calls, the provider's own units, and
`money` where the provider itself billed a figure. The engine folds that spend into the result's
`inner_agent` and `external` usage, for work still in flight when the cell ended as well as work that
delivered; a substituted entry reports no spend. Both shapes and both roles are
[provisional](#provisional-mechanisms).

**Payloads.** Anything else your kind wants kept with a result goes in `kind_payload`, which the engine
stores and never reads; register a measure for whatever should be compared.

**Spend.** Your kind reports its own calls' spend only as usage rows (`CandidateTelemetry.usage`). How
the engine turns those rows into a result's `cost_usd`, and what happens when a call cannot be priced, is
in [Cost and budgets: how a result's cost is counted](cost-and-budgets.md#how-a-results-cost-is-counted).

## Cassettes: recording and replaying tools

A [cassette](concepts.md#cassette) lets two arms face exactly the same tool answers. `run_eval` and `compare`
do the wiring below for you (`tools=`, `cassette_mode=`; `examples/cassettes.py`); this section is for a kind of
your own.

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
- **Background work.** A delivery seam reports each piece of work where it starts (`recorder.started(request)`)
  and settles it where it ends; under replay it takes `replay.next(request)` instead, reporting
  `substituted=True`.
- **Matching.** Every recording is keyed by what was asked and which time it was asked, so a repeated dice roll
  replays both rolls in order; an ask the capture never made, or made fewer times, stops the cell as the rig's
  failure rather than running live.
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
that errored: a judge's "can't tell" is an answer. Every dim the judge phase asked records the prompt that asked
it in `EvalResult.judge_config_ids`, an unfinished one included: a config's id with the reply format's
fingerprint (`<config id>@<sha12>`; `judge_config_of` reads the id back), or the built-in prompt's `builtin:`
identity. A re-judge refuses a dim whose recorded identity is not the prompt it would send today.

## Provisional mechanisms

Each mechanism below exists because the first host to adopt the engine needed it. Each was generalised out of
that host's vocabulary, and none has yet been used by a second host. Use whichever fits your subject. Your
adoption, as the second host, decides whether each one stays an engine concept. A mechanism no second host uses
may be removed in a later release, and that release's `MIGRATIONS.toml` notes say so (the core ones then move
with a core major and an upgrader). None is deprecated, and nothing here changes what is stored. Two of them hang
off [`host-contract`](#stability-tiers) names: `KnownUsageRole` carries `inner_agent` and `external`, and
`WorldDimension` reaches `Triggered` through its `when`. Removing either is a signature change to a contract name,
so it goes through the deprecation cycle, never straight out in one release's notes.

- **`AsyncDelivery` and `AsyncExternalSpend`** (on `EvalResult.async_deliveries`, a core document). They cover
  background work a candidate hands off and gets back turns later, with its own model spend and the paid
  non-LLM calls it made. Skip them and the engine does not see work that finishes after the turn that asked
  for it: its spend, its timing and whether a harness substituted its payload are yours to keep, in
  `kind_payload`, which the engine never reads ([Background work](#background-work-payloads-and-spend)).
- **The `inner_agent` and `external` usage roles** (`KnownUsageRole`). They put a background task's own model
  calls and its paid third-party calls on spend lines of their own. Skip them and that spend is not in the
  role breakdown or the blended cost.
- **`MeteredCallLedger`** (`run.metering`). It caps the third-party calls a run's tools may make, counted at
  your action dispatcher (`max_metered_calls`), for a provider quota the cost cap cannot see (search credits,
  a billed API). Skip it and a run's third-party quota is bounded only by your tools' own configuration.
- **Triggered world dimensions** (`Triggered`, `ConversationSpec.world_rounds`). State that arrives on a
  condition (a turn, an event, a person) rather than at the first turn: a case's seed arms it, and a world
  round or your kind fires it. Skip them and your world is set once, before the first turn, and no goal check
  can ask what `fired(...)` ([A world](#a-world-through-the-cells-session)).
- **`ChartFont`** (`kernel.host`). A brand face for chart text, declared with the per-character advance widths
  a renderer lays labels out by. Skip it and charts are drawn in the renderer's own face; nothing a chart
  states changes.
- **`SeedSection`'s zones** (`primacy`, `reference`, `recency`, in `kernel`). The sectioned prompt template the
  two proposer prompts are declared in, ordered by zone, each section with its own condition, formatter and
  cache class. Without a registry for sectioned templates, you render those sections into the system prompt
  you hand the proposer yourself (`plan_draft` reads the rendered prompt;
  [The engine's prompts](#the-engines-prompts-seeded-into-your-registry)).
