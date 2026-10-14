# Reference

**For:** people who already know the engine and agents that need exact names. It answers: what does this root export, what does this setting or field mean, what can a goal check say, which actions and commands exist, and what does each measure measure. To learn the engine, start with the [ladder](ladder.md) instead.

**Generated** from the package by `packages/evals/scripts/generate_reference.py`. Do not edit it by hand: change the docstring, field description or definition it came from, then run `uv run python packages/evals/scripts/generate_reference.py`. `tests/test_reference_doc.py` fails while this page is stale.

## Contents

- [Public API](#public-api)
- [Configuration](#configuration)
- [The report and the analysis bundle](#documents)
- [The goal-check language](#goal-checks)
- [The action catalogue (MCP)](#actions)
- [The command line](#cli)
- [Measures](#measures)

<a id="public-api"></a>
## Public API

Import only from these roots: a module under a root is internal and may move. The names below are the declared surface, mapped by rung in [the surface](surface.md). Within a root, names are grouped as functions, classes, types and constants, each sorted by name. A class's kind says what it is: a `model` is a Pydantic model, a `protocol` is something a host implements. A name marked *host contract* is one a host writes against: it moves only with a deprecation cycle. Every other name listed may move in any release, recorded in `MIGRATIONS.toml`. A root also exports *provisional* names, not listed here: most are types a signature below names, importable so your code can annotate them, and none is promised; `packages/evals/api/surface.json` lists them ([stability tiers](adopting-a-host.md#stability-tiers)).

- [`threetears.evals`](#api-threetears-evals)
- [`threetears.evals.schema`](#api-schema)
- [`threetears.evals.kernel`](#api-kernel)
- [`threetears.evals.kernel.host`](#api-kernel-host)
- [`threetears.evals.run`](#api-run)
- [`threetears.evals.analysis`](#api-analysis)
- [`threetears.evals.gen`](#api-gen)
- [`threetears.evals.storage`](#api-storage)
- [`threetears.evals.testing`](#api-testing)
- [`threetears.evals.quick`](#api-quick)
- [`threetears.evals.ops`](#api-ops)
- [`threetears.evals.actions`](#api-actions)
- [`threetears.evals.transports.fastmcp`](#api-transports-fastmcp) (fails to import without the `[fastmcp]` extra)
- [`threetears.evals.negotiation`](#api-negotiation)
- [`threetears.evals.providers`](#api-providers)

<a id="api-threetears-evals"></a>
### `threetears.evals`

The eval engine: templates, cases, runs and results; the trial loop, its judge and simulated user; and the analysis over what runs produced.

**Functions**

- **`declared_names`** · *host contract* · function · Every declared public name of the installed build (`host-contract` and `stable`), by `root.name`.
  <br>`declared_names() -> frozenset[str]`
- **`grader`** · *host contract* · function · Declare what a reading needs beside its grade: `@grader(guard=0.02)`, `grader("total", margin=0.05)`.
  <br>`grader(target: Callable[[Mapping[str, Any], Any], float | bool] | str | None = None, /, *, margin: float | None = None, guard: float | None = None, range: tuple[float, float] | None = None) -> Grader | Callable[[Callable[[Mapping[str, Any], Any], float | bool]], Grader]`
- **`surface_tier`** · *host contract* · function · A public name's stability tier in the installed build, by `root.name`.
  <br>`surface_tier(qualified_name: str) -> Literal['host-contract', 'stable', 'provisional']`

**Classes**

- **`Answer`** · *host contract* · dataclass · What a candidate returns to report its own spend beside its answer.
- **`Eval`** · *host contract* · class · What is evaluated: the cases, the runner under test, and what counts as right.
- **`Example`** · *host contract* · dataclass · An answer the judge must grade a known way: one that must fail a dimension, or one that looks wrong and must pass.
- **`Judge`** · *host contract* · class · A model that grades each answer on a rubric, one call per dimension: a grader in an eval's `grader` list.
- **`Log`** · *host contract* · class · A runner whose outputs an app pushed to a JSONL file, and the call that pushes one.
- **`Model`** · *host contract* · class · One model, as the engine's completion client and as three helpers for a runner of your own.
- **`Result`** · *host contract* · dataclass · What an eval ran — one arm, or every arm of a comparison — and what its campaign's report says.
- **`RootExtra`** · *host contract* · dataclass · What one optional public root needs installed before it imports: the extra, and the modules it brings.
- **`World`** · *host contract* · class · A world: state each case seeds and tools the agent changes it with, or a set of lookups the agent calls.

**Constants**

- **`__version__`** · *host contract* · constant (str) · The installed distribution's version, as its metadata states it (`importlib.metadata`): the build a host records beside what it stores, since no stored document or analysis bundle carries it.
- **`PUBLIC_ROOTS`** · *host contract* · constant (tuple) · The public roots, as absolute module names.
- **`ROOT_EXTRAS`** · *host contract* · constant (mappingproxy) · Each public root that fails to import without an extra installed, mapped to what it needs (`RootExtra`): the extra's name, and the modules that extra provides which the root imports.

<a id="api-schema"></a>
### `threetears.evals.schema`

The engine's stored shapes and the ports a host implements: what is written down, and what writes it.

**Functions**

- **`label_key_of`** · function · The label key of one judgement: `evidence` read on `dim`.
  <br>`label_key_of(evidence: JudgeEvidence, dim: RubricDim | str) -> LabelKey`
- **`nested_schemas`** · function · Every schema written directly inside `schema`, in declaration order.
  <br>`nested_schemas(schema: Mapping[str, Any]) -> Iterator[NestedSchema]`
- **`sent_settings_of`** · *host contract* · function · What the request that produced `completion` actually carried, as its client reports it, or `None`.
  <br>`sent_settings_of(completion: object) -> ClientRequestSettings | None`

**Classes**

- **`ActorPolicy`** · model · One simulated actor in a template's conversation.
- **`AsyncDelivery`** · model · One piece of background work the candidate started: acknowledged at once, delivered later.
- **`AsyncDocumentStore`** · *host contract* · protocol · `DocumentStore`, every method awaited: the port for a host whose database driver is async.
- **`AsyncExternalSpend`** · model · Paid non-LLM calls one piece of background work made at one provider, as the work reports them.
- **`AsyncStoreClock`** · *host contract* · protocol · `StoreClock` for an `AsyncDocumentStore`: the store's own time, awaited.
- **`BoundCompletionClient`** · protocol · A completion client built for one model, which it names.
- **`CalibrationRating`** · model · A rater's score for one judged dimension of one result — a person's, or an agent's.
- **`CallLedger`** · model · The calls one cell's candidate made that succeeded, across every tool, in recorded order, and those refused.
- **`CatalogRubricDim`** · model · A reusable rubric dimension in the shared catalog.
- **`CellIdentity`** · dataclass · Which cell execution a stretch of work belongs to.
- **`CellTrace`** · dataclass · What one cell's tracing yielded — filled by the sink, read by the engine.
- **`ClientRequestSettings`** · *host contract* · model · Request parameters for one apparatus role's LLM call: what the engine ASKED for, or what a request was SENT with.
- **`CompletionClient`** · *host contract* · protocol · The completion port eval is constructed with.
- **`CompletionProtocolDefect`** · *host contract* · exception · A host's completion does not meet the `CompletionResult` protocol, found while the engine read it.
- **`CompletionResult`** · *host contract* · protocol · What eval reads off one completion, whatever produced it.
- **`ControlEndState`** · model · An end state a template's author states, to prove its goal checks can tell outcomes apart.
- **`ConversationSpec`** · model · The simulated side of a conversing candidate: who talks to it, in what order, for how long.
- **`ConversationStopCause`** · enum · Why a conversation trial's turn loop stopped — one structural signal, never a reading of prose.
  <br>values: `'max_turns'`, `'user_done'`, `'participants_ended'`, `'simulator_error'`, `'apparatus_error'`, `'candidate_error'`, `'budget_stopped'`
- **`DocumentStore`** · *host contract* · protocol · A narrow document store over one `(scope_id, doc_type, id)`-keyed collection.
- **`EvalBaseModel`** · model · Base model for the eval engine's own Pydantic models.
- **`EvalCaseStratum`** · model · What the analysis reads off a test case: which case, and the stratum it declares.
- **`EvalResult`** · *host contract* · model · One test case x one model x one k-iteration.
- **`EvalRun`** · *host contract* · model · One execution of a template (or explicit test case set) against one candidate model.
- **`EvalTemplate`** · *host contract* · model · Abstract scenario blueprint — subject-agnostic, domain-level.
- **`EvalTestCase`** · *host contract* · model · Concrete, immutable inputs generated from an `EvalTemplate`.
- **`EvalTrace`** · *host contract* · model · The candidate's output, what its judge read, and the OTel spans — stored beside a result, not inside it.
- **`ExternalSpend`** · dataclass · One caller's report of what a provider call (or a batch of them) consumed.
- **`GoalCheckControl`** · model · One goal check's intent, and the end state that proves it discriminates.
- **`GoalCheckControls`** · model · Proof, at authoring, that each of a template's goal checks can tell its outcomes apart.
- **`GoalStateOutcome`** · model · One judge-free fact a candidate's execution established, and whether it held.
- **`IntervalScale`** · *host contract* · model · A real number with real spacing — `0.4`, `15`, `2000ms`.
- **`JudgeConfig`** · model · Versioned judge configuration for one rubric dimension.
- **`JudgedArtifact`** · *host contract* · enum · What a judge reads of a kind's output — the kind's declaration, which picks the judged axes.
  <br>values: `'transcript'`, `'document'`, `'unjudged'`
- **`JudgeEvidence`** · model · Everything a judge reads about one cell's candidate, rendered by the kind that ran it.
- **`LatencyMetrics`** · model · Per-result latency decomposition: harvested OTel spans, plus what they miss.
- **`OutOfRunSpend`** · model · One call the engine made outside any run, as it was admitted and as the provider reported it.
- **`Precondition`** · model · One thing a template presumes about the world before the subject's first turn.
- **`ProviderFailure`** · dataclass · What eval is allowed to know about a completion call that raised.
- **`ProviderFailureDescriber`** · *host contract* · protocol · The host's mapping from its own exceptions onto `ProviderFailure`.
- **`RecordedCall`** · model · One call a candidate made that succeeded — or, in a control end state, one it is stated to have made.
- **`RefusedCall`** · model · One call a candidate made that its tool refused: nothing changed, and the refusal said why.
- **`RoleUsage`** · *host contract* · model · Per-role token + cost observation for one `EvalResult`.
- **`RubricDim`** · *host contract* · model · One judge-scored rubric dimension.
- **`RubricProposal`** · model · The validated DRAFT the rubric proposer returns for operator review.
- **`RubricScore`** · model · Outcome of one rubric judge dimension.
- **`SecondJudge`** · model · A judge other than the one a run was scored by: a model, the prompt per dimension, and a temperature.
- **`StoreClock`** · *host contract* · protocol · A store's own clock: the optional capability a `DocumentStore` may offer beside its methods.
- **`StoreConflict`** · *host contract* · exception · A conditional write lost its race: the stored document no longer carries `if_match`.
- **`SubjectSnapshot`** · *host contract* · model · The subject a set of observations was taken against, frozen at capture.
- **`SweepableValue`** · *host contract* · model · One level of one swept input: what it *is*, what to call it, and what kind of axis it is on.
- **`TraceSink`** · *host contract* · protocol · The engine's whole reach into a host's tracing.
- **`VariationAxis`** · model · One axis along which test cases vary for a template.
- **`VariationCounts`** · model · How many test cases a launch asked generation for, and how many it froze.
- **`WorldSeed`** · *host contract* · model · Initial state for the eval's stateful world.

**Types**

- **`AsyncDeliveryStatus`** · literal · Where one piece of background work stood when the cell ended.
  <br>`'delivered'` | `'failed'` | `'undelivered'`
- **`KnownCellTimeoutOrigin`** · *host contract* · literal · A value of `CellTimeoutOrigin` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'launch'` | `'kind'` | `'default'`
- **`KnownCostCapOrigin`** · *host contract* · literal · A value of `CostCapOrigin` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'chosen'` | `'inherited'` | `'uncapped'`
- **`KnownMeteredCallOrigin`** · *host contract* · literal · A value of `MeteredCallOrigin` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'chosen'` | `'inherited'` | `'uncapped'` | `'none_declared'`
- **`KnownModelRoleOrigin`** · *host contract* · literal · A value of `ModelRoleOrigin` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'chosen'` | `'inherited'`
- **`KnownOutOfRunPurpose`** · *host contract* · literal · A value of `OutOfRunPurpose` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'variation'` | `'proposer'` | `'analysis'` | `'judge'` | `'second_judge'`
- **`KnownReasoningEffort`** · *host contract* · literal · A value of `ReasoningEffort` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'max'` | `'xhigh'` | `'high'` | `'medium'` | `'low'` | `'minimal'` | `'none'`
- **`KnownRefusalKind`** · *host contract* · literal · A value of `RefusalKind` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'refused'` | `'not_found'` | `'conflict'` | `'validation_failed'` | `'storage_failed'` | `'provider_refused'` | `'admission_refused'`
- **`KnownRoleModelOrigin`** · *host contract* · literal · A value of `RoleModelOrigin` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'chosen'` | `'inherited'` | `'alternate'`
- **`KnownSimulatorPurpose`** · *host contract* · literal · A value of `SimulatorPurpose` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'utterance'` | `'schedule'`
- **`KnownUsageRole`** · *host contract* · literal · A value of `UsageRole` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
  <br>`'candidate'` | `'judge'` | `'simulator'` | `'inner_agent'` | `'external'`
- **`RefusalKind`** · *host contract* · type alias · What refused a call or ended a job: the public engine error it was, by kind.
  <br>`str`
- **`RubricAxis`** · literal · The two rubric axes a dimension sits on: what a subject should DO (`capability`) and what it should refuse or withstand (`boundary`).
  <br>`'capability'` | `'boundary'`
- **`StopReason`** · *host contract* · literal · Why a completion stopped, in the engine's words.
  <br>`'end_turn'` | `'max_tokens'` | `'content_filter'` | `'truncated'` | `'error'`
- **`WorldPlacement`** · literal · What a RUN did with a dimension, computed from that run's own record and nothing else.
  <br>`'representable'` | `'judge_only'` | `'witnessed'` | `'out_of_play'`

**Constants**

- **`CASSETTE_SCHEMA_VERSION`** · *host contract* · constant (int) · The version every cassette is written under, and the only one a read of one accepts.
  <br>`= 1`
- **`CELL_TIMEOUT_ORIGINS`** · *host contract* · constant (tuple) · The values of `CellTimeoutOrigin` this build knows, in the order its readers list them.
- **`CORE_BASELINE_VERSION`** · *host contract* · constant (int) · The oldest core major this build reads: always `CORE_SCHEMA_VERSION`, since the core ships no upgrader and a core major is a baseline reset that moves both.
  <br>`= 15`
- **`CORE_SCHEMA_MINOR`** · *host contract* · constant (int) · The core minor this build writes within `CORE_SCHEMA_VERSION`.
  <br>`= 2`
- **`CORE_SCHEMA_VERSION`** · *host contract* · constant (int) · The core version this build writes. Every bump is a baseline reset, appends one line here and moves `CORE_BASELINE_VERSION` with it.
  <br>`= 15`
- **`COST_CAP_ORIGINS`** · *host contract* · constant (tuple) · The values of `CostCapOrigin` this build knows, in the order its readers list them.
- **`DEFAULT_JUDGE_TEMPERATURE`** · constant (float) · The temperature every judge call is requested at unless a `JudgeConfig` for its dimension says otherwise, and that config's own default (#633).
  <br>`= 0.0`
- **`DOC_TYPE_FACTS`** · constant (mappingproxy) · Every `doc_type` the engine stores, and its facts.
- **`DURABLE_BASELINE_VERSION`** · *host contract* · constant (int) · The first durable version, and the oldest every build reads, forever: it never moves, so a durable document any release since `.25` wrote stays readable (`tests/test_durable_schema_pins.py` pins it).
  <br>`= 15`
- **`DURABLE_DOC_TYPES`** · constant (frozenset) · What people author, by `doc_type`: the `durable` entries of `DOC_TYPE_FACTS`, derived as `CORE_DOC_TYPES` is.
- **`DURABLE_SCHEMA_MINOR`** · *host contract* · constant (int) · The durable minor this build writes within `DURABLE_SCHEMA_VERSION`, under the core's minor rules.
  <br>`= 2`
- **`DURABLE_SCHEMA_VERSION`** · *host contract* · constant (int) · The durable version this build writes.
  <br>`= 15`
- **`EVAL_DOC_TYPES`** · constant (tuple) · Every `doc_type` the engine writes — the set the operator wipe sweeps.
- **`METERED_CALL_ORIGINS`** · *host contract* · constant (tuple) · The values of `MeteredCallOrigin` this build knows, in the order its readers list them.
- **`MODEL_ROLE_ORIGINS`** · *host contract* · constant (tuple) · The values of `ModelRoleOrigin` this build knows, in the order its readers list them.
- **`NON_TERMINAL_RUN_STATUSES`** · constant (frozenset) · Statuses a run can still leave under its own power — it is either queued or executing, and something in-process is expected to write its terminal status.
- **`OUT_OF_RUN_PURPOSES`** · *host contract* · constant (tuple) · The values of `OutOfRunPurpose` this build knows, in the order its readers list them.
- **`PROSE_SCHEMA_KEY`** · constant (str) · The JSON Schema keyword marking a string property as model prose, for vocabularies declared as schema rather than as Pydantic models (a world dimension's value schema).
  <br>`= 'x-model-prose'`
- **`REASONING_EFFORTS`** · *host contract* · constant (tuple) · The values of `ReasoningEffort` this build knows, in the order its readers list them.
- **`REFUSAL_KINDS`** · *host contract* · constant (tuple) · The refusal kinds this build knows, one per public engine error class, in the order its readers list them.
- **`REGENERABLE_SCHEMA_VERSION`** · *host contract* · constant (int) · The version every regenerable document is written under, the cassette apart (`CASSETTE_SCHEMA_VERSION`), and the only one a read of one accepts.
  <br>`= 11`
- **`RESERVED_DIM_IDS`** · constant (frozenset) · The reserved dim ids, which are deliberately NOT namespaced: they identify the two dual-score axes rather than a rubric dimension scored in some context, so there is no context to name. `require_namespaced_dim_name` exempts them — a judge service is built for these ids on every run.
- **`ROLE_MODEL_ORIGINS`** · *host contract* · constant (tuple) · The values of `RoleModelOrigin` this build knows, in the order its readers list them.
- **`SIMULATOR_PURPOSES`** · *host contract* · constant (tuple) · The values of `SimulatorPurpose` this build knows, in the order its readers list them.
- **`USAGE_ROLES`** · *host contract* · constant (tuple) · The values of `UsageRole` this build knows, in the order its readers list them.

<a id="api-kernel"></a>
### `threetears.evals.kernel`

The engine's kernel: the behaviour every other package runs on, over the shapes in `threetears.evals.schema`.

**Functions**

- **`confusion_cell`** · function · The value a classification reports under `confusion_cell`: `expected → predicted`.
  <br>`confusion_cell(expected: str, predicted: str) -> str`
- **`default_job_timeout`** · *host contract* · function · Enforce a job budget with `asyncio.timeout` and nothing else.
  <br>`default_job_timeout(budget_s: float) -> AsyncIterator[float]`
- **`extract_json`** · function · Extract a JSON object from provider output.
  <br>`extract_json(content: str) -> dict[str, Any]`
- **`extract_json_array`** · function · Extract a JSON array from provider output, handling markdown code blocks.
  <br>`extract_json_array(text: str) -> list[dict[str, Any]]`
- **`fold_phase_timings`** · function · Fold one delivery's carried phase timings into a result's accumulator, and say what could not be folded.
  <br>`fold_phase_timings(accumulator: dict[str, float], *, source_tool: str, timings: Any) -> list[str]`
- **`prompt_drift`** · function · Read a stored copy of a catalogue prompt against the package's shipped default.
  <br>`prompt_drift(entry: SeedPrompt, *, shipped: SeedTemplate | str, stored: SeedTemplate | str, seeded_digest: str | None, reset: str, customized: bool = False) -> PromptDrift`
- **`seed_digest`** · function · The canonical digest of a seed's content: what a host records when it seeds a slot from it.
  <br>`seed_digest(seed: SeedTemplate | str) -> str`
- **`withhold_failure_detail`** · *host contract* · function · Describe a failure by its class and nothing else — the safe default.
  <br>`withhold_failure_detail(exc: BaseException) -> ProviderFailure`

**Classes**

- **`ActionSeam`** · protocol · A kind's synchronous tools, as the cassette lane records and replays them.
- **`AdmissionRefusedError`** · *host contract* · exception · A launch refused because it would admit more unfinished runs than the process may hold (status 429).
- **`AdmittedCall`** · dataclass · A planned call its budget admitted, at the ceiling it was priced at.
- **`AuthoredJudgeCase`** · model · A calibration case written by hand: an output, the criterion it is judged on, and the verdict it must get.
- **`BarOverride`** · model · A standard this campaign holds itself to, tighter than the registered one.
- **`CampaignDesign`** · model · The declaration: what this campaign set out to learn, before it learned anything.
- **`CandidateKind`** · *host contract* · protocol · The two operations that vary between one evaluable subject and another.
- **`CandidateKindDefect`** · *host contract* · exception · A kind handed back output that contradicts its own `CandidateKind.judged_artifact`.
- **`CandidateOutput`** · *host contract* · model · What one candidate produced, as everything below the dispatch sees it.
- **`CandidatePreparationFailed`** · *host contract* · exception · A candidate could not be built, and the cell is cleanly excluded rather than scored.
- **`CandidateTelemetry`** · *host contract* · model · What one candidate's execution cost, and the windows nothing else timed.
- **`CellCassettes`** · *host contract* · protocol · One cell's handle on its run's cassette lane, as the kind driving that cell sees it.
- **`CellSink`** · *host contract* · protocol · One cell's record of where it stands, as the kind driving that cell reports into it.
- **`CellSpanWindow`** · *host contract* · protocol · One cell's two tracing windows, as the kind driving that cell sees them.
- **`ConflictError`** · *host contract* · exception · State conflict (status 409).
- **`ControlDeclaration`** · model · What held still, stated — because an absent control is a fact, not a null.
- **`DeliveryRecorder`** · protocol · Captures one asynchronous tool's background work for one cell.
- **`DeliveryReplay`** · protocol · Serves one asynchronous tool's recorded work for one cell.
- **`DeliverySeam`** · protocol · One asynchronous tool of a kind, as the cassette lane records and replays its background work.
- **`DeliveryTicket`** · protocol · One piece of background work being captured, from its start to however it ends.
- **`EvalAnalysis`** · *host contract* · model · A generated, stored analysis of one campaign — the productized "Analysis" lens.
- **`EvalCampaign`** · *host contract* · model · A curated set of eval runs under one subject×behavior — the analysis hub.
- **`EvalInsight`** · model · A durable, subject-scoped insight extracted from an analysis.
- **`EvalJobTimeout`** · exception · A job outlived the wall-clock budget its timeout context was enforcing.
- **`EvalJudgeProfile`** · model · What one judge was measured to be on one criterion, over frozen cases, by a judge campaign — stored for reuse.
- **`EvalServiceError`** · *host contract* · exception · Structured error from the eval service layer.
- **`EvalStorage`** · *host contract* · class · Storage for eval documents over one document store.
- **`ExternalRateTable`** · dataclass · Operator-declared money per provider unit, resolved once for a whole run.
- **`JobLeases`** · class · The leases one process holds on its jobs' records, and the task that renews them.
- **`MeasureFamily`** · model · One family of measures and who produces its numbers — the engine's six, or one a host declares.
- **`MetricDescriptor`** · *host contract* · model · What a single measure is, independent of any particular value of it.
- **`NotFoundError`** · *host contract* · exception · Resource not found (status 404).
- **`OutOfRunBudget`** · dataclass · The cap one out-of-run unit of work is held to, and the ledger its calls are written to.
- **`PlannedCall`** · dataclass · One call an out-of-run unit of work means to make: the prompt pair and the directive it sends.
- **`PromptDrift`** · dataclass · How one stored prompt reads against the package's shipped default.
- **`ProviderRefusedError`** · *host contract* · exception · A paid provider call made on the caller's behalf raised (status 502).
- **`Question`** · model · Something this campaign is trying to find out.
- **`Recordable`** · protocol · The two members the cassette layer uses on whatever it records: an action result or a delivered payload.
- **`ReplayedDelivery`** · dataclass · One recorded piece of background work, served in place of running it.
- **`RoleUsageLedger`** · dataclass · Accumulates one role's LLM spend across every call it makes in a cell.
- **`SeedPrompt`** · dataclass · Where one shipped default lives, so a host can register it by import.
- **`SeedSection`** · dataclass · One section of a seeded prompt template.
- **`SeedTemplate`** · dataclass · A seeded prompt template: an ordered set of sections and its identity.
- **`StorageError`** · *host contract* · exception · Storage operation failed (status 503). Its remedy is the class's own (`STORAGE_REMEDY`).
- **`SweptAxis`** · model · One axis this campaign set out to vary, and the levels it meant to compare.
- **`UnreadableRunError`** · *host contract* · exception · A read that cannot leave a run out met one this build's model refuses (status 500).
- **`ValidationFailedError`** · *host contract* · exception · Validation failed (status 422).
- **`VariantConfig`** · *host contract* · model · The A/B'd stack one cell runs, as every kind alike receives it.
- **`WorldSession`** · *host contract* · class · One cell's handle on the host's world, and the record of what happened to it.

**Types**

- **`AuthoredVerdict`** · type alias · An expected verdict as its author states it: a score on the criterion's scale (1-5 ordinal; 1 pass, 0 fail), or `"pass"` / `"fail"` on a pass/fail criterion.
  <br>`Annotated[int, Field(strict=True)] | Literal['pass', 'fail']`
- **`ExclusionCause`** · literal · Why a result is left out of a measure.
  <br>`'infra'` | `'truncated'` | `'judge_cannot_tell'`
- **`MeritAxis`** · literal · The merit axes a measure can serve.
  <br>`'quality'` | `'cost'` | `'latency'` | `'reliability'`
- **`ToolWrap`** · type alias · What the action seam is handed: maps the candidate's tools, by name, to the tools it should use — each declared recorded tool wrapped for the cassette run, every other one unchanged.
  <br>`Callable[[Mapping[str, ToolLike]], dict[str, ToolLike]]`

**Constants**

- **`CONFUSION_CELL_MEASURE`** · constant (str) · The core measure a classification's confusion-matrix cell is reported under.
  <br>`= 'confusion_cell'`
- **`FIND_TEMPLATE`** · constant (str) · How a caller finds a template's id.
  <br>`= "templates_list names this scope's templates"`
- **`IDENTITY_VERSION`** · *host contract* · constant (int) · One counter covers both predicates, so a bump on either side re-derives the other's keys — conservative in the safe direction, since re-deriving unchanged inputs reproduces the same digest.
  <br>`= 24`
- **`JUDGE_KIND`** · constant (str) · The `candidate_kind` a judge template declares.
  <br>`= 'judge'`
- **`JUDGE_KIND_CONTRACT`** · constant (KindContract) · The judge kind's contract: its overlays, and the rig it fills.
- **`JUDGE_KIND_MEASURES`** · constant (tuple) · The measures a judge trial lands on `host_measures`, for a host to declare on its measure registry.
- **`KIND_TEMPLATE`** · constant (str) · A seed whose constant is a sectioned template, registered with a template registry.
  <br>`= 'template'`
- **`KIND_TEXT`** · constant (str) · A seed whose constant is a single string, registered with a shared-prompt registry.
  <br>`= 'text'`
- **`MATCH_MEASURE`** · constant (str) · The core measure one classification's verdict is reported under — the one a classifier kind lands on `host_measures`, a bool.
  <br>`= 'match'`
- **`SEED_PROMPTS`** · constant (tuple) · Every prompt the engine ships a default for, in the order a registry seeds them.
- **`SEED_REMEDY`** · constant (str) · What a template's author does when its world seed states world state a goal check's control cannot be built over.

<a id="api-kernel-host"></a>
### `threetears.evals.kernel.host`

The host contract — what a consuming product declares, and what the engine never interprets.

**Functions**

- **`check_seed`** · function · Walk a world seed against the registry: every write it would make, checked, or the first refusal.
  <br>`check_seed(registry: WorldRegistry, namespaces: Mapping[str, Any], *, attached: Collection[str] | None = None) -> tuple[SeedWrite, ...]`
- **`check_world_conformance`** · async function · Run every obligation this registry's declarations imply, against the host's own handles.
  <br>`check_world_conformance(registry: WorldRegistry, *, expressions: Sequence[str] = ()) -> WorldConformanceReport`
- **`default_cell_timeout`** · *host contract* · function · Enforce a cell budget with `asyncio.timeout` and nothing else.
  <br>`default_cell_timeout(budget_s: float) -> AsyncIterator[float]`
- **`load_bar_registry`** · *host contract* · function · Build a host's bar registry from a declaration kept as data — refusing any field it does not know.
  <br>`load_bar_registry(declaration: Mapping[str, Any]) -> BarRegistry`

**Classes**

- **`ActsOn`** · dataclass · Name the measure or covariate an overlay knob is supposed to move — its lever's `acts_on`.
- **`ApparatusError`** · exception · A fault in the eval measuring rig, not a failure of the world under test.
- **`Bar`** · *host contract* · dataclass · The incumbent standard for one behavior on one measure.
- **`BarRegistry`** · *host contract* · class · One host's registered bars, keyed by `(behavior, measure)` and validated at construction.
- **`CompletionClients`** · *host contract* · protocol · The host's completion-client factory: one client per role, model and temperature.
- **`EvalCellTimeout`** · exception · A cell outlived the wall-clock budget its timeout context was enforcing.
- **`EvalHost`** · *host contract* · dataclass · Everything the engine reads of one consuming product, as one frozen value.
- **`HostProfile`** · *host contract* · dataclass · Everything the engine knows about one consuming product.
- **`Interval`** · dataclass · Mark a numeric field's unit, rendered beside each level (`120words`).
- **`KindContract`** · *host contract* · dataclass · What one candidate kind's runs carry, declared as Pydantic models — see the module docstring.
- **`MeasureRegistry`** · *host contract* · class · One host's declared measures, validated at construction.
- **`Ordinal`** · dataclass · Mark a `Literal` or `Enum` field as ordered: its levels rank in the order they are declared.
- **`PassThreshold`** · *host contract* · dataclass · The 1–5 level a capability criterion must reach for an attempt to pass, for one behavior.
- **`ResolvesInto`** · dataclass · Name the fixed host lever this overlay knob is WRITTEN INTO — its lever's `resolves_into`.
- **`RolePins`** · dataclass · One pinned ROLE — who filled a seat in the rig, and the inputs that identify them.
- **`SeedRefused`** · *host contract* · exception · A seed asked for a write the registry or the subject cannot make.
- **`StyleProfile`** · dataclass · One host's bounded presentation contract.
- **`Sweepable`** · *host contract* · dataclass · One score-determining input: what it is called, who owns it, how to read it.
- **`SweepableRegistry`** · *host contract* · class · The declared inputs for one host, validated at construction.
- **`Triggered`** · dataclass · State that arrives on a condition rather than at t=0.
- **`UnsupportedSchemaError`** · exception · A schema uses a construct outside `HONOURED_KEYWORDS` — a gap to extend, never a value to pass.
- **`WorldConformanceError`** · exception · An ENGINE gap: a check the kit cannot run, or a record its own vocabulary forbids.
- **`WorldConformanceReport`** · dataclass · Every verdict from one run of the kit over one registry.
- **`WorldDimension`** · *host contract* · dataclass · One state dimension a run may set, may read back, or may only witness.
- **`WorldRegistry`** · *host contract* · class · One host's world declarations, its resolution table, and the algebra over them.

**Types**

- **`CompletionRole`** · *host contract* · literal · The apparatus roles the engine asks a host for a completion client in.
  <br>`'judge'` | `'simulator'` | `'analysis'` | `'variation'` | `'proposer'`

**Constants**

- **`CANDIDATE_MODEL_LEVER`** · constant (str) · The one name the candidate model answers to, everywhere — a DECLARED COORDINATE of every observation (`ScoreRecord.model`) as well as the core lever below, which is why it is the only lever a reporting lens resolves off the observation rather than off the run.
  <br>`= 'model'`
- **`SHARED_CORE`** · *host contract* · constant (SweepableRegistry) · The registry a host extends. Nothing here names a product.

<a id="api-run"></a>
### `threetears.evals.run`

The engine's run package: launching and executing a run, judging it, metering it, and storing it.

**Functions**

- **`assert_preconditions`** · function · Assert at t=0 that the world this template presumes actually holds.
  <br>`assert_preconditions(template: EvalTemplate, test_case: EvalTestCase, seeded: Mapping[str, Any], *, world: WorldRegistry | None) -> list[PreconditionOutcome]`
- **`build_judge_service`** · function · Build the `JudgeService` for a run, with its judge attribution.
  <br>`build_judge_service(host: EvalHost, template: EvalTemplate, judge_model: str, selection: dict[str, str] | None = None, *, judged_artifact: JudgedArtifact) -> RunJudge`
- **`create_template`** · function · Create and persist a template from an authoring definition.
  <br>`create_template(host: EvalHost, definition: dict[str, Any], *, scope_id: str, require_known_tools_allowed: Callable[[Sequence[str] | None], None], refuse_undeclared_world_seed: Callable[[EvalTemplate], None], refuse_undeliverable_template: Callable[[EvalTemplate], None]) -> EvalTemplate`
- **`delete_run`** · function · Delete a run, the results it owns, and its campaign memberships.
  <br>`delete_run(storage: CurationStore, run: EvalRun, scope_id: str, *, confirm: str | None = None) -> dict[str, Any]`
- **`drive_conversation`** · async function · Run `driver`'s conversation from its first turn until it stops on a structural signal.
  <br>`drive_conversation(driver: TurnDriver, candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]], post_user_turn: Callable[[SimulatorTurn], Awaitable[None]], *, llm: SimulatorLLM, sink: CellSink, world: WorldSession | None = None) -> ConversationStopCause`
- **`estimate_judge_repeat`** · async function · Price repeating a run's judge scores against the cap it would be held to, and make no call.
  <br>`estimate_judge_repeat(host: EvalHost, run_id: str, scope_id: str, *, out_of_run_cap_usd: float | None, result_ids: Sequence[str] | None = None) -> JudgeRepeatEstimate`
- **`execute_run`** · *host contract* · async function · Execute an entire `EvalRun` end-to-end.
  <br>`execute_run(host: EvalHost, *, run: EvalRun, template: EvalTemplate, test_cases: list[EvalTestCase], judge_service: JudgeService | None, options: RunnerOptions, callbacks: RunCallbacks | None = None, budget_gate: Callable[[float | None], CapBreach | None] | None = None, on_cost: Callable[[float | None], None] | None = None, cell_sink: list[CellSummary] | None = None) -> list[CellSummary]`
- **`fold_metered_cell`** · function · Fold one cell's action-seam tally into the external role's rows.
  <br>`fold_metered_cell(external_usage: RoleUsageLedger, cell: MeteredCallTally | None) -> None`
- **`freeze_judge_cases`** · function · Freeze (output, criterion, labels) cases from stored judged results, or written by hand, into a judge template.
  <br>`freeze_judge_cases(storage: EvalStorage, *, template: EvalTemplate, run_ids: Sequence[str] = (), scope_id: str, dims: Collection[str] | None = None, result_ids: Collection[str] | None = None, case_set: str | None = None, from_labels: bool = False, authored: Sequence[AuthoredJudgeCase] = ()) -> JudgeCaseFreezeReport`
- **`get_result_trace`** · *host contract* · function · Load one result's trace payload, or `None` when it stored none.
  <br>`get_result_trace(storage: ResultStore, result: EvalResult) -> EvalTrace | None`
- **`get_run`** · function · Load an eval run by id within its scope, whole.
  <br>`get_run(storage: RunStore, run_id: str, scope_id: str) -> EvalRun`
- **`grade_goal_checks`** · function · Grade goal checks against a call ledger, an end state and what fired: the one evaluation every caller uses.
  <br>`grade_goal_checks(expressions: Sequence[str], *, ledger: CallLedger, end_state: Mapping[str, Any], fired: Firings | None, variation: Mapping[str, Any], world: WorldRegistry | None) -> list[GoalStateOutcome]`
- **`judge_repeat_work`** · async function · Admit asking a finished run's judge to score its results' scored dims again — every call priced, none made.
  <br>`judge_repeat_work(host: EvalHost, run_id: str, scope_id: str, *, out_of_run_cap_usd: float | None, result_ids: Sequence[str] | None = None) -> JobWork[EvalActionJob]`
- **`judge_temperature_work`** · async function · Admit re-judging a finished run's borderline cases at each temperature — every call priced, none made — as work.
  <br>`judge_temperature_work(host: EvalHost, run_id: str, scope_id: str, *, out_of_run_cap_usd: float | None, selection: TemperatureSelection = 'borderline', repeats: int = 5, result_ids: Sequence[str] | None = None) -> JobWork[EvalJudgeTemperatureCheck]`
- **`launch_run`** · *host contract* · async function · Assemble, stamp, bound and hand over a run — the launch tail every candidate kind shares.
  <br>`launch_run(host: LaunchHost, request: LaunchRequest, wiring: KindWiring) -> EvalRun`
- **`launchable_judge_kind`** · function · The judge kind's entry in a host's launch registry: each arm one judge over the template's frozen cases.
  <br>`launchable_judge_kind(launch_host: Callable[[], LaunchHost]) -> LaunchableKind`
- **`list_results`** · *host contract* · function · List every result for one eval run within its scope.
  <br>`list_results(storage: ResultStore, run_id: str, scope_id: str) -> list[EvalResult]`
- **`list_runs`** · function · List eval runs in a scope, optionally filtered by status.
  <br>`list_runs(host: EvalHost, scope_id: str, *, status: str | None = None, include_archived: bool = False, label: str | None = None) -> list[EvalRun]`
- **`metered_cell_tally`** · function · One cell's slice of the run's metered-call tally, or `None` when nothing counted.
  <br>`metered_cell_tally(ledger: MeteredCallLedger | None, baseline: MeteredCallTally | None = None) -> MeteredCallTally | None`
- **`plan_judge`** · function · The judges one arm of `template` will be scored by, resolved as `build_judge_service` resolves them — building nothing.
  <br>`plan_judge(host: EvalHost, template: EvalTemplate, judge_model: str, selection: Mapping[str, str] | None = None, *, judged_artifact: JudgedArtifact) -> PlannedJudge`
- **`precondition_failure_text`** · function · One line naming what was presumed and what the world held instead.
  <br>`precondition_failure_text(outcomes: Sequence[PreconditionOutcome]) -> str`
- **`price_subject_call_floor`** · *host contract* · async function · A floor under a subject arm's largest judge or simulator call, for its `LaunchableKind.price_largest_call`.
  <br>`price_subject_call_floor(host: EvalHost, request: LaunchRequest, *, judged_artifact: JudgedArtifact) -> CallCeiling | None`
- **`rate_result`** · function · Record one rater's score for one judged dimension of one result — a person's, or an agent's.
  <br>`rate_result(storage: RatingStore, *, result_id: str, scope_id: str, rubric_dim: str, rater: str, rater_kind: RaterKind, score: int, reason: str) -> CalibrationRating`
- **`refuse_ignored_template_fields`** · function · Refuse a template stating a field its kind never acts on (`LaunchableKind.ignored_template_fields`).
  <br>`refuse_ignored_template_fields(template: EvalTemplate, launchable: LaunchableKind) -> None`
- **`require_candidate_model`** · function · The model `request`'s arm runs on — the one it named, or the kind's role default — or the refusal.
  <br>`require_candidate_model(request: LaunchRequest, default: str | None) -> str`
- **`resolve_judge_pin`** · function · The run-level judge pin an arm is scored under: the launch's, or the role default stepped off a candidate.
  <br>`resolve_judge_pin(request: LaunchRequest, role_default: str, *, candidate_model: str) -> str`
- **`run_blocking`** · *host contract* · async function · Run `fn(*args, **kwargs)` on `executor` and await its result.
  <br>`run_blocking(executor: Executor | None, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T`
- **`run_job_work`** · async function · Write the work's record, leased, and run the work to its end in the caller's own frame.
  <br>`run_job_work(work: JobWork[R], *, store: JobRecordStore, executor: Executor | None, leases: JobLeases) -> R`
- **`run_judge_llm`** · async function · Run an LLM judge call against a set of criteria and parse the response.
  <br>`run_judge_llm(client: Any, system_prompt: str, user_prompt: str, criteria_dicts: list[dict[str, Any]], label: str, case_id: str, *, cannot_tell_offered: bool = False, failure_describer: ProviderFailureDescriber | None = None) -> dict[str, Any] | None`
- **`second_judge_work`** · async function · Admit a second judge's pass over a finished run — every call priced and admitted, none made — as work to run.
  <br>`second_judge_work(host: EvalHost, run_id: str, scope_id: str, *, judge: SecondJudge, out_of_run_cap_usd: float | None, asked_by: SecondJudgeAskedBy = 'judge_second', sample_fraction: float = 1.0, seed: int = 0, result_ids: Sequence[str] | None = None) -> JobWork[EvalSecondJudgePass]`
- **`seed_eval_definitions`** · function · Create any corpus definition whose natural key is absent from the corpus's scope.
  <br>`seed_eval_definitions(host: EvalHost, corpus: SeedCorpus, *, require_known_tools_allowed: Callable[[Sequence[str] | None], None], refuse_undeclared_world_seed: Callable[[EvalTemplate], None], refuse_undeliverable_template: Callable[[EvalTemplate], None]) -> SeedOutcome`
- **`start_run`** · *host contract* · async function · Refuse what no kind can run, admit the launch, and dispatch each arm to its kind's launcher.
  <br>`start_run(host: LaunchHost, *, template_id: str, subject_id: str, models: list[str], k_runs: int = 3, n_variations: int = 0, variation_model: str | None = None, judge_model: str | None = None, judge_config_ids: dict[str, str] | None = None, simulator_model: str | None = None, cassette_mode: str | None = 'off', cassette_corpus_id: str | None = None, overlays: Mapping[str, Any] | None = None, apparatus_settings: Mapping[str, Any] | None = None, max_cost_usd: float | None = None, max_metered_calls: int | None = None, scope_id: str, launch_group: LaunchGroup | None = None, admission: AdmissionTicket | None = None, cell_timeout_s: float | None = None, case_set: CaseSetRef | None = None, margins: Mapping[str, float] | None = None, measure_latency: bool = False) -> list[EvalRun]`
- **`sweep_abandoned_runs`** · function · Write lost every run left not ended by a process that has gone, in the scopes the host names.
  <br>`sweep_abandoned_runs(storage: ReclaimStore, scopes: Iterable[str]) -> AbandonedRunSweepReport`

**Classes**

- **`AbandonedRunSweepReport`** · model · Outcome of one reclaim of runs left non-terminal by a process that has gone (`sweep_abandoned_runs`).
- **`ArmPlan`** · dataclass · What one arm of a launch will run, as its kind says before its launcher runs.
- **`ArmPrice`** · dataclass · What the host's pricer predicts one arm will cost, and how it knows.
- **`ArmQuote`** · dataclass · One arm of a launch, as the engine asks the host's pricer to price it — stored-case arms and generating ones alike.
- **`CallCeiling`** · dataclass · The largest reservation one call of an arm's cells will hold against its run's cap, priced before launch.
- **`CandidateTurn`** · dataclass · One candidate-side response captured by the kind and fed back into the simulator.
- **`CellContext`** · dataclass · What the runner knows about one cell when it asks a kind's factory for the cell's kind.
- **`CellExecutor`** · protocol · Runs a run's cells, at most `width` at once, starting them in the order given.
- **`FidelityContract`** · dataclass · One shared construction path and the callers required to reach it.
- **`GoalCheckUnevaluable`** · exception · A goal check raised while being evaluated — a fault of the rig, not a verdict on the candidate.
- **`JudgeRepeatReport`** · model · What a repeat did: which results it recorded a repeat on, what it spent, and what it left out.
- **`JudgeService`** · class · Stateless single-dim judge. See module docstring for the contract.
- **`KindFactory`** · *host contract* · protocol · Builds the candidate kind for one cell.
- **`KindWiring`** · *host contract* · dataclass · What one kind's launcher resolved for its arm — everything `launch_run` cannot know itself.
- **`LaunchableKind`** · *host contract* · dataclass · One entry of the launch registry a host hands `start_run`: how a kind launches, and what it refuses.
- **`LaunchHost`** · *host contract* · dataclass · An `EvalHost`, and what starting its runs needs.
- **`LaunchRequest`** · *host contract* · dataclass · One arm's launch as `start_run` resolved it, handed to the kind's own launcher.
- **`LaunchSettings`** · *host contract* · model · The host's launch settings, as one snapshot of values.
- **`MeteredCallLedger`** · class · In-memory per-run ceiling on metered third-party calls.
- **`MeteredCallTally`** · dataclass · What a run (or one cell's slice of it) metered, kept per provider.
- **`RunnerOptions`** · *host contract* · dataclass · The run's own knobs that don't fit on the run document — per-run values, never host wiring.
- **`SeedCorpus`** · dataclass · A host's authored definitions for one scope, already constructed through their production models.
- **`SeedJudgeCase`** · *host contract* · model · One authored judge case a seed corpus writes: `judge_case_author`'s arguments, its templates named by name.
- **`SeedOutcome`** · dataclass · What one seeding pass did, per doc type.
- **`SimulatorTurn`** · dataclass · One user-side utterance produced by the simulator.
- **`TurnDriver`** · dataclass · Multi-actor round scheduler and utterance generator.
- **`UnknownSubject`** · *host contract* · dataclass · A host's answer that it holds no subject of the id a launch named — and how its caller finds one it does.

**Types**

- **`KindLauncher`** · *host contract* · type alias · One kind's launcher: builds that kind's collaborators for one arm and hands `launch_run` its run.
  <br>`Callable[[LaunchRequest], Awaitable[EvalRun]]`
- **`SubjectResolver`** · *host contract* · type alias · How a kind answers whether its host holds the subject a launch of it names, before anything is priced: `(subject_id, scope_id) -> None` when it does, `UnknownSubject` when it does not.
  <br>`Callable[[str, str], UnknownSubject | None]`
- **`TemplateField`** · literal · The fields of a template a kind may never act on: what it states about tools, the cases it generates, a conversation, the world (its seed, what it presumes, the checks graded against it) and the rubric a judge scores.
  <br>`'tools_required'` | `'tools_allowed'` | `'variation_axes'` | `'conversation'` | `'world_seed'` | `'preconditions'` | `'goal_state_checks'` | `'goal_check_controls'` | `'rubric'`

**Constants**

- **`JUDGE_CALL_ATTEMPTS`** · constant (int) · Calls one judge dimension can make: the first, plus its parse retries.
  <br>`= 2`
- **`JUDGE_MAX_TOKENS`** · constant (int) · The eval judge's output cap: the reply schema's answer budget plus the declared reasoning headroom, DERIVED rather than chosen, so it always leaves room to think and then answer.
  <br>`= 10240`
- **`JUDGE_REQUEST_SETTINGS`** · *host contract* · constant (ClientRequestSettings) · The judge role's request settings as ONE value: what the host's client builder applies to every eval judge client, and what a run launched with a judge records as `judge_asked_settings`.
- **`SCHEDULER_CALL_ATTEMPTS`** · constant (int) · How many calls one `llm_decided` scheduling decision may make: the first, and one repair that names what was wrong with it.
  <br>`= 2`
- **`SIMULATOR_ANSWER_BUDGET_TOKENS`** · constant (int) · Output room kept for the simulated user's visible reply: the strict-schema JSON object holding one utterance or one scheduling pick.
  <br>`= 4096`
- **`SIMULATOR_MAX_TOKENS`** · constant (int) · The simulator's output cap: the reasoning allowance plus the answer budget, derived rather than chosen. It is the only bound in tokens a simulator call has.
  <br>`= 5120`
- **`SIMULATOR_REQUEST_SETTINGS`** · *host contract* · constant (ClientRequestSettings) · The simulator role's request settings as ONE value: what the host's client builder applies to the simulated user's client, and what a run launched with a simulated user records as `simulator_asked_settings`.

<a id="api-analysis"></a>
### `threetears.evals.analysis`

The engine's analysis package: campaigns, context bundles, generated analyses and the read lenses.

**Functions**

- **`analysis_gen_request_settings_for`** · *host contract* · function · The generator role's request settings as ONE value, from the host's answer budget, effort and headroom.
  <br>`analysis_gen_request_settings_for(*, answer_budget_tokens: int, reasoning_effort: KnownReasoningEffort = 'medium', reasoning_headroom_tokens: int = 8192) -> ClientRequestSettings`
- **`assemble_context_bundle`** · function · Assemble the closed context bundle for a campaign's runs.
  <br>`assemble_context_bundle(campaign: EvalCampaign, *, storage: CampaignReadStore, insights_as_of: str | None = None, profile: HostProfile) -> AnalysisContextBundle`
- **`campaign_report`** · function · The campaign's report — THE answer to "what is this campaign's report", for every caller.
  <br>`campaign_report(host: EvalHost, campaign_id: str, scope_id: str) -> Report`
- **`create_campaign`** · function · Create and persist a campaign from an authoring definition.
  <br>`create_campaign(storage: CampaignStore, definition: dict[str, Any], *, scope_id: str, created_by: str, profile: HostProfile, control_from_run_id: str | None = None, exploratory: bool = False) -> EvalCampaign`
- **`dollars_text`** · function · Spend as a person reads it: dollars and cents from ten cents up, three significant figures below.
  <br>`dollars_text(amount: float | None) -> str`
- **`format_number`** · *host contract* · function · Spell a number for a reader, the same way on every analysis surface.
  <br>`format_number(value: float | None, *, prefix: str = '') -> str`
- **`format_signed`** · *host contract* · function · Spell a change, carrying its sign — the direction is half of what a delta says.
  <br>`format_signed(value: float | None) -> str`
- **`frontier`** · function · Rank each subject's variants on quality x cost x latency, cheapest above bar.
  <br>`frontier(storage: LensStore, scope_id: str, *, list_runs: RunLister, bar: float | str | None = None, subject_id: str | None = None, status: str | None = 'completed', profile: HostProfile | None = None, control_variant_key: str | None = None) -> dict[str, Any]`
- **`gate_verdicts`** · function · Gate a report's typed verdicts: fail on the outcomes named, and never pass one the evidence left undecided.
  <br>`gate_verdicts(verdicts: Sequence[Verdict], *, fail_on: Iterable[str] = ('regressed', 'breached', 'undecided-guardrail'), readings: Iterable[str] | None = None) -> GateResult`
- **`generate_analysis`** · *host contract* · async function · Generate one campaign's analysis from its context bundle, in one LLM call or two.
  <br>`generate_analysis(bundle: AnalysisContextBundle, *, prompt: str, model: str, client: CompletionGenerator, prompt_id: str, bundle_assembled_at: str, prompt_version: str | None = None, tally: GenerationTally | None = None, admit: CallAdmission | None = None, profile: HostProfile, measuring_writers: bool = False) -> tuple[EvalAnalysis, list[EvalInsight]]`
- **`generation_ceiling_s`** · function · The wall-clock ceiling of one `generate_analysis`, derived from the ceilings it wraps.
  <br>`generation_ceiling_s(*, request_s: RequestCeiling, generator_max_tokens: int) -> float`
- **`inspect_campaign_bundle`** · *host contract* · function · Assemble a campaign's context bundle and return it, without generating.
  <br>`inspect_campaign_bundle(host: EvalHost, campaign_id: str, scope_id: str) -> BundleInspection`
- **`inter_judge_agreement`** · function · Pair each second judge's score with the first score it answers, and read agreement the way calibration does.
  <br>`inter_judge_agreement(results: Iterable[EvalResult], *, pass_id: str | None = None) -> InterJudgeAgreement`
- **`judge_agreement`** · function · Pair each rating with the judge's score on its dimension, on its result or the same output, and read agreement.
  <br>`judge_agreement(ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]) -> JudgeAgreement`
- **`judge_drift`** · function · Read how far each dimension's scores moved from the run's judge to a second judge on the same evidence.
  <br>`judge_drift(results: Iterable[EvalResult], *, pass_id: str | None = None) -> JudgeDrift`
- **`judge_evidence_tiers`** · function · Decide the evidence tier of every judge's readings on every dimension, from the two agreements.
  <br>`judge_evidence_tiers(agreement: JudgeAgreement, self_agreement: JudgeSelfAgreement, judged: Iterable[JudgeKey]) -> list[JudgeEvidenceTier]`
- **`judge_key`** · function · The judge behind `result`'s score on `dim`, or None when it holds no score there.
  <br>`judge_key(result: EvalResult, dim: str) -> JudgeKey | None`
- **`judge_kind_readings`** · function · Pool a judge campaign's trials into agreement with the labels, self-agreement and parse validity, per judge and criterion.
  <br>`judge_kind_readings(results: Iterable[EvalResult]) -> JudgeKindReadings`
- **`judge_phase_ceiling_s`** · function · The wall-clock ceiling of one cell's judge phase, derived from the ceilings of its requests.
  <br>`judge_phase_ceiling_s(*, judge_dims: int, judge_concurrency: int, judge_call_attempts: int, judge_max_tokens: int, request_s: RequestCeiling) -> float`
- **`judge_self_agreement`** · function · Pair each repeated score with the first score it repeated, and read agreement the way calibration does.
  <br>`judge_self_agreement(results: Iterable[EvalResult]) -> JudgeSelfAgreement`
- **`read_judge_temperatures`** · function · Read a judge temperature comparison's answers into per-dimension variance and self-agreement, side by side.
  <br>`read_judge_temperatures(answers: JudgeTemperatureAnswers) -> JudgeTemperatureComparison`
- **`refuse_undeliverable_reporter_template`** · function · Refuse a reporter template that asks for apparatus a memo writer cannot honour, or gives its judge nothing to score.
  <br>`refuse_undeliverable_reporter_template(template: EvalTemplate) -> None`
- **`report_html`** · *host contract* · function · Render a report as one standalone HTML page that needs no script.
  <br>`report_html(report: Report) -> str`
- **`report_markdown`** · *host contract* · function · Render a report as Markdown.
  <br>`report_markdown(report: Report) -> str`
- **`reporter_cell_timeout_s`** · function · The wall-clock ceiling of one reporter cell, derived from the ceilings it wraps.
  <br>`reporter_cell_timeout_s(*, judge_dims: int, judge_concurrency: int, generator_max_tokens: int, judge_call_attempts: int, judge_max_tokens: int, request_s: RequestCeiling) -> float`
- **`reporter_kind_contract`** · function · The reporter's contract, also seating apparatus dimensions of the host's own that its runs record.
  <br>`reporter_kind_contract(host_seats: Iterable[str] = ()) -> KindContract`
- **`tier_sentence`** · function · One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.
  <br>`tier_sentence(tier: JudgeEvidenceTier) -> str`
- **`variant_key_of_run`** · function · Which variant a run's observations carried — the authoring side of the control.
  <br>`variant_key_of_run(results: Sequence[EvalResult]) -> str | None`

**Classes**

- **`AnalysisContextBundle`** · *host contract* · model · The closed context bundle a generation prompt runs over.
- **`AsRecordedReporterKind`** · class · The memo each case's campaign actually got, replayed — the calibration candidate.
- **`CampaignReadStore`** · protocol · The nine reads assembling a campaign's context bundle needs.
- **`CaseResult`** · model · One case's answer on one repeat, every grade it got, and why it failed or was excluded.
- **`ChartBlock`** · *host contract* · model · One of the two charts a report draws, with every value it draws, so it prints as a table too.
- **`EvalSummary`** · *host contract* · model · One run, summarised.
- **`GateResult`** · dataclass · What a gate read and what it came to.
- **`LabelStatistics`** · model · One label's counts in a confusion matrix, and the precision, recall and F1 they give.
- **`Report`** · *host contract* · model · A campaign's report: one page, the bundle it was computed from, and the typed verdicts it states.
- **`ReporterKind`** · class · A (prompt preset, generator model) candidate: generates a memo over each case's bundle.
- **`ReporterOverlays`** · model · The one knob a reporter launch may turn: which preset of the analysis generator's prompt writes the memos.
- **`RequestSettingsReading`** · *host contract* · model · What one role asked for and what its calls were actually sent with, across the results read.
- **`SentSettingsShare`** · *host contract* · model · One value a role's calls were sent with, and how many of the role's units carried it.
- **`Verdict`** · model · One verdict the report states, typed: what a program reads instead of the words printed from it.

**Types**

- **`GateOutcome`** · literal · What a gate came to: `failed` (a verdict named in `fail_on`), `undecided` (none failed, and some verdict was not decided, or there was none), `passed` (every verdict decided, none failing).
  <br>`'passed'` | `'failed'` | `'undecided'`
- **`GateToken`** · literal · An outcome a gate can fail on; see the module docstring.
  <br>`'regressed'` | `'not-separated'` | `'untested'` | `'breached'` | `'undecided-guardrail'` | `'missed'` | `'undecided-bar'`
- **`SentSettingsState`** · *host contract* · literal · What a role's calls were sent with, read across a run's (or a campaign's) calls, for display only — never stored: `as_asked` (every call reported, one value, the one asked for), `sent_otherwise` (every call reported, one value, not the one asked for), `mixed` (every call reported, more than one value), `unverified` (some call's client did not report what it sent).
  <br>`'as_asked'` | `'sent_otherwise'` | `'mixed'` | `'unverified'`

**Constants**

- **`CELL_MODEL_VERSION`** · *host contract* · constant (int) · The cell model this module implements, stamped onto every analysis generated under it (`GenerationProvenance.cell_model_version`).
  <br>`= 13`
- **`EVAL_ANALYSIS_GEN_DEFAULT`** · *host contract* · constant (str) · The analysis generator's default system prompt.
- **`GATE_TOKENS`** · constant (tuple) · Every token, in the order the module docstring lists them.
- **`REPORTER_KIND`** · constant (str) · The `candidate_kind` a reporter template declares.
  <br>`= 'analysis_reporter'`

<a id="api-gen"></a>
### `threetears.evals.gen`

The engine's generation package: the prompts and expanders that author test material.

**Functions**

- **`generate_variations`** · async function · Generate up to `n_variations` test cases for a template.
  <br>`generate_variations(template: EvalTemplate, n_variations: int, *, storage: EvalTestCaseStore, scope_id: str, blocking_executor: Executor | None, generation: AdmittedGeneration | None = None, rng: random.Random | None = None, preview: bool = False) -> GeneratedVariations`
- **`plan_draft`** · function · The one call a rubric proposal sends, on either axis, built before anything is priced or made.
  <br>`plan_draft(*, axis: RubricAxis, subject_feed: str, catalog_feed: str, system_prompt: str | None = None, reset: str | None = None) -> PlannedCall`
- **`propose_draft`** · async function · Make an admitted rubric proposal on `axis` and read its draft.
  <br>`propose_draft(client: BoundCompletionClient, admitted: AdmittedCall, *, budget: OutOfRunBudget, axis: RubricAxis, subject_id: str) -> ProposedDraft`

**Constants**

- **`PROPOSER_MAX_TOKENS`** · constant (int) · Output-token ceiling for the capability/boundary proposer.
  <br>`= 16384`

<a id="api-storage"></a>
### `threetears.evals.storage`

Storage adapters the engine ships: implementations of the one port a host stores through.

**Functions**

- **`bridge_async_store`** · *host contract* · function · Present `store` as the engine's synchronous `DocumentStore`.
  <br>`bridge_async_store(store: AsyncDocumentStore, loop: asyncio.AbstractEventLoop) -> AsyncStoreBridge`

**Classes**

- **`CoreReset`** · dataclass · What `RoutedDocumentStore.reset_core` did: the core and regenerable documents deleted, and what moved.
- **`DocTypeRoute`** · *host contract* · dataclass · Which store each document type is kept in, decided by one of its facts.
- **`InMemoryDocumentStore`** · *host contract* · class · A `DocumentStore` over one dict.
- **`MisplacedDocument`** · exception · One id is held by two stores, or by a store its type is not placed in; the router never picks one.
- **`RouteChanged`** · exception · The router was started under placements its stores were not written under; `relocate` first.
- **`RoutedDocumentStore`** · *host contract* · class · One `DocumentStore` over several, each `doc_type` kept where its facts say.
- **`SqliteDocumentStore`** · *host contract* · class · A `DocumentStore` over one SQLite file.

**Constants**

- **`ROUTING_RECORD_DOC_TYPE`** · constant (str) · The `doc_type` of the placement record the router writes into every store, inside the routing scope its host names.
  <br>`= 'eval_routing_record'`

<a id="api-testing"></a>
### `threetears.evals.testing`

What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

**Functions**

- **`check_completion_conformance`** · *host contract* · function · Fail, naming the attribute, when a host's completion does not carry what eval reads off it.
  <br>`check_completion_conformance(completion: object, *, require_sent_settings: bool = False) -> None`
- **`check_refusal_negotiation`** · *host contract* · async function · Fail, naming each mistake, unless the client answers a router's refusals in order and reports what was served.
  <br>`check_refusal_negotiation(build_client: Callable[[RouterTransport], CompletionClient]) -> None`
- **`jobs_conformance_cases`** · *host contract* · function · Every case the catalogue's actions are held to: each action's call, and each spend action's happy path.
  <br>`jobs_conformance_cases(catalogue: ActionCatalogue, *, job_timeout_s: float = 60.0, call_return_timeout_s: float = 10.0) -> tuple[JobsConformanceCase, ...]`

**Classes**

- **`CompletionGuard`** · *host contract* · dataclass · The kit's watch on a host's completion clients: what was asked outside a job, and how often inside one.
- **`JobsConformanceHost`** · *host contract* · dataclass · What a host hands the kit for one case: its operations host, who calls, and its sample launch.
- **`NegotiationConformanceFailure`** · *host contract* · exception · A host's client mishandled the router's refusals: each mistake it made, named.
- **`RecordedCompletions`** · *host contract* · class · A completion client that records the calls an inner client answers, or replays a recording of them.
- **`RecordedPrompt`** · *host contract* · class · One call a recording is expected to serve, as `RecordedCompletions.check_covers` is handed it.
- **`RecordingRefused`** · *host contract* · exception · A recording cannot serve what was asked of it: a miss, a missing or unreadable file, a newer format.
- **`StoreConformanceCase`** · *host contract* · dataclass · One rule of the port, as a check over a store.

**Types**

- **`RouterTransport`** · *host contract* · type alias · A scripted router: the JSON body a client would POST in, the status and JSON body the router answers with out.
  <br>`Callable[[Mapping[str, object]], Awaitable[tuple[int, Mapping[str, object]]]]`

**Constants**

- **`STORE_CONFORMANCE_CASES`** · *host contract* · constant (tuple) · Every case, in the order the port states its rules: scoping, the strip on read, projection, querying, optimistic concurrency (with the re-read that recovers a lost race), merge, delete — and last, that every `doc_type` the engine writes is one the store stores.

<a id="api-quick"></a>
### `threetears.evals.quick`

The host bridge: an eval's host, the command line, and a measure declared on the function computing it.

**Functions**

- **`as_host`** · *host contract* · function · A host for these evals: the vocabulary their graders, levers and world need, over one store.
  <br>`as_host(evals: Sequence[Eval], *, store: DocumentStore | str | os.PathLike[str] | None, clients: CompletionClients | None = None) -> EvalHost`
- **`build_parser`** · function · The command line's parser.
  <br>`build_parser(prog: str = 'python -m threetears.evals', *, takes_host: bool = True, commands: Sequence[HostCommand] = ()) -> argparse.ArgumentParser`
- **`measure`** · *host contract* · function · Declare the measure a function computes, on the function: `@measure(higher_is_better=True, ...)`.
  <br>`measure(*, name: str | None = None, reader_name: str | None = None, description: str | None = None, family: str = 'mechanical', transferability_class: str = 'mechanical', attribution_scope: str = 'end_to_end', data_type: str = 'numeric', **declared: Any) -> Callable[[Callable[..., Any]], Measure]`
- **`run_cli`** · function · Parse `argv` and carry out the command, printing to stdout and refusals to stderr.
  <br>`run_cli(argv: Sequence[str] | None = None, *, host_factory: HostFactory | None = None, prog: str = 'python -m threetears.evals', commands: Sequence[HostCommand] = ()) -> int`

**Classes**

- **`Grader`** · dataclass · A function or a field with what its reading declares: a margin, a guardrail, or a range.
- **`HostCommand`** · dataclass · A subcommand a host adds beside the engine's own, mounted by `run_cli` under the same program.
- **`JudgeCalibration`** · dataclass · How a judge's scores on one dimension agree with a person's labels, and the evidence tier that bears.
- **`Measure`** · class · A function that computes a measure, carrying the measure's declaration.
- **`WorldOutcome`** · model · What a world runner's graders see: the agent's answer, the state it left, and the calls it made.
- **`WorldRunner`** · dataclass · A runner that hosts an agent on a world: `World(...)(agent)`.

**Types**

- **`HostFactory`** · type alias · What names the host the commands work in: called once, with no arguments, per invocation.
  <br>`Callable[[], EvalHost | LaunchHost]`

**Constants**

- **`DEFAULT_PROG`** · constant (str) · The program name the command line prints when it is run as `python -m threetears.evals`.
  <br>`= 'python -m threetears.evals'`
- **`ENGINE_COMMANDS`** · constant (tuple) · The commands the engine itself carries; a host command may take none of these names.
- **`EXIT_FAILED`** · constant (int) · Exit code: the command failed on an error nothing anticipated — a host factory, a launcher or a handler raising, or the engine's own fault.
  <br>`= 3`
- **`EXIT_GATE_FAILED`** · constant (int) · Exit code: `gate` read a verdict its `--fail-on` names.
  <br>`= 4`
- **`EXIT_OK`** · constant (int) · Exit code: every run completed, or the command read what it was asked for.
  <br>`= 0`
- **`EXIT_REFUSED`** · constant (int) · Exit code: the command was refused before it could do anything.
  <br>`= 2`
- **`EXIT_RUN_DID_NOT_COMPLETE`** · constant (int) · Exit code: a launched run did not complete.
  <br>`= 1`

<a id="api-ops"></a>
### `threetears.evals.ops`

Typed operations over a host: what every surface — a CLI, an MCP tool, a REST route — calls.

**Functions**

- **`admit_reporter_launch`** · function · Every refusal a reporter arm meets before it is priced, and what it resolved to — for its plan and its launcher alike.
  <br>`admit_reporter_launch(request: LaunchRequest, *, storage: EvalStorage, prompts: ReporterPrompts, resolve_model: ReporterModelResolver) -> ReporterAdmitted`
- **`campaign_create`** · function · Create a campaign over runs already in the scope, declared as it is created when the definition says so.
  <br>`campaign_create(host: EvalHost, definition: CampaignDefinition, scope_id: str, *, created_by: str) -> CampaignLine`
- **`campaigns_list`** · function · The scope's campaigns.
  <br>`campaigns_list(host: EvalHost, scope_id: str, *, archived: bool | None = None) -> CampaignListing`
- **`history_launch_pricer`** · function · The engine's launch pricer: an arm bounded from the scope's usage history of runs launched as it will be.
  <br>`history_launch_pricer(host: EvalHost) -> LaunchPricer`
- **`job_cancel`** · async function · Ask a running job to stop — from any process — then report where it stands.
  <br>`job_cancel(host: OpsHost, job_id: str, scope_id: str, *, reason: str | None = None) -> JobStatus`
- **`job_poll`** · async function · Where a job stands, read from the record its work writes — the same answer in every process.
  <br>`job_poll(host: OpsHost, job_id: str, scope_id: str) -> JobStatus`
- **`judge_profiles_record`** · function · Record the profile of every judge and criterion a judge campaign's runs measured.
  <br>`judge_profiles_record(host: EvalHost, record: JudgeProfilesRecord, scope_id: str) -> JudgeProfileRecording`
- **`plan_reporter_arm`** · function · A reporter arm's plan, for its `LaunchableKind.plan_arm`: every refusal first, then what it will run.
  <br>`plan_reporter_arm(host: EvalHost, request: LaunchRequest, *, prompts: ReporterPrompts, resolve_model: ReporterModelResolver) -> ArmPlan`
- **`price_largest_reporter_call`** · async function · A reporter arm's largest generator call, priced before launch, for its `LaunchableKind.price_largest_call`.
  <br>`price_largest_reporter_call(host: EvalHost, request: LaunchRequest, *, prompts: ReporterPrompts, resolve_model: ReporterModelResolver) -> CallCeiling | None`
- **`report_read`** · function · The campaign's report, serialized in one form — the same report the command line's `report` prints.
  <br>`report_read(host: EvalHost, campaign_id: str, scope_id: str, *, format: ReportFormat) -> ReportDocument`
- **`run_get`** · function · One run, summarised from its stored results.
  <br>`run_get(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary`
- **`run_label`** · function · Set a run's labels, replacing any it has; an empty list clears them (#648).
  <br>`run_label(host: EvalHost, run_id: str, scope_id: str, *, labels: list[str]) -> RunLine`
- **`run_launch`** · async function · Admit a template's launch — one run per model — and start its arms and runs as one job.
  <br>`run_launch(host: OpsHost, arguments: LaunchArguments, scope_id: str) -> JobsStarted`
- **`runs_list`** · function · The scope's runs, and how many it matched that this build cannot read.
  <br>`runs_list(host: EvalHost, scope_id: str, *, status: str | None = None, include_archived: bool = False, label: str | None = None) -> RunListing`
- **`scope_history`** · *host contract* · function · One measure over time for each contestant in the scope, with its regressions flagged.
  <br>`scope_history(host: EvalHost, scope_id: str, *, metric: str | None = None, min_absolute_change: float = 0.0, min_relative_change: float = 0.0, subject_id: str | None = None, status: str | None = 'completed') -> HistoryResult`
- **`start_action_job`** · async function · Start a host action's work as a job whose record (`EvalActionJob`) holds its result.
  <br>`start_action_job(host: OpsHost, scope_id: str, *, action: str, run: Callable[[], Awaitable[BaseModel]], budget_s: float, label: str = '', discard: Callable[[], Awaitable[None]] | None = None) -> JobsStarted`
- **`start_paid_call`** · async function · Start a host action's one paid model call: admitted against its cap in the call, made in a job.
  <br>`start_paid_call(host: OpsHost, scope_id: str, *, action: str, role: CompletionRole, model: str | None, purpose: KnownOutOfRunPurpose, budget: OutOfRunBudget, call: PlannedCall, make: Callable[[BoundCompletionClient, AdmittedCall], Awaitable[R]], label: str, budget_s: float) -> JobsStarted`
- **`templates_list`** · function · The scope's templates.
  <br>`templates_list(host: EvalHost, scope_id: str, *, archived: bool = False) -> TemplateListing`

**Classes**

- **`AnalysisGeneration`** · dataclass · How this host generates an analysis as a background job.
- **`CampaignDefinition`** · model · What creating a campaign names: what it is called, its subject and behaviour, its runs, and what it set out to learn.
- **`CampaignLine`** · model · One campaign, as a listing shows it.
- **`CampaignListing`** · model · A scope's campaigns, newest first.
- **`JobsStarted`** · model · What starting long work returns: one handle per job, in the order the work was asked for.
- **`JobStatus`** · model · Where one job stands, read from the record its work writes.
- **`JudgeProfilesRecord`** · model · What recording judge profiles names: the judge campaign's runs whose trials make up the measurement.
- **`LaunchArguments`** · model · What a launch names: the template, the subject, one arm per model, and the run's own limits.
- **`OpsHost`** · dataclass · The host the operations, and the actions over them, work in.
- **`RegressionWatch`** · dataclass · Checks a completed run's measures against its contestant's history, and delivers each regression.
- **`ReporterAdmitted`** · dataclass · What `admit_reporter_launch` resolved of a reporter arm it can honour.
- **`RunLine`** · model · One run, as a listing shows it.
- **`RunListing`** · model · A scope's runs, newest first as the store lists them.
- **`TemplateListing`** · model · A scope's templates.

**Types**

- **`ReporterModelResolver`** · type alias · The host's model resolution, as a reporter launch asks it: `(role, named)` returns the provider model the name resolves to, or the role's default when `named` is `None`.
  <br>`Callable[[ReporterModelRole, str | None], str]`
- **`ReporterModelRole`** · literal · Which model a reporter launch asks the host to resolve: the generator writing the memos, or the judge scoring them.
  <br>`'generator'` | `'judge'`

**Constants**

- **`REPORTER_UNHONOURED_LAUNCH_ARGUMENTS`** · constant (frozenset) · The launch arguments a reporter arm cannot honour, for its `LaunchableKind`: it drives no simulated user, calls no tool a cassette could record, and generates no case — each is frozen from a campaign.

**Also exported here**

`EvalSummary` ([`threetears.evals.analysis`](#api-analysis))

<a id="api-actions"></a>
### `threetears.evals.actions`

The action catalogue: every eval action, declared once, for any transport to mount.

**Functions**

- **`eval_catalogue`** · function · The engine's catalogue, with the host's own actions after the engine's.
  <br>`eval_catalogue(host_actions: tuple[Action, ...] = ()) -> ActionCatalogue`
- **`standard_tools`** · function · The two tools a host mounts by default: `<prefix>` and `<prefix>_admin`.
  <br>`standard_tools(prefix: str = 'evals') -> tuple[ToolSpec, ToolSpec]`

**Classes**

- **`Action`** · dataclass · One action: what it is called, what it takes and returns, who may call it, and how it reads back.
- **`ActionCatalogue`** · class · Every action a host offers: the engine's, and any the host contributes.
- **`ActionOutcome`** · dataclass · What one call came to: the text an agent reads, the typed result as data, and why it was refused.
- **`Caller`** · dataclass · Who is calling, and the scope they act in — resolved by the host for every call.
- **`ConformanceStage`** · dataclass · What a `HappyPath` is handed: the host the kit built, the caller, and the host's sample launch.

<a id="api-transports-fastmcp"></a>
### `threetears.evals.transports.fastmcp`

*Fails to import without the `[fastmcp]` extra (`pip install "3tears-evals[fastmcp]"`), which brings in `fastmcp`; see `threetears.evals.ROOT_EXTRAS`. Every root not so marked imports on the base install.*

The FastMCP transport: mounts the action catalogue as tools on a FastMCP server.

**Functions**

- **`mount_fastmcp`** · *host contract* · function · Add the catalogue's tools to a FastMCP server; or, handed a host alone, build the server and return it.
  <br>`mount_fastmcp(server: FastMCP | EvalHost | OpsHost, catalogue: ActionCatalogue | None = None, *, host: EvalHost | OpsHost | None = None, host_resolver: HostResolver | None = None, caller: CallerResolver | None = None, scope: str | None = None, identity: str = 'mcp', around: CallHook | None = None, annotate: CallAnnotator | None = None, tools: Sequence[ToolSpec] | None = None, call_timeout_s: CallTimeout = 60.0) -> FastMCP | tuple[str, ...]`

**Classes**

- **`CallTimedOut`** · exception · A call outlived the call timeout and was stopped before its action finished.
- **`HostResolution`** · dataclass · What a host resolver may answer in place of the bare host: the host, and state of the host's own for the hooks.
- **`ToolCall`** · dataclass · One call, as the `around` and `annotate` hooks see it once its caller and host are known.

**Types**

- **`CallerResolver`** · type alias · Resolves who is calling, and their scope, for one call — sync or async. The host reads its own request context (an access token, a session) to answer it.
  <br>`Callable[[], Caller | Awaitable[Caller]]`

<a id="api-negotiation"></a>
### `threetears.evals.negotiation`

Request negotiation: a client that never fails a request because a model lacks a setting.

**Functions**

- **`anthropic_thinking`** · *host contract* · function · The `thinking` an Anthropic request carries for `sent`, or `None` for none.
  <br>`anthropic_thinking(sent: WireSettings, facts: SettingFacts) -> dict[str, str | int] | None`
- **`closest_settings`** · *host contract* · function · What to send for `asked`: the closest settings `facts` say the model accepts, and why each one moved.
  <br>`closest_settings(asked: WireSettings, facts: SettingFacts, provider_family: ProviderFamily) -> tuple[WireSettings, tuple[str, ...]]`
- **`facts_from_router_catalogue`** · *host contract* · function · The facts a router's catalogue row states of a model: each parameter it does not list, as not taken.
  <br>`facts_from_router_catalogue(supported_parameters: Collection[str] | None) -> SettingFacts`
- **`negotiate`** · *host contract* · async function · Send the closest settings the model is known to take; on a recognised refusal, learn it and send again.
  <br>`negotiate(asked: WireSettings, *, provider_family: ProviderFamily, model: str, provider: str | None = None, send: Callable[[WireSettings, SettingFacts], Awaitable[R]], read_refusal: Callable[[Exception], SettingRefusal | None], learned: LearnedSettings, declared: SettingFacts | None = None) -> Negotiated[R]`
- **`read_anthropic_refusal`** · *host contract* · function · Whether Anthropic's answer refuses a request for a setting it carried, which, and in what words.
  <br>`read_anthropic_refusal(status: int, body: object) -> SettingRefusal | None`
- **`read_openai_refusal`** · *host contract* · function · Whether OpenAI's answer refuses a request for a setting it carried, which, and in what words.
  <br>`read_openai_refusal(status: int, body: object) -> SettingRefusal | None`
- **`read_router_refusal`** · *host contract* · function · Whether OpenRouter's answer refuses a request for a setting it carried, which, and in what words.
  <br>`read_router_refusal(status: int, body: object) -> SettingRefusal | None`
- **`wire_settings_for`** · *host contract* · function · What a client asks for when it is asked for `settings` at `temperature`: the engine's values, as wire ones.
  <br>`wire_settings_for(settings: ClientRequestSettings, temperature: float | None) -> WireSettings`

**Classes**

- **`LearnedSettings`** · *host contract* · class · What providers refused, per provider and model, kept for as long as the callers that share it live.
- **`Negotiated`** · *host contract* · dataclass · A served request: the caller's reply, and what the request that produced it carried.
- **`SettingFacts`** · *host contract* · dataclass · What is known of the request settings one model accepts: each fact `None` while nothing says.
- **`SettingRefusal`** · *host contract* · dataclass · A provider refusing a request for one setting it carried, in the provider's own words.
- **`WireSettings`** · *host contract* · dataclass · Request settings as a client is asked for them, or as one request carries them.

**Types**

- **`ProviderFamily`** · *host contract* · literal · The request shape a client speaks, which decides how a setting goes on the wire: `anthropic` (Anthropic's Messages API: thinking budgets, adaptive thinking with effort words, no temperature beside thinking), `openai` (OpenAI's `reasoning_effort`, no budget) and `router` (a router's chat completions: `reasoning.effort` with all seven words or a `reasoning.max_tokens` budget, never both, and enforcement as `provider.require_parameters`, under which a router that finds no endpoint taking every parameter refuses the request).
  <br>`'anthropic'` | `'openai'` | `'router'`
- **`RefusedSetting`** · *host contract* · literal · What a provider refused:
  <br>`'temperature'` | `'effort_word'` | `'effort'` | `'budget'` | `'adaptive'` | `'disable'` | `'format'` | `'max_tokens'` | `'enforcement'`

**Constants**

- **`ANTHROPIC_MIN_THINKING_BUDGET`** · *host contract* · constant (int) · Anthropic's floor on a thinking budget, in tokens: a budget below it is the caller's error, and a word sent as a budget is never below it.
  <br>`= 1024`
- **`MAX_RESENDS`** · *host contract* · constant (int) · The most times one call is resent for a refused setting: one per setting a client can drop or move (the temperature, the reasoning parameter, the reply format, enforcement).
  <br>`= 4`

<a id="api-providers"></a>
### `threetears.evals.providers`

The completion client: a model id in, the engine's completion port out, over each provider's own HTTP API.

**Functions**

- **`describe_failure`** · *host contract* · function · What the engine may know of a call through the client that raised: the host's `failure_describer`.
  <br>`describe_failure(exc: BaseException) -> ProviderFailure`

**Classes**

- **`Catalogue`** · *host contract* · class · OpenRouter's model catalogue, fetched at most once, and what providers refused, for every model that reads it.
- **`Clients`** · *host contract* · class · A host's role factory (`EvalHost.clients`): one `Model` per role, model id and temperature.
- **`Completion`** · dataclass · One reply: the completion port's ten members, and four of the client's own.
- **`ProviderError`** · *host contract* · exception · A request the provider refused or failed: who, the HTTP status, and the provider's own message.

**Types**

- **`Provider`** · literal · The providers the client speaks to: OpenRouter, Anthropic's Messages API, and any OpenAI-compatible chat endpoint.
  <br>`'openrouter'` | `'anthropic'` | `'openai'`

**Also exported here**

`Model` ([`threetears.evals`](#api-threetears-evals))

<a id="configuration"></a>
## Configuration

<a id="launch-settings"></a>
### `LaunchSettings`

The host's launch settings, as one snapshot of values.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_launch_arms` | `int` | required | How many runs one launch may start together. A group starts every member at once in one job slot, so this is the concurrency one launch adds. |
| `max_admitted_runs` | `int` | required | How many runs may be admitted and unfinished at once across every launch — the ceiling admission refuses past rather than queueing behind. |
| `judge_concurrency` | `int` | required | How many judge calls one cell makes at once. |
| `max_concurrent_cells` | `int` | `4` | The most of a run's cells that execute at once when its launch does not declare latency under test (`measure_latency`) — a ceiling, never a width a kind is pushed to: a run executes at the narrower of this and what its kind declares it is safe at (`KindWiring.max_concurrent_cells`), and a kind that declares nothing runs serially. A launch that declares latency runs its cells one at a time whatever either says. Defaults to `DEFAULT_MAX_CONCURRENT_CELLS`; `1` runs every run serially. |
| `enforcement_enabled` | `bool` | required | Whether the cost and metered-call ceilings are enforced at all. |
| `max_cost_usd` | `float` | required | The run cost ceiling a run inherits when its launch names none. |
| `max_metered_calls` | `int \| None` | required | The metered-call ceiling a run inherits when its launch names none, or `None` for a host that declares it has NO metered tools: its runs record a ceiling of `0` (origin `none_declared`), a metered call on one is refused and counted, and a launch naming a ceiling is refused, since it would bound nothing. |
| `max_out_of_run_cost_usd` | `float` | required | The most a launch's out-of-run calls — its case generation, which runs before any run exists and so under no run's cap — may together be priced at before they are made (`OutOfRunBudget`). Enforced exactly when `enforcement_enabled` is. Per LAUNCH, as `max_cost_usd` is per run: N launches may spend up to N times this out of run, as their runs may spend up to their count times their cap. An analysis generation is held to it too, per generation (`analysis_generate`): its calls run after the runs it reads, under no run's cap. |
| `judge_alternate_model` | `str \| None` | `None` | The judge a launch's arms are scored by instead of the judge role's default when that default IS one of the launch's candidate models — a model grading its own output — provided it is itself none of them (`resolve_judge_pin`), spelled as the host's clients name the model they resolve. It never overrides a judge the launch named, nor a model a judge config pins per dim: those are choices. `None` substitutes nothing, and a run judged on a candidate's model says so on every surface that lists its judges (`judges_sharing_a_candidate_model`). |
| `max_cell_timeout_s` | `float \| None` | `None` | The longest per-cell deadline a launch may name (`cell_timeout_s`), in seconds. `None` declares no ceiling of the host's own: a launch may then only LOWER its kind's deadline (the one the kind's launcher wires, or `DEFAULT_CELL_TIMEOUT_S`), so a host that declares nothing cannot be overridden upward. |
| `setting_names` | `dict[str, str]` | `{}` | What the host calls each of the fields above, keyed by field name, so a refusal names the knob an operator turns. A field the host does not name here is called by its own name; the engine names no host setting of its own. |

<a id="host-profile"></a>
### `HostProfile`

Everything the engine knows about one consuming product.

| Field | Type | Default | Description |
|---|---|---|---|
| `host_id` | `str` | required | Opaque. The engine never interprets or branches on it — it is for logs and error text. |
| `host_sweepables` | `SweepableRegistry` | required | What this host itself declares it sweeps and what holds its measurements: the shared core (`SHARED_CORE`) extended with the host's own levers, apparatus and labels. |
| `measures` | `MeasureRegistry` | required | What this host can see. The observability map. |
| `bars` | `BarRegistry` | `BarRegistry()` | The incumbent standards per behavior. A host with none registers an empty registry. |
| `style` | `StyleProfile` | `StyleProfile()` | The bounded presentation contract. No field of it is free text. |
| `world` | `WorldRegistry \| None` | `None` | What a run may set before the subject starts, and what it may only witness. |
| `caveat_kinds` | `frozenset[str]` | `frozenset()` | Caveat kinds this host declares, BEYOND the four the engine owns. |
| `variant_levers` | `VariantLeverReader \| None` | `None` | How this host resolves a run's level of each of its OWN fixed levers. See `VariantLeverReader`. |
| `observed_model_levers` | `Mapping[str, str]` | `dict()` | Levers whose INHERITED value is recoverable from what a run observably did. |
| `action_parameters` | `ActionParameterReader \| None` | `None` | How a goal check learns which of a call's recorded parameters are closed values. |
| `tool_actions` | `ToolActionReader \| None` | `None` | How a goal check learns which actions a tool has, so a misspelled one is refused at authoring. |
| `kinds` | `tuple[KindContract, ...]` | `()` | What each candidate kind's runs carry beyond the engine's own fields: its overlays and its spec. |
| `listing_elisions` | `frozenset[str]` | `frozenset()` | Paths inside `EvalRun.host_payload` that a read LISTING many runs leaves out. |
| `release_label` | `str \| None` | `None` | The `label` sweepable whose value names the BUILD of the product that ran — an app version. |
| `analysis_writer_models` | `tuple[str, ...]` | `()` | The model ids this host allows to write a campaign's analysis. Empty: any model. |
| `sweepables` | `SweepableRegistry` | derived | Every input this host's runs carry: `host_sweepables` plus each kind contract's levers. |

<a id="documents"></a>
## The report and the analysis bundle

<a id="report"></a>
### `Report`

A campaign's report: one page, the bundle it was computed from, and the typed verdicts it states.

| Field | Type | Default | Description |
|---|---|---|---|
| `basis` | `ReportBasis` | required | `analysis` over a generated memo, `code_only` over the evidence alone. |
| `title` | `str` | required | The title line: the campaign, its arms, the date and the basis. |
| `finding_count` | `int` | required | How many findings the memo holds. |
| `source` | `ReportSource` | required | What this is a report of. |
| `blocks` | `list[ReportBlock]` | required | The report, in reading order. |
| `verdicts` | `list[Verdict]` | `[]` | Every verdict code reaches, typed — the guardrails, then each bar on each cell, then the contrasts against the control — the source each printed verdict is rendered from. |
| `bundle` | `AnalysisContextBundle` | required | The evidence every number here was computed from, written beside the page. |

Its `source`, and each kind of block in `blocks`:

#### `ReportSource`

What the report is a report of, and how that analysis was generated — or that none was.

| Field | Type | Default | Description |
|---|---|---|---|
| `analysis_id` | `str \| None` | `None` | The analysis the report renders; None on a code-only report. |
| `campaign_id` | `str` | required | The campaign. |
| `campaign_name` | `str \| None` | `None` | The campaign's name, where known. |
| `scope_id` | `str` | required | The scope the campaign lives in. |
| `behavior` | `str` | required | The behaviour under analysis. |
| `generated_at` | `str` | required | When the analysis was generated (ISO-8601); on a code-only report, when the evidence was assembled. |
| `generator_model` | `str \| None` | `None` | The model that wrote the analysis; None on a code-only report. |
| `bundle_fingerprint` | `str` | required | The fingerprint of the bundle the report renders. |

#### `TextBlock`

What the analysis's author wrote, exactly as written, with the facts code states beside it.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `anchor` | `str \| None` | `None` | The id a link to this block names (`F1`, `case-3`, `T7`); None when none does. |
| `kind` | `Literal['text']` | `'text'` | Which kind of block this is. |
| `role` | `TextRole` | required | What the author wrote it as. |
| `body` | `ModelProse` | required | The author's words, Markdown allowed; empty where the author left a required field blank. |
| `facts` | `list[Fact]` | `[]` | Facts code states beside the words, in reading order. |

#### `TableBlock`

A table code laid out — its columns, its rows in their stated order, and how much of it is shown.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `anchor` | `str \| None` | `None` | The id a link to this block names (`F1`, `case-3`, `T7`); None when none does. |
| `kind` | `Literal['table']` | `'table'` | Which kind of block this is. |
| `name` | `str` | required | Which table this is, e.g. `scoreboard`, `comparisons`, `cases`. |
| `title` | `str` | required | The table's heading. |
| `columns` | `list[TableColumn]` | required | The columns, in display order. |
| `rows` | `list[dict[str, TableCell]]` | required | The rows shown, in the stated order, keyed by every column. |
| `order` | `str` | required | The order the rows are in, in words. |
| `total_rows` | `int` | required | How many rows the table has; more than are shown when it is truncated. |
| `details` | `bool` | `False` | Whether the table is drill-down: HTML folds it in a `<details>`, closed. |

#### `ChartBlock`

One of the two charts a report draws, with every value it draws, so it prints as a table too.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `anchor` | `str \| None` | `None` | The id a link to this block names (`F1`, `case-3`, `T7`); None when none does. |
| `kind` | `Literal['chart']` | `'chart'` | Which kind of block this is. |
| `chart` | `ChartKind` | required | `outcome_bar` or `interval_rows`. |
| `title` | `str` | required | What the chart shows. |
| `bars` | `list[OutcomeBar]` | `[]` | An outcome bar's arms, one bar each. |
| `rows` | `list[IntervalRow]` | `[]` | An interval chart's rows, in drawn order. |
| `unit` | `str` | `''` | The unit an interval chart's values are stated in; empty for none. |
| `zero_line` | `bool` | `False` | Whether the chart draws a line at zero (a difference's). |
| `margin` | `float \| None` | `None` | A margin band either side of zero, on the chart's ruler; None when none declared. |

#### `DisclosureBlock`

Something code must tell the reader that no author wrote — one idea.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `anchor` | `str \| None` | `None` | The id a link to this block names (`F1`, `case-3`, `T7`); None when none does. |
| `kind` | `Literal['disclosure']` | `'disclosure'` | Which kind of block this is. |
| `source` | `DisclosureSource` | required | What the disclosure speaks for. |
| `text` | `str` | required | The disclosure, one sentence or a few. |

<a id="analysis-bundle"></a>
### `AnalysisContextBundle`

The closed context bundle a generation prompt runs over.

Its top-level fields, in declaration order; each one's type is described in the API section.

| Field | Type | Default | Description |
|---|---|---|---|
| `schema_version` | `int` | `54` | Bundle-shape version, for future evolution + fingerprint clarity. |
| `campaign_id` | `str` | required | The campaign this bundle summarises. |
| `subject_id` | `str` | required | The analysed subject's stable id. |
| `subject_kind` | `str` | `''` | Discriminator; data, never a code branch. |
| `behavior` | `str` | required | Which aspect is under test, e.g. 'extraction'. |
| `template_id` | `str \| None` | `None` | Referenced eval_template (the Behavior), or None. |
| `scope_id` | `str` | required | Storage scope the member runs were loaded from. |
| `run_ids` | `list[str]` | `[]` | Resolved member-run ids (sorted). |
| `unresolved_run_ids` | `list[str]` | `[]` | Campaign run_ids not found in this scope — an honest gap, not dropped silently. |
| `archived_run_ids` | `list[str]` | `[]` | Member runs an operator archived, held out of every lens below. |
| `model_versions` | `dict[str, str]` | `{}` | Distinct models by role (candidate/judge/simulator). |
| `window` | `CampaignWindow \| None` | `None` | Derived [start, end] from member-run created_at, or None. |
| `run_summaries` | `list[RunSummary]` | `[]` | Per-run digests (run_id order). |
| `comparison` | `ComparisonSetsResult` | required | reporting.compute_comparison_sets over the member runs. |
| `frontier` | `FrontierResult` | required | reporting.compute_frontier — empty (no subjects) when the data can't seat a ranking yet. |
| `frontier_bar_withheld` | `str \| None` | `None` | Set when the frontier lens was given no bar: the campaign's effective bar (declared, else registered) is passed to it only when one names `pass_hat_k`, the measure it ranks on, and `frontier.bar` is then set and each point's `bar_decision` names the variants below it. |
| `telemetry` | `TelemetryRollup` | required | Campaign-wide descriptive telemetry. |
| `coverage` | `list[LeverCoverageInput]` | `[]` | Per-lever structural coverage map (the analysis's spine). |
| `scope_divergences` | `list[ScopeDivergence]` | `[]` | Lever changes where the whole-run measure moved by a different amount than the isolating measure, the difference itself tested and Holm-corrected within the lever — each one is a finding. |
| `whole_swings` | `list[WholeSwing]` | `[]` | Where each diverging whole-run movement went: one entry per lever, level pair and whole-run measure among `scope_divergences`, naming the component that carried it. |
| `divergences_omitted` | `int` | `0` | Gated divergences beyond the reporting cap, dropped weakest-first. Stated so a short list is not read as a complete one. |
| `divergences_tested` | `int` | `0` | Whole-and-part pairs across every lever whose divergence test carried a p, published or not. |
| `divergences_untested` | `int` | `0` | Whole-and-part pairs whose divergence could not be tested: fewer than two cases carrying both measures on a side, or a remainder (whole minus part) that moved by the same amount on every case where the two measures do not both declare value_range, so no test of the mean can call it; declare value_range on both for the bounded test to read it. |
| `declared_design` | `CampaignDesign \| None` | `None` | What the campaign SET OUT to do, carried whole rather than reduced to its control. |
| `design` | `RealizedDesign` | `RealizedDesign(control_arm=None, earlier_control=None, control_excluded=None, stale_control_disclosure=None, contrasts=[], unplaced_run_ids=[], shape='undesignated')` | What kind of experiment this is, inferred from the runs — never a declaration, and never to be called declared: the run carrying the declared control, each cell's moved levers, and whether the design is one-factor-at-a-time. |
| `incomplete_runs` | `dict[str, str]` | `{}` | Run id → status, for every resolved member run that did not reach 'completed'. |
| `short_runs` | `dict[str, str]` | `{}` | Run id → the disclosure sentence for every resolved member run that delivered fewer cells than it promised, whatever status it ended on. |
| `completeness_unknown_run_ids` | `list[str]` | `[]` | Resolved member runs carrying no completeness record, so whether they came up short is unknown rather than answered. |
| `short_cells` | `list[ShortCell]` | `[]` | Every cell holding fewer repetitions than the declaration's `intended_repetitions`, ordered by (variant_key, apparatus_class_id), each with the sentence to quote. |
| `cost_unmeasured_cells` | `list[CellCoordinate]` | `[]` | Every cell where no turn the candidate took observed spend — no usage row in its cost roles carried dollars — ordered by (variant_key, apparatus_class_id). |
| `cost_unmeasured` | `str \| None` | `None` | The sentence to quote about `cost_unmeasured_cells` — that cost was not measured there, and why a $0 would mean nothing. |
| `all_failed_cells` | `list[CellCoordinate]` | `[]` | Every cell where no result the harness did not fault took a turn — the candidate's model refused or errored on every call — ordered by (variant_key, apparatus_class_id). |
| `all_failed` | `str \| None` | `None` | The sentence to quote about `all_failed_cells` — that every result there failed, so there is no cost or latency to read. |
| `latency_contended_cells` | `list[CellCoordinate]` | `[]` | Every cell holding a result whose latency was read while other cells or runs executed beside it (`execution_mode` `concurrent`: its launch did not declare latency under test, or another run executed beside it), ordered by (variant_key, apparatus_class_id). |
| `latency_contended` | `str \| None` | `None` | The sentence to quote about `latency_contended_cells` — that latency read under concurrency is not compared, and how much was left out. |
| `held_fixed_reading` | `HeldFixedReading` | `HeldFixedReading(declared_stimulus=None, stimulus_reason='', declared_apparatus=None, run_provenance={}, contradicting_run_ids=[], disclosure=None)` | What the campaign declared held fixed beside the provenance every resolved run recorded, compared value for value, with the sentence to quote when they disagree, when runs mix commissioned and witnessed apparatus with nothing declared, or when the stimulus was declared uncontrolled. |
| `judge_agreement` | `JudgeAgreement` | `JudgeAgreement(ratings_read=0, dimensions=[], unpaired=[])` | How the judge's scores agreed with people's calibration ratings of the same results, per judged dimension, judge model and judge config: n, distinct results, exact agreement, Cohen's kappa and, on 1-5 dimensions, quadratic-weighted kappa, each pooled over people by result — over every resolved member run's results. |
| `judge_self_agreement` | `JudgeSelfAgreement` | `JudgeSelfAgreement(repeats_read=0, dimensions=[], unpaired=[])` | How the judge's repeated scores agreed with its own first scores of the same evidence, per judged dimension, judge model and judge config, read exactly as `judge_agreement` is (n, distinct results, exact agreement, kappa, weighted kappa, a "can't tell" repeat counted as a disagreement) — over every resolved member run's results. |
| `judge_evidence_tiers` | `list[JudgeEvidenceTier]` | `[]` | The evidence tier of each judge's readings on each judged dimension — a judge being a served model and a judge config — decided by code from `judge_agreement` and `judge_self_agreement`, each criterion on confidence bounds for its agreement and never the point estimate: `calibrated` (the one-sided 95% lower bound on agreement with people at or above 0.6, over at least 20 distinct results), `separation` (that bound on agreement with its own repeats at or above 0.8, over at least 120 distinct results), `incidental` (both upper bounds below their bars), or `undetermined` (not shown either way: too few results, or bounds across a bar). |
| `inter_judge_agreement` | `InterJudgeAgreement` | `InterJudgeAgreement(scores_read=0, dimensions=[], unpaired=[])` | How a second judge's scores of the member runs' stored evidence agreed with their judge's, per judged dimension, first judge and second judge: n, distinct results, exact agreement, and kappa (quadratic-weighted on 1-5, unweighted on pass/fail) with its confidence bounds — read exactly as `judge_agreement` is. |
| `judge_drift` | `JudgeDrift` | `JudgeDrift(dimensions=[], family_size=0, disclosure='Drift compares two judges on the same evidence: it detects whether the scores moved, never which judge is right. Whether either judge agrees with people is what calibration measures.')` | How far each judged dimension's scores moved when a second judge re-scored the member runs' stored evidence: the movement over cases, its interval at 1 − α/m over the dimensions, and separated (exactly when that interval excludes 0) / not separated / untested. |
| `judge_change` | `JudgeChange` | `JudgeChange(levels=[], drift_links=[], sentence=None)` | Each judge the judged member runs were scored by (model pin, configs, temperature), the runs and arms under each, and the drift readings that span a change. |
| `goal_check_proofs` | `list[GoalCheckProofReading]` | `[]` | Per goal check the member runs graded: whether it was shown, at launch, to tell its outcomes apart (`proven`), or not (`unproven`: no control, or a proof recorded under an earlier rule (`stale`); `refuted`: a control it does not beat, or a check the grammar refused at launch (`refused`)). |
| `multiple_comparisons` | `MultipleComparisons` | `MultipleComparisons(families=[], withheld=None)` | Each contrast tested against the control on every reading a live question asks about, per rig, with Holm correction inside each question's family: the family's size, each comparison's adjusted p and the verdict read off it. |
| `guardrails` | `GuardrailReadings` | `GuardrailReadings(measures=[], dimensions=[], checks=[], withheld=None)` | The guardrails — boundary judged dimensions and measures declared `guardrail`, what the candidate must not get worse on — each decided for every arm against the control on its own 95% interval: `held` (shown no worse than its margin), `breached` (shown worse) or `undecided`. |
| `run_margins` | `dict[str, float]` | `{}` | Margins on core rate measures (`accuracy`) that every member run declared alike at launch (`EvalRun.declared_margins`), by measure. |
| `launch_declarations` | `list[str]` | `[]` | One sentence per host measure that is read here otherwise than the reading host now declares it. |
| `run_margins_withheld` | `str \| None` | `None` | Why a margin some member runs declared on a core rate measure is read on none of its comparisons: the runs do not all declare it, or declare different ones, and a contrast between two arms read against a margin only one of them chose would be read against a margin nobody chose for the pair. |
| `scoreboard` | `list[ArmScoreboard]` | `[]` | Each arm's census, in the order of `variant_index`: trials ran, lost to the test setup by cause, usable, passed and failed by cause, the cases run and the cases with a usable trial, and what the arm cost. |
| `trials` | `list[TrialRow]` | `[]` | One row per trial — one case run once on one arm — with how it ended and why: its failed checks, the judge's scores and reasons, what the case asked, the tail of what the judge was handed and the recorded error, each capped. |
| `trials_omitted` | `int` | `0` | Trials beyond the cap, left out of `trials`; the scoreboard counts every one. |
| `case_sets` | `list[CaseSetRef]` | `[]` | The named, versioned case sets the member runs were launched against, distinct and sorted. |
| `design_frozen_at` | `str \| None` | `None` | When the campaign's design froze (ISO-8601): its first member run's `created_at`, archived runs included, after which a design edit is refused and a question asked confirms nothing (`reading_scope.post_hoc_questions`). |
| `package_version` | `str \| None` | `None` | The 3tears-evals version that assembled the bundle; None when the build carries no distribution metadata (a source tree on the path). |
| `reading_scope` | `ReadingScope` | `ReadingScope(questions_declared=False, post_hoc_questions=[], exploratory_measures=[], exploratory_dimensions=[], disclosure=None)` | Which readings no declared question asked about: exploratory, reportable as leads and never as confirmed answers. |
| `exclusions` | `list[ArmExclusions]` | `[]` | Per arm (cell), in coordinate order: how many of its trials were left out of its measures, by cause — `infra` (the harness faulted it), `truncated` (a judge reply cut off at the cap it was sent, named in `truncated_sent_caps`), `judge_cannot_tell` — so a shrunken n says why, arm by arm. |
| `differential_exclusions` | `list[DifferentialExclusion]` | `[]` | Every (arm, cause) where one cause took more than DIFFERENTIAL_EXCLUSION_SHARE (10%) of the arm's trials, with every other arm's share of the same cause beside it. |
| `verdict_order` | `VerdictOrder` | `VerdictOrder(merit_priority=[], tiers=[], unranked_bar_measure_ids=[], questions=[])` | The order verdicts are read in, as declared: the bars on each axis of `merit_priority`, strongest first, the bars on no ranked axis, and for each live question the bars on the axes it names. |
| `measurement_windows` | `list[MeasurementWindow]` | `[]` | Each resolved member run's wall-clock measurement span, derived from that run's own scored_at stamps. |
| `launch_disclosure` | `str \| None` | `None` | Set when the member runs were NOT all started by one campaign launch — some came from different launches, or were started on their own. |
| `model_identity_disclosure` | `str \| None` | `None` | Set when some of the member runs' candidate calls or judge scores came back from a response that named no model, so the record holds only the model id each asked for — and a provider may answer an id with a model of its choosing (a floating alias). |
| `request_settings_disclosure` | `str \| None` | `None` | Set when the member runs' judge or simulated-user calls were not shown to be sent as each run asked: sent other settings than asked (a client may send the closest setting a model accepts), sent more than one way (with the counts), sent without enforcement (the provider may have ignored any setting the serving model does not take), or not reported at all (unverified). |
| `measurement_window_disclosure` | `str \| None` | `None` | Set when AT LEAST ONE PAIR of member runs was measured over spans of wall-clock time that do not overlap — so anything that moved between those spans (a model revision, a provider's load, a rate limit) moved with the runs, and a difference between the two runs of such a pair is not attributable to the runs alone. |
| `apparatus_confounds` | `list[Confound]` | `[]` | Apparatus dimensions that varied across the WHOLE campaign, scanned independently of any lever. |
| `aliased_factors` | `list[AliasedFactors]` | `[]` | Factors that moved in lockstep across the campaign: each group's factors split the runs identically, so no comparison separates them. |
| `factor_pairs` | `FactorPairScan \| None` | `None` | Every pair of varying factors checked for co-varying, with a pivot (unrun combinations as not_run) for each co-varying pair outside a lockstep group, and how many pairs were examined. |
| `declared_crossing` | `DeclaredCrossing \| None` | `None` | Where the declared design says which combinations of its levels it meant to run: every cell, marked ran, not_run (a gap), skipped_by_design (left out on purpose: never a gap) or undetermined. |
| `arm_mechanisms` | `list[ArmMechanismReading]` | `[]` | Each arm's mean of every covariate read as an observed mechanism (today the candidate's reasoning share, `reasoning_ratio`), sorted by arm then covariate. |
| `arm_served_models` | `list[ArmServedModel]` | `[]` | Which model the provider's responses named as having answered each arm's candidate calls, sorted by arm. |
| `arm_production_footings` | `dict[str, PooledProductionFooting]` | `{}` | Arm (variant key) -> what each of its runs set away from the subject's production configuration, read off the host's sweepable declarations: `runs` maps run id -> that run's footing (`moved` with levels, `unchecked`, `held`; null for a run nobody could check). |
| `cell_model_version` | `int` | `13` | Which definition of a cell produced `cells`. |
| `host_declarations_digest` | `str` | required | sha256 of the host's declared sweepables (each one's name, role and blank rule) and world dimension names — the declarations that partition observations into apparatus classes and arms (`host_declarations_digest`). |
| `cells` | `list[Cell]` | `[]` | Every (variant, apparatus class) that any observation landed in, with how many observations pooled there and over how many cases and repeats per case. |
| `variant_index` | `list[VariantIndexEntry]` | `[]` | One entry per keyed variant — the variant key and the resolved lever map it was digested from. |
| `refused_merges` | `list[RefusedMerge]` | `[]` | Same-variant cells that did NOT pool, and which rule kept them apart. |
| `refused_merges_omitted` | `int` | `0` | Refused pairs beyond the reporting cap, dropped smallest-first (by the observations the two cells hold). Stated so a short list is not read as a complete one. |
| `next_experiments` | `list[NextExperiment]` | `[]` | What recording one unrecorded apparatus dimension would buy, in units of k. |
| `next_experiments_omitted` | `int` | `0` | Recordings beyond the reporting cap, dropped least-gain-first (by the observations recording would add). Stated so a short list is not read as a complete one. |
| `subject_key_instabilities` | `list[SubjectKeyInstability]` | `[]` | Subject keys and labels disagreeing about how many subjects there are — one key under two labels, or one label under two keys. |
| `measure_catalog` | `dict[str, MetricDescriptor]` | `{}` | What each measure name MEANS — the registry descriptor, carried once per campaign rather than repeated on every run's summary. |
| `judged_measures` | `list[JudgedMeasure]` | `[]` | Every judged dimension any result was scored on, with its per-cell scores. |
| `bar_adjudications` | `list[BarAdjudication]` | `[]` | Every bar this campaign is held to — its own declared bars, plus each registered incumbent for its behavior that no declared bar overrides — with a verdict per cell computed here, or the reason none exists. |
| `cell_measures` | `list[CellFacts]` | `[]` | Everything measured in each cell, one entry per cell, the declared control's cells first as the reference, then every other arm alphabetically by name (an order that is not a ranking): every measure over the cell's non-faulted observations — the population every bar verdict is read over, so a value here and a verdict on the same cell describe the same observations — every judged dimension scored there, its replication, and the notes on its member runs. |
| `time_axis` | `TimeAxis \| None` | `None` | The runs placed in time, when they span two or more builds (the host's release label) or, failing that, two or more days: each position names its runs and carries every cell measured there, computed exactly as `cell_measures` is over that position's runs alone. |
| `time_axis_withheld` | `str \| None` | `None` | Why there is no time axis — what every run shared — or None when there is one. |
| `confound_catalog` | `dict[str, str]` | `{}` | What a change in each confound dimension DOES to a measurement, carried once per campaign rather than repeated on every lever that names it. |
| `prior_insights` | `list[EvalInsight]` | `[]` | Subject-scoped prior insights (newest first), one per claim and at most the reporting cap of the newest. |
| `prior_insights_omitted` | `int` | `0` | Live prior insights the ledger holds that `prior_insights` does not carry: an older insight stating the same claim as a carried one, and every insight beyond the cap, oldest dropped first. |
| `retracted_insights` | `dict[str, str]` | `{}` | Insight id → the archived analysis that minted it, for every insight the ledger held that is NOT in `prior_insights` because its analysis was archived as shown false. |

<a id="goal-checks"></a>
## The goal-check language

Goal-state DSL — small expression language for eval scoring.

Expressions are evaluated against a cell's end state (`state`: declared world dimension name →
value), its `CallLedger` (read by the call
predicates), and a variation parameter dict (`variation`). The evaluation is three-valued: True,
False, or `Missing` — *not established* — when it rests on a value the end state does not hold.
A check whose value is `Missing` is never a pass; the grader records it as failed with a detail
saying it was not established and naming the path that held nothing (see *Missing values* below).

#### Surface

Path access (Python-style):

```python
state.inventory.orders           # list
state.inventory.orders.length    # len(...)
state.inventory.orders[-1]       # indexing
state.support.messages[-1].content
variation.region_pair            # variation parameter
```

`state.<path>` roots at a declared world dimension's NAME and nothing else, as the authoring gate
reads it: `state.inbox.messages` and a flat `state.ingest_backlog` both resolve through the
host's world registry, by the longest declared prefix, and whatever follows the dimension addresses
inside its value. A path naming no declared dimension is `Missing`. There is no reading of a
raw layout: a host that declares no world has no `state` to read, and a path under it raises.

Comparisons:

```python
state.inventory.orders.length >= 1
state.inventory.orders[0].sku == "SKU-42"
variation.tone != "hostile"
```

Predicates (function-call form):

```python
contains(state.inventory.orders, "SKU-42")              # value in path
contains(state.inventory.categories, variation.category) # a case parameter in path
intersects(state.inventory.categories, ["toys", "games"]) # non-empty intersection
length(state.inventory.orders) >= 2                     # equivalent to .length
```

Ordering predicates (across all tools' recorded calls — the cell's call ledger, never world state):

```python
called_before("inventory.search", "inventory.place_order")
called_after("inventory.place_order", "inventory.cancel_order")
call_count("inventory.place_order") >= 2
last_call_was("inventory.place_order")
```

Each ordering predicate is False when either action never happened, so "never acted" reads like "acted in
the wrong order", and `not called_before(a, b)` holds for a candidate that did neither. A check about order
pairs it with `call_count(a) >= 1`. `last_call_was` reads only the cell's final call, across the whole
cell, so a candidate that acted and then called something else answers False for the action it did take.

A deliberate pass (the engine's reserved ledger entry, recorded by `CallLedger.record_pass`) is not a call:
none of the call builtins sees it, and `passed()` reads it, the same for every host:

```python
passed()
not passed() and call_count("inventory.place_order") >= 1
```

`passed()` holds when the cell recorded a pass and no call: a candidate that acted and then passed did not
pass. A cell that did nothing and recorded no pass did not pass either.

Refused calls (`refused()` and `refused_count()` read the calls a tool refused — recorded apart by
`CallLedger.record_refusal`, never among the calls — and nothing else does):

```python
refused("inventory.place_order")
refused_count("inventory.place_order") == 0
call_count("inventory.place_order") >= 1 and not refused("inventory.place_order")
```

Every other call builtin counts only calls that succeeded, so a candidate whose every call was refused reads,
to them, as one that never called; these two are what tell the two apart. `refused(a)` is
`refused_count(a) >= 1`. A ledger that keeps no refusal record (a kind that does not record refusals, or a
cell stored before refusals were recorded) is not read as "none refused": evaluating either against it raises,
so the check cannot be graded rather than graded as a pass. `passed()` does not read refusals: it is an
end-state check, and a refused call is its own outcome, which these two grade.

Call parameters (`calls()` returns the matching calls' recorded parameters, in order):

```python
calls("inventory.place_order").length >= 3
any(it.priority == "rush" for it in calls("inventory.add_note"))
all(it.note_text.length > 0 for it in calls("inventory.place_order"))
```

World events (`fired()` reads which triggered dimensions fired during the cell, by the rig or in the
world — never world state, and never at t=0; `fired_armed()` reads only a firing of the event the
cell's seed armed, so the world's own firing on the same dimension does not satisfy it):

```python
fired("inventory.restock_alarm")
not fired("support.escalation")
fired_armed("inventory.restock_alarm")
```

Generator predicates (`it` binds to each element):

```python
any(it.sku == "X" for it in state.inventory.orders)
all(intersects(it.tags, ["toys", "games"]) for it in state.inventory.orders)
```

A case parameter (`variation.<name>`) is one string, as the case stores it, and never a list: a
case's variation parameters are a flat string map. So it is compared (`==`, `!=`) or looked for
(`contains(state.<path>, variation.<name>)`), and never read as a collection — `intersects` over
one, `contains` searching one, a generator over one or an index into one is refused where the
expression is parsed. Write a set of values in the check itself, as a list literal.

Boolean composition:

```python
state.inventory.orders.length >= 1 and any(it.sku == "X" for it in state.inventory.orders)
not state.support.messages[-1].content == ""
```

#### Missing values

A path that resolves to nothing — a dimension the end state does not hold, an index past the end,
a field a value does not carry — is `Missing`: *unknown*, not *absent* and not *empty*. It
propagates by Kleene's three-valued logic, so negating a question never turns "unknown" into "yes":

* a comparison with a `Missing` side is `Missing` (and so is `contains`/`intersects` over one,
  and `.length`/`length()` of one);
* a list or tuple literal holding a `Missing` element is `Missing` (`[state.x]` is not a
  one-element list when `state.x` resolved to nothing);
* `fired_armed(...)` is `Missing` for a cell whose firings cannot say which were armed — a
  witnessed cell, which no seed armed (`of`);
* `not Missing` is `Missing`;
* `a and b` is False when any operand is False, else `Missing` when any is `Missing`, else True;
  `a or b` is True when any operand is True, else `Missing` when any is `Missing`, else False;
* `any(...)`/`all(...)` over a `Missing` iterable is `Missing`; over elements, `any` is True
  when some element's body is True, else `Missing` when some is `Missing`, and `all` the dual.

So `not state.support.messages[-1].content == ""` holds when the last message has content, fails
when it is empty, and is *not established* when there are no messages at all — exactly as
`state.support.messages[-1].content != ""` is. A top-level `Missing` is never a pass:
`evaluate` returns False for it and `evaluate_with_detail` says it was not established.
Note that `.length` is always `len()` of the value, so a mapping field literally named
`length` cannot be reached by path.

#### Static extraction

`extract_paths` parses an expression and reports what it reads without evaluating
anything, so an authoring gate can resolve a path against a host's declared vocabulary
before a run exists:

```python
state.inventory.orders.length >= 1      # reads the path inventory.orders.length
state.inventory.orders[0].sku == "X"    # reads inventory.orders — an index addresses inside it
```

Roots are reported apart rather than merged: `variation` names a case parameter, not
world state, and a path below an index or bound to `it` addresses inside a value rather
than naming one. Every example in this docstring is extracted by
`tests/test_dsl.py`, so a surface documented here that the language cannot
actually parse fails there rather than misleading an author.

#### Text matches over model prose

`contains()` is generic: membership over a structured value (an array of orders) and a
substring test over a string. The first is a mechanical check; the second, over text a model
wrote, is a keyword test on prose — an eval dimension wearing a validator's clothes. The
language cannot tell them apart, because the difference is the operand's TYPE,
so `extract_text_matches` reports every text comparison with the field it reads and the
authoring gate asks the host's vocabulary which of those fields are prose
(`world_prose_matches`). Evaluation is unchanged: a stored template
keeps loading and running; the refusal is at authoring.

A call parameter is text the model wrote until its tool says otherwise. `calls()` exposes what
the subject passed, which includes what it said, so a text comparison over a parameter is allowed
only where the action's own parameter schema — the one the model is shown — closes the value
(`enum`, `const` or `pattern`); `call_parameter_matches` reports every other one, and
every one over an action or parameter the host does not describe. The schema is the host's
statement, read as given: the tool's own model-facing schema is the source. A `pattern` that admits free text would make that statement false, which is the host's
defect to fix in its schema; this module does not second-guess a regex.

#### Safety

The parser uses `ast` with a strict node allowlist. Disallowed: lambdas,
dict literals, conditional expressions, list comprehensions (only generator
expressions for `any`/`all`), attribute access on non-allowlisted root
names, function calls to anything except the DSL builtins above. Malformed
or disallowed expressions raise `DSLError` at parse time, not at
evaluation time — pinning down bad expressions in the template editor before
a run starts.

<a id="goal-check-functions"></a>
### Functions

| Function | Arguments | Description |
|---|---|---|
| `all` | `<body> for it in <path>` | Evaluate `any(<body> for it in <path>)` / `all(...)` three-valued. |
| `any` | `<body> for it in <path>` | Evaluate `any(<body> for it in <path>)` / `all(...)` three-valued. |
| `call_count` | `spec` | Return the count of recorded calls matching `tool.action`. |
| `called_after` | `first, second` | True when the last occurrence of `first` follows the last occurrence of `second`. |
| `called_before` | `first, second` | True when the first occurrence of `first` precedes the first occurrence of `second`. |
| `calls` | `spec` | Return the recorded parameters of every call matching `tool.action`, in ledger order. |
| `contains` | `haystack, needle` | Whether `needle` is in `haystack` — substring over a string, membership otherwise; `Missing` over one. |
| `fired` | `name` | Whether the triggered dimension `name` fired during the cell — for `fired_armed`, as the seed's armed event. |
| `fired_armed` | `name` | Whether the triggered dimension `name` fired during the cell — for `fired_armed`, as the seed's armed event. |
| `intersects` | `a, b` | Whether two sets-of-elements share at least one element; `Missing` when either is Missing. |
| `last_call_was` | `spec` | True when the most recent recorded call matches `tool.action`. |
| `length` | `value` | Return `len(value)` or Missing if unsizable. |
| `passed` | `` | True when the cell recorded a deliberate pass and made no call. |
| `refused` | `spec` | True when a tool refused at least one call matching `tool.action`; reads only refused calls. |
| `refused_count` | `spec` | Return the count of refused calls matching `tool.action`; a call that succeeded is never counted here. |

<a id="actions"></a>
## The action catalogue (MCP)

Every engine action, as every transport mounts it (the FastMCP tools, a host's own CLI or REST mapping). A mounted tool takes `action` plus the union of its actions' parameters; `help` is generated. A parameter marked `?` is optional. `job` marks long work: it returns jobs to poll with `job_poll`. A host may contribute actions of its own, which are not listed here.

| Tool (default prefix) | Classes it mounts | Description |
|---|---|---|
| `evals` | `read`, `spend`, `write` | Run, read and analyse evals: launch runs, poll their jobs, and read campaigns and reports. |
| `evals_admin` | `destructive` | Irreversibly delete eval records. Every action needs `confirm` echoing what it destroys. |

### Find what is there

| Action | Class | Parameters | Description |
|---|---|---|---|
| `templates_list` | `read` | — | List the scope's active templates — what a launch can run. |
| `judge_configs_list` | `read` | `dim_filter?` | List the scope's judge configs — the versioned prompts a second judge or a launch can name. |
| `runs_list` | `read` | `status?`, `include_archived?`, `label?` | List the scope's runs, newest first. |
| `runs_unreadable` | `read` | — | List the scope's stored runs this build cannot read, each with why. |
| `case_sets_list` | `read` | `case_set_filter?` | List the scope's named case sets, every version — what a launch can target by name. |
| `campaigns_list` | `read` | `include_archived?` | List the scope's campaigns — the sets of runs an analysis reads. |

### Run and watch

| Action | Class | Parameters | Description |
|---|---|---|---|
| `sweep_launch` | `spend`, job | `template_id`, `subject_id`, `campaign_name`, `campaign_behavior`, `campaign_description?`, `k_runs?`, `max_concurrent_arms?`, `judge_model?`, `simulator_model?`, `max_cost_usd?`, `cell_timeout_s?`, `case_set_name?`, `case_set_version?`, `measure_latency?`, `arms` | Launch arms that differ beyond the model, one after another, into one campaign, as one job. |
| `case_set_mint` | `write` | `case_set`, `template_id`, `test_case_ids`, `tracked?` | Store the next version of a named case set: a template's cases, frozen in order. |
| `run_launch` | `spend`, job | `template_id`, `subject_id`, `models?`, `k_runs?`, `n_variations?`, `variation_model?`, `overlays?`, `apparatus_settings?`, `max_cost_usd?`, `judge_model?`, `simulator_model?`, `cell_timeout_s?`, `case_set_name?`, `case_set_version?`, `measure_latency?` | Launch a template's runs, one per model, as one job to poll: its arms prepared, its runs run. |
| `launch_estimate` | `read` | `template_id`, `subject_id`, `models?`, `k_runs?`, `n_variations?`, `variation_model?`, `overlays?`, `apparatus_settings?`, `max_cost_usd?`, `judge_model?`, `simulator_model?`, `cell_timeout_s?`, `case_set_name?`, `case_set_version?`, `measure_latency?`, `n_test_cases?` | Estimate what a run_launch would cost, from what the scope's runs have spent. |
| `job_poll` | `read` | `job_id` | Read where a job stands — a launch, a run, a sweep, a generation, a judge pass — and its result. |
| `job_cancel` | `write` | `job_id`, `reason?` | Ask a running job to stop; poll to see it land as cancelled. |
| `run_get` | `read` | `run_id` | Summarise one run: how it ended, how its results came out, each measure's mean. |
| `results_list` | `read` | `run_id`, `condition_filter?`, `offset?`, `limit?` | List one run's results, a light row each — case, repeat, condition, headline measures — paged. |
| `result_get` | `read` | `result_id`, `part?` | Read one stored result: its record, usage rows and condition, and one part of its trace. |

### Analyse and report

| Action | Class | Parameters | Description |
|---|---|---|---|
| `campaign_create` | `write` | `name`, `subject_id`, `behavior`, `description?`, `run_ids?`, `template_id?`, `declared_design?`, `control_from_run_id?`, `exploratory?` | Create a campaign over runs in the scope, for an analysis to read. |
| `campaign_update` | `write` | `campaign_id`, `name?`, `description?`, `declared_design?`, `exploratory?`, `control_from_run_id?` | Rename or redescribe a campaign, or declare, amend or withdraw its design before its first run. |
| `campaign_runs_add` | `write` | `campaign_id`, `run_ids` | Put runs in a campaign — runs launched after it was created, or measured on their own first. |
| `campaign_runs_remove` | `write` | `campaign_id`, `run_ids` | Take runs out of a campaign; the runs and their results stay, in any other campaign too. |
| `analysis_generate` | `spend`, job | `campaign_id`, `model?` | Generate a campaign's analysis with a paid model call, as a job to poll. |
| `analysis_estimate` | `read` | `campaign_id`, `model?` | Price a campaign's analysis generation against the host's out-of-run cap, without spending. |
| `judge_second` | `spend`, job | `run_id`, `second_judge_model`, `second_judge_config_ids?`, `second_judge_temperature?`, `result_ids?`, `sample_fraction?`, `sample_seed?` | Ask a second judge to score a seeded share of a finished run's judged results, as a job; read agreement. |
| `judge_second_estimate` | `read` | `run_id`, `second_judge_model`, `second_judge_config_ids?`, `second_judge_temperature?`, `result_ids?`, `sample_fraction?`, `sample_seed?` | Price a second judge's pass over a run against the host's out-of-run cap, without spending. |
| `judge_drift_check` | `spend`, job | `run_id`, `second_judge_model`, `second_judge_config_ids?`, `second_judge_temperature?`, `result_ids?` | Re-score every judged result of a run under a changed judge, as a job; read how far each moved. |
| `judge_temperature` | `spend`, job | `run_id`, `selection?`, `repeats?`, `result_ids?` | Re-judge a run's borderline cases at temperature 0 and at the provider default, as a job; compare. |
| `judge_temperature_estimate` | `read` | `run_id`, `selection?`, `repeats?`, `result_ids?` | Price a judge temperature comparison against the host's out-of-run cap, without spending. |
| `analyses_list` | `read` | `campaign_id` | List a campaign's stored analyses. |
| `insights_list` | `read` | `subject_filter?`, `campaign_filter?` | List the scope's insights — the durable claims analyses minted — newest first. |
| `insight_get` | `read` | `insight_id` | Read one insight in full: its statement, evidence runs, provenance and standing. |
| `analyses_undescribable` | `read` | — | List the scope's analyses holding an arm whose levels this build cannot describe. |
| `report_read` | `read` | `campaign_id`, `format?` | Read a campaign's report — its analysis, else its evidence alone — as Markdown, JSON or HTML. |
| `bars_propose` | `read` | `campaign_id` | Propose bars from a single-cell baseline campaign's measured incumbent; registers nothing. |
| `reporter_case_freeze` | `write` | `template_id`, `campaign_id`, `recorded_analysis_id?`, `labels?`, `supersedes?` | Freeze a campaign's analysis bundle, and the memo it got, into a case of a reporter template. |
| `reporter_cases_list` | `read` | `template_id`, `include_archived?` | List a reporter template's cases: which each campaign and memo launches, superseded or retired. |
| `judge_template_create` | `write` | `template_name`, `intent` | Write a judge template: the judge kind and an intent, which judge cases are frozen into. |
| `judge_cases_freeze` | `write` | `template_id`, `judged_run_ids?`, `from_labels?`, `criteria?`, `judged_result_ids?`, `into_case_set?` | Freeze stored judged outputs, their criteria and their labels into cases of a judge template. |
| `judge_case_author` | `write` | `template_id`, `criterion_template_id`, `criterion`, `output`, `case_material`, `expected`, `note`, `scenario?`, `subject?`, `judged_artifact?` | Write a judge case by hand: an output, the rubric dimension it is judged on, and its verdict. |
| `judge_profiles_record` | `write` | `judge_run_ids` | Store what a judge campaign measured each judge to be, per criterion, for other campaigns to read. |
| `judge_profiles_list` | `read` | `criterion_dim?` | List the stored judge profiles: each judge and criterion measured, on what, when, and its tier. |
| `scope_pivot` | `read` | `row_factor`, `column_factor`, `metric?`, `weighting?`, `subject_filter?`, `run_status?`, `predicted_cost?`, `launched_run_ids?` | Aggregate one measure over the scope's observations by two coordinates, cell by cell. |
| `runs_compare` | `read` | `baseline_run_id`, `candidate_run_id` | Compare one run's arm against another's: pass^k, mean composite, their deltas and the test. |
| `runs_bisect` | `read` | `baseline_run_id`, `candidate_run_id` | Split the versioned inputs of two runs into those that differ, agree and cannot be decided. |
| `scope_out_of_run_spend` | `read` | `purpose_filter?`, `launch_group_filter?`, `template_filter?` | List what the engine spent outside any run — case generations, rubric proposals and analysis generations — with totals. |
| `scope_history` | `read` | `metric?`, `min_absolute_change?`, `min_relative_change?`, `subject_filter?`, `run_status?` | Series one measure over time for each contestant in the scope, flagging regressions. |
| `scope_frontier` | `read` | `bar?`, `subject_filter?`, `run_status?` | Rank each subject's variants on quality, cost and latency, and name the cheapest that clears a bar. |
| `scope_export` | `read` | `export_format?`, `run_status?`, `export_run_ids?` | Export the scope's observations as flat rows, CSV or JSON, for analysis elsewhere. |

### Curate

| Action | Class | Parameters | Description |
|---|---|---|---|
| `run_archive` | `write` | `run_id`, `archived?` | Archive a run (or restore it): out of every cohort, nothing destroyed. |
| `run_label` | `write` | `run_id`, `run_labels` | Name a run with labels (or clear them): runs_list lists runs by one, no key moves. |
| `campaign_archive` | `write` | `campaign_id`, `archived?` | Archive a campaign (or restore it): out of listings, nothing destroyed. |
| `result_rate` | `write` | `result_id`, `rubric_dim`, `score`, `rating_reason` | Rate one judged dimension of one result, as an agent — kept beside people's ratings, never pooled. |
| `analysis_archive` | `write` | `analysis_id`, `archived`, `archive_reason?` | Archive a stored analysis (or restore it): marked, its insights retracted, nothing destroyed. |
| `reporter_case_archive` | `write` | `reporter_case_id`, `archived?`, `archive_reason?` | Retire a reporter case (or restore it): no launch runs it again, nothing destroyed. |
| `run_delete` | `destructive` | `run_id`, `confirm` | Destroy a run, its results and its campaign memberships. Unrecoverable; archive instead. |
| `analysis_delete` | `destructive` | `analysis_id`, `confirm` | Destroy a stored analysis; the insights it minted remain. Unrecoverable; archive instead. |
| `insight_delete` | `destructive` | `insight_id`, `confirm` | Destroy one insight, so no later analysis reads it as prior context. Unrecoverable. |

<a id="action-parameters"></a>
### Parameters

| Parameter | Type | Description |
|---|---|---|
| `analysis_id` | `string` | A stored analysis's id, as analyses_list names it. |
| `apparatus_settings` | `object` or `null` | Host-declared apparatus values to set the runs' rig up with, by apparatus dimension (e.g. who sits in an adjudicator's seat) — each a string, a bool or a number, and one the template's kind reads; refused otherwise. Recorded on every run and part of its measurement context, so one template can be compared at two. |
| `archive_reason` | `string` or `null` | Why the record is archived (an analysis shown false or superseded, a reporter case that can no longer measure anything); cleared on restore. |
| `archived` | `boolean` | The state to set: true retires the record, false restores it. |
| `arms` | array of `object` | The arms, launched in this order: each {model, label?, overlays?, apparatus_settings?, judge_model?, simulator_model?} — its own model and what it sets differently from its siblings. |
| `bar` | `number` or `null` | The pass^k a variant must clear, from 0 to 1; omitted ranks the variants without a verdict. |
| `baseline_run_id` | `string` | The run read as the baseline (A), as runs_list names it. |
| `behavior` | `string` | Which aspect of the subject is under test. |
| `campaign_behavior` | `string` | What the sweep's campaign measures. |
| `campaign_description` | `string` | The sweep's campaign's description. |
| `campaign_filter` | `string` or `null` | List only the insights this campaign's analyses minted, by id; omitted lists every campaign's. |
| `campaign_id` | `string` | A campaign's id, as campaigns_list names it. |
| `campaign_name` | `string` | The campaign every arm's run joins as it is created. It is created exploratory, since a sweep takes no design; declare one afterwards with campaign_update. |
| `candidate_run_id` | `string` | The run read against the baseline (B), as runs_list names it. |
| `case_material` | `string` | What the output is judged against: its source, or its facts. |
| `case_set` | `string` | The case set's name; a new name starts at version 1. |
| `case_set_filter` | `string` or `null` | List only this case set's versions. |
| `case_set_name` | `string` or `null` | A named case set of the template to run, as case_sets_list names it; every arm runs exactly its cases and records it. Named with case_set_version, and refused beside n_variations. |
| `case_set_version` | `integer` or `null` | The version of case_set_name to run; required with it. |
| `cell_timeout_s` | `number` or `null` | The deadline each cell runs under, in seconds, in place of the kind's own; at or below the host's ceiling (a host declaring none allows only lowering the kind's deadline). Recorded on every run. |
| `column_factor` | `string` | The coordinate the columns are; a pooled ranking across the rows is checked for reversal on it. |
| `condition_filter` | `'ok'` \| `'candidate_fail'` \| `'infra_exclude'` or `null` | List only results in this condition: ok (delivered, scored), candidate_fail (scored as a hard fail) or infra_exclude (a harness fault, in no aggregate). |
| `confirm` | `string` | Must echo the id of what is destroyed, exactly. |
| `control_from_run_id` | `string` or `null` | One of the campaign's runs whose variant becomes the declared control, the cell every other is read against: its variant key is resolved from the run, so nobody types it. Requires a declaration — declared_design in the same call, or, on an update, the stored one — and is refused beside a design that types its own control. On an update, null clears the control and omitted leaves it as it is. |
| `criteria` | array of `string` or `null` | The criteria to freeze; omitted freezes every dim each result was judged on. |
| `criterion` | `string` | The rubric dimension's name on that template. |
| `criterion_dim` | `string` or `null` | Only the profiles of this criterion's dim; omitted lists every one. |
| `criterion_template_id` | `string` | The subject template whose rubric dimension the case is judged on, as templates_list names it. |
| `declared_design` | `object` or `null` | What the campaign sets out to learn: axes (at least one: {axis_id, values: [{content, display}], rationale?}, each axis_id a lever or open-family member this host declares), held_fixed ({stimulus: controlled\|uncontrolled, stimulus_reason (required when uncontrolled), apparatus: commissioned\|witnessed}), and optionally questions ([{id, text, merit_axes?}]), bars ([{measure_id, threshold, direction}], no looser than the registered ones), merit_priority and intended_repetitions. Validated and gated as every campaign declaration is. A campaign is created with a design or with exploratory: true, never neither and never both; its control is named by control_from_run_id. On an update it replaces the stored declaration, its questions reconciled with the stored ones by id; declaring one on an exploratory campaign makes it declared; null withdraws the declaration, only beside exploratory: true; omitted leaves it as it is. |
| `description` (on `campaign_create`) | `string` | A longer description of the campaign. |
| `description` (on `campaign_update`) | `string` or `null` | A longer description of the campaign. |
| `dim_filter` | `string` or `null` | List only the configs asking for this dimension. |
| `expected` | `integer` or `'pass'` \| `'fail'` | The verdict the output must get: 'pass' or 'fail' (or 1 / 0) on a pass/fail dimension, a score from 1 to 5 on an ordinal one. Refused, naming both, when the dimension's scale cannot hold it. |
| `exploratory` | `boolean` | True says the campaign declares no design: someone learning to run evals, or testing an intuition with no settled question. Its report and analysis say once that its readings confirm nothing, and its design is inferred from the runs. On a create, required without declared_design and refused beside one; on an update, required beside declared_design: null and refused beside a design, or on a declared campaign the update leaves declared. An assertion checked and never stored: a campaign is exploratory exactly while it declares no design, so declaring one later (campaign_update) ends it. |
| `export_format` | `'csv'` \| `'json'` | The export's form: csv (flat rows, a column per lever) or json (the rows and what was left out). |
| `export_run_ids` | array of `string` or `null` | Export only these runs, archived ones included since they are named; omitted exports all. |
| `format` | `'markdown'` \| `'json'` \| `'html'` | The report's form: markdown (the memo), json (the schema's form), html (script-free). |
| `from_labels` | `boolean` | Freeze from the scope's person labels instead of from runs: one case per output and criterion, from the content each label carries, reading no run or result, so it works after the runs are gone. A label that carries no content is skipped by name. |
| `include_archived` | `boolean` | List archived records too; they are left out by default. |
| `insight_id` | `string` | An insight's id, as insights_list names it. |
| `intent` | `string` | What its judge campaign measures, in a sentence a reader of its report is told. |
| `into_case_set` | `string` or `null` | A case set to mint the next version of, listing exactly the frozen cases — what a launch targets. |
| `job_id` | `string` | A job's id, exactly as the action that started it returned it. |
| `judge_model` | `string` or `null` | The judge model, where the kind is model-judged. |
| `judge_run_ids` | array of `string` | The finished judge-kind runs whose trials make up the measurement — every arm of the judge campaign, as runs_list names them. Each judge and criterion they measured is recorded from all of them together. |
| `judged_artifact` | `'document'` \| `'transcript'` | 'document' (judged against its material) or 'transcript' (a conversation's). |
| `judged_result_ids` | array of `string` or `null` | The results to freeze; omitted freezes every result of the runs. |
| `judged_run_ids` | array of `string` | The judged runs whose stored results to freeze, as runs_list names them; omitted with from_labels, and required without it. |
| `k_runs` | `integer` | Repeats of every case, for pass^k. |
| `label` | `string` or `null` | List only runs carrying this label (run_label sets them); every run if unset. |
| `labels` | array of `object` | Reader verdicts on the recorded memo, each {dimension, direction, quote, note?}: a dimension the template scores, a direction of low\|low_mid\|mid\|mid_high\|high, and the reader's words verbatim. They need recorded_analysis_id; the freeze stamps each with the template's criterion for its dimension. |
| `launch_group_filter` | `string` or `null` | Only the calls one launch's case generation made; its runs carry this id. |
| `launched_run_ids` | array of `string` or `null` | The runs the estimated launch made (the run ids its launch job names, as run_ids), with predicted_cost: each predicted cell then says how many of its observations came from other runs. |
| `limit` | `integer` | The most rows to return, up to 200. |
| `max_concurrent_arms` | `integer` | How many of a sweep's arms run at once. 1 (the default) runs them one after another, so no arm shares another's provider load. To compare latency, also declare measure_latency. |
| `max_cost_usd` | `number` or `null` | A per-run cost cap in dollars, at or below the host's ceiling; it can only lower that ceiling, and a value above it is refused. |
| `measure_latency` | `boolean` | Declare latency under test: each run executes its cases one at a time, with no other run beside it. False runs several cases at once — far faster — and the latency then recorded is marked read under concurrency and never compared. |
| `metric` | `string` or `null` | The measure to read; omitted reads the composite score. |
| `min_absolute_change` | `number` | The smallest move a regression flag counts, in the measure's unit; 0 lets the test decide. |
| `min_relative_change` | `number` | The smallest move from the baseline a flag counts, as a fraction; 0 lets the test decide. |
| `model` | `string` or `null` | The analysis generator's model; omitted for the host's. A model outside the host's allowed writers is refused before anything is spent. |
| `models` | array of `string` | Candidate models, one arm and one run each; empty runs the kind's own default. |
| `n_test_cases` | `integer` or `null` | A case count to price each planned arm at in place of its plan's, for a what-if grid. |
| `n_variations` | `integer` | New cases to generate from the template's variation axes; 0 runs its stored cases. |
| `name` (on `campaign_create`) | `string` | The campaign's name, as an operator reads it. |
| `name` (on `campaign_update`) | `string` or `null` | The campaign's name, as an operator reads it. |
| `note` | `string` | Why the output must get that verdict, in the author's words. |
| `offset` | `integer` | Skip this many rows; the previous page's next_offset. |
| `output` | `string` | The output under review, as the judge reads it. |
| `overlays` | `object` or `null` | The knobs this launch turns on the template's kind, by field. |
| `part` | `'record'` \| `'judge'` \| `'spans'` | Which part of the stored trace to return: record (the output the kind stored, its call ledger and end state, beside the result), judge (what the judge was sent) or spans (the stored spans). |
| `predicted_cost` | `object` or `null` | A cost pivot's plan: the structured result launch_estimate returned before these runs. The cell at each priced arm's model and template then shows its predicted cost beside the cost observed. |
| `purpose_filter` | `'variation'` \| `'proposer'` \| `'analysis'` \| `'judge'` \| `'second_judge'` or `null` | Only calls made for this purpose: variation (a launch's case generation), proposer (a rubric draft), analysis (an analysis generation), judge (a judge's repeat or temperature comparison) or second_judge (a second judge's pass). |
| `rating_reason` | `string` | Why that score, in the rater's own words. |
| `reason` | `string` or `null` | Why, recorded on a cancelled run. |
| `recorded_analysis_id` | `string` or `null` | A stored analysis of the campaign whose memo the case pins as the one it got, as analyses_list names it; omitted freezes the bundle alone, which only a generating candidate can run. |
| `repeats` | `integer` | How many times each dim is judged at each temperature. |
| `reporter_case_id` | `string` | A reporter case's id, as reporter_case_freeze or reporter_cases_list returns it. |
| `result_id` | `string` | A result's id, as results_list names it. |
| `result_ids` | array of `string` or `null` | The run's results to draw from; omitted for every result of the run. |
| `row_factor` | `string` | The coordinate the rows are: a declared one (model, template_id, ...) or a lever the host registers. |
| `rubric_dim` | `string` | The judged dimension rated, spelled as the result's score spells it. |
| `run_id` | `string` | A run's id, as runs_list or a launch's job names it. |
| `run_ids` | array of `string` | The runs, by id, all in the caller's scope: those a campaign is created with, or those added to or taken out of it. |
| `run_labels` | array of `string` | The run's labels, replacing any it has; [] clears them. Each 1 to 64 characters, stored sorted and each once. A label names the run (a baseline, a production snapshot) and moves no key or fingerprint. |
| `run_status` | `'pending'` \| `'running'` \| `'completed'` \| `'failed'` \| `'cancelled'` \| `'budget_stopped'` \| `'exhausted'` or `string` | Read only runs with this status, or 'all'. Completed unless named: a run still going is still adding results. |
| `sample_fraction` | `number` | The share of the run's judged results the second judge is asked about. |
| `sample_seed` | `integer` | The seed the share is drawn with; the same seed draws the same results. |
| `scenario` | `string` or `null` | The scenario the judge reads as what the case asks; omitted reads the criterion template's intent, as its runs' judge does. |
| `score` | `integer` | The score, on the dimension's scale: 1-5, or 1 (pass) / 0 (fail). |
| `second_judge_config_ids` | `object` or `null` | dim -> the versioned judge config whose prompt asks the second judge for that dim; omitted for the prompts the run recorded. A dim absent from a map given is asked with the built-in prompt. |
| `second_judge_model` | `string` | The model the second judge sends every judged dimension to. |
| `second_judge_temperature` | `number` or `null` | The temperature every second-judge call is requested at; omitted for what each prompt asks. |
| `selection` | `'borderline'` \| `'all'` | Which scored dims are re-judged: borderline (stored score inside the scale, or a recorded repeat or second judge disagreed) or all. |
| `simulator_model` | `string` or `null` | The simulated user's model, where the kind has one. |
| `status` | `'pending'` \| `'running'` \| `'completed'` \| `'failed'` \| `'cancelled'` \| `'budget_stopped'` \| `'exhausted'` or `null` | List only runs with this stored status. |
| `subject` | `string` or `null` | Who or what produced the output, as the judge should know it; omitted for none. |
| `subject_filter` | `string` or `null` | Read only this subject's runs or insights; omitted reads every subject. |
| `subject_id` | `string` | The subject the runs measure, as the host names it. |
| `supersedes` | array of `string` | The live case(s) of this campaign and memo the freeze replaces, by id — needed to revise a case's labels or re-freeze moved evidence; every live case of the pair when it holds several. |
| `template_filter` | `string` or `null` | Only calls made for this template, by id. |
| `template_id` (on `sweep_launch`, `case_set_mint`, `run_launch`, `launch_estimate`, `reporter_case_freeze`, `reporter_cases_list`, `judge_cases_freeze`, `judge_case_author`) | `string` | A template's id, as templates_list names it. |
| `template_id` (on `campaign_create`) | `string` or `null` | A template's id, as templates_list names it. |
| `template_name` | `string` | The judge template's name, unique in the scope. |
| `test_case_ids` | array of `string` | The cases, in order, each a stored case of the template; a list the latest version already holds is refused. |
| `tracked` | `boolean` | Whether the set is a standing suite followed over time, or made for one launch. |
| `variation_model` | `string` or `null` | The model that writes the template's llm variation axes' values; required when n_variations generates for such an axis, refused otherwise. |
| `weighting` | `string` or `null` | How a cell averages its observations; omitted takes equal per scenario. |

<a id="cli"></a>
## The command line

`python -m threetears.evals <command> --host MODULE:FACTORY --scope SCOPE [options]`

A product mounting the commands under its own CLI (`run_cli(host_factory=...)`) drops `--host`. Every command takes `--scope`.

`python -m threetears.evals FILE.py|FILE.toml [--verbose]` runs every eval the file declares instead: each prints its result, writes its report to `.evals/reports/<name>/` and exits with the codes below.

Exit codes, in full: 0 done; 1 a launched run did not complete; 2 refused; 3 failed with an unanticipated error; 4 the gate failed.

Each command's options, as its `--help` prints them:

### `run`

```text
usage: python -m threetears.evals run [-h] --host MODULE:FACTORY --scope SCOPE --template TEMPLATE
                                      --subject SUBJECT [--model MODEL] [--k K]
                                      [--max-cost-usd DOLLARS] [--judge-model MODEL]
                                      [--simulator-model MODEL] [--n-variations N]
                                      [--variation-model MODEL] [--apparatus-settings JSON]
                                      [--measure-latency]

Launch a template's runs, wait for them, and print each run's summary.

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --template TEMPLATE   the template to run, by id
  --subject SUBJECT     the subject the runs measure, as the host names it
  --model MODEL         a candidate model; repeat for one arm each. Omitted, the kind runs one arm
                        on its own default model, and a kind with no default refuses the launch
  --k K                 repeats per case (default 3)
  --max-cost-usd DOLLARS
                        a per-run cost cap in dollars, at or below the host's ceiling (as
                        run_launch's max_cost_usd)
  --judge-model MODEL   the judge model, where the kind is model-judged (as run_launch's
                        judge_model)
  --simulator-model MODEL
                        the simulated user's model, where the kind has one (as run_launch's
                        simulator_model)
  --n-variations N      generate N new cases from the template's variation axes first (as
                        run_launch's n_variations)
  --variation-model MODEL
                        the model that writes the template's llm axes' values when generating (as
                        run_launch's variation_model)
  --apparatus-settings JSON
                        host-declared apparatus values to set the runs' rig up with, as a JSON
                        object keyed by apparatus dimension, e.g. '{"adjudicator_seat":
                        "model:m"}' (as run_launch's apparatus_settings)
  --measure-latency     latency is under test: run each run's cells one at a time (as run_launch's
                        measure_latency)
```

### `ls`

```text
usage: python -m threetears.evals ls [-h] --host MODULE:FACTORY --scope SCOPE

List the scope's templates, runs and campaigns.

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
```

### `report`

```text
usage: python -m threetears.evals report [-h] --host MODULE:FACTORY --scope SCOPE
                                         [--format {markdown,json,html}] [--out DIR]
                                         campaign

Print a campaign's report — its analysis, else its evidence alone — without calling a model.

positional arguments:
  campaign              the campaign, by id

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --format {markdown,json,html}
                        markdown (default), html (one page, needs no script) or json (the bundle
                        the report was computed from)
  --out DIR             write report.html, report.md and bundle.json into DIR instead of printing
                        one form
```

### `bundle`

```text
usage: python -m threetears.evals bundle [-h] --host MODULE:FACTORY --scope SCOPE campaign

Print a campaign's analysis bundle as JSON — what a generation would read.

positional arguments:
  campaign              the campaign, by id

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
```

### `spend`

```text
usage: python -m threetears.evals spend [-h] --host MODULE:FACTORY --scope SCOPE
                                        [--purpose {variation,proposer,analysis,judge,second_judge}]
                                        [--launch-group ID] [--template ID]

Print what the engine spent outside any run — case generations, rubric proposals, analyses.

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --purpose {variation,proposer,analysis,judge,second_judge}
                        only this purpose's calls
  --launch-group ID     only one launch's case generation
  --template ID         only calls made for this template
```

### `gate`

```text
usage: python -m threetears.evals gate [-h] --host MODULE:FACTORY --scope SCOPE
                                       [--fail-on OUTCOMES] [--reading MEASURE]
                                       campaign

Exit 4 when the campaign's verdicts include an outcome --fail-on names — the CI gate. No model is
called.

positional arguments:
  campaign              the campaign, by id

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --fail-on OUTCOMES    comma-separated outcomes that fail the gate, of regressed, not-separated,
                        untested, breached, undecided-guardrail, missed, undecided-bar (default
                        regressed,breached,undecided-guardrail)
  --reading MEASURE     gate only this reading, by key or as the report heads it; repeat for more
                        (default: every reading)
```

### `frontier`

```text
usage: python -m threetears.evals frontier [-h] --host MODULE:FACTORY --scope SCOPE [--bar BAR]
                                           [--subject SUBJECT] [--json]

Rank each subject's variants on quality, cost and latency, and the cheapest that clears a bar.

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --bar BAR             the pass^k a variant must clear, from 0 to 1
  --subject SUBJECT     only this subject's variants
  --json                print the frontier as JSON
```

### `judge-temperature`

```text
usage: python -m threetears.evals judge-temperature [-h] --host MODULE:FACTORY --scope SCOPE
                                                    (--max-cost-usd DOLLARS | --no-cap)
                                                    [--repeats N] [--all] [--result ID]
                                                    [--estimate] [--json]
                                                    RUN

Re-judge a run's borderline cases at the pinned judge temperature and at the provider default, and
compare the spread of their scores. Spends, under the out-of-run cap it names.

positional arguments:
  RUN                   the finished run, by id

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --max-cost-usd DOLLARS
                        the most the comparison's calls may be priced at together; every call is
                        priced before the first
  --no-cap              enforce no out-of-run cap (calls are still ledgered)
  --repeats N           calls per dim at each temperature (default 5, at least 2)
  --all                 re-judge every scored dim, not only the borderline ones
  --result ID           only this result of the run; repeat for more (default: every result)
  --estimate            price it against the cap and call nothing
  --json                print the comparison (or estimate) as JSON
```

<a id="measures"></a>
## Measures

The engine's own measures (`METRIC_DESCRIPTORS`), grouped by family in the order the engine declares them. Direction is which end is better (`—` for a coordinate, condition or raw count); scale says whether a relative change means anything (`ratio`) or only a difference does (`interval`), `—` when undeclared. A host declares its own measures on its `MeasureRegistry`, and a template mints one per rubric dimension and goal check; neither is listed here.

### mechanical

| Measure | Direction | Scale | Unit | Description |
|---|---|---|---|---|
| `total_ms` | lower | — | ms | Wall-clock across the turn roots. |
| `llm_ms` | lower | — | ms | Time inside model calls — the model-attributable share of latency. |
| `tool_ms` | lower | — | ms | Time inside tool executions. |
| `orchestration_ms` | lower | — | ms | Turn wall-clock spent neither inside a model call nor inside a tool execution — assembling perceptions, building the prompt, parsing the decision, updating and persisting state. |
| `async_wait_ms` | lower | — | ms | Wall-clock the runner spent waiting on in-flight background tool work before the candidate could deliver it. |
| `judge_ms` | lower | — | ms | Wall-clock the judge phase took to score one result — every axis and dimension, including retries. |
| `cost_usd` | lower | — | usd | Blended program spend for the result. |
| `production_replicating_cost` | lower | — | usd | Observed dollars belonging to the production roles — measurement-only roles excluded. |
| `program_cost` | lower | — | usd | What the eval program spent to measure the candidate, including judge and simulator. |
| `prompt_tokens` | — | — | tokens | Input tokens a role consumed. |
| `completion_tokens` | — | — | tokens | Output tokens a role produced. |
| `reasoning_tokens` | — | — | tokens | The reasoning share of completion_tokens. |
| `call_count` | — | — | calls | How many calls a role made — and it counts a different thing per role: LLM rounds for the candidate, attempts including parse retries for the judge, searches for external. |
| `provider_units` | — | — | provider-defined | Units of a provider's own metered quantity that an external role consumed — a search API's credits, a video API's quota units — named by the row's provider_unit. |
| `context_tokens_in` | — | — | tokens | How much context the candidate was carrying — a stratification covariate, not a quality measure. |
| `reasoning_ratio` | — | — | — | How much of the candidate's output was reasoning. |
| `candidate_output_tokens_per_s` | — (diagnostic) | — | tokens/s | How fast the candidate's provider produced output, over the time spent inside its model calls. |
| `dropped_tool_calls` | — | — | calls | How many times the candidate emitted a tool call the client dropped before dispatch — a name that did not parse, so nothing ran and the model was not told. |
| `refused_tool_attaches` | — | — | requests | How many times the candidate asked to attach a tool outside the run's tools_allowed and the harness refused. |
| `truncated_rounds` | lower | — | rounds | How many of the candidate's LLM rounds the provider cut off at its output cap (its output reached the cap, or the provider reported finish_reason=length) — the turn's silence or half-answer is the cap's, not the model's decision, so every quality measure on the cell is confounded by it; a reasoning model that spends the cap thinking shows here with reasoning_ratio near 1. |
| `turns_ended_by_budget` | lower | — | turns | How many of the candidate's turns outlived the host's per-turn budget — the bound the host runs every turn under in production — and were ended by it. |
| `execution_mode` | — | — | — | Whether the observation was made while other eval work was executing — another run holding a concurrency slot (never one queued for it), or other cells of its own run, which executes its cells concurrently when its launch did not declare latency under test. |
| `n_results` | — | — | results | How many results the group aggregates — the denominator behind its means. |
| `n_total_ms` | — | — | results | How many results contributed a measured total_ms — the denominator behind mean/median/p95/max total_ms, which can be smaller than n_results because latency components are nulled independently. |
| `n_llm_ms` | — | — | results | How many results contributed a measured llm_ms — the denominator behind mean_llm_ms. |
| `n_tool_ms` | — | — | results | How many results contributed a measured tool_ms — the denominator behind mean_tool_ms. |
| `n_cost_usd` | — | — | results | How many results' spend was priced — the denominator behind mean_cost_usd. |
| `n_prod_cost_usd` | — | — | results | How many results measured a production-replicating cost — the denominator behind mean_prod_cost_usd. |
| `n_test_cases` | — | — | cases | How many test cases the model was run against. |
| `k` | — | — | iterations | The depth the headline pass_hat_k is read at. |
| `n_cannot_tell_excluded` | — | — | iterations | Iterations left out of pass^k because the judge answered it could not score a rubric dimension from the evidence. |
| `n_no_criterion_excluded` | — | — | iterations | Iterations left out of pass^k because they carried nothing for it to conjoin: no goal-state check and no judge, as a classifier scored only against its expected label. |
| `pass_hat_k_unmeasured_reason` | — | — | — | Why pass_hat_k is null when the reason is that no iteration carried a pass criterion; null otherwise. |
| `n_cases_at_k` | — | — | cases | The cases pass_hat_k averages over: those scored at least k times. |
| `mean_total_ms` | lower | — | ms | Average end-to-end wall-clock over the cells that MEASURE the candidate — a cell an apparatus fault produced is excluded, since a clock stopped by a cassette miss or a judge error times the harness. |
| `median_total_ms` | lower | — | ms | Typical end-to-end wall-clock, less sensitive to one slow outlier than the mean. |
| `p95_total_ms` | lower | — | ms | Tail end-to-end wall-clock: as likely above the true 95th percentile as below it. |
| `max_total_ms` | lower | — | ms | The slowest measured turn: the worst case seen, never a percentile. |
| `mean_llm_ms` | lower | — | ms | Average model-attributable latency — the share a model swap can move. |
| `mean_tool_ms` | lower | — | ms | Average time inside tool executions. |
| `total_cost_usd` | lower | — | usd | What the group's priced results cost in total: each result's cost_usd is a sum over the roles its cost_roles names. |
| `mean_cost_usd` | lower | — | usd | Average program spend per PRICED result (blended, incl. judge + simulator). |
| `total_prod_cost_usd` | lower | — | usd | Observed spend on the production roles — candidate + inner_agent + external, excluding the judge/simulator measurement scaffolding. |
| `mean_prod_cost_usd` | lower | — | usd | Average production-replicating spend per MEASURED result — the reporting default (what a config costs to run). |
| `n` | — | — | scores | How many judged scores a dimension row aggregates. |
| `async_deliveries` | — | — | deliveries | How many async deliveries the group made that describe a real run of the tool — not every delivery that happened, and the denominator any rate over them is taken against. |
| `async_deliveries_substituted` | — | — | deliveries | Deliveries EXCLUDED from async_deliveries because a harness supplied their payload (a seeded payload or a replayed capture). |
| `async_delivery_mean_elapsed_ms` | lower | — | ms | Average async delivery duration. |
| `async_delivery_median_elapsed_ms` | lower | — | ms | Typical async delivery duration. |
| `async_delivery_p95_elapsed_ms` | lower | — | ms | Tail async delivery duration. |
| `async_delivery_elapsed_n` | — | — | deliveries | How many non-substituted deliveries had a measured duration — the denominator behind the mean, median and p95. |
| `paired` | — | — | — | Whether the significance test paired the two runs' composites by test case. |
| `n_pairs` | — | — | cases | How many test cases were scored in BOTH runs — the paired test's sample size. |
| `n_cases` | — | — | cases | How many test cases contributed a composite — the pairing atom behind significance. |
| `status` | — | — | — | Where one piece of background work stood when the cell ended: delivered, failed with an error, or acknowledged and never delivered before the cell ended. |
| `acknowledged_turn` | — | — | turn index | The 0-based turn whose call a background tool acknowledged. |
| `delivered_turn` | — | — | turn index | The 0-based turn a background tool's payload reached the conversation on. |
| `elapsed_ms` | lower | — | ms | Wall-clock of one piece of background work, acknowledgement to delivery or failure. |
| `delivered_items` | higher | — | items | How many items one async delivery's payload carried. |
| `substituted` | — | — | — | Whether a harness supplied this delivery's payload instead of a background run producing it (a seeded payload or a replayed capture). |
| `count_a` | — | — | cases | How many cases contributed a composite in run A — the sample size behind its score. |
| `count_b` | — | — | cases | How many cases contributed a composite in run B. |
| `comparison_basis` | — | — | — | Whether the two runs were compared over the templates they share or taken as independent — the caveat that decides how much the deltas mean. |

### rubric

| Measure | Direction | Scale | Unit | Description |
|---|---|---|---|---|
| `score` | higher | — | — | What one judged score on ONE rubric dimension of ONE result counts as, on the dimension's own scale (ordinal: scored 1-5 against the template's scoring guide; pass_fail: answered pass (1) or fail (0); its mean is the pass rate): the judge's score, the scale's floor where the candidate failed (a turn the output cap ended included), or null where the harness faulted the cell — the per-observation measure mean_score / min_score / max_score aggregate, emitted per judged dimension by export_results and accepted as a pivot metric. |
| `mean_score` | higher | — | — | Average judged score for a single dimension — on its raw scale, NOT the 0-1 composite scale; on a pass/fail dimension it is the pass rate. |
| `min_score` | higher | — | — | Worst judged score observed for one rubric dimension, on the dimension's own scale (ordinal: scored 1-5 against the template's scoring guide; pass_fail: answered pass (1) or fail (0); its mean is the pass rate). |
| `max_score` | higher | — | — | Best judged score observed for one rubric dimension, on the dimension's own scale (ordinal: scored 1-5 against the template's scoring guide; pass_fail: answered pass (1) or fail (0); its mean is the pass rate). |

### composite

| Measure | Direction | Scale | Unit | Description |
|---|---|---|---|---|
| `hedges_g` | — | — | — | Effect size of a composite difference between two runs: Hedges' g, Cohen's d with its small-sample upward bias removed. |
| `significant` | — | — | — | Whether the composite difference cleared p &lt; 0.05. |
| `p` | — | — | — | The p-value the significance verdict was thresholded against. |
| `pass_hat_k` | higher | — | — | pass^k (τ-bench): the chance that k attempts at a case ALL pass — never pass@k, the chance that at least one does. |
| `mean_composite` | higher | — | — | Average judged quality, threshold-free — a regression often shows here before cases start failing pass^k. |
| `pass_hat_k_a` | higher | — | — | Reliability (pass^k at the shared depth k, each 1-5 criterion at or above the compare's recorded rubric_threshold of 5) for run A. |
| `pass_hat_k_b` | higher | — | — | Reliability (pass^k at the shared depth k, each 1-5 criterion at or above the compare's recorded rubric_threshold of 5) for run B. |
| `pass_hat_k_delta` | higher | — | — | Change in reliability (pass^k at the shared depth k, each 1-5 criterion at or above the compare's recorded rubric_threshold of 5) from run A to run B. |
| `composite_a` | higher | — | — | Mean composite quality for run A. |
| `composite_b` | higher | — | — | Mean composite quality for run B. |
| `composite_delta` | higher | — | — | Change in mean composite quality from run A to run B. |

### dual_axis

| Measure | Direction | Scale | Unit | Description |
|---|---|---|---|---|
| `__transcript__` | higher | interval | — | Decision quality given the context the candidate actually had. |
| `__outcome__` | higher | interval | — | Whether the final state satisfied the scenario's intent. |
| `mean_transcript_score` | higher | interval | — | Average decision quality given the context the candidate actually had — the aggregate an aggregating surface reports over __transcript__ rows, on the raw 1-5 scale and NOT the 0-1 composite scale. |
| `mean_outcome_score` | higher | interval | — | Average intent satisfaction — the aggregate an aggregating surface reports over __outcome__ rows, on the raw 1-5 scale. |

### goal_state

| Measure | Direction | Scale | Unit | Description |
|---|---|---|---|---|
| `goal_state` | higher | — | — | Whether one goal-state check passed on one observation — 1 or 0, keyed by the check it reports (goal_check). |
| `goal_state_pass_rate` | higher | — | — | How often a goal-state check passed — the aggregate an aggregating surface reports over goal_state rows, one check at a time (key the rows on goal_check). |

### classifier

| Measure | Direction | Scale | Unit | Description |
|---|---|---|---|---|
| `accuracy` | higher | — | — | Share of classifications matching the expected label — the classifier family's quality measure, derived by the engine from each observation's `match` (a host lands `match` and never this). |
| `precision` | higher | — | — | Per class: of the times this label was predicted, how often it was right. |
| `recall` | higher | — | — | Per class: of the times this label was correct, how often it was predicted. |
| `f1` | higher | — | — | Harmonic mean of precision and recall for a class. |
| `match` | higher | — | — | Whether one classification matched its expected label. |
| `confusion_cell` | — | — | — | Which cell of the confusion matrix one classification landed in. |
