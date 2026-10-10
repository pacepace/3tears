# Reference

**For:** people who already know the engine and agents that need exact names. It answers: what does this root export, what does this setting or field mean, what can a goal check say, which actions and commands exist, and what does each measure measure. To learn the engine, start with the [tutorial](tutorial.md) instead.

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

Import only from these roots, and only the names below: a module under a root is internal and may move. Within a root, names are grouped as functions, classes, types and constants, each sorted by name. A class's kind says what it is: a `model` is a Pydantic model, a `protocol` is something a host implements.

- [`threetears.evals.contracts`](#api-contracts)
- [`threetears.evals.contracts.host`](#api-contracts-host)
- [`threetears.evals.run`](#api-run)
- [`threetears.evals.analysis`](#api-analysis)
- [`threetears.evals.analysis.viz`](#api-analysis-viz)
- [`threetears.evals.gen`](#api-gen)
- [`threetears.evals.storage`](#api-storage)
- [`threetears.evals.testing`](#api-testing)
- [`threetears.evals.quick`](#api-quick)
- [`threetears.evals.ops`](#api-ops)
- [`threetears.evals.actions`](#api-actions)
- [`threetears.evals.transports.fastmcp`](#api-transports-fastmcp)
- [`threetears.evals.vega`](#api-vega)

<a id="api-contracts"></a>
### `threetears.evals.contracts`

The engine's contracts: the stored shapes, and the vocabulary every other package speaks.

**Functions**

- **`agreement_statistic`** · function · The one agreement figure both tiers are held to: weighted kappa on 1-5, kappa on pass/fail.
  <br>`agreement_statistic(scale: RubricScale, kappa: float | None, weighted_kappa: float | None) -> float | None`
- **`attribution_state`** · function · Collapse a run's attribution pair into the one state every consumer reads.
  <br>`attribution_state(effective_judges: dict[str, str] | None, source: str | None) -> JudgeAttributionState`
- **`blended_cost_roles`** · function · The roles a run's blended `EvalResult.cost_usd` sums, in stored order.
  <br>`blended_cost_roles(rate_table: ExternalRateTable | None) -> tuple[UsageRole, ...]`
- **`calibration_criterion`** · function · The calibration criterion over `n` judge–human pairs covering `results` results, at `agreement`.
  <br>`calibration_criterion(n: int, results: int, agreement: float | None, interval: tuple[float, float] | None = None) -> TierCriterion`
- **`candidate_failure_cause`** · function · Name why `result` is a candidate failure, or `None` when it is not one.
  <br>`candidate_failure_cause(result: EvalResult) -> CandidateFailureCause | None`
- **`canonical_digest`** · function · Return the full sha256 hex digest of `payload`'s canonical JSON.
  <br>`canonical_digest(payload: Any) -> str`
- **`classifier_label_measure`** · function · The measure name one label's precision, recall or F1 is reported under.
  <br>`classifier_label_measure(statistic: ClassifierStatistic, label: str) -> str`
- **`classifier_label_of`** · function · The `(statistic, label)` a name was minted for by `classifier_label_measure`, or None.
  <br>`classifier_label_of(name: str) -> tuple[ClassifierStatistic, str] | None`
- **`classify_result`** · function · Categorize a result for scoring by its error fields and its candidate's delivery.
  <br>`classify_result(result: EvalResult) -> ResultOutcome`
- **`confusion_cell`** · function · The value a classification reports under `confusion_cell`: `expected → predicted`.
  <br>`confusion_cell(expected: str, predicted: str) -> str`
- **`confusion_of`** · function · The `(expected, predicted)` labels of a value `confusion_cell` made, or None for anything else.
  <br>`confusion_of(cell: str) -> tuple[str, str] | None`
- **`count_substituted`** · function · Count the deliveries a harness supplied, over the delivery record itself.
  <br>`count_substituted(deliveries: Sequence[AsyncDelivery] | None) -> int`
- **`count_substituted_deliveries`** · function · Count the async deliveries in `result` whose payload a harness supplied.
  <br>`count_substituted_deliveries(result: EvalResult) -> int`
- **`counted_goal_verdicts`** · function · Each goal-state check on `result` paired with the verdict every rate counts for it.
  <br>`counted_goal_verdicts(result: EvalResult) -> list[tuple[GoalStateOutcome, bool]] | None`
- **`counted_score`** · function · What one judged score on `result` counts as in every measure — a rubric dim or a reserved axis.
  <br>`counted_score(result: EvalResult, score: RubricScore) -> int | None`
- **`criterion_state`** · function · How one criterion reads, decided on the agreement's interval and never on its point estimate.
  <br>`criterion_state(results: int, agreement: float | None, interval: tuple[float, float] | None, *, threshold: float, min_results: int) -> CriterionState`
- **`delivered_a_turn`** · function · Whether `result` is a turn the candidate took — the one population a cost or latency reading is over.
  <br>`delivered_a_turn(result: EvalResult) -> bool`
- **`derive_variant_identity`** · function · Derive a run's variant key through the host's profile, writing nothing.
  <br>`derive_variant_identity(*, run: EvalRun, profile: HostProfile) -> DerivedVariantIdentity`
- **`describe_and_log_failure`** · function · Describe a failed call (`describe_failure`) and log it (`log_provider_failure`).
  <br>`describe_and_log_failure(describer: ProviderFailureDescriber, exc: BaseException, *, logger: logging.Logger, where: str, message: str, args: tuple[object, ...] = (), level: int = 40) -> ProviderFailure`
- **`eval_trace_doc_id`** · function · Doc id for a result's `EvalTrace` sibling.
  <br>`eval_trace_doc_id(result_id: str) -> str`
- **`existing_axis_values`** · function · The values a template's stored cases already give `axis` — what its generation call asks the model to avoid.
  <br>`existing_axis_values(axis: VariationAxis, existing: Sequence[EvalTestCase]) -> set[str]`
- **`extract_json`** · function · Extract a JSON object from provider output.
  <br>`extract_json(content: str) -> dict[str, Any]`
- **`extract_json_array`** · function · Extract a JSON array from provider output, handling markdown code blocks.
  <br>`extract_json_array(text: str) -> list[dict[str, Any]]`
- **`extract_paths`** · function · Parse `expression` and return the paths it reads, without evaluating it.
  <br>`extract_paths(expression: str) -> ExtractedPaths`
- **`fold_phase_timings`** · function · Fold one delivery's carried phase timings into a result's accumulator.
  <br>`fold_phase_timings(accumulator: dict[str, float], *, source_tool: str, timings: Any) -> None`
- **`goal_check_of`** · function · The check a measure name was minted for by `goal_check_measure`, or None.
  <br>`goal_check_of(name: str) -> str | None`
- **`judges_sharing_a_candidate_model`** · function · The dims whose judge is one of the run's candidate models: a model grading its own output.
  <br>`judges_sharing_a_candidate_model(effective_judges: dict[str, str] | None, candidate_models: Sequence[str]) -> dict[str, str]`
- **`keep_fields`** · function · Return `document` reduced to the top-level `fields` it has — the meaning of `keep`.
  <br>`keep_fields(document: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]`
- **`list_metrics`** · function · List every measure a host can describe, optionally filtered by family and/or scope.
  <br>`list_metrics(measures: MeasureRegistry, family: MetricFamily | None = None, attribution_scope: AttributionScope | None = None) -> list[MetricDescriptor]`
- **`materiality`** · function · Whether a difference of `delta` in a measure is large enough to act on.
  <br>`materiality(threshold: float | None, delta: float) -> Materiality`
- **`omit_paths`** · function · Return `document` without each dotted path in `paths` — the meaning of `exclude`.
  <br>`omit_paths(document: dict[str, Any], paths: Sequence[str]) -> dict[str, Any]`
- **`percentile`** · function · Nearest-rank percentile of an already-sorted, non-empty list.
  <br>`percentile(sorted_values: list[float], pct: float) -> float`
- **`plan_variation_calls`** · function · The one call each of `template`'s `llm` axes makes to generate `n_variations` values, built before any is made.
  <br>`plan_variation_calls(template: EvalTemplate, n_variations: int, existing: Sequence[EvalTestCase]) -> dict[str, PlannedCall]`
- **`production_replicating_cost`** · function · Observed cost of the roles a production deployment would also pay for.
  <br>`production_replicating_cost(usage: list[RoleUsage], *, substituted_deliveries: int) -> float | None`
- **`program_cost`** · function · Cost of every role, i.e. what the eval program spent to produce this result.
  <br>`program_cost(usage: list[RoleUsage]) -> float | None`
- **`referenced_actions`** · function · Every `(tool, action)` a goal check names through a call builtin, in source order, each once.
  <br>`referenced_actions(expression: str) -> tuple[tuple[str, str], ...]`
- **`refuse_an_undeclarable_design`** · function · Refuse a declaration this host cannot honour — at authoring time, not at analysis time.
  <br>`refuse_an_undeclarable_design(design: CampaignDesign, *, behavior: str, template: EvalTemplate | None, profile: HostProfile) -> None`
- **`resolve_bar_name`** · function · Say what a bar names and where a result carries it, or why no result can.
  <br>`resolve_bar_name(name: str, *, rubric_dimensions: Mapping[str, RubricScale], goal_state_checks: Collection[str], measures: MeasureRegistry) -> BarName | UnreadableBarName`
- **`resolve_context_identity`** · function · Return the identity a read surface should show for `run`.
  <br>`resolve_context_identity(run: EvalRun, profile: HostProfile) -> DerivedContextIdentity`
- **`resolve_result_condition`** · function · Return the condition `result` is in, on every per-result axis.
  <br>`resolve_result_condition(result: EvalResult) -> ResultCondition`
- **`resolve_result_usage`** · function · Return the per-role usage a read surface should show for `result`.
  <br>`resolve_result_usage(result: EvalResult) -> ResolvedUsage`
- **`resolve_variant_identity`** · function · Return the variant identity a reader should show for `run`'s observations.
  <br>`resolve_variant_identity(*, run: EvalRun, profile: HostProfile) -> DerivedVariantIdentity`
- **`save_document`** · function · Upsert `document` through `repo`, raising on any failure.
  <br>`save_document(repo: DocumentStore, document: dict[str, Any], *, if_match: str | None = None) -> None`
- **`separation_criterion`** · function · The separation criterion over `n` first-score/repeat pairs covering `results` results, at `agreement`.
  <br>`separation_criterion(n: int, results: int, agreement: float | None, interval: tuple[float, float] | None = None) -> TierCriterion`
- **`summarize_completeness`** · function · Count what a finished run loop delivered against the matrix the run promised.
  <br>`summarize_completeness(run: EvalRun, cells: Sequence[CellSummary]) -> RunCompleteness`
- **`tier_of`** · function · The tier two criteria decide.
  <br>`tier_of(calibration: TierCriterion, separation: TierCriterion) -> JudgedEvidenceTier`
- **`utc_now_iso`** · function · Return the current UTC time in ISO-8601 format.
  <br>`utc_now_iso() -> str`
- **`weakest_judged_tier`** · function · What several judged readings can bear together: the weakest of them (see `JUDGED_TIERS_WEAKEST_FIRST`).
  <br>`weakest_judged_tier(tiers: list[JudgedEvidenceTier]) -> JudgedEvidenceTier`
- **`withhold_failure_detail`** · function · Describe a failure by its class and nothing else — the safe default.
  <br>`withhold_failure_detail(exc: BaseException) -> ProviderFailure`

**Classes**

- **`ActionSeam`** · protocol · A kind's synchronous tools, as the cassette lane records and replays them.
- **`ActorPolicy`** · model · One simulated actor in a template's conversation.
- **`AdmissionRefusedError`** · exception · A launch refused because it would admit more unfinished runs than the process may hold (status 429).
- **`AdmittedCall`** · dataclass · A planned call its budget admitted, at the ceiling it was priced at.
- **`ArmGuardrails`** · class · Where one arm stands on every guardrail checked against the control, under any rig.
- **`AsyncDelivery`** · model · One piece of background work the candidate started: acknowledged at once, delivered later.
- **`AsyncExternalSpend`** · model · Paid non-LLM calls one piece of background work made at one provider, as the work reports them.
- **`AuthoredAnalysis`** · model · A decision memo over one campaign's evidence.
- **`BarAdjudication`** · model · One bar the campaign is held to, adjudicated against every cell.
- **`BarName`** · dataclass · A bar name a result can carry: which kind it is, and the descriptor that says which way it runs.
- **`BarOverride`** · model · A standard this campaign holds itself to, tighter than the registered one.
- **`BarVerdict`** · model · Whether one cell cleared one bar — computed here, never by the reader.
- **`BoundCompletionClient`** · protocol · A completion client built for one model, which it names.
- **`CalibrationRating`** · model · A rater's score for one judged dimension of one result — a person's, or an agent's.
- **`CallLedger`** · model · The calls one cell's candidate made that succeeded, across every tool, in recorded order.
- **`CallUsage`** · dataclass · One LLM call's observed usage, for roles whose client result doesn't survive to the runner.
- **`CampaignDesign`** · model · The declaration: what this campaign set out to learn, before it learned anything.
- **`CampaignView`** · model · A campaign plus its read-time-derived window — the `campaign_get` payload.
- **`CampaignWindow`** · model · The [start, end] time span a campaign's runs cover — DERIVED, never stored.
- **`CandidateKind`** · protocol · The two operations that vary between one evaluable subject and another.
- **`CandidateKindDefect`** · exception · A kind handed back output that contradicts its own `CandidateKind.judged_artifact`.
- **`CandidateOutput`** · model · What one candidate produced, as everything below the dispatch sees it.
- **`CandidatePreparationFailed`** · exception · A candidate could not be built, and the cell is cleanly excluded rather than scored.
- **`CandidateTelemetry`** · model · What one candidate's execution cost, and the windows nothing else timed.
- **`CassetteCorrupt`** · exception · The corpus cannot be used as recorded.
- **`CassetteExhausted`** · exception · A replay asked for something one more time than its capture did.
- **`CassetteKey`** · dataclass · Everything that names one recorded answer — and the one place its document id is composed.
- **`CassetteMiss`** · exception · A replay asked for something its corpus never recorded.
- **`CassetteSeams`** · protocol · What a kind hands `CellCassettes.wire`: the seams its candidate exposes.
- **`CassetteStore`** · protocol · The recordings a capture run made and a replay run is served.
- **`CatalogRubricDim`** · model · A reusable rubric dimension in the shared catalog.
- **`Caveat`** · model · What qualifies a finding, and which class of qualification it is.
- **`CellCassettes`** · protocol · One cell's handle on its run's cassette lane, as the kind driving that cell sees it.
- **`CellFacts`** · model · Everything measured in one cell — one arm, under one rig.
- **`CellSink`** · protocol · One cell's record of where it stands, as the kind driving that cell reports into it.
- **`CellSpanWindow`** · protocol · One cell's two tracing windows, as the kind driving that cell sees them.
- **`CellSummary`** · dataclass · What a run retains per cell after the full result is durably persisted.
- **`Chart`** · model · A chart code draws from the named cells and measures; each type reads its lists by position.
- **`ClientRequestSettings`** · model · The request parameters a host applied to one apparatus role's LLM client, for one run.
- **`CompletionClient`** · protocol · The completion port eval is constructed with.
- **`CompletionGenerator`** · protocol · The one call a consumer of a completion client makes, without the client's lifecycle.
- **`CompletionResult`** · protocol · What eval reads off one completion, whatever produced it.
- **`ConflictError`** · exception · State conflict (status 409).
- **`ContextComponents`** · model · The separately-recorded pieces a run's `context_key` is composed from.
- **`ControlDeclaration`** · model · What held still, stated — because an absent control is a fact, not a null.
- **`ControlEndState`** · model · An end state a template's author states, to prove its goal checks can tell outcomes apart.
- **`ConversationSpec`** · model · The simulated side of a conversing candidate: who talks to it, in what order, for how long.
- **`ConversationStopCause`** · enum · Why a conversation trial's turn loop stopped — one structural signal, never a reading of prose.
  <br>values: `'max_turns'`, `'user_done'`, `'participants_ended'`, `'simulator_error'`, `'apparatus_error'`, `'candidate_error'`, `'budget_stopped'`
- **`CoverageLens`** · model · The coverage lens: per-lever measurement summaries.
- **`Decision`** · model · A proposal and the verdict on it.
- **`DecisionSurface`** · model · The campaign's measured cells and the bars they were held to — frozen at generation.
- **`DefinitionStore`** · protocol · What runs are launched FROM: templates, their test cases, rubric dimensions and judge configs.
- **`DeliveryRecorder`** · protocol · Captures one asynchronous tool's background work for one cell.
- **`DeliveryReplay`** · protocol · Serves one asynchronous tool's recorded work for one cell.
- **`DeliverySeam`** · protocol · One asynchronous tool of a kind, as the cassette lane records and replays its background work.
- **`DeliveryTicket`** · protocol · One piece of background work being captured, from its start to however it ends.
- **`DerivedContextIdentity`** · model · A run's context key as a value, never written back onto the run.
- **`DerivedVariantIdentity`** · model · A computed variant key plus what it was computable from.
- **`DocumentStore`** · protocol · A narrow document store over one `(scope_id, doc_type, id)`-keyed collection.
- **`DSLError`** · exception · Raised when a DSL expression is malformed or disallowed.
- **`EvalAnalysis`** · model · A generated, stored analysis of one campaign — the productized "Analysis" lens.
- **`EvalAnalysisAttempt`** · model · One run of the analysis generator for a campaign, whatever it came to.
- **`EvalBaseModel`** · model · Base model for the eval engine's own Pydantic models.
- **`EvalCampaign`** · model · A curated set of eval runs under one subject×behavior — the analysis hub.
- **`EvalCaseStratum`** · model · What the analysis reads off a test case: which case, and the stratum it declares.
- **`EvalCassette`** · model · One recorded answer — an action's result, or how a piece of background work ended.
- **`EvalDocumentModel`** · model · The base of every eval model serialized as a document — stored, or served to a reader.
- **`EvalInsight`** · model · A durable, subject-scoped insight extracted from an analysis.
- **`EvalResult`** · model · One test case x one model x one k-iteration.
- **`EvalRun`** · model · One execution of a template (or explicit test case set) against one candidate model.
- **`EvalRunStamp`** · model · What a reader of a run's place in a campaign reads off it: which run, curated out or not, and when.
- **`EvalServiceError`** · exception · Structured error from the eval service layer.
- **`EvalStorage`** · class · Storage for v1-shape eval documents over one document store.
- **`EvalTemplate`** · model · Abstract scenario blueprint — subject-agnostic, domain-level.
- **`EvalTestCase`** · model · Concrete, immutable inputs generated from an `EvalTemplate`.
- **`EvalTrace`** · model · The candidate's output, what its judge read, and the OTel spans — stored beside a result, not inside it.
- **`EvidenceRef`** · model · One reading at one cell. Code fills the number, its sample size and its spread.
- **`EvidenceRow`** · model · One reading at one cell, and the number code resolved it to, with its basis.
- **`ExternalRateTable`** · dataclass · Operator-declared money per provider unit, resolved once for a whole run.
- **`ExtractedPaths`** · dataclass · Where one expression reads from, grouped by the root it addresses through.
- **`Finding`** · model · One claim the evidence supports, in the author's words.
- **`FindingResolution`** · model · What code filled for one authored finding: the numbers its readings resolved to, and its chart.
- **`Firings`** · dataclass · What fired in a cell, as the goal language reads it: every dimension that fired, and the armed ones.
- **`GenerationProvenance`** · model · How an `EvalAnalysis` was generated — enables reproducible prompt A/B.
- **`GoalCheckControl`** · model · One goal check's intent, and the end state that proves it discriminates.
- **`GoalCheckControls`** · model · Proof, at authoring, that each of a template's goal checks can tell its outcomes apart.
- **`GoalStateOutcome`** · model · One judge-free fact a candidate's execution established, and whether it held.
- **`GuardrailCell`** · model · One side of a guardrail check: a cell, and the per-case values the check read.
- **`GuardrailCheck`** · model · One guardrail, one arm against the control under one rig: held, breached or undecided.
- **`GuardrailReadings`** · model · Every guardrail, decided for each arm against the control — the pillar kept apart from capability.
- **`JobStore`** · protocol · The run document's read-modify-write: what the job manager persists a run's status through.
- **`JudgeConfig`** · model · Versioned judge configuration for one rubric dimension.
- **`JudgeConfigTombstone`** · model · The record that a judge config slot was deleted, so a seed never writes it back.
- **`JudgedArtifact`** · enum · What a judge reads of a kind's output — the kind's declaration, which picks the judged axes.
  <br>values: `'transcript'`, `'document'`, `'unjudged'`
- **`JudgedDimensionFacts`** · model · What one judged dimension IS, frozen beside its scores — `MeasureFacts`' judged sibling.
- **`JudgedReading`** · model · One judged dimension's scores in one cell.
- **`JudgeEvidence`** · model · Everything a judge reads about one cell's candidate, rendered by the kind that ran it.
- **`JudgeEvidenceTier`** · model · The evidence tier of one judge's readings on one dimension, and the two measurements that decided it.
- **`JudgeRepeat`** · model · One repeat of a result's judge scores: the same judge asked the same question again, recorded beside them.
- **`JudgeRescore`** · model · One re-judge of a result's failed judge dimensions, recorded on the result it changed.
- **`LatencyMetrics`** · model · Per-result latency decomposition: harvested OTel spans, plus what they miss.
- **`LeverCoordinateError`** · exception · A host's per-observation lever map disagrees with its own registry, in either direction.
- **`LeverCoverage`** · model · Per-lever coverage summary — a point estimate is invalid without n + dispersion.
- **`MeasureCollection`** · model · Every measure a set of results carries, with the scopes that carry none named.
- **`MeasureFacts`** · model · What one measure IS, frozen beside its values — the catalogue entry a reader needs to read them.
- **`MeasureFamily`** · model · One family of measures and who produces its numbers — the engine's six, or one a host declares.
- **`MeasureRef`** · model · A measure or judged dimension, named in its namespace — which one is stated, never inferred.
- **`MeasureSummary`** · model · One measure's distribution across a set of results.
- **`MetricDescriptor`** · model · What a single measure is, independent of any particular value of it.
- **`NextStep`** · model · An experiment worth running next.
- **`NonTerminalRunScan`** · class · What one scope's scan for non-terminal runs found.
- **`NotFoundError`** · exception · Resource not found (status 404).
- **`OutOfRunBudget`** · dataclass · The cap one out-of-run unit of work is held to, and the ledger its calls are written to.
- **`OutOfRunSpend`** · model · One call the engine made outside any run, as it was admitted and as the provider reported it.
- **`OutOfRunSpendStore`** · protocol · The one write the out-of-run ledger makes.
- **`PassHatPoint`** · typed dict · One point of the pass^k curve: the estimate at one depth, and how many cases it rests on.
- **`PlannedCall`** · dataclass · One call an out-of-run unit of work means to make: the prompt pair and the directive it sends.
- **`Precondition`** · model · One thing a template presumes about the world before the subject's first turn.
- **`PreconditionOutcome`** · model · Outcome of one precondition asserted against the world at t=0.
- **`PricedCompletion`** · protocol · A completion the engine can price before it makes it: the call, the model, and the call's ceiling.
- **`ProposedDimSuggestion`** · model · A novel rubric dim the proposer invented (not a catalog reuse).
- **`ProposedTemplate`** · model · The drafted template fields for one subject (capability or boundary).
- **`ProviderFailure`** · dataclass · What eval is allowed to know about a completion call that raised.
- **`ProviderFailureDescriber`** · protocol · The host's mapping from its own exceptions onto `ProviderFailure`.
- **`ProviderRefusedError`** · exception · A paid provider call made on the caller's behalf raised (status 502).
- **`Question`** · model · Something this campaign is trying to find out.
- **`QuestionAnswer`** · model · Where one declared question stands on this evidence.
- **`Recordable`** · protocol · The two members the cassette layer uses on whatever it records: an action result or a delivered payload.
- **`RecordedCall`** · model · One call a candidate made that succeeded — or, in a control end state, one it is stated to have made.
- **`RecordedCompletion`** · class · What an admitted call returned, and the ledger row written for it.
- **`RepeatedScore`** · model · One dimension's stored judge score, asked again of the same judge from the same evidence.
- **`ReplayedDelivery`** · dataclass · One recorded piece of background work, served in place of running it.
- **`ResolvedUsage`** · model · The per-role usage a read surface should show for one result, and how it got there.
- **`ResultCondition`** · model · The condition one result is in, on every axis that is a property of the result.
- **`ResultOutcome`** · enum · How a result participates in scoring, by its error category.
  <br>values: `'ok'`, `'candidate_fail'`, `'infra_exclude'`
- **`ResultStore`** · protocol · The cells a run recorded: each `EvalResult` and the `EvalTrace` beside it.
- **`RoleUsage`** · model · Per-role token + cost observation for one `EvalResult`.
- **`RoleUsageLedger`** · dataclass · Accumulates one role's LLM spend across every call it makes in a cell.
- **`RubricDim`** · model · One judge-scored rubric dimension.
- **`RubricDimTombstone`** · model · The record that a rubric dim key was deleted, so a seed never writes it back.
- **`RubricProposal`** · model · The validated DRAFT the rubric proposer returns for operator review.
- **`RubricScore`** · model · Outcome of one rubric judge dimension.
- **`RunCompleteness`** · model · How much of its matrix a run actually delivered.
- **`RunIndexEntry`** · model · One row of the analysis's run index — a run's config + key metrics.
- **`RunRecordStore`** · protocol · A run together with its cells: for an operation that reads one and rewrites the other.
- **`RunStore`** · protocol · The eval runs of a scope: their documents, listings, stamps, archive flag and boot scan.
- **`ScaleSpec`** · class · What one rubric scale means, stated once.
- **`SeedPrompt`** · dataclass · Where one shipped default lives, so a host can register it by import.
- **`SeedSection`** · dataclass · One section of a seeded prompt template.
- **`SeedTemplate`** · dataclass · A seeded prompt template: an ordered set of sections and its identity.
- **`SimulatorLLM`** · protocol · The one-shot text-generation port the simulator role and the variation generator call.
- **`StorageError`** · exception · Storage operation failed (status 503).
- **`StoreConflict`** · exception · A conditional write lost its race: the stored document no longer carries `if_match`.
- **`StratumFacts`** · model · Everything measured in one stratum of one cell — the cell's figures again, over one kind of case.
- **`SweptAxis`** · model · One axis this campaign set out to vary, and the levels it meant to compare.
- **`SyncActionSeam`** · protocol · A kind's synchronous tools whose calls block rather than await, as the cassette lane records and replays them.
- **`SyncToolLike`** · protocol · The three members the cassette layer calls on a tool that answers as a plain blocking call.
- **`TierCriterion`** · model · One of the two measurements a judged tier is decided by, as it read.
- **`TimeAxis`** · model · The campaign's runs placed in time — present only when they span two builds or two days.
- **`TimePosition`** · model · One point on a campaign's time axis — the runs at one build or on one day, and what they measured.
- **`ToolLike`** · protocol · The three members the cassette layer calls on a synchronous tool it wraps.
- **`UnknownCandidateKind`** · exception · A template names a kind this host did not wire.
- **`UnreadableBarName`** · dataclass · A bar name no verdict can be given on, and why — the text an author or a reader is shown.
- **`ValidationFailedError`** · exception · Validation failed (status 422).
- **`VariantConfig`** · model · The A/B'd stack one cell runs, as every kind alike receives it.
- **`VariantIndexEntry`** · model · One observed variant, and the resolved lever map its key was digested from.
- **`VariationAxis`** · model · One axis along which test cases vary for a template.
- **`VariationCounts`** · model · How many test cases a launch asked generation for, and how many it froze.
- **`VariationLLM`** · protocol · The client the variation generator writes an `llm` axis's values with, naming the model it calls.
- **`Viz`** · model · A visualization spec attached to a finding.
- **`WorldEvent`** · model · One thing that moved a cell's world after it was seeded, in the order it happened.
- **`WorldSeed`** · model · Initial state for the eval's stateful world.
- **`WorldSession`** · class · One cell's handle on the host's world, and the record of what happened to it.
- **`WorldSessionError`** · exception · A kind asked its cell's world session for something the host's world, or the moment, cannot give.

**Types**

- **`ApparatusProvenance`** · literal · Whether a run's apparatus was set before the fact or found after it.
  <br>`'commissioned'` | `'witnessed'`
- **`ApparatusSettingValue`** · type alias · One host-declared apparatus value a launch sets (`EvalRun.apparatus_settings`): a string, a bool, or a finite number — a level two runs can be compared on, and hashed into the measurement context.
  <br>`Annotated[StrictStr | StrictBool | StrictInt | StrictFloat, AfterValidator(_finite_setting)]`
- **`AsyncDeliveryStatus`** · literal · Where one piece of background work stood when the cell ended.
  <br>`'delivered'` | `'failed'` | `'undelivered'`
- **`AttemptOutcome`** · literal · How one generation attempt ended.
  <br>`'stored'` | `'refused'` | `'failed'` | `'cancelled'`
- **`AttributionScope`** · literal · Whether a measure isolates one `subsystem` or reflects the whole `end_to_end` run.
  <br>`'subsystem'` | `'end_to_end'`
- **`BarDecision`** · literal · What a bar's verdict on one cell came to — see `BarVerdict.decision`.
  <br>`'cleared'` | `'missed'` | `'undecided'` | `'no_interval'` | `'no_data'`
- **`BarNameKind`** · literal · The three kinds of name a bar may carry, each read from a different place on a result.
  <br>`'measure'` | `'judged'` | `'goal_state'`
- **`BarNameRefusal`** · literal · Why a bar name has no verdict to give.
  <br>`'not_numeric'` | `'no_better_end'` | `'not_carried'`
- **`CandidateFailureCause`** · literal · Why a result is a candidate failure.
  <br>`'model_failed'` | `'turn_budget'` | `'output_cap'`
- **`CassetteMode`** · literal · A run's cassette mode. `'off'` runs every tool live and records nothing, so a cell of such a run is handed no `CellCassettes` at all.
  <br>`'capture'` | `'replay'` | `'off'`
- **`CassetteSeam`** · literal · Which interception point a cassette was recorded at. See `EvalCassette` for what each seam implies about `response`'s shape.
  <br>`'action'` | `'delivery'`
- **`CellPending`** · literal · What a running cell is waiting on — the component its deadline, striking now, would be waiting for.
  <br>`'apparatus'` | `'candidate'` | `'background_work'` | `'simulator'` | `'judge'`
- **`CellTermination`** · literal · How a cell's execution ended — the branch the runner took, recorded because the record cannot recover it.
  <br>`'completed'` | `'factory_failed'` | `'cell_timeout'` | `'seed_failed'` | `'precondition_failed'` | `'apparatus_failed'` | `'cancelled'`
- **`ClassifierStatistic`** · literal · The per-label statistics a classifier's confusion matrix yields, each minted per label.
  <br>`'precision'` | `'recall'` | `'f1'`
- **`CompletenessSource`** · literal · Where a run's completeness counts were taken: the run loop's own tally, or the results in storage (`RunCompleteness.counted_from`).
  <br>`'run_loop'` | `'stored_results'`
- **`Confidence`** · literal · How sure the author is of a finding or a decision.
  <br>`'very_high'` | `'high'` | `'medium'` | `'low'`
- **`ConfidenceTier`** · literal · How firmly the evidence settles a claim, as a qualitative tier — the tier a stored finding or decision carries.
  <br>`'very_high'` | `'high'` | `'medium'` | `'low'`
- **`CostCapOrigin`** · literal · How a run arrived at the spend ceiling it ran under.
  <br>`'chosen'` | `'inherited'` | `'uncapped'`
- **`CriterionState`** · literal · How one criterion read: its interval wholly at or above the threshold over enough results (`met`), wholly below it (`not_met`), across it (`undecided`), or too little evidence to say (fewer distinct results than its floor, or a kappa or interval that is undefined).
  <br>`'met'` | `'not_met'` | `'undecided'` | `'insufficient'`
- **`DeliveryOutcome`** · literal · How a recorded piece of background work ended when its capture cell did: it delivered a payload, it failed, or the cell ended with it still in flight.
  <br>`'delivered'` | `'failed'` | `'undelivered'`
- **`DimName`** · type alias · A rubric dimension's name as every model holds it: `<context>.<dim>`, or a reserved dual-score axis id.
  <br>`Annotated[str, AfterValidator(_namespaced_dim_name)]`
- **`EvalRunStatus`** · literal · `budget_stopped` is a budget the run was launched under binding — its cost cap or its wall-clock budget, `EvalRun.budget_stop_reason` says which: a designed stop that keeps what the run delivered, never `failed`.
  <br>`'pending'` | `'running'` | `'completed'` | `'failed'` | `'cancelled'` | `'budget_stopped'` | `'exhausted'`
- **`EvidenceTier`** · literal · What a finding's verdict stands on, as code reads it off the finding's resolved evidence rows — never chosen by the report writer: code assigns the tier.
  <br>`'mechanical'` | `'calibrated'` | `'separation'` | `'undetermined'` | `'incidental'` | `'none'`
- **`GoalCheckIntent`** · literal · What a goal check says about the behaviour it grades, and so which verdict "the candidate did nothing" must get.
  <br>`'act'` | `'hold'`
- **`GoalCheckProof`** · literal · Whether a goal check was shown, when its run launched, to tell its outcomes apart (`threetears.evals.run.check_controls.goal_check_proofs`).
  <br>`'proven'` | `'unproven'` | `'refuted'`
- **`GradedBy`** · literal · Who produced a family's numbers.
  <br>`'code'` | `'judge'`
- **`GuardrailDecision`** · literal · What a guardrail came to for one arm against the control, read off the interval on the difference against the guardrail's margin (`interval_clears`, three-valued).
  <br>`'held'` | `'breached'` | `'undecided'`
- **`JudgeAttributionSource`** · literal · Whether a run's per-dim judge attribution was captured at launch or reconstructed afterwards from stored `JudgeConfig` records.
  <br>`'recorded'` | `'derived'`
- **`JudgeAttributionState`** · literal · What a run can say about which model scored each dim.
  <br>`'recorded'` | `'derived'` | `'absent'`
- **`JudgedEvidenceTier`** · literal · The tier a judged reading stands on: one of the three the evidence can establish, or `undetermined` when it establishes none of them.
  <br>`'calibrated'` | `'separation'` | `'incidental'` | `'undetermined'`
- **`JudgedTierRule`** · literal · The rule a stored analysis's judged tiers were decided by.
  <br>`'interval_lower_bound'`
- **`JudgeTemperature`** · type alias · The temperature a judge call was actually sent at: a number, or `MODEL_DEFAULT_TEMPERATURE`.
  <br>`float | Literal['model_default']`
- **`JudgingState`** · literal · What the judge produced for this result.
  <br>`'scored'` | `'partial'` | `'failed'` | `'not_attempted'`
- **`Materiality`** · literal · How a difference in a measure reads against the measure's declared materiality threshold.
  <br>`'material'` | `'immaterial'`
- **`MeasurePopulation`** · literal · Which observations a measure is computed over.
  <br>`'scored'` | `'all_observed'` | `'delivered'`
- **`MeasureScale`** · literal · What a difference in a measure's values means.
  <br>`'ratio'` | `'interval'`
- **`MeritAxis`** · literal · The merit axes a measure can serve.
  <br>`'quality'` | `'cost'` | `'latency'` | `'reliability'`
- **`MeteredCallOrigin`** · literal · Which tier supplied a run's metered-call ceiling (`EvalRun.max_metered_calls_origin`): the three `CostCapOrigin` tiers, read with the currency swapped, and one more — `none_declared`, a host declaring it has no metered tools (`LaunchSettings.max_metered_calls` of `None`).
  <br>`'chosen'` | `'inherited'` | `'uncapped'` | `'none_declared'`
- **`MetricDataType`** · literal · What one observation of a measure is.
  <br>`'numeric'` | `'categorical'` | `'boolean'` | `'text'`
- **`MetricFamily`** · type alias · A measure family's name.
  <br>`str`
- **`ModelRoleOrigin`** · literal · How a run arrived at one of its resolved role models.
  <br>`'chosen'` | `'inherited'`
- **`OutOfRunOutcome`** · literal · How an out-of-run call ended: it returned a completion, or it raised.
  <br>`'completed'` | `'raised'`
- **`OutOfRunPurpose`** · literal · What an out-of-run call was for: `variation` writes a launch's generated cases (an `llm` variation axis's values), `proposer` drafts a rubric for operator review, `analysis` writes a campaign's analysis memo (its first call and the one repair round-trip a refused output buys), `judge` repeats a finished run's judge scores to measure the judge's agreement with itself (`repeat_judge_scores`).
  <br>`'variation'` | `'proposer'` | `'analysis'` | `'judge'`
- **`RaterKind`** · literal · Who wrote a calibration rating: a `person`, whose rating is the human side of judge calibration, or an `agent` (a model acting through a tool), whose rating is not.
  <br>`'person'` | `'agent'`
- **`Reading`** · literal · What a piece of evidence reads: a `measure`, or a `judged` dimension.
  <br>`'measure'` | `'judged'`
- **`ReasoningEffort`** · literal · A reasoning effort level, as the router's `reasoning.effort` takes it.
  <br>`'max'` | `'xhigh'` | `'high'` | `'medium'` | `'low'` | `'minimal'` | `'none'`
- **`RequestCeiling`** · type alias · The longest one request capped at `max_tokens` output tokens can take on the host's client, in seconds: every provider call the client makes for that one request and every wait between them (re-sends of a severed body, SDK retries, their back-off sleeps).
  <br>`Callable[[int], float]`
- **`RoleModelOrigin`** · literal · How a run arrived at one of its resolved role MODELS — `ModelRoleOrigin`'s two values plus `alternate`: the launch named no judge, and the host's alternate judge (`judge_alternate_model`) scored, because the judge role's default was one of the launch's candidates (`resolve_judge_pin`).
  <br>`'chosen'` | `'inherited'` | `'alternate'`
- **`RubricAxis`** · literal · The two rubric axes a dimension sits on: what a subject should DO (`capability`) and what it should refuse or withstand (`boundary`).
  <br>`'capability'` | `'boundary'`
- **`RubricScale`** · literal · How a criterion is answered: an integer from 1 to 5, or pass/fail.
  <br>`'ordinal'` | `'pass_fail'`
- **`SimulatorPurpose`** · literal · What one simulator-role call was for: an actor's line, or the `llm_decided` scheduler's pick of who speaks next.
  <br>`'utterance'` | `'schedule'`
- **`StopReason`** · literal · Why a completion stopped, in the engine's words.
  <br>`'end_turn'` | `'max_tokens'` | `'content_filter'` | `'error'`
- **`SyncToolWrap`** · type alias · What the synchronous action seam is handed: `ToolWrap` over `SyncToolLike` tools.
  <br>`Callable[[Mapping[str, SyncToolLike]], dict[str, SyncToolLike]]`
- **`TimeAxisBasis`** · literal · What a campaign's time positions are: the builds a host labels (`release`), or the UTC days its runs started on (`date`).
  <br>`'release'` | `'date'`
- **`ToolWrap`** · type alias · What the action seam is handed: maps the candidate's tools, by name, to the tools it should use — each declared recorded tool wrapped for the cassette run, every other one unchanged.
  <br>`Callable[[Mapping[str, ToolLike]], dict[str, ToolLike]]`
- **`TransferabilityClass`** · literal · How far a measure's meaning carries, loosest to strictest: `mechanical` is measured the same way everywhere, `judge_mediated` is comparable only under the same judge configuration, `scenario_bound` means nothing outside its scenario.
  <br>`'mechanical'` | `'judge_mediated'` | `'scenario_bound'`
- **`UsageRole`** · literal · The five roles an eval cell can spend on.
  <br>`'candidate'` | `'judge'` | `'simulator'` | `'inner_agent'` | `'external'`
- **`VizType`** · literal · Every visualization a stored finding can carry.
  <br>`'delta_table'` | `'frontier'` | `'timeseries'` | `'distribution'` | `'null_result'` | `'breakdown'` | `'attribution'` | `'sweep_ranking'`
- **`WorldEventCause`** · literal · Who made it happen.
  <br>`'rig'` | `'world'`
- **`WorldEventKind`** · literal · What moved the world: a triggered dimension's condition, by its trigger kind, or ambient perturbation.
  <br>`'turn'` | `'event'` | `'human'` | `'ambient'`

**Constants**

- **`ACCURACY_MEASURE`** · constant (str) · The classifier family's quality measure: 1.0 or 0.0 per observation, DERIVED by the engine from `MATCH_MEASURE` and never landed by a kind.
  <br>`= 'accuracy'`
- **`CALIBRATION_MIN_AGREEMENT`** · constant (float) · The least judge–human agreement (weighted kappa; kappa on pass/fail) a dimension's judge must reach to be `calibrated` on it. Owner ruling, 2026-10-06.
  <br>`= 0.6`
- **`CALIBRATION_MIN_RESULTS`** · constant (int) · The fewest distinct results a calibration must cover before it can decide anything: below it the calibration is `insufficient`, whatever its kappa.
  <br>`= 20`
- **`CANDIDATE_SPEAKER`** · constant (str) · The speaker label the candidate's own turns carry in a simulated transcript.
  <br>`= '__candidate__'`
- **`CLASSIFIER_FAMILY`** · constant (str) · A label compared against an expected one by code.
  <br>`= 'classifier'`
- **`COMPOSITE_FAMILY`** · constant (str) · A figure built over several results or runs: pass^k, the composite, a comparison's effect size.
  <br>`= 'composite'`
- **`CONFIDENCE_TIERS`** · constant (tuple) · The tiers, strongest first — the order a surface ranks by.
- **`CONFUSION_CELL_MEASURE`** · constant (str) · The core measure a classification's confusion-matrix cell is reported under.
  <br>`= 'confusion_cell'`
- **`DEFAULT_JUDGE_TEMPERATURE`** · constant (float) · The temperature every judge call is requested at unless a `JudgeConfig` for its dimension says otherwise, and that config's own default (#633).
  <br>`= 0.0`
- **`DEFAULT_LAUNCH_K_RUNS`** · constant (int) · Repeats per (case, model) a launch uses when the caller names none: one observation per case cannot tell a setting from the model's own variance.
  <br>`= 3`
- **`DROPPED_TOOL_CALLS_KEY`** · constant (str) · Covariate key: how many tool calls the candidate emitted that the LLM client dropped before dispatch (a `dropped_tool_calls` list on each turn record the kind dumps into the trace).
  <br>`= 'dropped_tool_calls'`
- **`DUAL_AXIS_FAMILY`** · constant (str) · The two reserved judge axes every judged run is scored on.
  <br>`= 'dual_axis'`
- **`ENGINE_FAMILIES`** · constant (dict) · The engine's own families, by name. A host family may not reuse one of these names.
- **`EVAL_DOC_TYPES`** · constant (tuple) · Every `doc_type` the engine writes — the set the operator wipe sweeps.
- **`EVAL_SCHEMA_VERSION`** · constant (int) · The schema version every stored eval document is written under, and the only one a read accepts.
  <br>`= 8`
- **`GOAL_STATE_FAMILY`** · constant (str) · A goal-state check's verdict: code compared against what the candidate did.
  <br>`= 'goal_state'`
- **`IDENTITY_VERSION`** · constant (int) · One counter covers both predicates, so a bump on either side re-derives the other's keys — conservative in the safe direction, since re-deriving unchanged inputs reproduces the same digest.
  <br>`= 24`
- **`INCOMPLETE_STOP_REASONS`** · constant (frozenset) · The `CompletionResult.stop_reason` values meaning the completion was CUT SHORT rather than finished.
- **`JSON_OBJECT_RESPONSE_FORMAT`** · constant (dict) · `response_format` directive forcing JSON-object output, passed to `CompletionClient.generate` by callers that parse a structured JSON *object* (the judge, the analysis generator, the proposer/boundary-proposer).
- **`JUDGED_TIER_RULE`** · constant (str) · The rule this build decides tiers by.
  <br>`= 'interval_lower_bound'`
- **`JUDGED_TIERS_WEAKEST_FIRST`** · constant (tuple) · Weakest first, for composing several judged readings into what all of them can bear.
- **`KIND_TEMPLATE`** · constant (str) · A seed whose constant is a sectioned template, registered with a template registry.
  <br>`= 'template'`
- **`KIND_TEXT`** · constant (str) · A seed whose constant is a single string, registered with a shared-prompt registry.
  <br>`= 'text'`
- **`MATCH_MEASURE`** · constant (str) · The core measure one classification's verdict is reported under — the one a classifier kind lands on `host_measures`, a bool.
  <br>`= 'match'`
- **`MECHANICAL_FAMILY`** · constant (str) · Measured the same way everywhere by code: wall-clock, tokens, spend, counts.
  <br>`= 'mechanical'`
- **`METRIC_DESCRIPTORS`** · constant (dict) · The engine's own measures, each one's descriptor keyed by its name.
- **`MODEL_DEFAULT_TEMPERATURE`** · constant (str) · A judge call SENT with no temperature, because its model refuses one (some reasoning models do): the model's own default applied.
  <br>`= 'model_default'`
- **`NON_TERMINAL_RUN_STATUSES`** · constant (frozenset) · Statuses a run can still leave under its own power — it is either queued or executing, and something in-process is expected to write its terminal status.
- **`NOT_ESTABLISHED`** · constant (str) · The prefix of every detail recording a check that rested on a value the world did not hold.
  <br>`= 'not established'`
- **`OUTCOME_DIM_ID`** · constant (str) · Reserved `rubric_dim_id` for the dual-score outcome axis.
  <br>`= '__outcome__'`
- **`PROSE_SCHEMA_KEY`** · constant (str) · The JSON Schema keyword marking a string property as model prose, for vocabularies declared as schema rather than as Pydantic models (a world dimension's value schema).
  <br>`= 'x-model-prose'`
- **`REASONING_RATIO_KEY`** · constant (str) · Covariate key: the share of the candidate's generated tokens that were reasoning — candidate `reasoning_tokens` over candidate `completion_tokens`, summed across the candidate's usage rows.
  <br>`= 'reasoning_ratio'`
- **`REFUSED_TOOL_ATTACHES_KEY`** · constant (str) · Covariate key: how many times the candidate asked to attach a tool outside its run's `tools_allowed` and the harness refused.
  <br>`= 'refused_tool_attaches'`
- **`RESERVED_DIM_IDS`** · constant (frozenset) · The reserved dim ids, which are deliberately NOT namespaced: they identify the two dual-score axes rather than a rubric dimension scored in some context, so there is no context to name. `require_namespaced_dim_name` exempts them — a judge service is built for these ids on every run.
- **`ROUND_DONE`** · constant (str) · The scheduler's answer that the current speaker round is over and the candidate answers next.
  <br>`= 'round_done'`
- **`RUBRIC_FAMILY`** · constant (str) · A judge's score on an authored rubric dimension.
  <br>`= 'rubric'`
- **`SCALES`** · constant (mappingproxy) · The scales, by name.
- **`SEPARATION_MIN_AGREEMENT`** · constant (float) · The least agreement a judge must reach with its own repeated scores (the same statistic as calibration) to earn `separation`. Owner ruling, 2026-10-06.
  <br>`= 0.8`
- **`SEPARATION_MIN_RESULTS`** · constant (int) · The fewest distinct results a self-agreement must cover before it can decide anything.
  <br>`= 120`
- **`STRATUM_MIN_CASES`** · constant (int) · The fewest distinct cases a stratum holds before its figures are read on their own.
  <br>`= 10`
- **`TRANSCRIPT_DIM_ID`** · constant (str) · Reserved `rubric_dim_id` for the dual-score transcript axis.
  <br>`= '__transcript__'`
- **`TRUNCATED_ROUNDS_KEY`** · constant (str) · Covariate key: how many of the candidate's LLM rounds the provider cut off at the output cap — rounds whose `stop_reason` was `max_tokens` (the output reached the cap, or the provider reported `finish_reason=length`; providers do not always say so).
  <br>`= 'truncated_rounds'`
- **`TURN_BUDGET_ENDED_KEY`** · constant (str) · Covariate key: how many of the candidate's turns the HOST's turn budget ended before they finished.
  <br>`= 'turns_ended_by_budget'`

<a id="api-contracts-host"></a>
### `threetears.evals.contracts.host`

The host contract — what a consuming product declares, and what the engine never interprets.

**Functions**

- **`check_seed`** · function · Walk a world seed against the registry: every write it would make, checked, or the first refusal.
  <br>`check_seed(registry: WorldRegistry, namespaces: Mapping[str, Any], *, attached: Collection[str] | None = None) -> tuple[SeedWrite, ...]`
- **`check_world_conformance`** · async function · Run every obligation this registry's declarations imply, against the host's own handles.
  <br>`check_world_conformance(registry: WorldRegistry, *, expressions: Sequence[str] = ()) -> WorldConformanceReport`
- **`default_cell_timeout`** · function · Enforce a cell budget with `asyncio.timeout` and nothing else.
  <br>`default_cell_timeout(budget_s: float) -> AsyncIterator[float]`
- **`freeze`** · function · What is recorded of values a kind's model validated: the model's JSON form, or nothing.
  <br>`freeze(validated: BaseModel | None) -> dict[str, Any]`
- **`nested_schemas`** · function · Every schema written directly inside `schema`, in declaration order.
  <br>`nested_schemas(schema: Mapping[str, Any]) -> Iterator[NestedSchema]`
- **`obligation_rows`** · function · Which rows of the obligations table this dimension's declared shape sits on.
  <br>`obligation_rows(declared: WorldDimension) -> frozenset[ObligationRow]`
- **`obligations`** · function · The per-dimension checks this dimension's shape owes, derived and never declared.
  <br>`obligations(declared: WorldDimension) -> tuple[CheckName, ...]`
- **`require_resolved_colour`** · function · The palette's one colour check: `value` is resolved sRGB hex (`#rrggbb`), or a refusal.
  <br>`require_resolved_colour(where: str, value: object) -> str`
- **`schema_violations`** · function · Every way `value` fails `schema`, each naming the path it fails at.
  <br>`schema_violations(schema: Mapping[str, Any], value: Any, *, at: str) -> list[str]`
- **`served_models_by_score`** · function · Who scored each of one result's stored scores, as the provider named it — `None` where it did not.
  <br>`served_models_by_score(result: EvalResult) -> list[str | None]`

**Classes**

- **`ActsOn`** · dataclass · Name the measure or covariate an overlay knob is supposed to move — its lever's `acts_on`.
- **`ApparatusError`** · exception · A fault in the eval measuring rig, not a failure of the world under test.
- **`Bar`** · dataclass · The incumbent standard for one behavior on one measure.
- **`BarProposal`** · dataclass · A bar the ratchet computed from a measurement, for a person to confirm or tighten.
- **`BarRegistrationError`** · exception · A bar declaration contradicts what this registry promises.
- **`BarRegistry`** · class · One host's registered bars, keyed by `(behavior, measure)` and validated at construction.
- **`CellIdentity`** · dataclass · Which cell execution a stretch of work belongs to.
- **`CellTimeoutFactory`** · protocol · Builds the context manager one cell runs inside, from that cell's budget.
- **`CellTrace`** · dataclass · What one cell's tracing yielded — filled by the sink, read by the engine.
- **`ChartFont`** · dataclass · A typeface a renderer draws chart text in, with the advance widths its layout is computed from.
- **`ChartPalette`** · dataclass · A host's chart colours, by role — what a renderer themes every chart it draws with.
- **`CompletionClients`** · protocol · The host's completion-client factory: one client per role, model and temperature.
- **`ConformanceResult`** · dataclass · One check's verdict on one dimension, or on the registry when the check is registry-wide.
- **`Coverage`** · dataclass · Whether eval reaches one precondition on one axis, and why when it does not.
- **`EvalCellTimeout`** · exception · A cell outlived the wall-clock budget its timeout context was enforcing.
- **`EvalHost`** · dataclass · Everything the engine reads of one consuming product, as one frozen value.
- **`ExternalSpend`** · dataclass · One caller's report of what a provider call (or a batch of them) consumed.
- **`HostProfile`** · dataclass · Everything the engine knows about one consuming product.
- **`Interval`** · dataclass · Mark a numeric field's unit, rendered beside each level (`120words`).
- **`IntervalScale`** · model · A real number with real spacing — `0.4`, `15`, `2000ms`.
- **`KindContract`** · dataclass · What one candidate kind's runs carry, declared as Pydantic models — see the module docstring.
- **`KindContractError`** · exception · A kind's model cannot be read the way the engine promises to read it — raised where it is declared.
- **`MeasureRegistrationError`** · exception · A measure declaration contradicts what this registry promises.
- **`MeasureRegistry`** · class · One host's declared measures, validated at construction.
- **`NestedSchema`** · class · One schema written inside another, and where it sits.
- **`NominalScale`** · model · Unordered categories. Two levels are different, and neither is larger.
- **`Ordinal`** · dataclass · Mark a `Literal` or `Enum` field as ordered: its levels rank in the order they are declared.
- **`OrdinalScale`** · model · Ordered but unspaced — `small` / `medium` / `large`.
- **`ProfileRegistrationError`** · exception · Two of a host's registries contradict each other, raised where both are in hand.
- **`RegistrationError`** · exception · A declaration contradicts what this module promises, raised where it is written.
- **`ResolvedLevers`** · dataclass · What one run's levers are called and what it ran them at.
- **`ResolvesInto`** · dataclass · Name the fixed host lever this overlay knob is WRITTEN INTO — its lever's `resolves_into`.
- **`RolePins`** · dataclass · One pinned ROLE — who filled a seat in the rig, and the inputs that identify them.
- **`SeedRefused`** · exception · A seed asked for a write the registry or the subject cannot make.
- **`SeedWrite`** · class · One write a checked seed makes: through `handle`, of `value`, into dimension `name`.
- **`StyleError`** · exception · A style profile contradicts the bounded contract this module promises.
- **`StyleProfile`** · dataclass · One host's bounded presentation contract.
- **`SubjectSnapshot`** · model · The subject a set of observations was taken against, frozen at capture.
- **`Sweepable`** · dataclass · One score-determining input: what it is called, who owns it, how to read it.
- **`SweepableRegistry`** · class · The declared inputs for one host, validated at construction.
- **`SweepableValue`** · model · One level of one swept input: what it *is*, what to call it, and what kind of axis it is on.
- **`TraceSink`** · protocol · The engine's whole reach into a host's tracing.
- **`Triggered`** · dataclass · State that arrives on a condition rather than at t=0.
- **`UnsupportedSchemaError`** · exception · A schema uses a construct outside `HONOURED_KEYWORDS` — a gap to extend, never a value to pass.
- **`WorldConformanceError`** · exception · An ENGINE gap: a check the kit cannot run, or a record its own vocabulary forbids.
- **`WorldConformanceReport`** · dataclass · Every verdict from one run of the kit over one registry.
- **`WorldDimension`** · dataclass · One state dimension a run may set, may read back, or may only witness.
- **`WorldRegistrationError`** · exception · A world declaration contradicts what this module promises, raised where it is written.
- **`WorldRegistry`** · class · One host's world declarations, its resolution table, and the algebra over them.

**Types**

- **`ActionParameterReader`** · type alias · `(tool, action) -> that action's parameter JSON Schema`, or None for an action the host does not describe.
  <br>`Callable[[str, str], 'Mapping[str, Any] | None']`
- **`CheckName`** · literal · One conformance check.
  <br>`'round_trip'` | `'perception_ab'` | `'perception_stillness'` | `'ambient_isolation'` | `'independence'` | `'vocabulary_completeness'`
- **`Comparability`** · literal · The three outcomes of comparing one input across a set of runs.
  <br>`'same'` | `'differs'` | `'unknown'`
- **`CompletionRole`** · literal · The apparatus roles the engine asks a host for a completion client in.
  <br>`'judge'` | `'simulator'` | `'analysis'` | `'variation'` | `'proposer'`
- **`CoverageState`** · literal · The answer to "does eval reach this precondition", per axis.
  <br>`'covered'` | `'uncovered'` | `'inapplicable'`
- **`Evidence`** · literal · Is a dimension's `read` computed from the world, or an authored claim about it?
  <br>`'machine'` | `'labeled'`
- **`FamilyMemberTest`** · type alias · Whether one bare name is a member of an open family, asked with no run in hand.
  <br>`Callable[[str], bool]`
- **`ObligationRow`** · literal · A row of the obligations table.
  <br>`'seedable_machine_read'` | `'seedable_labeled_read'` | `'perceivable_not_seedable'` | `'triggered_automatic'` | `'triggered_human'` | `'every_dimension'`
- **`Outcome`** · literal · What a check concluded.
  <br>`'passed'` | `'failed'` | `'unavailable'`
- **`Qualification`** · literal · Why a result is narrower than a clean proof. Present on every outcome that is not one.
  <br>`'plumbing_only'` | `'arming_only'` | `'not_instantiable_unattended'` | `'no_perturbation_binding'` | `'nothing_to_observe'` | `'schema_admits_too_few_values'` | `'nothing_to_resolve'` | `'seeding_did_not_take'`
- **`ResidualReader`** · type alias · The surface an open family resolves into, with some of its members taken back out.
  <br>`Callable[['EvalRun', 'Sequence[EvalResult]', frozenset[str]], Any]`
- **`Scale`** · type alias · What kind of axis a swept value sits on.
  <br>`Annotated[NominalScale | OrdinalScale | IntervalScale, Field(discriminator='kind')]`
- **`SeedRefusalKind`** · literal · Why a seed was refused, one value per question the walk asks.
  <br>`'malformed'` | `'undeclared'` | `'misplaced'` | `'unseedable'` | `'nonconforming'` | `'unattached'`
- **`SweepableReader`** · type alias · Reader signature. A reader is host code the engine calls and never inspects.
  <br>`Callable[['EvalRun', 'Sequence[EvalResult]'], Any]`
- **`SweepableRole`** · literal · Who owns a sweepable input: a `lever` a campaign sweeps, `apparatus` (the measuring rig, which should not move), or a `label` that identifies rather than varies.
  <br>`'lever'` | `'apparatus'` | `'label'`
- **`ToneRegister`** · literal · The registers a host may pick from. Engine-owned and closed — the point of the enum is that a host cannot write its own.
  <br>`'neutral'` | `'executive'` | `'technical'`
- **`ToolActionReader`** · type alias · `tool -> the names of its actions`, or None for a tool whose actions the host cannot list.
  <br>`Callable[[str], 'frozenset[str] | None']`
- **`TriggerKind`** · literal · What kind of condition brings a triggered dimension into being.
  <br>`'turn'` | `'event'` | `'human'`
- **`VariantLeverReader`** · type alias · Produce a run's level of each of the HOST'S OWN levers — its share of the variant key's pre-image.
  <br>`Callable[['EvalRun'], 'dict[str, SweepableValue]']`
- **`When`** · type alias · When a dimension's value comes into being. `initial` is the common case — set before the subject's first turn.
  <br>`Literal['initial'] | Triggered`
- **`WorldCapability`** · literal · What a run can do with a dimension, computed from its registration and nothing else.
  <br>`'representable'` | `'judge_only'` | `'witnessed'`
- **`WorldPlacement`** · literal · What a RUN did with a dimension, computed from that run's own record and nothing else.
  <br>`'representable'` | `'judge_only'` | `'witnessed'` | `'out_of_play'`

**Constants**

- **`CANDIDATE_KIND_LEVER`** · constant (str) · The name the candidate KIND answers to: the other half of what a run ran, beside its model.
  <br>`= 'candidate_kind'`
- **`CANDIDATE_MODEL_LEVER`** · constant (str) · The one name the candidate model answers to, everywhere — a DECLARED COORDINATE of every observation (`ScoreRecord.model`) as well as the core lever below, which is why it is the only lever a reporting lens resolves off the observation rather than off the run.
  <br>`= 'model'`
- **`CHART_FONT_CHARACTERS`** · constant (str) · The characters a `ChartFont`'s metrics must cover: printable ASCII, space to tilde.
- **`SERIES_SLOTS`** · constant (int) · How many categorical colour slots a palette supplies before it recycles — the width of the vocabulary, and so exactly how many `ChartPalette.series` colours a palette declares.
  <br>`= 8`
- **`SHARED_CORE`** · constant (SweepableRegistry) · The registry a host extends. Nothing here names a product.
- **`UNSEATED_LEVEL`** · constant (str) · The level an apparatus dimension reads at for a run whose rig has no such seat (`HostProfile.apparatus_level`), where the dimension is kept because another run in the same comparison does have it.
  <br>`= "(none — this run's rig has no such seat)"`
- **`VALIDATED_SLOTS`** · constant (int) · How many categorical colour slots a chart may assign with validated separation.
  <br>`= 4`

<a id="api-run"></a>
### `threetears.evals.run`

The engine's run package: launching and executing a run, judging it, metering it, and storing it.

**Functions**

- **`assert_preconditions`** · function · Assert at t=0 that the world this template presumes actually holds.
  <br>`assert_preconditions(template: EvalTemplate, test_case: EvalTestCase, seeded: Mapping[str, Any], *, world: WorldRegistry | None) -> list[PreconditionOutcome]`
- **`build_judge_context`** · function · The evidence a result's judge reads, built from the cell's own records.
  <br>`build_judge_context(*, template: EvalTemplate, test_case: EvalTestCase, goal_outcomes: list[GoalStateOutcome], judged_artifact: JudgedArtifact, judge_evidence: JudgeEvidence) -> JudgeContext`
- **`build_judge_service`** · function · Build the `JudgeService` for a run, with its judge attribution.
  <br>`build_judge_service(host: EvalHost, template: EvalTemplate, judge_model: str, selection: dict[str, str] | None = None, *, judged_artifact: JudgedArtifact) -> RunJudge`
- **`callers_missing_the_constructor`** · function · Name the declared callers whose source no longer reaches the constructor.
  <br>`callers_missing_the_constructor(contract: FidelityContract) -> list[str]`
- **`cancel_run`** · function · Cancel a pending/running eval run so it stops burning cost and quota.
  <br>`cancel_run(storage: RunRecordStore, run_id: str, scope_id: str, *, job_manager: EvalJobManager | None, reason: str | None = None) -> EvalRun`
- **`create_judge_config`** · function · Create and persist a judge config from an authoring definition.
  <br>`create_judge_config(storage: DefinitionStore, definition: dict[str, Any], *, scope_id: str) -> JudgeConfig`
- **`create_rubric_dim`** · function · Create and persist a catalog rubric dim from an authoring definition.
  <br>`create_rubric_dim(storage: DefinitionStore, definition: dict[str, Any], *, scope_id: str) -> CatalogRubricDim`
- **`create_template`** · function · Create and persist a template from an authoring definition.
  <br>`create_template(host: EvalHost, definition: dict[str, Any], *, scope_id: str, require_known_tools_allowed: Callable[[Sequence[str] | None], None], refuse_undeclared_world_seed: Callable[[EvalTemplate], None], refuse_undeliverable_template: Callable[[EvalTemplate], None]) -> EvalTemplate`
- **`default_job_timeout`** · function · Enforce a job budget with `asyncio.timeout` and nothing else.
  <br>`default_job_timeout(budget_s: float) -> AsyncIterator[float]`
- **`delete_analysis`** · function · Delete a stored analysis, leaving the insights it minted in place.
  <br>`delete_analysis(storage: CurationStore, analysis: EvalAnalysis, *, confirm: str | None = None) -> dict[str, Any]`
- **`delete_insight`** · function · Delete a single insight from the ledger.
  <br>`delete_insight(storage: CurationStore, insight_id: str, scope_id: str, *, confirm: str | None = None) -> dict[str, Any]`
- **`delete_judge_config`** · function · Delete a judge config by id, after an id-echo confirmation.
  <br>`delete_judge_config(storage: DefinitionStore, config_id: str, scope_id: str, *, confirm: str | None = None) -> None`
- **`delete_result`** · function · Delete a single eval result, leaving its run and siblings in place.
  <br>`delete_result(storage: CurationStore, result: EvalResult, scope_id: str, *, confirm: str | None = None) -> dict[str, Any]`
- **`delete_rubric_dim`** · function · Delete a catalog rubric dim by id, after an id-echo confirmation, and retire its key from the seed.
  <br>`delete_rubric_dim(storage: DefinitionStore, dim_id: str, scope_id: str, *, confirm: str | None = None) -> None`
- **`delete_run`** · function · Delete a run, the results it owns, and its campaign memberships.
  <br>`delete_run(storage: CurationStore, run: EvalRun, scope_id: str, *, confirm: str | None = None) -> dict[str, Any]`
- **`drive_conversation`** · async function · Run `driver`'s conversation from its first turn until it stops on a structural signal.
  <br>`drive_conversation(driver: TurnDriver, candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]], post_user_turn: Callable[[SimulatorTurn], Awaitable[None]], *, llm: SimulatorLLM, sink: CellSink) -> ConversationStopCause`
- **`estimate_judge_repeat`** · async function · Price repeating a run's judge scores against the cap it would be held to, and make no call.
  <br>`estimate_judge_repeat(host: EvalHost, run_id: str, scope_id: str, *, out_of_run_cap_usd: float | None, result_ids: Sequence[str] | None = None) -> JudgeRepeatEstimate`
- **`evaluate_goal_state`** · function · Run every expression in `template.goal_state_checks` and capture outcomes.
  <br>`evaluate_goal_state(*, template: EvalTemplate, test_case: EvalTestCase, ledger: CallLedger, end_state: Mapping[str, Any], fired: Firings | None, world: WorldRegistry | None) -> list[GoalStateOutcome]`
- **`execute_run`** · async function · Execute an entire `EvalRun` end-to-end.
  <br>`execute_run(host: EvalHost, *, run: EvalRun, template: EvalTemplate, test_cases: list[EvalTestCase], judge_service: JudgeService | None, options: RunnerOptions, callbacks: RunCallbacks | None = None, budget_gate: Callable[[float | None], CapBreach | None] | None = None, on_cost: Callable[[float | None], None] | None = None, cell_sink: list[CellSummary] | None = None) -> list[CellSummary]`
- **`fold_metered_cell`** · function · Fold one cell's action-seam tally into the external role's rows.
  <br>`fold_metered_cell(external_usage: RoleUsageLedger, cell: MeteredCallTally | None) -> None`
- **`get_judge_config`** · function · Load a judge config by id within a scope.
  <br>`get_judge_config(storage: DefinitionStore, config_id: str, scope_id: str) -> JudgeConfig`
- **`get_result`** · function · Load one eval result by id within its scope.
  <br>`get_result(storage: ResultStore, result_id: str, scope_id: str) -> EvalResult`
- **`get_result_trace`** · function · Load one result's trace payload, or `None` when it stored none.
  <br>`get_result_trace(storage: ResultStore, result: EvalResult) -> EvalTrace | None`
- **`get_rubric_dim`** · function · Load a catalog rubric dim by id within a scope.
  <br>`get_rubric_dim(storage: DefinitionStore, dim_id: str, scope_id: str) -> CatalogRubricDim`
- **`get_run`** · function · Load an eval run by id within its scope, whole.
  <br>`get_run(storage: RunStore, run_id: str, scope_id: str) -> EvalRun`
- **`get_template`** · function · Load a template by id within a scope.
  <br>`get_template(host: EvalHost, template_id: str, scope_id: str) -> EvalTemplate`
- **`grade_goal_checks`** · function · Grade goal checks against a call ledger, an end state and what fired: the one evaluation every caller uses.
  <br>`grade_goal_checks(expressions: Sequence[str], *, ledger: CallLedger, end_state: Mapping[str, Any], fired: Firings | None, variation: Mapping[str, Any], world: WorldRegistry | None) -> list[GoalStateOutcome]`
- **`launch_as_group`** · async function · Launch one group of runs: admit it, prepare every arm, and start them together — or start none.
  <br>`launch_as_group(host: LaunchHost, n_runs: int, *, settings: LaunchSettings, form: Callable[[], Awaitable[tuple[LaunchGroup, _Formed]]], prepare: Callable[[LaunchGroup, _Formed], Awaitable[list[EvalRun]]], event: str, admission: AdmissionTicket | None = None, attach: Callable[[list[EvalRun]], Awaitable[None]] | None = None, detach: Callable[[list[EvalRun]], Awaitable[None]] | None = None) -> list[EvalRun]`
- **`launch_run`** · async function · Assemble, stamp, bound and hand over a run — the launch tail every candidate kind shares.
  <br>`launch_run(host: LaunchHost, request: LaunchRequest, wiring: KindWiring) -> EvalRun`
- **`list_judge_configs`** · function · List judge configs in a scope.
  <br>`list_judge_configs(storage: DefinitionStore, scope_id: str, *, rubric_dim_id: str | None = None, archived: bool = False) -> list[JudgeConfig]`
- **`list_results`** · function · List every result for one eval run within its scope.
  <br>`list_results(storage: ResultStore, run_id: str, scope_id: str) -> list[EvalResult]`
- **`list_rubric_dims`** · function · List catalog rubric dims in a scope.
  <br>`list_rubric_dims(storage: DefinitionStore, scope_id: str, *, axis: str | None = None, universal: bool | None = None, archived: bool = False) -> list[CatalogRubricDim]`
- **`list_runs`** · function · List eval runs in a scope, optionally filtered by status.
  <br>`list_runs(host: EvalHost, scope_id: str, *, status: str | None = None, include_archived: bool = False) -> list[EvalRun]`
- **`list_templates`** · function · List templates in a scope.
  <br>`list_templates(storage: DefinitionStore, scope_id: str, *, archived: bool = False, required_tool: str | None = None, universal: bool | None = None) -> list[EvalTemplate]`
- **`load_run_as_listed`** · function · Load one run the way a listing loads it: without the payload paths the host declares a listing leaves out.
  <br>`load_run_as_listed(storage: CurationStore, run_id: str, scope_id: str, *, profile: HostProfile) -> EvalRun`
- **`load_seed_corpus`** · function · Read a corpus off disk, constructing each document through its model in `scope_id`.
  <br>`load_seed_corpus(seed_dir: Path, scope_id: str) -> SeedCorpus`
- **`metered_cell_tally`** · function · One cell's slice of the run's metered-call tally, or `None` when nothing counted.
  <br>`metered_cell_tally(ledger: MeteredCallLedger | None, baseline: MeteredCallTally | None) -> MeteredCallTally | None`
- **`no_launcher_for`** · function · The refusal for a template whose kind this host cannot launch — one wording, wherever it fires.
  <br>`no_launcher_for(template_id: str, candidate_kind: str, launchable: Mapping[str, Any]) -> ValidationFailedError`
- **`plan_judge`** · function · The judges one arm of `template` will be scored by, resolved as `build_judge_service` resolves them — building nothing.
  <br>`plan_judge(host: EvalHost, template: EvalTemplate, judge_model: str, selection: Mapping[str, str] | None = None, *, judged_artifact: JudgedArtifact) -> PlannedJudge`
- **`precondition_failure_text`** · function · One line naming what was presumed and what the world held instead.
  <br>`precondition_failure_text(outcomes: Sequence[PreconditionOutcome]) -> str`
- **`price_arms`** · async function · Plan and price every arm of one launch by the engine's one rule, refusing before any launcher runs.
  <br>`price_arms(host: LaunchHost, launchable: LaunchableKind, requests: Sequence[LaunchRequest]) -> list[LaunchRequest]`
- **`quote_launch`** · async function · What `start_run` with the same arguments would make of its arms' prices — read-only.
  <br>`quote_launch(host: LaunchHost, *, template_id: str, subject_id: str, models: list[str], k_runs: int = 3, n_variations: int = 0, variation_model: str | None = None, judge_model: str | None = None, judge_config_ids: dict[str, str] | None = None, simulator_model: str | None = None, cassette_mode: str | None = 'off', cassette_corpus_id: str | None = None, overlays: Mapping[str, Any] | None = None, apparatus_settings: Mapping[str, Any] | None = None, max_cost_usd: float | None = None, max_metered_calls: int | None = None, scope_id: str, case_count: int | None = None) -> LaunchQuote`
- **`rate_result`** · function · Record one rater's score for one judged dimension of one result — a person's, or an agent's.
  <br>`rate_result(storage: RatingStore, *, result_id: str, scope_id: str, rubric_dim: str, rater: str, rater_kind: RaterKind, score: int, reason: str) -> CalibrationRating`
- **`recheck_goal_states`** · function · Re-grade every result of one stored run from what its cells stored, and optionally store the new verdicts.
  <br>`recheck_goal_states(store: RecheckStore, run_id: str, scope_id: str, *, world: WorldRegistry | None, apply: bool) -> RunRecheck`
- **`recheck_result`** · function · Re-grade one stored result's goal checks against what its cell stored.
  <br>`recheck_result(result: EvalResult, *, ledger: CallLedger | None, end_state: Mapping[str, Any] | None, variation: Mapping[str, Any] | None, world: WorldRegistry | None, provenance: ApparatusProvenance) -> tuple[ResultRecheck, list[GoalStateOutcome] | None]`
- **`record_witnessed_cell`** · async function · Record one cell a host observed — a session it witnessed, not one the engine ran — as a result and its trace.
  <br>`record_witnessed_cell(host: EvalHost, run: EvalRun, test_case: EvalTestCase, output: CandidateOutput, *, k_iteration: int, result_id: str, scored_at: str, judged_artifact: JudgedArtifact, external_rates: ExternalRateTable | None = None, spans: CellTrace | None = None, world_events: Sequence[WorldEvent] | None = None, end_state: Mapping[str, Any] | None = None, judging: WitnessedJudging | None = None) -> tuple[EvalResult, EvalTrace]`
- **`refuse_raised_ceiling`** · function · Refuse a per-run override above the host's configured ceiling — the one rule every surface applies.
  <br>`refuse_raised_ceiling(override: Ceiling | None, *, configured: Ceiling, name: str, configured_name: str) -> None`
- **`refuse_stale_presumptions`** · function · Refuse a template presuming world state this host's registry no longer declares.
  <br>`refuse_stale_presumptions(template: EvalTemplate, *, profile: HostProfile) -> None`
- **`rejudge_result`** · async function · Re-score a finished result's failed judge dimensions from the evidence its judge read.
  <br>`rejudge_result(host: EvalHost, result_id: str, scope_id: str) -> EvalResult`
- **`repeat_judge_scores`** · async function · Ask a finished run's judge to score its results' scored dims again, and record each repeat on its result.
  <br>`repeat_judge_scores(host: EvalHost, run_id: str, scope_id: str, *, out_of_run_cap_usd: float | None, result_ids: Sequence[str] | None = None) -> JudgeRepeatReport`
- **`reproducible_judge_inputs`** · function · Load what a result's judge calls read, refusing any input its run did not record.
  <br>`reproducible_judge_inputs(storage: JudgeInputStore, result: EvalResult, run: EvalRun, scope_id: str, *, request_settings: RequestSettingsPolicy, config_dims: Collection[str] | None = None) -> ReproducibleJudgeInputs`
- **`require_candidate_model`** · function · The model `request`'s arm runs on — the one it named, or the kind's role default — or the refusal.
  <br>`require_candidate_model(request: LaunchRequest, default: str | None) -> str`
- **`require_delete_confirmation`** · function · Refuse a destructive eval delete unless the caller echoed the target's id.
  <br>`require_delete_confirmation(kind: str, object_id: str, confirm: str | None, *, cascade: str | None = None, alternative: str = 'archive it instead to exclude it from cohorts without destroying it') -> None`
- **`resolve_ceiling_origin`** · function · Return which tier of the cascade supplied the ceiling a run is bounded by.
  <br>`resolve_ceiling_origin(override: float | None, *, enforcement_enabled: bool) -> CostCapOrigin`
- **`resolve_constructor`** · function · Import a contract's constructor.
  <br>`resolve_constructor(contract: FidelityContract) -> object`
- **`resolve_effective_ceiling`** · function · Return the ceiling a run is actually bounded by, or `None` when nothing bounds it.
  <br>`resolve_effective_ceiling(override: Ceiling | None, *, configured: Ceiling, enforcement_enabled: bool) -> Ceiling | None`
- **`resolve_judge_pin`** · function · The run-level judge pin an arm is scored under: the launch's, or the role default stepped off a candidate.
  <br>`resolve_judge_pin(request: LaunchRequest, role_default: str, *, candidate_model: str) -> str`
- **`run_blocking`** · async function · Run `fn(*args, **kwargs)` on `executor` and await its result.
  <br>`run_blocking(executor: Executor | None, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T`
- **`run_judge_llm`** · async function · Run an LLM judge call against a set of criteria and parse the response.
  <br>`run_judge_llm(client: Any, system_prompt: str, user_prompt: str, criteria_dicts: list[dict[str, Any]], label: str, case_id: str, *, cannot_tell_offered: bool = False, failure_describer: ProviderFailureDescriber | None = None) -> dict[str, Any] | None`
- **`sample_concurrent_eval_jobs`** · function · Sample how many eval jobs are executing, keeping the busiest observation so far (R4).
  <br>`sample_concurrent_eval_jobs(options: RunnerOptions, previous: int | None = None) -> int | None`
- **`seed_eval_definitions`** · function · Create any corpus definition whose natural key is absent from the corpus's scope.
  <br>`seed_eval_definitions(host: EvalHost, corpus: SeedCorpus, *, require_known_tools_allowed: Callable[[Sequence[str] | None], None], refuse_undeclared_world_seed: Callable[[EvalTemplate], None], refuse_undeliverable_template: Callable[[EvalTemplate], None]) -> SeedOutcome`
- **`set_analysis_archived`** · function · Archive or un-archive a stored analysis — the alternative to destroying it.
  <br>`set_analysis_archived(storage: CurationStore, analysis_id: str, scope_id: str, *, archived: bool, reason: str | None = None) -> EvalAnalysis`
- **`set_campaign_archived`** · function · Archive or un-archive a campaign — retire it from listings without destroying anything.
  <br>`set_campaign_archived(storage: CurationStore, campaign_id: str, scope_id: str, *, archived: bool) -> EvalCampaign`
- **`set_run_archived`** · function · Archive or un-archive a run — reversible exclusion from every cohort.
  <br>`set_run_archived(storage: CurationStore, run_id: str, scope_id: str, *, archived: bool, profile: HostProfile) -> EvalRun`
- **`settable_apparatus`** · function · The apparatus dimensions a launch may set: every one the host declares, but none of the engine's.
  <br>`settable_apparatus(registry: SweepableRegistry) -> frozenset[str]`
- **`stamp_witnessed_judge`** · function · A witnessed run that names the template its cells are judged against, the judge apparatus that scores them, and the ceiling it is held to.
  <br>`stamp_witnessed_judge(host: EvalHost, run: EvalRun, template: EvalTemplate, *, judge_model: str, judged_artifact: JudgedArtifact, selection: dict[str, str] | None = None, configured_max_cost_usd: float, enforcement_enabled: bool, max_cost_usd: float | None = None) -> EvalRun`
- **`start_run`** · async function · Refuse what no kind can run, admit the launch, and dispatch each arm to its kind's launcher.
  <br>`start_run(host: LaunchHost, *, template_id: str, subject_id: str, models: list[str], k_runs: int = 3, n_variations: int = 0, variation_model: str | None = None, judge_model: str | None = None, judge_config_ids: dict[str, str] | None = None, simulator_model: str | None = None, cassette_mode: str | None = 'off', cassette_corpus_id: str | None = None, overlays: Mapping[str, Any] | None = None, apparatus_settings: Mapping[str, Any] | None = None, max_cost_usd: float | None = None, max_metered_calls: int | None = None, scope_id: str, launch_group: LaunchGroup | None = None, admission: AdmissionTicket | None = None) -> list[EvalRun]`
- **`start_universal_battery`** · async function · Launch the operator-curated boundary battery against one subject.
  <br>`start_universal_battery(host: LaunchHost, subject_id: str, *, scope_id: str, models: list[str], k_runs: int = 3, n_variations: int = 0, variation_model: str | None = None, judge_model: str | None = None, simulator_model: str | None = None, cassette_mode: str | None = 'off', apparatus_settings: Mapping[str, Any] | None = None, max_cost_usd: float | None = None, preflight: BatteryPreflight) -> list[str]`
- **`sweep_abandoned_runs`** · function · Cancel runs this process cannot own, left non-terminal by a previous one.
  <br>`sweep_abandoned_runs(storage: RunRecordStore, scopes: Iterable[str], *, job_manager: EvalJobManager | None) -> AbandonedRunSweepReport`
- **`update_judge_config`** · function · Re-author a judge config via archive-and-recreate (immutable versioning).
  <br>`update_judge_config(storage: DefinitionStore, config_id: str, scope_id: str, fields: dict[str, Any]) -> JudgeConfig`
- **`update_rubric_dim`** · function · Apply a partial update to a catalog rubric dim and persist it.
  <br>`update_rubric_dim(storage: DefinitionStore, dim_id: str, scope_id: str, fields: dict[str, Any]) -> CatalogRubricDim`
- **`update_template`** · function · Apply a partial update to a template and persist it.
  <br>`update_template(host: EvalHost, template_id: str, scope_id: str, fields: dict[str, Any], *, require_known_tools_allowed: Callable[[Sequence[str] | None], None], refuse_undeclared_world_seed: Callable[[EvalTemplate], None], refuse_undeliverable_template: Callable[[EvalTemplate], None]) -> EvalTemplate`
- **`validated_kind_spec`** · function · The template's `kind_spec` as its kind's spec model validates it, refused by field.
  <br>`validated_kind_spec(template: EvalTemplate, *, profile: HostProfile) -> BaseModel | None`

**Classes**

- **`AbandonedRunSweepReport`** · model · Outcome of one startup reclaim of runs left non-terminal by a dead process.
- **`AccountExhaustedError`** · exception · Raised by `execute_run` when the run's paying account refuses a call.
- **`AdmissionTicket`** · class · Room a launch has reserved for runs it has not yet handed to the manager.
- **`ArmPlan`** · dataclass · What one arm of a launch will run, as its kind says before its launcher runs.
- **`ArmPrice`** · dataclass · What the host's pricer predicts one arm will cost, and how it knows.
- **`ArmQuote`** · dataclass · One arm of a launch, as the engine asks the host's pricer to price it — stored-case arms and generating ones alike.
- **`ArmVerdict`** · dataclass · One arm of a launch as the engine's one pricing rule judged it: its plan, its price, its cap and the outcome.
- **`BudgetStoppedError`** · exception · Raised by `execute_run` when the cap trips mid-run.
- **`CandidateTurn`** · dataclass · One candidate-side response captured by the kind and fed back into the simulator.
- **`CapBreach`** · dataclass · The numbers that justify a graceful budget stop.
- **`CeilingRaisedError`** · exception · A launch named a per-run ceiling ABOVE the host's configured one.
- **`CellContext`** · dataclass · What the runner knows about one cell when it asks a kind's factory for the cell's kind.
- **`CheckFlip`** · model · One goal check whose verdict a re-grade changes.
- **`CurationStore`** · protocol · Everything the curation family reads, writes and destroys — and nothing else.
- **`ErrorLedger`** · dataclass · Accumulates a cell's errors, categorized candidate-vs-infra at the source.
- **`EvalJobManager`** · class · Manages background async eval jobs.
- **`EvalJobTimeout`** · exception · A job outlived the wall-clock budget its timeout context was enforcing.
- **`EvalRunCostCap`** · class · In-memory per-run cost cap for a single eval run.
- **`EveryCellApparatusFailedError`** · exception · Raised by `execute_run` when an apparatus fault excluded every cell of the run.
- **`FidelityContract`** · dataclass · One shared construction path and the callers required to reach it.
- **`GoalCheckUnevaluable`** · exception · A goal check raised while being evaluated — a fault of the rig, not a verdict on the candidate.
- **`JobTimeoutFactory`** · protocol · Builds the context manager a single job runs inside.
- **`JudgeContext`** · dataclass · Everything a single-dim judge call needs about the result under review.
- **`JudgeInputStore`** · protocol · The reads `reproducible_judge_inputs` makes, and no more.
- **`JudgeOutcome`** · dataclass · Result of one single-dim judge call.
- **`JudgeRepeatEstimate`** · model · What repeating a run's judge scores would be priced at, against the cap it would be held to — no call made.
- **`JudgeRepeatReport`** · model · What a repeat did: which results it recorded a repeat on, what it spent, and what it left out.
- **`JudgeRepeatSkip`** · model · A result of the run that is not repeated, and why.
- **`JudgeRequest`** · class · One judge call as it is sent: what `JudgeService._score` hands the client.
- **`JudgeService`** · class · Stateless single-dim judge. See module docstring for the contract.
- **`KeptOutcome`** · model · One stored outcome a re-check left as it was, and why.
- **`KindFactory`** · protocol · Builds the candidate kind for one cell.
- **`KindWiring`** · dataclass · What one kind's launcher resolved for its arm — everything `launch_run` cannot know itself.
- **`LaunchableKind`** · dataclass · One entry of the launch registry a host hands `start_run`: how a kind launches, and what it refuses.
- **`LaunchGroup`** · class · The runs of one launch, prepared together and started together — or not at all.
- **`LaunchHost`** · dataclass · An `EvalHost`, and what starting its runs needs.
- **`LaunchPricer`** · protocol · The host's prediction of what one arm of a launch will cost, before any launcher runs.
- **`LaunchQuote`** · dataclass · What a launch would make of its arms' prices, made by the launch's own steps and launching nothing.
- **`LaunchRequest`** · dataclass · One arm's launch as `start_run` resolved it, handed to the kind's own launcher.
- **`LaunchSettings`** · model · The host's launch settings, as one snapshot of values.
- **`MeteredCallLedger`** · class · In-memory per-run ceiling on metered third-party calls.
- **`MeteredCallTally`** · dataclass · What a run (or one cell's slice of it) metered, kept per provider.
- **`PlannedJudge`** · dataclass · The judge one arm will be scored by, resolved before anything is built: what `plan_judge` returns.
- **`RatingStore`** · protocol · The two calls a rating makes: read the rated result, write the rating.
- **`RecheckStore`** · protocol · The reads and the one rewrite a re-check makes, and no more.
- **`ReproducibleJudgeInputs`** · class · The stored records a result's judge calls are rebuilt from, each the one its run recorded.
- **`ResultRecheck`** · model · What a re-check found for one stored result.
- **`RunCallbacks`** · dataclass · Optional progress / persistence hooks for the run loop.
- **`RunJudge`** · dataclass · A judged run's judge, as `build_judge_service` resolved it from one config load.
- **`RunnerOptions`** · dataclass · The run's own knobs that don't fit on the run document — per-run values, never host wiring.
- **`RunRecheck`** · model · What a re-check of one run found, and whether it was written.
- **`SeedCorpus`** · dataclass · A host's authored definitions for one scope, already constructed through their production models.
- **`SeedOutcome`** · dataclass · What one seeding pass did, per doc type.
- **`SimulatorCall`** · dataclass · One simulator-role model call the driver made, for the cell's `simulator` usage row.
- **`SimulatorReplyInvalid`** · exception · A simulator-role reply did not match the schema it was sent.
- **`SimulatorTurn`** · dataclass · One user-side utterance produced by the simulator.
- **`TurnDriver`** · dataclass · Multi-actor round scheduler and utterance generator.
- **`WitnessedJudging`** · class · One judged witnessed run's judging in this process: its ceiling checks, one cell at a time, and what it judged unsaved.

**Types**

- **`ArmOutcome`** · literal · What the one pricing rule made of an arm: admitted under its cap; refused (predicted above it, or unpriceable under an inherited one); unpriceable and run under a cap its launch named; or not held to any cap, the host enforcing none.
  <br>`'admitted'` | `'refused'` | `'unpriced-under-chosen-cap'` | `'uncapped'`
- **`BatteryPreflight`** · type alias · Prepares a battery's pre-flight for one subject and the battery's models, once, and returns the per-template check.
  <br>`Callable[[str, Sequence[str]], Awaitable[TemplatePreflight]]`
- **`JudgeClientFactory`** · type alias · Builds a judge LLM client for a `(model, temperature)` pair.
  <br>`Callable[[str | None, float | None], Any]`
- **`KindLauncher`** · type alias · One kind's launcher: builds that kind's collaborators for one arm and hands `launch_run` its run.
  <br>`Callable[[LaunchRequest], Awaitable[EvalRun]]`
- **`LaunchArgument`** · literal · A launch argument a kind may declare it cannot honour.
  <br>`'n_variations'` | `'judge_model'` | `'judge_config_ids'` | `'simulator_model'` | `'cassette_mode'`
- **`RequestSettingsPolicy`** · literal · How a caller treats the run's recorded judge request settings.
  <br>`'today'` | `'as_recorded'`
- **`TemplatePreflight`** · type alias · Checks one of a battery's templates the way its launch would, before any template launches.
  <br>`Callable[['EvalTemplate', str], Awaitable[None]]`
- **`WorkFn`** · type alias · A job's work: an async function handed its progress callback.
  <br>`Callable[[ProgressFn], Awaitable[None]]`

**Constants**

- **`JUDGE_CALL_ATTEMPTS`** · constant (int) · Calls one judge dimension can make: the first, plus its parse retries.
  <br>`= 2`
- **`JUDGE_MAX_TOKENS`** · constant (int) · The eval judge's output cap: the reasoning budget plus the answer budget, DERIVED rather than chosen, so it always sits above the ceiling it wraps.
  <br>`= 10240`
- **`JUDGE_REQUEST_SETTINGS`** · constant (ClientRequestSettings) · The judge role's request settings as ONE value: what the host's client builder applies to every eval judge client, and what a run launched with a judge records as `judge_request_settings`.
- **`RUBRIC_DIM_SERVER_FIELDS`** · constant (tuple) · Identity/lifecycle fields the rubric-dim family owns — never taken from caller input.
- **`SCHEDULER_CALL_ATTEMPTS`** · constant (int) · How many calls one `llm_decided` scheduling decision may make: the first, and one repair that names what was wrong with it.
  <br>`= 2`
- **`SIMULATOR_ANSWER_BUDGET_TOKENS`** · constant (int) · Output room kept for the simulated user's visible reply: the strict-schema JSON object holding one utterance or one scheduling pick.
  <br>`= 4096`
- **`SIMULATOR_MAX_TOKENS`** · constant (int) · The simulator's output cap: the reasoning allowance plus the answer budget, derived rather than chosen. It is the only bound in tokens a simulator call has.
  <br>`= 5120`
- **`SIMULATOR_REASONING_ALLOWANCE_TOKENS`** · constant (int) · Room the output cap leaves for the simulated user's private reasoning at `SIMULATOR_REASONING_EFFORT`.
  <br>`= 1024`
- **`SIMULATOR_REASONING_EFFORT`** · constant (str) · How hard the simulated user reasons: the router's `reasoning.effort`, at its lowest level that still reasons.
  <br>`= 'minimal'`
- **`SIMULATOR_REQUEST_SETTINGS`** · constant (ClientRequestSettings) · The simulator role's request settings as ONE value: what the host's client builder applies to the simulated user's client, and what a run launched with a simulated user records as `simulator_request_settings`.
- **`TEMPLATE_SERVER_FIELDS`** · constant (tuple) · Identity/lifecycle fields the template family owns — never taken from caller input.

<a id="api-analysis"></a>
### `threetears.evals.analysis`

The engine's analysis package: campaigns, context bundles, generated analyses and the read lenses.

**Functions**

- **`add_runs_to_campaign`** · function · Attach runs to a campaign (de-duplicated, order-preserving) and persist.
  <br>`add_runs_to_campaign(storage: CampaignStore, campaign_id: str, scope_id: str, run_ids: list[str]) -> EvalCampaign`
- **`analysis_gen_request_settings_for`** · function · The generator role's request settings as ONE value, from the host's two budgets.
  <br>`analysis_gen_request_settings_for(*, answer_budget_tokens: int, reasoning_budget_tokens: int) -> ClientRequestSettings`
- **`analysis_report`** · function · Read one stored analysis as its report — the one document every surface renders.
  <br>`analysis_report(storage: AnalysisStore, analysis_id: str, scope_id: str) -> Report`
- **`assemble_context_bundle`** · function · Assemble the closed context bundle for a campaign's runs.
  <br>`assemble_context_bundle(campaign: EvalCampaign, *, storage: CampaignReadStore, insights_as_of: str | None = None, profile: HostProfile) -> AnalysisContextBundle`
- **`build_code_only_report`** · function · Lay a campaign's assembled evidence out as a report, when no analysis exists to report through.
  <br>`build_code_only_report(bundle: AnalysisContextBundle, *, measures: MeasureRegistry, assembled_at: str, campaign_name: str | None = None) -> Report`
- **`build_report`** · function · Lay one stored analysis out as a report.
  <br>`build_report(analysis: EvalAnalysis) -> Report`
- **`campaign_report`** · function · The campaign's report — THE answer to "what is this campaign's report", for every caller.
  <br>`campaign_report(host: EvalHost, campaign_id: str, scope_id: str) -> Report`
- **`cell_label`** · function · Name one cell — an arm under one rig — from the index and the multi-rig population.
  <br>`cell_label(variant_key: str, apparatus_class_id: str, *, index: Mapping[str, VariantIndexEntry], multi_rig: frozenset[str]) -> str`
- **`compare_two_runs`** · function · Diff two runs into the side-by-side compare view.
  <br>`compare_two_runs(storage: LensStore, run_a_id: str, run_b_id: str, scope_id: str, *, load_template: Callable[[str], EvalTemplate], subject_detail: Callable[[EvalRun], dict[str, dict[str, Any]]], rubric_threshold: int = 3) -> dict[str, Any]`
- **`comparison_sets`** · function · Group a scope's runs into sets that may honestly be compared.
  <br>`comparison_sets(storage: LensStore, scope_id: str, *, list_runs: RunLister, status: str | None = 'completed', full_windows: bool = False, campaign_id: str | None = None, run_ids: list[str] | None = None, profile: HostProfile) -> dict[str, Any]`
- **`completeness_disclosure`** · function · The one sentence a surface must show about a run that came up short.
  <br>`completeness_disclosure(completeness: RunCompleteness | None) -> str | None`
- **`component_carrier`** · function · The component SHOWN to carry the whole's movement, or None where the data cannot name one.
  <br>`component_carrier(whole: MeasureMovement, components: Sequence[MeasureMovement], level_a: Mapping[str, Mapping[str, Fraction]], level_b: Mapping[str, Mapping[str, Fraction]]) -> MeasureMovement | None`
- **`create_campaign`** · function · Create and persist a campaign from an authoring definition.
  <br>`create_campaign(storage: CampaignStore, definition: dict[str, Any], *, scope_id: str, created_by: str, profile: HostProfile, control_from_run_id: str | None = None) -> EvalCampaign`
- **`declarable_axes`** · function · The axes a campaign may declare on this host, read off the registry the gate decides with.
  <br>`declarable_axes(profile: HostProfile) -> DeclarableAxes`
- **`describe_insight_id_filters`** · function · Name the id filters an insight read was narrowed by, for a caller-facing sentence.
  <br>`describe_insight_id_filters(subject_id: str | None, source_campaign_id: str | None) -> str`
- **`difference_was_declared_at_launch`** · function · Whether every one of these runs got its value from its own launch declaration.
  <br>`difference_was_declared_at_launch(origins: Iterable[str | None]) -> bool`
- **`estimate_analysis_generation`** · async function · Price a generation's first call against the cap it would be held to, and make no call.
  <br>`estimate_analysis_generation(host: EvalHost, campaign_id: str, scope_id: str, *, model: str | None, resolve_prompt: Callable[[], Awaitable[str]], out_of_run_cap_usd: float | None) -> AnalysisGenerationEstimate`
- **`export_results`** · function · Serialize the projection's flat rows for a scope as CSV or JSON.
  <br>`export_results(storage: LensStore, scope_id: str, *, list_runs: RunLister, fmt: str | None = None, status: str | None = 'completed', run_ids: list[str] | None = None, profile: HostProfile) -> ScoreExport`
- **`finding_chart_intent`** · function · Decide one finding's chart, for a surface that draws a single chart at a time with its own renderer.
  <br>`finding_chart_intent(storage: AnalysisStore, analysis_id: str, scope_id: str, finding_id: str) -> ChartIntent`
- **`first_request`** · function · The exact first request a generation sends: system prompt, user message and the contract.
  <br>`first_request(bundle: AnalysisContextBundle, prompt: str, profile: HostProfile) -> tuple[str, str, dict[str, Any]]`
- **`format_number`** · function · Spell a number for a reader, the same way on every analysis surface.
  <br>`format_number(value: float | None) -> str`
- **`format_signed`** · function · Spell a change, carrying its sign — the direction is half of what a delta says.
  <br>`format_signed(value: float | None) -> str`
- **`format_significance`** · function · The read plus the statistics behind it, as one cell a surface prints verbatim.
  <br>`format_significance(*, significant: bool | None, paired: bool, p: float | None = None, effect: float | None = None, n: int | None = None, hedges: bool = False) -> str`
- **`freeze_reporter_case`** · function · Freeze a campaign's analysis bundle into a reporter case of `template_id`.
  <br>`freeze_reporter_case(host: EvalHost, *, template_id: str, campaign_id: str, scope_id: str, analysis_id: str | None = None, labels: Sequence[Any] = (), supersedes: Sequence[str] = (), load_template: Callable[[str], EvalTemplate]) -> EvalTestCase`
- **`frontier`** · function · Rank each subject's variants on quality x cost x latency, cheapest above bar.
  <br>`frontier(storage: LensStore, scope_id: str, *, list_runs: RunLister, bar: float | str | None = None, subject_id: str | None = None, status: str | None = 'completed') -> dict[str, Any]`
- **`frozen_case_receipt`** · function · Project a stored reporter case onto the receipt a freeze answers with.
  <br>`frozen_case_receipt(test_case: EvalTestCase) -> FrozenReporterCase`
- **`generate_analysis`** · async function · Generate one campaign's analysis from its context bundle, in one LLM call or two.
  <br>`generate_analysis(bundle: AnalysisContextBundle, *, prompt: str, model: str, client: CompletionGenerator, prompt_id: str, bundle_assembled_at: str, prompt_version: str | None = None, tally: GenerationTally | None = None, admit: CallAdmission | None = None, profile: HostProfile) -> tuple[EvalAnalysis, list[EvalInsight]]`
- **`generation_ceiling_s`** · function · The wall-clock ceiling of one `generate_analysis`, derived from the ceilings it wraps.
  <br>`generation_ceiling_s(*, request_s: RequestCeiling, generator_max_tokens: int) -> float`
- **`get_analysis`** · function · Load a stored analysis by id.
  <br>`get_analysis(storage: AnalysisStore, analysis_id: str, scope_id: str) -> EvalAnalysis`
- **`get_campaign`** · function · Load a campaign by id.
  <br>`get_campaign(storage: CampaignStore, campaign_id: str, scope_id: str) -> EvalCampaign`
- **`get_campaign_view`** · function · Load a campaign and enrich it with its read-time-derived window.
  <br>`get_campaign_view(storage: CampaignStore, campaign_id: str, scope_id: str) -> CampaignView`
- **`history`** · function · Series one measure over time per contestant, flagging real regressions.
  <br>`history(storage: LensStore, scope_id: str, *, list_runs: RunLister, metric: str | None = None, min_absolute_change: float = 0.0, min_relative_change: float = 0.0, subject_id: str | None = None, status: str | None = 'completed', profile: HostProfile) -> HistoryResult`
- **`insight_standing`** · function · Classify each insight by where its minting analysis stands, asking the store once per analysis.
  <br>`insight_standing(insights: Iterable[EvalInsight], analysis_archived: Callable[[str], bool | None]) -> InsightStanding`
- **`inspect_analysis_bundle`** · function · Re-assemble the context bundle a stored analysis was generated over.
  <br>`inspect_analysis_bundle(host: EvalHost, analysis_id: str, scope_id: str) -> BundleInspection`
- **`inspect_campaign_bundle`** · function · Assemble a campaign's context bundle and return it, without generating.
  <br>`inspect_campaign_bundle(host: EvalHost, campaign_id: str, scope_id: str) -> BundleInspection`
- **`judge_agreement`** · function · Pair each rating with the judge's score on the same dimension of the same result, and read agreement.
  <br>`judge_agreement(ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]) -> JudgeAgreement`
- **`judge_evidence_tiers`** · function · Decide the evidence tier of every judge's readings on every dimension, from the two agreements.
  <br>`judge_evidence_tiers(agreement: JudgeAgreement, self_agreement: JudgeSelfAgreement, judged: Iterable[JudgeKey]) -> list[JudgeEvidenceTier]`
- **`judge_key`** · function · The judge behind `result`'s score on `dim`, or None when it holds no score there.
  <br>`judge_key(result: EvalResult, dim: str) -> JudgeKey | None`
- **`judge_phase_ceiling_s`** · function · The wall-clock ceiling of one cell's judge phase, derived from the ceilings of its requests.
  <br>`judge_phase_ceiling_s(*, judge_dims: int, judge_concurrency: int, judge_call_attempts: int, judge_max_tokens: int, request_s: RequestCeiling) -> float`
- **`judge_self_agreement`** · function · Pair each repeated score with the first score it repeated, and read agreement the way calibration does.
  <br>`judge_self_agreement(results: Iterable[EvalResult]) -> JudgeSelfAgreement`
- **`list_analyses`** · function · List every analysis attached to a campaign in a scope, newest first.
  <br>`list_analyses(storage: AnalysisStore, campaign_id: str, scope_id: str) -> list[EvalAnalysis]`
- **`list_analysis_attempts`** · function · List every generation attempt recorded against a campaign in a scope, newest first — stored and failed alike.
  <br>`list_analysis_attempts(storage: AnalysisStore, campaign_id: str, scope_id: str) -> list[EvalAnalysisAttempt]`
- **`list_campaigns`** · function · List campaigns in a scope, newest first.
  <br>`list_campaigns(storage: CampaignStore, scope_id: str, *, subject_id: str | None = None, behavior: str | None = None, archived: bool | None = None) -> list[EvalCampaign]`
- **`list_insights`** · function · List insights in a storage scope, newest observation first.
  <br>`list_insights(storage: AnalysisStore, scope_id: str, *, subject_id: str | None = None, scope: str | None = None, source_campaign_id: str | None = None) -> list[EvalInsight]`
- **`measure_movement`** · function · Test one measure's movement between two levels against its own noise, and read it against what matters.
  <br>`measure_movement(descriptor: MetricDescriptor, at_a: Mapping[str, Fraction], at_b: Mapping[str, Fraction]) -> MeasureMovement`
- **`metric_help`** · function · Render the metrics a surface accepts, each glossed and beside its catalog name, for its help text.
  <br>`metric_help(accepted: frozenset[str]) -> str`
- **`multi_rig_variants`** · function · The variants measured under more than one rig among `cells`.
  <br>`multi_rig_variants(cells: Iterable[CellFacts]) -> frozenset[str]`
- **`orphaned_runs`** · function · Report the scope's runs that no campaign holds, with their spend.
  <br>`orphaned_runs(storage: LensStore, scope_id: str, *, list_runs: RunLister) -> dict[str, Any]`
- **`pivot`** · function · Aggregate a scope's observations over any two coordinates.
  <br>`pivot(storage: LensStore, scope_id: str, *, list_runs: RunLister, row_factor: str, column_factor: str, metric: str | None = None, weighting: str | None = None, subject_id: str | None = None, status: str | None = 'completed', predicted_cost: CostEstimate | Sequence[PlannedCost] | Mapping[str, Any] | None = None, profile: HostProfile) -> PivotTable`
- **`prepare_analysis_generation`** · async function · Check and build everything a generation needs before it spends anything, its first call priced and admitted.
  <br>`prepare_analysis_generation(host: EvalHost, campaign_id: str, scope_id: str, *, model: str | None, resolve_prompt: Callable[[], Awaitable[str]], out_of_run_cap_usd: float | None) -> PreparedGeneration`
- **`program_budget`** · function · Report program-lens spend over a scope, excluding no run.
  <br>`program_budget(storage: LensStore, scope_id: str, *, list_runs: RunLister) -> dict[str, Any]`
- **`prompt_content_version`** · function · The version a generation records for its prompt — a short hash of what the model is told.
  <br>`prompt_content_version(prompt: str, profile: HostProfile, *, time_axis: bool) -> str`
- **`propose_bars`** · function · Propose a bar on every measure the baseline campaign's incumbent was measured on.
  <br>`propose_bars(host: EvalHost, baseline_campaign_id: str, *, scope_id: str) -> BaselineBarProposals`
- **`published_report_schema`** · function · The schema as published in `schema.json`.
  <br>`published_report_schema() -> dict[str, Any]`
- **`remove_runs_from_campaign`** · function · Detach runs from a campaign — the inverse of `add_runs_to_campaign`.
  <br>`remove_runs_from_campaign(storage: CampaignStore, campaign_id: str, scope_id: str, run_ids: list[str]) -> EvalCampaign`
- **`render_memo_as_written`** · function · The memo as authored: the model's words, code's figures, in one canonical layout a reader follows.
  <br>`render_memo_as_written(analysis: EvalAnalysis) -> str`
- **`report_html`** · function · Render a report as a standalone HTML page that needs no script.
  <br>`report_html(report: Report) -> str`
- **`report_json_schema`** · function · The report's JSON Schema, as generated from the model — what `schema.json` must equal.
  <br>`report_json_schema() -> dict[str, Any]`
- **`report_markdown`** · function · Render a report as Markdown.
  <br>`report_markdown(report: Report) -> str`
- **`reporter_calibration`** · function · Read a reporter run against its cases' labels — the rubric's calibration.
  <br>`reporter_calibration(storage: AnalysisStore, run_id: str, scope_id: str) -> ReporterCalibration`
- **`reporter_case_bank`** · function · Derive which of a reporter template's stored cases is live, refusing when that is undecidable.
  <br>`reporter_case_bank(storage: AnalysisStore, template_id: str, scope_id: str) -> ReporterCaseBank`
- **`reporter_cell_timeout_s`** · function · The wall-clock ceiling of one reporter cell, derived from the ceilings it wraps.
  <br>`reporter_cell_timeout_s(*, judge_dims: int, judge_concurrency: int, generator_max_tokens: int, judge_call_attempts: int, judge_max_tokens: int, request_s: RequestCeiling) -> float`
- **`run_analysis_generation`** · async function · Make the paid generator call(s) for a prepared generation, then store and record the result.
  <br>`run_analysis_generation(host: EvalHost, prepared: PreparedGeneration, *, prompt_id: str, max_output_tokens: int) -> tuple[EvalAnalysis, list[EvalInsight]]`
- **`run_summary`** · function · Compose a run's verdict numbers — pass^k, latency, cost — per model.
  <br>`run_summary(storage: LensStore, run_id: str, scope_id: str, *, load_run_listed: Callable[[str, str], EvalRun], row_columns: RowColumns, rubric_threshold: int = 3) -> dict[str, Any]`
- **`set_campaign_control`** · function · Designate (or clear) the campaign's control, ADDRESSED from a member observation.
  <br>`set_campaign_control(storage: CampaignStore, campaign_id: str, scope_id: str, run_id: str | None, *, set_by: str, profile: HostProfile) -> EvalCampaign`
- **`set_reporter_case_archived`** · function · Retire (archive) a reporter case, or restore one — the answer to a case that can no longer measure anything.
  <br>`set_reporter_case_archived(storage: ReporterCaseStore, test_case_id: str, scope_id: str, *, archived: bool, reason: str | None = None) -> EvalTestCase`
- **`short_digest`** · function · Cut a variant key or an apparatus class id to the `DIGEST_CHARS` that tell it apart.
  <br>`short_digest(digest: str) -> str`
- **`significance_disclosure`** · function · The sentence naming the test and threshold a comparison's verdicts rest on.
  <br>`significance_disclosure(*, paired: bool) -> str`
- **`tier_for_judges`** · function · The tier a reading stands on when `judges` served its scores: the weakest of theirs.
  <br>`tier_for_judges(tiers: Iterable[JudgeEvidenceTier], judges: Iterable[JudgeKey]) -> JudgedEvidenceTier`
- **`tier_sentence`** · function · One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.
  <br>`tier_sentence(tier: JudgeEvidenceTier) -> str`
- **`update_campaign`** · function · Amend a campaign's authored fields — the write half the declaration needs.
  <br>`update_campaign(storage: CampaignStore, campaign_id: str, scope_id: str, updates: dict[str, Any], *, updated_by: str, profile: HostProfile) -> EvalCampaign`
- **`variant_key_of_run`** · function · Which variant a run's observations carried — the authoring side of the control.
  <br>`variant_key_of_run(results: Sequence[EvalResult]) -> str | None`

**Classes**

- **`AmbiguousPair`** · dataclass · A (campaign, recorded memo) pair holding more than one live case.
- **`AnalysisContextBundle`** · model · The closed context bundle a generation prompt runs over.
- **`AnalysisGenerationEstimate`** · model · What a generation would be priced at before it starts, against the cap it would be held to.
- **`AnalysisStore`** · protocol · The storage calls the analysis service makes — its own, plus the bundle assembly's it hands the store to.
- **`ArmLevel`** · model · One coordinate of an arm — which lever, and the level it carried.
- **`ArmMeasurement`** · model · One measurement that placed at this arm, and the finding it was read from.
- **`ArmMechanismReading`** · model · One arm's mean of one observed-mechanism covariate, or the statement that it was not measured.
- **`ArmRow`** · model · One contestant, where it stands, and the evidence that placed there.
- **`ArmServedModel`** · model · Which model answered one arm's candidate calls, as the provider's responses named it.
- **`ArmTable`** · model · The derived comparison — every arm the campaign observed, and where it stands.
- **`AsRecordedReporterKind`** · class · The memo each case's campaign actually got, replayed — the calibration candidate.
- **`BaselineBarProposals`** · dataclass · What a baseline campaign proposes as its behavior's bars, and what it could not propose a bar on.
- **`BudgetedGenerator`** · class · A generation's calls, each priced and admitted against its out-of-run budget before it is sent, and ledgered.
- **`BundleInspection`** · model · A bundle handed to a reader instead of to a model — the read surface's payload.
- **`CalibrationCase`** · model · One frozen case of the run, with everything its results say against its labels.
- **`CalibrationCell`** · model · One result of one case: a single (model, repeat) pass, read against the case's labels.
- **`CampaignReadStore`** · protocol · The six reads assembling a campaign's context bundle needs.
- **`CampaignStore`** · protocol · Everything the campaign family reads and writes — and nothing else.
- **`CaseSetIdentity`** · model · One distinct case set inside a group, and which runs executed it.
- **`Cell`** · model · Every observation sharing one variant and one apparatus class.
- **`CellCoordinate`** · model · A cell named by its two coordinates and nothing else — an entry in a list of cells a fact holds for.
- **`ChartBlock`** · model · A chart: a finding's (its intent, or why the stored chart cannot be drawn), or one code chose.
- **`ComparedCell`** · model · One side of a family comparison: a cell, and the per-case values its test read.
- **`ComparisonFamily`** · model · Every comparison one declared question could draw a verdict from, corrected as one family.
- **`ComparisonSet`** · model · A group of runs that may be compared with each other, and on what basis.
- **`ComparisonSetsResult`** · model · Comparability groups plus the runs the caller's own scope left out.
- **`Confound`** · model · One dimension that did not hold still, and what state that fact is in.
- **`ConfusionCount`** · model · One cell of a confusion matrix: how often a case expecting one label was given another (or the same).
- **`CostEstimate`** · model · A proposed run/campaign's predicted cost, per model and in total, banded where n allows.
- **`CostEstimateCell`** · model · The predicted cost of running one proposed model, from its historical per-observation cost.
- **`DeclarableAxes`** · model · What a campaign may declare it sweeps on this host — the vocabulary the authoring gate reads.
- **`DesignArm`** · model · One arm of the campaign, the runs that measured it, and what it moved off the control.
- **`DimensionAgreement`** · model · How one judge's scores on one dimension agreed with people's ratings of the same results.
- **`DimensionReading`** · model · A dimension the judge scored that no label speaks to — reported, never compared.
- **`DisclosureBlock`** · model · Something code must tell the reader that no author wrote — one idea.
- **`Fact`** · model · A labelled fact code states beside an author's words — a confidence, a disposition, a tier.
- **`FamilyComparison`** · model · One contrast against the control on one reading, tested and corrected within its family.
- **`FrontierCostTie`** · model · A contestant that cleared the bar with a cost, which the verdict's pick was NOT shown cheaper than.
- **`FrontierDominator`** · model · One contestant shown to beat another point on every axis it measured, named as a ROW is named.
- **`FrontierPoint`** · model · One contestant's position on quality x cost x latency, within a subject.
- **`FrontierResult`** · model · The verdict surface across every subject, plus its disclosures.
- **`FrontierVerdict`** · model · The cheapest variant clearing the operator's bar for one subject — or, where the data cannot pick one, the set it is among.
- **`FrozenReporterCase`** · model · What a reporter-case freeze stored — the receipt every surface renders.
- **`GenerationError`** · exception · The generator's output could not be turned into a valid analysis.
- **`GenerationTally`** · model · What one generation has sent, spent and been refused so far — kept by the CALLER.
- **`GoalCheckProofReading`** · model · Whether one goal check the campaign's runs graded was shown to beat doing nothing.
- **`HeldFixedReading`** · model · What the campaign declared held still, beside what its runs say about the apparatus.
- **`HistoryResult`** · model · Per-measure longitudinal series across contestants, with its disclosures.
- **`InsightStanding`** · class · Where each listed insight's minting analysis stands — the two states a reader must be told.
- **`JudgeAgreement`** · model · Every rating read, paired with the judge where it can be, and agreement per dimension and judge.
- **`JudgedArm`** · model · One judged dimension's scores in one cell — the arm, measured under one rig.
- **`JudgedMeasure`** · model · A judged dimension, measured — carried so its exclusion from ranking is not read as its absence.
- **`JudgeKey`** · class · Who judged a reading: the dimension, its scale, the model that served the score, the config that asked and the temperature the call was sent at.
- **`JudgeSelfAgreement`** · model · Every repeated score read, paired with the first score where it can be, and agreement per dimension and judge.
- **`LabelCriterion`** · model · The rubric criterion a label was written against, frozen as text when its case was frozen.
- **`LabelReading`** · model · One reader's verdict on a dimension, beside what the judge scored there.
- **`LabelStatistics`** · model · One label's counts in a confusion matrix, and the precision, recall and F1 they give.
- **`LensStore`** · protocol · The storage reads the lenses make — results by scope and by run, campaigns, and one whole run.
- **`LeverCoverageInput`** · model · Structural coverage of one lever, as the bundle computes it.
- **`MeasurementWindow`** · model · The wall-clock span a run's cells were actually measured over — DERIVED.
- **`MeasureMovement`** · model · How one measure moved between two levels of a lever, and whether that movement separates from noise.
- **`MeasureSeries`** · model · One contestant's measure over time, within a subject.
- **`MechanismCheck`** · model · Whether the measure a swept lever declares it acts on measurably moved across the lever's levels.
- **`MeritTier`** · model · One axis of the declared merit priority, and the bars that give verdicts on it.
- **`MultipleComparisons`** · model · The campaign's comparisons, one corrected family per live declared question — or one for the whole campaign.
- **`NextExperiment`** · model · What recording one dimension would buy, in pooled observations.
- **`OpenAxisFamily`** · model · An open family of axes: a container whose members are declarable, though not enumerable.
- **`PivotCell`** · model · One (row, column) cell: the number, and everything needed to trust it.
- **`PivotTable`** · model · A two-factor pivot over one measure, with its disclosures attached.
- **`PlannedCost`** · model · One planned model's predicted cost over its planned observations, as a cost pivot's plan reads it.
- **`PredictedValue`** · model · A modelled estimate, kept beside the observed value it predicts and never fused with it.
- **`PreparedGeneration`** · class · Everything an analysis generation needs before it spends anything, checked and built.
- **`PreparedReporter`** · dataclass · One cell's reporter candidate: the model it binds, and the cell's tracing windows.
- **`ProjectionExclusions`** · model · What the projection dropped on the way to producing rows, and why.
- **`QuestionScope`** · model · Which verdicts bear on one declared question — read off the axes the question names.
- **`ReadingScope`** · model · Which readings the campaign's declared questions asked about — and which it reads only exploratorily.
- **`RealizedDesign`** · model · What kind of experiment this campaign turned out to be — DERIVED from the runs.
- **`RefusedMerge`** · model · Two cells that share a variant and did not pool, and which rule stopped them.
- **`RegressionFlag`** · model · A descriptive verdict on the change from the previous point in the series.
- **`Report`** · model · One analysis, as a document every surface renders. See the module docstring for the contract.
- **`ReporterCalibration`** · model · A reporter run read against its cases' labels — the rubric's calibration, as data.
- **`ReporterCase`** · model · What one reporter case pins: a frozen bundle, optionally the memo it got, and its labels.
- **`ReporterCaseBank`** · dataclass · One template's stored reporter cases, with which of them is live — derived once, read by every consumer.
- **`ReporterCaseStore`** · protocol · The three storage calls a reporter case's retirement and restore make — and nothing else.
- **`ReporterKind`** · class · A (prompt preset, generator model) candidate: generates a memo over each case's bundle.
- **`ReporterLabel`** · model · One reader's written verdict on a recorded memo, mapped to the dimension it bears on.
- **`ReportSource`** · model · What the report is a report of, and how that analysis was generated — or that none was.
- **`RunSummary`** · model · One run's compact digest — the levers it RAN at + its key telemetry.
- **`ScopeDivergence`** · model · Two scopes disagreeing about what one lever change did — a finding, not a caveat.
- **`ScoreExport`** · model · A projection's rows serialized for analysis elsewhere, with the account a CSV body cannot carry.
- **`SelfAgreementDimension`** · model · How one judge's repeated scores on one dimension agreed with its first scores of the same evidence.
- **`SeriesPoint`** · model · One run's aggregate for the measure — the trend's unit, with its denominators.
- **`ShortCell`** · model · A cell holding fewer repetitions than the declaration intended — a short run, stated per cell.
- **`SimpsonsFlag`** · model · A pooled column ranking that the per-row rankings mostly contradict.
- **`SoundnessRefusal`** · exception · A finished generator call whose OUTPUT was refused — the repairable half.
- **`SubjectFrontier`** · model · One subject's frontier: its points, and the verdict over them.
- **`SubjectKeyInstability`** · model · A subject key and its label disagreeing about how many things there are.
- **`SurfaceColumn`** · model · One numeric column of the table — an adjudicated bar, or a cost or latency measure.
- **`SurfaceRow`** · model · One measured cell — one arm under one rig — laid out against the table's columns.
- **`SurfaceRunNote`** · model · One member run of a cell that ran short of its design, or did not complete.
- **`SurfaceTable`** · model · The decision surface laid out — derived on every read, never stored.
- **`SurfaceUnadjudicatedBar`** · model · A bar no cell could be read against — listed below the table, never drawn as a column of misses.
- **`SurfaceValue`** · model · One cell's number in one column, already in the column's unit.
- **`TableBlock`** · model · A table code laid out — its columns, its rows in their stated order, and how much of it is shown.
- **`TableColumn`** · model · One column of a report table.
- **`TelemetryRollup`** · model · Campaign-wide descriptive telemetry — the trustworthy-signal layer.
- **`TextBlock`** · model · What the analysis's author wrote, exactly as written, with the facts code states beside it.
- **`TokenRollup`** · model · Summed token usage across results — visible cost of generation (§Visible Costs).
- **`TwoPillarDisclosure`** · model · Why the verdict rests on one quality pillar, stated on every answer.
- **`UnpairedRating`** · model · A rating with no judge score to set it against, and why.
- **`UnrepeatedScore`** · model · A repeated score with no pair to read, and why.
- **`VerdictOrder`** · model · The order verdicts are read in, as the campaign declared it — never as the writer would choose.

**Types**

- **`ArmStatus`** · literal · Where an arm stands, according to this analysis.
  <br>`'winner'` | `'contradicted'` | `'ruled_out'` | `'replaced_incumbent'` | `'unresolved'`
- **`CallAdmission`** · type alias · Asked before each generator call is counted or sent, with `(system, user, response_format)`; raises to refuse that call.
  <br>`Callable[[str, str, 'dict[str, Any] | None'], None]`
- **`ChangeLabel`** · literal · What a change between two paired samples reads as — see `ChangeVerdict`.
  <br>`'improved'` | `'regressed'` | `'equivalent'` | `'below_threshold'` | `'not_separated'` | `'untested'`
- **`ComparisonVerdict`** · literal · What one comparison in a family came to, read off its ADJUSTED p's.
  <br>`'improved'` | `'regressed'` | `'equivalent'` | `'not_separated'` | `'untested'`
- **`CriterionDrift`** · literal · How a label's criterion compares with the template's today — see `LabelReading.criterion_drift`.
  <br>`'unchanged'` | `'changed'` | `'no_live_criterion'`
- **`DisclosureSource`** · literal · Who a disclosure speaks for.
  <br>`'chart'` | `'arms'` | `'surface'` | `'time_axis'` | `'generation'` | `'runs'` | `'measurement'` | `'apparatus'` | `'comparisons'` | `'strata'` | `'guardrails'` | `'scope'`
- **`ExportFormat`** · literal · The two on-demand serializations.
  <br>`'csv'` | `'json'`
- **`FrontierCostDecision`** · literal · How a frontier verdict's pick stands on cost against the other contestants that cleared the bar with a cost — see `FrontierVerdict.cost_decision`.
  <br>`'shown_cheapest'` | `'not_separated'` | `'untested'` | `'only_cleared'`
- **`FrontierDominance`** · literal · Whether the frontier lens shows a contestant dominated: `dominated` — another is shown better on every axis it measured; `not_separated` — tested against at least one other and no domination shown, which says nothing about whether one exists; `untested` — nothing could be tested against it.
  <br>`'dominated'` | `'not_separated'` | `'untested'`
- **`LabelDirection`** · literal · Where a person reading a memo places it on a rubric dimension, low to high.
  <br>`'low'` | `'low_mid'` | `'mid'` | `'mid_high'` | `'high'`
- **`MechanismUncheckedReason`** · literal · Why a swept lever's mechanism could not be checked: `not_declared` = the lever names no measure it acts on; `not_swept` = it was observed at one level, so there is nothing to compare; `levels_unobserved` = some level observed none of the measure, so no pair of levels separated and whether it held still at every level cannot be shown; `too_few_observations` = every level observed it, but some pair of levels has too few cases on a side for the separation test to run, or a gap with no spread over too few cases for an exact test to call it at alpha.
  <br>`'not_declared'` | `'not_swept'` | `'levels_unobserved'` | `'too_few_observations'`
- **`ReportBasis`** · literal · What a report is of: a generated analysis, or the campaign's evidence alone with no analysis.
  <br>`'analysis'` | `'code_only'`
- **`ReportBlock`** · type alias · A report block, discriminated by `kind`.
  <br>`Annotated[TextBlock | TableBlock | ChartBlock | DisclosureBlock, Field(discriminator='kind')]`
- **`ReportSection`** · literal · Where a block sits, in reading order.
  <br>`'summary'` | `'questions'` | `'decisions'` | `'guardrails'` | `'findings'` | `'arms'` | `'surface'` | `'next'` | `'methods'`
- **`RowColumns`** · type alias · The host's own columns for each `(model, run_id)` group of a run's results, laid over a `run_summary` row after the engine's aggregates.
  <br>`Callable[['list[EvalResult]'], Mapping[tuple[str, str], Mapping[str, Any]]]`
- **`RunLister`** · type alias · Lists a scope's runs: `list_runs(scope_id, *, status=None, include_archived=False)`.
  <br>`Callable[..., 'list[EvalRun]']`
- **`SurfaceState`** · literal · Where a table stands. Two states rather than an optional table, because "the surface froze no cell" is a fact with its own sentence, and is not an empty table.
  <br>`'no_cells'` | `'measured'`
- **`SurfaceVerdict`** · literal · A bar's verdict on one cell, as a word — never a colour alone, and one per `decision`.
  <br>`'clears'` | `'misses'` | `'undecided'` | `'no_interval'` | `'no_data'`
- **`TextRole`** · literal · What a text block's author wrote it as.
  <br>`'summary'` | `'answer'` | `'decision'` | `'revisit_when'` | `'finding_title'` | `'finding_body'` | `'caveat'` | `'carried_forward'` | `'next_step'` | `'next_step_why'`
- **`UnpairedReason`** · literal · Why a rating has no judge score to be read against.
  <br>`'result_unresolved'` | `'dimension_unscored'` | `'scale_changed'` | `'rated_by_an_agent'`
- **`UnrepeatedReason`** · literal · Why a repeated score has no pair to be read in.
  <br>`'repeat_failed'` | `'judge_changed'` | `'config_changed'` | `'temperature_changed'`
- **`WriterMessageCheck`** · literal · Whether a case's frozen writer message is the one its recorded memo's generator was sent — see `writer_message_check`.
  <br>`'verified'` | `'differs'`

**Constants**

- **`ABSENT`** · constant (str) · What an absent or non-finite number reads as. Never `0`, which would say something was measured.
  <br>`= '—'`
- **`AS_RECORDED_MODEL`** · constant (str) · The candidate model name the as-recorded candidate runs under.
  <br>`= 'as-recorded'`
- **`CELL_MEASURED`** · constant (str) · A pivot cell state: observations landed here and carried a value for the measure.
  <br>`= 'measured'`
- **`CELL_NOT_RUN`** · constant (str) · A pivot cell state: no observation landed here — the combination was never run, which is not a zero.
  <br>`= 'not_run'`
- **`CELL_WITHHELD`** · constant (str) · A fourth state, and not a kind of the other three: the cell HAS measured observations, and its mean is withheld because it would pool two quantities that are not one distribution — today a cost cell pooling replayed results with live ones (#658).
  <br>`= 'withheld'`
- **`COST_ESTIMATE_MIN_BASIS`** · constant (int) · The fewest past observations a cost estimate publishes a band from; below it, the point estimate alone.
  <br>`= 3`
- **`COST_PREDICTION_METHOD`** · constant (str) · The `method_id` of the cost estimator's predictions: a planned cell priced from the corpus's own per-observation usage history (`compute_estimate_cost`).
  <br>`= 'usage-history'`
- **`DECLARED_INPUT_ORIGIN`** · constant (str) · The one origin value that means *this launch said so*.
  <br>`= 'chosen'`
- **`DEFAULT_WEIGHTING`** · constant (str) · The weighting a pivot uses when none is named: equal per scenario.
  <br>`= 'equal_per_scenario'`
- **`EQUIVALENCE_TEST_NAME`** · constant (str) · The equivalence test the change classifier runs beside the paired test, named for the same reason: an `equivalent` label names the statistics it rests on.
- **`EVAL_ANALYSIS_GEN_DEFAULT`** · constant (str) · The analysis generator's default system prompt.
- **`HISTORY_METRICS`** · constant (frozenset) · The measures `history` can series; any other is refused rather than answered with an empty series.
- **`LABEL_BANDS`** · constant (dict) · The 1-5 judge scores each label direction agrees with, inclusive.
- **`METRIC_COMPOSITE`** · constant (str) · The observation-level measure holding a result's composite quality score.
  <br>`= 'composite'`
- **`METRIC_OUTCOME`** · constant (str) · The observation-level measure holding the dual-score outcome axis.
  <br>`= '__outcome__'`
- **`METRIC_SCORE`** · constant (str) · The observation-level measure holding one judged rubric dimension's score: one row per dimension.
  <br>`= 'score'`
- **`METRIC_TRANSCRIPT`** · constant (str) · The observation-level measure holding the dual-score transcript axis.
  <br>`= '__transcript__'`
- **`NO_ANALYSIS`** · constant (str) · What the code-only report says in place of an analysis: that none was generated, and whose the numbers are.
- **`PAIRED_TEST_NAME`** · constant (str) · The paired test the change classifier discloses, so a regression flag names the statistics it rests on rather than presenting a bare verdict.
- **`PROJECTED_METRICS`** · constant (frozenset) · Every measure `project_score_records` can emit.
- **`REPORT_VERSION`** · constant (int) · The report shape's version.
  <br>`= 5`
- **`REPORTER_KIND`** · constant (str) · The `candidate_kind` a reporter template declares.
  <br>`= 'analysis_reporter'`
- **`SCOPED_METRICS_HELP`** · constant (str) · How each scoped metric must be read, in one sentence per metric, for every surface's help text (REST and MCP alike) — rendered from the table rather than written beside it, so no surface can describe a subset.
- **`SECTION_TITLES`** · constant (dict) · Each section's heading, in reading order — the order every serializer lays the sections out in.
- **`WEIGHTING_EQUAL_PER_SCENARIO`** · constant (str) · Weighting mode: every scenario (case) gets an equal vote in a cell's number, however many observations it has.
  <br>`= 'equal_per_scenario'`

<a id="api-analysis-viz"></a>
### `threetears.evals.analysis.viz`

Finding charts — eval's own chart intent, and the seam a renderer sits behind.

**Functions**

- **`assert_renderer_conforms`** · function · Raise unless `renderer` draws every one of `intents` in agreement with its values.
  <br>`assert_renderer_conforms(renderer: ChartRenderer[DrawingT], intents: Sequence[ChartIntent]) -> None`
- **`chart_intent`** · function · Decide a finding's chart: validate its payload, build its intent, and hold it to the policy rules.
  <br>`chart_intent(viz_type: str, payload: dict[str, Any]) -> ChartIntent`
- **`check_intent`** · function · Check a chart intent against the presentation rules.
  <br>`check_intent(intent: ChartIntent) -> list[str]`
- **`renderer_disagreements`** · function · Where `renderer`'s drawing of `intent` disagrees with the intent's values — empty when it agrees.
  <br>`renderer_disagreements(renderer: ChartRenderer[DrawingT], intent: ChartIntent) -> list[str]`
- **`table_disagreements`** · function · Where the values table states a value no mark of the same identity carries — empty when it agrees.
  <br>`table_disagreements(intent: ChartIntent) -> list[str]`

**Classes**

- **`ChartAxis`** · model · One ruler a chart measures against.
- **`ChartColours`** · model · One colour scheme a chart uses: which field it carries, and the slot each value takes.
- **`ChartColumn`** · model · One column of the values-as-drawn table.
- **`ChartEncoding`** · model · What one field of `data` encodes, and against which axis.
- **`ChartIdentity`** · model · What a chart's rows are named by, and the order they are drawn in.
- **`ChartIntent`** · model · One chart's intent: what it draws and what it must say, for any renderer to draw.
- **`ChartReference`** · model · A standard drawn across one axis — the bar the marks are read against, not one of them.
- **`ChartRenderer`** · protocol · A chart renderer: a host's palette, bound at construction, applied to any intent.
- **`IntentPolicyError`** · exception · A chart intent breaks a presentation rule — a data problem, never a rendering one.
- **`PayloadError`** · exception · A viz payload does not match the contract declared for its type.

**Types**

- **`Cell`** · type alias · A values-table or data cell: a JSON scalar.
  <br>`str | int | float | bool | None`
- **`ChartType`** · literal · The chart types — eval's own, and the only ones an intent can carry.
  <br>`'delta_table'` | `'frontier'` | `'timeseries'` | `'distribution'` | `'null_result'` | `'breakdown'` | `'attribution'` | `'sweep_ranking'`
- **`EncodingRole`** · literal · What a field encodes.
  <br>`'identity'` | `'length'` | `'position'` | `'interval_low'` | `'interval_high'` | `'level'` | `'class'` | `'ordinal'` | `'count'` | `'label'`

**Constants**

- **`INTENT_VERSION`** · constant (int) · The chart-intent shape's version.
  <br>`= 1`
- **`SERIES_SLOTS`** · constant (int) · How many categorical colour slots a palette supplies before it recycles — the width of the vocabulary, and so exactly how many `ChartPalette.series` colours a palette declares.
  <br>`= 8`
- **`VALIDATED_SLOTS`** · constant (int) · How many categorical colour slots a chart may assign with validated separation.
  <br>`= 4`

<a id="api-gen"></a>
### `threetears.evals.gen`

The engine's generation package: the prompts and expanders that author test material.

**Functions**

- **`generate_variations`** · async function · Generate up to `n_variations` test cases for a template.
  <br>`generate_variations(template: EvalTemplate, n_variations: int, *, storage: EvalTestCaseStore, scope_id: str, blocking_executor: Executor | None, llm: VariationLLM | None = None, budget: OutOfRunBudget | None = None, rng: random.Random | None = None, preview: bool = False) -> GeneratedVariations`
- **`price_variations`** · async function · Price the calls `generate_variations` would make, refusing exactly as it would — and make none.
  <br>`price_variations(template: EvalTemplate, n_variations: int, *, storage: EvalTestCaseStore, scope_id: str, blocking_executor: Executor | None, llm: VariationLLM, budget: OutOfRunBudget) -> list[float | None]`
- **`propose_draft`** · async function · Draft a rubric on `axis` from a subject feed and a catalog feed.
  <br>`propose_draft(client: BoundCompletionClient, *, budget: OutOfRunBudget, axis: RubricAxis, subject_id: str, system_prompt: str, subject_feed: str, catalog_feed: str) -> ProposedDraft`

**Classes**

- **`EvalTestCaseStore`** · protocol · The one read and one write generation needs.
- **`GeneratedVariations`** · class · The cases a generation produced, the counts the launch records beside them, and what its calls spent.
- **`ProposedDraft`** · class · A rubric draft, and the ledger row of the call that wrote it.

**Constants**

- **`EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT`** · constant (SeedTemplate) · The seeded default `eval_boundary_gen` template: drafts a battery of simulated actors that pressure a subject's boundaries, and the refusal dimensions they are scored on.
- **`EVAL_PROPOSER_TEMPLATE_DEFAULT`** · constant (SeedTemplate) · The seeded default `eval_proposer` template: drafts a capability rubric and the variation axes that stress it.
- **`PROPOSER_MAX_TOKENS`** · constant (int) · Output-token ceiling for the capability/boundary proposer.
  <br>`= 16384`

<a id="api-storage"></a>
### `threetears.evals.storage`

Storage adapters the engine ships: implementations of the one port a host stores through.

**Classes**

- **`InMemoryDocumentStore`** · class · A `DocumentStore` over one dict.

<a id="api-testing"></a>
### `threetears.evals.testing`

What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

**Functions**

- **`check_completion_conformance`** · function · Fail, naming the attribute, when a host's completion does not carry what eval reads off it.
  <br>`check_completion_conformance(completion: object) -> None`
- **`nonpublic_evals_imports`** · function · Every `threetears.evals` import under `sources` that reaches below a public root.
  <br>`nonpublic_evals_imports(*sources: Path | str) -> tuple[NonPublicImport, ...]`

**Classes**

- **`CompletionConformanceFailure`** · exception · A host's completion lacks an attribute eval reads, or reports a stop reason outside eval's vocabulary.
- **`NonPublicImport`** · dataclass · One import of a `threetears.evals` name that does not come from a public root.
- **`ReaderConformanceCase`** · dataclass · One promise a reader makes, as a check over a sample.
- **`ReaderConformanceFailure`** · exception · A host's reader broke a promise the engine relies on; the message names the case, the reader and the run.
- **`ReaderSample`** · dataclass · What the kit checks a host's readers over: the host's profile and runs it produced.
- **`StoreConformanceCase`** · dataclass · One rule of the port, as a check over a store.
- **`StoreConformanceFailure`** · exception · A store broke a rule of the `DocumentStore` port; the message names the case and the rule.

**Constants**

- **`READER_CONFORMANCE_CASES`** · constant (tuple) · Every promise, as a case. Parametrise over this tuple.
- **`STORE_CONFORMANCE_CASES`** · constant (tuple) · Every case, in the order the port states its rules: scoping, the strip on read, projection, querying, optimistic concurrency (with the re-read that recovers a lost race), merge, delete — and last, that every `doc_type` the engine writes is one the store stores.

<a id="api-quick"></a>
### `threetears.evals.quick`

Batteries: run an eval in one call, and drive the engine from a command line.

**Functions**

- **`build_parser`** · function · The command line's parser.
  <br>`build_parser(prog: str = 'python -m threetears.evals', *, takes_host: bool = True, commands: Sequence[HostCommand] = ()) -> argparse.ArgumentParser`
- **`callable_host`** · function · The least host there is: the shared core, one measure per scorer, no world, an in-memory store.
  <br>`callable_host(scorers: Sequence[Scorer] = (), *, levers: Sequence[str] = (), world: World | None = None, arms: bool = False) -> EvalHost`
- **`callable_kind_contracts`** · function · The contracts of both callable kinds, declaring `levers` as each run's levels beside its model.
  <br>`callable_kind_contracts(levers: Sequence[str] = ()) -> tuple[KindContract, KindContract]`
- **`compare`** · async function · Run each candidate over every case `k` times as one arm, test every arm against `control`, and report.
  <br>`compare(cases: Sequence[Mapping[str, Any]], candidates: Mapping[str, Candidate | ToolUsingCandidate | WorldCandidate] | Mapping[tuple[str, ...], Candidate | ToolUsingCandidate | WorldCandidate], scorers: Sequence[Scorer] = (), *, control: ArmKey, scope_id: str, expected: ExpectedLabel | None = None, judge: Judge | None = None, intent: str | None = None, host: EvalHost | None = None, k: int = 3, name: str | None = None, created_by: str = 'compare', factors: Sequence[str] | None = None, tools: Mapping[str, Tool] | None = None, cassette_mode: CassetteMode = 'off', cassette_corpus_id: str | None = None, world: World | None = None, seed: CaseSeed | None = None, goal_checks: Sequence[str] = (), max_cost_usd: float | None = None) -> Comparison`
- **`run_cli`** · function · Parse `argv` and carry out the command, printing to stdout and refusals to stderr.
  <br>`run_cli(argv: Sequence[str] | None = None, *, host_factory: HostFactory | None = None, prog: str = 'python -m threetears.evals', commands: Sequence[HostCommand] = ()) -> int`
- **`run_eval`** · async function · Run `candidate` on every case `k` times, grade each answer with every scorer and the judge, and summarise.
  <br>`run_eval(cases: Sequence[Mapping[str, Any]], candidate: Candidate | ToolUsingCandidate | WorldCandidate, scorers: Sequence[Scorer] = (), *, scope_id: str, expected: ExpectedLabel | None = None, judge: Judge | None = None, intent: str | None = None, world: World | None = None, seed: CaseSeed | None = None, goal_checks: Sequence[str] = (), host: EvalHost | None = None, k: int = 3, model: str | None = None, levers: Mapping[str, str] | None = None, tools: Mapping[str, Tool] | None = None, cassette_mode: CassetteMode = 'off', cassette_corpus_id: str | None = None, max_cost_usd: float | None = None) -> EvalSummary`
- **`summarize_run`** · function · Summarise one stored run and its results.
  <br>`summarize_run(host: EvalHost, run_id: str, scope_id: str, *, case_names: Mapping[str, str] | None = None) -> EvalSummary`

**Classes**

- **`Answer`** · dataclass · What a candidate returns to report its own spend beside its answer.
- **`CaseResult`** · model · One case's answer on one repeat, every grade it got, and why it failed or was excluded.
- **`Comparison`** · dataclass · What `compare` ran and what its campaign's report says.
- **`Dimension`** · dataclass · One piece of the world's state.
- **`DimensionSummary`** · model · One judged rubric dimension over a run's results.
- **`EvalSummary`** · model · One run, summarised.
- **`GoalCheckSummary`** · model · One goal-state check over a run's results.
- **`HostCommand`** · dataclass · A subcommand a host adds beside the engine's own, mounted by `run_cli` under the same program.
- **`Judge`** · dataclass · A model that grades each answer on a rubric, one call per dimension.
- **`JudgeGrade`** · model · One rubric dimension's score on one answer, with the judge's reason.
- **`MeasureSummary`** · model · One measure over a run's results.
- **`ToolRefused`** · exception · A tool call the world did not make: no such tool, or parameters its schema refuses. Nothing changed.
- **`World`** · class · A small world: named state each case seeds, and tools the candidate changes it with.
- **`WorldTool`** · class · One action the candidate can take on the world: a function of the state and its parameters.
- **`WorldTools`** · class · What the candidate holds of its cell's world: the tools, bound to it, and a view of it.

**Types**

- **`ArmKey`** · type alias · An arm's key: its name, which is its model, when `compare` is given no `factors`; with them, its level of each factor, in the order `factors` names them.
  <br>`str | tuple[str, ...]`
- **`Candidate`** · type alias · The candidate under test: a callable taking one case and returning its answer — usually `async def`; a plain `def` is called in a worker thread (`run_eval`), and a callable returning an awaitable has it awaited.
  <br>`Callable[[Mapping[str, Any]], Awaitable[Any] | Any]`
- **`CandidateTools`** · type alias · What a tool-using candidate is handed beside its case: each declared tool by name, as an async function of keyword arguments returning the tool's JSON answer.
  <br>`Mapping[str, Callable[..., Awaitable[Any]]]`
- **`CaseMaterial`** · type alias · Renders the material one case's answer is judged against: takes the case, returns non-blank text.
  <br>`Callable[[Mapping[str, Any]], str]`
- **`CaseOutcome`** · literal · How one result came out, as `classify_result` classifies it: graded normally, failed by the candidate (it counts against the candidate), or excluded as a fault of the rig (it counts for nothing).
  <br>`'scored'` | `'failed'` | `'excluded'`
- **`CaseSeed`** · type alias · A case's starting state: takes the case, returns dimension name to value.
  <br>`Callable[[Mapping[str, Any]], Mapping[str, Any]]`
- **`ExpectedLabel`** · type alias · A classifier's expected label for one case: takes the case, returns the label a correct answer gives.
  <br>`Callable[[Mapping[str, Any]], str]`
- **`HostFactory`** · type alias · What names the host the commands work in: called once, with no arguments, per invocation.
  <br>`Callable[[], EvalHost | LaunchHost]`
- **`Scorer`** · type alias · One grade: takes the case and the candidate's answer, returns a number (`True`/`False` count as 1 and 0).
  <br>`Callable[[Mapping[str, Any], Any], float | bool]`
- **`Tool`** · type alias · A tool a candidate calls: a function of keyword arguments returning a JSON value, sync or async.
  <br>`Callable[..., Any]`
- **`ToolUsingCandidate`** · type alias · A candidate that calls tools: an async callable taking one case and its tools, returning its answer.
  <br>`Callable[[Mapping[str, Any], CandidateTools], Awaitable[Any]]`
- **`WorldCandidate`** · type alias · A world candidate: an async callable taking one case and the tools on its cell's world.
  <br>`Callable[[Mapping[str, Any], WorldTools], Awaitable[Any]]`

**Constants**

- **`ARM_LEVER`** · constant (str) · The lever a single-factor `compare` names its arms on, declared by `callable_host(arms=True)`: each arm's run states its name as its level, so the report calls the arm `candidate=<name>` rather than calling the name a model.
  <br>`= 'candidate'`
- **`CALLABLE_KIND`** · constant (str) · The kind `run_eval` launches, as its template names it.
  <br>`= 'callable'`
- **`CALLABLE_KIND_CONTRACT`** · constant (KindContract) · The callable kind's contract: no overlays, no spec, and no rig seat — nothing in a `run_eval` run is graded by a model or talks to a simulated user.
- **`CALLABLE_UNSEATED`** · constant (frozenset) · What a `run_eval` run never has as a level of its rig, whatever host it runs in, so a callable-kind contract may seat none of it: the engine's judge and simulator (by role or by any pinned dimension) — the callable kind is unjudged and simulates nobody — and the spend ceiling, which is off unless a call caps it (`max_cost_usd=`) and is then a condition stated beside the run: a run it stopped says so by its status, and two runs under different caps measured the same candidate the same way until one stopped.
- **`DEFAULT_PROG`** · constant (str) · The program name the command line prints when it is run as `python -m threetears.evals`.
  <br>`= 'python -m threetears.evals'`
- **`ENGINE_COMMANDS`** · constant (tuple) · The commands the engine itself carries; a host command may take none of these names.
- **`EXIT_FAILED`** · constant (int) · Exit code: the command failed on an error nothing anticipated — a host factory, a launcher or a handler raising, or the engine's own fault.
  <br>`= 3`
- **`EXIT_OK`** · constant (int) · Exit code: every run completed, or the command read what it was asked for.
  <br>`= 0`
- **`EXIT_REFUSED`** · constant (int) · Exit code: the command was refused before it could do anything.
  <br>`= 2`
- **`EXIT_RUN_DID_NOT_COMPLETE`** · constant (int) · Exit code: a launched run did not complete.
  <br>`= 1`
- **`JUDGED_CALLABLE_KIND`** · constant (str) · The kind `run_eval` launches when it is handed a judge: the callable, its answers judged as documents.
  <br>`= 'callable-judged'`
- **`JUDGED_CALLABLE_KIND_CONTRACT`** · constant (KindContract) · The judged callable kind's contract: no overlays, no spec, and the engine's judge seated — its runs are graded by a model, so who judged them is part of what two of them are compared on, and a run judged by another model or under other judge configs is a confound rather than a blank.
- **`JUDGED_CALLABLE_UNSEATED`** · constant (frozenset) · What a judged `run_eval` run never has as a level of its rig: the simulator (by role or by any pinned dimension) and the spend ceiling, a condition beside the run as `CALLABLE_UNSEATED` says.
- **`SHARED_ARM_MODEL`** · constant (str) · The candidate model every arm of a single-factor `compare` shares when its arms are named on `ARM_LEVER`: the arms' names are not models, and a lever every arm shares splits none of them.
  <br>`= 'callable'`
- **`UNUSABLE_ANSWER`** · constant (str) · The predicted label a classifier's answer is counted under when it is not a usable label: not a string, or a blank one.
  <br>`= '(unusable answer)'`

**Also exported here**

`ConfusionCount` ([`threetears.evals.analysis`](#api-analysis)), `LabelStatistics` ([`threetears.evals.analysis`](#api-analysis))

<a id="api-ops"></a>
### `threetears.evals.ops`

Typed operations over a host: what every surface — a CLI, an MCP tool, a REST route — calls.

**Functions**

- **`analyses_list`** · function · A campaign's stored analyses.
  <br>`analyses_list(host: EvalHost, campaign_id: str, scope_id: str) -> AnalysisListing`
- **`analysis_archive`** · function · Archive or restore a stored analysis — the reversible answer to deleting it.
  <br>`analysis_archive(host: EvalHost, analysis_id: str, scope_id: str, *, archived: bool, reason: str | None = None) -> AnalysisLine`
- **`analysis_delete`** · function · Destroy a stored analysis — its insights stay; archive is the reversible answer.
  <br>`analysis_delete(host: EvalHost, analysis_id: str, scope_id: str, *, confirm: str | None) -> AnalysisDeleted`
- **`analysis_estimate`** · async function · What a campaign's generation would be priced at, against the cap it would be held to — making no call.
  <br>`analysis_estimate(host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None) -> AnalysisGenerationEstimate`
- **`analysis_generate`** · async function · Check a campaign's generation, price its first call against the host's out-of-run cap, then start it as a job.
  <br>`analysis_generate(host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None) -> JobsStarted`
- **`analysis_job_id`** · function · The job id of an analysis generation: its campaign, then its attempt.
  <br>`analysis_job_id(campaign_id: str, attempt_id: str) -> str`
- **`campaign_archive`** · function · Archive or restore a campaign — retired from listings, nothing destroyed.
  <br>`campaign_archive(host: EvalHost, campaign_id: str, scope_id: str, *, archived: bool) -> CampaignLine`
- **`campaign_create`** · function · Create a campaign over runs already in the scope, declared as it is created when the definition says so.
  <br>`campaign_create(host: EvalHost, definition: CampaignDefinition, scope_id: str, *, created_by: str) -> CampaignLine`
- **`campaigns_list`** · function · The scope's campaigns.
  <br>`campaigns_list(host: EvalHost, scope_id: str, *, archived: bool | None = None) -> CampaignListing`
- **`dollars_text`** · function · Spend as a person reads it: dollars and cents from ten cents up, three significant figures below.
  <br>`dollars_text(amount: float) -> str`
- **`estimate_text`** · function · An estimate as text: the launch priced, each arm's price and outcome, and the total.
  <br>`estimate_text(estimate: LaunchEstimate) -> str`
- **`export_text`** · function · An export as text: a line of its row count and what it left out, then the body itself.
  <br>`export_text(export: ScoreExport) -> str`
- **`generation_key`** · function · The exclusivity key one campaign's generations share: one runs at a time, and only its scope sees it.
  <br>`generation_key(campaign_id: str, scope_id: str) -> str`
- **`history_launch_pricer`** · function · The engine's launch pricer: an arm bounded from the scope's usage history of runs launched as it will be.
  <br>`history_launch_pricer(host: EvalHost) -> LaunchPricer`
- **`history_text`** · function · A history as text: each contestant's series, oldest first, with each step's verdict and its test.
  <br>`history_text(result: HistoryResult) -> str`
- **`job_cancel`** · async function · Ask a running job to stop, then report where it stands.
  <br>`job_cancel(host: OpsHost, job_id: str, scope_id: str, *, reason: str | None = None) -> JobStatus`
- **`job_poll`** · async function · Where a job stands, read from the record its work writes.
  <br>`job_poll(host: OpsHost, job_id: str, scope_id: str) -> JobStatus`
- **`judge_repeat`** · async function · Repeat a finished run's judge scores — the measurement the `separation` evidence tier reads.
  <br>`judge_repeat(host: OpsHost, run_id: str, scope_id: str, *, result_ids: list[str] | None = None) -> JudgeRepeatReport`
- **`judge_repeat_estimate`** · async function · What repeating a finished run's judge scores would be priced at, against the host's out-of-run cap — no call made.
  <br>`judge_repeat_estimate(host: OpsHost, run_id: str, scope_id: str, *, result_ids: list[str] | None = None) -> JudgeRepeatEstimate`
- **`launch_estimate`** · async function · What `run_launch` with `arguments` would cost, priced by the launch's own rule.
  <br>`launch_estimate(host: OpsHost, arguments: LaunchArguments, scope_id: str, *, n_test_cases: int | None = None) -> LaunchEstimate`
- **`out_of_run_spend_text`** · function · The scope's out-of-run spend as an operator reads it: the totals, per purpose and launch, then each call.
  <br>`out_of_run_spend_text(report: OutOfRunSpendReport) -> str`
- **`parse_job_id`** · function · Read a job id back into what it names.
  <br>`parse_job_id(job_id: str) -> tuple[JobKind, str, str | None]`
- **`pivot_text`** · function · A pivot as text: what was computed, each cell with its denominators, and every caveat the table carries.
  <br>`pivot_text(table: PivotTable) -> str`
- **`report_read`** · function · The campaign's report, serialized in one form — the same report the command line's `report` prints.
  <br>`report_read(host: EvalHost, campaign_id: str, scope_id: str, *, format: ReportFormat) -> ReportDocument`
- **`reporter_case_archive`** · function · Retire a reporter case, or restore one: a retired case is never launched again and stays readable.
  <br>`reporter_case_archive(host: EvalHost, test_case_id: str, scope_id: str, *, archived: bool, reason: str | None = None) -> FrozenReporterCase`
- **`reporter_case_freeze`** · function · Freeze a campaign's analysis bundle, and optionally the memo it got, into a case of a reporter template.
  <br>`reporter_case_freeze(host: EvalHost, freeze: ReporterCaseFreeze, scope_id: str) -> FrozenReporterCase`
- **`reporter_cases_list`** · function · A reporter template's cases: each with whether a launch runs it, then what cannot be read or decided.
  <br>`reporter_cases_list(host: EvalHost, template_id: str, scope_id: str, *, include_archived: bool = False) -> ReporterCaseListing`
- **`result_get`** · function · One stored result and one part of its trace — what the cell recorded, as its kind stored it.
  <br>`result_get(host: EvalHost, result_id: str, scope_id: str, *, part: ResultPart = 'record') -> ResultDetail`
- **`result_rate`** · function · Record an agent's rating of one judged dimension of one result — kept beside people's, never pooled with them.
  <br>`result_rate(host: EvalHost, result_id: str, scope_id: str, *, rubric_dim: str, score: int, reason: str, rater: str) -> ResultRated`
- **`results_list`** · function · One run's results, a light row each, paged in a stable order.
  <br>`results_list(host: EvalHost, run_id: str, scope_id: str, *, condition: ResultOutcome | None = None, offset: int = 0, limit: int | None = None) -> ResultListing`
- **`run_archive`** · function · Archive or restore a run — reversible exclusion from every cohort.
  <br>`run_archive(host: EvalHost, run_id: str, scope_id: str, *, archived: bool) -> RunLine`
- **`run_delete`** · function · Destroy a run, its results and its campaign memberships — unrecoverable; archive is the safe answer.
  <br>`run_delete(host: EvalHost, run_id: str, scope_id: str, *, confirm: str | None) -> RunDeleted`
- **`run_get`** · function · One run, summarised from its stored results.
  <br>`run_get(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary`
- **`run_job_id`** · function · The job id of a launched run.
  <br>`run_job_id(run_id: str) -> str`
- **`run_launch`** · async function · Launch a template's runs, one per model, and return a job per run.
  <br>`run_launch(host: OpsHost, arguments: LaunchArguments, scope_id: str) -> JobsStarted`
- **`runs_compare`** · function · Compare two runs' arms, with each run's completeness, clock and cassette disclosures beside the numbers.
  <br>`runs_compare(host: EvalHost, baseline_run_id: str, candidate_run_id: str, scope_id: str) -> RunsCompared`
- **`runs_compared_text`** · function · Two runs compared as text: the arms, each reading with its delta and test, then every disclosure.
  <br>`runs_compared_text(compared: RunsCompared) -> str`
- **`runs_list`** · function · The scope's runs.
  <br>`runs_list(host: EvalHost, scope_id: str, *, status: str | None = None, include_archived: bool = False) -> RunListing`
- **`scope_export`** · function · The scope's observations as flat rows, in CSV or JSON, for analysis elsewhere.
  <br>`scope_export(host: EvalHost, scope_id: str, *, format: str | None = None, status: str | None = 'completed', run_ids: list[str] | None = None) -> ScoreExport`
- **`scope_history`** · function · One measure over time for each contestant in the scope, with its regressions flagged.
  <br>`scope_history(host: EvalHost, scope_id: str, *, metric: str | None = None, min_absolute_change: float = 0.0, min_relative_change: float = 0.0, subject_id: str | None = None, status: str | None = 'completed') -> HistoryResult`
- **`scope_out_of_run_spend`** · function · What the engine spent outside any run in a scope, call by call and summed, optionally narrowed.
  <br>`scope_out_of_run_spend(host: EvalHost, scope_id: str, *, purpose: OutOfRunPurpose | None = None, launch_group_id: str | None = None, template_id: str | None = None) -> OutOfRunSpendReport`
- **`scope_pivot`** · function · One measure over the scope's observations, aggregated over two coordinates.
  <br>`scope_pivot(host: EvalHost, scope_id: str, *, row_factor: str, column_factor: str, metric: str | None = None, weighting: str | None = None, subject_id: str | None = None, status: str | None = 'completed', predicted_cost: CostEstimate | LaunchEstimate | Mapping[str, Any] | None = None, launched_run_ids: Sequence[str] = ()) -> PivotTable`
- **`serialize_report`** · function · A report in one of its three forms.
  <br>`serialize_report(report: Report, format: ReportFormat) -> str`
- **`templates_list`** · function · The scope's templates.
  <br>`templates_list(host: EvalHost, scope_id: str, *, archived: bool = False) -> TemplateListing`

**Classes**

- **`AmbiguousReporterPair`** · model · A (campaign, recorded memo) pair holding more than one live case — which every launch of the template refuses.
- **`AnalysisDeleted`** · model · What deleting an analysis removed: the analysis, never the insights it minted.
- **`AnalysisGeneration`** · dataclass · How this host generates an analysis as a background job.
- **`AnalysisLine`** · model · One stored analysis, as a listing shows it.
- **`AnalysisListing`** · model · A campaign's stored analyses.
- **`ArmEstimate`** · model · One arm of a launch estimate: what its kind planned, what the host's pricer predicted, and what the launch would do.
- **`CampaignDefinition`** · model · What creating a campaign names: what it is called, its subject and behaviour, its runs, and what it set out to learn.
- **`CampaignLine`** · model · One campaign, as a listing shows it.
- **`CampaignListing`** · model · A scope's campaigns, newest first.
- **`JobHandle`** · model · A started job: the id to poll, and what it is working on.
- **`JobsStarted`** · model · What starting long work returns: one handle per job, in the order the work was asked for.
- **`JobStatus`** · model · Where one job stands, read from the record its work writes.
- **`LaunchArguments`** · model · What a launch names: the template, the subject, one arm per model, and the run's own limits.
- **`LaunchEstimate`** · model · What a launch would cost and whether it would launch, priced by the launch's own rule.
- **`OpsHost`** · dataclass · The host the operations, and the actions over them, work in.
- **`OutOfRunSpendReport`** · model · The calls the engine made outside any run in a scope — case generations, rubric proposals and analysis generations — and their totals.
- **`OutOfRunSpendTotals`** · model · What a set of out-of-run calls spent, summed — with what could not be summed counted beside it.
- **`ReportDocument`** · model · A campaign's report, serialized in one form.
- **`ReporterCaseEntry`** · model · One readable case of a reporter template, with whether a launch runs it.
- **`ReporterCaseFreeze`** · model · What freezing a reporter case names: the reporter template, the campaign, and optionally its memo and labels.
- **`ReporterCaseListing`** · model · A reporter template's cases, in storage order.
- **`ResultDetail`** · model · One part of one stored result: the record and its condition always, and the trace part asked for.
- **`ResultLine`** · model · One result, as a run's listing shows it: where it sits, the condition it is in, and its headline measures.
- **`ResultListing`** · model · One page of a run's results, in a stable order: by case, then repeat, then id.
- **`ResultRated`** · model · A rating an agent wrote: what it rated, and that it is an agent's, never read as a person's.
- **`RunDeleted`** · model · What deleting a run removed.
- **`RunLine`** · model · One run, as a listing shows it.
- **`RunListing`** · model · A scope's runs, newest first as the store lists them.
- **`RunsCompared`** · model · One run's arm against another's, with what either run could not deliver said beside the numbers.
- **`TemplateLine`** · model · One template, as a listing shows it.
- **`TemplateListing`** · model · A scope's templates.
- **`TraceJudge`** · model · The `judge` part of a stored trace: what the judge was sent, as the kind rendered it.
- **`TraceRecord`** · model · The `record` part of a stored trace: what the kind stored about the cell, without the two heavy parts.
- **`UnreadableReporterCase`** · model · A stored case carrying a reporter case this build cannot read.

**Types**

- **`JobKind`** · literal · What a job's work is: a launched run, or an analysis generation.
  <br>`'run'` | `'analysis'`
- **`JobState`** · literal · Where a job stands.
  <br>`'running'` | `'completed'` | `'stopped'` | `'failed'` | `'cancelled'` | `'lost'`
- **`ReportFormat`** · literal · The forms a report is read in: Markdown (the memo, and what an agent reads), its canonical JSON (what the published schema validates) and HTML that reads without any script.
  <br>`'markdown'` | `'json'` | `'html'`
- **`ResultPart`** · literal · Which part of a stored result `result_get` returns.
  <br>`'record'` | `'judge'` | `'spans'`
- **`TraceState`** · literal · Whether a result's trace can be read: `stored`; `none` — the cell wrote none (its record says so); or `missing` — the record says one was written and no document backs it, which is a fault, never an ordinary absence (`get_result_trace` logs it).
  <br>`'stored'` | `'none'` | `'missing'`

**Constants**

- **`ANALYSIS_JOB_PREFIX`** · constant (str) · The prefix of an analysis generation's job id.
  <br>`= 'analysis:'`
- **`RUN_JOB_PREFIX`** · constant (str) · The prefix of a launched run's job id.
  <br>`= 'run:'`
- **`TERMINAL_JOB_STATES`** · constant (frozenset) · The states a job does not leave.

**Also exported here**

`AnalysisGenerationEstimate` ([`threetears.evals.analysis`](#api-analysis)), `CaseResult` ([`threetears.evals.quick`](#api-quick)), `CostEstimate` ([`threetears.evals.analysis`](#api-analysis)), `DimensionSummary` ([`threetears.evals.quick`](#api-quick)), `EvalSummary` ([`threetears.evals.quick`](#api-quick)), `FrozenReporterCase` ([`threetears.evals.analysis`](#api-analysis)), `HistoryResult` ([`threetears.evals.analysis`](#api-analysis)), `JudgeGrade` ([`threetears.evals.quick`](#api-quick)), `MeasureSummary` ([`threetears.evals.quick`](#api-quick)), `PivotTable` ([`threetears.evals.analysis`](#api-analysis)), `ScoreExport` ([`threetears.evals.analysis`](#api-analysis)), `summarize_run` ([`threetears.evals.quick`](#api-quick))

<a id="api-actions"></a>
### `threetears.evals.actions`

The action catalogue: every eval action, declared once, for any transport to mount.

**Functions**

- **`engine_actions`** · function · The engine's actions, in the order help lists them.
  <br>`engine_actions() -> tuple[Action, ...]`
- **`eval_catalogue`** · function · The engine's catalogue, with the host's own actions after the engine's.
  <br>`eval_catalogue(host_actions: tuple[Action, ...] = ()) -> ActionCatalogue`
- **`read_only_tools`** · function · One tool carrying only the `read` actions, for an agent that may look but not act.
  <br>`read_only_tools(prefix: str = 'evals') -> tuple[ToolSpec]`
- **`standard_tools`** · function · The two tools a host mounts by default: `<prefix>` and `<prefix>_admin`.
  <br>`standard_tools(prefix: str = 'evals') -> tuple[ToolSpec, ToolSpec]`

**Classes**

- **`Action`** · dataclass · One action: what it is called, what it takes and returns, who may call it, and how it reads back.
- **`ActionCatalogue`** · class · Every action a host offers: the engine's, and any the host contributes.
- **`ActionOutcome`** · dataclass · What one call came to: the text an agent reads, the typed result as data, and whether it was refused.
- **`Caller`** · dataclass · Who is calling, and the scope they act in — resolved by the host for every call.
- **`MountedTool`** · dataclass · A tool cut from the catalogue: the actions its classes admit, and how a call to it is carried out.
- **`ToolHints`** · dataclass · The MCP behaviour hints a mounted tool's actions earn.
- **`ToolSpec`** · dataclass · One tool a transport mounts: its name, its short description and the classes of action it carries.

**Types**

- **`ActionHandler`** · type alias · Carries an action out: handed the host, the caller and the validated parameters, it returns the action's result model.
  <br>`Callable[[OpsHost, Caller, Any], Awaitable[BaseModel]]`
- **`PermissionClass`** · literal · What an action does to the world, which decides the tools that may mount it.
  <br>`'read'` | `'spend'` | `'write'` | `'destructive'`

**Constants**

- **`ACTION_NAME`** · constant (Pattern) · An action's name: `noun_verb` — lowercase words joined by underscores, at least two of them.
  <br>`= '^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+$'`
- **`DEFAULT_PREFIX`** · constant (str) · The default prefix a host's eval tools carry.
  <br>`= 'evals'`
- **`HELP_ACTION`** · constant (str) · The action every mounted tool answers, generated from the catalogue rather than declared in it.
  <br>`= 'help'`
- **`PERMISSION_CLASSES`** · constant (tuple) · The classes, in the order help lists them.
- **`RESERVED_PARAMETERS`** · constant (frozenset) · Parameter names a tool reserves: `action` selects the action, `topic` is help's.
- **`TOOL_NAME`** · constant (Pattern) · A tool's name: a lowercase identifier.
  <br>`= '^[a-z][a-z0-9_]*$'`

<a id="api-transports-fastmcp"></a>
### `threetears.evals.transports.fastmcp`

The FastMCP transport: mounts the action catalogue as tools on a FastMCP server.

**Functions**

- **`mount_fastmcp`** · function · Add the catalogue's tools to a FastMCP server.
  <br>`mount_fastmcp(server: FastMCP, catalogue: ActionCatalogue, *, host: OpsHost, caller: CallerResolver, tools: Sequence[ToolSpec] | None = None) -> tuple[str, ...]`

**Classes**

- **`CatalogueTool`** · model · One mounted tool, as FastMCP serves it: every call handed to the catalogue.
- **`ToolBinding`** · class · What a `CatalogueTool` hands each call to: the mounted tool, the host and the caller resolver.

**Types**

- **`CallerResolver`** · type alias · Resolves who is calling, and their scope, for one call — sync or async. The host reads its own request context (an access token, a session) to answer it.
  <br>`Callable[[], Caller | Awaitable[Caller]]`

<a id="api-vega"></a>
### `threetears.evals.vega`

The Vega-Lite chart renderer — an optional adapter over eval's chart intent.

**Functions**

- **`check_spec`** · function · Check a Vega-Lite spec against the presentation rules.
  <br>`check_spec(spec: dict[str, Any]) -> list[str]`
- **`compile_chart`** · function · Compile a finding's viz into a Vega-Lite spec, through the chart's intent.
  <br>`compile_chart(viz_type: str, payload: dict[str, Any], *, font: ChartFont | None = None) -> CompiledChart`
- **`draw_intent`** · function · Draw a decided chart intent as a Vega-Lite spec — this renderer's one entry point.
  <br>`draw_intent(intent: ChartIntent, *, font: ChartFont | None = None) -> CompiledChart`
- **`load_chart_font`** · function · Read a metrics artifact a host measured for its own face, as the font to declare.
  <br>`load_chart_font(path: Path) -> ChartFont`
- **`packaged_font`** · function · The face this renderer draws in when its host declares none, with the table measured for it.
  <br>`packaged_font() -> ChartFont`
- **`packaged_palette`** · function · The palette packaged with this renderer, in one of its two variants.
  <br>`packaged_palette(theme: Theme) -> ChartPalette`
- **`register_fonts`** · function · Make a directory of font files available to the rasteriser.
  <br>`register_fonts(font_dir: Path) -> None`
- **`render_png`** · function · Rasterise a compiled spec to PNG.
  <br>`render_png(spec: dict[str, Any], *, palette: ChartPalette, scale: int = 2, font_dir: Path | None = None, font: ChartFont | None = None) -> bytes`
- **`render_svg`** · function · Render a compiled spec to SVG.
  <br>`render_svg(spec: dict[str, Any], *, palette: ChartPalette, font_dir: Path | None = None, font: ChartFont | None = None) -> str`
- **`vega_config`** · function · Build the Vega-Lite config that themes a compiled spec in `palette`, set in `font`.
  <br>`vega_config(palette: ChartPalette, font: ChartFont | None = None) -> dict[str, Any]`
- **`write_font_metrics`** · function · Commit a fresh measurement as a table `text_width` reads.
  <br>`write_font_metrics(advances: dict[str, float], *, fallback_advance: float, worst_label: str, worst_ratio: float, font: str, measured_with: str, probe_size: int, weights: list[int], path: Path = Path('<package>/src/threetears/evals/vega/font_metrics.json')) -> Path`

**Classes**

- **`CompiledChart`** · dataclass · One finding's chart: its intent, and the Vega-Lite spec drawn from it.
- **`CompiledColumn`** · typed dict · One column of the values-as-drawn table, as the compilation hands it on.
- **`PaletteError`** · exception · The chart palette artifact is missing or cannot be drawn with.
- **`SpecPolicyError`** · exception · A chart's spec breaks one or more of the report's presentation rules.
- **`TextMetricsError`** · exception · The font metrics artifact is missing or unusable.
- **`VegaRenderer`** · dataclass · Draw chart intents as Vega-Lite, in one palette and one typeface.

**Types**

- **`Theme`** · literal · The colour theme a chart is drawn for.
  <br>`'light'` | `'dark'`

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
| `enforcement_enabled` | `bool` | required | Whether the cost and metered-call ceilings are enforced at all. |
| `max_cost_usd` | `float` | required | The run cost ceiling a run inherits when its launch names none. |
| `max_metered_calls` | `int \| None` | required | The metered-call ceiling a run inherits when its launch names none, or `None` for a host that declares it has NO metered tools: its runs record a ceiling of `0` (origin `none_declared`), a metered call on one is refused and counted, and a launch naming a ceiling is refused, since it would bound nothing. |
| `max_out_of_run_cost_usd` | `float` | required | The most a launch's out-of-run calls — its case generation, which runs before any run exists and so under no run's cap — may together be priced at before they are made (`OutOfRunBudget`). Enforced exactly when `enforcement_enabled` is. Per LAUNCH, as `max_cost_usd` is per run: a battery is one launch per template, so a battery of N generating templates may spend up to N times this out of run, as its runs may spend up to their count times their cap. An analysis generation is held to it too, per generation (`analysis_generate`): its calls run after the runs it reads, under no run's cap. |
| `judge_alternate_model` | `str \| None` | `None` | The judge a launch's arms are scored by instead of the judge role's default when that default IS one of the launch's candidate models — a model grading its own output — provided it is itself none of them (`resolve_judge_pin`), spelled as the host's clients name the model they resolve. It never overrides a judge the launch named, nor a model a judge config pins per dim: those are choices. `None` substitutes nothing, and a run judged on a candidate's model says so on every surface that lists its judges (`judges_sharing_a_candidate_model`). |
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
| `sweepables` | `SweepableRegistry` | derived | Every input this host's runs carry: `host_sweepables` plus each kind contract's levers. |

<a id="documents"></a>
## The report and the analysis bundle

<a id="report"></a>
### `Report`

One analysis, as a document every surface renders. See the module docstring for the contract.

| Field | Type | Default | Description |
|---|---|---|---|
| `report_version` | `Literal[5]` | `5` | This shape's version. |
| `basis` | `ReportBasis` | required | `analysis` when the report renders a generated analysis; `code_only` when no analysis exists and the report is the campaign's evidence as code computed it — no headline, no findings, no author's words. |
| `headline` | `ModelProse` | required | The author's headline, as written; empty when the author wrote none, and on a code-only report. |
| `finding_count` | `int` | required | How many findings the document holds — the range every position is in. |
| `source` | `ReportSource` | required | What this is a report of. |
| `blocks` | `list[ReportBlock]` | required | The report, in reading order. |

Its `source`, and each kind of block in `blocks`:

#### `ReportSource`

What the report is a report of, and how that analysis was generated — or that none was.

| Field | Type | Default | Description |
|---|---|---|---|
| `analysis_id` | `str \| None` | `None` | The analysis the report renders; None on a code-only report. |
| `campaign_id` | `str` | required | The campaign the analysis is of. |
| `campaign_name` | `str \| None` | `None` | The campaign's name, which a code-only report's title reads by; None where the report was built without it, and the title then names the campaign by its id. |
| `scope_id` | `str` | required | The scope both live in. |
| `subject_id` | `str` | required | The analysed subject. |
| `subject_kind` | `str` | required | The subject's kind; empty when the campaign declared none. |
| `behavior` | `str` | required | The behaviour under analysis. |
| `generated_at` | `str` | required | When the analysis was generated (ISO-8601); on a code-only report, when the evidence was assembled for it. |
| `generator_model` | `str \| None` | `None` | The model that wrote the analysis, as the provider reported it; None on a code-only report. |
| `bundle_fingerprint` | `str` | required | The fingerprint of the evidence bundle the analysis was written over; on a code-only report, of the bundle the report was computed from. |

#### `TextBlock`

What the analysis's author wrote, exactly as written, with the facts code states beside it.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position in the document (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `kind` | `Literal['text']` | `'text'` | Which kind of block this is. |
| `role` | `TextRole` | required | What the author wrote it as. |
| `body` | `ModelProse` | required | The author's words, Markdown allowed; empty where the author left a required field blank. |
| `facts` | `list[Fact]` | `[]` | Facts code states beside the words, in reading order. |

#### `TableBlock`

A table code laid out — its columns, its rows in their stated order, and how much of it is shown.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position in the document (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `kind` | `Literal['table']` | `'table'` | Which kind of block this is. |
| `name` | `str` | required | Which table this is: `evidence`, `arms`, `surface`, `unadjudicated_bars`, `comparisons` (the contrasts against the control, as code tested them), `questions` (the declared questions, on a code-only report), `strata` (each arm's figures per stratum of its cases, beside its pooled figure, when its cases declare strata) or `labels` (a classifier's per-label precision, recall and F1, a row per label and arm, on a code-only report). |
| `title` | `str` | required | The table's heading. |
| `columns` | `list[TableColumn]` | required | The columns, in display order. |
| `rows` | `list[dict[str, Cell]]` | required | The rows shown, in the stated order, keyed by column key. |
| `order` | `str` | required | The order the rows are in, in words. |
| `total_rows` | `int` | required | How many rows the table has; more than are shown when it is truncated. |

#### `ChartBlock`

A chart: a finding's (its intent, or why the stored chart cannot be drawn), or one code chose.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position in the document (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `kind` | `Literal['chart']` | `'chart'` | Which kind of block this is. |
| `viz_type` | `ChartType` | required | The chart type the finding carries. |
| `intent` | `ChartIntent \| None` | `None` | What the chart draws and must say; None when it cannot be drawn. |
| `error` | `str` | `''` | Why the stored chart cannot be drawn, naming the offending field; empty when it can. |

#### `DisclosureBlock`

Something code must tell the reader that no author wrote — one idea.

| Field | Type | Default | Description |
|---|---|---|---|
| `section` | `ReportSection` | required | The section the block sits in. |
| `finding` | `int \| None` | `None` | The finding this block belongs to, by its position in the document (0 is the first); None when it belongs to none. |
| `rests_on` | `list[int]` | `[]` | Positions of the findings this block rests on, as the author linked them. |
| `kind` | `Literal['disclosure']` | `'disclosure'` | Which kind of block this is. |
| `source` | `DisclosureSource` | required | What the disclosure speaks for. |
| `text` | `str` | required | The disclosure, one sentence or a few. |

<a id="analysis-bundle"></a>
### `AnalysisContextBundle`

The closed context bundle a generation prompt runs over.

Its top-level fields, in declaration order; each one's type is described in the API section.

| Field | Type | Default | Description |
|---|---|---|---|
| `schema_version` | `int` | `47` | Bundle-shape version, for future evolution + fingerprint clarity. |
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
| `frontier_bar_withheld` | `str \| None` | `None` | Set when the frontier lens was given no bar, which is how this bundle always assembles it. |
| `telemetry` | `TelemetryRollup` | required | Campaign-wide descriptive telemetry. |
| `coverage` | `list[LeverCoverageInput]` | `[]` | Per-lever structural coverage map (the analysis's spine). |
| `scope_divergences` | `list[ScopeDivergence]` | `[]` | Lever changes where the whole-run measure moved by a different amount than the isolating measure, the difference itself tested and Holm-corrected within the lever — each one is a finding. |
| `divergences_omitted` | `int` | `0` | Gated divergences beyond the reporting cap, dropped weakest-first. Stated so a short list is not read as a complete one. |
| `divergences_tested` | `int` | `0` | Whole-and-part pairs across every lever whose divergence test carried a p, published or not. |
| `divergences_untested` | `int` | `0` | Whole-and-part pairs whose divergence could not be tested: fewer than two cases carrying both measures on a side, or a remainder with no spread over too few cases for an exact test. |
| `declared_design` | `CampaignDesign \| None` | `None` | What the campaign SET OUT to do, carried whole rather than reduced to its control. |
| `design` | `RealizedDesign` | `RealizedDesign(control_arm=None, control_excluded=None, contrasts=[], unplaced_run_ids=[], shape='undesignated')` | What kind of experiment this is — the run carrying the declared control, each cell's moved levers, and whether the design is one-factor-at-a-time. |
| `incomplete_runs` | `dict[str, str]` | `{}` | Run id → status, for every resolved member run that did not reach 'completed'. |
| `short_runs` | `dict[str, str]` | `{}` | Run id → the disclosure sentence for every resolved member run that delivered fewer cells than it promised, whatever status it ended on. |
| `completeness_unknown_run_ids` | `list[str]` | `[]` | Resolved member runs carrying no completeness record, so whether they came up short is unknown rather than answered. |
| `short_cells` | `list[ShortCell]` | `[]` | Every cell holding fewer repetitions than the declaration's `intended_repetitions`, ordered by (variant_key, apparatus_class_id), each with the sentence to quote. |
| `cost_unmeasured_cells` | `list[CellCoordinate]` | `[]` | Every cell where no turn the candidate took observed spend — no usage row in its cost roles carried dollars — ordered by (variant_key, apparatus_class_id). |
| `cost_unmeasured` | `str \| None` | `None` | The sentence to quote about `cost_unmeasured_cells` — that cost was not measured there, and why a $0 would mean nothing. |
| `all_failed_cells` | `list[CellCoordinate]` | `[]` | Every cell where no result the harness did not fault took a turn — the candidate's model refused or errored on every call — ordered by (variant_key, apparatus_class_id). |
| `all_failed` | `str \| None` | `None` | The sentence to quote about `all_failed_cells` — that every result there failed, so there is no cost or latency to read. |
| `held_fixed_reading` | `HeldFixedReading` | `HeldFixedReading(declared_stimulus=None, stimulus_reason='', declared_apparatus=None, run_provenance={}, contradicting_run_ids=[], disclosure=None)` | What the campaign declared held fixed beside the provenance every resolved run recorded, compared value for value, with the sentence to quote when they disagree, when runs mix commissioned and witnessed apparatus with nothing declared, or when the stimulus was declared uncontrolled. |
| `judge_agreement` | `JudgeAgreement` | `JudgeAgreement(ratings_read=0, dimensions=[], unpaired=[])` | How the judge's scores agreed with people's calibration ratings of the same results, per judged dimension, judge model and judge config: n, distinct results, exact agreement, Cohen's kappa and, on 1-5 dimensions, quadratic-weighted kappa, each pooled over people by result — over every resolved member run's results. |
| `judge_self_agreement` | `JudgeSelfAgreement` | `JudgeSelfAgreement(repeats_read=0, dimensions=[], unpaired=[])` | How the judge's repeated scores agreed with its own first scores of the same evidence, per judged dimension, judge model and judge config, read exactly as `judge_agreement` is (n, distinct results, exact agreement, kappa, weighted kappa, a "can't tell" repeat counted as a disagreement) — over every resolved member run's results. |
| `judge_evidence_tiers` | `list[JudgeEvidenceTier]` | `[]` | The evidence tier of each judge's readings on each judged dimension — a judge being a served model and a judge config — decided by code from `judge_agreement` and `judge_self_agreement`, each criterion on confidence bounds for its agreement and never the point estimate: `calibrated` (the one-sided 95% lower bound on agreement with people at or above 0.6, over at least 20 distinct results), `separation` (that bound on agreement with its own repeats at or above 0.8, over at least 120 distinct results), `incidental` (both upper bounds below their bars), or `undetermined` (not shown either way: too few results, or bounds across a bar). |
| `goal_check_proofs` | `list[GoalCheckProofReading]` | `[]` | Per goal check the member runs graded: whether it was shown, at launch, to tell its outcomes apart (`proven`), or not (`unproven`: no control, or a proof recorded under an earlier rule (`stale`); `refuted`: a control it does not beat, or a check the grammar refused at launch (`refused`)). |
| `multiple_comparisons` | `MultipleComparisons` | `MultipleComparisons(families=[], withheld=None)` | Each contrast tested against the control on every reading a live question asks about, per rig, with Holm correction inside each question's family: the family's size, each comparison's adjusted p and the verdict read off it. |
| `guardrails` | `GuardrailReadings` | `GuardrailReadings(measures=[], dimensions=[], checks=[], withheld=None, unstamped_dimensions=[])` | The guardrails — boundary judged dimensions and measures declared `guardrail`, what the candidate must not get worse on — each decided for every arm against the control on its own 95% interval: `held` (shown no worse than its margin), `breached` (shown worse) or `undecided`. |
| `reading_scope` | `ReadingScope` | `ReadingScope(questions_declared=False, exploratory_measures=[], exploratory_dimensions=[], disclosure=None)` | Which readings no declared question asked about: exploratory, reportable as leads and never as confirmed answers. |
| `verdict_order` | `VerdictOrder` | `VerdictOrder(merit_priority=[], tiers=[], unranked_bar_measure_ids=[], questions=[])` | The order verdicts are read in, as declared: the bars on each axis of `merit_priority`, strongest first, the bars on no ranked axis, and for each live question the bars on the axes it names. |
| `measurement_windows` | `list[MeasurementWindow]` | `[]` | Each resolved member run's wall-clock measurement span, derived from that run's own scored_at stamps. |
| `launch_disclosure` | `str \| None` | `None` | Set when the member runs were NOT all started by one campaign launch — some came from different launches, or were started on their own. |
| `measurement_window_disclosure` | `str \| None` | `None` | Set when AT LEAST ONE PAIR of member runs was measured over spans of wall-clock time that do not overlap — so anything that moved between those spans (a model revision, a provider's load, a rate limit) moved with the runs, and a difference between the two runs of such a pair is not attributable to the runs alone. |
| `apparatus_confounds` | `list[Confound]` | `[]` | Apparatus dimensions that varied across the WHOLE campaign, scanned independently of any lever. |
| `arm_mechanisms` | `list[ArmMechanismReading]` | `[]` | Each arm's mean of every covariate read as an observed mechanism (today the candidate's reasoning share, `reasoning_ratio`), sorted by arm then covariate. |
| `arm_served_models` | `list[ArmServedModel]` | `[]` | Which model the provider's responses named as having answered each arm's candidate calls, sorted by arm. |
| `cell_model_version` | `int` | `11` | Which definition of a cell produced `cells`. |
| `cells` | `list[Cell]` | `[]` | Every (variant, apparatus class) that any observation landed in, with how many observations pooled there and over how many cases and repeats per case. |
| `variant_index` | `list[VariantIndexEntry]` | `[]` | One entry per keyed variant — the variant key and the resolved lever map it was digested from. |
| `refused_merges` | `list[RefusedMerge]` | `[]` | Same-variant cells that did NOT pool, and which rule kept them apart. |
| `next_experiments` | `list[NextExperiment]` | `[]` | What recording one unrecorded apparatus dimension would buy, in units of k. |
| `subject_key_instabilities` | `list[SubjectKeyInstability]` | `[]` | Subject keys and labels disagreeing about how many subjects there are — one key under two labels, or one label under two keys. |
| `measure_catalog` | `dict[str, MetricDescriptor]` | `{}` | What each measure name MEANS — the registry descriptor, carried once per campaign rather than repeated on every run's summary. |
| `judged_measures` | `list[JudgedMeasure]` | `[]` | Every judged dimension any result was scored on, with its per-cell scores. |
| `bar_adjudications` | `list[BarAdjudication]` | `[]` | Every bar this campaign is held to — its own declared bars, plus each registered incumbent for its behavior that no declared bar overrides — with a verdict per cell computed here, or the reason none exists. |
| `cell_measures` | `list[CellFacts]` | `[]` | Everything measured in each cell, one entry per cell, the declared control's cells first as the reference, then every other arm alphabetically by name (an order that is not a ranking): every measure over the cell's non-faulted observations — the population every bar verdict is read over, so a value here and a verdict on the same cell describe the same observations — every judged dimension scored there, its replication, and the notes on its member runs. |
| `time_axis` | `TimeAxis \| None` | `None` | The runs placed in time, when they span two or more builds (the host's release label) or, failing that, two or more days: each position names its runs and carries every cell measured there, computed exactly as `cell_measures` is over that position's runs alone. |
| `time_axis_withheld` | `str \| None` | `None` | Why there is no time axis — what every run shared — or None when there is one. |
| `confound_catalog` | `dict[str, str]` | `{}` | What a change in each confound dimension DOES to a measurement, carried once per campaign rather than repeated on every lever that names it. |
| `prior_insights` | `list[EvalInsight]` | `[]` | Subject-scoped prior insights (newest first). Every insight minted by an ARCHIVED analysis is left out and named in `retracted_insights` instead. |
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
| `runs_list` | `read` | `status?`, `include_archived?` | List the scope's runs, newest first. |
| `campaigns_list` | `read` | `include_archived?` | List the scope's campaigns — the sets of runs an analysis reads. |

### Run and watch

| Action | Class | Parameters | Description |
|---|---|---|---|
| `run_launch` | `spend`, job | `template_id`, `subject_id`, `models?`, `k_runs?`, `n_variations?`, `variation_model?`, `overlays?`, `apparatus_settings?`, `max_cost_usd?`, `judge_model?`, `simulator_model?` | Launch a template's runs, one per model, each as a job to poll. |
| `launch_estimate` | `read` | `template_id`, `subject_id`, `models?`, `k_runs?`, `n_variations?`, `variation_model?`, `overlays?`, `apparatus_settings?`, `max_cost_usd?`, `judge_model?`, `simulator_model?`, `n_test_cases?` | Estimate what a run_launch would cost, from what the scope's runs have spent. |
| `job_poll` | `read` | `job_id` | Read where a job stands — a launched run or an analysis generation. |
| `job_cancel` | `write` | `job_id`, `reason?` | Ask a running job to stop; poll to see it land as cancelled. |
| `run_get` | `read` | `run_id` | Summarise one run: how it ended, how its results came out, each measure's mean. |
| `results_list` | `read` | `run_id`, `condition_filter?`, `offset?`, `limit?` | List one run's results, a light row each — case, repeat, condition, headline measures — paged. |
| `result_get` | `read` | `result_id`, `part?` | Read one stored result: its record, usage rows and condition, and one part of its trace. |

### Analyse and report

| Action | Class | Parameters | Description |
|---|---|---|---|
| `campaign_create` | `write` | `name`, `subject_id`, `behavior`, `description?`, `run_ids?`, `declared_design?`, `control_from_run_id?` | Create a campaign over runs in the scope, for an analysis to read. |
| `analysis_generate` | `spend`, job | `campaign_id`, `model?` | Generate a campaign's analysis with a paid model call, as a job to poll. |
| `analysis_estimate` | `read` | `campaign_id`, `model?` | Price a campaign's analysis generation against the host's out-of-run cap, without spending. |
| `analyses_list` | `read` | `campaign_id` | List a campaign's stored analyses. |
| `report_read` | `read` | `campaign_id`, `format?` | Read a campaign's report — its analysis, else its evidence alone — as Markdown, JSON or HTML. |
| `reporter_case_freeze` | `write` | `template_id`, `campaign_id`, `recorded_analysis_id?`, `labels?`, `supersedes?` | Freeze a campaign's analysis bundle, and the memo it got, into a case of a reporter template. |
| `reporter_cases_list` | `read` | `template_id`, `include_archived?` | List a reporter template's cases: which each campaign and memo launches, superseded or retired. |
| `scope_pivot` | `read` | `row_factor`, `column_factor`, `metric?`, `weighting?`, `subject_filter?`, `run_status?`, `predicted_cost?`, `launched_run_ids?` | Aggregate one measure over the scope's observations by two coordinates, cell by cell. |
| `runs_compare` | `read` | `baseline_run_id`, `candidate_run_id` | Compare one run's arm against another's: pass^k, mean composite, their deltas and the test. |
| `scope_out_of_run_spend` | `read` | `purpose_filter?`, `launch_group_filter?`, `template_filter?` | List what the engine spent outside any run — case generations, rubric proposals and analysis generations — with totals. |
| `scope_history` | `read` | `metric?`, `min_absolute_change?`, `min_relative_change?`, `subject_filter?`, `run_status?` | Series one measure over time for each contestant in the scope, flagging regressions. |
| `scope_export` | `read` | `export_format?`, `run_status?`, `export_run_ids?` | Export the scope's observations as flat rows, CSV or JSON, for analysis elsewhere. |

### Curate

| Action | Class | Parameters | Description |
|---|---|---|---|
| `run_archive` | `write` | `run_id`, `archived?` | Archive a run (or restore it): out of every cohort, nothing destroyed. |
| `campaign_archive` | `write` | `campaign_id`, `archived?` | Archive a campaign (or restore it): out of listings, nothing destroyed. |
| `result_rate` | `write` | `result_id`, `rubric_dim`, `score`, `rating_reason` | Rate one judged dimension of one result, as an agent — kept beside people's ratings, never pooled. |
| `analysis_archive` | `write` | `analysis_id`, `archived`, `archive_reason?` | Archive a stored analysis (or restore it): marked, its insights retracted, nothing destroyed. |
| `reporter_case_archive` | `write` | `reporter_case_id`, `archived?`, `archive_reason?` | Retire a reporter case (or restore it): no launch runs it again, nothing destroyed. |
| `run_delete` | `destructive` | `run_id`, `confirm` | Destroy a run, its results and its campaign memberships. Unrecoverable; archive instead. |
| `analysis_delete` | `destructive` | `analysis_id`, `confirm` | Destroy a stored analysis; the insights it minted remain. Unrecoverable; archive instead. |

<a id="action-parameters"></a>
### Parameters

| Parameter | Type | Description |
|---|---|---|
| `analysis_id` | `string` | A stored analysis's id, as analyses_list names it. |
| `apparatus_settings` | `object` or `null` | Host-declared apparatus values to set the runs' rig up with, by apparatus dimension (e.g. who sits in an adjudicator's seat) — each a string, a bool or a number, and one the template's kind reads; refused otherwise. Recorded on every run and part of its measurement context, so one template can be compared at two. |
| `archive_reason` | `string` or `null` | Why the record is archived (an analysis shown false or superseded, a reporter case that can no longer measure anything); cleared on restore. |
| `archived` | `boolean` | The state to set: true retires the record, false restores it. |
| `baseline_run_id` | `string` | The run read as the baseline (A), as runs_list names it. |
| `behavior` | `string` | Which aspect of the subject is under test. |
| `campaign_id` | `string` | A campaign's id, as campaigns_list names it. |
| `candidate_run_id` | `string` | The run read against the baseline (B), as runs_list names it. |
| `column_factor` | `string` | The coordinate the columns are; a pooled ranking across the rows is checked for reversal on it. |
| `condition_filter` | `'ok'` \| `'candidate_fail'` \| `'infra_exclude'` or `null` | List only results in this condition: ok (delivered, scored), candidate_fail (scored as a hard fail) or infra_exclude (a harness fault, in no aggregate). |
| `confirm` | `string` | Must echo the id of what is destroyed, exactly. |
| `control_from_run_id` | `string` or `null` | One of run_ids whose variant becomes the declared control, the cell every other is read against: its variant key is resolved from the run, as designating a control on an existing campaign resolves it. Requires declared_design. |
| `declared_design` | `object` or `null` | What the campaign sets out to learn, declared before it learns anything: axes (at least one: {axis_id, values: [{content, display}], rationale?}, each axis_id a lever or open-family member this host declares), held_fixed ({stimulus: controlled\|uncontrolled, stimulus_reason (required when uncontrolled), apparatus: commissioned\|witnessed}), and optionally questions ([{id, text, merit_axes?}]), bars ([{measure_id, threshold, direction}], no looser than the registered ones), merit_priority and intended_repetitions. Validated and gated as every campaign declaration is; omitted, the campaign is undeclared and its analysis infers the design from the runs. Its control is named by control_from_run_id. |
| `description` | `string` | A longer description of the campaign. |
| `export_format` | `'csv'` \| `'json'` | The export's form: csv (flat rows, a column per lever) or json (the rows and what was left out). |
| `export_run_ids` | array of `string` or `null` | Export only these runs, archived ones included since they are named; omitted exports all. |
| `format` | `'markdown'` \| `'json'` \| `'html'` | The report's form: markdown (the memo), json (the schema's form), html (script-free). |
| `include_archived` | `boolean` | List archived records too; they are left out by default. |
| `job_id` | `string` | A job's id, exactly as the action that started it returned it. |
| `judge_model` | `string` or `null` | The judge model, where the kind is model-judged. |
| `k_runs` | `integer` | Repeats of every case, for pass^k. |
| `labels` | array of `object` | Reader verdicts on the recorded memo, each {dimension, direction, quote, note?}: a dimension the template scores, a direction of low\|low_mid\|mid\|mid_high\|high, and the reader's words verbatim. They need recorded_analysis_id; the freeze stamps each with the template's criterion for its dimension. |
| `launch_group_filter` | `string` or `null` | Only the calls one launch's case generation made; its runs carry this id. |
| `launched_run_ids` | array of `string` or `null` | The runs the estimated launch made (the run ids its jobs name), with predicted_cost: each predicted cell then says how many of its observations came from other runs. |
| `limit` | `integer` | The most rows to return, up to 200. |
| `max_cost_usd` | `number` or `null` | A per-run cost cap in dollars, at or below the host's ceiling; it can only lower that ceiling, and a value above it is refused. |
| `metric` | `string` or `null` | The measure to read; omitted reads the composite score. |
| `min_absolute_change` | `number` | The smallest move a regression flag counts, in the measure's unit; 0 lets the test decide. |
| `min_relative_change` | `number` | The smallest move from the baseline a flag counts, as a fraction; 0 lets the test decide. |
| `model` | `string` or `null` | The analysis generator's model; omitted for the host's. |
| `models` | array of `string` | Candidate models, one arm and one run each; empty runs the kind's own default. |
| `n_test_cases` | `integer` or `null` | A case count to price each planned arm at in place of its plan's, for a what-if grid. |
| `n_variations` | `integer` | New cases to generate from the template's variation axes; 0 runs its stored cases. |
| `name` | `string` | The campaign's name, as an operator reads it. |
| `offset` | `integer` | Skip this many rows; the previous page's next_offset. |
| `overlays` | `object` or `null` | The knobs this launch turns on the template's kind, by field. |
| `part` | `'record'` \| `'judge'` \| `'spans'` | Which part of the stored trace to return: record (the output the kind stored, its call ledger and end state, beside the result), judge (what the judge was sent) or spans (the stored spans). |
| `predicted_cost` | `object` or `null` | A cost pivot's plan: the structured result launch_estimate returned before these runs. The cell at each priced arm's model and template then shows its predicted cost beside the cost observed. |
| `purpose_filter` | `'variation'` \| `'proposer'` \| `'analysis'` \| `'judge'` or `null` | Only calls made for this purpose: variation (a launch's case generation), proposer (a rubric draft) or analysis (an analysis generation). |
| `rating_reason` | `string` | Why that score, in the rater's own words. |
| `reason` | `string` or `null` | Why, recorded on a cancelled run. |
| `recorded_analysis_id` | `string` or `null` | A stored analysis of the campaign whose memo the case pins as the one it got, as analyses_list names it; omitted freezes the bundle alone, which only a generating candidate can run. |
| `reporter_case_id` | `string` | A reporter case's id, as reporter_case_freeze or reporter_cases_list returns it. |
| `result_id` | `string` | A result's id, as results_list names it. |
| `row_factor` | `string` | The coordinate the rows are: a declared one (model, template_id, ...) or a dotted lever. |
| `rubric_dim` | `string` | The judged dimension rated, spelled as the result's score spells it. |
| `run_id` | `string` | A run's id, as runs_list or a launch's job names it. |
| `run_ids` | array of `string` | The runs to put in the campaign, all in the caller's scope. |
| `run_status` | `'pending'` \| `'running'` \| `'completed'` \| `'failed'` \| `'cancelled'` \| `'budget_stopped'` \| `'exhausted'` or `string` | Read only runs with this status, or 'all'. Completed unless named: a run still going is still adding results. |
| `score` | `integer` | The score, on the dimension's scale: 1-5, or 1 (pass) / 0 (fail). |
| `simulator_model` | `string` or `null` | The simulated user's model, where the kind has one. |
| `status` | `'pending'` \| `'running'` \| `'completed'` \| `'failed'` \| `'cancelled'` \| `'budget_stopped'` \| `'exhausted'` or `null` | List only runs with this stored status. |
| `subject_filter` | `string` or `null` | Read only this subject's runs; omitted reads every subject. |
| `subject_id` | `string` | The subject the runs measure, as the host names it. |
| `supersedes` | array of `string` | The live case(s) of this campaign and memo the freeze replaces, by id — needed to revise a case's labels or re-freeze moved evidence; every live case of the pair when it holds several. |
| `template_filter` | `string` or `null` | Only calls made for this template, by id. |
| `template_id` | `string` | A template's id, as templates_list names it. |
| `variation_model` | `string` or `null` | The model that writes the template's llm variation axes' values; required when n_variations generates for such an axis, refused otherwise. |
| `weighting` | `string` or `null` | How a cell averages its observations; omitted takes equal per scenario. |

<a id="cli"></a>
## The command line

`python -m threetears.evals <command> --host MODULE:FACTORY --scope SCOPE [options]`

A product mounting the commands under its own CLI (`run_cli(host_factory=...)`) drops `--host`. Every command takes `--scope`.

Exit codes, in full: 0 done; 1 a launched run did not complete; 2 refused; 3 failed with an unanticipated error.

Each command's options, as its `--help` prints them:

### `run`

```text
usage: python -m threetears.evals run [-h] --host MODULE:FACTORY --scope SCOPE --template TEMPLATE
                                      --subject SUBJECT [--model MODEL] [--k K]
                                      [--max-cost-usd DOLLARS] [--judge-model MODEL]
                                      [--simulator-model MODEL] [--n-variations N]
                                      [--variation-model MODEL] [--apparatus-settings JSON]

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
                                         [--format {markdown,json,html}] [--out PATH]
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
                        markdown (default), html (needs no script) or json (the published schema's
                        form)
  --out PATH            write the report to PATH instead of stdout
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
                                        [--purpose {variation,proposer,analysis,judge}]
                                        [--launch-group ID] [--template ID]

Print what the engine spent outside any run — case generations, rubric proposals, analyses.

options:
  -h, --help            show this help message and exit
  --host MODULE:FACTORY
                        the host to work in
  --scope SCOPE         the scope to read and write in
  --purpose {variation,proposer,analysis,judge}
                        only this purpose's calls
  --launch-group ID     only one launch's case generation
  --template ID         only calls made for this template
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
| `execution_mode` | — | — | — | Whether the observation was made while other eval jobs were executing — a condition, not a result. |
| `n_results` | — | — | results | How many results the group aggregates — the denominator behind its means. |
| `n_total_ms` | — | — | results | How many results contributed a measured total_ms — the denominator behind mean/median/p95/max total_ms, which can be smaller than n_results because latency components are nulled independently. |
| `n_llm_ms` | — | — | results | How many results contributed a measured llm_ms — the denominator behind mean_llm_ms. |
| `n_tool_ms` | — | — | results | How many results contributed a measured tool_ms — the denominator behind mean_tool_ms. |
| `n_cost_usd` | — | — | results | How many results' spend was priced — the denominator behind mean_cost_usd. |
| `n_prod_cost_usd` | — | — | results | How many results measured a production-replicating cost — the denominator behind mean_prod_cost_usd. |
| `n_test_cases` | — | — | cases | How many test cases the model was run against. |
| `k` | — | — | iterations | The depth the headline pass_hat_k is read at. |
| `n_cannot_tell_excluded` | — | — | iterations | Iterations left out of pass^k because the judge answered it could not score a rubric dimension from the evidence. |
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
| `async_delivery_elapsed_n` | — | — | deliveries | How many deliveries had a measured duration — absent rather than zero when none did. |
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
| `pass_hat_k_a` | higher | — | — | Reliability (pass^k at the shared depth k) for run A. |
| `pass_hat_k_b` | higher | — | — | Reliability (pass^k at the shared depth k) for run B. |
| `pass_hat_k_delta` | higher | — | — | Change in reliability (pass^k at the shared depth k) from run A to run B. |
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
| `support` | — | — | cases | How many cases carried this expected label — the denominator behind its precision and recall. |
| `match` | higher | — | — | Whether one classification matched its expected label. |
| `confusion_cell` | — | — | — | Which cell of the confusion matrix one classification landed in. |
