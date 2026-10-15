# The surface: every declared name, by rung

**For** a reader who wants the whole declared API on one page. **Answers:** which names exist at each rung and
where each is imported from. What a name does is in the [reference](reference.md#public-api), which lists the
same names; what each tier promises is in [stability tiers](adopting-a-host.md#stability-tiers).

About 450 names are declared: the `host-contract` and `stable` tiers. A root exports more, as `provisional`
names, most of them the types a declared signature reaches; they import, and nothing about them is promised.
Each line below is one root, `threetears.evals.` dropped from its name. `tests/test_api_tiers.py` holds this page
to `HOST_CONTRACT` and `STABLE` in `packages/evals/scripts/public_surface.py`.

## Rung 0: a TOML file

Run an eval written as a file from the command line, or wire a host's own commands into the same entry; the exit codes are what a CI step reads.

- `quick`: `EXIT_FAILED`, `EXIT_GATE_FAILED`, `EXIT_OK`, `EXIT_REFUSED`, `EXIT_RUN_DID_NOT_COMPLETE`, `HostCommand`, `HostFactory`, `run_cli`

## Rung 1: an eval and a grader

Cases, the code under test and what counts as right, run k times; the result, read whole or case by case.

- `threetears.evals`: `Answer`, `Eval`, `grader`, `Model`, `Result`
- `analysis`: `CaseResult`, `GLOSSARY`, `humanize`, `LabelStatistics`, `lint_prose`, `StyleViolation`, `Verdict`
- `quick`: `Grader`, `Measure`, `measure`

## Rung 2: a judge, examples and calibration

A model grading on a rubric, the examples it must grade a known way, and the repeat, re-judge, agreement, drift and evidence-tier readings that say whether to trust it.

- `threetears.evals`: `Example`, `Judge`
- `schema`: `CalibrationRating`, `DEFAULT_JUDGE_TEMPERATURE`, `label_key_of`, `RubricDim`, `RubricScore`, `SecondJudge`
- `kernel`: `AuthoredJudgeCase`, `AuthoredVerdict`, `EvalJudgeProfile`, `JobLeases`, `JUDGE_KIND`
- `run`: `estimate_judge_repeat`, `freeze_judge_cases`, `judge_repeat_work`, `judge_temperature_work`, `JudgeRepeatReport`, `JudgeService`, `rate_result`, `regrade`, `run_job_work`, `second_judge_work`
- `analysis`: `inter_judge_agreement`, `judge_agreement`, `judge_drift`, `judge_evidence_tiers`, `judge_key`, `judge_kind_readings`, `judge_self_agreement`, `read_judge_temperatures`, `tier_sentence`
- `quick`: `JudgeCalibration`
- `ops`: `judge_profiles_record`, `JudgeProfilesRecord`

## Rung 3: comparing, reporting and gating

Two or more arms as a campaign, its bundle, report and frontier, the classifier measures, and the gate a pipeline fails on.

- `schema`: `EvalCaseStratum`, `LatencyMetrics`
- `kernel`: `BarOverride`, `CampaignDesign`, `confusion_cell`, `CONFUSION_CELL_MEASURE`, `ControlDeclaration`, `EvalAnalysis`, `EvalInsight`, `MATCH_MEASURE`, `Question`, `SweptAxis`
- `run`: `get_result_trace`, `get_run`, `list_results`, `list_runs`
- `analysis`: `AnalysisContextBundle`, `assemble_context_bundle`, `campaign_report`, `CELL_MODEL_VERSION`, `ChartBlock`, `create_campaign`, `EVAL_ANALYSIS_GEN_DEFAULT`, `EvalSummary`, `format_number`, `format_signed`, `frontier`, `GATE_TOKENS`, `gate_verdicts`, `GateOutcome`, `GateResult`, `GateToken`, `generate_analysis`, `inspect_campaign_bundle`, `Report`, `report_html`, `report_markdown`, `RequestSettingsReading`, `SentSettingsShare`, `SentSettingsState`
- `ops`: `scope_history`

## Rung 4: a world, a log, a host and serving

An agent acting on declared state, an app's own logged outputs, the host a product adopts the engine through, and the catalogue and MCP mount that serve it.

- `threetears.evals`: `Log`, `World`
- `schema`: `BoundCompletionClient`, `CallLedger`, `CellIdentity`, `CellTrace`, `ControlEndState`, `GoalCheckControl`, `GoalCheckControls`, `GoalStateOutcome`, `JudgeEvidence`, `Precondition`, `RecordedCall`, `WorldPlacement`
- `kernel`: `MeasureFamily`
- `kernel.host`: `ActsOn`, `ApparatusError`, `check_world_conformance`, `Interval`, `Ordinal`, `RolePins`, `StyleProfile`, `Triggered`, `WorldConformanceError`
- `run`: `ArmPlan`, `ArmPrice`, `ArmQuote`, `CellExecutor`, `create_template`, `FidelityContract`, `GoalCheckUnevaluable`, `grade_goal_checks`, `require_candidate_model`
- `quick`: `as_host`, `WorldOutcome`, `WorldRunner`
- `ops`: `OpsHost`, `RegressionWatch`
- `actions`: `Action`, `ActionCatalogue`, `ActionOutcome`, `Caller`, `ConformanceStage`, `eval_catalogue`, `standard_tools`
- `transports.fastmcp`: `CallerResolver`, `CallTimedOut`, `HostResolution`, `mount_fastmcp`, `ToolCall`

## The host contract

What a host writes against beyond the rungs: the host and profile, registries, the kind seam, launching, stored documents, the store port and its kits, refusals, versions, vocabularies and negotiation.

- `threetears.evals`: `__version__`, `declared_names`, `PUBLIC_ROOTS`, `ROOT_EXTRAS`, `RootExtra`, `surface_tier`
- `schema`: `AsyncDocumentStore`, `AsyncStoreClock`, `CASSETTE_SCHEMA_VERSION`, `CELL_TIMEOUT_ORIGINS`, `ClientRequestSettings`, `CompletionClient`, `CompletionProtocolDefect`, `CompletionResult`, `CORE_BASELINE_VERSION`, `CORE_SCHEMA_MINOR`, `CORE_SCHEMA_VERSION`, `COST_CAP_ORIGINS`, `DocumentStore`, `DURABLE_BASELINE_VERSION`, `DURABLE_SCHEMA_MINOR`, `DURABLE_SCHEMA_VERSION`, `EvalResult`, `EvalRun`, `EvalTemplate`, `EvalTestCase`, `EvalTrace`, `IntervalScale`, `JudgedArtifact`, `KnownCellTimeoutOrigin`, `KnownCostCapOrigin`, `KnownMeteredCallOrigin`, `KnownModelRoleOrigin`, `KnownOutOfRunPurpose`, `KnownReasoningEffort`, `KnownRefusalKind`, `KnownRoleModelOrigin`, `KnownSimulatorPurpose`, `KnownUsageRole`, `METERED_CALL_ORIGINS`, `MODEL_ROLE_ORIGINS`, `OUT_OF_RUN_PURPOSES`, `ProviderFailureDescriber`, `REASONING_EFFORTS`, `REFUSAL_KINDS`, `RefusalKind`, `REGENERABLE_SCHEMA_VERSION`, `ROLE_MODEL_ORIGINS`, `RoleUsage`, `sent_settings_of`, `SIMULATOR_PURPOSES`, `StopReason`, `StoreClock`, `StoreConflict`, `SubjectSnapshot`, `SweepableValue`, `TraceSink`, `USAGE_ROLES`, `WorldSeed`
- `kernel`: `AdmissionRefusedError`, `CandidateKind`, `CandidateKindDefect`, `CandidateOutput`, `CandidatePreparationFailed`, `CandidateTelemetry`, `CellCassettes`, `CellSink`, `CellSpanWindow`, `ConflictError`, `default_job_timeout`, `EvalCampaign`, `EvalServiceError`, `EvalStorage`, `IDENTITY_VERSION`, `MetricDescriptor`, `NotFoundError`, `ProviderRefusedError`, `StorageError`, `UnreadableRunError`, `ValidationFailedError`, `VariantConfig`, `withhold_failure_detail`, `WorldSession`
- `kernel.host`: `Bar`, `BarRegistry`, `CompletionClients`, `CompletionRole`, `default_cell_timeout`, `EvalHost`, `HostProfile`, `KindContract`, `load_bar_registry`, `MeasureRegistry`, `PassThreshold`, `SeedRefused`, `SHARED_CORE`, `Sweepable`, `SweepableRegistry`, `WorldDimension`, `WorldRegistry`
- `run`: `execute_run`, `JUDGE_REQUEST_SETTINGS`, `KindFactory`, `KindLauncher`, `KindWiring`, `launch_run`, `LaunchableKind`, `LaunchHost`, `LaunchRequest`, `LaunchSettings`, `price_subject_call_floor`, `run_blocking`, `RunnerOptions`, `SeedCaseSet`, `SeedJudgeCase`, `SIMULATOR_REQUEST_SETTINGS`, `start_run`, `SubjectResolver`, `UnknownSubject`
- `analysis`: `analysis_gen_request_settings_for`, `MIN_CASES`
- `storage`: `bridge_async_store`, `DocTypeRoute`, `InMemoryDocumentStore`, `RoutedDocumentStore`, `SqliteDocumentStore`
- `testing`: `check_completion_conformance`, `check_refusal_negotiation`, `CompletionGuard`, `jobs_conformance_cases`, `JobsConformanceHost`, `NegotiationConformanceFailure`, `RecordedCompletions`, `RecordedPrompt`, `RecordingRefused`, `RouterTransport`, `STORE_CONFORMANCE_CASES`, `StoreConformanceCase`
- `negotiation`: `ANTHROPIC_MIN_THINKING_BUDGET`, `anthropic_thinking`, `closest_settings`, `facts_from_router_catalogue`, `LearnedSettings`, `MAX_RESENDS`, `negotiate`, `Negotiated`, `ProviderFamily`, `read_anthropic_refusal`, `read_openai_refusal`, `read_router_refusal`, `RefusedSetting`, `SettingFacts`, `SettingRefusal`, `wire_settings_for`, `WireSettings`

## What a production host binds besides

Declared because the production host imports it: launch and judge plumbing, the reporter kind, prompt registry, metering, the routed store and the operations its routes call.

- `schema`: `ActorPolicy`, `AsyncDelivery`, `AsyncDeliveryStatus`, `AsyncExternalSpend`, `CatalogRubricDim`, `ConversationSpec`, `ConversationStopCause`, `DOC_TYPE_FACTS`, `DURABLE_DOC_TYPES`, `EVAL_DOC_TYPES`, `EvalBaseModel`, `ExternalSpend`, `JudgeConfig`, `nested_schemas`, `NON_TERMINAL_RUN_STATUSES`, `OutOfRunSpend`, `PROSE_SCHEMA_KEY`, `ProviderFailure`, `RefusedCall`, `RESERVED_DIM_IDS`, `RubricAxis`, `RubricProposal`, `VariationAxis`, `VariationCounts`
- `kernel`: `ActionSeam`, `AdmittedCall`, `DeliveryRecorder`, `DeliveryReplay`, `DeliverySeam`, `DeliveryTicket`, `EvalJobTimeout`, `ExclusionCause`, `ExternalRateTable`, `extract_json`, `FIND_TEMPLATE`, `JUDGE_KIND_CONTRACT`, `JUDGE_KIND_MEASURES`, `KIND_TEMPLATE`, `KIND_TEXT`, `MeritAxis`, `OutOfRunBudget`, `PlannedCall`, `prompt_drift`, `PromptDrift`, `Recordable`, `ReplayedDelivery`, `RoleUsageLedger`, `seed_digest`, `SEED_PROMPTS`, `SEED_REMEDY`, `SeedPrompt`, `SeedSection`, `SeedTemplate`, `ToolWrap`
- `kernel.host`: `check_seed`, `EvalCellTimeout`, `ResolvesInto`, `UnsupportedSchemaError`
- `run`: `AbandonedRunSweepReport`, `assert_preconditions`, `build_judge_service`, `CallCeiling`, `CandidateTurn`, `CellContext`, `delete_run`, `drive_conversation`, `fold_metered_cell`, `JUDGE_CALL_ATTEMPTS`, `JUDGE_MAX_TOKENS`, `launchable_judge_kind`, `metered_cell_tally`, `MeteredCallLedger`, `MeteredCallTally`, `plan_judge`, `precondition_failure_text`, `refuse_ignored_template_fields`, `resolve_judge_pin`, `run_judge_llm`, `SCHEDULER_CALL_ATTEMPTS`, `seed_eval_definitions`, `SeedCorpus`, `SeedOutcome`, `SIMULATOR_ANSWER_BUDGET_TOKENS`, `SIMULATOR_MAX_TOKENS`, `SimulatorTurn`, `sweep_abandoned_runs`, `TemplateField`, `TurnDriver`
- `analysis`: `AsRecordedReporterKind`, `CampaignReadStore`, `dollars_text`, `generation_ceiling_s`, `judge_phase_ceiling_s`, `refuse_undeliverable_reporter_template`, `reporter_cell_timeout_s`, `REPORTER_KIND`, `reporter_kind_contract`, `ReporterKind`, `ReporterOverlays`
- `gen`: `generate_variations`, `plan_draft`, `propose_draft`, `PROPOSER_MAX_TOKENS`
- `storage`: `CoreReset`, `MisplacedDocument`, `RouteChanged`, `ROUTING_RECORD_DOC_TYPE`
- `ops`: `admit_reporter_launch`, `AnalysisGeneration`, `campaign_create`, `CampaignDefinition`, `CampaignLine`, `CampaignListing`, `campaigns_list`, `EvalSummary`, `history_launch_pricer`, `job_cancel`, `job_poll`, `JobsStarted`, `JobStatus`, `LaunchArguments`, `plan_reporter_arm`, `price_largest_reporter_call`, `report_read`, `REPORTER_UNHONOURED_LAUNCH_ARGUMENTS`, `ReporterAdmitted`, `ReporterModelResolver`, `ReporterModelRole`, `run_get`, `run_label`, `run_launch`, `RunLine`, `RunListing`, `runs_list`, `start_action_job`, `start_paid_call`, `TemplateListing`, `templates_list`

## Providers

The shipped completion client: a model, the role factory a host hands its clients, the shared catalogue, the reply and the one error it raises.

- `providers`: `Catalogue`, `Clients`, `Completion`, `describe_failure`, `Model`, `PriceSource`, `Provider`, `ProviderError`
