# The surface: every declared name, by role

Each line is a name, `threetears.evals.` dropped, and the first sentence of its docstring; the
[reference](reference.md#public-api) has the rest.

## Writing an eval

### Your first eval: these 2

- `Eval`: What is evaluated: the cases, the runner under test, and what counts as right.
- `Model`: One model, as the engine's completion client and as three helpers for a runner of your own.

### [Rung 2: a judge, checked against a person](ladder.md#rung-2-a-judge-checked-against-a-person) (adds 2)

- `Judge`: A model that grades each answer on a rubric, one call per dimension: a grader in an eval's `grader` list.
- `Example`: An answer the judge must grade a known way: one that must fail a dimension, or one that looks wrong and must pass.

### [Rung 3: two prompts, compared](ladder.md#rung-3a-two-prompts-compared) (adds 1)

- `grader`: Declare what a reading needs beside its grade: `@grader(guard=0.02)`, `grader("total", margin=0.05)`.

### [Rung 4: an agent, graded on the world it leaves](ladder.md#rung-4a-an-agent-graded-on-the-world-it-leaves) (adds 3)

- `World`: A world: state each case seeds and tools the agent changes it with, or a set of lookups the agent calls.
- `quick.as_host`: A host for these evals: the vocabulary their graders, levers and world need, over one store.
- `transports.fastmcp.mount_fastmcp`: Add the catalogue's tools to a FastMCP server; or, handed a host alone, build the server and return it.

### [Rung 5: evaluate the evaluator](ladder.md#rung-5-evaluate-the-evaluator) (adds 3)

- `harness.AnalystSkill`: The analyst skill's files, checked whole, and each role's system text built from them.
- `harness.Writer`: One arm of a writer eval: the model that writes, the skill it starts from, and its reasoning effort.
- `harness.WriterEval`: The report writer, measured: arms of `Writer` over portable writer cases, replayed from recordings.

### Written in an eval, taught on a page rather than a rung (1)

- `Answer`: What a candidate returns to report its own spend beside its answer. Taught in [cost-and-budgets](cost-and-budgets.md).

## The host's contract: 87 names a host implements or hands the engine

### Ports a host implements (18)

- `schema.AsyncDocumentStore`: `DocumentStore`, every method awaited: the port for a host whose database driver is async.
- `schema.AsyncStoreClock`: `StoreClock` for an `AsyncDocumentStore`: the store's own time, awaited.
- `schema.BoundCompletionClient`: A completion client built for one model, which it names.
- `schema.CompletionClient`: The completion port eval is constructed with.
- `schema.CompletionResult`: What eval reads off one completion, whatever produced it.
- `schema.DocumentStore`: A narrow document store over one `(scope_id, doc_type, id)`-keyed collection.
- `schema.ProviderFailureDescriber`: The host's mapping from its own exceptions onto `ProviderFailure`.
- `schema.StoreClock`: A store's own clock: the optional capability a `DocumentStore` may offer beside its methods.
- `schema.ToolTurnClient`: The client a tool-calling loop is handed: one turn at a time, priced before it is sent.
- `schema.TraceSink`: The engine's whole reach into a host's tracing.
- `kernel.ActionSeam`: A kind's synchronous tools, as the cassette lane records and replays them.
- `kernel.CandidateKind`: The two operations that vary between one evaluable subject and another.
- `kernel.DeliverySeam`: One asynchronous tool of a kind, as the cassette lane records and replays its background work.
- `kernel.Recordable`: The two members the cassette layer uses on whatever it records: an action result or a delivered payload.
- `kernel.host.CompletionClients`: The host's completion-client factory: one client per role, model and temperature.
- `run.CellExecutor`: Runs a run's cells, at most `width` at once, starting them in the order given.
- `run.KindFactory`: Builds the candidate kind for one cell.
- `providers.Pricer`: What a call is billed at, as the host knows it: the port a `Model` is handed (`pricer=`), or `AnthropicPublishedRates` by default (the module docstring says when).

### What a host builds and hands the engine (69)

- `schema.CellTrace`: What one cell's tracing yielded — filled by the sink, read by the engine.
- `schema.EvalBaseModel`: Base model for the eval engine's own Pydantic models.
- `schema.ProviderFailure`: What eval is allowed to know about a completion call that raised.
- `schema.RunDisclosure`: What the launching host discloses about the launch a run came from: the machine, its load, how it measures.
- `kernel.default_job_timeout`: Enforce a job budget with `asyncio.timeout` and nothing else.
- `kernel.EvalStorage`: Storage for eval documents over one document store.
- `kernel.JobLeases`: The leases one process holds on its jobs' records, and the task that renews them.
- `kernel.RoleUsageLedger`: Accumulates one role's LLM spend across every call it makes in a cell.
- `kernel.ToolWrap`: What the action seam is handed: maps the candidate's tools, by name, to the tools it should use — each declared recorded tool wrapped for the cassette run, every other one unchanged.
- `kernel.withhold_failure_detail`: Describe a failure by its class and nothing else — the safe default.
- `kernel.WorldSession`: One cell's handle on the host's world, and the record of what happened to it.
- `kernel.host.ActsOn`: Name the measure or covariate an overlay knob is supposed to move — its lever's `acts_on`.
- `kernel.host.BarRegistry`: One host's registered bars, keyed by `(behavior, measure)` and validated at construction.
- `kernel.host.default_cell_timeout`: Enforce a cell budget with `asyncio.timeout` and nothing else.
- `kernel.host.EvalHost`: Everything the engine reads of one consuming product, as one frozen value.
- `kernel.host.HostProfile`: Everything the engine knows about one consuming product.
- `kernel.host.Interval`: Mark a numeric field's unit, rendered beside each level (`120words`).
- `kernel.host.KindContract`: What one candidate kind's runs carry, declared as Pydantic models — see the module docstring.
- `kernel.host.MeasureRegistry`: One host's declared measures, validated at construction.
- `kernel.host.ResolvesInto`: Name the fixed host lever this overlay knob is WRITTEN INTO — its lever's `resolves_into`.
- `kernel.host.SweepableRegistry`: The declared inputs for one host, validated at construction.
- `kernel.host.TemplateChecks`: The questions about a template, and about one of its own cases, that only the host can answer, and the kinds it launches, read wherever a template or a template case is admitted.
- `kernel.host.Triggered`: State that arrives on a condition rather than at t=0.
- `kernel.host.WorldRegistry`: One host's world declarations, its resolution table, and the algebra over them.
- `run.ArmPlan`: What one arm of a launch will run, as its kind says before its launcher runs.
- `run.CallCeiling`: The most one call of an arm's cells can cost, priced before launch: its largest call's ceiling, or a floor.
- `run.JudgeService`: Stateless single-dim judge.
- `run.KindLauncher`: One kind's launcher: builds that kind's collaborators for one arm and hands `launch_run` its run.
- `run.KindWiring`: What one kind's launcher resolved for its arm — everything `launch_run` cannot know itself.
- `run.launchable_judge_kind`: The judge kind's entry in a host's launch registry: each arm one judge over the template's frozen cases.
- `run.LaunchableKind`: One entry of the launch registry a host hands `start_run`: how a kind launches, and what it refuses.
- `run.LaunchGroup`: The runs of one launch, prepared together and started together — or not at all.
- `run.LaunchHost`: An `EvalHost`, and what starting its runs needs.
- `run.MeteredCallLedger`: In-memory per-run ceiling on metered third-party calls.
- `run.SubjectResolver`: How a kind answers whether its host holds the subject a launch of it names, before anything is priced: `(subject_id, scope_id) -> None` when it does, `UnknownSubject` when it does not.
- `run.TurnDriver`: Multi-actor round scheduler and utterance generator.
- `run.UnknownSubject`: A host's answer that it holds no subject of the id a launch named — and how its caller finds one it does.
- `run.UnpricedCall`: A call an arm's cells are bound to make that the clients they call through cannot price before launch.
- `storage.bridge_async_store`: Present `store` as the engine's synchronous `DocumentStore`.
- `storage.DocTypeRoute`: Which store each document type is kept in, decided by one of its facts.
- `storage.InMemoryDocumentStore`: A `DocumentStore` over one dict.
- `storage.RoutedDocumentStore`: One `DocumentStore` over several, each `doc_type` kept where its facts say.
- `storage.SqliteDocumentStore`: A `DocumentStore` over one SQLite file.
- `quick.HostCommand`: A subcommand a host adds beside the engine's own, mounted by `run_cli` under the same program.
- `quick.HostFactory`: What names the host the commands work in: called once, with no arguments, per invocation.
- `ops.AnalysisGeneration`: How this host generates an analysis as a background job: the report writer, on the host's model and budget.
- `ops.AnalystLimits`: What the analyst's rounds are held to, so every draft and revision ends within bounded calls and bounces.
- `ops.EditorLimits`: What each review round is held to, so every review ends within a bounded number of calls.
- `ops.history_launch_pricer`: The engine's launch pricer: an arm bounded from the scope's usage history of runs launched as it will be.
- `ops.OpsHost`: The host the operations, and the actions over them, work in.
- `ops.RegressionWatch`: Checks a completed run's measures against its contestant's history, and delivers each regression.
- `ops.ReportBudget`: The host's cap for one report, shared by every session that writes it: dollars, or tokens, and either with turns.
- `ops.SessionLimits`: What every turn of a session is held to, so no turn's reply or prompt grows past what its budget admitted.
- `ops.WriterLimits`: What the writing loop is held to, so every report ends within bounded rounds, whatever the reviews say.
- `actions.Action`: One action: what it is called, what it takes and returns, who may call it, and how it reads back.
- `actions.ActionCatalogue`: Every action a host offers: the engine's, and any the host contributes.
- `actions.analyst_tools`: The one tool carrying the analyst's actions: `<prefix>_analyst`, cut for the analyst or for its editor.
- `actions.Caller`: Who is calling, and the scope they act in — resolved by the host for every call.
- `actions.definition_drift_actions`: `definitions_drift` and `definitions_reset`, for the host's catalogue.
- `actions.eval_catalogue`: The engine's catalogue, the analyst's actions after the engine's, and the host's own after both.
- `actions.standard_tools`: The two tools a host mounts by default: `<prefix>` and `<prefix>_admin`.
- `transports.fastmcp.CallerResolver`: Resolves who is calling, and their scope, for one call — sync or async.
- `transports.fastmcp.HostResolution`: What a host resolver may answer in place of the bare host: the host, and state of the host's own for the hooks.
- `negotiation.LearnedSettings`: What providers refused, per provider and model, kept for as long as the callers that share it live.
- `providers.AnthropicPublishedRates`: The default pricer: Anthropic's published rates for a direct Anthropic call, the catalogue for anything else.
- `providers.Catalogue`: OpenRouter's model catalogue, refreshed every `refresh_after`, and what providers refused, for every model that reads it.
- `providers.Clients`: A host's role factory (`EvalHost.clients`): one `Model` per role, model id and temperature.
- `providers.describe_failure`: What the engine may know of a call through the client that raised: the host's `failure_describer`.
- `providers.Model`: the same as `Model`.

## Functions a host's code calls: 76

Each is `invoked`: your code imports it and calls it. None is an action; the actions follow.

### `threetears.evals` (1)

- `surface_tier`: A public name's stability tier in the installed build, by `root.name`.

### `schema` (6)

- `schema.bind_cell_identity`: Make `identity` the answer of `current_cell_identity` for the extent of the block.
- `schema.check_after`: Return `DocumentStore.by_doc_type`'s `after` checked, as a pair — the calls every implementation refuses.
- `schema.current_cell_identity`: Return the cell the calling task is working for, or `None` outside any cell.
- `schema.field_in_values`: Return `field_in` checked and each field's values with repeats dropped — the meaning of `field_in`.
- `schema.label_key_of`: The label key of one judgement: `evidence` read on `dim`.
- `schema.nested_schemas`: Every schema written directly inside `schema`, in declaration order.

### `kernel` (7)

- `kernel.confusion_cell`: The value a classification reports under `confusion_cell`: `expected → predicted`.
- `kernel.extract_json`: Extract a JSON object from provider output.
- `kernel.extract_json_array`: Extract a JSON array from provider output, handling markdown code blocks.
- `kernel.fold_phase_timings`: Fold one delivery's carried phase timings into a result's accumulator, and say what could not be folded.
- `kernel.prompt_drift`: Read a stored copy of a catalogue prompt against the package's shipped default.
- `kernel.seed_digest`: The canonical digest of a seed's content: what a host records when it seeds a slot from it.
- `kernel.strict_response_format`: The strict `json_schema` directive that asks a provider for exactly `schema`'s shape.

### `kernel.host` (1)

- `kernel.host.check_seed`: Walk a world seed against the registry: every write it would make, checked, or the first refusal.

### `run` (34)

- `run.assert_preconditions`: Assert at t=0 that the world this template presumes actually holds.
- `run.build_judge_service`: Build the `JudgeService` for a run, with its judge attribution.
- `run.corpus_digests`: The digest of every definition of a corpus, by its key: what a seed of it records.
- `run.create_template`: Create and persist a template from an authoring definition.
- `run.definition_drift`: Read every definition of a seed corpus against its scope's stored one.
- `run.delete_run`: Delete a run, the results it owns, and its campaign memberships.
- `run.drive_conversation`: Run `driver`'s conversation from its first turn until it stops on a structural signal.
- `run.estimate_judge_repeat`: Price repeating a run's judge scores against the cap it would be held to, and make no call.
- `run.execute_run`: Execute an entire `EvalRun` end-to-end.
- `run.fold_metered_cell`: Fold one cell's action-seam tally into the external role's rows.
- `run.freeze_judge_cases`: Freeze (output, criterion, labels) cases from stored judged results, or written by hand, into a judge template.
- `run.get_result_trace`: Load one result's trace payload, or `None` when it stored none.
- `run.get_run`: Load an eval run by id within its scope, whole.
- `run.grade_goal_checks`: Grade goal checks against a call ledger, an end state and what fired: the one evaluation every caller uses.
- `run.judge_repeat_work`: Admit asking a finished run's judge to score its results' scored dims again — every call priced, none made.
- `run.judge_temperature_work`: Admit re-judging a finished run's borderline cases at each temperature — every call priced, none made — as work.
- `run.launch_run`: Assemble, stamp, bound and hand over a run — the launch tail every candidate kind shares.
- `run.list_results`: List every result for one eval run within its scope, read a page at a time in the store's order.
- `run.list_runs`: List eval runs in a scope, optionally filtered by status.
- `run.metered_cell_tally`: One cell's slice of the run's metered-call tally, or `None` when nothing counted.
- `run.plan_judge`: The judges one arm of `template` will be scored by, resolved as `build_judge_service` resolves them — building nothing.
- `run.precondition_failure_text`: One line naming what was presumed and what the world held instead.
- `run.price_subject_call_floor`: A floor under a subject arm's largest judge or simulator call, for its `LaunchableKind.price_largest_call`.
- `run.rate_result`: Record one rater's score for one judged dimension of one result — a person's, or an agent's.
- `run.require_candidate_model`: The model `request`'s arm runs on — the one it named, or the kind's role default — or the refusal.
- `run.reset_definitions`: Write the corpus's content for the named definitions, each as a new revision or record, never in place.
- `run.resolve_judge_pin`: The run-level judge pin an arm is scored under: the launch's, or the role default stepped off a candidate.
- `run.run_blocking`: Run `fn(*args, **kwargs)` on `executor` and await its result.
- `run.run_job_work`: Write the work's record, leased, and run the work to its end in the caller's own frame.
- `run.run_judge_llm`: Run an LLM judge call against a set of criteria and parse the response.
- `run.second_judge_work`: Admit a second judge's pass over a finished run — every call priced and admitted, none made — as work to run.
- `run.seed_eval_definitions`: Create any corpus definition whose natural key is absent from the corpus's scope.
- `run.sweep_abandoned_runs`: Write lost every run left not ended by a process that has gone, in the scopes the host names.
- `run.update_template`: Apply a partial update to a template and persist it.

### `analysis` (16)

- `analysis.campaign_report`: The campaign's page — THE answer to "what is this campaign's report", for every caller and every campaign.
- `analysis.create_campaign`: Create and persist a campaign: one question, its arms, its aspects and its floors.
- `analysis.dollars_text`: Spend as a person reads it: two significant figures, and dollars and cents from ten cents up.
- `analysis.format_number`: Spell a number for a reader, the same way on every analysis surface.
- `analysis.frontier`: Rank each subject's variants on quality x cost x latency, cheapest above bar.
- `analysis.humanize`: An identifier as a reader is shown it after its first mention.
- `analysis.inspect_campaign_bundle`: Assemble one aspect's context bundle of a campaign and return it, without generating.
- `analysis.inter_judge_agreement`: Pair each second judge's score with the first score it answers, and read agreement the way calibration does.
- `analysis.judge_agreement`: Pair each rating with the judge's score on its dimension, on its result or the same output, and read agreement.
- `analysis.judge_drift`: Read how far each dimension's scores moved from the run's judge to a second judge on the same evidence.
- `analysis.judge_evidence_tiers`: Decide the evidence tier of every judge's readings on every dimension, from the two agreements.
- `analysis.judge_key`: The judge behind `result`'s score on `dim`, or None when it holds no score there.
- `analysis.judge_kind_readings`: Pool a judge campaign's trials into agreement with the labels, self-agreement and parse validity, per judge and criterion.
- `analysis.judge_self_agreement`: Pair each repeated score with the first score it repeated, and read agreement the way calibration does.
- `analysis.read_judge_temperatures`: Read a judge temperature comparison's answers into per-dimension variance and self-agreement, side by side.
- `analysis.tier_sentence`: One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.

### `gen` (3)

- `gen.generate_variations`: Generate up to `n_variations` test cases for a template.
- `gen.plan_draft`: The one call a rubric proposal sends, on either axis, built before anything is priced or made.
- `gen.propose_draft`: Make an admitted rubric proposal on `axis` and read its draft.

### `ops` (1)

- `ops.start_paid_call`: Start a host action's one paid model call: admitted against its cap in the call, made in a job.

### `negotiation` (7)

- `negotiation.anthropic_thinking`: The `thinking` an Anthropic request carries for `sent`, or `None` for none.
- `negotiation.facts_from_router_catalogue`: The facts a router's catalogue row states of a model: each parameter it does not list, as not taken.
- `negotiation.negotiate`: Send the closest settings the model is known to take; on a recognised refusal, learn it and send again.
- `negotiation.read_anthropic_refusal`: Whether Anthropic's answer refuses a request for a setting it carried, which, and in what words.
- `negotiation.read_openai_refusal`: Whether OpenAI's answer refuses a request for a setting it carried, which, and in what words.
- `negotiation.read_router_refusal`: Whether OpenRouter's answer refuses a request for a setting it carried, which, and in what words.
- `negotiation.wire_settings_for`: What a client asks for when it is asked for `settings` at `temperature`: the engine's values, as wire ones.

## The functions behind actions: 13

Each is `action`: the function an action of the catalogue (below) runs, under the same name. An agent or an
operator invokes the action by name; a host's code may also call the function.

- `ops.campaign_create`: Create a campaign over runs already in the scope, declared as it is created when the definition says so.
- `ops.campaign_record`: Record a person's call on a campaign — the arm decided, why, and each broken floor and missed bar it overrides.
- `ops.campaigns_list`: One page of the scope's campaigns, newest first, and the next page's cursor.
- `ops.job_cancel`: Ask a running job to stop — from any process — then report where it stands.
- `ops.job_poll`: Where a job stands, read from the record its work writes — the same answer in every process.
- `ops.judge_profiles_record`: Record the profile of every judge and criterion a judge campaign's runs measured.
- `ops.report_read`: The campaign's report, serialized in one form — the same report the command line's `report` prints.
- `ops.run_get`: One run, summarised from its stored results.
- `ops.run_label`: Set a run's labels, replacing any it has; an empty list clears them (#648).
- `ops.run_launch`: Admit a template's launch — one run per model — and start its arms and runs as one job.
- `ops.runs_list`: One page of the scope's runs, newest first, how many on it this build cannot read, and the next page's cursor.
- `ops.scope_history`: One measure over time for each contestant in the scope, with its regressions flagged.
- `ops.templates_list`: The scope's templates, each with its revision: a reset template's older revisions are archived, so the active listing shows each name once, at the revision a launch by name runs.

## The action catalogue: 87 actions an agent calls

`actions.eval_catalogue()` holds them; `transports.fastmcp.mount_fastmcp` serves them over MCP. An agent or
an operator invokes each by its name, with no Python, and the action does the work.

### Find what is there (6)

- `templates_list`: List the scope's active templates — what a launch can run.
- `judge_configs_list`: List the scope's judge configs — the versioned prompts a second judge or a launch can name.
- `runs_list`: List the scope's runs, newest first.
- `runs_unreadable`: List the scope's stored runs this build cannot read, each with why.
- `case_sets_list`: List the scope's named case sets, every version — what a launch can target by name.
- `campaigns_list`: List the scope's campaigns — the sets of runs an analysis reads.

### Run and watch (11)

- `sweep_launch`: Launch arms that differ beyond the model into one campaign, as one job.
- `case_set_mint`: Store the next version of a named case set: a template's cases, frozen in order.
- `run_launch`: Launch a template's runs, one per model, as one job to poll: its arms prepared, its runs run.
- `launch_estimate`: Estimate what a run_launch would cost, from what the scope's runs have spent.
- `job_poll`: Read where a job stands — a launch, a run, a sweep, a generation, a judge pass — and its result.
- `job_cancel`: Ask a running job to stop; poll to see it land as cancelled.
- `job_resume`: Resume an interrupted run, sweep or launch as itself: lost, cancelled, or stopped by a budget.
- `jobs_resumable`: List the interrupted runs, sweeps and launches job_resume would take up, with how each ended.
- `run_get`: Summarise one run: how it ended, how its results came out, each measure's mean.
- `results_list`: List one run's results, a light row each — case, repeat, condition, headline measures — paged.
- `result_get`: Read one stored result: its record, usage rows and condition, and one part of its trace.

### Analyse and report (37)

- `campaign_create`: Create a campaign: one question asked of its arms, measured on one or more aspects.
- `campaign_update`: Rename or redescribe a campaign, or declare, amend or withdraw its declaration before its first run.
- `campaign_runs_add`: Put runs in a campaign's aspect — runs launched after it was created, or measured on their own first.
- `campaign_runs_remove`: Take runs out of a campaign; the runs and their results stay, in any other campaign too.
- `campaign_record`: Record a person's call on a campaign: the arm decided, why, and each warning it overrides.
- `analysis_generate`: Write a campaign's report with the analyst and the report editor (paid model turns), as a job to poll.
- `analysis_estimate`: State the report budget a campaign's analysis generation would be held to, without spending.
- `judge_second`: Ask a second judge to score a seeded share of a finished run's judged results, as a job; read agreement.
- `judge_second_estimate`: Price a second judge's pass over a run against the host's out-of-run cap, without spending.
- `judge_drift_check`: Re-score every judged result of a run under a changed judge, as a job; read how far each moved.
- `judge_temperature`: Re-judge a run's borderline cases at temperature 0 and at the provider default, as a job; compare.
- `judge_temperature_estimate`: Price a judge temperature comparison against the host's out-of-run cap, without spending.
- `analyses_list`: List a campaign's stored analyses.
- `insights_list`: List the scope's insights — the durable claims analyses minted — newest first.
- `insight_get`: Read one insight in full: its statement, evidence runs, provenance and standing.
- `analyses_undescribable`: List the scope's analyses holding an arm whose levels this build cannot describe.
- `report_read`: Read a campaign's report — its analysis, else its evidence alone — as Markdown, JSON or HTML.
- `analyst_gaps_list`: List the capability gaps analysts noted, and code's notices, grouped and ranked as issue text.
- `bars_propose`: Propose bars from a single-cell baseline campaign's measured incumbent; registers nothing.
- `writer_template_create`: Write a report writer template, its scoring rubric seeded from the editor rubric this build ships.
- `writer_case_mint`: Freeze a campaign's report evidence into a writer case, listed in its side's dev or held-out set.
- `writer_promotion`: Read whether a report writer campaign's candidate arm should replace its control.
- `writer_pairwise`: Ask the scoring judge which page is better, candidate or control, for each case, as a job.
- `judge_template_create`: Write a judge template: the judge kind and an intent, which judge cases are frozen into.
- `judge_cases_freeze`: Freeze stored judged outputs, their criteria and their labels into cases of a judge template.
- `judge_cases_list`: List a judge template's stored cases: each id, criterion, expected verdict and whether it is live.
- `judge_case_author`: Write a judge case by hand: an output, the rubric dimension it is judged on, and its verdict.
- `judge_profiles_record`: Store what a judge campaign measured each judge to be, per criterion, for other campaigns to read.
- `judge_profiles_list`: List the stored judge profiles: each judge and criterion measured, on what, when, and its tier.
- `scope_pivot`: Aggregate one measure over the scope's observations by two coordinates, cell by cell.
- `runs_compare`: Compare one run's arm against another's: pass^k, mean composite, their deltas and the test.
- `runs_bisect`: Split the versioned inputs of two runs into those that differ, agree and cannot be decided.
- `scope_out_of_run_spend`: List what the engine spent outside any run — case generations, rubric proposals and analysis generations — with totals.
- `scope_history`: Series one measure over time for each contestant in the scope, flagging regressions.
- `scope_frontier`: Rank each subject's variants on quality, cost and latency, and name the cheapest that clears a bar.
- `scope_export`: Export the scope's observations as flat rows, CSV or JSON, for analysis elsewhere.
- `campaign_export`: Export a campaign's stored documents, its host's profile and a manifest: what a writer case is minted from.

### Curate (14)

- `cases_rename`: Rename a stored case: the name the report calls it by, and nothing else about it.
- `run_archive`: Archive a run (or restore it): out of every cohort, nothing destroyed.
- `run_label`: Name a run with labels (or clear them): runs_list lists runs by one, no key moves.
- `campaign_archive`: Archive a campaign (or restore it): out of listings, nothing destroyed.
- `result_regrade`: Grade one stored result again from its trace: report every verdict that moves, store on apply.
- `run_remeasure`: Read one stored run's host measures again under the host's current definitions: report, or store.
- `runs_attest`: State one coordinate of stored runs that they did not record: their measure definitions or served model.
- `result_rate`: Rate one judged dimension of one result, as an agent — kept beside people's ratings, never pooled.
- `result_rate_person`: Record a person's rating of one judged dimension, relayed by a caller the host attests for them.
- `analysis_archive`: Archive a stored analysis (or restore it): marked, its insights retracted, nothing destroyed.
- `writer_case_archive`: Retire a writer case (or restore it): no launch runs it again, nothing destroyed.
- `run_delete`: Destroy a run, its results and its campaign memberships.
- `analysis_delete`: Destroy a stored analysis; the insights it minted remain.
- `insight_delete`: Destroy one insight, so no later analysis reads it as prior context.

### Investigate: snapshot (1)

- `report_snapshot_freeze`: Freeze a campaign's evidence, or a stored analysis's, as the snapshot every analyst tool reads.

### Investigate: facts (7)

- `campaign_overview`: Read what a snapshot holds first: its campaign, arms, aspects, measures, criteria and caveats.
- `readings_list`: List each aspect's readings against the control, pair readings, floors and bars, by their keys.
- `contrasts_list`: List the corrected contrasts of each arm against the control, and the families correcting them.
- `measure_read`: Read one measure's or judged dimension's statistics, by arm or within each stratum, task or cell.
- `examples_list`: List the failed trials the snapshot froze: each trial's reply, reason and stored result.
- `caveats_list`: List every caveat, stated or implied: where it may land, whether it matters, and what it touches.
- `figure_keys_check`: Read each figure key a report cites, as the page would print it, or why it reads nothing.

### Investigate: analyses (7)

- `options_dominance`: Which options survive the gates, which dominates on cost and the asked aspects, and why.
- `decision_sensitivity`: How code's call and per-task picks move as a declared margin, floor or threshold, or a hypothetical line, moves over a range.
- `measure_breakdown`: A measure slice by slice: each candidate against the control in each stratum, task or cell.
- `arms_compare`: Two chosen arms on one measure: the difference overall and in each slice, with its interval.
- `settle_cases`: The cases that would settle an arm against its line, or two arms apart, and how often today's design reads the wrong way.
- `price_tradeoff`: Price the trade between the surviving options: cost per point gained on one axis, and where the best option flips as the price of a point moves.
- `levers_effects`: What each setting the arms vary does to a measure, between arms differing in it alone, and whether that changes with another setting.

### Investigate: blocks (3)

- `block_catalog`: List every kind of evidence block, the sources each draws, and the sources this snapshot holds.
- `block_render`: Draw one evidence block as a page would, and read its values table, level, unit and line.
- `report_check`: Check a draft report document: every verdict code finds, and the page as the editor reads it.

### Investigate: gaps (1)

- `capability_gap`: Note a capability you needed and no tool gave, or a result that looks wrong, and the workaround you go on with.

## A host's own tests: 17 kits, recorders and guards

- `kernel.host.check_world_conformance`: Run every obligation this registry's declarations imply, against the host's own handles.
- `run.FidelityContract`: One shared construction path and the callers required to reach it.
- `testing.check_completion_conformance`: Fail, naming the attribute, when a host's completion does not carry what eval reads off it.
- `testing.check_refusal_negotiation`: Fail, naming each mistake, unless the client answers a router's refusals in order and reports what was served.
- `testing.CompletionGuard`: The kit's watch on a host's completion clients: what was asked outside a job, and how often inside one.
- `testing.jobs_conformance_cases`: Every case the catalogue's actions are held to: each action's call, and each spend action's happy path.
- `testing.JobsConformanceCase`: One action held to one rule.
- `testing.JobsConformanceHost`: What a host hands the kit for one case: its operations host, who calls, and its sample launch.
- `testing.NegotiationConformanceFailure`: A host's client mishandled the router's refusals: each mistake it made, named.
- `testing.RecordedCompletions`: A completion client that records the calls an inner client answers, or replays a recording of them.
- `testing.RecordedPrompt`: One call a recording is expected to serve, as `RecordedCompletions.check_covers` is handed it.
- `testing.RecordingRefused`: A recording cannot serve what was asked of it: a miss, a missing or unreadable file, a newer format, a call already served to another eval, or a call refused while recording.
- `testing.RouterTransport`: A scripted router: the JSON body a client would POST in, the status and JSON body the router answers with out.
- `testing.STORE_CONFORMANCE_CASES`: Every case, in the order the port states its rules: scoping, the strip on read, projection, querying, optimistic concurrency (with the re-read that recovers a lost race), merge, delete — and last, that every `doc_type` the engine writes is one the store stores.
- `testing.StoreConformanceCase`: One rule of the port, as a check over a store.
- `actions.ConformanceSample`: One launch a host can run end to end, which the engine's spend actions' happy paths build on.
- `actions.ConformanceStage`: What a `HappyPath` is handed: the host the kit built, the caller, and the host's sample launch.

## Shapes: 258 values a reader is handed

### Protocols the engine implements and hands a host (10)

- `kernel.CellCassettes`: One cell's handle on its run's cassette lane, as the kind driving that cell sees it.
- `kernel.CellSink`: One cell's record of where it stands, as the kind driving that cell reports into it.
- `kernel.CellSpanWindow`: One cell's two tracing windows, as the kind driving that cell sees them.
- `kernel.DeliveryRecorder`: Captures one asynchronous tool's background work for one cell.
- `kernel.DeliveryReplay`: Serves one asynchronous tool's recorded work for one cell.
- `kernel.DeliveryTicket`: One piece of background work being captured, from its start to however it ends.
- `kernel.JobStore`: The run document's read-modify-write: what the job manager persists a run's status through.
- `kernel.ResultStore`: The cells a run recorded: each `EvalResult` and the `EvalTrace` beside it.
- `kernel.RunStore`: The eval runs of a scope: their documents, listings, stamps, archive flag and boot scan.
- `kernel.host.TemplateCaseFields`: What a host's check over a template case reads of it: the fields of a seed corpus's `threetears.evals.run.SeedTemplateCase`, which is what the check is handed.

### Runs, results and traces (26)

- `Result`: What an eval ran — one arm, or every arm of a comparison — and what its campaign's report says.
- `schema.AsyncDelivery`: One piece of background work the candidate started: acknowledged at once, delivered later.
- `schema.AsyncDeliveryStatus`: Where one piece of background work stood when the cell ended.
- `schema.CellIdentity`: Which cell execution a stretch of work belongs to.
- `schema.EvalResult`: One test case x one model x one k-iteration.
- `schema.EvalRun`: One execution of a template (or explicit test case set) against one candidate model.
- `schema.EvalTrace`: The candidate's output, what its judge read, and the OTel spans — stored beside a result, not inside it.
- `schema.KnownOutOfRunPurpose`: A value of `OutOfRunPurpose` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
- `schema.OutOfRunSpend`: One call the engine made outside any run, as it was admitted and as the provider reported it.
- `schema.ToolResultClearing`: The provider clearing old tool results from a long conversation itself: server-side context editing.
- `kernel.CandidateOutput`: What one candidate produced, as everything below the dispatch sees it.
- `kernel.CandidateTelemetry`: What one candidate's execution cost, and the windows nothing else timed.
- `kernel.OutOfRunBudget`: The cap one out-of-run unit of work is held to, and the ledger its calls are written to.
- `kernel.ReplayedDelivery`: One recorded piece of background work, served in place of running it.
- `kernel.VariantConfig`: The A/B'd stack one cell runs, as every kind alike receives it.
- `run.AbandonedRunSweepReport`: Outcome of one reclaim of runs left non-terminal by a process that has gone (`sweep_abandoned_runs`).
- `run.CandidateTurn`: One candidate-side response captured by the kind and fed back into the simulator.
- `run.CellContext`: What the runner knows about one cell when it asks a kind's factory for the cell's kind.
- `run.LaunchRequest`: One arm's launch as `start_run` resolved it, handed to the kind's own launcher.
- `run.LaunchSettings`: The host's launch settings, as one snapshot of values.
- `run.RunnerOptions`: The run's own knobs that don't fit on the run document — per-run values, never host wiring.
- `analysis.GateResult`: What a gate read and what it came to.
- `quick.WorldRunner`: A runner that hosts an agent on a world: `World(...)(agent)`.
- `ops.LaunchArguments`: What a launch names: the template, the subject, one arm per model, and the run's own limits.
- `ops.RunLine`: One run, as a listing shows it.
- `ops.RunListing`: One page of a scope's runs, newest first by created_at and id.

### Judges, graders and labels (20)

- `schema.CalibrationRating`: A rater's score for one judged dimension of one result — a person's, or an agent's.
- `schema.CatalogRubricDim`: A reusable rubric dimension in the shared catalog.
- `schema.JudgeConfig`: Versioned judge configuration for one rubric dimension.
- `schema.JudgedArtifact`: What a judge reads of a kind's output — the kind's declaration, which picks the judged axes.
- `schema.JudgeEvidence`: Everything a judge reads about one cell's candidate, rendered by the kind that ran it.
- `schema.RubricAxis`: The two rubric axes a dimension sits on: what a subject should DO (`capability`) and what it should refuse or withstand (`boundary`).
- `schema.RubricDim`: One judge-scored rubric dimension.
- `schema.RubricProposal`: The validated DRAFT the rubric proposer returns for operator review.
- `schema.RubricScore`: Outcome of one rubric judge dimension.
- `schema.SecondJudge`: A judge other than the one a run was scored by: a model, the prompt per dimension, and a temperature.
- `kernel.AuthoredJudgeCase`: A calibration case written by hand: an output, the criterion it is judged on, and the verdict it must get.
- `kernel.AuthoredVerdict`: An expected verdict as its author states it: a score on the criterion's scale (1-5 ordinal; 1 pass, 0 fail), or `"pass"` / `"fail"` on a pass/fail criterion.
- `kernel.EvalJudgeProfile`: What one judge was measured to be on one criterion, over frozen cases, by a judge campaign — stored for reuse.
- `run.JudgeRepeatReport`: What a repeat did: which results it recorded a repeat on, what it spent, and what it left out.
- `run.SeedJudgeCase`: One authored judge case a seed corpus writes: `judge_case_author`'s arguments, its templates named by name.
- `analysis.Verdict`: One verdict the report states, typed: what a program reads instead of the words printed from it.
- `quick.Grader`: A function or a field with what its reading declares: a margin, a guardrail, or a range.
- `quick.JudgeCalibration`: How a judge's scores on one dimension agree with a person's labels, and the evidence tier that bears.
- `ops.JudgeCaseAuthor`: What authoring one judge case by hand names: the judge template, the output, its criterion and its verdict.
- `ops.JudgeProfilesRecord`: What recording judge profiles names: the judge campaign's runs whose trials make up the measurement.

### Campaigns and comparisons (26)

- `schema.ControlEndState`: An end state a template's author states, to prove its goal checks can tell outcomes apart.
- `schema.GoalCheckControl`: One goal check's intent, and the end state that proves it discriminates.
- `schema.SweepableValue`: One level of one swept input: what it *is*, what to call it, and what kind of axis it is on.
- `schema.VariationAxis`: One axis along which test cases vary for a template.
- `kernel.Aspect`: One part of what a campaign asks, read on its own runs: its template, its margin, its design and its runs.
- `kernel.AspectDesign`: What one aspect of a campaign set out to vary and hold itself to — the declaration, one aspect's worth.
- `kernel.BarOverride`: A standard this campaign holds itself to, tighter than the registered one.
- `kernel.CampaignCall`: What a person decided over a campaign's readings: the arm, why, and every broken floor and missed bar they chose to override.
- `kernel.ControlDeclaration`: What held still, stated — because an absent control is a fact, not a null.
- `kernel.EvalCampaign`: What a person set out to learn, and the runs gathered to learn it: one question, its arms, its aspects.
- `kernel.Floor`: A line a candidate is warned for crossing on one aspect: a typed clause, read on that aspect's readings.
- `kernel.MeritAxis`: The merit axes a measure can serve.
- `kernel.OverriddenBar`: An absolute bar a call's arm missed on one aspect, named as the writer's `bars_missed` names one: the aspect and the bar's measure.
- `kernel.SweptAxis`: One axis this campaign set out to vary, and the levels it meant to compare.
- `kernel.host.Bar`: The incumbent standard for one behavior on one measure.
- `kernel.host.PassThreshold`: The 1–5 level a capability criterion must reach for an attempt to pass, for one behavior.
- `kernel.host.RolePins`: One pinned ROLE — who filled a seat in the rig, and the inputs that identify them.
- `kernel.host.Sweepable`: One score-determining input: what it is called, who owns it, how to read it.
- `run.ArmPrice`: What the host's pricer predicts one arm will cost, and how it knows.
- `run.ArmQuote`: One arm of a launch, as the engine asks the host's pricer to price it — stored-case arms and generating ones alike.
- `analysis.CampaignReadings`: Everything code reads off a campaign, by its kind: for a change, one row per aspect × candidate and one status per floor × candidate; for a choice, one reading per pair of arms per aspect; for a measurement, each arm's measures; for every kind, each arm against each absolute bar.
- `analysis.CampaignReport`: A campaign's page: the decision first, then what code read across its aspects, the findings and next steps, how the campaign was set up, then one fold per aspect.
- `ops.CampaignDefinition`: What creating a campaign names: what it is called, its subject and behavior, its question, arms, aspects and floors, and the campaigns whose aspects it adopts.
- `ops.CampaignExport`: One campaign's closure: its documents, the host's profile declaration and the manifest.
- `ops.CampaignLine`: One campaign, as a listing shows it.
- `ops.CampaignListing`: One page of a scope's campaigns, newest first by created_at and id.

### Reports and analyses (9)

- `kernel.EvalAnalysis`: A generated, stored analysis of one campaign — the productized "Analysis" lens.
- `kernel.EvalInsight`: A durable, subject-scoped insight extracted from an analysis.
- `kernel.host.WorldConformanceReport`: Every verdict from one run of the kit over one registry.
- `analysis.AnalysisContextBundle`: The closed context bundle a generation prompt runs over.
- `analysis.ChartBlock`: One of the charts a report draws, with every value it draws, so it prints as a table too.
- `analysis.EvalSummary`: One run, summarised.
- `analysis.Report`: A campaign's report: one page, the bundle it was computed from, and the typed verdicts it states.
- `analysis.RequestSettingsReading`: What one role asked for and what its calls were actually sent with, across the results read.
- `ops.EvalSummary`: the same as `analysis.EvalSummary`.

### Cost and budgets (12)

- `schema.AsyncExternalSpend`: Paid non-LLM calls one piece of background work made at one provider, as the work reports them.
- `schema.ExternalSpend`: One caller's report of what a provider call (or a batch of them) consumed.
- `schema.KnownUsageRole`: A value of `UsageRole` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
- `schema.RoleUsage`: Per-role token + cost observation for one `EvalResult`.
- `schema.TurnCeiling`: The most one tool turn can cost, as its client bounds it before it is sent.
- `kernel.CostLower`: A clause of a floor: the arm must cost less per reply than the control.
- `kernel.ExternalRateTable`: Operator-declared money per provider unit, resolved once for a whole run.
- `kernel.RateDrop`: A clause of a floor: no pass rate may drop by more than `max_points` against the control.
- `run.Ceilings`: The package's inner ceilings a host derives its outer budgets from, so a budget cannot drift below one.
- `run.MeteredCallTally`: What a run (or one cell's slice of it) metered, kept per provider.
- `providers.PriceRefresh`: What one read of the price catalogue found: what `Model.refresh_prices()` and `Clients.refresh_prices()` return, and `Model.last_price_refresh` keeps.
- `providers.PriceSource`: The words the shipped client writes as a completion's `price_source`: who priced the call.

### Cases, templates and seeds (19)

- `schema.CaseSetRef`: Which named, versioned case set a run was launched against: a label on its frozen case ids.
- `schema.EvalTemplate`: Abstract scenario blueprint — subject-agnostic, domain-level.
- `schema.EvalTestCase`: Concrete, immutable inputs generated from an `EvalTemplate`.
- `schema.VariationCounts`: How many test cases a launch asked generation for, and how many it froze.
- `schema.WorldSeed`: Initial state for the eval's stateful world.
- `kernel.PromptDrift`: How one stored prompt reads against the package's shipped default.
- `kernel.SeedPrompt`: Where one shipped default lives, so a host can register it by import.
- `kernel.SeedSection`: One section of a seeded prompt template.
- `kernel.SeedTemplate`: A seeded prompt template: an ordered set of sections and its identity.
- `run.DefinitionReset`: What a reset wrote, and how the corpus reads after it.
- `run.SeedCaseSet`: One named case set a seed corpus mints: a judge template's authored cases, listed by name, in order.
- `run.SeedCorpus`: A host's authored definitions for one scope, already constructed through their production models.
- `run.SeedOutcome`: What one seeding pass did, per doc type.
- `run.SeedTemplateCase`: One of a template's own cases a seed corpus writes with it: a stimulus the host authored, under a name.
- `run.TemplateField`: The fields of a template a kind may never act on: what it states about tools, the cases it generates, a conversation, the world (its seed, what it presumes, the checks graded against it) and the rubric a judge scores.
- `ops.CaseRenamed`: A renamed case: which, and its name before and after.
- `ops.CaseSetLine`: One version of a named case set.
- `ops.CaseSetMint`: What minting a case set's next version names: the set, its template and its cases in order.
- `ops.TemplateListing`: A scope's templates.

### Worlds, agents and conversations (16)

- `schema.ActorPolicy`: One simulated actor in a template's conversation.
- `schema.ConversationSpec`: The simulated side of a conversing candidate: who talks to it, in what order, for how long.
- `schema.ConversationStopCause`: Why a conversation trial's turn loop stopped — one structural signal, never a reading of prose.
- `schema.GoalCheckControls`: Proof, at authoring, that each of a template's goal checks can tell its outcomes apart.
- `schema.GoalStateOutcome`: One judge-free fact a candidate's execution established, and whether it held.
- `schema.KnownSimulatorPurpose`: A value of `SimulatorPurpose` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
- `schema.Precondition`: One thing a template presumes about the world before the subject's first turn.
- `schema.SubjectSnapshot`: The subject a set of observations was taken against, frozen at capture.
- `schema.TurnMessage`: One message of the conversation a turn sends.
- `schema.TurnReply`: The model's next message, why it stopped, what it used and what it cost.
- `schema.TurnToolSpec`: One tool the model may call: its name, what it does, and its arguments' JSON schema.
- `schema.WorldPlacement`: What a RUN did with a dimension, computed from that run's own record and nothing else.
- `kernel.host.WorldDimension`: One state dimension a run may set, may read back, or may only witness.
- `run.SimulatorTurn`: One user-side utterance produced by the simulator.
- `quick.WorldOutcome`: What a world runner's graders see: the agent's answer, the state it left, and the calls it made.
- `transports.fastmcp.ToolCall`: One call, as the `around` and `annotate` hooks see it once its caller and host are known.

### Model calls (21)

- `schema.CallLedger`: The calls one cell's candidate made that succeeded, across every tool, in recorded order, and those refused.
- `schema.ClientRequestSettings`: Request parameters for one apparatus role's LLM call: what the engine ASKED for, or what a request was SENT with.
- `schema.FailureOwner`: Whose failure a raised completion call was, as the host's describer reads it (`ProviderFailure.owner`): the calling `account` (refused for every model behind the key), the `provider` (it declined to serve the call at that moment: a rate limit, 429, or the provider overloaded or unavailable, 529 or 503), `unserved` (the provider does not serve the model the call named: retired, unknown, or not a valid model id, so every call naming it gets the same answer), or the `candidate` (anything else, charged to whoever made the call).
- `schema.KnownRefusalKind`: A value of `RefusalKind` this build knows, as a type: what a caller names (an action's argument, a filter) and what this build writes, refused outside the set as any `Literal` is.
- `schema.RecordedCall`: One call a candidate made that succeeded — or, in a control end state, one it is stated to have made.
- `schema.RefusalKind`: What refused a call or ended a job: the public engine error it was, by kind.
- `schema.StopReason`: Why a completion stopped, in the engine's words.
- `schema.ThinkingBinding`: The provider checking that a replayed thinking block's conversation is unchanged: the Messages API's `thinking.block_binding`, under the `thinking-binding-controls-2026-08-01` beta.
- `kernel.AdmittedCall`: A planned call its budget admitted, at the ceiling it was priced at.
- `kernel.PlannedCall`: One call an out-of-run unit of work means to make: the prompt pair and the directive it sends.
- `kernel.host.CompletionRole`: The apparatus roles the engine asks a host for a completion client in.
- `analysis.SentSettingsShare`: One value a role's calls were sent with, and how many of the role's units carried it.
- `analysis.SentSettingsState`: What a role's calls were sent with, read across a run's (or a campaign's) calls, for display only — never stored: `as_asked` (every call reported, one value, the one asked for), `sent_otherwise` (every call reported, one value, not the one asked for), `mixed` (every call reported, more than one value), `unverified` (some call's client did not report what it sent).
- `actions.Refusal`: Why a call was refused, as data: the kind of error and the status it carries.
- `negotiation.Negotiated`: A served request: the caller's reply, and what the request that produced it carried.
- `negotiation.ProviderFamily`: The request shape a client speaks, which decides how a setting goes on the wire: `anthropic` (Anthropic's Messages API: thinking budgets, adaptive thinking with effort words, no temperature beside thinking), `openai` (OpenAI's `reasoning_effort`, no budget) and `router` (a router's chat completions: `reasoning.effort` with all seven words or a `reasoning.max_tokens` budget, never both, and enforcement as `provider.require_parameters`, under which a router that finds no endpoint taking every parameter refuses the request).
- `negotiation.RefusedSetting`: What a provider refused:
- `negotiation.SettingFacts`: What is known of the request settings one model accepts: each fact `None` while nothing says.
- `negotiation.SettingRefusal`: A provider refusing a request for one setting it carried, in the provider's own words.
- `negotiation.WireSettings`: Request settings as a client is asked for them, or as one request carries them.
- `providers.Completion`: One reply: the completion port's ten members, and seven of the client's own.

### Measures (4)

- `schema.IntervalScale`: A real number with real spacing — `0.4`, `15`, `2000ms`.
- `schema.LatencyMetrics`: Per-result latency decomposition: harvested OTel spans, plus what they miss.
- `kernel.MeasureFamily`: One family of measures and who produces its numbers — the engine's six, or one a host declares.
- `kernel.MetricDescriptor`: What a single measure is, independent of any particular value of it.

### Actions and jobs (4)

- `ops.JobHandle`: A started job: the id to poll, and what it is working on.
- `ops.JobsStarted`: What starting long work returns: one handle per job, in the order the work was asked for, and what the call told its caller beside starting it.
- `ops.JobStatus`: Where one job stands, read from the record its work writes.
- `actions.ActionOutcome`: What one call came to: the text an agent reads, the typed result as data, and why it was refused.

### Storage (1)

- `storage.CoreReset`: What `RoutedDocumentStore.reset_core` did: the core and regenerable documents deleted, and what moved.

### Errors (29)

- `schema.CompletionProtocolDefect`: A host's completion does not meet the `CompletionResult` protocol, found while the engine read it.
- `schema.StoreCalledOnEventLoop`: A store call was made on a thread that is running an event loop.
- `schema.StoreConflict`: A conditional write lost its race: the stored document no longer carries `if_match`.
- `kernel.AdmissionRefusedError`: A launch refused because it would admit more unfinished runs than the process may hold (status 429).
- `kernel.CandidateKindDefect`: A kind handed back output that contradicts its own `CandidateKind.judged_artifact`.
- `kernel.CandidatePreparationFailed`: A candidate could not be built, and the cell is cleanly excluded rather than scored.
- `kernel.ConflictError`: State conflict (status 409).
- `kernel.EvalJobTimeout`: A job outlived the wall-clock budget its timeout context was enforcing.
- `kernel.EvalServiceError`: Structured error from the eval service layer.
- `kernel.NotFoundError`: Resource not found (status 404).
- `kernel.ProviderRefusedError`: A paid provider call made on the caller's behalf raised (status 502).
- `kernel.StorageError`: Storage operation failed (status 503).
- `kernel.UnportableSchema`: A schema keyword no strict structured-output mode is known to accept, and none that a reply's validation stands in for: a structural keyword (`allOf`, `not`, `if`), which dropping would change the shape sent; or an `enum` or `const` holding a value that is not a string, which not every strict mode enforces.
- `kernel.UnreadableRunError`: A read that cannot leave a run out met one this build's model refuses (status 500).
- `kernel.ValidationFailedError`: Validation failed (status 422).
- `kernel.WorldSessionError`: A kind asked its cell's world session for something the host's world, or the moment, cannot give.
- `kernel.host.ApparatusError`: A fault in the eval measuring rig, not a failure of the world under test.
- `kernel.host.BarRegistrationError`: A bar declaration contradicts what this registry promises.
- `kernel.host.EvalCellTimeout`: A cell outlived the wall-clock budget its timeout context was enforcing.
- `kernel.host.MeasureRegistrationError`: A measure declaration contradicts what this registry promises.
- `kernel.host.ReplayMiss`: A recording that cannot serve this run, so the run fails.
- `kernel.host.SeedRefused`: A seed asked for a write the registry or the subject cannot make.
- `kernel.host.UnsupportedSchemaError`: A schema uses a construct outside `HONOURED_KEYWORDS` — a gap to extend, never a value to pass.
- `kernel.host.WorldConformanceError`: An ENGINE gap: a check the kit cannot run, or a record its own vocabulary forbids.
- `run.GoalCheckUnevaluable`: A goal check raised while being evaluated — a fault of the rig, not a verdict on the candidate.
- `storage.MisplacedDocument`: One id is held by two stores, or by a store its type is not placed in; the router never picks one.
- `storage.RouteChanged`: The router was started under placements its stores were not written under; `relocate` first.
- `transports.fastmcp.CallTimedOut`: A call outlived the call timeout and was stopped before its action finished.
- `providers.ProviderError`: A request the provider refused or failed: who, the HTTP status, and the provider's own message.

### Other (1)

- `RootExtra`: What one optional public root needs installed before it imports: the extra, and the modules it brings.

### Constants (60)

- `__version__`: The installed distribution's version, as its metadata states it (`importlib.metadata`): the build a host records beside what it stores, since no stored document or analysis bundle carries it.
- `PUBLIC_ROOTS`: The public roots, as absolute module names.
- `ROOT_EXTRAS`: Each public root that fails to import without an extra installed, mapped to what it needs (`RootExtra`): the extra's name, and the modules that extra provides which the root imports.
- `schema.CASSETTE_SCHEMA_VERSION`: The version every cassette is written under, and the only one a read of one accepts.
- `schema.CORE_BASELINE_VERSION`: The oldest core major this build reads: always `CORE_SCHEMA_VERSION`, since the core ships no upgrader and a core major is a baseline reset that moves both.
- `schema.CORE_RESET_DOC_TYPES`: Every `doc_type` a core baseline reset deletes: the core and every regenerable type, and never a durable one.
- `schema.CORE_SCHEMA_MINOR`: The core minor this build writes within `CORE_SCHEMA_VERSION`.
- `schema.CORE_SCHEMA_VERSION`: The core version this build writes.
- `schema.COST_CAP_ORIGINS`: The values of `CostCapOrigin` this build knows, in the order its readers list them.
- `schema.DEFAULT_JOB_LEASE_S`: How long a job lease lasts unless renewed, in seconds, when the launching host sets none of its own (`LaunchHost.job_lease_s`): a job is read lost this long after its process last renewed it.
- `schema.DEFAULT_JUDGE_TEMPERATURE`: The temperature every judge call is requested at unless a `JudgeConfig` for its dimension says otherwise, and that config's own default (#633).
- `schema.DOC_TYPE_FACTS`: Every `doc_type` the engine stores, and its facts.
- `schema.DURABLE_BASELINE_VERSION`: The first durable version, and the oldest every build reads, forever: it never moves, so a durable document any release since `.25` wrote stays readable (`tests/test_durable_schema_pins.py` pins it).
- `schema.DURABLE_DOC_TYPES`: What people author, by `doc_type`: the `durable` entries of `DOC_TYPE_FACTS`, derived as `CORE_DOC_TYPES` is.
- `schema.DURABLE_SCHEMA_MINOR`: The durable minor this build writes within `DURABLE_SCHEMA_VERSION`, under the core's minor rules.
- `schema.DURABLE_SCHEMA_VERSION`: The durable version this build writes.
- `schema.EVAL_DOC_TYPES`: Every `doc_type` the engine writes — the set the operator wipe sweeps.
- `schema.JOB_LEASE_RENEWAL_FRACTION`: How often a held lease is renewed, as a share of its duration: a third, so two renewals in a row may fail before it lapses.
- `schema.JUDGE_WORDING_CHANGE_SENTENCE`: The sentence a release's `MIGRATIONS.toml` notes carry, verbatim, when it changes the wording of a built-in judge prompt (`builtin_judge_config_id`) or of the reply-format suffix a configured judge is sent (`configured_judge_config_id`).
- `schema.LEASE_POOL_WORKERS`: How many threads the job leases' own pool runs store round-trips on: enough that every lease a busy launch holds (a launch and each of its runs) renews at once; renewals past it wait only for each other.
- `schema.NON_TERMINAL_RUN_STATUSES`: Statuses a run can still leave under its own power — it is either queued or executing, and something in-process is expected to write its terminal status.
- `schema.PROSE_SCHEMA_KEY`: The JSON Schema keyword marking a string property as model prose, for vocabularies declared as schema rather than as Pydantic models (a world dimension's value schema).
- `schema.REFUSAL_KINDS`: The refusal kinds this build knows, one per public engine error class, in the order its readers list them.
- `schema.REGENERABLE_SCHEMA_VERSION`: The version every regenerable document is written under, the cassette apart (`CASSETTE_SCHEMA_VERSION`), and the only one a read of one accepts.
- `schema.RESERVED_DIM_IDS`: The reserved dim ids, which are deliberately NOT namespaced: they identify the two dual-score axes rather than a rubric dimension scored in some context, so there is no context to name. `require_namespaced_dim_name` exempts them — a judge service is built for these ids on every run.
- `schema.USAGE_ROLES`: The values of `UsageRole` this build knows, in the order its readers list them.
- `kernel.CONFUSION_CELL_MEASURE`: The core measure a classification's confusion-matrix cell is reported under.
- `kernel.FIND_TEMPLATE`: How a caller finds a template's id.
- `kernel.IDENTITY_VERSION`: One counter covers both predicates, so a bump on either side re-derives the other's keys — conservative in the safe direction, since re-deriving unchanged inputs reproduces the same digest.
- `kernel.JUDGE_KIND`: The `candidate_kind` a judge template declares.
- `kernel.JUDGE_KIND_CONTRACT`: The judge kind's contract: its overlays, and the rig it fills.
- `kernel.JUDGE_KIND_MEASURES`: The measures a judge trial lands on `host_measures`, for a host to declare on its measure registry.
- `kernel.KIND_TEMPLATE`: A seed whose constant is a sectioned template, registered with a template registry.
- `kernel.KIND_TEXT`: A seed whose constant is a single string, registered with a shared-prompt registry.
- `kernel.MATCH_MEASURE`: The core measure one classification's verdict is reported under — the one a classifier kind lands on `host_measures`, a bool.
- `kernel.METRIC_DESCRIPTORS`: The engine's own measures, each one's descriptor keyed by its name.
- `kernel.SEED_PROMPTS`: Every prompt the engine ships a default for, in the order a registry seeds them.
- `kernel.SEED_REMEDY`: What a template's author does when its world seed states world state a goal check's control cannot be built over.
- `kernel.host.SHARED_CORE`: The registry a host extends.
- `run.JUDGE_CALL_ATTEMPTS`: Calls one judge dimension can make: the first, plus its parse retries.
- `run.JUDGE_MAX_TOKENS`: The eval judge's output cap: the reply schema's answer budget plus the declared reasoning headroom, DERIVED rather than chosen, so it always leaves room to think and then answer.
- `run.JUDGE_REQUEST_SETTINGS`: The judge role's request settings as ONE value: what the host's client builder applies to every eval judge client, and what a run launched with a judge records as `judge_asked_settings`.
- `run.SIMULATOR_REQUEST_SETTINGS`: The simulator role's request settings as ONE value: what the host's client builder applies to the simulated user's client, and what a run launched with a simulated user records as `simulator_asked_settings`.
- `analysis.CELL_MODEL_VERSION`: The cell model this module implements, stamped onto every analysis generated under it (`GenerationProvenance.cell_model_version`).
- `analysis.DECLARED_NAME_CHARS`: The most characters a declared display name holds: an aspect's short name, a goal check's name, a stratum's name.
- `analysis.GOAL_CHECK`: What a page calls a goal check its template names nothing and whose expression names no subject a reader can read (`_check_subject`): numbered where an aspect reads several such (`check_names`).
- `analysis.READ_TIER_ROW_BUDGET`: The readings readers' budget (`READING_BUDGET`) under the name a host already reads; also how many projected score records one call may carry before its observer line is logged at WARNING (#652), a signal set to fire before a page is slow, not a latency limit (`elapsed_ms` rides on the same line for a host that would rather alert on time).
- `gen.EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT`: The seeded default `eval_boundary_gen` template: drafts a battery of simulated actors that pressure a subject's boundaries, and the refusal dimensions they are scored on.
- `gen.EVAL_PROPOSER_TEMPLATE_DEFAULT`: The seeded default `eval_proposer` template: drafts a capability rubric and the variation axes that stress it.
- `storage.ROUTING_RECORD_DOC_TYPE`: The `doc_type` of the placement record the router writes into every store, inside the routing scope its host names.
- `quick.EXIT_FAILED`: Exit code: the command failed on an error nothing anticipated — a host factory, a launcher or a handler raising, or the engine's own fault.
- `quick.EXIT_GATE_FAILED`: Exit code: `gate` read a verdict its `--fail-on` names.
- `quick.EXIT_OK`: Exit code: every run completed, or the command read what it was asked for.
- `quick.EXIT_REFUSED`: Exit code: the command was refused before it could do anything.
- `quick.EXIT_RUN_DID_NOT_COMPLETE`: Exit code: a launched run did not complete.
- `ops.ANALYSIS_PREPARATION_BUDGET_S`: The default wall-clock ceiling on a generation job's preparation, used when a host supplies none (`AnalysisGeneration.preparation_budget_s`, read on every call): building the `analyst` client, holding it to the host's allowed writers and asking whether its turns can be priced — in seconds.
- `ops.LIST_PAGE_MAX`: The most rows one listing page reads; a caller wanting more reads the next page.
- `ops.LIST_PAGE_SIZE`: How many rows a listing page reads when its caller names no `limit`.
- `negotiation.MAX_RESENDS`: The most times one call is resent for a refused setting: one per setting a client can drop or move (the temperature, the reasoning parameter, the reply format, enforcement).
- `providers.CATALOGUE_FETCH_TIMEOUT_S`: How long the catalogue fetch may take, in seconds, on a client the catalogue opens itself.

## Awaiting demotion (#1199): 22

Declared, and used by no production code and no shipped example (admission criteria 2 and 3): only tests, the
toy host, a page's prose or the action catalogue reach each. Where an action runs one, the action is the door
to use. Issue #1199 demotes them together with the namespace re-root, so a host migrates once; do
not start using one.

- `kernel.CappedCompletion`: One cell's view of a completion client: every call's cost counted in the cell's spend as it returns. (bound)
- `kernel.host.load_bar_registry`: Build a host's bar registry from a declaration kept as data — refusing any field it does not know. (bound)
- `run.admit_template`: Every refusal a template meets before it is first written, in `create_template`'s order. (invoked)
- `run.launch_as_group`: Launch one group of runs: admit it, prepare every arm, and start them together — or start none. (invoked)
- `run.load_seed_corpus`: Read a corpus off disk, constructing each document through its model in `scope_id`. (invoked)
- `run.mint_case_set`: Store the next version of the case set `name`: version 1 for a new name, else one past the latest. (invoked)
- `run.quote_launch`: What `start_run` with the same arguments would make of its arms' prices — read-only. (invoked)
- `run.regrade`: Grade one stored result again from what its cell stored, and report every verdict that moved. (invoked)
- `run.start_run`: Refuse what no kind can run, admit the launch, and dispatch each arm to its kind's launcher. (invoked)
- `analysis.assemble_context_bundle`: Assemble the closed context bundle for a campaign's runs. (invoked)
- `analysis.gate_verdicts`: Gate a report's typed verdicts: fail on the outcomes named, and never pass one the evidence left undecided. (invoked)
- `analysis.report_html`: Render a report as one standalone HTML page whose one script is the kit's, or, `static`, with none. (invoked)
- `quick.run_cli`: Parse `argv` and carry out the command, printing to stdout and refusals to stderr. (invoked)
- `ops.await_job`: Await started jobs to their end in the foreground, and return how each ended — its result included. (invoked)
- `ops.campaign_export`: Assemble one campaign's closure, read-only: the documents, the profile declaration and the manifest a writer case is minted from. (action)
- `ops.case_rename`: Rename a stored case: write its `name`, and nothing else but the seed name drift would have found it by. (invoked)
- `ops.case_set_mint`: Store the next version of a named case set: append-only, so a change is a new version, never an edit. (action)
- `ops.case_sets_list`: The scope's case sets — every version, or every version of one name. (action)
- `ops.judge_case_author`: Write one judge case by hand into a judge template: an output, its criterion, and the verdict it must get. (action)
- `actions.render_jobs_started`: What a start returned: each job, and how to poll it. (bound)
- `transports.fastmcp.CatalogueTool`: One mounted tool, as FastMCP serves it: every call handed to the catalogue. (bound)
- `negotiation.closest_settings`: What to send for `asked`: the closest settings `facts` say the model accepts, and why each one moved. (invoked)

## Needs a shipped example (#1199): 4

Each is the only door to something a host must be able to do, and no shipped example uses it yet: use it as
its line says. Issue #1199 adds the example.

- `kernel.failed_call_output`: A host's own kind routes a failed call through it, so an account refusal stops the run, a decline excludes the cell and an unserved model is never judged.
- `kernel.host.MemberActsOn`: A host marks a map field of its kind's overlay with it, so each entry is attributed to what it acts on.
- `kernel.host.Ordinal`: A host marks a `Literal` or `Enum` knob of its kind's overlay with it, so the knob's values are ranked.
- `ops.start_action_job`: A host's own action runs work that is not one paid call as a job through it (`start_paid_call` covers one paid call).

## Reading states: 18 reasons a reading is not counted, and what closes each

| State | Reason | Remedy |
|---|---|---|
| `set_aside` | `identical_arms_separated` | `repair_rig` |
| `set_aside` | `instrument_below_bar` | `replace_instrument` |
| `set_aside` | `instrument_leans` | `replace_instrument` |
| `set_aside` | `instrument_unchecked` | `check_against_people` |
| `set_aside` | `people_disagree` | `sharpen_criterion` |
| `withheld` | `identity_unrecorded` | `remeasure_or_attest` |
| `withheld` | `measured_under_concurrency` | `rerun_serially` |
| `withheld` | `split` | `rerun_on_one_cell` |
| `withheld` | `trials_lost` | `repair_rig` |
| `withheld` | `unfinished` | `resume_or_rerun` |
| `not_available` | `identity_unrecorded` | `remeasure_or_attest` |
| `not_available` | `no_variation` | `add_cases` |
| `not_available` | `not_drawn_at_random` | `draw_at_random` |
| `not_available` | `not_recorded` | `record_and_rerun` |
| `not_available` | `replayed` | `run_live` |
| `not_available` | `skipped_past_limit` | `finish_read` |
| `not_available` | `too_few` | `add_cases` |
| `not_available` | `trials_lost` | `repair_rig` |

## Near-duplicate shapes: 2, each saying why both exist

- `schema.AsyncExternalSpend` and `schema.ExternalSpend`
- `schema.ClientRequestSettings` and `negotiation.WireSettings`

## The counts

A name is public when it meets the admission criteria, not to meet a number: these counts are reported each
wave as a trend. The goals are the adopter's, a first eval's names and rung 3's.

| Kind | Names | Who meets it |
|---|---|---|
| `typed` | 12 | a newcomer writes it in an eval file |
| `bound` | 93 | a host implements it or hands it to the engine |
| `action` | 17 | the function behind an action of the catalogue, which an agent or operator invokes by name |
| `invoked` | 92 | a function a host's own code calls, to make the engine act or do one step its way |
| `received` | 258 | a value someone is handed and reads |
| `tests` | 17 | a kit, recorder or guard a host runs in its own tests |
| all | 489 | |

Of these, 22 await demotion and 4 need a shipped example
(issue #1199), each listed in its register's section above.

| Typed names | Names | Goal |
|---|---|---|
| A first eval (rungs 0 and 1) | 2 | 2 |
| Through rung 3 | 5 | 5 |
| The whole ladder | 11 | |
| Taught on a page rather than a rung | 1 | |

The hold-back ledger, [`tests/fixtures/declared_holdbacks.txt`](../tests/fixtures/declared_holdbacks.txt),
lists every declared name the tree alone does not hold; `tests/test_declared_holdbacks.py` holds it.

| Reason | What holds the name | Names |
|---|---|---|
| `port-host-implements` | A protocol a host implements and hands the engine | 18 |
| `protocol-engine-provides` | A protocol the engine implements and hands a host, or its storage satisfies | 10 |
| `unreached` | Nothing | 0 |

Generated by `uv run python packages/evals/scripts/generate_surface.py`: change a docstring or `scripts/surface_kinds.py`, never this page.
