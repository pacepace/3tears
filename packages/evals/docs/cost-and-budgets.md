# Cost and budgets

**For** anyone launching runs that call paid models or tools. **Answers:** what a launch will cost, what stops
it spending, how long it takes, and how spend is counted. If you are writing a host, read
[Adopting the engine](adopting-a-host.md) first; the terms used here (arm, run, launcher, cell) are
defined in [Concepts](concepts.md).

## The rules, in plain words

**The engine never pretends a cost it could not count was zero.** A call nobody could price is recorded as
**unpriced**, not as $0. Every surface spells an amount through `dollars_text`, which reads `None` as `unpriced`
(`UNPRICED`), so the word is the same wherever a spend is printed.

**Caps bind only on a host that enforces them** (`LaunchSettings.enforcement_enabled`, which every host sets).
There, the engine does not spend money it has not priced first:

- Every **run** has a dollar cap. Before a launch starts, the engine predicts what each arm will cost and
  refuses an arm it predicts will blow its cap — or one it cannot predict at all. A capped run stops at its
  first unpriced result.
- Calls made **outside any run** (writing new cases, proposing a rubric, writing an analysis, re-asking a
  judge) are priced before they are sent, against a separate out-of-run cap, and written to a ledger you
  can read back.

With enforcement off, runs record themselves as `uncapped` (below) and nothing is refused on price. **The quick
path (`run_eval`, `compare`) runs with enforcement off:** its candidate is your own function, whose spend the
engine sees only afterwards and only if you report it (an `Answer`), so no cap could price it first or stop
it. Bound a quick run's spend inside your candidate or your client.

The sections below state each rule exactly.

## Caps

**The per-run cap.** A launch's runs are held to the host's per-run ceiling, which a launch's
`max_cost_usd` may only lower — one above it is refused on every surface. A run the cap stops
mid-flight ends `budget_stopped`, with the results it already delivered kept: an honest outcome, not
an infrastructure failure.

**What the per-run cap counts while a cell runs.** Each cell's sink keeps one running total per source —
the candidate, its background work, the simulator, the judge — each only ever added to, plus the ceilings of
the priced calls in flight. Every question of the cap, in a cell or between cells, counts the sum over every
running cell beside what the run has already recorded. So no part of a cell reporting its spend can hide
another part's, and a sibling cell sees all of it.

- **A priced call is reserved before it is made.** Its ceiling (the client's `price_ceiling` times every attempt
  the engine makes for it) is held against the cap while it is in flight, and settled to what it actually cost
  when it returns, or released if it raised. A call whose ceiling would take the run past its cap is refused
  before it is made. **A run whose paid calls are all reserved never spends past its cap**, however many cells
  run at once. The engine reserves its own calls this way: every simulator call (a conversation's
  `drive_conversation`), every judge call in a cell's judge phase, the judge kind's call and the reporter kind's
  generator calls. A host's kind reserves its candidate's calls through `CellSink.reserve`, or wraps the client
  in `CappedCompletion`.
- **A call whose ceiling is above the whole cap can never be made.** Its reservation is refused at any spend, so the
  run stops `budget_stopped` on reaching it and its case is excluded, after the cases before it were paid for. A
  launch therefore refuses such an arm before anything is spent (step 3 below), when its kind prices its calls
  before launch. The judge kind's call reserves the client's `price_ceiling` for the case's prompt times 2 (the
  judge's parse retry, `JUDGE_CALL_ATTEMPTS`), and the judge asks up to `JUDGE_MAX_TOKENS` (10,240) output
  tokens, so a judge-role `price_ceiling` well above what 10,240 output tokens plus the prompt cost at the model's
  rates is pricing a larger output cap than the judge sends.
- **A call nobody can price** (a client with no `price_ceiling`, or one answering `None`) holds nothing; it is
  made only while the cap is not yet reached. Such calls can pass the cap by about one call per running cell.
- **Spend nothing reported** is counted only once the cell's result lands, so a kind that neither reserves its
  calls nor reports them (`CellSink.add_spend`) can pass the cap by up to one whole cell per running cell.

A call the cap refuses stops the cell: a conversation ends `budget_stopped` — a kind's candidate turn raises
`CostCapReached` to end it the same way — and a one-shot kind returns what it has. The cell is excluded, no cell
starts after it, and the run ends `budget_stopped` once the cells running are saved. A judge call the cap refuses
leaves its dim unscored with an error naming the cap, which a re-judge can repair; the candidate's work is kept.
When a refused reservation is what stopped the run, the stop says so: the call's ceiling, and what the run would
have spent with it.

**An uncapped run** (the host's cost enforcement off) records `max_cost_usd` as null and
`max_cost_usd_origin` as `uncapped`. When an analysis or a bisection compares spend ceilings, it reads that
run's ceiling as the level `uncapped`, so two uncapped runs agree and an uncapped run differs from a capped
one. Only a run that recorded no ceiling and no origin is undecided.

**The quick path's cap.** `run_eval` and `compare` run uncapped unless given `max_cost_usd=`, and the summary
says which. The cap is the per-run cap above, counting each result's reported spend (the candidate's `Answer`
and the judge's); `compare` splits it equally across arms. The candidate cannot be priced before it is called, so
the cap is asked before each call, and what its `Answer` reports is counted as soon as it returns; the judge's
calls are reserved. A run can overshoot by the candidate calls in flight when the cap is reached — one per case
running. A candidate that returns no `Answer` is invisible to it, and the summary says so.

**The wall-clock budget.** Each run's job also runs under a time budget sized to its matrix (cases × k ×
the cell timeout, plus a margin, clamped between a floor and an 8-hour cap). When it binds, the run ends
`budget_stopped` too, not `failed`: like the cost cap, it is a bound someone set doing its job, and the
results already delivered are kept and counted in the run's completeness record. `budget_stop_reason` tells
the two apart (a clock stop's reason opens with `wall-clock budget`), and `error_details` stays empty,
because that list counts faults. A run stopped by its clock under 0.66.0 or earlier stays stored `failed`
(`Job timed out after Ns`).

**The cell deadline.** Each cell runs under a deadline: the kind's own (its launcher's
`KindWiring.cell_timeout_s`, else 600 s). A cell that outlives it is excluded as `cell_timeout`. A launch can
name `cell_timeout_s` for a long-horizon run, up to the host's `LaunchSettings.max_cell_timeout_s`. A host that
declares no ceiling lets a launch only lower the kind's deadline. The job's wall-clock budget is sized from
the deadline in force. The run records it as `cell_timeout_s`, with `cell_timeout_s_origin` (`launch`, `kind`
or `default`). A run stored before 0.66.0 records neither, and its deadline reads as unknown.

**Wall time.** A run executes its cells (case × repeat) in a shuffled order, several at once: as many as its
kind declares it is safe at (`CandidateKind.max_concurrent_cells`, handed to the launch as
`KindWiring.max_concurrent_cells`; a kind that declares nothing runs one at a time), never more than the host's
ceiling `LaunchSettings.max_concurrent_cells` (4 by default). A cell's judge calls run up to `judge_concurrency`
at once. The arms of one launch run side by side in one concurrency slot. Estimate a
run's wall time as cases × k × the time one cell takes, divided by the width: 30 cases at k=3 and 6 s a cell
is a little over 2 minutes at a width of 4. A launch that declares latency under test (`measure_latency=True`)
runs its cells one at a time and its arms one after another, with no other run beside them: the same run
then takes about 9 minutes, times the number of arms. The cost cap and the metered-call ceiling hold at any
width: reserved calls never pass the cap, and what can (above) is bounded per running cell. A host with its own
pool or queue supplies a `CellExecutor` on `LaunchHost.cell_executor`.

**The out-of-run cap** is `LaunchSettings.max_out_of_run_cost_usd`. It bounds every call the engine makes
outside a run (below).

**A host with no metered tools** sets `LaunchSettings.max_metered_calls=None`: its runs record a
ceiling of `0` with origin `none_declared`, and a metered call that happens anyway is refused and counted.

## Every arm is priced before any launcher runs

**Every arm is priced before any launcher runs, by one rule.** In order:

1. **Plan.** The engine asks the kind what each arm will run (`LaunchableKind.plan_arm` → `ArmPlan`: for
   an arm over stored cases, how many of the template's stored cases it plays, for a generating arm at
   most `n_variations`; the model; the judges it will be scored by, resolved with `plan_judge`; and its
   simulator).
2. **Price.** Under an enforced cap, it prices each arm with the host's `LaunchHost.launch_pricer`
   (`ArmQuote.case_source` says which). `threetears.evals.ops.history_launch_pricer` bounds it by the upper
   end of the band of runs launched the same way — template, model, cassette mode, the model each scored
   dim was judged by, the simulator that ran, resolved apparatus settings. The band is a 95% prediction band
   read on the log scale, since costs are positive and a few long conversations cost several times the rest.
   With three past results it is wide on the high side (three costs of $0.10, $0.20 and $0.30 put a
   15-observation arm's upper end near $59), so a capped launch on thin history can be refused. It treats
   each observation as independent, which repeats of one case are not, and says so (`band_basis`).
3. **Refuse.** First, it refuses an arm whose largest call is reserved above its run's cap on its own
   (`ArmPlan.call_ceiling`): the run could never make that call, whatever the arm is predicted to cost in
   all. The kind prices it right after planning (`LaunchableKind.price_largest_call`), on the clients the
   cells will call through, exactly as a cell reserves it. Two kinds price their largest call exactly, each
   over every case, keeping the largest:
   - **The judge kind** prices each case's judge call. What lowers it: a cheaper judge model, shorter frozen
     evidence on the largest case, or a judge-role client priced at the output cap the judge asks for.
   - **The analysis reporter** prices each case's first generator request, built from the frozen bundle, the
     preset's text and the admitted model exactly as its cell builds it. What lowers it: a cheaper generator
     model, a smaller bundle on the largest case, or a generator-role client that retries less or prices a
     smaller output cap. The engine ships no ready reporter entry, so a host wires it into its own:
     `price_largest_call=lambda request: price_largest_reporter_call(eval_host, request, prompts=..., resolve_model=...)`
     (from `threetears.evals.ops`), with the same hooks as its `plan_arm` and the `eval_host` its launcher
     builds the generator client on. That client comes from the host's factory in the `analysis` role. Two
     calls stay unpriced: the memo's judge call, which judges a memo that does not exist yet, and the
     generation's repair round-trip, which carries the first call's output. The replay
     (`AS_RECORDED_MODEL`) calls no generator and prices nothing.

   - **A subject kind** (a persona, a classifier, any kind whose launcher builds its judge with
     `build_judge_service`) cannot price its largest call: its judge and simulator prompts carry a transcript
     that does not exist yet. It prices a **floor** under it instead, with `price_subject_call_floor` (from
     `threetears.evals.run`), and the ceiling says so (`CallCeiling.lower_bound`, reported on the estimate as
     `call_ceiling_lower_bound`). Each call is priced with every part its cell fills left empty — the
     variation, the goal-state outcomes, the kind's subject, case material and transcript — and everything
     else as sent: the dim's instructions and reply contract, the template's intent, every heading, the reply
     format and every parse-retry attempt. That is the smallest request the cell could send, and every real one
     is it with text added. The judge's floor is the largest dim's, since every judged case makes every dim's
     call. The simulated user's is the smallest actor's utterance call, counted only when the conversation is
     bound to make one: not when every round up to the turn limit is the world's, or round one is the only
     actors' round and an opener can end it. `llm_decided` scheduling calls are not priced. The arm's floor is
     the larger of the two.

     The floor is sound on a client whose `price_ceiling` does not fall as a prompt grows, which holds for
     any ceiling priced from a bound on the prompt's length and a fixed output cap. It catches the ceilings the
     output cap and the retry multiplier dominate, as in the incident above, whatever the transcript. A ceiling
     a long transcript dominates it cannot see: a floor under the cap admits the arm, and that call is held to
     the cap as the run reserves it. A candidate's own calls are the kind's and are not priced. Wire it as
     `price_largest_call=lambda request: price_subject_call_floor(eval_host, request, judged_artifact=...)`,
     with the `eval_host` your launcher builds its judge and simulator clients on and the declaration it hands
     `build_judge_service`. The simulator client comes from the host's factory in the `simulator` role, for
     the planned simulator model.

   The refusal names the call, its ceiling and the cap one such call needs (rounded up to the cent, plus the
   arm's predicted spend for room around it, or the host's operator when that is above the host's ceiling),
   and what lowers the ceiling. A floor's refusal says "even the smallest such call this arm could make" is
   reserved above the cap, and its remedy names the floor as a cap to exceed, not one that would be enough,
   since the real call adds its transcript on top. A kind that does not price its calls before launch
   (`price_largest_call=None`) is not checked here: such a call is refused only when the run reaches it. Then
   it refuses an arm predicted above its run's cap, or one
   nothing can predict — no pricer, no plan, or no history to bound — whose cap the run would merely inherit.
   An unpriceable arm under a cap the launch named runs under it. The estimate reports each arm's largest
   call as `call_ceiling_usd`, and `call_ceiling_lower_bound` when that figure is a floor.

Every arm is planned before any is priced, so `plan_arm` is where a kind makes its request-level refusals
(no model and no default — `require_candidate_model`, the tail's own refusal — a judge or simulator it
needs): the operator hears those before any "cannot be priced". The launch tail holds each launcher to its
plan (no more cases, no other model, judges or simulator), and refuses an arm that was never priced under
an enforced cap — a host composing its own launch through `launch_as_group` prices its arms with
`price_arms`.

**Batteries.** A battery prices each template's arms once, in its pre-flight, prepares every template
before starting any, and launches each template as it priced it.

**Estimating without launching.** `launch_estimate` (`quote_launch` in `run`) runs the same steps
read-only and reports each arm's price and the launch's verdict, word for word. Hand its result to a cost
pivot as `predicted_cost`, and each prediction sits only in the cell of its model and template. Once the
launch ran, pass its run ids as `launched_run_ids` too, and each predicted cell says how many of its
observations came from other runs — the history the prediction was drawn from among them.

**What the launch could detect.** Beside the price, the estimate's `detectable_effect` block says, per
reading a comparison would test, "with N cases and k repeats this campaign can detect Δ ≥ x (from runs …)".
Δ is the smallest true difference the paired test finds four times in five, at the α/m the first step of
Holm's correction asks of the family. Its variance comes from earlier runs of the same template: the repeat
noise, and how far two arms disagree about one case (measured where two earlier runs launched differently
shared cases; otherwise one arm's case-to-case spread, counted twice). With no earlier run, or no repeated
cases, a reading says it cannot estimate and why; no default variance is ever used. The block states its
assumptions, among them near-normal per-case differences and that only the one comparison differs.

A host therefore prices no arm itself: a wrapper that priced assembled runs would be a second rule, and a
second pricing of the same arm.

## Spend outside any run

These engine calls have no run around them: a launch's case generation (an `llm` variation axis's writer),
the rubric proposer (`propose_draft`), an analysis generation, a judge repeat, a second judge and a judge
temperature comparison. Under enforcement, each is bounded in dollars before it spends; every one is ledgered
either way.

**Priced in the call, spent in the job.** No transport call waits on a model, so every one of these that an
action starts is split where its spending starts. The action's call makes the refusals, prices every call against
the cap and admits them — a launch also prices every arm and takes its room under the admission ceiling — and
refuses there, with nothing spent. The work that spends runs as a job (see
[Driving it from an agent](agents-and-mcp.md#operations-and-jobs)): a launch's case generation runs in the
launch's job, before its runs start; a second judge's and a temperature comparison's calls in theirs. A job refused
part-way — a kind's launcher refusing an arm, a provider refusing the account — ends `failed` on its record, and
what it had spent is in the ledger as it went.

**Planned, admitted in the call, made in the job.** Every one of these calls is a public plan before it is
anything else, so the call can price it. The package never plans, admits and makes a call in one step: a
planner returns the `PlannedCall` (`gen.plan_draft` for a proposal, `kernel.plan_variation_calls` for a
generation), the caller quotes it and admits it under its `OutOfRunBudget` in its call, and the executor
(`gen.propose_draft`, `gen.generate_variations`) takes the admitted call and only makes it, on the client it was
priced on. The engine's own jobs (an analysis generation, a second judge, a judge repeat, a temperature
comparison) admit in their call the same way, and each has an estimate that quotes the same plan.

**Case generation.** Generation runs before any run exists, so its calls are outside every run's cost cap
and metered-call ceiling. In the launch's call, the engine builds the writer from the host's client factory,
plans every `llm` axis's call, and admits them together on that writer (`price_ceiling`) against
`LaunchSettings.max_out_of_run_cost_usd`, so a generation the cap cannot pay for is refused before the launch
returns. Each arm's request carries the admission (`LaunchRequest.generation`); the launcher makes those calls
and no others, and each is ledgered as an `OutOfRunSpend` under the launch's group. The launcher's side is in
[Generating cases at launch](adopting-a-host.md#generating-cases-at-launch).

**A rubric proposal.** A host plans it with `plan_draft(axis=..., system_prompt=..., ...)`, which refuses a
stored proposer prompt written for another draft shape before anything is priced (see
[the engine's prompts](adopting-a-host.md#the-engines-prompts-seeded-into-your-registry)), admits the call with
`budget.admit(client, "proposer", [call])` in its action's call (refusing there, with nothing called), and makes
it in the action's job with `propose_draft(client, admitted, budget=..., axis=..., subject_id=...)`, which
releases the client. A client admitted and never made is the host's to release.

**Batteries that generate.** A battery prices every template's arms and every template's writer calls (on
the host's `variation` client, against the budget each launch will be held to) before any template
launches, so it pays for no template's cases until all have been priced; its caps are per launch, as a
launch's are (`start_universal_battery(max_cost_usd=...)` names the per-run cap).

**Analysis generation.** An analysis generation is held to the host's out-of-run cap
(`LaunchSettings.max_out_of_run_cost_usd`): its first call is priced before the job starts
(`analysis_estimate` prices it without spending), its one repair round-trip before that is sent, and each
call is ledgered under purpose `analysis`, so `scope_out_of_run_spend` reads it.

**Judge repeats.** A judge repeat (`judge_repeat`) is held to the same cap, every call — parse retries
included — priced and admitted before the first is sent, and ledgered under purpose `judge` with the run's
id. (What a judge repeat is for is in
[Reading reports](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers).)

**Second judges.** A second judge (`judge_second`, `judge_drift_check`) is held to the same cap, priced and
admitted the same way, and ledgered under purpose `second_judge` with the run's id. It is measurement cost on its
own line: never in a result's `cost_usd`, never in the run's judge spend. `run_get` shows it as
`second_judge_cost_usd`. The pass's own report — what it called and what that cost — is on its record
(`EvalSecondJudgePass`), which `job_poll` reads.

**Judge temperature comparisons.** `judge_temperature` is held to the same cap the same way and ledgered under
purpose `judge`; its record (`EvalJudgeTemperatureCheck`) keeps every answer and its spend.

**A host's own `spend` actions.** A host's own `spend` action is held to one rule the engine checks: its call
makes no completion request, so it prices and refuses in the call and spends in a job (`start_action_job`). How
far it meters what that job spends is the host's handler's; the jobs conformance kit
(`jobs_conformance_cases`) shows the call spends nothing and the job reaches the work.

**Reading it back.** What was spent out of run is read back by `scope_out_of_run_spend` — the
`scope_out_of_run_spend` action, and the CLI's `spend` — narrowed by purpose, launch group or template.

**A ledger write that fails.** The row of a call the store would not take is logged at ERROR — and raised, for a
call that completed — and kept on the budget (`OutOfRunBudget.unrecorded`), so a judge repeat's, second judge's
or temperature comparison's own report still counts that call and its cost rather than a total missing it. What
reads the store alone, `scope_out_of_run_spend`, cannot see it: the log line is where it is.

## How a result's cost is counted

Each completion your client returns names its own `price_source`; the engine stores what it is told and
never assumes a provider. Your kind reports its own calls' spend only as usage rows
(`CandidateTelemetry.usage`), each call's dollars as your client priced them; the engine derives a
result's `cost_usd` from those rows and its background work's (`async_deliveries`, see
[background work](adopting-a-host.md#background-work-payloads-and-spend)).

**A `run_eval` or `compare` candidate reports its spend by returning an `Answer`.** The engine sees only what a
plain function returns. Return `Answer(value, model=..., input_tokens=..., output_tokens=..., cost_usd=...)`
and the call becomes the cell's `candidate` usage row: `value` is graded as a plain return would be, the
result's `cost_usd` is derived from the row, the summary prints `candidate spend: $... over N call(s)`, and
`compare` tests the arms' spend against the control. A field left `None` is unreported, not zero.
Pass `served_model=` too, the model the provider's response named: without it the run records only the model
asked for, and its summary says which model answered is unconfirmed
([Reading reports](reading-reports.md#which-model-answered)).
`examples/compare_two_models.py` prices each call this way.

**Every surface that states what an arm costs reads `production_replicating_cost`**, the spend of the roles
production runs: the contrasts, each decision-surface cell, its cost column and a frontier's cost axis (which
refuses `cost_usd`). `cost_usd` and `program_cost` also sum what a judge spent, which is measuring cost: never
tested between arms, and labelled measuring spend wherever drawn. An analysis stored before cells carried the
candidate's spend shows its frozen `cost_usd` column under that label.

**What an arm costs and what the program spent count different results.** A comparison cost — the
candidate's own spend, `production_replicating_cost`, and every mean of it (`mean_prod_cost_usd` in a run
summary, a frontier point's cost, an analysis cell's) — reads only the turns the candidate took. It leaves
out a result the harness faulted, since a cell an apparatus fault cut short spent less than a whole one and
would let the rig make an arm look cheaper. It also leaves out a call the candidate's model refused straight
away, which took no turn. Program spend (`cost_usd`, `total_cost_usd`, `mean_cost_usd`, a `cost_usd` pivot,
the `cost_usd` history series, an analysis cell's or bar's `cost_usd`, the budget view) keeps both, because those
dollars were spent. `n_prod_cost_usd` counts what the comparison figure rests on.

**A production-replicating cost names what its run moved off production.** It is what production
would spend only for a run that set nothing: a model the launch named, an overlay set off its default, or
an apparatus setting that strips the candidate's learned state each make it the cost of something
production does not run, and the error has no reliable sign. So every surface that shows it carries the
run's production footing, read off the host's sweepable declarations
(`SweepableRegistry.production_footing`): a bundle's `RunSummary.production_footing`, the
`prod_cost_footing` on each `run_summary` row and in an analysis run index's key metrics. It names the
inputs the run `moved` (with their levels), the levers whose departure nothing records (`unchecked`), and
the inputs `held`; only an empty `moved` and an empty `unchecked` say the cost was measured at production's
configuration. A lever with no `departs_production` reader is read by the engine's rule: unset is the
subject's own setting, a value is unchecked, so a lever you declare reaches the disclosure with nothing
else edited. Declare `departs_production` on a lever to decide its value against production, and on an
apparatus input that moves the candidate off its production footing. A run launched before the candidate
model's origin was recorded reads its model as unchecked; a summary assembled before the disclosure
carries `None`, "nobody checked".

A cost pooled over runs carries every run's footing (`PooledProductionFooting`, run id to footing), and says
where the runs ran at different footings rather than merging them: a frontier point's cost axis and its
verdict (`FrontierPoint.production_footing`, when the frontier is given the host's profile; the bundle's
frontier always is), and each analysis arm (`AnalysisContextBundle.arm_production_footings`), whose cells,
contrasts and bars on `production_replicating_cost` pool its runs. A code-only report says per arm what its
runs set, left unchecked, or that they moved nothing.

**A cost pivot says what its cells pool.** `scope_pivot(metric="cost_usd")` averages measuring spend, so each
cell names the role sets its dollars were summed over (`cost_compositions`, with `cost_compositions_differ` when
they are not one set: a cheaper cell may only have priced fewer things). A cell pooling a replayed run with a
live one is `withheld`, because the mean of the two is neither one's spend; put `cassette_mode` on an axis to
read each alone. A cell whose observations carried a seeded or
replayed delivery counts them (`n_substituted`) and says its dollars leave that spend out; one built only from
such observations says it is no live run's spend.

**Unpriced is a state, never zero.** A call your client could not price (a local model, say), or
background work's paid calls that report no `money` and that a run with declared rates has no rate
for, leaves the result's `cost_usd` as `None`. A reported `money` wins over the run's rate.

- Every cost aggregate leaves such a result out of its dollars and counts it beside them (`n_cost_usd`,
  `n_unpriced`).
- A capped run stops on its first unpriced result, because a cap cannot enforce a ceiling on spend it
  cannot count.
- An uncapped run carries on and counts them.
