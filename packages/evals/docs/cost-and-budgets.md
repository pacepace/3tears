# Cost and budgets

Read this if you launch runs that call paid models or tools, and want to know what a launch will cost, what
stops it spending, and how spend is counted. If you are writing a host, read
[Adopting the engine](adopting-a-host.md) first; the terms used here (arm, run, launcher, cell) are
defined in [Concepts](concepts.md).

## The rules, in plain words

The engine holds one line everywhere: **it does not spend money it has not priced first, and it never
pretends a cost it could not count was zero.**

- Every **run** has a dollar cap. Before a launch starts, the engine predicts what each arm will cost and
  refuses an arm it predicts will blow its cap — or one it cannot predict at all.
- Calls made **outside any run** (writing new cases, proposing a rubric, writing an analysis, re-asking a
  judge) are priced before they are sent, against a separate out-of-run cap, and written to a ledger you
  can read back.
- A call nobody could price is recorded as **unpriced**, not as $0, and a capped run stops at the first
  one.

The sections below state each rule exactly.

## Caps

**The per-run cap.** A launch's runs are held to the host's per-run ceiling, which a launch's
`max_cost_usd` may only lower — one above it is refused on every surface. A run the cap stops
mid-flight ends `budget_stopped`, with the results it already delivered kept: an honest outcome, not
an infrastructure failure. A conversing kind's loop checks the cap before every paid simulator call (see
[conversations](adopting-a-host.md#what-the-judge-reads-and-conversations)).

**An uncapped run** (the host's cost enforcement off) records `max_cost_usd` as null and
`max_cost_usd_origin` as `uncapped`. When an analysis or a bisection compares spend ceilings, it reads that
run's ceiling as the level `uncapped`, so two uncapped runs agree and an uncapped run differs from a capped
one. Only a run that recorded no ceiling and no origin is undecided.

**The wall-clock budget.** Each run's job also runs under a time budget sized to its matrix (cases × k ×
the cell timeout, plus a margin, clamped between a floor and an 8-hour cap). When it binds, the run ends
`budget_stopped` too, not `failed`: like the cost cap, it is a bound someone set doing its job, and the
results already delivered are kept and counted in the run's completeness record. `budget_stop_reason` tells
the two apart (a clock stop's reason opens with `wall-clock budget`), and `error_details` stays empty,
because that list counts faults. A run whose time budget fired under 0.66.0 or earlier is stored `failed`
with `Job timed out after Ns` in `error_details`, and is not rewritten.

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
3. **Refuse.** It refuses an arm predicted above its run's cap, or one nothing can predict — no pricer, no
   plan, or no history to bound — whose cap the run would merely inherit. An unpriceable arm under a cap
   the launch named runs under it.

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

A host therefore prices no arm itself: a wrapper that priced assembled runs would be a second rule, and a
second pricing of the same arm.

## Spend outside any run

Four engine calls have no run around them: a launch's case generation (an `llm` variation axis's writer),
the rubric proposer (`propose_draft`), an analysis generation, and a judge repeat. Every spend operation
is bounded in dollars before it spends.

**Case generation.** Generation runs before any run exists, so its calls are outside every run's cost cap
and metered-call ceiling — which is one reason the engine prices every arm before any launcher runs
(above). The launcher hands `generate_variations` the request's `budget=request.generation_budget`: every
`llm` axis's call is priced on the writer's client (`price_ceiling`, the host's answer) against
`LaunchSettings.max_out_of_run_cost_usd` before the first is made, and each is ledgered as an
`OutOfRunSpend` document (`EvalStorage.query_out_of_run_spend`) under the launch's group, written on the
host's blocking executor (an `OutOfRunBudget` names its `blocking_executor`, as an `EvalHost` does).
`propose_draft` takes a budget the same way.

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

**A host's own `spend` actions.** A host's own `spend` action carries no such obligation: the class is a
label a tool cut splits on, metered only as far as the host's handler meters it.

**Reading it back.** What was spent out of run is read back by `scope_out_of_run_spend` — the
`scope_out_of_run_spend` action, and the CLI's `spend` — narrowed by purpose, launch group or template.

## How a result's cost is counted

Each completion your client returns names its own `price_source`; the engine stores what it is told and
never assumes a provider. Your kind reports its own calls' spend only as usage rows
(`CandidateTelemetry.usage`), each call's dollars as your client priced them; the engine derives a
result's `cost_usd` from those rows and its background work's (`async_deliveries`, see
[background work](adopting-a-host.md#background-work-payloads-and-spend)).

**A `run_eval` or `compare` candidate reports its spend by returning an `Answer`.** The quick layer's
candidate is a plain async function, so the engine sees what it returns and nothing of what it spent.
Return `Answer(value, model=..., input_tokens=..., output_tokens=..., cost_usd=...)` instead of the bare
value and the call becomes the cell's `candidate` usage row: `value` is graded and stored as a plain return
would be, the result's `cost_usd` is derived from the row, the summary prints `candidate spend: $... over N
call(s)`, and `compare`'s report tests the arms' spend against the control like any other reading. The
contrast is on `production_replicating_cost`, the spend of the roles production runs: `cost_usd` and
`program_cost` also sum what a judge spent, which is measuring cost and is never tested between arms.
Every surface that states what an arm costs reads the same figure: each cell of an analysis's decision
surface carries `production_replicating_cost`, its table's cost column and a frontier chart's cost axis are
that measure (a frontier refuses `cost_usd` there), and wherever `cost_usd` or `program_cost` is drawn it is
labelled measuring spend. An analysis stored before cells carried the candidate's spend froze `cost_usd` on the
cost axis; its table still shows that column, now labelled measuring spend. A
field left `None` is unreported, not zero, so an `Answer` with no `cost_usd` leaves its result's cost
unknown. A candidate that returns anything else reports no spend, as before.
`examples/compare_two_models.py` prices each Claude call this way.

**What an arm costs and what the program spent count different results.** A comparison cost — the
candidate's own spend, `production_replicating_cost`, and every mean of it (`mean_prod_cost_usd` in a run
summary, a frontier point's cost, an analysis cell's) — reads only the turns the candidate took. It leaves
out a result the harness faulted, since a cell an apparatus fault cut short spent less than a whole one and
would let the rig make an arm look cheaper. It also leaves out a call the candidate's model refused straight
away, which took no turn. Program spend (`cost_usd`, `total_cost_usd`, `mean_cost_usd`, a `cost_usd` pivot,
the `cost_usd` history series, the budget view) keeps both, because those dollars were spent. `n_prod_cost_usd` counts what the comparison figure rests on.

**A cost pivot says what its cells pool.** `scope_pivot(metric="cost_usd")` averages measuring spend, so
each cell names the role sets its dollars were summed over (`cost_compositions`), and the table sets
`cost_compositions_differ` when they are not all one set: a cheaper cell may only have priced fewer things.
A cell that pools results from a run that replayed its third party with results from a live run is
`withheld` with the reason, because the mean of the two is neither one's spend; put `cassette_mode` on an
axis to read each alone. A cell whose observations carried a seeded or replayed background delivery counts
them (`n_substituted`) and says its dollars leave that delivery's spend out (`substitution_disclosure`); a cell
built only from such observations says it is no live run's spend. The table carries the same cassette-mode
sentence `runs_compare` does, and the export carries `cassette_mode`, `substituted_deliveries` and `cost_roles`
as columns. `runs_compare` states no
dollars, so it has no compositions to name.

**Unpriced is a state, never zero.** A call your client could not price (a local model, say), or
background work's paid calls that report no `money` and that a run with declared rates has no rate
for, leaves the result's `cost_usd` as `None`. A reported `money` wins over the run's rate.

- Every cost aggregate leaves such a result out of its dollars and counts it beside them (`n_cost_usd`,
  `n_unpriced`).
- A capped run stops on its first unpriced result, because a cap cannot enforce a ceiling on spend it
  cannot count.
- An uncapped run carries on and counts them.
