# Design: the engine owns the remedies (one call per next step)

Status: design, revision 2, 2026-11-15.

Read against:
- `evals/bp-s5a2` @1fdc83582 (S5a-2 round 5), and `evals/main` f9ab6f74d;
- `evals/bp-k` @de8623349 for the pooling check;
- the production host at `/opt/venv` @a78b15b66.

Nothing here is built. Paths are under `packages/evals/src/threetears/evals/` unless they start with `tests/`
(meaning `packages/evals/tests/`) or `discodon/` (the production host). Line numbers are at the commits above.

## Revision 2 (2026-11-15)

An independent review (`design-remedy-actions-review.md`, verdict "build with changes") found five blocking problems
and ten smaller ones. Every finding was checked against the code and every one holds; none was rejected. The owner
ruled on both questions.

| # | What changed | Why | Finding |
|---|---|---|---|
| 1 | An extend whose held contrast is not fully paired first **cross-fills**: each arm runs the held cases only the other arm ran. The size is then planned on the design that will actually be read | `contrast_cases` (analysis/stats.py:1581) reads only shared cases once two exist. The old plan subtracted held cases that would leave the contrast | F1 |
| 2 | **One conditional-power projection**, built over the existing `bounded_cases_to_clear` (analysis/stats.py:2975). The held cases are fixed and the stated probability is what a seeded simulation of the real path measures. The two-point projection is gone | "Needed − held" lifts about 88–95%, not 80%. A second projection would drift from the first | F2 |
| 3 | The per-case values the projection reads go in a **new top-level part, `per_case`**, outside the digest by the existing rule (report_input.py:1429). `Reading` is unchanged | `_OUTSIDE_THE_DIGEST` drops only top-level keys (:1454), so a field on `Reading` moves every digest | F4 |
| 4 | An arm "as launched" now uses **one mapping of a stored run to its launch**, factored out of `admit_resumed_run` (run/launch.py:2278). Before any spend it is checked twice: resume's drift check, then a **would-pool** check using K's shared cell comparison (refusal `would_not_pool`). After the runs end, a served-model change is outcome `split` | New runs that record measure definitions do not pool with old unrecorded ones (run/resume.py:372), so the spend would lift nothing | F5 |
| 5 | Spend safety. **`max_total_usd` is required**, and the per-run `max_cost_usd` parameter is gone. The extend **joins once at the end**. An unfinished remedy blocks the next one. `closes` is matched **by the caveat's identity**, not its id string, and `Gate` gets an id | Spend could exceed the estimate. A crash left a partial look in the campaign. Ids change when a sibling caveat is raised (caveats.py:815) | F6, F7, F8 |
| 6 | Owner rulings recorded (§10): an extend on an unplanned campaign is allowed with the looks caveat stated truthfully, at the rate measured for planner-sized extends, and on the first screen of existing campaigns too. No default `planned_looks`. No budget-limited smaller extend | Owner, 2026-11-15 | Q1, Q2 |
| 7 | The D1 interface gains L5 (a look a remedy added is raised at report reach on any campaign) and L6 (planner-sized extend rows in D1's simulation). The pure step types `raises: [looks]` | Adaptive sizing inflates more than fixed batches. Pre-D1 campaigns raise looks at method reach only | F3, F9 |
| 8 | `looks → confirm` gets a fresh plan. "Unseen" means never run by any run that was ever a member; a confirm also excludes the compared variants' runs on the template | `plan_of` had no `looks` branch. Unseen was read from current membership | F10, F11 |
| 9 | Arms that differ in `measure_latency` are refused. A latency re-run runs the control in the same sequential group. "Successive groups" was a misread: an all-latency group runs sequentially (run/arm_schedule.py:48) | | F12 |
| 10 | The production host's launch path was read. Both its kinds honour a case set, and its one generating kind calls the engine's `generate_variations`, so the host binds nothing new | | F13 |
| 11 | Chunks renamed M0–M3. M1 is split into M1a (estimate, spends nothing) and M1b (spend). `EvalRemedy` lives in `run/remedy_job.py`. M1–M3 are built on one branch and merged once | | F14 |
| 12 | Minor fixes: measured best-of-n bias; near-zero sizes stopped by the budget; ordering refusal `first`; the point-estimate row; `aspects` on the record; subtraction counts corrected | | F15–F20 |

## 0. Summary

A next step is a promise: taken as written, it can close its caveat by the rule that raised it, at the stated
probability. Four Critic rounds each found another way a hand-assembled sequence of catalogue actions broke that
promise. The report builder wrote recipes for actions that know nothing of the campaign, so every fact the engine
holds and the reading lacks became another step field: `then`, `after`, `as_launched`, `alike`, `call_arms`,
`limits`, `unpaired_arms`.

The engine now owns the procedure, behind one spend action and its estimate:

| Action | Permission | Takes | Does |
|---|---|---|---|
| `campaign_remedy` | spend | `campaign_id`, `closes`, `max_total_usd` (required) | Re-reads the campaign, finds the caveat by identity, plans it on the design that will be read, and runs one engine procedure: **extend**, **rerun** or **confirm**. Returns `JobsStarted` (`remedy:<id>`) |
| `campaign_remedy_estimate` | read | `campaign_id`, `closes` | The same plan, priced, with every refusal the spend would make. Spends nothing. Returns `RemedyEstimate` |

A launching step becomes:

```
{action: "campaign_remedy", calls: [{campaign_id, closes}], needs: ["max_total_usd"],
 priced_by: "campaign_remedy_estimate", raises: ["looks"]?}
```

The commonest A/B (one model, two prompts, unsettled) gets a one-call step that lifts it. The held cases are paired
by cross-fill, the remainder is one shared new case set, both arms run in one launch group, and the job joins once
and counts as one look.

## 1. The remedies, from the code

Remedies are mapped in `REMEDY_OF_REASON` (analysis/caveats.py:191) and `_REMEDY_OF_KIND` (:215), and taken through
`STEP_OF_REMEDY` (:351).

| Caveat | Remedy | Today's step | What closes it | Proposed step |
|---|---|---|---|---|
| gate `too_small` (`too_few`) | add_cases | `run_launch` + `campaign_runs_add`; driven on an llm axis (tests/test_caveat_steps_driven.py:354); limit `axis_may_not_generate` | Every arm reaches the floor on cases it did not hold | extend |
| `borderline` | add_cases | Same sequence | The two-sided interval at its table's level clears the line | extend |
| `point_estimate`, a mean below the floor | add_cases | Same, sized to the floor (caveats.py:1232) | An interval exists. Whether it then clears is a separate question; `borderline` may follow | extend, promising an interval only |
| `point_estimate`, a percentile or a mean at the floor | add_cases | No call, `no_interval_to_gain` (:1241) | Nothing | unchanged |
| `unsettled`, arms differing only in model | add_cases | The control's call only, then a join (:1484) | `within_interval` clears the margin's line (answer.py:417) | extend |
| `unsettled`, a beyond-model arm (**one model, two prompts**) | add_cases | **No call**: `lift_not_shown`, `unpaired_arms` (:1625; test_caveat_list.py:672) | Same | extend, with cross-fill where held cases are unshared |
| `no_variation` | add_cases | Unsized, `no_spread` (:1303) | Nothing at today's estimate | unchanged |
| `apparatus:<dim>` | rerun_balanced | Launch (`alike`), add, remove; driven (driven.py:447) | The compared arms at one value of `<dim>` | rerun |
| `confound:<dim>` | rerun_balanced | No call (:1601) | Hold the dimension, where a non-lever setting holds it | rerun for a non-lever apparatus dimension; otherwise none |
| `measured_under_concurrency`, `latency` | rerun_serially | No call (:356) | Every compared arm, the control included, re-run with `measure_latency` in one sequential group | rerun |
| `replayed` | run_live | No call (:368) | The arms re-run live | rerun |
| `late:*`, `best_of_n` | confirm_on_fresh_cases | `campaign_create`, launch, add, `campaign_update`; driven (driven.py:512); three limits | Nothing on this report: a new campaign declared before its results, on unseen cases | confirm |
| `looks` (past plan) | declare_looks | `campaign_create` with needs: an empty campaign (:371) | Nothing on this report: a fresh read | confirm, with a fresh plan (§2.4) |
| `split` | rerun_on_one_cell | No call (:354) | The provider serving one model; no setting does it | none: a host or person acts |
| `not_recorded`, `untimed`, `request_settings`, `judge_unstated` | record_and_rerun | No call (:367) | The host recording what it did not | none: the host's recording |
| `unfinished`, `short_run` / `identity_unrecorded` / judge kinds / `no_control` / `nothing_declared` | resume / remeasure / check / name / declare | One call (or none) | Unchanged | unchanged |
| below bar, leans, people disagree, identical arms, trials lost, not random, skipped | replace, sharpen, repair, draw, finish | None | Acts outside the engine | unchanged |

## 2. The actions

### 2.1 One action, keyed by the caveat

| Option | Verdict |
|---|---|
| Three actions taking arms and a size | Rejected: the taker's size is a second planner. It also costs six names (three spends, three estimates) |
| Three actions keyed by the caveat | Three names for one contract |
| **One action keyed by the caveat** | Adopted. Identity over inputs: the engine derives everything else from stored facts at call time. A new re-run kind is a new procedure behind the same name, admitted by its own driven test |

Steps that are already one call keep their actions (`job_resume`, `run_remeasure`, `result_rate_person`,
`campaign_update`), so there are not two doors. No free-form extension is offered: growing a campaign is the step a
caveat names, sized by its rule, or a new campaign. That is the IA §9.2 D guard.

### 2.2 Parameters, identity, result

| Parameter | Type | Meaning |
|---|---|---|
| `campaign_id` | `CampaignId` | The campaign whose report raised the caveat |
| `closes` | `str` | A caveat or gate id, in any form that names it. Matched by identity (below) |
| `max_total_usd` | `float`, gt 0, **required on the spend** | The person's consent to the whole spend: runs and generation together. The step lists it in `needs`. Absent from the estimate |

**Matching `closes` by identity.** A typed caveat's id changes form with its siblings: `<state>:<reading>`, then
`:<reason>`, then `:<coordinate>` (`_fact_id`, caveats.py:815). So `closes` resolves on the identity tuple instead:
kind, state, reading, reason, coordinate (and the criterion, for a figure). Any id form naming that tuple matches.
The canonical id is returned in `RemedyEstimate.closes`. When a re-raise regroups the caveat's arms, the plan reads
the arms as they are now.

`Gate` (report_input.py:1076) gains an `id`, built by the same rule (`gate:<rule>`), so the gate is addressed like any
caveat. `campaign_remedy` returns `JobsStarted` with job id `remedy:<id>`. `job_poll`, `job_cancel`, `job_resume`
and `jobs_resumable` take that id.

**`RemedyEstimate`.** Typed; words are rendered at read.

| Field | Content |
|---|---|
| `procedure`, `closes` (canonical), `aspects` | What it runs, on which aspects |
| `first` | Present when a caveat of a higher class on the same arms (`apparatus:`, `confound:`, `split`, `identity_unrecorded`) would leave the result unreadable: that caveat's id. The spend then refuses `first` |
| `arms` | Per arm: label, variant key, the stored run it is launched from, the settings that mapping yields, the schedule the group imposes |
| `cross_fill` | Per arm: held cases it will run that only the other side ran |
| `planned_cases` | Per arm, new cases after cross-fill. For confirm, the whole design |
| `lifts` | The rule that lifts the caveat; the conditional probability measured (§6); the unconditional rate as a reported row; for best-of-n, the measured selection rate (§2.3) |
| `cases` | Stored unseen cases used, cases to generate, the writer, the case set to mint |
| `looks` | Taken before and after, planned, whether the reading after it raises `looks`, and the false-separation rate D1 measured for this procedure (§5) |
| `held`, `replaces`, `creates` | For rerun: settings held and runs replaced. For confirm: the new campaign |
| `price` | Per arm (cross-fill and new), generation, total, and the pricer's basis |

### 2.3 The procedures

**The common head, both actions:**

1. Assemble the bundle and report input as the report does.
2. Raise the caveat list and match `closes` (§2.2), else `caveat_not_raised`.
3. Refuse `procedure_none` when the caveat's remedy is not one of the three, naming the step it does take.
4. Refuse `remedy_running` when a live remedy holds the aspect.
5. Refuse `remedy_unfinished` when a failed or lost one holds it, naming `job_resume remedy:<id>` or `job_cancel`.
6. Refuse `first` (above).
7. Refuse `looks_exhausted` when the aspect declares `planned_looks` and this would pass it.
8. Plan with `plan_of` on that fresh input. It is the same planner the report's step uses. Refuse `unsized:<reason>`.
9. Check every arm's launch (§2.6).
10. Price. On the spend only, refuse `over_budget` when the quote exceeds `max_total_usd`.

**Extend** (`add_cases`):

1. **Arms.** For a contrast, the caveat's arms and the control. For a gate or figure, every arm the aspect holds.
2. **Cross-fill.** Where the held contrast shares fewer cases than its arms ran, each arm runs, in a seeded order,
   up to the planned count of the held cases only the other side ran. No generation is needed. Held cases left
   unpaired leave the contrast, and the estimate says how many.
3. **New cases.** The remainder is drawn as unseen cases (§3). Ids go on the record first, then are minted as case set
   `remedy:<id>` v1 (`mint_case_set`, run/case_sets.py:67).
4. **Launch.** Every arm runs as **one** launch-group member on its cross-fill cases plus the set. Each case list is
   minted as the arm's own set so `launch_run` enforces it (run/launch.py:4067).
5. **Join.** Once every run has ended, the runs join in **one** campaign write, idempotent by run id. A run that
   failed or was `budget_stopped` fails the job, and nothing joins. Lost cells are taken up by resume first.
6. **Served model.** If any arm's results show a served model other than its held runs', the outcome is `split`.
   Nothing joins, and the estimate's promise is void, stated.

**Rerun** (`rerun_balanced`, `rerun_serially`, `run_live`). What is held comes from the caveat:

| Caveat | Held |
|---|---|
| `apparatus:<dim>` | `<dim>` at the control's recorded value. Refuse `no_control` or `control_not_one_value` |
| `measured_under_concurrency`, `latency` | `measure_latency` true. The control re-runs too, in the same sequential group: latency against a control measured hours apart reads provider load |
| `replayed` | Cassette off |

Then:

1. Refuse `held_setting_is_a_lever` when the held dimension is in an arm's `variant_levers` (schema/models.py:4570).
   Holding it would define a different arm, and arms are frozen once runs exist (actions/engine.py:2402).
2. Cases are the control's cases on the aspect.
3. Launch one group. After every new run completes, one campaign write swaps the new runs in and the replaced ones
   out, recording the departures (§5 L3). A failed re-run leaves membership as it was.

**Confirm** (`confirm_on_fresh_cases`):

1. **Create the campaign now, before any run.** It copies the original's question, the asked aspects' designs, floors
   and margins, and its `planned_looks`. For `late`, the late criterion is now dated before the results.
2. **Name the control at create.** The original's `CampaignArm` record is copied whole: variant key, levers, identity
   version (kernel/declaration.py:822). Nothing can then be named late.
3. Store `confirms: {campaign_id, closes}` on the new campaign.
4. **Arms.** For `best_of_n`, the pick and the control. For `late`, the surviving options and the control. For
   `looks`, the aspect's arms.
5. **Cases.** Unseen by the stricter confirm rule (§3), one group per aspect, joined at the end.
6. **Size.** The confirm reads its survivors under the new campaign's family correction, so they are sized at the level
   the new campaign reads them at.
7. **The original.** Its caveat stays raised. Its step reads the pointer at read time and names the confirming
   campaign.

**Best-of-n bias.** `size_bias` stays and is measured: M3's seeded selection simulation (N options, pick the best,
confirm at the planned size) gives the realised lift rate. The estimate states that number.

### 2.4 `looks` goes to confirm, with a plan

Looks past the plan cannot be undeclared, so `looks` maps to `confirm_on_fresh_cases`. `plan_of` (caveats.py:1287)
gains a `looks` branch: a fresh plan of the aspect's readings, sized as `best_of_n` is but with no bias flag. When the
report input already shows `{planned, taken}` with the plan spent, the step for an unsettled reading on that aspect
names this confirm, not an extend that would refuse.

### 2.5 The record, idempotency, resumption, cost

**`EvalRemedy`** is a kept leased-job record in `run/remedy_job.py`, beside `EvalLaunch` (run/launch_job.py:55). It
holds:

- `campaign_id`, `aspects`, `closes` (canonical), `procedure`, `planned_cases`;
- per arm: the source run, `cross_fill_case_ids`, and its per-run cap;
- `held`, `case_ids`, `case_sets`, the generation's counts and writer;
- `launch_group_ids`, `run_ids`, `replaced_run_ids`, `created_campaign_id`;
- `phase` (`planned`, `cases`, `launched`, `ended`, `joined`), `outcome` (`completed`, `failed`, `split`,
  `cancelled`, `lost`), lease, faults, refusal.

**Resumption, by phase:**

- `planned`: draw again. Persisted generated cases are found by the ids on the record, so they are never generated or
  paid for twice.
- `cases`: mint, then launch.
- `launched`: each run is taken up as itself (`take_up_arms`, run/resume.py:1075).
- `ended`: join.

A second call is refused while a remedy is live or unfinished. Once one has joined, the evidence has moved. A new call
re-plans on what is now there, and is one more look.

**Cost.** The report's step is pure and carries no price. The estimate prices each arm with the launch's own pricer
(`quote_launch`, run/launch.py:3489) and adds generation through `plan_variation_calls` under the out-of-run budget's
quote.

The spend enforces `max_total_usd` as a running total through caps that already exist, with no new budget mechanism:

- each run's cap is its quoted share of `max_total_usd`;
- the generation's out-of-run cap is its share;
- the caps sum to `max_total_usd`;
- a run stopped at its cap is `budget_stopped`: the job fails, nothing joins, and `job_resume` with a raised
  `max_total_usd` continues.

A plan near zero with no margin can ask for a huge size (`_ON_THE_LINE`, caveats.py:1097). It is stated as computed;
the estimate prices it, and the budget refuses it. There is no invented cap.

### 2.6 Launching an arm again: one mapping, two checks

**One mapping.** `admit_resumed_run` (run/launch.py:2278) maps a stored run to its launch: template, subject, model,
repeats, cassette and corpus, overlays, apparatus settings, margins, latency, schedule, judge and config pins,
simulator, metered-call ceiling, deadline. It is factored into one function, `launch_of_stored_run(run, *,
cases)`, used by resume and remedy alike. The case list is the only argument that differs. A run with no template
(ad hoc) is refused there today (:2316); the remedy surfaces that as `arm_not_launched_by_engine`. An arm whose runs
disagree on any mapped setting is refused `arm_not_one_launch`, naming each.

**Check 1: it would measure as it did.** Resume's drift check (`_resume_drift`, run/resume.py:329): context
components, variant levers, pins, frozen template content and digest. The case list is excluded. Refusal:
`arm_would_measure_otherwise`.

**Check 2: it would pool.** The drift check lets an unrecorded measure definition through, because a take-up
continues the same run (run/resume.py:372, :385). A remedy's runs are new runs, and readers keep apart runs that differ
on a coordinate. So the would-be run, assembled through the mapping as a take-up check assembles it (run/resume.py:1049)
and never started, is compared with the arm's held runs. The comparison is the readers' own shared cell check (K:
`pool_observations` and `unread_arms`, analysis/cells.py:778, :644 at bp-k): context key, recorded or unrecorded
definitions, and served model where one is pinned. Refusal: `would_not_pool`, naming the coordinate.

On the production host's existing campaigns this refuses until their runs are attested. Its old runs are
unrecorded, which K's verify ruled. The estimate's `first` then names the `identity_unrecorded` caveat, whose step is
K's `runs_attest`. Both checks run in the estimate, so a refusal is seen before any spend.

**One condition per group.** `LaunchGroup.add` refuses mixed `measure_latency` (run/launch.py:1015), so the estimate
refuses `arms_not_one_condition` first. An all-latency group runs sequentially (`resolve_arm_schedule`,
run/arm_schedule.py:48). Otherwise the group interleaves. The schedule is a disclosure, not identity, so the estimate
states the one imposed.

**The production host** (read at a78b15b66) needs nothing new bound:

- both its kinds pick cases through `arm_cases`, which plays the launch's case set when one is named
  (discodon/evals/kinds/cases.py:18–33);
- its one generating kind, the classifier, calls the engine's own `generate_variations` with the same arguments
  (discodon/evals/kinds/classifier.py:1187, :1274), so engine-side generation gives the same cases;
- persona has no generating axis, so its extends draw stored unseen cases only.

## 3. Unseen cases

**Extend** needs new pairs, not unseen cases. Cross-filled cases are held cases chosen in a seeded order, never for
their outcome. New cases, though, must be unseen, or they would be repeats rather than cases.

**The rule.** A case is unseen when no run that was **ever** a member of the campaign ran it. That means current
members plus the departures D1 records (§5 L3).

**Confirm** also excludes any case a run of a compared arm's variant ran on this template, in any campaign, since a
pick can be informed there too.

Supply comes in order: stored unseen cases (seeded by the remedy id, so a resume draws the same), then generated ones.
`generate_variations` gains `exclude_params`. Today enum and sample axes reuse stored combinations
(gen/variation_gen.py:220, :244).

| Template | Guarantee | Where it cannot be met |
|---|---|---|
| Any `llm` axis | The writer is asked for values not stored (:232, :354) | A short writer: the job records `kept < requested` and fails with `cases_short` before any launch. Resume asks again, within the budget |
| `enum` / `sample` only | Draw from the axis product minus the seen params | `too_few_unseen_cases`, with `unseen` and `needed` |
| No axis, or a kind that declines generation (persona) | Stored unseen cases only | `too_few_unseen_cases`: a person authors cases and calls again |
| A replayed arm | None | `replayed_arm`: run it live first |

When the action refuses, the step still names it and the estimate states the numbers. The report does not read every
template's cases at assembly (#996): the fact is checked where it is held, before any spend.

## 4. Pairing

The contrast pairs over the cases both sides ran, once there are two (analysis/stats.py:1581). Pairing becomes
engine-enforced:

1. **Held cases** become pairs by cross-fill (§2.3). The same template's stored cases run on another arm, so nothing
   new is needed from the host.
2. **New cases** exist before any launch, as one minted set that every arm runs. `launch_run` refuses an arm whose
   frozen cases differ (run/launch.py:4067), so shared generation no longer rests on a host launcher calling
   `LaunchGroup.resolve_once`.
3. **Every arm is one member of one group**, with its own model, overlays, rig and pins.

**Post-condition, before the join.** Every arm's **completed results** cover its case list, apart from cells lost for
a recorded rig cause, which the reading treats as it always does. Frozen ids alone prove nothing. The record states
the pairs gained.

**Why cross-fill rather than plan fresh.** Correctness: both give a fully paired read. Cost: cross-fill pairs each
held case at one arm-run, against two for a new pair, and the user's earlier spend keeps counting. Selection:
cross-fill picks no case by outcome. The reading over the cross-filled cases is the look D1 counts, like any extend.
Plan fresh only where cross-fill is refused (a replayed arm, drift on the held runs): `fresh=True`, with the typed
statement that the held cases leave the contrast.

## 5. Looks: the D1 interface

D1 counts a look as "the data behind a verdict changed" (design-judges.md §6.2). It reads
`AspectDesign.planned_looks` (kernel/declaration.py:723) and fills `aspect.looks`, which the raiser reads
(caveats.py:895).

| # | Need | Why |
|---|---|---|
| L1 | `looks_of(bundle, aspect) -> (planned, taken)`, the reader's own count, callable before spending | The estimate's `looks` block; `looks_exhausted` |
| L2 | One joined `EvalRemedy` is exactly one look on each aspect it touched, and its launch groups are not counted again | One remedy may use more than one group across aspects |
| L3 | Looks counted from membership history. Every departure (by `campaign_runs_remove`, or a remedy's swap) is recorded append-only on the aspect | A swap must not erase the look at the confounded verdict. The same record defines "ever a member" for §3 |
| L4 | A confirm campaign's first launch is look 1 of its own | It is a new campaign |
| L5 | A look a remedy added is raised at **report reach** on any campaign, pre-D1 campaigns included | The owner's ruling: the looks caveat shows on the first screen of existing campaigns too |
| L6 | New rows in `tests/test_simulated_looks.py`: planner-sized extends ×1, ×2, ×3, undeclared and declared at L = 2 and 3, a pass rate under the bounded interval, no true difference | The rate stated must be the one measured for this procedure. Where the declared rows exceed α, "within plan raises nothing" does not ship |

**The step and the estimate.** The pure step types `raises: ["looks"]` wherever taking it would raise one by the
input's own `{planned, taken}`. The estimate states the L6 rate for this procedure and this count of looks.

**For D1 to cost.** A stage-wise betting product (stage 1 at its launched size, stage 2 at the planned m, stakes fixed
before each stage) is a test supermartingale. By Ville it holds α under any planner-chosen continuation. If its power
cost is modest, it removes the looks caveat for remedy extends by principle.

## 6. Sizing: one conditional-power projection

**The stored fact.** A new top-level part:

```
ReportInput.per_case: dict[str, CaseSamples]
```

It is keyed `<aspect>/<measure>/<arm>` for a contrast and `criterion:<id>/<arm>[/<task|cell>]` for a figure, and is
added to `_OUTSIDE_THE_DIGEST` (report_input.py:1429). `CaseSamples` holds:

- per side, the case ids and per-case values, higher better (the values `RateReading.samples` already carries,
  analysis/floors.py:1362);
- `paired`;
- `value_range`;
- `confidence`: the level the reading was actually read at, so D1's α/L flows through.

A stored input without it plans as today and states "at least".

**The rule** (`analysis/separation.py`, new `cases_to_lift`). The held cases are fixed, the truth is taken as today's
estimate d̂, and new cases are drawn like the held ones. With s the per-case spread, L the line, n₀ the held pairs
(after cross-fill) and m = N − n₀, find the smallest N ≥ n₀ with

> reach(N) ≤ (d̂ − L) − z(POWER) · s · √m / N

where reach(N) is the bounded interval's reach on the held values repeated to N: exactly what
`bounded_cases_to_clear` (analysis/stats.py:2975) computes, called unedited with that room. The search tests a
candidate N through `bounded_cases_to_clear(room=…) ≤ N`, doubling then bisecting as that function does.

- Figures (`_figure_plan`) use the same rule in one-sample form.
- On an unpaired held contrast, the cross-filled halves are new draws: they count toward N but enter the variance term
  as new.
- At zero spread the projection returns the observed count. Near the line it returns the computed size, which the
  budget stops (§2.5).

**The seeded simulation fixes, on the real code path:**

- truth = today's estimate; held sample fixed; new cases drawn from the held per-case values;
- read with `bounded_difference_interval` at the reading's level.

It **holds** the conditional lift rate in [0.78, 0.84] at POWER 0.8:

- for n₀ ∈ {3, 5, 10, 30, 100};
- on a pass rate (k = 1 and k = 3), a bounded mean and a gate figure;
- for a paired held contrast, and for a cross-filled one with ρ = 0 between arms (ρ = 0.3 and 0.6 are recorded, and
  lift more).

It **records**, and does not hold, the unconditional rate: a fresh initial sample each replicate, truth fixed. The
estimate states both.

## 7. What it subtracts, and what it adds

**Public surface.**

- `RemedyStep` (report_input.py:1146) loses seven fields: `then` (:1213), `after` (:1205), `as_launched` (:1187),
  `alike` (:1194), `call_arms` (:1200), `limits` (:1263, a four-value literal), `unpaired_arms` (:1283).
- `NoActionReason` loses `lift_not_shown` (:1140). The no-call re-runs become `StepRule(None)`.
- About six of the fourteen checks in `_priced_and_planned` (:1290) go.
- It gains `raises: list[Literal["looks"]]`.
- `needs`, `calls` and `per_run` stay for the single-action steps.

**Code removed** (analysis/caveats.py), about 400 lines:

- the `StepRule` fields `launches`, `replaces`, `new_campaign`, `balances` and `limits`;
- the recipe constants (:282–350);
- `ArmRuns` through `_launching` (:1329–1533);
- the launching branch of `next_step` (:1597–1641);
- the arms half of `evidence_of` (:1754);
- `Setup` threading;
- the recipe paragraphs of the module docstring (:1–80).

**Tests removed or inverted** (tests/test_caveat_list.py): :583, :640, :693 and :847 go, and :672 inverts into the
driven A/B. The driven file's follower (:245) collapses to one call. Its four tests are re-pointed.

**Added:**

- about 3,700 gross lines: M0 about 600, M1a about 1,100, M1b about 1,100, M2 about 800, M3 about 900, all including
  tests;
- the engine's own `launch_of_stored_run`, factored rather than duplicated.

Net is about +3,300 lines. The engine gains capability; the public surface still shrinks.

| Kind | Effect | Admitting criterion (#1177, 2026-11-15) |
|---|---|---|
| Action | +2: `campaign_remedy`, `campaign_remedy_estimate` | (1) an agent invokes them; (2) every first-screen launching step names them, and a shipped example calls them (M3); (3) they replace the multi-call recipes of ten caveats; (5) the step contract |
| Received shape | +2: `RemedyEstimate`, `CaseSamples` (in `per_case`). `JobsStarted` is reused. `RemedyStep` −7 fields; `NoActionReason` −1 value | |
| Typed, bound | 0 | |
| Job kind | +1 prefix, `remedy:` | |

An agent goes from up to six actions in order, plus seven step fields, to two actions and the step's `calls`.

## 8. The commonest A/B, for a non-expert

A control prompt `c` and a candidate prompt `b`, on one model, launched separately: each ran 20 different generated
cases. The reading on accuracy is unsettled.

1. **The first screen.** S2 shows one reason, class E, "unsettled". S3 shows the step:
   - remedy `add_cases`, `raises: [looks]`;
   - one call, `campaign_remedy {campaign_id, closes: "unsettled:accuracy"}`;
   - `needs: [max_total_usd]`, priced by `campaign_remedy_estimate`.
2. **The estimate** (an agent or the page). Extend:
   - cross-fill: `b` runs `c`'s 20 cases and `c` runs `b`'s 20;
   - then, say, 12 new shared cases, generated, with the writer named;
   - the measured lift probability;
   - looks 1 → 2, no plan, at the false-separation rate D1 measured for one planner-sized extend;
   - prices.

   It is shown to the person: spending is their call.
3. **The spend.** `campaign_remedy` with `max_total_usd` at or above the quote. `job_poll remedy:<id>` shows the
   phases. Nothing joins until every run has ended.
4. **Read again.** At the stated probability, `b` has a verdict over 52 pairs. The trust slot carries `looks` with the
   reference rate simulated for its test, plan and count, and that simulation's setting, unless `planned_looks` was
   declared and `plan_kept` holds for every test the aspect uses; a t-test aspect extended by the planner carries it
   even within its plan (corrected 2026-11-15 after package D1's simulation: a declared plan is not shown to hold
   the level on a t-test with planner-sized extends, 5.59% at a plan of 2, upper bound 6.06%).

## 9. Build chunks

Order: **M0** (now) ∥ D1 → **M1a** → **M1b** → **M2** ∥ **M3**. M1a–M3 are built on one integration branch,
`evals/remedy`, and merged once to `evals/main` with M3's text map. M0 is additive and merges to `evals/main` on its
own.

| Chunk | Scope | Files owned | Size | Depends on |
|---|---|---|---|---|
| **M0 sizing** | The conditional-power projection; the `per_case` part; the planner uses it, including the cross-fill count | `analysis/separation.py` (new `cases_to_lift`; `stats.bounded_cases_to_clear` called, not edited); `analysis/report_input.py` (new `CaseSamples`, `ReportInput.per_case`, `_OUTSIDE_THE_DIGEST`; no other class touched, clear of S5a-3's arm, contrast and measure methods); `analysis/report_input_build.py` (fill `per_case`); `analysis/caveats.py` **only** `Plan` (gains `cross_fill`), `_settling`, `_reading_plan`, `_readings_plan`, `_figure_plan`, `_criteria_plan`, and the `unsettled`, `borderline` and `point_estimate` branches of `plan_of`. Not `_looks` (D1), not the step section. Tests: `tests/test_caveat_list.py` :1066 and :1140 replaced; new `tests/test_simulated_remedy_sizing.py` | ~600 | Branch from `evals/bp-s5a2` @1fdc83582; rebase after S5a-2 merges |
| **M1a estimate** | `EvalRemedy` shape; `campaign_remedy_estimate`; identity matching and the `Gate` id; every refusal of §2.3 and §2.6; `launch_of_stored_run` factored and used by resume; the would-pool check; §3 supply with `exclude_params`; the `looks → confirm` plan branch | `run/remedy_job.py`, `ops/remedies.py`, `run/launch.py` (the mapping only), `run/resume.py` (calls it), `gen/variation_gen.py`, actions catalogue entry, `caveats.py` (identity resolver, `plan_of` `looks` branch) | ~1,100 | M0, D1 (L1, L3), K merged |
| **M1b extend** | The spend: phases, cross-fill, mint, one group, cap allocation, join at end, `split` outcome, resume; add_cases steps become one call with `raises` and `needs` | `ops/remedies.py` (extend), `run/remedy_job.py` (job), `caveats.py` step section for `add_cases`, `report_input.py` `RemedyStep.raises` | ~1,100 | M1a, D1 (L2, L5) |
| **M2 rerun** | Held apparatus, serial with the control, live; atomic swap with recorded departures; `held_setting_is_a_lever` | `ops/remedies.py` (rerun), `analysis/campaigns.py` internal `replace_runs`, the re-run rows of `caveats.py` | ~800 | M1b |
| **M3 confirm and cut-over** | New campaign with the copied control record and the `confirms` pointer; the measured best-of-n bias; delete the seven `RemedyStep` fields and `lift_not_shown`; schema doc; a ladder example | `ops/remedies.py` (confirm), `kernel/campaign.py` `confirms` (after D1 lands), `bundle/evidence.py` census entry, `report_input.py` `RemedyStep` deletion, examples | ~900 | M1b, D1 merged |

**M0 done when:**

- The §6 simulation holds [0.78, 0.84] in every held row, and records the unconditional and ρ rows.
- `report_input_digest`, every snapshot id, every held-out frozen digest, every fingerprint, body, frame and recording
  are byte-identical. Only the held-out `first_screen` steps' `planned_cases` and the new `per_case` part are allowed
  to move, and are re-exported.
- A stored input without `per_case` plans as before.
- The planner reads the reading's own `confidence`, never the `INTERVAL_LEVEL` constant.
- An unpaired held A/B plans its cross-fill and remainder.
- A zero-spread figure returns the observed count with no division by zero.
- A near-line contrast plans its computed size.

**M1a done when** — each case driven through `MountedTool.call` on the toy-host fixture, $0, nothing spent:

1. The estimate refuses `would_not_pool` on an arm whose held runs recorded no definitions.
2. It refuses `arm_would_measure_otherwise` after a template edit.
3. It refuses `arms_not_one_condition` on mixed latency.
4. It refuses `first` under an `apparatus:` caveat on the same arms.
5. It refuses `too_few_unseen_cases` with counts on an enum template, and draws unseen combinations where enough
   remain.
6. A `closes` given in a sibling-dependent id form matches after a sibling caveat is raised.
7. The resume path still passes through the factored mapping: the existing resume tests are unchanged.

**M1b done when:**

1. **The one-model two-prompt unsettled A/B with unshared held cases** is one call. It cross-fills, every arm's
   completed results cover its list, the job joins once, and read again the caveat lifts by `unsettled_arms`' own rule.
2. The estimate's plan equals what the spend ran, and the estimate's price is at least the fixture's spend.
3. A spend is refused above `max_total_usd`.
4. A `budget_stopped` arm fails the job, and nothing joins.
5. A job killed after minting resumes without generating or paying again.
6. A second call while it is live or unfinished is refused.
7. D1's count reads exactly one more look, at report reach on a pre-D1 campaign.
8. The structural test (test_caveat_list.py:506) still accepts every step's call.

**M2 done when:** the J-2 scenario (driven.py:447) runs through one call. The confound is gone and the arms are the
same. A lever dimension refuses. A latency re-run re-runs the control in one sequential group. A failed re-run leaves
membership unchanged. The departures stay counted as looks and as seen cases.

**M3 done when:**

- The J-3 scenario (driven.py:512) runs through one call. No late criterion or control is read, the cases are disjoint
  from every run ever in the original, and the original's caveat stays raised with its step naming the confirming
  campaign.
- A `looks` confirm runs at its fresh plan.
- The best-of-n selection simulation's rate is stated.
- The census classifies `confirms`.
- No §7 field remains.

**Goldens.**

- The held-out `first_screen` is re-exported at M0, M1b and M3: `held_out/{s1,s2,s3,s4,toyhost,courierhost,
  hot-path-scaled}.json.gz`.
- Surface files are regenerated once, at the `evals/remedy` merge.
- No recording, fingerprint, snapshot id, body or frame moves.

**Stored schema.**

- M0: report input only, outside the digest.
- M1a: a core minor for `EvalRemedy` (kept) and the `remedy:` job kind.
- M3: a durable minor for `confirms`. D1 owns the departures record.
- Each change is in MIGRATIONS and classified in the evidence census as not evidence.

**The production host:**

- Nothing to implement (§2.6).
- One cut-over: S5a-2 merges with the multi-call `RemedyStep` fields marked provisional. The host re-pins to consume
  steps only at the single `evals/remedy` merge.
- That merge ships a text map:
  - each removed field → "one call: `campaign_remedy {campaign_id, closes}` with `max_total_usd`";
  - `lift_not_shown` → `StepRule(None)` with its remedy;
  - the two actions, `remedy:`, `RemedyEstimate`, `per_case`.
- Before an extend can run on its existing campaigns, those runs must be attested (`runs_attest`). The estimate names
  that as `first`.

## 10. Decisions

**Owner, 2026-11-15:**

1. **An extend on a campaign that planned no looks is allowed.** The looks caveat is stated truthfully, at the rate
   measured for planner-sized extends (L6, not the fixed-batch figure), and it shows on the first screen of existing
   campaigns too (L5). No default of two planned looks.
2. **No budget-limited smaller extend.** The price before spending and the required `max_total_usd` are the brake.

**By principle:**

1. One action keyed by the caveat's identity. Principles: identity over inputs, one planner, fewer things to know.
2. No size or arm overrides, and no free-form extension. A promise holds only on the engine's plan.
3. **Cross-fill before generation.** It gives the same fully paired read at about half the spend per pair, keeps the
   user's earlier spend, and picks no case by its outcome. Its read is counted as a look.
4. One mapping of a stored run to its launch, for resume and remedy. Two would drift (criterion 3).
5. Would-pool is checked with the readers' own cell rule. A spend that cannot pool lifts nothing: no pooling by
   assumption.
6. Join once, at the end. A partial extension is never read, and never counted as no look.
7. `max_total_usd` is enforced through per-run and out-of-run caps that already exist, rather than a new budget.
8. No action for split, record-and-rerun, mechanism and lever confounds. No setting addresses them, and a driven test
   admits every procedure.
9. The rerun holds the control's value and swaps atomically. The confirm copies the control's arm record at create.
10. A precondition is checked where its fact is held, in the estimate, before spend. A price lives in the estimate,
    never in the report.

There are no open owner questions.

## 11. Not verified

1. That assembling the would-be run through the mapping (as `_check_take_up`, run/resume.py:1049, does) has no side
   effect when the run is never started.
2. K's function names and the exact cell comparison (`pool_observations`, `unread_arms`, read at bp-k @de8623349,
   which is still merging).
3. The conditional search's behaviour for n₀ < N < 2n₀, where the cushion grows with N and the predicate may not be
   monotone. The simulation band is its test.
4. The one-sample form of `bounded_cases_to_clear` for a gate figure (the line as a constant control on a widened
   range).
5. The size of `per_case` on a large campaign (#996), and the cost of one full report-input build per estimate.
6. That stored runs can be queried by template and variant key, for confirm's exclusion.
7. That per-run caps allocated by quoted share leave a large enough margin for arms whose real cost runs above the
   quote. A run at its cap fails the job rather than overspending, so the risk is a failed job, not money.
8. That `cassette_mode` off is accepted for every kind (`run_live`).
9. D1's final shape. §5 is the interface this design needs.
