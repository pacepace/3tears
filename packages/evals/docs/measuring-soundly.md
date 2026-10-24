# Measuring soundly

**For** anyone about to spend on a campaign, choose its arms, k and a case count, pick a judge, or decide whether to
believe a surprising result. **Answers:** which design answers which question and what it cannot answer; how to
declare a campaign, its question, arms, aspects and floors; how cases and repeats trade off; which read gives the answer; which levers
are worth sweeping; and which practices keep an eval's numbers honest, with what went wrong without them. Arm, lever, variant, k and campaign are as
[Concepts](concepts.md) defines them.

The second half is practice notes from building the engine against one production host between May and October
2026: a conversational agent with tools, its three-label classifier, a retrieval sub-agent it calls, and the engine's
own analysis writer. Each finding is a rule, its reason and an *Evidence:* line; "replicated" means two or more
independent campaigns or arm pairs showed it. The Evidence lines summarise campaigns run in that private application;
the campaigns themselves are not published. Every "in simulation" figure is reproducible from the package's
known-answer suite (`packages/evals/tests/test_sim_*.py` and `test_simulated_*.py`, on
`tests/simulation_support.py`). **pass^k** is the chance that k attempts at a case all pass: per case C(c, k) / C(n, k)
from c passes in n ≥ k repeats, averaged over the cases measured that deep (field `pass_hat_k`). It is not pass@k, the
chance that at least one of k passes.

The code blocks run in order, in one async context (`python -m asyncio`, or inside `async def main()`), each using
names an earlier one defined. The frontier block also uses `ROOM`, the world declared in the first block of
[Evaluating a tool-using agent, step 1](evaluating-agents.md#step-1-declare-the-world): run that block first.

Two more terms, and a rule:

- **Control:** the arm every other arm is tested against. It is a variant (one configuration), not a run.
- **Shape:** what the engine derives from what ran (`bundle.design.shape`). `one_factor_at_a_time` means each
  arm moves at most one lever off the control. `multi_factor` means some arm moves more than one.
  `undesignated` means no control resolved, so nothing is known to be controlled.
- Only goal-state checks and judged dimensions decide a pass for pass^k; guardrails take no part.

## Choose which levers to sweep

Before the design comes the choice of levers, and a host with many has more than a campaign can afford. Read where
the evidence is already thin. Each aspect's fold on `report_read`'s page has a Not tested section that lists every
`thin` and `unswept` lever. The full rows, with `n` and dispersion, are each aspect's `bundle.coverage`, in the
`bundle-<aspect>.json` written beside a saved page (the command line's `report --out`).

Sweep the `thin` or `unswept` levers that could change a decision, and leave out one whose every level would be
shipped the same. Predicting an unrun cell and ranking what to run next is
[#657](https://github.com/pacepace/3tears/issues/657).

## Start from the question

| Your question | Run these arms | Shape | Read the answer from | It cannot tell you |
|---|---|---|---|---|
| Is B better than A? | A and B, differing in one lever; A is the control | `one_factor_at_a_time` | [the contrasts against the control](reading-reports.md#reading-a-comparison) | why B differs, or whether another level would do better |
| Which of several single changes helps? | the control, plus one arm per change, each moving one lever | `one_factor_at_a_time` | the contrasts, Holm-corrected across the arms | how the changes combine |
| Which combination wins? (say three models by three research-round settings by three search limits, once coverage shows those three levers are the thin ones: 27 arms at k of 3 or more, a control declared up front) | every combination of the levers' levels (a full grid) | `multi_factor` | the contrasts, read again against other controls (`Result.against`) | which lever made a combination's gain, except where the grid holds that lever's two levels with all else equal |
| Which arm is cheapest among those good enough? | arms that differ in cost (model, effort), all run at one k, under one subject | any | the [frontier](#the-frontier-passk-against-cost): pass^k, cost and latency against a bar | whether two arms differ on a measure; that is a contrast |

**"One lever per arm" is a rule for the first two questions, not a law.** It is what makes each contrast
attributable to its lever. In a grid, arms that move several levers are the point. A lever's effect is clean
only between two arms that differ in that lever alone. A 2×2 grid holds two such pairs for each lever: the
report tests the lever at the control's level of the other lever, and `Result.against` tests it at the
other level. The engine does not test the interaction itself. When those two effects differ, the combination
matters more than either lever. Each `against` is its own Holm family. The bundle's row for each lever
(`bundle.coverage`) lists in `confounded_by` every other lever that varied across its runs. While that list
is non-empty, the row is a comparison averaged over the other levers, not a controlled one.
[`examples/3-invoices/extract.py`](../examples/3-invoices/extract.py) says in its docstring how to name
the arms of a grid, prompt by reasoning effort.

**The winner of a grid needs a test against the runner-up.** The contrasts test every combination against the
control, never against each other. To test the leader against the rest, `await comparison.against(leader)`. You
chose that control after seeing the results, so treat what it shows as a lead to confirm in a fresh campaign. No
table on the report ranks the combinations.

## Declare the campaign: its question, arms and aspects

**On the quick path, an `Eval` over a mapping of arms declares it for you.** It sets one axis per factor, with the
levels the arms ran, and the control from the mapping's first key. It sets `intended_repetitions` to `k`, and states the stimulus held
fixed and the rig commissioned. It launches every arm together, so time and load do not differ between them.

```python
from threetears.evals.analysis import inspect_campaign_bundle
from threetears.evals import Eval
from threetears.evals.run import run_blocking

CASES = [
    {"ticket": "I was charged twice for March.", "queue": "billing"},
    {"ticket": "Refund the plan I bought yesterday.", "queue": "billing"},
    {"ticket": "The export button does nothing.", "queue": "bug"},
    {"ticket": "The app crashes on large photos.", "queue": "bug"},
    {"ticket": "The reset email never arrives.", "queue": "account"},
    {"ticket": "Change the email on my account.", "queue": "account"},
]

def classifier(prompt):  # an offline stand-in for a model call under each prompt
    async def classify(case):
        text = case["ticket"].lower()
        if "email" in text and prompt == "v2":
            return "account"
        return "billing" if "charged" in text or "refund" in text else "bug"
    return classify

comparison = await Eval(CASES, {"v1": classifier("v1"), "v2": classifier("v2")}, ["queue"], k=3).run(
    store=":memory:", scope="triage"
)
host = comparison.host
# A store read, so it runs off the event loop, on the host's executor.
inspected = await run_blocking(
    host.blocking_executor, inspect_campaign_bundle, host, comparison.campaign_id, comparison.scope_id
)
design = inspected.bundle.design
print(design.shape, [arm.moved for arm in design.contrasts])  # one_factor_at_a_time [{'candidate': 'v2'}]
```

**On your own host, declare it when you create the campaign.** A campaign asks **one question** of its arms, and
measures it on one or more **aspects**: an aspect is one template, the runs gathered on it, the margin a candidate
may fall below the control by there, and the design those runs were gathered under. A decision that rests on two
behaviours (routing and the replies, say) is one campaign of two aspects, not two campaigns. Call `create_campaign`
(the `campaign_create` action takes the same fields) with the `question`, the `arms` (name the control by one of
its runs with `from_run_ids`, so you never type a digest; a candidate nobody names is read off the runs) and the
`aspects`, each with its runs, a `margin` and a `design`. A campaign you create without a design is one you say is
exploratory (`exploratory=True`); creating one that says neither is refused, so no campaign is exploratory by
accident. A design that names an axis your host cannot vary, or a bar looser than the registered one, is refused
when the campaign is created. Create it before you launch, with no runs, then add each run to its aspect with
`add_runs_to_campaign` (the `campaign_runs_add` action; a run goes to the aspect of the template it ran, or the one
you name). This declares the runs above by hand:

```python
from threetears.evals.analysis import create_campaign

runs = {arm: summary.run_id for arm, summary in comparison.arms.items()}
levels = [{"content": arm, "display": arm} for arm in runs]
campaign = await run_blocking(  # a store write, so it runs off the event loop too
    comparison.host.blocking_executor,
    create_campaign,
    comparison.host.storage,
    {
        "name": "triage v2", "subject_id": "triage", "behavior": "routing",
        "question": "Does v2 route more tickets right?",
        "arms": [{"label": "v1", "role": "control", "from_run_ids": [runs["v1"]]}],
        "aspects": [
            {
                "label": "routing", "run_ids": list(runs.values()), "margin": 5.0,
                "design": {
                    "axes": [{"axis_id": "candidate", "values": levels}],
                    "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"},
                    "intended_repetitions": 3,
                    "merit_axes": ["quality"],
                },
            }
        ],
    },
    scope_id="triage", created_by="you", profile=comparison.host.profile,
)
```

| Field | What it does | Needed for |
|---|---|---|
| `question` | The one thing the campaign asks of its arms, in your words: what would make a candidate the one to adopt. Empty, it asks nothing, and every reading is [exploratory](reading-reports.md#readings-no-question-asked-about-exploratory) though a design is declared. | a confirmatory answer |
| `arms` | `{label, role, from_run_ids}` each, the control first. The server reads each arm's variant key from its runs and refuses runs of one arm that carry two keys, naming each. The arms decide the [kind](concepts.md#campaign-kind): one arm is a measurement, several with no control a choice (every pair read both ways), a control and candidates a change. | A/B, single changes, grid |
| `aspects[].margin` | How far, in points of the pass rate, a candidate may fall below the control on this aspect before it is shown worse. A declared aspect states one when the campaign asks a question and names a control; with no control a margin is refused, and the remedy is an absolute bar. | a change |
| `aspects[].design.axes` (required) | The levers and levels you meant to compare. Each declared axis gets a row in `bundle.coverage`, `unswept` if it held one level; compare the row's `levels` with yours. | every declared aspect |
| `aspects[].design.held_fixed` (required) | Whether the cases held the stimulus fixed, and whether the rig was commissioned or found. | every declared aspect |
| `aspects[].design.merit_axes` | Which axes the question asks about on this aspect: they set the aspect's Holm family. Omitted, the question asks about every axis. | a scoped question |
| `aspects[].design.intended_repetitions` | The repeats per case you meant each cell to get. A cell that falls short is named in `bundle.short_cells`. | the frontier, and any cell you want checked |
| `aspects[].design.bars`, `merit_priority` | Thresholds tighter than the host's, and the tie-break order between merit axes. | a bar verdict, a tie-break |
| `aspects[].design.crossing`, `skipped_cells` | Which combinations of levels (cells) you meant to run: `full`, or `star` (each axis's first level is the centre; cells moving more than one axis from it are skipped), less any `skipped_cells`. `bundle.declared_crossing` then reads a skipped cell as `skipped_by_design`, never a gap, and an unrun cell you meant to run as `not_run`. With neither, no cell is read either way. | a partial grid |
| `floors` | Lines a candidate is warned for crossing, each `{aspect, clause}` on a declared aspect: `{kind: "rate_drop", max_points, reading?}` or `{kind: "cost_lower"}`, typed, never words. A floor reads a candidate against the control, so a campaign naming none is refused one. A broken floor warns on the page's first screen; no floor decides anything ([declaring them](adopting-a-host.md#declaring-what-a-campaign-asks)). | a warning nobody can miss |

**A criterion set after its readings is said, not demoted; data age is never a defect.** What pre-registration
guards against is criterion selection: a margin, floor, bar, control or aspect chosen once the readings it judges
could be seen. Each is dated where it is stored (the aspect's `asked_at`, which also dates its bars and a margin or
floor declared with it; `margin_declared_at`; `Floor.declared_at`; the control's `named_at`) and read against the
first stored result of its aspect. One set later reads as one set before, since the operator is trusted to have
fixed it in good faith, and every page saying the aspect's readings names it once with its time
(`reading_scope.late_criteria`); one stored by an older build with no time says its time is not recorded. When the
runs ran, or when the campaign holding them was declared, never counts: a run whose measurement context is unchanged
reads exactly like a fresh one, and rerunning it to make it count only buys the same reading again. The declaration
is frozen once any aspect holds a run: `update_campaign` (the `campaign_update` action) refuses `question`,
`aspects` and `floors` in any form from then on, and names a control only while none is. To amend a question after
the runs, create a new campaign that adopts the aspects it keeps (`aspects_from`): each adopted aspect keeps its
times, so it reads as the campaign declared before the runs did, and whatever the new campaign adds (a margin, a
floor, an arm) is dated now and named on the page where it came after its readings. An aspect adopted from a
campaign that asked no question is the known exception: it keeps its source's time when the new campaign first puts
it to a question, so that is not named. Runs added to an aspect (`campaign_runs_add`) date nothing: where their results
predate the aspect's criteria, those criteria are named as set after their readings.

**Declare a design when you know the question; explore when you do not.** Either is a choice you state when the
campaign is created: a `design` on an aspect, or `exploratory=True`. Declare
one when the question is settled before the runs: you know which lever you are comparing, against which
control, and what answer would change what you do. Only a declared question can confirm anything, so a
decision you will act on needs one; declare it before the runs where you can, since one set after them is disclosed
as such. Leave it out when you are learning how to run evals, or testing an
intuition you cannot yet put as a question. A campaign with no design is **exploratory**: its report and its
bundle say so once, at the top, and its readings confirm nothing. Its analysis reads the design inferred from
the runs, and never calls that design declared. Do not declare a design you do not have just to get past this:
a made-up question makes leads read as answers, which is worse than declaring none. When an exploratory
campaign finds something worth knowing, declare the campaign that tests it. A design that names
an axis your host cannot vary is refused naming `declarable_axes()`, which lists the levers and open families
your host accepts.

**The control is a variant, not an earlier baseline run.** It is a configuration, resolved through any member
run that carries it. A run of the incumbent from last month is not a control: run the control in the same
launch as the other arms, on the same rig. A run of the same configuration from another launch pools into the
control arm, and the report discloses runs that did not overlap in time.

**Check the design before you read it.** `inspect_campaign_bundle` costs nothing. Check `design.shape`, and
each arm's `moved` in `design.contrasts`. If you meant single changes and the shape says `multi_factor`, some
arm moved a lever you did not intend. `design.control_excluded` says why a declared control did not resolve,
and `short_cells` names the cells that got fewer repeats than intended.

## How many cases, how many repeats

**For a difference in means, add cases first.** That covers the A/B, the single changes and the grid. Every
interval is computed over cases, so six cases at k = 3 are as wide as six draws, not eighteen. Repeats only
steady each case's mean, and k = 1 to 3 is enough. Size the bank from an effect you have measured:
[Variance, k and how many cases](#variance-k-and-how-many-cases) gives the numbers. A grid
multiplies the cost, since each combination is a cell: 2 × 3 levels at 20 cases and k = 3 is 360 trials.
[Cost and budgets](cost-and-budgets.md) prices a launch before it runs.

**For the frontier, set k first, then add cases.** pass^k counts a case only once it has run at least k times.
The frontier reads every arm of a subject at one depth, the smallest k among its runs, so one arm launched at
k = 1 reads the whole subject at pass^1. So:

1. Choose the depth the question needs (k = 3 for "passes three times out of three").
2. Launch every arm at that k, and declare it as `intended_repetitions`.
3. Then add cases: the bar is decided on the interval, and the interval narrows with cases.

An arm that passed every attempt reads [0.70, 1.00] on 12 cases, which straddles a bar of 0.8, so it is
undecided. On 20 cases it reads [0.82, 1.00] and clears the bar. Repeats beyond k add little when a case's
attempts agree.

## The frontier: pass^k against cost

`frontier` ranks the arms of every completed run in a scope, not one campaign's, on pass^k, cost and latency,
one subject at a time. With a `bar`, it decides each arm's pass^k against the bar on its interval
(`bar_decision`). It then picks the cheapest arm that cleared, and says whether that arm is shown cheaper than
the others that cleared (`verdict.cost_decision`). Here two stand-in agents act on the room from
[Evaluating a tool-using agent](evaluating-agents.md#step-1-declare-the-world). The high-effort one costs more
and never slips:

```python
from functools import partial

from threetears.evals.analysis import frontier
from threetears.evals import Answer
from threetears.evals.run import list_runs

def agent(cost, skips):  # an offline stand-in that leaves the light alone in the rooms named in `skips`
    async def act(case, room):
        seen = await room.view()
        wanted = "on" if seen["daylight"] == "dark" else "off"
        if seen["light"] != wanted and case["room"] not in skips:
            await room["switch_light"](to=wanted)
        return Answer("Done.", model="m", cost_usd=cost)
    return act

ROOMS = [{"room": f"{light}-{day}-{i}", "world": {"light": light, "daylight": day}}
         for light in ("on", "off") for day in ("dark", "bright") for i in range(5)]
arms = {  # named as k=v pairs: each pair a factor, the first arm the control
    "model=m, effort=high": ROOM(agent(0.002, set())),
    "model=m, effort=low": ROOM(agent(0.0005, {"off-dark-0", "off-dark-1"})),
}
rooms = await Eval(ROOMS, arms, ['(state.light == "on") == (state.daylight == "dark")'], k=3).run(
    store=":memory:", scope="lights"
)
read = await run_blocking(  # a store read
    rooms.host.blocking_executor, frontier, rooms.host.storage, "lights",
    list_runs=partial(list_runs, rooms.host), bar=0.8,
)
for point in read["subjects"][0]["points"]:
    print(point["production_replicating_cost"], point["pass_hat_k"], round(point["pass_hat_k_ci_low"], 2),
          point["bar_decision"])
# 0.002 1.0 0.82 cleared       high effort: passed all 20 rooms, every attempt
# 0.0005 0.9 0.67 undecided    low effort: missed 2 rooms of 20
print(read["subjects"][0]["verdict"]["cost_decision"])  # only_cleared: high is the only arm that cleared
```

The cheaper arm may still be good enough; 20 rooms cannot say. To decide it, add rooms.

Before you rely on it, check two things:

- **Every arm must run under one subject.** The frontier never ranks across subjects. On the quick path,
  every arm of one comparison runs under the comparison's subject (its name), so its models and its named arms
  rank on one frontier. On your own host, run every arm under the subject it measures.
- **pass^k needs goal checks or a judge.** An arm graded by neither, such as a classifier scored only by
  a field grader, has no pass^k: `pass_hat_k` is `None`, `n_pass_no_criterion` counts its attempts, and no bar
  is decided on it. Read such an arm from the contrasts.

The same frontier is the `scope_frontier` action and the `frontier` command
([Serving](serving.md#frontier)); `scope_frontier` in `threetears.evals.ops` returns it as a
`FrontierResult`.

The bundle carries a frontier too (`bundle.frontier`). Check `bundle.frontier_bar_withheld` before reading
its clearing counts.

## Variance, k and how many cases

**Measure the noise floor before reading any difference.** Re-run an unchanged arm and see how far its own
readings move. A smaller difference is not a finding. Stability tracks how mechanical a judged dimension is:
one anchored on an observable order reproduced exactly, and one judging "the right artefact" moved most.
*Evidence:* agent with tools and retrieval sub-agent, re-runs at n=12 per dimension and at k=1 then k=3, 2 campaigns, 2026-08 and 2026-09, 4 of 6 dimensions within 0.2 and 2 moving 0.5–0.7, latency moving 12%, replicated.

**Add cases before repeats, unless the reading is pass^k.** k (default 3) measures consistency and cases
measure coverage, so five cases at k=3 are not fifteen independent draws. The engine computes every reading's
interval over the cases, so five cases at k=3 get the width of five draws, not fifteen; repeats narrow it only
as far as they steady each case.
At n=5 a paired t-test reaches p < 0.05 only once the observed standardised difference d_z passes about 1.24,
and it has 80% power only for a true d_z near 1.7, so "not separated" there says almost nothing. Size the bank
from a measured effect: 80% power needs about 17 cases for d_z ≈ 0.74 and about 34 for d_z ≈ 0.50. The
exception is pass^k, and with it the frontier: a case counts toward pass^k only once it has run k times, so set
k to the depth the question needs first, then add cases
([above](#how-many-cases-how-many-repeats)).
*Evidence:* agent with tools, two independent arm pairs at n=5, 2026-08, pass^k 0.2 → 0.8 gave p=0.174 and p=0.629, replicated; in simulation (2026-09) a 95% interval over observations from 5 cases × k=3 covered the truth 70.3% of the time, against 94.3% over case means; from 2 to 15 cases at k=1 to 5 the engine's interval over cases covered 95% on average for a mean, every configuration within Monte-Carlo error, and at least 96% for a rate, against as little as 51% and 67% over observations (`tests/test_sim_reading_intervals.py`, 2026-10).

**A small bank misses whole failure classes.** A hand-picked handful covers the cases its author thought of. See
[How many cases](building-an-inventory.md#how-many-cases).
*Evidence:* three-label classifier, k=3, 8 then 20 cases, 2026-09: the 20-case bank held a failure class (5 cases, all failed) the 8-case bank had no case of. The prompt also changed between the two banks, so the score's fall (0.875 → 0.750) is not attributable to the bank alone; single campaign.

**Pair by case.** Between-case variance dominates small banks, so compare per-case differences, as the engine
does where levels share cases. Two standard errors over five pairs is still loose: t with 4 degrees of
freedom needs about 2.8.
*Evidence:* retrieval sub-agent, 2 arms, n=5, k=3, 2 campaigns, 2026-09, pooled means showed nothing while paired differences gave −8.1 s at 2.9 se and then −8.7 s at 5.7 se; replicated for latency, though quality signs flipped between the campaigns.

**A bar on a mean over mixed cases measures the case mix.** Set bars per
[stratum](reading-reports.md#results-by-kind-of-case-strata) or over a homogeneous population.
*Evidence:* retrieval sub-agent, 5 cases, every arm, 2026-09, two cases scored 4–5 and three scored 3 under every configuration, so no arm cleared 4.0, single campaign.

**A bar decides on the interval, against a margin, three ways.** Seeded at the incumbent's mean and read on a
cell's mean, a bar fails the unchanged incumbent half the time. The engine reads the cell's interval against
the threshold less the measure's materiality threshold: cleared when the whole interval is on the good side,
missed when it is wholly on the bad side, undecided when it straddles. It seeds the threshold at the
incumbent's mean moved √2 − 1 of the way to its interval's permissive end, which makes a miss of an
unchanged incumbent measured on as many cases a one-sided test at the interval's 2.5%. On a handful of
cases most verdicts are undecided, and that is the finding: the bank cannot tell.
*Evidence:* seeded simulation (`tests/test_sim_bars_and_change.py`), σ = 1, one observation per case, no margin, 2026-10: the mean rule missed an unchanged incumbent 49–50% of the time at n = 3, 6 and 15. The three-way rule missed it 1.3%, 2.0% and 2.2% of the time and cleared it 21%, 19% and 21%; a candidate 1.6σ worse was cleared 1.6%, 0.05% and 0% of the time and missed 15%, 60% and 98%.

### A floor is held only on the bounded interval

**"No worse by more than the margin" is a claim of safety, so it reads an interval that holds its error rate on
coarse values.** A rate floor (`RateDrop(max_points=10)`: no pass rate drops by more than 10 points) reads `held`
only where each rate's bounded interval (the bounded test's, on the 0-1 range, at the nominal 95% and uncorrected)
keeps the drop within the margin, as a guardrail's `held` does. The t interval the Contrasts table prints overstates
its confidence on a pass rate's few coarse per-case shares: a candidate that breaks one case in ten leaves most small
samples agreeing with the control. The price is honest: on seven cases nothing short of a sweep shows a 10-point
floor held, and an unclear floor's `To show the floors hold` table says how many cases would.
*Evidence:* simulated at a true drop of exactly the 10-point margin, 3 repeats per case, 20,000 replicates per row,
2026-10; how often the t interval read held, against the one tail it states (`tests/test_simulated_switch.py`
checks the 12-case row at 1,000 replicates: about 14% on the t interval, none on the bounded):

| Shape | Cases | Held read on the t interval | Its tail |
|---|---|---|---|
| control passes 0.95; the candidate breaks 1 case in 10 | 7 | 12.1% | 2.5% |
| the same | 12 | 13.9% | 2.5% |
| every trial passes; the candidate drops 0.4 on 1 case in 4 | 12 to 30 | 7.8 to 11.9% | 2.5% |

### One family per candidate

**A campaign over several aspects tests each candidate once per aspect, so it corrects across them.** Each
candidate's rows, one per aspect, are one family at α = 0.05, with gains (the change against zero) and drops (the
change against the aspect's margin) corrected apart by Holm's step-down. Each direction's claim is one-sided, read off
a two-sided bounded interval, so a candidate's gains and its drops are each shown wrongly at most 2.5% of the time.
A choice's pairs have no direction, so every pair on every aspect is one family.

- **A reading and its interval never disagree.** A row's interval is read at the level its step tested it at, so a
  row is shown exactly where its interval clears.
- **One aspect reads as a floor reads.** A family of one is read at 95%, the floor's level, on the floor's bounded
  interval, so a one-aspect campaign's floor and its reading on the same margin are one computation.
- **There is no correction across candidates.** Each candidate is its own decision; with several, one of them is
  shown better or worse by chance more often than 5%, and the page says what its correction covers.
- **Floors are not corrected.** A floor warns, and a warning that fires less often for every floor added is no
  warning, so each reads at its nominal 95%.
- **No worse than the margin is not corrected either.** A row neither shown better nor shown worse reads
  `within_margin` where the floor's own reading of its pass rate, at 95%, keeps the drop within the margin. What
  that reading feeds is a claim made only when every part holds: a candidate is no worse only where it holds on
  every asked aspect, and such an intersection-union claim errs at most α however many aspects it has, so correcting
  each part would only make the claim harder to reach without making it safer. The cases that would settle an
  unclear row are counted at the same level, so a row's count and its floor's are one count.

## Order and time

**Never let running order follow the lever; interleaving absorbs drift for free.** A provider's throughput and
load drift over hours, so a sweep run in lever order can credit the drift to the lever. The engine shuffles cell order
within a run, and interleaves a launch's arms by default, taking their cases in turn, which costs nothing and absorbs
that drift across every arm. Arms run apart are not a weaker experiment: measured minutes, hours or days apart, they
are one experiment while nothing that decides their result changed (owner, 2026-10-23), so no page discounts them. The
lens tools (`runs_compare`, comparison sets) state when two runs' spans did not overlap, as a fact.
*Evidence:* agent with tools, a 5-hour sweep run in lever order, 2026-08, Spearman ρ = 0.87 between running position and latency; retrieval sub-agent, 2026-09, throughput drifting 9–25 tok/s across an afternoon explained a ~205 s tail credited to a lever; replicated.

## Metrics that mislead

**Keep the pass^k conjunction small and reliable.** A trial passes only when every goal-state check passes
and every judged capability dimension reaches the pass threshold (`compute_pass_hat_k`): 3 of 5 unless the
host declares another for the behavior (`PassThreshold`, registered with its `BarRegistry`), and every pass^k
records the threshold it was computed at (`rubric_threshold`). So pass^k collapses to its weakest member, and
judge noise counts as candidate unreliability. Conjoin checks a competent candidate clears and dimensions
whose retest agreement you have measured. Ranking and diagnosis are two jobs.
*Evidence:* agent with tools, 2 campaigns, 2026-07 and 2026-08, one never-clearing dimension held pass^k to 0.0 or 0.2 for every arm, and a strict check set pinned it at 0 for every arm; in simulation (2026-09) a perfect candidate with five dimensions, each falsely failing 5% of the time, shows pass^3 ≈ 0.46.

**Pair quality with coverage and latency with a delivery rate, and read the floor.** An arm delivering
almost nothing has little to get wrong; one can be fast by declining the work. Rank latency on its mean, which
has an interval and a paired test at these sizes, and read the tail beside it. A tail cannot rank: a 95% interval
on a p95 has no upper end below 72 observations, and the engine gives a p95 only from 13 (median-unbiased,
Hyndman–Fan type 8). Below that it reports the slowest observation as `max`, the worst case seen, never as a
percentile.
*Evidence:* retrieval sub-agent, 2026-07, an arm delivering almost nothing scored grounding and honesty 5.0, and one with a 69 s p95 declined 60% of hard cases; agent with tools, 2026-08, an incumbent averaging 4.3 had a minimum of 1 against a challenger's 5; 2 campaigns. In simulation (2026-10), a "p95" read nearest-rank from 5 latencies, which is their maximum, fell below the true p95 77% of the time, and one read by linear interpolation still did 68% of the time at 30.

**Price per token is not cost.** Reasoning tokens bill at the output rate. Compare measured cost per
observation ([how a result's cost is counted](cost-and-budgets.md#how-a-results-cost-is-counted)).
*Evidence:* retrieval sub-agent, 2026-08, a model 25% cheaper per output token was 1.6× costlier and 1.8× slower per observation, single campaign.

**Rank on the measure at the lever's scope.** An end-to-end measure moves with everything, not only the lever.
Each measure carries an `attribution_scope` (`end_to_end` or `subsystem`). Report both and rank on the
subsystem's.
*Evidence:* agent with tools, 2026-07, a written analysis credited a sub-agent setting with halving end-to-end p95 (98.3 → 48.5 s) while the sub-agent's own p95 went from 36.6 to 37.6 s, and recommended an arm failing 13% against 0%, single campaign.

## Did the lever move

**Assert the lever moved before reading the outcome.** A lever that never took effect reads like one that
changed nothing. Declare what it acts on
([mechanism checks](reading-reports.md#did-a-lever-take-effect-mechanism-checks-and-observed-mechanisms)).
*Evidence:* retrieval sub-agent, 3 arms, 2026-09, a reasoning effort was sent on the wire and the model returned 0 reasoning tokens in every arm, single campaign.

**A reasoning effort is a label, not a bound. Bound reasoning rather than raising the cap.** Each vendor maps
the word to its own budget, and the engine names a confound when two arms' reasoning shares differ by
`REASONING_SHARE_DIVERGENCE` (0.20). A provider refusing to turn reasoning off (HTTP 400) is a permanent rig
result, not a low score; never retry it as transient.
*Evidence:* three subjects, 2026-08 to 2026-10, replicated. A classifier parsed 3 of 6 at a 32-token cap with reasoning on, 6 of 6 with it off, and 6 of 6 at a 512 cap for 10–90× the tokens. At effort "low" one model spent 0.28–0.48 of its completion reasoning and another 0.68–1.00. Turning reasoning off cut the agent's p95 time to first action from 163 s to 37 s on one model.

**In a before/after, change one thing at a time; a judge change voids it.** The engine never pools
observations across [apparatus classes](concepts.md#apparatus-class). A grid that varies several levers on
purpose is a different design ([Start from the question](#start-from-the-question)).
*Evidence:* agent with tools, 2026-07, a dimension rose 2.7 → 4.4 when a fix and a judge swap landed together, while the four dimensions judged identically moved between −0.3 and +0.5; ruled unanswerable, single campaign.

**Give every arm the same inputs by construction.** A key-wise overlay leaves keys it omits as they were. The
variation generator's default random source is unseeded, so subsets drawn per arm differ. Generate cases
once and run every arm over the stored cases.
*Evidence:* retrieval sub-agent, 2026-09, single campaign.

## Rig failures that score as candidate failures

Each defect below produced plausible numbers, some for months. A rig fault is excluded, never scored
([Rig failures](adopting-a-host.md#rig-failures-a-broken-rig-costs-one-cell-never-the-run)); the converse
matters as much: **a swept lever's own failure is the outcome.** Excluding a swept sub-model's
timeouts as harness failures dropped 10 of 15 cells and left pass^k at 1.0 over 2 cases. The engine charges a
candidate for a turn truncated at its output cap, and never excludes a trial a failed check already decided. A call
the provider declined (a 429 that is not a spent quota, or a 529 overloaded or 503 unavailable) is not the model's
failure but the provider declining to serve it, as it would any request, so it is excluded, never charged
([design](design.md#a-failure-is-charged-to-whoever-caused-it)); its lost trials are bounded in every verdict, and an
arm that loses more than a tenth of its trials to one cause is flagged.

- **The candidate was told something the seed contradicted.** A "you were just restarted" notice fired at
  every cell start, so for about four months quality numbers measured which model ignores a contradiction.
  Anything the candidate perceives must be seeded world.
- **Templates assumed a state the world did not show.** Six of seven described a system mid-task while the
  seed showed it idle; they measured the confound.
- **Tools were never started under eval.** Every call returned "not started", the judge scored grounding
  1/5, and a cassette recorded the failure as the tool's answer. When a probe confirms a predicted absence,
  demand one positive observation of the path running.
- **Muted actions succeeded where production refuses them** (2 of 199 failures). The fix: production's
  handlers on seeded production state, replacing only what leaves the process.
- **Goal checks counted refused calls.** One trial passed 7 of 7 checks with every call refused. The call
  ledger now records succeeded calls only.
- **An account limit failed candidate and judge alike,** and negative checks (`not any(...)`,
  `call_count(...) <= 1`) passed vacuously on empty conversations; 54 runs were void. An account refusal is
  now a rig failure, and a failed candidate fails every check.
- **The judge could not see results delivered asynchronously,** so it scored grounding 1 every time.
- **The eval measured a stub,** and the classifier scored 32.8% against a 33.3% chance floor. See
  [Feed the classifier exactly what production feeds it](building-an-inventory.md#feed-the-classifier-exactly-what-production-feeds-it).

*Evidence:* agent with tools and its classifier, 2026-05 to 2026-10, eight separate defects, each reproduced live.

**Prove a check can fail before trusting it passing.** A goal check giving the same verdict whether the
candidate acted or did nothing grades nothing. Authoring refuses a check that gives one verdict on its control end
state and on the untouched seed, and every run marks a check not so proven (`unproven`, `refuted`) beside its pass
rate; a quick world run also states how many cases a do-nothing candidate would pass
([the world model](evaluating-agents.md#the-rules-the-contract-enforces)). Calibrate any script that reads verdicts on a
known pass and a known fail, and classify on reported counts, not exit codes: a run that never happened and a run
that failed both exit non-zero.
*Evidence:* agent with tools, 2026-10, an audit found a hold check reading a parameter rather than its effect, and goal checks reading the static seed rather than the end state, single audit.

**Test failure paths, not more samples.** The defects that change verdicts live in the code deciding what an
incomplete run may contribute. The engine now discloses cells short of their intended repetitions, and gives
no cost band below 3 observations. Above that the band is read on the log scale, because costs are positive and
skewed, so with only a few past results it is wide on the high side.
*Evidence:* agent with tools, 3 rounds, 2026-08, defects found rose 4 → 7 → 10; the failure-path round found a budget-stopped run flipping a frontier ranking and a ±1.5% cost band from 2 observations missing the actual by 13%; single program. In simulation (2026-10), the normal-theory band from 5 lognormal costs (log-SD 1.0) covered a 15-observation sweep 82% of the time against its stated 95%; the log-scale band covers 97%.

## Judges

**Choose a judge by self-consistency first, then agreement with people, and check the people's labels can
beat a constant.** Lenient judges compress toward the ceiling, where a high exact retest is not reliability.
The engine labels judged readings with an
[evidence tier](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers).
*Evidence:* analysis writer, 6 judges × 5 memos × k=3 × 7 dimensions, 2026-09: two judges ordered memos repeatably (self-ρ 0.80, 0.86), four were lenient (self-ρ −0.12 to 0.39), and a constant 3 landed in band on 29 of 32 human labels. Agent with tools, 2026-10: judges agreed with themselves 66% and 83% exact, and the more consistent sided with the code check at a fifth of the cost. The same judge won on both subjects; replicated.

**Give the judge the evidence the candidate acted on, and let code grade what code can check.**
*Evidence:* retrieval sub-agent, 2026-07, the judge called a tool-grounded answer "fabrication" while scoring its grounding 5, and its "parse errors" vanished once it saw the trace, single campaign.

**Calibrating a judge mostly repairs the apparatus.** Render what the judge reads as a reader sees it. A rule
folded into a composite 1–5 scale is outweighed by everything else; give it a narrow pass/fail dimension.
*Evidence:* analysis writer, 5 frozen cases, 3 calibration rounds, 2026-09: the judge scored the layout's field names, then charged re-assembly artifacts to the memo, then truncated as a reasoning judge and was re-billed; single campaign.

## The analysis writer

**The writer model, not the prompt's form, limits groundedness, and judge noise limits what you can tune.**
Subtract rubric criteria before adding them. Blind human ratings are a separate measurement from the judge's.
*Evidence:* analysis writer, a latitude A/B with a pre-stated rule and a 2×2 form × input study, 2026-09: groundedness near 2 in every condition, the latitude arm failing its rule, and judge test–retest noise (0.42–0.46) exceeding every effect ranked; single campaign.

**Code renders every number, and the model's prose references them.** A deterministic gate over prose never
converges. The verdict line and every count on a report are code's; a recommendation names the findings it rests
on; a finding's title stating a null its finding established none of is the prompt's rule to prevent, not a phrase code refuses
([Having a model write the analysis](reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence)).
Never build a prompt's worked example from real output: it fossilises that output's errors.
*Evidence:* analysis writer, 2026-07 to 2026-09, 10 of 10 checkable claims in two production memos were false, a prose fact gate grew to 4,099 lines in 33 days and was deleted, and an example taken from a real memo had its false claim reproduced near-verbatim; single program.

**Runs are cheap; the memo is the bill.** A refused generation is billed and stores nothing. Price analyses
like runs ([Spend outside any run](cost-and-budgets.md#spend-outside-any-run)).
*Evidence:* three campaigns, ≈ $34 in total, 2026-09: over 99% of the classifier campaign's spend was memo generation and ≈ $9.4 bought refused or discarded generations; only the multi-turn agent campaign's runs dominated (≈ $11 of $17).

## How an eval program drifts

From an audit of one eval program's history (about 1,900 commits over six months to 2026-09, single program),
each mechanism with its counter-measure:

- **Every defect was answered with an addition** (2.74 lines inserted per line deleted). Name what each new
  mechanism replaces; delete at the third rework.
- **Locally right rules made every finding mandatory machinery.** Fix an apparatus defect now only if it would
  change the current decision or corrupt stored evidence, and cap review at two rounds.
- **The product was defined as prose,** the part that cannot converge. Code checks structure; an eval measures
  prose.
- **Generality was built for absent consumers:** the identity version moved 15 times and the bundle schema 24
  times in two months, each bump shortening stored evidence's life. Build for a consumer you can name.
- **"Trusted enough" had no test:** no product decision reached production in five weeks while about 150
  apparatus defects surfaced. Declare the decision, bar and inconclusive default before launch, and give each
  campaign spend and repair budgets.
