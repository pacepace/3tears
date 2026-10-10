# Choosing a campaign design

**For** anyone about to spend on a campaign: you have a question, and need to choose which arms to run, how many
cases and repeats, and how to declare it. **Answers:** which design answers which question, and what it cannot
answer; how to declare the design and its control; how cases and repeats trade off for each question; and which
read gives the answer. Arm, lever, variant, k and campaign are as [Concepts](concepts.md) defines them.
[Measuring soundly](measuring-soundly.md) has the evidence behind the sizing advice.

Three more terms:

- **Control:** the arm every other arm is tested against. It is a variant (one configuration), not a run.
- **Shape:** what the engine derives from what ran (`bundle.design.shape`). `one_factor_at_a_time` means each
  arm moves at most one lever off the control. `multi_factor` means some arm moves more than one.
  `undesignated` means no control resolved, so nothing is known to be controlled.
- **pass^k:** the chance that k attempts at a case all pass. Only goal-state checks and judged dimensions
  decide a pass; guardrails take no part.

## Start from the question

| Your question | Run these arms | Shape | Read the answer from | It cannot tell you |
|---|---|---|---|---|
| Is B better than A? | A and B, differing in one lever; A is the control | `one_factor_at_a_time` | [the contrasts against the control](reading-reports.md#reading-a-comparison) | why B differs, or whether another level would do better |
| Which of several single changes helps? | the control, plus one arm per change, each moving one lever | `one_factor_at_a_time` | the contrasts, Holm-corrected across the arms | how the changes combine |
| Which combination wins? | every combination of the levers' levels (a full grid) | `multi_factor` | the contrasts, read again against other controls (`Comparison.against`) | which lever made a combination's gain, except where the grid holds that lever's two levels with all else equal |
| Which arm is cheapest among those good enough? | arms that differ in cost (model, effort), all run at one k, under one subject | any | the [frontier](#the-frontier-passk-against-cost): pass^k, cost and latency against a bar | whether two arms differ on a measure; that is a contrast |

**"One lever per arm" is a rule for the first two questions, not a law.** It is what makes each contrast
attributable to its lever. In a grid, arms that move several levers are the point. A lever's effect is clean
only between two arms that differ in that lever alone. A 2×2 grid holds two such pairs for each lever: the
report tests the lever at the control's level of the other lever, and `Comparison.against` tests it at the
other level. The engine does not test the interaction itself. When those two effects differ, the combination
matters more than either lever. Each `against` is its own Holm family. The bundle's row for each lever
(`bundle.coverage`) lists in `confounded_by` every other lever that varied across its runs. While that list
is non-empty, the row is a comparison averaged over the other levers, not a controlled one.
[`examples/prompt_x_model.py`](../examples/prompt_x_model.py) runs a 2×2 and reads both effects.

**The winner of a grid needs a test against the runner-up.** The contrasts test every combination against
the control, never against each other. To test the leader against the rest, call `against(leader)`. You chose
that control after seeing the results, so treat what it shows as a lead to confirm in a fresh campaign. An
analysis can draw a `sweep_ranking` chart, which orders the combinations by their mean and tests nothing. The
decision surface's row order is not a ranking either.

## Declare the design and its control

**On the quick path, `compare` declares it for you.** It sets one axis per factor, with the levels the arms
ran, and the control from `control=`. It sets `intended_repetitions` to `k`, and states the stimulus held
fixed and the rig commissioned. It launches every arm together, so time and load do not differ between them.

```python
from threetears.evals.analysis import inspect_campaign_bundle
from threetears.evals.quick import compare

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

comparison = await compare(
    CASES, {"v1": classifier("v1"), "v2": classifier("v2")}, control="v1",
    expected=lambda case: case["queue"], scope_id="triage", k=3,
)
design = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle.design
print(design.shape, [arm.moved for arm in design.contrasts])  # one_factor_at_a_time [{'candidate': 'v2'}]
```

**On your own host, declare it when you create the campaign.** Call `create_campaign` (the `campaign_create`
action takes the same fields). Pass `declared_design`, and name the control by one of its runs with
`control_from_run_id`. The engine resolves that run's variant key, so you never type a digest. To change the
control later, call `set_campaign_control`. A design that names an axis your host cannot vary, or a bar looser
than the registered one, is refused when the campaign is created. So create it before you launch, with no runs,
then add each run with `add_runs_to_campaign` and set the control with `set_campaign_control`; a run can be
named as the control before it has any results. This declares the runs above by hand:

```python
from threetears.evals.analysis import create_campaign

runs = {arm: summary.run_id for arm, summary in comparison.arms.items()}
levels = [{"content": arm, "display": arm} for arm in runs]
campaign = create_campaign(
    comparison.host.storage,
    {
        "name": "triage v2", "subject_id": "triage", "behavior": "routing", "run_ids": list(runs.values()),
        "declared_design": {
            "axes": [{"axis_id": "candidate", "values": levels}],
            "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"},
            "intended_repetitions": 3,
            "questions": [{"text": "Does v2 route more tickets right?", "merit_axes": ["quality"]}],
        },
    },
    control_from_run_id=runs["v1"], scope_id="triage", created_by="you", profile=comparison.host.profile,
)
```

| `declared_design` field | What it does | Needed for |
|---|---|---|
| `axes` (required) | The levers and levels you meant to compare. Each declared axis gets a row in `bundle.coverage`, `unswept` if it held one level; compare the row's `levels` with yours. | every question |
| `held_fixed` (required) | Whether the cases held the stimulus fixed, and whether the rig was commissioned or found. | every question |
| `control` | The reference variant, set from `control_from_run_id` or `set_campaign_control`. With none, the shape is `undesignated` and nothing is tested against a control. | A/B, single changes, grid |
| `intended_repetitions` | The repeats per case you meant each cell to get. A cell that falls short is named in `bundle.short_cells`. | the frontier, and any cell you want checked |
| `questions` | What you set out to learn. Their `merit_axes` set the Holm families, and a reading no question covers is [exploratory](reading-reports.md#readings-no-question-asked-about-exploratory). | a confirmatory answer |
| `bars`, `merit_priority` | Thresholds tighter than the host's, and the tie-break order between merit axes. | a bar verdict, a tie-break |

**Declare a design when you know the question; explore when you do not.** The design is optional. Declare
one when the question is settled before the runs: you know which lever you are comparing, against which
control, and what answer would change what you do. Only a declared question can confirm anything, so a
decision you will act on needs one. Leave it out when you are learning how to run evals, or testing an
intuition you cannot yet put as a question. A campaign with no design is **exploratory**: its report and its
bundle say so once, at the top, and its readings confirm nothing. Its analysis reads the design inferred from
the runs, and never calls that design declared. Do not declare a design you do not have just to get past this:
a made-up question makes leads read as answers, which is worse than declaring none. When an exploratory
campaign finds something worth knowing, declare a design for the campaign that tests it. A design that names
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
[Variance, k and how many cases](measuring-soundly.md#variance-k-and-how-many-cases) gives the numbers. A grid
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
from threetears.evals.quick import Answer
from threetears.evals.run import list_runs

def agent(cost, skips):  # an offline stand-in that leaves the light alone in the rooms named in `skips`
    async def act(case, room):
        seen = await room.view()
        wanted = "on" if seen["daylight"] == "dark" else "off"
        if seen["light"] != wanted and case["id"] not in skips:
            await room["switch_light"](to=wanted)
        return Answer("Done.", model="m", cost_usd=cost)
    return act

ROOMS = [{"id": f"{light}-{day}-{i}", "light": light, "daylight": day}
         for light in ("on", "off") for day in ("dark", "bright") for i in range(5)]
rooms = await compare(
    ROOMS, {("m", "high"): agent(0.002, set()), ("m", "low"): agent(0.0005, {"off-dark-0", "off-dark-1"})},
    factors=("model", "effort"), control=("m", "high"), world=ROOM, scope_id="lights", k=3,
    seed=lambda case: {"light": case["light"], "daylight": case["daylight"]},
    goal_checks=['(state.light == "on") == (state.daylight == "dark")'],
)
read = frontier(rooms.host.storage, "lights", list_runs=partial(list_runs, rooms.host), bar=0.8)
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
  every arm of one `compare` runs under the comparison's subject (its name), so its models and its named arms
  rank on one frontier. On your own host, run every arm under the subject it measures.
- **pass^k needs goal checks or a judge.** An arm graded by neither, such as a classifier scored only by
  `expected=`, has no pass^k: `pass_hat_k` is `None`, `n_pass_no_criterion` counts its attempts, and no bar
  is decided on it. Read such an arm from the contrasts.

The same frontier is the `scope_frontier` action and the `frontier` command
([The command line](command-line.md#frontier)); `scope_frontier` in `threetears.evals.ops` returns it as a
`FrontierResult`.

The bundle carries a frontier too (`bundle.frontier`). Check `bundle.frontier_bar_withheld` before reading
its clearing counts.

## Where to go next

- [Reading reports](reading-reports.md): what the contrasts, guardrails and decision surface say.
- [Measuring soundly](measuring-soundly.md): the noise floor, order effects, and the traps behind these rules.
- [Cost and budgets](cost-and-budgets.md): what a design will cost before it runs.
- [Adopting the engine](adopting-a-host.md): a host of your own, so a campaign can declare any lever.
