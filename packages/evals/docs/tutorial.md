# Tutorial: your first eval, end to end

**For** anyone new to 3tears-evals, or new to evals. **Answers:** how to write cases, run them, read what went
wrong, compare two versions, read the verdict, grade with an LLM judge, and hold a line no version may cross. It runs offline and costs nothing:
small scripted stand-ins play the model. Every term is defined where it first appears,
and [Concepts](concepts.md) is the glossary.

The running example is a support-ticket router. It reads a ticket and picks one queue: `billing`, `bug`,
`account` or `other`.

**Setup.** Install the package (`pip install 3tears-evals`, Python 3.14+), or run `uv sync` in a checkout of this
repo. Build one file, `triage.py`, as you go. Step 1 starts it. Each later step adds code below what is there and
replaces the `main()` function and the `asyncio.run(main())` line at the bottom. Run it with
`python triage.py` (`uv run python triage.py` in the checkout).

## 1. Write the cases

A **case** is one input plus what a good answer looks like. Here, a ticket and the queue it belongs in. The
**candidate** is what you are evaluating: any async function that takes one case and returns an answer.

```python
import asyncio
import json
from types import SimpleNamespace

from threetears.evals.quick import Judge, compare, run_eval

CASES = [
    {"id": "double-charge", "ticket": "I was charged twice for March.", "queue": "billing"},
    {"id": "refund", "ticket": "Please refund the plan I bought yesterday.", "queue": "billing"},
    {"id": "invoice", "ticket": "Where do I download last month's invoice?", "queue": "billing"},
    {"id": "export", "ticket": "The export button does nothing.", "queue": "bug"},
    {"id": "crash", "ticket": "The app crashes when I upload a large photo.", "queue": "bug"},
    {"id": "reset-email", "ticket": "The password reset email never arrives.", "queue": "account"},
    {"id": "change-email", "ticket": "How do I change the email on my account?", "queue": "account"},
    {"id": "locked", "ticket": "I'm locked out after too many login attempts.", "queue": "account"},
    {"id": "hours", "ticket": "What are your support hours?", "queue": "other"},
    # A hard case on purpose: it names a crash, but the customer lost money, and our ruling is that money wins.
    {"id": "charged-crash", "ticket": "The app crashed during checkout and I was charged anyway.", "queue": "billing"},
]


async def route_v1(case: dict) -> str:
    """Route a support ticket to billing, bug, account or other."""
    # In your code, this is where the prompt and the model call go. A keyword script stands in here.
    text = case["ticket"].lower()
    if "crash" in text or "button" in text:
        return "bug"
    if "charged" in text or "refund" in text or "invoice" in text:
        return "billing"
    return "other"
```

A case is a plain dict. Its `id` names it in everything you read later; without one, a case is named by its
position. Nothing is graded against `queue` until step 2 says it holds the right answer.

Ten cases is enough to learn with, not to decide with. A set worth trusting is built mostly from hard cases like
`charged-crash`: boundaries between labels, lookalikes, and pairs that differ in one detail.
[Designing a classifier eval set](designing-classifier-evals.md) says which cases to write and how many.

## 2. Run it

`run_eval` plays the candidate on every case `k` times. **k** is the number of repeats per case. A model can
answer the same input differently from one call to the next, so one play per case hides that variance.

```python
async def main() -> None:
    summary = await run_eval(CASES, route_v1, expected=lambda case: case["queue"], k=2)
    print(summary.render())


asyncio.run(main())
```

```
run 01a12445-eaee-73db-abe3-97da2b88bbb7 completed: route_v1 over 10 case(s) x k=2
  20 result(s): 20 scored
  match: mean 0.6 (n=20, min 0, max 1)
  confusion_cell: n=20, counted in the confusion matrix below
  confusion (expected → predicted):
    account → other: 6
    billing → billing: 6
    billing → bug: 2
    bug → bug: 4
    other → other: 2
  per label:
    account: precision none (n=0), recall 0 (0/6 over 3 cases, 95% CI 0-0.82), f1 none
    billing: precision 1 (6/6 over 3 cases, 95% CI 0.18-1), recall 0.75 (6/8 over 4 cases, 95% CI 0.19-0.97), f1 0.857
    bug: precision 0.667 (4/6 over 3 cases, 95% CI 0.081-0.98), recall 1 (4/4 over 2 cases, 95% CI 0.018-1), f1 0.8
    other: precision 0.25 (2/8 over 4 cases, 95% CI 0.026-0.81), recall 1 (2/2 over 1 case), f1 0.4
```

How to read it:

- **`expected=`** makes this a classifier eval: code compares each answer with the case's label. That is the
  special case. Most answers can't be checked by code, and step 6 grades those with a judge. A **scorer** is the
  other code grade: any `(case, answer) -> bool | float` function, passed in a list after the candidate
  ([`rung_zero.py`](../examples/rung_zero.py) has one).
- **A result** is one play of one case. **Scored** results were graded. When there are any, the line also counts
  results **failed by the candidate** (your function raised; they count against it) and **excluded** ones (a
  fault of the measuring setup, such as a scorer that raised; they count for nothing).
- **`match`** is 1 when the answer was the expected label. Its mean, 0.6, is the accuracy.
- **The confusion matrix** shows which labels were mistaken for which. Every `account` ticket went to `other`.
- **Per label**, precision is how often a predicted label was right, and recall is how often a true label was
  found. Each has a 95% interval: the range the true rate plausibly lies in. The intervals count cases, not
  repeats ("over 3 cases"), because repeating a case does not make it a new case. A rate from a single case has
  no interval at all, and ten cases give wide ones.
- **Where the run is stored**: in memory, for as long as the call. With a host of your own, `scope_id=` names
  where its runs are kept and found again ([Adopting the engine](adopting-a-host.md)).

## 3. Read your misses

This is the most important habit in evals. An accuracy figure tells you how much went wrong. Only the misses
tell you what went wrong, and whether the eval itself is right. Read every miss before you read any number
again.

A **miss** is a result the candidate got wrong: a wrong label, a scorer that gave 0, a failed check, a failed
pass/fail judgement, or a result the candidate failed outright (it raised, say). `summary.misses()` returns
them, each saying why. `summary.results()` returns every result, misses or not.

```python
async def main() -> None:
    summary = await run_eval(CASES, route_v1, expected=lambda case: case["queue"], k=2)
    for miss in summary.misses():
        print(f"{miss.case} (repeat {miss.repeat}): {'; '.join(miss.missed_because)}")
    excluded = [result for result in summary.results() if result.outcome == "excluded"]
    print(f"{len(excluded)} result(s) excluded")


asyncio.run(main())
```

```
reset-email (repeat 1): answered 'other', expected 'account'
reset-email (repeat 2): answered 'other', expected 'account'
change-email (repeat 1): answered 'other', expected 'account'
change-email (repeat 2): answered 'other', expected 'account'
locked (repeat 1): answered 'other', expected 'account'
locked (repeat 2): answered 'other', expected 'account'
charged-crash (repeat 1): answered 'bug', expected 'billing'
charged-crash (repeat 2): answered 'bug', expected 'billing'
0 result(s) excluded
```

For each miss, decide whose fault it is. There are three answers, and each needs a different fix:

1. **The candidate is wrong.** The three `account` tickets fail because `route_v1` has no rule for that queue.
   This is the finding you ran the eval for. Fix the candidate, then run again.
2. **The case is wrong.** Is `charged-crash` really `billing`? It names a crash. The label is right only because
   of a ruling, money wins, which the comment in step 1 records. Without one, ask whoever owns the labels. If
   they disagree, change the case, not the candidate. Either way, write the ruling down beside the case.
3. **The grading is wrong.** A scorer that is too strict, or an expected label with a trailing space, makes
   right answers read as misses. Fix the grader.

Look at the repeats too. This stand-in always answers the same way, so each miss appears on both repeats. A real
model often misses a case on one repeat and not the other. That case is unstable, which is a finding in itself.

Excluded results are not misses. They say nothing about the candidate. When they are not zero, read them in
`summary.results()`, where each one's `errors` says why.

## 4. Compare two versions

`route_v2` adds the missing queue and checks for money before crashes. Add it below `route_v1`:

```python
async def route_v2(case: dict) -> str:
    """Route a support ticket to billing, bug, account or other."""
    text = case["ticket"].lower()
    if "charged" in text or "refund" in text or "invoice" in text:
        return "billing"
    if "crash" in text or "button" in text:
        return "bug"
    if "email" in text or "password" in text or "login" in text:
        return "account"
    return "other"


async def main() -> None:
    result = await compare(
        CASES,
        {"v1": route_v1, "v2": route_v2},
        expected=lambda case: case["queue"],
        control="v1",
        k=2,
    )
    for row in result.contrasts("accuracy"):
        print(f"{row['contrast']} against {row['control']}: delta {row['delta']:+.2f}, interval {row['interval']}, "
              f"p {row['p_adjusted']:.3f}, {row['verdict']}")
    print(f"v2 misses: {len(result.misses('v2'))}")


asyncio.run(main())
```

```
candidate=v2 against candidate=v1: delta +0.40, interval [0.03059, 0.7694] at 95%, p 0.037, improved on the control
v2 misses: 0
```

Each candidate becomes an **arm**: one version, run over every case. `compare` runs every arm over the same cases,
so each case is compared with itself across the arms. The **control** is the arm the others are tested
against: usually what runs in production today. The report names each arm by its key, as `candidate=v2`. The
runs land together as one **campaign**, the set of runs you want compared.

`result.render()` prints the whole report as Markdown. `result.contrasts()` gives its "Contrasts against the
control" table as rows you can read in code. `result.misses("v2")` works as in step 3: a better number is no
reason to stop reading misses.

## 5. Read the verdict

Each row of the contrasts table compares one arm with the control on one reading:

- **Delta** is the arm's mean minus the control's: here, 40 points more accuracy.
- **The interval on delta** is where the true difference plausibly lies. When several rows are tested together,
  each interval is widened so that all of them hold together, 95% of the time.
- **p (Holm-adjusted)** is the p-value corrected for every row tested alongside it. Use this p, never a raw one.
- **The verdict** says what the evidence supports:

| Verdict | Means | What to do |
|---|---|---|
| separated: printed **improved on the control** or **regressed from the control** | the adjusted p is below 0.05 | act on it, unless the row says *immaterial*: real, but smaller than a margin you declared |
| **not separated from the control** | the cases could not tell the arms apart | add cases, above all hard ones. It never means "no difference" |
| **equivalent to the control** | a test showed the difference is inside a margin you declared | treat the arms as interchangeable on this reading. Only this verdict says "good enough", and it needs a margin: see below |
| **untested** | no test could decide, for example with fewer than two cases on a side | fix what the row names, usually too few cases |

In code, branch on `row["outcome"]` (`improved`, `regressed`, `equivalent`, `not_separated`, `untested`), never
on the words. `result.gate()` turns the verdicts into a CI check that fails on a regression or a guardrail not
shown held, and `python -m threetears.evals gate` does the same from a pipeline
([The command line](command-line.md#gate)).

Now try a smaller change. `route_v1_login` fixes only the locked-out ticket. Add it, and compare all three:

```python
async def route_v1_login(case: dict) -> str:
    """Route a support ticket to billing, bug, account or other."""
    if "login" in case["ticket"].lower():
        return "account"
    return await route_v1(case)


async def main() -> None:
    result = await compare(
        CASES,
        {"v1": route_v1, "v1-login": route_v1_login, "v2": route_v2},
        expected=lambda case: case["queue"],
        control="v1",
        k=2,
    )
    for row in result.contrasts("accuracy"):
        print(f"{row['arm']}: delta {row['delta']:+.2f}, interval {row['interval']}, "
              f"p {row['p_adjusted']:.3f}, {row['verdict']}")


asyncio.run(main())
```

```
v2: delta +0.40, interval [-0.03846, 0.8385] at 97.5%, p 0.074, not separated from the control
v1-login: delta +0.10, interval [-0.1685, 0.3685] at 97.5%, p 0.343, not separated from the control
```

Two lessons here:

- **`v1-login` is not separated.** It got one more case right out of ten. That may be a real improvement, but ten
  cases cannot show it. "Not separated" claims nothing either way.
- **`v2` lost its separation without changing.** Adding an arm put two tests in one family, so each interval
  widened (to 97.5%) and each p was corrected for two. Every arm you add costs every other arm some power.
  Compare only the arms that answer your question.

To show a cheaper or simpler version is **good enough**, you need `equivalent`, and that needs a **margin**: the
largest difference too small to matter. Declare it with `margins=`, by measure: on accuracy, it is recorded on
every arm's run; on a scorer of your own, by the scorer's name. Replace `main()`:

```python
async def main() -> None:
    result = await compare(
        CASES,
        {"v1": route_v1, "v1-login": route_v1_login},
        expected=lambda case: case["queue"],
        control="v1",
        k=2,
        margins={"accuracy": 0.25},
    )
    for row in result.contrasts("accuracy"):
        print(f"{row['arm']}: delta {row['delta']:+.2f}, interval {row['interval']}, {row['verdict']}")


asyncio.run(main())
```

```
v1-login: delta +0.10, interval [-0.1262, 0.3262] at 95%, not separated from the control — immaterial: the observed delta is inside the margin of ±0.25 on Accuracy, which does not show the true difference is that small
```

Still not separated, and not equivalent either: ten cases cannot show a pass rate within 0.25, even when the
arms agree on every case. That takes at least 12, and more when they disagree on some
([how many](reading-reports.md#reading-a-comparison)). Without a margin, the report says so in a line above
its contrasts table.

[Reading reports](reading-reports.md#reading-a-comparison) explains each column, the tests behind them, and the
rest of the report.

## 6. Add a judge

Most answers have no label to compare against. Is this reply polite? Does it say only what the policy says? A
**judge** is a model that reads each answer and scores it against a **rubric**: one written description per
quality you care about, each scored 1 to 5 by default.

Give `run_eval` a `Judge`: a completion client, the model it calls, the rubric, and the material each answer is
judged against. The client below is an offline stand-in that looks for an apology. A real one calls your model
provider; [`llm_judge.py`](../examples/llm_judge.py) has a small adapter for the `anthropic` SDK.

```python
async def draft_reply(case: dict) -> str:
    """Draft the first reply to a support ticket."""
    if case["queue"] == "other":
        return "Our support hours are 9 to 5, Monday to Friday."
    return "Sorry about that. We've passed your ticket to the right team."


async def stand_in_judge(*, system: str, user: str, response_format: dict | None = None) -> SimpleNamespace:
    """An offline stand-in for the judge's model. It reads the prompt the engine sends, as a model would."""
    score = 5 if "sorry" in user.lower() else 2
    reply = {"reasoning": "stand-in: looks for an apology", "criteria_scores": {"answer.acknowledges": score}}
    return SimpleNamespace(
        content=json.dumps(reply), input_tokens=None, output_tokens=None, reasoning_tokens=None, cost_usd=0.0,
        price_source="stand-in", model="stand-in", served_model=None, stop_reason="end_turn", temperature=0.0,
    )


async def main() -> None:
    judge = Judge(
        client=SimpleNamespace(generate=stand_in_judge),
        model="stand-in",
        rubric={"acknowledges": "The reply acknowledges the customer's problem before anything else."},
        case_material=lambda case: f"Ticket: {case['ticket']}",
    )
    summary = await run_eval(CASES, draft_reply, judge=judge, k=2)
    print(summary.render())
    for result in summary.results():
        for grade in result.judged:
            if grade.score <= 2:
                print(f"{result.case}: {grade.dimension} {grade.score}, because {grade.reasoning}")


asyncio.run(main())
```

```
run 01a12446-2d38-77e1-8185-6b938c406abb completed: draft_reply over 10 case(s) x k=2
  20 result(s): 20 scored, 0 excluded
  spend cap: uncapped — no spend ceiling was in force (cost enforcement was off)
  intent (from draft_reply's docstring): Draft the first reply to a support ticket.
  answer.acknowledges (judged 1-5): mean 4.7 (n=20, min 2, max 5)
  judge spend: $0 over 20 call(s)
hours: answer.acknowledges 2, because stand-in: looks for an apology
hours: answer.acknowledges 2, because stand-in: looks for an apology
```

- **The intent** is what the judge is told each case asks. Here it came from the candidate's docstring, and the
  summary says so. Pass `intent=` to state it outright: its wording can move the scores.
- **With a judge in play** the summary also counts results excluded for a judge's fault, states the spend cap
  (none here; `max_cost_usd=` sets one) and what the judge spent.
- **`answer.acknowledges`** is the rubric's dimension, placed under the judge's default context, `answer`.
- **The judge reads every field of the case** beside the material, `queue` included. Keep out of the case
  anything the judge should not see.
- **Read the low scores the way you read misses.** The stand-in marked down `hours`, and a model reading this
  rubric would have reason to as well: the ticket asks a question and reports no problem, so there is nothing to
  acknowledge. The rubric is at fault, not the reply. Reword it ("acknowledges the customer's problem, if there
  is one").
- `judge=` works on `compare` too, and code grading (`expected=`, scorers) can run beside it.

A judge's scores are only as good as the judge. Use a different model from the candidate's, and check it
against people's ratings before you lean on it: [Judges and calibration](judges-and-calibration.md).

## 7. Hold a guardrail

Some things a reply must never do, however much better it gets elsewhere: here, promise a refund, which only
the billing team may grant. A **guardrail** is a reading no arm may get worse on. It is kept out of the contrasts,
so a gain there cannot pay for a loss here, and each arm is decided against the control on it alone.

Add two new drafts, a scorer for the rule, and declare the scorer a guardrail with `Guardrail`. Add `Guardrail`
to the `threetears.evals.quick` import at the top:

```python
async def draft_reply_v2(case: dict) -> str:
    """Draft the first reply to a support ticket."""
    if case["queue"] in ("billing", "bug"):
        return "So sorry about that! We'll refund you, and we've passed your ticket to the right team."
    return await draft_reply(case)


async def draft_reply_v3(case: dict) -> str:
    """Draft the first reply to a support ticket."""
    if case["queue"] in ("billing", "bug"):
        return "So sorry about that! We've passed your ticket to the right team, who will be in touch today."
    return await draft_reply(case)


def promises_nothing(case: dict, reply: str) -> bool:
    """The reply makes no promise of a refund, which only the billing team may make."""
    return "refund" not in reply.lower()


async def main() -> None:
    result = await compare(
        CASES,
        {"v1": draft_reply, "v2": draft_reply_v2, "v3": draft_reply_v3},
        [promises_nothing],
        guardrails={"promises_nothing": Guardrail(margin=0.1, direction="higher_is_better")},
        control="v1",
        k=2,
    )
    for row in result.guardrails():
        print(f"{row['arm']}: {row['control_mean']:.2f} -> {row['arm_mean']:.2f}, interval {row['interval']}, "
              f"{row['outcome']}")


asyncio.run(main())
```

```
v2: 1.00 -> 0.40, interval [-0.9694, -0.2306] at 95%, breached
v3: 1.00 -> 1.00, interval [-0.3085, 0.3085] at 95% (bounded: every case moved alike), undecided
```

- **The margin** (`margin=0.1`) is how much worse than the control you would tolerate: here, a promise in one
  more reply in ten. **The direction** says which way is better. You declare both; neither is assumed.
- **Breached**: the whole interval is beyond the margin. `v2` is not adopted, whatever else it gained, and the
  report says so.
- **Undecided**: `v3` made no promise, and still is not shown safe. Ten cases cannot show that a promise in one
  reply in ten would not appear, so the interval reaches past the margin. Undecided is never read as held. At
  a margin of 0.1 it takes about 40 cases, none of them with a promise, to read **held**.

`result.render()` prints the full "Guardrails against the control" table, and `result.guardrail_standing("v2")`
says what one arm breached, is undecided on, and held. A judged rubric dimension can be a guardrail too: name it
in `guardrails=`. [Reading the guardrails](reading-reports.md#reading-the-guardrails) has the details.

## 8. Where next

| To | Read | Example |
|---|---|---|
| run against a real model | [Running the examples](../examples/README.md#running-them) | [`llm_judge.py`](../examples/llm_judge.py) |
| hold a guardrail across a real change, to held | [Reading the guardrails](reading-reports.md#reading-the-guardrails) | [`guardrails.py`](../examples/guardrails.py) |
| weigh accuracy against cost | [Cost and budgets](cost-and-budgets.md) | [`compare_two_models.py`](../examples/compare_two_models.py) |
| vary several things at once, such as prompt and model | [Choosing a campaign design](choosing-a-design.md) | [`prompt_x_model.py`](../examples/prompt_x_model.py) |
| evaluate an agent by what it does, not what it says | [Evaluating a tool-using agent](evaluating-agents.md) | [`world.py`](../examples/world.py) |
| write reports to files | [Reading reports](reading-reports.md) | [`reports.py`](../examples/reports.py) |
| build a classifier set worth trusting | [Designing a classifier eval set](designing-classifier-evals.md) | |
| keep runs in your own storage, over weeks | [Adopting the engine](adopting-a-host.md) | |

## Exploring, or declaring a design

Everything above was **exploratory**: you compared arms without saying in advance what you wanted to learn, and
the report says so. Every finding in it is a lead, not a confirmed answer. A campaign may be exploratory because
you are still figuring out how to do evals, or because you have an intuition and want to test some things
without a clear goal. Both are good reasons, and exploring needs no planning.

Declaring a design pays off once you will act on the answer. `compare` already declares the axes (the levers
you varied, and their levels) and the control for you; on a host of your own you declare those too, so the
report can check that each lever moved as you meant. Two more things are always yours to declare, before the
runs:

- **the question**: what you set out to learn, and which merit axes (`quality`, `cost`, `latency`,
  `reliability`) it is about. Only the readings on those axes are corrected together, so the correction is not
  spread over readings you never cared about. A reading outside them is still reported, as a lead;
- **the bar**: the level an arm must reach to be good enough. Each bar then reads **cleared**, **missed** or
  **undecided**.

A good path is to explore until you have a hunch, then run a fresh campaign that declares it. A pattern found by
looking at results is confirmed only by evidence gathered after you named it.
[Choosing a campaign design](choosing-a-design.md) shows how to declare each part.
