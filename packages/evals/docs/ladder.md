# The ladder: five examples, one idea each

**For** anyone who ran rung 0 (the TOML file on the [README](../README.md)) and wants the next step. **Answers:** how
to point an eval at your own function, grade with a judge you can check, compare two versions with a verdict a
reviewer can act on, grade an agent by the world it leaves, then serve the runs to your team, and evaluate the model
that writes your reports. Each rung is one directory under [`examples/`](../examples/), in its own domain. Rungs 1 to
4 are recorded on cheap models and rung 5 on Claude Sonnet, and every rung replays its recordings, so each runs
offline and free. Every block of Python below is a verbatim excerpt of the example it names, and every output shown
is what the recording replays (`tests/test_doc_blocks.py` holds both).

**Running a rung.** In a checkout, after `uv sync` at the repository root, from `packages/evals`:

```bash
uv run python -m threetears.evals examples/1-triage/triage.py
```

The command finds every eval the file defines at module level (an `Eval`, or rung 5's `WriterEval`), runs it, prints
it quietly, writes its report under `.evals/reports/<name>/`, keeps its runs in `.evals/evals.sqlite` (both under the
directory you run from), and exits with its code. The report is code's: built from the readings, with no model
called. A second door, at rung 3, has a model write one too. Paths a file names (its cases, its recordings) are read
beside the file. A recording that exists is replayed: no key, no network, no cost. `RECORD=1` with
`OPENROUTER_API_KEY` set records it again, live. After a pip install, fetch the examples first ([how](../examples/README.md#running-them)).

The ladder's line counts are of non-blank lines: rung 1's program is 15 of them, rung 4's agent and server together
stay under 100.

## Rung 1: your own function

[`examples/1-triage/`](../examples/1-triage/) triages 30 support emails, as rung 0 did, with your own function as the
runner. Twelve emails are lookalikes, the hard negatives: a phishing mail shouting about your account is `spam`, a
calm note that customer data is exposed is urgent.

```python
from typing import Literal

from pydantic import BaseModel

from threetears.evals import Eval, Model

model = Model("deepseek/deepseek-v4-flash", recording="recordings/triage.jsonl")


class Triage(BaseModel):
    category: Literal["billing", "bug", "account", "sales", "spam"]
    urgent: bool
    needs_human: bool


async def triage(case: dict) -> Triage:
    """Triage an inbound support email."""
    return await model.json(f"Triage this support email:\n\n{case['email']}", Triage)


def escalates(case: dict, out: Triage) -> bool:
    """Never leaves alone an email a person must see."""
    return out.needs_human or not case["needs_human"]


triage_eval = Eval("emails.jsonl", runner=triage, grader=["category", "urgent", escalates], k=3)
```

- **Three slots.** `Eval(cases, runner, grader)`: a cases file (or a list of dicts), any function from one case to an
  answer (async or not), and what counts as right. `k=3` plays each case three times, because a model can answer the
  same input differently from one call to the next.
- **`Model`** is the shipped client (`deepseek/deepseek-v4-flash` through OpenRouter here). `model.json` sends the
  pydantic schema as a strict JSON schema and validates the reply; a reply that fails validation is a miss whose
  reason is pydantic's message, never a silent default.
- **A field name is a grader.** `"category"` grades `out.category` against the case's `category`. The first field
  is read as a classifier: a confusion matrix, and each label's precision and recall with an interval counted over
  cases, not repeats. Any other field is a pass rate.
- **A function is a grader.** `escalates` fails only an email a person must see that the triage left alone. That
  rule is asymmetric, which a field name cannot say. Its name names the reading and its docstring describes it.
- **Reserved keys.** A case's `id` names it, its `stratum` (here `plain` or `lookalike`) splits every reading in the
  report, and neither is handed to the runner.

What it printed:

```text
emails: 18/30 cases passed; 12 misses: double-charge, refund-after-cancel, gdpr-delete, book-demo, phish-suspend, and 7 more
90 result(s): 90 scored
  urgent: mean 0.8333 (n=90, min 0, max 1)
  escalates: mean 0.9889 (n=90, min 0, max 1)
  category: mean 0.9 (n=90, min 0, max 1)
  per label:
    account: precision 1 (17/17 over 7 cases, 95% CI 0.5687-1), recall 0.8095 (17/21 over 7 cases, ...
    ...
  total spend: $0.00192, the results' cost
top misses (5 of 20):
  case double-charge (repeat ...): urgent gave 0
  case refund-after-cancel (repeat ...): urgent gave 0
  ...
stored in .evals/evals.sqlite, scope 'emails'
report: .evals/reports/emails/report.html
```

The misses print in the report's order, the order its Trials table lists the failed trials in. Which repeat of a
case missed can differ between replays: a recorded answer may serve another repeat of the same case.

**The lesson: the rule you write decides what you measured, so read the misses before the means.** Rung 0 graded
`needs_human` both ways and read 0.81; 16 of its 17 misses were emails escalated that did not need a person, a minute
of someone's time, not a customer left alone. `escalates` asks the question that matters and reads 0.99. Then read
the misses: a case missed on every repeat (rung 0 marks `refund-after-cancel` not urgent three times in three) is a
stable disagreement, so check the case's label before the model; a case missed on one repeat in three is unstable,
which is a finding of its own. A miss is the
runner's fault, the case's, or the grader's, and each has a different fix. The intervals are wide on purpose: 30
emails are enough to learn with, not to decide with ([how many cases](building-an-inventory.md#how-many-cases)).

## Rung 2: a judge, checked against a person

[`examples/2-replies/`](../examples/2-replies/) drafts a reply to each of 20 customer emails from a support policy
([`policy.md`](../examples/2-replies/policy.md)). No code can say whether a reply is right, so the grader is a `Judge`:
a model of another family (`google/gemini-2.5-flash-lite`) reading each reply against the policy.

```python
async def reply(case: dict) -> str:
    """Draft a reply to a customer's email that follows the support policy."""
    return await drafter.text(case["email"], system=f"Reply to this customer email for Acme Notes.\n\n{POLICY}")


def policy_and_email(case: dict) -> str:
    return f"{POLICY}\n\n# The customer's email\n{case['email']}"


late = {"email": "My annual plan renewed two months ago. Please refund the whole year."}
locked = {"email": "My account is locked. Can you check my password for me?"}
judge = Judge(
    judge_model,  # another family; Judge(drafter, ...) would print the self-judging warning under the scores
    {
        "grounded": "Every fact, price, deadline and promise in the reply is stated in the support policy.",
        "answers": "The reply answers what the customer asked, or says plainly what happens next.",
        "safe": "The reply never asks for a password or a full card number, and never grants what the policy refuses.",
    },
    examples=[  # hard negatives: each looks like the other of its pair
        Example("grounded", "Done: your year is refunded in full.", "fail", "no refund after 30 days", late),
        Example("grounded", "No refund after 30 days; I can stop the renewal.", "pass", "curt, but true", late),
        Example("safe", "Reply with your password and I'll check it.", "fail", "asks for a password", locked),
        Example("safe", "We never ask for passwords. It unlocks in 30 minutes.", "pass", "asks for nothing", locked),
    ],
    labels="labels.csv",  # not there until a person rates to_label.csv; until then the judge is uncalibrated
    case_material=policy_and_email,
)
replies_eval = Eval("emails.jsonl", runner=reply, grader=[judge], k=2, name="replies")
```

- **A judge is a grader.** It scores one rubric dimension per call, pass/fail by default (the scale people label most
  reliably), gives its reasoning before its score, and may answer "can't tell", which excludes that trial from that
  dimension rather than scoring it.
- **`case_material`** is what the judge reads beside each answer: here the policy and the email, the material the
  runner answered from.
- **What the judge reads of an answer that is not a string**: `json.dumps` of it, or its `repr` when JSON cannot
  hold it. A pydantic object is one JSON cannot hold, so a runner whose answer a judge reads returns
  `model_dump(mode="json")` of it, or builds a plain dict, and the judge reads the fields rather than a repr.
- **A judge of another family.** A model tends to rate its own output higher, so judging with the runner's own model
  prints the self-judging warning under the scores, a warning no quiet print drops.
- **`Example`s are hard negatives in pairs**: an answer that looks right and must fail, beside one that looks wrong
  and must pass. A run stores them as cases a judge campaign can measure the judge on, calling no model to do it.
- **`labels=`** names a person's labels. They are not there yet.

What it printed:

```text
replies: 20/20 cases passed; 0 misses
judge: 0 labels: judge uncalibrated (examples/2-replies/labels.csv is not there yet); next step: rate about 20 outputs of each dimension in the to_label.csv the report writes, which holds 40 per dimension, without looking at the judge's verdict
40 result(s): 40 scored, 0 excluded
  answer.grounded (judged pass/fail): mean 1 (n=40, min 1, max 1)
  answer.answers (judged pass/fail): mean 1 (n=40, min 1, max 1)
  answer.safe (judged pass/fail): mean 1 (n=40, min 1, max 1)
  total spend: $0.0168, the results' cost (candidate $0.00175, judge $0.0151)
stored in .evals/evals.sqlite, scope 'replies'
report: .evals/reports/replies/report.html
```

**The lesson: a judge is a model too, and a perfect score from an unchecked judge is a question, not an answer.** The
recorded judge passes all 40 replies on all three dimensions. Read a few and you will find claims the policy never
makes. The run says what is missing in a line it never quiets, and the rung ships with no person's labels on purpose:
the package provides the mechanism, and collecting labels is yours to do.

1. The report directory holds `to_label.csv`: every judged reply on every dimension, with the case it answered and the
   judge's verdict withheld ([the recorded one](../examples/2-replies/to_label.csv)).
2. Fill in `label` (`pass` or `fail`) and `rater` for about 20 rows of a dimension, without opening the report's
   scores, and save it beside `replies.py` as `labels.csv`.
3. Run again. `Judge(labels="labels.csv")` pairs each label with the judge's score of the same reply (by a
   fingerprint of what was read, so a replayed run pairs with the rows the recorded one wrote) and prints, per
   dimension, how many paired, κ with its lower bound, and the evidence tier it earns.

Twenty labels is the floor for the `calibrated` tier; about 50 is a realistic plan for a good judge. Reword a
dimension and its labels no longer pair, so settle the wording first. Until then every judged reading is flagged with
the tier it stands on, never hidden: [Judges and calibration](judges-and-calibration.md) goes deeper.

## Rung 3: two prompts, compared

[`examples/3-invoices/`](../examples/3-invoices/) reads 30 invoices in three strata (clean, multi-currency, noisy OCR)
with two prompts. The answer has a field per thing to extract:

```python
class Invoice(BaseModel):
    number: str
    currency: Literal["USD", "EUR", "GBP", "JPY", "CHF", "CAD"]
    due: str
    total: float | None


V1 = "Extract the invoice number, currency, due date (YYYY-MM-DD) and total due from this invoice:\n\n{text}"
V2 = V1 + (
    "\n\nThe currency is the one the total is charged in, never one quoted for reference. Read dates in the "
    "issuer's own order (09/03 is 9 March in Europe, 3 September in the US). Undo OCR slips (O for 0, l for 1). "
    "Leave total null unless the invoice prints the amount due: never add up the lines yourself."
)
```

and the comparison is a runner mapping:

```python
@grader(guard=0.02)  # a guardrail: v2 must not do worse here by more than 2 points, whatever else it gains
def no_invented_total(case: dict, out: Invoice) -> bool:
    """Never states a total the invoice does not print."""
    return case["total"] is not None or out.total is None


invoices = Eval(
    "invoices.jsonl",  # each case's stratum splits every reading by kind of invoice in the report
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},  # the first key is the control
    # What the comparison answers; the first grader (currency) is its measure, which the verdict and the page lead on.
    question="Does v2 read the currency an invoice charges in right more often than v1?",
    # A field each; a case's null total means none may be stated. A difference under 5 points on total is too small
    # to act on, so v2 can read "equivalent" there.
    grader=["currency", "number", "due", grader("total", margin=0.05), no_invented_total],
)

if __name__ == "__main__":
    result = asyncio.run(invoices.run())
    print(result)  # the verdict line, then each arm's readings; every delta and its interval is in the report
    result.write("report")  # report.html for the reviewer, beside report.md, campaign.json and the bundle
    raise SystemExit(4 if result.gate().failed else 0)  # in CI: a regression or a guardrail not shown held fails
```

- **A mapping is a comparison.** Each runner is an **arm**, run over the same cases, so each case is compared with
  itself across the arms. The first key is the **control**, usually what runs today, and every other arm is read
  against it on every reading: the difference (delta), its interval, and a verdict.
- **`question=`** is the one sentence the comparison answers, asked before any reading is seen. Its measure is the
  first grader, here `currency` (the classifier's accuracy): the printed first line and the page lead on it, and the
  page quotes the question under its title. Asked alone it is a superiority question ("does v2 do better?"), so no
  margin is filed and v2's pass rate reads shown better, shown worse or not separated, against zero. A question of
  "no worse than v1" writes its own margin, in the measure's units: `Eval(question=..., margin=0.05)` is 5 points.
  Leave the question out and the readings are leads: the page says "Exploratory comparison." once and leads on the
  same first grader.
- **`grader("total", margin=0.05)`** is the `total` field, graded with a margin: the largest difference too small to
  act on. Only with a margin can a reading come out **equivalent**, the one verdict that says "good enough". A case
  whose `total` is `null` expects none stated, so an invented total misses there too.
- **`grader(guard=0.02)`** makes a guardrail: something v2 must not do worse on, decided apart from everything else,
  so a gain elsewhere cannot pay for a loss here. Every grader is higher-is-better, so no direction is stated.
- **`result.write("report")`** writes the report for a reviewer (`report.html`, `report.md`, `campaign.json`, `bundle-<aspect>.json`), and
  `result.gate()` fails a CI step on a regression, a breach, or a guardrail not shown held.
- **Two levers at once?** The file's docstring names arms as `k=v` pairs (`"prompt=v2, effort=high"`), so each
  pair is a factor of the design, and `result.against(...)` reads the same runs against another control.

What the recording printed (made at `max_tokens=8192`), first lines, errors and the last line, which says why it
exits 4:

```text
invoices: Accuracy: v2 and v1 not separated; nothing else separated; guardrail No invented total score undecided for v2
v1 (control): 90 result(s): 90 scored
  number: mean 0.8333 (n=90, min 0, max 1)
  ...
v2: 90 result(s): 89 scored, 1 failed by the candidate
  number: mean 1 (n=89, min 1, max 1)
  ...
  error: case c02: the candidate raised ValidationError: 1 validation error for Invoice Invalid JSON: expected `:` at line 1 column 7 ...
...
exit 4: the gate failed on guardrail No invented total score, v2 vs v1: undecided: not shown held, so not known to be safe (the interval reaches both sides of -2 pts, the declared margin, so the arm is shown neither within it nor beyond it)
```

And what the report (`.evals/reports/invoices/report.md`, the same page as `report.html`) said: its first screen's
title, the question, the lead and the guardrail's warning beside it, the reading, the contrast row for `number`, the
guardrail's row, and the trials' census:

```text
# v1 vs v2 · classify (invoices)
"Does v2 read the currency an invoice charges in right more often than v1?"
**Accuracy: v2 and v1 not separated.**
**v2 is not yet shown safe on the no invented total guardrail on classify: v2 failed it on 1 of 30 cases and v1 on 1, a case counted once however many of its trials failed it, too few cases to show v2 within its 2-point margin or beyond it.**
| classify | v2 | 30 | 73 of 90 trials | 88 of 90 trials | 16.7 | [-1.5, 25.9] | — | not separated |
| Number score | 83.3% [66.8%, 92.5%] | 100% [88.1%, 100%] | 16.7 pts | [-7.8, 35.4] pts | v2 and v1 not separated | 30 paired |
| No invented total score | v2 | 98.9% | 98.9% | 30 paired | 0 pts | [-13.4, 13.4] pts | 2 pts | undecided: not shown held, so not known to be safe (the interval reaches both sides of -2 pts, the declared margin, so the arm is shown neither within it nor beyond it) |
180 trials ran. Of them 161 passed and 19 failed.
```

The page leads on the question's measure, the first grader's accuracy, as the printed line does, and says beside it
what leaves the guardrail undecided. Because the question was asked before the readings were seen, each reading is an
answer, set without ", a lead". v2's pass rate, 16.7 points above v1's, is not separated from it: its interval reaches
below zero.

**The lesson: a report that cannot decide says which question it could not answer, and what would answer it,
rather than declaring a winner.** v2 read every invoice number it answered where v1 missed six of the ten noisy ones,
and still v2 and v1 are not separated: on 30 invoices the corrected interval on that gain reaches below zero.
The guardrail is the question the first screen's warning names as unanswered: each arm stated a total once on a noisy invoice that prints
none, the interval on the difference is ±13 points against a margin of 2, and an undecided guardrail is never read as
held, so the gate fails and the run exits 4. What would answer it is a narrower interval on that guardrail, and only
an invoice that prints no amount due can fail it (7 of the 30 do): more of those hard-negative cases. One v2 trial
failed on malformed JSON from the model, and it counts against v2, stated, never excluded: a model's unusable answer
is the candidate's failure. [Reading reports](reading-reports.md#reading-a-comparison) explains each verdict, and
[Measuring soundly](measuring-soundly.md#how-many-cases-how-many-repeats) how many cases a question needs.

### Have a model write the analysis

Every page so far is code's. The harness door runs the same file, then hands the campaign to the report writer: an
analyst model that reads the frozen evidence through tools and writes the page, and an editor that reviews it before
it is stored.

```bash
ANTHROPIC_API_KEY=... uv run python -m threetears.evals.harness examples/3-invoices/extract.py --write-report claude-sonnet-5-5
```

| What | The harness door |
|---|---|
| Key | `ANTHROPIC_API_KEY`: the writer's every turn is live |
| Cost | at most `--budget-usd` a report, $3 unless you name less |
| Page | `.evals/reports/invoices/analysis.html`, beside code's `report.html` |
| Served | the campaign's analysis from then on: `report` and `report_read` show this page |
| Turns | recorded under `examples/3-invoices/recordings/report/` |

The writer's turns never replay here: each run of the file is a new campaign with a new snapshot, so no earlier
recording is of it, and with no key the door refuses, naming the file it would record. A writer replays offline only
over frozen campaigns, which is rung 5.

## Rung 4: an agent, graded on the world it leaves

[`examples/4-booking/`](../examples/4-booking/) books meetings. Each request seeds one day's calendar under its
reserved `world` key; the agent gets two tools; the graders read the day it left.

```python
class Day(BaseModel):
    busy: list[int] = Field(description="The hours already taken; 9 is 9:00 to 10:00.")
    booked: list[int] = Field([], description="The hours the agent booked.")


def find_free(day: Day) -> list[int]:
    """List the day's free hours, 9 to 16."""
    return [hour for hour in range(9, 17) if hour not in day.busy + day.booked]


def book(day: Day, hour: Annotated[int, Field(ge=9, le=16)]) -> str:
    """Book the one-hour meeting at an hour; the calendar does not stop a double booking."""
    day.booked.append(hour)
    return f"booked {hour}:00"

```

A tool is a plain function of the state and its typed parameters: its name, docstring and types are what the model is
shown, and a call that breaks its schema changes nothing and tells the agent why. The agent is your own loop:

```python
async def agent(case: dict, tools: Mapping) -> str:
    """Book the meeting a request asks for, or say why it cannot be booked."""
    said = f"Request: {case['ask']}"
    for _ in range(5):  # a loop that never says done cannot run up a bill
        turn = await model.json(said, Turn, system=SYSTEM)
        if turn.tool == "done":
            return turn.reply
        try:
            answer = await (tools["book"](hour=turn.hour) if turn.tool == "book" else tools["find_free"]())
        except ValueError as refused:  # a call the world refuses changes nothing, and the agent is told why
            answer = f"refused: {refused}"
        said += f"\n\n{turn.tool}(hour={turn.hour}) said: {answer}"
    return "ran out of turns"
```

```python
def in_window(case: dict, out) -> bool:
    """Every booking falls inside the hours the request asked for."""
    return all(case["start"] <= hour < case["end"] for hour in out.state.booked)


booking = Eval(
    "requests.jsonl",  # each case's "world" is the day it starts with
    runner=World(Day, tools=[find_free, book])(agent),  # the model's turns replay; tools that move state run live
    grader=[
        '(state.booked.length >= 1) == (variation.expect == "book")',  # booked exactly when a free hour fits
        "not intersects(state.booked, state.busy)",  # never over a taken hour: doing nothing passes this too
        'call_count("day.book") <= 1',  # and so does this: only the first check beats the did-nothing control
        in_window,
    ],
    name="booking",
)
```

- **`World(Day, tools=[...])(agent)`** is a runner: each cell gets a fresh world seeded from the case, and the agent
  is handed the tools bound to it. Its output, which function graders see, has `.answer`, `.state` and `.calls`.
- **A string grader is a goal check**: an expression over the end state (`state.booked`), the calls that succeeded
  (`call_count`) and the case's own parameters (`variation.expect`).
- **Every goal check must beat doing nothing.** The engine derives the do-nothing control (the seed untouched, no
  calls) and says, beside each check's pass rate, how many cases an agent that did nothing would pass.

What it printed:

```text
booking: 9/12 cases passed; 3 misses: r08, r09, r10
36 result(s): 36 scored
  in_window: mean 1 (n=36, min 1, max 1)
  goal check (state.booked.length >= 1) == (variation.expect == "book"): passed 32/36 — unproven: no control shows it tells acting from doing nothing; a candidate that did nothing passes it in 4 of 12 case(s)
  goal check not intersects(state.booked, state.busy): passed 32/36 — NOT A MEASUREMENT: a candidate that did nothing passes it in 12 of 12 case(s), so this pass rate does not beat doing nothing
  goal check call_count("day.book") <= 1: passed 35/36 — NOT A MEASUREMENT: a candidate that did nothing passes it in 12 of 12 case(s), so this pass rate does not beat doing nothing
```

**The lesson: grade what the agent did, and only with checks that can fail.** Most misses are the agent booking a
taken hour on the full-day hard negatives (`r09`, `r10`), where the right move is to say no: its reply could read
fine while the calendar is wrong. The second and third checks are guardrails worth keeping, but an agent that did
nothing passes them in every case, so their pass rates are not measurements of capability; only the first beats the
did-nothing control. [Evaluating a tool-using agent](evaluating-agents.md) covers proving a check with an authored
control, and declaring a production world of your own.

### Serve it to the team

An eval's runs are kept in its store, so anyone can read them back without running anything.
[`serve.py`](../examples/4-booking/serve.py) hands `as_host([booking], store=...)` to the MCP transport
(`mount_fastmcp`, the `[fastmcp]` extra) so an agent can list and read the runs, and the same host serves the
command line:

```bash
cd examples/4-booking  # serve.py imports agent.py as a module, so both run from here, where the runs are stored
uv run python -m threetears.evals agent.py
uv run python -m threetears.evals ls --host serve:host --scope booking
```

`report CAMPAIGN --host serve:host --scope booking` prints a campaign's page, and `report_read` over MCP reads the
same page: the report the harness door wrote, once it has written one, and code's page until then. The store is any
`DocumentStore`: the SQLite file by default, or your own database to share it.
[Serving](serving.md) covers the commands, the actions an agent calls, and the MCP transport.

## Rung 5: evaluate the evaluator

[`examples/5-writer/`](../examples/5-writer/) turns the eval on the report writer itself: the analyst model that reads
a campaign through tools and writes its page, and the editor that reviews the page before it is stored: the model
behind rung 3's second door. It gets the same treatment as anything else you ship: arms, cases, a judge, a verdict.
The arms are two versions of the analyst's skill, the Markdown it is told to work from:

```python
control = AnalystSkill.at("skills/control")  # frozen: recordings are keyed by the skill's digest
variant = (Path(__file__).parent / "skills" / "lead-first.md").read_text(encoding="utf-8")  # SKILL.md, lead claim first
lead_first = control.with_file("SKILL.md", variant)
judge = Model(  # another family; low effort leaves a score and a short reason room in 2,048 tokens, the priced cap
    "google/gemini-3.8-flash", temperature=0, max_tokens=2048, effort="low", recording="recordings/judge.jsonl"
)
```

and the eval is a `WriterEval` over a directory of frozen campaigns:

```python
writer_eval = WriterEval(
    "cases",  # eleven frozen campaigns: {name, group, stratum, withhold, snapshot}, one .json.gz each
    arms={  # the first is the control; the arms differ in the skill alone
        "shipped": Writer("claude-sonnet-5-5", skill=control, effort="medium"),
        "lead-first": Writer("claude-sonnet-5-5", skill=lead_first, effort="medium"),
    },
    judge=judge,
    recordings="recordings/turns",  # one file per arm and case, keyed by skill, model, effort and snapshot
    budget_usd=2.5,  # each report's cap
    turns=45,  # each report's turn cap, the analyst's and the editor's together
    max_cost_usd=39.0,  # the whole eval's, judge and pairwise pass included
    pairwise_share=0.05,  # of max_cost_usd, for the pairwise pass: $1.95; each arm gets half the rest, $18.52
    held_out_fraction=0.35,  # 7 held-out cases of 11: the 0.3 default leaves 4, under the 5 a rate needs
    name="writer",
)
```

- **A case is a campaign, frozen.** Each file in `cases/` holds one campaign's evidence exactly as the writer's tools
  read it: the four campaigns rungs 1 to 4 store when they replay, a larger campaign the package ships as a fixture
  (the `kairo` stratum), and six of the package's report inputs reshaped to fit no exemplar (the `misfit` stratum).
  Each case's `stratum` is its domain, so the readings split by it, and its `withhold` names the skill's exemplar
  notes the analyst is not given: the one reasoned on that same campaign.
- **Cases are split before anything runs.** A case's split group (related cases share one) lands on the dev side,
  where a change is tuned, or the held-out side, where it is decided. The promotion is read on held-out cases only.
- **`Writer(model, skill=, effort=)`** is an arm. A writer can also differ in its revisions or its editor's own model
  (`editor_model=`); each lever the arms differ in becomes an axis of the campaign's design.
- **The control is a copy, frozen.** A recording is keyed by the skill's digest, so a recording made over the skill
  the package ships today would go stale with its next edit. The rung measures the copy that was recorded.
- **The judge is another family** than the writers, as in rung 2, and the eval refuses one of the writers' own.

What the recording printed (made 2026-11-03), replayed from `packages/evals/` with no key: the arms, the first
readings and the judged ones, the judge's preference between the two arms' pages, the promotion read, and where the
turns came from (each path printed whole):

```text
writer: lead-first against shipped, 7 held-out case(s) x 1, judged by google/gemini-3.8-flash
| reading | shipped | lead-first |
|---|---|---|
| Report finished | 0.14 | 0.43 |
| Bounces | 0.86 | 1.00 |
...
| Must-fix findings open | 0.43 | 0.29 |
| Should-fix findings open | 2.43 | 1.57 |
...
| Writer turns | 12.57 | 12.29 |
...
| report_writer.first_screen (judged) | 5.00 | 5.00 |
| report_writer.coherent_story (judged) | 4.86 | 5.00 |
| report_writer.honest_emphasis (judged) | 5.00 | 5.00 |
| report_writer.sound_elision (judged) | 4.86 | 5.00 |
| report_writer.confounds_handled (judged) | 5.00 | 5.00 |
| report_writer.actionable_next_steps (judged) | 5.00 | 5.00 |
| report_writer.evidence_fits_claims (judged) | 4.43 | 4.86 |
...
| Candidate | Pairs | Reads | Share (%) | Difference (pts) | Interval, 95% (pts) |
|---|---|---|---|---|---|
| lead-first | every pair | not separated | 68 | 36 | -28 to 100 |
| lead-first | both pages finished | untested | — | — | — |

| Candidate | Order consistency | First-slot lean (pts) | Can't tell | Held-out cases to settle | Spent |
|---|---|---|---|---|---|
| lead-first | 86% (6 of 7) | 7 | 0 | 39 | $0.1404 of $1.95 |
...
promotion of lead-first: Undecided: the primary reading does not separate lead-first from shipped, and only an improved primary reading promotes.
shipped: writer turns replayed from .../examples/5-writer/recordings/turns/shipped-4e14e276b568-claude-sonnet-5-5-medium-r2
lead-first: writer turns replayed from .../examples/5-writer/recordings/turns/lead-first-af11aac84890-claude-sonnet-5-5-medium-r2
```

A report finished is one the editor cleared, so it is stored finished rather than as a draft. The report
(`.evals/reports/writer/report.md`) opens with the answer, the promotion reading and the judge's preference in one
sentence, and the same two side-by-side tables head its Scorecard:

```text
**Undecided, so shipped stays: across 7 held-out cases the judge gave lead-first 68% of its preference over shipped (50% is even), too few cases to tell from an even split; about 39 cases would settle it.**
```

Further down it reads the two code measures that moved most:

```text
| Report finished | 14.3% [2.6%, 51.3%] | 42.9% [15.8%, 75%] | 28.6 pts | [-74.6, 100] pts | lead-first and shipped not separated | 7 paired |
| Must-fix findings open | 0.4286 [0, 0.9229] | 0.2857 [0, 0.9848] | -0.1429 | [-1.313, 1.027] | lead-first and shipped not separated | 7 paired |
```

| What | Recorded |
|---|---|
| Reports | 14: 2 arms x 7 held-out cases, once each |
| Writer turns | about 12.4 a report |
| Writer | $6.90, about $0.49 a report |
| Judge | $0.66, 112 calls: 98 scores and the pass's 14 |
| Total | $7.56, under the $39 `max_cost_usd` |

[`recordings/README.md`](../examples/5-writer/recordings/README.md) holds the one command that records it again, what
it may spend, and what to do when a recording run is cut short.

**The lesson: a judge that scores one page at a time sits at the ceiling here, so the reading that decides is the
judge's preference between two pages.** Every judged item reads 5 or near it for both arms: the writer revises each page
until the editor passes it on the same rubric, so the scores bunch at the top and cannot tell the arms apart. Read side
by side, the same judge gave lead-first 68% of its preference (50% is even), and its two orders agreed on 6 of the 7
cases. Two code measures moved the same way: 3 of 7 lead-first reports finished against 1 of 7, with fewer must-fix
findings left open (0.29 a report against 0.43). None of it separates. That preference is the promotion read's primary
reading, fixed before the run: it reads `promote` only when the preference favours the candidate on enough held-out
cases, no gated reading regressed and every guardrail held. 7 held-out cases, each written once (k=1), separate only a
large effect, and the pass gives the size that would settle this one: 39 held-out cases. So the read is `Undecided`,
which never promotes, and recording the call stays a person's act. That is the answer at this size, stated rather than
tuned around: what would answer it is more cases or more repeats of each, and each costs a recorded report. [Pairwise
preference](judges-and-calibration.md#pairwise-preference-for-the-report-writer) covers the share and the promotion
rule, and [Measuring soundly](measuring-soundly.md#how-many-cases-how-many-repeats) how many cases a question needs.

## Where next

| To | Read |
|---|---|
| decide what to test before writing cases, and how many | [Building an inventory](building-an-inventory.md) |
| trust a judge, and measure it as a subject | [Judges and calibration](judges-and-calibration.md) |
| read every part of a report | [Reading reports](reading-reports.md) |
| have a model write a comparison's report, and read it | [The analyst's report](reading-reports.md#the-analysts-report-written-over-frozen-evidence) |
| promote a change to the report writer | [Pairwise preference for the report writer](judges-and-calibration.md#pairwise-preference-for-the-report-writer) |
| choose a design, k and a case count, and avoid known traps | [Measuring soundly](measuring-soundly.md) |
| price a run before it spends | [Cost and budgets](cost-and-budgets.md) |
| evaluate an agent in your production world | [Evaluating a tool-using agent](evaluating-agents.md) |
| keep runs in your own database and launch from your app | [Adopting the engine](adopting-a-host.md) |
