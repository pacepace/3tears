# The ladder: four examples, one idea each

**For** anyone who ran rung 0 (the TOML file on the [README](../README.md)) and wants the next step. **Answers:** how
to point an eval at your own function, grade with a judge you can check, compare two versions with a verdict a
reviewer can act on, and grade an agent by the world it leaves, then serve the runs to your team. Each rung is one
directory under [`examples/`](../examples/), in its own domain, recorded on cheap models, so every rung runs offline
and free. Every block of Python below is a verbatim excerpt of the example it names (`tests/test_doc_blocks.py`
holds it so), and every output is what the recording replays.

**Running a rung.** In a checkout, after `uv sync` at the repository root, from `packages/evals`:

```bash
uv run python -m threetears.evals examples/1-triage/triage.py
```

The command finds every `Eval` the file defines at module level, runs it, prints it quietly, writes its report under
`.evals/reports/<name>/`, keeps its runs in `.evals/evals.sqlite` (both under the directory you run from), and exits
with its code. Paths a file names (its cases, its recordings) are read beside the file. A recording that exists is
replayed: no key, no network, no cost. `RECORD=1` with `OPENROUTER_API_KEY` set records it again, live. After a pip
install, fetch the examples first ([how](../examples/README.md#running-them)).

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
    account: precision 1 (17/17 over 7 cases, 95% CI 0.5687-1), recall 0.8095 (17/21 over 7 cases, ...)
    ...
  total spend: $0.00192, the results' cost
top misses (5 of 20):
  case double-charge (repeat 2): urgent gave 0
  case refund-after-cancel (repeat 1): urgent gave 0
  ...
stored in .evals/evals.sqlite, scope 'emails'
report: .evals/reports/emails/report.html
```

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
    return await writer.text(case["email"], system=f"Reply to this customer email for Acme Notes.\n\n{POLICY}")


def policy_and_email(case: dict) -> str:
    return f"{POLICY}\n\n# The customer's email\n{case['email']}"


late = {"email": "My annual plan renewed two months ago. Please refund the whole year."}
locked = {"email": "My account is locked. Can you check my password for me?"}
judge = Judge(
    judge_model,  # another family; Judge(writer, ...) would print the self-judging warning under the scores
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
- **A judge of another family.** A model tends to rate its own output higher, so judging with the runner's own model
  prints the self-judging warning under the scores, a warning no quiet print drops.
- **`Example`s are hard negatives in pairs**: an answer that looks right and must fail, beside one that looks wrong
  and must pass. A run stores them as cases a judge campaign can measure the judge on, calling no model to do it.
- **`labels=`** names a person's labels. They are not there yet.

What it printed:

```text
replies: 20/20 cases passed; 0 misses
judge: 0 labels: judge uncalibrated (examples/2-replies/labels.csv is not there yet); next step: rate about 20 outputs in the to_label.csv the report writes, without looking at the judge's verdict
40 result(s): 40 scored, 0 excluded
  answer.grounded (judged pass/fail): mean 1 (n=40, min 1, max 1)
  answer.answers (judged pass/fail): mean 1 (n=40, min 1, max 1)
  answer.safe (judged pass/fail): mean 1 (n=40, min 1, max 1)
  total spend: $0.0168, the results' cost
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
    # A field each; a case's null total means none may be stated. A difference under 5 points on total is too small
    # to act on, so v2 can read "equivalent" there.
    grader=["currency", "number", "due", grader("total", margin=0.05), no_invented_total],
)

if __name__ == "__main__":
    result = asyncio.run(invoices.run())
    print(result)  # the verdict line, then each arm's readings; every delta and its interval is in the report
    result.write("report")  # report.html for the reviewer, beside report.md and bundle.json
    raise SystemExit(4 if result.gate().failed else 0)  # in CI: a regression or a guardrail not shown held fails
```

- **A mapping is a comparison.** Each runner is an **arm**, run over the same cases, so each case is compared with
  itself across the arms. The first key is the **control**, usually what runs today, and every other arm is read
  against it on every reading: the difference (delta), its interval, and a verdict.
- **`grader("total", margin=0.05)`** is the `total` field, graded with a margin: the largest difference too small to
  act on. Only with a margin can a reading come out **equivalent**, the one verdict that says "good enough". A case
  whose `total` is `null` expects none stated, so an invented total misses there too.
- **`grader(guard=0.02)`** makes a guardrail: something v2 must not do worse on, decided apart from everything else,
  so a gain elsewhere cannot pay for a loss here. Every grader is higher-is-better, so no direction is stated.
- **`result.write("report")`** writes the report for a reviewer (`report.html`, `report.md`, `bundle.json`), and
  `result.gate()` fails a CI step on a regression, a breach, or a guardrail not shown held.
- **Two levers at once?** The file's docstring names arms as `k=v` pairs (`"prompt=v2, effort=high"`), so each
  pair is a factor of the design, and `result.against(...)` reads the same runs against another control.

What the recording printed (made at `max_tokens=8192`), first lines and errors; it exits 4:

```text
invoices: no arm separated from the control 'v1'
v1 (control): 90 result(s): 90 scored
  number: mean 0.8333 (n=90, min 0, max 1)
  ...
v2: 90 result(s): 89 scored, 1 failed by the candidate
  number: mean 1 (n=89, min 1, max 1)
  ...
  error: case c02: the candidate raised ValidationError: 1 validation error for Invoice
  Invalid JSON: expected `:` at line 1 column 7 [type=json_invalid, input_value='{"{  "currency": "GBP", ...7",  "total": 270.00  }', input_type=str]
```

And what the report (`.evals/reports/invoices/report.md`, the same page as `report.html`) said: its first screen,
the contrast row for `number`, the guardrail's row, and the trials' census:

```text
# invoices: does v2 do better than v1?
v1 (control) vs v2 · 15 Oct 2026 · 30 cases × 3 repeats
**Exploratory: no difference found between v2 and v1 on any reading. Not yet shown safe: v2 on no invented total.**
| Number score | 83.3% [66.8%, 92.5%] | 100% [88.1%, 100%] | 16.7 pts | [-1.4, 34.7] pts | not separated from the control | 30 paired |
| No invented total score | v2 | 98.9% | 98.9% | 30 paired | 0 pts | [-13.4, 13.4] pts | 2 pts | undecided: not shown held, so not known to be safe (the interval on the difference, [-0.1343, 0.1343], reaches both sides of -0.02 (the declared margin), so the arm is shown neither within it nor beyond it) |
180 trials ran. Of them 179 passed and 1 failed.
```

**The lesson: a report that cannot decide says which question it could not answer, and what would answer it,
rather than declaring a winner.** v2 read every invoice number it answered where v1 missed six of the ten noisy ones,
and still no arm separated: on 30 invoices the corrected interval on that gain reaches just below zero. The guardrail
is the question the first screen names as unanswered: each arm stated a total once on a noisy invoice that prints
none, the interval on the difference is ±13 points against a margin of 2, and an undecided guardrail is never read as
held, so the gate fails and the run exits 4. What would answer it is a narrower interval on that guardrail, and only
an invoice that prints no amount due can fail it (7 of the 30 do): more of those hard-negative cases. One v2 trial
failed on malformed JSON from the model, and it counts against v2, stated, never excluded: a model's unusable answer
is the candidate's failure. [Reading reports](reading-reports.md#reading-a-comparison) explains each verdict, and
[Measuring soundly](measuring-soundly.md#how-many-cases-how-many-repeats) how many cases a question needs.

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

The store is any `DocumentStore`: the SQLite file by default, or your own database to share it.
[Serving](serving.md) covers the commands, the actions an agent calls, and the MCP transport.

## Where next

| To | Read |
|---|---|
| decide what to test before writing cases, and how many | [Building an inventory](building-an-inventory.md) |
| trust a judge, and measure it as a subject | [Judges and calibration](judges-and-calibration.md) |
| read every part of a report | [Reading reports](reading-reports.md) |
| choose a design, k and a case count, and avoid known traps | [Measuring soundly](measuring-soundly.md) |
| price a run before it spends | [Cost and budgets](cost-and-budgets.md) |
| evaluate an agent in your production world | [Evaluating a tool-using agent](evaluating-agents.md) |
| keep runs in your own database and launch from your app | [Adopting the engine](adopting-a-host.md) |
