# The ladder: five examples, one idea a file

**For** anyone who ran rung 0 (the TOML file on the [README](../README.md)) and wants the next step. **Answers:** how
to point an eval at your own function, grade with a judge you can check, compare two versions with a verdict a
reviewer can act on, grade an agent by the world it leaves, then serve the runs to your team, and evaluate the model
that writes your reports. Each rung is one directory under [`examples/`](../examples/), in its own domain, and each
file in it one idea: rung 3's six steps are 3a to 3f, rung 4's two are 4a and 4b. Rungs 1 to
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
called. A second door, at rung 3f, has a model write one too. Paths a file names (its cases, its recordings) are read
beside the file. A recording that exists is replayed: no key, no network, no cost. A call it does not hold is never
sent live: it fails the run, which exits 1, and the print opens with that failure, naming the call, and no verdict;
the readings after it cover only the results recorded before it failed. Any run that did not finish (cancelled,
stopped by its budget) opens the same way, in the print and on its report. `RECORD=1` with
`OPENROUTER_API_KEY` set records it again, live. After a pip install, fetch the examples first ([how](../examples/README.md#running-them)).

The ladder's line counts are of non-blank lines: rung 1's program is 15 of them, each rung-3 file stays under 30 (its
CI step under 10), and rung 4's agent stays under 80 and its server under 20.

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
emails: triage passed 70 of 90 trials on 30 cases. Cases that failed a trial: double-charge, refund-after-cancel, gdpr-delete, book-demo, phish-suspend, and 7 more.
90 result(s): 90 scored
  Urgent score: 83%, 75 of 90 trials
  Escalates score: 99%, 89 of 90 trials
  Accuracy: 90%, 81 of 90 trials
  per label:
    account: precision 100% (17 of 17 trials over 7 cases, 57 to 100%), recall 81% (17 of 21 trials over 7 cases, ...
    ...
  total spend: $0.0019, the results' cost
top misses (5 of 20 failed trials):
  case double-charge (repeat ...): Urgent score is 0
  case refund-after-cancel (repeat ...): Urgent score is 0
  ...
stored in .evals/evals.sqlite, scope 'emails'
report: .evals/reports/emails/report.html
```

The misses print in the report's order, the order its Trials table lists the failed trials in. Which repeat of a
case missed is the same on every replay: each repeat is served the answer it was recorded with.

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
CRITERIA = {
    "grounded": "Nothing in the reply contradicts the support policy, and it invents no specific fact the policy does "
    "not support (a price, a plan, a deadline, an account state). Where the policy says nothing, that is no failure.",
    "answers": "The reply answers what the customer asked, or says plainly what happens next.",
    "safe": "The reply never asks for a password or a full card number, and never grants what the policy refuses.",
}
late = {"email": "My annual plan renewed two months ago. Please refund the whole year."}
locked = {"email": "My account is locked. Can you check my password for me?"}
EXAMPLES = [  # graded in the run: each looks like the other of its pair, and must get the other verdict
    Example("grounded", "Done: your year is refunded in full.", "fail", "no refund after 30 days", late),
    Example("grounded", "No refund after 30 days; I can stop the renewal.", "pass", "curt, but true", late),
    Example("safe", "Reply with your password and I'll check it.", "fail", "asks for a password", locked),
    Example("safe", "We never ask for passwords. It unlocks in 30 minutes.", "pass", "asks for nothing", locked),
]
judge = Judge(
    judge_model,  # another family; Judge(drafter, ...) would print the self-judging warning under the scores
    CRITERIA,
    examples=EXAMPLES,
    labels="labels.csv",  # read blind on adjudication/blind_read_as_read.html; adjudicated; re-checked 2026-11-17
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
  and must pass. The run grades each one twice (`k=2`), through the same call it grades the replies with, and says
  how many the judge got right; a miss would print under it.
- **`grounded` is consistent, not exhaustive.** The policy was never meant to be the whole of the business: a reply
  fails only if it contradicts the policy or invents a specific fact it does not support. Silence is no failure.
- **`labels=`** names the reference the judge is checked against, described next.

**The reference.** Two readers read the same 20 replies blind, under an earlier, exhaustive wording of `grounded`
("Every fact, price, deadline and promise in the reply is stated in the support policy."): Brooks, a person, and
`claude-fable-5-2`, a model. They disagreed on 13 of the 60 reply and dimension pairs. The person reader adjudicated
those 13 blind to who said what, under the wording above: 9 decided (7 for the person's reading, 2 for the model's) and
4 unclear. On 2026-11-17 the owner re-checked three `grounded` items under this wording and re-ruled two: the
locked-out reply ("I've unlocked your account now") is a fail, reversing its adjudicated pass, since it asserts an
unlock the policy allows only after an email check nothing shows happened; the search-bug reply, which both readers
failed under the earlier wording, is left out as borderline; the annual-late reply keeps its pass. `labels.csv` holds
55 labels and leaves out 5, each said on its row: the 4 unclear and the borderline one. The reference is adjudicated by
the person reader; no independent adjudicator. [`adjudication/`](../examples/2-replies/adjudication/) holds both reads,
the rulings and the re-checks.

**The re-checks were chosen after the judges were read.** The three re-checks were picked after both judges' verdicts
were seen, and two of them, locked-out and annual-late, were put to the owner because the stronger judge had failed
those replies (the cheaper judge failed locked-out too). One re-ruling, locked-out, moved the reference toward both
judges. So on those items the reference is not independent of the judges: re-reading only replies a judge failed and
the reference passed can raise agreement with that judge and cannot lower it, and here it raised it. The `grounded` rows
as they stood before the re-checks, beside the rows after (the full table is under the print below):

| Judge | Dimension | Reference: failed / read | Judge failed of those the reference failed | Judge failed of those it passed |
|---|---|---|---|---|
| `gemini-2.5-flash-lite`, before | grounded | 1 / 16 | 0 of 1 (0 to 97%) | 1 of 15 (0 to 32%) |
| `gemini-2.5-flash-lite`, after | grounded | 2 / 16 | 1 of 2 (1 to 99%) | 0 of 14 (0 to 23%) |
| `claude-sonnet-5-5`, before | grounded | 1 / 16 | 1 of 1 (3 to 100%) | 5 of 15 (12 to 62%) |
| `claude-sonnet-5-5`, after | grounded | 2 / 16 | 2 of 2 (16 to 100%) | 4 of 14 (8 to 58%) |

`answers` and `safe` were not re-checked, and their rows are the same before and after.

**Passes carried to the new wording.** 9 of the 14 `grounded` passes were read by both readers only under the earlier
wording, and `labels.csv` binds them to this wording's fingerprint. That carry-over is sound one way only: a reply that
states nothing the policy does not (the earlier wording) contradicts nothing and invents nothing (this one), so a pass
under the stricter wording is a pass under the more lenient one. A fail does not carry, so the one `grounded` reply both
readers failed was read again under this wording rather than carried. The labels pair by the reply they were given on
and the wording they answer, so the person's earlier read, as saved, does not pair on `grounded` at all. They were read
on an earlier draw, whose page is
[`adjudication/blind_read_as_read.html`](../examples/2-replies/adjudication/blind_read_as_read.html); the run's own
`blind_read.html` draws the other repeat on 10 of the 20 cases, so the labels print as not drawn at random.

What it printed:

```text
replies: reply passed 39 of 40 trials on 20 cases. Cases that failed a trial: locked-out.
judge: 55 label(s) in labels.csv by Brooks (adjudicated by the person reader; no independent adjudicator; 3 grounded items re-checked after the judges were read) are not on this run's draw (the answers changed since, or it is an older sheet): kept, each paired with the same output, and not drawn at random
  answer answers (judged)   20 results from 20 cases: the judge caught 0 of the 1 the reader failed (exact 95% interval 0 to 97%); the judge failed 0 of the 19 the reader passed (exact 95% interval 0 to 18%); the judge gave every answer one verdict, pass
  answer grounded (judged)  16 results from 16 cases: the judge caught 1 of the 2 the reader failed (exact 95% interval 1 to 99%); the judge failed 0 of the 14 the reader passed (exact 95% interval 0 to 23%)
  answer safe (judged)      19 results from 19 cases: the reader failed no answer: no catch rate; the judge failed 0 of the 19 the reader passed (exact 95% interval 0 to 18%); the reader gave every answer one verdict, pass; the judge gave every answer one verdict, pass
judge: 4 of 4 examples graded as they must be on every one of 2 repeats ($0.00090 over 8 call(s))
40 result(s): 40 scored, 0 excluded
  answer grounded (judged), scored pass/fail: 97.5%, 39 of 40 trials
  answer answers (judged), scored pass/fail: 100%, 40 of 40 trials
  answer safe (judged), scored pass/fail: 100%, 40 of 40 trials
  total spend: $0.016, the results' cost (candidate $0.0017, judge $0.014)
```

Each line is counts (owner ruling, #1208): of the replies the reference failed, how many the judge caught by failing
them too (the catch rate), and of those it passed, how many the judge failed (the false-fail rate), each with its exact
95% interval.

**One tier up.** `stronger_eval` judges the same recorded replies on the same words, examples and reference with
`claude-sonnet-5-5`, sent to Anthropic with no temperature:

```python
stronger = Judge(
    Model("claude-sonnet-5-5", provider="anthropic", recording="recordings/judge_stronger.jsonl"),
    CRITERIA,
    examples=EXAMPLES,
    labels="labels.csv",
    case_material=policy_and_email,
)
stronger_eval = Eval("emails.jsonl", runner=reply_again, grader=[stronger], k=2, name="replies-stronger")
```

```text
replies-stronger: reply_again passed 30 of 40 trials on 20 cases. Cases that failed a trial: monthly-refund, annual-late-refund, annual-early-refund, locked-out, send-password, and 5 more.
...
  answer answers (judged)   20 results from 20 cases: the judge caught 0 of the 1 the reader failed (exact 95% interval 0 to 97%); the judge failed 3 of the 19 the reader passed (exact 95% interval 3 to 40%)
  answer grounded (judged)  16 results from 16 cases: the judge caught 2 of the 2 the reader failed (exact 95% interval 16 to 100%); the judge failed 4 of the 14 the reader passed (exact 95% interval 8 to 58%)
...
```

| Judge | Dimension | Reference: failed / read | Judge failed of those the reference failed | Judge failed of those it passed |
|---|---|---|---|---|
| `gemini-2.5-flash-lite` | grounded | 2 / 16 | 1 of 2 (1 to 99%) | 0 of 14 (0 to 23%) |
| | answers | 1 / 20 | 0 of 1 (0 to 97%) | 0 of 19 (0 to 18%) |
| | safe | 0 / 19 | no catch rate | 0 of 19 (0 to 18%) |
| `claude-sonnet-5-5` | grounded | 2 / 16 | 2 of 2 (16 to 100%) | 4 of 14 (8 to 58%) |
| | answers | 1 / 20 | 0 of 1 (0 to 97%) | 3 of 19 (3 to 40%) |
| | safe | 0 / 19 | no catch rate | 0 of 19 (0 to 18%) |

**What these numbers cannot show.** The reference failed two replies on `grounded`, one on `answers` and none on
`safe`, so no catch rate here says how often either judge catches a bad reply: two catches of two read 16% to 100%,
one of two 1% to 99%. The cheaper judge failed one of the two `grounded` failures and nothing the reference passed;
the stronger one failed both, and 4 of the 14 the reference passed. The two judges' counts are read on the same replies
but not tested against each other, and all are read on the 20 replies the reference was built from, once each, against
one person's adjudication. Neither judge is shown to catch the reference's failures. The rubric's wording was not
tuned toward either number, but the reference was revisited after the judges' verdicts were read, as said above.

**Your own judge needs your own read.** The reference pairs with these recorded replies and these words only:
re-record the replies or reword a dimension and it does not pair.

1. Run your eval. Its report directory holds `blind_read.html`, one page that opens from disk
   ([the recorded one](../examples/2-replies/blind_read.html)): the replies one at a time in a random order, each with
   the email it answers, and no judge verdict.
2. Read blind, from the top: answer each case, stop when you like, press **Download labels.csv**, and save the file
   where `Judge(labels=...)` names. What you answered from the top is a random sample of the run's replies, however
   many that is; skip ahead and the sample is not drawn at random, which still counts and is said.
3. Run again. The print shows how far you read and, per dimension, the counts: how many of the replies you failed the
   judge failed, and how many of those you passed it failed, each with its exact interval. A `reviewer=` model's read
   prints beside yours, apart, never pooled.

A catch rate needs failures to count: read enough cases that you fail some. Nothing on the page hides the judge's
scores from you: they are in the report beside it, so reading blind is your discipline, not the package's lock. A
model's read is paid for; a file store keeps it, and `store=":memory:"` pays for it again on every run. [Judges and
calibration](judges-and-calibration.md) goes deeper.

## Rung 3a: two prompts, compared

[`examples/3-invoices/`](../examples/3-invoices/) reads 30 invoices in three strata (clean, multi-currency, noisy OCR)
with two prompts, one idea a file: 3a compares them, 3b adds a margin, 3c a guardrail, 3d gates CI on it, 3e crosses
two levers, and 3f has a model write the analysis. Every file reads into one answer, a field per thing to extract, from
[`invoice.py`](../examples/3-invoices/invoice.py), which declares no eval:

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

and [`compare.py`](../examples/3-invoices/compare.py) is the comparison, a runner mapping:

```python
compare = Eval(
    "invoices.jsonl",  # each case's stratum splits every reading by kind of invoice in the report
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},  # the first key is the control
    # What the comparison answers; the first grader (currency) is its measure, which the verdict and the page lead on.
    question="Does v2 read the currency an invoice charges in right more often than v1?",
    grader=["currency", "number", "due", "total"],  # a field each; a case's null total means none may be stated
    name="compare",
)
```

- **A mapping is a comparison.** Each runner is an **arm**, run over the same cases, so each case is compared with
  itself across the arms. The first key is the **control**, usually what runs today, and every other arm is read
  against it on every reading: the difference (delta), its interval, and a verdict.
- **`question=`** is the one sentence the comparison answers, asked before any reading is seen. Its measure is the
  first grader, here `currency` (the classifier's accuracy): the printed first line and the page lead on it, and the
  page quotes the question under its title. Asked alone it is a superiority question ("does v2 do better?"): the
  question is decided on that one reading, with the bet staked toward "better", and every other reading is shown with
  its change, its interval and its evidence and decides nothing. A question of "no worse than v1" writes its own
  margin, in the measure's units:
  `Eval(question=..., margin=0.05)` is 5 points. Leave the question out and the readings are leads: the page says
  "Exploratory comparison." once and leads on the same first grader.
- **`name=`** names the comparison: run it again under the same name, into the same store and scope, and it continues
  (only what it has not recorded runs); a new name starts a fresh one.
- **Each eval gets its own `Model`.** A replay serves each recorded call once per model, so two evals in one process
  never share one: `compare.py`, `margin.py` and `guardrail.py` each build one over the same recording, `grid.py`
  builds two (one per effort, each over its own recording), `ci.py` runs `guardrail.py`'s eval on its model, and
  `invoice.py` holds no eval and builds none.

What it printed (made at `max_tokens=8192`), first lines and errors:

```text
compare: Accuracy: v2 behind v1, undecided. Over 30 paired cases, v2 is 1 point behind v1. The interval allows v2 to be as much as 14 points worse or 12 better. The evidence that v2 is worse is weak (1.2). Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2.
v1 (control): 90 result(s): 90 scored
  Number score: 83%, 75 of 90 trials
  ...
v2: 90 result(s): 89 scored, 1 failed by the candidate
  Number score: 100%, 89 of 89 trials
  ...
  error: case c02: the candidate raised ValidationError: 1 validation error for Invoice Invalid JSON: expected `:` at line 1 column 7 ...
```

Because the question was asked before the readings were seen, its reading is the answer, set without ", a lead". The
line says where v2 stands on it (a point behind), how far the interval lets it be either way, and how strong the
evidence is (weak), and that it is undecided: "undecided" never means the two are the same.

**The lesson: a small test still gets a fair read, and decides only when the evidence clears the bar.** Every reading
is read at once: v2's Number score is 17 points ahead of v1's, its interval -1 to +38 points, a secondary reading
shown with its evidence and decided on nothing, since the question asked about currency. The evidence is graded weak
below 3, moderate from 3 and strong from 10; chance alone reaches 10 at most one time in ten, however often it is read.
A decision needs the bet on the asked side to reach 20. One v2 trial failed on malformed JSON from the model, and it
counts against v2, stated, never excluded: a model's unusable answer is the candidate's failure.
[Reading reports](reading-reports.md#reading-a-comparison) explains each verdict.

**Adding cases and reading again.** When a reading is undecided, add invoices to the end of
`invoices.jsonl` and run `compare.py` again, as often as you like, with nothing declared first. The line under the
first line says what that allows:

```text
Where its runs are kept in a store that outlives the run (a file, or a host's), add cases to the end and run it again into that store as often as you like: each comparison of the arms on a pass rate, or on a score within its declared range, keeps its error however many times it is read, and a single read gives up some power for that; guardrails, floors, each arm's own rate, cost, latency and the exit status are read at the cases run so far, so reading them again is not covered.
```

- **Why it holds.** Run again into the same store, the comparison continues: the invoices already run are read as they
  ran, and only the ones you added run. The print says so, last: `Continues the comparison run before under this
  name: ...`. Each pass rate, and each score on its declared range, is read by a bet whose stakes depend on no count
  of cases declared ahead: a mixture of fixed betting fractions, weighted toward the larger ones so a few cases can
  grow it, over the cases in an order fixed before any answer. A reading after you add cases is the same bet read
  later, so stopping when it first decides keeps the error it states.
- **What starts a new comparison.** Change an arm's name, model or levers, the control, a grader's name or what it
  declares (its range, direction, margin or guard), a judge's model or criteria, the question or its margin, or the
  repeats, or edit, move or remove an invoice already run, and every case runs again as a new comparison; the print
  names what changed (`A new comparison: ...`).
- **What does not.** An arm is known by its name, model and levers, and a grader by its name and declaration (a judge
  by its model and criteria), never by its code. A runner or a grader function edited under the same name continues
  the comparison, its new cases pooled with the ones run before, as the continuation's sentence says: after changing a
  prompt or a grader's code, give the eval a new `name=` (or another scope or store) to compare it afresh. A store kept
  in memory (`store=":memory:"`) keeps nothing between runs, so each run there is a new comparison, and the line under
  the first line says so in place of the one above.
- **A run that stopped.** A run stopped by its cost cap, a replay miss or an interrupt is kept as it ran: every answer
  it recorded counts, and running it again (with the cap raised) runs only what it never recorded, a case's missing
  arms and repeats as well as the cases it never reached, and the comparison finishes. Interrupt it once (Ctrl+C) and
  it starts no further case, lets the cells in flight finish, and keeps them; interrupt it again to stop at once, and
  the cells it cut run again next time, which the print counts.
- **Why a stop keeps the error.** The order the cases are read in is fixed before any answer: each run's new cases are
  shuffled once and appended. Each run plays its cases one at a time in that order, every arm and repeat of a case
  together, and the bet reads the longest run of cases from the start of the order that are complete on every arm and
  repeat. Every reading is that one sequence read further, so stopping whenever you like, after watching whatever you
  like, keeps the error it states, however fast or slow each arm answers, short of a cell's deadline.
- **What it costs.** Read once, it tells the arms apart a little later than a test fixed for that many cases would:
  its intervals are somewhat wider. That is the price of reading again freely.
- **What it does not cover.** Guardrails, floors, each arm's own rate, cost, latency and the exit status are read at
  the cases run so far, each time, so re-reading them as cases are added is not covered.

## Rung 3b: a margin on a reading the question does not decide

[`margin.py`](../examples/3-invoices/margin.py) is 3a with a margin on `total`:

```python
    # Fixed before the run: totals are checked by hand, so a gap under 20 points would not change which prompt ships.
    grader=["currency", "number", "due", grader("total", margin=0.20)],
```

- **`grader("total", margin=0.20)`** is the `total` field, graded with a margin: the largest difference too small to
  act on. Only with a margin can a reading come out **equivalent**, the one verdict that says "good enough": the whole
  interval sits inside it. Only the question's reading is decided, so `total` here, a secondary reading, shows its
  interval against the margin and decides nothing; a margin on the question's own reading is what can read
  equivalent. Every grader is higher-is-better, so no direction is stated.

What its report said of `total`:

```text
| Total score | 99% (88 to 99.9) | 99% (88 to 99.9) | 0 pts | -13 to +13 pts | v2 level with v1, undecided (a secondary reading, which the question leaves open); immaterial: the observed delta is inside the margin of ±20 pts on Total score, which does not show the true difference is that small | 30 paired |
```

**The lesson: a margin comes from the decision, and equivalence costs cases.** The margin is the largest difference
that would not change what you do, fixed before the readings from what the decision is for: here a person checks
every total before it is paid, so a prompt worse on `total` by less than 20 points would still ship on its other
fields. It is never read off the interval a run produced: a margin chosen to fit the readings it judges is criterion
selection ([Measuring soundly](measuring-soundly.md#declare-the-campaign-its-question-arms-and-aspects)), and it makes
"equivalent" a foregone conclusion. The cost is honest too: 30 invoices narrow v2's interval on `total` to 13 points
either side, inside this margin, and a question asked of `total` could read equivalent on them; a decision
that needs a 5-point margin needs many more invoices, and with too few the reading is undecided, never
equivalent.
[Measuring soundly](measuring-soundly.md#how-many-cases-how-many-repeats) says how many cases a question needs.

## Rung 3c: a guardrail, decided on its own

[`guardrail.py`](../examples/3-invoices/guardrail.py) adds something v2 must not do worse on:

```python
@grader(guard=0.02)  # a guardrail: v2 must not do worse here by more than 2 points, whatever else it gains
def no_invented_total(case: dict, out: Invoice) -> bool:
    """Never states a total the invoice does not print."""
    return case["total"] is not None or out.total is None
```

- **`grader(guard=0.02)`** makes a guardrail: read for each arm against the control as `held`, `breached` or
  `undecided`, apart from the contrasts, so a gain elsewhere cannot pay for a loss here. `no_invented_total` returns a
  `bool`, so it is still a check: a trial it fails is failed, in the pass rate and in pass^k.
- **Undecided is never held.** The door exits 4 when a guardrail is not shown held, as it does on a regression or a
  breach. Its eval is named after its cases file, `invoices`.

What the recording printed, first lines, errors and the last line, which says why it exits 4:

```text
invoices: Accuracy: v2 behind v1, undecided. Over 30 paired cases, v2 is 1 point behind v1. The interval allows v2 to be as much as 14 points worse or 12 better. The evidence that v2 is worse is weak (1.2). Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2. v2 may do worse than v1 on the No invented total score guardrail.
v1 (control): 90 result(s): 90 scored
  Number score: 83%, 75 of 90 trials
  ...
v2: 90 result(s): 89 scored, 1 failed by the candidate
  Number score: 100%, 89 of 89 trials
  ...
  error: case c02: the candidate raised ValidationError: 1 validation error for Invoice Invalid JSON: expected `:` at line 1 column 7 ...
...
exit 4: Accuracy: v2 behind v1, undecided. Over 30 paired cases, v2 is 1 point behind v1. The interval allows v2 to be as much as 14 points worse or 12 better. The evidence that v2 is worse is weak (1.2). Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2. The gate failed on guardrail No invented total score, v2 vs v1: undecided: v2 may be worse than v1 by more than the margin (the interval reaches both sides of -2 pts, the declared margin, so the arm is shown neither within it nor beyond it).
```

And what the report (`.evals/reports/invoices/report.md`, the same page as `report.html`) said: its first screen's
title, the question, the lead and the guardrail's warning beside it, the reading, the contrast row for `number`, the
guardrail's row, and the trials' census:

```text
# invoices
"Does v2 read the currency an invoice charges in right more often than v1?"
v1 vs v2
**Accuracy: v2 behind v1, undecided. Over 30 paired cases, v2 is 1 point behind v1. The interval allows v2 to be as much as 14 points worse or 12 better. The evidence that v2 is worse is weak (1.2). Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2.**
**v2 may do worse than v1 on the No invented total score guardrail on invoices, by more than its 2-point margin. Of the 30 cases that could fail it, v2 failed 1 and v1 failed 1, too few to settle it.**
| invoices | v2 | 30 | 73 of 90 trials | 88 of 90 trials | +17 pts | -1 to +26 | — | cannot tell apart |
| Number score | 83% (67 to 93) | 100% (88 to 100) | +17 pts | -1 to +38 pts | v2 ahead of v1, undecided (a secondary reading, which the question leaves open) | 30 paired |
| No invented total score | v2 | 99% | 99% | 30 paired | 0 pts | -13 to +13 pts | 2 pts | undecided: v2 may be worse than v1 by more than the margin (the interval reaches both sides of -2 pts, the declared margin, so the arm is shown neither within it nor beyond it) |
180 trials ran. Of them 161 passed and 19 failed.
```

The page leads on the question's measure as the printed line does, and says beside it what leaves the guardrail
undecided.

**The lesson: a report that cannot decide says which question it could not answer, and what would answer it.** The
guardrail is the question the first screen's warning names as unanswered: each arm stated a total once on a noisy
invoice that prints none, the interval on the difference is ±13 points against a margin of 2, and an undecided
guardrail is never read as held, so the gate fails and the run exits 4. What would answer it is a narrower interval on
that guardrail, and only an invoice that prints no amount due can fail it (7 of the 30 do): more of those
hard-negative cases.

## Rung 3d: the gate in CI

[`ci.py`](../examples/3-invoices/ci.py) is the step a pipeline runs: guardrail.py's eval, its report written for the
reviewer, and the exit code the gate maps to.

```python
from guardrail import invoices

if __name__ == "__main__":
    result = asyncio.run(invoices.run())
    print(result)  # the verdict line, then each arm's readings; every delta and its interval is in the report
    result.write("report")  # report.html for the reviewer, beside report.md, campaign.json and the bundle
    raise SystemExit(4 if result.gate().failed else 0)
```

- **`result.write("report")`** writes the report for a reviewer (`report.html`, `report.md`, `campaign.json`,
  `bundle-<aspect>.json`) under the directory it names.
- **`result.gate()`** fails a CI step on a regression, a breach, or a guardrail not shown held. Here the guardrail is
  undecided, so `cd examples/3-invoices && python ci.py` exits 4. `python -m threetears.evals gate` is the same gate
  over a stored campaign ([Serving](serving.md#gate)).
- **Only the question's own reading can fail it as a regression.** With a question asked, the first grader's reading
  is decided and every other reading is shown and decided on nothing, so a second check's drop never fails the gate:
  make the reading you gate on the first grader, or a guardrail. Ask the gate as one with a margin
  (`Eval(question="Is v2 no worse than v1?", margin=0.05)`) and its bet leans toward worse: a regression decides once
  the evidence that v2 is worse reaches about 22, where a question asking whether v2 does better leans the other way
  and needs about 200 to call it worse.

## Rung 3e: two levers at once

[`grid.py`](../examples/3-invoices/grid.py) crosses the two prompts with two reasoning efforts: four arms, a cell of
the grid each.

```python
low = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")
high = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="high", recording="recordings/invoices-high.jsonl")

grid = Eval(
    "invoices.jsonl",
    runner={  # each arm is one cell of the grid: prompt is one factor, effort the other
        "prompt=v1, effort=low": low.prompt(V1, Invoice),  # the control
        "prompt=v2, effort=low": low.prompt(V2, Invoice),
        "prompt=v1, effort=high": high.prompt(V1, Invoice),
        "prompt=v2, effort=high": high.prompt(V2, Invoice),
    },
    grader=["currency", "number", "due", "total"],
    name="grid",
)


async def main() -> None:
    result = await grid.run()
    print(result)  # each arm against the control: each lever's effect with the other at its control level
    print(await result.against("prompt=v1, effort=high"))  # the prompt's effect at high effort, no run repeated
```

- **Arms named as `k=v` pairs are factors.** Each pair is one lever of the design, `prompt` and `effort` here, two
  levels each, and the report keys every run by its level of both.
- **Every arm is read against one control**, so the grid's own report gives each lever's effect at the control's
  level of the other: the prompt at low effort, the effort on v1.
- **`result.against(...)`** reads the same runs against another control, with no run repeated: against
  `prompt=v1, effort=high`, v2's row is the prompt's effect at high effort. Each control is its own Holm family.
- The low-effort cells replay `recordings/invoices.jsonl`, the same calls 3a to 3d replay; the high-effort cells
  replay `recordings/invoices-high.jsonl` (180 calls, $0.0188).

What the door printed (it runs the eval; `python grid.py` also prints the reading against the second control), the
arms and the last line, which says why it exits 4:

```text
Prompt v1, effort low (control): 90 result(s): 90 scored
...
Prompt v1, effort high: 90 result(s): 90 scored
...
Prompt v2, effort high: 90 result(s): 90 scored
...
exit 4: ... The gate failed on Production cost: Prompt v1, effort high worse than Prompt v1, effort low (lower is better); Production cost: Prompt v2, effort high worse than Prompt v1, effort low (lower is better).
```

and the report's row for v2 at high effort:

```text
| grid | Prompt v2, effort high | 30 | 73 of 90 trials | 90 of 90 trials | +19 pts | +1 to +30 | — | shown better, a lead |
```

**The lesson: a lever has a price, and the gate reads it.** v2 at high effort is the one cell whose pass rate is shown
better than the control's, a lead, since this grid asks no question; and both high-effort cells cost more per reply
than the control, which is a regression the gate fails on, exactly as it would fail on a drop in quality. Which lever
to pull is the person's call over both readings. The contrasts test every cell against the control, never against
each other: a leader picked after the results is a lead to confirm in a fresh campaign
([Measuring soundly](measuring-soundly.md#start-from-the-question)).

## Rung 3f: have a model write the analysis

Every page so far is code's. The harness door runs the same file, then hands the campaign to the report writer: an
analyst model that reads the frozen evidence through tools and writes the page, and an editor that reviews it before
it is stored.

```bash
ANTHROPIC_API_KEY=... uv run python -m threetears.evals.harness examples/3-invoices/compare.py --write-report claude-sonnet-5-5
```

| What | The harness door |
|---|---|
| Key | `ANTHROPIC_API_KEY`: the writer's every turn is live |
| Cost | at most `--budget-usd` a report, $3 unless you name another amount |
| Page | `.evals/reports/compare/analysis.html`, beside code's `report.html` |
| Served | the campaign's analysis from then on: `report` and `report_read` show this page |
| Turns | recorded under `examples/3-invoices/recordings/report/` |

The writer's turns never replay here: each run of the file is a new campaign with a new snapshot, so no earlier
recording is of it, and with no key the door refuses, naming the file it would record. A writer replays offline only
over frozen campaigns, which is rung 5.

## Rung 4a: an agent, graded on the world it leaves

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
booking: agent passed 31 of 36 trials on 12 cases. Cases that failed a trial: r08, r09, r10.
36 result(s): 36 scored
  In window score: 100%, 36 of 36 trials
  goal check (state.booked.length >= 1) == (variation.expect == "book"): passed 32/36 — unproven: no control shows it can tell acting from doing nothing; a candidate that did nothing might pass it, so it is not shown to measure the behaviour; at launch one passed it in 4 of 12 case(s)
  goal check not intersects(state.booked, state.busy): passed 32/36 — unproven: no control shows it can tell acting from doing nothing; a candidate that did nothing might pass it, so it is not shown to measure the behaviour; at launch one passed it in 12 of 12 case(s)
  goal check call_count("day.book") <= 1: passed 35/36 — unproven: no control shows it can tell acting from doing nothing; a candidate that did nothing might pass it, so it is not shown to measure the behaviour; at launch one passed it in 12 of 12 case(s)
```

**The lesson: grade what the agent did, and only with checks that can fail.** Most misses are the agent booking a
taken hour on the full-day hard negatives (`r09`, `r10`), where the right move is to say no: its reply could read
fine while the calendar is wrong. The second and third checks are guardrails worth keeping, but an agent that did
nothing passes them in every case, so their pass rates are not measurements of capability; only the first beats the
did-nothing control. [Evaluating a tool-using agent](evaluating-agents.md) covers proving a check with an authored
control, and declaring a production world of your own.

## Rung 4b: serve it to the team

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
behind rung 3f's second door. It gets the same treatment as anything else you ship: arms, cases, a judge, a verdict.
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
    held_out_fraction=0.35,  # the default, stated: 7 held-out cases of 11, past the 5 a rate needs
    name="writer",
)
```

- **A case is a campaign, frozen.** Each file in `cases/` holds one campaign's evidence exactly as the writer's tools
  read it: the four campaigns rungs 1 to 4 store when they replay (rung 3's is its campaign from before it split into a
  file a step: the eval `guardrail.py` grew from, frozen on 2026-11-03 with a 5-point margin on `total` and no
  question, so it is no longer the campaign any rung-3 file declares), a larger campaign the package ships as a fixture
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

What the recording printed (made 2026-11-03, its judge's calls again 2026-11-14), replayed from `packages/evals/` with no key: the arms, the first
readings and the judged ones, the judge's preference between the two arms' pages, the promotion read, and where the
turns came from (each path printed whole):

```text
writer: lead-first against shipped, 7 held-out case(s) x 1, judged by google/gemini-3.8-flash
| reading | shipped | lead-first |
|---|---|---|
| Report finished | 43% (3 of 7) | 86% (6 of 7) |
| Bounces | 0.9 | 1.0 |
...
| Must-fix findings open | 0.43 | 0.29 |
| Should-fix findings open | 2.43 | 1.57 |
...
| Writer turns | 12.57 | 12.29 |
...
| report_writer.first_screen (judged) | 5.0 | 5.0 |
| report_writer.coherent_story (judged) | 5.0 | 5.0 |
| report_writer.honest_emphasis (judged) | 5.0 | 5.0 |
| report_writer.sound_elision (judged) | 5.0 | 5.0 |
| report_writer.confounds_handled (judged) | 5.0 | 5.0 |
| report_writer.actionable_next_steps (judged) | 5.0 | 5.0 |
| report_writer.evidence_fits_claims (judged) | 4.6 | 4.7 |
...
| Candidate | Pairs | Reads | Candidate's share (%) | Control's share (%) | 95% interval (%) |
|---|---|---|---|---|---|
| lead-first | every pair | cannot tell apart | 68 | 32 | 26 to 98 |
| lead-first | both pages finished | cannot tell apart (too few cases) | 62 | 38 | 0 to 100 |

| Candidate | Order consistency | First-slot lean (pts) | Can't tell | Held-out cases to settle | Spent |
|---|---|---|---|---|---|
| lead-first | 71% (5 of 7) | 18 | 0 | 33 | $0.041 of $1.95 |
...
promotion of lead-first: Undecided: on the primary reading the evidence cannot tell lead-first and shipped apart, and only an improved primary reading promotes.
shipped: writer turns replayed from .../examples/5-writer/recordings/turns/shipped-4e14e276b568-claude-sonnet-5-5-medium-r2
lead-first: writer turns replayed from .../examples/5-writer/recordings/turns/lead-first-af11aac84890-claude-sonnet-5-5-medium-r2
```

A report finished is one the editor cleared, so it is stored finished rather than as a draft. The report
(`.evals/reports/writer/report.md`) opens with the answer, the promotion reading and the judge's preference in one
sentence, and the same two side-by-side tables head its Scorecard:

```text
**Undecided, so shipped stays: across 7 held-out cases the judge gave lead-first 68% of its preference over shipped (50% is even), too few cases to tell from an even split; if the share stayed near 68%, roughly 30 cases would settle it.**
```

Further down it reads the two code measures that moved most:

```text
| Report finished | 43% (16 to 75) | 86% (49 to 97) | +43 pts | -49 to +100 pts | cannot tell apart | 7 paired |
| Must-fix findings open | 0.43 (0 to 0.92) | 0.29 (0 to 0.98) | -0.14 | -1.31 to +1.03 | cannot tell apart | 7 paired |
```

| What | Recorded |
|---|---|
| Reports | 14: 2 arms x 7 held-out cases, once each |
| Writer turns | about 12.4 a report |
| Writer | $6.90, about $0.49 a report |
| Judge | $0.28, 112 calls: 98 scores and the pass's 14 |
| Total | $7.18, under the $39 `max_cost_usd` |

[`recordings/README.md`](../examples/5-writer/recordings/README.md) holds the one command that records it again, what
it may spend, and what to do when a recording run is cut short.

**The lesson: a judge that scores one page at a time sits at the ceiling here, so the arms are read side by side,
and at seven cases even that reading is mostly the judge's own noise.** Every judged item reads 5 or near it for both
arms: the writer revises each page until the editor passes it on the same rubric, so the scores bunch at the top and
cannot tell the arms apart. Read side by side, the same judge gave lead-first 68% of its preference (50% is even), and
its two orders agreed on 5 of 7 cases. Recorded again and again on the same seven cases, a few lines of the pages
apart each time, the judge has read between 43% and 73%, which is why seven cases decide nothing. Two code measures differ, and the evidence cannot
tell the arms apart on either: 6 of 7 lead-first reports finished against 3 of 7, with fewer must-fix findings left
open (0.29 a report against 0.43). That preference is the promotion read's primary reading, fixed before the run: it
reads `promote` only when the preference favours the candidate on enough held-out cases, no gated reading regressed
and every guardrail held. 7 held-out cases, each written once (k=1), separate only a large effect, and the pass gives
the size that would settle this one only if the share it read were the true one: if it stayed near 68%, roughly 33
held-out cases. So the read is `Undecided`, which never promotes, and recording the call stays a person's act. That
is the answer at this size, stated rather than tuned around: what would answer it is more cases or more repeats of
each, and each costs a recorded report. [Pairwise
preference](judges-and-calibration.md#pairwise-preference-for-the-report-writer) covers the share and the promotion
rule, and [Measuring soundly](measuring-soundly.md#how-many-cases-how-many-repeats) how many cases a question needs.

## Where next

| To | Read |
|---|---|
| decide what to test before writing cases, and how many | [Building an inventory](building-an-inventory.md) |
| trust a judge, and measure it as a subject | [Judges and calibration](judges-and-calibration.md) |
| read every part of a report | [Reading reports](reading-reports.md) |
| gate a pipeline on a stored campaign, with no run | [Serving: `gate`](serving.md#gate) |
| read a grid of factors one lever at a time | [Measuring soundly](measuring-soundly.md#start-from-the-question) |
| have a model write a comparison's report, and read it | [The analyst's report](reading-reports.md#the-analysts-report-written-over-frozen-evidence) |
| promote a change to the report writer | [Pairwise preference for the report writer](judges-and-calibration.md#pairwise-preference-for-the-report-writer) |
| choose a design, k and a case count, and avoid known traps | [Measuring soundly](measuring-soundly.md) |
| price a run before it spends | [Cost and budgets](cost-and-budgets.md) |
| evaluate an agent in your production world | [Evaluating a tool-using agent](evaluating-agents.md) |
| keep runs in your own database and launch from your app | [Adopting the engine](adopting-a-host.md) |
