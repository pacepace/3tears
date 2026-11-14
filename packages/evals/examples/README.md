# Examples

The ladder comes first: one directory per rung, each in its own domain. Rungs 1 to 4 are recorded on cheap models
through OpenRouter and rung 5 on Claude Sonnet, and each runs offline and free from the recordings beside it. Climb
it in order; each file adds one idea to the one before, so rungs 3 and 4 are steps of a file each (3a to 3f, 4a
and 4b).

## The ladder

| Rung | Where | Why you would climb it |
|---|---|---|
| 0 | [`1-triage/triage.toml`](1-triage/triage.toml), after [Building an inventory](../docs/building-an-inventory.md) | Write down what your app must never get wrong, then run it with one TOML file. |
| 1 | [`1-triage/triage.py`](1-triage/triage.py) | Point it at your own function: 15 lines of Python. |
| 2 | [`2-replies/replies.py`](2-replies/replies.py) | When no code can grade the answer, let a model judge it, and check that the judge is right. |
| 3a | [`3-invoices/compare.py`](3-invoices/compare.py) | Changing a prompt? Compare it against the current one, and read the verdict. |
| 3b | [`3-invoices/margin.py`](3-invoices/margin.py) | Is it good enough? Declare a margin, so a reading can come out equivalent. |
| 3c | [`3-invoices/guardrail.py`](3-invoices/guardrail.py) | Must it not get worse somewhere? A guardrail, decided on its own. |
| 3d | [`3-invoices/ci.py`](3-invoices/ci.py) | Gate CI on it, with a report you can hand to a reviewer. |
| 3e | [`3-invoices/grid.py`](3-invoices/grid.py) | Two levers at once? A grid of factors, read by one lever. |
| 3f | `--write-report` over [`compare.py`](3-invoices/compare.py) | Have a model write the analysis. |
| 4a | [`4-booking/agent.py`](4-booking/agent.py) | Is your agent doing the right thing to the world, not only saying it? |
| 4b | [`4-booking/serve.py`](4-booking/serve.py) | Serve your evals, and the reports over them, to the team. |
| 5 | [`5-writer/writer.py`](5-writer/writer.py) | Is the model that writes your reports writing them well? A writer is a model too: compare two of its skills on frozen campaigns, and let the engine read the promotion. Its Sonnet recording replays offline, free. |

One door runs every rung: `python -m threetears.evals <file>`, from any directory. It runs every eval the file
defines (an `Eval`, or rung 5's `WriterEval`). The paths a rung names (its cases, its recordings, a judge's labels)
are read beside its file; the store (`.evals/evals.sqlite`) and the report (`.evals/reports/<name>/report.html`) are
written under the directory you run from. That report is code's, built from the readings with no model called. A
second door, `python -m threetears.evals.harness <file> --write-report MODEL`, also has the report writer write one
(rung 3f). In a checkout, from
`packages/evals/`, after `uv sync`:

```bash
uv run python -m threetears.evals examples/1-triage/triage.toml    # rung 0
uv run python -m threetears.evals examples/1-triage/triage.py      # rung 1
uv run python -m threetears.evals examples/2-replies/replies.py    # rung 2
uv run python -m threetears.evals examples/3-invoices/compare.py   # rung 3a
uv run python -m threetears.evals examples/3-invoices/margin.py    # rung 3b
uv run python -m threetears.evals examples/3-invoices/guardrail.py # rung 3c
uv run python -m threetears.evals examples/3-invoices/grid.py      # rung 3e
uv run python -m threetears.evals examples/4-booking/agent.py      # rung 4a
uv run python -m threetears.evals examples/5-writer/writer.py      # rung 5
```

Two files run from their own directory instead, because each is run with plain `python` rather than through the
door, and outside the door a relative path is read from the working directory, not beside the file: rung 3d's
[`ci.py`](3-invoices/ci.py) (`cd examples/3-invoices && python ci.py`; the eval it imports names `invoices.jsonl` and
`recordings/invoices.jsonl`) and rung 4b's [`serve.py`](4-booking/serve.py) (from `examples/4-booking/`, whose store
is `.evals/evals.sqlite` there). Importing a sibling is not the reason: plain `python` puts a script's own directory
on `sys.path`, and the door puts a rung's there too.

A recording that exists is replayed: no key, no network, no cost. `RECORD=1` with `OPENROUTER_API_KEY` set records it
again, live (`RECORD=1 uv run python -m threetears.evals examples/1-triage/triage.py`). A recording holds one model at one
temperature, and an email the recording does not hold fails the run, naming the call, and is never sent to a model.

### Rungs 0 and 1: email triage, `1-triage/`

[`emails.jsonl`](1-triage/emails.jsonl) is 30 support emails, each with the triage a person gave it: its `category`,
whether it is `urgent`, and whether it `needs_human`. Twelve are in the `lookalike` stratum, the hard negatives: a
phishing mail shouting about your account is spam, a calm note that customer data is exposed is urgent, a vendor
selling a lead list is not a sales lead. `stratum` is a reserved key, so the runner never sees it and the report reads
each stratum apart.

**Rung 0, [`triage.toml`](1-triage/triage.toml)**, is the eval with no Python: the cases, a prompt to
`deepseek/deepseek-v4-flash` whose `{email}` is filled from each case, the answer's fields, and three field graders.
`category` is graded as a label, so it gets a confusion matrix and each label's precision and recall; `urgent` and
`needs_human` are pass rates. Each email is triaged three times (`k = 3`).

#### The keys a TOML file takes

Every path is read relative to the TOML file.

| Key | What it holds |
|---|---|
| `cases` | A case file (`.jsonl`, `.json`, `.csv`, `.toml`), or the cases inline as `[[cases]]` |
| `grader` | Field names, expressions, and judges (`{judge = {model = "...", rubric = {...}}}`) |
| `recording` | The runner model's recording; each judge's sits beside it (`triage.judge.jsonl`, `triage.judge2.jsonl`) |
| `name`, `k` | The eval's name (the file's stem if not given) and how many times each case is asked |
| `[runner] model` | The model's id |
| `[runner] prompt` or `prompt_file` | The prompt, or a file holding it; `{field}` is filled from each case |
| `[runner] system` or `system_file` | The system message, or a file holding it, sent as written; optional |
| `[runner] max_tokens` | The output cap, a positive number of tokens; optional |
| `[runner] effort` | The reasoning effort, a word (`none` to `max`) or a token budget below `max_tokens`; optional |
| `[runner.output]` | The answer's fields: a type name (`str`, `int`, `float`, `bool`) or the list of values allowed |

Point `prompt_file` and `system_file` at the files your app sends, and set `max_tokens` and `effort` as your app does,
so rung 0 measures the request your app makes. The files are read each time the eval loads, and the recording is
keyed by the text sent, so an edit to either file replays as a prompt the recording does not hold, by name.

**Rung 1, [`triage.py`](1-triage/triage.py)**, is the same eval with your own function as the runner, in 15 lines.
`Eval(cases, runner, grader)` takes the three slots. The new grader is `escalates`, a plain function: it fails only
an email a person must see that the triage left alone. That rule is asymmetric, which a field name cannot say. In
rung 0's recorded run, `needs_human`, graded both ways, passes 0.81 of the time, and 16 of its 17 misses are emails
escalated that did not need a person: a minute of someone's time, not a customer left alone. Rung 1's `escalates`
passes 0.99. Read the misses each run prints before its means.

Recorded on `deepseek/deepseek-v4-flash`: 90 calls for each file, about $0.002 each; `triage.py`'s docstring gives
both figures as the recordings' cost fields sum them.

### Rung 2: replies a judge grades, `2-replies/`

[`replies.py`](2-replies/replies.py) drafts a reply to each of 20 customer emails
([`emails.jsonl`](2-replies/emails.jsonl)) from a support policy ([`policy.md`](2-replies/policy.md)), twice each.
No code can say whether a reply is right, so the grader is a `Judge`: `google/gemini-2.5-flash-lite` grades every
reply on three pass/fail dimensions, `grounded`, `answers` and `safe`, one call per dimension, reading the policy and
the email beside the reply (`case_material`). The judge is another family than the runner
(`deepseek/deepseek-v4-flash`) on purpose: a model tends to rate its own output higher, and judging with the runner's
own model prints the self-judging warning under the scores, a warning no quiet print drops.

**A judge is a model too, so check that it is right, two ways.** Its four `Example`s are hard negatives in pairs, one
that must fail and one that looks like it and must pass. The run grades each one twice, through the same call it
grades the replies with, and the recorded judge gets all four right:

```text
judge: 4 of 4 examples graded as they must be on every one of 2 repeats ($0.00086 over 8 call(s))
```

A miss would print under it, with what the judge gave and what it must get. But the judge also passes all 40 replies
on all three dimensions, and read a few, you will find claims the policy never makes: telling four written pairs
apart is not grading the run's replies right. That takes a person's labels on the replies, and the run says they are
missing in the line it never quiets:

```text
judge: 0 of 20 items read blind: judge uncalibrated (labels.csv is not there yet); next step: open the blind_read.html the report writes, answer its items from the top, and save its labels as labels.csv
```

Calibrating it is yours to do; no person's labels ship with the rung.
1. Run it. The report directory (`.evals/reports/replies/`) holds `blind_read.html`: the replies one at a time in
   a random order, each with the email it answers, and no judge verdict on the page.
   [`blind_read.html`](2-replies/blind_read.html) beside the example is that file, as the recorded run wrote it
   ([`blind_read.csv`](2-replies/blind_read.csv) is the same order as rows, for a spreadsheet).
2. Open it and answer from the top. Stop when you like, press **Download labels.csv**, and save the file beside
   `replies.py`. What you answered from the top is a random sample; answers out of order read `selected`.
3. Run it again. `Judge(labels="labels.csv")` reads your labels back, pairs each with the judge's score of the same
   reply (by its fingerprint, so a replayed run draws the same order and pairs), and prints how far you read and,
   per dimension, the results paired, κ with its bounds, and the evidence tier it earns. Below 20 results the tier
   says the judge is not yet calibrated; reword a rubric dimension and its labels no longer pair.

Recorded: 40 reply calls on `deepseek/deepseek-v4-flash` ($0.0017) and 128 judge calls on
`google/gemini-2.5-flash-lite` ($0.0159, 8 of them on the examples), the runner's and the judge's in recordings of
their own (`recordings/replies.jsonl`, `recordings/judge.jsonl`).

### Rung 3a: two prompts compared, `3-invoices/compare.py`

Rung 3 is one directory and six steps, five of them a file each, over 30 invoices ([`invoices.jsonl`](3-invoices/invoices.jsonl))
in three strata (clean, multi-currency, noisy OCR) and two prompts. What every file reads an invoice into, and the two
prompts, are in [`invoice.py`](3-invoices/invoice.py), which declares no eval and builds no `Model`. Every eval gets its
own `Model`, since a replay serves each recorded call once per model: `grid.py` builds two (one per effort), and
`ci.py` runs `guardrail.py`'s eval on its model.

[`compare.py`](3-invoices/compare.py) compares the prompts, as `runner={"v1": model.prompt(V1, Invoice), "v2": ...}`;
the first is the control, and v2 is read against it on every reading: the delta, its interval and a verdict.
`question="Does v2 read the currency an invoice charges in right more often than v1?"` is the one sentence it
answers, asked before any reading is seen: its measure is the first grader, `currency`, and the page quotes it. What
the recorded run said: accuracy did not separate v2 from v1, and nothing else did.

```text
compare: Accuracy: the evidence cannot tell v2 and v1 apart. Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2. Over 30 paired cases, the interval allows v2 to be as much as 20 points worse or 18 better.
```

One v2 trial failed, its recorded error `the candidate raised ValidationError: 1 validation error for Invoice` on
`Invalid JSON`: malformed JSON from the model, counted against v2.

Recorded on `deepseek/deepseek-v4-flash` at `max_tokens=8192`: 180 calls, $0.0105 (`recordings/invoices.jsonl`),
which 3b, 3c, 3d and the grid's low-effort cells replay too.

### Rung 3b: a margin, `3-invoices/margin.py`

[`margin.py`](3-invoices/margin.py) is 3a with `grader("total", margin=0.20)`, and only with a margin can a reading
come out equivalent, the one verdict that says "good enough". The margin comes from the decision, fixed before any
reading: a person checks every total before it is paid, so a prompt worse on `total` by under 20 points would still
ship on its other fields. It is never read off the interval the run produced. 30 invoices narrow the interval on
`total` to 19 points either side, so they can show equivalence only at a margin this wide; a tighter margin needs
more invoices:

```text
| Total score | 99% (88 to 99.9) | 99% (88 to 99.9) | 0 pts | -19 to +19 pts | v2 and v1 equivalent, within the measure's margin (margin ±20 pts) | 30 paired |
```

### Rung 3c: a guardrail, `3-invoices/guardrail.py`

[`guardrail.py`](3-invoices/guardrail.py) adds `@grader(guard=0.02) no_invented_total`, a guardrail: a pass/fail check
that fails each trial it fails, with its own held/breached reading against v1 on top, decided apart from the contrasts.
It grades the fields as compare.py does, with no margin. The guardrail is undecided for v2, so it
exits 4, its last line saying why in the report's words
(`exit 4: Accuracy: the evidence cannot tell v2 and v1 apart. ... The gate failed on guardrail No invented total score, v2 vs v1: undecided: ...`). The report's first
screen quotes the question and leads on its measure, the classifier's accuracy, with the guardrail's warning beside
it, and reads v2 against v1 in one row

```text
# invoices
"Does v2 read the currency an invoice charges in right more often than v1?"
v1 vs v2
**Accuracy: the evidence cannot tell v2 and v1 apart. Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2. Over 30 paired cases, the interval allows v2 to be as much as 20 points worse or 18 better.**
...
**v2 may do worse than v1 on the No invented total score guardrail on invoices, by more than its 2-point margin. Of the 30 cases that could fail it, v2 failed 1 and v1 failed 1, too few to settle it.**
...
| Aspect | Cases | ▸ v1 · pass rate | v2 · pass rate vs v1 |
|---|---|---|---|
| invoices | 30 | 81% (73 of 90 trials, 30 cases) | 98% · +17 pts (cannot tell apart) |
```

with the guardrail's row saying why, on the row's ruler beside its interval of `-13 to +13 pts`: `undecided: v2
may be worse than v1 by more than the margin (the interval reaches both sides of -2 pts, the declared margin, so the
arm is shown neither within it nor beyond it)`. That is the lesson: rather than declare a winner, the report names the
question it could not answer, and what would answer it is more hard-negative cases, invoices that print no amount due
(7 of the 30). [The ladder](../docs/ladder.md#rung-3c-a-guardrail-decided-on-its-own) quotes the rest.

### Rung 3d: the gate in CI, `3-invoices/ci.py`

[`ci.py`](3-invoices/ci.py) is the step a pipeline runs, from `3-invoices/`: `python ci.py` runs guardrail.py's eval,
`result.write("report")` writes the HTML report for a reviewer under `report/`, and `gate()` fails the step, exit 4,
on a regression, a breach or a guardrail not shown held, as guardrail.py's is.

### Rung 3e: two levers at once, `3-invoices/grid.py`

[`grid.py`](3-invoices/grid.py) crosses the prompts with two reasoning efforts, four arms named as `k=v` pairs
(`"prompt=v2, effort=high"`), so each pair is a factor of the design. Its low-effort cells replay
`recordings/invoices.jsonl`; its high-effort cells replay `recordings/invoices-high.jsonl`. Every arm is read against
the control, `prompt=v1, effort=low`; `python grid.py` also reads the same runs against `prompt=v1, effort=high` with
`result.against(...)`, the prompt's effect at high effort, with no run repeated. What the recorded run said: v2 at high
effort is the one cell shown better than the control, a lead, and both high-effort cells cost more per reply, which
fails the gate:

```text
Prompt v2, effort high: 90 result(s): 90 scored
...
exit 4: ... The gate failed on Production cost: Prompt v1, effort high worse than Prompt v1, effort low (lower is better); Production cost: Prompt v2, effort high worse than Prompt v1, effort low (lower is better).
```

Recorded on `deepseek/deepseek-v4-flash` at high effort: 180 calls, $0.0188 (`recordings/invoices-high.jsonl`).

### Rung 3f: a model writes the analysis, `--write-report`

The harness door runs a file, then the report writer (an analyst model and its editor) writes the campaign's report
over its frozen evidence:

```bash
ANTHROPIC_API_KEY=... uv run python -m threetears.evals.harness examples/3-invoices/compare.py --write-report claude-sonnet-5-5
```

It is live on every run, at most `--budget-usd` a report ($3 unless you name another amount), and writes
`.evals/reports/compare/analysis.html`, its turns under `3-invoices/recordings/report/`. The report is stored as the
campaign's analysis, so `report` and `report_read` serve it. Each run is a new campaign, so no recording is ever
replayed here; rung 5 is the writer offline.

### Rung 4a: an agent graded on the world it leaves, `4-booking/agent.py`

[`agent.py`](4-booking/agent.py) books meetings: each request ([`requests.jsonl`](4-booking/requests.jsonl)) seeds one
day's calendar, `World(Day, tools=[find_free, book])` hands the agent two tools in a JSON tool loop over
`model.json`, and the graders are goal checks on the day it left, beside the did-nothing control that only the first
check beats.

Recorded on `deepseek/deepseek-v4-flash`: 100 calls, $0.0080 (`recordings/agent.jsonl`).

### Rung 4b: serve it to the team, `4-booking/serve.py`

[`serve.py`](4-booking/serve.py) serves the eval's runs to the team: `as_host`, an MCP server in one
call (`mount_fastmcp(host(), scope="booking").run()`, the `[fastmcp]` extra, read-only;
[Serving](../docs/serving.md#serving-the-runs-to-the-team)) and
[the command line](../docs/serving.md#the-host-commands) (`python -m threetears.evals ls --host serve:host --scope booking`),
both from `4-booking/`, where `agent.py`'s runs are stored when it is run there. `report CAMPAIGN` and `report_read`
read a campaign's page there: the report the harness door wrote, once it has written one, and code's page until then.

### Rung 5: the report writer evaluated, `5-writer/`

[`writer.py`](5-writer/writer.py) evaluates the report writer, the analyst model and its editor behind rung 3f's
harness door, as the thing under test. Its two arms are two versions of the analyst's skill: a frozen copy of the
shipped one ([`skills/control/`](5-writer/skills/control/), frozen because every recording is keyed by the skill's
digest; [`skills/README.md`](5-writer/skills/README.md) says which) and [`skills/lead-first.md`](5-writer/skills/lead-first.md),
which puts each report's lead claim first. Both write on `claude-sonnet-5-5` at effort `medium`; the judge is
`google/gemini-3.8-flash`, another family.

Its eleven cases ([`cases/`](5-writer/cases/), one `.json.gz` each) are frozen campaigns, by stratum:

| Stratum | Cases | Side |
|---|---|---|
| `support-triage` | rung 1's campaign | held-out |
| `support-replies`, `invoices`, `booking` | rung 2's, rung 3's from before it split (frozen 2026-11-03, the eval `guardrail.py` grew from; no rung-3 file declares it now), and rung 4a's | dev |
| `kairo` | a larger campaign the package ships as a fixture | held-out |
| `misfit` | five of the package's report inputs reshaped to fit no exemplar | held-out |
| `misfit` | one more, in the r3 group | dev |

A case's side is its split group's, and `held_out_fraction=0.35` puts 7 of the 11 on the held-out side, where a
promotion is read. `tests/fixtures/writer_cases/export_writer_cases.py` wrote the files; a ladder case is frozen
once, since a rung run again is new evidence with a new snapshot id.

It replays its recordings with no key: 14 reports, the judge's scores and its preference between the two arms' pages.
The read is `Undecided`: the judge gave lead-first's pages a share of 68%, its two reading orders agreeing on 5 of 7
cases, and 7 held-out cases cannot tell that from an even split. [`recordings/README.md`](5-writer/recordings/README.md) says when each recording was made and what it cost, and
holds the one command that records them again (both `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`, under the $39 cap).
[The ladder](../docs/ladder.md#rung-5-evaluate-the-evaluator) walks it.

## Running them

Every recorded rung replays the recordings beside it when it runs with no key, so it calls nothing and costs nothing,
and its output is a real model's, as of that recording. There is no scripted stand-in for a model anywhere in them.

**Outside a checkout** (you installed `3tears-evals` with pip): the examples are not part of the installed
package, so fetch this whole directory from the repository, at the tag of the version you installed (the
[README's install step](../README.md#install) shows the tag form). This fetches only the examples:

```bash
git clone --depth 1 --branch <tag> --filter=blob:none --sparse https://github.com/pacepace/3tears
cd 3tears && git sparse-checkout set packages/evals/examples && cd packages/evals/examples
```

Keep each rung's directory together: a rung reads its cases and replays its `recordings/` from beside itself. Then
`python -m threetears.evals 1-triage/triage.py`, or with `export OPENROUTER_API_KEY=...` and `RECORD=1` first, any
rung live.

**A rung file imports the modules beside it.** Its directory is put first on `sys.path` before it loads and stays there for the process, as
`python FILE` puts it, so app code several rung files share (an adapter like the one below) is written once, in a
module beside them, and each imports it (`from router_adapter import RecordedClient`). A module is imported once per
process, so helpers in two rung directories need distinct names.

**Exit codes.** The command exits with the worst code of the evals the file declares:

| Code | Meaning |
|---|---|
| 0 | Every run completed |
| 1 | A run did not complete |
| 2 | Refused before it ran: the file, or what it declares |
| 3 | An error nothing anticipated, its traceback printed |
| 4 | A comparison's gate failed, with a line saying why |

A test of a rung asserts the code its recording produces, and 4 is a verdict, not a crash: a recorded comparison
whose gate fails exits 4 on every replay.

**Your own client.** A client of your own, for another provider or a gateway, is anything with an async
`generate(*, system, user, response_format=None)` that returns an object carrying ten attributes: `content`,
`input_tokens`, `output_tokens`, `reasoning_tokens`, `cost_usd`, `price_source`, `model`, `served_model`,
`stop_reason` and `temperature` (a `SimpleNamespace` with the ten is one), and optionally `sent_settings`, what the
request actually carried, which the engine records as unverified when it is absent. `check_completion_conformance` (from
`threetears.evals.testing`) is the check: hand it one completion your client returned, and it names each attribute
missing and a `stop_reason` the engine does not know (`end_turn`, `max_tokens`, `content_filter`, `truncated` or
`error`). An absent or unknown finish reason is `truncated` (output that stopped early, cause unnamed); `error` is for
a provider that reported a failure in place of a completion. When to keep your own rather than the shipped client:
[Adopting the engine](../docs/adopting-a-host.md#bringing-a-provider).

**The package's `Model` inside your app's client.** The other way round: your app owns a client protocol (a class
its own code calls, handed in through a constructor), and the recording `Model` goes inside it, so every call is
recorded once and replayed after while the app's code stays the app's. An adapter for an app whose client takes a
list of messages, built with the app's own model, `max_tokens` and effort, which the recording is keyed by:

```python
from threetears.evals import Model

model = Model("openai/gpt-5-nano", max_tokens=600, effort="minimal", recording="recordings/router.jsonl")


class RecordedClient:
    """The app's client protocol over the package's Model: each call recorded once, replayed after."""

    async def complete(self, request):  # the app's own call and its own request type
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        # The app sends two user messages (the table, then the player's); Model takes one, so they are joined, and
        # the eval measures a request that differs from production by this join.
        user = "\n\n".join(m.content for m in request.messages if m.role == "user")
        reply = await model.generate(system=system, user=user)
        if reply.stop_reason != "end_turn":  # text() and json() refuse anything else; an adapter must too, or a
            raise RuntimeError(f"the reply did not finish: {reply.stop_reason}")  # cut-off reply scores as a wrong answer
        return AppReply(text=reply.content, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens)

    async def aclose(self):
        await model.aclose()


router = Router(MeteredClient(RecordedClient(), ...))  # the app's own code, unchanged, handed this client
```

`recording=` is read relative to the rung file, as every path an eval file hands `Model` is; `RECORD=1` records it
again. `generate` returns the completion with the attributes above, and a structured reply passes the app's
`json_schema` directive as `response_format=`. Where the app wants only the reply's text, or a pydantic object
validated from it, `Model.text(user, system=...)` and `Model.json(user, Schema, system=...)` are the shorter calls.
A call made inside a runner counts as the runner's spend.

The limit: `Model.generate` takes one system message and one user message. An app that sends more joins them, as
above, and says so. A multi-message turn with tool calls is recorded by the harness's turn recording
(`Model(turn_recording=...)` and its `turn_session`, on Anthropic only), not by this adapter.

## Recording

A recording is made once, live, with `OPENROUTER_API_KEY` set, and replaces the rung's file in its `recordings/`
(`RECORD=1` records afresh; it never appends to an old recording). From this directory:

```bash
RECORD=1 python -m threetears.evals 1-triage/triage.toml   # 1-triage/recordings/triage-toml.jsonl
RECORD=1 python -m threetears.evals 1-triage/triage.py     # 1-triage/recordings/triage.jsonl
RECORD=1 python -m threetears.evals 2-replies/replies.py   # 2-replies/recordings/replies.jsonl and judge.jsonl
RECORD=1 python -m threetears.evals 3-invoices/compare.py  # 3-invoices/recordings/invoices.jsonl, which 3a to 3e replay
python -m threetears.evals 3-invoices/grid.py              # 3-invoices/recordings/invoices-high.jsonl, once deleted
RECORD=1 python -m threetears.evals 4-booking/agent.py     # 4-booking/recordings/agent.jsonl
```

The grid's high-effort recording is made without `RECORD=1`: delete `invoices-high.jsonl` and run `grid.py` with the
key set, and only the missing file is recorded live while the low-effort cells replay. `RECORD=1` would record
`invoices.jsonl` afresh too, and pay for it again.

Rung 5 records differently: with `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` set, and no `RECORD=1`, it
records whatever is missing and replays the rest ([`5-writer/recordings/README.md`](5-writer/recordings/README.md)).
Rung 3f's harness door needs `ANTHROPIC_API_KEY` alone, and records every run, since each is a new campaign.

`RECORD=1` with no key is refused: a recording is of a live run. Record again whenever a rung's prompt, cases or
request settings change: the rung's test checks the recording still covers every prompt the rung sends before it
replays, and names each one that changed.

[The ladder](../docs/ladder.md) walks every rung through its code, with what each report said and its lesson;
[Serving](../docs/serving.md#keep-your-runs-and-read-them-back) covers keeping your runs in a file of your own.

## Not covered by an example yet

These are documented in the guides:

- Judge reliability and evidence tiers:
  [Reading reports](../docs/reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers).
- Declaring a campaign (one question, its arms, the aspects it is measured on and the floors that warn), and when
  exploring without one is the right call:
  [Measuring soundly](../docs/measuring-soundly.md#declare-the-campaign-its-question-arms-and-aspects).
- A campaign over several aspects, its one page and the person's recorded call:
  [Reading reports](../docs/reading-reports.md#the-campaigns-first-screen) and
  [Recording the call](../docs/adopting-a-host.md#recording-the-call).
- Budgets and spend caps: [Cost and budgets](../docs/cost-and-budgets.md).
- Conversations with a simulated user, generating cases, storage, and writing your own host:
  [Adopting the engine](../docs/adopting-a-host.md).
- How a bar is decided: [Reading reports](../docs/reading-reports.md#methods) and
  [Measuring soundly](../docs/measuring-soundly.md#variance-k-and-how-many-cases).
