# Examples

The ladder comes first: one directory per rung, each in its own domain and recorded on cheap models through
OpenRouter, so every rung runs offline and free from the recordings beside it. Climb it in order; each rung adds one
idea to the one before.

## The ladder

| Rung | Where | Why you would climb it |
|---|---|---|
| 0 | [`1-triage/triage.toml`](1-triage/triage.toml), after [Building an inventory](../docs/building-an-inventory.md) | Write down what your app must never get wrong, then run it with one TOML file. |
| 1 | [`1-triage/triage.py`](1-triage/triage.py) | Point it at your own function: 15 lines of Python. |
| 2 | [`2-replies/replies.py`](2-replies/replies.py) | When no code can grade the answer, let a model judge it, and check the judge against a person. |
| 3 | [`3-invoices/extract.py`](3-invoices/extract.py) | Changing a prompt? Compare it against the current one, and get a report you can hand to a reviewer. |
| 4 | [`4-booking/agent.py`](4-booking/agent.py), then [`serve.py`](4-booking/serve.py) | Is your agent doing the right thing to the world, not only saying it? Then serve your evals to the team. |

One door runs every rung: `python -m threetears.evals <file>`, from any directory. The paths a rung names (its
cases, its recordings, a judge's labels) are read beside its file; the store (`.evals/evals.sqlite`) and the report
(`.evals/reports/<name>/report.html`) are written under the directory you run from. In a checkout, from
`packages/evals/`, after `uv sync`:

```bash
uv run python -m threetears.evals examples/1-triage/triage.toml    # rung 0
uv run python -m threetears.evals examples/1-triage/triage.py      # rung 1
uv run python -m threetears.evals examples/2-replies/replies.py    # rung 2
uv run python -m threetears.evals examples/3-invoices/extract.py   # rung 3
uv run python -m threetears.evals examples/4-booking/agent.py      # rung 4
```

The one exception is rung 4's [`serve.py`](4-booking/serve.py), which runs from `examples/4-booking/` because it imports `agent.py` as a module.

A recording that exists is replayed: no key, no network, no cost. `RECORD=1` with `OPENROUTER_API_KEY` set records it
again, live (`RECORD=1 uv run python -m threetears.evals examples/1-triage/triage.py`). A recording holds one model at one
temperature, and an email the recording does not hold is excluded from the run by name, never sent to a model.

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
own model prints the self-judging warning under the scores, a warning no quiet print drops. Its four `Example`s are
hard negatives in pairs, one that must fail and one that looks like it and must pass; a run stores them for a judge
campaign to measure the judge on, and calls no model to do it.

**A judge is a model too, so check it against a person.** The recorded judge passes all 40 replies on all three
dimensions. That is what an unchecked judge looks like, not proof the replies are right: read a few, and you will find
claims the policy never makes. The run says so in the line it never quiets:

```
judge: 0 labels: judge uncalibrated (examples/2-replies/labels.csv is not there yet); next step: rate about 20 outputs in the to_label.csv the report writes, without looking at the judge's verdict
```

Calibrating it is yours to do; no person's labels ship with the rung.
1. Run it. The report directory (`.evals/reports/replies/`) holds `to_label.csv`: every judged reply on every
   dimension, with the case it answered and the judge's verdict withheld. [`to_label.csv`](2-replies/to_label.csv)
   beside the example is that file, as the recorded run wrote it.
2. Fill in `label` (`pass` or `fail`), and `rater` with your name, for about 20 rows of each dimension, without
   opening the report's scores. Save it beside `replies.py` as `labels.csv`.
3. Run it again. `Judge(labels="labels.csv")` reads your labels back, pairs each with the judge's score of the same
   reply (by its fingerprint, so a replayed run pairs with the rows the recorded one wrote), and prints, per
   dimension, how many labels paired, κ with its lower bound, and the evidence tier it earns. Below 20 labels the
   tier says the judge is not yet calibrated; reword a rubric dimension and its labels no longer pair.

Recorded: 40 reply calls on `deepseek/deepseek-v4-flash` ($0.0017) and 120 judge calls on
`google/gemini-2.5-flash-lite` ($0.0151), the runner's and the judge's in recordings of their own
(`recordings/replies.jsonl`, `recordings/judge.jsonl`).

### Rung 3: two prompts compared, `3-invoices/`

[`extract.py`](3-invoices/extract.py) reads 30 invoices ([`invoices.jsonl`](3-invoices/invoices.jsonl)) in three
strata (clean, multi-currency, noisy OCR) with two prompts, as `runner={"v1": model.prompt(V1, Invoice), "v2": ...}`;
the first is the control, and v2 is read against it on every reading: the delta, its interval and a verdict. A field
grader per field, `grader("total", margin=0.05)` declaring a margin for equivalence (a case whose `total` is `null`
expects none stated), and `@grader(guard=0.02) no_invented_total`, a guardrail decided apart from the rest. `result.write("report")` writes the HTML report for a reviewer, and `gate()`
fails the run in CI on a regression or a guardrail not shown held. Its closing comment names arms as `k=v` pairs for a
grid of factors, read against another control with `result.against()`.

What the recorded run said: no arm separated, and it exits 4. The report reads

```text
**Leads only: no arm separated from the control.**
> Guardrails left undecided, so not known to be safe: No invented total score for candidate=v2.
```

with the guardrail's row saying why: `undecided: not shown held, so not known to be safe (the interval on the
difference, [-0.1343, 0.1343], reaches both sides of -0.02 (the declared margin), so the arm is shown neither within
it nor beyond it)`. That is the lesson: rather than declare a winner, the report names the question it could not
answer, and what would answer it is more hard-negative cases, invoices that print no amount due (7 of the 30). One
v2 trial is `failed: a model the candidate runs failed or refused`, its recorded error `the candidate raised
ValidationError: 1 validation error for Invoice` on `Invalid JSON`: malformed JSON from the model, counted against v2.
[The ladder](../docs/ladder.md#rung-3-two-prompts-compared) quotes the rest.

Recorded on `deepseek/deepseek-v4-flash` at `max_tokens=8192`: 180 calls, $0.0105 (`recordings/invoices.jsonl`).

### Rung 4: an agent graded on the world it leaves, `4-booking/`

[`agent.py`](4-booking/agent.py) books meetings: each request ([`requests.jsonl`](4-booking/requests.jsonl)) seeds one
day's calendar, `World(Day, tools=[find_free, book])` hands the agent two tools in a JSON tool loop over
`model.json`, and the graders are goal checks on the day it left, beside the did-nothing control that only the first
check beats. [`serve.py`](4-booking/serve.py) serves the eval's runs to the team: `as_host`, an MCP server in one
call (`mount_fastmcp(host(), scope="booking").run()`, the `[fastmcp]` extra, read-only;
[Serving](../docs/serving.md#serving-the-runs-to-the-team)) and
[the command line](../docs/serving.md#the-host-commands) (`python -m threetears.evals ls --host serve:host --scope booking`),
both from `4-booking/`, where `agent.py`'s runs are stored when it is run there.

Recorded on `deepseek/deepseek-v4-flash`: 100 calls, $0.0080 (`recordings/agent.jsonl`).

## Running them

Every rung replays the recordings beside it when it runs with no key, so it calls nothing and costs nothing, and its
output is a real model's, as of that recording. There is no scripted stand-in for a model anywhere in them.

**Outside a checkout** (once 0.68.0 is on PyPI and you installed `3tears-evals` with pip): the examples are
not part of the installed package, so fetch this whole directory from the repository, at the tag you installed (the
[README's install step](../README.md#install) names it). This fetches only the examples:

```bash
git clone --depth 1 --branch <tag> --filter=blob:none --sparse https://github.com/pacepace/3tears
cd 3tears && git sparse-checkout set packages/evals/examples && cd packages/evals/examples
```

Keep each rung's directory together: a rung reads its cases and replays its `recordings/` from beside itself. Then
`python -m threetears.evals 1-triage/triage.py`, or with `export OPENROUTER_API_KEY=...` and `RECORD=1` first, any
rung live.

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

## Recording

A recording is made once, live, with `OPENROUTER_API_KEY` set, and replaces the rung's file in its `recordings/`
(`RECORD=1` records afresh; it never appends to an old recording). From this directory:

```bash
RECORD=1 python -m threetears.evals 1-triage/triage.toml   # 1-triage/recordings/triage-toml.jsonl
RECORD=1 python -m threetears.evals 1-triage/triage.py     # 1-triage/recordings/triage.jsonl
RECORD=1 python -m threetears.evals 2-replies/replies.py   # 2-replies/recordings/replies.jsonl and judge.jsonl
RECORD=1 python -m threetears.evals 3-invoices/extract.py  # 3-invoices/recordings/invoices.jsonl
RECORD=1 python -m threetears.evals 4-booking/agent.py     # 4-booking/recordings/agent.jsonl
```

`RECORD=1` with no key is refused: a recording is of a live run. Record again whenever a rung's prompt, cases or
request settings change: the rung's test checks the recording still covers every prompt the rung sends before it
replays, and names each one that changed.

[The ladder](../docs/ladder.md) walks every rung through its code, with what each report said and its lesson;
[Serving](../docs/serving.md#keep-your-runs-and-read-them-back) covers keeping your runs in a file of your own.

## Not covered by an example yet

These are documented in the guides:

- Judge reliability and evidence tiers: [Reading reports](../docs/reading-reports.md). Results by kind of case
  (strata) are read in [`3-invoices/`](3-invoices/extract.py), and explained in
  [Reading reports](../docs/reading-reports.md#results-by-kind-of-case-strata).
- Declaring a campaign's design (axes, question, bar), and when exploring without one is the right call:
  [Measuring soundly](../docs/measuring-soundly.md#start-from-the-question).
- Budgets and spend caps: [Cost and budgets](../docs/cost-and-budgets.md).
- Conversations with a simulated user, generating cases, storage, and writing your own host:
  [Adopting the engine](../docs/adopting-a-host.md).
- How a bar is decided: [Reading reports](../docs/reading-reports.md#methods) and
  [Measuring soundly](../docs/measuring-soundly.md#variance-k-and-how-many-cases).
