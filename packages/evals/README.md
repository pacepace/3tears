# 3tears-evals

Evaluate an LLM-backed product the way you would run an experiment: declare the levers you can
change, run trials of each variant against a corpus of cases, grade each trial with code checks and
model judges, and read an analysis report that says which variant is better, by how much, at what
cost — and when the evidence cannot tell.

**Use it when** you have a feature built on a model (a classifier, an extractor, an assistant) and a
change you want to make to it — a new prompt, a different model, a new setting — and you want evidence,
not a hunch, that the change is better. Start with one function call; grow into a full integration with
your own store, launch path and reports when you need to compare runs over time.

**Status: being extracted.** This package was cut from a production app's in-tree eval engine and is
being reshaped into a library any app can adopt. Its public API will change without notice until the
first release that says otherwise.

## Install

```bash
pip install 3tears-evals                 # the engine, the in-memory store, run_eval and the CLI
pip install "3tears-evals[vega]"         # + the Vega-Lite chart renderer's PNG/SVG rasteriser
pip install "3tears-evals[fastmcp]"      # + the FastMCP transport for the agent tool catalogue
```

Python 3.14+.

## The mental model

```
  template  ──▶  cases  ──▶  run (one arm = one variant × every case × k repeats)
                                │
                                ▼
                             results, graded by measures (code) and judged dimensions (an LLM judge)
                                │
  campaign (the runs you compare)  ──▶  analysis bundle (the numbers)  ──▶  report
```

| Term | Meaning, with a support-ticket triage classifier as the example |
|---|---|
| **case** | One input and what a good answer looks like: a ticket and its right queue. |
| **template** | The blueprint the cases belong to: what is tested and how it is scored. |
| **variant** | One complete configuration under test: "prompt v2 on model-x". |
| **arm** | One variant as a contestant. A run measures exactly one arm. |
| **run** | One arm played over every case, `k` times each: 40 tickets × 3 = 120 trials. |
| **result** | One trial (one case, one repeat) with every grade it got. |
| **measure** | A number code computes about a result: did the queue match, latency, cost. |
| **judged dimension** | A quality an LLM judge scores against a rubric, such as tone. |
| **campaign** | The runs you want compared — v1 against v2 — analysed together. |
| **report** | The document you read: tables, charts, and findings if an analysis was generated. |

Every other term (kind, lever, apparatus, cell, scope, stratum, ...) is defined in
**[Concepts](docs/concepts.md)**, with the full diagram.

## Rung zero: one call

A function to test, cases to test it on, and code that grades an answer are enough:

```python
from threetears.evals.quick import run_eval

async def extract_total(case: dict) -> float: ...

def exact(case: dict, total: float) -> bool:
    return total == case["total"]

summary = await run_eval(cases, extract_total, [exact], scope_id="dev", k=2)
print(summary.render())
```

`run_eval` builds the rest — a kind over the function, a host with one measure per scorer, the
in-memory store — launches one run through the engine's own launch path, and returns its
`EvalSummary`. A candidate that raises fails its cell (one case at one repeat); a scorer that raises
excludes that cell.

**A classifier** is graded by each case's expected label rather than by a scorer. Pass `expected=`, a
function from a case to the label a correct answer gives, and `classify` returns a label:

```python
summary = await run_eval(cases, classify, scope_id="dev", expected=lambda case: case["expected"], k=2)
```

What you get back:

- Each cell lands the core `match` and `confusion_cell` measures a classifier kind lands, so the summary
  carries the confusion matrix (`summary.confusion`, one `ConfusionCount` per expected and predicted
  label) and each label's counts, precision and recall with their Wilson intervals, and F1
  (`summary.labels`, one `LabelStatistics` per label); `render()` prints both.
- `match`'s mean is the share of answers that matched; the analysis derives `accuracy` from it.

The rules it holds:

- An answer that is not a non-blank string (`None`, `""`, a number) is counted under a predicted label of
  its own, `UNUSABLE_ANSWER`, and never matches; any other string is a label exactly as written, so
  `"positive "` is not `"positive"`.
- `run_eval` refuses an `expected=` that raises or gives a case a blank, non-string or `UNUSABLE_ANSWER`
  label.
- Scorers may run beside `expected=`, except one named `match`, `confusion_cell` or `accuracy`.
- A classifier's expected labels are part of its case set: two calls share a template only when they
  expect the same labels.

**Keeping runs to compare.** Pass `host=callable_host(scorers)` (`callable_host()` for a classifier with
no scorers) to keep the store and compare several candidates' runs, or your own host. Your own host must
declare a measure per scorer and a contract for the callable kind — `CALLABLE_KIND_CONTRACT`, or a
`KindContract(CALLABLE_KIND, seats=...)` seating only apparatus of your own that the runs read, never the
judge, the simulator or the spend ceiling (`CALLABLE_UNSEATED`) — or `run_eval` refuses it, since without
one every such run's blank judge and simulator read as unrecoverable and no two of them compare. A
classifier's `match` and `confusion_cell` are core measures, so a host declares neither.

[`examples/rung_zero.py`](examples/rung_zero.py) is the whole thing in one file: a sentiment classifier
graded by its expected labels, beside one scorer.

## Comparing two variants

<!-- TODO(onramp): filled in when examples/compare_two_prompts.py lands -->

## Grading with an LLM judge

When no code can grade an answer (is it helpful? does it stick to its source?), give `run_eval` a
`Judge`: a completion client, the model it calls, and a rubric. The engine's own judge scores each answer
on each dimension (1-5 by default) and records the judge's spend as the client prices it. Scorers and
`expected=` still work beside it.

```python
from threetears.evals.quick import Judge, run_eval

judge = Judge(
    client=my_client,            # any CompletionClient; you own it, and the run never closes it
    model="claude-haiku-5-5",
    rubric={"helpful": "Resolves the question.", "grounded": "Claims only what the policy says."},
    case_material=lambda case: f"Policy:\n{POLICY}\n\nQuestion: {case['question']}",
)
summary = await run_eval(cases, answer, [concise], judge=judge, scope_id="faq")
print(summary.render())   # adds "answer.helpful (judged 1-5): mean ..." and "judge spend: $..."
```

A bare rubric name is placed under the judge's `context` (`answer` by default), so `helpful` is reported
as `answer.helpful`. Only the judge's spend reaches the summary: the candidate is your code, so the engine
never sees what it spent. `examples/llm_judge.py` is the whole thing in one file, including a small
adapter from the `anthropic` SDK; it calls Claude when `ANTHROPIC_API_KEY` is set and runs labelled
offline stand-ins otherwise.

## What's in the package

| Subpackage | What it holds |
|---|---|
| `threetears.evals.contracts` | the data models, the host contract an app implements, identity and scoring rules |
| `threetears.evals.run` | the trial loop, judges, simulated users, budgets and metering |
| `threetears.evals.gen` | case and rubric generation |
| `threetears.evals.analysis` | the analysis bundle, report generation and charts |
| `threetears.evals.storage` | the storage adapters the engine ships: the in-memory reference store |
| `threetears.evals.testing` | conformance kits an app runs in its own test suite: the store kit |
| `threetears.evals.quick` | the batteries: `run_eval` in one call, and the `python -m threetears.evals` command line |
| `threetears.evals.ops` | typed operations over a host, and one job contract for long work |
| `threetears.evals.actions` | the action catalogue every transport mounts: `evals` and `evals_admin` |
| `threetears.evals.transports.fastmcp` | the catalogue as FastMCP tools (extra `fastmcp`) |

Import from those roots and from `threetears.evals.contracts.host`, never from a module below
them. Every engine type a public signature hands you — a protocol you implement, a value you
receive, an exception you catch, a literal you annotate with — is exported from one of those roots.

## Where to go next

| If you want to... | Read |
|---|---|
| look up a term, or see how the pieces fit | [Concepts](docs/concepts.md) |
| build a good classifier eval set: the labels, the kinds of case a set needs (boundaries, lookalikes, contrast pairs, context), how many, and how to read the results. Start here if you have not built an eval before | [Designing a classifier eval set](docs/designing-classifier-evals.md) |
| run evals from a terminal, or under your own CLI | [The command line](docs/command-line.md) |
| wire the engine into your app: host, store, kind, launcher, worlds, cassettes | [Adopting the engine](docs/adopting-a-host.md) |
| know what a launch will cost, and what stops it | [Cost and budgets](docs/cost-and-budgets.md) |
| read a campaign's report, strata and evidence tiers, or draw its charts | [Reading reports](docs/reading-reports.md) |
| let an agent launch and read evals over MCP | [Driving it from an agent](docs/agents-and-mcp.md) |
| see a complete host in code | [`tests/fixtures/courierhost/`](tests/fixtures/courierhost/__init__.py) (minimal), then [`tests/fixtures/toyhost/`](tests/fixtures/toyhost/README.md) (every shape) |
