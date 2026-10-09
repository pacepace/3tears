# 3tears-evals

3tears-evals measures what a change to an LLM-backed feature actually does. Change a prompt, a model or a
setting, run each version over the same cases, and see what moved: how often it's right on each kind of
case, how an LLM judge scores its answers, what it costs, and what it did to the systems it acts on. Each
figure comes with its uncertainty, so you can tell a real difference from noise. Most changes trade one
thing for another (a cheaper model that misses more edge cases, a stricter prompt that refuses more), and
the report lays those trade-offs side by side so the decision gets made with all of them in view.

It can also **model a world**: the state your feature acts on (a room's lights, a calendar, an order
queue) and the tools that change it. Each case starts the world in a known state, the model acts through
the tools, and the engine reads the world back and measures what the model changed. That evaluates an
agent by its effect on the world as well as by what it says. See
[Modeling a world](#modeling-a-world-and-measuring-the-models-impact).

Use it when you have a feature built on a model (a classifier, an extractor, an assistant, an agent) and
want to know what a change will do before you ship it. It starts as one function call and grows into a
full integration with your own store, launch path and reports.

> **The public API is still changing.** This package was extracted from a production app's eval engine,
> and its API will change without notice until a release says otherwise.

## Contents

- [Getting started](#getting-started)
- [The examples](#the-examples)
- [The mental model](#the-mental-model)
- [Using it in your code](#using-it-in-your-code)
- [What's in the package](#whats-in-the-package)
- [Where to go next](#where-to-go-next)

## Getting started

1. **Install.** In a checkout of this repo, `uv sync` at the repo root installs the package and everything
   the examples use, including the `anthropic` SDK. Outside the repo (Python 3.14+):

   ```bash
   pip install 3tears-evals anthropic       # the engine, plus the SDK the examples call Claude through
   pip install "3tears-evals[vega]"         # adds SVG and PNG chart rendering
   pip install "3tears-evals[fastmcp]"      # adds the FastMCP transport, for driving evals from an agent
   ```

2. **Run the first example offline.** Every example runs with no API key: small scripted stand-ins play the
   model, so you see the real shape of the output. Their numbers say nothing about any model.

   ```bash
   uv run python packages/evals/examples/rung_zero.py
   ```

3. **Add your API key to run them live.** Put an Anthropic API key in `packages/evals/examples/.env`:

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

   Git ignores `.env` files, so the key stays out of commits. Pass the file to `uv run`:

   ```bash
   uv run --env-file packages/evals/examples/.env python packages/evals/examples/compare_two_prompts.py
   ```

   Outside the repo, `export ANTHROPIC_API_KEY=...` works the same way. Each example prints
   `Running against Claude (<model>).` when it calls the API and `running OFFLINE` when it doesn't. They use
   the cheapest Haiku models over a dozen or so cases, so a live run costs cents.

4. **Work through [the examples](#the-examples) in order,** then use [Using it in your code](#using-it-in-your-code)
   as the reference.

## The examples

Each example is one short file that shows one capability, and each builds on the one before.
[`examples/README.md`](examples/README.md) gives the question each one answers and what it adds.

| Example | What it shows |
|---|---|
| [`rung_zero.py`](examples/rung_zero.py) | One function evaluated against cases, graded by expected labels and a scorer. |
| [`llm_judge.py`](examples/llm_judge.py) | Open-ended answers graded by an LLM judge against a rubric. |
| [`compare_two_prompts.py`](examples/compare_two_prompts.py) | Two prompts measured on the same cases, with a verdict on whether the difference is real. |
| [`compare_two_models.py`](examples/compare_two_models.py) | A cheaper model weighed on accuracy against cost. |
| [`prompt_x_model.py`](examples/prompt_x_model.py) | Two things varied at once (prompt and model), with results for every combination. |
| [`cassettes.py`](examples/cassettes.py) | Tool results captured once and replayed, so every arm sees the same ones. |
| [`world.py`](examples/world.py) | A modeled world (a room's light), with the model's impact on it measured. |
| [`reports.py`](examples/reports.py) | A finished campaign written out as verdicts, Markdown, HTML and charts. |
| [`llm_analysis.py`](examples/llm_analysis.py) | A model writing the analysis from frozen, fingerprinted evidence. |

## The mental model

```
  template  ──▶  cases  ──▶  run (one arm = one variant × every case × k repeats)
                                │
                                ▼
                             results, graded by an LLM judge (judged dimensions) and by code (measures)
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
| **judged dimension** | A quality an LLM judge scores against a rubric, such as tone. Most grading happens here. |
| **measure** | A number code computes about a result, when there is something exact to check: did the queue match, cost. |
| **world** | The state a model acts on, seeded for each case and read back after: a room's light and daylight. |
| **campaign** | The runs you want compared — v1 against v2 — analysed together. |
| **report** | The document you read: tables, charts, and findings if an analysis was generated. |

Every other term (kind, lever, apparatus, cell, scope, stratum, ...) is defined in
**[Concepts](docs/concepts.md)**, with the full diagram.

## Using it in your code

### Your first eval

An eval needs three things: cases, the function you're evaluating, and a way to grade its answers. Here
the function sorts product reviews into `positive`, `negative` or `neutral`, and each case says which
label is right:

```python
import asyncio

from threetears.evals.quick import run_eval

# The cases: each one an input, plus what a right answer looks like.
CASES = [
    {"review": "Works perfectly, I love it.", "expected": "positive"},
    {"review": "It broke after two days.", "expected": "negative"},
    {"review": "It arrived on Tuesday.", "expected": "neutral"},
]


# What you're evaluating: any async function from a case to an answer. This is where your
# prompt and model call go.
async def classify(case: dict) -> str:
    return await my_model(f"Classify this review as positive, negative or neutral: {case['review']}")


async def main() -> None:
    summary = await run_eval(
        CASES,
        classify,
        expected=lambda case: case["expected"],  # the right answer for each case, so code can grade it
        scope_id="dev",  # where the runs are stored; runs you want to compare share a scope
        k=2,  # ask twice per case, since a model's answer can change between calls
    )
    print(summary.render())


asyncio.run(main())
```

The summary reports how often the answer was right, a confusion matrix of which labels got mistaken for
which, and each label's precision and recall with their intervals. A function that raises counts as that
case failing; the run carries on.

This eval grades with code because a review's label is either right or wrong. Most answers aren't like
that (is this reply helpful? does it stick to the policy?), and an LLM judge grades those against a rubric:
see [Grading with an LLM judge](#grading-with-an-llm-judge). Code grading is the special case for answers
with something exact to check: a label, a number, a field. Besides `expected=`, any function
`(case, answer) -> bool | float` passed in a list after the candidate becomes a score of its own.

The engine also reads your functions' names and first docstring lines, for example as a score's
description. [What the engine reads from your code](docs/concepts.md#what-the-engine-reads-from-your-code)
lists each, and how to set it explicitly instead.

**Classifier details.** An answer that isn't a non-blank string (`None`, `""`, a number) counts under its
own predicted label, `UNUSABLE_ANSWER`, and never matches. Any other string is compared exactly, so
`"positive "` is not `"positive"`. The summary carries the confusion matrix as `summary.confusion` and each
label's statistics as `summary.labels`. Scores may run beside `expected=`. No score may take the name of
an engine core measure (`match`, `accuracy`, `score`, `f1`, `cost_usd` and the rest): `run_eval` refuses it
and asks you to rename the function.

**Keeping runs to compare.** Pass `host=callable_host(scorers)` (`callable_host()` when there are no
scorers) and reuse it, so several runs share one store. Your own host must declare a measure per scorer and
a contract for the callable kind (`CALLABLE_KIND_CONTRACT`, or a `KindContract(CALLABLE_KIND, seats=...)`
that seats only apparatus of your own, never the judge, the simulator or the spend ceiling:
`CALLABLE_UNSEATED`), or `run_eval` refuses it.

Example: [`examples/rung_zero.py`](examples/rung_zero.py).

### Grading with an LLM judge

Most answers are graded this way: no code can say whether a reply is helpful or sticks to its source, but
a model reading it against a rubric can. Give `run_eval` a `Judge`: a completion client, the model it
calls, and a rubric. The engine's own judge scores each answer
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
summary = await run_eval(
    cases, answer, [concise], judge=judge, intent="Answer a customer's question from the policy.", scope_id="faq"
)
print(summary.render())   # adds "intent: ...", "answer.helpful (judged 1-5): mean ..." and "judge spend: $..."
```

The judge reads `intent=` beside every answer as what each case asks, so its wording can move the scores.
Left out, it is the first line of the candidate's docstring, and `render()` says so:
`intent (from answer's docstring): ...`.

A bare rubric name is placed under the judge's `context` (`answer` by default), so `helpful` is reported
as `answer.helpful`. The judge's spend reaches the summary as its client prices it; a candidate that calls
a model reports its own by returning an `Answer`, as in
[Comparing models: accuracy and cost](#comparing-models-accuracy-and-cost).

Example: [`examples/llm_judge.py`](examples/llm_judge.py), which includes a small adapter from the
`anthropic` SDK to the engine's completion client.

### Comparing two versions

`compare` runs each candidate over the same cases as one arm, files the runs as one campaign, and tests
every arm against the one you name as the control. It returns each arm's summary and the campaign's report.

```python
from threetears.evals.quick import compare

result = await compare(
    CASES,
    {"baseline": classify_v1, "candidate": classify_v2},  # arm name -> async candidate
    expected=lambda case: case["label"],                   # or scorers=[...]
    control="baseline",
    scope_id="dev",
    k=2,
)
print(result.render())  # "Contrasts against the control": each difference, its interval, a Holm-adjusted p, a verdict
```

Each row of that table is one arm against the control on one reading, and says:

- **Delta**: the arm's mean minus the control's, over the cases the test read. The test is a **paired
  t-test on per-case means** (each case's repeats averaged first) over the cases both arms ran; when they
  share fewer than two, Welch's t statistic on the conservative `min(n) − 1` degrees of freedom. A case only
  one arm ran is left out of a paired test, and "Cases tested" says how many.
- **Interval on delta**: where the true difference plausibly lies, widened for the number of rows tested
  together so that all of them hold at once, 95% of the time.
- **Hedges' g**: the difference in standard deviations, corrected for small samples.
- **p (Holm-adjusted)**: corrected over every row in its family: one per declared question, or, when none
  is declared, every reading on a merit axis across the campaign. Use only this p, never a raw one.

The verdict is one of five:

- **improved on the control** / **regressed from the control**: the adjusted p is below 0.05. If it says
  *immaterial*, the move is real but smaller than the measure's declared margin. Don't act on it.
- **equivalent to the control**: an equivalence test (TOST) shows the difference inside the measure's
  declared margin. This is the only verdict that says two arms are alike, so it is how "the cheaper model is
  good enough" gets shown. It needs a margin (`materiality_threshold`) on the measure.
- **not separated from the control**: the cases could not tell the arms apart. It does not mean they are
  equal. Add cases (above all hard ones), or declare a margin so equivalence can be tested.
- **untested**: no test could decide (fewer than two cases on a side, or no spread over too few cases for an
  exact test to reach 0.05). The row says why.

[Reading a comparison](docs/reading-reports.md#reading-a-comparison) has the details.
`result.arms["candidate"]` is that arm's `EvalSummary`, and `result.campaign_id` names the campaign holding
every run.

Example: [`examples/compare_two_prompts.py`](examples/compare_two_prompts.py).

### Comparing models: accuracy and cost

Asking whether a cheaper model is good enough means weighing what each gets right against what it
costs. The engine cannot see what a plain candidate spends, so have the candidate return an `Answer`:
the label plus the call's tokens and dollars. Each arm's summary then prints its `candidate spend`, and
the report tests the arms' spend (`production_replicating_cost`, the candidate's own) against the control
the same way it tests their accuracy.

```python
from threetears.evals.quick import Answer, compare

async def classify_cheaper(case: dict) -> Answer:
    label, tokens_in, tokens_out = await call_model(case)   # your call
    return Answer(label, model="cheap-model", input_tokens=tokens_in, output_tokens=tokens_out,
                  cost_usd=(tokens_in * 0.10 + tokens_out * 0.50) / 1e6)

result = await compare(CASES, {"current": classify_current, "cheaper": classify_cheaper},
                       expected=lambda case: case["label"], control="current", scope_id="dev", k=2)
print(result.render())  # verdicts on accuracy and on spend
```

A field left `None` is unreported, not zero. A candidate that returns a plain value still works, and
reports no spend: the report then says cost was not measured rather than charting zeros.

Example: [`examples/compare_two_models.py`](examples/compare_two_models.py).

### Varying several things at once

Changes rarely come one at a time: you might try two prompts on three models at two temperatures. Name
each thing you vary (a **factor**, such as `model` or `prompt`) and key each arm by its level of every
factor. `compare` runs every combination as its own arm, so the results have as many dimensions as the
things you varied, and you can read any combination against any other.

```python
result = await compare(
    CASES,
    {(model, prompt): make(model, prompt) for model in (OLD, NEW) for prompt in ("v1", "v2")},
    factors=("model", "prompt"),          # each key is (model level, prompt level)
    control=(OLD, "v1"),
    expected=lambda case: case["queue"],
    scope_id="dev",
    k=2,
)
on_old = result.contrasts("accuracy")                       # v2 against v1 on OLD is in here
on_new = result.against((NEW, "v1")).contrasts("accuracy")  # v2 against v1 on NEW, no re-run
```

`compare` takes any number of factors, as long as `model` is one of them. The report names every arm by
all of its levels (`callable.prompt=v2, model=...`) and tests each one against the control.
`against(arm)` re-reads the same runs against any other combination, and each reading corrects its own
contrasts. Two things aren't tested yet: a factor's effect pooled over all the others (a main effect), and
whether factors interact. The pivot read (`ops.scope_pivot`) averages a scope's results over any two
factors, without a significance test. It says what a cell pools that is not one quantity: a cost cell names
the role sets its dollars covered, a cost cell that pools replayed results with live ones is withheld, and a
cell grouped on `variant_key` that spans an identity-version bump names the versions. Whether a lever actually took effect, rather than just being set, is a
[mechanism check](docs/reading-reports.md#did-a-lever-take-effect-mechanism-checks-and-observed-mechanisms).

Example: [`examples/prompt_x_model.py`](examples/prompt_x_model.py).

### Capturing & replaying tool results

A candidate that calls tools (a search, a price lookup) is compared fairly only when every arm got the
same tool answers. Declare the tools as plain functions (`tools=`) and the candidate is called as
`candidate(case, tools)`. Capture one run with the tools live, then replay that recording to every arm:
the tools are not called, and every arm faces identical answers. A cassette records the tools only; a model
behind the candidate still runs live. A replay that asks something the capture never recorded excludes
that cell as the rig's failure, and the tool is not called live.

```python
host = callable_host([correct])
capture = await run_eval(cases, reader, [correct], scope_id="s", host=host, k=1,
                         tools={"search": search}, cassette_mode="capture")
comparison = await compare(cases, {"a": reader_a, "b": reader_b}, [correct], control="a",
                           scope_id="s", host=host, tools={"search": search},
                           cassette_mode="replay", cassette_corpus_id=capture.run_id)
```

The replay reads the capture from the same host and scope.
[Adopting the engine](docs/adopting-a-host.md#cassettes-recording-and-replaying-tools) says how a replay
matches each ask.

Example: [`examples/cassettes.py`](examples/cassettes.py).

### Modeling a world and measuring the model's impact

Many features act rather than answer: an assistant that books meetings, a support agent that issues
refunds, a home assistant that controls the lights. What such a model says matters less than what it did,
so 3tears-evals lets you model the world it acts on and measure its impact there.

1. **Declare the world.** A `World` is a set of `Dimension`s, each a piece of state with a schema (the
   room's `light`, whether it's `daylight`), and the `WorldTool`s that change them.
2. **Seed it per case.** Each case starts the world in a known state (`seed=`). Every cell gets a fresh
   copy, set before the model's first turn.
3. **Let the model act.** The candidate is handed the world's tools and changes the world only through
   them. Every call is recorded.
4. **Measure the impact.** After the last turn the engine reads the world back, and goal-state checks grade
   it: code over the end state (`state.light`), the calls made (`calls("room.switch_light")`) and the
   case's starting point (`variation.light`). They are code, so no judge is needed.

```python
def switch_light(room: dict, to: str) -> str:
    """Turn the room's light on or off."""   # the model sees this as the tool's description
    room["light"] = to
    return f"The light is now {to}."

ROOM = World("room", [Dimension("light", {"enum": ["on", "off"]}, "The lamp."),
                      Dimension("daylight", {"enum": ["dark", "bright"]}, "Whether the lamp is needed.")],
             tools=[WorldTool(switch_light, to={"enum": ["on", "off"]})])

summary = await run_eval(
    cases, assistant,                                       # assistant(case, room) acts through room's tools
    world=ROOM,
    seed=lambda case: {"light": case["light"], "daylight": case["daylight"]},
    goal_checks=[
        '(state.light == "on") == (state.daylight == "dark")',                # the room ended right
        'all(it.to != variation.light for it in calls("room.switch_light"))',  # and no needless switching
    ],
    scope_id="world",
)
```

The summary reports, per check, in how many cells the world ended as it should. `compare` takes the same
`world=`, `seed=` and `goal_checks=`, so two prompts or models can be compared on their impact. A world
run takes no `tools=` and no cassette: its tools are the world's own, and replaying them would skip the
change being measured. In a full integration a world can span several systems, record the events that
fired (`fired(...)`), and grade sessions observed in production: see
[Adopting the engine](docs/adopting-a-host.md).

Example: [`examples/world.py`](examples/world.py), which drives Claude through a tool-use loop.

### Writing reports to files

A campaign's report is a typed document, so a script can read its verdicts as data. It also writes out
for each reader: Markdown for a pull request, script-free HTML for a person, the evidence bundle its
numbers came from, and each chart as a Vega-Lite spec (SVG too, with the `[vega]` extra).

```python
from threetears.evals.analysis import report_html, report_markdown

verdicts = {(row["contrast"], row["reading"]): row["verdict"] for row in comparison.contrasts()}
Path("report.md").write_text(report_markdown(comparison.report))
Path("report.html").write_text(report_html(comparison.report))
```

[Reading reports](docs/reading-reports.md) says what each part of a report means.

Example: [`examples/reports.py`](examples/reports.py), which writes into `./eval-report/`.

### LLM-written analysis over frozen evidence

An analysis is written from one input only, the campaign's **analysis bundle**: every number code
computed, with a sha256 fingerprint. The model never types a figure. It names a reading, code fills in the
number from the bundle, and a reading the bundle lacks is refused. Save the bundle and you can regenerate
over the same evidence with another prompt; each analysis records the fingerprint it read, so two prompts
are compared fairly. Figures typed straight into a sentence are not checked; only cited readings are.

```python
bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
Path("bundle.json").write_text(bundle.to_json(indent=2))          # freeze
frozen = AnalysisContextBundle.from_json(Path("bundle.json").read_text())
assert frozen.fingerprint() == bundle.fingerprint()
analysis, _ = await generate_analysis(frozen, prompt=my_prompt, model=MODEL, client=writer,
                                      prompt_id="eval_analysis_gen", bundle_assembled_at=assembled_at,
                                      profile=comparison.host.profile)
analysis.generation.bundle_fingerprint  # == frozen.fingerprint()
```

Example: [`examples/llm_analysis.py`](examples/llm_analysis.py), which writes two analyses over one saved
bundle.

## What's in the package

| Subpackage | What it holds |
|---|---|
| `threetears.evals.contracts` | the data models, the host contract an app implements, identity and scoring rules |
| `threetears.evals.run` | the trial loop, judges, simulated users, budgets and metering |
| `threetears.evals.gen` | case and rubric generation |
| `threetears.evals.analysis` | the analysis bundle, report generation and charts |
| `threetears.evals.storage` | the storage adapters the engine ships: the in-memory reference store |
| `threetears.evals.testing` | conformance kits an app runs in its own test suite: the store and reader kits, and the completion-type check |
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
| know the rules the engine keeps, and why | [Principles](docs/principles.md) |
| understand why arms, levers, confounds, identity and the analysis are shaped as they are | [Design rationale](docs/design-rationale.md) |
| understand why a subject runs in a seeded world, and what the world contract enforces | [The world model](docs/world-model.md) |
| see what the field recommends for evals and what this engine took from it | [Prior art](docs/prior-art.md) |
| avoid the measurement traps campaigns have hit: variance, misleading metrics, rigs, judges | [Measuring soundly](docs/measuring-soundly.md) |
| find known gaps worth building next | [Open problems](docs/open-problems.md) |
| see each capability in one short file, in order | [The examples](examples/README.md) |
| see a complete host in code | [`tests/fixtures/courierhost/`](tests/fixtures/courierhost/__init__.py) (minimal), then [`tests/fixtures/toyhost/`](tests/fixtures/toyhost/README.md) (every shape) |
