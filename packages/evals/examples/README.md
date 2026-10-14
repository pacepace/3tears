# Examples

Each file here answers one question about your LLM-backed feature, and adds one capability. Read them in
order: each builds on the ones before it. Those that call a model call Claude when you set an API key, and replay a
recording of a live run when you do not, so every one runs offline and free out of the box.

## Running them

The examples that call a model run **live by default**: with `ANTHROPIC_API_KEY` set they call Claude
(`claude-haiku-5-5`, the only model they use) through the package's own completion client,
[`threetears.evals.providers`](../src/threetears/evals/providers/__init__.py). Without a key they replay
`recordings/<example>.jsonl`, what Claude answered when the run was last recorded, so they call nothing and cost
nothing, and their output is a real model's, as of that recording. There is no scripted stand-in for a model
anywhere in them. Each prints which way it ran, first:

- `Running live against Claude (<model>).` means it is calling the API (`..., recording it into recordings/...`
  when `RECORD=1` is set too).
- `ANTHROPIC_API_KEY is not set: replaying recordings/<example>.jsonl, Claude (<model>) as recorded on <date>.`
  means no model is called: every answer is the recorded one. A prompt the recording does not hold is refused by
  name with the command that records it again, never sent live.
- `No model is called: ...` means the example never calls one, live or not (`cassettes.py`, `reports.py`).
  `rung_zero.py` calls no model either, and prints no such line.

The examples need nothing beyond the base install: the client speaks Anthropic's API over `httpx`, which the
package depends on.

**In a checkout of this repo**, from the repo root, after `uv sync`:

```bash
uv run python packages/evals/examples/llm_judge.py
```

To run live, put an Anthropic API key in `packages/evals/examples/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Git ignores `.env` files, so the key stays out of commits. Then pass the file to `uv run`:

```bash
uv run --env-file packages/evals/examples/.env python packages/evals/examples/llm_judge.py
```

**Outside a checkout** (once 0.68.0 is on PyPI and you installed `3tears-evals` with pip): the examples are
not part of the installed package, so fetch this whole directory from the repository, at the tag you installed (the
[README's install step](../README.md#first-result-in-10-minutes) names it). This fetches only the examples:

```bash
git clone --depth 1 --branch <tag> --filter=blob:none --sparse https://github.com/pacepace/3tears
cd 3tears && git sparse-checkout set packages/evals/examples && cd packages/evals/examples
```

Keep all of it together: the examples replay `recordings/` and read `tickets.jsonl` and
`llm_analysis_first_draft.json` from beside themselves. Then, from inside it, `python rung_zero.py`, or with
`export ANTHROPIC_API_KEY=sk-ant-...` first, any example live.

**How an example builds its client.** Each model-calling example has the same few lines, so read any one: with the
key, `Model("claude-haiku-5-5", api_key=..., provider="anthropic", max_tokens=..., effort=...)` from
`threetears.evals.providers`, built from the `ClientRequestSettings` the example declares (`SETTINGS`, or one
per arm); without it, `RecordedCompletions.replay(recording, settings=...)` from
`threetears.evals.testing`, given the same settings, since a recording serves only a client built the way it was.
No temperature is sent: `claude-haiku-5-5` takes none, so the model's own sampling applies. The candidates ask for
`low` effort; `compare_two_efforts.py` and `prompt_x_effort.py` compare `low` with `medium`, and `llm_judge.py`'s
judge is asked as a judged run records it (`JUDGE_REQUEST_SETTINGS`: an output cap and `high` effort).

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

Live runs call `claude-haiku-5-5` over 5 to 14 cases (50 for `guardrails.py`), so each costs cents. Each file's
docstring gives its call count and rough live cost. A replayed run reports the dollars each call cost when it was
recorded. `reports.py` and `llm_analysis.py` write their files under the directory you run from (`./eval-report/`
and `./eval-analysis/`).

## Recording

A recording is made once, live, with the key set, and replaces the example's file in `recordings/`
(`RECORD=1` records afresh; it never appends to an old recording). From this directory:

```bash
RECORD=1 python llm_judge.py            # recordings/llm_judge.jsonl
(cd 3-invoices && RECORD=1 python extract.py)                   # 3-invoices/recordings/invoices.jsonl, on OPENROUTER_API_KEY
(cd 4-booking && RECORD=1 python -m threetears.evals agent.py)  # 4-booking/recordings/agent.jsonl, on OPENROUTER_API_KEY
```

The whole ladder, in a checkout, from the repo root:

```bash
cd packages/evals/examples && for example in llm_judge; do RECORD=1 uv run --env-file .env python "$example.py" || break; done
```

`RECORD=1` with no key is refused: a recording is of a live run. Record again whenever an example's prompt, cases or
request settings change: the example's test (`tests/test_<example>_example.py`) checks the recording still covers
every prompt the example sends before it replays, and names each one that changed. `llm_analysis.py` also keeps the
bundle its recording was made from: a recording answers only the prompt it was made for, the bundle is the prompt,
and a fresh campaign mints fresh run ids and times, so a replay hands the writer the recorded bundle.

Every example that grades ends by reading what it measured, not only the rate: `summary.misses()` (each miss and
why), or, in a comparison, where the arms disagree (`comparison.results(arm)`). Read them before you trust the
numbers. `cassettes.py`, `reports.py` and `llm_analysis.py` grade nothing new: they show what to do with a run.

## The ladder

| # | File | The question it answers | What's new |
|---|---|---|---|
| 0 | [`rung_zero.py`](rung_zero.py) | Does my function give the right answer on my cases? | `Eval(cases, runner, grader)`; a field grader and a function; `summary.misses()` |
| 1 | [`llm_judge.py`](llm_judge.py) | Is each answer helpful, and does it say only what its source supports? | `Judge`, among the graders: a model grades each answer against a rubric |
| 3 | [`3-invoices/`](3-invoices/extract.py) | Changing a prompt? Compare it against the current one, and get a report you can hand to a reviewer. | two prompts as `runner={"v1": model.prompt(V1, Invoice), "v2": ...}`, the first the control (delta, interval, verdict); a field grader per field, `grader(margin=0.05)` for equivalence and `@grader(guard=0.02) no_invented_total`; strata; `result.write()` for the HTML report; `gate()` in CI; `result.against()` and `k=v` arm names in its closing comment |
| 4 | [`4-booking/`](4-booking/agent.py) | Is your agent doing the right thing to the world, not only saying it? Then serve your evals to the team. | `World(Day, tools=[find_free, book])` and a JSON tool loop over `model.json`; goal checks and the did-nothing control; [`serve.py`](4-booking/serve.py): `as_host`, MCP (`mount_fastmcp`) and the command line |

The [tutorial](../docs/tutorial.md) covers rungs 0 to 3 and 9 (in its own order, the judge after the comparison),
and keeping your runs in a file. The other rungs are here alone.

## Not covered by an example yet

These are documented in the guides:

- Judge reliability and evidence tiers: [Reading reports](../docs/reading-reports.md). Results by kind of case
  (strata) are read in [`3-invoices/`](3-invoices/extract.py), and explained in
  [Reading reports](../docs/reading-reports.md#results-by-kind-of-case-strata).
- Declaring a campaign's design (axes, question, bar), and when exploring without one is the right call:
  [Choosing a design](../docs/choosing-a-design.md).
- Budgets and spend caps: [Cost and budgets](../docs/cost-and-budgets.md).
- Conversations with a simulated user, generating cases, storage, and writing your own host:
  [Adopting the engine](../docs/adopting-a-host.md).
- How a bar is decided: [Reading reports](../docs/reading-reports.md#methods) and
  [Measuring soundly](../docs/measuring-soundly.md#variance-k-and-how-many-cases).
- The command line: [The command line](../docs/command-line.md).
- Driving evals from an agent over MCP: [Driving it from an agent](../docs/agents-and-mcp.md).
