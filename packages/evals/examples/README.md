# Examples

Each file here answers one question about your LLM-backed feature, and adds one capability. Read them in
order: each builds on the ones before it. Those that call a model call Claude when you set an API key, and replay a
recording of a live run when you do not, so every one runs offline and free out of the box.

## Running them

The examples that call a model run **live by default**: with `ANTHROPIC_API_KEY` set they call Claude
(`claude-haiku-5-5`, the only model they use) through the package's shipped provider adapter,
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

The examples need the `[models]` extra, which brings in `3tears-models` and the Anthropic SDK through it. Outside a
checkout that is `pip install "3tears-evals[models]"`, which resolves only once 0.68.0 is on PyPI. Until then, use a
checkout of this repo and `uv sync --extra models` at its root.

**In a checkout of this repo**, from the repo root, after `uv sync --extra models`:

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

**Outside a checkout** (once 0.68.0 is on PyPI and you installed `3tears-evals[models]` with pip): the examples are
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
key, `models_client("claude-haiku-5-5", api_key=..., provider="anthropic", max_tokens=..., reasoning_effort=...)`
from `threetears.evals.providers`, built from the `ClientRequestSettings` the example declares (`SETTINGS`, or one
per arm); without it, `RecordedCompletions.replay(recording, settings=..., model=...)` from
`threetears.evals.testing`, given the same settings, since a recording serves only a client built the way it was.
No temperature is sent: `claude-haiku-5-5` refuses one, so the model's own sampling applies. The candidates ask for
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
a provider that reported a failure in place of a completion. When to keep your own rather than the shipped adapter:
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
RECORD=1 python compare_two_prompts.py  # recordings/compare_two_prompts.jsonl
RECORD=1 python compare_two_efforts.py  # recordings/compare_two_efforts.jsonl
RECORD=1 python prompt_x_effort.py      # recordings/prompt_x_effort.jsonl
RECORD=1 python world.py                # recordings/world.jsonl
RECORD=1 python llm_analysis.py         # recordings/llm_analysis.jsonl and recordings/llm_analysis.bundle.json
RECORD=1 python guardrails.py           # recordings/guardrails.jsonl
```

The whole ladder, in a checkout, from the repo root:

```bash
cd packages/evals/examples && for example in llm_judge compare_two_prompts compare_two_efforts prompt_x_effort \
  world llm_analysis guardrails; do RECORD=1 uv run --env-file .env python "$example.py" || break; done
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
| 0 | [`rung_zero.py`](rung_zero.py) | Does my function give the right answer on my cases? | `run_eval`; `expected=` and a scorer; `summary.misses()` |
| 1 | [`llm_judge.py`](llm_judge.py) | Is each answer helpful, and does it say only what its source supports? | `Judge`: a model grades each answer against a rubric |
| 2 | [`compare_two_prompts.py`](compare_two_prompts.py) | What does the new prompt change, and is the difference real or noise? | `compare`: arms, a control, delta, its interval, a verdict; each arm per kind of case (`stratum_key=`, `comparison.strata()`); the cases kept as a file, [`tickets.jsonl`](tickets.jsonl), that every prompt version runs against |
| 3 | [`compare_two_efforts.py`](compare_two_efforts.py) | Is less effort good enough for this prompt, given what it saves? Two arms that differ in one setting, `low` against `medium` reasoning effort | `Answer` (each arm's spend), `margins={"accuracy": ...}`: only `equivalent` says "good enough"; typed verdicts |
| 4 | [`prompt_x_effort.py`](prompt_x_effort.py) | What does changing the prompt do at each effort? | `factors=`, and `Comparison.against` for a second control |
| 5 | [`cassettes.py`](cassettes.py) | How do I compare candidates fairly when the tool they call answers differently every time? | `tools=`, `cassette_mode=`: capturing and replaying tool results |
| 6 | [`world.py`](world.py) | Does the model turn the light on when the room is dark? | `world=`, `seed=`, `goal_checks=`: grade what the model did to a world |
| 7 | [`reports.py`](reports.py) | How do I turn a finished campaign into files people read? | `report_markdown`, `report_html`, Vega-Lite charts, `Comparison.gate` |
| 8 | [`llm_analysis.py`](llm_analysis.py) | Can a model write the analysis from a frozen copy of the evidence? | `generate_analysis` over a fingerprinted bundle; a canned first draft the check refuses, and the model's repair |
| 9 | [`guardrails.py`](guardrails.py) | The new prompt answers more questions. Does it still keep the customer's card private? | `guardrails=`: a guardrail with its margin, held / breached / undecided per arm, apart from the contrasts |

The [tutorial](../docs/tutorial.md) covers rungs 0 to 3 and 9 (in its own order, the judge after the comparison),
and keeping your runs in a file. The other rungs are here alone.

## Not covered by an example yet

These are documented in the guides:

- Judge reliability and evidence tiers: [Reading reports](../docs/reading-reports.md). Results by kind of case
  (strata) are read in [`compare_two_prompts.py`](compare_two_prompts.py), and explained in
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
