# Examples

Each file here answers one question about your LLM-backed feature, and adds one capability. Read them in
order: each builds on the ones before it. They all run offline with no API key, and against Claude when you add one.

## Running them

From the repo root, after `uv sync`:

```bash
uv run python packages/evals/examples/rung_zero.py
```

To run live, put an Anthropic API key in `packages/evals/examples/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Git ignores `.env` files, so the key stays out of commits. Then pass the file to `uv run`:

```bash
uv run --env-file packages/evals/examples/.env python packages/evals/examples/llm_judge.py
```

An example that calls a model prints which way it ran:

- `Running against Claude (<model>).` means it is calling the API. The model ids are in each file's
  `MODEL` or `LIVE` constant, and the client they share is [`_live.py`](_live.py): Claude behind the engine's
  completion-client protocol, priced at list price. It is not an example of its own.
- `ANTHROPIC_API_KEY is not set: running OFFLINE, ...` means small scripted stand-ins play the model. They are
  named as stand-ins (`cheaper-stand-in`, `offline-judge`), the output has the real shape, and its numbers say
  nothing about any model.
- `No model is called: ...` means the example never calls one, live or not (`cassettes.py`, `reports.py`).
  `rung_zero.py` calls no model either, and prints no such line.

Live runs use the cheapest Haiku models over 5 to 14 cases, so each costs cents. Each file's
docstring gives its call count and rough live cost. `reports.py` and `llm_analysis.py` write their files under
the directory you run from (`./eval-report/` and `./eval-analysis/`).

Every example ends by reading what it measured, not only the rate: `summary.misses()` (each miss and why), or,
in a comparison, where the arms disagree (`comparison.results(arm)`). Read them before you trust the numbers.

## The ladder

| # | File | The question it answers | What's new |
|---|---|---|---|
| 0 | [`rung_zero.py`](rung_zero.py) | Does my function give the right answer on my cases? | `run_eval`; `expected=` and a scorer; `summary.misses()` |
| 1 | [`llm_judge.py`](llm_judge.py) | Is each answer helpful, and does it say only what its source supports? | `Judge`: a model grades each answer against a rubric |
| 2 | [`compare_two_prompts.py`](compare_two_prompts.py) | What does the new prompt change, and is the difference real or noise? | `compare`: arms, a control, delta, its interval, a verdict |
| 3 | [`compare_two_models.py`](compare_two_models.py) | Is the cheaper model good enough for this prompt, given what it saves? | `Answer` (each arm's spend), and `margins=`: only `equivalent` says "good enough" |
| 4 | [`prompt_x_model.py`](prompt_x_model.py) | What does changing the prompt do on each model? | `factors=`, and `Comparison.against` for a second control |
| 5 | [`cassettes.py`](cassettes.py) | How do I compare candidates fairly when the tool they call answers differently every time? | `tools=`, `cassette_mode=`: capturing and replaying tool results |
| 6 | [`world.py`](world.py) | Does the model turn the light on when the room is dark? | `world=`, `seed=`, `goal_checks=`: grade what the model did to a world |
| 7 | [`reports.py`](reports.py) | How do I turn a finished campaign into files people read? | `report_markdown`, `report_html`, Vega-Lite charts |
| 8 | [`llm_analysis.py`](llm_analysis.py) | Can a model write the analysis from a frozen copy of the evidence? | `generate_analysis` over a fingerprinted bundle |

The [tutorial](../docs/tutorial.md) walks the same capabilities in the same order.

## Not covered by an example yet

These are documented in the guides:

- Judge reliability and evidence tiers, and results by kind of case (strata):
  [Reading reports](../docs/reading-reports.md).
- Guardrails, what a candidate must never do, decided held, breached or undecided apart from capability:
  [Reading the guardrails](../docs/reading-reports.md#reading-the-guardrails).
- Declaring a campaign's design (axes, question, bar), and when exploring without one is the right call:
  [Choosing a design](../docs/choosing-a-design.md).
- Budgets and spend caps: [Cost and budgets](../docs/cost-and-budgets.md).
- Conversations with a simulated user, generating cases, storage, and writing your own host:
  [Adopting the engine](../docs/adopting-a-host.md).
- How a bar is decided: [Reading reports](../docs/reading-reports.md#methods) and
  [Measuring soundly](../docs/measuring-soundly.md#variance-k-and-how-many-cases).
- The command line: [The command line](../docs/command-line.md).
- Driving evals from an agent over MCP: [Driving it from an agent](../docs/agents-and-mcp.md).
