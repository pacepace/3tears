# Examples

Each file here answers one question about your LLM-backed feature, and each builds on the one before.
Read them in order. They all run offline with no API key, and against Claude when you add one.

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
uv run --env-file packages/evals/examples/.env python packages/evals/examples/rung_zero.py
```

Each example prints which way it ran:

- `Running against Claude (<model>).` means it is calling the API. The model ids are in each file's
  `MODEL` or `MODELS` constant.
- `ANTHROPIC_API_KEY is not set: running OFFLINE, ...` means small scripted stand-ins play the model. The
  output has the real shape, and its numbers say nothing about any model.
- `No model is called: ...` means the example never calls one, live or not.

Live runs use the cheapest Haiku models over a dozen or so cases, so each costs cents. Each file's
docstring gives its rough live cost. `reports.py` and `llm_analysis.py` write their files under the
directory you run from (`./eval-report/` and `./eval-analysis/`).

## The ladder

| # | File | The question it answers | What's new |
|---|---|---|---|
| 0 | [`rung_zero.py`](rung_zero.py) | Does my function give the right answer on my cases? | `run_eval`, a classifier's `expected=`, a scorer |
| 1 | [`llm_judge.py`](llm_judge.py) | Is each answer helpful, and does it say only what its source supports? | `Judge`: a model grades against a rubric; `Answer`: a candidate reports its spend |
| 2 | [`compare_two_prompts.py`](compare_two_prompts.py) | What does the new prompt change, and is the difference real or noise? | `compare`: arms, a control, a verdict |
| 3 | [`compare_two_models.py`](compare_two_models.py) | Is the cheaper model good enough for this prompt, given what it saves? | each arm's spend, and cost tested against the control |
| 4 | [`prompt_x_model.py`](prompt_x_model.py) | What does changing the prompt do on each model? | `factors=`: vary several things, read every combination |
| 5 | [`cassettes.py`](cassettes.py) | How do I compare candidates fairly when the tool they call answers differently every time? | `tools=`, `cassette_mode=`: capturing & replaying tool results |
| 6 | [`world.py`](world.py) | Does the model turn the light on when the room is dark? | `world=`, `seed=`, `goal_checks=`: model a world, measure the model's impact on it |
| 7 | [`reports.py`](reports.py) | How do I turn a finished campaign into files people read? | `report_markdown`, `report_html`, Vega-Lite charts |
| 8 | [`llm_analysis.py`](llm_analysis.py) | Can a model write the analysis from a frozen copy of the evidence? | `generate_analysis` over a fingerprinted bundle |

The [package README](../README.md#using-it-in-your-code) covers the same capabilities as a reference,
in the same order.

## Not covered by an example yet

These are documented in the guides:

- Judge reliability and evidence tiers, and results by kind of case (strata):
  [Reading reports](../docs/reading-reports.md).
- Budgets and spend caps: [Cost and budgets](../docs/cost-and-budgets.md).
- Conversations with a simulated user, generating cases, bars, storage, and writing your own host:
  [Adopting the engine](../docs/adopting-a-host.md).
- The command line: [The command line](../docs/command-line.md).
- Driving evals from an agent over MCP: [Driving it from an agent](../docs/agents-and-mcp.md).
