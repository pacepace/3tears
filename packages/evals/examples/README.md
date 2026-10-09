# Examples

Each example is one file that answers one question, and each builds on the one before it. Read them in
this order.

| # | File | The question it answers | What's new |
|---|---|---|---|
| 0 | [`rung_zero.py`](rung_zero.py) | Does my function give the right answer on my cases? | `run_eval`, a classifier's `expected=`, a scorer |
| 1 | [`compare_two_prompts.py`](compare_two_prompts.py) | Is the new prompt better than the old one, or is the difference noise? | `compare`: arms, a control, a verdict |
| 2 | [`compare_two_models.py`](compare_two_models.py) | Is the cheaper model good enough for this prompt, given what it saves? | `Answer`: a candidate reports its spend |
| 3 | [`prompt_x_model.py`](prompt_x_model.py) | Does the better prompt help on both models, or only on one? | `factors=`, `Comparison.against` |
| 4 | [`llm_judge.py`](llm_judge.py) | Is each answer helpful, and does it say only what its source supports? | `Judge`: a model grades against a rubric |
| 5 | [`cassettes.py`](cassettes.py) | How do I compare candidates fairly when the tool they call answers differently every time? | `tools=`, `cassette_mode=` capture and replay |
| 6 | [`world.py`](world.py) | Does the model turn the light on when the room is dark, judged by what it does rather than what it says? | `world=`, `seed=`, `goal_checks=` |
| 7 | [`reports.py`](reports.py) | How do I turn a finished campaign into the files people read: verdicts, report files and charts? | `report_markdown`, `report_html`, `VegaRenderer` |
| 8 | [`llm_analysis.py`](llm_analysis.py) | Can a model write a campaign's analysis from a frozen copy of the evidence, so it can be checked and redone? | `generate_analysis` over a fingerprinted bundle |

## Running one

From the repository root:

```bash
python packages/evals/examples/compare_two_prompts.py
```

`reports.py` writes to `./eval-report/` and `llm_analysis.py` to `./eval-analysis/`, both under the
directory you run from.

## Offline or live

Every example runs with no API key. Without `ANTHROPIC_API_KEY`, the ones that call a model print
`ANTHROPIC_API_KEY is not set: running OFFLINE, ...` and use small scripted stand-ins: the output has the
real shape, and its numbers say nothing about any model. With the key set (and `pip install anthropic`)
they print `Running against Claude (<model>).` and make real calls. The model ids are in each file's
`MODEL` or `MODELS` constant.

Live, each costs a few US cents at most: `prompt_x_model.py` about two (112 calls), `llm_analysis.py`
about one, and the others well under one. `rung_zero.py`, `cassettes.py` and `reports.py` call no model
and cost nothing.

## Not shown by an example yet

Judge reliability and evidence tiers, and strata, are in [Reading reports](../docs/reading-reports.md);
budgets and caps in [Cost and budgets](../docs/cost-and-budgets.md); conversations with a simulated user,
case generation, bars, storage and writing your own host in [Adopting the engine](../docs/adopting-a-host.md);
the CLI in [The command line](../docs/command-line.md); MCP in [Driving it from an agent](../docs/agents-and-mcp.md).
