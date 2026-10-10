# 3tears-evals

3tears-evals measures what a change to an LLM-backed feature does, from a classifier to a tool-using agent. Run
each version over the same cases and it reports what moved, with the uncertainty that says whether the move is
real. For an agent, it measures what the agent did to a world your application declares, not only what it said:
the open tickets it closed, the meeting it booked, the light it switched on.

> **The public API is still changing.** This package was extracted from a production app's eval engine, and its
> API will change without notice until a release says otherwise.

## What is different

Evaluating an agent in a seeded starting state is established practice. Three things here are new; we found no
prior system that combines them ([The world model](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/world-model.md#what-is-new-here)):

1. **Your application declares the world to the engine.** Each piece of state (a dimension) has a JSON Schema,
   seed and read handles that use production's own write paths, and a list of the surfaces where the agent can
   see it.
2. **Every run records, per dimension, whether the case set it and whether the agent could see it.** State the
   agent saw but the run did not set is disclosed as a confound (a rival explanation), never averaged with state
   the case set.
3. **A conformance kit your application runs against its own world** checks that seeds read back, that each
   named surface moves when its dimension does while the others hold still, and that dimensions are
   independent. A check it cannot run is reported as unavailable, with its reason, never waived.

It also holds a set of measurement rules, each explained on its page:

- **Paired arms on frozen cases.** Every version (an arm) runs over the same stored cases and is compared case
  by case ([why](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/design-rationale.md#the-arm-is-the-unit)).
- **Identity keys for configurations.** A configuration is keyed by the content that ran, so an edited prompt is
  never averaged in with its old version ([why](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/design-rationale.md#identity-is-content-with-a-rendering)).
- **"Not separated", never "no difference".** Evidence that cannot tell two versions apart is reported as exactly
  that ([why](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/principles.md#numbers)).
- **A swept lever is checked against its mechanism.** A lever (a setting you vary) can name what it acts on, so
  a setting that never took effect does not read as one that changed nothing
  ([why](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/measuring-soundly.md#did-the-lever-move)).
- **Every confound is disclosed.** When something besides the lever moved, the comparison says so rather than
  hiding it ([why](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/design-rationale.md#confounds-qualify-never-suppress)).
- **Guardrails are kept apart from capability.** What the agent must never do is decided on its own, so a gain
  cannot pay for a breach ([why](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/design-rationale.md#guardrails-are-a-pillar-apart-from-capability)).

## First result in 10 minutes

1. **Install** (Python 3.14+). In a checkout of this repo, `uv sync` at the root does it.

   ```bash
   pip install 3tears-evals
   pip install "3tears-evals[vega]"      # adds SVG and PNG chart rendering
   pip install "3tears-evals[fastmcp]"   # adds the FastMCP transport, for driving evals from an agent
   ```

2. **Run the first example.** It grades a small sentiment classifier on five cases, offline and free:

   ```bash
   uv run python packages/evals/examples/rung_zero.py
   ```

3. **Point it at your own function.** Any async function from a case to an answer works. Put your prompt and
   model call where the comment says:

   ```python
   import asyncio

   from threetears.evals.quick import run_eval

   CASES = [
       {"text": "Exactly what I ordered, and it arrived early.", "expected": "positive"},
       {"text": "The box came crushed.", "expected": "negative"},
       {"text": "It works.", "expected": "neutral"},
   ]


   async def classify(case: dict) -> str:
       """Label a review's sentiment as positive, negative or neutral."""
       return "positive"  # your prompt and model call go here


   async def main() -> None:
       summary = await run_eval(CASES, classify, expected=lambda case: case["expected"], scope_id="dev", k=2)
       print(summary.render())
       for miss in summary.misses():
           print(miss.case, miss.missed_because)


   asyncio.run(main())
   ```

Then do the [tutorial](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/tutorial.md): it is the
one starting path. It covers reading misses, comparing two versions, reading the verdict, and adding a judge.
To run the examples against Claude, see
[the examples](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/README.md).

## Twelve words to know

Each links to its entry in [Concepts](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md),
the glossary of record.

| Term | In one line |
|---|---|
| [case](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#case-test-case) | One input and what a good answer looks like. |
| [candidate](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#candidate) | The code under test, which answers each case. |
| [scorer](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#scorer) | A function that grades an answer with a number, when code can check it. |
| [judge](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#judge) | A model that grades an answer against a rubric, when code cannot. |
| [rubric](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#judged-dimension-rubric-dimension) | The written qualities a judge scores, one dimension each. |
| [k](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#k-repeats) | How many times each case is played, because model answers vary. |
| [arm](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#arm) | One version under test, run over every case. |
| [control](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#control) | The arm every other arm is tested against. |
| [verdict](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#verdict) | Separated, not separated or equivalent: what the evidence supports. |
| [interval](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#delta-interval-and-adjusted-p) | Where the true value or difference plausibly lies. |
| [p](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#delta-interval-and-adjusted-p) | The Holm-adjusted p-value a verdict is decided on. |
| [world](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/concepts.md#world) | The state an agent acts on, set for each case and read back after. |

## Docs

**Learn**

- [Tutorial](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/tutorial.md): your first eval, end to end. Start here.
- [Designing a classifier eval set](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/designing-classifier-evals.md): which cases to write, and how many.
- [Reading reports](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reading-reports.md): what each part of a comparison and a report means.

**Build**

- [Evaluating a tool-using agent](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/evaluating-agents.md): a world, seeds, goal checks and the conformance kit.
- [Judges and calibration](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/judges-and-calibration.md): rubrics, and whether to trust the judge.
- [Adopting the engine](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/adopting-a-host.md): your own host, store, kind and launcher.
- [Choosing a campaign design](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/choosing-a-design.md): which arms answer your question, how many cases, and when to stay exploratory.
- [Cost and budgets](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/cost-and-budgets.md): what a launch will cost, and what stops it.
- [Driving it from an agent](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/agents-and-mcp.md): operations, actions and MCP.
- [The command line](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/command-line.md): launching and reading from a terminal.

**Background**

- [Principles](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/principles.md): the rules the engine keeps, and why.
- [Measuring soundly](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/measuring-soundly.md): the traps campaigns have hit.
- [The world model](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/world-model.md): why an agent runs in a declared world.
- [Design rationale](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/design-rationale.md): why arms, identity, confounds and the analysis are shaped as they are.
- [Prior art](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/prior-art.md): what the field recommends, and what this engine took from it.
- [Open problems](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/open-problems.md): known gaps.

**Reference**

- [Reference](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reference.md): every public name, setting, report field, goal-check builtin and action, generated from the code.

## For experts

| Capability | API | Doc | Example |
|---|---|---|---|
| World contract | `WorldRegistry`; quick path: `World`, `Dimension`, `WorldTool`, `seed=`, `goal_checks=` | [Evaluating a tool-using agent](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/evaluating-agents.md), [The world model](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/world-model.md) | [`world.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/world.py) |
| World conformance kit | `check_world_conformance` (`threetears.evals.contracts.host`) | [Step 5](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/evaluating-agents.md#step-5-run-the-conformance-kit-against-your-world) | |
| Guardrails, decided apart from capability | quick path: `compare(guardrails={name: Guardrail(margin=..., direction=...)})`, `Comparison.guardrails()`; on a host: `MetricDescriptor(guardrail=True)`, `RubricDim(axis="boundary")`; held / breached / undecided | [Reading the guardrails](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reading-reports.md#reading-the-guardrails), [Judges and calibration](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/judges-and-calibration.md#step-3-guardrails-are-not-capabilities) | [`guardrails.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/guardrails.py) |
| Equivalence | `MetricDescriptor.materiality_threshold` as the margin and `value_range` as the range (with no range it is never tested); verdict `equivalent` | [Reading a comparison](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reading-reports.md#reading-a-comparison) | |
| Cassettes | `tools=`, `cassette_mode="capture"` / `"replay"`, `cassette_corpus_id=` | [Adopting the engine](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/adopting-a-host.md#cassettes-recording-and-replaying-tools) | [`cassettes.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/cassettes.py) |
| Grids of factors | `compare(..., factors=...)`, `Comparison.against` | [Choosing a campaign design](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/choosing-a-design.md#start-from-the-question) | [`prompt_x_model.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/prompt_x_model.py) |
| Campaigns and declared designs | `create_campaign`, `set_campaign_control`, `declared_design` | [Choosing a campaign design](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/choosing-a-design.md#declare-the-design-and-its-control) | [`compare_two_prompts.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/compare_two_prompts.py) |
| Cost against quality | `Answer`, `max_cost_usd=`, the frontier | [Cost and budgets](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/cost-and-budgets.md), [The frontier](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/choosing-a-design.md#the-frontier-passk-against-cost) | [`compare_two_models.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/compare_two_models.py) |
| The analysis bundle and reports | `inspect_campaign_bundle`, `AnalysisContextBundle`, `report_markdown`, `report_html` | [Reading reports](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reading-reports.md) | [`reports.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/reports.py) |
| Analysis written by a model over frozen evidence | `generate_analysis` | [Having a model write the analysis](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence) | [`llm_analysis.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/llm_analysis.py) |
| Judge calibration and evidence tiers | `rate_result`, `judge_agreement`, `JudgeEvidenceTier` | [Judges and calibration](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/judges-and-calibration.md) | [`llm_judge.py`](https://github.com/pacepace/3tears/blob/develop/packages/evals/examples/llm_judge.py) |
| MCP | `threetears.evals.actions`, `mount_fastmcp` (extra `fastmcp`) | [Driving it from an agent](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/agents-and-mcp.md) | |
| A complete host | `EvalHost`, `HostProfile`, `CandidateKind` | [Adopting the engine](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/adopting-a-host.md#read-the-reference-hosts-in-this-order) | [`courierhost`](https://github.com/pacepace/3tears/blob/develop/packages/evals/tests/fixtures/courierhost/__init__.py), then [`toyhost`](https://github.com/pacepace/3tears/blob/develop/packages/evals/tests/fixtures/toyhost/README.md) |

Import only from the public roots the [reference](https://github.com/pacepace/3tears/blob/develop/packages/evals/docs/reference.md#public-api)
lists: a module below a root is internal and may move.
