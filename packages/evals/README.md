# 3tears-evals

3tears-evals measures what a change to an LLM-backed feature does, from a classifier to a tool-using agent. Run
each version over the same cases and it reports what moved, with the uncertainty that says whether the move is
real. For an agent, it measures what the agent did to a world your application declares, not only what it said:
the open tickets it closed, the meeting it booked, the light it switched on.

> **The public API is still changing.** This package was extracted from a production app's eval engine. From
> 0.68.0 only its host contract is promised: a name on it moves with a deprecation cycle, and any other public name
> may change in any release, recorded in `MIGRATIONS.toml`
> ([stability tiers](docs/adopting-a-host.md#stability-tiers)).

## First result in 10 minutes

1. **Install** (Python 3.14+). **3tears-evals is not on PyPI until 0.68.0 ships**, so `pip install 3tears-evals`
   finds nothing yet. Until then, install it from the repository at a tag. The `TAG=` line is the one place the
   docs name that tag, and every other page's install line points here:

   ```bash
   TAG=v0.67.0-evaladopt.25
   pip install "3tears-evals @ git+https://github.com/pacepace/3tears.git@$TAG#subdirectory=packages/evals"
   # adds SVG and PNG chart rendering
   pip install "3tears-evals[vega] @ git+https://github.com/pacepace/3tears.git@$TAG#subdirectory=packages/evals"
   # adds the FastMCP transport, for driving evals from an agent
   pip install "3tears-evals[fastmcp] @ git+https://github.com/pacepace/3tears.git@$TAG#subdirectory=packages/evals"
   ```

   Once 0.68.0 is on PyPI these are `pip install 3tears-evals`, `pip install "3tears-evals[vega]"` and
   `pip install "3tears-evals[fastmcp]"`. In a checkout of this repo, `uv sync` at the root does it.
   `threetears.evals.__version__` says which version is installed.

   The `[models]` extra (the shipped provider adapter, `threetears.evals.providers`) installs from PyPI once 0.68.0
   ships, as `pip install "3tears-evals[models]"`. Until then it has no install line from a tag: it needs
   3tears-models and its family packages at 0.68.0, none of them on PyPI yet, so use a checkout of this repo at the
   tag and run `uv sync --extra models` at the root.

2. **Run the first example.** It grades a small sentiment classifier on six cases, offline and free. In a
   checkout:

   ```bash
   uv run python packages/evals/examples/rung_zero.py
   ```

   After a pip install, copy the [`examples/`](examples/) directory from the repository and run
   `python rung_zero.py` inside it; [examples/README.md](examples/README.md) says how, and what a live run needs.

3. **Point it at your own function.** Any async function from a case to an answer works. Put your prompt and
   model call where the comment says:

   ```python
   import asyncio

   from threetears.evals.quick import run_eval

   CASES = [
       {"id": "early", "text": "Exactly what I ordered, and it arrived early.", "expected": "positive"},
       {"id": "crushed", "text": "The box came crushed.", "expected": "negative"},
       {"id": "works", "text": "It works.", "expected": "neutral"},
   ]


   async def classify(case: dict) -> str:
       """Label a review's sentiment as positive, negative or neutral."""
       return "positive"  # your prompt and model call go here


   async def main() -> None:
       summary = await run_eval(CASES, classify, expected=lambda case: case["expected"], k=2)
       print(summary.render())
       for miss in summary.misses():
           print(f"{miss.case} (repeat {miss.repeat}): {'; '.join(miss.missed_because)}")


   asyncio.run(main())
   ```

Then do the [tutorial](docs/tutorial.md): it is the
one starting path. It covers reading misses, comparing two versions, reading the verdict, and adding a judge.
To run the examples against Claude, see
[the examples](examples/README.md).

## What is different

Evaluating an agent in a seeded starting state is established practice. Three things here are new; we found no
prior system that combines them ([The world model](docs/world-model.md#what-is-new-here)):

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
  by case ([why](docs/design-rationale.md#the-arm-is-the-unit)).
- **Identity keys for configurations.** A configuration is keyed by the content that ran, so an edited prompt is
  never averaged in with its old version ([why](docs/design-rationale.md#identity-is-content-with-a-rendering)).
- **"Not separated", never "no difference".** Evidence that cannot tell two versions apart is reported as exactly
  that ([why](docs/principles.md#numbers)).
- **A swept lever is checked against its mechanism.** A lever (a setting you vary) can name what it acts on, so
  a setting that never took effect does not read as one that changed nothing
  ([why](docs/measuring-soundly.md#did-the-lever-move)).
- **Every confound is disclosed.** When something besides the lever moved, the comparison says so rather than
  hiding it ([why](docs/design-rationale.md#confounds-qualify-never-suppress)).
- **Guardrails are kept apart from capability.** What the agent must never do is decided on its own, so a gain
  cannot pay for a breach ([why](docs/design-rationale.md#guardrails-are-a-pillar-apart-from-capability)).

## Words to know

Sixteen terms carry a first eval: case, candidate, scorer, measure, judge, judged dimension, k, run, result, arm,
control, campaign, verdict, interval and p, report, world. Each is one line in
[Terms to learn first](docs/concepts.md#terms-to-learn-first),
in Concepts, the glossary of record.

## Docs

These links are relative to the package, so on GitHub each opens the page at the ref you are reading; on PyPI,
where this README is the package's description, they do not resolve, so read the docs in the repository at the tag
you installed.

**Learn**

- [Tutorial](docs/tutorial.md): your first eval, end to end. Start here.
- [Designing a classifier eval set](docs/designing-classifier-evals.md): which cases to write, and how many.
- [Reading reports](docs/reading-reports.md): what each part of a comparison and a report means.

**Build**

- [Evaluating a tool-using agent](docs/evaluating-agents.md): a world, seeds, goal checks and the conformance kit.
- [Judges and calibration](docs/judges-and-calibration.md): rubrics, and whether to trust the judge.
- [Adopting the engine](docs/adopting-a-host.md): your own host, store, kind and launcher.
- [Choosing a campaign design](docs/choosing-a-design.md): which arms answer your question, how many cases, and when to stay exploratory.
- [Cost and budgets](docs/cost-and-budgets.md): what a launch will cost, and what stops it.
- [Driving it from an agent](docs/agents-and-mcp.md): operations, actions and MCP.
- [The command line](docs/command-line.md): launching and reading from a terminal.

**Background**

- [Principles](docs/principles.md): the rules the engine keeps, and why.
- [Measuring soundly](docs/measuring-soundly.md): the traps campaigns have hit.
- [The world model](docs/world-model.md): why an agent runs in a declared world.
- [Design rationale](docs/design-rationale.md): why arms, identity, confounds and the analysis are shaped as they are.
- [Prior art](docs/prior-art.md): what the field recommends, and what this engine took from it.
- [Open problems](docs/open-problems.md): known gaps.

**Reference**

- [Reference](docs/reference.md): every public name, setting, report field, goal-check builtin and action, generated from the code.

## For experts

| Capability | API | Doc | Example |
|---|---|---|---|
| World contract | `WorldRegistry`; quick path: `World`, `Dimension`, `WorldTool`, `seed=`, `goal_checks=` | [Evaluating a tool-using agent](docs/evaluating-agents.md), [The world model](docs/world-model.md) | [`world.py`](examples/world.py) |
| World conformance kit | `check_world_conformance` (`threetears.evals.kernel.host`) | [Step 5](docs/evaluating-agents.md#step-5-run-the-conformance-kit-against-your-world) | |
| Guardrails, decided apart from capability | quick path: `compare(guardrails={name: Guardrail(margin=..., direction=...)})`, `Comparison.guardrails()`; on a host: `MetricDescriptor(guardrail=True)`, `RubricDim(axis="boundary")`; held / breached / undecided | [Reading the guardrails](docs/reading-reports.md#reading-the-guardrails), [Judges and calibration](docs/judges-and-calibration.md#step-3-guardrails-are-not-capabilities) | [`guardrails.py`](examples/guardrails.py) |
| Equivalence, the only verdict that says "good enough" | A margin by measure: `compare(margins={"accuracy": 0.05})` (recorded on the runs) or a scorer's, `margins={"correct": 0.05}`; a scorer returning a number, not a bool, needs its range too (`ranges={"rating": (1, 5)}`). On a host: `MetricDescriptor.materiality_threshold` and `value_range` (with no range it is never tested) | [Reading a comparison](docs/reading-reports.md#reading-a-comparison) | [`compare_two_efforts.py`](examples/compare_two_efforts.py) |
| Typed verdicts and a CI gate | `Comparison.verdicts()`, `Comparison.gate()`, `python -m threetears.evals gate --fail-on regressed,breached` | [The command line](docs/command-line.md#gate) | [`reports.py`](examples/reports.py) |
| Cassettes | `tools=`, `cassette_mode="capture"` / `"replay"`, `cassette_corpus_id=` | [Adopting the engine](docs/adopting-a-host.md#cassettes-recording-and-replaying-tools) | [`cassettes.py`](examples/cassettes.py) |
| Grids of factors | `compare(..., factors=...)`, `Comparison.against` | [Choosing a campaign design](docs/choosing-a-design.md#start-from-the-question) | [`prompt_x_effort.py`](examples/prompt_x_effort.py) |
| Campaigns and declared designs | `create_campaign`, `update_campaign`, `add_runs_to_campaign`, `set_campaign_control`, `declared_design`, `exploratory`, `control_from_run_id` | [Choosing a campaign design](docs/choosing-a-design.md#declare-the-design-and-its-control) | [`compare_two_prompts.py`](examples/compare_two_prompts.py) |
| Cost against quality | `Answer`, `max_cost_usd=`, the frontier | [Cost and budgets](docs/cost-and-budgets.md), [The frontier](docs/choosing-a-design.md#the-frontier-passk-against-cost) | [`compare_two_efforts.py`](examples/compare_two_efforts.py) |
| The analysis bundle and reports | `inspect_campaign_bundle`, `AnalysisContextBundle`, `report_markdown`, `report_html` | [Reading reports](docs/reading-reports.md) | [`reports.py`](examples/reports.py) |
| Analysis written by a model over frozen evidence | `generate_analysis` | [Having a model write the analysis](docs/reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence) | [`llm_analysis.py`](examples/llm_analysis.py) |
| Judge calibration and evidence tiers | `rate_result`, `judge_agreement`, `JudgeEvidenceTier` | [Judges and calibration](docs/judges-and-calibration.md) | [`llm_judge.py`](examples/llm_judge.py) |
| MCP | `threetears.evals.actions`, `mount_fastmcp` (extra `fastmcp`) | [Driving it from an agent](docs/agents-and-mcp.md) | |
| Keeping runs | `run_eval(..., store=SqliteDocumentStore(path))`, the same on `compare`; any `DocumentStore`, proven with `STORE_CONFORMANCE_CASES` | [Keep your runs](docs/tutorial.md#8-keep-your-runs), [The store](docs/adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it), [What is kept](docs/adopting-a-host.md#stored-data-what-is-kept) | |
| A complete host | `EvalHost`, `HostProfile`, `CandidateKind` | [Adopting the engine](docs/adopting-a-host.md#read-the-reference-hosts-in-this-order) | [`courierhost`](tests/fixtures/courierhost/__init__.py), then [`toyhost`](tests/fixtures/toyhost/README.md) |

Import only from the public roots the [reference](docs/reference.md#public-api)
lists: a module below a root is internal and may move. A public name that leaves a root is recorded with its new
home in each release's notes ([Upgrading: names that move](docs/adopting-a-host.md#upgrading-names-that-move)).
