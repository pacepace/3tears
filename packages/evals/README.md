# 3tears-evals

Evals for LLM features, from a classifier to a tool-using agent: it runs each version over the same cases and says
what moved, with the uncertainty that says whether the move is real. Python 3.14+.

```bash
pip install "3tears-evals @ git+https://github.com/pacepace/3tears.git@v0.68.0-evaladopt.27#subdirectory=packages/evals"
```

**Rung 0 is one TOML file and no Python.** [`examples/1-triage/triage.toml`](examples/1-triage/triage.toml) triages
30 support emails, each with the triage a person gave it, three times each:

```toml
# Rung 0, no Python: python -m threetears.evals triage.toml
cases = "emails.jsonl"
grader = ["category", "urgent", "needs_human"]
recording = "recordings/triage-toml.jsonl"
k = 3

[runner]
model = "deepseek/deepseek-v4-flash"
prompt = "Triage this support email:\n\n{email}"

[runner.output]
category = ["billing", "bug", "account", "sales", "spam"]
urgent = "bool"
needs_human = "bool"
```

`python -m threetears.evals examples/1-triage/triage.toml` replays the recorded model, offline and free. It prints
each field's pass rate (`needs_human: mean 0.8111 (n=90, min 0, max 1)`), a confusion matrix for `category`, the top
misses, and where its report is. Then climb [the ladder](docs/ladder.md), one idea per rung:

| Rung | Example | Climb it when |
|---|---|---|
| [1](docs/ladder.md#rung-1-your-own-function) | [`1-triage/triage.py`](examples/1-triage/triage.py) | Point it at your own function: 15 lines of Python. |
| [2](docs/ladder.md#rung-2-a-judge-checked-against-a-person) | [`2-replies/`](examples/2-replies/replies.py) | When no code can grade the answer, let a model judge it, and check the judge against a person. |
| [3](docs/ladder.md#rung-3-two-prompts-compared) | [`3-invoices/`](examples/3-invoices/extract.py) | Changing a prompt? Compare it against the current one, and get a report you can hand to a reviewer. |
| [4](docs/ladder.md#rung-4-an-agent-graded-on-the-world-it-leaves) | [`4-booking/`](examples/4-booking/agent.py) | Is your agent doing the right thing to the world, not only saying it? Then serve your evals to the team. |

**Where the science lives:**
- [Measuring soundly](docs/measuring-soundly.md): which design answers your question, how many cases, and the traps.
- [Judges and calibration](docs/judges-and-calibration.md): whether to trust a judge, and how far.
- [Reading reports](docs/reading-reports.md): every verdict, interval and disclosure a report prints.

**Wiring it into your app** (your store, your launcher, your world): [Adopting the engine](docs/adopting-a-host.md).

## Install

**3tears-evals is not on PyPI until 0.68.0 ships**, so `pip install 3tears-evals` finds nothing yet. Until then,
install it from the repository at a tag. The line above is the one place the docs name that tag, and every other
page's install line points here. The FastMCP transport, for serving evals to an agent, is the `[fastmcp]` extra:

```bash
pip install "3tears-evals[fastmcp] @ git+https://github.com/pacepace/3tears.git@v0.68.0-evaladopt.27#subdirectory=packages/evals"
```

Once 0.68.0 is on PyPI these are `pip install 3tears-evals` and `pip install "3tears-evals[fastmcp]"`. In a checkout
of this repo, `uv sync` at the root does it, and the commands above run from `packages/evals` with `uv run` in front.
`threetears.evals.__version__` says which version is installed. The completion client (`Model`) is on the base
install: it speaks each provider's HTTP API over `httpx`, with no SDK and no other 3tears package. The examples run on
cheap models through OpenRouter (`OPENROUTER_API_KEY`); `Model(id, provider="anthropic")` (`ANTHROPIC_API_KEY`) and
`provider="openai"` (`OPENAI_API_KEY`) are options, and `base_url=` takes any OpenAI-compatible endpoint. After a pip install, fetch the examples from the repository first
([how](examples/README.md#running-them)).

> **The public API is still changing.** From 0.68.0 only its host contract is promised: a name on it moves with a
> deprecation cycle, and any other declared name may change in any release, recorded in `MIGRATIONS.toml`. About 450
> names are declared, [mapped by rung](docs/surface.md); the rest a root exports are provisional
> ([stability tiers](docs/adopting-a-host.md#stability-tiers)).

## What is different

- **A verdict, not a mean.** Every version is run over the same frozen cases and tested case by case against a control,
  and evidence that cannot tell two versions apart says "not separated", never "no difference"
  ([why](docs/design.md#numbers)).
- **Guardrails apart from capability.** What a version must never do is decided on its own, so a gain cannot pay for
  a breach ([why](docs/design.md#guardrails-are-a-pillar-apart-from-capability)).
- **A judge is measured, not trusted.** Every judged reading carries the evidence tier its judge's agreement with
  people and with itself reached ([how](docs/judges-and-calibration.md#step-7-evidence-tiers-and-how-many-ratings-they-take)).
- **An agent is graded on the world it leaves,** a world your app declares, with checks proven to beat doing nothing
  and a conformance kit your app runs against its own world ([what is new](docs/evaluating-agents.md#what-is-new-here)).

## Docs

These links are relative to the package, so on GitHub each opens the page at the ref you are reading; on PyPI, where
this README is the package's description, they do not resolve, so read the docs in the repository at your tag.

**Climb**

- [Building an inventory](docs/building-an-inventory.md): what to test before a line of code, ending in rung 0; then
  which cases a classifier needs, and how many.
- [The ladder](docs/ladder.md): rungs 1 to 4, through the four examples.
- [Concepts](docs/concepts.md): the six concepts, and the glossary of every other word.

**Go deeper**

- [Judges and calibration](docs/judges-and-calibration.md): rubrics, labels, and whether to trust the judge.
- [Reading reports](docs/reading-reports.md): what each part of a comparison and a report means.
- [Measuring soundly](docs/measuring-soundly.md): which design answers a question, how many cases, and the traps
  campaigns have hit.
- [Evaluating a tool-using agent](docs/evaluating-agents.md): a world, seeds, goal checks, the conformance kit, and
  why the world contract is shaped as it is.
- [Cost and budgets](docs/cost-and-budgets.md): what a launch will cost, and what stops it.
- [Serving](docs/serving.md): the command line, keeping and reading runs, the actions an agent calls, and MCP.
- [Adopting the engine](docs/adopting-a-host.md): your own host, store, kind and launcher.

**Background and reference**

- [Design](docs/design.md): the principles the engine keeps, why each concept is shaped as it is, and the prior art.
- [Open problems](docs/open-problems.md): known gaps.
- [The surface](docs/surface.md): the declared names on one page, by rung.
- [Reference](docs/reference.md): every declared name, setting, report field, goal-check builtin and action, generated
  from the code.

## For experts

| Capability | API | Doc | Example |
|---|---|---|---|
| World contract | `World(State, tools=[...])`, each case's `world` key, expression graders; on a host, `WorldRegistry` | [Evaluating a tool-using agent](docs/evaluating-agents.md) | [`4-booking/`](examples/4-booking/agent.py) |
| World conformance kit | `check_world_conformance` (`threetears.evals.kernel.host`) | [Step 5](docs/evaluating-agents.md#step-5-run-the-conformance-kit-against-your-world) | |
| Guardrails, decided apart from capability | `grader(fn, guard=...)`, `Judge(guard={dim: ...})`, `Result.guardrails()`; on a host, `MetricDescriptor(guardrail=True)`, `RubricDim(axis="boundary")` | [Reading the guardrails](docs/reading-reports.md#reading-the-guardrails) | [`3-invoices/`](examples/3-invoices/extract.py) |
| Equivalence, the only verdict that says "good enough" | `grader("label", margin=0.05)`, `grader(fn, margin=...)`, `grader(rating, range=(1, 5))`; on a host, `MetricDescriptor.materiality_threshold` and `value_range` | [Reading a comparison](docs/reading-reports.md#reading-a-comparison) | [`3-invoices/`](examples/3-invoices/extract.py) |
| Typed verdicts and a CI gate | `Result.verdicts()`, `Result.gate()`, `python -m threetears.evals gate --fail-on regressed,breached` | [`gate`](docs/serving.md#gate) | [`3-invoices/`](examples/3-invoices/extract.py) |
| Grids of factors | arms named as `k=v` pairs (`"prompt=v2, effort=low"`), `Result.against` | [Start from the question](docs/measuring-soundly.md#start-from-the-question) | [`3-invoices/`](examples/3-invoices/extract.py) |
| Campaigns and declared designs | `create_campaign`, `update_campaign`, `add_runs_to_campaign`, `declared_design`, `exploratory`, `control_from_run_id` | [Declare the design](docs/measuring-soundly.md#declare-the-design-and-its-control) | |
| Cost against quality | `Answer`, `max_cost_usd=`, the frontier | [Cost and budgets](docs/cost-and-budgets.md), [the frontier](docs/measuring-soundly.md#the-frontier-passk-against-cost) | |
| Judge calibration and evidence tiers | `Judge(labels=...)`, `rate_result`, `judge_agreement`, `JudgeEvidenceTier` | [Judges and calibration](docs/judges-and-calibration.md) | [`2-replies/`](examples/2-replies/replies.py) |
| Analysis written by a model over frozen evidence | `generate_analysis` | [Having a model write the analysis](docs/reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence) | [`llm_analysis.py`](tests/fixtures/generate_analysis/llm_analysis.py) |
| Cassettes | `World(tools=[...], recording=...)`; on a host, `cassette_mode="capture"` / `"replay"` | [Step 4](docs/evaluating-agents.md#step-4-cassettes-for-tool-results), [on a host](docs/adopting-a-host.md#cassettes-recording-and-replaying-tools) | |
| MCP | `threetears.evals.actions`, `mount_fastmcp` (extra `fastmcp`) | [Serving](docs/serving.md#fastmcp) | [`4-booking/serve.py`](examples/4-booking/serve.py) |
| Keeping runs | `Eval(...).run(store=path)` (by default `.evals/evals.sqlite`); any `DocumentStore`, proven with `STORE_CONFORMANCE_CASES` | [Keep your runs](docs/serving.md#keep-your-runs-and-read-them-back), [the store](docs/adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it) | |
| A complete host | `EvalHost`, `HostProfile`, `CandidateKind` | [Adopting the engine](docs/adopting-a-host.md#read-the-reference-hosts-in-this-order) | [`courierhost`](tests/fixtures/courierhost/__init__.py), then [`toyhost`](tests/fixtures/toyhost/README.md) |

Import only from the public roots the [reference](docs/reference.md#public-api) lists: a module below a root is
internal and may move. A public name that leaves a root is recorded with its new home in each release's notes
([Upgrading: names that move](docs/adopting-a-host.md#upgrading-names-that-move)).
