# Driving it from an agent: operations, actions and MCP

**For** anyone wiring an agent, a REST route or their own CLI to launch runs, poll them and read reports.
**Answers:** how operations, the action catalogue and the FastMCP transport fit together, and how to read one
result. It assumes you already have a `LaunchHost` (see [Adopting the engine](adopting-a-host.md)) and uses
the terms in [Concepts](concepts.md).

## Three layers, in plain words

- An **operation** is one Python function per thing an operator does (launch a run, read a report, poll a
  job). Every surface calls the same ones, so the CLI, an MCP tool and a REST route cannot disagree.
- The **action catalogue** describes each operation once for an agent: a name, a permission class, flat
  parameters with descriptions, and how to render the result.
- A **transport** mounts the catalogue on a server. The package ships one for FastMCP.

## Operations and jobs

The operations (`threetears.evals.ops`) run over an `OpsHost`: the `LaunchHost`, plus `AnalysisGeneration`
(the prompt, output cap and budget a background generation runs under). Each returns a typed model.

Long work is a **job**: `run_launch`, `sweep_launch` and `analysis_generate` return `JobsStarted`, and
`job_poll` / `job_cancel` take any job id they returned. A job id names the durable record its work writes, so
it is still answerable after a restart. A job is answered only in the caller's scope: another scope's
generation or sweep reads `lost` on poll and is refused on cancel.

**A sweep** (`sweep_launch`) is for arms that differ in more than the model: each arm names its own model,
overlays and apparatus settings, and every arm shares the template, subject, `k` and the other settings.
Every arm is refused up front if its launch would be. The arms run one at a time, so no arm shares another's
provider load; `max_concurrent_arms` raises that. To compare latency, also declare `measure_latency`, which
every arm's launch takes. Each run joins the named campaign
as it is created. Its one job's progress counts the arms launched and finished, and one `job_cancel` stops
the arm in flight and launches none after it. An arm refused at its launch, or whose run ends `failed`, ends
the sweep `failed`. The judge and simulator are the sweep's. Where it names none, the first arm's resolved
ones hold for the rest, so a role default that moves mid-sweep cannot split the arms.

On a host that enforces its ceilings (`LaunchSettings.enforcement_enabled`), every spend operation is bounded
in dollars before it spends: a launch by the per-run ceiling, an analysis generation and a judge repeat by the
host's out-of-run cap. With enforcement off, none is capped. The full rules, including that a host's own
`spend` actions carry no such obligation, are in [Cost and budgets](cost-and-budgets.md).

## The action catalogue

The catalogue (`threetears.evals.actions`) names each action `noun_verb`, with a permission class (`read`,
`spend`, `write`, `destructive`). A host adds its own actions and cuts tools by class:

```python
from fastmcp import FastMCP
from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.ops import OpsHost
from threetears.evals.transports.fastmcp import mount_fastmcp

server = FastMCP("myapp")
mount_fastmcp(
    server,
    eval_catalogue(my_actions),               # the engine's actions, then the host's
    host=OpsHost(launch=launch_host, generation=my_generation),
    caller=lambda: Caller(scope_id=current_scope(), identity=current_user()),
    tools=standard_tools("evals"),            # `evals` (read, spend, write) and `evals_admin` (destructive)
)
```

An agent calls `action='help'` for the actions grouped by workflow and `action='help', topic=<action>` for
one action's parameters and an example. A parameter the action does not declare is refused, naming the
ones it accepts. `read_only_tools(prefix)` mounts a tool an agent can only read through.

## Reading one result

`run_get` says how a run came out; it does not say what one cell did. Two read actions do:

- `results_list` pages one run's results as light rows (ordered by case, then repeat): each row's case,
  repeat, model, variant, condition (`ok`, `candidate_fail` or `infra_exclude`), cost, `total_ms`, goal checks as every
  rate counts them, judge scores and host measures. `condition_filter` narrows to one condition; `total`,
  `next_offset` and `limit` (default 50, at most 200) page it.
- `result_get` reads one result back as stored, with one `part` of its trace:
  - `record`, the default: the result record with its per-role usage rows and every error field, and its
    condition with the sentence every surface shows for it. Its latency follows: the five stored components
    (an unmeasured one reads `absent`, never zero) and the `orchestration_ms` remainder of `total_ms`, or the
    sentence saying why that split is withheld. Each goal check is shown as evaluated, and
    also as counted when the two differ: a candidate failure counts every check failed, and a harness
    fault counts none. Then come the output documents exactly as the kind stored them, the call ledger
    and the world's end state.
  - `judge`: what the judge was sent.
  - `spans`: the stored spans.

  Every part carries the result record and its condition. The default leaves the judge's evidence and the
  spans out, since either can outweigh everything else, and says how to ask for them. A refused call shows
  in the output documents only if the kind records failures; the call ledger holds only succeeded calls. A
  trace the record says was written but whose document is missing reads as missing, not as none stored.

A run or a result outside the caller's scope is not found, and so is an id of another type, such as a run's
id passed to `result_get`. A listing never comes back empty in place of not found.

## FastMCP

The FastMCP transport (`threetears.evals.transports.fastmcp`) needs the extra; the core does not:

```bash
pip install "3tears-evals[fastmcp]"
```

## Reporter runs: evaluating the analysis writer itself

A reporter run (the `analysis_reporter` kind, which measures the analysis writer itself) starts from a
frozen case, never a generated one.

- `reporter_case_freeze` (write) freezes a campaign's analysis bundle into a case of a reporter template —
  with `recorded_analysis_id`, the memo the campaign got too, and any reader `labels` on it — and answers
  with the case id, its bundle fingerprint and the `limits` the freeze recorded.
- `reporter_cases_list` reads the template's bank: which case each campaign and memo launches, and what
  superseded or retired the rest.
- `reporter_case_archive` retires a case or restores it.

The template then launches through `run_launch`, by the reporter kind the host registers.

## Judge campaigns: evaluating a judge itself

A judge campaign (the `judge` kind) makes a judge configuration the subject: its model, its prompt per criterion and
its temperature. It starts from frozen cases, never generated ones.

- `judge_cases_freeze` (write) freezes judged runs (`judged_run_ids`)'s stored outputs into cases of a judge template: per judged dim of
  each result, the evidence its judge read and the person ratings given on it as labels. With `into_case_set`, it mints
  the set a launch targets. It answers with each case and every result it skipped, with why.
- Then `run_launch` the judge template, one arm per judge: the arm's model, and its `config_ids` and `temperature`
  as overlays. The trials call only the judge, and their spend is recorded under the `judge` role.
  See [judges and calibration](judges-and-calibration.md#evaluating-a-judge-as-a-subject).
