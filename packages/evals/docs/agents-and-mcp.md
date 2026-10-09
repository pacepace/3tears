# Driving it from an agent: operations, actions and MCP

Read this if you want an agent (or a REST route, or your own CLI) to launch runs, poll them and read
reports. It assumes you already have a `LaunchHost` — see [Adopting the engine](adopting-a-host.md) — and
uses the terms in [Concepts](concepts.md).

## Three layers, in plain words

- An **operation** is one Python function per thing an operator does (launch a run, read a report, poll a
  job). Every surface calls the same ones, so the CLI, an MCP tool and a REST route cannot disagree.
- The **action catalogue** describes each operation once for an agent: a name, a permission class, flat
  parameters with descriptions, and how to render the result.
- A **transport** mounts the catalogue on a server. The package ships one for FastMCP.

## Operations and jobs

Every surface calls the same **operations** (`threetears.evals.ops`): one function per thing an operator
does, over an `OpsHost` — the `LaunchHost`, plus `AnalysisGeneration` (the prompt, output cap and budget
a background generation runs under) — returning a typed model.

Long work is a **job**: `run_launch` and `analysis_generate` return `JobsStarted`, and `job_poll` /
`job_cancel` take any job id either returned. A job id names the durable record its work writes, so it is
still answerable after a restart. A job is answered only in the caller's scope: another scope's
generation reads `lost` on poll and is refused on cancel.

Every spend operation is bounded in dollars before it spends: a launch by the per-run ceiling, an analysis
generation and a judge repeat by the host's out-of-run cap. The full rules, including that a host's own
`spend` actions carry no such obligation, are in [Cost and budgets](cost-and-budgets.md#spend-outside-any-run).

## The action catalogue

The **action catalogue** (`threetears.evals.actions`) declares each operation once for an agent: a
`noun_verb` name, a permission class (`read`, `spend`, `write`, `destructive`), flat described
parameters, a result and its rendering. A host adds its own actions and cuts tools by class:

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

- `results_list` pages one run's results as light rows, ordered by case, then repeat, then id: each row's
  id, case, repeat, model and variant, its condition (`ok`, `candidate_fail` or `infra_exclude`), how the
  cell ended, its cost, its goal checks as every rate counts them, its judge scores and its host measures.
  `condition_filter` narrows the rows to one condition, `total` counts the rows that match across every
  page, and `next_offset` is the offset of the next page. A page holds 50 rows unless `limit` asks for
  fewer or more, and never more than 200.
- `result_get` reads one result back as stored, with one `part` of its trace:
  - `record`, the default: the result record with its per-role usage rows and every error field, and its
    condition with the sentence every surface shows for it. Each goal check is shown as evaluated, and
    also as counted when the two differ: a candidate failure counts every check failed, and a harness
    fault counts none. Then come the output documents exactly as the kind stored them, the call ledger
    and the world's end state.
  - `judge`: what the judge was sent.
  - `spans`: the stored spans.

  Every part's data carries the result record and its condition, so a reply always says whose part it is.

  The default leaves the judge's evidence and the spans out of both the text and the data, because either
  can outweigh everything else, and its text says how to ask for them.

  The output documents hold whatever the kind wrote about each call. A failed call, with what the tool
  returned, appears there only when the kind records failures. A kind that records only successes leaves
  no sign of a refused call. The call ledger holds only the calls the kind recorded as succeeded.

  A trace the record says was written but whose document is missing reads as missing, not as none stored.

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
