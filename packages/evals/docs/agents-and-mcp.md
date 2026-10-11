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

**No transport call waits on a model.** Every operation whose work calls one is a **job**: `run_launch`,
`sweep_launch`, `analysis_generate`, `judge_second`, `judge_drift_check` and `judge_temperature` return
`JobsStarted`, and `job_poll` / `job_cancel` take any job id they returned. The call does everything that can
refuse before money is spent — the refusals, the pricing against the cap the work is held to, and the admission
ceiling — and refuses there with nothing spent; the job does the spending. So a call is a few store round-trips,
and a transport holds every call to one fixed timeout (`mount_fastmcp(..., call_timeout_s=...)`, 60s by default)
rather than one sized to what a call might spend.

Each job writes its own record, and polling reads it:

| Job id | Started by | Its record | What `job_poll` returns once it completes |
|---|---|---|---|
| `launch:<id>` | `run_launch` | `EvalLaunch`: the case generation and every arm's preparation, then the runs to their end | `run_ids`, in arm order (from the moment they start), and each run's status in `progress` |
| `run:<id>` | one run of a launch | the `EvalRun` | the run's status; read it with `run_get` |
| `sweep:<id>` | `sweep_launch` | `EvalSweep` | the arms launched and finished |
| `analysis:<campaign>:<attempt>` | `analysis_generate` | `EvalAnalysisAttempt` | the stored analysis's id; read it with `report_read` |
| `second-judge:<id>` | `judge_second`, `judge_drift_check` | `EvalSecondJudgePass`, its id the pass id every paired score names | `result`: the pass's report, agreement and drift |
| `judge-temperature:<id>` | `judge_temperature` | `EvalJudgeTemperatureCheck`, every answer kept | `result`: the comparison |
| `action:<id>` | a host action ([Adopting the engine](adopting-a-host.md)) | `EvalActionJob` | `result`: the action's typed result |

A launch is one job now, not one per arm: its arms are prepared inside it — where a generating launch
(`n_variations`) calls its variation writer — and its record names its runs once they start, so an agent polls
the launch and reads each run with `run_get`. One `job_cancel` of a launch stops every run it started.

A job id names the durable record its work writes, so
every process sharing the store answers it alike, and it is still answerable after a restart. Whether a job is
still running is its record's answer: the process running it holds a **lease** on the record and renews it
(every third of `LaunchHost.job_lease_s`, 60s by default). A job whose record has not ended and whose lease has
lapsed reads `lost`, and the first poll to see that writes it on the record. A cancel is written on the record
too, so `job_cancel` works from any process: the job ends `cancelled` within one renewal interval. A graceful
shutdown writes every job its process holds `lost` at once. A job is answered only in the caller's scope:
another scope's job reads not found, on poll and on cancel.

**In the foreground.** A caller that is not a transport — a script, a test, a host's own batch — may await
what it started instead of polling: `await_job(host, started, scope_id)` waits for the jobs to end and returns
each one's final `JobStatus`, result included (cancelling the wait cancels the jobs). Below the operations,
`run_eval` and the command line await the same work directly (`start_run`, and
`run_job_work(judge_temperature_work(...))` for `judge-temperature`), so a short-lived consumer loses nothing.

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
in dollars before it spends: a launch by the per-run ceiling and its case generation by the host's out-of-run
cap, an analysis generation and the judge actions by the host's out-of-run cap. With enforcement off, none is
capped. The full rules are in [Cost and budgets](cost-and-budgets.md).

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

Every `spend` action — the engine's and a host's — declares a `happy_path`: the parameters (and what it arranges
first) that reach its work. One without is refused when it is declared. The jobs conformance kit
(`threetears.evals.testing.jobs_conformance_cases`) runs every action of a catalogue under a guarded client
factory: each call must make no completion request, and each spend action's happy path must return
`JobsStarted` and its job make at least one. See [Adopting the engine](adopting-a-host.md).

An agent calls `action='help'` for the actions grouped by workflow and `action='help', topic=<action>` for
one action's parameters and an example. A parameter the action does not declare is refused, naming the
ones it accepts. `read_only_tools(prefix)` mounts a tool an agent can only read through.

## Reading one result

`run_get` says how a run came out; it does not say what one cell did. Its structured result (an `EvalSummary`)
carries the words its text prints beside the numbers, so a client shows them rather than re-spelling them:
`headline`, `results_text`, `stopped_text` (beside `stopped_because`), `spend_cap_text`, `measure_lines` (each
measure's `state` — `measured`, `not_a_number`, `in_confusion_matrix`, `no_successful_results`, `all_excluded`
or `not_carried` — and its line), `confusion_lines`, `label_lines`, `intent_text`, `dimension_lines`,
`self_judging_warning`, `judge_spend_text`, `candidate_spend_text`, `goal_check_lines`, and `text`, the whole of it
as `render()` prints it. A field with nothing to say is null or empty, and the summary prints no line for it.

Two read actions do say what one cell did:

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

## Refusals as data

Every refused call has two forms. The text teaches an agent what was wrong and what would be right. The
`refusal` on the `ActionOutcome` is the same refusal for your code: a `Refusal` with a `kind` and a `status`.
Branch on the kind, never on the text.

| `kind` | Raised as | `status` |
|---|---|---|
| `not_found` | `NotFoundError` | 404 |
| `conflict` | `ConflictError` | 409 |
| `validation_failed` | `ValidationFailedError`, and every call the catalogue turns away itself | 422 |
| `admission_refused` | `AdmissionRefusedError` | 429 |
| `provider_refused` | `ProviderRefusedError` | 502 |
| `storage_failed` | `StorageError` | 503 |
| `refused` | `EvalServiceError` itself | whatever it was raised with (400 by default) |

The set is closed: one kind per public error class in `threetears.evals.kernel`, listed in
`REFUSAL_KINDS`. An error class your own actions derive refuses as its nearest public ancestor's kind, and
keeps the status it was raised with. The catalogue's own refusals (no action, an unknown one, an undeclared
parameter, a value that does not validate, a help topic that is no action) are the call failing the tool's
declared input, so they are `validation_failed`. A call that succeeded, and help, carry no refusal.

A handler that raises anything else has a bug. It is not a refusal, and it propagates to the transport.

## FastMCP

The FastMCP transport (`threetears.evals.transports.fastmcp`) needs the extra; the core does not:

```bash
pip install "3tears-evals[fastmcp]"
```

A result carries its text and the result model's JSON as structured content. A refused call is an MCP
error result (`isError`) carrying the teaching text, with the refusal as structured content:

```json
{"refusal": {"kind": "not_found", "status": 404}}
```

### Serving several tenants

`mount_fastmcp` (and `CatalogueTool.over`, for one tool) takes the hooks a multi-tenant server needs, so you
never rebuild the tool yourself:

```python
from collections.abc import AsyncGenerator
from threetears.evals.actions import ActionOutcome, Caller, eval_catalogue
from threetears.evals.transports.fastmcp import ToolCall, mount_fastmcp

async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
    with tracer.start_as_current_span(f"{call.tool}.{call.action}"):
        outcome = yield                       # the call runs here
        calls.add(1, {"action": outcome.action, "refusal": outcome.refusal.kind if outcome.refusal else "none"})

mount_fastmcp(
    server,
    eval_catalogue(my_actions),
    host_resolver=lambda: tenant_hosts[current_tenant()],   # in place of host=
    caller=lambda: Caller(scope_id=current_tenant(), identity=current_user()),
    around=around,
)
```

- **`host_resolver`** answers which `OpsHost` this call works in, per call, from the same request context as
  the caller. It may be sync or async, and is asked after the caller. Pass `host` for a server with one host,
  or `host_resolver`: exactly one of the two, or the mount is refused before any tool is added.
- **A resolver that raises fails the call before anything runs.** No action runs and the hook is not
  entered. The client sees the transport's error result, not a refusal. A resolver that answers with
  something other than a `Caller` or an `OpsHost` fails the same way, with a `TypeError` naming what it
  answered.
- **`around`** wraps every call to every tool, for tracing, metrics or an audit log. It is an async generator
  function. It receives a `ToolCall` (the tool, the `action` argument as given, and the caller), yields once,
  and receives the `ActionOutcome` at the yield. Anything it opens before the yield is current while the
  action runs, so a span it starts is the parent of the action's own. A call that raises a defect, rather
  than refusing, is thrown in at the yield, so a `with` or `finally` closes. The defect still propagates,
  whatever the hook does.
- **The one call timeout holds the whole call**, the caller and host resolvers and the hook with it. A call it
  stops is closed in the hook at its yield, so the hook's `finally` runs, and the client gets the timeout's
  error result.

**The hook cannot change the outcome.** What it yields is discarded, the outcome is frozen, and the client
gets exactly what the catalogue returned.

**A hook that raises is logged and otherwise ignored.** If it fails before its yield, the call runs
unobserved. If it fails after, or yields a second time, the outcome stands. This is deliberate. A hook
observes. If its failure could fail the call, an exporter outage would refuse every eval call. Worse, a
failure after the action ran would report an error for work that was done (a launch that spent, a write
that landed), and an agent would repeat it. An audit that must hold *before* an action runs belongs in the
host or the action, which can refuse. A hook is not a gate.

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

- `judge_template_create` (write) writes the judge template (`template_name`, `intent`): the judge kind and an intent,
  nothing else, since its cases carry everything a trial reads. Refused on a host that launches no judge kind.
- `judge_cases_freeze` (write) freezes judged runs (`judged_run_ids`)'s stored outputs into cases of a judge template: per judged dim of
  each result, the evidence its judge read and the person ratings given on it as labels. With `into_case_set`, it mints
  the set a launch targets. It answers with each case and every result it skipped, with why.
- Then `run_launch` the judge template, one arm per judge: the arm's model, and its `config_ids` and `temperature`
  as overlays. The trials call only the judge, and their spend is recorded under the `judge` role.
  See [judges and calibration](judges-and-calibration.md#evaluating-a-judge-as-a-subject).
- `judge_profiles_record` (write) stores, from the campaign's finished runs (`judge_run_ids`), one profile per judge
  and criterion: both agreements, parse validity, the cases and runs it was measured on, and when. Any campaign that
  judge scores then reads its evidence tier from the profile where its own evidence decides none, and says so.
  `judge_profiles_list` (read) lists them. See
  [storing a judge's profile](judges-and-calibration.md#storing-a-judges-profile-for-other-campaigns).
