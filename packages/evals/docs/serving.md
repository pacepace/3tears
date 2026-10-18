# Serving: the command line, the actions and MCP

**For** anyone at rung 4 who wants the team to read an eval's runs, run evals from CI, or let an agent launch and
read them, and anyone mounting the same commands under their own app. **Answers:** how a file runs from the command
line, how runs are kept and read back, how [`examples/4-booking/serve.py`](../examples/4-booking/serve.py) serves
them, what each host command does and exits with, and how operations, the action catalogue and the FastMCP
transport fit together. Evals are administrative: these surfaces are for the people and agents who build and judge
your app, never its end users. The terms are those of [Concepts](concepts.md).

## Running a file

`python -m threetears.evals FILE` runs every eval a `.py` or `.toml` file declares:

- **A TOML file** declares one eval ([rung 0](building-an-inventory.md#7-write-it-down-and-run-it-15-min)).
- **A Python file** is imported, and every `Eval` it defines at module level runs, in the order it defines them. No
  `asyncio.run` is needed in it ([the ladder](ladder.md)).

Each eval runs into its default store (`.evals/evals.sqlite` under the directory you run from, in a scope named after
the eval), prints its result quietly (`--verbose` prints it whole), and writes its report to `.evals/reports/<name>/`.
The exit code is the worst of the evals' (the [table below](#exit-codes)): `4` when a comparison's gate failed, so a
CI step that runs a comparison file fails on a regression, a breach, or a guardrail not shown held. Any other first
argument is a host command ([below](#the-host-commands)).

## Keep your runs, and read them back

A kept run is found again by its scope (`scope=`, the eval's name when you name none), and kept runs have a history:
how a reading moved from one run to the next. This script keeps its runs in a file of its own and reads that history:

```python
import asyncio

from threetears.evals import Eval
from threetears.evals.ops import scope_history
from threetears.evals.quick import as_host
from threetears.evals.run import run_blocking
from threetears.evals.storage import SqliteDocumentStore

CASES = [
    {"id": "double-charge", "ticket": "I was charged twice for March.", "queue": "billing"},
    {"id": "export", "ticket": "The export button does nothing.", "queue": "bug"},
    {"id": "locked", "ticket": "I'm locked out after too many login attempts.", "queue": "account"},
]


async def route(case: dict) -> str:
    """Route a support ticket to billing, bug or account."""
    text = case["ticket"].lower()  # your prompt and model call go here; a keyword script stands in
    return "billing" if "charged" in text else "bug" if "button" in text else "account"


def correct(case: dict, answer: str) -> bool:
    """Whether the ticket went to its queue."""
    return answer == case["queue"]


async def main() -> None:
    store = SqliteDocumentStore("triage.sqlite")
    ev = Eval(CASES, route, [correct], k=2)
    await ev.run(store=store, scope="tickets")
    host = as_host([ev], store=store)
    history = await run_blocking(host.blocking_executor, scope_history, host, "tickets", metric="correct")
    for series in history.series:
        for point in series.points:
            step = f", {point.regression.label} from the run before" if point.regression else ""
            print(f"{point.created_at} {series.model}: correct {point.value:.2f} over {point.n} result(s){step}")


asyncio.run(main())
```

Run it twice. The second run reads the first run's results from the file, because they are stored there:

```
2026-10-14T18:28:17.246834+00:00 route: correct 1.00 over 6 result(s)
2026-10-14T18:28:19.915241+00:00 route: correct 1.00 over 6 result(s), not_separated from the run before
```

- **`as_host([ev], store=store)`** is the host the eval built for its run. Any later program reads the runs through
  it: the same graders, over a store opened on the same file.
- **`scope_history`** follows one reading over time, one series per runner, oldest run first; name a grader as
  `metric=`. It reads the store, which blocks, so async code calls it through `run_blocking` on the host's executor:
  no store call is ever made on the event loop. Each run is tested against the run before it, over the cases both
  ran. `regressed` is the step to act on; `not_separated`, as here, means the two runs could not be told apart, never
  that nothing changed.
- **A comparison's `run(store=store)`** keeps its arms and their campaign the same way, and a later program reads the
  comparison on the margins, ranges and guardrails it ran under, whatever its own host declares.
- **Keep the file across releases.** Runs, results and cases are the evidence core: a later release reads them as
  stored. A comparison's campaign and analysis are not kept that way: when their format changes, a release refuses
  them, and you make them again over the kept runs ([what is kept](adopting-a-host.md#stored-data-what-is-kept)).

## Serving the runs to the team

[`serve.py`](../examples/4-booking/serve.py), beside rung 4's agent, is the whole server. Its `host()` is
`as_host([booking], store=".evals/evals.sqlite")`, the store every run of `agent.py` fills; hand `as_host` any
`DocumentStore` instead (your own database) to share the runs. It serves them two ways:

- **Over MCP** (`python serve.py`, the `[fastmcp]` extra): `mount_fastmcp` is handed that host, builds the operations
  host with its defaults, and mounts the read-only tools, so an agent can list the runs, read them and gate a
  comparison. Nothing is launched over it.
- **On the command line**, the same host by name: `python -m threetears.evals ls --host serve:host --scope booking`,
  and `report`, `bundle` and `gate` beside it ([the host commands](#the-host-commands)).

A REST transport is not built yet. A host of your own serves the same catalogue with your own launch path and tenancy
([FastMCP](#fastmcp), [Adopting the engine](adopting-a-host.md)).

## The host commands

Past a file, the command line works in a host you name as `module:factory`: a zero-argument callable returning an
`EvalHost`, or a `LaunchHost` for `run`.

```
python -m threetears.evals run    --host myapp.evals:build_host --scope dev --template T --subject S [--model M ...]
                                  [--k N] [--max-cost-usd DOLLARS] [--judge-model MODEL] [--simulator-model MODEL]
                                  [--n-variations N] [--variation-model MODEL] [--apparatus-settings JSON]
                                  [--measure-latency]
python -m threetears.evals ls     --host myapp.evals:build_host --scope dev
python -m threetears.evals report CAMPAIGN --host myapp.evals:build_host --scope dev [--format markdown|html|json] [--out DIR]
python -m threetears.evals bundle CAMPAIGN --host myapp.evals:build_host --scope dev
python -m threetears.evals spend  --host myapp.evals:build_host --scope dev [--purpose P] [--launch-group ID] [--template ID]
python -m threetears.evals frontier --host myapp.evals:build_host --scope dev [--bar B] [--subject S] [--json]
python -m threetears.evals gate   CAMPAIGN --host myapp.evals:build_host --scope dev [--fail-on OUTCOMES] [--reading MEASURE ...]
python -m threetears.evals judge-temperature RUN --host myapp.evals:build_host --scope dev (--max-cost-usd DOLLARS | --no-cap)
                                  [--repeats N] [--all] [--result ID ...] [--estimate] [--json]
```

**Runs you made with an `Eval`** are read the same way: the factory returns the host the evals run in, over the
store you kept them in ([Keep your runs](#keep-your-runs-and-read-them-back)), with the same graders:

```python
# myapp/evals.py
from threetears.evals import Eval
from threetears.evals.quick import as_host
from threetears.evals.storage import SqliteDocumentStore

from myapp.triage import CASES, correct, triage

TRIAGE = Eval(CASES, triage, [correct])


def build_host():
    return as_host([TRIAGE], store=SqliteDocumentStore("triage.sqlite"))
```

`ls`, `report`, `bundle`, `spend`, `frontier` and `gate` then read those runs. `run` needs a launch host, and
`judge-temperature` a host with completion clients (`as_host(..., clients=...)`).

### `run`

`run` launches, waits and prints each run's summary.

- Each `--model` is one arm and one run; with no `--model` the kind runs one arm on its own default
  model, and a kind with no default refuses the launch.
- `--k` is the repeats per case (the launch default, 3, when omitted).
- `--max-cost-usd` caps each run at or below the host's ceiling (a launch may only lower that ceiling; a
  value above it is refused).
- `--judge-model` and `--simulator-model` pin the judge and the simulated user (omitted, the kind's
  defaults apply).
- `--n-variations` and `--variation-model` generate that many cases first (priced against the host's
  out-of-run cap, outside the runs' caps; see [Cost and budgets](cost-and-budgets.md#spend-outside-any-run)).
- `--apparatus-settings` sets host-declared apparatus values as a JSON object.
- `--measure-latency` says latency is under test, so each run's cells run one at a time.

Each flag passes to one `start_run` argument: `--model` to `models`, `--k` to `k_runs`, `--template` to
`template_id`, `--subject` to `subject_id`, `--scope` to `scope_id`, and the rest to the argument of the same name.

### `report` and `bundle`

`report` prints the campaign's report (see [Reading reports](reading-reports.md#the-report-one-page-result-first)) —
its analysis, or, when it has none, a code-only report of its evidence. `bundle` prints, as JSON, the analysis
bundle a generation would read inside a wrapper (`BundleInspection`) that adds its campaign, scope and
fingerprint; the bundle itself is the wrapper's `bundle` field. Neither command calls a model.

### `spend`

`spend` prints what the engine spent outside any run in the scope — case generations, rubric proposals,
analysis generations, judge repeats and judge temperature comparisons
(`--purpose variation|proposer|analysis|judge|second_judge`) — narrowed by its flags.

### `frontier`

`frontier` prints each subject's variants ranked on pass^k, cost and latency, and, with `--bar`, the cheapest
that clears it ([The frontier](measuring-soundly.md#the-frontier-passk-against-cost)). `--subject` narrows it to
one subject, and `--json` prints the `FrontierResult` the `scope_frontier` action returns.

### `judge-temperature`

`judge-temperature` measures what temperature does to a run's judge: it re-judges the run's borderline cases
`--repeats` times (default 5) at the pinned judge temperature and as many times at the provider's default, and
prints each dimension's score variance, unstable cases and self-agreement at the two side by side
([Step 10](judges-and-calibration.md#step-10-measure-what-temperature-does-to-the-judge)). It is the one command
here besides `run` that spends: name the out-of-run cap with `--max-cost-usd`, or waive it with `--no-cap`. Every
call is priced against it before the first is sent and ledgered under purpose `judge`. `--estimate` prices it and
calls nothing. `--all` re-judges every scored dimension, `--result` narrows it to those results, and `--json` prints
the `JudgeTemperatureComparison` (or estimate). Nothing is written to the results; every answer is kept on the
comparison's own record (`EvalJudgeTemperatureCheck`) in the host's store, which the command writes and holds as the
`judge_temperature` action's job does, awaiting it in the foreground. A comparison
whose client did not send the temperature a side names prints `NOT COMPARABLE` with the reason and still exits `0`.

### `gate`

`gate` is the CI gate. It reads the campaign's typed verdicts as code reaches them on its evidence now (never an
analysis's words), prints the outcome and every verdict that failed or was not decided, and exits `4` when an
outcome `--fail-on` names occurred:

| Outcome | Fails on |
|---|---|
| `regressed` | a contrast shown worse than the control |
| `not-separated`, `untested` | a contrast the evidence could not decide |
| `breached` | a guardrail shown worse than the control by more than its margin |
| `undecided-guardrail` | a guardrail neither shown held nor breached |
| `missed`, `undecided-bar` | a bar a cell is shown to miss, or not shown to clear or miss |

The default is `--fail-on regressed,breached,undecided-guardrail`: a regression and a breach must not ship, and a
guardrail not shown held is not known to be safe. A verdict left undecided that `--fail-on` does not name exits
`0`, and the gate prints it as undecided, never as passed. To require every arm be shown improved or equivalent,
add `not-separated,untested`. `--reading` gates only the readings it names. A campaign with no verdict at all
(no control, no bar, no guardrail) is refused. In code, `Result.gate()` and `gate_verdicts()` do the same.

## Exit codes

| Code | Name | Meaning |
|---|---|---|
| `0` | `EXIT_OK` | done |
| `1` | `EXIT_RUN_DID_NOT_COMPLETE` | a launched run did not complete |
| `2` | `EXIT_REFUSED` | refused: a host that cannot be loaded, a template that is not there, a launch the engine refuses, a malformed command line |
| `3` | `EXIT_FAILED` | failed on an error nothing anticipated — a host factory, launcher or host command raising — with its traceback on stderr |
| `4` | `EXIT_GATE_FAILED` | `gate` read an outcome `--fail-on` names |

The names are in `threetears.evals.quick`.

## Mounting it under your own CLI

Mount the same commands under your own CLI with `run_cli(argv, host_factory=build_host, prog="myapp
evals")`; your users then never name the host.

## Operations, actions and transports: three layers

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
`JobsStarted`, and `job_poll` / `job_cancel` take any job id they returned. The call does what it can
refuse cheaply before money is spent — the refusals it can name from a few store reads, and the admission
ceiling — and refuses there with nothing spent; the job does the spending. An analysis generation does less in
its call: assembling its bundle reads every member run's results, so its job assembles it, builds the client and
prices the first call against the cap, and a refusal at any of those ends the job `failed` with
`refused before any call: <why>` for `job_poll` to read, with nothing spent. That preparation is bounded: one
that has not finished in `AnalysisGeneration.preparation_budget_s` (default `ANALYSIS_PREPARATION_BUDGET_S`, 900 s; the host may set it per call) ends the same way.
So a call is a few store round-trips, and a transport holds every call to one fixed timeout (`mount_fastmcp(..., call_timeout_s=...)`, 60s by default)
rather than one sized to what a call might spend. The timeout may be a function in place of the number, read once
as each call starts, so an operator retunes it from live configuration without remounting.

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

A launch is one job, not one per arm: its arms are prepared inside it — where a generating launch
(`n_variations`) calls its variation writer — and its record names its runs once they start, so an agent polls
the launch and reads each run with `run_get`. One `job_cancel` of a launch stops every run it started.

A job id names the durable record its work writes, so
every process sharing the store answers it alike, and it is still answerable after a restart. Whether a job is
still running is its record's answer: the process running it holds a **lease** on the record and renews it
(every third of `LaunchHost.job_lease_s`, 60s by default; all held leases at once, on a pool of its own that
cell work never queues behind). A job whose record has not ended and whose lease has
lapsed reads `lost`, and the first read to see that writes it on the record: not only `job_poll`, but every read
of the record — `run_get`, `runs_list`, a campaign's or a report's read of its runs, a sweep's, an analysis
attempt's — so a job whose process died reads `lost` everywhere once its lease lapses, and nothing counts it as
still running. A run whose stored document this build's model refuses cannot be written lost: the boot reclaim counts
it `unreadable` and leaves it, and `runs_unreadable` names each such run of the scope with why. `runs_list` leaves
such a run out and says how many it left out (`unreadable`). Every other read that reaches one (`run_get`,
`job_poll`, a comparison, a campaign's report, a launch's pricing) refuses with `UnreadableRunError` (status 500),
whose remedy names `runs_unreadable`: an answer over the runs that happened to read would be a wrong one.
`run_delete` removes such a run from its stored row, with what a readable run's delete removes. A `job_poll` of
another job whose record this build cannot read (a sweep's, a launch's, an analysis attempt's) refuses with
`UnreadableRecordError` (status 500), naming the record and why. A launch or a sweep
written lost settles the runs it names with it. A cancel is written on the record
too, so `job_cancel` works from any process: the job ends `cancelled` within one renewal interval. A graceful
shutdown writes every job its process holds `lost` at once. A job is answered only in the caller's scope:
another scope's job reads not found, on poll and on cancel.

**In the foreground.** A caller that is not a transport — a script, a test, a host's own batch — may await
what it started instead of polling: `await_job(host, started, scope_id)` waits for the jobs to end and returns
each one's final `JobStatus`, result included (cancelling the wait cancels the jobs). Below the operations,
`Eval.run` and the command line await the same work directly (`start_run`, and
`run_job_work(judge_temperature_work(...))` for `judge-temperature`), so a short-lived consumer loses nothing.

**A sweep** (`sweep_launch`) is for arms that differ in more than the model: each arm names its own model,
overlays and apparatus settings, and every arm shares the template, subject, `k` and the other settings.
Every arm is refused up front if its launch would be. How the arms share time is `arm_schedule`, the launch's own
argument (below). By default they are interleaved: every arm is prepared, then all start as one launch. `sequential`
launches one arm after another and `concurrent` launches them all at once, each its own launch. To compare latency,
also declare `measure_latency`, which every arm's launch takes and which runs the arms sequential. Launched one after
another or all at once, each run joins the named campaign as it is created; interleaved, every run joins once all
the arms are prepared and started. That campaign is created exploratory, since a sweep takes no design: declare what the arms were
swept to learn with `campaign_update` once they are known. Its one job's progress counts the arms launched and finished, and one `job_cancel` stops
the arms in flight and launches none after them. An arm refused at its launch, or whose run ends `failed`, ends
the sweep `failed`. The judge and simulator are the sweep's. Where it names none, the role's default is resolved
once for every arm (by the one interleaved launch, or from the first arm's run when they launch one by one), so a
role default that moves mid-sweep cannot split the arms.

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

**A campaign is declared or exploratory, and the caller says which.** `campaign_create` takes a
`declared_design` or `exploratory: true` — exactly one; neither is refused naming both, and both are refused as a
contradiction. `exploratory` is checked, never stored: a campaign is exploratory exactly while it declares no
design. `campaign_update` changes a campaign's `name`, `description` or `declared_design` after it exists, so an
operator who started exploring declares the design when the question settles, and the campaign is declared from
that write on. Going back is said too: `declared_design: null` withdraws the declaration only beside
`exploratory: true`.

**Everything a campaign needs is an action.** An agent can start ad hoc and end with a declared, controlled
campaign without a line of Python:

1. `run_launch` the arms, then `campaign_create` with their `run_ids` and `exploratory: true`;
2. `campaign_runs_add` the runs launched later (`campaign_runs_remove` takes one out; its results stay);
3. `campaign_update` with the `declared_design` once the question settles, and `control_from_run_id` naming the
   member run whose variant is the control — one write, so a refused control leaves the design unwritten too.
   `control_from_run_id` alone re-names the control on a declared campaign, and `control_from_run_id: null`
   clears it.

`judge_configs_list` names the judge configs a second judge's `second_judge_config_ids` can map a dimension to.

## Reading one result

`run_get` says how a run came out; it does not say what one cell did. Its structured result (an `EvalSummary`)
carries the words its text prints beside the numbers, so a client shows them rather than re-spelling them:
`headline`, `results_text`, `stopped_text` (beside `stopped_because`), `spend_cap_text`, `measure_lines` (each
measure's `state` — `measured`, `not_a_number`, `in_confusion_matrix`, `no_successful_results`, `all_excluded`
or `not_carried` — and its line), `confusion_lines`, `label_lines`, `stratum_lines`, `intent_text`,
`dimension_lines`, `self_judging_warning`, `model_identity_warning` (beside `unconfirmed_models`),
`request_settings_warning` (beside `request_settings`, each role's asked and sent settings), `spend_lines`,
`total_spend_text`, `goal_check_lines`, and `text`, the whole of it
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

**Refused or raised.** A refusal at a 5xx status (`provider_refused`, `storage_failed`, a host's own error raised
with one, a host that is still booting) is not the call turned away: the engine or the host failed to carry it out.
`ActionOutcome.ending` says which, `ok`, `refused` or `raised` (and `Refusal.raised` on the data), the text is headed
`raised:` in place of `refused:`, and the MCP transport's structured content carries it as `ending` beside the
`refusal`. A host recording an agent's refused calls on its ledger (`CallLedger.record_refusal`) records an outcome
whose `ending` is `refused` and no other: a raised call is the rig's fault, never the agent's.

A handler that raises anything else has a bug. It is not a refusal, and it propagates to the transport.

A job can be refused too, after its call returned the job's id: a provider refusing the account its work calls on,
a store failing, a cap refusing a call. The job ends `failed`, and its record keeps the refusal as data
(`schema.JobRefusal`: the same `kind` and `status` as above), which `job_poll` serves as `JobStatus.refusal` —
`{"refusal": {"kind": "provider_refused", "status": 502}}` in its JSON, and a `refusal:` line in its text. Branch on
that kind, never on `detail`'s words. A job broken by a defect carries none; a run's `status` is its own ending's
word (`exhausted` when its provider account refused it).

## FastMCP

The FastMCP transport (`threetears.evals.transports.fastmcp`) needs the extra; the core does not:

```bash
pip install "3tears-evals[fastmcp]"
```

To install the extra from a pre-release or an unreleased checkout, use the repository form at a tag
[the README shows](../README.md#install):
`pip install "3tears-evals[fastmcp] @ git+https://github.com/pacepace/3tears.git@<tag>#subdirectory=packages/evals"`.

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

def annotate(call: ToolCall, outcome: ActionOutcome) -> str | None:
    if outcome.refusal is not None and call.action == "run_launch" and pricing.stale:
        return "the pricing catalogue is stale, so every model is unpriced"
    return None

mount_fastmcp(
    server,
    eval_catalogue(my_actions),
    host_resolver=lambda: tenant_hosts[current_tenant()],   # in place of host=
    caller=lambda: Caller(scope_id=current_tenant(), identity=current_user()),
    around=around,
    annotate=annotate,
    call_timeout_s=lambda: settings.eval_call_timeout_s,    # read as each call starts
)
```

- **`host_resolver`** answers which `OpsHost` this call works in, per call, from the same request context as
  the caller. It may be sync or async, and is asked after the caller. Pass `host` for a server with one host,
  or `host_resolver`: exactly one of the two, or the mount is refused before any tool is added.
- **The host resolver can hand state to the hooks.** Answer a `HostResolution(host, context=...)` in place of the
  bare `OpsHost`, and its `context` reaches both `around` and `annotate` as `call.context`, beside the resolved host
  as `call.host`: the tenant you resolved, an audit record you started, for this call alone. No context variable to
  set and reset. A resolver answering a bare `OpsHost` gives `call.context` `None`, as does a mount with `host=`.

  ```python
  from threetears.evals.transports.fastmcp import HostResolution

  def host_resolver() -> HostResolution:
      tenant = current_tenant()
      return HostResolution(tenant_hosts[tenant], context=AuditRecord(tenant=tenant))

  async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
      outcome = yield
      audit.write(call.context, call.action, outcome)  # call.context is None if the host resolver refused
  ```
- **A resolver that refuses answers the call as a refusal.** A caller or host resolver that raises an
  `EvalServiceError` (your own subclass among them, say a 503 for a host still booting, with a remedy telling the
  agent when to come back) is answered exactly as an action's refusal is: the `refused:` text with its `remedy:`
  line, and `{"refusal": {"kind": ..., "status": ...}}` with the error flag. No action runs. A host resolver's
  refusal still enters the `around` hook, because the caller is known: the hook runs to its yield and is handed the
  refusal as its outcome, so your audit log records "no host could serve this call" from the same hook as every
  other call. Its `ToolCall` has `host` and `context` both `None`, and `call.refused` is the very `EvalServiceError`
  the resolver raised, so both `around` and `annotate` can read your subclass's own attributes. The outcome carries
  only the kind and status, which two cases of one error class share; `call.refused` tells them apart with no
  context variable set before the raise:

  ```python
  async def around(call: ToolCall) -> AsyncGenerator[None, ActionOutcome]:
      outcome = yield
      if isinstance(call.refused, HostUnavailable):
          log.log(logging.INFO if call.refused.booting else logging.ERROR, "no host: %s", call.refused)
  ```

  `call.refused` is `None` in every other case: a call that ran, a mount with `host=`, and a resolver that answered.
  A caller resolver's refusal enters no hook, since there is no `ToolCall` without a caller.
- **A resolver that raises anything else fails the call before anything runs.** No action runs and the hook is
  not entered. The client sees the transport's error result, not a refusal. A resolver that answers with
  something other than a `Caller`, or an `OpsHost` or a `HostResolution` holding one, fails the same way, with a `TypeError` naming what it
  answered.
- **`around`** wraps every call to every tool, for tracing, metrics or an audit log. It is an async generator
  function. It receives a `ToolCall` (the tool, the `action` argument as given, the caller, the resolved host, the
  timeout the call read, the host resolver's `context`, and the error it raised if it refused), yields once,
  and receives the `ActionOutcome` at the yield. Anything it opens before the yield is current while the
  action runs, so a span it starts is the parent of the action's own; a `ContextVar` it sets, or a `with` it
  enters, before the yield is what the action sees, and its code after the yield undoes it in the call's own
  task. A call that raises a defect, rather
  than refusing, is thrown in at the yield, so a `with` or `finally` closes. The defect still propagates,
  whatever the hook does.
- **`annotate`** adds your own word to what a call returns. It receives the `ToolCall` and the `ActionOutcome`
  once the outcome exists, sync or async, and a non-empty string it answers is appended as a text block of its
  own, after the outcome's. The outcome's text, structured content and error flag are untouched, so a client
  branching on the refusal's kind is never misled by your note. It sees every outcome reached once the caller is
  known (a host resolver's refusal among them), never a call refused while the caller was being resolved, nor one
  that raised or timed out. One that raises, answers something other than a string, or overruns
  `HOOK_TAIL_TIMEOUT_S` is logged, and the call returns as if there were no note.
- **The one call timeout holds the whole call**, the caller and host resolvers and the hook with it. A call it
  stops has `CallTimedOut` thrown in at the hook's yield, so the hook's `except CallTimedOut` and `finally` run and
  it can tell a timeout from a cancel: a call cancelled from outside (the client gave up, the process is stopping)
  closes the hook with nothing thrown. `call.timeout_s` is the timeout this call read. The client gets the timeout's
  error result, naming the timeout that call read and saying the action may have completed, or begun a
  write, before it was stopped: read before calling again (`job_poll` for a job it may have started, or the list
  for what it was writing), or a retried launch is paid for twice. `call_timeout_s` takes a number, checked positive at mount,
  or a function, read once as each call starts: one answering anything but a positive number fails that call
  with a `ValueError` naming the value, before anything runs.

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
- `judge_cases_list` (read) lists the template's stored cases (`template_id`), in stored order: each case's id, its
  criterion, its author and expected verdict when it was written by hand, how many labels it carries, and whether it is
  live (not retired). It is how an operator finds the ids `case_set_mint` takes to list all of them. A case this build
  cannot read is listed apart. A case's name is stored with it but not listed here; `cases_rename` changes it.
- Then `run_launch` the judge template, one arm per judge: the arm's model, and its `config_ids` and `temperature`
  as overlays. The trials call only the judge, and their spend is recorded under the `judge` role.
  See [judges and calibration](judges-and-calibration.md#evaluating-a-judge-as-a-subject).
- `judge_profiles_record` (write) stores, from the campaign's finished runs (`judge_run_ids`), one profile per judge
  and criterion: both agreements, parse validity, the cases and runs it was measured on, and when. Any campaign that
  judge scores then reads its evidence tier from the profile where its own evidence decides none, and says so.
  `judge_profiles_list` (read) lists them. See
  [storing a judge's profile](judges-and-calibration.md#storing-a-judges-profile-for-other-campaigns).
