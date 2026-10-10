# The command line

**For** anyone who already has a host and wants to launch runs and read reports from a terminal, or mount the
same commands under their app's own CLI. **Answers:** what each command does, what it maps to, and its exit
codes. With no host yet, start with the [tutorial](tutorial.md), which needs
none, then [Adopting the engine](adopting-a-host.md).

## Commands

**The command line** works in a host you name as `module:factory` — a zero-argument callable
returning an `EvalHost`, or a `LaunchHost` for `run`:

```
python -m threetears.evals run    --host myapp.evals:build_host --scope dev --template T --subject S [--model M ...]
                                  [--k N] [--max-cost-usd DOLLARS] [--judge-model MODEL] [--simulator-model MODEL]
                                  [--n-variations N] [--variation-model MODEL] [--apparatus-settings JSON]
python -m threetears.evals ls     --host myapp.evals:build_host --scope dev
python -m threetears.evals report CAMPAIGN --host myapp.evals:build_host --scope dev [--format markdown|html|json] [--out PATH]
python -m threetears.evals bundle CAMPAIGN --host myapp.evals:build_host --scope dev
python -m threetears.evals spend  --host myapp.evals:build_host --scope dev [--purpose P] [--launch-group ID] [--template ID]
python -m threetears.evals frontier --host myapp.evals:build_host --scope dev [--bar B] [--subject S] [--json]
python -m threetears.evals gate   CAMPAIGN --host myapp.evals:build_host --scope dev [--fail-on OUTCOMES] [--reading MEASURE ...]
python -m threetears.evals judge-temperature RUN --host myapp.evals:build_host --scope dev (--max-cost-usd DOLLARS | --no-cap)
                                  [--repeats N] [--all] [--result ID ...] [--estimate] [--json]
```

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

Each flag passes to one `start_run` argument: `--model` to `models`, `--k` to `k_runs`, `--template` to
`template_id`, `--subject` to `subject_id`, `--scope` to `scope_id`, and the rest to the argument of the same name.

### `report` and `bundle`

`report` prints the campaign's report (see [Reading reports](reading-reports.md#the-campaigns-report)) —
its analysis, or, when it has none, a code-only report of its evidence. `bundle` prints, as JSON, the analysis
bundle a generation would read inside a wrapper (`BundleInspection`) that adds its campaign, scope and
fingerprint; the bundle itself is the wrapper's `bundle` field. Neither command calls a model.

### `spend`

`spend` prints what the engine spent outside any run in the scope — case generations, rubric proposals,
analysis generations, judge repeats and judge temperature comparisons
(`--purpose variation|proposer|analysis|judge|second_judge`) — narrowed by its flags.

### `frontier`

`frontier` prints each subject's variants ranked on pass^k, cost and latency, and, with `--bar`, the cheapest
that clears it ([The frontier](choosing-a-design.md#the-frontier-passk-against-cost)). `--subject` narrows it to
one subject, and `--json` prints the `FrontierResult` the `scope_frontier` action returns.

### `judge-temperature`

`judge-temperature` measures what temperature does to a run's judge: it re-judges the run's borderline cases
`--repeats` times (default 5) at the pinned judge temperature and as many times at the provider's default, and
prints each dimension's score variance, unstable cases and self-agreement at the two side by side
([Step 10](judges-and-calibration.md#step-10-measure-what-temperature-does-to-the-judge)). It is the one command
here besides `run` that spends: name the out-of-run cap with `--max-cost-usd`, or waive it with `--no-cap`. Every
call is priced against it before the first is sent and ledgered under purpose `judge`. `--estimate` prices it and
calls nothing. `--all` re-judges every scored dimension, `--result` narrows it to those results, and `--json` prints
the `JudgeTemperatureComparison` (or estimate). Keep that output: nothing is written to the results. A comparison
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
(no control, no bar, no guardrail) is refused. In code, `Comparison.gate()` and `gate_verdicts()` do the same.

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
