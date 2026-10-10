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
analysis generations and judge repeats (`--purpose variation|proposer|analysis|judge`) — narrowed by its flags.

## Exit codes

| Code | Name | Meaning |
|---|---|---|
| `0` | `EXIT_OK` | done |
| `1` | `EXIT_RUN_DID_NOT_COMPLETE` | a launched run did not complete |
| `2` | `EXIT_REFUSED` | refused: a host that cannot be loaded, a template that is not there, a launch the engine refuses, a malformed command line |
| `3` | `EXIT_FAILED` | failed on an error nothing anticipated — a host factory, launcher or host command raising — with its traceback on stderr |

The names are in `threetears.evals.quick`.

## Mounting it under your own CLI

Mount the same commands under your own CLI with `run_cli(argv, host_factory=build_host, prog="myapp
evals")`; your users then never name the host.
