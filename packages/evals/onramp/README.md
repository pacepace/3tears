# The on-ramp agent test

An eval of the package's docs, built on the package. It asks: can a coding agent that has never seen this package add
a working eval to an unfamiliar app, from the docs and the error messages alone?

- **Cases:** three small apps (`apps/`), each with a planted defect, and a fixed twin of each (`twins/`).
- **Arms:** docs builds. Each arm is a wheel of `3tears-evals` and the `llms.txt` inside it. The control arm is the
  wheel built from `ONRAMP_CONTROL_REF` (default `f9ab6f74d`).
- **Runner:** a model acting as the agent. It reads, writes and runs `python` in a Docker container with no network,
  as a non-root user, holding only a copy of the app and the installed wheel (`isolation.py`).
- **Checks** (`checks.py`), each re-run by the harness after the agent stops, with no model call:

| Check | Passes when |
|---|---|
| `eval_runs` | the agent's eval file, run by the eval door on a fresh copy of the app, exits 0 or 4, stores a finished run, and scores at least one result |
| `measures_the_app` | the eval reads better on the fixed twin than on the app, and a runner that does nothing fails at least half its cases |
| `read_right` | the agent's report gives the run's first line verbatim, the named measure's value within 0.005, and whether the run finished |

Nothing under `src/` imports this directory, and the wheel does not ship it.

## Running it

All commands run from the repo root. Every form but `--dry-run` and `--scripted ... --host` needs a Docker daemon:
the agent's commands and every re-grade run only in trial containers, never on this machine.

| Command | What it does | Cost |
|---|---|---|
| `./scripts/onramp-agent.sh --dry-run` | prints the arms and the quote; builds nothing | free |
| `./scripts/onramp-agent.sh --scripted succeeds --k 1` | fixed transcripts that should pass every check on every app | free |
| `./scripts/onramp-agent.sh --scripted stuck --k 1` | a fixed transcript that gets stuck: every check fails | free |
| `... --scripted succeeds --host` | the same, with commands run on this machine (scripted only) | free |
| `./scripts/onramp-agent.sh --regrade packages/evals/onramp/runs/<stamp>` | grades a stored run again with today's checks | free |
| `ONRAMP_LIVE=1 ./scripts/onramp-agent.sh --cap 4.50 --k 7 --arm docs@candidate=PATH.whl` | the live comparison: cheap tier, 3 apps, 2 arms | see below |

Add `--arm docs@candidate=PATH.whl` to any form to measure a second build against the control. Before a live or
replayed run starts, the harness runs `--scripted succeeds` against each arm and refuses the run unless every check
passes on every app.

The isolation canaries: `./scripts/test-integration.sh --here -k onramp -rs`.

## What a live run costs

The cheap tier is `deepseek/deepseek-v4-flash` through OpenRouter (`OPENROUTER_API_KEY`). The ceiling tier
(`--tier ceiling`) is Claude Haiku 5.5, direct to Anthropic (`ANTHROPIC_API_KEY`).

The live runs of 2026-11-15 and 2026-11-16, all at `--cap 4.50`, cheap tier:

| Run | Trials | Most billed |
|---|---|---|
| `20261115T224108Z` | 9 | $0.058 |
| `20261116T004246Z` | 42 | $0.614 |
| `20261116T020937Z` | 42 | $0.292 |
| `20261116T023325Z` | 42 | $0.359 |
| `20261116T030232Z` | 42 | $0.307 |

How the cap holds:

- A live run is refused unless `ONRAMP_LIVE=1` is set and `--cap` names a dollar amount. It is also refused when a
  key-fallback variable is set, since a fallback key could resend a call.
- Each trial gets the smaller of the tier's limit ($0.50 cheap, $1.50 ceiling) and its share of the cap.
- Before each model call, the harness prices the call's most expensive reply. It sends the call only if what the trial
  has been charged, plus 6 of those prices, stays within the trial's share.
- The harness counts every send the provider may have billed (a reply, a timeout, a 504 or 524) itself, and refuses a
  seventh. A call that failed or was cut off is charged at 6 times its price.

So a run is billed at most its cap. The one gap: a provider endpoint priced above the catalogue row, about $0.02 a run.

## Where results land

`runs/<UTC stamp>/` (git-ignored): `onramp.sqlite` (the store), `report/`, `trials.jsonl` (every trial's record and
checks), `census.txt`, and per trial `<arm>/<app>-<n>/` with `steps.jsonl` (every turn), `record.json` and the app as
the agent left it. Built wheels, images and environments are cached in `runs/wheels/`, `runs/images/`, `runs/venvs/`.

## Comparing arms

Compare each check's k of n between arms, app by app, and nothing else. Never compare `errors_hit`, and never the
count of trials the harness failed none of: both change with how each arm's agents behave. With two arms the census
tallies each arm apart (arms are never pooled), and with more than one app each line ends with its tally by app.

### What N trials on 3 apps can and cannot show

The unit that generalises is the app, not the trial. Trials on one app are repeats of that app: the same code, the
same planted defect, the same docs, one model. A comparison of 2 arms x 3 apps x k trials holds 3 units, however
large k is.

It can show:

- On each of these apps, with this model and this harness, whether one arm's agents passed a check more often than
  the other's, and whether the apps agree on the direction.
- Where agents get stuck, on which message, and what a run costs.
- That a docs change did not break anything the scripted pre-flight checks.

It cannot show:

- That a difference holds for an app nobody looked at, or for another model. Three apps are three draws: if the
  arms did not differ, one arm would still come out ahead on all three apps one time in eight, so no reading of 3
  apps reaches the 5% level.
- A significance level computed over trials. A test that counts 42 trials as 42 independent units (Fisher exact, a
  chi-square, a pooled proportion) has no term for the app, nor for two runs of the same design that disagree, and
  states a p-value the design cannot support. The package's own verdict on such a run ("too few cases to call a
  difference") is the right reading.
- A rate to rely on. The census's exact 95% range is per arm over trials, and trials on one app share the app, so
  even that range is too narrow: at 3 of 9 it already runs from 0.07 to 0.70.
- Whether a failure belongs to the docs or to the model.

### Development set and held-out set

A docs change written from the transcripts of these three apps was tuned on them. A comparison on the same apps is
development-set evidence: it says the change does what its authors meant on the apps they read, and nothing about any
other app. It does not admit, adopt or ship a docs change on its own.

What can support a claim beyond these apps:

- Apps nobody read transcripts of while writing the change, added to `apps/` with their twins before the run.
- The app as the unit of analysis: each app's check rate per arm, compared across apps.
- The plan (arms, apps, k, the check compared, the test, what counts as a result) committed to the repo or posted on
  the tracking issue before the run starts. A plan in a working file with no history cannot be verified as written
  first.

History: on issue #1177, a docs change to `llms.txt` was described as established by a Fisher exact test over 42
trials a side on the three apps here. The correction of 2026-11-16 withdrew that: the trials were 3 apps by 14 repeats,
the change had been written from those apps' transcripts, and the two halves of the confirmatory run disagreed (15
and 8 of 21). Read any comparison this harness produces under the same limits.
