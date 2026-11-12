# Testing 3tears

The suites, why several of them sit outside `./scripts/test.sh`, how to make the integration
suite run everything it can, and how test fakes declare what they stand in for. The rules are in
`CLAUDE.md`; this is the detail behind them.

## Run only what you're working on

```bash
./scripts/test.sh scrape          # one package: packages/scrape/tests/
./scripts/test.sh agent/memory    # one agent-family package: packages/agent/memory/tests/
./scripts/test.sh agent           # the whole agent family: packages/agent/
./scripts/test.sh core -k lease   # one package, narrowed; pytest's options pass through
./scripts/test.sh evals packages/evals/tests/test_fastmcp_transport.py   # one file, not all of evals
./scripts/test.sh evals packages/evals/tests/test_fastmcp_transport.py::test_name   # one test
./scripts/test.sh --not-area evals   # everything but evals, on the pool
./scripts/test.sh                 # everything
```

`./scripts/test.sh <package>` runs that package's `tests/` and nothing else, so evals stays out
unless you name it (`./scripts/test.sh evals`). Name test files or `file.py::name` node ids after
the package and only those run, on the pool as locally, so a one-file run takes seconds rather
than the whole package's minutes. A bare `./scripts/test.sh` runs every package and
the repo-level `tests/`, and evals is most of that test time (roughly four fifths of a full run;
see "Where the unit suite's time goes" below).

To run everything but evals, select by testament area: `./scripts/test.sh --not-area evals` (or
`testament run --not-area evals`). Areas are named slices of the one unit suite, declared in
`.testament.toml`'s `[areas]`: `evals`, `enforcement` (`packages/enforcement/` and
`tests/enforcement/`), `core`, `agent` and `platform` (every other package and the repo-level
`tests/test_*.py`). `--area NAME` runs one; both repeat. `test.sh` passes them to testament ahead of
the pytest selection, and they combine with a package or test paths. Only the pool knows the
areas, so a run that would go local (`TESTAMENT=off`, an environment gate, a flag that keeps it
local, CI, no testament) refuses them and exits 2; a `-m` goes to the pool with them. On the pool an area costs what its folder does: `--area core` collected
the same 5,223 tests as `packages/core/tests/` in about the same time (31 s against 30 s), so a
package run stays a path, which also works locally.

Run everything at the gates: before you open a PR (`./scripts/check-all.sh`, which runs
`./scripts/test.sh` and the suites below that it does not), and on the integration branch before it
merges into `develop`. CI runs everything on every PR as well.

`check-all.sh` also runs the two suites CI's `check` job runs beside the unit suite:
`./scripts/test-mcp-v1.sh` (the MCP packages on mcp 1.x, overlaid on the locked 2.x) and
`./scripts/test-otel-floor.sh` (observe's OpenTelemetry tests at its declared floors, in an isolated
environment). Both set their own dependency versions, so they run locally, never on the pool, and
`test.sh` alone does not cover them: PR #1087 went red in CI on `test-mcp-v1.sh` after every local
gate was green, because none of them ran it.

## The sidecar suite

`./scripts/test-sidecar.sh` runs the nodriver sidecar's own tests. nodriver is AGPL-3.0 and never
enters the workspace venv, so `test.sh` carries `--ignore` for the sidecar and cannot run these.
Separate but not optional: `check-all.sh` runs it. Until it existed, a ruff autofix wrote a syntax
error into `hitl.py` that passed lint, mypy, and the entire workspace suite.

## The unit suite runs on four workers

`./scripts/test.sh` with no package, and CI, run the unit suite under pytest-xdist as
`-n 4 --dist loadgroup`. The root `conftest.py` puts every test in its package's xdist group, so
each package runs in one worker in its usual order, and packages run side by side. A
single-package run (`./scripts/test.sh core`) stays serial. On the local route (`TESTAMENT=off`, or
no testament) pass `-n 0` to run the whole suite serially, for a debugger or to rule out the
parallel run; on the pool route `test.sh` refuses `-n` on anything wider than named test files.

On GitHub's runners CI runs the same thing as parallel shards, through
`./scripts/ci-unit-tests.sh --shard <name>`: `evals-1` to `evals-8`, `core` (with datasources),
`tests`, and `rest` (the agent family and every other package). A worker's memory only grows over
its session, and one session over the whole suite grew four workers past the runner's 16 GB, so
the job was killed with no summary. Each session starts fresh workers, and together they collect
exactly the tests one session would.

An evals shard runs on one worker (evals is one xdist group), so its wall clock is the sum of its
tests. pytest-split cuts evals into eight runs of about equal recorded duration, from
`.test_durations` at the repo root. Refresh that file from the testament pool's history with
`./scripts/update-test-durations.sh` when the `unit (evals-N)` jobs drift apart; stale entries
only cost balance, never a test.

Where `testament` is installed (outside CI), `test.sh` hands the run to its
runners instead (testament 0.45.1 or later: an older client refuses the manifest's
`fallback` key and runs nothing; upgrade with
`uv tool install --force --python 3.14 git+ssh://git@github.com/brookstalley/testament@v<version>`,
where `testament status --json` names the pool's `coordinator_version`), and none of the above applies. Testament distributes the tests itself, and a run
there leaves out the integration and `loop_timing` tests (`.testament.toml` selects
`-m "not integration and not loop_timing"`; CI runs `loop_timing` serially on its own runner). A
`-m` given to `test.sh` narrows that selection: testament sends `-m "(<manifest's>) and
(<yours>)"`. The manifest is `fallback = "pool-only"` (testament 0.45.1 or later): when the pool
can't take a run, it exits 2 naming the key instead of running the suite on your machine.

`-x`, `--maxfail`, `--lf`, `--ff`, `--sw` and `-n` need one local pytest process, and `-s`,
`--capture=no`, `--pdb` and `--trace` need this terminal (testament runs those here whatever the
manifest says), so the pool can't take them either. With only named test files (`.py` files or
node ids) the run goes local with them; any wider run (a package, a directory) is refused, exit 2,
with how to run it. A run gated by an environment variable (`RECORD`, `THREETEARS_LIVE_*`) stays
local, since its tests would skip on runners without it. Any other run goes local only with
`TESTAMENT=off` and a stated reason, `TESTAMENT_OFF_REASON="..."`, which `test.sh` prints; without
the reason it refuses and exits 2.

### When the pool is down

The rule while the pool can't take runs: **CI's GitHub shards gate the unit suite, and a PR's
author runs everything else locally.** Nobody runs the whole unit suite on a laptop.

**How it shows.** `./scripts/test.sh` exits 2 and ends with "testament gave no verdict". From
testament 0.46, testament itself exits 75 when the pool could not serve the run, and 2 for every
other stop. `test.sh` turns its 75 into 2 and first prints "testament exit 75", so a 75 is always the
pool. Testament names the cause:
- `coordinator_unreachable`, `no_runner`, `connection_lost`, `batches_lost` or `env_build_failed`:
  the pool;
- `machine_not_set_up`: this machine has no key; `unauthorized` or `revoked`: its key was revoked;
- `bad_manifest` with "a key it doesn't know may need a newer one": this machine's testament is
  older than 0.45.1.

`testament run --probe` and `testament status` say whether the pool answers.

**Iterating.** Name the files and run them here:
`TESTAMENT=off TESTAMENT_OFF_REASON="pool down" ./scripts/test.sh core packages/core/tests/test_x.py`.
On a machine with room for it (not the laptop), a package works the same way.

**Before a PR.** Run `./scripts/check-all.sh`. With the pool down it still runs lint, typecheck,
the mcp 1.x, OpenTelemetry-floor and sidecar suites, then fails at the end with "Unit suite NOT
run". That line is expected only when the testament error above it is one of the pool or machine
causes listed here; any other exit 2 (a `TESTAMENT=off` with no reason, `path_not_found`, a crash
in the code under test) is yours to fix. Every other check above it must be green. Run
`./scripts/test-integration.sh -rs` as always. Push: CI's `route` job finds the pool down and runs
the unit suite on the GitHub shards, and their verdict is the unit gate. To run the whole unit
suite yourself anyway, use a machine with more memory than a 16 GB CI runner, run it the way CI
does (`TESTAMENT_OFF_REASON="pool down" ./scripts/ci-unit-tests.sh`, in sessions), and expect
hours. Never `TESTAMENT=off ./scripts/check-all.sh`: its one session over every package is the run
that OOM-killed four workers on 16 GB.

**CI.** CI takes the shards on its own when its probe fails, when the client can't be installed,
or when the run is lost. One case stays red instead: the pool answers the probe and then fails
every run for its own reasons, for example `env_build_failed` on every branch, or a manifest
refused after the probe. Then set the repo variable `TESTAMENT_CI` to `off` (*Settings → Secrets
and variables → Actions → Variables*) and re-run the checks. Delete it when the pool is back.

**Releasing.** Unchanged: `release.yml` runs the GitHub shards and never uses the pool.

**On another machine.**
- With no testament installed, `test.sh` runs locally, as it always has, and so does
  `check-all.sh`'s unit step: one session over every package, the run that OOM-killed four workers
  on 16 GB. On such a machine run the unit suite as CI does instead,
  `TESTAMENT_OFF_REASON="no pool here" ./scripts/ci-unit-tests.sh`, and the rest of `check-all.sh`'s
  steps by hand (`lint.sh`, `typecheck.sh`, `test-mcp-v1.sh -q`, `test-otel-floor.sh -q`,
  `test-sidecar.sh`).
- With testament installed but not enrolled, holding a revoked key, or older than 0.45.1, every
  run exits 2. Run named files with `TESTAMENT=off TESTAMENT_OFF_REASON="..."`, or upgrade (above)
  and enrol (`testament login --console <the pool's console address>`).

**Agents** stop and tell the person. They never rerun a package or the whole suite with
`TESTAMENT=off`, never run `ci-unit-tests.sh` or raw `pytest`, and never edit `.testament.toml`.
Named test files under `TESTAMENT=off` are fine.

### CI on the testament pool

CI's unit + enforcement suite takes one of two routes, and `check` is green only when the route
that ran to a verdict passed:

- **The testament pool**, one job, `unit (testament pool)`, running
  `./scripts/ci-unit-pool.sh run`. GitHub's runner only waits.
- **The GitHub shards** (above), otherwise.

The `route` job chooses, with `./scripts/ci-unit-pool.sh route`. It takes the shards, without
asking the pool, when the repo variable `TESTAMENT_CI` is `off`; when no `TESTAMENT_KEY` secret
reached the run (a fork's pull request gets none); when the event is neither a push nor a pull
request from a branch of this repo (ci.yml withholds the key then too, whatever GitHub's fork
policy does); or when no `TESTAMENT_COORDINATOR` variable names the pool. Otherwise it installs
the client and runs `testament run --probe`, which asks the coordinator whether a run is possible:
the project allowed, a runner connected that can take it, disk to record it. A probe that says no,
one with no answer within 60 seconds (`TESTAMENT_PROBE_TIMEOUT`), or a client it cannot get, routes
to the shards too.

**When the pool is down or lost, CI falls back to the shards**, never to testament's own local
fallback (`--no-fallback`), which would run the whole suite in one session on one runner: that
OOMs its xdist workers (`ci-unit-tests.sh`'s header) and takes about two hours.

| The probe | The pool's run | What runs | `check` |
|---|---|---|---|
| not asked: `TESTAMENT_CI=off`, no key, a fork's PR, no coordinator | | the shards | the shards' verdict |
| fails, gives no answer within 60 s, or the client can't be had | | the shards | the shards' verdict |
| passes | passed | the pool | green |
| passes | failed: a test failed or errored, nothing collected, or any other refusal (`forbidden`, `upgrade_required`, ...) | the pool only | red |
| passes | unreachable: `coordinator_unreachable`, `connection_lost`, `batches_lost` or `no_runner`, and no test failed; or the client could not be had, so nothing ran | the pool, then the shards | the shards' verdict |
| passes | the job broke, timed out (75 min) or was cancelled | the pool only | red |
| `route` itself broke | | nothing | red |

A retryable ending (`coordinator_unreachable`, `connection_lost`, `batches_lost`) runs once more
before it counts. A fallback found during the run starts the shards only once `unit-pool` ends, so
that run takes the pool's time plus the shards'. A fallback is never silent: `route` or
`unit (testament pool)` says why in an annotation and its job summary, and `check` names the route
that ran and why in both. `tests/test_ci_unit_pool_script.py` pins the script's choices and
`tests/enforcement/test_ci_unit_shards.py` runs `check`'s verdict against each row.

**To take the shards on purpose**, set the repo's Actions variable `TESTAMENT_CI` to `off`
(*Settings → Secrets and variables → Actions → Variables*); delete it to go back. The secret stays.

**What the pool runs.** `testament run --fresh --no-cache --no-fallback -- -m "not loop_timing"`.
`.testament.toml` selects `packages/` and `tests/` with `-m "not integration and not loop_timing"`:
the tests `ci-unit-tests.sh` partitions, less the `loop_timing` ones (below), which this `-m`
names again. `--fresh` runs every batch in a container started cold from the project's image (one
per runner, removed when the run ends), so nothing an earlier run left in a warm container reaches
the required check; it costs a cold start and a full `[setup]` (`uv sync`) on each runner it
reaches. `--fresh` does not turn off the cache, so `--no-cache` makes every test run rather than
serving an earlier passing result on the same tree. Every difference from the shard route:

- **Batches, not xdist workers.** The pool runs each batch with `-n 0`, its runners being the
  parallelism; the shards run `-n 4 --dist loadgroup` inside each session. An `xdist_group` still
  keeps its tests together on the pool.
- **`loop_timing`** stays off the pool in CI. Its tests bound the event loop's lateness by wall
  clock, and in a pool batch beside other work they fail on the runner's load, not the code
  (#1140). The pool run passes `-m "not loop_timing"`, which testament narrows the manifest's
  expression with rather than replacing it, and the same `unit (testament pool)` job then runs them
  serially on its own runner (`./scripts/ci-unit-tests.sh --loop-timing`), as the `rest` shard
  does. When the pool is lost mid-run it skips that step, since the shards run them. The manifest
  leaves them out of every pool run too, so a local `test.sh` pool run never meets them;
  `./scripts/ci-unit-tests.sh --loop-timing` runs them serially here.
- **The environment.** The pool's image (`docker/testament.Dockerfile`: `python:3.14` slim, a
  pinned uv) with `uv sync --frozen --extra vision`; the shards use `ubuntu-latest`, the latest uv
  and `uv sync --extra vision` (not frozen).
- **The split.** The pool batches by its own planner; the shards split evals by
  `.test_durations` and the rest by package.
- **Limits.** A pool batch stops at `batch_timeout_s` (30 min); the pool job at 75 min. A shard has
  no per-batch limit.
- **Nothing collected** is a failure on the pool (testament's exit 5); a shard session tolerates it
  (a session of only integration tests).
- **History.** The pool records every result under `TESTAMENT_BRANCH`, so `testament` can report
  on CI runs; the shards record nothing outside the job log.
- **The browser tests** (the evals render checks, marker `browser`) run on both routes. The pool's
  image installs headless Chromium with the locked playwright, plus DejaVu, so the package's pages
  (a local font stack) draw in the dev box's faces rather than a CJK fallback. The evals shards run
  `playwright install --with-deps --only-shell chromium`, cached by playwright's version. Both run
  `./scripts/check-fallback-fonts.sh`, which fails unless the generic families resolve to DejaVu, and
  both set `THREETEARS_REQUIRE_CHROMIUM=1`, so a missing browser fails them instead of skipping them.

**The client.** testament is private and not on PyPI. The console serves its current release's
wheel at `/api/v1/cli/wheel` to any client key, so no GitHub token for the private repo is needed,
and the client is always the pool's own version. The response names the wheel's sha256 in a
header, and the script refuses a file that does not match it or is not a testament wheel: an
integrity check of the download, not provenance, since one HTTPS response carries both. The trust
is HTTPS to the console, so the script refuses a `TESTAMENT_CONSOLE` that is not `https://` before
sending the key, and sends the key to curl on stdin, never in its arguments. The console defaults
to `https://testament.rodeo.illuminati.org/`; an Actions variable `TESTAMENT_CONSOLE` overrides it.

**The key** is a client key the owner mints in the testament console: People, their own page,
**+ New client key**, named `ci-3tears`. Store it as the repo's Actions secret `TESTAMENT_KEY`
(*Settings → Secrets and variables → Actions*), and revoke it there in the console when CI no
longer needs it. The pool's address is the Actions variable `TESTAMENT_COORDINATOR`. Integration,
lint, typecheck, the mcp 1.x and OpenTelemetry-floor suites and the attribution check stay on
GitHub either way.

### Where the unit suite's time goes

testament records every test it runs, so it can say which area is slow, large or sparse. The
unit suite is one testament suite (`default`), divided into the areas above:

```bash
testament time --by area --project 3tears --since 7d                 # share of recorded test time per area
testament trend --by area --project 3tears                           # each area day by day: size, cost of one full run
testament time --by folder --area evals --depth 4 --project 3tears   # inside one area, file by file
testament tests slowest --project 3tears --since 7d                  # the slowest tests, mean, median and p90
testament run --json -- packages/scrape/tests/                       # report.items[]: each test's outcome and duration
testament areas check --strict                                       # tests in no area, in two, and globs matching none
```

Without `--project` they cover every project on the pool. A run's summary breaks its time down by
area, and history is re-sliced by the map in your checkout, so a changed `[areas]` needs no rerun.
`time` shares are of recorded time, so an area that runs more often weighs more. `trend`'s "cost s"
counts each test once (its inventory times its typical duration), so it is the figure for one full
run, and the one behind "evals is about four fifths": 13,904 s of 16,887 s (82%) on 2026-11-10.

`platform` names its packages one by one rather than ending in a catch-all `packages/**`: a
catch-all overlaps every area above it, and `areas check --strict` reports each overlap. A new
package therefore needs its glob in `[areas]`; `tests/enforcement/test_testament_areas_cover_the_tree.py`
fails until it has one. `areas check` reads the tests recorded in the last 14 days, branches
included, so a package removed recently can show as unassigned until it ages out (`--since 3d`).

**Totals recorded before 2026-11-09 can be wrong under freezegun.** A test that froze time
recorded the frozen clock as its duration (one test showed 1.78e9 s, and `packages/core` read as
88.9% of the suite). testament now times with a monotonic clock (brookstalley/testament#553); if a
folder reads as billions of seconds, narrow the window with `--since` or read the median.

Two things break under a parallel run that a serial one hides. A fixed `asyncio.sleep` that
waits for background work loses the race on a loaded machine: wait for the event itself. And a
`conftest.py` fixture named like the root one replaces it rather than adding to it, which drops
the root's per-test reset of process-wide state.

## The integration suite

`./scripts/test-integration.sh` runs the integration tests. `test.sh` excludes them with
`-m "not integration"`, and `check-all.sh` does not run them either: they spin real NATS and
Postgres containers and need Docker, so folding them into the default gate would break it
wherever Docker is absent. **CI runs only part of them:** its `integration` job covers core,
nats, epoch, scheduled-jobs and agent/acl against Postgres and NATS services, so the rest of the
packages are executed by nothing but a developer, locally.

The evals render tests are not here. They need headless Chromium, not Docker, so they carry the
`browser` marker and run in the unit suite: on the testament pool and on the GitHub evals shards of
`ci.yml` and `release.yml` (above).
A local run without Chromium skips them; `-m "not browser"` leaves them out.

**Run them before any PR.** Cross-pod behaviour lives entirely there, and a green
`check-all.sh` says nothing about it. `project-state.yaml` lists this as the third declared test
command, so recorded evidence that omits it covers two suites out of three.

### A skip is not a pass

The suite exits 0 with tests skipped, so a skipped test reads exactly like a passing one in the
summary line. Run it as `./scripts/test-integration.sh -rs` and account for every skip.

This is not hypothetical: `test_a_real_display_is_driven_through_the_pipe` was unpassable on
every machine for three weeks after the 2026-08-18 change that defaulted the sidecar's
`BIND_HOST` to loopback (correct for the shipping Kubernetes shape, wrong for a testcontainer with
its own network namespace). Nobody saw it, because the test skips when the sidecar image is absent
and CI never runs the suite at all -- the skip that hid the breakage was also the reason nobody
noticed. It surfaced only when a release stopped to ask why 35 tests were skipping.

### Build the sidecar image first

Or its integration tests skip:

```bash
docker buildx bake --file docker-bake.hcl nodriver-sidecar \
  --set nodriver-sidecar.platform=linux/$(docker version --format '{{.Server.Arch}}') --load
```

The bake target is multi-platform and the local `docker` driver refuses that ("Multi-platform
build is not supported for the docker driver"), hence `--set ... platform` and `--load`. A bare
`docker buildx bake nodriver-sidecar` exits non-zero having built nothing -- and piping it through
`tee` masks that exit code, which is how it looked like it had worked.

**Build for the Docker host's own architecture**, which is what the `docker version`
substitution picks (`arm64` on Apple silicon, `amd64` on an Intel box). A `linux/amd64` image on
an arm64 host builds and loads, then runs under emulation and never becomes healthy, so its
integration tests fail rather than skip. A native build passes.

### Legitimate skips on a dev box

They need credentials or tools this repo does not ship: the Redshift live tests
(`OTS_REDSHIFT_PASSWORD`), the backup suites (`pg_dump`/`pg_restore`/`psql` on PATH), the live
Claude CLI batch (below), and one deliberate manual microbenchmark. Anything else is a test turned off by accident. When reporting
results, state the pass count AND the remaining skips with their reasons -- "integration green"
on its own is not a report.

## The live Claude CLI batch: structured output and the session pool

`./scripts/test-live-claude-cli.sh` makes real calls through the real Claude CLI on a
subscription (`packages/models/tests/live/`): a batch of structured calls that must all answer in
their schema, and the session pool's live proofs -- a reused pooled CLI shows the next call
nothing of the last, and one pooled CLI switches between system prompts, each checked to have run
on the pool. A change to `providers/claude_cli.py` or `claude_cli_pool.py` is covered by it.
`test.sh` collects these tests and skips them:
it needs a subscription token (`CLAUDE_CODE_OAUTH_TOKEN`, from `claude setup-token`) and spends
real usage, so it is opt-in (`THREETEARS_LIVE_CLAUDE_CLI=1`, which the script sets). Every other
test of the subscription route fakes the CLI's messages and can only replay shapes somebody
already saw; this is the one that shows what a real model does. The release runs it before
tagging anything that touches `packages/models`, and pastes the line the script records into
the release PR (`docs/releasing.md`, "Cutting a release").

## The hot-path oracle's goldens

`packages/evals/tests/test_hot_path_oracle.py` reads five campaigns the way a host reads them and
compares every bundle, fingerprint and report body with its golden in
`packages/evals/tests/fixtures/hot_path_oracle/<campaign>.json`. Its module docstring says what each
golden holds.

**Goldens move only when a change to a page is intended** (owner ruling, 2026-11-06):

1. Regenerate with `HOT_PATH_ORACLE_REGEN=1`, locally and on one process:
   `TESTAMENT=off TESTAMENT_OFF_REASON="regenerating the hot-path goldens" HOT_PATH_ORACLE_REGEN=1
   ./scripts/test.sh evals -k hot_path_oracle`. Each
   regenerating test fails, naming the entries it moved with their sizes before and after.
2. Review the golden diff line by line (`git diff packages/evals/tests/fixtures/hot_path_oracle`).
3. State the reason in the commit and in the chunk report.
4. Never regenerate to make a red go away. A red means a change moved what a host reads.

A campaign with a platform override (its pool slots and the dev container's differ) has its pool slots
written from the pool run's own red: copy each slot's `OBSERVED` record into the golden, and only when
its byte count matches the local regeneration of the same entry.

A release bump moves no golden. The version of 3tears-evals that read a page is the build, not
the page's content. The oracle puts one fixed version in its place, in the bundle's
`package_version` and in the code-only page's methods table, before it compares.

## Test fakes

A test fake is any class named `Fake<Name>` or `_Fake<Name>` under a `tests/` directory. Every one
declares what production protocol it stands in for. Three routes, in order of preference:

1. **Subclass it.** `class _FakeKv(KvBucketLike):`. The walker accepts any non-`object` base and
   checks nothing further, on the theory that a type checker covers it. **In this repo it does
   not:** mypy runs over `packages/*/src` only, so a subclassed fake's surface is unverified by
   anything. Prefer it anyway for a real Protocol, because the base documents intent and an IDE
   follows it. Reach for route 2 when you want the surface actually compared.
2. **`# parity-with: <fully.qualified.name>`** on the line above the class. The walker imports
   the target and compares method surfaces. This is the only route that verifies anything.
3. **`# parity-exempt: <rationale>`** on the line above the class, for a hand-rolled subset stub
   with no single production protocol to name. The rationale must be at least 30 characters and
   must not be a blanket phrase like "tests need this" or "temporary". **Keep it on one line,
   however long.** The walker reads the first non-blank line above the class and stops, so a
   wrapped rationale exempts nothing.

Workspace tests centralise their asyncpg and workspace-entity shells under
`packages/agent/workspace/tests/helpers/`, so per-test inline fakes need only a one-line
subclass declaration.

**Exempt in place, not in `tests/enforcement/_fake_parity_exemptions.txt`.** That file still
parses and is deliberately empty. Its entries are keyed `path:LINE:symbol`, so one added import
shifts every fake below it and the gate fails with `no_declaration` for a fake nobody touched. A
marker on the class moves with the class.

`tests/enforcement/test_fake_protocol_parity.py` enforces this. It is a thin shell over the
canonical walker in `packages/enforcement/src/threetears/enforcement/fake_parity/`.
`FAKE_PARITY_ENFORCEMENT_MODE` defaults to `strict`. It catches the drift class where production
protocols evolve while test fakes rot silently, until some downstream test happens to call the
missing method.
