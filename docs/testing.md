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
./scripts/test.sh                 # everything
```

`./scripts/test.sh <package>` runs that package's `tests/` and nothing else, so evals stays out
unless you name it (`./scripts/test.sh evals`). Name test files or `file.py::name` node ids after
the package and only those run, on the pool as locally, so a one-file run takes seconds rather
than the whole package's minutes. A bare `./scripts/test.sh` runs every package and
the repo-level `tests/`, and evals is most of that test time (roughly four fifths, by testament's per-test
medians from a full run on 2026-11-03; its duration totals are unreliable, see below).

Run everything at the gates: before you open a PR (`./scripts/test.sh`, or `./scripts/check-all.sh`,
which runs it), and on the integration branch before it merges into `develop`. CI runs everything on
every PR as well.

## The sidecar suite

`./scripts/test-sidecar.sh` runs the nodriver sidecar's own tests. nodriver is AGPL-3.0 and never
enters the workspace venv, so `test.sh` carries `--ignore` for the sidecar and cannot run these.
Separate but not optional: `check-all.sh` runs it. Until it existed, a ruff autofix wrote a syntax
error into `hitl.py` that passed lint, mypy, and the entire workspace suite.

## The unit suite runs on four workers

`./scripts/test.sh` with no package, and CI, run the unit suite under pytest-xdist as
`-n 4 --dist loadgroup`. The root `conftest.py` puts every test in its package's xdist group, so
each package runs in one worker in its usual order, and packages run side by side. A
single-package run (`./scripts/test.sh core`) stays serial. Pass `-n 0` to run the whole suite
serially, for a debugger or to rule out the parallel run.

CI runs the same thing as five sessions, through `./scripts/ci-unit-tests.sh`: evals, core with
datasources, the agent family, every other package, then `tests/`. A worker's memory only grows
over its session, and one session over the whole suite grew four workers past the runner's 16 GB,
so the job was killed with no summary. Each session starts fresh workers, and together they
collect exactly the tests one session would.

Where `testament` is installed (outside CI), `test.sh` hands the run to its
runners instead, and none of the above applies: testament distributes the tests itself, `-n`,
`-x`, `--maxfail` or `--lf` make it run locally, and a package run there skips its integration
tests (`.testament.toml` selects `-m "not integration"`). Two kinds of run never go to the pool:
one gated by an environment variable (`RECORD`, `THREETEARS_LIVE_*`, `THREETEARS_EVALS_PAID*`),
whose tests would skip on runners without it, and one that passes its own `-m`. Any other run
goes local only with `TESTAMENT=off` and a stated reason, `TESTAMENT_OFF_REASON="..."`, which
`test.sh` prints; without the reason it refuses and exits 2.

### Where the unit suite's time goes

testament records every test it runs, so it can say which area is slow, large or sparse:

```bash
testament time --by folder --depth 3 --project 3tears --since 7d     # share of test time per package
testament trend --project 3tears                                     # the suite day by day: size, cost, full runs
testament tests slowest --project 3tears --since 7d                  # the slowest tests, mean, median and p90
testament run --json -- packages/scrape/tests/                       # report.items[]: each test's outcome and duration
```

Without `--project` they cover every project on the pool. The unit suite is one testament suite
(`default`), so a per-area view is a folder view; testament's own per-area reporting is pending
(brookstalley/testament#549).

**Totals and means are wrong under freezegun.** A test that freezes time records the frozen clock
as its duration (one test showed 1.78e9 s), so a folder holding one reads as billions of seconds:
`packages/core` showed 88.9% of the suite for that reason alone. Read the median, or leave such a
folder out, until testament fixes it (brookstalley/testament#553).

Two things break under a parallel run that a serial one hides. A fixed `asyncio.sleep` that
waits for background work loses the race on a loaded machine: wait for the event itself. And a
`conftest.py` fixture named like the root one replaces it rather than adding to it, which drops
the root's per-test reset of process-wide state.

## The integration suite

`./scripts/test-integration.sh` runs the integration tests. `test.sh` excludes them with
`-m "not integration"`, and `check-all.sh` does not run them either: they spin real NATS and
Postgres containers and need Docker, so folding them into the default gate would break it
wherever Docker is absent. **CI cannot run them at all -- GitHub Actions has no Docker -- so
nothing but a developer, locally, ever executes them.**

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

## The evals paid lane: one analysis by a real model

`packages/evals/tests/test_paid_analysis_generation.py` (marker `paid`) assembles the toy host's
bundle and has a real model write its analysis with the seed writer prompt and the authored
response schema. It asserts the analysis round-trips the stored contract, every finding's evidence
resolved to a reading code filled in, and no repair loop exhausted; a second test sends the contract
with a required field removed and asserts the lane fails. Every other generator test answers with a
fixtured completion, so a prompt, schema or fixture defect shows up only here.

It spends money, so it is opt-in: `THREETEARS_EVALS_PAID=1 ./scripts/test.sh evals -m paid -rs`,
with `ANTHROPIC_API_KEY` set. Without the flag, or without a key, both tests skip; they never fail
for want of one. `THREETEARS_EVALS_PAID_MODEL` picks the writer (default `claude-opus-5-5`). Each call
is priced at its worst case before it is sent and refused unsent past the lane's ceiling
(`SPEND_CEILING_USD`, $5; a typical run is two calls at well under a dollar each on the default
writer). CI runs it (`.github/workflows/evals-paid.yml`) on a PR touching `analysis/generator.py`,
`analysis/gen_prompt.py`, `kernel/authored.py` or the toy-host fixture, behind the `evals-paid`
environment's approval gate; there a missing key fails the job rather than skipping. The job is
skipped unless the repository variable `EVALS_PAID_ENABLED` is `true`, which it is not until the
environment has its reviewers and key.

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
