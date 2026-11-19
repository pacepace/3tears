# Build plan: the typecheck on the testament pool, served from its cache

Owner request, 2026-11-19: "Type check must move to testament, and be cacheable such that running it from
here (using testament) and then immediately from GitHub ci will reuse the results rather than running
twice on identical code." Also: "make sure that our typechecks are running".

## Goals

1. mypy runs as a testament suite (`typecheck`), locally through `./scripts/typecheck.sh` and in CI.
2. Its results are served from testament's cache: a run on a tree that already passed, from the laptop
   or from CI, is not run again (apart from testament's own sampled re-checks).
3. CI's verdict on types is always reported. Today the `Typecheck` step sits after the unit verdict in
   `check`, so a red unit suite skips it (run of #1278, 2026-11-19: `Typecheck=skipped`), and lint with it.
4. The checked set stays what it is: every package's import roots, plus the evals example hosts.

## Facts this rests on (testament v0.56.0, confirmed with the testament session 2026-11-19)

- A `files` suite is never served from the cache (`has_recorder=False`), so the suite is pytest.
- A pytest test that starts a subprocess, reads the clock or reads a file outside the tree, the venv and
  temp is uncacheable. So mypy runs in-process (`mypy.api.run`) with `time.time` frozen; mypyc calls it
  through the module attribute (probe: 406 calls caught on one package), `--cache-dir` under pytest's
  temp, and an explicit `--config-file` so mypy probes no user config.
- The cache key is (project, suite, tree, env); env holds the manifest and every `v*` tag in the
  submitting checkout. CI's and the laptop's keys share results. Same tree = a clean committed checkout
  that already contains the base (a PR's merge commit then has the head's tree).
- 3tears is granted serving. About 15% of would-be hits re-run as the cache's own check.

## Inferences (alignment pass: proceed, citing the request)

- [DECISION] The tests live in a new top-level `typecheck/` directory, outside the unit suite's
  `packages/ tests/` selection, so no marker has to keep them out of the unit suite, its local form or
  the GitHub shards.
- [DECISION] One test per package (ids `core`, `agent.memory`, ...) plus one for the example hosts. The
  roots are read off each package's `src/threetears/` tree, which replaces `typecheck.sh`'s hard-coded
  list (identical today: 32 packages, 33 roots). A test pins `[tool.mypy] files` to the same set.
- [DECISION] Each test reports every error mypy prints, a dependency's included: a dependency's error
  fails its dependents as well as itself, which is never a false pass.
- [DECISION] `typecheck.sh` keeps `[package]`; its remaining arguments now go to pytest, not mypy. Raw
  mypy flags (`--no-incremental`) no longer pass through: the run is cached and owns its mypy flags.
  Stated here so it is not a silent drop.
- [DECISION] `typecheck.sh` goes to the pool where testament is installed (outside CI), `fallback =
  "pool-only"` as for test.sh; `TESTAMENT=off` with a stated reason runs it here (cheap, unlike the unit
  suite).
- [DECISION] The suite ships no release tags (`[suites.typecheck.git] tags = []`, testament 0.57.0, built
  for this as brookstalley/testament#886). The project's tags otherwise join every suite's cache key, and
  a laptop whose tags differ from origin's (this one had four) never shares a result with CI. This
  replaced a tag check in `typecheck.sh` that fetched origin's tags and named those that differed.
  Testament 0.57.0 becomes the minimum for the repo.
- [DECISION] CI: a `typecheck` job beside the unit jobs. On the pool when `route` chose it
  (`ci-unit-pool.sh typecheck`: no `--fresh`, no `--no-cache`, so results are served), else, or when the
  pool is lost, `./scripts/typecheck.sh` on GitHub's runner. `check` needs it and reports its verdict.
- [DECISION] `check`'s steps after the unit verdict run unless the run was cancelled, so lint and the
  floor suites report beside a red unit suite. The unit verdict stays the first step (pinned by tests).
- [DECISION, alignment-pass miss] `release.yml`'s gate ran its own `mypy` over eight packages, not
  `typecheck.sh` as this plan first assumed (Critic, cumulative review). It now runs `typecheck.sh` (CI set,
  so on its runner): every package. An enforcement test holds every workflow to the script.
- [DECISION] mypy's incremental cache lives under pytest's temp and is dropped with the run: testament's
  result cache does that work, and a cache carried between runs on a runner is state the key never covers.
- [DECISION] mypy's fallback stub directory `/usr/local/lib/mypy` (listed on every POSIX run, for a broken
  install) is reported missing to it: a host's stubs there are a read outside the tree, and would make a
  result depend on the runner.

## Chunks

1. `typecheck/test_mypy.py`, `[suites.typecheck]`, `typecheck.sh`, its tests.
2. CI: `ci-unit-pool.sh typecheck`, the `typecheck` job, `check`'s verdict and step conditions, their tests.
3. Docs: CLAUDE.md, README, docs/testing.md, CHANGELOG.

## Done when

- The typecheck suite passes on the pool, and a second run of the same tree is served from the cache
  (`report.cache`: served, none uncacheable).
- Tests: the script hand-off (pool, local, CI, the TESTAMENT=off reason, package selection, tag
  warning); the CI pool mode's flags; the workflow's typecheck job, `check`'s needs and verdict.
- lint, the enforcement suite and the touched tests pass; Critic run.
