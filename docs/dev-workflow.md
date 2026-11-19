# Repository layout, dev environment and scripts

Moved out of `CLAUDE.md` so every session's load stays within its budget; the rules that bind stay there, each pointing here for the detail.

## Structure

```
packages/
  core/               # PyPI: 3tears
  models/             # PyPI: 3tears-models
  nats/               # PyPI: 3tears-nats
  ...                 # top-level packages live directly under packages/
  agent/
    memory/           # PyPI: 3tears-agent-memory
    tools/            # PyPI: 3tears-agent-tools
    ...               # the agent-* family lives under packages/agent/
```

Each package has its own `pyproject.toml`, `src/`, and `tests/`.

## Dev environment

uv workspaces, Python 3.14+.

```bash
uv sync                    # install all packages in dev mode
```

## Scripts

**Always use the scripts.** Never run pytest, ruff, or mypy directly.

| Script | Purpose |
|---|---|
| `./scripts/test.sh` | Run tests. One package while you work (`./scripts/test.sh core`, `agent/memory`), which leaves evals out unless you name it; all packages with no argument. Before a PR, `./scripts/check-all.sh`, which runs it and CI's other suites |
| `./scripts/test-sidecar.sh` | Run the nodriver sidecar's own tests |
| `./scripts/test-integration.sh` | Run the integration suite on the testament pool; `--here` runs the docker and credentialed tests on this machine |
| `./scripts/test-live-claude-cli.sh` | Run the live Claude CLI batch -- structured output and the session pool -- and record the result (needs `CLAUDE_CODE_OAUTH_TOKEN`; before tagging a `packages/models` release) |
| `./scripts/lint.sh` | ruff check plus format check. `--fix` to auto-fix |
| `./scripts/typecheck.sh` | mypy, strict, on every package's sources and tests, or one package's: `./scripts/typecheck.sh evals`; `--src` / `--tests` for one half |
| `./scripts/check-all.sh` | lint plus typecheck plus tests, including CI's mcp-1.x and OpenTelemetry-floor suites |

Extra args pass through: `./scripts/test.sh core -v -k lease`

Where `testament` is installed, `test.sh` hands the run to testament's runners (outside CI), and only there: the manifest is `fallback = "pool-only"`, so a pool that can't take the run fails it rather than running it on this machine. `TESTAMENT=off` forces a local run, with the reason stated in `TESTAMENT_OFF_REASON`. A pool run leaves out the integration tests, which `test-integration.sh` covers, and the `loop_timing` tests, which CI runs serially. A `-m` narrows the pool's selection. `-x`, `--maxfail`, `--lf`, `-n` and their kin need one local pytest process, and `-s`, `--pdb` and `--trace` this terminal: with only named test files (`./scripts/test.sh core packages/core/tests/test_x.py -x`) the run goes local, and a wider run is refused. A run gated by an environment variable (`RECORD`, `THREETEARS_LIVE_*`, `TAVILY_API_KEY`) stays local, since on the pool its tests would skip.

**When the pool can't take a run** (`test.sh` exits 2, "testament gave no verdict"), stop and tell the person. Never rerun a unit-suite package or the whole unit suite with `TESTAMENT=off`, never run `ci-unit-tests.sh` or raw `pytest` to get past it, and never edit `.testament.toml`. Named test files under `TESTAMENT=off` are fine. The integration suite is the one exception the runbook names: with the pool down, the person may run it one package at a time through testcontainers (`TESTAMENT=off TESTAMENT_OFF_REASON="pool down" ./scripts/test-integration.sh <package> -rs`), on a machine with Docker; an agent asks first. What gates a PR meanwhile: [docs/testing.md](docs/testing.md), "When the pool is down".

`test-sidecar.sh` and `test-integration.sh` sit outside `test.sh` on purpose (an AGPL dependency, and the pool's integration suite). Neither is optional:

- **Run the integration suite before any PR, as `./scripts/test-integration.sh -rs`** (`check-all.sh` runs it where the pool is reachable, and fails saying so where it isn't). It runs on the testament pool, where Postgres, NATS and S3 run as processes and a skip fails. When a change touches scrape, search, the nats lame-duck handover or a credentialed package, also run `./scripts/test-integration.sh --here -rs` on this machine: it runs the `docker` and `secrets` tests, which never reach the pool. CI's `integration` job covers only core, nats, epoch, scheduled-jobs and agent/acl. When reporting, give the pass count AND each skip with its reason. The evals render tests are not in it: they need headless Chromium, carry the `browser` marker and run in the unit suite, on the testament pool and CI's evals shards.
- **Before `--here`, build the sidecar image for the Docker host's own architecture**, or its sidecar test skips (or fails under emulation).

The command, the incident behind each rule, and the list of legitimate dev-box skips: [docs/testing.md](docs/testing.md).
