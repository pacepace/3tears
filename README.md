# 3tears

Three-tier data framework for Python applications with LLM agent support.

## Packages

| Package | PyPI | Import | Description |
|---------|------|--------|-------------|
| [3tears](packages/core/) | `pip install 3tears` | `threetears.core` | Three-tier caching — L1 SQLite, L2 NATS KV, L3 PostgreSQL |
| [3tears-agent-memory](packages/agent-memory/) | `pip install 3tears-agent-memory` | `threetears.agent.memory` | Memory system for LLM agents |
| [3tears-agent-tools](packages/agent-tools/) | `pip install 3tears-agent-tools` | `threetears.agent.tools` | Tool framework for LLM agents |

## Architecture

```
L1 (SQLite, in-process, sync)  →  L2 (NATS KV, shared, async)  →  L3 (PostgreSQL, persistent, async)
```

Reads promote up the stack. Writes flow down with configurable flush strategies.

## Development

```bash
uv sync                      # install all packages in dev mode
./scripts/check-all.sh       # lint + typecheck + tests
./scripts/test.sh             # tests only
./scripts/test.sh core        # single package
./scripts/lint.sh             # ruff check + format
./scripts/typecheck.sh        # mypy strict
./scripts/bump-version.sh     # bump patch (or: major, minor)
```

## Releasing

```bash
./scripts/bump-version.sh patch   # or: minor, major
git add -A && git commit -m "bump version to X.Y.Z"
git tag vX.Y.Z
git push origin develop --tags
```

Pushing a tag triggers CI: lint, typecheck, test, build, and publish to PyPI.
