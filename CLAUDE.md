# CLAUDE.md -- 3tears

## Project

Three-tier data object framework. A uv-workspace monorepo of independently-versioned packages, all sharing the `threetears.*` import namespace and each published to PyPI on its own. Three families:

| Import root | Example packages | Purpose |
|---|---|---|
| `threetears.core` | `3tears` | Three-tier data objects (L1/L2/L3 caching), DataStore, MigrationRunner |
| `threetears.agent.*` | `3tears-agent-memory`, `3tears-agent-tools`, `3tears-agent-acl`, ... | Chat agent framework |
| `threetears.*` | `3tears-models`, `3tears-nats`, `3tears-langgraph`, ... | Infrastructure and integrations |

See `README.md` for the full package list.

## Dev Environment

Uses **uv workspaces**. Python 3.14+.

```bash
uv sync                    # install all packages in dev mode
```

## Scripts

**Always use the scripts.** Never run pytest, ruff, or mypy directly.

| Script | Purpose |
|---|---|
| `./scripts/test.sh` | Run tests (all packages, or specify one: `./scripts/test.sh core`) |
| `./scripts/lint.sh` | Run ruff check + format check (`--fix` to auto-fix) |
| `./scripts/typecheck.sh` | Run mypy on all packages |
| `./scripts/check-all.sh` | Run lint + typecheck + tests |

Extra args pass through: `./scripts/test.sh core -v -x`

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

## Namespace Packages

The `threetears/` and `threetears/agent/` directories must **never** have `__init__.py` files. Only leaf packages (`threetears/core/`, `threetears/agent/memory/`, and every other leaf) get `__init__.py`. This is required for implicit namespace packages to work when packages are installed independently.

## Conventions

- Build backend: hatchling
- Linting: ruff (line-length 120, target py314)
- Type checking: mypy (strict)
- Testing: pytest
- No poetry -- uv only

## Test Fakes

Every test fake (a class named `Fake<Name>` or `_Fake<Name>` under any `tests/` directory) MUST declare what production protocol it stands in for, via subclass declaration, a `# parity-with: <fully.qualified.name>` marker comment, or an exemption with `# rationale:` line in `tests/enforcement/_fake_parity_exemptions.txt`. Workspace tests centralise their asyncpg + workspace-entity shells under `packages/agent/workspace/tests/_helpers/{asyncpg_shims,workspace_shims}.py` so per-test inline fakes only need a one-line subclass declaration.

Enforced by `tests/enforcement/test_fake_protocol_parity.py` (thin shell over the canonical walker in `packages/enforcement/src/threetears/enforcement/fake_parity/`). Mode is controlled by `FAKE_PARITY_ENFORCEMENT_MODE` -- defaults to `strict`. Catches the drift bug class where production protocols evolve while test fakes silently rot until a downstream test happens to call the missing method.
