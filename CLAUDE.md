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

## Git / PR Workflow

- **NEVER squash-merge, ever.** Every PR merge — feature→develop, develop→main,
  any of them — MUST use a real merge commit (`gh pr merge --merge`). Never
  `--squash`, never `--rebase`. Squashing collapses commit history and can
  silently drop or corrupt file content relative to what the branch actually
  contained, with no diff-review step catching it before or after. If a PR
  needs cleaner history, fix it on the branch before merging (interactive
  rebase there is fine) — never let the merge step itself do the squashing.
- Never force-push anything, ever (no `--force`, no `--force-with-lease`);
  restructure via a new branch + new PR instead.
- Feature-branch all medium+ work; merge order respects PR stacking.
- Releases: bump version → PR into develop → PR develop into main (no
  version bump on that second PR) → tag from main. Don't cut a release
  tag on a plain develop→main sync that isn't meant to ship a release.
- **Republishing an already-tagged version** (a package missed the upload,
  a partial publish needs completing): do NOT move the tag and do NOT bump
  the version to carry one artifact. Land the fix on `main` via a hotfix
  branch — never through `develop`, which usually holds unreleased work —
  then run the `Release` workflow manually (`gh workflow run release.yml
  --ref main -f version=X.Y.Z`) and approve the `pypi` environment gate.
  `skip-existing` means everything already on PyPI is skipped, so the only
  possible effect is that a genuinely absent artifact uploads. This is
  written here rather than only in `release.yml` because v0.18.0 shipped
  26 of 27 packages while the instruction that would have prevented it sat
  in a comment inside the step it was telling you to delete.

## Conventions

- Build backend: hatchling
- Linting: ruff (line-length 120, target py314)
- Type checking: mypy (strict)
- Testing: pytest
- No poetry -- uv only

## Test Fakes

Every test fake (a class named `Fake<Name>` or `_Fake<Name>` under any `tests/` directory) MUST declare what production protocol it stands in for, via subclass declaration, a `# parity-with: <fully.qualified.name>` marker comment, or an exemption with `# rationale:` line in `tests/enforcement/_fake_parity_exemptions.txt`. Workspace tests centralise their asyncpg + workspace-entity shells under `packages/agent/workspace/tests/_helpers/{asyncpg_shims,workspace_shims}.py` so per-test inline fakes only need a one-line subclass declaration.

Enforced by `tests/enforcement/test_fake_protocol_parity.py` (thin shell over the canonical walker in `packages/enforcement/src/threetears/enforcement/fake_parity/`). Mode is controlled by `FAKE_PARITY_ENFORCEMENT_MODE` -- defaults to `strict`. Catches the drift bug class where production protocols evolve while test fakes silently rot until a downstream test happens to call the missing method.
