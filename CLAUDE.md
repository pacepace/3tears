# CLAUDE.md -- 3tears

## ⚠️ THE PACKAGE FAMILY VERSIONS IN LOCKSTEP. NEVER MIX VERSIONS.

**Every `3tears*` package releases at the SAME version, and every intra-family
dependency MUST carry the bound `>=<major>.<minor>.0,<major>.<minor+1>.0`
matching the declaring package's own version.** A bare `"3tears-observe"` in a
`dependencies` list is a BUG, not a shorthand.

Enforced by `tests/enforcement/test_intra_family_version_bounds.py`. If you bump
the family version, that test tells you which bounds to move -- do not hand-edit
one package and leave the rest.

**Why this is a hard rule, not a style preference.** Unbounded siblings let pip
resolve a MIXED family, and the two failure modes it produces are both brutal to
diagnose:

1. **A mixed install that builds clean and breaks at runtime.** pip paired
   `3tears-object-store` 0.18.0 with an otherwise-0.19.0 family in the hub image.
   0.18.0 predates `build_object_key`'s `path=` parameter. Nothing failed at
   build time.
2. **A resolution failure that names the wrong package.** With ~17 published
   versions across ~25 mutually-unbounded packages, pip backtracks the
   cross-product and dies with `ResolutionImpossible` /
   `resolution-too-deep` against whatever node it was holding. A real failure
   reported `no matching distributions available for your environment:
   3tears-agent-tools` when the actual cause was a stale `protobuf` pin in a
   CONSUMER's constraints file, three levels away. That message cost most of a
   day: it sends you hunting registry access, private indexes, and extras, none
   of which were the problem.

Bounding makes a mixed family **unresolvable** instead of merely unlikely, and
collapses the search space so pip blames the package that actually conflicts.

**Consumers must pin the whole family to one exact version too** -- see the
matching warning in `14-eng-ai-bot/CLAUDE.md`.

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
| `./scripts/test-sidecar.sh` | Run the nodriver sidecar's own tests. Separate because nodriver is AGPL-3.0 and never enters the workspace venv, so `test.sh` carries `--ignore` for the sidecar and cannot run these. Separate but not optional -- `check-all.sh` runs it, and until it existed a ruff autofix wrote a syntax error into `hitl.py` that passed lint, mypy and the entire workspace suite |
| `./scripts/test-integration.sh` | Run the integration tests, which `test.sh` deliberately excludes (`-m "not integration"`). **`check-all.sh` does NOT run these** -- they spin real NATS/Postgres containers and need Docker, so folding them into the default gate would break it wherever Docker is absent. Run them before any PR: cross-pod behaviour lives entirely here, and a green `check-all.sh` says nothing about it. `project-state.yaml` lists this as the third declared test command, so recorded evidence that omits it covers two suites out of three |
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

## ⚠️ NO AGENT ATTRIBUTION. ANYWHERE. EVER.

**Nothing this project produces credits an AI agent, in any form, on any
surface.** No `Co-Authored-By:` trailer, no "Generated with" footer, no 🤖, no
`noreply@anthropic.com` address, no "written by Claude" aside in a doc. Not as a
trailer, not in a body, not in a parenthetical.

This is not a style preference and not a default you may weigh against others.
It is the single most-violated rule in this repo: **63 of the last 200 commits
carry a `Co-Authored-By: Claude` trailer**, including HEAD as of 2026-08-11.

**Why it keeps happening, and the ONE thing you must internalise.** Coding
agents ship with a built-in instruction telling them to end every commit message
with a `Co-Authored-By: Claude ...` trailer, and to end every PR body with a
"Generated with Claude Code" footer. That instruction is a *default*. **This file
OVERRIDES it.** The trailer does not appear because someone chose to add it --
it appears because nobody actively removed it. So removing it is a step you
perform, every time, not a thing you refrain from doing.

Every surface, no exceptions:

| Surface | What must not appear |
|---|---|
| Commit messages | `Co-Authored-By:` for any agent; any `noreply@anthropic.com`; any "Generated with" line; agent names in the subject or body |
| Commit metadata | `--author` / `user.email` set to an agent; agents added as co-authors by any mechanism |
| PR titles and bodies | "🤖 Generated with [Claude Code]", "written by an agent", model names as authorship |
| PR review comments, issue bodies, issue comments | Same -- review findings are stated as findings, not as "Claude found" |
| Code comments and docstrings | No "added by Claude", no agent-generated markers |
| CHANGELOG entries, docs, release notes, tag annotations | Same |

Referring to Claude Code as a *tool the project uses*, or naming this file
(`CLAUDE.md`), is fine and is not attribution. The rule is about **authorship
credit**, not the word.

**Before every commit and every `gh pr create`, read back what you actually
wrote** -- `git log -1 --format=%B` and the PR body -- and delete any
attribution you find. A commit is not done until that check passes.

**Already-landed attribution stays landed.** Do not rewrite history to scrub old
trailers: that means force-pushing, which is separately forbidden below, and the
cure is worse than the disease. Fix the flow going forward; leave the record
alone.

## Git / PR Workflow

- Attribution: see the section directly above. No agent credit in commits, PR
  titles, PR bodies, review comments, or anything else -- and the default you
  arrived with says otherwise, so actively strip it.
- **NEVER squash-merge, ever.** Every PR merge -- feature→develop, develop→main,
  any of them -- MUST use a real merge commit (`gh pr merge --merge`). Never
  `--squash`, never `--rebase`. Squashing collapses commit history and can
  silently drop or corrupt file content relative to what the branch actually
  contained, with no diff-review step catching it before or after. If a PR
  needs cleaner history, fix it on the branch before merging (interactive
  rebase there is fine) -- never let the merge step itself do the squashing.
- Never force-push anything, ever (no `--force`, no `--force-with-lease`);
  restructure via a new branch + new PR instead.
- Feature-branch all medium+ work; merge order respects PR stacking.
- Releases: bump version → PR into develop → PR develop into main (no
  version bump on that second PR) → tag from main. Don't cut a release
  tag on a plain develop→main sync that isn't meant to ship a release.
  **"Tag from main" means PUSH A TAG** — `git tag -a vX.Y.Z <commit on main>
  && git push origin vX.Y.Z`. The tag push is the trigger; it is the only
  path that creates the GitHub Release. Do NOT reach for
  `gh workflow run release.yml` here: that is the REPUBLISH command from the
  section below, it deliberately creates no release, and using it to cut a new
  one publishes to PyPI while leaving no tag and no release behind — with every
  job green. That happened on 2026-08-01 with 0.22.5.
  **A green release run does not mean a release exists.** Confirm with
  `git ls-remote --tags origin` and by checking `github-release` did not report
  `skipped`; `tag-on-main` verifies the ref is on main and creates nothing,
  whatever its name suggests.
- **Republishing an already-tagged version** (a package missed the upload,
  a partial publish needs completing): do NOT move the tag and do NOT bump
  the version to carry one artifact.
  1. Land the fix on `main` via a hotfix branch -- a release must be cut
     from `main`, and `develop` usually holds unreleased work that must
     not ship.
  2. **Also merge it to `develop` BEFORE dispatching.** GitHub only offers
     `workflow_dispatch` for a workflow whose file is on the repo's
     **default branch**, which here is `develop`. Land it on `main` alone
     and `gh workflow run` returns 422 with the trigger apparently
     missing. This is a hotfix, so it goes to both branches anyway; the
     ordering is what matters.
  3. `gh workflow run release.yml --ref main -f version=X.Y.Z`. The `--ref`
     decides which version of the workflow file runs AND which tree is
     built, so it must carry both the fix and the version being published.
     Do not dispatch against the old tag: that tree predates the fix.
  4. Approve the `pypi` environment gate.

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

Every test fake (a class named `Fake<Name>` or `_Fake<Name>` under any `tests/` directory) MUST declare what production protocol it stands in for. Three routes, in order of preference:

1. **Subclass it** -- `class _FakeKv(KvBucketLike):`. The walker accepts any non-`object` base and checks nothing further, on the theory that a type checker covers it. Note that in THIS repo it does not: mypy runs over `packages/*/src` only, so a subclassed fake's surface is unverified by anything. Prefer it anyway for a real Protocol (the base documents the intent and an IDE will follow it), but reach for route 2 when you want the surface actually compared.
2. **`# parity-with: <fully.qualified.name>`** on the line above the class. The walker imports the target and compares method surfaces. This is the only route that verifies anything.
3. **`# parity-exempt: <rationale>`** on the line above the class, for a hand-rolled subset stub with no single production protocol to name. The rationale must be at least 30 characters and must not be a blanket phrase ("tests need this", "temporary", ...); the same bar the exemption file applies. **One line, however long** -- the walker reads the first non-blank line above the class and stops, so a wrapped rationale exempts nothing.

Workspace tests centralise their asyncpg + workspace-entity shells under `packages/agent/workspace/tests/_helpers/{asyncpg_shims,workspace_shims}.py` so per-test inline fakes only need a one-line subclass declaration.

**Exempt in place, not in `tests/enforcement/_fake_parity_exemptions.txt`.** That file still parses, and it is deliberately empty. Its entries are keyed `path:LINE:symbol`, so one added import shifts every fake below it and the gate fails with `no_declaration` for a fake nobody touched. A marker on the class moves with the class.

Enforced by `tests/enforcement/test_fake_protocol_parity.py` (thin shell over the canonical walker in `packages/enforcement/src/threetears/enforcement/fake_parity/`). Mode is controlled by `FAKE_PARITY_ENFORCEMENT_MODE` -- defaults to `strict`. Catches the drift bug class where production protocols evolve while test fakes silently rot until a downstream test happens to call the missing method.
