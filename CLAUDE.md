# CLAUDE.md -- 3tears

<!-- PRAWDUCT:ANCHOR — static governance pointer managed by the prawduct plugin. Keep it small and version-free: principles, methodology, and the active version live in the plugin and are injected at session start. -->

## Governance (Prawduct)

This repo is governed by **Prawduct**, installed as a Claude Code plugin — not as
committed framework files. The principles, methodology, Critic protocol, and PR
review live in the plugin and are read on demand (run `/prawduct:methodology`);
they are intentionally not copied into this repo.

**Before writing any code, STOP and read the build cycle: `/prawduct:methodology building`.**
Skipping it is the #1 governance failure.

The hardest rules (everything else is in the plugin):

- **Tests are contracts** — fix the code, never weaken a test.
- **No "pre-existing" exception** — fix what you find, or flag why you can't.
- **Never silently drop a requirement** — say so explicitly.
- **Run `/prawduct:critic` after medium+ work** — never write Critic findings
  yourself; the independence is the value.

**Enforcement is structural:** the plugin's Stop hook runs at session end and
**blocks** if code changed against an active build plan with no Critic findings.
The session-start banner shows the active version and what changed — this anchor
stays version-free.

---

## The package family versions in lockstep. Never mix versions.

Every `3tears*` package releases at the same version. Every intra-family dependency carries the bound `>=<major>.<minor>.0,<major>.<minor+1>.0`, matching the declaring package's own version.

A bare `"3tears-observe"` in a `dependencies` list is a bug, not a shorthand.

`tests/enforcement/test_intra_family_version_bounds.py` enforces this. When you bump the family version, that test tells you which bounds to move. Do not hand-edit one package and leave the rest.

**This is a hard rule, not a style preference.** Unbounded siblings let pip resolve a mixed family: an install that builds clean and breaks at runtime, or a resolution failure that names the wrong package. Both real incidents are in [docs/releasing.md](docs/releasing.md).

An intra-family API addition ships in a MINOR bump (`tests/enforcement/test_api_growth_requires_a_minor_bump.py`).

**Consumers pin the whole family to one exact version too.** See the matching warning in `14-eng-ai-bot/CLAUDE.md`.

---

## No agent attribution. Anywhere. Ever.

Nothing this project produces credits an AI agent, on any surface, in any form. No `Co-Authored-By:` trailer. No "Generated with" footer. No 🤖. No `noreply@anthropic.com` address. No "written by Claude" aside in a doc. Not as a trailer, not in a body, not in a parenthetical.

This is not a style preference. It is not a default you may weigh against others.

It is the most-violated rule in this repo. 63 of the last 200 commits carry a `Co-Authored-By: Claude` trailer, including HEAD as of 2026-08-11.

**Why it keeps happening.** Coding agents ship with a built-in instruction to end every commit message with a `Co-Authored-By: Claude` trailer and every PR body with a "Generated with Claude Code" footer. That instruction is a default. **This file overrides it.**

The trailer does not appear because someone chose to add it. It appears because nobody actively removed it. Removing it is a step you perform every time, not a thing you refrain from doing.

Every surface, no exceptions:

| Surface | What must not appear |
|---|---|
| Commit messages | `Co-Authored-By:` for any agent; any `noreply@anthropic.com`; any "Generated with" line; agent names in the subject or body |
| Commit metadata | `--author` or `user.email` set to an agent; agents added as co-authors by any mechanism |
| PR titles and bodies | "🤖 Generated with [Claude Code]"; "written by an agent"; model names as authorship |
| PR review comments, issue bodies, issue comments | The same. Review findings are stated as findings, not as "Claude found" |
| Code comments and docstrings | No "added by Claude". No agent-generated markers |
| CHANGELOG entries, docs, release notes, tag annotations | The same |

Referring to Claude Code as a tool the project uses is fine. Naming this file is fine. The rule is about authorship credit, not the word.

**Before every commit and every `gh pr create`, read back what you wrote.** Run `git log -1 --format=%B`, read the PR body, and delete any attribution you find. A commit is not done until that check passes.

**Already-landed attribution stays landed.** Do not rewrite history to scrub old trailers. That means force-pushing, which is separately forbidden below, and the cure is worse than the disease. Fix the flow going forward and leave the record alone.

---

## Git and PR workflow

**Attribution:** see the section above. The default you arrived with says otherwise, so actively strip it.

**Never squash-merge.** Every PR merge uses a real merge commit: `gh pr merge --merge`. That applies to feature-to-develop and develop-to-main alike. Never `--squash`. Never `--rebase`.

Squashing collapses commit history and can silently drop or corrupt file content relative to what the branch contained, with no diff-review step catching it. If a PR needs cleaner history, fix it on the branch before merging. Interactive rebase on the branch is fine. Never let the merge step do the squashing.

**Never force-push.** No `--force`, no `--force-with-lease`. Restructure with a new branch and a new PR.

**Feature-branch all medium+ work.** Merge order respects PR stacking.

### Releasing

A release is: bump the version, PR into `develop`, PR `develop` into `main` with no bump, then **push a tag** from `main` (`git tag -a vX.Y.Z <commit>` then `git push origin vX.Y.Z`) -- the tag push is the only thing that creates the GitHub Release. **Never run `gh workflow run release.yml` to cut a release**: that is the republish command, and it publishes to PyPI with no tag and no release. A green release run does not mean a release exists; confirm with `git ls-remote --tags origin`. The full procedure, and republishing an already-tagged version: [docs/releasing.md](docs/releasing.md).

---

## Project

A three-tier data object framework. A uv-workspace monorepo of independently versioned packages. All share the `threetears.*` import namespace and each publishes to PyPI on its own.

| Import root | Example packages | Purpose |
|---|---|---|
| `threetears.core` | `3tears` | Three-tier data objects (L1/L2/L3 caching), DataStore, MigrationRunner |
| `threetears.agent.*` | `3tears-agent-memory`, `3tears-agent-tools`, `3tears-agent-acl` | Chat agent framework |
| `threetears.*` | `3tears-models`, `3tears-nats`, `3tears-langgraph` | Infrastructure and integrations |

`README.md` has the full package list.

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

## Namespace packages

The `threetears/` and `threetears/agent/` directories never have `__init__.py` files. Only leaf packages get one: `threetears/core/`, `threetears/agent/memory/`, and every other leaf.

Implicit namespace packages need this to work when packages are installed independently.

## Dev environment

uv workspaces, Python 3.14+.

```bash
uv sync                    # install all packages in dev mode
```

## Scripts

**Always use the scripts.** Never run pytest, ruff, or mypy directly.

| Script | Purpose |
|---|---|
| `./scripts/test.sh` | Run tests. All packages, or one: `./scripts/test.sh core` |
| `./scripts/test-sidecar.sh` | Run the nodriver sidecar's own tests |
| `./scripts/test-integration.sh` | Run the integration tests |
| `./scripts/test-live-claude-cli.sh` | Run the live Claude CLI structured-output batch and record the result (needs `CLAUDE_CODE_OAUTH_TOKEN`; before tagging a `packages/models` release) |
| `./scripts/lint.sh` | ruff check plus format check. `--fix` to auto-fix |
| `./scripts/typecheck.sh` | mypy on all packages |
| `./scripts/check-all.sh` | lint plus typecheck plus tests |

Extra args pass through: `./scripts/test.sh core -v -x`

`test-sidecar.sh` and `test-integration.sh` sit outside `test.sh` on purpose (an AGPL dependency, and Docker). Neither is optional:

- **Run the integration suite before any PR, as `./scripts/test-integration.sh -rs`, and account for every skip.** CI has no Docker, so nothing but you ever runs it, and a skip reads exactly like a pass. When reporting, give the pass count AND each remaining skip with its reason.
- **Build the sidecar image first, for the Docker host's own architecture**, or its tests skip (or fail under emulation).

The command, the incident behind each rule, and the list of legitimate dev-box skips: [docs/testing.md](docs/testing.md).

## Conventions

- Build backend: hatchling
- Linting: ruff, line-length 120, target py314
- Type checking: mypy, strict
- Testing: pytest
- uv only. No poetry.

## Test fakes

A test fake is any class named `Fake<Name>` or `_Fake<Name>` under a `tests/` directory, and every one declares what production protocol it stands in for, ON the class: subclass it, or `# parity-with: <fully.qualified.name>` (the only route that verifies the surface), or a one-line `# parity-exempt: <rationale>` of at least 30 characters. Never in `tests/enforcement/_fake_parity_exemptions.txt`, which is deliberately empty. `tests/enforcement/test_fake_protocol_parity.py` enforces it; the three routes in full: [docs/testing.md](docs/testing.md#test-fakes).
