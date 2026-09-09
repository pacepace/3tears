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

**This is a hard rule, not a style preference.** Unbounded siblings let pip resolve a mixed family. That produces two failure modes, both brutal to diagnose.

**A mixed install that builds clean and breaks at runtime.** pip paired `3tears-object-store` 0.18.0 with an otherwise-0.19.0 family in the hub image. 0.18.0 predates `build_object_key`'s `path=` parameter. Nothing failed at build time.

**A resolution failure that names the wrong package.** With about 17 published versions across about 25 mutually-unbounded packages, pip backtracks the cross-product and dies with `ResolutionImpossible` or `resolution-too-deep` against whatever node it was holding. One real failure reported `no matching distributions available for your environment: 3tears-agent-tools`. The actual cause was a stale `protobuf` pin in a consumer's constraints file, three levels away. That message cost most of a day. It sends you hunting registry access, private indexes, and extras, none of which were the problem.

Bounding makes a mixed family unresolvable rather than merely unlikely. It also collapses the search space, so pip blames the package that actually conflicts.

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

### Cutting a release

1. Bump the version.
2. PR into `develop`.
3. PR `develop` into `main`, with no version bump on that second PR.
4. Tag from `main`.

Do not cut a release tag on a plain develop-to-main sync that is not meant to ship.

**"Tag from main" means push a tag.** Run `git tag -a vX.Y.Z <commit on main>` then `git push origin vX.Y.Z`. The tag push is the trigger. It is the only path that creates the GitHub Release.

**Do not run `gh workflow run release.yml` to cut a release.** That is the republish command below. It deliberately creates no release. Using it to cut a new one publishes to PyPI while leaving no tag and no release behind, with every job green. That happened on 2026-08-01 with 0.22.5.

**A green release run does not mean a release exists.** Confirm with `git ls-remote --tags origin`, and check that `github-release` did not report `skipped`. `tag-on-main` verifies the ref is on main and creates nothing, whatever its name suggests.

### Republishing an already-tagged version

Use this when a package missed the upload, or a partial publish needs completing. Do not move the tag. Do not bump the version to carry one artifact.

1. **Land the fix on `main` via a hotfix branch.** A release is cut from `main`, and `develop` usually holds unreleased work that must not ship.
2. **Merge it to `develop` before dispatching.** GitHub only offers `workflow_dispatch` for a workflow whose file is on the repo's default branch, which here is `develop`. Land it on `main` alone and `gh workflow run` returns 422 with the trigger apparently missing. This is a hotfix, so it goes to both branches anyway. The ordering is what matters.
3. Run `gh workflow run release.yml --ref main -f version=X.Y.Z`. The `--ref` decides which version of the workflow file runs and which tree is built, so it must carry both the fix and the version being published. Do not dispatch against the old tag: that tree predates the fix.
4. Approve the `pypi` environment gate.

`skip-existing` means everything already on PyPI is skipped. The only possible effect is that a genuinely absent artifact uploads.

This is written here rather than only in `release.yml` because v0.18.0 shipped 26 of 27 packages while the instruction that would have prevented it sat in a comment inside the step it was telling you to delete.

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
| `./scripts/lint.sh` | ruff check plus format check. `--fix` to auto-fix |
| `./scripts/typecheck.sh` | mypy on all packages |
| `./scripts/check-all.sh` | lint plus typecheck plus tests |

Extra args pass through: `./scripts/test.sh core -v -x`

**Why the sidecar is separate.** nodriver is AGPL-3.0 and never enters the workspace venv, so `test.sh` carries `--ignore` for the sidecar and cannot run these. Separate but not optional: `check-all.sh` runs it. Until it existed, a ruff autofix wrote a syntax error into `hitl.py` that passed lint, mypy, and the entire workspace suite.

**Why integration tests are separate.** `test.sh` excludes them with `-m "not integration"`. `check-all.sh` does not run them either: they spin real NATS and Postgres containers and need Docker, so folding them into the default gate would break it wherever Docker is absent. **CI cannot run them at all — GitHub Actions has no Docker — so nothing but you, locally, ever executes them.**

**Run them before any PR.** Cross-pod behaviour lives entirely there, and a green `check-all.sh` says nothing about it. `project-state.yaml` lists this as the third declared test command, so recorded evidence that omits it covers two suites out of three.

**A skip is not a pass.** The suite exits 0 with tests skipped, so a skipped test reads exactly like a passing one in the summary line. Run it as `./scripts/test-integration.sh -rs` and account for every skip. This is not hypothetical: `test_a_real_display_is_driven_through_the_pipe` was unpassable on every machine for three weeks after the 2026-08-18 change that defaulted the sidecar's `BIND_HOST` to loopback (correct for the shipping Kubernetes shape, wrong for a testcontainer with its own network namespace). Nobody saw it, because the test skips when the sidecar image is absent and CI never runs the suite at all — the skip that hid the breakage was also the reason nobody noticed. It surfaced only when a release stopped to ask why 35 tests were skipping.

**Build the sidecar image first**, or its integration tests skip:

```bash
docker buildx bake --file docker-bake.hcl nodriver-sidecar \
  --set nodriver-sidecar.platform=linux/amd64 --load
```

The bake target is multi-platform and the local `docker` driver refuses that ("Multi-platform build is not supported for the docker driver"), hence `--set ... platform` and `--load`. A bare `docker buildx bake nodriver-sidecar` exits non-zero having built nothing — and piping it through `tee` masks that exit code, which is how it looked like it had worked.

**Legitimate skips on a dev box** (they need credentials or tools this repo does not ship): the Redshift live tests (`OTS_REDSHIFT_PASSWORD`), the backup suites (`pg_dump`/`pg_restore`/`psql` on PATH), and one deliberate manual microbenchmark. Anything else is a test you have turned off by accident. When reporting results, state the pass count AND the remaining skips with their reasons — "integration green" on its own is not a report.

## Conventions

- Build backend: hatchling
- Linting: ruff, line-length 120, target py314
- Type checking: mypy, strict
- Testing: pytest
- uv only. No poetry.

## Test fakes

A test fake is any class named `Fake<Name>` or `_Fake<Name>` under a `tests/` directory. Every one declares what production protocol it stands in for. Three routes, in order of preference:

1. **Subclass it.** `class _FakeKv(KvBucketLike):`. The walker accepts any non-`object` base and checks nothing further, on the theory that a type checker covers it. **In this repo it does not:** mypy runs over `packages/*/src` only, so a subclassed fake's surface is unverified by anything. Prefer it anyway for a real Protocol, because the base documents intent and an IDE follows it. Reach for route 2 when you want the surface actually compared.
2. **`# parity-with: <fully.qualified.name>`** on the line above the class. The walker imports the target and compares method surfaces. This is the only route that verifies anything.
3. **`# parity-exempt: <rationale>`** on the line above the class, for a hand-rolled subset stub with no single production protocol to name. The rationale must be at least 30 characters and must not be a blanket phrase like "tests need this" or "temporary". **Keep it on one line, however long.** The walker reads the first non-blank line above the class and stops, so a wrapped rationale exempts nothing.

Workspace tests centralise their asyncpg and workspace-entity shells under `packages/agent/workspace/tests/_helpers/`, so per-test inline fakes need only a one-line subclass declaration.

**Exempt in place, not in `tests/enforcement/_fake_parity_exemptions.txt`.** That file still parses and is deliberately empty. Its entries are keyed `path:LINE:symbol`, so one added import shifts every fake below it and the gate fails with `no_declaration` for a fake nobody touched. A marker on the class moves with the class.

`tests/enforcement/test_fake_protocol_parity.py` enforces this. It is a thin shell over the canonical walker in `packages/enforcement/src/threetears/enforcement/fake_parity/`. `FAKE_PARITY_ENFORCEMENT_MODE` defaults to `strict`.

This catches the drift class where production protocols evolve while test fakes rot silently, until some downstream test happens to call the missing method.
