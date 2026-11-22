# Project Preferences

Developer preferences for how code is written in this project. Captured during discovery, updated as preferences evolve. Every session should read this before writing code.

## Language & Runtime

- **Language**: Python (the `web/` browser packages are JavaScript, published to npm as `@3tears/web-*`)
- **Version**: 3.14+ (`.python-version`, ruff `target-version = "py314"`)
- **Package manager**: uv workspaces (`uv sync`); never poetry. Build backend: hatchling

## Code Style

- **Naming**: pep8-naming (ruff `N`), with two house exceptions recorded in the root `pyproject.toml`; exceptions are named for the condition (`LeaseTimeout`, `AccessDenied`), never suffixed `Error` (N818 off)
- **Formatting**: ruff format, line length 120
- **Linting**: ruff, selecting `SLF`, `F`, `N`, `TC004`; run through `./scripts/lint.sh`, never ruff directly
- **Type annotations**: required; mypy strict, run through `./scripts/typecheck.sh`
- **Imports**: absolute under the `threetears.*` namespace; `threetears/` and `threetears/agent/` never get an `__init__.py` (implicit namespace packages), only leaf packages do
- **Underscores**: a leading underscore is a stability contract: nothing outside the defining module binds to it, tests included; no public API exists only for a test (CLAUDE.md, owner ruling 2026-10-01)

## Testing

- **Framework**: pytest, through `./scripts/test.sh` only (never raw pytest); suites run on the testament pool (`.testament.toml`)
- **Style**: tests are contracts; a test fake (`Fake<Name>`) declares the production protocol it stands in for (`tests/enforcement/test_fake_protocol_parity.py`)
- **Coverage expectations**: behaviour through the front door (constructor, fixture, exposed behaviour); repo-wide enforcement tests under `tests/enforcement/`
- **Testing strategies**: property-based (hypothesis); integration against real Postgres, NATS and S3 (`./scripts/test-integration.sh`)
- **Test location**: each package's own `tests/`; repo-wide rules in `tests/enforcement/`
- **Parallelization**: testament distributes on the pool; markers `integration`, `docker`, `secrets`, `browser`, `live`, `loop_timing` select what runs where (root `pyproject.toml`)

## Architecture Patterns

- **Data modeling**: three-tier entities and collections (L1 SQLite, L2 NATS KV, L3 PostgreSQL or a git working tree); Pydantic and dataclasses for value types
- **Error handling**: a failure reaches the caller or is recorded where an operator sees it; broad handlers re-raise or call `threetears.observe.contain` (`docs/exceptions.md`, `tests/enforcement/test_no_silent_swallow.py`)
- **Async**: async/await throughout; `CancelledError` always re-raises, tasks stop with `cancel_and_wait`
- **File organization**: one package per directory under `packages/` (the agent family under `packages/agent/`), each with its own `pyproject.toml`, `src/` and `tests/`

## Tooling

- **Key libraries**: nats-py (third-party internals confined to one module, `nats/_nats_py_internals.py`), testament (test pool), hypothesis
- **Dev commands**: `./scripts/test.sh <package>`, `./scripts/lint.sh`, `./scripts/typecheck.sh`, `./scripts/check-all.sh`, `./scripts/test-integration.sh -rs`
- **Versioning**: the whole `3tears*` family releases in lockstep at one version; intra-family bounds `>=X.Y.0,<X.Y+1.0` (`docs/releasing.md`)

## Workflow

- **Branching**: feature-branches (default: feature-branches — create a branch for medium+ work, direct commits to protected branches only for trivial fixes; set to "direct" for solo projects where committing to main is OK)
- **Protected branches**: main, develop (branches that should not receive direct commits unless branching is "direct")
- **PR creation**: wait_for_user — open a PR only when asked
- **PR merge**: never from a session (owner 2026-11-22: "we cannot merge PR's"); the maintainer merges
- **PR merge strategy**: merge commit (default: merge commit — `gh pr merge --merge`; preserves each commit's identity so a reused branch's merge-base stays correct and the review/PR gates don't re-review already-merged work; set to "squash" for one linear commit per PR, or "rebase" — with either, branches are single-use: delete after merge and never reuse, because the rewritten history strands a reused branch's merge-base)
- **Branch cleanup**: propose (default: propose — when merged branches or dead worktrees accumulate, the session start proposes removing them, each with the proof that nothing is lost, and the agent removes them on your yes; `prawduct-hook git-tidy` shows the list any time. Set to "off" to silence the proposal and keep a branch after its PR merges, except under a squash or rebase merge strategy, where deleting it is required)
- **Long-lived branches**: (unset — branches never proposed for deletion, beyond `main`, `master`, `develop`, `trunk`, `release/*`, the remote's default branch and the configured base. Name any environment branch kept level with one of those, such as `staging`, comma-separated, because a branch at the same commit as `main` otherwise reads as merged)
- **Commit attribution**: none (default: none — no `Co-Authored-By`, `Signed-off-by`, or "Generated with …" trailers on commits or PR bodies; set to "co-authored" to add a Claude `Co-Authored-By` trailer)
- **Upstream filing**: ask-user (default: ask-user — governs `/prawduct:report-bug`, which files a bug about *prawduct itself* into prawduct's own public tracker. Under ask-user you are shown the exact outbound bytes and nothing is sent without your per-report approval of them; set to "never-file" for a standing no, which the adapter refuses on mechanically and without exception; set to "always-file" for standing pre-consent, which stops the asking — and then the drafting step is the only thing between your session's words and a public issue in someone else's repository)
- **Stats contribution**: (unset — reads as `never`, so nothing is sent. Governs `prawduct-hook contribute`, which sends anonymous governance stats to prawduct's collector to help improve prawduct. A report carries only numbers and fixed-list values from a public allowlist: no names, paths, branches, text or identifiers. Write `never` to decline, `ask` to approve the exact bytes of every send, or `always` for standing consent. `/prawduct:janitor` asks once while this is unset. `prawduct-hook contribute` previews what would be sent, and sends nothing.)
- **Delegation**: on, by default when it finishes sooner without collision. Delegates do light testing only. When several delegates run at once, a dedicated integrator subagent maintains an integration branch and merges their work into it; the main session judges when that is more efficient than integrating itself (owner 2026-11-22)
- **Delegate verification**: light: the test files the delegate's change touches, named explicitly (`./scripts/test.sh <package> <test file>`), run serially; never `-n`, a whole package, the whole suite, loops or load repros. Heavy is everything else: package suites, the unit suite, the integration suite, typecheck of all packages; those run once, on the merged result, on the testament pool
- **Inner-loop verification**: everything that can run on testament runs on testament (owner 2026-11-22): `./scripts/test.sh <package>` (optionally `-k`) while iterating, which hands off to the pool; only named test files or env-gated runs go local. The `docker` and `secrets` markers never reach the pool and run with `./scripts/test-integration.sh --here -rs` when a change touches them

---

**What belongs here**: How you want code written. Conventions, tools, style preferences, workflow preferences.

**What doesn't belong here**: What to build (product-brief), system design (data-model, architecture), performance targets (nonfunctional-requirements), or deployment (operational-spec).

## Enforcement

Each preference above should be enforced by one of three mechanisms — assign the mechanism when you add the preference so it doesn't quietly become aspirational.

| Mechanism | Where it lives | What it catches | Trade-off |
|---|---|---|---|
| **Linter** | Project's configured linter (ruff, eslint, swiftlint, etc.) | Mechanical style/naming rules | Best tool when configured. If no linter, preferences in this category fall through to Critic. |
| **Test** | `tests/preferences/test_*.py` (or equivalent) | Structural rules with named exceptions (AST checks, config-presence checks) | Bakes the rule into CI; refuses to be silent. Cost: re-validate when the rule's shape changes. |
| **Critic** | `/prawduct:critic` review (Goal 3: norm departures) | Judgment-required rules (semantic naming, "appropriate" anything, what counts as a "boundary") | No false-confidence test. Cost: requires reviewer per chunk; misses violations between reviews. |

This per-preference table is the product's **norm index** (`/prawduct:methodology norms`): each row assigns a norm its **mechanism** (linter / test / Critic) and its **audit home** — `janitor` (only the deep sweep sees it) or `advisory` (a mechanical probe fires on it). A row may be a **pointer** to a `## Direction` section instead of restating the norm, and every norm carries its **why** (a whyless norm is unenforceable at its edges).

**Every populated row here *is* a homed norm** — the `norm-health-sweep-overdue` advisory reads
these rows to decide whether this product has norms worth auditing, so never leave an example or
placeholder row in the table: it would claim a norm registry that has not been ratified. Two row
shapes go in — an ordinary row naming the convention, its mechanism, its enforcement artifact, its
audit home (`janitor` or `advisory`) and its why; and a **pointer** row whose first cell reads
`norm lives in <artifact> § Direction`, whose enforcement artifact is `—`, and whose why lives in
the Direction entry it points at.

| Preference / norm | Mechanism | Enforcement artifact | Audit home | Why |
|---|---|---|---|---|
| Delegation policy: see § Workflow `Delegation` and `Delegate verification` | Critic | — | janitor | "yes light testing for delegates, and if several delegates are running a dedicated integrator subagent should be used" (owner 2026-11-22) |
| Inner-loop verification: see § Workflow `Inner-loop verification` | Critic | — | janitor | "everything that can run on testament should run on testament" (owner 2026-11-22) |

**A filled `Delegation` / `Delegate verification` / `Inner-loop verification` row states a norm, and each takes `Critic`** — a policy stated in prose is judgment-required by construction, so no linter or test can grade it; audit home `janitor`, and the why is the sentence the owner gave for it. One row covers the policy `Delegation` and `Delegate verification` state together; `Inner-loop verification` governs your own loop rather than a delegate's, so it gets its own row. The row is written when the policy is **ratified** — the owner confirms it, and `/prawduct:doctor` proposes a starting point for all three — never shipped here, because this table ships empty.

**Rule for adding a new preference:** assign a mechanism. If the preference can be expressed as "every file/function/config matches pattern X with named exceptions" → write a test. If a linter rule already exists for it → configure the linter. If it requires understanding intent → assign to Critic. Never leave a preference unassigned.

**False-confidence guardrail:** if a generated test would pass on conforming code but couldn't reliably catch a real violation (e.g., greppy heuristics for semantic rules), prefer Critic over a weak test. A green test that doesn't actually check the rule is worse than no test.
