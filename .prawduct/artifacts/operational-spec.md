---
artifact: operational-spec
version: 1
depends_on:
  - artifact: nonfunctional-requirements
  - artifact: security-model
last_validated: null
---

# Operational Specification

## Summary

3tears is a library family. It deploys nothing. "Operations" has two halves:

1. What a host operator must provide and can rely on when embedding the packages (Postgres, NATS, optionally S3), and the recovery behaviour the code supports.
2. How this project releases itself and gates its own changes (lockstep versions, tag-driven PyPI release, CI shards and the testament pool).

No SLO, RTO or RPO is stated anywhere in the repo. None is invented here.

| Term | Meaning |
|---|---|
| Host | An application that embeds 3tears and runs pods |
| L1 / L2 / L3 | In-process cache, NATS KV cache, Postgres (or Yugabyte) store (`README.md`, `docs/integration-guide.md`) |
| Testament pool | Remote runners that execute the unit and integration suites (`.testament.toml`) |
| Lockstep | Every `3tears*` package at one version (`CLAUDE.md`) |

## Deployment

### What a host provides

| Dependency | Requirement | Source |
|---|---|---|
| Python | 3.14 or later | `CLAUDE.md` |
| PostgreSQL (L3) | Required. DSN, a schema, and the `vector` extension only if agent-memory is used. Tests use Postgres 16 with pgvector | `docs/integration-guide.md` Step 1; `docs/testing.md` |
| Migrations | One one-shot run per schema, not from every pod; idempotent, keyed `(version, package)`, serialised on the database DDL lock | `docs/integration-guide.md` Step 2 |
| NATS (L2) | Required for more than one pod. JetStream on, file storage so KV survives restarts. Tested against nats-server 2.12.6 and 2.14.2 | `docs/integration-guide.md` Step 3; `docs/testing.md` |
| One declarer per bucket | One process declares `{namespace}-collections`; all others bind with `create_if_missing=False` and die at wiring if it is missing | `docs/integration-guide.md` Step 3 |
| KV grants | The `{ns}-epochs` bucket: hub and gateway writable, agent pod, tool pod and registry read only. Missing grant does not raise; it shows as a timeout | `packages/epoch/README.md` |
| S3 | Needed only by object-store and backup users; S3 is the first driver, filesystem the second | `packages/backup/README.md` |
| Env vars | `THREETEARS_NATS_URL`, optional `THREETEARS_NATS_PROXY_TIMEOUT_MS`, `THREETEARS_LOG_LEVEL` | `docs/integration-guide.md` Step 4 |
| Shutdown window | At least the NATS drain timeout (about 30 s default) | `docs/integration-guide.md` Step 6 |

Shutdown order on SIGTERM: stop the invalidation listener, `nats.shutdown()`, then `pool.close()` (`docs/integration-guide.md` Step 6).

Pod count: one pod needs no NATS (L1 plus L3). More than one needs NATS and `start_invalidation_listener` on every pod, or pods serve stale L1 (`docs/integration-guide.md` Step 5).

### What the host owns, not this repo

Worker deployment, grants, alert rules, settings and DDL wiring for L2-first collections are the hub's (`docs/l2-first-operations.md`, "What the hub adds"). Container images, Kubernetes manifests and environments are not in this repo.

### The sidecar image

The nodriver scrape sidecar is not a workspace member and not on PyPI. It is AGPL-3.0, isolated, and ships as a container to GHCR on a tag push (`.github/workflows/release.yml`, job `sidecar-image`).

## Backup & Recovery

| Item | What the code gives | Source |
|---|---|---|
| Database backups | `3tears-backup`: client-side AES-256-GCM encrypted streams to an injected `ObjectStore`; Postgres and Yugabyte dump drivers, autodetected from `version()` | `packages/backup/README.md` |
| Retention | Grandfather-father-son: N daily, weekly, monthly; the rest pruned | `packages/backup/README.md` |
| Restore proof | Restore into a throwaway database, with an optional hook to run a stack against it | `packages/backup/README.md` |
| Config | Frozen `BackupConfig`; `from_env` reads `THREETEARS_BACKUP_*` | `packages/backup/README.md` |
| L2 loss | The L3 epoch log (`threetears_epoch_log`) restores the L2 change log and rows | `docs/l2-first-operations.md` |
| Epoch counters | Memory-backed KV by design; a lost counter is a lost coherence signal, not a lost fact. Durable epochs (tile family) keep a `config_epochs` row | `packages/epoch/README.md` |

Not in the repo: backup schedule, retention numbers, restore drill cadence, RTO and RPO. The host sets them.

A write that did not wait for L3 and had not reached it is lost when NATS loses data, until its source sends it again. A write that waited and returned is in L3 (`docs/l2-first-operations.md`).

## Monitoring

3tears emits signals; the host collects and alerts (`.prawduct/artifacts/observability-strategy.md`).

| Watch | Why | Source |
|---|---|---|
| `/healthz/live`, `/healthz/ready` | Restart versus rotation | `packages/observe/src/threetears/observe/health.py` |
| `collection_epoch_tables_behind` | Followers behind on a table | `docs/l2-first-operations.md` |
| `threetears_epoch_changes_fill_ratio` at 0.75 or above | Past 1.0 the log refuses records and writes needing one fail | `docs/l2-first-operations.md` |
| Increase in `threetears_l2_first_l3_refused_total` | L3 refused a committed non-waiting write | `docs/l2-first-operations.md` |
| Increase in `threetears_l2_first_records_missing_total` | Change records missing | `docs/l2-first-operations.md` |
| `threetears_l2_first_refills_total` and its WARNING | L2 was refilled from L3 | `docs/l2-first-operations.md` |
| `contained_errors_total{site, exc_type}` | Absorbed failures per site | `docs/exceptions.md` |
| `threetears_scheduled_jobs_failures_total`, `unrouted_kind` ERROR | Scheduled fires failing or unrouted | `packages/scheduled-jobs/README.md` |

## Failure Recovery

### Failure modes the code handles

| Failure | Behaviour | Source |
|---|---|---|
| NATS lame-duck (rolling restart) | Each `NatsClient` connection starts a make-before-break move to a successor on the lame-duck notice, retried every `REAUTH_RETRY_SECONDS`; proven live on a two-node 2.14.2 cluster with no lost request or message | `docs/nats-credential-renewal.md` |
| NATS credential expiry | User JWT TTL is a one-day backstop; the same handover renews about daily. Access is removed by a kick via the system account plus a refusing auth callout | `docs/nats-credential-renewal.md` |
| A TTL too short to hold the longest request | `unsafe_renewal_reason` names it each cycle; the hub refuses to start | `docs/nats-credential-renewal.md` |
| NATS restart loses JetStream state | The declaring `NatsClient` re-creates declared buckets, streams and durable consumers, empty, after every reconnect; raw `jetstream_context()` creations are not remembered | `docs/integration-guide.md` Step 3 |
| NATS loses data under L2-first tables | Pods keep serving what they hold; missing keys raise `L2LostError`; writes raise `L2LostError`; the worker's next pass refills from L3 | `docs/l2-first-operations.md` |
| Two L2-first workers at once | Safe, only wasteful | `docs/l2-first-operations.md` |
| L2-first writer dies mid-write | The saver applies the row from the epoch change record | `docs/l2-first-operations.md` |
| Lock held / KV error in a tick | `LockHeld` skips silently; `KvError` degrades open, with optimistic CAS as the guard | `packages/scheduled-jobs/README.md` |
| Process dies mid-fire | The row is left to the reaper (default 15-minute baseline, per-kind override) | `packages/scheduled-jobs/README.md` |
| Unrouted job kind | Refused with a metric and an ERROR; the occurrence is not claimed and survives | `packages/scheduled-jobs/README.md` |
| Startup order | Retry with exponential backoff for services that start before dependencies | `packages/observe/src/threetears/observe/resilience.py` |
| A periodic tick raises | Logged at WARNING; the loop continues | `packages/observe/src/threetears/observe/periodic.py` |
| Epoch counter lost | Consumers need `on_reset` and `catchup_tick`; a broker restart replaces the counter silently | `packages/epoch/README.md` |

Limits stated in the sources: a kick lost for good, or a partition from the server, leaves a credential valid until the backstop expiry (`docs/nats-credential-renewal.md`). A run whose order is its meaning must take a `PublishPin` to stay on one connection during a handover (same file).

### Runbooks that exist

| Runbook | Trigger | Source |
|---|---|---|
| Measure what a write generation costs | Before switching a busy table on; after any change to the epoch write path; before a release touching it | `docs/runbook-epoch-write-cost.md` (script `scripts/measure-epoch-write-cost.py`; no test lane runs it) |
| Running L2-first collections | Wiring, status read, after NATS loses data, change log filling, L3 refusal | `docs/l2-first-operations.md` |
| NATS credential renewal | Decision record, not an incident procedure | `docs/nats-credential-renewal.md` |
| Cutting a release | Every release | `docs/releasing.md` |
| Republishing a tagged version | A package missed the upload | `docs/releasing.md` |
| Testament pool down | `test.sh` exits 2, "testament gave no verdict" | `docs/testing.md`, "When the pool is down" |

## Release procedure

Source for the whole section: `docs/releasing.md` and `CLAUDE.md`.

1. `./scripts/bump-version.sh minor` for new public API, `patch` otherwise. It moves every lockstep location, moves `## Unreleased` in `CHANGELOG.md`, relocks, and refuses an untagged base, an existing tag or empty notes (`scripts/bump-version.sh`).
2. In the same PR: rename the `unreleased` section of `packages/evals/MIGRATIONS.toml`, and roll `packages/evals/tests/test_judge_identity_pins.py` forward.
3. PR into `develop`, then PR `develop` into `main` with no bump. Merges use a real merge commit, never squash or rebase, never force-push (`CLAUDE.md`).
4. Push a tag from `main`: `git tag -a vX.Y.Z <commit>` then `git push origin vX.Y.Z`. The tag push is the only trigger that creates the GitHub Release.
5. `release.yml` runs `tag-on-main` (the commit must be an ancestor of `main`), the unit shards, `check`, and `live-claude-cli` (needs secret `CLAUDE_CODE_OAUTH_TOKEN`; a missing token fails the job). Then `build` (lockstep verify, `scripts/verify-dist-complete.sh`), `pypi-publish` (the `pypi` environment gate, `skip-existing`), `npm-publish` (only when variable `NPM_PUBLISH` is `true`), `github-release` (tag pushes only) and `sidecar-image`.
6. Confirm with `git ls-remote --tags origin` and that `github-release` did not skip. A green run alone proves nothing.

New package in a release: register its PyPI pending trusted publisher first. New web package: the first publish uses a short-lived `NPM_TOKEN`, then add the npm trusted publisher and delete the token.

Do not run `gh workflow run release.yml` to cut a release. It is the republish command and leaves no tag (0.22.5 shipped that way on 2026-08-01).

### Republishing a tagged version

1. Land the fix on `main` through a hotfix branch.
2. Merge it to `develop` before dispatching; `workflow_dispatch` is offered only for workflow files on the default branch (`develop`).
3. `gh workflow run release.yml --ref main -f version=X.Y.Z`.
4. Approve the `pypi` environment.

The tag does not move and the version is not bumped (`docs/releasing.md`; v0.18.0 shipped 26 of 27 packages).

## CI and test gates

| Gate | What runs | Source |
|---|---|---|
| `check` (required status) | Attribution check, lint (`ruff`), typecheck (`./scripts/typecheck.sh`), OTel floor, MCP 1.x suite, plus the unit verdict | `.github/workflows/ci.yml` |
| Unit route | Testament pool when the key, event and a probe allow it; otherwise eleven GitHub shards (`evals-1`..`evals-8`, `core`, `tests`, `rest`). A pool lost mid-run falls back to the shards | `.github/workflows/ci.yml`, `docs/testing.md` |
| `loop_timing` tests | Serial on the job's own runner, never on the pool | `.github/workflows/ci.yml`, `.testament.toml` |
| `integration` job | core, nats, epoch, scheduled-jobs, agent/acl against Postgres (pgvector pg16) and NATS 2.11 with JetStream | `.github/workflows/ci.yml` |
| Doc-only changes | Skip lint, typecheck and the suites | `scripts/ci-scope.sh` |
| Concurrency | A newer run of the same ref cancels the older | `.github/workflows/ci.yml` |

Local scripts (`CLAUDE.md`): `./scripts/test.sh`, `./scripts/test-integration.sh`, `./scripts/lint.sh`, `./scripts/typecheck.sh`, `./scripts/check-all.sh`. Never raw `pytest`, `ruff` or `mypy`.

Testament pool (`.testament.toml`, `docs/testing.md`): manifest version 3, `fallback = "pool-only"`, batch timeout 1800 s, pool job limit 75 minutes. The image holds Postgres 16 with pgvector and nats-server 2.14.2 and 2.12.6. When the pool cannot take a run, `test.sh` exits 2. Stop and tell the person; do not rerun with `TESTAMENT=off` except for named test files. Integration with the pool down is one package at a time through testcontainers, and an agent asks first (`CLAUDE.md`).

`docker` and `secrets` tests never reach the pool. `./scripts/test-integration.sh --here -rs` runs them when a change touches scrape, search, the lame-duck handover or a credentialed package (`docs/testing.md`). A skip is not a pass: run with `-rs` and account for every skip.

## Direction

Ratified 2026-11-26 from declarations the owner already made; each Why cites where.

- **Nothing this project produces credits an AI agent, on any surface:** commit messages and
  metadata, PR titles and bodies, review and issue comments, code comments and docstrings,
  CHANGELOG entries, docs, release notes and tag annotations. Naming Claude Code as a tool the
  project uses is fine; the rule is about authorship credit. Already-landed attribution stays
  landed: history is never rewritten to scrub it.
  Why: coding agents ship with a default that adds attribution, so it appears unless removed
  every time. `CLAUDE.md` § No agent attribution: "This is not a style preference. It is not a
  default you may weigh against others."
- **Every PR merges with a real merge commit (`gh pr merge --merge`), feature-to-develop and
  develop-to-main alike; never `--squash`, never `--rebase`.** Cleaner history is made on the
  branch before the merge, never by the merge step.
  Why: squashing collapses commit history and can silently drop or corrupt file content relative
  to what the branch contained, with no diff-review step catching it. `CLAUDE.md` § Git and PR
  workflow.
- **Never force-push: no `--force`, no `--force-with-lease`.** Restructure with a new branch and
  a new PR.
  Why: a pushed branch is shared (a PR branch here takes commits from more than one session), and
  rewriting it discards what others built on it. `CLAUDE.md` § Git and PR workflow: "Never
  force-push."; `scripts/check-attribution.sh`: "never force-push a shared branch".
- **Sessions push work branches and never merge a PR; the maintainer merges.**
  Why: owner 2026-11-22, "yes push, but we cannot merge PR's." (`.prawduct/owner-rulings.md`).
- **A release is cut only by pushing an annotated tag from `main`; `gh workflow run release.yml`
  is the republish command and never cuts a release.** A green release run is not a release;
  `git ls-remote --tags origin` confirms one.
  Why: the tag push is the only thing that creates the GitHub Release; the workflow run publishes
  to PyPI with no tag and no release. `CLAUDE.md` § Releasing; `docs/releasing.md`.
- **Medium-or-larger work goes on a feature branch, and merge order respects PR stacking.**
  Why: `CLAUDE.md` § Git and PR workflow; `develop` and `main` are protected
  (`project-preferences.md` § Workflow).
- **Tests, lint and typecheck run through `./scripts/` only; never pytest, ruff or mypy
  directly.**
  Why: the scripts carry the testament pool handoff, the CI selection and the suites
  `check-all.sh` adds; a direct run skips all three. `CLAUDE.md` § Scripts: "Always use the
  scripts."
- **When the testament pool can't take a run (`test.sh` exits 2, "testament gave no verdict"),
  stop and tell the person.** Never rerun a unit-suite package or the whole unit suite with
  `TESTAMENT=off`, never run `ci-unit-tests.sh` or raw `pytest` to get past it, and never edit
  `.testament.toml`. Named test files under `TESTAMENT=off` are fine; the integration suite's
  one-package-at-a-time fallback is the person's to run, and an agent asks first.
  Why: the manifest is `fallback = "pool-only"`, so a pool that can't take the run fails it rather
  than running it on this machine. `CLAUDE.md` § Scripts; owner 2026-11-22, "everything that can run on
  testament should run on testament" (`.prawduct/owner-rulings.md`); `docs/testing.md` § When the
  pool is down.
- **Before any PR, the integration suite runs as `./scripts/test-integration.sh -rs`;** a change
  touching scrape, search, the nats lame-duck handover or a credentialed package also runs
  `./scripts/test-integration.sh --here -rs`. The report gives the pass count and each skip with
  its reason.
  Why: CI's `integration` job covers only core, nats, epoch, scheduled-jobs and agent/acl, and
  the `docker` and `secrets` tests never reach the pool. `CLAUDE.md` § Scripts: "Neither is
  optional."
- **The change record is `.prawduct/changes/`, one fragment per scope; there is no
  `.prawduct/change-log.md`.**
  Why: owner 2026-11-22, "Please also make sure change logs are the modern fragment style."
  (`.prawduct/owner-rulings.md`).

- **Every integration test runs on the pool or in `--here`, and the two lists say exactly which.**
  Why: a test in neither runs nowhere, and both the pool and the laptop go green without it. Gate:
  `tests/enforcement/test_integration_suite_selection.py`. Owner 2026-11-26 bulk ratification.
- **A test that drives a Docker daemon carries the `docker` marker, and a test that reads a
  credential from the environment carries the `secrets` marker.**
  Why: the pool has no Docker daemon and ships no credentials, and its suites deselect both
  markers; an unmarked test would run there and fail or skip. Gates:
  `tests/enforcement/test_docker_marker.py`, `tests/enforcement/test_secrets_marker.py`. Owner
  2026-11-26 bulk ratification.
- **The CI unit shards together run the whole suite, and `check` waits for every one.**
  Why: a shard missing from a matrix is a slice of the suite that silently never runs, with every
  job green. Gate: `tests/enforcement/test_ci_unit_shards.py`. Owner 2026-11-26 bulk ratification.
- **Wall-clock tests carry the `loop_timing` marker and run serially.**
  Why: they assert the event loop's responsiveness by wall clock, so they run apart and serially
  (the marker's definition in the root `pyproject.toml`); the pool leaves them out
  (`CLAUDE.md` § Scripts). Gates: `tests/enforcement/test_ci_unit_shards.py`,
  `scripts/ci-unit-tests.sh`. Owner 2026-11-26 bulk ratification.

## Candidate norms (unratified)

| Rule the code enforces | Enforcing file |
|---|---|
| Every intra-family dependency carries `>=X.Y.0,<X.(Y+1).0` | `tests/enforcement/test_intra_family_version_bounds.py` |
| API growth ships in a MINOR bump | `tests/enforcement/test_api_growth_requires_a_minor_bump.py` |
| No agent attribution in commits | `tests/enforcement/test_no_agent_attribution.py`, `scripts/check-attribution.sh` |
| Every integration test is in the suite or the `--here` list | `tests/enforcement/test_integration_suite_selection.py` |
| `docker` and `secrets` marks are applied to the tests that need them | `tests/enforcement/test_docker_marker.py`, `tests/enforcement/test_secrets_marker.py` |
| CI `check` verdict matches each unit-route outcome | `tests/enforcement/test_ci_unit_shards.py` |
| A KV bucket grant names the bucket the client opens | `tests/enforcement/test_kv_bucket_grant_naming.py` |
| A leading underscore is not reached from outside its module | `tests/enforcement/test_underscore_access.py` |
| Test fakes declare the protocol they stand in for | `tests/enforcement/test_fake_protocol_parity.py` |
| The OTel floor step stays in CI | `tests/enforcement/test_otel_floor.py` |

## Open questions

1. No availability, latency, RTO or RPO targets exist in the sources. Does the maintainer want any stated for hosts, or does each host own them?
2. Supported version ranges for Postgres and NATS are not stated as a support policy. Sources give only what tests run against (Postgres 16; NATS 2.11 in CI, 2.12.6 and 2.14.2 on the pool). Is that the supported range?
3. The Yugabyte version the L3 targets is not stated in the docs read.
4. Whether `docs/runbook-epoch-write-cost.md` and the others meet the prawduct runbook standard (`/prawduct:runbook`) was not assessed.
5. Backup restore drills: no cadence or evidence of one in the repo.
6. S3 and object-store requirements (versions, IAM) were not read beyond the backup README.
7. `CHANGELOG.md` and `docker-bake.hcl` handling in a release were not read; only the bump script's description is cited.
