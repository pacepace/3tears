# Full collection support — shard sequence

**Goal:** a tool pod holds a `BaseCollection` on the tiers it needs — L1 and L2
mandatory, L3 conditional — on the same substrate an agent pod uses, and cannot
reach any other principal's data.

Today a non-agent pod gets L1 (pod-local SQLite, no grant involved, works for
any principal) and neither of the other two.

Facts, sites and ratified decisions live in
`14-eng-ai-bot/docs/collection-support-evidence.md`. Shards cite
it rather than restating; where both carry a fact, the ledger wins.

This file is a sequence, not a task. It has no shard number.

---

## Order

The `NN` suffix is execution order.

| shard | repos | what it lands |
|---|---|---|
| `coll-task-01-invalidation-lifecycle` | 3tears + SDK | subscription handle, `stop_invalidation_listener`, idempotent start |
| `coll-task-02-hub-consumes-invalidation` | hub | hub processes subscribe and tear down |
| `coll-task-03-scope-substrate` | 3tears | the scope segment in `l2_key`, the scope helper, wiring-time validation, **L2 eviction on invalidation** |
| `coll-task-04a-bucket-config-reconcile` | 3tears + hub | `allow_direct: true` via a real reconcile primitive, and the reconnect self-heal |
| `coll-task-04b-delete-nats-kv-client` | 3tears + SDK | deletes the retired `NatsKvClient`, including a patch target the SDK's shared conftest names by dotted path |
| `coll-task-05a-grant-shape` | 3tears | the minted grant: scoped `$KV.` publish, `$KV.` off subscribe, the four JetStream bypasses |
| `coll-task-05b-static-user-grants` | hub | the nine static NATS users — the larger half of the isolation |
| `coll-task-06a-consumer-wiring-3tears` | 3tears | two registries in the registry server; the wiring enforcement rule |
| `coll-task-06b-consumer-wiring-hub` | hub | every hub registry site; the identity-fence cutover |
| `coll-task-06c-consumer-wiring-sdk` | SDK | agent pod and devx runtime, scoped by `agent_id` |
| `coll-task-07a-stack-consolidation` | 3tears + SDK | delete the duplicated ACL wiring, call the existing shared subscriber |
| `coll-task-07c-tool-pod-grant` | 3tears | the tool-pod grant, proven by a refusal probe |

There is no `-07b`. It carried `tool_namespaces` into the callout claims so a
tool pod could derive a scope — unnecessary once the scope is `tool_pods.id`,
which is already `claims.sub`. The genuine bug it found (`tool_namespaces` is
never passed, so tool-pod HITL and pipe grants are empty tuples in production)
belongs to `build-plan-principal-convergence.md` Chunk 11, which has the better
analysis: `allowed_namespaces` holds prefixes while `hitl_forward_family` needs
full names, so wiring the existing parameter cannot work.

`design-l3-for-non-agent-principals.md` is a design doc, not a shard. Build it
only if a pod needs durable state.

---

## Why this order

**`-01` before `-02`.** `start_invalidation_listener` discards the
`Subscription` it is handed and there is no stop method, so "unsubscribe on
shutdown" is not expressible in the hub repo. `-02` is therefore not a hub-only
landing.

**`-03` before `-04` before `-05a`.** The key prefix is inert until the grant
narrows; the grant is unenforceable on reads until the bucket runs
`allow_direct: true`, because with it false the key rides in the request body
where no subject permission can see it.

**`-06x` after those.** They supply the scope every process needs.

**`-07x` last, in order.** `-07a` changes no grant *narrowing*, and is reviewable
on its own — though it carries two stated behaviour changes (CON-04, CON-05) and
depends on `-05a` GRANT-13 for the deadletter grant `subscribe_typed` needs. `-07c` grants the bucket, which must come after the grant
surface is safe.

`-01`/`-02` are separable from the rest: the invalidation gap is a live
multi-replica staleness bug on its own. Everything from `-03` on is one landing;
each piece alone is inert or breaking.

---

## Landing mechanics

Three repos against a PyPI-pinned dependency. **Step 2 is now landed** — the
paragraph below described the pre-step-2 state and read as if it still held.

Verified state at this commit: all three consumers pin `3tears*==0.27.0` and
resolve the whole family from the local checkout, via ~30 per-package
`[tool.uv.sources]` entries pointing through a gitignored `.3tears` symlink in
each repo. `3tears-search==0.27.0` rides in `constraint-dependencies`. The hub
depends on the SDK and admin as **editable path sources**, so all three resolve
together. `0.27.0` is unreleased — no `v0.27.0` tag exists on `3tears` locally
or on `origin`. Steps 3 through 5 are outstanding.

**Three symlink roots, not one.** uv canonicalizes each shared editable source
to whichever repo's `.3tears` it resolved through, so the hub's own `uv.lock`
names all three. At this commit:

| root | packages resolved through it |
|---|---|
| `14-eng-ai-bot-agent-admin/.3tears` | `3tears` (core), `-agent-acl`, `-epoch`, `-langgraph`, `-mcp`, `-models`, `-nats`, `-observe` |
| `14-eng-ai-bot-agents/.3tears` | `-agent-audit`, `-agent-knowledge`, `-agent-memory`, `-agent-tools`, `-agent-workspace`, `-channels`, `-conversations`, `-datasources`, `-iam`, `-media-contracts`, `-registry` |
| `14-eng-ai-bot/.3tears` (the hub's own) | `-enforcement`, `-geo`, `-object-store`, `-scheduled-jobs`, `-search` |

Two consequences. **`uv sync` in any one repo needs all three siblings present
and all three symlinks created** — `scripts/link-3tears.sh` only ever links its
own repo root, so it must be run in each. And **nothing requires the three to
point at the same checkout**: aim the hub's at one 3tears worktree and the SDK's
at another and you install a silently mixed family, the exact failure
`14-eng-ai-bot/CLAUDE.md` opens by forbidding, with no error at install time.

There is also **no producer-side matrix gate** — `matrix-fan-out.yml` was never
committed and the `override-matrix-gate` label is read by no workflow. Consumer
CI checks out 3tears by the version in its own `uv.lock` as a tag (`v${VERSION}`),
so it is blind to unreleased 3tears work — and while the block is in at an
unreleased version, that checkout step has no tag to resolve.

1. 3tears feature branch. Pick the target version now; use it everywhere.
2. **In hub + SDK + admin, in one commit each:** bump every `3tears*` pin *and*
   every `constraint-dependencies` entry to the target version, and add the
   per-package `[tool.uv.sources]` path entries. Do not add the override without
   the bump — a path source pointing at 0.27.0 fails against a `==0.26.1`
   requirement. Do not leave admin behind; it participates in the hub's
   resolution.
   The override is ~30 per-package entries per repo, not one line, and **there
   is no generator for them**. `scripts/link-3tears.sh` only runs `ln -sfn` to
   create the repo's own `.3tears` symlink; it writes no `pyproject.toml`
   entries and it touches no sibling repo, so run it in each of the three.
   Neither it nor the hub `ci.yaml` block is vestigial: the committed
   `[tool.uv.sources]` paths resolve through `.3tears`, and ci.yaml's "Link
   .3tears in every aibots repo to the 3tears checkout" step exists precisely
   because the lock canonicalizes the shared sources across all three roots.
   Recover the entries from `git show 540fcfcd -- pyproject.toml` in
   the hub, which added 29 — **it omits `3tears-search`**, the one every consumer pins via
   `constraint-dependencies`, so add that by hand or you ship the mixed family
   this section warns about.
3. 3tears: bump the family in lockstep with `scripts/bump-version.sh`, merge to
   develop, then main, tag, publish.
4. Consumers: `uv lock`, commit the lock.
5. **Remove the sources block** and re-verify the build resolves from PyPI alone
   (`uv export --locked`). This is the step most likely to be skipped, and
   skipping it ships a repo that builds only on a machine with three sibling
   checkouts, each symlinked. Note that `--locked` is only a valid check *after*
   the block is gone: while it is in, `--locked` re-resolves through the path
   deps and fails naming innocent packages, which is why the Dockerfile uses
   `--frozen`.
6. Merge consumers. Fire `matrix-nightly.yml` by hand — nothing runs on a
   schedule.

`14-eng-ai-bot/CLAUDE.md`'s "3tears Library Dependency" section has been
reconciled to this: it now states the PyPI default, flags that the branch is in
the opt-in state, and requires all three repos' symlinks to be aimed at one
checkout. Whoever lands step 5 removes that branch-state flag in the same
commit.

---

## Relationship to `build-plan-principal-convergence.md`

That plan is active and its Chunks 11-13 cover this ground. **These shards
govern**, and the plan is reconciled to them — with two exceptions where the plan
is right on evidence and the shards absorb its finding:

- **Chunk 11's HITL analysis.** `allowed_namespaces` holds name prefixes while
  `hitl_forward_family` digests full registered namespace names, unknown at
  connect. "Wire the existing parameter" cannot work. Chunk 11 keeps that problem.
- **Chunk 13's `ns_<hex>` schema derivation** and its write-side isolation bar,
  which are further developed than the L3 design doc's sketch.

One apparent conflict is not one: Chunk 11's *"not a per-row opt-in"* is about
**which principals** get collections and invalidation (answer: all of them), while
`coll-task-05a`'s per-bucket opt-in is about **which bucket** gets a scoped
grant (answer: collections only, because nothing else writes a prefix). Different
axes.

**The dead-`Principal` resolution moved to `coll-task-05a` GRANT-11.** Chunk 11
also claims it, but it sits behind ten unbuilt chunks and both `coll-task-05b`
and `-06b` block on it, so `-05a` takes it — that shard already owns
`subject_permissions.py`. Chunk 11 is annotated accordingly.

Playbook: `14-eng-ai-bot/docs/runbook-matrix-gate.md`.

---

## Decisions

All ratified decisions live in the ledger's Part 4, and all open questions in
its Part 5. Two are worth repeating here because they shape the whole set:

**There is no SHARED tier.** Every key is scoped to one principal. A shared tier
was designed and dropped — see the ledger for why. Anything that proposes
reintroducing it must first answer the three problems recorded there.

**`$KV.` is publish-only.** Read authority is `$JS.API.DIRECT.GET`.

---

## Out of scope, recorded so the residual is not read as closed

The intra-bucket-isolation residual `-03` quotes names **three** shared buckets.
This landing closes one.

- **`checkpoints`** — granted to every agent pod, keyed by thread id via its own
  separate `l2_key`, holding full conversation state, cross-agent and
  cross-customer.
- **`{ns}_agent_config`** — granted to every agent pod, keyed by `agent_id`, so
  any agent pod can read or overwrite any other agent's config. It is also the
  one bucket anything actually watches.

`-05` must keep both at `>` rather than narrowing them, or every read on them
dies as a broker timeout.

### The enforcement blind spot `-07c` found — now closed

`threetears.enforcement.common.find_local_src_roots` used to walk `packages/*/src`
only. On this repo that silently omitted the ten NESTED `packages/agent/*`
packages — `acl`, `audit`, `identity`, `intention`, `knowledge`, `memory`,
`skills`, `tools`, `wake`, `workspace` — **233 of 702 python files, a third of the
src surface**. Every walker built on the helper reported a clean tree over that
third: `test_kv_grant_capability`, `test_l2_scope_wiring`,
`test_cache_primitive_usage`, `test_no_bespoke_reuse`, `test_underscore_access`,
`test_kv_bucket_open_discipline`, `test_l2_scope_discipline`. A walker that scans
nothing reports exactly what a walker that finds nothing reports, which is why it
survived.

The helper now recurses to ANY depth under `packages/`, skipping
`SKIP_DIRS` + dot-directories (so `packages/registry/.mypy_cache/3.14/src` is not
mistaken for a package) and never descending into a `src/` tree it has already
claimed. Depth is unbounded on purpose: the `agent/` grouping is a naming choice,
and encoding a guess about it is what produced the hole. `discover_src_roots` was
NOT affected — it expands `[tool.uv.workspace].members`, which already listed
`packages/agent/*`.

**The guard that matters is the non-vacuity one.**
`packages/enforcement/tests/common/test_repo_layout.py` now asserts against the
LIVE repo that the result is non-empty, is at least of the expected order of
magnitude, covers every `[tool.uv.workspace]` member carrying a `src/` (read from
the pyproject independently, so a package added tomorrow is required
automatically), and covers the whole nested `agent/` family. A regression that
returns nothing fails loudly instead of reading green.

What the widening surfaced, and how each landed:

| Finding | Resolution |
| --- | --- |
| 8 × `cache.missing_collection` (`identity_versions`, `intentions`, `memory_consolidations`, `agent_skills`, `agent_skill_invocations`, `agent_wake_schedules`, `wake_fires`, `webhook_subscriptions`) | Every one already had a real Collection; the per-repo `collection_table_allowlist` simply never named them because no migration under `packages/agent/` had ever been opened. Eight allowlist entries, no new classes, no exemptions. |
| 1 × `cache.pool_access` on `memory_chunks` (`agent/memory/tools.py`) | Fixed: new `MemoryChunkCollection.find_by_chunk_indexes`; the tool routes through it. The inline SQL had also dropped `customer_id` from the auth triple. |
| 6 × `cache.pool_access` in `agent/wake` (second order — naming the wake tables is what turned the pool-access walker on for them) | `webhook_adapter` now routes through `WakeFireCollection.count_in_window` (extended with a `webhook_subscription_id` narrowing); the three `rate_limit.py` aggregate/JOIN counts carry `# cache-bypass:` with specific reasons, matching what the wake Collections already do for the same shapes. |
| 4 × `underscore_access.E` in `agent/wake` | Promoted: `_check_rate_limit` → `check_rate_limit`, `_check_active_schedule_cap` → `check_active_schedule_cap`. Both were already re-exported from the package `__all__`, so public was the truthful reading. No alias. |
| 2 × `reuse.*` in `agent/tools` (not predicted) | `McpClient` held a raw `httpx.AsyncClient` → now owns a `TracedHttpClient` (`max_attempts=1`, because `tools/call` is not idempotent). `ToolServer` is a false positive — a tool REGISTRY dict plus a heartbeat loop, nothing buffered or flushed — and carries an exemption with that rationale. |

The two "stale fixtures" the earlier attempt predicted did not appear: they were
stale only against an implementation that required a `pyproject.toml` beside
`src/`. The landed helper keeps the original contract (a `src/` directory is
enough), so `test_packages_monorepo` and `test_mixed_layout` still hold.

`-07c`'s own gate composed the nested roots itself while the helper was narrow;
that local workaround is retired and it now calls the shared helper directly. Its
non-vacuity test still names `agent/tools/bootstrap.py` explicitly.

**Still explicitly two-level, by choice:** `dependency_alignment`'s
`package_globs`, `test_runtime_version_is_not_hardcoded._PACKAGE_GLOBS`, and
`fake_parity`'s tests-dir resolver each enumerate `packages/*` and
`packages/agent/*` in the open. They cover today's layout and a reader can SEE
the depth they assume, which is the opposite of the failure above; a third
grouping level would need them updated.

---

## Before writing code

Every shard gets an adversarial review first, and the review must demand a
**live probe** for each security claim rather than a reading of the gate.

Successive rounds of independent review over this set have each found bypasses
of the same class as the one the work started from, plus claims whose cited
evidence did not hold — including counts stated with confidence and no source. Reasoning over a partial read is the recurring failure mode here. Running
the thing is what catches it, and a probe that is collected but skipped is not a
probe — the 3tears integration suite does not run under `check-all.sh`.
