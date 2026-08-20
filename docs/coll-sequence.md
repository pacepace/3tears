# Full collection support — shard sequence

**Goal:** a tool pod holds a `BaseCollection` on the tiers it needs — L1 and L2
mandatory, L3 conditional — on the same substrate an agent pod uses, and cannot
reach any other principal's data.

Today a non-agent pod gets L1 (pod-local SQLite, no grant involved, works for
any principal) and neither of the other two.

Facts, sites and ratified decisions live in
`14-eng-ai-bot/.prawduct/artifacts/collection-support-evidence.md`. Shards cite
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
| `coll-task-04-bucket-config-reconcile` | 3tears + hub + SDK | `allow_direct: true` via a real reconcile primitive; deletes a class the SDK's test conftest patches |
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

Three repos against a PyPI-pinned dependency, and the mechanics are not
currently in place. Verified state: all three consumers pin `3tears*==0.26.1`
from PyPI **and** carry `3tears-search==0.26.1` in `constraint-dependencies`;
hub `[tool.uv.sources]` has no `../3tears` entry; 3tears is already on `0.27.x`
intra-family bounds; the hub depends on the SDK and admin as **editable path
sources**, so all three resolve together.

There is also **no producer-side matrix gate** — `matrix-fan-out.yml` was never
committed and the `override-matrix-gate` label is read by no workflow. Consumer
CI checks out 3tears by the tag in its own `uv.lock`, so it is blind to
unreleased 3tears work.

1. 3tears feature branch. Pick the target version now; use it everywhere.
2. **In hub + SDK + admin, in one commit each:** bump every `3tears*` pin *and*
   every `constraint-dependencies` entry to the target version, and add the
   per-package `[tool.uv.sources]` path entries. Do not add the override without
   the bump — a path source pointing at 0.27.0 fails against a `==0.26.1`
   requirement. Do not leave admin behind; it participates in the hub's
   resolution.
   The override is ~30 per-package entries per repo, not one line, and **there
   is no generator for them**. `scripts/link-3tears.sh` only runs `ln -sfn` to
   create a `.3tears` symlink; it writes no `pyproject.toml` entries, no
   pyproject in any repo references `.3tears` any more, and the hub has no such
   symlink at all. The script and the matching block in the hub's `ci.yaml` are
   vestigial. Recover the entries from `git show 540fcfcd -- pyproject.toml` in
   the hub, which added 29 — **it omits `3tears-search`**, the one every consumer pins via
   `constraint-dependencies`, so add that by hand or you ship the mixed family
   this section warns about.
3. 3tears: bump the family in lockstep with `scripts/bump-version.sh`, merge to
   develop, then main, tag, publish.
4. Consumers: `uv lock`, commit the lock.
5. **Remove the sources block** and re-verify the build resolves from PyPI alone
   (`uv export --locked`). This is the step most likely to be skipped, and
   skipping it ships a repo that builds only on a machine with a sibling
   checkout.
6. Merge consumers. Fire `matrix-nightly.yml` by hand — nothing runs on a
   schedule.

Whoever lands steps 2 and 5 also fixes `14-eng-ai-bot/CLAUDE.md`'s "3tears
Library Dependency" section, which is wrong on **both** lines: there is no local
path dependency, and CI/deployment does not use a git dependency with a pinned
tag — all three locks resolve from PyPI.

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

---

## Before writing code

Every shard gets an adversarial review first, and the review must demand a
**live probe** for each security claim rather than a reading of the gate.

Successive rounds of independent review over this set have each found bypasses
of the same class as the one the work started from, plus claims whose cited
evidence did not hold — including counts stated with confidence and no source. Reasoning over a partial read is the recurring failure mode here. Running
the thing is what catches it, and a probe that is collected but skipped is not a
probe — the 3tears integration suite does not run under `check-all.sh`.
