# coll-task-07a: Consolidate the Duplicated Three-Tier Stacks

## Objective

Delete the duplicated ACL invalidation wiring and make every caller use the
shared implementation that **already exists**.

No *grant-surface* change, and reviewable on its own -- which is why it is split
out from the tool-pod grant work in `-07c`. It does carry two deliberate
behaviour changes (CON-04, CON-05), and it depends on `coll-task-05a` GRANT-13
for the deadletter grant that adopting `subscribe_typed` requires.

---

## The extraction is already done; nobody calls it

`threetears.agent.acl.invalidation_bus` defines `AclInvalidationSubscriber` as a
Protocol, exported from the package `__init__`, and `subscribe_acl_invalidation`
already performs exactly the wiring that **three** call sites duplicate -- the
same three handlers, the same payload models.

None of `packages/registry/.../rbac_stack.py`, the SDK's
`runtime/three_tier_stack.py`, or the hub's `broker/acl.py` calls it. The hub's
copy already pairs `subscribe_invalidations` with `unsubscribe_invalidations`,
which is the shape CON-02 wants.

So the instruction is **not** "extract a shared subscriber" -- an earlier
formulation of this work said that, and it would have collided with a Protocol of
that exact name in that exact package. It is: **delete all three copies and call
the existing function.**

`rbac_stack.py`'s own opening docstring already says it "mirrors the agent SDK's
`build_three_tier_stack` with the agent-specific bits stripped" -- the duplication
is acknowledged in source, just never resolved.

---

## The copies have already drifted, and one drift is a live bug

This is the argument for doing it now rather than recording it.

- **Warning text and control flow differ.** The registry copy logs with a
  `registry acl cache:` prefix and returns early; the SDK copy logs
  `agent acl cache:` and uses `try/except/else`. Near-identical modulo the
  prefix, but not identical. Only the bus and the SDK carry a queue-group
  rationale, and their texts differ; the registry has none.
- **There are three copies of the rbac L1 metadata, not two**, and **two of the
  three are missing the same five columns** -- `tool_eligible`, `skill_eligible`,
  `face_api`, `face_mcp`, `face_platform_tool` -- against the canonical
  `NamespaceCollection.schema`. The registry copy is one. The other is the
  **hub's** `common/l1_cache.py`, which is live under `HubNamespaceCollection` in
  both `hub/app.py` and `gateway/acl.py`. Only the SDK copy is current. That is
  the `column does not exist` failure the SDK copy documents in its own comments,
  sitting in production in the repo that owns those columns.
- The registry uses `TIMESTAMP(timezone=True)` where the SDK uses bare
  `TIMESTAMP`.

**This shard takes the hub half.** It crosses a repo, as `-01` and `-04` already
do, and the alternative -- folding it into `coll-task-06b` -- does not work:
`hub/common/l1_cache.py` is not a registry-construction site, so `-06b` never
visits it. Leaving "lives once" as a criterion while a third copy stands is the
failure this section exists to name.

The rbac L1 table metadata should move beside the ACL Collections in
`threetears.agent.acl` -- and be **generated**, not retyped.
`TableSchema.to_sqlalchemy_table(metadata)` already performs this conversion,
`NamespaceCollection` is a `SchemaBackedCollection`, and
`core/testing/sqla_parity.py` is the established drift guard for any table that
must stay hand-written. Consolidating three hand-maintained copies into one
hand-maintained copy just moves the drift a column later -- which is exactly the
bug this section opens by describing. The registry's stated reason for keeping its own copy -- 
avoiding a dependency on the agents package -- does not block that: the registry
already imports from `threetears.agent.acl`.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| CON-01 | **All three** callers use `subscribe_acl_invalidation`; every local copy is deleted | P0 |
| CON-02 | Callers unsubscribe the handles the function already returns; add `unsubscribe_acl_invalidation(subscriptions)` beside it so the idiom is one, and type the return | P0 |
| CON-03 | The rbac L1 table metadata lives once, in `threetears.agent.acl` | P0 |
| CON-04 | The consolidated metadata is **generated** from `NamespaceCollection.schema` via `TableSchema.to_sqlalchemy_table`, not retyped | P0 |
| CON-05 | The only behavioural changes are CON-04's bug fix and the malformed-payload semantics below -- both stated in the commit | P0 |

**CON-02 is smaller than it looks.**
`subscribe_acl_invalidation` does **not** lack teardown: it returns the three
`Subscription` handles, its docstring says they are "for the caller to
unsubscribe at shutdown", and it already unwinds them itself on partial failure.
Both existing callers run that loop. The real defects are the untyped `list[Any]`
return and two competing idioms -- the bus unwinds via `subscription.unsubscribe()`
while consumers call `nats_client.unsubscribe(sub)`. Those are not equivalent -- 
the client form also removes the handle from `client._subscriptions`, the
`Subscription` form does not. Pick the client form (what all three consumers use
today) and say so, or the helper silently changes teardown bookkeeping and CON-05
is wrong again. No state lives on the subscriber; do not give it any.

**CON-05 is not "no behavioural change".** The local copies use raw `subscribe`
with hand-rolled `model_validate_json` and log WARNING + continue on a malformed
payload. `subscribe_acl_invalidation` uses `subscribe_typed`, which routes
validation failures and callback exceptions to `{ns}.deadletter.{subject}` and
**never calls the handler**. So adopting it deletes the local WARNING and adds a
deadletter publish each principal must be granted. On a security-cache path that
must be stated, not discovered.

CON-04 is also a deliberate behaviour change -- see below.

---

## Files to Modify

- `packages/agent/acl/src/threetears/agent/acl/invalidation_bus.py` -- the teardown counterpart.
- `packages/agent/acl/` -- the rbac L1 table metadata, beside the ACL Collections.
- `packages/registry/src/threetears/registry/rbac_stack.py` -- delete the local subscriber and teardown; call the shared one. Also fix its handler docstring, which claims unparseable payloads are logged **and the cache invalidated** ("the canonical fail-safe behaviour") -- every handler returns after the warning without invalidating.
- `14-eng-ai-bot/src/aibots/hub/broker/acl.py` -- the **third** copy, with its own `subscribe_invalidations` / `unsubscribe_invalidations`.
- `14-eng-ai-bot/src/aibots/hub/common/l1_cache.py` -- the third rbac-metadata copy, also missing the five columns, and live under `HubNamespaceCollection`. Its comment claims it mirrors the canonical `TableSchema` byte-for-byte.
- `packages/registry/src/threetears/registry/l1_cache.py` -- delete the duplicated metadata.
- `14-eng-ai-bot-agents/src/aibots_agents/runtime/three_tier_stack.py` and `runtime/l1_cache.py` -- same.

Genuinely out of scope, and correctly left alone: `ThreeTierStack`'s two-pool
`NatsProxyL3Backend` split, its knowledge and datasource collections, and
`PLATFORM_RBAC_READ_NAMESPACE`. Those are agent and hub policy, not shared
substrate.

---

## Quoting hazard

Several docstrings in `rbac_stack.py` and `registry/l1_cache.py` mis-name the SDK
module as `3tears_agents.runtime.three_tier_stack` (it is `aibots_agents`). Quote
them verbatim in a commit message or a comment and it reads as a typo you
introduced. Fix them in place while the files are open.

---

## Anti-patterns

- DO NOT create a new `AclInvalidationSubscriber`. One exists, as a Protocol, in the target package -- and note it degrades to plain `object` when the acl `[bus]` extra is absent, which is why the extra is required rather than optional.
- DO NOT restructure the subscriber into a stateful lifecycle object. The handles are already returned; `unsubscribe_acl_invalidation` is a thin symmetric helper, not a redesign.
- DO NOT leave one copy "for now". CLAUDE.md bans the parallel path.
- DO NOT adopt the registry's or the hub's column list. Both are behind; `NamespaceCollection.schema` is the source.
- DO NOT widen the scope to the L3 backend split or the agent-specific collections.

---

## Success criteria

- [ ] `subscribe_acl_invalidation` is the only implementation; both local copies gone
- [ ] It has a teardown counterpart and both callers use it
- [ ] The rbac L1 metadata exists once and carries all columns of `NamespaceCollection.schema`
- [ ] Registry and agent-pod ACL invalidation behave as before, minus the missing-column bug
- [ ] `./scripts/check-all.sh` and `./scripts/test-integration.sh` green
- [ ] `cd 14-eng-ai-bot-agents; uv run pytest tests/unit/ -q` green

---

## Verification

```bash
cd 3tears
./scripts/check-all.sh
./scripts/test-integration.sh
cd ../14-eng-ai-bot-agents
uv run pytest tests/unit/ -q
```

Behavioural: publish a membership, assignment and role invalidation; confirm both
the registry and an agent pod evict as they did before. Then confirm a namespace
row carrying `tool_eligible` round-trips through the registry's L1 -- that is the
column-list bug, and it is the reason to prefer a live check over a diff read.
