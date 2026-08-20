# coll-task-07a: Consolidate the Duplicated Three-Tier Stacks

## Objective

Delete two copies of the same ACL invalidation wiring and make both callers use
the shared implementation that **already exists**.

No security change. This shard is pure consolidation and can be reviewed on its
own — which is why it is split out from the tool-pod grant work in `-07c`.

---

## The extraction is already done; nobody calls it

`threetears.agent.acl.invalidation_bus` defines `AclInvalidationSubscriber` as a
Protocol, exported from the package `__init__`, and `subscribe_acl_invalidation`
already performs exactly the wiring that two repos duplicate — the same three
handlers, the same payload models, the same no-queue-group rationale.

Neither `packages/registry/.../rbac_stack.py` nor the SDK's
`runtime/three_tier_stack.py` calls it.

So the instruction is **not** "extract a shared subscriber" — an earlier
formulation of this work said that, and it would have collided with a Protocol of
that exact name in that exact package. It is: **delete both copies, call the
existing function, and add the teardown half it lacks.**

`rbac_stack.py`'s own opening docstring already says it "mirrors the agent SDK's
`build_three_tier_stack` with the agent-specific bits stripped" — the duplication
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
  three are missing the same five columns** — `tool_eligible`, `skill_eligible`,
  `face_api`, `face_mcp`, `face_platform_tool` — against the canonical
  `NamespaceCollection.schema`. The registry copy is one. The other is the
  **hub's** `common/l1_cache.py`, which is live under `HubNamespaceCollection` in
  both `hub/app.py` and `gateway/acl.py`. Only the SDK copy is current. That is
  the `column does not exist` failure the SDK copy documents in its own comments,
  sitting in production in the repo that owns those columns.
- The registry uses `TIMESTAMP(timezone=True)` where the SDK uses bare
  `TIMESTAMP`.

The hub copy is outside this shard's repos. Either give CON-03/CON-04 a hub half
or fold it into `coll-task-06b`, which already visits every hub registry site —
but do not leave "lives once" as a criterion while a third copy stands.

The rbac L1 table metadata should move beside the ACL Collections in
`threetears.agent.acl`. The registry's stated reason for keeping its own copy —
avoiding a dependency on the agents package — does not block that: the registry
already imports from `threetears.agent.acl`.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| CON-01 | Both callers use `subscribe_acl_invalidation`; both local copies are deleted | P0 |
| CON-02 | Callers unsubscribe the handles the function already returns; add `unsubscribe_acl_invalidation(subscriptions)` beside it so the idiom is one, and type the return | P0 |
| CON-03 | The rbac L1 table metadata lives once, in `threetears.agent.acl` | P0 |
| CON-04 | The consolidated metadata carries every column of `NamespaceCollection.schema` | P0 |
| CON-05 | The only behavioural changes are CON-04's bug fix and the malformed-payload semantics below — both stated in the commit | P0 |

**CON-02 is smaller than it looks, and an earlier draft got it backwards.**
`subscribe_acl_invalidation` does **not** lack teardown: it returns the three
`Subscription` handles, its docstring says they are "for the caller to
unsubscribe at shutdown", and it already unwinds them itself on partial failure.
Both existing callers run that loop. The real defects are the untyped `list[Any]`
return and two competing idioms — the bus unwinds via `subscription.unsubscribe()`
while consumers call `nats_client.unsubscribe(sub)`. Do **not** restructure a
deliberately stateless handle-returning function into a `coll-task-01`-style
lifecycle object; no state lives on the subscriber.

**CON-05 is not "no behavioural change".** The local copies use raw `subscribe`
with hand-rolled `model_validate_json` and log WARNING + continue on a malformed
payload. `subscribe_acl_invalidation` uses `subscribe_typed`, which routes
validation failures and callback exceptions to `{ns}.deadletter.{subject}` and
**never calls the handler**. So adopting it deletes the local WARNING and adds a
deadletter publish each principal must be granted. On a security-cache path that
must be stated, not discovered.

CON-04 is also a deliberate behaviour change — see below.

---

## Files to Modify

- `packages/agent/acl/src/threetears/agent/acl/invalidation_bus.py` — the teardown counterpart.
- `packages/agent/acl/` — the rbac L1 table metadata, beside the ACL Collections.
- `packages/registry/src/threetears/registry/rbac_stack.py` — delete the local subscriber and teardown; call the shared one.
- `packages/registry/src/threetears/registry/l1_cache.py` — delete the duplicated metadata.
- `14-eng-ai-bot-agents/src/aibots_agents/runtime/three_tier_stack.py` and `runtime/l1_cache.py` — same.

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

- DO NOT create a new `AclInvalidationSubscriber`. One exists, as a Protocol, in the target package — and note it degrades to plain `object` when the acl `[bus]` extra is absent, which is why the extra is required rather than optional.
- DO NOT build a teardown function. The handles are already returned.
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
row carrying `tool_eligible` round-trips through the registry's L1 — that is the
column-list bug, and it is the reason to prefer a live check over a diff read.
