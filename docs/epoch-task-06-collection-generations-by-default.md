# epoch-task-06: Every Collection Carries a Write Generation by Default

**Status:** DESIGN, nothing built. The direction was decided by the product owner on
2026-10-08 and is recorded under "The Decision"; it is not re-argued here. What this note
adds is the model, the costs, the rollout, and the questions that are still his.
**Scope:** `3tears-core` (`collections/base.py`, `collections/registry.py`,
`collections/generation.py`, `collections/caller_transaction.py`, `collections/flush.py`),
`3tears-epoch` (`generation.py`, a generation catch-up pass beside `tick.py`), `3tears-nats`
(`subject_permissions.py`), `3tears-agent-acl` (`cache.py`, `invalidation_bus.py`),
`3tears-agent-knowledge`. Out of repo: the hub, the SDK, identity-core, provider tool pods.
**Depends on:** epoch-task-01, 02, 03 (the epoch bucket, replaced-counter detection, the
consumer-driven tick). Revisits epoch-task-04 and 05; see "The L1 Max Age".

> Citations name the symbol, not the line, for the reason epoch-task-05 gives. Where
> something could not be confirmed from the code, it says so in one line.

---

## Objective

Make "this pod's cache is behind" something a pod can detect for every collection table,
without a timer, and let anything derived from collection tables use the same signal instead
of its own invalidation subject.

## The Decision

1. Every collection carries a write generation by default. It is opt-out. A forgotten
   opt-in is silent staleness, and for access tables a security problem; a forgotten opt-out
   is visible write latency.
2. A collection built with `NO_L2` gets no generation: it shares no cache across pods.
3. A hot table that does share a cache takes an explicit, named opt-out.
4. A cache derived from collection tables keys on those tables' generations. First the
   access cache (`AclCache` in the hub and agents), then a per-caller "which states may this
   person see" cache in a tool pod.
5. No timeout or other safety net goes on top of the epoch system. If a change is not
   covered, the epoch system is extended to cover it.
6. Tables with several independent classes are cleaned up to one class per table.

## What a Write Does Today

`BaseCollection.save_entity` writes L3, then L2 (`_cache_committed_row`, or `_save_to_l2`
on the write-behind path), then calls `_publish_invalidation`, which reaches
`CollectionRegistry.publish_invalidation`: one `CacheInvalidationMessage` per row on
`Subjects.cache_invalidate`. A peer's `start_invalidation_listener` handler drops dependent
scans (`ScanCache.drop_for_table`), deletes its own scoped L2 key (`delete_l2_entry`) and
evicts the row from L1 (`evict_from_cache_sync`). The subject is core NATS: at most once.

Two things were decided about a lost message. epoch-task-04 closed the publisher-side
queue: nats-py already buffers and replays a publish across a reconnect, and the residue
(outbound buffer overflow, and a pod whose *subscription* is partitioned) is out of any
publisher's reach. epoch-task-05 bounded that residue with a lazy L1 max age
(`CollectionRegistry.set_l1_max_age`, `BaseCollection._row_is_expired`), off unless a table
asks. The one production caller found is the hub's `data_version_fence`. So today a missed
invalidation on almost every table is served until the pod restarts.

## The Generation That Already Exists

`threetears.core.collections.generation.GenerationSource` is a two-method protocol,
`current(table)` and `advance(table)`. `threetears.epoch.generation.EpochGenerationSource`
implements it over the `{ns}-epochs` memory KV bucket, one key per table
(`Subjects.collection_generation_epoch`, `{ns}.collections.{table}.epoch`), holding
`{incarnation}:{count}` in a single value. The incarnation is a `uuid7` minted when the key
is first created, so a broker restart that empties the bucket yields a token no old one can
equal. One value, not two keys, so a read cannot straddle a wipe.

It is used for exactly one thing: absence caching. A collection that sets
`negative_cache_max_age` stamps each recorded absence with the token, read in
`_pull_through_watched` (`_current_generation`) before the L3 lookup that found nothing, and
advances it in `save_entity` and `_persist_cas_result` (`_advance_generation`) after the L3
commit. One collection sets it, `CoordinationRevocationsCollection`. The source is wired
with `CollectionRegistry.set_generation_source`, and `_refuse_unsound_negative_cache` refuses
construction without one.

**Cost, counted from `EpochGenerationSource`.** `advance` is `get_entry` then `update` at
that revision: two KV round trips uncontended. Each lost compare-and-swap adds two more and
a jittered sleep of up to `_CAS_RETRY_BACKOFF_SECONDS`, to a budget of `_MAX_CAS_ATTEMPTS`.
`current` is one `get`; the first read of a table adds a `create`. The bucket handle is
cached by `NatsClient.kv_bucket`.

**Write paths that do not advance today**, because absence caching did not need them:
`BaseCollection.delete`, the write-behind branch of `save_entity`, subscript writes
(`_async_propagate_write`), `invalidate_cache_many`, `bypassing_write`, and
`CallerTransaction._settle`. A negative-caching collection refuses `conn=` and subscript
writes outright. All of these have to be covered before a generation means "any write".

**Who wires a source.** Only identity-core, in `identity_core/server.py`
(`set_generation_source(EpochGenerationSource(self._nc))`). The hub builds an `EpochClient`
and a registry in `aibots/hub/app.py` and no source. The SDK's agent bootstrap
(`build_three_tier_stack`), its granted-writer stack (`build_owner_data_stack`) and the
framework's tool pod stack (`ToolServerBootstrap.install_collection_stack`, which
`ProviderToolPod` extends) build registries with no source and no epoch client at all. Each
needs one `set_generation_source` call after `configure`, plus a broker grant it may not
hold, below.

## Who May Reach the Bucket

From `threetears.nats.subject_permissions`: `_hub` and `_gateway` hold
`JsResource.kv(f"{ns}-epochs", scope=None, writable=True)`. `_agent_pod` holds
`JsResource.kv_bucket_keys(f"{ns}-epochs", writable=False)`: read only, with the reason
written beside it (a pod that could write could fake a bump the fleet acts on). `_tool_pod`,
`_registry`, `_channel_adapter`, `_agent_router` and `_dataset_executor` hold nothing on the
bucket. Epoch *subjects* are granted to agent pods (`mcp_rbac_epoch`), the hub and the
gateway; a tool pod holds none. The agent pod, tool pod, hub and gateway each already hold
`CROSS_PLATFORM_CACHE_INVALIDATE`. How identity-core's service credential is granted the
bucket was not found in either repo.

This is the hard part of "by default". An agent pod writes collection tables and cannot
advance a generation. No existing `JsCapability` expresses "write these literal keys of an
unscoped bucket"; `KV_KEY_READ` is the read half only. Two ways through, and the choice is
an open question: add the write sibling and grant each pod the keys of the tables it may
write, or have the hub's L3 broker advance after it commits a pod's write, so no pod writes
the bucket. Whether the broker sees commit boundaries was not confirmed.

## The Model

**A generation is per L3 table.** It answers "has any write to this table committed since
token X". It cannot say which row.

**Each registry keeps, per followed table, a mark:** the last token whose writes this pod
has accounted for. A pod is *behind* when the bucket's token is not its mark and it cannot
account for the difference.

**The per-row broadcast stays the fast path, and now accounts for bumps.**
`GenerationSource.advance` returns the token it wrote. `CacheInvalidationMessage` gains two
optional fields, `generation` and `bump_rows` (how many rows that bump covered). A receiver
evicts the row exactly as today and records that it heard one row of that count. When every
count between its mark and a token has been heard in full, the mark moves. A writer accounts
for its own bumps directly, since its registry skips its own broadcasts (`origin`).

**The catch-up pass is the judge.** One pass reads each followed table's token:

- incarnation differs from the mark's: the bucket was replaced. Drop, record the new token.
- same incarnation, every count up to it heard: nothing to do.
- same incarnation, a count unheard: a broadcast was missed. Drop, record the new token.

**Drop means the table, in this pod.** Its L1 rows, its scans (`drop_local_scans`), and a
notification to every derived cache registered on it. The pod's own scoped L2 keys for the
table cannot be dropped as a set without listing the bucket; whether a table drop must also
clear them, or a re-read through L3 is forced another way, is not designed here and must be
before build (the listener deletes the L2 key per row today for exactly this reason).

A finer rule than "drop the table" needs the missed rows' identities, which means a durable
log of invalidations. That is epoch-task-04's option 2 (JetStream) and its cost; not
proposed.

A pass that reads a token while that bump's broadcast is still in flight sees an unheard
count and drops needlessly. That is the safe direction and should be rare on tables left on
the default; the measurement below counts it.

**Broker restart.** The per-key incarnation already covers it, so collection generations do
not need `EpochListener._bucket_was_replaced`. `GenerationUnavailableError` is not a reset:
keep the mark, serve as before, try again next pass.

## Read on Every Lookup, or Follow

**Read on every lookup** costs one KV round trip per read, turns an in-process L1 hit into a
network call, and is impossible for the synchronous readers (`get_row_sync`,
`get_field_sync`, `__getitem__`). It is what absence caching does, correctly, because it
reads only on the miss path.

**Follow** costs nothing on a read. It costs one KV `get` per followed table per pass per
pod, and a missed broadcast is caught at the next pass.

**Recommendation: follow.** Absence caching keeps its read-before-lookup, unchanged.

The existing consumer side does not fit as it is. `EpochListener.catch_up` reads
`EpochClient.current`, which decodes a `DistributedCounter` integer and cannot parse
`{incarnation}:{count}`; `Subjects.collection_generation_epoch` is documented as never
broadcast; `EpochClient.observe_broadcast` and `versions(max_age=)` serve the durable tile
family only. So the follower is new: the per-table mark lives in core beside
`GenerationSource`, and one pure-async pass lives in `3tears-epoch` beside `catchup_tick`,
with the same contract (one pass per call, the consumer schedules it, one table's failure
does not abandon the rest). Today `catchup_tick` has one caller, `threetears.mcp.auth`.
epoch-task-03 asked that the wiring requirement be recorded: every bootstrap that wires a
generation source must also schedule this pass, and the enforcement test below checks it.

`NatsKvBucket.watch_key` is the alternative to a pass: push delivery, and it redelivers the
latest value after a restart. It costs one named consumer per table per pod, which is wrong
for a registry following every table and plausible for a pod following four keys.

## Batching

One bump per table per commit, never one per row.

- A single `save_entity`, `delete`, or won `l2_cas_mutate`: one bump, `bump_rows=1`.
- `CallerTransaction._settle` already groups enrolled keys by collection and calls
  `invalidate_cache_many` once each. It bumps once per table after the transaction ends, and
  every row message of that settle carries the token and the row count. This also lifts the
  reason `save_entity` refuses `conn=` on a negative-caching collection.
- `bypassing_write` on the collection's own pool: one bump when the body ends, unless
  `BypassingWrite.unchanged` was called.
- Write-behind: `flush_pending` bumps once per table per flush, after the batch commits, and
  re-announces that flush's rows with the token. Not at save time: the row is not in L3
  until the flush, and derived caches read L3. A write-behind table left on the default
  therefore pays a second broadcast per row. Most belong on the opt-out list.
- Subscript writes are fire-and-forget and can only log a failed advance.

## The Write Cost, and How It Is Measured

No numbers exist yet and none are invented here. Before the default flips:

- **Harness.** A real NATS (the `nats_container` fixture) and a real Postgres, with
  `EpochGenerationSource` wired.
- **Tables.** One config-like (`role_assignments`), the six undecided tables below, and two
  from the opt-out list as the expected worst case (`usage_records`, `context_items`).
- **Load.** Each table's write path at 1, 8 and 32 concurrent writers to the same table.
- **Compared.** `save_entity` latency at p50 and p99 with and without the advance; lost
  compare-and-swap rounds per write; the bucket's operations per second; and, on a second
  registry following the table, needless table drops per minute.
- **Production shape.** Per-table write rates from the deployed hub, which this note does
  not have, to say which tables see concurrent writers at all.

**Rule for the opt-out list.** A table opts out when, at its observed production write
rate, the advance adds more than an agreed fraction to its p99 write latency, or exhausts
`_MAX_CAS_ATTEMPTS` at all, or its followers drop the table more often than they hit it. The
fraction is the owner's to set. A table that governs access never opts out on cost.

## The Opt-Out

A typed class-level declaration, in the style of `NO_L2`, with the reason as a field so a
test can require one (as `_DURABLE_FAMILIES` does):

    write_generation: ClassVar[WriteGeneration | NoWriteGeneration] = WRITE_GENERATION

    class SomeMeteringCollection(...):  # illustrative
        write_generation = NoWriteGeneration(reason="append-only; no pod reads a row by key")

`NO_L2` stays a constructor argument and needs no declaration. An explicit
`nats_client=None` is not `NO_L2`: it still advances when the registry has a source, for the
reason `_negative_cache_writes_advance` gives (the caches to invalidate are other pods').
`__init_subclass__` refuses `negative_cache_max_age` together with an opt-out. A collection
with no L3 (`HeartbeatCollection`, the presence rooms) shares a cache through L2 and must
declare one way or the other; both are hot.

**Initial list. A starting point, to be confirmed by the measurement.** The survey behind
it classified most tables from docstrings, not from write rates.

| Tables | Reason |
|---|---|
| `job_fires`, `wake_fires`, `agent_skill_invocations` | one row per firing; append-only |
| `coordination_counters`, `coordination_claims`, `coordination_redemptions` | compare-and-swap state written on the request path |
| `context_items`, `conversation_memory_refs` | written every turn |
| `memories`, `memory_chunks`, `memory_consolidations` | written by agents during turns and consolidation |
| `workspace_files`, `workspace_file_versions` | written per file operation |
| `scrape_extractions`, `scrape_target_health` | one row per fetch |
| `dataset_runs`, `dataset_run_inventory` | per-run progress rows |
| `audit_events`, `backup_operations` | append-only logs |
| `usage_records`, `usage_cost_records`, `usage_export_watermark` | metering; the watermark moves every export |
| `eval_runs`, `eval_case_results` | one row per case |
| `geo_features_*`, `geo_tiles_*` | derived caches with their own durable tile epoch (`datasource_tile_epoch`) |

**To be decided per table**, config-like rows with a per-event write path: `scheduled_jobs`,
`agent_wake_schedules`, `webhook_subscriptions`, `agent_skills`, `conversations`,
`intentions`. Splitting the per-event columns off the row is the alternative to opting out.

## Derived Caches

**The rule.** A cache derived from collection tables records those tables' tokens before it
reads them, and is invalid when any has moved. It declares no invalidation subject of its
own. `ScanCache.begin_read` is the existing shape for "take the token before the query,
refuse the store if it moved".

**`AclCache` today** (`threetears.agent.acl.cache`) is three in-process layers with a
`ttl_seconds` of 60. `subscribe_acl_invalidation` binds it to
`Subjects.acl_invalidate("membership" | "assignment" | "role")`. The hub publishes from
`publish_acl_invalidation` in `aibots/hub/rbac/collections.py`, called by the `Hub*`
collections' `save_entity` and `delete` overrides, best-effort, with the TTL named as the
fallback. A nested-group edit announces only the child group: `MembershipInvalidatePayload`
with `actor_type="group"` evicts that group's own entry, which is right inside `AclCache`
(the walk happens at read) and useless to anything keyed by the user above it. A group
delete has to read and announce each cascaded member by hand, because no collection sees the
cascade.

**On generations.** `AclCache` follows `groups`, `group_members`, `roles` and
`role_assignments`. When any of the four moves, it calls `invalidate_all`. That is coarser
than today's per-actor eviction and it is the price of exactness: the row broadcast carries
the primary key `(group_id, id)`, not the member, so it cannot name an actor's entry. The
nested-group and cascade cases stop being special. Whether `namespaces` belongs in the set
was not confirmed. The refill cost under a burst of rbac writes (bootstrap, reconcilers,
agent access materialization) is part of the measurement.

**The `acl.*.invalidate` subjects** stay through expand and migrate and are removed at
contract: subscribers first, then `publish_acl_invalidation` and
`evict_after_rbac_write`'s publish half, then the grants. `ttl_seconds` goes with them, per
decision 5.

**A tool pod** follows the same four tables for its per-caller cache. It needs
`JsResource.kv_key_read(f"{ns}-epochs", key=...)` for each of the four keys (the shape it
already holds on the data-versions bucket), and it already hears the row broadcasts. It
needs no epoch subject and no `acl.*` subject.

## The L1 Max Age

Once a table carries a generation, a missed invalidation is detected at the next pass, which
is what epoch-task-05 was built for. Stated plainly, a max age would still cover only:

- a table with no generation (opted out, or `NO_L2`);
- a write that moves no generation: raw SQL outside a collection, a database cascade, a
  migration or backfill, a writer on an old release, a write whose advance failed.

Each item in the second line is, under decision 5, a gap to close in the epoch system, not a
reason for a timer. With the write-path inventory complete, a max age covers nothing a
generation-carrying table lacks.

**Recommendation:** leave `set_l1_max_age` in place and off by default through expand and
migrate, add no new callers, and take **removing it as a named decision for the owner** at
contract, together with the hub's one existing caller. `ScanCache`'s
`DEFAULT_SCAN_TTL_SECONDS` is the same question and should be answered with it.

## One Class per Table

`playbook_entries` and `concepts` each have three unrelated classes, all subclassing
`SchemaBackedCollection` directly: `PlaybookEntryCollection` and `ConceptCollection` in
`threetears.agent.knowledge.collections`, in the hub (`aibots.hub.knowledge.
playbook_collections`, `concept_collections`) and in the SDK
(`aibots_agents.runtime.knowledge`). A class-level `write_generation` set on one does not
bind the other two, and they live in three repos, so no in-process check can see the
disagreement. Their schemas were not compared column by column.

**Cleanup.** The framework class is the one class. The hub and SDK delete theirs and import
it; admin-only or agent-only queries go in a subclass, the way `HubGroupMemberCollection`
extends `GroupMemberCollection`, so the declaration is inherited. Do it before the default
flips. The registry also refuses to hold two collections for one table that disagree.

## Rollout

Expand, migrate, contract. Old pods that neither bump nor follow coexist with new ones at
every stage.

1. **Expand, release N.** `advance` returns its token. The message fields, the mark, the
   pass, the opt-out declaration and the grants ship. Nothing bumps and nothing follows
   unless a table is switched on. Old receivers ignore the new fields; a message without
   them is handled as today.
2. **Migrate writers.** Each deployment wires a source and its tables bump. Readers still
   rely on the row broadcast, the `acl.*` subjects and the TTLs. Safe in any order.
3. **Migrate readers.** Followers switch on, table by table. A table drop is always safe.
   **A follower must not read "the token did not move" as "nothing changed" while any
   writer of that table is on an old release:** the old writer committed and bumped nothing.
   So following is additive here. No older mechanism is removed for a table until every
   deployment that writes it is past stage 2, and that is a fact about releases, recorded
   per table, not something a reader infers from the bucket.
4. **Flip the default.** After the measurement and the one-class cleanup. From here a new
   collection bumps unless it declares otherwise.
5. **Contract.** Remove the `acl.*.invalidate` subscribers, publishers and grants and
   `AclCache.ttl_seconds`. The L1 max age and the scan TTL wait on the owner's decision.

The public API grows (`advance`'s return, the declaration, the follower), so this takes a
minor bump across the family.

## Tests and Enforcement

- A missed broadcast is caught: two registries, drop one message, run one pass, the
  follower's row is gone. No sleep.
- A broker restart is caught: delete the bucket's stream (the one-liner epoch-task-02
  uses), write, run one pass, the table drops.
- An opted-out table bumps nothing, and a `NO_L2` collection bumps nothing: assert zero KV
  writes on the epoch bucket.
- One bump per transaction: a `CallerTransaction` saving many rows of one table moves the
  count by one, and a follower that hears all of them does not drop.
- A flush bumps once per table.
- `GenerationUnavailableError` on a pass does not drop and does not move the mark.
- A derived cache: a `group_members` write on one registry empties an `AclCache` on another
  with no `acl.*` subscription bound.
- **Enforcement.** Enumerate every `BaseCollection` subclass in the family: each carries
  the default or a `NoWriteGeneration` with a non-empty reason, and no two classes share a
  `table_name`. A second test fails when a bootstrap wires a generation source and
  schedules no pass. A grant test pairs each principal's epoch-bucket keys with the keys
  `EpochGenerationSource` opens, as epoch-task-01 did for the bucket.

## Open Questions for the Owner

1. **Who advances for a pod.** A new key-scoped write grant per pod, or the hub's broker
   advancing after it commits the pod's write.
2. **A failed advance after a committed write.** Raise, as `save_entity` does today for
   absence caching, or log. Recommended: raise; it is the only way the write is covered.
3. **The pass interval for the access tables,** or a `watch_key` on those four keys instead.
   It is the bound on a missed broadcast.
4. **The generation key for per-agent tables.** The key carries the table name only, so
   every agent's `memories` would share one generation. Right for platform tables; for
   per-agent namespaces it is correct but noisy. Most such tables are on the opt-out list.
5. **The opt-out threshold:** the fraction of p99 write latency that puts a table on the
   list.
6. **Remove the L1 max age and the scan TTL at contract, or keep them.**
7. **The six undecided tables:** opt out, or split the per-event columns off.
8. **`AclCache` granularity:** is `invalidate_all` on any of the four tables acceptable, or
   must the row broadcast carry enough to evict one actor.
