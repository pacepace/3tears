# epoch-task-06: Every Collection Carries a Write Generation by Default

**Status:** STAGES 1 AND 2 OF 5 BUILT (expand, migrate writers). Every deployment now wires a
generation source and the hub's L3 broker advances a pod's switched-on tables after each commit;
no table is switched on outside tests, and nothing follows. What exists, what each build decided,
where this note was wrong against the code, and what the later stages need are under "Built in
the Expand Stage" and "Built in the Migrate-Writers Stage" at the end. The direction was decided by the product owner on
2026-10-08 and is recorded under "The Decision"; it is not re-argued here. What this note
adds is the model, the costs, the rollout, his answers to the questions it raised, and the
three that wait on the measurement.
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
bucket was not found in either repo. (Found in the expand build: the hub's
`aibots.hub.security.static_nats_grants` declares the `identity` user's grants explicitly, and
`_kv("epochs")` is among them.)

This is the hard part of "by default". An agent pod writes collection tables and cannot
advance a generation. No existing `JsCapability` expresses "write these literal keys of an
unscoped bucket"; `KV_KEY_READ` is the read half only. Two ways through: add the write
sibling and grant each pod the keys of the tables it may write, or have the hub's L3 broker
advance after it commits a pod's write, so no pod writes the bucket. **Decided: the broker
advances.** Whether the broker sees commit boundaries was not confirmed when this was written.
The build confirmed it does; see "The Broker Sees Every Commit" at the end.

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
clear them, or a re-read through L3 is forced another way, was not designed here (the listener
deletes the L2 key per row today for exactly this reason). The build decided it: a drop must
stop trusting them, and does so key by key through L3. See "The Table Drop and L2" at the end.

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

**On generations, row by row.** `AclCache` follows `groups`, `group_members`, `roles` and
`role_assignments`, and a change evicts exactly the entries it reaches. It does not empty the
cache for a change (owner, 2026-10-08: "we cannot be invalidating entire caches for a change,
this has to be row by row or ALL if that is needed").

The row broadcast cannot do that today: it carries the primary key, which for
`group_members` is `(group_id, id)` and does not name the member. So a collection may declare
the columns its invalidation message carries beyond the key:

    invalidation_columns: ClassVar[tuple[str, ...]] = ("member_type", "member_id")

`CacheInvalidationMessage` gains one optional field for them, written from the row the write
saw (the row being deleted, for a delete), so a receiver needs no read to know what the row
was. A derived cache registers, per table, how a row maps to its own entries:

- `group_members`: the member. A person's row evicts that person. A group's row (a bundle
  nested in a state group) evicts every cached actor whose resolved groups include that
  group; `AclCache` keeps, for each cached actor, the groups its walk passed through, which
  it computes at read today and throws away.
- `role_assignments`: the group it grants to, by the same index.
- `roles`: every actor holding an assignment of that role, through the assignment index; a
  role whose reach cannot be bounded this way is the one routine case that may take ALL.
- `groups`: a delete evicts the actors resolved through it; the cascade to its members
  arrives as `group_members` rows once deletes carry their row (the hub's hand-written
  per-member announce in the group delete goes away).

**ALL is for when the reach is unknown, not for convenience:** a missed broadcast (the pass
finds a count unheard, so some row changed and the pod cannot say which), a replaced bucket,
or a change a cache has no index for. Those drop the table's cached rows and every entry
derived from it, as "The Model" describes. A heard change never does.

Whether `namespaces` belongs in the set was not confirmed. The eviction cost under a burst of
rbac writes (bootstrap, reconcilers, agent access materialization) is part of the
measurement, counted in entries evicted, not caches emptied.

**The `acl.*.invalidate` subjects** stay through expand and migrate and are removed at
contract: subscribers first, then `publish_acl_invalidation` and
`evict_after_rbac_write`'s publish half, then the grants. `ttl_seconds` goes with them, per
decision 5.

**A tool pod** follows the same four tables for its per-caller cache, by the same rule: a
person's membership row drops that person's entry, a group's row drops the callers resolved
through it. It needs a read of the four generation keys, and it already hears the row
broadcasts. It needs no epoch subject and no `acl.*` subject.

> **Corrected in the expand build.** This said the pod needs
> `JsResource.kv_key_read(f"{ns}-epochs", key=...)` for each of the four keys, "the shape it
> already holds on the data-versions bucket". That grant cannot be built for these keys.
> `kv_key_read` takes one subject token and refuses a dot (a unit test holds it to that), and a
> generation key is four tokens, `{ns}.collections.{table}.epoch`. Its read is also the direct
> get, which the epoch bucket is not declared for. The tool pod is granted what the agent pod
> already holds on this bucket, `JsResource.kv_bucket_keys(f"{ns}-epochs", writable=False)`:
> the whole bucket, read only. A grant of four literal keys would need a new capability.
>
> **Decided (owner, 2026-10-09): the tool pod's read of the whole bucket is accepted**, and no
> capability for four literal keys will be built. The values are counters with nothing secret in
> them, and every write stays with the hub.

## The L1 Max Age

Once a table carries a generation, a missed invalidation is detected at the next pass, which
is what epoch-task-05 was built for. Stated plainly, a max age would still cover only:

- a table with no generation (opted out, or `NO_L2`);
- a write that moves no generation: raw SQL outside a collection, a database cascade, a
  migration or backfill, a writer on an old release, a write whose advance failed.

Each item in the second line is, under decision 5, a gap to close in the epoch system, not a
reason for a timer. With the write-path inventory complete, a max age covers nothing a
generation-carrying table lacks.

**Decided (owner, 2026-10-08):** `set_l1_max_age` stays in place and off by default through
expand and migrate, takes no new callers, and is removed at contract together with the hub's
one existing caller. `ScanCache`'s
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
   `AclCache.ttl_seconds`, and the L1 max age and the scan TTL with the hub's one caller.

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
- A derived cache, row by row: a `group_members` write for one person on one registry evicts
  that person's entry in an `AclCache` on another, with no `acl.*` subscription bound, and
  leaves every other entry; nesting a group evicts exactly the actors resolved through it; a
  missed broadcast evicts all.
- **Enforcement.** Enumerate every `BaseCollection` subclass in the family: each carries
  the default or a `NoWriteGeneration` with a non-empty reason, and no two classes share a
  `table_name`. A second test fails when a bootstrap wires a generation source and
  schedules no pass. A grant test pairs each principal's epoch-bucket keys with the keys
  `EpochGenerationSource` opens, as epoch-task-01 did for the bucket.

> **Corrected in the expand build.** The second test is keyed on following, not on wiring a
> source. Stage 2 has every writer wire a source while following nothing, so "wires a source
> and schedules no pass" would fail each of them for doing what the stage asks. What is unsafe
> is following a table with nothing scheduled to judge the mark, and that is what
> `tests/enforcement/test_write_generation_declarations.py` refuses. The enumeration in the
> first sentence is built for every class in this repository; the hub's and the SDK's classes
> are out of its reach. See "Enforcement, and What Waits" at the end.

## Decided on the Open Questions (Owner, 2026-10-08)

1. **Who advances for a pod: the hub's broker,** after it commits the pod's write. No pod
   writes the epoch bucket, so a pod still cannot fake a bump. Whether the broker sees commit
   boundaries is the first thing to confirm in the build.
2. **A failed advance after a committed write raises.**
3. **The four access tables are followed by `watch_key`,** not a timed pass.
4. **The L1 max age and the scan TTL are removed at contract,** with the hub's one caller.
5. **Derived caches are invalidated row by row, or ALL only when that is needed.** Not
   `invalidate_all` on any change. See "Derived Caches".

## Still Open, After the Measurement

1. **The opt-out threshold:** the fraction of p99 write latency that puts a table on the
   list.
2. **The six undecided tables:** opt out, or split the per-event columns off.
3. **The generation key for per-agent tables.** The key carries the table name only, so
   every agent's `memories` would share one generation. Right for platform tables; for
   per-agent namespaces it is correct but noisy. Most such tables are on the opt-out list.

---

## Built in the Expand Stage

Built on `feat/collection-generations-expand`, 2026-10-08. Nothing in the hub, the SDK,
identity-core or a product repository changed.

### The Broker Sees Every Commit

Read from the hub (`aibots.hub.broker.proxy`, `statement_classifier`, `sql_inspection`) and
from `NatsProxyL3Backend`. Nothing in the hub was changed.

- **Three shapes of write reach the broker, and it ends each one itself.** `l3.query` is one
  statement, autocommitted, or run in a per-request transaction when row-level security is on.
  `l3.batch` is a list of statements, either in one transaction the broker commits when the
  last has run, or each on its own. `l3.tx.begin` opens a session the broker pins to one
  connection and one hub replica by `tx_id`; `l3.tx.execute`, `fetchrow` and `fetch` run in it,
  and `l3.tx.commit` or `l3.tx.rollback` ends it. In every shape the commit is a line of the
  broker's own code, with the outcome in hand.
- **It knows the tables each statement wrote, from the parsed statement.** Every statement on
  every door is parsed once by `classify_statement` before any grant is consulted, and a
  statement it cannot parse is refused. `ClassifiedStatement.targets` names each written table
  with its verb, including an `INSERT`, `UPDATE` or `DELETE` inside a CTE under a `SELECT`. So
  the set is as reliable as the parse, and nothing unparsed runs. What it cannot see: rows a
  trigger or a foreign-key cascade writes, and a table written inside a function.
- **A transaction session does not keep that set today.** `_TxSession` records only whether a
  non-owner wrote the agent's data (`wrote_agent_data`). `_admit_tx_statement` classifies each
  statement and keeps only the `AgentDataStatement` a non-owner's gate returns; the
  `ClassifiedStatement`, and its targets, go out of scope there. Advancing at `l3.tx.commit`
  needs the session to collect the written tables as its statements are admitted. A session the
  sweeper or shutdown ends goes through `force_rollback` and must advance nothing.
- **Where each commit is.** `l3.query`: `_execute_query`, where the RLS path leaves
  `conn.transaction()` and the autocommit path returns from `conn.execute`. `l3.batch`:
  `_execute_batch_transaction`, leaving its one `conn.transaction()`, and
  `_execute_batch_independent`, once per item. `l3.tx.commit`: `_complete_verified_tx`, after
  `session.commit()` returns; a refused commit raises there and is answered as such.
- **Names.** A target is the table as the statement names it, with a schema only when the
  statement qualified it; a bare name resolves through the namespace's `search_path`. The
  generation key carries the table name alone, so the broker's map is "written table name to
  key", the same for every namespace, which is "Still Open" 3.
- **The broker holds no generation source today.** `QueryProxy` has no reference to the epoch
  bucket. The hub's own principal may write it.

So "the broker advances" needs, in the hub: a generation source on `QueryProxy`; the written
tables kept per request and per session; one advance per table after each successful commit;
and the tokens sent back, because the pod stamps them on its row broadcasts. The wire changes
are under "What the Migrate Stages Need".

### What Exists Now

- **`GenerationSource.advance` returns the token it wrote** (`threetears.core.collections.
  generation`), and `EpochGenerationSource.advance` returns the value its own compare-and-swap
  put in the bucket. A source that returns nothing still works: the write advances and its row
  messages name no generation.
- **`CacheInvalidationMessage` has three more optional fields**: `generation`, `bump_rows`, and
  `columns` for the declared invalidation columns' values, as strings. A receiver one release
  back ignores them. A message without them evicts its row exactly as before and counts nothing.
- **The declaration**: `BaseCollection.write_generation`, one of `WRITE_GENERATION`,
  `NoWriteGeneration(reason=...)` or `WRITE_GENERATION_UNDECLARED`. Undeclared is the default in
  this stage and changes nothing. **Switching a table on is declaring `WRITE_GENERATION` on its
  class.** Stage 4 flips the one default on `BaseCollection`. `__init_subclass__` refuses
  `negative_cache_max_age` with an opt-out, and anything that is not one of the three. A
  `NO_L2` collection advances nothing whatever it declares; an explicit `nats_client=None` still
  advances. With no generation source on the registry nothing can advance, and a switched-on
  collection then writes as an undeclared one does.
- **`BaseCollection.invalidation_columns`**, a tuple of column names. Their values ride on the
  row message, from the row the write saw. A delete reads the row before it deletes it, from L3
  when there is one, and only for a collection that declares columns. A bulk delete
  (`SchemaBackedCollection.delete_rows`) does the same, one read per key, on the caller's
  transaction.
- **Every write path advances once per commit on a switched-on collection**: `save_entity`,
  `delete`, a won `l2_cas_mutate`, a subscript write, `invalidate_cache`,
  `invalidate_cache_many`, `bypassing_write`, `CallerTransaction` settling (once per collection,
  with every row message carrying the token and the row count), and `flush_pending` (once per
  table per flush, announcing each landed row again with the token). A failed advance raises
  `GenerationUnavailableError` after the rest of the path has run. A subscript write logs it.
- **The mark**: `GenerationMarks` in core, one per registry, reached through
  `CollectionRegistry.follow_generation`, `generation_marks`, `account_generation` and
  `settle_generation`.
- **The pass and the watcher** in `threetears.epoch.generation_tick`:
  `generation_catchup_tick(registry, reader)`, one pass per call, and
  `follow_generation_key(registry, reader, table)`, which follows one table by `watch_key`.
  Both read through `EpochGenerationReader`, which binds the bucket and never writes it.
- **`CollectionRegistry.drop_table`** and `BaseCollection.drop_cached_table`.
- **`CollectionRegistry.register_derived_cache(table, on_row=..., on_table_dropped=...)`.**
  `on_row` gets every row message for the table, peers' and this process's own. The cache is
  told to drop everything only when the table drops. `AclCache` is not converted.
- **The registry refuses two collections for one table that disagree on `write_generation`.**
- **Grants**: a tool pod reads the epoch bucket, as an agent pod already did. No pod writes it.

### Decided in the Build

**Switched on is a class declaration, not a registry flag.** The illustration under "The
Opt-Out" shows `WRITE_GENERATION` as the default; that is stage 4. Until then the default is a
third declaration that says nothing was decided, so a collection that bumps is one whose class
says so, and the flip is one line.

**A pod's first sight of a table drops it.** A registry that starts following has no mark, so
it cannot vouch for anything it cached before. The first pass records the generation and drops
the table once. A cache that has just started is empty, so this costs nothing where it matters.

**A new incarnation always drops, including the first one.** A follower that saw no generation
for a table, and then sees the first one a writer mints, drops the table once more even if it
heard every row. Two bucket losses between passes could otherwise hide writes under the lost
incarnation. It happens once per table per bucket lifetime.

**The watcher gives broadcasts time.** A writer advances and then publishes, so the broker
pushes the new generation ahead of its rows. Judged at once, every heard write would read as a
missed one. `follow_generation_key` waits up to `grace` (two seconds by default) for the rows of
an advance before it calls them missed. A generation under a new incarnation is not waited on.
This is a wait before a drop inside the follower, not an age on any cached value.

**What a watch cannot see.** The broker pushes nothing when the bucket is emptied. A watcher
learns of a lost bucket at the table's next advance, which arrives under a new incarnation. An
advance whose push and whose broadcasts were all lost in one restart is seen then, and not
before. A pod for which that window matters also runs the pass. Decision 3 stands; this is its
limit, stated.

**`invalidate_cache` advances too.** "Write paths that do not advance today" lists
`invalidate_cache_many` and not the single-key form. Callers use the single form to announce a
row changed by SQL the collection did not see, so it is a write path. The write paths' own use
of it, to withdraw a row whose L3 write did not land, goes through a private form that does not
advance.

**A settled transaction advances whichever way it ended.** `CallerTransaction` does not know
whether the commit succeeded, and evicts either way for that reason. It advances either way too.
A needless advance after a rollback is the safe direction.

**A flush announces its rows with the writer's L2 entry marked current.** L2 took each row when
it was saved, so peers sharing the scope keep the entry instead of deleting it.

**A failed advance in a flush raises from `flush_pending`,** after every table has been
attempted and every landed row acknowledged, so nothing is replayed for it. The coordination
flusher (`PeriodicFlusher`) logs it as what it is: the rows were written, and the generation
did not move. On an interval there is no caller to raise to, so the loop carries on, as a
subscript write does. The final flush in `aclose` raises it to whoever closes the flusher
(`CollectionRegistry.close_collections` logs it and closes the rest). Another caller of
`flush_pending` must expect it once it switches on a write-behind table.

**`save_entity(conn=)` stays refused on a collection that caches absences,** and so do subscript
writes. "Batching" says settling after the transaction lifts the reason for the first. It does,
but lifting it changes what an existing collection accepts, and nothing in this stage needs it.
It is left for the stage that switches such a table on.

**A derived cache that raises does not cost the row its eviction.** The row is evicted, the
failure surfaces, and the message is not counted as heard, so the next pass drops the table.

### The Table Drop and L2

**Decided: a table drop must stop trusting the pod's own L2 entries for the table, and it
does.** Without that the drop heals nothing. L2 keys are per principal
(`{scope}.{table}.{body}`), so a writer's save touches only its own key. The broadcast is what
makes a peer delete its key, and a table drop is what happens when that broadcast was lost.
The peer's L1 row goes in the drop, its next read pulls through, L2 is read first, and the
stale row is cached again. `test_table_drop_and_derived_caches.py` holds this: without the
mechanism its reader is served the stale row again after the drop.

**How.** The entries cannot be deleted as a set: that needs a listing of the bucket, and a
pod's grant on the collections bucket carries no consumer. So after a drop each key is
distrusted until this process has read it through from L3 once. A live L2 row for a distrusted
key is read past, as an expired row already is, and replaced by the L3 row at the revision it
was read at, so a writer's newer value still wins. If L3 no longer holds the row, the stale
entry is deleted at that revision. The key is trusted again once the entry has moved on, unless
another drop landed while it was being read through. A read or write in flight when the table
drops does not cache what it read. The record of keys trusted again since the last drop is
bounded (`BaseCollection.L2_READ_THROUGH_LIMIT`, 10,000 keys by default). Past the bound the
oldest key is forgotten, which costs it one more read through L3 and nothing else.

**A row withdrawn because its write did not land advances nothing.** A subscript write whose L3
write raised, or that the store refused, takes its row out of L1 and L2 and announces it with no
generation: nothing committed. One that committed and could not be read back is withdrawn the
same way and carries its advance. The public `invalidate_cache` always advances, because its
callers use it for a row changed by SQL the collection did not see.

**Where it does not apply.** A write-behind table and a table whose rows a compare-and-swap
orders keep L2 ahead of L3 on purpose, so reading L3 there would put an older row over a newer
one. A collection with no L3 has nothing to read through from. For those a drop removes L1, the
scans and the derived caches, and leaves L2 trusted. A missed broadcast on such a table can
therefore still be re-read from the pod's own stale L2 entry. Most of them are on the opt-out
list; any that is switched on and followed needs this closed first.

### Enforcement, and What Waits

Built, in `tests/enforcement/test_write_generation_declarations.py`: a `write_generation` is
one of the three declarations and an opt-out's reason is a non-empty literal written where it
is declared; a switched-on class publishes no row message of its own without its advance; a
module that follows a table schedules the pass or the watcher.

**The enumeration, for this repository.** The same test imports every module of every package,
in a process of its own (`tests/enforcement/_collection_census.py`), and walks every
`BaseCollection` subclass: 56 classes today. Each carries one of the three declarations, an
opt-out with a reason. The classes that name one table, whether by a `TableSchema` or a
`table_name` property, descend from one class that names it, and they all declare the same
thing. A subclass that adds queries is the same class for this purpose, the shape
`HubGroupMemberCollection` has. A concrete class whose table is named per instance is listed
with why: `TileCollection`, `FeatureCache` and the coordination tables' shared base. The family
passes today, so nothing in this repository has to change before the migrate stage.

**What it cannot reach.** The hub's and the SDK's classes for `playbook_entries` and `concepts`
live in other repositories, and no check here sees them. They do not disagree today, because
every class is undeclared. They can only start to disagree when one of them is switched on. So
the one-class cleanup goes before the first of these tables is switched on, as "One Class per
Table" says, and each of those repositories runs the census over its own classes. At run time
the registry refuses two disagreeing collections for one table, which covers any pair that
shares a process.

`packages/epoch/tests/unit/test_generation_grants.py` pairs the minted grants with the keys the
source and the reader open. `packages/epoch/tests/integration/test_generation_grants_live.py`
runs the reader and the source under the minted grants, on a live broker, as an agent pod and
as a tool pod. Each pod binds the bucket, reads, and is pushed a generation through
`watch_key`, and its own advance is refused.

**Write paths in subclasses that the base class does not see.** These publish a row message from
a method of their own and advance nothing: the presence rooms (`threetears.channels.presence.
collection`), `HeartbeatCollection`, the tool collections in `threetears.agent.tools.
collections`, and `ObjectResolutionCollection`. None is switched on. The enforcement test fails
any of them that is switched on before its own publishes carry an advance.

### Test Evidence

Run at `823f5689` on 2026-10-09 with the workspace's locked tools, serially:

- every unit suite, `pytest packages/ tests/ -m "not integration"`: none failed. None of the skips
  is in core, epoch or nats except one baseline enforcement skip. Recorded as this repository's
  test evidence (`prawduct-hook test-evidence record`).
- integration, `-m integration -rs` over core, epoch and nats, against docker: none failed, no
  skips.
- `ruff check`, `ruff format --check` and `mypy` (868 files) clean.

Each test added for review `rev-20261009T010436Z-d86af247` was checked by breaking the code it
names and watching it fail:
- each bump return in `_persist_cas_result`, and the raise after a failed swap advance;
- each of `_distrusts_l2`'s three exclusions;
- the guard against re-trusting a key during a second drop;
- the bump on a withdrawn row that committed, and routing either failure withdrawal through
  `invalidate_cache`.

### What the Migrate Stages Need

**Writers (stage 2).**

- *Hub, broker.* A generation source on `QueryProxy`. The written tables collected per request,
  and per `_TxSession` as statements are admitted. One advance per written table after the
  commit of `l3.query`, of each `l3.batch` transaction or item, and of `l3.tx.commit`. Never
  after a rollback.
- *Wire, reply.* A new optional field on `L3QueryResponse`, `L3BatchResponse` and
  `L3TxCompleteResponse` carrying the token written for each table. The 3tears client reads
  replies as dictionaries, so an old pod ignores it. Expand: the hub sends it. Migrate: pods
  read it. Nothing to contract.
- *Wire, request.* The broker does not know which tables are switched on, so either it advances
  every table it sees written, hot ones included, or the pod names the tables to advance in a
  new optional request field, which the broker checks against the tables it parsed. The hub's
  request models refuse unknown fields (`extra="forbid"`), so the hub must accept the field one
  release before any pod sends it.
- *Wire, failure.* A commit that succeeded with an advance that failed needs its own reply
  code, so the pod raises `GenerationUnavailableError` and does not retry the write.
- *3tears.* A `GenerationSource` for a pod whose `advance` returns the token the broker's reply
  carried for the commit just made, with `NatsProxyL3Backend` surfacing it. The collection code
  built here calls `advance` after the commit and needs no change.
- *Hub, its own writes.* `set_generation_source(EpochGenerationSource(nc))` on the hub's
  registry after `configure`. The hub writes Postgres directly, not through the broker.
- *Hub, minted grants.* Pick up the tool pod's read of the epoch bucket by taking this release.
  `aibots.hub.security.static_nats_grants` resolves each static pod user through
  `build_permissions`, so it follows without an edit there.
- *SDK.* The pod-side source wired in `build_three_tier_stack`, `build_owner_data_stack` and
  `ToolServerBootstrap.install_collection_stack`.
- *identity-core.* Nothing new to wire: it already sets `EpochGenerationSource`, and the hub's
  static grants give its user the epoch bucket. Its collection classes declare
  `WRITE_GENERATION` when their tables are switched on.
- *Every repository.* `write_generation = WRITE_GENERATION` on each class to switch on, the same
  on every class for a table. The four access tables first.

**Readers (stage 3).**

- *Hub and agent pods.* `follow_generation` for each followed table, an `EpochGenerationReader`,
  and `follow_generation_key` per access table as decided, with `generation_catchup_tick`
  beside it where the watch's limit matters.
- *Tool pods.* The same, for the four access tables behind the per-caller cache.
- *`AclCache`.* Registered with `register_derived_cache` for `groups`, `group_members`, `roles`
  and `role_assignments`, with the indexes "Derived Caches" describes, and `invalidation_columns`
  declared on `GroupMemberCollection` and the others so a row names its member.
- *Per table, recorded.* The releases after which every writer of the table advances. No
  follower reads "the generation did not move" as "nothing changed" before that.
- *Enforcement in each repository.* The rule that a module which follows schedules a pass, run
  over that repository's own bootstraps.

**For the owner.**

- ~~Whether a tool pod's read of the whole epoch bucket is acceptable.~~ **Decided (owner,
  2026-10-09): accepted.** `JsResource.kv_bucket_keys(f"{ns}-epochs", writable=False)` stays; no
  capability for literal multi-token keys is built. The values are counters with nothing secret in
  them, and writes stay with the hub.
- Whether the epoch bucket should be declared for direct gets. Only the narrower grant needed it
  for a read, and that grant is not being built, so this falls away; a watch needs neither.

---

## Built in the Migrate-Writers Stage

Built on `feat/generations-migrate-writers` in 3tears, the hub (`14-eng-ai-bot-reports`) and the
SDK (`14-eng-ai-bot-agents-reports`), 2026-10-09. No table is switched on outside tests.

### What Exists Now

- **The hub's broker advances.** `QueryProxy` takes a `generation_source` (the hub passes the same
  `EpochGenerationSource(nc)` its own registry is wired with, after `configure`). After each
  commit it advances, once per table, every table the commit wrote that is switched on:
  `l3.query` after `_execute_query` returned (the RLS transaction left, or the autocommit), each
  `l3.batch` request after its one transaction committed or after its statements ran one by one,
  and `l3.tx.commit` after `session.commit()`. `_TxSession.written_tables` collects the tables of
  every statement admitted to the session (`_admit_tx_statement`). A rollback, a refused commit,
  a refused statement, a rolled-back batch, and a session the sweeper or shutdown force-rolls back
  advance nothing. The written tables are `ClassifiedStatement.targets` that are not `select`
  (`written_tables`), so a write in a CTE under a `SELECT` counts; a trigger, a cascade or a
  function's write does not.
- **The reply names the tokens.** `L3QueryResponse`, `L3BatchResponse` and `L3TxCompleteResponse`
  gain `generations` (table to token), `generations_failed`, and, on the two that lacked them,
  `error_code` and `error_message`. All optional; absent unless a switched-on table was written.
  The field names and the code are spelled once, in `threetears.core.backends.broker_generation`
  (`GENERATIONS_REPLY_FIELD`, `GENERATIONS_FAILED_REPLY_FIELD`, `GENERATION_UNAVAILABLE_ERROR_CODE`),
  and the hub imports them.
- **A pod's source.** `threetears.core.backends.BrokerGenerationSource`. `NatsProxyL3Backend` hands
  every reply that ends a commit to `record_reply_generations` (a successful `l3.query` or
  `l3.tx.commit`, and every `l3.batch` reply, a statement-by-statement batch that failed partway
  included); `advance(table)` returns the token the broker's reply carried for that table, as often
  as that commit's settling asks. `current` reads through an optional `GenerationReader`
  (`EpochGenerationReader` satisfies it) and raises when there is none or the table has no
  generation yet, because a pod cannot mint one. Without a reader the source says it cannot read
  (`reads_generations`), and absence caching stays off (see "Reading and Advancing Are Two
  Capabilities").
- **Wired everywhere.** Every SDK registry is built by
  `aibots_agents.runtime.broker_registry.broker_collection_registry`, which sets
  `BrokerGenerationSource()`: `build_three_tier_stack`, `build_owner_data_stack` and the devx
  workspace runtime (`DevxWorkspaceRuntime.connect`), and an SDK enforcement test refuses a
  registry built anywhere else. The framework's `build_tool_pod_collection_stack` (which
  `ToolServerBootstrap.install_collection_stack` and `ProviderToolPod` reach) sets it too. Every
  hub-family registry takes its source from `aibots.hub.common.generation_sources` (see "Every
  hub-family registry has a source"). identity-core already wires `EpochGenerationSource`. Stage 3
  gives the pod source its reader in those two places.
- **How each side knows a table's writes advance: from the one class.** The pod: its collection
  class, which calls `advance` when it declares `WRITE_GENERATION` or caches absences
  (`negative_cache_max_age`). The broker: `threetears.core.collections.tables_with_write_generation()`,
  the tables named on every live collection class imported in the hub process that does either,
  read through `table_named_by_class`, the one derivation the census uses too. No request field, so the hub's
  `extra="forbid"` request models are untouched.
- **One class per table, for the two tables that had three.** See "Playbook Entries and Concepts".
- **The census ships.** `threetears.enforcement.collection_census` (`run_census`,
  `find_census_problems`) replaces `tests/enforcement/_collection_census.py`. With
  `framework=True` it imports the installed 3tears packages too and marks their classes, so a
  product's census catches a class of its own for a table the framework already has a class for.
  The hub and the SDK each run it (`tests/enforcement/test_one_class_per_table.py`); run over the
  hub's and the SDK's trees before this stage, it reports exactly the `playbook_entries` and
  `concepts` duplicates.

### Decided in the Build

**A failed advance is answered as a success.** The note asked for "a distinct reply code" so the
pod raises and does not retry. A failed reply (`success: false`) would not do that for a pod one
release back: it reads every failed reply as `DataLayerUnavailableError`, which callers retry, and
the write has already committed. So the reply stays `success: true`, carries its rows, and adds
`generations_failed` and `error_code: GENERATION_UNAVAILABLE`. An old pod ignores both; a new pod's
collection raises `GenerationUnavailableError` from its `advance`, after the rest of its write path
ran, exactly as over the epoch bucket. The broker logs the failure, and never turns an advance's
failure (of any kind) into a failed commit.

**Which commit a token belongs to: the calling task's own.** The record is a context variable.
Two tasks writing one table each get their own token. A task started while one is held (the
`asyncio.shield` around `CallerTransaction._settle`) reads its starter's tokens, because they share
the record object; a task that makes a write request of its own starts its own record and leaves
the one it inherited as it was.

**One commit, one advance per table, with one count (verify of the owner's 2026-10-09 rule).** The
owner's rule was that every advance of a table for one commit is handed that commit's token. The
verify (rev-20261009T044338Z-669a118d) showed what that cost: each advance stamped its own row
count, so N advances under one token published N groups each claiming `bump_rows` of its own
group, and a follower counted the advance complete after the first group and moved on before the
others landed. So one commit's rows are settled in one advance that stamps them all with their
total: `AgentSkillCollection.bump_use_count` evicts its rows in one `invalidate_cache_many`, and
`CallerTransaction._settle` shares one advance (`SharedAdvance`) among the collection instances of
one table, the first advancing for the rows of all of them. The token is handed out once; a second
advance of the table for the same commit raises `GenerationUnavailableError` rather than publish
rows under a count already given, and its rows are still evicted and broadcast, naming no
generation.

**A token belongs to the commit that produced it.** Every reply that ends a write (an `l3.query`
whose statement writes, an `l3.batch`, an `l3.tx.commit`) replaces the task's record, naming
generations or not, so an advance after a commit whose reply named nothing for the table raises
`GenerationUnavailableError` (the broker named nothing, whose cause is the broker's) instead of
being handed an earlier commit's token. It is not `GenerationNotCommittedError`: that one says the
commit landed nothing and is logged at INFO, and this commit landed. A read's reply ends no write
and leaves the record. A rolled-back transaction (`tx.rollback`, sent by the pod, including
the acquire exit's safety net), a refused commit, and a commit whose request raised before any
reply came back (a timeout, a closed client) leave a record that says so. A collection settling
after any of those is not handed an earlier commit's token: its advance raises
`GenerationNotCommittedError`, a `GenerationUnavailableError` that says the commit landed nothing,
which the collection logs at INFO rather than as a failed advance. A table the broker named nothing
for raises the plain error, whose cause is the broker's.

**A statement-by-statement batch advances once per table, not once per statement.** "What the
Migrate Stages Need" said "each l3.batch transaction or item". The pod gets one reply and stamps
one token per table on the rows of its batch, so a second advance for a second item would be an
advance no row names, and every follower would read it as missed. One advance after the batch,
for every table a committed item wrote, says the same thing: a write committed since any token
read before it.

**Several commits before one advance.** A collection that writes a table twice before it advances
once (a `bypassing_write` body of two autocommitted statements) is handed the later token; the
earlier advance has no row, and a follower drops the table once. The safe direction.

**An autocommitted write refused for too many returned rows still advances.** Without RLS the
statement committed before `RESULT_TOO_LARGE` was raised; under RLS the refusal rolled it back.

**A session's written tables are collected when a statement is admitted**, not after it ran. A
statement that then fails aborts the transaction, so its commit lands nothing; an advance for it
would be needless, the safe direction.

**Reading and Advancing Are Two Capabilities.** A collection that caches absences needs a
generation it can READ to stamp them with; a switched-on collection needs one it can ADVANCE. A
source declares the first with `reads_generations` (absent means it reads), checked through
`source_reads`; `CollectionRegistry.readable_generation_source` is the registry's source when it
reads. Every absence-caching check, including the refusal at construction and the advance an
absence-caching class makes, uses the readable source; the switched-on advance uses any source. So
a pod's reader-less `BrokerGenerationSource` leaves `CoordinationRevocationsCollection` (through
`RevocationGuard`) exactly as with no source, and nothing on a pod caches absences until stage 3
wires a reader.

**The broker's advance, after review.** Only a success reply is stamped: a failed reply (a
statement-by-statement batch whose later item was refused) keeps its own `error_code`, so a
constraint violation still reaches the pod typed and is not retried as unavailability; its
advances are still made and logged. The advances share the time left before the pod stops
waiting for the commit's reply (`ReplyDeadline.seconds_left_to_answer`), and a table not reached
in it is named failed, so a stalled epoch bucket neither turns a committed write into a timeout
the pod retries nor holds an in-flight slot. The failure is logged with the door, principal,
namespace, correlation id and session. A transaction session records how it ended (open,
committing, committed, rolled back): a commit asked of a session that shutdown, the sweeper or a
fence refusal rolled back while the commit waited is refused (`TX_SESSION_CLOSED`) and advances
nothing.

**Every hub-family registry has a source.** `aibots.hub.common.generation_sources` picks it by how
the process writes: directly with the epoch bucket's write (the hub, the gateway):
`EpochGenerationSource`; through the broker: `BrokerGenerationSource`; directly without the grant
(the agent router, the dataset executor, the channel adapters, an operator's audit replay): a
source that reads nothing and refuses every advance at once, naming the process. A table one of
those processes writes cannot be switched on until the process is granted the bucket's write; the
refusal says so rather than letting an ungranted JetStream call block to its deadline. An
enforcement test holds every `CollectionRegistry()` in the hub to a source from that module.

**The hub's knowledge subclasses inherit the framework's agent reads (accepted).** The framework's
`PlaybookEntryCollection` and `ConceptCollection` carry the agent pod's reads
(`list_visible_to_user`, `list_own_drafts`, `fetch_embeddings`, over the rbac proxy pool with
`customer_scope`) beside the table's declaration, so the hub's subclasses inherit reads that do
not run on the hub's pool. The hub's own reads are named `list_entities_visible_to_user` and
`list_own_draft_entities`, and every hub docstring names those. Splitting the framework class into
a declaration base and an agent-pod subclass would let the hub keep the shorter name; it is a
3tears change with no behaviour in it, and is left for the stage that touches those classes again.

**Pods read no generation yet.** The SDK and `3tears-agent-tools` do not depend on `3tears-epoch`,
and adding it is a lock change this stage does not need: nothing on a pod caches absences.
`BrokerGenerationSource()` is built without a reader, so `current` raises and a collection that
caches absences on a pod trusts none. Stage 3 adds the dependency, because a follower needs
`EpochGenerationReader` anyway, and passes the reader in.

**What a reply that names no generation means.** A switched-on pod collection whose commit's reply
names no token for its table raises `GenerationUnavailableError`, not `GenerationNotCommittedError`: the hub is older than this stage,
has no source, or has not imported the class that switches the table on (a class whose table is
named per instance cannot be read off the class at all). Loud, never a claimed advance. One gap is
not loud: during a rolling hub upgrade, a token left unclaimed from a new replica's reply can be
handed to a later commit an old replica served. That commit advanced nothing, which is what any
old writer does, and the stage 3 rule already covers it: no follower trusts an unmoved generation
until every writer of the table is past stage 2.

### Playbook Entries and Concepts

Compared column by column (name, type, nullability, immutability, default, primary key,
`cas_column` and every other schema attribute and class variable):

- **`concepts`**: the framework's, the hub's and the SDK's schemas are identical.
- **`playbook_entries`**: the SDK's is identical to the framework's. **The hub's has one more
  column, `enforcement`** (nullable JSONB, migration v020, query-enforcement-task-01), which the
  hub's routes write and its governance scan reads. The framework's class leaves it out, as it
  leaves out `embedding`. So `HubPlaybookEntryCollection` declares the framework's schema plus
  `enforcement` (`dataclasses.replace` of the framework's, so nothing else can drift), and the
  pods' projection is unchanged. Adding the column to the framework's class would hand it to every
  agent's read; that is the owner's call, not this stage's.
- **Entities**: all six were bare `BaseEntity` subclasses with `primary_key_field = "id"`. The hub
  and the SDK now use the framework's, and the SDK's `DraftView` (identical fields) is the
  framework's.

The hub's `HubPlaybookEntryCollection` and `HubConceptCollection` and the SDK's
`AgentPlaybookEntryCollection` and `AgentConceptCollection` subclass the framework's classes and
inherit schema, table name and declaration. The hub's two reads that shared a name with the
framework's agent reads but not their signature (an entity list, no `customer_scope`) are renamed
`list_entities_visible_to_user` and `list_own_draft_entities`, so the subclass does not break the
contract the framework's callers rely on. The SDK's subclass overrides only
`list_visible_to_user`, which refuses a named datasource and reads uncached; its `list_own_drafts`
and `fetch_embeddings` were the framework's line for line and are inherited.

### What Stage 3 Needs

- *Per table, recorded.* The releases after which every writer of each access table advances: the
  hub (direct, through its registry), identity-core, agent pods and tool pods (through the
  broker). Only after all of them are on this stage may a follower read "did not move" as
  "nothing changed".
- *Switch the four on.* `write_generation = WRITE_GENERATION` on `GroupCollection`,
  `GroupMemberCollection`, `RoleCollection` and `RoleAssignmentCollection` (the hub's `Hub*`
  subclasses inherit it), with `invalidation_columns` declared so a row names its member; the hub
  must import those classes, which it does.
- *Readers.* `3tears-epoch` as a dependency of the SDK and `3tears-agent-tools`;
  `BrokerGenerationSource(EpochGenerationReader(nc))` in the three pod stacks; `follow_generation`
  and `follow_generation_key` per access table, `generation_catchup_tick` where the watch's limit
  matters; `AclCache` registered with `register_derived_cache` and its indexes.
- *The write paths the base class does not see* (listed under the expand stage) before any of those
  tables is switched on.
- *Open:* whether the framework's `PlaybookEntryCollection` should carry `enforcement`; a hub
  table written only by pods whose class the hub does not import is never advanced, which the
  pod's advance reports, but no test enumerates such tables yet.

### Test Evidence

Run on 2026-10-09 with each main checkout's locked tools and the three worktrees first on the path,
serially (the hub's sets at `-n 4`):

- 3tears, after the review fixes (`54a1a221`), serially: the unit suites of the packages this stage
  touches (core, epoch, agent tools, enforcement, and `tests/enforcement`), and the integration
  suites of core and nats against Docker, none failed and no integration test skipped; recorded in
  the evidence store (`prawduct-hook test-evidence record`). `ruff` and `mypy` (872 files) clean.
- Hub, `tests/unit tests/enforcement -m "not integration"`: all passed after rebasing on
  `feature/reports`. `ruff` clean; `mypy src` clean.
- SDK, `tests/unit tests/enforcement`: all passed. `ruff` and `mypy src` clean.

Each new test was checked against the code it names: the pod-side tests fail with the proxy's
recording removed, with either rollback's forgetting removed, with the unanswered commit's
forgetting removed, with the per-task owner check removed, with a token removed on its first take,
and with the readable-source gate removed; the broker's fail with each of
the seven advance sites or conditions broken; the wiring tests fail with the wiring removed; the
census tests report exactly the old duplicates when run over the hub's and the SDK's trees as they
were before this stage.
