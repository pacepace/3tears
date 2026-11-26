# Running the change log's recovery, and L2-first collections

The epoch change log (`{ns}-epoch-changes`) is in NATS, memory storage, three replicas. It is the only
clock and the only catch-up path. To survive NATS losing it, every table carries its own **epoch**: a versioned
table on every version it appends, a stamped table on the one row a key it updates. One **log keeper** process
marks the log's position every 30 s and, after a loss, puts the log back from the tables: every version and stamped
row written since a mark. Followers never read the tables to catch up; they walk the log.

An **L2-first** collection commits each write in NATS (one epoch record in the change log, carrying the
row), then a saver makes L3 match.

This page is for whoever wires, runs and watches it. The design is in the module docstrings of
`threetears.core.collections.log_keeper`, `threetears.core.collections.table_restore` and
`threetears.core.collections.l2_first`. The process that runs the keeper installs `3tears[arrow]`: a restore folds
what it reads as Arrow.

## Facts every section uses

| What | Value |
|---|---|
| Change-log keys | `{tenant}.{table}.e` (`threetears.core.tenancy.epoch_change_key`): `s-{schema}` for an agent-schema table, `v-{value}` for a platform table's declared tenant column (a digest `h-...` when the value is not one subject token), `_all` for a platform-wide table, a platform table's whole-table or keys-unknown change, or an undeclared table. An agent-schema table's whole-table or keys-unknown change goes under its schema's `s-` key, named by the writer (`advance_with_change(tenant=...)`) |
| Tenancy | declared on the collection or its `TableSchema` (`tenancy = TENANT_BY_SCHEMA`, `tenant_column("customer_id")`, `no_tenant(reason)`); never guessed. The collection census (`find_census_problems`, run by the framework's and every consumer's `test_one_class_per_table` gate) fails on a concrete class with no declaration. At runtime an undeclared table still goes on `_all`, so nothing is lost, with one WARNING per table (`collection class declares no tenancy`) and `threetears_collection_tenancy_undeclared_records_total{table}` per record |
| Who reads what | a platform service (hub, registry, gateway) reads every tenant (`EpochGenerationReader(nc)`, the wildcard `*.{table}.e`). A pod reads by each table's declared tenancy: `EpochGenerationReader(nc, tenants=PodTenants(schema=..., customer=...))` reads a schema table under `s-{its schema}` only, a platform table with a tenant column under `v-{its customer}` and `_all`, any other under `_all`: exactly the keys the grants its host names cover (`build_permissions(epoch_change_grants=...)`). `PodTenants` is the only form a pod's reader takes |
| Row values | `ChangedRow.value` (an L2-first commit's row) rides under its tenant's key, so it is readable only by whoever may read that tenant, as in the collections bucket |
| Keeper | `LogKeeper(pool, schema=..., log=source)`, run under its lease by `LogKeeping(keeper, nats, lease=...)` (`start()` once, `claim_and_keep(holder)` every `LOG_KEEPER_CLAIM_EVERY`, 15 s) |
| Keeper alive | `threetears_change_notes_keeper_last_pass_timestamp_seconds`, set by every whole keeper pass: alert when its max across processes is older than 30 min |
| L2-first worker alive | `threetears_l2_first_worker_last_pass_timestamp_seconds`, set by every whole worker pass: alert when its max across processes is older than 10 passes |
| Marks | `threetears_change_note_marks` in the keeper's schema: the log's position every 30 s, and at once whenever the keeper takes a log. Kept for the log's whole window (the newest mark before the log's oldest record, and every one after), never by age; `threetears_change_log_marks_dropped_total` |
| The log the keeper knows | the incarnation its newest mark names; no ring row holds it |
| Restore reads | every versioned table's versions and stamped table's rows with `epoch >= X`, X = (T - M) << 15 from the lost log's oldest mark (T its time, M the margin: 10 min) |
| Every source of the deployment | `EpochGenerationSource(..., restored_from_l3=True)` |
| Follower quiet pass | every `quiet_pass_interval()`: the log's retention (24 h) / 4 = 6 h |
| Unrestorable records | `threetears_change_notes_missing_total{table}`: a record of a table that carries no epoch, which a restore cannot read back; alert on any increase |
| Margin violations | `threetears_restore_margin_violations_total{table}`: a record after a mark whose lowest stamp is below the mark's X; alert on any increase |
| Restores | `threetears_change_notes_restores_total`; `threetears_restore_unvouched_total` (no mark of the lost log: every follower behind) |
| Restore truncated | `threetears_change_notes_restore_truncated_total`: only the changes after a later mark fit; followers older than its `covered_from` are behind; alert on any increase |
| Restore refused | `threetears_change_notes_restores_refused_total`: not even the changes after the newest mark fit; alert on any increase |
| A second database's part | `threetears_change_notes_put_backs_total`; `threetears_change_notes_put_back_short_total{reason}` (`too_large`, `floor_after_label`): alert on any increase |
| Scrub lag | `threetears_change_notes_scrub_behind` (records the log holds after the scrub's position) |
| Quiet marks moved | `collection_epoch_quiet_marks_moved_total{table}` |
| Pods behind | `collection_epoch_tables_behind`; `collection_epoch_behind_total{reason}` |
| L2-first refusals | `threetears_l2_first_l3_refused_total{table}`: alert on any increase |
| L2-first records missing | `threetears_l2_first_records_missing_total{table}`: alert on any increase |
| L2-first refills | `threetears_l2_first_refills_total{table}`, with a WARNING per refill |
| Change log fill | `threetears_epoch_changes_fill_ratio`, warn at `EPOCH_CHANGES_FILL_WARN_RATIO` (0.75) |
| New env vars or settings | none in 3tears; see "What the hub adds" |

## Tenants: who reads which change records

Every record is keyed by tenant first (see the facts table). A writer splits one commit's rows by tenant: each
row's `ChangedRow.tenant`, named by the collection from its declared tenant column, or by the writer that knows
the row's schema (the hub's broker, for an agent-schema table, through `advance_for_write(source, table, rows, schema=...)`, which applies the table's declaration). A whole-table or keys-unknown change carries no
row: an agent-schema table's goes under the schema the writer names (it belongs to that schema alone), a
platform table's on `_all`. A restore names each record's tenant from the table's declared tenancy (its schema, or
its tenant column, by the same token rule), so it puts each tenant's records back under its own key. A record of a schema table that names
no schema still goes on `_all`, where no pod reads that table, and is said once per table and counted
(`threetears_epoch_change_schema_record_untenanted_total{table}`).

The access machinery (groups, group_members, roles, role_assignments, namespaces) is `no_tenant`: every pod's
access evaluator reads those rows across customers by id, so every follower must see every customer's change,
and a pod keyed to its own customer would decide access on a row it never evicted. An agent's schema is
`threetears.core.namespaces.build_agent_schema_name(agent_id)` (`agent_<hex>`), the one spelling the broker,
the callout and the pod's `PodTenants(schema=...)` share. A tool pod's provider space (`tools.<provider>`) lives
in `threetears.core.namespaces.build_namespace_schema_name(namespace_id)` (`ns_<hex>`, from the namespace row's
id), and its collection stack (`build_tool_pod_collection_stack(tenants=...)`) reads as
`PodTenants(schema=<that schema>)`: `s-ns_<hex>.<table>.e` for its own tables, `_all.<table>.e` for the access
machinery. A tool pod with no space reads as `PodTenants()`: `_all` keys only, and a schema table is refused.
A tool pod never reads through the `*.<table>.e` wildcard, which a per-tenant grant cannot admit.

A host names its pod's tenants by overriding `ToolServerBootstrap.collection_tenants(server, nats_client)`
(default `PodTenants()`), which the base connected callback awaits on the live connection before it builds the
stack. The SDK's `ProviderToolPod` overrides it: it runs the hub handshake there and returns
`PodTenants(schema=<the schema the handshake names>)`. That needs hub `d10d71d36` or later, the first hub whose
handshake reply names the space's schema and whose grant names `s-ns_<hex>.>`; against an older hub the pod
reads as `PodTenants()`. A pod that owns two or more spaces has no one schema to read as, so it runs as
`PodTenants()` too: its own tables have no readable generation (a documented limit).

A pod's reader picks each table's keys from the table's declared tenancy (`PodTenants.keys_for`), so the pod
asks only for the keys its grant names. A key outside the grant is refused by the broker, which answers the
read with nothing: the first such read waits to its deadline, then the reader remembers the table as refused
(`ChangeLogKeyRefusedError`) and answers later reads of it at once, until a new credential (a reconnect, a
renewal) clears the client's record of the refusal and the reader asks again. A collection logs the refusal at
ERROR once per table, naming the keys, the tenants and the fix, and counts each read it answers from L3
(`threetears_collection_change_log_refused_reads_total{table}`); a real outage stays a rate-limited warning.

A follower keeps one mark per table, and walks each key it may read after it, merged by the log's sequence.
Detection is the same per key as it was per table: the log's first sequence (`aged_out`) and its incarnation
(`replaced`, `record_lost`). An interior record of a live log cannot vanish: the stream denies deletes and
refuses a full append. A pod's walk and watch use named consumers filtered on its keys (and its newest record a
direct get by subject), so its grant can name exactly those keys; it holds no message get by sequence on this
log.

What a pod's host grants (`EpochChangeGrant`):

- its own agent schema, whole: `EpochChangeGrant(tenant=schema_tenant(<its schema>))` (`s-...`, every table in
  it is its own);
- for a tool pod that owns a provider space, that space's schema, whole:
  `EpochChangeGrant(tenant=schema_tenant(build_namespace_schema_name(<the space's namespace id>)))`
  (`s-ns_<hex>`, every table in it is the pod's own), which the pod reads as `PodTenants(schema=<that schema>)`;
- for each platform table its role already reads (from the same source that grants it those rows): the table
  for its customer, `EpochChangeGrant(tenant=value_tenant(<customer id>), table=<table>)`, and for `_all`,
  `EpochChangeGrant(tenant=ALL_TENANTS, table=<table>)`. Never a customer's or `_all`'s whole prefix;
- for a tool pod, each agent table it was granted (`agent_table_grants`):
  `EpochChangeGrant(tenant=schema_tenant(<that agent's schema>), table=<table>)`, which its owner stack reads
  with `PodTenants(schema=<that agent's schema>)`.

A host that names no grants (`epoch_change_grants=None`) keeps the pod's read of the whole log: the expand step
of the rollout, until the host names them.

## Wire every process that writes

```python
from threetears.epoch import EpochGenerationSource, EpochChangeRetention

# restored_from_l3: the keeper alone starts a lost log; set it in EVERY process of the deployment
source = EpochGenerationSource(
    nats_client, change_retention=EpochChangeRetention(replicas=3), restored_from_l3=True
)
registry.set_generation_source(source)
```

A source without `restored_from_l3` declares a lost log again under a new incarnation, and every follower
is then behind (`replaced`) until it restarts. With it, a write while the log is lost waits up to
`restore_wait` (10 s) for the keeper, then fails with an ERROR (its rows are committed, with their epochs); a
read of the generation fails at once and its caller asks L3.

Every table a restore must put back carries its epoch: declare it versioned (`TableSchema(row_epoch=row_versions(...))`)
or stamped (`row_epoch=ROW_STAMPED`). A switched-on table without one is said at WARNING at its migration run (refused
once `REFUSE_TABLES_WITHOUT_EPOCH` is on), and its records are counted by the scrub.

## Run the keeper

```python
from threetears.core.collections.log_keeper import LogKeeper, LogKeeping, LOG_KEEPER_CLAIM_EVERY

keeping = LogKeeping(LogKeeper(l3_pool, schema="platform", log=source), nats_client, lease="change-note-keeper")
await keeping.start()                          # the marks table, then one pass; before this process writes
# then every LOG_KEEPER_CLAIM_EVERY (15 s), in a periodic task:
await keeping.claim_and_keep(holder="hub-<replica>")   # a pass every second while it holds the lease
```

A second database whose writers share the log (identity-core's) runs its own keeper beside it,
`LogKeeper(pool, schema=<its schema>, log=<its source, database=<its schema>>, keeps_log=False)`, under a lease of its
own.

Each pass, in order:

1. **Log absent** (NATS lost it) and the newest mark names a log: put it back from the tables (see "After NATS loses
   the change log"). No mark at all (a database's first start): declare a fresh log and mark it.
2. **A restore left unlabelled** (its keeper died part-way): done again in full into the same log.
3. **A log the keeper does not know** (its newest mark names another): taken, said at WARNING unless it is this
   keeper's own restore, and marked at once.
4. **Mark**: every 30 s, the log's position after the database's clock; then drop the marks older than the log's
   window's anchor.
5. **Scrub**, every minute, not every pass: up to 1000 records after the scrub's position, up to the log's head one
   scrub ago. Each record of this database is checked against the tables a restore reads back (an ERROR and
   `threetears_change_notes_missing_total{table}` for one that carries no epoch) and against the newest mark before it
   (an ERROR and `threetears_restore_margin_violations_total{table}` for a lowest stamp below its X). Never repair.
   The log's owner reads by sequence, 64 at a time; a keeper beside reads its schema's tables by their keys, never by
   sequence (a pod's grant cannot name that read).

Two keepers at once mark and scrub twice, and never restore twice (the log is declared once); the lease keeps it to
one.

## After NATS loses the change log

1. Pods keep serving what they hold. Followers count themselves on `collection_epoch_tables_behind`.
   L2-first writes raise `L2LostError`; other writes wait for the restore, and no process starts a log.
2. The keeper's next pass first declares a new log under a NEW incarnation, marked as being restored
   (`threetears_restoring` in the stream's metadata) and continuing nothing yet: writes waiting for the log
   append to it at once. It then reads the tables' changes since the lost log's oldest mark, sized first against the
   restore budget (what the log holds plus every record must stay under 75 % of `max_bytes`), each table a scope or a
   bucket at a time, folded in the restore's worker; appends them; and only then labels the log as continuing the
   lost one from that mark's position (`continues`, `covered_from`). It logs one WARNING with the counts and counts
   `threetears_change_notes_restores_total`. Until the label is written every follower of the lost log is behind
   (`replaced`), never current.
   - A restore that stops part-way (a refused append, a NATS error, the keeper losing its lease or its
     process) is unlabelled: the next pass, by any keeper, does it again in full into the same log, under
     fresh append ids. A key appended twice is evicted twice.
   - Changes too big for the log are put back from a later mark, the oldest whose part fits. One ERROR
     (`only the tables' changes after a later mark`) names `covered_from`, and
     `threetears_change_notes_restore_truncated_total` counts it (alert on any). Followers at or past it
     catch up; older ones are behind (`record_lost`) until they restart.
   - Only when not even the changes after the newest mark fit is the restore refused: one ERROR
     (`was NOT restored`), `threetears_change_notes_restores_refused_total` (alert on any), and the new log
     is taken as a fresh one. Every follower of the lost log stays behind until it restarts. Nothing reloads.
   - A keeper beside (identity-core's) puts its own database's tables back into the same log from its own marks,
     labelling nothing: `threetears_change_notes_put_backs_total`.
3. Once the label is written, a follower whose mark under the lost incarnation is at or past `covered_from`
   carries its mark to the start of the new log and walks every record, the writes made during the restore
   included: it evicts each key once (some it already held current) and is caught up. A run deleted whole comes
   back as its prefix.
4. A follower whose mark is older than `covered_from` is behind (`record_lost`), serving what it holds, until it
   restarts.
5. For L2-first tables the worker's refiller then puts L2's rows back from L3 (see below).
6. Verify: `collection_epoch_tables_behind` returns to 0; the keeper's WARNING names the restored
   incarnation; `l2_first_status` shows every L2-first table `complete=True`.

What a loss costs: a write whose epoch append failed while the log was absent is in L3 with its epoch, and a restore
that reads after it puts it back. An L2-first write that did not wait and had not reached L3 is gone until its source
sends it again. A table carrying no epoch is not put back at all: the scrub reports it before a loss.

## Alert: a record a restore cannot read back

`threetears_change_notes_missing_total` increased. The ERROR names the table and the record. A restore after a NATS
loss would not put that table's changes back: no versioned or stamped table of that name carries an epoch in this
database.

Do: declare the table versioned or stamped (its migration then adds the epoch and its restore index). The scrub never
writes anything.

## Alert: a record below its mark's floor

`threetears_restore_margin_violations_total` increased. A writer held a transaction, or its append, longer than the
margin M (10 min): a restore from that mark would read the table from X and miss what it wrote below X. The ERROR names
the table, the scopes, the keys and the stamps. Find the writer that holds its transaction that long.

## Followers: the quiet pass

A pod that follows tables by watching their keys (`TableFollower`, `follow_tables`) is pushed only when a
table is written. A quiet table would keep its mark while the log's retention moves past it, and then be
judged behind (`aged_out`) for a stretch in which nothing happened. So the follower runs one quiet pass for
all its tables every `quiet_pass_interval()`, a quarter of the log's own retention: it reads only NATS (the
log's state and each table's newest epoch), moves a quiet table's mark to the log's head, and walks and
applies any table the log holds an unapplied epoch of. It logs one INFO line per pass and counts
`collection_epoch_quiet_marks_moved_total{table}`. A pod that runs `generation_catchup_tick` can run
`quiet_marks_tick` instead, at the same interval.

## Run the L2-first worker

```python
from threetears.core.collections.l2_first import L2Refiller, L3Saver, run_l2_first_worker

held = await lease.hold("l2-first-worker", ttl=timedelta(seconds=30), renew_every=timedelta(seconds=10))
saver = L3Saver(registry, epochs=source)
refiller = L2Refiller(registry, epochs=source)
await run_l2_first_worker(held, saver=saver, refiller=refiller)
```

Each pass refills any table L2 does not hold complete (once the keeper has put a lost log back), then saves
every epoch after each table's saved position. A failed pass is logged at ERROR and the next one resumes.
Two workers at once are safe, only wasteful. The worker's registry holds every L2-first collection with L2
and L3 wired. `every` bounds how long a write that does not wait stays unsaved.

The process that commits L2-first writes needs write access to the epoch bucket and the change log. A pod
today has neither; its broker advances for it.

## See where each L2-first table stands

```python
from threetears.core.collections.l2_first import l2_first_status

for status in await l2_first_status(registry, epochs=source):
    print(status)
```

| Reading | Meaning | Do |
|---|---|---|
| `generation=None`, `complete=False` | NATS lost the log; the keeper has not put it back | check the keeper holds its lease and is passing |
| `complete=False` with a `generation` | the refiller has not run since the restore | check the L2-first worker |
| `complete=True`, `unsaved_epochs` small | normal | nothing |
| `unsaved_epochs` growing | no saver running, or L3 failing | check the worker's ERROR lines |

## Alert: the epoch change log is filling

`threetears_epoch_changes_fill_ratio` at 0.75 or above. The L2-first saver measures it every pass and logs
a WARNING. Past 1.0 the log refuses records, and every write that needs one fails.

Do: raise `EpochChangeRetention.max_bytes` on every source that declares the log, or find the writer
producing so many epochs.

## Alert: L3 refused a committed L2-first write

`threetears_l2_first_l3_refused_total` increased. Only a collection that does not wait gets here: L2 holds
a row L3's constraints refuse. The ERROR names the table, the epoch and the key. A waiting write's refusal
goes to its caller instead, and L2 is withdrawn.

Do: fix the row at its source and send it again.

## What the hub adds

3tears ships the library pieces above. The hub owns:

- **The keeper.** One process (the hub, under a lease) runs `LogKeeping`: `start()` before it writes,
  `claim_and_keep` every `LOG_KEEPER_CLAIM_EVERY`; it builds every source with `restored_from_l3=True`.
- **The scripts.** `aibots` CLI commands that print the keeper's newest mark, `l2_first_status` per table,
  and run a keeper or refill pass, each with its runbook step.
- **The L2-first worker deployment.** One process per principal holding L2-first collections runs
  `run_l2_first_worker` under a lease.
- **The grants.** Epoch and change-log writes for the principal that commits L2-first writes.
- **The alert rules.** On the metrics in the facts table, at the thresholds given.
- **Settings.** If the hub makes the mark interval, `every`, a lease TTL or `max_bytes` a per-environment setting,
  each goes in 1Password through the existing env-var pattern, with its provisioning step.

The write-mark trigger is never installed on an L2-first table, and the reconciler deletes any mark on one
without advancing it: its epoch already named its rows before L3 had them.
