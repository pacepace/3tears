# Running L2-first collections

An L2-first collection commits each write in NATS: one epoch record in the change log, carrying the row.
The row is applied to L2 right after; if the writer dies first, the saver applies it from the record. The
saver then makes L3 (Yugabyte) match. Pods never read L3. When NATS loses data, one
claimed worker puts L2 back from L3.

This page is for whoever wires, runs and watches it. The design is in the module docstring of
`threetears.core.collections.l2_first`.

## Facts every section uses

| What | Value |
|---|---|
| Declaration | `l2_first = L2_FIRST` on the collection class (waits for L3) |
| Not waiting | `l2_first = L2First(wait_for_l3=False, refetchable_from="<the source that resends it>")` |
| Default | waits for L3 before the write returns |
| Without NATS | the same collection writes straight to L3 |
| Generation source | `threetears.epoch.EpochGenerationSource` on the registry (`set_generation_source`) |
| L3 epoch log | table `threetears_epoch_log` in a schema you name; `SqlEpochLogStore(pool, schema=...)` |
| Registry wiring for writers | `registry.set_epoch_log(SqlEpochLogStore(...))` |
| Bookkeeping in L2 | keys `{scope}._l2first.{table}.complete`, `.saved`, `.settled.{incarnation}.{count}` in the collections bucket |
| Worker | `run_l2_first_worker(held_lease, saver=..., refiller=..., every=timedelta(seconds=1))` |
| Fill gauge | `threetears_epoch_changes_fill_ratio`, warn at `EPOCH_CHANGES_FILL_WARN_RATIO` (0.75) |
| Refusals | `threetears_l2_first_l3_refused_total{table}`: alert on any increase |
| Records missing | `threetears_l2_first_records_missing_total{table}`: alert on any increase |
| Refills | `threetears_l2_first_refills_total{table}`, with a WARNING per refill |
| Pods behind | `collection_epoch_tables_behind` (existing) |
| New env vars or settings | none in 3tears; see "What the hub adds" |

## Wire a process that writes

```python
from threetears.core.collections.l2_first import SqlEpochLogStore
from threetears.epoch import EpochGenerationSource, EpochChangeRetention

source = EpochGenerationSource(nats_client, change_retention=EpochChangeRetention(replicas=3))
registry.set_generation_source(source)
epoch_log = SqlEpochLogStore(l3_pool, schema="platform")
await epoch_log.install()          # once per database; safe to run again
registry.set_epoch_log(epoch_log)
```

The process needs write access to the epoch bucket and the epoch change log. A pod today has neither; its
broker advances for it. Until the grants exist, L2-first writes belong in a process that has them.

## Run the one claimed worker

```python
from datetime import timedelta
from threetears.core.coordination.lease import KVLease
from threetears.core.collections.l2_first import L2Refiller, L3Saver, run_l2_first_worker

lease = KVLease(nats_client, bucket_name="leases")
held = await lease.hold("l2-first-worker", ttl=timedelta(seconds=30), renew_every=timedelta(seconds=10))
saver = L3Saver(registry, epochs=source, log=epoch_log)
refiller = L2Refiller(registry, epochs=source, log=epoch_log)
await run_l2_first_worker(held, saver=saver, refiller=refiller)
```

Each pass refills any table L2 does not hold complete, then saves every epoch after each table's saved
position. A failed pass is logged at ERROR and the next one resumes. Two workers at once are safe, only
wasteful. The worker's registry holds every L2-first collection with L2 and L3 wired.

Tuning: `every` is the only knob. It bounds how long a write that does not wait stays unsaved.

## See where each table stands

```python
from threetears.core.collections.l2_first import l2_first_status

for status in await l2_first_status(registry, epochs=source, log=epoch_log):
    print(status)
```

| Reading | Meaning | Do |
|---|---|---|
| `generation=None`, `complete=False` | NATS lost the table; the refiller has not run | check the worker holds its lease and is passing |
| `complete=True`, `unsaved_epochs` small | normal | nothing |
| `unsaved_epochs` growing | no saver running, or L3 failing | check the worker's ERROR lines |
| `generation` equals `l3_logged` | caught up with the L3 epoch log | nothing |

## After NATS loses data

1. Pods keep serving what they hold. A key a pod does not hold raises `L2LostError`. Followers count
   themselves on `collection_epoch_tables_behind`. Writes raise `L2LostError`.
2. The worker's next pass puts the change log back from the L3 epoch log: every table's newest 4096
   records, at their old sequences, in one restore. Then, per table, the rows L2 holds nothing for, the
   saver's position, and the "complete" mark. It logs one WARNING for the log and one per table, and
   counts `threetears_l2_first_refills_total`.
3. Pods catch up through the normal epoch path. They never read L3.
4. Verify with the status read above: every table `complete=True` and a `generation`.

To run a refill now rather than wait for the next pass, run one pass from the worker's process:
`await refiller.run_pass()`. It does nothing for a table L2 already holds complete.

What a loss costs: a write that did not wait and had not reached L3 is gone until its source sends it
again. A waiting write that returned is in L3.

## Alert: the epoch change log is filling

`threetears_epoch_changes_fill_ratio` at 0.75 or above. The saver measures it every pass and logs a
WARNING. Past 1.0 the log refuses records, and every write that needs one fails.

Do: raise `EpochChangeRetention.max_bytes` on every source that declares the log, or find the writer
producing so many epochs.

## Alert: L3 refused a committed write

`threetears_l2_first_l3_refused_total` increased. Only a collection that does not wait gets here: L2 holds
a row L3's constraints refuse. The ERROR names the table, the epoch and the key. A waiting write's refusal
goes to its caller instead, and L2 is withdrawn.

Do: fix the row at its source and send it again.

## What the hub adds

3tears ships the library pieces above. The hub owns:

- **The script.** An `aibots` CLI command that prints `l2_first_status` per table and one that runs a
  refill pass, each with its runbook step.
- **The worker deployment.** One process per principal holding L2-first collections runs
  `run_l2_first_worker` under a lease.
- **The grants.** Epoch and epoch-change-log writes for the principal that commits L2-first writes.
- **The alert rules.** On the three metrics above, at the thresholds given.
- **Settings.** If the hub makes `every`, the lease TTL or `max_bytes` a per-environment setting, each
  goes in 1Password through the existing env-var pattern, with its provisioning step.
- **The DDL.** `SqlEpochLogStore.install()` in its migration path.

The write-mark trigger is never installed on an L2-first table, and the reconciler deletes any mark on
one without advancing it: its epoch already named its rows before L3 had them.
