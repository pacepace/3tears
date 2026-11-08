# Running the change log's recovery, and L2-first collections

The epoch change log (`{ns}-epoch-changes`) is in NATS, memory storage, three replicas. It is the only
clock and the only catch-up path. To survive NATS losing it, every committed write also leaves a **change
note** in L3, in the same transaction as its rows: the table, the row's key and its invalidation columns,
never the row's data. One **keeper** process runs the ring of note tables and, after a loss, puts the log
back from the notes. Followers never read the notes; they walk the log.

An **L2-first** collection commits each write in NATS (one epoch record in the change log, carrying the
row), then a saver makes L3 match. Its L3 writes leave notes like any other table's.

This page is for whoever wires, runs and watches it. The design is in the module docstrings of
`threetears.core.collections.change_notes` and `threetears.core.collections.l2_first`.

## Facts every section uses

| What | Value |
|---|---|
| Note tables | `threetears_change_notes_0`, `_1`, `_2` in the platform schema: one current, one previous, one emptied and waiting |
| Ring row | `threetears_change_note_ring` (one row; only the keeper writes it, once per switch) |
| Which tables are noted | every table a class declares `write_generation`, absence caching or `l2_first` for: `noted_tables()` |
| Triggers | `threetears_change_note` (row) and `threetears_change_note_truncate` on each noted table, in every schema |
| Keeper | `ChangeNoteKeeper(pool, notes_schema=..., log=source)`, run by `run_change_note_keeper(held_lease, keeper)` |
| Ring switch | when the current table has taken one change log's worth (the log's `max_bytes`, 64 MiB on the hub) |
| Fence | the keeper's `fence`, 10 min by default: the least time between switches, and so before a table is emptied |
| Restore reads | the current and the previous note tables |
| Every source of the deployment | `EpochGenerationSource(..., restored_from_l3=True)` |
| Follower quiet pass | every `quiet_pass_interval()`: the log's retention (24 h) / 4 = 6 h |
| Missing notes | `threetears_change_notes_missing_total{table}`: alert on any increase |
| Ring | `threetears_change_notes_switches_total`, `threetears_change_notes_emptied_total`, `threetears_change_notes_restores_total` |
| Scrub lag | `threetears_change_notes_scrub_behind` (epochs the log holds after the scrub's position) |
| Quiet marks moved | `collection_epoch_quiet_marks_moved_total{table}` |
| Pods behind | `collection_epoch_tables_behind`; `collection_epoch_behind_total{reason}` |
| L2-first refusals | `threetears_l2_first_l3_refused_total{table}`: alert on any increase |
| L2-first records missing | `threetears_l2_first_records_missing_total{table}`: alert on any increase |
| L2-first refills | `threetears_l2_first_refills_total{table}`, with a WARNING per refill |
| Change log fill | `threetears_epoch_changes_fill_ratio`, warn at `EPOCH_CHANGES_FILL_WARN_RATIO` (0.75) |
| New env vars or settings | none in 3tears; see "What the hub adds" |

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
`restore_wait` (10 s) for the keeper, then fails with an ERROR (its rows and their notes are committed); a
read of the generation fails at once and its caller asks L3.

## Run the keeper

```python
from datetime import timedelta
from threetears.core.coordination.lease import KVLease
from threetears.core.collections.change_notes import ChangeNoteKeeper, run_change_note_keeper
from threetears.core.collections.generation import noted_tables

keeper = ChangeNoteKeeper(l3_pool, notes_schema="platform", log=source)
await keeper.install()                          # the ring; once per database, safe to run again
await keeper.install_triggers(noted_tables())   # also run it periodically, as the write-mark install is

lease = KVLease(nats_client, bucket_name="leases")
held = await lease.hold("change-note-keeper", ttl=timedelta(seconds=30), renew_every=timedelta(seconds=10))
await run_change_note_keeper(held, keeper)      # a pass every second
```

Each pass, in order:

1. **Log absent** (NATS lost it): put it back from the notes (see "After NATS loses the change log").
2. **New log** the keeper did not put back: take it; its notes cover it from its start.
3. **Fence**: once the fence has passed since the last switch, record the log's position. Every epoch
   after it has its note in the current table or a later one.
4. **Empty**: right after a switch, `TRUNCATE` the table after the new current one. It was current two
   switches back, at least one fence ago, so no statement that read it as current is still running. Never
   row by row.
5. **Switch**: when the fence has passed, the waiting table is empty, and the current table has taken
   one log's worth of records (bytes per sequence the log holds, times sequences since the switch), make
   the waiting table current. The fence's position becomes the coverage's **floor**: every epoch after it
   has its note in the current or the previous table.
6. **Scrub**: check up to `scrub_batch` (1000) epochs after the floor, up to the log's head one pass
   ago, against the notes. Report each one whose note is not where the coverage needs it. Never repair.

Two keepers at once never switch twice (every ring update names the switch it read), but empty and scrub
twice; the lease keeps it to one. Tuning: `fence` (keep it far beyond the longest statement any writer
runs), `switch_bytes` (default: the log's `max_bytes`; never set it below what the log takes between two
passes), `every`.

## After NATS loses the change log

1. Pods keep serving what they hold. Followers count themselves on `collection_epoch_tables_behind`.
   L2-first writes raise `L2LostError`; other writes wait for the restore, and no process starts a log.
2. The keeper's next pass reads the current and previous note tables, folds them into one record per
   table (each distinct key once, with its invalidation columns; a truncate or file load as one
   whole-table record), and declares a new log under a NEW incarnation that names the lost one and the
   floor (`continues`, `covered_from`). It logs one WARNING with the counts and counts
   `threetears_change_notes_restores_total`.
3. A follower whose mark under the lost incarnation is at or past the floor carries its mark to the start
   of the new log and walks every restored record: it evicts each noted key once (some it already held
   current) and is caught up. Nothing is dropped except a table a restored whole-table record names: that
   is a truncate or load that happened in the coverage, re-applied.
4. A follower whose mark is older than the floor is behind (`record_lost`), serving what it holds, until it
   restarts.
5. For L2-first tables the worker's refiller then puts L2's rows back from L3 (see below).
6. Verify: `collection_epoch_tables_behind` returns to 0; the keeper's WARNING names the restored
   incarnation; `l2_first_status` shows every L2-first table `complete=True`.

What a loss costs: a write whose epoch append failed while the log was absent is in L3 with its note, but
not in the restored log unless it committed before the keeper read the notes. A follower that missed it
serves its old copy of those rows until they are written again. An L2-first write that did not wait and
had not reached L3 is gone until its source sends it again.

## Alert: an epoch has no change note

`threetears_change_notes_missing_total` increased. The ERROR names the table, the epoch, the key and the
note tables looked in. A restore after a NATS loss would not put that epoch back.

Causes, most likely first:

- the table has no note trigger yet (created since the last `install_triggers`, or its class names its
  table per instance so `noted_tables()` cannot name it): run `install_triggers`;
- an L2-first table whose saver is behind: the record is in NATS and its row (and note) not yet in L3;
  see "unsaved_epochs growing" below;
- a writer stalled between its commit and its epoch append for longer than the fence: raise the fence.

Do: fix the cause. The scrub never writes a note.

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
producing so many epochs. The note ring switches at the same size, so it follows.

## Alert: L3 refused a committed L2-first write

`threetears_l2_first_l3_refused_total` increased. Only a collection that does not wait gets here: L2 holds
a row L3's constraints refuse. The ERROR names the table, the epoch and the key. A waiting write's refusal
goes to its caller instead, and L2 is withdrawn.

Do: fix the row at its source and send it again.

## What the hub adds

3tears ships the library pieces above. The hub owns:

- **The keeper.** One process (the hub, under a lease) runs `run_change_note_keeper`, calls
  `keeper.install()` at start and `install_triggers(noted_tables())` periodically beside the write-mark
  install, and builds every source with `restored_from_l3=True`.
- **The broker's file loads.** The hub's `l3.load_files` copies on its own connection; it should set
  `NOTE_WHOLE_SETTING` (`SELECT set_config('threetears.note_whole', '1', true)`) in each load transaction,
  as `SqlL3Backend.load_files` does, so a load is one whole-table note rather than one per row.
- **The scripts.** `aibots` CLI commands that print the ring (`keeper.ring()`), `l2_first_status` per table,
  and run a keeper or refill pass, each with its runbook step.
- **The L2-first worker deployment.** One process per principal holding L2-first collections runs
  `run_l2_first_worker` under a lease.
- **The grants.** Epoch and change-log writes for the principal that commits L2-first writes.
- **The alert rules.** On the metrics in the facts table, at the thresholds given.
- **Settings.** If the hub makes the fence, `every`, a lease TTL or `max_bytes` a per-environment setting,
  each goes in 1Password through the existing env-var pattern, with its provisioning step.

The write-mark trigger is never installed on an L2-first table, and the reconciler deletes any mark on one
without advancing it: its epoch already named its rows before L3 had them. The note trigger is installed on
it like any other table.
