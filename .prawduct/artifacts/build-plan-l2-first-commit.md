# Build plan: the L2-first write path

Owner decisions of 2026-11-08 (Pace), from the reports design `design-streaming-loads.md`, sections
"One write order for every collection" and "Into the shared epoch layer". Ships in 0.95.0, stacked on
#1121 (`advance_with_change`, the epoch change log).

## Goals

1. A collection that declares `l2_first` commits a write in NATS: the row lands in L2 and its epoch
   record (naming the changed keys) lands in the epoch change log. That record landing is the commit.
2. A saver makes L3 match L2 for every committed key, and saves the epoch record to an L3 epoch log.
   Saving twice is harmless.
3. By default the caller waits for the L3 save (durability, and L3's constraints decide "saved").
   `L2First(wait_for_l3=False, refetchable_from=...)` declares a collection whose data its source can
   send again; its caller returns at the commit.
4. Without L2 the same API writes straight to L3 (the existing path).
5. Pods never read L3 for an L2-first collection: an L2 miss is an absence while L2 is known complete,
   and `L2LostError` while it is not. An L3->L2 refiller (one claimed worker) restores rows, epoch records
   and the generation from L3; pods serve what they hold, report themselves behind, and catch up through
   the normal epoch path.
6. No rebuilds, no whole-table drops, no TTLs; L2 failures go through `degrade_l2`.
7. `collection_epoch_changes_fill_ratio` gauge, with a warn threshold, before the change log fills.

## Inferences (alignment pass: proceed, citing the owner's decisions)

- [DECISION] **Declared, not yet the default.** `BaseCollection.l2_first` defaults to `None`. Making it
  every collection's default needs every writing principal granted the epoch writes and a saver/refiller
  running in every deployment; a pod today has neither (its broker advances for it). The flip is one line
  once the hub/SDK declarations land. Flagged to the owner, not dropped.
- [DECISION] The saver's position and the "settled by the writer" markers live in the collection's own L2
  bucket under `{scope}._l2first.{table}.`, which every writer and pod of the principal is already granted;
  `_l2first` cannot be a table name (it holds a digit), so no row key collides.
- [DECISION] The L3 epoch log is a protocol (`EpochLogStore`) with a SQL implementation and its DDL;
  installing it is the consumer's migration step, like the write-mark queue.
- [DECISION] A rejected wait-flag write is withdrawn by restoring L2's prior value at the revision it won,
  then advancing once more naming the key, so a follower that took the withdrawn value evicts it.
- Known limit: a no-wait write that never reached L3 is lost with the broker; a pod that heard it holds a
  newer value than the refill until its source re-sends. Counted and logged by the refiller.

## Chunks

1. Core: `l2_first` module (declaration, protocols, saver, refiller, SQL epoch log), `ChangedRow.l3_pending`,
   BaseCollection write/read/delete branches, `SchemaBackedCollection.l3_pages`.
2. Epoch: `EpochGenerationSource` implements `L2FirstEpochs`; change-log fill gauge and warning.
3. NATS: `held_bytes()` on the KV bucket and the fake.

## Done when

- Tests: crash between the NATS commit and the L3 save (saver resumes, nothing lost, saving twice
  harmless); a wait-flag caller sees an L3 rejection and L2 is withdrawn; a no-wait rejection is an
  ERROR, counted, and never wedges the saver; L2 lost -> refiller restores -> the pod catches up through
  the normal path with no L3 reads; no-L2 mode writes straight to L3; the fill gauge warns.
- CI lint, format, mypy and the enforcement suite pass as the workflow runs them; Critic run.
