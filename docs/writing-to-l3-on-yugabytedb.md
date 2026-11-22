# Writing to L3 on YugabyteDB

**A bulk write's change notes go into a per-connection buffer and reach the ring in one insert at the commit.** On
YugabyteDB, inserting a note into the ring for each row is a distributed write per row. That insert, not the row
itself, was most of the cost of a 1000-row write. The notes are unchanged: the same trigger writes the same note for
each row, so followers, the write-mark reconciler and a restore read exactly what they read before.

## What a written row costs

Measured on 2026-11-21 on YugabyteDB 2026.1, one node, in a scratch container. The table is the reports pod's
`latest_results_candidates_by_id`: 34 columns, a 7-column text key and the by-state index. The upsert moves one
invalidation column on every row. Times are the median of five runs, from begin to commit, with the range in
brackets.

| Write | Bare (no triggers) | Today: note into the ring per row | Buffered, one flush | One note per statement |
|---|---|---|---|---|
| insert 1000 | 29 ms (28–33) | 471 ms (451–550) | 103 ms (86–111) | 40 ms (36–60) |
| insert 5000 | 145 ms (123–187) | 2375 ms (2152–2868) | 376 ms (348–460) | 133 ms (120–160) |
| upsert 1000 | 203 ms (150–233) | 1402 ms (1201–1586) | 684 ms (597–780) | 737 ms (530–823) |

Every case except bare also has today's write-mark trigger.

The upsert again, isolating the write mark (1000 rows):

| Upsert 1000 | Median (range) |
|---|---|
| no triggers | 146 ms (110–151) |
| only the per-row write-mark trigger | 479 ms (440–533) |
| only a statement-level write mark | 183 ms (171–190) |
| one note per statement and a statement-level mark | 189 ms (174–231) |
| append 1000 new versions (no `ON CONFLICT`, no triggers) | 30 ms (30–33) |

What the numbers say:

- **The per-row insert is the cost, not calling a trigger.** Two row triggers that do nothing cost 0.16–0.19 s
  against 0.1 s bare; one that inserts a note costs 1.0–1.2 s (the reports stack, 1000 rows).
- **Any per-row AFTER trigger defeats an upsert's batched existence check.** The write-mark trigger inserts one row a
  transaction and otherwise only checks a setting, yet it alone takes the upsert from 146 to 479 ms.
- **Transition tables are not an option.** YugabyteDB has no `REFERENCING NEW TABLE` for statement triggers
  (YugabyteDB issue 1668), so a statement trigger cannot see the rows it wrote.

## The rules

1. **No per-row insert into the ring on a bulk write.** A writer that writes many rows sets the buffer's flag, and the
   trigger writes each note into the buffer instead (`NOTE_BUFFER_SETTING`, `note_buffer_open_sql`).
2. **One flush, just before the commit.** `note_flush_sql` moves the buffered notes into the ring's current table in
   one insert, before a notes bracket is read. Each note keeps the time its row was noted.
3. **A flagged transaction cannot commit without its flush.** The buffer's guard is a deferred foreign key that fails
   the COMMIT until the flush deletes its row (`NOTE_BUFFER_GUARD`). It is a key, not a constraint trigger, because
   YugabyteDB 2026.1 refuses `CREATE CONSTRAINT TRIGGER`.
4. **No flag means today's behaviour.** A write that does not come through a buffering writer (raw SQL, an older hub)
   is noted row by row into the ring, as always.
5. **The write mark stays as it is.** It already writes one row per table per transaction, and only when a row
   changed, so no advance is ever recorded for a write that changed nothing. Moving it out of the trigger would save
   nothing while any per-row trigger remains.

The hub's broker opens the buffer at a session's first write and flushes it in the commit (`_TxSession.commit`).
`tests/enforcement/test_change_notes_buffer_bulk_writes.py` fails if a keyed note can reach the ring outside the
buffer's check, or if the speed budget loosens. The budget itself, `test_note_buffer_live.py`, is a YugabyteDB timing
test that allows at most 5x the bare write.

## Rolling out

The trigger keeps its name and its function's name. Only the body changes, gaining the buffer's branch. During a
rollout, an older installer may write back the older body, and a newer one the new body. Either way each row is
noted exactly once:

- An older body ignores the flag and writes the note into the ring itself. The flush then finds nothing in the
  buffer.
- A newer body writes the note into the buffer when the flag is set. The flush moves it into the ring.

So no note is missing and none is doubled, whichever body a table has at any moment. The only cost is that installers
of two versions may each replace the other's body until every one is upgraded.

## Why not one note per statement

One note per statement, naming the statement's reach, is the fastest note (40 ms above). But it breaks the followers'
keyed catch-up. A commit too large to record names a notes bracket, and every follower reads the keys from the notes
in it (`noted_keys_page`). A note with no key makes that read answer "the whole table", so every such commit would
have dropped the table from every follower, the whole-table read the platform forbids. Its reach would also have to
name every key to stay exact, which grows by about 100 bytes a row past the change record's bound. A restore after a
NATS loss would put back the prefixes, not the rows, so a follower would drop every row under each prefix.

## Where this goes next: insert-only, newest version wins

The cheapest write measured is an append: 1000 new row versions in 30 ms, against 146 ms for the same rows as a bare
upsert. With no conflict check and no update, an append needs no per-row trigger. Its notes and its mark can be one
per statement, because the versions it wrote name exactly what changed. Readers take the newest version of each key,
and a bounded cleanup removes older versions later. This is the planned next step, not built yet.

## See also

- `threetears.core.ddl.change_notes`: the ring, the trigger body, the buffer and the flush.
- `threetears.core.ddl.write_marks`: the write-mark queue and its trigger.
- [Runbook: measure what a declared write generation costs](runbook-epoch-write-cost.md).
- [L2-first operations](l2-first-operations.md): the change notes' restore and scrub alerts.
