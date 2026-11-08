# Runbook: measure what a declared write generation costs

`scripts/measure-epoch-write-cost.py` measures `save_entity` on a table that declares
`write_generation = WRITE_GENERATION` against one that does not. It is the evidence for "may every
collection declare a write generation", and the check that a change to the epoch path did not make
writes slower.

## When to run it

- Before deciding to switch a busy table on, or to make `WRITE_GENERATION` the default.
- After any change to `EpochGenerationSource.advance_with_change`, the epoch change log, the
  write-mark trigger, or the collection write path (`BaseCollection._bump_generation` and its callers).
- Before a release that changes any of those, on the release candidate's commit. Name the commit with
  the numbers: a result counts only for the commit it ran on.

Run it on a quiet machine. Another test suite or build running beside it inflates every row.

## How to run it

```bash
# throwaway NATS (JetStream) and Postgres containers; needs Docker
uv run python scripts/measure-epoch-write-cost.py

# fewer or other writer counts, more saves per writer
uv run python scripts/measure-epoch-write-cost.py --writers 1 8 32 64 --rounds 400

# against a broker and database you already run
uv run python scripts/measure-epoch-write-cost.py --nats nats://localhost:4222 --db postgresql://u:p@localhost/postgres
```

No settings or secrets: with no URLs it starts its own containers. No test lane runs it.

## What it measures

Three tables, one per variant, all on one broker and one database. Each writer is its own process
(its own NATS connection, database pool, L1 and registry), and every writer writes distinct rows of
the same table, so they contend only where the platform makes them.

| variant | what a save does beyond the row write |
|---|---|
| `undeclared` | nothing (today's default) |
| `declared` | one append of the epoch's change record to the change log (its stream sequence is the epoch; there is no counter to contend for) |
| `declared+marks` | the same, plus the write-mark trigger's insert in the write's transaction and the read and delete of the marks the advance covered: the hub's production shape |
| `declared+notes` | `declared`, plus the change-note trigger's note row in the write's transaction. A keeper process runs the note ring beside the writers with a 200 ms fence and a switch size of a fifth of the run's records, so the ring switches and `TRUNCATE`s its oldest table mid-run |

Pick variants with `--variants` (e.g. `--variants declared declared+notes`).

The script times each append through the source's public `advance_with_change`, so its append columns
measure the shipped path.

## How to read the result

The output is one markdown table:

- **p50 / p99 ms** per save. Compare each declared row with the `undeclared` row at the same writer count.
- **writes/s**, all writers together.
- **append p50 / p99 ms**: the epoch append alone, inside each declared save. It should stay flat as
  writers grow: appends do not contend. A p99 that climbs with writers points at the broker, not the table.
- **advances**: appends made; one per declared save.
- **append failed** / **failed**: appends the log refused, and saves that raised. A refused append's row
  **committed** and the save raised, so a follower that missed the broadcast cannot apply it. Any value
  above 0 is a correctness problem, not only a speed one.

Rule of thumb for a decision: declaring is cheap for a table where both failure columns stay 0 and the
declared p99 stays within a few milliseconds of the undeclared p99 at the most concurrent writers that
table really has. Count writers per table, not per pod: on the platform every agent pod's write to a
table is advanced by the hub's broker, so the hub's concurrent commits to one table are the writers.

## Recorded results

### The change note (`declared+notes`), 2026-11-08

Apple M4 (10 cores), Docker Desktop, `postgres:17-alpine`, `nats:2.14.2-alpine`, on the change-note branch
(`release/0.95.0-fold` after 12a01170f). Two runs, the second with 400 saves per writer. Read each notes row
against the `declared` row above it.

| run | writers | variant | p50 ms | p99 ms | writes/s | switches | emptied | saves during an emptying | their p99 ms | their max ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | declared | 2.68 | 4.53 | 361 | - | - | - | - | - |
| 1 | 1 | declared+notes | 3.11 | 4.99 | 311 | 3 | 3 | 104 | 4.99 | 5.62 |
| 1 | 8 | declared | 5.45 | 13.99 | 1338 | - | - | - | - | - |
| 1 | 8 | declared+notes | 6.20 | 10.14 | 1233 | 4 | 4 | 486 | 8.13 | 8.74 |
| 1 | 32 | declared | 17.66 | 50.39 | 1501 | - | - | - | - | - |
| 1 | 32 | declared+notes | 18.24 | 69.26 | 1416 | 2 | 2 | 12 | 72.41 | 72.41 |
| 2 | 8 | declared | 5.29 | 8.24 | 1439 | - | - | - | - | - |
| 2 | 8 | declared+notes | 5.80 | 10.95 | 1282 | 5 | 5 | 1198 | 18.77 | 40.42 |
| 2 | 32 | declared | 19.05 | 74.83 | 1363 | - | - | - | - | - |
| 2 | 32 | declared+notes | 21.68 | 77.00 | 1249 | 1 | 0 | 0 | - | - |

No append or save failed in any row. The note costs about 0.4 to 0.8 ms at p50 at one and eight writers and
does not grow with writers at p50 (0.6 ms at 32 in run 1, 2.6 ms in run 2). At 32 writers on a 10-core laptop
p99 is dominated by the machine (two runs of the same `declared` row differ by 24 ms), so its +2 to +19 ms is
not separable from noise. During an emptying on Postgres the overlapping saves stayed at the run's p99 in run
1 and rose to an 18.8 ms p99 (40 ms max) in run 2 at eight writers: `TRUNCATE` invalidates every session's
cached plan of the note function, so each replans once.

**YugabyteDB 2026.1.0.0-b118 (local `yugabyted`): crashes on `TRUNCATE` under writes.** The tserver
SIGSEGVed during the noted run at eight writers, and a probe with no notes and no triggers at all -- eight
writers inserting into one table while a loop `TRUNCATE`s another -- crashed it four times in two minutes. The
same probe on **2026.1.3.0** ran 84 truncates with no error and no crash. Production must run a YB that passes
the probe before the keeper empties a note table there.

YugabyteDB 2026.1.3.0 (local `yugabyted`, one node), same machine, two runs:

| run | writers | variant | p50 ms | p99 ms | writes/s | switches | emptied | saves during an emptying | their p99 ms | their max ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | declared | 3.42 | 5.96 | 276 | - | - | - | - | - |
| 1 | 1 | declared+notes | 4.18 | 9.01 | 222 | 4 | 4 | 151 | 9.01 | 9.83 |
| 1 | 8 | declared | 6.89 | 13.11 | 1068 | - | - | - | - | - |
| 1 | 8 | declared+notes | 8.70 | 20.18 | 821 | 4 | 4 | 1275 | 20.51 | 46.34 |
| 1 | 32 | declared | 23.00 | 79.19 | 1185 | - | - | - | - | - |
| 1 | 32 | declared+notes | 27.12 | 72.56 | 1018 | 1 | 1 | 4335 | 78.95 | 151.36 |
| 2 | 8 | declared | 7.00 | 14.63 | 1066 | - | - | - | - | - |
| 2 | 8 | declared+notes | 9.06 | 17.01 | 817 | 2 | 2 | 2310 | 16.76 | 27.56 |
| 2 | 32 | declared | 21.47 | 49.55 | 1304 | - | - | - | - | - |
| 2 | 32 | declared+notes | 26.76 | 77.51 | 1013 | 2 | 2 | 1430 | 58.14 | 83.30 |

On YB the note costs about 0.8 ms p50 at one writer, 2 ms at eight and 4 to 5 ms at 32: it grows with
writers, unlike on Postgres, and throughput drops 20 to 23 %. Each noted statement makes two extra
distributed operations, the note insert and the read of the one-row ring table, every writer's read landing
on that one tablet. The emptying itself does not stand out: the saves that overlapped an emptying had a p99
at or below the run's.

**These are from before the change log (pre-#1121): the advance was a compare-and-swap on a per-table
counter key.** That path is gone, so the CAS columns and the conclusion below describe what the change
log replaced. They are kept as the baseline it was built against. Re-run on the release commit for the
change log's numbers, and record them here with that commit.

Apple M4 (10 cores), Docker Desktop, `postgres:17-alpine`, `nats:2.14.2-alpine`, 200 saves per writer.
Run on 36d876a79 with nothing else on the machine. The write path is the same at e9e9503dc
(pacepace/3tears#1121's head; that commit changes only the follower side). Two later runs on e9e9503dc
had other test suites beside them: the absolute numbers were 2x to 20x higher, the ratios held, and
the failed saves at 32 writers were 16/30 and 115/40.

| writers | variant | writes | p50 ms | p99 ms | writes/s | CAS losses/write | advances | failed |
|---|---|---|---|---|---|---|---|---|
| 1 | undeclared | 200 | 1.51 | 4.75 | 563 | 0.000 | 0 | 0 |
| 1 | declared | 200 | 2.66 | 5.06 | 351 | 0.000 | 200 | 0 |
| 1 | declared+marks | 200 | 3.47 | 4.98 | 280 | 0.000 | 200 | 0 |
| 8 | undeclared | 1600 | 4.03 | 6.72 | 1894 | 0.000 | 0 | 0 |
| 8 | declared | 1600 | 5.99 | 82.65 | 554 | 0.561 | 1600 | 0 |
| 8 | declared+marks | 1600 | 10.08 | 73.48 | 446 | 0.516 | 1600 | 0 |
| 32 | undeclared | 6400 | 15.63 | 47.55 | 1727 | 0.000 | 0 | 0 |
| 32 | declared | 6400 | 45.19 | 312.50 | 445 | 3.572 | 6394 | 6 |
| 32 | declared+marks | 6400 | 59.55 | 352.89 | 360 | 3.645 | 6390 | 10 |

What it says: one writer pays about +1 ms (declared) to +2 ms (with marks) per save. With several
concurrent writers on one table, the single epoch key is the limit: a declared table tops out near
450 to 550 saves/s, against 1,700 to 1,900 undeclared. p99 rises about 12x at 8 writers and 7x at 32,
and at 32 writers some saves exhaust their compare-and-swap rounds and raise after their row committed.
Declaring every collection was cheap for tables with few concurrent writers; for a table with tens of
concurrent writers it was not acceptable with the per-commit compare-and-swap. That is why the change
log appends instead: the measurement that decides the default is the re-run above.
