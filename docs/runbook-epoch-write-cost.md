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
