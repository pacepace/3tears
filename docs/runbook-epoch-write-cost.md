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

### YugabyteDB on Docker Desktop

On a Linux kernel from 6.19 up to 7.0.14 on arm64 (Docker Desktop's linuxkit kernel on Apple silicon is one),
YugabyteDB's tcmalloc corrupts its heap through its per-CPU caches (yugabyte-db #33458) and the tserver
SIGSEGVs under ordinary concurrent writes. Block the `rseq` syscall for the container: Docker's default
seccomp profile with `rseq` answered `ENOSYS`.

```bash
curl -sSL https://raw.githubusercontent.com/moby/profiles/main/seccomp/default.json -o /tmp/seccomp-default.json
python3 - <<'PY'
import json
d = json.load(open("/tmp/seccomp-default.json"))
for rule in d["syscalls"]:
    if "rseq" in rule.get("names", []):
        rule["names"].remove("rseq")
d["syscalls"] = [rule for rule in d["syscalls"] if rule.get("names")]
d["syscalls"].append({"names": ["rseq"], "action": "SCMP_ACT_ERRNO", "errnoRet": 38})
json.dump(d, open("/tmp/seccomp-no-rseq.json", "w"))
PY
docker run -d --name yb --security-opt seccomp=/tmp/seccomp-no-rseq.json -p 15433:5433 \
  yugabytedb/yugabyte:2026.1.0.0-b118 bin/yugabyted start --background=false --tserver_flags="ysql_max_connections=400"
# must print "TCMalloc per cpu caches active: 0"
docker exec yb bash -c 'grep -rh "per cpu caches active" /root/var/logs/'
uv run python scripts/measure-epoch-write-cost.py --db postgresql://yugabyte:yugabyte@localhost:15433/yugabyte
```

Bench against the version production runs (2026.1.0.0-b118 on 2026-11-08).

## What it measures

Three tables, one per variant, all on one broker and one database. Each writer is its own process
(its own NATS connection, database pool, L1 and registry), and every writer writes distinct rows of
the same table, so they contend only where the platform makes them.

| variant | what a save does beyond the row write |
|---|---|
| `undeclared` | nothing (today's default) |
| `declared` | one append of the epoch's change record to the change log (its stream sequence is the epoch; there is no counter to contend for) |
| `declared+marks` | the same, plus the write-mark trigger's insert in the write's transaction and the read and delete of the marks the advance covered: the hub's production shape |
| `declared+notes` | `declared`, plus the change-note trigger's note row in the write's transaction. A keeper process runs the note ring beside the writers with a 1 s fence (`--fence-seconds`) and a switch size of a fifth of the run's records, so the ring switches and empties its idle table (batched deletes) mid-run. Each writer's pool carries the cached ring slot, as the hub's does |

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

## The scrub's cost

The keeper's scrub is not on any write's path. It runs every tenth of the fence (1 min at the 10 min
default), on the keeper's replica only, and per run reads at most `scrub_batch` (1000) records from the
change log, 64 at a time, plus, per note table in the coverage (two), one index lookup of the checked keys
and one of whole-table notes, both led by the `(key_text, table_name)` index: a run's cost follows the
epochs it checks, not how many notes the coverage holds. The index is one more entry per note on the write
path; the recorded results below were taken before it was added (3tears 44216b89e). One query is not indexed: an epoch that names no rows
(a whole-table or keys-unknown write) asks whether its table has any note, which stops at the first one and
scans the note table only when there is none (the missing-note alert case).

## Recorded results

### The cached slot and batched-delete emptying, 2026-11-08 (3tears 44216b89e)

Apple M4 (10 cores), Docker Desktop, `nats:2.14.2-alpine`, 1000 saves per writer, keeper fence 1 s (a writer's
slot is re-read every 0.25 s; with production's 10 min fence it is every 2.5 min), each cell its own process
run. Read each notes row against the `declared` row above it.

**YugabyteDB 2026.1.0.0-b118** (the version production runs), local `yugabyted` with `rseq` blocked
(`TCMalloc per cpu caches active: 0`), one node, one tablet per table. No tserver crash in either run.

| run | writers | variant | p50 ms | p99 ms | writes/s | failed | switches | emptied | saves during deletes | their p99 ms | their max ms | slot refreshes | notes that read the ring |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | declared | 2.59 | 3.91 | 362 | 0 | - | - | - | - | - | 12 | - |
| 1 | 1 | declared+notes | 3.21 | 6.53 | 297 | 0 | 3 | 3 | 72 | 4.56 | 4.56 | 14 | 0 |
| 1 | 8 | declared | 5.77 | 10.47 | 1308 | 0 | - | - | - | - | - | 200 | - |
| 1 | 8 | declared+notes | 6.91 | 13.78 | 1076 | 0 | 5 | 5 | 611 | 11.13 | 13.00 | 240 | 0 |
| 1 | 32 | declared | 18.36 | 44.97 | 1573 | 0 | - | - | - | - | - | 2470 | - |
| 1 | 32 | declared+notes | 20.79 | 52.35 | 1371 | 0 | 2 | 2 | 7678 | 69.30 | 162.41 | 2816 | 0 |
| 2 | 1 | declared | 2.85 | 3.95 | 337 | 0 | - | - | - | - | - | 13 | - |
| 2 | 1 | declared+notes | 3.15 | 5.15 | 307 | 0 | 3 | 3 | 70 | 4.30 | 4.30 | 14 | 0 |
| 2 | 8 | declared | 5.63 | 9.54 | 1350 | 0 | - | - | - | - | - | 192 | - |
| 2 | 8 | declared+notes | 8.25 | 15.18 | 907 | 0 | 5 | 5 | 1261 | 13.62 | 49.99 | 281 | 0 |
| 2 | 32 | declared | 27.33 | 83.40 | 1011 | 0 | - | - | - | - | - | 3693 | - |
| 2 | 32 | declared+notes | 34.88 | 85.49 | 836 | 0 | 3 | 3 | 10622 | 79.09 | 127.05 | 4504 | 0 |

**Postgres 17** (`postgres:17-alpine`):

| run | writers | variant | p50 ms | p99 ms | writes/s | failed | switches | emptied | saves during deletes | their p99 ms | their max ms | slot refreshes | notes that read the ring |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | declared | 1.90 | 3.12 | 503 | 0 | - | - | - | - | - | 8 | - |
| 1 | 1 | declared+notes | 2.37 | 3.75 | 406 | 0 | 3 | 3 | 75 | 3.57 | 3.57 | 10 | 0 |
| 1 | 8 | declared | 5.66 | 10.71 | 1315 | 0 | - | - | - | - | - | 200 | - |
| 1 | 8 | declared+notes | 7.26 | 28.67 | 957 | 0 | 4 | 4 | 438 | 64.49 | 74.07 | 267 | 0 |
| 1 | 32 | declared | 21.28 | 86.87 | 1269 | 0 | - | - | - | - | - | 3017 | - |
| 1 | 32 | declared+notes | 18.24 | 83.96 | 1446 | 0 | 3 | 3 | 6525 | 94.36 | 186.41 | 2613 | 0 |
| 2 | 1 | declared | 1.57 | 2.28 | 619 | 0 | - | - | - | - | - | 7 | - |
| 2 | 1 | declared+notes | 1.72 | 3.31 | 539 | 0 | 2 | 2 | 59 | 2.35 | 2.35 | 8 | 0 |
| 2 | 8 | declared | 4.86 | 9.50 | 1537 | 0 | - | - | - | - | - | 168 | - |
| 2 | 8 | declared+notes | 5.49 | 12.56 | 1345 | 0 | 3 | 3 | 471 | 17.17 | 17.73 | 192 | 0 |
| 2 | 32 | declared | 16.64 | 79.64 | 1581 | 0 | - | - | - | - | - | 2343 | - |
| 2 | 32 | declared+notes | 18.10 | 83.94 | 1467 | 0 | 2 | 2 | 798 | 76.79 | 115.33 | 2560 | 0 |

What it says:

- No save failed and no note read the ring row: every write took the cached slot. The slot refreshes are the
  bench's 1 s fence (about 4 per writer per second); the ring row is otherwise never read.
- YugabyteDB: the note's p50 cost is +0.3 to +0.6 ms at one writer, +1.1 to +2.6 ms at eight and +2.4 to
  +7.5 ms at 32, against +0.8 / +2 / +4 to 5 ms with the per-statement ring read. Lower at one and eight
  writers; at 32 the two runs' `declared` rows alone differ by 9 ms, so the change is inside the noise. It
  still grows with writers here. Locally every table is one tablet, so every writer's note insert lands on
  the note table's one tablet, the same shape the ring row had; on a cluster with several tablets per table
  that insert spreads. The slot refreshes add, at 32 writers, one ring read per 7 to 11 saves, which
  production's fence makes one per 2.5 min per process.
- Postgres: +0.2 to +0.5 ms at one writer, +0.6 to +1.6 ms at eight, -3 to +1.5 ms at 32: flat.
- Emptying by batched deletes: at one and eight writers on YugabyteDB the saves that overlapped a delete
  batch had a p99 at or below the run's (13.6 against 15.2 ms, 11.1 against 13.8 ms). At 32 writers run 1's
  overlapping saves reached 69 ms against the run's 52 ms; run 2's did not (79 against 85 ms). On Postgres at
  eight writers run 1's overlapping saves rose to a 64 ms p99 against the run's 29 ms; run 2's to 17 against
  13 ms. Each emptying here deletes a fifth of the run's notes in 1000-row batches with a 20 ms pause.

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

**Correction (2026-11-08): the local YugabyteDB crashes were the kernel, not `TRUNCATE` and not b118.** An
earlier version of this page said 2026.1.0.0-b118 crashes on `TRUNCATE` under writes. It does not. The tserver
SIGSEGVs were yugabyte-db issue #33458: on Linux kernels from 6.19 up to the fix (7.0.14 on arm64), YugabyteDB's
bundled tcmalloc corrupts its heap through its per-CPU caches (`rseq`). Docker Desktop on this machine runs
kernel 7.0.12-linuxkit on aarch64, inside that range. With no `TRUNCATE` at all -- the `declared` variant alone,
plain upserts -- the same container crashed four to six times in two minutes. Dev and prod run amd64 on kernel
6.12 and are not affected. **Every local YugabyteDB number on this page taken before this correction, the
2026.1.3.0 table below included, was taken on that heap-corrupting kernel.** Run a local YugabyteDB with `rseq`
blocked (see "YugabyteDB on Docker Desktop" below) and check that its tserver log says
`TCMalloc per cpu caches active: 0`.

YugabyteDB 2026.1.3.0 (local `yugabyted`, one node), same machine, two runs, **on the heap-corrupting kernel, before the cached slot** (kept as the baseline the slot was built against):

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
