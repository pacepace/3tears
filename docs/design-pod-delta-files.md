# Design: pod copies catch up from delta files

Status: agreed with Pace, 2026-11-23, one question at a time. Planned in `build-plan-pod-delta-files.md` and built for
0.95.0; not yet released or proven (the plan's chunk 11). Sizes measured on the election-night capture; fold cost
under the copy's retention still to measure (Open questions).

Home: 3tears implements this design, so this file is the copy its code and tests cite. It came from the reports
repo (`bluelabsio/14-eng-ai-bot-agent-reports`, `.prawduct/artifacts/design-pod-delta-files.md`), where its build plan and
the reports-side names it mentions (`pod/lsf.py`, `pod/tables.py`, ...) live.

The writing load saves every change to an opted-in table as a file on a shared volume. Pod copies apply every change
from those files instead of reading L3. Nothing expires on a timer, so a restarted pod can always catch up. Built in
0.95.0. These are the pods' own files, per table; the browser does not read them. The browser's per-report files are
plan 2 (`design-frontend-full-and-delta-loads.md`), a separate design and its own release after this one, which uses
the same full-plus-delta shape.

## The problem

A restarted pod can get stuck on old data for good, because three clocks in today's design were never tied together.

| What | Kept for | Set by |
| --- | --- | --- |
| The log that says a change happened | 24 hours | the epoch log's max age |
| The list of which rows a big write changed | about 5 to 20 minutes | the change-notes ring, sized to save space |
| The pod's checkpoint | until enough rows are applied | rows applied, with no link to the row lists |

A restarted pod loads its checkpoint and asks what changed since. If the checkpoint is older than the row lists, it
cannot learn which rows changed, and nothing gets it out. One stuck table also stops every other table from updating.

On the burst replay (2026-11-23) the pod froze for an hour on a follower bug, since fixed. It never saved a newer
checkpoint, so its restart found the row lists gone and stayed stuck on all six AP tables.

The rule the design lacked: **a change is kept until a newer full copy covers it.** Plan 2 already has that rule for
the browser.

## Decisions

1. **Two kinds of files.** Pod files are per table. Browser files (plan 2) are per report and come in the next release.
2. **The writing load makes every file,** deltas and fulls. Pod copies only read them. No new lock: there is one
   writing load at a time, under the write lease it already holds.
3. **Copies apply every change from the files,** not only after a restart. L3 is read only on a true cold start or if
   the volume is lost. This keeps L3's read load flat however many pods run. A missing or unreadable file is recreated by the writer
   (decision 29), never read from L3 by every copy.
4. **The epoch record names the version's delta files.** A pod fetches exactly those; nothing lists the volume. The
   hub's broker writes epoch records at commit, so the pod hands it the file name with the commit (a small hub change).
5. **Opt-in per table.** A table's declaration turns files on. L3 stays the default everywhere in 3tears; the files
   apply only to followed copies in epoch-ring mode.
6. **One delta file per table per version,** holding every state that version changed.
7. **Files live on a shared multi-read volume** (ReadWriteMany; EFS on EKS) mounted in every reports pod, not in NATS.
   Fulls (today's checkpoints) move there too. No bundles of deltas: pods read off the mount.
8. **A full is per (table, state) part files plus a manifest.** Folding writes new parts only for the states that
   changed, writes a new manifest, then renames it over the current pointer in one step. Nothing is changed in place,
   so a reader of the old full is never disturbed. Old parts go only when no manifest in use names them (the previous
   manifest's parts are kept until the next fold). Any plain shared filesystem works: it needs an atomic rename in one
   directory, and a closed file fully visible to a reader that opens it after.
9. **Fold rules.** Deltas carry delete markers. History tables (AP snapshots, LSF runs) append; every other table keeps
   the newest row per key.
10. **Fold per state** once that state's changes since its last fold reach half its part's size.
11. **The delta file is saved before the commit,** under a unique name the commit's epoch record then names. The name
    leads with the write's version (`v<version>-<unique>`, decision 53). A crash before the commit leaves an orphan
    nothing reads, swept later; a crash after it leaves the file in place. Every committed version has its file.
12. **Every write to an opted-in table makes its file,** enforced by 3tears on the table, not by each caller. A reset
    or schema change starts the table fresh: one read of it from L3, its old files dropped.
13. **The fold applies a keep rule the table declares.** 3tears gives the fold a per-table hook; reports plugs in its
    LSF run thinning (`run_thinning.py`: every run for two hours, then one per 15 minutes, then hourly, plus the first
    and newest). The rule is monotone, so thinning the old part plus the deltas equals thinning every run: a full
    holds exactly what a pod copy holds, a few GB a month, never every run.
14. **When only L3 is left, the writer rebuilds the full, once.** A writer that finds no manifest on the volume (a
    lost volume, or a new environment) reads each opted-in table from L3 once, the one allowed whole read (a true cold
    start), writes the parts and the manifest, then makes deltas as usual. Copies never read L3 whole: a copy that
    finds no manifest waits, saying so in its status, and loads the full once it is published. A fresh bring-up takes
    this same path, so it is exercised every time, not only in a disaster.
15. **One writer, fenced by term.** The rebuild and every fold run only under the write lease the writer already
    holds; no other lock. Each lease handover gets a term that only goes up, and every manifest carries its term in its
    name. Readers take the manifest of the highest term, and the newest within it. A writer that lost its lease and
    wakes to publish writes an older term's manifest, which readers ignore and the sweep removes: nothing is ever
    overwritten, so a stale writer can never replace the current writer's work.
16. **A fold reuses a delta file as the part when it replaces the part whole.** When a delta holds only one state and,
    after the keep rule, replaces that state's part entirely (LSF county projections keep each race's newest run only,
    and LSF commits one state at a time), the new manifest names the delta file itself as the part instead of copying
    its rows. Files never change once written, so this is safe; a delta a manifest names is not deleted as covered. On
    election night this saves up to the 3.9 GB the county projections' folds would write, of 8.8 GB in all.
17. **A bulk file load builds its delta from the same files.** LSF loads on dev and prod copy the warehouse's export
    files from S3 straight into L3 (the hub's `load_files`), so no row passes through the pod. The pod reads those
    S3 files itself and writes them as the version's delta file before handing the copy to the hub. One extra S3
    read per load; L3 is not involved.
18. **Files are Arrow IPC with zstd inside.** An open Apache standard the pod's Arrow copy loads without decoding.
    zstd keeps them close to Parquet's size on the volume; plain IPC was only ever needed for the browser's files,
    which these are not.
19. **A delta records which columns each write set, and is applied the way L3 merged it.** An upsert of a row L3
    already holds keeps the old value of every column the write left out, and keeps the row's original
    `date_created`. So the delta carries the columns each write set; the copy and the fold apply it the same way
    (columns set overwrite, columns left out keep their value, `date_created` keeps the held row's). The copy ends
    exactly as L3 does, and L3 is never read.
20. **What the database decides is never guessed.** A column whose database default is a fixed value gets that value
    written into the delta file by the writer, as the database would fill it. A table with a default the database
    computes (a time, a sequence) or with a guarded write (an update that applies only if a condition holds, which the
    database may skip) cannot turn delta files on: 3tears refuses the declaration at startup, since the file is
    written before the commit and cannot know the database's answer.
21. **A commit too big to list its rows is applied from its whole delta file.** Its record carries only a time
    window and, today, its rows are found through the short-lived change notes and read from L3. When its record
    names a delta file, the copy applies the whole file instead: no notes, no L3. Nothing a copy needs to catch up
    then expires, which closes the restart gap of 2026-11-23 at its root.
22. **A copy learns a change's file from that change's record in the NATS epoch log.** One in-memory read per change,
    no L3, and no change to the live row broadcast every pod shares.
23. **A load that replaces a key slice names the slices in its delta file.** A bulk file load (LSF's, decision 17)
    replaces every row under a key prefix, and L3 deletes the old rows first; the writer cannot list those rows without
    reading L3. So the delta file names the prefixes it replaces; a copy, and the fold, clear their own rows under each
    prefix, then apply the file's rows. Same result as L3, no L3 read, no per-row delete markers.
24. **A restarting writer re-learns its unfolded deltas from the epoch log.** The manifest says which version each part
    is current to; every change since has a record in the NATS epoch log naming its delta file (decision 22). The
    writer reads those records at start and resumes folding. No listing of the volume, no L3.
25. **A writer is never stopped mid-commit.** On a stop (a deploy, a node drain) the writer starts no new commit, lets
    the commits in flight finish, then releases its lease so another replica takes the load over at once. Each commit
    is whole with its delta file, so finishing the commits in flight, seconds, is enough; the restart path
    (decisions 11, 24) stays as the safety net. Rollouts take one replica at a time.
26. **A bulk file load the hub is still copying does not hold up a stop.** The writer waits up to 8 s for commits in
    flight (inside the SDK's 15 s storage close and the pod's 90 s grace); a `load_files` copy that runs for minutes
    is left to the hub, which commits it anyway. Its delta file was saved before the copy began (decision 11) and its
    record names it, so copies apply it, and the next writer re-learns and folds it (decision 24). Deploys stay quick;
    nothing is lost.
27. **A delete marker carries its row's state when the table's key does not name it.** Fulls are cut per state
    (decision 8), and a delete marker otherwise carries only the key. The AP results tables are keyed by race
    (`office_key`), whose codes have no single shape a state can be read from. The writer deletes within one state's
    load, so it writes that state into the marker, and the fold files the delete under that state's part.
28. **A lost epoch log is recovered with one recovery delta, never a whole read.** If the NATS epoch log is gone (all
    replicas lost, or the writer down past its 24 h retention), the writer cannot tell the order of the delta files
    left on the volume, nor whether a file's commit landed. The full on the volume is still right up to its version,
    so the writer keeps it: it gathers every key the leftover delta files the full does not cover name (each committed
    change has its file, decision 11; a file the full covers by its name is skipped unopened, decision 53, so kept
    history costs no read and a damaged old file never fails the recovery), reads those keys' current rows from L3 by
    key (a key L3 no longer holds
    becomes a delete marker; a file whose commit never landed simply reads back as L3 has it), writes them as one
    recovery delta, folds it into the full, publishes, and (in delete mode, decision 54) deletes the leftover files it
    covers. Copies, which lost the log too,
    wait for that manifest. A whole read stays only for a true cold start, with no full anywhere.
29. **A missing or corrupt delta file is recreated once by the writer; copies wait.** A copy that finds a change's file
    missing or unreadable reads nothing from L3: it marks that table waiting for the file, keeps serving the rest,
    and asks the writer (a NATS request) for that file. The writer recreates it once: that change's rows read from L3
    by key (the keys from its record) written under the same name, temp name then rename; a commit too big to list its
    keys has them only in the bad file, so the writer recovers that table with decision 28's recovery delta instead.
    Copies then apply the file and go on; a recreated file may hold newer values than the original, which later
    changes apply over, newest per key. Counted and logged. L3 is read once, by the writer, never by every pod.
30. **Which pods may send and serve recreate requests is a per-pod registry setting, like `object_store`.** A tool pod
    row gains a `delta_files` opt-in (a migration, and an `aibots tool-pod` command to turn it on, run by reports'
    provisioning); the hub carries it into the pod's NATS login, and 3tears' `build_permissions` grants the
    `delta-files` forward family, publish and subscribe, in the pod's own namespace only. Not borrowed from another
    flag, not every tool pod, and not a storage bucket used as a message channel.
31. **A lost file of a commit too big to list its rows is recovered from the row lists while they last, else by its
    states.** Such a commit's keys live only in its delta file (decision 21), so neither a recreate (29) nor the
    recovery delta (28) can name them. Its epoch record also names the states it touched. Recovery reads the keys
    from the change-note row lists in the record's window while they still cover it; past that, it re-reads just
    those states' rows from L3, by state. Only on this double failure; counted and logged.
32. **Drift is caught by two checks, on the dashboard.** (A) Each full part's manifest entry carries a fingerprint of
    its rows; every pod periodically fingerprints its own copy of the same state at the same version and compares,
    with no L3. (B) The writer alone, every 15 minutes by default (a per-deployment setting), reads a few hundred
    random keys per table from L3 by key and compares them with its full. Mismatches are metrics naming the table
    (and state, for A); alerts come later, in Grafana.
33. **Every derived cache drops only the rows under a cleared slice.** Caches built without a notion of slices (the
    permissions cache, a copy's version-stamp table, row-follower tables) learn to take a slice, so a slice clear drops
    only what is under it; the epoch rule's exemption for whole drops of those caches is deleted. The shared 3tears
    interface changes, and every product's cache that implements it is updated with it.
34. **Old and new versions never run side by side.** An upgrade may scale the pods to zero, upgrade, then scale up;
    no older writer runs beside a newer one, and the old version never runs again. So nothing keeps a path for an
    older writer's shapes: a slice clear whose prefix does not lead the key (only an older writer sent those) is
    refused loudly instead of walked.
35. **A big file costs the event loop a few milliseconds, not tens.** Per-key bookkeeping for a delta file (keys heard,
    epochs per key) is kept as Arrow off the loop, not as a Python object per key on it: 200k keys left 30 ms turns
    and a 32 ms garbage-collection pause. Loop-turn tests at real size bound it at 10 ms, under the SDK watchdog's
    50 ms stall alarm.
36. **A local copy's read of one state is a declared primitive.** A table's declared state column (decision 27's
    `delta_scope`, or a followed table's scope) is marked `DeclaredScope`; 3tears' `read_state_arrow` reads one state
    of a local copy table by it and refuses outright if pointed at L3. The whole-read rule accepts that read as
    bounded and still flags a filter on any undeclared column. Used by the drift check for tables whose key does not
    name the state.
37. **A copy's working state has one owner; everything else sends it messages.** One task owns the copy's
    bookkeeping (changes heard and not yet applied, waits, what is staged). Heard changes, resets, file arrivals and
    recreate answers are messages on its queue; nothing else touches that state. The bookkeeping is immutable Arrow
    batches, one per change heard: a new change adds a batch, none is ever edited. A pass, a reload or a drift check
    gets a snapshot, the current list of batch references (a few pointers, no row data copied), and reports back
    "applied up to here"; the owner drops what was applied and judges anything that arrived since, a reset included,
    so a stale result is discarded and never applied. Per-key state is never one Python object per key, and no dict or
    set outlives a function except read-only data. The writer's fold gets the same shape: one owner task with commands
    on a queue. "One owner" is whoever holds the copy's pass lock, one at a time (a pass, a reload's apply, a file arrival's
    handling, the drift check); callbacks only post messages, and the holder applies them in order before its work
    (Pace, 2026-11-24: same guarantee as a dedicated task, without one task everything depends on).
38. **Keyed reads take keys column by column, and never slowly.** 3tears' `KeyBatch` holds one array per key column
    instead of one Python tuple per key, each column bound as one array value. Whenever pyarrow is present the arrays
    are Arrow (Arrow mode); a plain sequence is only for processes without pyarrow, which stays optional in core (the
    2026-11-15 ruling). Measured against today's binding at 1k-50k keys on PostgreSQL, YugabyteDB and DuckDB, plans
    unchanged; any path that would be slower runs in Arrow mode instead, with a speed-budget test guarding it. Keys that arrive as Arrow (every bulk path: copies, folds, loads) bind in Arrow mode; small key lists that
    program logic already holds as Python values (a request's ids, an ACL lookup) bind as they are, since converting
    them to Arrow text only to send them measured slower (2,000 UUIDs: 1.27-1.34x on PostgreSQL) (Pace, 2026-11-24).
39. **Keyed statements are always planned with their real values.** PostgreSQL and YugabyteDB switch a prepared
    statement to one generic plan after five runs; for keyed composite reads that plan collapsed (YugabyteDB about
    15 ms to about 430 ms; PostgreSQL 20 to 41 ms). 3tears sets `plan_cache_mode = force_custom_plan` where it opens
    the connections that run keyed statements: about a millisecond of planning each run, never the collapse. A test
    runs the same keyed read many times on YugabyteDB and PostgreSQL and fails if a later run is much slower than the
    first.
40. **A null-safe keyed statement stays on the index.** `KeyBatch(null_safe=True)` used `IS NOT DISTINCT FROM`, which
    no index serves (11 s for 1,000 keys on 200k rows). Nothing in production sets it today (every key column is NOT
    NULL in L3), but it becomes the split form anyway: keys with every value present go through the indexed lookup, keys
    with an empty part through `IS NULL` pins, as 3tears already does for warehouse keys. A guard test fails if the
    scanning form is ever generated against L3.
41. **The copy follower holds no mutable dicts or sets** (Pace, 2026-11-23, after the boundary review). Its per-table
    bookkeeping (file waits, versions per state, staged scopes, drops in flight) is immutable values the owner replaces
    whole, as `DeltaFolds` does. "Small, owner-only, under the pass lock" is not an exception, and the dict-state rule
    is tightened to catch a mutable attribute changed outside `__init__`.
42. **Keyed reads send no per-column bounds on YugabyteDB** (Pace, 2026-11-23). Beside the batched join, the
    `col = ANY(...)` bounds cost about 50 ms of YSQL CPU per 1,000-key page: 50k composite keys took 1.8-2.1 s with
    them and 0.24-0.29 s without; 1,000 prefixes (30k rows) 1,020 ms with, 37 ms without. PostgreSQL keeps them (they
    help its prefix reads, 14 vs 22.5 ms). A live speed budget on the YugabyteDB pool holds it.
43. **A load that may have committed part of its rows names its file as not confirmed** (Pace, 2026-11-23, option
    C). The hub commits a bulk load in 50k-row transactions, so a failure after the first commit leaves rows in L3.
    The epoch record names the request's file marked not confirmed (a new field, expand/contract); copies never apply
    it and never read L3 for it, they wait; the writer reads that request's keys (taken from the file) from L3 once,
    rewrites the file under the same name with what L3 holds, and confirms it. The pod writes no no-file record of its
    own for a request the hub owns; a hub crash with no record is found and re-read by the writer the same way. For an
    opted-in table, a partition drop from a no-file record refuses loudly instead of reading L3 in every copy.
44. **A bulk load writes its intent before its first commit** (Pace, 2026-11-23). The hub writes the unconfirmed
    record, naming the file and the hub instance, before it commits any rows, and the confirmed record once the load
    finishes. A hub that crashes mid-load leaves the intent behind. The writer confirms it (decision 43) only once the
    hub's own record supersedes it or that hub instance is provably gone by the hub's NATS liveness, never by elapsed
    time. Cost: one more record write per bulk load; copies still read nothing from L3.
45. **All of 3tears holds no mutable dicts or sets** (Pace, 2026-11-23, extending decision 41). The dict-state rule
    covers every 3tears package with no allowlist; state is immutable values replaced whole through the one shared
    helper `threetears.core.frozen_state` (stdlib only), and key batches without pyarrow through
    `threetears.core.packed_keys`. Where rebuilding whole would cost O(all entries) per change, the state is chunked:
    `shared_arrow`'s file tally is one Arrow batch per directory; the L1 ledger's recency is per-table Arrow batches
    handed to the ledger thread; the L1 fence and write buffer hold packed key runs; the LangGraph saver's bypassed L2
    keys and the git change set's index are append-only immutable chunks. Existing 3tears pieces are reused before
    anything new (`ObjectResolutionCache`, `KeyedTaskRegistry`).
46. **The rule's scope is state that is shared or outlives its call** (Pace, 2026-11-24): "I never wanted to restrict
    function local use of them, those are safe, but everything else that isn't is where we need to fix." A dict, set,
    list or deque built and changed inside one function, or inside an object that never outlives the call that made
    it (a per-call builder, a CLI report filled by its helpers), is allowed. Everything else, attributes of long-lived
    objects, module state, class-body fields, anything another coroutine or thread can reach, is changed only by
    replacing an immutable value whole, in the approved shapes (decision 45's table). The checker computes the
    difference as part of the rule, never as a waiver list.
47. **A FIFO hand-off between coroutines is an `asyncio.Queue`** (Pace, 2026-11-24), behind the bound the queue
    already had: one owner per end, safe on the loop, never a tuple rebuilt per put and get (quadratic at a 64 MiB
    read-ahead or a 524k-message backlog). The checker counts an `asyncio.Queue` attribute as an approved hand-off.
48. **3tears knows its tables from a declared manifest, never from what happens to be imported** (Pace,
    2026-11-24). The tenancy, write-generation and L2-first registries filled only as collection classes were
    imported, so a process that never imported a class (the hub and MemoriesCollection) wrote a tenanted table's
    record as `_all`, and could treat a table as having no write generation. Each package and app lists its collection
    modules once as a `threetears.collections` entry point; 3tears imports every listed module at startup, so the
    registries are complete. Two steps: this release logs loudly and counts a table no listed module declares; the
    next release, once every 3tears app has its list, refuses it.
49. **The dict and race rules cover serving code; non-serving code is declared by root** (Pace, 2026-11-24,
    refining decision 45's "every package"). Code no serving process can reach is listed once per repo as a
    non-serving root, each with its why, in `tests/enforcement/serving_code_support.py`, which the dict-state and
    await_race checks both read; never per site. The roots were found from an import graph seeded by the Dockerfiles,
    Procfiles and console scripts, so anything a serving path imports stays in scope: 3tears `enforcement`,
    `core/testing` and `evals` (nothing serving imports them); the hub's five operator CLIs; reports' 44 operator,
    preview and CLI modules (36 ops modules `pod.serve` imports stay in); the SDK's CLI-only, devx-only and testing
    modules.
50. **The recovery that recreates bad files also detects no files** (Pace, 2026-11-24: "something needs to check for
    the files and create them if they are not there"). A local burst replay started a pod on an empty volume while L3
    held the data; its first load had nothing new to write, never began a write, so the fold (started at a write's
    first begin, decision 24 and the 11-23 audit) never rebuilt the full, and every copy waited for ever. Now a copy
    whose volume's full lacks a table asks the writer through decision 29's request (`{"table", "want": "full"}`,
    3tears `DeltaRecreateRequester.full`); the writer's store answers `recovering` and has the fold rebuild the full
    from L3 (decision 14's cold start, its one whole read) with no write begun, at the highest version any writer took
    (the epochs' committed or begun), which its next begin passes. The writer also checks as a holding begins, off the
    load and only while the fold's start is owed (one manifest read), so one replica recovers with no copy asking. A
    volume holding its fulls costs nothing more; no copy reads L3.
    Only a load takes the lease, so a pod restarted on an empty volume whose start needed no load had no writer: its
    copy heard `no_writer` until the self-check's load 60 s later (local stack, 2026-11-24). Now the copy's first
    `no_writer` of its wait starts the pod's load as the self-check asks it (`writer_holding.PodRecreate`,
    `RefreshService.load_for_full`), once per wait and never when a load runs or is queued, beside the copy's look;
    that load's holding check rebuilds the full.
51. **Delta files are kept by default for reports; deleting them is an option** (Pace, 2026-11-24: "sometimes we might
    want to delete deltas; it should be an option and we will happen to run in don't-delete mode"). The fold had
    deleted each delta the moment a new full covered it, with no input about copies, against decision 21 ("nothing a
    copy needs to catch up then expires"): on the 2026-11-24 replay copies lagging their writer by a median 7.7 s found
    536 files gone, each a lost file, and 91 whole-table reloads of 339k rows followed. Now 3tears' fold takes a
    required `retention` (`DeltaFolds(retention="keep" | "delete")`, no default: which a deployment runs changes what a
    lagging copy finds on the volume). The reports pod runs `"keep"`: nothing deletes a delta file (not the fold, its
    rebuild or recovery, nor the orphan sweep), because the files are history across report sets and the front end's
    time travel; a later plan moves old ones to cold storage. Kept, the fold deletes only its own scratch: manifests
    and parts no manifest in use names, temporary files of writes that died, and its rebuild's and recovery's work
    files. `"delete"` deletes a covered file only once no copy can need it (decision 54).

52. **A whole-table reload never stalls another table's copy** (Pace, 2026-11-24). On a local burst replay (one
    reports pod) writer commit to copy publish was a median 0.5 s with no reload between and a median 9.4 s, up to
    45 s, with one: the copy's reload held the pass lock through its stage and apply (big reloads a median 12.6 s,
    up to 25 s), and the pod's one apply worker ran reload, merge and change one after another. Now (3tears
    `CopyFollower._reload_from_volume`) a reload reads and stages its table's full beside the copy in a stage of its
    own (`reload_stage`) with no lock held, has the backend rewrite the table (`PreparesStaged.prepare_staged`), and
    takes the lock only to judge and swap; a change to that table heard meanwhile stays held in order and is applied
    over the full after the swap, as before. In the pod (`SharedArrowBackend`) a reload has its own lane: its pages
    and the rewrite of the table's every part go to a reload worker of their own (`enr-reload`, one process, started
    on the first reload, retired past the apply worker's private-memory bound), outside `_applying`; only its settle
    over the version current then and one publish of the table with what the settle derives hold the apply worker.
    The settle stays on the apply lane because it reads every other table at the version it publishes on; settled
    earlier, a table would be published from a mix of old and new. Each table's epochs still apply in order, and a
    table is never published half old, half new. In the same code, keys stop being a Python object each (decision
    35, ~10 s of event-loop stalls over a 40-minute run): a local write's keys reach the copy as one Arrow batch
    (`CollectionRegistry.keys_told`, no row message built for it), and a run of a peer's row messages is read into
    Arrow off the messages (`CopyFollower._flush_rows`), never a list a key. The bus envelope still names one row.

The checkpoint trigger added on 2026-11-23 (3tears ab07a6764: a checkpoint due a quarter note fence after an epoch
taken from the notes) is no longer needed once this lands, since nothing expires on a timer; it is removed then.

53. **Delta files are named by their write's version, and each full says exactly which it covers** (Pace,
    2026-11-24, replacing a clock-based coverage that was tried and rejected: no clock heuristics, no TTLs, trust
    epochs). With deltas kept forever (decision 51), decision 28's recovery listed a table's whole delta folder and
    read every file's keys, and one damaged old file failed it every time. Now a file's name carries the version
    `ScopeEpochs.begin` gave its write (`v<version>-<unique>`; every data write already holds it before it saves its
    file: `ScopeEpochs.touch` notes it on the transaction, a load from files passes `write_version`). Each full's
    manifest records per table `folded_below` V and an `open` list: every file below V whose commit landed is folded,
    except those listed. V is the writer's term, or the oldest write it has a commit in flight of when lower
    (`DeltaFolds.committed(in_flight_from=)`, required, replacing `in_flight_since`'s time; the reports pod's commit
    gate tracks each commit's version). Every older write's commits have ended: told to the fold (held unfolded, so
    listed open until folded) or never landed. Open also lists a failed load request's file waiting to be confirmed
    and this process's saves not yet ended. The recovery reads only files at or above V, the open ones, any the fold
    holds told, and old-named ones; a rebuild or recovery deletes (delete mode) only by this coverage or a file told
    before its read, never by a file's time on the volume. **Expand/contract:** this release writes versioned names and
    reads both; a file of the older name (a bare id, `recovery-<id>`) is never covered, so it is read as before, and a
    manifest an older writer published (no `folded_below`) claims nothing. The old form is dropped in the release after
    this one has run everywhere (decision 34: old and new never run side by side, so every old-named file then predates
    every running writer): 3tears makes the version required at every save (`save_delta_files`, `load_files`), and the
    first full a writer of that release publishes with no old-named file open records that every old-named file is
    folded, after which the recovery skips them too.

54. **Delete mode deletes a covered delta only once no copy can need it** (Pace, 2026-11-24). **First rule: when in
    doubt, keep** ("really try to make sure we don't delete the deltas until all the pods don't need it"). Every doubt
    keeps every file that round: (1) a copy registers a `starting` beat before it loads or follows anything, and the
    fold deletes nothing while any copy is starting; (2) a beat that does not decode, lacks the table, or is of another
    incarnation than the file's, or a read of the beats that fails, deletes nothing that round; (3) a copy counts as
    gone only once its beat is removed, and only by the confirmed path that clears a dead writer's: the replica holding
    the recovery lease removes a beat not renewed for the writers' stale time (30 s, six 5 s beats), at the revision
    it read, so a beat renewed meanwhile is never taken; no reader judges a beat by its age; (4) the fold reads the
    beats again immediately before each delete, never once a round; (5) a delete is idempotent and logged per file with
    the beats that allowed it. **The shape.** (a) Each copy keeps one beat in the pod's own key scope of the `leases`
    bucket (`<pod scope>.delta-copies.<copy>`, 3tears `CopyBeats`): the newest epoch it has applied per table, renewed
    every 5 s and when its marks change, deleted on a clean stop. A live copy is one whose beat is kept. No grant
    changes: the pod's replicas already read and write their own scope of that bucket through their lease factory, and
    the fold reads only its own pod's copies. (b) `DeltaFolds(retention="delete")` requires `copies=` (refuses to start
    without it, saying how to fix it). The fold's workers never delete a delta file; the owner deletes a covered one only
    when the current and the previous full both cover it by its name (decision 53), no manifest in use names it, it is
    not in flight (told and unfolded, waiting to be confirmed, a save of this process not ended), and every live beat
    for its table is at or past its epoch (its told epoch, else the previous full's version of the table) in the same
    incarnation. A beat behind keeps that file only. (c) A copy that finds a file gone asks the writer as before; the
    writer answers `covered` (no L3 read, nothing written) when the current full covers it, and the copy, on that
    answer, stops asking and waits for a full at or past the file's epoch, never reading L3. Keep mode is unchanged: no
    delta file is ever deleted. **The fallback loads only the touched states** (Pace, 2026-11-25; 3tears
    `work/0950-covered-load`). On `covered`, or a lost file of a commit too big to list, the copy waits for a full
    holding the file's epoch and loads from it only the parts of the states the commit wrote, never the whole table and
    never L3: the copy's rows of those states leave and the parts' rows go in, in one change staged beside the passes
    on the reload lane, and every other state stays as the copy holds it. What it heard of those states while it waited
    at an epoch the part holds is dropped; the rest applies from its files. The table's mark is not moved; the file
    wait ends, so the held changes apply. **Only on states known complete.** A record named the scopes its writer
    touched, which nothing tied to the rows written: a record naming a strict subset would leave a copy with the old
    rows of a state it missed. So a commit marks its record's states complete (`EVERY_STATE_NAMED`) only when it reads
    them off its own delta files' rows, every row and delete marker carrying its state (`TableSchema.
    delta_state_scope`, its column a key column or `delta_scope`), and then names every one. A record without the mark
    (an older writer, a table declaring no state scope, a load) reloads the whole table, as does a full not split by
    state, a table whose key does not name its state, or a cleared prefix heard that names no state. A failed swap drops
    both of its stages, the keys as well as the rows. **For reports:** every delta-files table declares
    `delta_state_scope=(STATE_COLUMN, STATE_SCOPE)` (`tables.py`; each is keyed by its state or carries it as
    `delta_scope`; an LSF table keyed by no state would declare none), so a page write's record names its states
    marked complete (`tests/pod/test_commit_states.py`). The fold's own recovery by states (decision 31) trusts a
    record's states only when marked; an unmarked record of a scoped table is recovered by every state the table holds
    (the full's and L3's, counted `every_state`; 3tears `work/0950-scope-mark`). The reports pod runs keep mode, so
    neither path runs there yet.
55. **AP marks are kept per state; a load is a bounded slice; one failed page never throws a load away** (Pace,
    2026-11-25: "a load's time tracks the rows it writes, never the point in the night; one slow or failed page never
    throws away a whole load"). **Before:** one mark per table; one `_PagesFailed` cancelled every state's task and the
    load committed nothing, so the next load named the whole backlog again, and a backlog grew each load's time with the
    night. **Now:** (a) **Marks per state.** Each report table keeps rows `(<table>#mark, <state>)` (and `#changed` for
    the stamped candidates) beside its `WHOLE_TABLE` row, which is the low-water mark (the oldest state's), read by the
    records' key prefix, never a scan. The change statement ORs each state's own mark (3tears
    `counts_by_relations(ahead=)`, still one statement, every part bounded by the table's mark), so a state already
    ahead names no rows. A state written moves to the newest stamps named (never back); one failed or left keeps its
    mark; a state no longer ahead has its row dropped. (b) **A failed state fails alone.** `_PagesFailed` (after every
    retry; deadlines are the page-deadline work's) is caught per state for a marked state (a cold start still fails the
    load). The commit saves the other states' marks and commits the version withholding the failed state's scopes and
    any shared scope its writes touched (3tears `ScopeEpochs.commit(withheld=)`: they stay pending, the copy follower
    holds them as a write in progress's, so no reader sees the rows left part way). The failed state is recorded
    (`#withheld`, its scopes in the digest), so every later commit, of either side, on any replica, withholds it too,
    until the AP load that writes it whole: that commit touches its scopes again (they move with it) and drops the
    record. Side by side, the failure reaches the shared version's gate while the state still holds it, so no LSF
    checkpoint commit moves its scopes first. (c) **Bounded slices.** A load takes states oldest mark first until their
    change-statement rows reach `LOAD_SLICE_ROWS` (`FOURTEENAIBOTS_ENR_LOAD_SLICE_ROWS`); the rest keep their marks for
    the next load. A state alone past it is cut on the append-only snapshots by upload time (oldest uploads first, never
    between two of one upload stamp), its mark moving to the cut; the replaced tables (candidates, geos) do not grow with
    the night and are taken whole. An operator's reread is always taken, its own mark dropped first. A reset clears every
    state's mark and every `#withheld` record by key prefix; the cold start is unchanged. **N from replay 6**
    (2026-11-24, `page written` lines of the pod log): the AP tables wrote 1,383,161 rows in 2,333.7 s of page time, 593
    rows/s a page slot (snapshots 571,555 rows in 564.1 s, 1,013/s; candidates 776,771 in 1,597.8 s, 486/s; geos 34,835
    in 171.8 s, 203/s; LSF projections about 920/s beside them); x `PAGE_WRITERS_AT_ONCE` 4 = 2,372 rows/s; x a 60 s
    load = 142,300: **N = 140,000**. The count is rows the change statement names, at least the rows written (only rows
    that differ are written), so a load stays at or under the target. **Not built** (said, not dropped): a cold start
    committing per state after its first per-state commit; the cold start still loads every state in one commit.

## How it works

The writing load, for each commit to an opted-in table:

1. Write the version's delta file to the volume under a unique name: the rows written, with delete markers.
2. Commit to L3; the epoch record names the delta file.
3. After the commit, off the serving process (a worker process): for each state whose changes reached half its part,
   write its new part from the old part plus the deltas, then the new manifest, then rename it over the pointer.
   Delete the parts and manifests no manifest in use names. The deltas the new full covers are kept (decision 51:
   reports runs the fold with `retention="keep"`; a lagging copy still reads them).

A pod copy:

1. On start: read the manifest of the highest term and load each (table, state) part. With no manifest, wait for the
   writer to publish one (it rebuilds from L3 once); a copy never reads L3 whole.
2. Then apply, in order, every delta newer than each part's version.
3. Running: each epoch record names its delta files; read them off the mount and apply them. If a named file is
   missing or unreadable, wait on that table and ask the writer to recreate it (decision 29); never read L3.

## Sizes on the election-night capture

Measured on the capture of 2026-11-03 21:45 to 2026-11-08 02:15 (716 AP uploads, 2,009 LSF runs), as parquet with
zstd, which is close to the Arrow IPC the files would be.

| Table | Rows over the night | Per version (median / p90 / max rows) | Size over the night | Per version (median / max) |
| --- | --- | --- | --- | --- |
| LSF projections | 265,167,258 | 151,133 / 161,214 / 162,345 per run | 3,720 MB | 2.0 / 2.2 MB per run |
| LSF feed_why_sources | 7,109,902 | 1,800 / 11,687 / 29,030 per run | 152 MB | 37 / 618 KB |
| LSF feed_why | 2,711,740 | 688 / 4,916 / 9,772 per run | 146 MB | 39 / 536 KB |
| LSF feed_county_targets | 3,593,874 | 1,980 / 3,600 / 3,600 per run | 126 MB | 60 / 153 KB |
| LSF feed_county | 1,489,730 | 432 / 2,177 / 4,576 per run | 54 MB | 14 / 180 KB |
| LSF feed_race, feed_state, runs | 178,740 | 72 per run | 33 MB | under 20 KB |
| AP result_snapshots_by_id | 948,033 | 322 median, 48,480 max per upload | 19 MB | about 20 B a row |
| AP latest_results_candidates_by_id | 339,640 held at the end | changed rows per upload | 14 MB | about 43 B a row |
| AP latest_results_geos | 16,371 held at the end | changed rows per upload | 9 MB | about 535 B a row |

What this says:

- **Every delta of the night together is about 4.3 GB,** 3.7 GB of it LSF projections. That is the most the volume
  holds in deltas if nothing ever folded; folding deletes covered deltas, so it holds less.
- **A typical AP delta is a few KB to tens of KB** (322 rows); the largest, the night's first upload, about 1-2 MB.
- **A typical LSF version is about 2 MB,** almost all projections. The pod commits LSF a state at a time, so each
  state's commit is a slice of that.
- **The real night's AP uploads are small.** The burst replay's 150k-460k-row AP loads were hundreds of uploads piled
  up while the pod's first loads ran (the harness sends the next upload about once a second, without waiting for the
  pod); at real speed a load is hundreds to about 15,000 rows.
- **EFS sizing is comfortable:** a night is single-digit GB of deltas plus fulls about twice the copy's size.

## Open questions

1. **Fold cost: measured (2026-11-23).** Folds write 8.8 GB over the night, 1.86 times the 4.7 GB of deltas. That
   matches the estimate of about twice. The rate was wrong: the capture covers 100 hours, so 8.8 GB averages 0.02 MB/s.
   The busiest hour (2026-11-04 03Z) wrote 385 MB, 0.11 MB/s. The volume never held more than 74 MB of deltas at once.
   At the end, the fulls of every table and state come to 186 MB. They stay that small because the thinning keeps the
   parts small. As a result, folds are frequent and small: about 50,000 over the night.

   | Table | Versions | Deltas | Folds | Fold bytes | Peak deltas held | Largest part | Fulls at end |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | LSF county projections | 81,404 | 2,148 MB | 34,407 | 3,887 MB | 2.6 MB | 0.6 MB (TX) | 6 MB |
   | LSF projections | 83,413 | 1,572 MB | 2,115 | 3,160 MB | 55 MB | 10.7 MB (US) | 96 MB |
   | LSF feed_why_sources | 39,115 | 152 MB | 1,729 | 296 MB | 5.1 MB | 2.4 MB (TX) | 11 MB |
   | LSF feed_why | 45,559 | 146 MB | 2,201 | 285 MB | 6.3 MB | 8.4 MB (US) | 10 MB |
   | LSF feed_county_targets | 44,060 | 126 MB | 2,931 | 246 MB | 2.4 MB | 0.2 MB (SC) | 5 MB |
   | LSF feed_county | 44,060 | 54 MB | 1,242 | 111 MB | 4.1 MB | 3.7 MB (US) | 11 MB |
   | LSF feed_race, feed_state, runs | 48,816 | 33 MB | 1,469 | 71 MB | 3 MB | 4.1 MB (runs) | 8 MB |
   | AP latest_results_candidates_by_id | 9,787 | 300 MB | 1,791 | 429 MB | 3.9 MB | 1.4 MB (TX) | 14 MB |
   | AP latest_results_geos | 9,787 | 190 MB | 1,783 | 270 MB | 2.5 MB | 0.6 MB (TX) | 9 MB |
   | AP result_snapshots_by_id | 9,787 | 19 MB | 364 | 43 MB | 3.2 MB | 1.1 MB (TX) | 16 MB |
   | **Night** | 415,788 | 4,739 MB | 50,032 | 8,798 MB | 74 MB | 10.7 MB | 186 MB |

   The five busiest states by fold bytes: TX 607 MB (983 folds), GA 550 MB (986), US (national) 327 MB (265), KS
   321 MB (988), FL 309 MB (1,040).

   **The county projections table folds almost every version.** Its keep rule holds only each race's newest run. So
   every new run is at least half of the part, and the fold writes the delta out again. That table accounts for 69%
   of the folds and 44% of the fold bytes.

   **How it was measured.** The script replayed the capture's events in order. LSF versions came at each run's
   `finished` and AP versions at each upload's `uploaded`. LSF had one version per run and state, with rows placed as
   `pod/lsf.py` places them. A feed row went to its race's state, and a pooled row to `US`. Each new part applied the
   tables' declared `Retention` from `pod/tables.py`. Thinning used the real `Thinning.kept_runs`, and the windows used
   `FEED_WINDOW_MINUTES`. The script ran over every run seen so far, which equals old part plus deltas because the
   rule is monotone.

   It made three approximations:
   - **The holding rule** (`minutes=0`, `holding=geo_level, geo_id, quantity`) was taken as the newest run per race
     and model. Checked exactly in DuckDB at the end of the night, it keeps the same rows: GA 4,854 of 1,600,592; TX
     5,212 of 1,820,078.
   - **AP rows per upload.** The snapshots are split exactly by state and upload. The capture holds only the final
     latest tables, so a latest-candidates delta was taken as that upload's snapshot rows times the state's metric
     rows per candidate-geo (about 7). That is an upper bound: it assumes every metric row changed. A geos delta was
     taken as the upload's distinct (race, county).
   - **Bytes** are rows times each table's parquet+zstd bytes a row over the whole capture: projections 14 B, feeds
     21-237 B, AP per state from its own file. A small slice compresses worse, so small deltas and parts are somewhat
     understated.

   An empty part folds on its first delta. That is the cold start, counted here as a fold. The scripts are in the
   session scratchpad (`agg_lsf.py`, `sim.py`, `holdcheck.py`).
2. **Storage class: from the config, not confirmed live.** No `aibots` command lists StorageClasses or PVCs.
   `cluster show` reads Deployments and Argo, and `resize-volumes` reads a StatefulSet's claims. So this comes from
   `bluelabs-eng-k8s-deploy/tf/cobalt/storage.tf`, the same Terraform for dev and prod with
   `dev.tfvars`/`prod.tfvars` and separate state backends.
   - **The file system.** Each cluster has one EFS file system (`module "efs"`, terraform-aws-modules/efs `~> 1.1`,
     encrypted). It sets neither `throughput_mode` nor `performance_mode`, so AWS's API defaults apply: **bursting**
     throughput and generalPurpose performance. Bursting baseline grows with the bytes stored (about 50 KiB/s per GiB
     held), and credits allow bursts to 100 MiB/s. The folds' 0.11 MB/s peak and each pod reading about 0.06 MB/s of
     deltas in the busiest hour sit far below that.
   - **The classes.** Four StorageClasses share that one file system, so they share its throughput. All use
     `efs.csi.aws.com` with `provisioningMode: efs-ap` and `reclaimPolicy: Retain`:
     - `efs`: directory perms 700, `iam` mount.
     - `efs-logs`: `/dynamic/<namespace>/<pvc>`, perms 777, uid/gid 0. DIPP, Loki and Prometheus use it.
     - `efs-postgre`: uid 1001.
     - `efs-universal`: `/dynamic/<namespace>/<pvc>`, perms 777, uid/gid left to the pod's `fsGroup`, `iam`+`tls`
       mounts.
   - **The reports pods.** Today they mount no volume: `.devops/chart` has only an `emptyDir` in the tool pod. They
     would get a new ReadWriteMany PVC. Of the four classes, **`efs-universal`** fits non-root pods. `efs-logs` makes
     root-owned directories.
   - **Confirmed live (2026-11-23, read-only `aws efs describe-file-systems`):** cobalt dev `fs-0d2cdaae146ffd593`
     and cobalt prod `fs-07464b5f16cb746b2` are both generalPurpose, **bursting**. Bursting's baseline is about
     50 KiB/s per GiB stored: dev holds 335 GiB (about 17 MB/s baseline), prod 549 GiB (about 27 MB/s). The night's
     busiest hour of folds is 0.11 MB/s, so no throughput change is needed.
3. **Local stack and replay:** a shared Docker volume stands in for EFS, so the burst replay proves the same paths.
