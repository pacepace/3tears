"""the writer's fold: each state's new part of a full, from its old part and the delta files since.

**Why.** A table that opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`)
saves every commit's rows as a file a pod copy applies (:mod:`~threetears.core.collections.delta_files`), and a copy
starts from the full on the same volume (:mod:`~threetears.core.collections.volume_fulls`). Nothing expires on a timer,
so without a fold a restarted copy would apply every delta since the full was first written, and the volume would hold
every one. The writing load folds them: a change is kept until a newer full covers it (``design-pod-delta-files.md``,
decisions 2, 8, 9, 10, 13, 15, 16, 19 and "How it works", the writer's step 3).

**When a state folds** (decision 10). :meth:`DeltaFolds.committed` is told each commit's delta files after the commit.
A delta file holds every state its commit wrote (decision 6), so each file's bytes are shared among its states by their
rows. Once a state's bytes since its last fold reach half its part's size on the volume, the state folds; a state with
no part yet folds at its first delta.

**How a state folds** (decisions 9 and 19). The old part's rows, then each delta's rows of the state in commit order,
merged as L3 merged them (:func:`~threetears.core.testing.delta_files.apply_delta_row`: a column the write left out
keeps its value, ``date_created`` keeps the held row's, a delete marker deletes), the newest per key. A delta that
clears key prefixes (a load that replaced them, a key-prefix delete; decision 23) first takes every row of the state
under each out, then its own rows apply: a state a prefix names by its scope column shares the file, and every state
of the table does for a prefix that names none. A history table
(AP snapshots, LSF runs) has a key unique to each run or upload, so the same rule appends its rows. Then the table's
keep rule (decision 13, :class:`~threetears.core.collections.delta_keep.DeltaKeepRule`), the one its copy keeps by: a
table that declares none keeps every row.

**A delta as the part** (decision 16). When the last delta holds only the state and, after the keep rule, the folded
rows are exactly its own (a race's newest run replacing the last), the manifest names that delta file as the part and
no row is copied. Files never change once written, and :func:`~threetears.core.collections.volume_fulls.
deletable_full_files` never lists a delta a manifest in use names; the fold keeps such a delta in its list until none
does, then deletes it (when it deletes at all: ``retention``, below).

**Published by term** (decision 15). The new parts, then a manifest naming them and every part kept, at the writer's
term, through :func:`~threetears.core.collections.volume_fulls.publish_full_manifest`. The term is the version of the
writer's own write (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`): the fencing version taken
under the write lease, durable in L3 and only ever higher for a later writer, never the NATS KV's revision, which a
memory-only bucket resets. A writing load takes its lease and then begins, so it is the first ``begin`` after the
lease; a writer that lost its lease publishes an older term than the next writer's, which no reader takes, or is
refused (:class:`~threetears.core.collections.volume_fulls.StaleTermError`) and changes nothing.

**Each full says exactly which delta files it covers** (decision 53). A delta file's name carries its write's version
(:func:`~threetears.core.collections.delta_files.versioned_write_id`: the version ``ScopeEpochs.begin`` gave the write,
which every data write of it holds before it saves its file). Each publish records per table
(:attr:`~threetears.core.collections.volume_fulls.FullTable.folded_below` and ``open``): every file below version V
whose commit landed is folded, but the files it lists open. V is the writer's term, or the oldest write it has a commit
in flight of when lower (:meth:`DeltaFolds.committed`'s ``in_flight_from``, required): every commit of an older write
has ended, told to the fold (held unfolded, so listed open, until folded) or never landed. Open: the files queued, in
the round, in shares not folded yet, a failed load request's file waiting to be confirmed, and this process's saves
whose commit has not ended. Versions, never times: no clock is compared. A file of the older, unversioned name is
never covered (read as before; decision 53's expand step).

**Kept or deleted after the publish: the fold's ``retention``** (:data:`~threetears.core.collections.delta_files.
DeltaRetention`, required, no default; ``design-pod-delta-files.md`` decision 51). Always deleted: the fulls' own old
files, every part and manifest no current or previous manifest names (a reader holds one of those two; a part is
written only for the next manifest of the one writer of the term) and a temporary file a write that died left, and the
rebuild's and recovery's work files beside the fulls. With ``"keep"`` (the reports pod's) nothing else: no delta file
is deleted by the fold, its rebuild or its recovery, so a copy that lags its writer always finds the file its next
epoch names, and the files stay as history. With ``"delete"``, also each delta the new full covers (every state it
held is folded past it) and every orphan delta file this process saved for a commit that did not land (:data:`~threetears.core.collections.delta_files.OWN_DELTA_SAVES`,
:func:`~threetears.core.collections.delta_files.sweep_orphan_delta_files`), opened by its name: no directory is listed
and no file's time compared with the writer's clock, so a file of work in flight, this writer's or another's, is never
taken. A file of a process that died before its commit is left (a rebuild or a recovery of its table drops it). Kept,
an orphan's temporary file alone is swept.

**Never on the commit path.** :meth:`DeltaFolds.committed` only notes the files and starts a round in the background,
off the commit; each round runs in a worker process (:class:`~threetears.core.process_workers.ProcessWorkers`), since
reading, merging and writing parts is CPU, and hands back only names, versions and counts. A round that fails is
logged and counted (``delta_fold_rounds_total{outcome="failed"}``) and its files are kept for the round the next commit
starts; the write it followed has committed already and never fails for it.

**The rebuild from L3** (decisions 12, 14 and 15). A table the current full does not hold is never folded: a full made
from deltas alone would lack every row written before them. The writer rebuilds it instead, once: :meth:`DeltaFolds.
start`, called at the writer's start under its write lease, finds each folded table the current full lacks (no full
at all: a lost volume, or a new environment; or a full without the table: one newly opted in) and reads it from L3
whole, the one whole read a writer makes, declared a cold start; :meth:`DeltaFolds.start_fresh` marks a table reset or
recreated by a migration (its files no longer valid) for the same, read declared as its situation. The table's
generation is read first, then its rows: a commit landing during the read is in the rows and its delta is applied
again over them by every copy, which ends where L3 does (decision 19's merge, applied in order, leaves a row as its
last write left it). Every table's generation is read before any table's rows, and the tables are read several at
once (:class:`RebuildConcurrency`: up to 4 tables, 6 reads in all), a table past one page split by its key's lead (its
states) into ranges of it (an ordered lead) or sets of its values (YugabyteDB's hashed lead), each read writing its own
file. The rows are written beside the full as they are read (I/O, the serving process's), then a
worker process splits them by state, applies the keep rule, writes the parts and publishes the manifest at the
writer's term, before any fold of the table: :meth:`DeltaFolds.committed`'s rounds run a rebuild owed first, and a
table owed one is not folded meanwhile. Then its old parts go once the previous manifest no longer names them, and,
deleting, every delta of the table that neither manifest in use names and its full covers by name (decision 53; an unversioned file is never dropped
here). A table with no generation yet (never written) is rebuilt before the round
its first commit starts. A writer whose term is no longer the highest reads nothing and publishes nothing; one whose
lease is taken during its read is refused at the publish, deletes only its own new files, and changes nothing a
reader sees. A copy never reads L3 whole: it waits for the manifest (:class:`~threetears.core.collections.
volume_fulls.VolumeFulls`).

**No full, and no write begun** (Pace's ruling of 2026-11-24: the mechanism that recovers bad files also detects no
files). A writer whose load has nothing new to write never begins, so a fold started only at a write's begin never
rebuilt a full the volume lacks, and every copy waited for it for ever (a pod started on an empty volume while L3 held
the data: local burst replay, 2026-11-24). Now :meth:`DeltaFolds.lacking` names the tables the current full lacks (one
manifest read: a volume holding its fulls costs nothing more), and :meth:`DeltaFolds.full_owed` answers a copy's ask
for a table's full (``delta_recreate``): a fold already started looks again before its next round, and its rebuild
(the same cold start, the one whole read) publishes the full with no commit told. A fold not started yet is started by
its writer at the term its lease holds (the consumer's: the reports pod's store starts it when it takes the lease and
the full lacks a table), never per load.

**A restarted writer re-learns its unfolded deltas** (decisions 22 and 24). The fold's memory of each state's deltas
since its last fold dies with its process. :meth:`DeltaFolds.start` learns it again from the NATS epoch log, never
from a listing of the volume and never from L3: the current full says which version each part is current to, and
every commit since has a record in the log naming its delta file. Each table's records after its oldest part's version
are walked (``changes_after``), each file named is noted as a commit is, and a round drops each state's share a part
is already current to (folded before the restart); a file left with no share is deleted with the round's covered
files when the fold deletes. So folding resumes where it stopped, at the same half-size point.

**A lost log is recovered with one recovery delta, never a whole read** (decision 28). A table whose generation is of
a new incarnation (its store replaced: the memory-only log lost), or whose walk misses a record, has parts current to
versions no record orders, so the writer cannot tell the order of the delta files left on the volume nor whether each
one's commit landed. The full is still right up to its version, so it is kept: the writer lists the table's own delta
folder (here, and only here, the volume is listed; names only), skips every file the kept full covers by its name
(decision 53: below its claim and not open, so never opened: the kept history costs no read and a damaged old file
never fails the recovery), gathers every key the files left name and every key prefix they clear, and reads those keys' current rows from L3 by key (:func:`~threetears.core.keyed.read_keyed_arrow`), and
each prefix's rows by a key-led read of the prefix: a read the size of the changes, never of the table. A key L3 no
longer holds becomes a delete marker (carrying its state, decision 27); a file whose commit never landed simply reads
back as L3 has it. The rows are written as one recovery delta file (the prefixes cleared first, then the rows, every
column set, then the markers), folded in a worker process into every state it reaches, the other parts kept as they
are, every part current to the generation read before the keys; published at the writer's term, and the leftover
files deleted when the fold deletes (kept, the listing holds every file the table ever saved: decision 51). Logged at WARNING and counted (``delta_fold_recoveries_total``). The same for a round that finds a
commit of another incarnation than the table's full. A whole read stays only for no full at all (decision 14).

**A lost file of a commit too big to list is recovered from the notes, else by its states** (decision 31, with 21, 28
and 29). Such a commit's record lists no rows, so its keys were only in the file the copies could not read
(:meth:`DeltaFolds.recover_table`). Its recovery adds, beside the leftover files' keys: (a) the keys the change notes
name in the record's bracket (``noted_between``), read a page at a time while the ring still covers it (the notes
reader, keyed), and then only those keys' rows from L3 by key; else (b) every row of the states the record names
(``EpochChange.scopes``), read from L3 a keyset page at a time led by the state, which replace those states' parts
whole (a key L3 no longer holds simply is not there). The record's states are trusted only when it marks them complete
(:func:`~threetears.core.collections.epoch_change.complete_scopes`, decision 54 (c)): before the mark a record named
the states its writer touched, which nothing tied to the rows written, so one naming a strict subset left the full
with the old rows of a state it missed. A record of a scoped table without the mark is recovered by every state
instead: those the full holds and those L3 holds, read and replaced the same way (``every_state``). Counted
(``delta_fold_lost_file_recoveries_total``, why ``notes``, ``states``, ``every_state``, or ``unnamed`` when the record
names neither) and logged at WARNING. Never a copy: only the writer reads L3.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Collection, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal

from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge, histogram

from threetears.core.collections.delta_files import (
    _DELETED_COLUMN,
    _DELTA_FORMAT,
    _FORMAT_METADATA,
    _PREFIXES_METADATA,
    _WRITTEN_COLUMN,
    _WRITTEN_METADATA,
    _key_texts,
    _text_column,
    _under_texts,
    read_volume_file,
    DeltaFileUnreadableError,
    DeltaTable,
    DELTA_RETENTIONS,
    OWN_DELTA_SAVES,
    DeltaRetention,
    build_file_table,
    delta_file_version,
    versioned_write_id,
    fold_delta_tables,
    joined_key_texts,
    key_text_table,
    read_delta_file_keys,
    sweep_orphan_delta_files,
    write_table_file,
)
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE
from threetears.core.collections.delta_copy_beats import CopyPositionsReader
from threetears.core.collections.copy_follower import typed_text
from threetears.core.collections.delta_keep import DeltaKeepRule
from threetears.core.collections.epoch_change import RESET_REASON, EpochChange, EpochChangeReader, complete_scopes
from threetears.core.frozen_state import with_entry, without_entry
from threetears.core.collections.noted_keys import NotedKeysPage, NotedKeysReader, noted_key_pages
from threetears.core.collections.generation import GenerationSource, split_generation_token
from threetears.core.collections.schema_backed import DATETIMETZ_TYPE, TableSchema, rail_arrow_types, sql_cast
from threetears.core.declared import whole_read
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.keyed import KeyBatch, read_keyed_arrow
from threetears.core.keyset import keyset_arrow_pages, keyset_scope_pages, keyset_scope_values
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.backends.protocol import ArrowL3Reader
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.rows import DeclaredScope
from threetears.core.collections.volume_fulls import (
    FullManifest,
    FullPart,
    FullTable,
    StaleTermError,
    current_full_manifest,
    deletable_full_files,
    other_columns_parts,
    publish_full_manifest,
    rows_fingerprint,
    write_full_part,
)
from threetears.core.process_workers import ProcessWorkers

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    from threetears.core.arrow_keys import key_codes

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a fold refuses without it, saying so
    _PYARROW = False

__all__ = ["DeltaFolds", "FoldedTable", "RebuildConcurrency"]

log = get_logger(__name__)

#: a round's outcome: ``folded`` (a full published), ``nothing_due`` (no state reached half its part), ``no_full`` (no
#: full on the volume holds the table yet: the writer's rebuild publishes one), ``stale`` (a later writer's term is
#: current: this writer's lease was taken), ``failed`` (an error; the round's files are kept for the next), and
#: ``no_generation`` (a commit's file came with no generation to version it by: it is not folded)
_ROUNDS = counter(
    "delta_fold_rounds_total",
    description="writer fold rounds over tables that opt in to delta files, by outcome; failed rounds keep their files "
    "for the round the next commit starts",
    label_names=("outcome",),
)

#: states folded, by table and how: ``written`` (a new part file) or ``reused`` (a delta file named as the part)
_STATES = counter(
    "delta_fold_states_total",
    description="states folded into a new part of a full, by table and how (written: a new part file; reused: the "
    "delta file named as the part, no row copied)",
    label_names=("table", "how"),
)

#: rebuilds of a table's full from L3, by table, why (``cold_start``: no full holds it; ``reset``; ``schema_change``)
#: and outcome: ``rebuilt`` (published), ``stale`` (a later writer's term is current: nothing read or published),
#: ``no_generation`` (the table has none yet: rebuilt before its first commit's round), ``failed`` (retried before
#: the next round)
_REBUILDS = counter(
    "delta_fold_rebuilds_total",
    description="writer rebuilds of a table's full from L3 (its one whole read), by table, why and outcome",
    label_names=("table", "why", "outcome"),
)

#: covered delta files deleted once every copy was past them (decision 54), by table
_DELETED = counter(
    "delta_fold_deltas_deleted_total",
    description="covered delta files the fold deleted (retention delete), by table: each covered by the current and "
    "the previous full, and every live copy past it, its beats read right before",
    label_names=("table",),
)

#: covered delta files kept at a delete's look, by table and why: ``behind`` (a copy lags it: that file kept),
#: ``starting``, ``unreadable``, ``read_failed``, ``missing_table``, ``other_incarnation`` (every file kept that round)
_DELETES_KEPT = counter(
    "delta_fold_deltas_kept_total",
    description="covered delta files the fold kept at a delete's look (retention delete), by table and why: behind "
    "keeps the file; starting, unreadable, read_failed, missing_table, other_incarnation keep every file that round",
    label_names=("table", "why"),
)

#: delta files a restarted writer re-learned from the epoch log (:meth:`DeltaFolds.start`), by table
_RELEARNED = counter(
    "delta_fold_relearned_files_total",
    description="delta files a restarted writer re-learned from the NATS epoch log at its start, to fold, by table",
    label_names=("table",),
)

#: recoveries of a table's full from the keys its leftover delta files name, read from L3 by key (decision 28), by
#: table, why (``log_lost``: a record of the epoch log is missing; ``new_incarnation``: the log was replaced, its
#: generation of a new incarnation) and outcome: ``recovered`` (published), ``stale`` (a later writer's term is
#: current), ``no_generation`` (none to version it by yet: tried again before the next round), ``failed`` (tried again
#: before the next round)
_RECOVERIES = counter(
    "delta_fold_recoveries_total",
    description="writer recoveries of a table's full after its epoch log was lost: the keys its leftover delta files "
    "name read from L3 by key, folded as one recovery delta; by table, why and outcome",
    label_names=("table", "why", "outcome"),
)

#: lost files of commits too big to list recovered (decision 31), by table and why: ``notes`` (the record's notes
#: bracket still covered: its keys read from the notes, their rows from L3 by key), ``states`` (past the notes: the
#: states the record names read from L3 by state), ``every_state`` (past the notes, its record not marking its states
#: complete: every state the table holds read from L3 by state, decision 54 (c)), ``unnamed`` (the record names
#: neither, or is gone: only what the leftover files name is recovered)
_LOST_FILE_RECOVERIES = counter(
    "delta_fold_lost_file_recoveries_total",
    description="writer recoveries of a lost delta file of a commit too big to list its rows, by table and why: "
    "notes (keys from the change notes), states (the record's states read from L3), every_state (its states not "
    "marked complete: every state read from L3), unnamed (neither)",
    label_names=("table", "why"),
)

#: how a lost file's commit was recovered (decision 31)
_LostWhy = Literal["notes", "states", "every_state", "unnamed"]

#: rounds cut short (:meth:`DeltaFolds.close` past its bound, or :meth:`DeltaFolds.start` of a newer term finding
#: this process's round of an older one): ``stop`` or ``new_term``. A cut round's files go back to the queue (the next
#: writer re-learns them from the epoch log), its half-written rebuild file is deleted, and its worker terminated
_CUTS = counter(
    "delta_fold_rounds_cut_total",
    description="writer fold rounds cut short, by why: stop (close past its bound) or new_term (a newer holding's "
    "start found this process's round of an older term); their files are kept for the next round",
    label_names=("why",),
)

#: tables whose queued files were handed to a recovery by key after straight failed rounds (:data:`_FAILED_ROUNDS_CAP`)
_QUEUE_CAPPED = counter(
    "delta_fold_queue_capped_total",
    description="tables whose fold queue was handed to a recovery by key after straight failed rounds, so the queue "
    "never grows without bound; by table",
    label_names=("table",),
)

#: the commits queued for the next round, every table: grows only while rounds fail, until the cap hands them over
_QUEUED = gauge(
    "delta_fold_queued_files",
    description="commits' delta files queued for the writer's next fold round, every table together",
)

#: what the writer holds unfolded, by table, from the fold's own bookkeeping (gap 3): the delta files told and not yet
#: in a published full, and the bytes of their shares a round has measured (a file told counts in the files at once,
#: in the bytes once the round its commit starts has measured it). Zero while every delta is folded; a value that keeps
#: growing is a fold falling behind, or a state never due, before any copy's start grows slow
_UNFOLDED_FILES = gauge(
    "delta_fold_unfolded_files",
    description="delta files the writer's fold holds unfolded (told, not yet in a published full), by table; zero "
    "while every delta is folded",
    label_names=("table",),
)
_UNFOLDED_BYTES = gauge(
    "delta_fold_unfolded_bytes",
    description="bytes of the delta files the writer's fold holds unfolded, as its rounds measured each state's share, "
    "by table; zero while every delta is folded",
    label_names=("table",),
)

#: each fold round's seconds, the worker's whole round (note the commits' files, fold each state due, publish, delete),
#: as the writer waits for it, by outcome (:data:`_ROUNDS`, ``failed`` for one that raised): observed as the round
#: ends, a histogram observe on the loop and nothing more
_ROUND_SECONDS = histogram(
    "delta_fold_round_seconds",
    description="seconds of each writer fold round in its worker process, by outcome (folded, nothing_due, no_full, "
    "stale, no_generation, failed)",
    label_names=("outcome",),
)

#: the bytes the delta files' volume holds per table: its current full's parts plus the bytes of the delta files the
#: fold holds unfolded (:data:`_UNFOLDED_BYTES`). The full's bytes are the sizes the worker already had of the parts it
#: read, wrote or published (a round's, a rebuild's, a recovery's): never a walk or a stat of the directory on the loop.
#: Set once the writer's first round, rebuild or recovery has measured the table's full
_VOLUME_BYTES = gauge(
    "delta_files_volume_bytes",
    description="bytes on the delta files' volume per table: the current full's parts plus the delta files the "
    "writer's fold holds unfolded, from the sizes its worker measured (no directory walk)",
    label_names=("table",),
)

#: why a table starts fresh: it was reset, or recreated by a migration (:meth:`DeltaFolds.start_fresh`)
_FreshWhy = Literal["reset", "schema_change"]

#: why a table is recovered (decision 28): a record of its epoch log missing, or its log replaced (a new incarnation)
#: why a table is recovered by key: its log lost (a record missing, or the log replaced), or a commit too big to list
#: its rows lost its delta file (decision 29: its keys were only in the file)
_RecoveryWhy = Literal["log_lost", "new_incarnation", "file_lost", "rounds_failed"]

#: straight failed rounds past which the tables whose files they kept queued are recovered by key instead (decision
#: 28's recovery reads every file the full does not name, so the queue's entries of them are let go): what bounds the
#: queue when every round fails (a volume refusing writes), never a timer
_FAILED_ROUNDS_CAP: Final = 3

#: the write id a recovery delta's file is named by, after its term and before its unique part:
#: ``<table>/v<term>-recovery-<hex>.arrow`` (decision 53)
_RECOVERY_WRITE: Final = "recovery"

#: where a rebuild's rows are written as they are read, under the fulls' directory: one file a stream of a table's read,
#: removed once its parts are published (or the rebuild fails)
_REBUILD_DIR: Final = ".fulls/rebuild"


@dataclass(frozen=True, slots=True)
class RebuildConcurrency:
    """how much of L3 the writer's rebuild of missing fulls reads at once (:meth:`DeltaFolds._fresh_fulls`).

    Why: read one table after another, one statement at a time, a cold rebuild waited out every page's round trip
    through the L3 broker: 17 tables, ~20.1M rows at ~37k rows/s, 532 s before a pod restarted on an empty volume
    served (local stack, 2026-11-24; three LSF tables ~86% of the rows). Now up to ``tables`` tables are read at once,
    and a table of more than one page is split by its key's lead (a state) into reads of a set of lead values each;
    at most ``streams`` reads are in flight in all, each writing its pages to a file of its own as they come, so the
    rows held in memory are about ``streams`` pages, never a table.

    The rebuild's pages are bulk reads: each statement is one page under the broker's reply cap, sent with the
    backend's own deadline (``timeout_ms``) like any read, and the hub runs each in one of its read workers' connections
    (aibots hub ``read_worker.ReadWorkers``: 2 workers x 4 connections a hub by default, FIFO). The hub has no
    priority lane, so the rebuild is held under that room instead: ``streams`` 6 leaves at least 2 of a single hub's 8
    to serving reads at every moment (a quarter of the 24 of dev's and prod's 3 hubs), and a serving read waits at
    most one page behind a rebuild's.

    :param tables: the most tables read at once; at least one
    :ptype tables: int
    :param streams: the most reads (statements in flight, files written) at once across every table; at least one
    :ptype streams: int
    :raises ValueError: when either is under one
    """

    tables: int = 4
    streams: int = 6

    def __post_init__(self) -> None:
        if self.tables < 1 or self.streams < 1:
            raise ValueError(f"a rebuild reads at least one table and one stream at once, not {self}")


#: the rebuild's concurrency a fold is made with when its writer names none
_REBUILD_CONCURRENCY: Final = RebuildConcurrency()

#: one table and one statement at a time: a rebuild over an L3 that answers dict rows (:meth:`DeltaFolds._read_tables`)
_ONE_AT_A_TIME: Final = RebuildConcurrency(tables=1, streams=1)

#: a state folds once the bytes of its deltas since its last fold reach this share of its part's size (decision 10:
#: half), as a numerator over :data:`_DUE_OF`, so the test is whole numbers: ``bytes * _DUE_OF >= part * _DUE_AT``
_DUE_AT: Final = 1
_DUE_OF: Final = 2

#: the start of every file the fulls write themselves (``volume_fulls``): a part named outside it is a delta file
_FULLS_PREFIX: Final = ".fulls/"

#: one (delta file, state) share a round carries: table, scope, file, generation token, the share's bytes
_Share = tuple[str, str | None, str, str, int]


# --- what callers post to the fold's owner (decision 37) ---------------------------------------------------------------
#
# The fold's bookkeeping (the queue, the shares not folded, the tables owed a rebuild or a recovery, the asks, the lost
# files) has one owner: the round task, of which only one runs at a time. A caller (a commit told, the writer's start, a
# reset, a copy's lost file) only posts one of these, frozen as it hands it, and starts the round if none runs; the
# round applies every posted command in order before each step of its work (:meth:`DeltaFolds._drain`), and between
# two of its awaits nothing else changes what it reads. So a rebuild that read a table's generation judges, once it
# publishes, every ask that came after that read; and a restarted writer's walk of its log sees the queue as it was
# when the walk began, with a commit told meanwhile applied after it, never twice and never ahead of the older deltas.


@dataclass(frozen=True, slots=True)
class _Told:
    """a commit told (:meth:`DeltaFolds.committed`): its files and generations, copied when told."""

    files: Mapping[str, str]
    generations: Mapping[str, str]
    version: int
    in_flight_from: int | None


@dataclass(frozen=True, slots=True)
class _Started:
    """the writer's start (:meth:`DeltaFolds.start`): how L3 and the log are read, its term, and the answer it awaits:
    the tables published before the rounds it started end."""

    reader: _Reader
    changes: EpochChangeReader
    notes: NotedKeysReader | None
    version: int
    answer: asyncio.Future[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _Fresh:
    """a table reset or recreated by a migration (:meth:`DeltaFolds.start_fresh`)."""

    table: str
    why: _FreshWhy


@dataclass(frozen=True, slots=True)
class _LostFile:
    """a copy's lost file of a commit too big to list (:meth:`DeltaFolds.recover_table`, decision 29)."""

    table: str
    epoch: str
    file: str


@dataclass(frozen=True, slots=True)
class _Confirmed:
    """a failed load request's file the writer confirmed from L3 (:meth:`DeltaFolds.confirmed`, ruling C)."""

    table: str
    file: str
    epoch: str


@dataclass(frozen=True, slots=True)
class _Recheck:
    """a copy asked for a table's full the current one lacks (:meth:`DeltaFolds.full_owed`): the round looks for every
    table the full lacks before its next fold, and rebuilds them."""


_Command = _Told | _Started | _Fresh | _LostFile | _Confirmed | _Recheck


@dataclass(frozen=True, slots=True)
class FoldedTable:
    """one table the writer folds, as its schema declares it.

    :ivar name: the table
    :ivar key: its key columns, in key order
    :ivar columns: its columns, in the order its delta files and parts hold them
    :ivar scope: the scope each row belongs to, ``(column, prefix)`` as the writer's seqlock names it (a row's state is
        ``prefix + str(row[column])``); one part a state. The column is a key column, or the row column every delete
        marker of the table carries its state in (:attr:`delta_scope`), so each marker names its state. None: one part
        for the whole table
    :ivar delta_scope: the row column a delete marker carries its state in, for a table keyed without its state (AP
        results, keyed by the race, the state in ``join_state_code``;
        :attr:`~threetears.core.collections.schema_backed.TableSchema.delta_scope`, decision 27); None when the key
        names it
    :ivar keep: the rows a part keeps (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_keep`); None
        keeps every row
    :ivar kinds: each column's type tag, ``(column, column_type)`` in :attr:`columns`' order: how the writer's rebuild
        reads the table from L3 (:meth:`DeltaFolds.start`); empty for a table never rebuilt
    """

    name: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    scope: Annotated[tuple[str, str] | None, DeclaredScope] = None
    keep: type[DeltaKeepRule] | None = None
    kinds: tuple[tuple[str, str], ...] = ()
    delta_scope: str | None = None

    def __post_init__(self) -> None:
        missing = [column for column in self.key if column not in self.columns]
        if missing:
            raise ValueError(f"{self.name}: key columns {missing} are not among its columns")
        if self.scope is not None and self.scope[0] not in self.key and self.scope[0] != self.delta_scope:
            # a delete marker carries the key, and the state only in the table's delta_scope: a scope from any other
            # column leaves a delete's state unknown, so no one state's part could be folded past it
            raise ValueError(
                f"{self.name}: its scope column {self.scope[0]!r} must be a key column, so each delete marker names "
                f"its state, or the row column the table's markers carry the state in (TableSchema.delta_scope)"
            )

    @classmethod
    def of(cls, schema: TableSchema, *, scope: tuple[str, str] | None = None) -> FoldedTable:
        """a table as its schema declares it: its key, its columns as its delta files type them, its keep rule.

        :param schema: the table's schema; it must opt in to delta files
        :ptype schema: TableSchema
        :param scope: the scope each row belongs to, ``(column, prefix)``, as its copy follows it
            (:attr:`~threetears.core.collections.copy_follower.FollowedTable.scope`); None folds the table whole
        :ptype scope: tuple[str, str] | None
        :return: the table
        :rtype: FoldedTable
        :raises ValueError: when the table does not opt in to delta files
        """
        if not schema.delta_files:
            raise ValueError(f"{schema.name}: only a table that opts in to delta files (delta_files=True) is folded")
        return cls(
            name=schema.name,
            key=tuple(schema.pk_columns),
            columns=tuple(rail_arrow_types(schema.columns)),
            scope=scope,
            keep=schema.delta_keep,
            kinds=tuple((column.name, column.column_type) for column in schema.columns),
            delta_scope=schema.delta_scope,
        )

    @property
    def instant_keys(self) -> frozenset[str]:
        """the key columns of a ``timestamptz`` column (:func:`instant_key_columns`).

        :return: the columns
        :rtype: frozenset[str]
        """
        return instant_key_columns(self.key, self.kinds)

    def plain(self) -> tuple[Any, ...]:
        """the table as a worker process is handed it: plain data, and its keep rule's class by name.

        :return: ``(name, key, columns, scope column, scope prefix, keep, instant keys)``
        :rtype: tuple
        """
        column, prefix = self.scope if self.scope is not None else (None, None)
        return (self.name, self.key, self.columns, column, prefix, self.keep, tuple(sorted(self.instant_keys)))


def instant_key_columns(key: Sequence[str], kinds: Sequence[tuple[str, str]]) -> frozenset[str]:
    """the key columns of a ``timestamptz`` column (``DATETIMETZ_TYPE``), as
    :attr:`~threetears.core.collections.copy_follower.FollowedTable.instant_keys` names a copy's: each key text of one
    is its instant's one canonical text (``key_text(..., instant=True)``). Why: the rail and a delta file spell an
    instant ``...Z``, an epoch record and a change note ``....000000+00:00``; the fold and its recovery matched them
    as plain text, so a recovery of ``result_snapshots_by_id`` failed every round (2026-11-24 replay: 270 failures in
    20 minutes, pending files 8 -> 37 a round, fold rounds 197 -> 55 a 10 minutes).

    :param key: the key columns
    :ptype key: Sequence[str]
    :param kinds: each column's type tag, :attr:`FoldedTable.kinds`
    :ptype kinds: Sequence[tuple[str, str]]
    :return: the columns
    :rtype: frozenset[str]
    """
    return frozenset(name for name, kind in kinds if name in key and kind == DATETIMETZ_TYPE)


class DeltaFolds:
    """the writing load's fold of the tables that opt in to delta files: call :meth:`committed` after each commit.

    One a writer, made once and kept: it remembers each state's deltas since its last fold. Only the writer holding
    the write lease makes commits, so only it folds; no other lock. That memory has one owner, the round running
    (decision 37): :meth:`committed`, :meth:`start`, :meth:`start_fresh` and :meth:`recover_table` only post a command
    and start the round if none runs; the round applies the commands in order before each step (:meth:`_drain`).

    :param root: the delta files' root (``delta_files_root()``), the shared volume's mount
    :ptype root: Path
    :param tables: the tables the writer folds
    :ptype tables: Sequence[FoldedTable]
    :param retention: what becomes of a delta file once a full covers it (:data:`~threetears.core.collections.
        delta_files.DeltaRetention`), required: ``"keep"``, nothing deletes a delta file (decision 51), only the fold's
        own scratch goes; ``"delete"``, deleted as the module docstring says
    :ptype retention: DeltaRetention
    :param workers: the worker processes the rounds run in; None makes one worker of its own, shut by :meth:`close`
    :ptype workers: ProcessWorkers | None
    :param copies: the copies' positions (:class:`~threetears.core.collections.delta_copy_beats.CopyBeats`), required
        with ``retention="delete"``: a covered delta file is deleted only once every live copy is past it (decision 54)
    :ptype copies: CopyPositionsReader | None
    :param rebuild: how much of L3 a rebuild of missing fulls reads at once (:class:`RebuildConcurrency`)
    :ptype rebuild: RebuildConcurrency
    :raises ValueError: when ``retention`` is neither, or ``"delete"`` comes with no ``copies``
    """

    def __init__(
        self,
        root: Path,
        tables: Sequence[FoldedTable],
        *,
        retention: DeltaRetention,
        workers: ProcessWorkers | None = None,
        copies: CopyPositionsReader | None = None,
        rebuild: RebuildConcurrency = _REBUILD_CONCURRENCY,
    ) -> None:
        if not _PYARROW:
            raise ImportError("a fold needs pyarrow: install 3tears[snapshot]")
        if retention not in DELTA_RETENTIONS:
            raise ValueError(f"a fold's retention is one of {DELTA_RETENTIONS}, not {retention!r}")
        if retention == "delete" and copies is None:
            # decision 54: a fold that deletes must know where every copy is, or it deletes files a lagging copy needs
            raise ValueError(
                "a fold that deletes its delta files (retention='delete') needs the copies' positions: pass "
                "copies=CopyBeats(lease) (threetears.core.collections.delta_copy_beats) over the pod's lease factory, "
                "and have each copy register and beat (CopyBeats.register/beat); or run retention='keep'"
            )
        #: where every copy of the tables is (decision 54): read immediately before each delete; None when keeping
        self._copies = copies
        self.root = root
        #: how much of L3 a rebuild reads at once
        self.rebuild: Final = rebuild
        #: what becomes of a delta file once a full covers it: every round, rebuild and recovery is handed it
        self.retention: Final[DeltaRetention] = retention
        self.tables: Final = MappingProxyType({table.name: table for table in tables})
        self._plain_tables = tuple(table.plain() for table in tables)
        self._owns_workers = workers is None
        self._workers = workers if workers is not None else ProcessWorkers(workers=1, name="delta folds")
        #: (table, file, token) of each commit not yet handed to a round that finished, in commit order
        self._queued: tuple[tuple[str, str, str], ...] = ()
        #: the commits the round running now took off the queue: still unfolded, for the gauges
        self._taken: tuple[tuple[str, str, str], ...] = ()
        #: the oldest write version the writer had a commit in flight of, as the newest commit told said (None: none
        #: was): a full claims no file at or above it folded (decision 53), since such a file's commit may land later
        self._in_flight_from: int | None = None
        #: each (table, file) a restarted writer's walk left out as a failed load request's file not yet confirmed from
        #: L3: never folded until confirmed, so every full lists it open (decision 53), and a recovery reads it
        self._unconfirmed: tuple[tuple[str, str], ...] = ()
        self._term: int | None = None
        #: each (delta file, state) share not folded yet, the deltas named as parts, the covered deltas not deleted
        self._pending: tuple[_Share, ...] = ()
        self._reused: tuple[str, ...] = ()
        self._doomed: tuple[str, ...] = ()
        #: each doomed file's epoch, where the fold heard it (a commit told): what every copy must be past before it goes
        #: (decision 54). A read-only snapshot replaced whole, of the doomed files alone
        self._doomed_epochs: Mapping[str, str] = MappingProxyType({})
        self._round: asyncio.Task[None] | None = None
        #: fold rounds failed in a row: past :data:`_FAILED_ROUNDS_CAP` their tables are recovered by key
        self._failed_rounds = 0
        #: the term the running round folds at: a start of a newer term never waits on it (:meth:`start`)
        self._round_term: int | None = None
        self._unfolded_warned: frozenset[str] = frozenset()
        #: how the writer reads L3 for a rebuild, from :meth:`start`: None until the writer's start calls it
        self._reader: _Reader | None = None
        #: tables reset or recreated (:meth:`start_fresh`), by why: rebuilt before any round folds them
        self._fresh: Mapping[str, _FreshWhy] = MappingProxyType({})
        #: tables whose epoch log was lost, by why: recovered by key (decision 28) before any round folds them
        self._recover: Mapping[str, _RecoveryWhy] = MappingProxyType({})
        #: each table's delta files lost (missing or unreadable) that decision 29 handed to a recovery, onto the epoch
        #: whose record named each: never read for the recovery's keys, deleted once it is published; the record's
        #: notes bracket or states name the keys instead (decision 31)
        self._lost: Mapping[str, Mapping[str, str]] = MappingProxyType({})
        #: the epoch log and the change notes, from :meth:`start`: what a lost file's record and keys are read from
        self._changes: EpochChangeReader | None = None
        self._notes: NotedKeysReader | None = None
        #: each table's recovery asks (:meth:`recover_table`): one asked while a recovery of the table runs is owed
        #: again once it ends, since the generation that recovery read may be older than the lost file's epoch
        self._asked: Mapping[str, int] = MappingProxyType({})
        #: each table's current full's bytes, as the last round, rebuild or recovery measured its parts: a read-only
        #: snapshot replaced whole, the full's half of :data:`_VOLUME_BYTES`
        self._full_bytes: Mapping[str, int] = MappingProxyType({})
        #: whether a rebuild may be owed: the next round looks before it folds
        self._recheck = False
        #: whether a table was marked fresh since the last rebuild began
        self._marked = False
        #: how many rebuilds and recoveries published: a count, never a list growing with each
        self.published = 0
        #: the tables a rebuild or a recovery published during the :meth:`start` running, in order; None outside one
        self._starting: tuple[str, ...] | None = None
        #: what a rebuild from L3 is reading now, ``rebuilding from L3: n of m tables read`` or ``publishing ...``, for a copy's
        #: ask (:attr:`rebuilding`); None while none runs. A str replaced whole by the round, its one writer
        self._rebuilding: str | None = None
        #: the commands callers posted that the round has not applied yet, in order: the one thing a caller adds to
        self._inbox: tuple[_Command, ...] = ()
        #: the starts applied whose walk of the epoch log the round runs next (:meth:`_relearn`)
        self._relearning: tuple[_Started, ...] = ()
        #: the starts applied whose answer is the tables published once the rounds end
        self._answers: tuple[asyncio.Future[tuple[str, ...]], ...] = ()

    @property
    def unfolded(self) -> frozenset[str]:
        """the delta files told (or re-learned) and not folded yet, for a status line and the tests.

        :return: the files, under the root
        :rtype: frozenset[str]
        """
        posted = frozenset(
            file
            for command in self._inbox
            if isinstance(command, _Told)
            for table, file in command.files.items()
            if table in self.tables
        )
        return self._told() | posted

    def _told(self) -> frozenset[str]:
        """the delta files the owner holds unfolded: queued, in the round running, or in shares not folded yet.

        :return: the files, under the root
        :rtype: frozenset[str]
        """
        return frozenset(file for _, file, _ in (*self._queued, *self._taken)) | frozenset(
            share[2] for share in self._pending
        )

    def _folded_below(self) -> int | None:
        """the write version below which a full this fold publishes now claims every delta file folded, but those it
        lists open (decision 53): the oldest write the writer still had open (begun, not committed) or a commit in
        flight of, as the newest commit told said; past the writer's term (the newest write told) when none was. Every
        commit of an older write has ended: told (so held unfolded, and listed open, until folded), or never landed (an
        orphan, no change to hold). A write begun since the newest told is past the term: never claimed. A writer's
        start holds its own write open until a commit told says otherwise. None before a start or a commit: nothing is
        claimed.

        :return: the version, or None
        :rtype: int | None
        """
        if self._term is None:
            return None
        return self._term + 1 if self._in_flight_from is None else min(self._term + 1, self._in_flight_from)

    def _open_files(self, table: str, *, covered: frozenset[str] = frozenset()) -> tuple[str, ...]:
        """the delta files of ``table`` a full published now does not hold (decision 53's ``open``): queued, in the
        round running, in shares not folded yet, waiting for the writer to confirm them, or saved by this process for a
        commit not yet ended; but ``covered``, which the publish holds (a rebuild's or a recovery's, told before its
        generation was read).

        :param table: the table
        :ptype table: str
        :param covered: the files the publish holds
        :ptype covered: frozenset[str]
        :return: the files, sorted
        :rtype: tuple[str, ...]
        """
        held = {file for name, file, _ in (*self._queued, *self._taken) if name == table}
        held |= {share[2] for share in self._pending if share[0] == table}
        held |= self._outside_round(table)
        return tuple(sorted(held - covered))

    def _outside_round(self, table: str) -> set[str]:
        """the delta files of ``table`` a full does not hold that no round carries: waiting for the writer to confirm
        them, or saved by this process for a commit not yet ended.

        :param table: the table
        :ptype table: str
        :return: the files
        :rtype: set[str]
        """
        held = {file for name, file in self._unconfirmed if name == table}
        held |= {file for file in OWN_DELTA_SAVES.in_flight if file.startswith(f"{table}/")}
        return held

    def _measure(self) -> None:
        """set the unfolded gauges of every folded table from the fold's bookkeeping: the files queued, in the round
        running and held in shares not folded yet, and the bytes of those shares. Called wherever that bookkeeping
        changes; a dict walk of what the fold already holds, no file read.

        :return: nothing
        :rtype: None
        """
        files: dict[str, set[str]] = {name: set() for name in self.tables}
        sizes: dict[str, int] = dict.fromkeys(self.tables, 0)
        for table, file, _ in (*self._queued, *self._taken):
            files.setdefault(table, set()).add(file)
        for table, _, file, _, size in self._pending:
            files.setdefault(table, set()).add(file)
            sizes[table] = sizes.get(table, 0) + size
        for table in self.tables:
            _UNFOLDED_FILES.labels(table=table).set(len(files[table]))
            _UNFOLDED_BYTES.labels(table=table).set(sizes[table])
            full = self._full_bytes.get(table)
            if full is not None:
                _VOLUME_BYTES.labels(table=table).set(full + sizes[table])
        _QUEUED.set(len(self._queued))

    def _measured_fulls(self, full_bytes: Sequence[tuple[str, int]]) -> None:
        """keep the full's bytes per table a worker answered, and set the volume gauge from them.

        :param full_bytes: each table's ``(table, bytes of its current full's parts)``, from the worker
        :ptype full_bytes: Sequence[tuple[str, int]]
        :return: nothing
        :rtype: None
        """
        if full_bytes:
            self._full_bytes = MappingProxyType({**self._full_bytes, **dict(full_bytes)})
        self._measure()

    def committed(
        self,
        files: Mapping[str, str],
        generations: Mapping[str, str],
        *,
        version: int,
        in_flight_from: int | None,
    ) -> None:
        """note a commit's delta files and start a round of folds in the background; never waits, never raises for
        a fold. Call it after the commit, on the event loop.

        :param files: table -> its delta file, as the commit's transaction names them
            (:attr:`~threetears.core.collections.caller_transaction.CallerTransaction.delta_files`)
        :ptype files: Mapping[str, str]
        :param generations: table -> the generation token the commit's advance wrote: the version a part folded past
            the file is current to
        :ptype generations: Mapping[str, str]
        :param version: the write's version (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`),
            taken under the write lease: the writer's term (module docstring)
        :ptype version: int
        :param in_flight_from: the oldest write version the writer may still save a delta file for: a write still open
            (begun with ``ScopeEpochs.begin``, not yet committed; this commit's own while its write goes on) or a
            commit in flight; None when no write is open. A full the fold publishes claims every delta file below it
            (or past ``version`` when None) folded but those it lists open (decision 53), so a file a write may still
            save is never claimed: a recovery reads it, a fold that deletes never takes it. A version, never a time: no clock is compared (the volume's is not the writer's).
            Required, no default: a writer that left it out would have a file of a commit still in flight claimed
            covered, so a consumer must say it at the call
        :ptype in_flight_from: int | None
        :return: nothing
        :rtype: None
        """
        self._post(
            _Told(
                files=MappingProxyType(dict(files)),
                generations=MappingProxyType(dict(generations)),
                version=version,
                in_flight_from=in_flight_from,
            )
        )

    def _apply_told(self, told: _Told) -> None:
        """the owner's part of :meth:`committed`: queue the commit's files, each once (a restarted writer's walk of the
        log may have queued it already, in its place among the older ones), and take its term.

        :param told: the commit
        :ptype told: _Told
        :return: nothing
        :rtype: None
        """
        held = self._told()
        for table, file in told.files.items():
            if table not in self.tables:
                if table not in self._unfolded_warned:
                    self._unfolded_warned = self._unfolded_warned | {table}
                    log.warning(
                        "a commit named a delta file of a table the writer does not fold; its files are never folded "
                        "or deleted",
                        extra={"extra_data": {"table": table, "file": file}},
                    )
                continue
            token = told.generations.get(table)
            if token is None or split_generation_token(token) is None:
                _ROUNDS.labels(outcome="no_generation").inc()
                log.error(
                    "a commit's delta file came with no generation to version a part by; it is not folded",
                    extra={"extra_data": {"table": table, "file": file, "generation": token}},
                )
                continue
            if file in held:
                continue  # re-learned from the epoch log first, in commit order
            self._queued = (*self._queued, (table, file, token))
        self._term = told.version if self._term is None else max(self._term, told.version)
        self._in_flight_from = told.in_flight_from

    async def start(
        self,
        l3: Any,
        generations: GenerationSource,
        *,
        changes: EpochChangeReader,
        version: int,
        page_size: int = DEFAULT_PAGE_SIZE,
        notes: NotedKeysReader | None = None,
    ) -> tuple[str, ...]:
        """the writer's start: re-learn from the epoch log the delta files not yet folded (module docstring, "A
        restarted writer re-learns"), then rebuild from L3, once, each folded table the current full on the volume
        lacks (no full at all: a lost volume or a new environment; or one without the table), and recover by key each
        whose log was lost (module docstring, "A lost log"), publishing them at ``version``; then fold each state
        already due.
        Call it after taking the write lease and beginning the first write, before that write's commit is told to
        :meth:`committed`. Kept for the rebuilds a later round owes (:meth:`start_fresh`, a table with no generation
        yet).

        :param l3: the writer's L3, read whole once for each table rebuilt (the rail's proxy, or a pool)
        :ptype l3: Any
        :param generations: the writer's generation source: each table's generation is read before its rows, the
            version its parts are current to
        :ptype generations: GenerationSource
        :param changes: the NATS epoch log, as a copy reads it: each table's records since its full's parts name the
            delta files not folded yet (decision 24); never L3, never a listing of the volume
        :ptype changes: EpochChangeReader
        :param version: the write's version (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`),
            taken under the write lease: the writer's term
        :ptype version: int
        :param page_size: rows a page of the read holds
        :ptype page_size: int
        :param notes: the change notes' reader, which names a lost file's keys while its bracket is covered (decision
            31); None takes ``l3`` when it reads notes (the rail's proxy: the broker's ``l3.noted_keys``)
        :ptype notes: NotedKeysReader | None
        :return: the tables rebuilt or recovered and published, in order; empty when the full held every one and every
            log was whole (or this writer's term is no longer the highest, or a read failed: logged and counted,
            retried before the next round)
        :rtype: tuple[str, ...]
        :raises ValueError: when ``page_size`` is under one, or a table names no column types (``FoldedTable.of``)
        :raises GenerationUnavailableError: when the epoch log cannot be read: a fold without the deltas it would
            not know of would publish parts lacking their rows, so the writer's start fails and is tried again
        """
        if page_size < 1:
            raise ValueError("page_size must be at least one")
        for table in self.tables.values():
            _table_types(table)  # refuses a table it could not read
        running = self._round is not None and not self._round.done()
        if running and self._round_term is not None and self._round_term < version:
            # this process's own round of an older holding: a newer writer wrote since, so it is stale (its publish
            # is refused or never read); a session's start never waits on it. Its files go back to the queue
            await self._cut("new_term")
        answer: asyncio.Future[tuple[str, ...]] = asyncio.get_running_loop().create_future()
        self._post(
            _Started(
                reader=_Reader(l3=l3, generations=generations, page_size=page_size),
                changes=changes,
                notes=notes if notes is not None else l3 if isinstance(l3, NotedKeysReader) else None,
                version=version,
                answer=answer,
            )
        )
        return await answer

    def _apply_started(self, started: _Started) -> None:
        """the owner's part of :meth:`start`: how L3 and the log are read, the term, a rebuild looked for, and the walk
        of the log owed before the round's next fold.

        :param started: the start
        :ptype started: _Started
        :return: nothing
        :rtype: None
        """
        self._reader = started.reader
        self._changes = started.changes
        self._notes = started.notes
        self._term = started.version if self._term is None else max(self._term, started.version)
        # the start is called within the write it began: that write is open until a commit told says otherwise
        self._in_flight_from = (
            started.version if self._in_flight_from is None else min(self._in_flight_from, started.version)
        )
        self._recheck = True
        if self._starting is None:
            self._starting = ()
        self._relearning = (*self._relearning, started)
        self._answers = (*self._answers, started.answer)

    async def _relearn(self, generations: GenerationSource, changes: EpochChangeReader) -> None:
        """queue, as commits told, each delta file the epoch log names since each table's oldest part (module
        docstring, "A restarted writer re-learns"); a table whose log was lost is owed a recovery instead.

        :param generations: the writer's generation source: how far each table's log reaches
        :ptype generations: GenerationSource
        :param changes: the epoch log
        :ptype changes: EpochChangeReader
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        current = await asyncio.to_thread(current_full_manifest, self.root)
        if current is None or self._term is None or current.term > self._term:
            return  # no full to continue (the rebuild publishes one), or a later writer's
        # the owner's own queue, which nothing else changes across the walk's awaits: a commit told meanwhile waits
        # in the inbox and is applied after the walk, once (:meth:`_apply_told`)
        told = self._told()
        for name in sorted(self.tables):
            entry = current.tables.get(name)
            if entry is None or name in self._fresh or name in self._recover:
                continue  # rebuilt or recovered before any round: its parts are then current to L3, no delta owed
            other = other_columns_parts(entry, self.tables[name].kinds)
            if other:
                # a newer table list changed the table's columns in place (the hub adds a column with no recreate and
                # no reset record): its parts hold the old columns, so it starts fresh, never folded onto them
                self._owe_fresh(
                    name,
                    "schema_change",
                    detail=f"{len(other)} parts over other columns than the schema ({other[0].fingerprint})",
                )
                continue
            try:
                now = await generations.current(name)
            except (
                GenerationUnavailableError
            ):  # NOSILENT: never written since its store was made; the round of its first commit looks
                continue
            held, at = split_generation_token(entry.version), split_generation_token(now)
            if held is None or at is None:
                continue
            if held[0] != at[0]:
                self._owe_recovery(
                    name, "new_incarnation", detail=f"the full is of incarnation {held[0]}, the log of {at[0]}"
                )
                continue
            low = min(
                (
                    parsed[1]
                    for parsed in (split_generation_token(part.version) for part in entry.parts)
                    if parsed is not None and parsed[0] == held[0]
                ),
                default=held[1],
            )
            if at[1] <= low:
                continue
            walk = await changes.changes_after(name, f"{held[0]}:{low}", now)
            resets = [change.token for change in walk.changes if change.whole_table and change.reason == RESET_REASON]
            if resets:
                # a reset's record names no delta file by design: the table starts fresh (decision 12), its rows read
                # from L3 once, never recovered by key as a lost log
                self._owe_fresh(name, "reset", detail=f"reset records {resets[:5]} since {held[0]}:{low}")
                continue
            unnamed = [change.token for change in walk.changes if change.delta_file is None]
            if not walk.complete or unnamed:
                self._owe_recovery(
                    name, "log_lost", detail=f"first missing {walk.first_missing}, records naming no file {unnamed[:5]}"
                )
                continue
            files: dict[str, str] = {}
            newest_unconfirmed: dict[str, bool] = {}
            for change in walk.changes:
                assert change.delta_file is not None  # noqa: S101 -- refused above
                files[change.delta_file] = change.token  # a commit of several records: its newest
                newest_unconfirmed[change.delta_file] = change.delta_file_unconfirmed
            # a failed load request's file, named unconfirmed, is folded only once the writer has confirmed it from L3
            # (ruling C): one not confirmed yet is left out here, and told to the fold when a copy's ask confirms it.
            # Judged by the file's newest record: a load's intent record that the hub's confirmed record superseded is
            # folded at that record, as any whole load's
            unconfirmed = [file for file, still in newest_unconfirmed.items() if still]
            if unconfirmed:
                waiting = await asyncio.to_thread(_not_confirmed, self.root, name, unconfirmed)
                if waiting:
                    log.warning(
                        "a restarted writer left out failed load requests' files not yet confirmed from L3: each is "
                        "folded once a copy's ask confirms it",
                        extra={"extra_data": {"table": name, "files": waiting[:20], "count": len(waiting)}},
                    )
                files = {file: token for file, token in files.items() if file not in waiting}
                # listed open by every full until confirmed (decision 53): never folded, so never claimed covered
                held_waiting = set(self._unconfirmed)
                self._unconfirmed = (
                    *self._unconfirmed,
                    *((name, file) for file in waiting if (name, file) not in held_waiting),
                )
            present = await asyncio.to_thread(_present, self.root, list(files))
            # a file gone was covered by a published full and deleted by a fold that deletes (nothing else deletes a
            # committed delta)
            learned = [(name, file, files[file]) for file in present if file not in told]
            self._queued = (*self._queued, *learned)
            self._measure()
            _RELEARNED.labels(table=name).inc(len(learned))
            log.info(
                "a restarted writer re-learned its unfolded delta files from the epoch log",
                extra={
                    "extra_data": {
                        "table": name,
                        "since": f"{held[0]}:{low}",
                        "upto": now,
                        "records": len(walk.changes),
                        "files": len(learned),
                        "gone": len(files) - len(present),
                    }
                },
            )

    def start_fresh(self, table: str, *, why: _FreshWhy) -> None:
        """start ``table`` fresh: it was reset (``SchemaBackedCollection.reset``) or dropped and recreated by a
        migration, so its delta files and parts no longer hold it. Rebuilt from L3 under the writer's term before any
        round folds it again, its old files dropped (decision 12). Call it once the reset or the migration has
        committed; never waits.

        :param table: the table
        :ptype table: str
        :param why: ``"reset"`` or ``"schema_change"``
        :ptype why: str
        :return: nothing
        :rtype: None
        :raises KeyError: when the writer does not fold the table
        :raises ValueError: for another why
        """
        if table not in self.tables:
            raise KeyError(table)
        if why not in ("reset", "schema_change"):
            raise ValueError(f"a table starts fresh after a reset or a schema change, not {why!r}")
        self._post(_Fresh(table=table, why=why))

    def _owe_fresh(self, table: str, why: _FreshWhy, *, detail: str = "") -> None:
        """mark ``table`` owed a rebuild from L3 before any round folds it again, its old files dropped.

        :param table: the table
        :ptype table: str
        :param why: why it starts fresh
        :ptype why: str
        :param detail: what was found, for the log line
        :ptype detail: str
        :return: nothing
        :rtype: None
        """
        log.warning(
            "a folded table starts fresh: it is rebuilt from L3 and its old files dropped",
            extra={"extra_data": {"table": table, "why": why, "detail": detail}},
        )
        self._fresh = with_entry(self._fresh, table, why)
        self._recheck = True
        self._marked = True

    def _owe_recovery(self, table: str, why: _RecoveryWhy, *, detail: str) -> None:
        """mark ``table`` owed a recovery by key (decision 28) before any round folds it again: its epoch log was lost,
        so the order of its delta files left on the volume, and whether each one's commit landed, is unknown.

        :param table: the table
        :ptype table: str
        :param why: ``"log_lost"`` (a record missing) or ``"new_incarnation"`` (the log replaced)
        :ptype why: str
        :param detail: what was found, for the log line
        :ptype detail: str
        :return: nothing
        :rtype: None
        """
        log.warning(
            "a folded table's epoch log was lost: its full is kept and recovered from the keys its leftover delta "
            "files name, read from L3 by key",
            extra={"extra_data": {"table": table, "why": why, "detail": detail}},
        )
        self._recover = with_entry(self._recover, table, why)
        self._recheck = True
        self._marked = True

    async def recover_table(self, table: str, *, epoch: str, file: str) -> None:
        """decision 29's hand-off (:class:`~threetears.core.collections.delta_recreate.LostDeltaRecovery`): a commit
        too big to list its rows lost its delta file, so its keys were only in the file; the table is recovered by key
        (decision 28) and a full published whose every part is current to a generation read after the request, so at
        or past ``epoch``, which a copy waiting for the file reloads from. Coalesced: asks for one table while its
        recovery is owed join it, and one asked while it runs is owed again after it. Returns once owed, never waits
        for the publish. Only the writer holding the write lease calls it (the recreator is served under it).

        The lost file is never read for the recovery's keys (it is gone, or its bytes are damaged), and is deleted once
        the recovery is published. What it alone named is recovered from its epoch's record (decision 31, module
        docstring): the keys the change notes name in its bracket while the ring covers it, else every row of the
        states it names, read from L3 by state.

        :param table: the table
        :ptype table: str
        :param epoch: the epoch whose file was lost
        :ptype epoch: str
        :param file: the lost file, under the root
        :ptype file: str
        :return: nothing
        :rtype: None
        :raises KeyError: when the writer does not fold the table
        """
        if table not in self.tables:
            raise KeyError(table)
        self._post(_LostFile(table=table, epoch=epoch, file=file))

    def _apply_lost(self, lost: _LostFile) -> None:
        """the owner's part of :meth:`recover_table`: the lost file noted, the ask counted, the recovery owed.

        :param lost: the ask
        :ptype lost: _LostFile
        :return: nothing
        :rtype: None
        """
        table, epoch, file = lost.table, lost.epoch, lost.file
        self._lost = with_entry(self._lost, table, with_entry(self._lost.get(table, MappingProxyType({})), file, epoch))
        self._asked = with_entry(self._asked, table, self._asked.get(table, 0) + 1)
        if self._recover.get(table) is not None:
            # owed already (perhaps failed on this very file): tried again now, without reading it
            self._recheck = True
            self._marked = True
            return
        self._owe_recovery(
            table,
            "file_lost",
            detail=f"{file} of epoch {epoch}: a commit too big to list lost its file; what only it named is recovered "
            "from its record's notes, else its states",
        )

    @property
    def rebuilding(self) -> str | None:
        """what a rebuild of a missing full is doing now, for a copy waiting for it: ``rebuilding from L3: n of m
        tables read`` while it reads the tables from L3, ``publishing the rebuilt full (m tables)`` while its worker
        writes the parts; None while no rebuild runs.

        Why: a cold rebuild of ~20M rows read for 532 s with no line between its start and its publish (local stack,
        2026-11-24), and a copy waiting on it could not tell slow from stalled.

        :return: the step, or None
        :rtype: str | None
        """
        return self._rebuilding

    @property
    def started(self) -> bool:
        """whether a writer's :meth:`start` has run or been posted: only then does the fold know how to read L3 and at
        what term to publish, so only then can it rebuild a full on its own.

        :return: True once started
        :rtype: bool
        """
        return self._reader is not None or any(isinstance(command, _Started) for command in self._inbox)

    async def lacking(self) -> tuple[str, ...]:
        """the folded tables the current full on the volume does not hold (none published: every one), or holds over
        other columns than their schema's (decision 12's schema change, found here as :meth:`_relearn` finds it): the
        ones a rebuild from L3 is owed for. One read of the current manifest, on a thread; no part is opened.

        :return: the tables, sorted; empty when the full holds every one
        :rtype: tuple[str, ...]
        :raises ValueError: when the current manifest does not read back
        """
        current = await asyncio.to_thread(current_full_manifest, self.root)
        return _lacking(current, self.tables)

    async def full_owed(self, table: str) -> bool:
        """a copy's ask for ``table``'s full (:class:`~threetears.core.collections.delta_recreate._MissingFulls`): when
        the current full lacks it, a fold already started looks again before its next round, so its rebuild (the cold
        start, decision 14) publishes the full with no commit told; never waits for that. A fold not started cannot
        (it knows neither L3 nor its term): said, and left to the writer's start, which rebuilds every table lacking.

        :param table: the table
        :ptype table: str
        :return: whether a full of the table is owed (False: the current full holds it)
        :rtype: bool
        :raises KeyError: when the writer does not fold the table
        :raises ValueError: when the current manifest does not read back
        """
        if table not in self.tables:
            raise KeyError(table)
        lacking = await self.lacking()
        if table not in lacking:
            return False
        if self.started:
            self._post(_Recheck())
        else:
            log.warning(
                "a copy asked for a table's full before the writer's fold started: the writer's start rebuilds it",
                extra={"extra_data": {"table": table, "lacking": list(lacking)}},
            )
        return True

    def confirmed(self, table: str, *, file: str, epoch: str) -> None:
        """a failed load request's file, its record naming it unconfirmed, written again by the writer from L3 and
        marked confirmed (``delta_recreate.DeltaRecreator`` tells it, ruling C of
        2026-11-23): the rows of the request that committed are folded as any commit's, at its record's epoch. Never
        waits, never raises for a fold; a table the writer does not fold is skipped, said.

        Why: the load hands its caller only the requests that landed whole (:meth:`committed`), so the fold never heard
        the rows a failed request committed; they would be missing from every full published after.

        :param table: the table
        :ptype table: str
        :param file: the confirmed file, under the root
        :ptype file: str
        :param epoch: the epoch whose record names it
        :ptype epoch: str
        :return: nothing
        :rtype: None
        """
        self._post(_Confirmed(table=table, file=file, epoch=epoch))

    def _apply_confirmed(self, confirmed: _Confirmed) -> None:
        """the owner's part of :meth:`confirmed`: the file queued once, at its record's epoch.

        :param confirmed: the confirmed file
        :ptype confirmed: _Confirmed
        :return: nothing
        :rtype: None
        """
        table, file, token = confirmed.table, confirmed.file, confirmed.epoch
        # confirmed from L3: queued below (or learned by the start), so no longer waiting (decision 53's open)
        self._unconfirmed = tuple(item for item in self._unconfirmed if item != (table, file))
        if table not in self.tables:
            log.warning(
                "a confirmed load request's file is of a table the writer does not fold; it is never folded",
                extra={"extra_data": {"table": table, "file": file}},
            )
            return
        if split_generation_token(token) is None:
            _ROUNDS.labels(outcome="no_generation").inc()
            log.error(
                "a confirmed load request's file came with no generation to version a part by; it is not folded",
                extra={"extra_data": {"table": table, "file": file, "generation": token}},
            )
            return
        if file in self._told():
            return  # re-learned from the epoch log first (a restarted writer's walk), or told twice
        if self._term is None:
            # no write of this holding told yet: the writer's start walks the log and learns the file, confirmed on the
            # volume by then (decision 24); a round never runs without a term
            log.warning(
                "a confirmed load request's file was told before the writer's first write: its start learns it",
                extra={"extra_data": {"table": table, "file": file}},
            )
            return
        # a confirmation is no new write: the writer's term and its oldest open transaction are left as they are
        self._queued = (*self._queued, (table, file, token))

    def _post(self, command: _Command) -> None:
        """post a command to the owner and start the round if none runs: all a caller does (decision 37).

        :param command: the command
        :ptype command: _Command
        :return: nothing
        :rtype: None
        """
        self._inbox = (*self._inbox, command)
        self._run_rounds()

    def _drain(self) -> None:
        """the owner applies every command posted since it last looked, in order. Never awaits, so nothing is posted
        between two of its lines; the round calls it before each step of its work and before it ends.

        :return: nothing
        :rtype: None
        """
        commands, self._inbox = self._inbox, ()
        for command in commands:
            if isinstance(command, _Told):
                self._apply_told(command)
            elif isinstance(command, _Started):
                self._apply_started(command)
            elif isinstance(command, _Confirmed):
                self._apply_confirmed(command)
            elif isinstance(command, _Fresh):
                # an ask of the table: a rebuild that read its generation before it is owed again (:meth:`_fresh_fulls`)
                self._asked = with_entry(self._asked, command.table, self._asked.get(command.table, 0) + 1)
                self._owe_fresh(command.table, command.why)
            elif isinstance(command, _Recheck):
                # _fresh_fulls reads the manifest and rebuilds each table it lacks: no table named, none to forget
                self._recheck = True
            else:
                self._apply_lost(command)
        if commands:
            self._measure()

    def _run_rounds(self) -> None:
        """start the background rounds unless they are running.

        :return: nothing
        :rtype: None
        """
        if self._round is None or self._round.done():
            self._round = asyncio.get_running_loop().create_task(self._rounds(), name="delta folds")

    def _kept_files(self, *, covered: frozenset[str]) -> tuple[str, ...]:
        """the delta files a rebuild never drops, whatever their time on the volume (whose clock is not the
        writer's): each this process saved whose commit has not ended, and each commit's file queued or held unfolded
        but those the rebuild covers (told before its generation was read), which a commit newer than the rebuild's
        generation is folded from after it.

        :param covered: the files told before the rebuild's generations were read
        :ptype covered: frozenset[str]
        :return: their names under the root
        :rtype: tuple[str, ...]
        """
        told = {file for _, file, _ in self._queued} | {share[2] for share in self._pending}
        return tuple(sorted(OWN_DELTA_SAVES.in_flight | (told - covered)))

    def _published_one(self, table: str) -> None:
        """count a rebuild's or a recovery's publish, and name it to the :meth:`start` running.

        :param table: the table published
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        self.published += 1
        if self._starting is not None:
            self._starting = (*self._starting, table)

    async def wait(self) -> None:
        """wait for the round in progress, and any it starts, to end.

        :return: nothing
        :rtype: None
        """
        while self._round is not None and not self._round.done():
            await asyncio.shield(self._round)

    async def close(self, *, within: float | None = None) -> None:
        """end the fold: wait for the round in progress, then shut the worker this fold made.

        ``within`` bounds the wait (a serving process stopping inside its grace period): past it the round is cut
        (:meth:`_cut`): its files go back to the queue, so nothing is lost (the next writer re-learns them from the
        epoch log), its half-written rebuild file is deleted, and the worker running it is terminated, never waited
        for.

        :param within: the most seconds to wait for the round; None waits for it to end
        :ptype within: float | None
        :return: nothing
        :rtype: None
        """
        cut = False
        if within is None:
            await self.wait()
        else:
            try:
                await asyncio.wait_for(self.wait(), timeout=max(within, 0.0))
            except TimeoutError:
                cut = await self._cut("stop", within=within)
        if self._owns_workers:
            await self._workers.close(wait=not cut)

    async def _cut(self, why: Literal["stop", "new_term"], *, within: float | None = None) -> bool:
        """cut the round running short: the task cancelled (its own handling puts its files back in the queue and
        deletes a rebuild's half-written rows), and the worker this fold made terminated, never waited for (a worker
        handed in is its owner's: its job runs on). Counted and logged.

        :param why: ``stop`` (:meth:`close` past its bound) or ``new_term`` (:meth:`start` of a newer holding)
        :ptype why: str
        :param within: the bound that passed, for the log line
        :ptype within: float | None
        :return: whether a round was cut
        :rtype: bool
        """
        running = self._round
        if running is None or running.done():
            return False
        running.cancel()
        if self._owns_workers:
            await self._workers.close(wait=False)
        # its cancel handling: the queue restored, a rebuild's rows file unlinked (a thread's one unlink each)
        await asyncio.wait({running})
        _CUTS.labels(why=why).inc()
        log.warning(
            "a delta fold round was cut short: its files are kept for the next round, its worker terminated",
            extra={
                "extra_data": {
                    "why": why,
                    "within": within,
                    "term": self._round_term,
                    "queued": len(self._queued),
                    "worker_terminated": self._owns_workers,
                }
            },
        )
        return True

    async def _rounds(self) -> None:
        """the owner's turn (decision 37): run rounds while there is work, then answer each start applied. Cancelled
        (:meth:`_cut`), it applies what was posted and cancels the starts' answers, as their wait was cut; ended by
        an error, it hands the starts that error and raises it.

        :return: nothing
        :rtype: None
        """
        try:
            await self._owned_rounds()
        except BaseException as error:
            self._drain()
            self._relearning = ()
            answers, self._answers = self._answers, ()
            for answer in answers:
                if answer.done():
                    continue
                if isinstance(error, asyncio.CancelledError):
                    answer.cancel()
                else:
                    answer.set_exception(error)  # a start waiting raises what ended the rounds
            self._starting = None
            raise
        answers, self._answers = self._answers, ()
        published, self._starting = self._starting or (), None
        for answer in answers:
            if not answer.done():
                answer.set_result(published)

    async def _owned_rounds(self) -> None:
        """run rounds while commits are queued, each after the commands posted are applied, the walk of the log a start
        owes, and any rebuild or recovery owed; a failed round keeps its files and waits for the next commit. Between
        its last look at the inbox and its return nothing awaits, so a command posted is never left for no one.

        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: never: a start's walk that fails answers that start with it, and the
            rounds end
        """
        while True:
            self._drain()
            if self._relearning:
                if not await self._relearned():
                    self._drain()
                    if not self._relearning:
                        return
                continue
            self._round_term = self._term
            if self._reader is not None and self._recheck:
                self._recheck = False
                self._marked = False
                await self._fresh_fulls(self._reader)
                await self._recoveries(self._reader)
                await self._delete_covered()
                self._drain()
                if self._relearning:
                    continue
            if not self._queued:
                if self._marked and self._reader is not None:
                    continue  # a table marked fresh during that rebuild is rebuilt now, not at the next commit
                return
            taken, self._queued = self._queued, ()
            self._taken = tuple(taken)
            pending = self._pending
            term = self._term
            assert term is not None  # noqa: S101 -- set by the commit that queued the files
            self._round_term = term
            began = time.monotonic()
            try:
                answer = await self._workers.run(
                    fold_round,
                    str(self.root),
                    term,
                    self._plain_tables,
                    tuple(taken),
                    self._pending,
                    self._reused,
                    self._doomed,
                    OWN_DELTA_SAVES.orphans,
                    tuple(sorted(set(self._fresh) | set(self._recover))),
                    self.retention,
                    self._folded_below(),
                    tuple((name, tuple(sorted(self._outside_round(name)))) for name in sorted(self.tables)),
                )
            except BaseException as error:  # prawduct:allow prawduct/broad-except -- a fold never fails a write: logged, counted, retried at the next commit
                self._queued = (*taken, *self._queued)
                self._taken = ()
                if isinstance(error, asyncio.CancelledError):
                    raise
                _ROUNDS.labels(outcome="failed").inc()
                _ROUND_SECONDS.labels(outcome="failed").observe(time.monotonic() - began)
                self._failed_rounds += 1
                log.error(
                    "a delta fold round failed; its files are kept for the round the next commit starts",
                    extra={
                        "extra_data": {
                            "files": len(taken),
                            "failed_in_a_row": self._failed_rounds,
                            "error": repr(error),
                        }
                    },
                    exc_info=error,
                )
                if self._failed_rounds >= _FAILED_ROUNDS_CAP:
                    self._hand_queue_to_recovery()
                self._drain()
                self._measure()
                if self._relearning:
                    continue  # a start posted meanwhile walks its log and folds now, as it always did
                return
            self._failed_rounds = 0
            _ROUND_SECONDS.labels(outcome=answer[3]).observe(time.monotonic() - began)
            self._took(answer)
            # each folded file's epoch, as told: what every copy must be past before it goes (decision 54)
            self._doom((), {file: token for _, file, token in taken} | {share[2]: share[3] for share in pending})
            await self._delete_covered()

    async def _relearned(self) -> bool:
        """walk the epoch log for each start applied (:meth:`_relearn`). A walk that fails answers every start waiting
        with its error, and the rounds end: a fold without the deltas it would not know of would publish parts lacking
        their rows, so the writer's start fails and is tried again.

        :return: whether every walk was read
        :rtype: bool
        """
        starts, self._relearning = self._relearning, ()
        try:
            for started in starts:
                await self._relearn(started.reader.generations, started.changes)
        except GenerationUnavailableError as error:
            # the log could not be read: handed to the starts waiting, which raise it, and the writer's start is
            # tried again
            answers, self._answers = self._answers, ()
            self._starting = None
            for answer in answers:
                if not answer.done():
                    answer.set_exception(error)
            return False
        return True

    def _hand_queue_to_recovery(self) -> None:
        """after :data:`_FAILED_ROUNDS_CAP` straight failed rounds, hand each queued table to a recovery by key
        (decision 28) and let its queued entries go: the recovery reads every delta file of the table the full does not
        name, so each queued file's change is in it, and the queue no longer grows with every failed round. Counted
        and logged. A fold not started (no L3 reader to recover with) keeps its queue: only a writer's start makes one.

        :return: nothing
        :rtype: None
        """
        if self._reader is None:
            return
        tables = sorted({table for table, _, _ in self._queued})
        for table in tables:
            _QUEUE_CAPPED.labels(table=table).inc()
            if table not in self._recover and table not in self._fresh:
                self._owe_recovery(
                    table,
                    "rounds_failed",
                    detail=f"{self._failed_rounds} fold rounds failed in a row: its queued files are recovered by key",
                )
        self._queued = tuple(item for item in self._queued if item[0] not in tables)
        self._failed_rounds = 0

    def _took(self, answer: tuple[Any, ...]) -> None:
        """keep what a round answered, and count and log it.

        :param answer: :func:`fold_round`'s answer
        :ptype answer: tuple
        :return: nothing
        :rtype: None
        """
        pending, reused, doomed, outcome, folded, deleted, swept, manifest, pid, renewed, full_bytes = answer
        self._pending, self._reused, self._doomed = pending, reused, doomed
        OWN_DELTA_SAVES.swept(swept)
        self._taken = ()
        self._measured_fulls(full_bytes)
        _ROUNDS.labels(outcome=outcome).inc()
        for table in renewed:
            # a commit of another incarnation than the table's full: never folded across, recovered by key instead
            if table not in self._fresh and table not in self._recover:
                self._owe_recovery(table, "new_incarnation", detail="a commit's generation is of another incarnation")
        if outcome == "no_full":
            # no full on the volume (lost since the writer's start): rebuilt before the next round
            self._recheck = True
        for table, _scope, how, _rows in folded:
            _STATES.labels(table=table, how=how).inc()
        if outcome == "stale":
            log.warning(
                "a later writer's full is current: this writer's lease was taken, so its folds publish nothing",
                extra={"extra_data": {"term": self._term, "manifest": manifest}},
            )
        elif folded:
            log.info(
                "delta files folded into a new full",
                extra={
                    "extra_data": {
                        "manifest": manifest,
                        "states": len(folded),
                        "reused": sum(1 for _, _, how, _ in folded if how == "reused"),
                        "deleted": deleted,
                        "swept": len(swept),
                        "pending": len(pending),
                        "worker": pid,
                    }
                },
            )

    async def _fresh_fulls(self, reader: _Reader) -> None:
        """rebuild each table owed one (no full holds it, or :meth:`start_fresh` marked it), then drop what the
        rebuild covers from the folds' queue. Never raises but on cancellation: a failure is logged, counted and
        retried before the next round.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :return: nothing
        :rtype: None
        """
        term = self._term
        assert term is not None  # noqa: S101 -- set by start() before any round looks
        current = await asyncio.to_thread(current_full_manifest, self.root)
        owed: dict[str, str] = {
            name: "cold_start" for name in self.tables if current is None or name not in current.tables
        }
        owed.update(self._fresh)
        if not owed:
            return
        if current is not None and current.term > term:
            for name, why in owed.items():
                _REBUILDS.labels(table=name, why=why, outcome="stale").inc()
            log.warning(
                "a later writer's full is current: this writer's lease was taken, so it rebuilds nothing",
                extra={"extra_data": {"term": term, "manifest": current.name, "tables": sorted(owed)}},
            )
            return
        staged: tuple[tuple[str, tuple[str, ...], str, int], ...] = ()
        #: each table's files told before its generation was read: covered by its rebuild whatever their incarnation
        known: dict[str, frozenset[str]] = {}
        #: each table's asks (a reset, a lost file) applied before its generation was read: one after is owed again
        asked: dict[str, int] = {}
        # a rebuild's rows left by a writer that died mid-read: only the writer holding the lease writes there
        await asyncio.to_thread(_cleared, self.root / _REBUILD_DIR)
        log.info(
            "rebuilding fulls from L3: each table read whole once, several at once, then one full published",
            extra={
                "extra_data": {
                    "tables": sorted(owed),
                    "count": len(owed),
                    "term": term,
                    "tables_at_once": self.rebuild.tables,
                    "streams": self.rebuild.streams,
                }
            },
        )
        try:
            #: each table read, with its generation: every generation read before any table's rows
            reads: list[tuple[str, str, FoldedTable, str]] = []
            for name, why in sorted(owed.items()):
                # what was posted before the generation is read is judged covered by it; what comes after, not
                self._drain()
                known[name] = frozenset(file for table_name, file, _ in self._queued if table_name == name) | {
                    share[2] for share in self._pending if share[0] == name
                }
                asked[name] = self._asked.get(name, 0)
                try:
                    # the generation before the rows: a commit during the read is in the rows and applied again over
                    # them (module docstring), never missed. All read first, so the tables' rows can be read at once
                    token = await reader.generations.current(name)
                except GenerationUnavailableError as error:
                    # never written: its first commit makes one, and the round that commit starts looks again
                    self._recheck = True
                    _REBUILDS.labels(table=name, why=why, outcome="no_generation").inc()
                    log.warning(
                        "a table to rebuild has no generation yet: rebuilt before the round its first commit starts",
                        extra={"extra_data": {"table": name, "why": why, "error": str(error)}},
                    )
                    continue
                reads.append((name, why, self.tables[name], token))
            if reads:
                staged = await self._read_tables(reader, tuple(reads))
            if not staged:
                self._rebuilding = None
                return
            self._rebuilding = f"publishing the rebuilt full ({len(staged)} tables)"
            # every ask posted while the rows were read applied first: a file told meanwhile is open, never claimed
            self._drain()
            answer = await self._workers.run(
                rebuild_round,
                str(self.root),
                term,
                tuple(self.tables[name].plain() for name, _, _, _ in staged),
                tuple(staged),
                self._folded_below(),
                tuple((name, self._open_files(name, covered=known[name])) for name, _, _, _ in staged),
                tuple((name, tuple(sorted(known[name]))) for name, _, _, _ in staged),
                self._kept_files(covered=frozenset().union(*known.values())),
                self.retention,
            )
        except BaseException as error:  # prawduct:allow prawduct/broad-except -- a rebuild never fails a write: logged, counted, retried before the next round
            self._rebuilding = None
            self._recheck = True
            # every stream's file, of the tables read and those cut mid-read: the rebuild's directory is this writer's
            await asyncio.to_thread(_cleared, self.root / _REBUILD_DIR)
            if isinstance(error, asyncio.CancelledError):
                raise
            for name, why in owed.items():
                _REBUILDS.labels(table=name, why=why, outcome="failed").inc()
            log.error(
                "a rebuild of the full from L3 failed; it is tried again before the next round",
                extra={"extra_data": {"tables": sorted(owed), "error": repr(error)}},
                exc_info=error,
            )
            return
        self._rebuilding = None
        outcome, manifest, rebuilt, deleted, full_bytes, covered = answer
        self._measured_fulls(full_bytes)
        self._doom(covered, {})
        # every ask posted while the rebuild read and published, judged against the generation each table's read began at
        self._drain()
        for name, token, rows in rebuilt:
            why = owed[name]
            _REBUILDS.labels(table=name, why=why, outcome=outcome).inc()
            if outcome != "rebuilt":
                continue
            if self._asked.get(name, 0) == asked[name]:
                # nothing asked of the table since its generation was read: the rows hold every change asked of it, its
                # leftover and lost files' too, so neither a rebuild nor a recovery is owed after it
                self._fresh = without_entry(self._fresh, name)
                self._recover = without_entry(self._recover, name)
                self._lost = without_entry(self._lost, name)
            else:
                # a reset, or a lost file whose epoch may be past the generation read, asked while the rebuild read:
                # owed again, never dropped by a rebuild that could not hold it
                self._recheck = True
                self._marked = True
            self._published_one(name)
            self._drop_covered(name, token, known[name])
            log.info(
                "a table's full rebuilt from L3 and published",
                extra={"extra_data": {"table": name, "why": why, "rows": rows, "generation": token}},
            )
        if outcome == "rebuilt":
            log.info(
                "rebuilt full published", extra={"extra_data": {"manifest": manifest, "deleted": deleted, "term": term}}
            )
        else:
            log.warning(
                "a later writer's full is current: this writer's rebuild publishes nothing",
                extra={"extra_data": {"term": term, "manifest": manifest}},
            )

    async def _read_tables(
        self, reader: _Reader, reads: tuple[tuple[str, str, FoldedTable, str], ...]
    ) -> tuple[tuple[str, tuple[str, ...], str, int], ...]:
        """read each table's rows from L3 beside the full, up to :attr:`rebuild` ``.tables`` tables and ``.streams``
        reads at once (:class:`RebuildConcurrency`), saying each table's read as it ends.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :param reads: each table's ``(table, why, its FoldedTable, generation)``, the generation read before its rows
        :ptype reads: tuple
        :return: each table's ``(table, files of its rows, generation, rows)``, in the tables' order
        :rtype: tuple
        """
        # an L3 that answers dict rows (a plain pool: each row decoded and made Arrow under the interpreter lock) is
        # read one statement at a time, as before: read at once its reads only contend for the lock (Postgres 16, 1M
        # rows: 1.8 s one at a time, 4.8 s six at once, 2026-11-25); the rail's proxy parses its replies off it
        concurrency = self.rebuild if isinstance(reader.l3, ArrowL3Reader) else _ONE_AT_A_TIME
        streams = asyncio.Semaphore(concurrency.streams)
        tables = asyncio.Semaphore(concurrency.tables)
        began = time.monotonic()
        read: list[tuple[str, tuple[str, ...], str, int]] = []
        self._rebuilding = f"rebuilding from L3: 0 of {len(reads)} tables read"
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._read_table(reader, table, why, concurrency.streams, streams, tables))
                for _, why, table, _ in reads
            ]
            tokens = {name: token for name, _, _, token in reads}
            for index, done in enumerate(asyncio.as_completed(tasks), start=1):
                name, why, files, rows, seconds = await done
                # the round's own status, its one writer: each table said as its read ends, so a stall is told from
                # slow (2026-11-24: 532 s, silent between its start and its publish)
                self._rebuilding = f"rebuilding from L3: {index} of {len(reads)} tables read"
                log.info(
                    "a table read from L3 for its full's rebuild",
                    extra={
                        "extra_data": {
                            "table": name,
                            "why": why,
                            "rows": rows,
                            "seconds": round(seconds, 3),
                            "rows_per_second": round(rows / seconds) if seconds > 0 else None,
                            "streams": len(files),
                            "of": f"{index} of {len(reads)}",
                        }
                    },
                )
                read.append((name, files, tokens[name], rows))
        seconds = time.monotonic() - began
        rows = sum(count for _, _, _, count in read)
        log.info(
            "every table of the rebuild read from L3",
            extra={
                "extra_data": {
                    "tables": len(read),
                    "tables_at_once": concurrency.tables,
                    "streams": concurrency.streams,
                    "rows": rows,
                    "seconds": round(seconds, 3),
                    "rows_per_second": round(rows / seconds) if seconds > 0 else None,
                }
            },
        )
        return tuple(sorted(read))

    async def _read_table(
        self,
        reader: _Reader,
        table: FoldedTable,
        why: str,
        split: int,
        streams: asyncio.Semaphore,
        tables: asyncio.Semaphore,
    ) -> tuple[str, str, tuple[str, ...], int, float]:
        """one table's rows from L3, beside the full: its first page by its plain keyset read, and when more follow,
        the rest split by its states (the lead of its key) into up to ``streams`` reads of a part of them each, all at
        once. A table its state does not lead is read whole by one stream.

        Why split by the lead: the rebuild's rows are ~86% three LSF tables (a cold rebuild of 2026-11-24), so reading
        tables at once alone leaves the biggest table's one stream the whole rebuild's time; its states (the lead) split
        it into reads that each seek their own states.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :param table: the table
        :ptype table: FoldedTable
        :param why: why it is rebuilt (``cold_start``, ``reset``, ``schema_change``)
        :ptype why: str
        :param split: the most parts its states are read in (one: read whole by its plain read)
        :ptype split: int
        :param streams: the reads at once, across every table
        :ptype streams: asyncio.Semaphore
        :param tables: the tables at once
        :ptype tables: asyncio.Semaphore
        :return: ``(table, why, files of its rows, rows, seconds)``
        :rtype: tuple
        """
        async with tables:
            began = time.monotonic()
            async with streams:
                if split > 1 and table.scope is not None and table.key[0] == table.scope[0]:
                    first, rows, after = await self._probed_beside(
                        table, _declared_pages(reader, table, why), reader.page_size
                    )
                else:
                    # one stream, or a table its state does not lead (or with no state): read whole by its one stream
                    first, rows = await self._written_beside(table, _declared_pages(reader, table, why))
                    after = None
            files = [first]
            if after is not None:
                async with streams:
                    assert table.scope is not None  # noqa: S101 -- probed only when its state leads its key
                    states = await keyset_scope_values(
                        reader.l3,
                        table.name,
                        scope_column=table.scope[0],
                        scope_type=_table_types(table)[table.scope[0]],
                        page_size=reader.page_size,
                        after=after[0],
                    )
                # an ordered lead (Postgres) is split into runs of its states, each one index range read in key order;
                # a hashed one (YugabyteDB) into sets of them dealt in turn, each walked state by state (keyset's "A
                # hashed lead a set filter pins"). Measured on Postgres 16, a 766k-row table in six: runs 0.16 s, sets
                # 0.32 s (2026-11-25)
                ranged = not await known_on_yugabyte(reader.l3)
                # a failed or cut stream's file goes with it; the others' with the rebuild's directory (_fresh_fulls)
                async with asyncio.TaskGroup() as group:
                    # the first part holds the state the first page ended in: read past its last key
                    parts = [
                        group.create_task(
                            self._stream_beside(
                                table,
                                _state_part_pages(reader, table, part, after if index == 0 else None, ranged=ranged),
                                streams,
                            )
                        )
                        for index, part in enumerate(_spread(states, split, contiguous=ranged))
                    ]
                files.extend(task.result()[0] for task in parts)
                rows += sum(task.result()[1] for task in parts)
            return table.name, why, tuple(files), rows, time.monotonic() - began

    async def _stream_beside(
        self, table: FoldedTable, pages: AsyncIterator[Any], streams: asyncio.Semaphore
    ) -> tuple[str, int]:
        """:meth:`_written_beside` holding one of the rebuild's ``streams`` while it reads.

        :param table: the table
        :ptype table: FoldedTable
        :param pages: its pages
        :ptype pages: AsyncIterator[pyarrow.Table]
        :param streams: the reads at once
        :ptype streams: asyncio.Semaphore
        :return: the file and its rows
        :rtype: tuple[str, int]
        """
        async with streams:
            return await self._written_beside(table, pages)

    async def _probed_beside(
        self, table: FoldedTable, pages: AsyncGenerator[Any, None], page_size: int
    ) -> tuple[str, int, tuple[Any, ...] | None]:
        """a table's first page written beside the full, as :meth:`_written_beside` writes them, which ends its plain
        read there when it is full: more may follow, read split by the lead. After a short one the plain read is asked
        once more: a table it ended is read whole in one statement; a second page (the broker cut the first by its
        bytes, or a walk of YugabyteDB's hashed lead read one lead's rows) ends it there instead.

        Why the first page alone: the second is sized to the broker's reply cap (keyset's "How many rows a page asks
        for"), a fifth of a 766k-row table read by one stream before the split (2026-11-25: six streams' read 2.0x one
        stream's with it).

        :param table: the table
        :ptype table: FoldedTable
        :param pages: its plain keyset read
        :ptype pages: AsyncGenerator[pyarrow.Table, None]
        :param page_size: the rows its first page asks for
        :ptype page_size: int
        :return: the file, its rows, and the last key written when more may follow (None: the table is read whole)
        :rtype: tuple[str, int, tuple | None]
        """
        types = _table_types(table)
        file = f"{_REBUILD_DIR}/{table.name}/{uuid.uuid7().hex}.arrow"
        sink = await asyncio.to_thread(_RowsFile, self.root / file, pa.schema(list(types.items())))
        after: tuple[Any, ...] | None = None
        taken = 0
        try:
            async with aclosing(pages) as held:
                async for page in held:
                    await asyncio.to_thread(sink.write, page)
                    taken += 1
                    if page.num_rows and (page.num_rows >= page_size or taken == 2):
                        # more may follow: the rest is read split by the lead, from past this page's last key
                        after = _last_key(page, table.key)
                        break
        except BaseException:
            await asyncio.to_thread(_closed_and_unlinked, sink)
            raise
        await asyncio.to_thread(sink.close)
        return file, sink.rows, after

    async def _written_beside(self, table: FoldedTable, pages: AsyncIterator[Any]) -> tuple[str, int]:
        """write a table's rows beside the full as its read from L3 hands them, a page at a time on a thread (Arrow
        writes with the interpreter lock released), for the worker process to split.

        :param table: the table
        :ptype table: FoldedTable
        :param pages: its pages, read whole
        :ptype pages: AsyncIterator[pyarrow.Table]
        :return: the file the rows were written to, under the root, and how many
        :rtype: tuple[str, int]
        """
        types = _table_types(table)
        file = f"{_REBUILD_DIR}/{table.name}/{uuid.uuid7().hex}.arrow"
        sink = await asyncio.to_thread(_RowsFile, self.root / file, pa.schema(list(types.items())))
        try:
            async for page in pages:
                await asyncio.to_thread(sink.write, page)
        except BaseException:
            # a read that failed or was cut (:meth:`close` past its bound): its half-written rows go with it, never
            # left for the next writer's start to clear
            await asyncio.to_thread(_closed_and_unlinked, sink)
            raise
        await asyncio.to_thread(sink.close)
        return file, sink.rows

    async def _recoveries(self, reader: _Reader) -> None:
        """recover each table owed one (its epoch log lost, decision 28) from the keys its leftover delta files name,
        read from L3 by key; a table the current full lacks is the rebuild's, read whole (decision 14). Never raises but
        on cancellation: a failure is logged, counted and retried before the next round.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :return: nothing
        :rtype: None
        """
        if not self._recover:
            return
        term = self._term
        assert term is not None  # noqa: S101 -- set by start() before any round looks
        for name, why in sorted(self._recover.items()):
            if name in self._fresh:
                continue  # owed a rebuild from L3 (a reset): it holds every change, and drops the recovery
            current = await asyncio.to_thread(current_full_manifest, self.root)
            if current is None or name not in current.tables:
                # no full holds the table: the rebuild reads it whole, the one cold start, before the next round
                self._recover = without_entry(self._recover, name)
                self._recheck = True
                continue
            if current.term > term:
                _RECOVERIES.labels(table=name, why=why, outcome="stale").inc()
                log.warning(
                    "a later writer's full is current: this writer's lease was taken, so it recovers nothing",
                    extra={"extra_data": {"term": term, "manifest": current.name, "table": name}},
                )
                return
            try:
                await self._recover_table(reader, name, why, term)
            except BaseException as error:  # prawduct:allow prawduct/broad-except -- a recovery never fails a write: logged, counted, retried before the next round
                self._recheck = True
                if isinstance(error, asyncio.CancelledError):
                    raise
                _RECOVERIES.labels(table=name, why=why, outcome="failed").inc()
                log.error(
                    "a recovery of a table's full by key failed; it is tried again before the next round",
                    extra={"extra_data": {"table": name, "why": why, "error": repr(error)}},
                    exc_info=error,
                )

    async def _recover_table(self, reader: _Reader, name: str, why: _RecoveryWhy, term: int) -> None:
        """one table's recovery (decision 28): its generation read first, then the keys and prefixes its leftover delta
        files name (its own delta folder listed, the one listing of the volume), their rows read from L3 by key, written
        as one recovery delta, folded into the kept full and published in a worker process; the leftover files saved
        before it began deleted.

        A commit landing during the reads is in the rows and its delta is folded again over them later, which ends
        where L3 does (decision 19's merge), as for the rebuild.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :param name: the table
        :ptype name: str
        :param why: why it is recovered
        :ptype why: str
        :param term: the writer's term
        :ptype term: int
        :return: nothing
        :rtype: None
        :raises Exception: any failure of a read or a write; the caller logs, counts and retries it
        """
        table = self.tables[name]
        # what was posted before the generation is read is judged covered by it; what comes after, not
        self._drain()
        asked = self._asked.get(name, 0)
        lost = MappingProxyType(dict(self._lost.get(name, {})))
        known = frozenset(file for table_name, file, _ in self._queued if table_name == name) | {
            share[2] for share in self._pending if share[0] == name
        }
        # this process's saves not ended when the recovery began: never dropped, whatever their name
        saving = OWN_DELTA_SAVES.in_flight
        try:
            # the generation before the keys' rows: every part is current to it once the recovery is folded
            token = await reader.generations.current(name)
        except GenerationUnavailableError as error:
            self._recheck = True
            _RECOVERIES.labels(table=name, why=why, outcome="no_generation").inc()
            log.warning(
                "a table to recover has no generation yet: recovered before the round its first commit starts",
                extra={"extra_data": {"table": name, "why": why, "error": str(error)}},
            )
            return
        leftovers, skipped = await asyncio.to_thread(_leftover_delta_files, self.root, name, known)
        # the recovery's own files beside the fulls: the keys reached, the notes' keys, the rows read; never held whole
        work = f"{_REBUILD_DIR}/{name}/{uuid.uuid7().hex}"
        try:
            file, rows, markers, named, replaced, keyed = await self._recovery_delta(
                reader, table, work, [left for left in leftovers if left not in lost], lost, term
            )
        finally:
            await asyncio.to_thread(_unlink_work, self.root, work)
        # every ask posted while the recovery read applied first: a file told meanwhile is open, never claimed
        self._drain()
        below = self._folded_below()
        # dropped when the fold deletes: a leftover the recovery's full covers by its name (below its claim, not told
        # since the generation was read, no save of this process in flight; decision 53), or a lost file. A file of the
        # older, unversioned name is never dropped here: no name says its commit ended, and no clock is compared
        told = self._told()
        doomable = tuple(
            sorted(
                {
                    left
                    for left in leftovers
                    if left != file
                    and left not in saving
                    and left not in OWN_DELTA_SAVES.in_flight
                    and (
                        # told before the generation was read: its commit landed, so the rows read hold it
                        left in known
                        or (
                            left not in told
                            and (version := delta_file_version(left)) is not None
                            and below is not None
                            and version < below
                        )
                    )
                }
                | {left for left in leftovers if left in lost}
            )
        )
        try:
            answer = await self._workers.run(
                recovery_round,
                str(self.root),
                term,
                table.plain(),
                file,
                token,
                doomable,
                replaced,
                self.retention,
                below,
                self._open_files(name, covered=known | {file}),
            )
        except BaseException:
            await asyncio.to_thread(_unlink_all, self.root, [file])
            raise
        outcome, manifest, states, deleted, full_bytes, covered = answer
        self._measured_fulls(full_bytes)
        self._doom(covered, {})
        # every ask posted while the recovery read and published, judged against the generation it read
        self._drain()
        _RECOVERIES.labels(table=name, why=why, outcome=outcome).inc()
        if outcome != "recovered":
            await asyncio.to_thread(_unlink_all, self.root, [file])
            if outcome == "no_full":
                self._recover = without_entry(self._recover, name)
                self._recheck = True
            else:
                log.warning(
                    "a later writer's full is current: this writer's recovery publishes nothing",
                    extra={"extra_data": {"term": term, "manifest": manifest, "table": name}},
                )
            return
        if self._asked.get(name, 0) == asked:
            self._recover = without_entry(self._recover, name)
        else:
            # asked again while it ran (decision 29): that file's epoch may be past the generation read; owed again
            self._recheck = True
            self._marked = True
        if lost:
            left = {held: epoch for held, epoch in self._lost.get(name, {}).items() if held not in lost}
            if left:
                self._lost = with_entry(self._lost, name, MappingProxyType(left))
            else:
                self._lost = without_entry(self._lost, name)
        for lost_reach in named:
            _LOST_FILE_RECOVERIES.labels(table=name, why=lost_reach.why).inc()
            log.warning(
                "a lost delta file of a commit too big to list was recovered "
                + {
                    "notes": "from the keys its change notes name, read from L3 by key",
                    "states": "by its states, read from L3 by state: the notes no longer cover it",
                    "every_state": "by every state of the table, read from L3 by state: the notes no longer cover it "
                    "and its record does not mark its states complete",
                    "unnamed": "only as far as the leftover files name: its record names no notes bracket the ring "
                    "covers and no states",
                }[lost_reach.why],
                extra={
                    "extra_data": {
                        "table": name,
                        "epoch": lost_reach.epoch,
                        "why": lost_reach.why,
                        "keys": lost_reach.noted,
                        "states": list(lost_reach.scopes),
                        "detail": lost_reach.detail,
                    }
                },
            )
        self._published_one(name)
        self._drop_covered(name, token, known)
        log.warning(
            "a folded table's full recovered from its leftover delta files' keys, read from L3 by key, and published",
            extra={
                "extra_data": {
                    "table": name,
                    "why": why,
                    "generation": token,
                    "manifest": manifest,
                    "leftover_files": len(leftovers),
                    "covered_files_skipped": skipped,
                    "keys": keyed[0],
                    "prefixes": keyed[1],
                    "states_replaced": len(replaced),
                    "rows": rows,
                    "delete_markers": markers,
                    "states": states,
                    "deleted": deleted,
                }
            },
        )

    async def _recovery_delta(
        self,
        reader: _Reader,
        table: FoldedTable,
        work: str,
        files: Sequence[str],
        lost: Mapping[str, str],
        term: int,
    ) -> tuple[str, int, int, list[_LostReach], tuple[str, ...], tuple[int, int]]:
        """the recovery delta (decision 28) of ``table``: the keys and prefixes its leftover ``files`` name and the keys
        the lost files' notes name, gathered in a worker process into a file of Arrow (:func:`recovery_reach_job`);
        their rows read from L3 by key a page of keys at a time, each page written beside the fulls as it comes
        (:class:`_RowsFile`), never held whole; then the delta built, compressed and saved in the worker
        (:func:`recovery_delta_job`). Plain paths in, a file name and counts out.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :param table: the table
        :ptype table: FoldedTable
        :param work: the recovery's own files' stem, under the root
        :ptype work: str
        :param files: the leftover delta files to read the keys of
        :ptype files: Sequence[str]
        :param lost: each lost file onto its epoch: its keys come from its record (decision 31)
        :ptype lost: Mapping[str, str]
        :param term: the writer's term: the recovery delta's file is named by it (decision 53), so a later full that
            claims past it covers it by its name
        :ptype term: int
        :return: the recovery delta's file, its rows and markers, each lost file's reach, the states replaced whole, and
            the keys and prefixes read by key
        :rtype: tuple
        """
        types = _table_types(table)
        kinds = dict(table.kinds)
        # what only the lost files named, from their records (decision 31): keys from the notes, else whole states
        named: list[_LostReach] = []
        noted_files: list[str] = []
        for index, epoch in enumerate(lost.values()):
            noted_file = f"{work}.noted{index}.arrow"
            lost_reach = await self._lost_commit_reach(table, epoch, noted_file)
            named.append(lost_reach)
            if lost_reach.why == "notes":
                noted_files.append(noted_file)
        reach_file = f"{work}.reach.arrow"
        prefixes, count = await self._workers.run(
            recovery_reach_job,
            str(self.root),
            table.plain(),
            table.kinds,
            table.delta_scope,
            tuple(files),
            tuple(noted_files),
            reach_file,
            reader.page_size,
        )
        held = [dict(prefix) for prefix in prefixes]
        pages_file = f"{work}.pages.arrow"
        pages = await asyncio.to_thread(_RowsFile, self.root / pages_file, pa.schema(list(types.items())))
        try:
            casts = tuple(sql_cast(kinds[column]) for column in table.key)
            # one page of keys at a time, its Arrow key columns bound as they are (KeyBatch's Arrow mode)
            async for chunk in _reach_pages(self.root / reach_file, table.key):
                columns = [chunk.column(column) for column in table.key]
                for page in await read_keyed_arrow(
                    reader.l3, table.name, keys=KeyBatch(table.key, columns, casts), types=types
                ):
                    await asyncio.to_thread(pages.write, page)
            for width, values in _prefixes_by_width(table.key, held).items():
                # every row under a prefix a leftover file cleared, by a key-led read of the prefix: the load's own size
                for page in await read_keyed_arrow(
                    reader.l3,
                    table.name,
                    keys=KeyBatch(
                        table.key[:width], values, tuple(sql_cast(kinds[column]) for column in table.key[:width])
                    ),
                    types=types,
                ):
                    await asyncio.to_thread(pages.write, page)
        finally:
            await asyncio.to_thread(pages.close)
        noting = [lost_reach for lost_reach in named if lost_reach.why == "notes"]
        unstated = False
        if table.delta_scope is not None and noting:
            unstated = await self._workers.run(
                recovery_unstated_job, str(self.root), table.key, table.kinds, reach_file, pages_file
            )
            if unstated and all(lost_reach.fallback or lost_reach.fallback_every for lost_reach in noting):
                named = [
                    replace(
                        lost_reach,
                        why="every_state" if lost_reach.fallback_every else "states",
                        scopes=lost_reach.fallback,
                        detail="a key its notes name is gone from L3 and no file names its state",
                    )
                    if lost_reach.why == "notes"
                    else lost_reach
                    for lost_reach in named
                ]
            else:
                # none gone of an unknown state; or one commit names no states, and its marker is refused (retried)
                unstated = False
        replaced = tuple(sorted({scope for lost_reach in named for scope in lost_reach.scopes}))
        if any(lost_reach.why == "every_state" for lost_reach in named):
            # a record not marking its states complete (decision 54 (c)): every state the table holds, replaced whole
            replaced = tuple(sorted({*replaced, *await self._every_state(reader, table)}))
        state_files: tuple[str, ...] = ()
        if replaced:
            states_file = f"{work}.states.arrow"
            states = await asyncio.to_thread(_RowsFile, self.root / states_file, pa.schema(list(types.items())))
            try:
                async for page in _state_rows(reader, table, types, replaced):
                    await asyncio.to_thread(states.write, page)
            finally:
                await asyncio.to_thread(states.close)
            state_files = (states_file,)
        file, rows, markers = await self._workers.run(
            recovery_delta_job,
            str(self.root),
            table.plain(),
            table.kinds,
            table.delta_scope,
            reach_file,
            (pages_file, *state_files),
            prefixes,
            replaced,
            unstated,
            term,
        )
        return file, rows, markers, named, replaced, (count, len(held))

    async def _lost_commit_reach(self, table: FoldedTable, epoch: str, noted_file: str) -> _LostReach:
        """what a lost file's commit reached, from its epoch's record (decision 31): the keys the change notes name in
        its bracket while the ring covers it (written to ``noted_file`` a page at a time, kept only when every page
        was read); else the states it names; else nothing.

        :param table: the table
        :ptype table: FoldedTable
        :param epoch: the epoch whose file was lost
        :ptype epoch: str
        :param noted_file: where its noted keys are written, under the root: a text column a key column
        :ptype noted_file: str
        :return: how many keys the notes named, or the states, and why
        :rtype: _LostReach
        :raises GenerationUnavailableError: when the epoch log cannot be read (the recovery is tried again)
        """
        record = await self._changes.change(table.name, epoch) if self._changes is not None else None
        if record is None:
            return _LostReach(epoch=epoch, why="unnamed", detail="the epoch log holds no record of it")
        sink = await asyncio.to_thread(
            _RowsFile, self.root / noted_file, pa.schema([(column, pa.string()) for column in table.key])
        )
        try:
            noted = await self._noted_keys(table, record, sink)
        finally:
            await asyncio.to_thread(sink.close)
        if not isinstance(noted, int):
            # the notes could not name every key: none of the pages read stands for the commit's reach
            await asyncio.to_thread(_unlink_all, self.root, [noted_file])
        scopes = _table_scopes(table, record)
        if isinstance(noted, int):
            return _LostReach(
                epoch=epoch, why="notes", noted=noted, fallback=scopes or (), fallback_every=scopes is None
            )
        if scopes is None:
            # the record does not mark its states complete: they may be a strict subset of what the commit wrote
            return _LostReach(
                epoch=epoch, why="every_state", detail=f"{noted}; its record does not mark its states complete"
            )
        if scopes:
            return _LostReach(epoch=epoch, why="states", scopes=scopes, detail=noted)
        return _LostReach(epoch=epoch, why="unnamed", detail=f"{noted}; the record names none of the table's states")

    async def _every_state(self, reader: _Reader, table: FoldedTable) -> tuple[str, ...]:
        """every state ``table`` holds, as its scopes name them, for a lost file whose record does not mark its states
        complete (decision 54 (c)): the states the current full has parts of (one L3 no longer holds then has no part),
        and the distinct states L3 holds (one the full lacks: a state new since it), read a page of states at a time.

        :param reader: how L3 is read
        :ptype reader: _Reader
        :param table: the table, which has a scope
        :ptype table: FoldedTable
        :return: the states
        :rtype: tuple[str, ...]
        """
        assert table.scope is not None  # noqa: S101 -- _table_scopes answers None only for a scoped table
        column, prefix = table.scope
        current = await asyncio.to_thread(current_full_manifest, self.root)
        entry = current.tables.get(table.name) if current is not None else None
        held = {part.scope for part in entry.parts if part.scope is not None} if entry is not None else set()
        values = await keyset_scope_values(
            reader.l3,
            table.name,
            scope_column=column,
            scope_type=_table_types(table)[column],
            page_size=reader.page_size,
        )
        return tuple(sorted(held | {f"{prefix}{value}" for value in values if value is not None}))

    async def _noted_keys(self, table: FoldedTable, record: EpochChange, sink: _RowsFile) -> int | str:
        """write every key the change notes name in ``record``'s bracket to ``sink``, a page at a time as Arrow; how
        many, or why they cannot (the caller then drops what was written).

        :param table: the table
        :ptype table: FoldedTable
        :param record: the lost file's record
        :ptype record: EpochChange
        :param sink: the noted keys' file
        :ptype sink: _RowsFile
        :return: the keys written; or why not (``no_bracket``, ``no_notes_reader``, ``whole_table``, or the reader's
            word: ``not_covered`` once the ring no longer holds the bracket)
        :rtype: int | str
        """
        if record.noted_between is None:
            return "no_bracket"
        if self._notes is None:
            return "no_notes_reader"
        count = 0
        async with aclosing(noted_key_pages(self._notes, table.name, record.noted_between)) as pages:
            async for page in pages:
                if not isinstance(page, NotedKeysPage):
                    return page
                if page.whole_table:
                    # another commit's load or truncate in the bracket: the keys are not the reach
                    return "whole_table"
                if page.keys:
                    # the reader's page as its Arrow key batch (a text column a key column, off the loop), named by
                    # the table's key: the page's own size, never a tuple a key here
                    batch = await asyncio.to_thread(page.key_batch)
                    keys = batch.rename_columns(list(table.key))
                    await asyncio.to_thread(sink.write, keys)
                    count += keys.num_rows
        return count

    def _doom(self, files: Sequence[str], tokens: Mapping[str, str]) -> None:
        """remember ``files`` as covered delta files to delete once every copy is past them (decision 54), and each
        doomed or reused file's epoch where ``tokens`` names it. Kept (``retention="keep"``), nothing is remembered.

        :param files: the files newly covered
        :ptype files: Sequence[str]
        :param tokens: file -> the epoch its commit was told at, where known
        :ptype tokens: Mapping[str, str]
        :return: nothing
        :rtype: None
        """
        if self.retention != "delete":
            return
        if files:
            self._doomed = tuple(sorted(set(self._doomed) | set(files)))
        held = set(self._doomed) | set(self._reused)
        epochs = {file: epoch for file, epoch in self._doomed_epochs.items() if file in held}
        epochs.update({file: tokens[file] for file in held if file not in epochs and file in tokens})
        self._doomed_epochs = MappingProxyType(epochs)

    async def _delete_covered(self) -> None:
        """delete each covered delta file no copy can still need (decision 54; its first rule: when in doubt, keep).
        A file goes only when the current and the previous full both cover it by its name (decision 53), no manifest in
        use names it, it is not in flight (told and unfolded, waiting to be confirmed, or a save of this process not
        ended), and every live copy's beat, read again immediately before this delete, is at or past its epoch in the
        same incarnation (its told epoch; else the previous full's version of the table, which is at or past it). A copy
        behind keeps that file; a copy starting, a beat unreadable, lacking the table or of another incarnation, or a
        read that fails keeps every file this round. Idempotent (a file gone already is no error) and logged per file
        with the beats that allowed it.

        :return: nothing
        :rtype: None
        """
        if self.retention != "delete" or self._copies is None:
            return
        candidates = tuple(sorted(set(self._doomed) | set(self._reused)))
        if not candidates:
            return
        current, previous = await asyncio.to_thread(_current_and_previous, self.root)
        if current is None or previous is None:
            return
        named = current.files() | previous.files()
        busy = self._told() | OWN_DELTA_SAVES.in_flight | {file for _, file in self._unconfirmed}
        gone: set[str] = set()
        for file in candidates:
            table = file.split("/", 1)[0]
            now, before = current.tables.get(table), previous.tables.get(table)
            if (
                file in named
                or file in busy
                or now is None
                or before is None
                or not (now.covers(file) and before.covers(file))
            ):
                continue
            epoch = self._doomed_epochs.get(file, before.version)
            # read again right before each delete, never once a round: a copy that started meanwhile keeps it
            positions = await self._copies.positions()
            verdict = positions.judge(table, epoch)
            if verdict == "behind":
                _DELETES_KEPT.labels(table=table, why=verdict).inc()
                continue
            if verdict != "delete":
                _DELETES_KEPT.labels(table=table, why=verdict).inc()
                log.info(
                    "the fold deletes no delta file this round: a copy's position is not known",
                    extra={
                        "extra_data": {"table": table, "file": file, "why": verdict, "beats": positions.said(table)}
                    },
                )
                break
            await asyncio.to_thread(_unlink_all, self.root, [file])
            gone.add(file)
            _DELETED.labels(table=table).inc()
            log.info(
                "a covered delta file deleted: every copy is past it",
                extra={"extra_data": {"table": table, "file": file, "epoch": epoch, "beats": positions.said(table)}},
            )
        if gone:
            self._doomed = tuple(file for file in self._doomed if file not in gone)
            self._reused = tuple(file for file in self._reused if file not in gone)
            self._doomed_epochs = MappingProxyType(
                {file: epoch for file, epoch in self._doomed_epochs.items() if file not in gone}
            )

    def _drop_covered(self, table: str, token: str, known: frozenset[str]) -> None:
        """drop from the folds' queue each of ``table``'s commits its rebuild at ``token`` holds (of that incarnation,
        at or below its count; of another, one told before the generation was read: a commit of the lost log): a copy
        applies only the deltas newer than a part, and a fold of an older one would undo a later write. Their files
        are deleted with the next round's when the fold deletes (``retention``), else kept.

        :param table: the table rebuilt
        :ptype table: str
        :param token: the generation its parts are current to
        :ptype token: str
        :param known: the table's files told before that generation was read
        :ptype known: frozenset[str]
        :return: nothing
        :rtype: None
        """
        held = split_generation_token(token)

        def covered(name: str, file: str, share_token: str) -> bool:
            if name != table or held is None:
                return False
            parsed = split_generation_token(share_token)
            if parsed is not None and parsed[0] != held[0]:
                return file in known
            return parsed is not None and parsed[1] <= held[1]

        dropped = {file for name, file, share_token in self._queued if covered(name, file, share_token)}
        dropped |= {share[2] for share in self._pending if covered(share[0], share[2], share[3])}
        # each dropped file's epoch, as told, before the queue lets it go
        told_tokens = {file: share_token for name, file, share_token in self._queued if file in dropped}
        told_tokens.update({share[2]: share[3] for share in self._pending if share[2] in dropped})
        self._queued = tuple(item for item in self._queued if not covered(*item))
        self._pending = tuple(share for share in self._pending if not covered(share[0], share[2], share[3]))
        left = {item[1] for item in self._queued} | {share[2] for share in self._pending}
        # kept (decision 51): nothing is remembered to delete (_doom keeps nothing)
        self._doom(tuple(sorted(dropped - left)), told_tokens)
        self._measure()


@dataclass(frozen=True, slots=True)
class _LostReach:
    """what a lost file's commit reached, from its record (decision 31).

    :ivar epoch: the epoch whose file was lost
    :ivar why: ``notes``, ``states`` or ``unnamed``
    :ivar noted: for ``notes``, how many keys its notes name (written to the recovery's noted keys' file)
    :ivar scopes: for ``states``, the table's states the record names, as its scopes name them (``state:CA``)
    :ivar detail: why the notes could not name the keys, for the log line
    :ivar fallback: for ``notes``, the states the record names: read instead when a key the notes name is gone from L3
        and no file names its state (a table keyed without its state), so no marker of it could be filed
    :ivar fallback_every: for ``notes``, the record does not mark its states complete: every state is read instead
    """

    epoch: str
    why: _LostWhy
    noted: int = 0
    scopes: tuple[str, ...] = ()
    detail: str = ""
    fallback: tuple[str, ...] = ()
    fallback_every: bool = False


def _lacking(current: FullManifest | None, tables: Mapping[str, FoldedTable]) -> tuple[str, ...]:
    """the tables ``current`` does not hold, or holds over other columns than their schema's (:meth:`DeltaFolds.
    lacking`).

    :param current: the current manifest; None when none is published
    :ptype current: FullManifest | None
    :param tables: the folded tables, by name
    :ptype tables: Mapping[str, FoldedTable]
    :return: their names, sorted
    :rtype: tuple[str, ...]
    """
    if current is None:
        return tuple(sorted(tables))
    return tuple(
        sorted(
            name
            for name, table in tables.items()
            if name not in current.tables or other_columns_parts(current.tables[name], table.kinds)
        )
    )


def _table_scopes(table: FoldedTable, record: EpochChange) -> tuple[str, ...] | None:
    """the states of ``table`` a record names, when it marks them complete
    (:func:`~threetears.core.collections.epoch_change.complete_scopes`): its scopes under the table's scope prefix (a
    scope of another prefix is another table's, since each row's scope is its prefix and its scope column's value).

    Why only marked (decision 54 (c)): a record without the mark names the states its writer touched, which nothing
    ties to the rows its commit wrote; recovering only those would keep the old rows of a state it missed.

    :param table: the table
    :ptype table: FoldedTable
    :param record: the record
    :ptype record: EpochChange
    :return: the states, as the table's scopes name them; None for a scoped table whose record does not mark its states
        complete (every state is recovered); empty for a table with no scope
    :rtype: tuple[str, ...] | None
    """
    if table.scope is None:
        return ()
    complete = complete_scopes(record.scopes)
    if complete is None:
        return None
    return tuple(scope for scope in complete if scope.startswith(table.scope[1]))


async def _state_rows(
    reader: _Reader, table: FoldedTable, types: Mapping[str, Any], scopes: Sequence[str]
) -> AsyncIterator[Any]:
    """every row L3 holds of the states ``scopes`` name (decision 31): one keyset read led by the state column, the
    states a set filter on it, so each page seeks the states' index range (the key when it leads with the state, the
    table's state index otherwise), never the table.

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table, which has a scope
    :ptype table: FoldedTable
    :param types: its columns' Arrow types, in order
    :ptype types: Mapping[str, Any]
    :param scopes: the states, as its scopes name them
    :ptype scopes: Sequence[str]
    :return: the pages, as read (the caller writes each beside the fulls)
    :rtype: AsyncIterator[pyarrow.Table]
    """
    assert table.scope is not None  # noqa: S101 -- _table_scopes names states only of a scoped table
    kind = dict(table.kinds)[table.scope[0]]
    async for page in keyset_scope_pages(
        reader.l3,
        table.name,
        table.key,
        types=types,
        scope_column=table.scope[0],
        scope_values=[typed_text(kind, scope.removeprefix(table.scope[1])) for scope in scopes],
        scope_array=f"{sql_cast(kind)}[]",
        page_size=reader.page_size,
    ):
        yield page


@dataclass(frozen=True, slots=True)
class _Reader:
    """how the writer reads L3 for a rebuild (:meth:`DeltaFolds.start`)."""

    l3: Any
    generations: GenerationSource
    page_size: int


@dataclass(frozen=True, slots=True)
class _Kind:
    """a column's name and type tag: all :func:`~threetears.core.collections.schema_backed.rail_arrow_types` reads of
    a column. Never a :class:`~threetears.core.collections.schema_backed.Column` built from the tag alone: that is
    refused for a column whose declaration needs more (a ``NUMERIC`` its precision and scale, an enum its values, a
    vector its size), which refused the writer's start and its rebuild of a full for every such table (2026-11-23)."""

    name: str
    column_type: str


def _kinds_types(kinds: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """columns' Arrow types as the L3 rail answers them, in order, from their ``(column, column_type)`` tags.

    :param kinds: each column's type tag, :attr:`FoldedTable.kinds`
    :ptype kinds: Sequence[tuple[str, str]]
    :return: column -> its Arrow type
    :rtype: dict[str, Any]
    """
    return rail_arrow_types([_Kind(name, kind) for name, kind in kinds])


def _table_types(table: FoldedTable) -> dict[str, Any]:
    """a table's columns' Arrow types as the L3 rail answers them, in order.

    :param table: the table
    :ptype table: FoldedTable
    :return: column -> its Arrow type
    :rtype: dict[str, Any]
    :raises ValueError: when the table names no column types (``FoldedTable.kinds``)
    """
    if not table.kinds:
        raise ValueError(f"{table.name}: a rebuild reads the table by its column types; make it FoldedTable.of")
    return _kinds_types(table.kinds)


def _lead_array(table: FoldedTable) -> str:
    """the SQL array cast of a table's lead key column, for its keyset read (a declared lead is hashed on YugabyteDB).

    :param table: the table
    :ptype table: FoldedTable
    :return: the cast
    :rtype: str
    """
    return f"{sql_cast(dict(table.kinds)[table.key[0]])}[]"


def _declared_pages(
    reader: _Reader,
    table: FoldedTable,
    why: str,
) -> AsyncGenerator[Any, None]:
    """a table's whole read for its rebuild, declared as its situation (each read function holds its own declaration,
    and its call says it again): its first page, the rest read by its states when it has more
    (:meth:`DeltaFolds._read_table`).

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table
    :ptype table: FoldedTable
    :param why: why it is rebuilt: ``schema_change``, else a cold start (``cold_start``, ``reset``)
    :ptype why: str
    :return: its keyset pages
    :rtype: AsyncGenerator[pyarrow.Table, None]
    """
    if why == "schema_change":
        return whole_read(
            _schema_change_pages(reader, table),
            kind="schema_change",
            reason="a folded table was dropped or recreated by a migration: the writer reads that table alone once and publishes its "
            "full",
            ruling="Pace 2026-11-12: reset, schema change, rolling upgrade; Pace 2026-11-23: a schema change starts the table fresh, "
            "one read of it from L3 (design-pod-delta-files decision 12)",
            proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
            speed="packages/core/tests/integration/test_delta_fold_rebuild_streams_live.py::test_rebuild_streams_speed_at_real_size",
        )
    return whole_read(
        _cold_start_pages(reader, table),
        kind="cold_start",
        reason="no full on the delta-files volume holds the table (a lost volume, a new environment), or it was reset: the writer "
        "reads it once and publishes its full; copies wait for that, never reading L3 whole",
        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; Pace 2026-11-23: with only L3 left the writer "
        "rebuilds the full, once (design-pod-delta-files decisions 12 and 14); Pace 2026-11-25: several tables and states read at once",
        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
        speed="packages/core/tests/integration/test_delta_fold_rebuild_streams_live.py::test_rebuild_streams_speed_at_real_size",
    )


def _spread(values: Sequence[Any], streams: int, *, contiguous: bool) -> tuple[tuple[Any, ...], ...]:
    """``values`` split into at most ``streams`` parts, the first holding the first value: runs of them in order
    (``contiguous``, each read as a range), or dealt one to each part in turn, so the big states (their sizes unknown
    before the read) fall in different parts.

    :param values: the lead values, in the database's order
    :ptype values: Sequence[Any]
    :param streams: the most parts
    :ptype streams: int
    :param contiguous: runs in order, not dealt
    :ptype contiguous: bool
    :return: the parts, none empty
    :rtype: tuple[tuple[Any, ...], ...]
    """
    count = min(len(values), streams)
    if contiguous:
        return tuple(
            tuple(values[index * len(values) // count : (index + 1) * len(values) // count]) for index in range(count)
        )
    return tuple(tuple(values[index::count]) for index in range(count))


def _last_key(page: Any, key: Sequence[str]) -> tuple[Any, ...]:
    """the key of a page's last row, read off the table (one row).

    :param page: a non-empty page
    :ptype page: pyarrow.Table
    :param key: the key columns
    :ptype key: Sequence[str]
    :return: its key
    :rtype: tuple[Any, ...]
    """
    last = page.slice(page.num_rows - 1, 1).select(list(key)).to_pylist()[0]
    return tuple(last[column] for column in key)


def _state_part_pages(
    reader: _Reader, table: FoldedTable, states: Sequence[Any], after: Sequence[Any] | None, *, ranged: bool
) -> AsyncGenerator[Any, None]:
    """one part of a table's read for its rebuild: the rows of a part of its states (its key's lead), past ``after``;
    the table's first page was its declared whole read (:func:`_declared_pages`).

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table, scoped, its key led by its state column
    :ptype table: FoldedTable
    :param states: the states, in the database's order
    :ptype states: Sequence[Any]
    :param after: the key the read starts past; the first row when None
    :ptype after: Sequence[Any] | None
    :param ranged: read the states as the range from the first to the last (an ordered lead), not as a set
    :ptype ranged: bool
    :return: the pages
    :rtype: AsyncGenerator[pyarrow.Table, None]
    """
    assert table.scope is not None  # noqa: S101 -- only a table led by its state is read by its states
    return keyset_scope_pages(
        reader.l3,
        table.name,
        table.key,
        types=_table_types(table),
        scope_column=table.scope[0],
        scope_values=states,
        scope_array=_lead_array(table),
        page_size=reader.page_size,
        after=after,
        ranged=ranged,
    )


def _cold_start_pages(
    reader: _Reader,
    table: FoldedTable,
) -> AsyncGenerator[Any, None]:
    """a table's whole read for its rebuild: no full on the volume holds it (a
    lost volume, a new environment, a table newly opted in), or an operator reset it.

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table
    :ptype table: FoldedTable
    :return: its keyset pages
    :rtype: AsyncGenerator[pyarrow.Table, None]
    """
    types = _table_types(table)
    selected = ", ".join(quote_identifier(column) for column in types)
    return whole_read(
        keyset_arrow_pages(
            reader.l3,
            f"SELECT {selected} FROM {quote_identifier(table.name)}",  # noqa: S608 - trusted identifiers
            table.key,
            types=types,
            page_size=reader.page_size,
            lead_array=_lead_array(table),
            table=table.name,
        ),
        kind="cold_start",
        reason="no full on the delta-files volume holds the table (a lost volume, a new environment), or it was reset: the writer "
        "reads it once and publishes its full; copies wait for that, never reading L3 whole",
        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; Pace 2026-11-23: with only L3 left the writer "
        "rebuilds the full, once (design-pod-delta-files decisions 12 and 14); Pace 2026-11-25: several tables and states read at once",
        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
        speed="packages/core/tests/integration/test_delta_fold_rebuild_streams_live.py::test_rebuild_streams_speed_at_real_size",
    )


def _schema_change_pages(
    reader: _Reader,
    table: FoldedTable,
) -> AsyncGenerator[Any, None]:
    """a table's whole read for its rebuild: a migration dropped or recreated it.

    :param reader: how L3 is read
    :ptype reader: _Reader
    :param table: the table
    :ptype table: FoldedTable
    :return: its keyset pages
    :rtype: AsyncGenerator[pyarrow.Table, None]
    """
    types = _table_types(table)
    selected = ", ".join(quote_identifier(column) for column in types)
    return whole_read(
        keyset_arrow_pages(
            reader.l3,
            f"SELECT {selected} FROM {quote_identifier(table.name)}",  # noqa: S608 - trusted identifiers
            table.key,
            types=types,
            page_size=reader.page_size,
            lead_array=_lead_array(table),
            table=table.name,
        ),
        kind="schema_change",
        reason="a folded table was dropped or recreated by a migration: the writer reads that table alone once and publishes its "
        "full",
        ruling="Pace 2026-11-12: reset, schema change, rolling upgrade; Pace 2026-11-23: a schema change starts the table fresh, "
        "one read of it from L3 (design-pod-delta-files decision 12)",
        proof="packages/core/tests/unit/collections/test_delta_fold.py::test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild",
        speed="packages/core/tests/integration/test_delta_fold_rebuild_streams_live.py::test_rebuild_streams_speed_at_real_size",
    )


class _RowsFile:
    """an Arrow IPC file a rebuild's rows are written to as they are read; blocking, each call on a thread.

    :param path: the file
    :ptype path: Path
    :param schema: the rows' schema
    :ptype schema: pyarrow.Schema
    """

    def __init__(self, path: Path, schema: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.rows = 0
        self._sink = pa.OSFile(str(path), "wb")
        self._writer = pa_ipc.new_file(self._sink, schema)

    def write(self, page: Any) -> None:
        """append ``page``, cast to the schema, and count its rows.

        :param page: rows to append
        :ptype page: pyarrow.Table
        """
        self._writer.write_table(page.cast(self.schema))
        self.rows += page.num_rows

    def close(self) -> None:
        """finish the IPC file and close it, the file even when the writer fails."""
        try:
            self._writer.close()
        finally:
            self._sink.close()


def _closed_and_unlinked(sink: _RowsFile) -> None:
    """close a rows file and remove it; blocking.

    :param sink: the rows file
    :ptype sink: _RowsFile
    :return: nothing
    :rtype: None
    """
    try:
        sink.close()
    finally:
        sink.path.unlink(missing_ok=True)


def _unlink_all(root: Path, files: Sequence[str]) -> None:
    """remove each of ``files`` under ``root``, those already gone included; blocking.

    :param root: the root
    :ptype root: Path
    :param files: the files, relative to it
    :ptype files: Sequence[str]
    :return: nothing
    :rtype: None
    """
    for file in files:
        (root / file).unlink(missing_ok=True)


def _not_confirmed(root: Path, table: str, files: Sequence[str]) -> list[str]:
    """each of ``files`` (failed load requests' files, named unconfirmed) that does not read as confirmed from L3:
    missing, unreadable, or still the request's own; blocking. Each is opened by its name, never listed.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table whose files they are
    :ptype table: str
    :param files: the files, relative to it
    :ptype files: Sequence[str]
    :return: those not confirmed
    :rtype: list[str]
    """
    waiting: list[str] = []
    for file in files:
        try:
            confirmed = read_delta_file_keys(root / file, table=table).confirmed
        except FileNotFoundError, DeltaFileUnreadableError:
            confirmed = False  # NOSILENT: left out, said by the caller; the copies' ask recovers it (decision 28)
        if not confirmed:
            waiting.append(file)
    return waiting


def _current_and_previous(root: Path) -> tuple[FullManifest | None, FullManifest | None]:
    """the current full's manifest and the one it replaced, each a reader may hold; blocking.

    :param root: the delta files' root
    :ptype root: Path
    :return: both; either None when not on the volume
    :rtype: tuple[FullManifest | None, FullManifest | None]
    """
    current = current_full_manifest(root)
    if current is None or current.previous is None:
        return current, None
    path = root / ".fulls" / "manifests" / current.previous
    return current, FullManifest.decode(path.read_bytes()) if path.is_file() else None


def _present(root: Path, files: Sequence[str]) -> list[str]:
    """each of ``files`` that is on the volume, in order; blocking. Each is opened by its name, never listed.

    :param root: the delta files' root
    :ptype root: Path
    :param files: the files, relative to it
    :ptype files: Sequence[str]
    :return: those present
    :rtype: list[str]
    """
    return [file for file in files if (root / file).is_file()]


def _cleared(directory: Path) -> None:
    """remove every file under each table's directory of ``directory``; blocking.

    :param directory: the rebuilds' directory
    :ptype directory: Path
    :return: nothing
    :rtype: None
    """
    if not directory.is_dir():
        return
    for table in directory.iterdir():
        if table.is_dir():
            for path in table.iterdir():
                path.unlink(missing_ok=True)


def rebuild_round(
    root: str,
    term: int,
    tables: tuple[tuple[Any, ...], ...],
    staged: tuple[tuple[str, tuple[str, ...], str, int], ...],
    below: int | None,
    opens: tuple[tuple[str, tuple[str, ...]], ...],
    told: tuple[tuple[str, tuple[str, ...]], ...],
    kept: tuple[str, ...],
    retention: DeltaRetention,
) -> tuple[Any, ...]:
    """a rebuild, in a worker process: each table's rows read from L3 split into its parts by state, the keep rule
    applied, the parts written and the manifest published at the writer's term; then the fulls' old files dropped, and
    the tables' old delta files too when ``retention`` is ``"delete"``.

    Plain data in and out (:mod:`~threetears.core.process_workers`): names, versions and counts, never rows.

    :param root: the delta files' root
    :ptype root: str
    :param term: the writer's term
    :ptype term: int
    :param tables: each rebuilt table, :meth:`FoldedTable.plain`
    :ptype tables: tuple
    :param staged: each table's ``(table, files of its rows, generation, rows)``: a file a stream of its read, the
        generation read before its rows
    :ptype staged: tuple
    :param below: the version below which each rebuilt table's full claims every delta file folded but its open ones
        (:meth:`DeltaFolds._folded_below`, decision 53); None claims nothing
    :ptype below: int | None
    :param opens: each rebuilt table's open files: told after its generation was read, waiting for the writer to
        confirm them, or saved for a commit not yet ended (:meth:`DeltaFolds._open_files`)
    :ptype opens: tuple
    :param told: each rebuilt table's files told before its generation was read: their commits landed, so the rows read
        hold them, whatever their name
    :ptype told: tuple
    :param kept: delta files never dropped (:meth:`DeltaFolds._kept_files`)
    :ptype kept: tuple[str, ...]
    :param retention: ``"keep"``: no delta file is dropped, the tables' directories never listed (decision 51);
        ``"delete"``: each of the tables' delta files the rebuilt full covers by its name (below ``below``, not open:
        decision 53), or told before its generation was read, that nothing in use names. Any other file of the older,
        unversioned name is never dropped here: no name says its commit ended, and no clock is compared (the volume's
        is not the writer's)
    :ptype retention: DeltaRetention
    :return: ``(outcome, manifest, rebuilt, deleted, full_bytes, covered)``: ``"rebuilt"`` or ``"stale"``, the
        manifest published (or the current one), each table as ``(table, generation, rows)``, the fulls' files
        deleted, each rebuilt table's ``(table, bytes of its parts)`` (none when stale), and, deleting, the old delta
        files it covers, for the owner to delete once every copy is past them (decision 54)
    :rtype: tuple
    """
    base = Path(root)
    config = {plain[0]: _Folding.of(plain) for plain in tables}
    rebuilt = tuple((name, token, rows) for name, _, token, rows in staged)
    open_of = dict(opens)
    told_of = {name: frozenset(files) for name, files in told}
    written: list[str] = []
    try:
        current = current_full_manifest(base)
        if current is not None and current.term > term:
            return ("stale", current.name, rebuilt, 0, (), ())
        tables_now = dict(current.tables) if current is not None else {}
        for name, files, token, _ in staged:
            # its streams' files as one table, mapped, never copied
            rows = pa.concat_tables([_read_rows_file(base / file) for file in files])
            parts = _parts_of_rows(base, config[name], rows, token)
            written.extend(part.file for part in parts)
            tables_now[name] = FullTable(version=token, parts=parts).covered_as(
                folded_below=below, open_files=open_of.get(name, ())
            )
        try:
            published = publish_full_manifest(base, term=term, tables=tables_now)
        except StaleTermError:
            # a later writer published since this one looked: its parts are this writer's alone, named by none
            _unlink_all(base, written)
            stale = current_full_manifest(base)
            return ("stale", stale.name if stale is not None else "", rebuilt, 0, (), ())
    except BaseException:
        _unlink_all(base, written)
        raise
    finally:
        _unlink_all(base, [file for _, files, _, _ in staged for file in files])
    named = published.files() | set(kept)
    if published.previous is not None and (base / ".fulls" / "manifests" / published.previous).is_file():
        named |= FullManifest.decode((base / ".fulls" / "manifests" / published.previous).read_bytes()).files()
    old: list[str] = []
    # kept (decision 51): only the fulls' own old manifests and parts go, never a delta file
    for name in config if retention == "delete" else ():
        directory = base / name
        if not directory.is_dir():
            continue
        entry = published.tables[name]
        for path in directory.iterdir():
            file = f"{name}/{path.name}"
            # covered by its name (decision 53) or told before the read, never by its time on the volume
            if path.is_file() and file not in named and (entry.covers(file) or file in told_of.get(name, ())):
                old.append(file)
    deleted = 0
    # the fulls' own files only: the old delta files are handed back, deleted by the owner once every copy is past them
    # (decision 54)
    for file in deletable_full_files(base):
        try:
            (base / file).unlink(missing_ok=True)
        except OSError as error:  # NOSILENT: logged; the file is listed again by the next round
            log.warning(
                "a file the rebuilt full no longer names could not be deleted",
                extra={"extra_data": {"file": file, "error": repr(error)}},
            )
            continue
        deleted += 1
    return ("rebuilt", published.name, rebuilt, deleted, _full_bytes(base, published, config), tuple(sorted(old)))


def _leftover_delta_files(root: Path, table: str, told: Collection[str]) -> tuple[list[str], int]:
    """every delta file of ``table`` on the volume the current full does not cover (decision 53: its name's version at
    or above the full's claim, or listed open; or of the older, unversioned name, read as before for one release) and
    no full in use names; blocking. The one listing of the volume a writer makes: only a recovery (decision 28), when no
    epoch log names the files. Only names are listed: a covered file is never opened, so the kept history (decision 51)
    costs no read, and a damaged old file never fails the recovery.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: str
    :param told: the table's files the fold holds unfolded (told, or re-learned): read whatever the full claims, as
        they are not folded into it
    :ptype told: Collection[str]
    :return: each file to read, under the root, and how many the full covers (never read)
    :rtype: tuple[list[str], int]
    """
    directory = root / table
    if not directory.is_dir():
        return [], 0
    current = current_full_manifest(root)
    named = current.files() if current is not None else set()
    if current is not None and current.previous is not None:
        previous = root / ".fulls" / "manifests" / current.previous
        if previous.is_file():
            named |= FullManifest.decode(previous.read_bytes()).files()
    entry = current.tables.get(table) if current is not None else None
    found: list[str] = []
    covered = 0
    for path in sorted(directory.iterdir()):
        file = f"{table}/{path.name}"
        # a name starting with a dot is a file being written (``.partial``), never a delta a commit named
        if not path.is_file() or path.suffix != ".arrow" or path.name.startswith(".") or file in named:
            continue
        if entry is not None and entry.covers(file) and file not in told:
            covered += 1
            continue
        found.append(file)
    return found, covered


#: the recovery's reach file's own columns beside the key's (:func:`recovery_reach_job`): each key's one text, its
#: state as a file carried it, whether a cleared prefix holds it, and whether the notes named it
_TEXT: Final = "_3tears_text"
_STATE: Final = "_3tears_state"
_UNDER: Final = "_3tears_under"
_NOTED: Final = "_3tears_noted"


def _typed_texts(column: Any, kind: str, arrow_type: Any) -> Any:
    """a key column given as text (a note's) as its column binds (:func:`~threetears.core.collections.copy_follower.
    typed_text`), each distinct text converted once, then taken a row at a time in Arrow. A column the rail answers as
    text (a ``timestamptz`` that is no ``date_`` column, a ``NUMERIC``, a UUID) stays text: an instant's in its one
    canonical text (:func:`~threetears.core.collections.delta_files._text_column` with ``instant``), anything else as
    given (``key_text``'s spelling already).

    :param column: the texts
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :param kind: the column's type tag
    :ptype kind: str
    :param arrow_type: the column's Arrow type
    :ptype arrow_type: pyarrow.DataType
    :return: the values
    :rtype: pyarrow.Array
    """
    if pa.types.is_string(arrow_type):
        # Why: typed_text made a datetime of a timestamptz key's text (a Decimal of a NUMERIC's) and Arrow refused it
        # as a text column (ArrowTypeError: Expected bytes, got a 'datetime.datetime'), so a recovery of a table keyed
        # by a timestamp (result_snapshots_by_id) failed every round (2026-11-24 replay)
        return _text_column(column, instant=kind == DATETIMETZ_TYPE)
    encoded = pc.dictionary_encode(column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column)
    values = pa.array([typed_text(kind, text) for text in encoded.dictionary.to_pylist()], arrow_type)
    return values.take(encoded.indices)


def _under_prefixes(
    texts: Mapping[str, Any], count: int, prefixes: Sequence[Mapping[str, str]], instants: Collection[str]
) -> Any:
    """which rows lie under one of ``prefixes`` (:func:`~threetears.core.collections.delta_files.under_prefix`,
    compared as text), in Arrow.

    :param texts: each key column's values as text
    :ptype texts: Mapping[str, pyarrow.Array]
    :param count: the rows
    :ptype count: int
    :param prefixes: the prefixes, as a delta file names them
    :ptype prefixes: Sequence[Mapping[str, str]]
    :param instants: the key columns of a ``timestamptz`` column, ``texts`` holding each in its canonical text
    :ptype instants: Collection[str]
    :return: one flag a row
    :rtype: pyarrow.Array
    """
    return pa.array(_under_texts(texts, count, prefixes, instants))


def _first_each(rows: Any, scope: str | None) -> Any:
    """each key of ``rows`` once, in the order first held, by its text (:data:`_TEXT`): its first row's values, and
    the last state a row of it carried (``scope``), as the leftover files' reach was read in file order.

    :param rows: rows holding :data:`_TEXT` and the key columns, and ``scope`` when given
    :ptype rows: pyarrow.Table
    :param scope: the state column a file's markers carry; None for none
    :ptype scope: str | None
    :return: the keys, each once
    :rtype: pyarrow.Table
    """
    if not rows.num_rows:
        return rows
    # by Arrow's hash kernels and NumPy over the codes, never Acero's group_by (:mod:`threetears.core.arrow_keys`)
    (codes,) = key_codes([rows], [_TEXT])
    _, at, each = np.unique(codes, return_index=True, return_inverse=True)
    order = np.argsort(at, kind="stable")
    first = rows.take(pa.array(at[order]))
    if scope is not None:
        # each key's last row carrying a state (a row of none carries no state; a key of none, none)
        carried = np.flatnonzero(rows.column(scope).is_valid().to_numpy(zero_copy_only=False))
        last = np.full(len(at), -1, dtype=np.int64)
        np.maximum.at(last, each[carried], carried)
        picked = last[order]
        states = pc.take(rows.column(scope), pa.array(picked, pa.int64(), mask=picked < 0))
        first = first.set_column(first.schema.get_field_index(scope), scope, states)
    return first


def recovery_reach_job(
    root: str,
    plain: tuple[Any, ...],
    kinds: tuple[tuple[str, str], ...],
    delta_scope: str | None,
    files: tuple[str, ...],
    noted_files: tuple[str, ...],
    reach_file: str,
    page: int,
) -> tuple[tuple[tuple[tuple[str, str], ...], ...], int]:
    """what a recovery reaches (decision 28), in a worker process, as Arrow end to end: the keys and prefixes the
    leftover ``files`` name (their key columns, and the state a table keyed without it carries, read alone; each
    file's checksum verified first) and the keys the lost files' notes name, each key once by its text, written to
    ``reach_file`` (``page`` keys a batch): its key columns as typed, :data:`_TEXT`, :data:`_STATE`, :data:`_UNDER`
    (a cleared prefix holds it: its rows come with the prefix's read) and :data:`_NOTED`.

    :param root: the delta files' root
    :ptype root: str
    :param plain: the table, :meth:`FoldedTable.plain`
    :ptype plain: tuple
    :param kinds: its columns' type tags, :attr:`FoldedTable.kinds`
    :ptype kinds: tuple
    :param delta_scope: the state column its markers carry; None when its key names it
    :ptype delta_scope: str | None
    :param files: its leftover delta files, under the root
    :ptype files: tuple[str, ...]
    :param noted_files: the noted keys' files, under the root: a text column a key column
    :ptype noted_files: tuple[str, ...]
    :param reach_file: where the reach is written, under the root
    :ptype reach_file: str
    :param page: keys a batch of the reach file: one keyed read's worth
    :ptype page: int
    :return: the prefixes the files clear (each its ``(column, text)`` pairs, in order first named), and how many keys
        no prefix holds (read by key)
    :rtype: tuple
    :raises ValueError: when a file is not of the delta files' format
    """
    base = Path(root)
    table = _Folding.of(plain)
    types = _kinds_types(kinds)
    kind_of = dict(kinds)
    read = [*table.key, *((delta_scope,) if delta_scope is not None else ())]
    schema = pa.schema([(column, types[column]) for column in read])
    prefixes: list[dict[str, str]] = []
    held: list[Any] = []
    for file in files:
        # its checksum verified before a key is read: a damaged file raises, never names other keys
        rows = read_volume_file(base / file, columns=read)
        rows.validate(full=True)
        metadata = rows.schema.metadata or {}
        if metadata.get(_FORMAT_METADATA) != _DELTA_FORMAT:
            raise ValueError(f"{file}: a delta file of format {metadata.get(_FORMAT_METADATA)!r}")
        for prefix in json.loads(metadata[_PREFIXES_METADATA]):
            if dict(prefix) not in prefixes:
                prefixes.append(dict(prefix))
        held.append(rows.select(read).replace_schema_metadata(None).cast(schema))
    from_files = pa.concat_tables(held) if held else schema.empty_table()
    # every key text in its one canonical spelling (key_text, an instant's with ``instant``): a file's ``...Z`` and a
    # note's ``...+00:00`` of one instant are one key, as the writer's records and the copy name it
    instants = table.instants
    file_texts = _key_texts(
        [_text_column(from_files.column(column), instant=column in instants) for column in table.key]
    )
    reach = _first_each(from_files.append_column(_TEXT, file_texts), delta_scope)
    if delta_scope is not None:
        reach = reach.rename_columns([_STATE if name == delta_scope else name for name in reach.column_names])
    else:
        reach = reach.append_column(_STATE, pa.nulls(reach.num_rows, pa.string()))
    noted_parts = [_read_rows_file(base / file) for file in noted_files]
    noted_texts = pa.concat_tables(noted_parts) if noted_parts else None
    if noted_texts is not None and noted_texts.num_rows:
        typed = pa.table(
            {column: _typed_texts(noted_texts.column(column), kind_of[column], types[column]) for column in table.key}
        )
        texts = _key_texts([_text_column(typed.column(column), instant=column in instants) for column in table.key])
        noted = _first_each(typed.append_column(_TEXT, texts), None)
        named = pc.is_in(reach.column(_TEXT), value_set=noted.column(_TEXT))
        reach = reach.append_column(_NOTED, pc.fill_null(named, False))
        extra = noted.filter(pc.invert(pc.is_in(noted.column(_TEXT), value_set=reach.column(_TEXT))))
        extra = extra.append_column(_STATE, pa.nulls(extra.num_rows, reach.schema.field(_STATE).type))
        extra = extra.append_column(_NOTED, pa.repeat(True, extra.num_rows))
        reach = pa.concat_tables([reach, extra.select(reach.column_names)])
    else:
        reach = reach.append_column(_NOTED, pa.repeat(False, reach.num_rows))
    key_texts = {column: _text_column(reach.column(column), instant=column in instants) for column in table.key}
    reach = reach.append_column(_UNDER, _under_prefixes(key_texts, reach.num_rows, prefixes, instants))
    path = base / reach_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink, pa_ipc.new_file(sink, reach.schema) as writer:
        # one batch a page of keys: a keyed read's worth each (the reach's pieces joined first)
        for batch in reach.combine_chunks().to_batches(max_chunksize=max(page, 1)):
            writer.write_batch(batch)
    keyed = int(pc.sum(pc.invert(reach.column(_UNDER))).as_py() or 0) if reach.num_rows else 0
    return tuple(tuple(prefix.items()) for prefix in prefixes), keyed


def recovery_unstated_job(
    root: str, key: tuple[str, ...], kinds: tuple[tuple[str, str], ...], reach_file: str, pages_file: str
) -> bool:
    """whether a key the notes named is gone from L3 (no row of it among the rows read) and no leftover file names its
    state (decision 31: its delete marker could be filed under no state's part), in a worker process, in Arrow.

    :param root: the delta files' root
    :ptype root: str
    :param key: the table's key columns
    :ptype key: tuple[str, ...]
    :param kinds: its columns' type tags
    :ptype kinds: tuple
    :param reach_file: the reach (:func:`recovery_reach_job`), under the root
    :ptype reach_file: str
    :param pages_file: the rows read from L3 by key and prefix, under the root
    :ptype pages_file: str
    :return: whether one is
    :rtype: bool
    """
    base = Path(root)
    reach = _read_rows_file(base / reach_file)
    rows = _read_rows_file(base / pages_file)
    instants = instant_key_columns(key, kinds)
    found = _key_texts([_text_column(rows.column(column), instant=column in instants) for column in key])
    gone = pc.invert(pc.is_in(reach.column(_TEXT), value_set=found))
    unstated = pc.and_(pc.and_(reach.column(_NOTED), gone), pc.is_null(reach.column(_STATE)))
    return bool(pc.any(unstated).as_py())


def _prefixes_by_width(key: Sequence[str], prefixes: Sequence[Mapping[str, str]]) -> dict[int, list[list[str]]]:
    """prefixes grouped by how many of the key's leading columns they name (a delta file's prefixes lead the key,
    :meth:`~threetears.core.collections.delta_files.TableDelta.clears`), column by column as a
    :class:`~threetears.core.keyed.KeyBatch` takes them: one keyed read a width. A recovery's few cleared prefixes (a
    leftover file's clears), function-local.

    :param key: the table's key columns, in key order
    :ptype key: Sequence[str]
    :param prefixes: the prefixes, as text
    :ptype prefixes: Sequence[Mapping[str, str]]
    :return: width -> one list a leading key column, each prefix's value at the same place in each
    :rtype: dict[int, list[list[str]]]
    """
    grouped: dict[int, list[list[str]]] = {}
    for prefix in prefixes:
        width = len(prefix)
        columns = grouped.setdefault(width, [[] for _ in range(width)])
        for place, column in enumerate(key[:width]):
            columns[place].append(prefix[column])
    return grouped


def recovery_delta_job(
    root: str,
    plain: tuple[Any, ...],
    kinds: tuple[tuple[str, str], ...],
    delta_scope: str | None,
    reach_file: str,
    pages_files: tuple[str, ...],
    prefixes: tuple[tuple[tuple[str, str], ...], ...],
    replaced: tuple[str, ...],
    unstated: bool,
    term: int,
) -> tuple[str, int, int]:
    """save the recovery delta (decision 28), in a worker process, in Arrow: the prefixes the leftover files cleared,
    then every row L3 holds under them or of their keys (every column set, so a held row takes L3's whole), then a
    delete marker for each key reached (by key) that L3 no longer holds, carrying its state.

    :param root: the delta files' root
    :ptype root: str
    :param plain: the table, :meth:`FoldedTable.plain`
    :ptype plain: tuple
    :param kinds: its columns' type tags
    :ptype kinds: tuple
    :param delta_scope: the state column its markers carry; None when its key names it
    :ptype delta_scope: str | None
    :param reach_file: the reach (:func:`recovery_reach_job`), under the root
    :ptype reach_file: str
    :param pages_files: the rows read from L3 (by key and prefix; the replaced states'), under the root
    :ptype pages_files: tuple[str, ...]
    :param prefixes: the prefixes the leftover files cleared
    :ptype prefixes: tuple
    :param replaced: the states whose every row the pages hold (decision 31): their parts are replaced whole, so a key
        of one L3 no longer holds needs no marker
    :ptype replaced: tuple[str, ...]
    :param unstated: a key gone of no known state is in a replaced state: no marker
    :ptype unstated: bool
    :param term: the writer's term, naming the file (:func:`~threetears.core.collections.delta_files.
        versioned_write_id`, decision 53)
    :ptype term: int
    :return: the file, under the root; the rows it holds; the delete markers
    :rtype: tuple[str, int, int]
    :raises ValueError: when a key L3 no longer holds carries no state, for a table keyed without its state: its
        delete marker could be filed under no state's part
    """
    base = Path(root)
    table = _Folding.of(plain)
    types = _kinds_types(kinds)
    schema = pa.schema(list(types.items()))
    read = [_read_rows_file(base / file).select(list(types)).cast(schema) for file in pages_files]
    rows = pa.concat_tables(read) if read else schema.empty_table()
    reach = _read_rows_file(base / reach_file)
    found = _key_texts([_text_column(rows.column(column), instant=column in table.instants) for column in table.key])
    gone = pc.and_(pc.invert(reach.column(_UNDER)), pc.invert(pc.is_in(reach.column(_TEXT), value_set=found)))
    state = reach.column(_STATE)
    if replaced and table.scope_column is not None:
        known = state if delta_scope is not None else reach.column(table.scope_column)
        scope_text = pc.binary_join_element_wise(pa.scalar(table.scope_prefix or ""), pc.cast(known, pa.string()), "")
        # its state's part is replaced whole by L3's rows: gone from them already
        gone = pc.and_(gone, pc.invert(pc.fill_null(pc.is_in(scope_text, value_set=pa.array(replaced)), False)))
    if delta_scope is not None:
        stateless = pc.and_(gone, pc.is_null(state))
        if unstated:
            # a noted key of no known state: its commit's states are replaced whole, so it is gone with them
            gone = pc.and_(gone, pc.invert(stateless))
        elif pc.any(stateless).as_py():
            first = reach.filter(stateless).slice(0, 1).column(_TEXT)[0].as_py()
            raise ValueError(
                f"{table.name}: key {first!r} is gone from L3 and no leftover file names its state: its delete marker "
                f"could be filed under no state's part"
            )
    markers = reach.filter(gone)
    count = markers.num_rows
    columns = []
    for name, kind in types.items():
        if name in table.key:
            columns.append(markers.column(name).cast(kind))
        elif delta_scope is not None and name == delta_scope:
            columns.append(markers.column(_STATE).cast(kind))
        else:
            columns.append(pa.nulls(count, kind))
    marked = pa.Table.from_arrays(columns, schema=schema)
    segments: list[tuple[Any, ...]] = []
    if rows.num_rows:
        segments.append(("typed", rows, list(types)))
    segments.append(("deleted", marked))
    write_id = versioned_write_id(term, f"{_RECOVERY_WRITE}-{uuid.uuid7().hex}")
    built = build_file_table(
        table.name, types, table.key, {}, [dict(prefix) for prefix in prefixes], segments, write_id
    )
    file = write_table_file(base, table.name, write_id, built)
    return file, int(rows.num_rows), int(count)


async def _reach_pages(path: Path, key: Sequence[str]) -> AsyncIterator[Any]:
    """the reach's keys no cleared prefix holds, a batch of the reach file at a time, as Arrow key columns: one page
    of keys at a time, let go once read.

    :param path: the reach file
    :ptype path: Path
    :param key: the key columns
    :ptype key: Sequence[str]
    :return: each page's keys, a column each
    :rtype: AsyncIterator[pyarrow.RecordBatch]
    """
    reader = await asyncio.to_thread(_opened_ipc, path)
    for index in range(reader.num_record_batches):
        batch = await asyncio.to_thread(reader.get_batch, index)
        keyed = batch.filter(pc.invert(batch.column(_UNDER)))
        if keyed.num_rows:
            yield keyed.select(list(key))


def _opened_ipc(path: Path) -> Any:
    """an Arrow IPC file opened over its memory map, its batches read one at a time. Blocking.

    :param path: the file
    :ptype path: Path
    :return: its reader
    :rtype: pyarrow.ipc.RecordBatchFileReader
    """
    return pa_ipc.open_file(pa.memory_map(str(path)))


def _unlink_work(root: Path, work: str) -> None:
    """remove a recovery's own files (its stem's): its reach, noted keys and rows read. Blocking.

    :param root: the delta files' root
    :ptype root: Path
    :param work: their stem, under the root
    :ptype work: str
    :return: nothing
    :rtype: None
    """
    directory = (root / work).parent
    stem = (root / work).name
    if not directory.is_dir():
        return
    for path in directory.glob(f"{stem}.*"):
        path.unlink(missing_ok=True)


def recovery_round(
    root: str,
    term: int,
    plain: tuple[Any, ...],
    file: str,
    token: str,
    doomable: tuple[str, ...],
    replaced: tuple[str, ...],
    retention: DeltaRetention,
    below: int | None,
    open_files: tuple[str, ...],
) -> tuple[Any, ...]:
    """a recovery's fold, in a worker process (decision 28): the recovery delta folded into every state it reaches
    (its rows' and markers' states, and those its cleared prefixes reach), every other part kept as it is; every part
    current to ``token``, the generation read before the keys; the manifest published at the writer's term, then the
    fulls' old files deleted, and, when ``retention`` is ``"delete"``, the leftover files it replaces and the recovery
    delta itself (unless a manifest in use names one).

    The kept parts may be of an older incarnation of the table's generation (its log replaced): the full is right up
    to its version and the recovery holds every change since, so the whole table is current to ``token``.

    Plain data in and out (:mod:`~threetears.core.process_workers`): names, versions and counts, never rows.

    :param root: the delta files' root
    :ptype root: str
    :param term: the writer's term
    :ptype term: int
    :param plain: the table, :meth:`FoldedTable.plain`
    :ptype plain: tuple
    :param file: the recovery delta file, under the root
    :ptype file: str
    :param token: the generation every part is current to after it
    :ptype token: str
    :param doomable: the leftover files the recovery replaces, deleted once it is published (unless a manifest in use
        names one)
    :ptype doomable: tuple[str, ...]
    :param replaced: the states whose every row L3 holds the recovery delta holds (decision 31): each one's part is
        those rows alone, the held part dropped; a state L3 holds no row of has no part
    :ptype replaced: tuple[str, ...]
    :param retention: ``"keep"``: no delta file is deleted, the leftovers, a lost file and the recovery delta kept
        (decision 51); ``"delete"``: they go once the recovery is published
    :ptype retention: DeltaRetention
    :param below: the version below which the recovered full claims every delta file folded but ``open_files``
        (decision 53); None claims nothing
    :ptype below: int | None
    :param open_files: the table's files the recovery does not hold (told after its generation was read, waiting to
        be confirmed, saved for a commit not yet ended)
    :ptype open_files: tuple[str, ...]
    :return: ``(outcome, manifest, states, deleted, full_bytes, covered)``: ``"recovered"``, ``"stale"`` or
        ``"no_full"``; the manifest published (or the current one); the states folded; the fulls' files deleted; the
        table's ``(table, bytes of its parts)`` once recovered; and, deleting, the delta files it covers, for the owner
        to delete once every copy is past them (decision 54)
    :rtype: tuple
    """
    base = Path(root)
    table = _Folding.of(plain)
    current = current_full_manifest(base)
    if current is None or table.name not in current.tables:
        return ("no_full", current.name if current is not None else "", 0, 0, (), ())
    if current.term > term:
        return ("stale", current.name, 0, 0, (), ())
    entry = current.tables[table.name]
    sizes = {part.scope: (base / part.file).stat().st_size for part in entry.parts}
    by_scope: dict[str | None, list[_Share]] = {}
    for share in _shares(base, table, file, token, sizes):
        by_scope.setdefault(share[1], []).append(share)
    parts: list[FullPart] = []
    written: list[str] = []
    for scope, shares in sorted(by_scope.items(), key=lambda item: str(item[0])):
        held = None if scope in replaced else next((part for part in entry.parts if part.scope == scope), None)
        new_part, how = _fold_state(base, table, scope, held, shares, across=True)
        if how == "written":
            written.append(new_part.file)
        parts.append(new_part)
    # the parts the recovery does not reach hold no change since their version: current to the recovery's, their
    # rows and so their fingerprint as they were (decision 32: a copy still compares them)
    parts.extend(
        FullPart(scope=part.scope, file=part.file, version=token, rows=part.rows, fingerprint=part.fingerprint)
        for part in entry.parts
        if part.scope not in by_scope and part.scope not in replaced
    )
    tables_now = dict(current.tables)
    tables_now[table.name] = FullTable(version=token, parts=tuple(parts)).covered_as(
        folded_below=below, open_files=open_files
    )
    try:
        published = publish_full_manifest(base, term=term, tables=tables_now)
    except StaleTermError:
        _unlink_all(base, written)
        stale = current_full_manifest(base)
        return ("stale", stale.name if stale is not None else "", 0, 0, (), ())
    deleted = 0
    # the fulls' own files only: kept (decision 51) no delta file goes; deleting, the leftovers it covers and its own
    # delta are handed back, deleted by the owner once every copy is past them (decision 54)
    covered = tuple(sorted({*doomable, file})) if retention == "delete" else ()
    for doomed in deletable_full_files(base):
        try:
            (base / doomed).unlink(missing_ok=True)
        except OSError as error:  # NOSILENT: logged; a later round lists it again
            log.warning(
                "a file the recovered full replaces could not be deleted",
                extra={"extra_data": {"file": doomed, "error": repr(error)}},
            )
            continue
        deleted += 1
    return ("recovered", published.name, len(by_scope), deleted, _full_bytes(base, published, (table.name,)), covered)


def _parts_of_rows(root: Path, table: _Folding, rows: Any, token: str) -> tuple[FullPart, ...]:
    """a table's parts from its rows read whole: one a state (one for a table with no scope), each kept by the
    table's keep rule; a state the rule keeps no row of has no part.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: _Folding
    :param rows: every row L3 held
    :ptype rows: pyarrow.Table
    :param token: the generation the rows are current to
    :ptype token: str
    :return: the parts, written
    :rtype: tuple[FullPart, ...]
    """
    rows = rows.select(list(table.columns)).replace_schema_metadata(None)
    states: list[tuple[str | None, Any]] = []
    if table.scope_column is None:
        states.append((None, rows))
    else:
        values = rows.column(table.scope_column)
        # the values grouped by the state they name (each value's text, under the scope's prefix): one part a state
        by_state: dict[str, list[Any]] = {}
        for value in pc.unique(values).to_pylist():
            by_state.setdefault(table.scope_of(value), []).append(value)
        for state, named in by_state.items():
            states.append((state, rows.filter(pc.is_in(values, value_set=pa.array(named, values.type)))))
    parts: list[FullPart] = []
    for scope, held in sorted(states, key=lambda state: str(state[0])):
        kept = table.keep.keep(held) if table.keep is not None else held
        if kept.num_rows:
            parts.append(write_full_part(root, table.name, scope, kept.select(list(table.columns)), token))
    return tuple(parts)


def fold_round(
    root: str,
    term: int,
    tables: tuple[tuple[Any, ...], ...],
    committed: tuple[tuple[str, str, str], ...],
    pending: tuple[_Share, ...],
    reused: tuple[str, ...],
    doomed: tuple[str, ...],
    orphans: tuple[str, ...],
    fresh: tuple[str, ...],
    retention: DeltaRetention,
    below: int | None,
    outside: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[Any, ...]:
    """one round of folds, in a worker process: note the commits' files, fold each state due, publish, delete the
    fulls' old files, and the covered delta files when ``retention`` is ``"delete"``.

    Plain data in and out (:mod:`~threetears.core.process_workers`): names, versions and counts, never rows.

    :param root: the delta files' root
    :ptype root: str
    :param term: the writer's term
    :ptype term: int
    :param tables: each folded table, :meth:`FoldedTable.plain`
    :ptype tables: tuple
    :param committed: each new commit's (table, delta file, generation token), in commit order
    :ptype committed: tuple
    :param pending: each (delta file, state) share not folded yet, from the round before
    :ptype pending: tuple
    :param reused: the delta files named as parts, kept until no manifest in use names them
    :ptype reused: tuple[str, ...]
    :param doomed: covered delta files a round before could not delete
    :ptype doomed: tuple[str, ...]
    :param orphans: the files the writer saved whose commit did not land (:data:`~threetears.core.collections.
        delta_files.OWN_DELTA_SAVES`): swept by name unless a record or a full names them, nothing else
    :ptype orphans: tuple[str, ...]
    :param fresh: tables owed a rebuild (reset or recreated): their files are kept, never folded onto the old parts
    :ptype fresh: tuple[str, ...]
    :param retention: ``"keep"``: no delta file is deleted (decision 51), so none is remembered as reused or doomed
        and an orphan's temporary file alone is swept; ``"delete"``: covered deltas go after the publish
    :ptype retention: DeltaRetention
    :param below: the version below which the full claims every delta file folded but those open
        (:meth:`DeltaFolds._folded_below`, decision 53); None claims nothing
    :ptype below: int | None
    :param outside: each table's open files no round carries (:meth:`DeltaFolds._outside_round`); the shares left
        unfolded after this round are open too
    :ptype outside: tuple
    :return: ``(pending, reused, doomed, outcome, folded, deleted, swept, manifest, pid, renewed, full_bytes)``: the
        state for the next round, the round's outcome (:data:`_ROUNDS`), each state folded as ``(table, scope, how,
        rows)``, the files deleted, the orphans swept (by name), the manifest published (or the current one), this
        worker's process id, each table a commit of another incarnation than its full's was told of (its log lost:
        rebuilt, never folded), and each committed or folded table's ``(table, bytes of its current full's parts)``
        from the sizes the round already took (:data:`_VOLUME_BYTES`)
    :rtype: tuple
    """
    base = Path(root)
    config = {plain[0]: _Folding.of(plain) for plain in tables}
    current = current_full_manifest(base)
    shares = list(pending)
    # each committed table's full's bytes, from the part sizes its shares are measured against: the volume gauge's
    full_bytes: dict[str, int] = {}
    for table, file, token in committed:
        # a cleared prefix reaches the states the full holds of the table, each counted at its part's size
        held = current.tables.get(table) if current is not None else None
        sizes = {part.scope: (base / part.file).stat().st_size for part in held.parts} if held is not None else {}
        if held is not None:
            full_bytes[table] = sum(sizes.values())
        shares.extend(_shares(base, config[table], file, token, sizes))
    renewed: tuple[str, ...] = ()

    def answer(
        outcome: str, manifest: str, *, folded: Sequence[Any] = (), deleted: int = 0, swept: Sequence[str] = ()
    ) -> tuple[Any, ...]:
        return (
            tuple(shares),
            reused,
            doomed,
            outcome,
            tuple(folded),
            deleted,
            tuple(swept),
            manifest,
            os.getpid(),
            renewed,
            tuple(sorted(full_bytes.items())),
        )

    if current is None:
        return answer("no_full", "")
    if current.term > term:
        return answer("stale", current.name)
    # a share a part is already current to was folded before the writer restarted (its files re-learned from the
    # epoch log, decision 24): dropped, and a file left with no share deleted with the round's covered files
    named = {share[2] for share in shares}
    shares = [share for share in shares if not _folded_in(current, share)]
    deletes = retention == "delete"
    # kept (decision 51): nothing is remembered to delete, so reused and doomed stay empty, never growing a round
    doomed = tuple(sorted(set(doomed) | (named - {share[2] for share in shares} - current.files()))) if deletes else ()
    renewed = tuple(
        sorted(
            {
                share[0]
                for share in shares
                if share[0] in current.tables
                and share[0] not in fresh
                and not _one_incarnation(current.tables[share[0]].version, share[3])
            }
        )
    )
    # a delta a full names as a part is kept until no manifest in use names it, then deleted: a writer that started
    # after the publish that named it learns it here, from the manifest
    reused = (
        tuple(sorted(set(reused) | {file for file in current.files() if not file.startswith(_FULLS_PREFIX)}))
        if deletes
        else ()
    )
    by_state: dict[tuple[str, str | None], list[_Share]] = {}
    for share in shares:
        if share[0] in current.tables and share[0] not in fresh and share[0] not in renewed:
            by_state.setdefault((share[0], share[1]), []).append(share)
    tables_now = dict(current.tables)
    outside_of = dict(outside)
    folded: list[tuple[str, str | None, str, int]] = []
    folded_shares: set[_Share] = set()
    named_parts: list[str] = []
    for (table, scope), state_shares in by_state.items():
        entry = tables_now[table]
        part = next((p for p in entry.parts if p.scope == scope), None)
        held_bytes = (base / part.file).stat().st_size if part is not None else 0
        if sum(share[4] for share in state_shares) * _DUE_OF < held_bytes * _DUE_AT:
            continue
        new_part, how = _fold_state(base, config[table], scope, part, state_shares)
        if how == "reused":
            named_parts.append(new_part.file)
        tables_now[table] = FullTable(
            version=_newest(table, [entry.version, new_part.version]),
            parts=tuple(p for p in entry.parts if p.scope != scope) + (new_part,),
        )
        folded.append((table, scope, how, new_part.rows))
        folded_shares.update(state_shares)
    if not folded:
        # an orphan is no file a share, a part or the full names: swept by name, as after a publish
        unnamed = {share[2] for share in shares} | set(reused) | set(doomed) | current.files()
        return answer(
            "nothing_due",
            current.name,
            swept=sweep_orphan_delta_files(base, orphans=orphans, named=unnamed, retention=retention),
        )
    unfolded = [share for share in shares if share not in folded_shares]
    # each table's coverage (decision 53): every file below ``below`` folded but the shares left and the files no round
    # carries. A table owed a rebuild or a recovery keeps the coverage it had: its files are not folded, and its
    # recovery reads by that coverage
    for name, entry in list(tables_now.items()):
        if name in config and name not in fresh and name not in renewed:
            tables_now[name] = entry.covered_as(
                folded_below=below,
                open_files={*outside_of.get(name, ()), *(share[2] for share in unfolded if share[0] == name)},
            )
    try:
        published = publish_full_manifest(base, term=term, tables=tables_now)
    except StaleTermError:
        # another writer published a later term since this round read the current full: its parts are orphans
        # the current writer's next round deletes
        return answer("stale", current.name)
    shares = unfolded
    left = {share[2] for share in shares}
    covered = {share[2] for share in folded_shares} - left
    kept_reused = set(reused) | set(named_parts) if deletes else set()
    # only the fulls' own old manifests and parts go here: a delta file never (kept, decision 51; deleting, the owner
    # deletes each covered one once every copy is past it, reading the copies' beats right before, decision 54)
    listed = deletable_full_files(base)
    deleted: set[str] = set()
    for file in listed:
        try:
            (base / file).unlink(missing_ok=True)
        except OSError as error:  # NOSILENT: logged; the file is listed again by the next round
            log.warning(
                "a file the full no longer names could not be deleted",
                extra={"extra_data": {"file": file, "error": repr(error)}},
            )
            continue
        deleted.add(file)
    reused = tuple(sorted(kept_reused - deleted))
    doomed = tuple(sorted((covered | set(doomed)) - deleted - kept_reused)) if deletes else ()
    named = left | set(reused) | set(doomed) | published.files()
    swept = sweep_orphan_delta_files(base, orphans=orphans, named=named, retention=retention)
    # the folded tables' new fulls: the parts it wrote and kept, sized once each here in the worker
    full_bytes.update(_full_bytes(base, published, {table for table, _, _, _ in folded}))
    return answer("folded", published.name, folded=folded, deleted=len(deleted), swept=swept)


def _full_bytes(base: Path, manifest: FullManifest, tables: Collection[str]) -> tuple[tuple[str, int], ...]:
    """each of ``tables``' bytes in ``manifest``: the sizes of the parts it names, in a worker process (never the loop).
    A part already gone (a later round's delete) counts as nothing.

    :param base: the delta files' root
    :ptype base: Path
    :param manifest: the full
    :ptype manifest: FullManifest
    :param tables: the tables to size
    :ptype tables: Collection[str]
    :return: each table held as ``(table, bytes)``
    :rtype: tuple[tuple[str, int], ...]
    """
    sized: list[tuple[str, int]] = []
    for table in sorted(tables):
        entry = manifest.tables.get(table)
        if entry is None:
            continue
        total = 0
        for part in entry.parts:
            try:
                total += (base / part.file).stat().st_size
            except FileNotFoundError:  # NOSILENT: a part a later round deleted holds no bytes on the volume
                continue
        sized.append((table, total))
    return tuple(sized)


def _one_incarnation(token: str, other: str) -> bool:
    """whether two generation tokens are of one incarnation of the table's log, so their counts order them.

    :param token: one token
    :ptype token: str
    :param other: the other
    :ptype other: str
    :return: whether both parse and their incarnations match
    :rtype: bool
    """
    first, second = split_generation_token(token), split_generation_token(other)
    return first is not None and second is not None and first[0] == second[0]


def _folded_in(current: FullManifest, share: _Share) -> bool:
    """whether the current full already holds a share: its state's part (or, for a state with no part, the table,
    which then holds no row of the state) is current to the share's generation or a later one of its incarnation.

    :param current: the current full
    :ptype current: FullManifest
    :param share: the share
    :ptype share: _Share
    :return: whether it was folded before
    :rtype: bool
    """
    entry = current.tables.get(share[0])
    if entry is None:
        return False
    part = next((p for p in entry.parts if p.scope == share[1]), None)
    held = split_generation_token(part.version if part is not None else entry.version)
    mine = split_generation_token(share[3])
    return held is not None and mine is not None and held[0] == mine[0] and mine[1] <= held[1]


@dataclass(frozen=True, slots=True)
class _Folding:
    """a folded table as a round reads it from :meth:`FoldedTable.plain`."""

    name: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    scope_column: str | None
    scope_prefix: str | None
    keep: type[DeltaKeepRule] | None
    instants: frozenset[str]

    @classmethod
    def of(cls, plain: tuple[Any, ...]) -> _Folding:
        """the folding from :meth:`FoldedTable.plain`'s tuple.

        :param plain: the tuple
        :ptype plain: tuple
        :return: the folding
        :rtype: _Folding
        """
        name, key, columns, column, prefix, keep, instants = plain
        return cls(name, tuple(key), tuple(columns), column, prefix, keep, frozenset(instants))

    def scope_of(self, value: Any) -> str:
        """a row's state from its scope column's value (a delete marker's included: the key's, or the state the
        table's markers carry), as the writer's seqlock names it."""
        return f"{self.scope_prefix}{value}"


def _read_ipc(path: Path) -> Any:
    """a volume file's table (a delta file, a part), read whole, its checksum verified first.

    :param path: the file
    :ptype path: Path
    :return: its table
    :rtype: pyarrow.Table
    :raises DeltaFileUnreadableError: when its bytes do not match its checksum
    """
    return read_volume_file(path)


def _read_rows_file(path: Path) -> Any:
    """a rebuild's rows file (:class:`_RowsFile`), read whole: written moments before by this writer beside its
    rebuild, never named by a record or a manifest, so it carries no checksum (its rows are hashed as the parts they
    become).

    :param path: the file
    :ptype path: Path
    :return: its table
    :rtype: pyarrow.Table
    """
    with pa.memory_map(str(path)) as source:
        return pa_ipc.open_file(source).read_all()


def _shares(
    root: Path, table: _Folding, file: str, token: str, sizes: Mapping[str | None, int] | None = None
) -> list[_Share]:
    """a delta file's bytes shared among the states its rows belong to, by rows, and every state a prefix it clears
    reaches (decision 23): the prefix's state when it names the scope column, else every state the full holds.

    A cleared state's share is its part's size: the clear may take every row of the part out in a file of a few
    hundred bytes, so the state folds at once (decision 10's half is reached) rather than keeping a part of rows L3
    no longer holds until other writes add up.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the file's table
    :ptype table: _Folding
    :param file: the file, under the root
    :ptype file: str
    :param token: the generation its commit wrote
    :ptype token: str
    :param sizes: each state the current full holds of the table, and its part's bytes
    :ptype sizes: Mapping[str | None, int] | None
    :return: one share a state
    :rtype: list
    """
    size = (root / file).stat().st_size
    if table.scope_column is None:
        return [(table.name, None, file, token, max(1, size))]
    rows = _read_ipc(root / file)
    # each distinct value's rows (a delta's few states), counted in Arrow
    counts: Counter[str] = Counter()
    for counted in pc.value_counts(rows.column(table.scope_column)).to_pylist():
        counts[table.scope_of(counted["values"])] += counted["counts"]
    total = max(1, rows.num_rows)
    shared = {scope: max(1, size * count // total) for scope, count in counts.items()}
    held = dict(sizes or {})
    for prefix in _cleared_prefixes(rows):
        reached = [table.scope_of(prefix[table.scope_column])] if table.scope_column in prefix else list(held)
        for scope in reached:
            if scope is not None:
                shared[scope] = max(shared.get(scope, 1), held.get(scope, 1))
    return [(table.name, scope, file, token, share) for scope, share in shared.items()]


def _cleared_prefixes(delta: Any) -> list[dict[str, str]]:
    """the key prefixes a delta file clears before its rows, as it names them (decision 23).

    :param delta: the file's table, its schema's metadata read with it
    :ptype delta: pyarrow.Table
    :return: the prefixes
    :rtype: list[dict[str, str]]
    """
    named = (delta.schema.metadata or {}).get(_PREFIXES_METADATA)
    return [dict(prefix) for prefix in json.loads(named)] if named else []


def _held_rows(path: Path, columns: Sequence[str]) -> Any:
    """a part's rows: its table's columns, a delta file named as the part read without its delete markers.

    :param path: the part's file
    :ptype path: Path
    :param columns: the table's columns
    :ptype columns: Sequence[str]
    :return: its rows
    :rtype: pyarrow.Table
    """
    rows = _read_ipc(path)
    if _DELETED_COLUMN in rows.column_names:
        rows = rows.filter(pc.invert(pc.fill_null(rows.column(_DELETED_COLUMN), False)))
    return rows.select(list(columns)).replace_schema_metadata(None)


def _fold_state(
    root: Path,
    table: _Folding,
    scope: str | None,
    part: FullPart | None,
    shares: Sequence[_Share],
    *,
    across: bool = False,
) -> tuple[FullPart, str]:
    """one state's new part: the old part, each delta's rows of the state applied in order, then the keep rule;
    the last delta named as the part when it is exactly what the fold keeps (decision 16), else a new part file.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: _Folding
    :param scope: the state; None for a table folded whole
    :ptype scope: str | None
    :param part: the state's part in the current full; None when it has none
    :ptype part: FullPart | None
    :param shares: the state's shares since its last fold, in commit order
    :ptype shares: Sequence
    :param across: a recovery's fold (:func:`recovery_round`): the delta may be of a newer incarnation than the part
        (the log replaced), which it brings current to L3
    :ptype across: bool
    :return: the new part, and ``"reused"`` or ``"written"``
    :rtype: tuple[FullPart, str]
    :raises ValueError: when a delta's generation is of another incarnation than the part's (a reset: the rebuild's)
    """
    schema: Any = None
    deltas: list[DeltaTable] = []
    last: Any = None
    last_only_state = False
    for index, (_, _, file, _, _) in enumerate(shares):
        delta = _read_ipc(root / file)
        if schema is None:
            schema = pa.schema([delta.schema.field(name).with_nullable(True) for name in table.columns])
        mine = delta if scope is None else _of_state(table, scope, delta)
        if index == len(shares) - 1:
            last_only_state = mine.num_rows == delta.num_rows
            last = mine.filter(pc.invert(pc.fill_null(mine.column(_DELETED_COLUMN), False)))
        deltas.append(
            DeltaTable(
                table=table.name,
                write=file,
                key=table.key,
                rows=mine.select(list(table.columns)),
                deleted=mine.column(_DELETED_COLUMN),
                written=mine.column(_WRITTEN_COLUMN),
                sets=tuple(frozenset(named) for named in json.loads(delta.schema.metadata[_WRITTEN_METADATA])),
                prefixes=tuple(_cleared_prefixes(delta)),
            )
        )
    held = schema.empty_table() if part is None else _held_rows(root / part.file, table.columns).cast(schema)
    version = shares[-1][3]
    if part is not None and not across:
        # refused across a reset: the part and the delta must be of one incarnation
        _newest(table.name, [part.version, version])
    folded = fold_delta_tables(
        table.key, held, deltas, {field.name: field.type for field in schema}, instants=table.instants
    )
    if table.keep is not None:
        folded = table.keep.keep(folded)
    if last_only_state and last is not None and _same_rows(table, schema, folded, last):
        last_file = shares[-1][2]
        # the delta named as the part is fingerprinted as a written part is (decision 32): by its rows, here
        fingerprint = rows_fingerprint(folded.select(list(table.columns)))
        return FullPart(
            scope=scope, file=last_file, version=version, rows=last.num_rows, fingerprint=fingerprint
        ), "reused"
    return write_full_part(root, table.name, scope, folded.select(list(table.columns)), version), "written"


def _of_state(table: _Folding, scope: str, rows: Any) -> Any:
    """the rows of one state: those whose scope column's value names it (:meth:`_Folding.scope_of`), judged once a
    distinct value (a delta's few states), the rows filtered in Arrow.

    :param table: the table, which has a scope column
    :ptype table: _Folding
    :param scope: the state
    :ptype scope: str
    :param rows: a delta's rows
    :ptype rows: pyarrow.Table
    :return: the state's rows
    :rtype: pyarrow.Table
    """
    assert table.scope_column is not None  # noqa: S101 -- a scoped share is a scoped table's
    values = rows.column(table.scope_column)
    distinct = pc.unique(values)
    named = [value for value in distinct.to_pylist() if table.scope_of(value) == scope]
    return rows.filter(pc.is_in(values, value_set=pa.array(named, distinct.type)))


def _same_rows(table: _Folding, schema: Any, folded: Any, rows: Any) -> bool:
    """whether the folded rows are exactly ``rows`` (a delta's rows written), one a key: compared in Arrow, each side in
    its keys' order.

    :param table: the table
    :ptype table: _Folding
    :param schema: the part's schema
    :ptype schema: pyarrow.Schema
    :param folded: the rows a part would hold
    :ptype folded: pyarrow.Table
    :param rows: the last delta's rows written
    :ptype rows: pyarrow.Table
    :return: whether they are the same rows
    :rtype: bool
    """
    if folded.num_rows != rows.num_rows:
        return False
    written = joined_key_texts(key_text_table(rows, table.key), table.key)
    if pc.count_distinct(written).as_py() != rows.num_rows:
        # a key written twice in one commit: the file holds both rows, a part may hold only one
        return False
    held = joined_key_texts(key_text_table(folded, table.key), table.key)
    columns = list(table.columns)
    ours = folded.select(columns).cast(schema).take(pc.sort_indices(held)).combine_chunks()
    theirs = rows.select(columns).cast(schema).take(pc.sort_indices(written)).combine_chunks()
    return bool(ours.equals(theirs))


def _newest(table: str, tokens: Sequence[str]) -> str:
    """the newest of a table's generation tokens.

    :param table: the table, for the message
    :ptype table: str
    :param tokens: its tokens
    :ptype tokens: Sequence[str]
    :return: the newest
    :rtype: str
    :raises ValueError: when one is not a token, or two are of different incarnations (a reset: the table starts
        fresh from the writer's rebuild, never from a fold)
    """
    split = []
    for token in tokens:
        parsed = split_generation_token(token)
        if parsed is None:
            raise ValueError(f"{table}: {token!r} is not a generation token")
        split.append((parsed, token))
    if len({incarnation for (incarnation, _), _ in split}) != 1:
        raise ValueError(
            f"{table}: generations of incarnations {sorted({i for (i, _), _ in split})}: a reset starts the table "
            f"fresh from the writer's rebuild, never from a fold"
        )
    return max(split)[1]
