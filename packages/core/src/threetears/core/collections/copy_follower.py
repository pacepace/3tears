"""A complete copy of collection tables in DuckDB, kept current by the keys each epoch changed.

**Why.** A pod that answers analytic queries over whole tables holds every row of them in a DuckDB L1
(:mod:`~threetears.core.collections.complete_copy`). :class:`~threetears.core.collections.complete_copy.CompleteCopy`
is voided by any change and warmed whole again, so a change costs the size of the table. A copy kept by
:class:`CopyFollower` costs what changed: it hears the keys of every committed write of its tables through the
registry's standard path (each row broadcast, and each epoch's change record a catch-up applies:
:meth:`~threetears.core.collections.registry.CollectionRegistry.register_derived_cache`), reads exactly those keys'
rows from L3, and swaps them in. Nothing here is a side system: the keys come from the epoch change log, the rows
from L3, and the copy is the registry's L1 backend's own tables.

**One state of every table.** A change is applied only at a settled stamp of the writer's seqlock (``settled``, as
:class:`~threetears.core.collections.complete_copy.BufferedCopies` takes it; on the reports pod ``ScopeEpochs``): the
stamp is read, every table is caught up to its generation read AFTER it, the keys heard are read from L3 and staged
beside the copy (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.stage_rows`), the stamp is read again, and only
when it has not moved is everything staged applied in ONE DuckDB transaction
(:meth:`~threetears.core.cache.duckdb.DuckDBBackend.apply_staged`) and the copy labelled with the stamp. A stamp that
moved, or a write in progress, keeps every key pending for the next pass and the copy as it was: stale, never mixed.
The label is exact when the writer's seqlock commit comes after the generation advances of the data it covers (a
writer whose last data transaction commits the stamp itself races that transaction's advance).

**From delta files, for a table that opts in** (:attr:`FollowedTable.delta_files`). Each key heard keeps the epoch
it was heard at; the epoch's record names the commit's delta file (``EpochChange.delta_file``), read off the shared
volume (``THREETEARS_DELTA_FILES_DIR``), and the key's row is the row the copy holds with each of its epochs' changes
applied in order, merged as L3 merged them (:func:`~threetears.core.collections.delta_files.apply_delta_row`): no
statement reaches L3, however many pods follow (``design-pod-delta-files.md``, decisions 3, 4 and 19). The scope rules
are the same: a key left out of a change keeps its epochs and is staged again from its files. A key whose epoch names
no file (an older writer's record) is read from L3 by key as for any table, counted by why
(``copy_follower_delta_file_fallback_keys_total``). A commit too big to list its rows (a keys-unknown record) that
names its file is applied whole from it: the registry names every key the file holds, stamped with the epoch, and
each is staged from the file as above, so neither the change notes nor L3 is read for it and nothing a restarted copy
needs expires (decision 21). A commit of several records names one file in each; a key is applied from it once.
A load's or a clear's record that names its file (a bulk load from files, a key-prefix delete) is applied from it
too (decisions 17 and 23): each key the file names is heard at the epoch, and each prefix the file clears names every
key of it the copy holds or has heard, at the epoch, all staged as above, a key under a cleared prefix deleted before
the file's own change of it applies. L3 is read for none of it.

**A missing or unreadable file is waited for, never read from L3** (decision 29, which amends decision 3). A pass
that finds an epoch's file not on the volume, or there and not a delta file of the table
(:class:`~threetears.core.collections.delta_files.DeltaFileUnreadableError`), reads nothing from L3: the table waits for
that file (:attr:`FollowerStatus.waiting_for_file` names it), every change heard of it meanwhile is held, in order, as
the reset wait holds what it hears, and every other table is followed as before. The copy asks the writer once to
recreate the file (``recreate``, a NATS request the lease holder alone answers:
:mod:`~threetears.core.collections.delta_recreate`), reads it again when the writer answers or every
``file_poll_seconds``, and once it reads, applies the held changes in order. A load's or a clear's file, and a commit
too big to list, are found missing by the registry (:class:`~threetears.core.collections.registry.WantedFile`) and
waited for the same way; a commit too big to list cannot be recreated by key (its keys were only in the file), so its
table waits for the writer's recovery full (decision 28) when it starts from the volume. Counted by why
(``copy_follower_delta_file_waits_total``) and logged at WARNING; L3 is read once, by the writer, never by every pod.

**Bounded.** What this process holds of a change is a page of rows at a time: each page is staged as it is read.
The keys pending are the change's. A first warm reads each table a page at a time, by an order an index serves
(:attr:`FollowedTable.order`).

**Never a drop, never a rebuild.** An epoch the log cannot give (lost, aged out, keys unknown) leaves the registry
behind on the table (``CollectionRegistry.behind_on``): the copy keeps the stamp it has and says it is behind until
the log is put back (``ChangeNoteKeeper``) and the catch-up applies it. A record that names the whole table (a
truncate, a file load) reads that table again beside the copy and swaps it in whole, in the same one transaction as
everything else.

**Drift, checked against the full** (decision 32). Every :data:`DRIFT_CHECK_SECONDS` (staggered, so pods do not
check at once), a copy that starts from the volume fingerprints its own rows of each (table, state) it holds at exactly
the version of the current full's part of that state, and compares with the fingerprint the part's manifest entry
carries (:func:`~threetears.core.collections.volume_fulls.rows_fingerprint`, one definition both sides use): a local
read of the copy by key and a worker process's hash, never L3. A state whose scope column leads the key is read by that
prefix, every row the copy holds of it; any other by the keys its part holds, so a row the copy holds there and the
part does not is not seen. A state the copy holds at another version is not comparable and is skipped.
A mismatch sets ``copy_follower_drift_parts{table,state}`` to 1 until a check finds the state equal again, and is
logged at WARNING; every check is counted by outcome (``copy_follower_drift_checks_total``). The version a copy holds a
state at is the newest epoch it applied of that state, from the full it loaded and each key's epochs since
(:meth:`CopyFollower._hold_state_versions`).

**Start.** :meth:`CopyFollower.start` registers with the registry and takes each table's mark: from a checkpoint
(:mod:`~threetears.core.collections.copy_checkpoint`), whose marks the registry adopts so the catch-up walks the log
from them, or, when there is none, from the head, after which every table is read whole once (the first start only).
A table that opts in to delta files starts instead from the full on the shared volume
(:mod:`~threetears.core.collections.volume_fulls`): with none published the follower waits for it, its status saying so,
and never reads that table whole from L3.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from array import array
from datetime import UTC, datetime
from decimal import Decimal
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Final, Protocol, TYPE_CHECKING

from threetears.interfaces.keyed.rows import DeclaredKey, KeyColumns
from threetears.observe import PeriodicTask, contain, get_logger, retry_until_done
from threetears.observe.metrics import counter, gauge

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange, WholeKind
from threetears.core.cache.duckdb_binds import KEYS, run_joined
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled, rail_arrow_types
from threetears.core.collections.delta_files import (
    DeltaFileUnreadableError,
    DeltaTable,
    HeldDeltaRows,
    delta_files_root,
    read_delta_file_reach,
    read_delta_table,
    under_prefix,
)
from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.generation import split_generation_token
from threetears.core.collections.registry import CacheInvalidationMessage, WantedFile
from threetears.core.collections.scope_epochs import ScopeEpochs
from threetears.core.collections.volume_fulls import (
    FullAbsent,
    FullManifest,
    FullPart,
    StagedFull,
    VolumeFulls,
    current_full_manifest,
    fingerprint_columns,
    fingerprint_ipc,
    fingerprint_kinds,
    full_table_floor,
    read_full_part,
)
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    Column,
    TableSchema,
    sql_cast,
)
from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER
from threetears.core.declared import whole_read
from threetears.core.keyed import KeyBatch, WholeReads, read_keyed_arrow
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.process_workers import ProcessWorkers
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.core.cache.duckdb import DuckDBBackend
    from threetears.core.collections.copy_checkpoint import CopyCheckpoints
    from threetears.core.collections.epoch_change import EpochChangeReader
    from threetears.core.collections.registry import CollectionRegistry

try:
    import numpy as np
    import pyarrow as pa
except ImportError:  # NOSILENT: the snapshot extra is optional; a follower says so when built
    pa = None

__all__ = [
    "GATED_WHOLE_READ_WHYS",
    "WHOLE_READ_WHYS",
    "CopyFollower",
    "HubCapabilityMissingError",
    "FollowedTable",
    "FollowerBackend",
    "FollowerPhase",
    "FileRecreate",
    "FollowerStatus",
    "PartialStamp",
    "require_generation_by_trigger",
    "typed_text",
]

log = get_logger(__name__)

#: why a follower reads a table whole from L3, counted by table and why (``copy_follower_whole_reads_total``). Only
#: ``cold_start`` (a first start with no checkpoint) and ``reset`` (an operator's reset of that table) are expected;
#: every other is a change some writer did not record by key or key prefix
_WHOLE_READS: Final = WholeReads(
    "copy_follower_whole_reads_total",
    subject="tables a copy follower read whole from L3",
    whys={
        "cold_start": "a first start with no checkpoint: every table is read once",
        "reset": "an operator's reset of the table (SchemaBackedCollection.reset): that table's cold start",
        "schema_change": "a table dropped or recreated (CopyFollower.schema_changed): only that table is read again",
        # TODO(3tears 0.96.0 contract step, dated 2026-11-09): remove once no 0.94-or-older writer is left to record one
        "legacy_whole_record": "an epoch record naming the whole table, from a writer older than 0.95.0 mid-roll",
    },
    expected={"cold_start", "reset"},
)

# TODO(3tears 0.96.0 contract step, dated 2026-11-10): the two names below are _WHOLE_READS.whys and .gated, kept one
# release for their importers (reports' replay and prove scripts)
WHOLE_READ_WHYS: Final[Mapping[str, str]] = _WHOLE_READS.whys

#: the whys a replay or a soak requires to stay at zero: every whole read but a true cold start and an operator's reset
GATED_WHOLE_READ_WHYS: Final = _WHOLE_READS.gated

#: behind reasons the change notes have already failed to answer this pass: asking them again would only wait again
_NOTES_CANNOT: Final = frozenset({"notes_unavailable", "notes_not_covered", "notes_whole"})

#: the registry's behind reasons that mean the copy can no longer walk a table's epoch log from its mark: the log was
#: replaced under a new incarnation (every NATS replica lost), a record of it is gone, the restored log does not reach
#: the mark, or its retention passed the mark. For a table that opts in to delta files and starts from the volume the
#: copy reads nothing from L3 for it: it waits for the writer's recovered full (decision 28 of the delta-files design,
#: :meth:`CopyFollower._wait_for_recovery`). A record whose file cannot name its keys (``keys_unknown``, ``notes_*``)
#: is decision 29's, not a lost log
_LOG_LOST: Final = frozenset({"replaced", "record_missing", "record_lost", "aged_out"})

#: the most values of a key prefix's last column one partition read names
_PREFIX_BATCH: Final = 1000


#: failed passes, by what failed: ``stages_lost`` (the apply worker died and its stages with it: the change is read
#: again) or ``pass`` (any other failure: the stages are kept and the next pass tries again)
_PASS_FAILURES = counter(
    "copy_follower_pass_failures_total",
    description="copy follower passes that failed, by what failed; the copy keeps serving its last change",
    label_names=("what",),
)

#: when the copy stopped taking changes: unix time of the first failed pass since the last good one, 0 while current
_BEHIND_SINCE = gauge(
    "copy_follower_behind_since_timestamp_seconds",
    description="unix time of the first failed copy follower pass since the last change applied, 0 while passes "
    "succeed; alert when it is non-zero for several minutes (the copy serves an old change)",
)

#: consecutive failed passes after which the follower says FAILING (it keeps trying, at the longest wait)
FAILING_AFTER: Final = 5

#: the wait before a failed pass is tried again: doubled each failure from the first, up to the second (seconds)
RETRY_WAITS: Final = (1.0, 60.0)


#: passes that only relabelled the copy: nothing heard or staged, the writer's stamp moved. Such a pass relabels the
#: change the copy already holds in place and asks the backend for no apply (see ``CopyFollower._relabel``)
_RELABELS = counter(
    "copy_follower_relabels_total",
    description="copy follower passes that only relabelled the copy with a new stamp (nothing to apply): no apply, "
    "no settle, no publish",
)


#: keys of a table that opts in to delta files read from L3 by key instead of from their files, by table and why.
#: Only an older writer's records, mid-roll, reach here (decision 29: anything lost is waited for, never read):
#: ``no_file_named`` (the record names no file: a writer before delta files, 3tears < 0.95.0, or a write before the
#: table opted in) and ``no_epoch`` (the key was heard with no epoch that orders it: a row broadcast from a publisher
#: before epoch-stamped broadcasts, the keys the change notes name for a keys-unknown record of such a writer, or
#: epochs of two incarnations of the log, which the registry's ``replaced`` behind reason recovers). Every one is an L3
#: read the files were meant to save, and stays at zero once every writer runs 0.95.0
_DELTA_FILE_FALLBACKS = counter(
    "copy_follower_delta_file_fallback_keys_total",
    description="keys of a table that opts in to delta files read from L3 by key instead of from their files, by "
    "table and why; a missing or unreadable file is never one (copy_follower_delta_file_waits_total)",
    label_names=("table", "why"),
)


#: waits for a delta file, by table and why: ``file_missing`` (the record names a file the volume does not hold, or no
#: volume is mounted) or ``file_unreadable`` (it holds one under the name that is not the table's delta file:
#: truncated, garbage, another format, a damaged zstd buffer, another table's or write's), ``key_not_in_file`` (it reads,
#: and holds no change of a key its record named: read again only once the writer answered, which recreates it from
#: the record) or ``record_missing`` (the log holds no record of the epoch, so no file can be named: the writer recovers
#: the table, decision 28, and the copy waits for that full). Each is a file the writer recreated, or must; the copy
#: read nothing from L3 for it (decision 29)
_FILE_WAITS = counter(
    "copy_follower_delta_file_waits_total",
    description="times a copy follower's table began waiting for a missing or unreadable delta file, by table and why; "
    "the copy asks the writer to recreate it and reads nothing from L3 meanwhile",
    label_names=("table", "why"),
)

#: the copy's requests to the writer to recreate a delta file, by table and outcome: ``recreated``, ``present`` (the
#: writer found it whole), ``recovering`` (a commit too big to list: the writer recovers the table, decision 28),
#: ``no_writer`` (no lease holder answered: asked again), ``failed`` (the writer's recreation raised: asked again)
_RECREATE_ASKS = counter(
    "copy_follower_delta_file_recreate_requests_total",
    description="a copy follower's requests to the writer to recreate a missing or unreadable delta file, by table and "
    "outcome",
    label_names=("table", "outcome"),
)

#: the writer's answers after which the copy does not ask again for the same file: it reads it, or waits for the full
_ANSWERED: Final = frozenset({"recreated", "present", "recovering"})

#: how often a table waiting for a delta file looks for it again, besides when the writer answers (seconds)
FILE_POLL_SECONDS: Final = 5.0

#: the most changes a table waiting for a delta file holds, heard while it waits (a key at an epoch each), before it
#: stops holding them and waits for the writer's next full instead (:meth:`CopyFollower._held_past_cap`). Why: every
#: change heard meanwhile is held until the file reads, with no end if it never does (2026-11-23 audit). Measured
#: 2026-11-23 (Python 3.14): 310 bytes a held change (its pending key and epoch set), about 400 with the key's own
#: tuple, so 500,000 hold about 200 MB at most a table
HELD_WHILE_WAITING_CAP: Final = 500_000

#: how many held changes apart the held-while-waiting gauge is set (a set a change would cost the hearing a label
#: lookup each)
_HELD_GAUGE_EVERY: Final = 1000

#: changes a table waiting for a delta file holds, heard while it waits: set every :data:`_HELD_GAUGE_EVERY`, and at 0
#: once it no longer waits
_HELD_WHILE_WAITING = gauge(
    "copy_follower_held_while_waiting_changes",
    description="changes a copy follower's table holds, heard while it waits for a delta file (a key at an epoch "
    "each, about 400 bytes); past HELD_WHILE_WAITING_CAP the table waits for the writer's next full instead",
    label_names=("table",),
)

#: the most file names one fallback's log line lists
_FILES_NAMED: Final = 10

#: how often a copy that starts from the volume checks its states against the full's fingerprints (seconds; decision
#: 32), unless ``drift_check_seconds`` or :data:`_DRIFT_CHECK_ENV` says otherwise: a quarter hour, a local read of each
#: comparable state and its hash in a worker process, never L3
DRIFT_CHECK_SECONDS: Final = 900.0

#: the per-deployment setting of the check's period, in seconds; ``0`` turns it off
_DRIFT_CHECK_ENV: Final = "THREETEARS_COPY_DRIFT_CHECK_SECONDS"

#: (table, state) parts whose copy rows differ from the full's fingerprint at the same version: 1 while the last check
#: of the state found it so, 0 once one finds it equal (decision 32). The dashboard's; alerts are Grafana's
_DRIFT_PARTS = gauge(
    "copy_follower_drift_parts",
    description="1 while a copy's rows of a (table, state) differ from the full's fingerprint of the part at the same "
    "version, 0 once a check finds them equal; a local check, never L3",
    label_names=("table", "state"),
)

#: the drift checks of each part, by table and outcome: ``equal``, ``drift``, ``skipped`` (the copy holds the state at
#: another version than the part's: not comparable), ``not_current`` (the table has a change pending, staged, waiting
#: or behind), ``version_unknown`` (a change of the state was applied with no epoch), ``no_fingerprint`` (an older
#: writer's part), ``other_columns`` (the copy holds other columns)
_DRIFT_CHECKS = counter(
    "copy_follower_drift_checks_total",
    description="a copy's checks of its states against the full's fingerprints, by table and outcome (equal, drift, "
    "skipped: held at another version, not_current, version_unknown, no_fingerprint, other_columns)",
    label_names=("table", "outcome"),
)

#: a state's held version once a change of it was applied with no epoch: never a part's version
_VERSION_UNKNOWN: Final = ""


def _drift_check_seconds(given: float | None) -> float:
    """the drift check's period: ``given``, else :data:`_DRIFT_CHECK_ENV`, else :data:`DRIFT_CHECK_SECONDS`.

    :param given: the follower's argument
    :ptype given: float | None
    :return: seconds; 0 turns the check off
    :rtype: float
    :raises ValueError: when the period is negative or the setting is not a number (a start refuses it)
    """
    if given is None:
        named = os.environ.get(_DRIFT_CHECK_ENV)
        given = DRIFT_CHECK_SECONDS if not named else float(named)
    if given < 0:
        raise ValueError(f"the drift check's period is not negative, got {given}")
    return given


def _newest_epoch(epochs: Iterable[str | None]) -> str | None:
    """the newest of a key's epochs; None when one is missing or two are of different incarnations (no order).

    :param epochs: the epochs
    :ptype epochs: Iterable[str | None]
    :return: the newest, or None
    :rtype: str | None
    """
    parsed = [split_generation_token(epoch) if epoch is not None else None for epoch in epochs]
    if not parsed or None in parsed or len({p[0] for p in parsed if p is not None}) != 1:
        return None
    incarnation, count = max(p for p in parsed if p is not None)
    return f"{incarnation}:{count}"


def _later_epoch(held: str | None, epoch: str) -> str:
    """a state's held version after a change of it at ``epoch`` applies: the later of the two, or ``epoch`` when they
    are of different incarnations (a table reset is reloaded from the volume, so the older one is gone).

    :param held: the state's held version; None when it held none
    :ptype held: str | None
    :param epoch: the change's
    :ptype epoch: str
    :return: the held version
    :rtype: str
    """
    if held is None:
        return epoch
    if held == _VERSION_UNKNOWN:
        return held
    old, new = split_generation_token(held), split_generation_token(epoch)
    if old is None or new is None or old[0] != new[0]:
        return epoch
    return held if old[1] >= new[1] else epoch


def _warms() -> Any:
    return counter(
        "copy_follower_warms_total",
        description="copy follower warms from L3 (every table, or one table again), by why; zero but the cold start",
        label_names=("why",),
    )


#: how many applied changes' stamps are kept for readers whose snapshot is older than the newest
_STAMPS_KEPT: Final = 256


class HubCapabilityMissingError(RuntimeError):
    """the hub's broker lacks a capability a copy follower needs: the copy would be silently stale, so it never serves.

    :ivar capability: the capability's name
    """

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(
            f"the hub does not announce {capability!r}: it advances no generation for a pod's own tables switched on "
            f"by their trigger, so a copy following them would never hear a change. Deploy the hub that announces it "
            f"before this pod; the pod stays not ready until then"
        )


async def require_generation_by_trigger(l3: Any) -> None:
    """refuse, before a copy follower serves, a hub whose broker does not advance trigger-switched tables.

    An L3 reached through the hub's broker (anything with ``hub_capabilities``) must announce
    :data:`~threetears.core.backends.broker_generation.GENERATION_BY_TRIGGER`. An L3 reached directly (a SQL pool, a
    test's) has no broker between: its writers advance their own generations, and nothing is asked.

    :param l3: the pod's L3
    :ptype l3: Any
    :return: nothing
    :rtype: None
    :raises HubCapabilityMissingError: when the hub does not announce it
    """
    probe = getattr(l3, "hub_capabilities", None)
    if probe is None:
        return
    if GENERATION_BY_TRIGGER not in await probe():
        raise HubCapabilityMissingError(GENERATION_BY_TRIGGER)


def _text_keys(rows: Any, key: Sequence[str]) -> list[tuple[str, ...]]:
    """each row's key, its values as text in key order (as a change record names a key).

    :param rows: rows holding the key columns
    :ptype rows: pyarrow.Table
    :param key: the key columns
    :ptype key: Sequence[str]
    :return: the keys
    :rtype: list[tuple[str, ...]]
    """
    return [tuple(str(v) for v in values) for values in zip(*(rows.column(c).to_pylist() for c in key), strict=True)]


#: keys a run of the loop's work names before it lets the loop run again (a pass's snapshot of each key's epochs)
_KEYS_PER_TURN: Final = 2000


def _no_record(table: str, epoch: str) -> str:
    """what a wait for an epoch whose record the log lost is named by: no file can be (the record named it); the
    writer, asked with it, finds no record and recovers the table (decision 28).

    :param table: the table
    :ptype table: str
    :param epoch: the epoch
    :ptype epoch: str
    :return: the name
    :rtype: str
    """
    return f"{table}/no record of epoch {epoch}"


@dataclass(frozen=True, slots=True)
class _WaitedReach:
    """what a waited-for delta file reaches: the prefixes it clears and its keys, each as text, once, in order.

    :ivar prefixes: the key prefixes it clears, as text
    :ivar keys: each key's values as text (``key_text``)
    """

    prefixes: tuple[dict[str, str], ...]
    keys: list[tuple[str, ...]]


def _waited_reach(path: Any, table: str) -> _WaitedReach:
    """a waited-for file's reach, its keys as text. Blocking: the read and a loop over every key, run on a thread.

    :param path: the file
    :ptype path: Path
    :param table: the table whose file it must be
    :ptype table: str
    :return: its reach
    :rtype: _WaitedReach
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when it is not the table's delta file, or does not decode
    """
    reach = read_delta_file_reach(path, table=table)
    return _WaitedReach(prefixes=reach.prefixes, keys=[tuple(key_text(value) for value in key) for key in reach.keys])


@dataclass(frozen=True, slots=True)
class _EpochPlan:
    """the epochs a pass's keys of one opted-in table were heard at, each epoch's keys, oldest epoch first.

    :ivar order: every epoch heard, oldest first (one incarnation's counts order them)
    :ivar keys_at: each epoch's keys, as their positions in the pass's keys
    :ivar no_epoch: per key, heard with no epoch that orders it (none, or two incarnations')
    """

    order: list[str]
    keys_at: dict[str, Any]
    no_epoch: Any


def _epoch_plan(heard: Sequence[frozenset[str | None]]) -> _EpochPlan:
    """which keys each epoch names, and which keys no epoch orders. Blocking: a loop over every key, run on a thread.

    :param heard: each key's epochs, by position
    :ptype heard: Sequence[frozenset[str | None]]
    :return: the plan
    :rtype: _EpochPlan
    """
    no_epoch = np.zeros(len(heard), dtype=bool)
    parsed: dict[str, tuple[str, int] | None] = {}
    keys_at: dict[str, array[int]] = {}
    for position, epochs in enumerate(heard):
        parts = []
        for epoch in epochs:
            if epoch is not None and epoch not in parsed:
                parsed[epoch] = split_generation_token(epoch)
            parts.append(None if epoch is None else parsed[epoch])
        if not parts or None in parts or len({part[0] for part in parts if part is not None}) != 1:
            # heard with no epoch, or under two incarnations of the log, whose counts do not order them
            no_epoch[position] = True
            continue
        for part in parts:
            assert part is not None  # noqa: S101 -- checked above
            keys_at.setdefault(f"{part[0]}:{part[1]}", array("q")).append(position)
    order = sorted(keys_at, key=lambda epoch: split_generation_token(epoch) or ("", 0))
    return _EpochPlan(
        order=order,
        keys_at={epoch: np.frombuffer(keys_at[epoch], dtype=np.int64) for epoch in order},
        no_epoch=no_epoch,
    )


@dataclass(frozen=True, slots=True)
class _FromFiles:
    """a pass's rows of one opted-in table, merged from its epochs' files onto the copy's rows (on a thread).

    :ivar rows: the rows that exist after the change, typed as the rail answers them, in key position order
    :ivar at: each row's key's position in the pass's keys, ascending
    :ivar heard: each key's epochs it was merged from, by position (an apply holds those)
    :ivar from_l3: per key, read from L3 by key instead (a fallback, counted)
    :ivar fallbacks: the keys read from L3, by why: their count and the files named
    :ivar applied: how many changes were applied from the files
    """

    rows: Any
    at: Any
    heard: list[frozenset[str | None]]
    from_l3: Any
    fallbacks: dict[str, tuple[int, set[str]]]
    applied: int


def _rail_column(column: Any, kind: Any) -> Any:
    """a column the copy holds as the L3 rail answers it (:func:`_rail_python`): kept when already of its type, cast
    when it is text or an integer to be text, else made value by value (only such a column's values become Python).

    :param column: the copy's column
    :ptype column: pyarrow.ChunkedArray
    :param kind: the column's Arrow type as the rail answers it
    :ptype kind: pyarrow.DataType
    :return: the column, typed
    :rtype: pyarrow.Array
    """
    column = column.combine_chunks()
    given = column.type
    if given == kind:
        return column
    textual = pa.types.is_string(given) or pa.types.is_large_string(given)
    if pa.types.is_string(kind) and (textual or pa.types.is_integer(given)):
        return column.cast(kind)
    return pa.array([_rail_python(value, kind) for value in column.to_pylist()], kind)


def _rail_python(value: Any, kind: Any) -> Any:
    """a value the copy holds as the L3 rail answers its column (``rail_arrow_types``): text where the rail answers
    text (a ``NUMERIC``'s digits, an instant in its key spelling, a UUID), an instant where it answers an instant (the
    copy may keep one as its ISO text), else as it is.

    Why: a row merged from a delta file takes the file's values, typed as the rail answers them, beside the copy's own
    of every column its write left out, and the stage takes one type a column, as it takes an L3 read's.

    :param value: the copy's value
    :ptype value: Any
    :param kind: the column's Arrow type as the rail answers it
    :ptype kind: pyarrow.DataType
    :return: the value as the rail answers it
    :rtype: Any
    """
    if value is None:
        return value
    if pa.types.is_timestamp(kind) and isinstance(value, str):
        return datetime.fromisoformat(value)
    if not pa.types.is_string(kind) or isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    return key_text(value)


def _key_prefix(table: FollowedTable, partition: Mapping[str, str]) -> bool:
    """whether a partition names the leading columns of the table's key: the one shape an index reads exactly (a key
    range), never the table.

    :param table: the table
    :ptype table: FollowedTable
    :param partition: the partition's columns and values
    :ptype partition: Mapping[str, str]
    :return: whether its columns are a key prefix
    :rtype: bool
    """
    return bool(partition) and set(partition) == set(table.key[: len(partition)])


def typed_text(kind: str, text: str) -> Any:
    """a value given as text (a record's partition, a key an epoch record names) as the L3 rail binds its column's type.

    Shared with the writer's recreation of a delta file (:mod:`~threetears.core.collections.delta_recreate`), which binds
    a record's keys and prefixes the same way.

    :param kind: the column's type tag
    :ptype kind: str
    :param text: the value
    :ptype text: str
    :return: the value
    :rtype: Any
    """
    if kind in (INT_TYPE, BIGINT_TYPE):
        return int(text)
    if kind == BOOL_TYPE:
        return text.lower() in ("true", "t", "1")
    if kind == NUMERIC_TYPE:
        return Decimal(text)
    if kind == DATETIMETZ_TYPE:
        return datetime.fromisoformat(text)
    return text


def _prefix_batches(
    key: Annotated[Sequence[str], DeclaredKey], parts: Sequence[Mapping[str, str]]
) -> list[tuple[tuple[str, ...], tuple[str, ...], list[str]]]:
    """partitions grouped into reads: those of one shape (their columns, in key order, and the values of all but the
    last) are one read, every value of the last column bound at once, at most :data:`_PREFIX_BATCH` a read.

    :param key: the table's key, whose order a prefix's columns take
    :ptype key: Sequence[str]
    :param parts: the partitions, each a key prefix or a held scope's column
    :ptype parts: Sequence[Mapping[str, str]]
    :return: ``(columns, leading values, the last column's values)`` per read
    :rtype: list[tuple[tuple[str, ...], tuple[str, ...], list[str]]]
    """
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, None]] = {}
    for part in parts:
        columns = tuple(sorted(part, key=lambda c: key.index(c) if c in key else len(key)))
        lead = tuple(part[c] for c in columns[:-1])
        grouped.setdefault((columns, lead), {})[part[columns[-1]]] = None
    return [
        (columns, lead, list(values)[start : start + _PREFIX_BATCH])
        for (columns, lead), values in grouped.items()
        for start in range(0, len(values), _PREFIX_BATCH)
    ]


class FileRecreate(Protocol):
    """asks the writer to recreate one epoch's delta file, and answers how it went: the copy's side of
    :class:`~threetears.core.collections.delta_recreate.DeltaRecreateRequester` (a NATS request the lease holder alone
    answers), kept a protocol so the follower carries no NATS client.
    """

    async def __call__(self, table: str, file: str, epoch: str) -> str:
        """ask once; never raises for an answer the copy can wait out.

        :param table: the table
        :ptype table: str
        :param file: the file's name under the delta files' directory, as the epoch's record names it
        :ptype file: str
        :param epoch: the epoch whose record names it
        :ptype epoch: str
        :return: ``recreated``, ``present``, ``recovering``, ``no_writer`` or ``failed``
        :rtype: str
        """
        ...


@dataclass(slots=True)
class _FileWait:
    """one delta file a table waits for (decision 29): every change of the table is held until it reads.

    :ivar epoch: the epoch whose record names it
    :ivar file: its name under the delta files' directory
    :ivar why: ``file_missing`` or ``file_unreadable``
    :ivar reach: what the file brings once it reads: ``keys`` (the keys the copy heard at the epoch, held pending),
        ``prefixes`` (a load's or a clear's: its prefixes and keys are heard from the file, as the registry would have
        told them) or ``keys_unknown`` (a commit too big to list: its keys are heard from the file)
    :ivar watch: whether the file is looked for; False for a commit too big to list whose table waits for the writer's
        recovery full instead (its file cannot be recreated by key)
    :ivar answer: the writer's last answer; None before it is asked
    """

    epoch: str
    file: str
    why: str
    reach: str
    watch: bool = True
    answer: str | None = None

    def say(self) -> str:
        """the wait, for the status.

        :return: a line
        :rtype: str
        """
        asked = "not yet asked the writer" if self.answer is None else f"the writer answered {self.answer}"
        return f"{self.file} of epoch {self.epoch}: {self.why}; {asked}"


#: every scope the stamp names: a waiting table whose held changes' scopes are not known holds them all
#: (:meth:`CopyFollower._waiting_scopes`); compared by identity
_EVERY_SCOPE: Final = frozenset({"\x00every scope"})


class _FileWaitError(RuntimeError):
    """a pass found a key's epoch file missing or unreadable: the table waits for it, its keys held (decision 29).

    Not a ``LookupError``: a pass reads that as its stages lost with the apply worker.
    """

    def __init__(self, table: str, file: str, epoch: str, why: str) -> None:
        super().__init__(f"{table}: delta file {file} of epoch {epoch}: {why}")
        self.table = table
        self.file = file
        self.epoch = epoch
        self.why = why


@dataclass(frozen=True, slots=True)
class FollowedTable:
    """one table a :class:`CopyFollower` keeps whole.

    :ivar name: the table, as L3 and the copy both name it
    :ivar key: its primary key's columns, in declared order: the order a change record names a key's values in
    :ivar columns: the columns the copy holds, each read from L3 (``name`` and ``column_type``); every key column is one
    :ivar order: a unique column order an index of L3 serves, for the first warm's pages; the key when empty
    :ivar partition_order: a unique column order an index led by a load's constant columns serves, for the pages of a
        load's rows (a run's, a state's at a version); the key when empty
    """

    name: str
    key: KeyColumns
    columns: tuple[Column, ...]
    order: tuple[str, ...] = ()
    partition_order: tuple[str, ...] = ()
    #: every column L3 holds and its type tag, the copy's or not: a load's partition may name a column the copy does
    #: not hold (a version stamp), which the read of its rows still filters on
    l3_kinds: tuple[tuple[str, str], ...] = ()
    #: the scope each row belongs to, as the writer's seqlock names scopes: ``(column, prefix)``, a row's scope being
    #: ``prefix + str(row[column])`` (``("join_state_code", "state:")``). A write touches the scope of every row it
    #: writes, so while a write is in progress the follower still applies every scope it does not hold. None: the
    #: table's changes apply only at a settled stamp (a write in progress holds all of them back)
    scope: tuple[str, str] | None = None
    #: the table opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`):
    #: each epoch's changes are applied from the file its record names, read off the shared volume, merged onto the
    #: row the copy holds, never read from L3 (``design-pod-delta-files.md``, decisions 3, 4 and 19). A missing or
    #: unreadable file is waited for while the writer recreates it, never read from L3 (decision 29)
    delta_files: bool = False

    def __post_init__(self) -> None:
        held = {column.name for column in self.columns}
        missing = [column for column in self.key if column not in held]
        if missing:
            raise ValueError(f"{self.name}: the copy must hold every key column; {missing} is not among its columns")
        if self.scope is not None and self.scope[0] not in held:
            raise ValueError(f"{self.name}: the copy must hold its scope column {self.scope[0]!r}")
        # a keyset read pages past the last row's values of its order: an order that is not unique skips every row
        # sharing the cursor's values at each page boundary, so the copy would hold fewer rows than L3 under L3's
        # epochs. Unique here means it holds every key column
        for named, order in (("order", self.order), ("partition_order", self.partition_order)):
            short = [column for column in self.key if order and column not in order]
            if short:
                raise ValueError(
                    f"{self.name}: {named} {list(order)} is not unique: a keyset read by it skips rows at page "
                    f"boundaries; it must hold every key column, missing {short}"
                )

    def scope_of(self, value: Any) -> str:
        """the scope a row whose scope column holds ``value`` belongs to.

        :param value: the scope column's value
        :ptype value: Any
        :return: the scope, as the writer's seqlock names it
        :rtype: str
        """
        assert self.scope is not None  # noqa: S101 -- asked only of a scoped table
        return f"{self.scope[1]}{value}"

    @property
    def scope_in_key(self) -> int | None:
        """where the scope column sits in the key, or None when the key does not hold it (or the table has no scope).

        :return: its index
        :rtype: int | None
        """
        if self.scope is None or self.scope[0] not in self.key:
            return None
        return self.key.index(self.scope[0])

    @classmethod
    def of(
        cls, schema: TableSchema, *, columns: Sequence[str] | None = None, order: Sequence[str] = ()
    ) -> FollowedTable:
        """a table as its schema declares it.

        :param schema: the table's schema
        :ptype schema: TableSchema
        :param columns: the columns the copy holds, by name; every column when None
        :ptype columns: Sequence[str] | None
        :param order: a unique column order an index serves, for the first warm
        :ptype order: Sequence[str]
        :return: the table
        :rtype: FollowedTable
        """
        key = schema.primary_key if isinstance(schema.primary_key, tuple) else (schema.primary_key,)
        wanted = None if columns is None else set(columns)
        held = tuple(c for c in schema.columns if wanted is None or c.name in wanted)
        kinds = tuple((c.name, c.column_type) for c in schema.columns)
        return cls(
            name=schema.name,
            key=tuple(key),
            columns=held,
            order=tuple(order),
            l3_kinds=kinds,
            delta_files=schema.delta_files,
        )

    @property
    def read_order(self) -> tuple[str, ...]:
        """the columns the first warm pages on.

        :return: :attr:`order`, or the key
        :rtype: tuple[str, ...]
        """
        return self.order or self.key

    def kind(self, column: str) -> str:
        """a column's type tag.

        :param column: a column the copy holds, or one its L3 table holds
        :ptype column: str
        :return: its type tag
        :rtype: str
        :raises ValueError: when neither names it
        """
        for held in self.columns:
            if held.name == column:
                return held.column_type
        for name, kind in self.l3_kinds:
            if name == column:
                return kind
        raise ValueError(f"{self.name}: neither the copy nor its schema names {column!r}")

    def key_cast(self, column: str) -> str:
        """the SQL type a key column's text is cast to.

        :param column: a key column
        :ptype column: str
        :return: the type
        :rtype: str
        """
        kind = next(c.column_type for c in self.columns if c.name == column)
        return sql_cast(kind)


@dataclass(frozen=True, slots=True)
class _KeyedRead:
    """how some keys' rows of one table are read from L3: :func:`~threetears.core.keyed.read_keyed_arrow` over the
    table's key, each key column's values bound as one text array and joined to the table on its key.

    On PostgreSQL and YugabyteDB the join is answered by the primary key, a lookup a key (``EXPLAIN`` shows an
    index scan of the key, never a scan of the table): its cost is the keys', whatever the table holds.

    :ivar casts: each key column's SQL type, its text cast back to
    :ivar types: each column's Arrow type, as the rail answers it
    """

    casts: tuple[str, ...]
    types: Mapping[str, Any]

    @classmethod
    def of(cls, table: FollowedTable) -> _KeyedRead:
        """the read of ``table``'s rows by key.

        :param table: the table
        :ptype table: FollowedTable
        :return: the read
        :rtype: _KeyedRead
        """
        return cls(casts=tuple(table.key_cast(column) for column in table.key), types=rail_arrow_types(table.columns))


def _stamp_table_of(
    stamp_table: str | None,
    settled: Callable[[], Awaitable[Any]],
    tables: Sequence[FollowedTable],
) -> str | None:
    """the table the writer's stamp is kept in, whose row messages wake the follower: the one named, else derived from
    ``settled`` when it is a ``ScopeEpochs.settled``; None for a follower none of whose tables has a scope, whose keys
    no write can hold.

    :raises ValueError: when a table has a scope and the stamp table is neither named nor derivable
    """
    if stamp_table is not None:
        return stamp_table
    epochs = getattr(settled, "__self__", None)
    if isinstance(epochs, ScopeEpochs):
        return epochs.table_name
    scoped = sorted(table.name for table in tables if table.scope is not None)
    if not scoped:
        return None
    raise ValueError(
        f"{scoped} have a scope a write may hold: pass stamp_table=, the table the writer's stamp is kept in "
        "(ScopeEpochs' own, ScopeEpochs.table_name), or settled=ScopeEpochs.settled to derive it; without it a "
        "follower holding a write's keys never hears another replica commit it"
    )


@dataclass(frozen=True, slots=True)
class PartialStamp:
    """what a copy holds while a write is in progress: every scope the write does not hold at its committed epoch, and
    each scope it holds at the epoch the copy last applied it at.

    A reader of a held scope reads its last settled version; a reader of any other reads the newest committed. The
    stamp the follower labels a change with when its ``settled`` answered :class:`Unsettled` (with each scope's epoch
    and the scopes the write touched).

    :ivar epochs: each scope's epoch as the copy holds it
    :ivar held: the scopes the write in progress holds, each at an older epoch than L3's rows may be
    :ivar version: the last committed write's version, when the writer's record names it
    """

    epochs: Mapping[str, int]
    held: frozenset[str]
    version: int | None = None

    def epoch(self, scope: str) -> int:
        """one scope's epoch as the copy holds it; 0 for a scope it holds at none.

        :param scope: the scope
        :ptype scope: str
        :return: the epoch
        :rtype: int
        """
        return self.epochs.get(scope, 0)


def _scoped(stamp: Any) -> tuple[Mapping[str, int], frozenset[str]] | None:
    """a writer's record as each scope's epoch and the scopes a write in progress holds; None for a stamp that names
    no scope (the follower then applies only at a settled stamp, compared whole).

    :param stamp: what ``settled`` answered, or a label
    :ptype stamp: Any
    :return: the epochs and the held scopes
    :rtype: tuple[Mapping[str, int], frozenset[str]] | None
    """
    if isinstance(stamp, PartialStamp):
        return stamp.epochs, stamp.held
    epochs = getattr(stamp, "epochs", None)
    if not isinstance(epochs, Mapping):
        return None
    held = frozenset(getattr(stamp, "pending", frozenset())) if isinstance(stamp, Unsettled) else frozenset()
    return epochs, held


class FollowerPhase(StrEnum):
    """what a follower is doing."""

    STARTING = "starting"
    WARMING = "warming"
    FOLLOWING = "following"
    WAITING = "waiting"
    BEHIND = "behind"
    FAILING = "failing"
    STOPPED = "stopped"


@dataclass(slots=True)
class FollowerStatus:
    """what a follower says of itself.

    :ivar phase: what it is doing
    :ivar stamp: the writer's stamp the copy holds every table at; None until it holds one
    :ivar behind: the tables whose log the registry is behind on, with why: its reason
        (:data:`~threetears.core.collections.registry.BEHIND_REASONS`), as ``collection_epoch_behind_total`` labels it
    :ivar pending: keys heard and not yet applied, by table
    :ivar rows: each table's rows, as last counted
    :ivar applied: changes applied since start
    :ivar relabelled: passes since start that only relabelled the copy (nothing to apply, a new stamp): no apply
    :ivar last_apply_seconds: how long the last change took, from the first stamp read to the commit
    :ivar failures: passes failed in a row since the last good one; at :data:`FAILING_AFTER` the phase is FAILING
    :ivar behind_since: when the first of those failed (None while passes succeed): the copy serves the change it held
        then
    :ivar waiting_for_full: each opted-in table reset or recreated while the follower ran, waiting for the writer to
        publish its full on the delta-files volume (never read from L3), onto why it is still waiting; every other
        table keeps being followed meanwhile
    :ivar waiting_for_file: each opted-in table waiting for a missing or unreadable delta file the writer recreates
        (decision 29), onto which file and why; its changes are held meanwhile, every other table followed
    :ivar detail: a line for an operator
    """

    phase: FollowerPhase = FollowerPhase.STARTING
    stamp: Any = None
    behind: dict[str, str] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    applied: int = 0
    relabelled: int = 0
    last_apply_seconds: float | None = None
    failures: int = 0
    behind_since: datetime | None = None
    waiting_for_full: dict[str, str] = field(default_factory=dict)
    waiting_for_file: dict[str, str] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _FullAfter:
    """the full an opted-in table reset or recreated while the follower ran must be reloaded from: one that holds the
    table at or past the reset (decisions 12 and 14 of the delta-files design: the writer rebuilds the table from L3
    once and publishes it; a copy never reads it from L3). A full from before the reset holds the table the reset
    replaced, so it is never taken.

    :ivar why: ``reset`` or ``schema_change``
    :ivar token: the reset's generation token when the record named it; else the newest generation the copy had heard
        of the table, which the reset came after (a reset in this process is told before its advance is written);
        None when the copy had heard none
    :ivar inclusive: whether ``token`` is the reset's own (a full at it holds the reset table) or one it came after
    :ivar incarnation: for a lost epoch log (decision 28), the incarnation of the log the copy reads now: a full is
        taken only once it holds the table under it, which only the writer's recovery (or a later full) does; a full
        of it is past a ``token`` of the lost log's incarnation. None for a reset or a schema change
    """

    why: str
    token: str | None
    inclusive: bool
    incarnation: str | None = None

    def short_of(self, table: str, manifest: FullManifest | None) -> str | None:
        """why ``manifest`` cannot be reloaded from; None when it can.

        :param table: the table
        :ptype table: str
        :param manifest: the current manifest, None when none is published
        :ptype manifest: FullManifest | None
        :return: why not, for the status
        :rtype: str | None
        :raises ValueError: when the table's versions in it are not generation tokens of one incarnation
        """
        if manifest is None:
            return "no full on the volume; the writer publishes one"
        entry = manifest.tables.get(table)
        if entry is None:
            return f"full {manifest.name} does not hold the table; the writer publishes one that does"
        if self.token is None and self.incarnation is None:
            return None
        floor = full_table_floor(table, entry)
        held = split_generation_token(floor)
        if self.incarnation is not None and (held is None or held[0] != self.incarnation):
            return (
                f"full {manifest.name} holds the table at {floor}, not yet under incarnation {self.incarnation}: the "
                f"writer's recovery of the {self.why} publishes one"
            )
        if self.token is None:
            return None
        after = split_generation_token(self.token)
        if self.incarnation is not None and after is not None and after[0] != self.incarnation:
            # the mark is of the lost log's incarnation: every full of the log read now holds the table past it
            return None
        if held is None or after is None or held[0] != after[0]:
            # a count under one incarnation is never compared with one under another: the reset's own full is awaited
            return f"full {manifest.name} holds the table at {floor}, of another incarnation than the {self.why} at {self.token}"
        if held[1] > after[1] or (self.inclusive and held[1] == after[1]):
            return None
        return f"full {manifest.name} holds the table at {floor}, from before the {self.why} at {self.token}"


_Heard = tuple[
    dict[tuple[str, ...], set[str | None]],
    list[dict[str, str]],
    list[tuple[str, list[dict[str, str]]]],
]


def _held_by_full(epoch: str | None, floor: tuple[str, int] | None, after: _FullAfter) -> bool:
    """whether the full reloaded at ``floor`` already holds the change of ``epoch``.

    :param epoch: the epoch a change was heard at; None when it was heard with none (never judged held)
    :ptype epoch: str | None
    :param floor: the full's version of the table, split; None when it is not a generation token
    :ptype floor: tuple[str, int] | None
    :param after: what the table waited for: after a lost log, an epoch of the lost incarnation is held too
    :ptype after: _FullAfter
    :return: whether it is held
    :rtype: bool
    """
    of = split_generation_token(epoch) if epoch is not None else None
    if of is None or floor is None:
        return False
    if of[0] != floor[0]:
        return after.incarnation is not None and of[0] != after.incarnation
    return of[1] <= floor[1]


def _past_floor(heard: _Heard, floor: str, after: _FullAfter) -> _Heard:
    """what was heard of a table while it waited, less the changes the full it reloaded from already holds: applied
    again over the full, an older change would undo a newer one the full holds (a row changed twice, the second in a
    record the copy never read).

    :param heard: the keys with their epochs, the partitions and the cleared prefixes with their epochs
    :ptype heard: _Heard
    :param floor: the full's version of the table (its oldest)
    :ptype floor: str
    :param after: what the table waited for
    :ptype after: _FullAfter
    :return: the same, without what the full holds; a key every one of whose epochs it holds is left out
    :rtype: _Heard
    """
    keys, partitions, clears = heard
    split = split_generation_token(floor)
    kept: dict[tuple[str, ...], set[str | None]] = {}
    for key, epochs in keys.items():
        left = {epoch for epoch in epochs if not _held_by_full(epoch, split, after)}
        if left or not epochs:
            kept[key] = left
    return (
        kept,
        partitions,
        [(epoch, prefixes) for epoch, prefixes in clears if not _held_by_full(epoch, split, after)],
    )


class _StagesLostError(LookupError):
    """a pass's stages were lost with the worker holding them: forgotten, and the change is read again."""


class FollowerBackend(Protocol):
    """what a :class:`CopyFollower` keeps its copy in: a backend that stages a change beside the copy and applies
    everything staged in one transaction (:class:`~threetears.core.cache.duckdb.DuckDBBackend` is one; a pod's own
    backend that publishes the copy to its worker processes is another). Each method is called on a thread.
    """

    def stage_rows(self, stage: str, table: str, rows: Any, key: Sequence[str]) -> int:
        """stage some rows of ``table`` under ``stage``; the rows staged."""
        ...

    def stage_keys(
        self, stage: str, table: str, key: Sequence[str], keys: Any, *, rows_stage: str | None = None
    ) -> int:
        """stage the keys read under ``stage`` (a key not among ``rows_stage``'s rows is deleted); the keys staged."""
        ...

    def unstage_keys(self, stages: Sequence[str], table: str, key: Sequence[str], keys: Any) -> int:
        """take these keys out of these stages of ``table`` (their keys, their rows), the rest of each kept; the keys
        given."""
        ...

    def drop_stage(self, *stages: str) -> None:
        """drop these stages."""
        ...

    def staged_stages(self) -> list[str]:
        """every stage held."""
        ...

    def apply_staged(self, changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        """apply every change staged in one transaction, labelled with ``mark``; the rows applied. Raises
        ``LookupError`` when the stages are gone (the worker holding them died; nothing applied): the follower then
        reads the change again. Any other failure keeps the stages for the apply to be tried again."""
        ...

    def read_snapshot(self) -> AbstractContextManager[Any]:
        """a cursor over one applied state of the copy, for as long as the block runs."""
        ...

    def staged_mark(self, cursor: Any) -> int | None:
        """the mark of the change the state ``cursor`` reads was applied with; None before any."""
        ...


class CopyFollower:
    """keeps tables whole in a DuckDB backend, current to the writer's settled stamp, by the keys each epoch changed.

    :param registry: the registry whose epoch catch-up names the keys; each table is followed there
    :ptype registry: CollectionRegistry
    :param tables: the tables, each created on ``backend`` already
    :ptype tables: Sequence[FollowedTable]
    :param backend: the copy
    :ptype backend: FollowerBackend
    :param l3: L3, the rail's proxy or anything with an asyncpg-shaped ``fetch``
    :ptype l3: Any
    :param settled: the writer's seqlock: a stamp, or :class:`Unsettled` while a write is in progress
    :ptype settled: Callable[[], Awaitable[Any]]
    :param generation: a table's current write generation, or None when it has none
    :ptype generation: Callable[[str], Awaitable[str | None]]
    :param changes: reads the epoch change log; None leaves every unheard epoch behind
    :ptype changes: EpochChangeReader | None
    :param page_size: rows a statement reads; under the rail's cap
    :ptype page_size: int
    :param checkpoints: where the copy's checkpoints are kept (:mod:`~threetears.core.collections.copy_checkpoint`);
        None keeps none, and every start reads L3 whole
    :ptype checkpoints: CopyCheckpoints | None
    :param stamp_table: the table the writer's stamp is kept in (``ScopeEpochs``' own), whose row messages the
        registry hears from every replica: each one asks a pass, so keys a write held are applied once another
        replica's commit moves its scopes, with nothing else heard. None derives it from ``settled`` when that is a
        ``ScopeEpochs.settled``, and hears none when no table has a scope (no write can hold its keys)
    :ptype stamp_table: str | None
    :param retry_waits: after a failed pass, the first wait before :meth:`run` tries again and the longest it doubles to
        (seconds)
    :ptype retry_waits: tuple[float, float]
    :param fulls: the full on the delta-files volume (:mod:`~threetears.core.collections.volume_fulls`) for the tables
        that opt in to delta files: a start loads them from it, waiting for one when none is published, and never
        reads them whole from L3; ``checkpoints`` then holds only the other tables. None: every table starts as before
    :ptype fulls: VolumeFulls | None
    :param recreate: asks the writer to recreate a missing or unreadable delta file (decision 29); None asks no one,
        and a table waiting for a file looks for it every ``file_poll_seconds`` until it reads
    :ptype recreate: FileRecreate | None
    :param file_poll_seconds: how often a table waiting for a delta file looks for it again, besides when the writer
        answers; None takes :data:`FILE_POLL_SECONDS` as it is when the follower is built
    :ptype file_poll_seconds: float | None
    :param drift_check_seconds: how often :meth:`run` checks the copy's states against the full's fingerprints
        (:meth:`check_drift`), the first check at a random point within one period; None reads
        ``THREETEARS_COPY_DRIFT_CHECK_SECONDS``, else :data:`DRIFT_CHECK_SECONDS`; 0 turns it off. Only with ``fulls``
    :ptype drift_check_seconds: float | None
    :param drift_workers: the worker processes a state's fingerprint is taken in; None makes one worker of the
        follower's own, closed when :meth:`run` ends
    :ptype drift_workers: ProcessWorkers | None
    :raises ValueError: when a table has a scope and ``stamp_table`` is neither given nor derivable, or a table is both
        in ``fulls`` and in ``checkpoints``, or ``fulls`` holds a table the follower does not follow
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        tables: Sequence[FollowedTable],
        backend: FollowerBackend,
        l3: Any,
        *,
        settled: Callable[[], Awaitable[Any]],
        generation: Callable[[str], Awaitable[str | None]],
        changes: EpochChangeReader | None,
        page_size: int = DEFAULT_PAGE_SIZE,
        checkpoints: CopyCheckpoints | None = None,
        stamp_table: str | None = None,
        retry_waits: tuple[float, float] = RETRY_WAITS,
        fulls: VolumeFulls | None = None,
        recreate: FileRecreate | None = None,
        file_poll_seconds: float | None = None,
        drift_check_seconds: float | None = None,
        drift_workers: ProcessWorkers | None = None,
    ) -> None:
        if pa is None:
            raise ImportError("a copy follower needs pyarrow: install 3tears[snapshot]")
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self._stamp_table = _stamp_table_of(stamp_table, settled, tables)
        self._registry = registry
        self._tables: dict[str, FollowedTable] = {table.name: table for table in tables}
        self._reads = {table.name: _KeyedRead.of(table) for table in tables}
        self._backend = backend
        self._l3 = l3
        self._settled = settled
        self._generation = generation
        self._changes = changes
        self._page_size = page_size
        self._retry_waits = retry_waits
        #: keys heard and not yet applied, by table, each the key's values as text
        self._pending: dict[str, set[tuple[str, ...]]] = {name: set() for name in self._tables}
        #: tables to read again whole at the next apply, and why (:data:`WHOLE_READ_WHYS`): a cold start, a schema
        #: change, or an older writer's whole-table record mid-roll; a second drop before the read is the same read
        self._whole: dict[str, str] = {}
        #: the loads' partitions heard and not yet applied, by table: each the columns its rows hold and their values
        self._partitions: dict[str, list[dict[str, str]]] = {name: [] for name in self._tables}
        #: the loads' partitions staged and not yet applied, by table, put back to be read again if a pass fails
        self._partitions_staged: dict[str, list[dict[str, str]]] = {}
        self._lock = asyncio.Lock()
        self._poked = asyncio.Event()
        self._removers: list[Callable[[], None]] = []
        self._status = FollowerStatus()
        self._listeners: list[Callable[[Any], None]] = []
        #: rows applied since :meth:`take_rows_since_mark` was last called (the checkpoint's trigger)
        self._rows_since_mark = 0
        #: what is staged beside the copy and not yet applied, by table, and the keys each stage was read for
        self._staged: dict[str, StagedChange] = {}
        self._staged_keys: dict[str, set[tuple[str, ...]]] = {}
        #: the keys each stage of a table with no scope replaces (keyed and partition changes alike): what a
        #: checkpoint patches the table's one part by
        self._staged_touched: dict[str, set[tuple[str, ...]]] = {}
        #: the tables staged whole, and why: put back to be read again only when their stage is forgotten
        self._staged_whole: dict[str, str] = {}
        #: the scopes each staged whole read left out because a write held them (their partitions queued, read once
        #: the write commits), with the rows left out of each: a partition of one is never dropped as covered by
        #: that read (:meth:`_stage_partitions`), and each step that reads or keeps them says so
        self._whole_left_out: dict[str, dict[str, int]] = {}
        #: the scopes each table's stage holds rows of (None: rows of no scope, a table without one)
        self._staged_scopes: dict[str, set[str | None]] = {}
        #: the scopes of each key a keyed stage was read for, by table: a scope the stamp moved under while a pass
        #: read (or a write took) leaves the stage by these keys alone, and the rest of the stage is applied
        #: (:meth:`_still`). Kept for keyed stages only: a whole read's or a partition's stage is not split
        self._staged_key_scopes: dict[str, dict[tuple[str, ...], frozenset[str | None]]] = {}
        #: the scopes the last :meth:`_still` left out of the change (their keys put back pending): the label keeps
        #: each at the epoch the copy holds it at, as it keeps a held one (:meth:`_label`)
        self._left_scopes: frozenset[str] = frozenset()
        #: the keys of each table the last :meth:`_still` left out of the change, for :meth:`_leave_out`
        self._leaving: dict[str, set[tuple[str, ...]]] = {}
        #: the mark of the last change applied, and the stamp of each recent one (a reader's snapshot names its mark)
        self._mark = 0
        #: the mark of the change the copy holds now (the last apply that succeeded); None before the first. A pass
        #: that only relabels the copy relabels this one in place (:meth:`_relabel`)
        self._applied_mark: int | None = None
        self._stamps: dict[int, tuple[Any, Mapping[str, str | None]]] = {}
        #: each table's mark when the keys being staged were taken: what the next change applied holds
        self._staging_marks: dict[str, str | None] = {}
        #: each table's mark the copy holds every key of, at or below it: what a checkpoint of the copy records. A table
        #: whose keys a write in progress holds (left pending) keeps the mark it was last current at, so a start from
        #: that checkpoint walks the log from there and hears those keys again
        self._current_marks: dict[str, str | None] = {}
        self._checkpoints = checkpoints
        self._fulls = fulls
        if fulls is not None:
            # an opted-in table's full is on the volume alone: the NATS checkpoint store keeps every other table
            unknown = sorted(set(fulls.tables) - set(self._tables))
            if unknown:
                raise ValueError(f"the volume's fulls name {unknown}, which this follower does not follow")
            both = sorted(set(fulls.tables) & set(checkpoints.table_names if checkpoints is not None else ()))
            if both:
                raise ValueError(f"{both} opt in to delta files: their full is on the volume, never in a checkpoint")
        if checkpoints is not None:
            # every other table the checkpoint holds is derived by the follower's owner: loaded with these, never warmed
            checkpoints.follow(self._tables)
        #: stages of forgotten tables, dropped before the next is staged; the table a pass reads now
        self._dropping: set[str] = set()
        self._reading: str | None = None
        self._checkpointing: asyncio.Task[Any] | None = None
        #: when each table was last current in the registry (its mark caught up, nothing behind): a table the registry
        #: falls behind on is resolved from the change notes since then (:meth:`CollectionRegistry
        #: .resolve_behind_from_notes`), never by a warm
        self._current_at: dict[str, datetime] = {}
        #: each table whose keys the registry named from the change notes (an epoch recorded keys unknown, or a
        #: resolve of being behind), onto where the oldest such notes bracket begins: handed to the checkpoint once an
        #: apply holds the table current past it, so a checkpoint past that epoch is due before the notes can go
        self._noted: dict[str, datetime] = {}
        #: asks for a checkpoint when an epoch applied from the notes makes one due with no change to apply then
        self._checkpoint_wake: asyncio.TimerHandle | None = None
        #: for each table that opts in to delta files, the epochs each key was heard at and not yet applied (None: heard
        #: with no epoch). Kept until an apply holds them, so a key left out of a change, forgotten with its stage or
        #: named again is staged again from the copy's row and every one of its epochs' files, in order, never from L3
        self._epochs_of: dict[str, dict[tuple[str, ...], set[str | None]]] = {
            table.name: {} for table in tables if table.delta_files
        }
        #: the epochs each staged key of such a table was staged from: what an apply of the stage holds
        self._staged_epochs: dict[str, dict[tuple[str, ...], frozenset[str | None]]] = {}
        #: each opted-in table reset or recreated while the follower runs, onto the full it waits for: it is reloaded
        #: from the volume alone, never read from L3 (:meth:`_wait_for_reload`); every other table is followed meanwhile
        self._full_after: dict[str, _FullAfter] = {}
        #: what is heard of each such table while it waits (its keys with their epochs, its partitions, and the
        #: prefixes its files cleared with their epochs): changes to the table the reset left, applied over the full
        #: once it is reloaded, never over the table it replaced
        self._heard_waiting: dict[
            str,
            tuple[
                dict[tuple[str, ...], set[str | None]],
                list[dict[str, str]],
                list[tuple[str, list[dict[str, str]]]],
            ],
        ] = {}
        #: the task looking for each such table's full
        self._reloads: dict[str, asyncio.Task[None]] = {}
        #: for each table that opts in: the key prefixes a delta file cleared, with the epoch its record names, heard
        #: and not yet turned into the keys they reach (:meth:`_keys_of_cleared_prefixes`)
        self._cleared: dict[str, list[tuple[str, list[dict[str, str]]]]] = {
            table.name: [] for table in tables if table.delta_files
        }
        self._recreate = recreate
        self._file_poll_seconds = FILE_POLL_SECONDS if file_poll_seconds is None else file_poll_seconds
        #: each opted-in table waiting for delta files (decision 29), onto each file it waits for by name: nothing of the
        #: table is staged or applied until every one reads (its keys, partitions and prefixes stay held, in order)
        self._file_waits: dict[str, dict[str, _FileWait]] = {}
        #: the task looking for each such table's files and asking the writer for them
        self._file_tasks: dict[str, asyncio.Task[None]] = {}
        #: changes each table waiting for files has heard while it waits (:data:`HELD_WHILE_WAITING_CAP`)
        self._held_counts: dict[str, int] = {}
        self._drift_seconds = _drift_check_seconds(drift_check_seconds)
        self._owns_drift_workers = drift_workers is None
        self._drift_workers = (
            drift_workers if drift_workers is not None else ProcessWorkers(workers=1, name="copy drift")
        )
        #: for each opted-in table, the version the copy holds each state at (decision 32): the full's part's at the
        #: load, then the newest epoch applied of the state; :data:`_VERSION_UNKNOWN` once a change of it applied with
        #: no epoch. A table read whole or by partition since its load holds none (:meth:`_hold_state_versions`)
        self._state_versions: dict[str, dict[str | None, str]] = {}

    # --- what the registry tells it -------------------------------------------------------------

    def _on_row(self, table: str, *, held: bool = True) -> Callable[[CacheInvalidationMessage], None]:
        epochs = self._epochs_of.get(table)

        def heard(message: CacheInvalidationMessage) -> None:
            key = tuple(message.ids)
            waiting = self._heard_waiting.get(table)
            if waiting is not None:
                # a change to the table a reset left: applied over its full once reloaded (:meth:`_reload_from_volume`)
                waiting[0].setdefault(key, set()).add(message.generation)
                return
            self._pending[table].add(key)
            if epochs is not None:
                # the epoch names the record, and the record the file the key's change is applied from
                epochs.setdefault(key, set()).add(message.generation)
            if held and table in self._file_waits:
                self._held_while_waiting(table)
            self._poked.set()

        return heard

    def _held_while_waiting(self, table: str) -> None:
        """count a change heard while ``table`` waits for a delta file; past :data:`HELD_WHILE_WAITING_CAP` stop
        holding them (:meth:`_held_past_cap`).

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        count = self._held_counts.get(table, 0) + 1
        self._held_counts[table] = count
        if not count % _HELD_GAUGE_EVERY:
            _HELD_WHILE_WAITING.labels(table=table).set(count)
        if count > HELD_WHILE_WAITING_CAP:
            self._held_past_cap(table, count)

    def _held_past_cap(self, table: str, count: int) -> None:
        """a table waiting for a delta file holds more changes than :data:`HELD_WHILE_WAITING_CAP`: it waits for the
        writer's next full at or past the newest epoch heard instead (:meth:`_wait_for_reload`), forgetting what it held
        and the files it waited for, which that full holds. Without fulls for the table there is nothing else to wait
        for: it keeps holding, said at WARNING once a cap's worth.

        :param table: the table
        :ptype table: str
        :param count: the changes it holds
        :ptype count: int
        :return: nothing
        :rtype: None
        """
        if self._fulls is None or table not in self._fulls.tables:
            if not count % HELD_WHILE_WAITING_CAP:
                log.warning(
                    "a copy follower's table waiting for a delta file holds past its cap and has no full to wait for "
                    "instead: it keeps holding",
                    extra={"extra_data": {"table": table, "held": count, "cap": HELD_WHILE_WAITING_CAP}},
                )
            return
        log.warning(
            "a copy follower's table waiting for a delta file held past its cap: it stops holding and waits for the "
            "writer's next full instead, reading nothing from L3",
            extra={"extra_data": {"table": table, "held": count, "cap": HELD_WHILE_WAITING_CAP}},
        )
        self._wait_for_reload(table, "held_past_cap", None)
        self._end_held_count(table)

    def _end_held_count(self, table: str) -> None:
        """``table`` holds nothing heard while waiting for a file any more: its count and gauge are zero.

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        if self._held_counts.pop(table, None) is not None:
            _HELD_WHILE_WAITING.labels(table=table).set(0)

    def _on_dropped(self, table: str) -> Callable[[], None]:
        def dropped() -> None:
            # every writer of 0.95.0 records keys or a partition; a whole-table record is an older writer mid-roll
            log.warning(
                "an epoch record names a whole table: a writer older than 3tears 0.95.0 recorded neither keys nor a "
                "partition; this one table is read again whole, and counted",
                extra={"extra_data": {"table": table, "why": "legacy_whole_record", "writer": "3tears < 0.95.0"}},
            )
            self._read_whole(table, "legacy_whole_record")

        return dropped

    def _read_whole(self, table: str, why: str) -> None:
        """read ``table`` again whole at the next apply; drops heard before that read are one read.

        :param table: the table
        :ptype table: str
        :param why: a :data:`WHOLE_READ_WHYS` key
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        self._whole.setdefault(table, why)
        self._pending[table].clear()
        self._epochs_of.get(table, {}).clear()
        self._partitions[table].clear()
        self._poked.set()

    def schema_changed(self, table: str, token: str | None = None) -> None:
        """a table was dropped or recreated (a migration): read it again whole, once, beside the copy. The one change
        a follower reads a table whole for, counted (``copy_follower_whole_reads_total{why="schema_change"}``). A table
        that opts in to delta files and starts from the volume is never read from L3: it waits for the writer's
        rebuild of it (:meth:`_wait_for_reload`).

        :param table: the table
        :ptype table: str
        :param token: the table's generation token once recreated, when known: an opted-in table's full is taken only
            at or past it
        :ptype token: str | None
        :return: nothing
        :rtype: None
        :raises KeyError: when the follower does not keep it
        """
        if table not in self._tables:
            raise KeyError(table)
        if self._fulls is not None and table in self._fulls.tables:
            self._wait_for_reload(table, "schema_change", token)
            return
        log.warning(
            "a followed table's schema changed; it alone is read again whole", extra={"extra_data": {"table": table}}
        )
        self._read_whole(table, "schema_change")

    def _on_partition(self, table: str) -> Callable[[Mapping[str, str]], None]:
        def loaded(partition: Mapping[str, str]) -> None:
            waiting = self._heard_waiting.get(table)
            if waiting is not None:
                # a load into the table a reset left: read once its full is reloaded
                if dict(partition) not in waiting[1]:
                    waiting[1].append(dict(partition))
                return
            if not _key_prefix(self._tables[table], partition):
                # an older writer's partition on columns no key leads: no index reads it exactly, so it is the table's
                self._on_dropped(table)()
                return
            if table in self._whole:
                return  # the whole read to come covers it
            if dict(partition) not in self._partitions[table]:
                self._partitions[table].append(dict(partition))
            self._poked.set()

        return loaded

    def _on_file_prefixes(self, table: str) -> Callable[[Sequence[Mapping[str, str]], str], None]:
        def cleared(prefixes: Sequence[Mapping[str, str]], epoch: str) -> None:
            waiting = self._heard_waiting.get(table)
            if waiting is not None:
                # a clear in the table a reset left: its keys are named over the full once reloaded, never the old copy
                waiting[2].append((epoch, [dict(prefix) for prefix in prefixes]))
                return
            if table in self._whole:
                return  # the whole read to come covers it
            # the keys they reach are named at the next pass (a local read of the copy), before its keys are taken
            self._cleared[table].append((epoch, [dict(prefix) for prefix in prefixes]))
            self._poked.set()

        return cleared

    def _on_reset(self, table: str) -> Callable[[str | None], None]:
        def reset(token: str | None) -> None:
            if self._fulls is not None and table in self._fulls.tables:
                self._wait_for_reload(table, "reset", token)
                return
            log.info(
                "a followed table was reset by an operator; it alone is read again whole, as on a first start",
                extra={"extra_data": {"table": table, "why": "reset"}},
            )
            self._whole[table] = "reset"
            self._read_whole(table, "reset")

        return reset

    def _wait_for_reload(self, table: str, why: str, token: str | None, *, keep: bool = False) -> None:
        """an opted-in table was reset or recreated: never read it from L3 (decisions 12 and 14 of the delta-files
        design: the writer rebuilds it from L3 once and publishes its full). Forget everything the copy heard or staged
        of the table the reset replaced, hold what is heard of it from now on, say in the status that it waits, and look
        for a full holding it at or past the reset (:meth:`_reload_when_published`). The copy keeps serving the table as
        it held it, and keeps following every other table. A second reset while it waits raises the bar to the newer.

        :param table: the table
        :ptype table: str
        :param why: ``reset``, ``schema_change``, ``file_lost`` or ``held_past_cap`` (a table waiting for a file held
            more changes than :data:`HELD_WHILE_WAITING_CAP`: the full at or past the newest heard holds them all)
        :ptype why: str
        :param token: the reset's generation token, when the record named it
        :ptype token: str | None
        :param keep: keep what the copy heard of the table and has not applied, to apply over the full (``file_lost``:
            the table was not replaced, a commit too big to list lost its file and the writer recovers the table,
            decision 28); a reset or a schema change replaced the table, so what was heard of the old one is forgotten
        :ptype keep: bool
        :return: nothing
        :rtype: None
        """
        after = (
            _FullAfter(why=why, token=token, inclusive=True)
            if token is not None
            else _FullAfter(why=why, token=self._registry.generation_marks.highest_heard(table), inclusive=False)
        )
        waiting = self._full_after.get(table)
        if keep and waiting is not None and waiting.token is not None and after.token is not None:
            have, new = split_generation_token(waiting.token), split_generation_token(after.token)
            if have is not None and new is not None and have[0] == new[0] and have[1] >= new[1]:
                after = waiting  # the full it waits for already holds the newer epoch: the bar is never lowered
        log.warning(
            "a followed table that opts in to delta files was reset or recreated: it is reloaded from the writer's "
            "full on the volume once published, never read from L3; the copy serves it as it held it meanwhile",
            extra={"extra_data": {"table": table, "why": why, "token": token, "full_after": after.token}},
        )
        self._full_after[table] = after
        # a stage of the table from before the reset holds the table it replaced: never applied (kept: its keys are
        # put back pending, to be held below)
        self._forget_tables_now([table])
        self._whole.pop(table, None)
        if keep:
            keys, partitions, clears = self._heard_waiting.get(table, ({}, [], []))
            epochs = self._epochs_of.get(table, {})
            for key in self._pending[table]:
                # a key heard with no epoch has no change to apply from a file; its epochs say what it waits for
                if epochs.get(key):
                    keys.setdefault(key, set()).update(epochs[key])
            partitions += [part for part in self._partitions[table] if part not in partitions]
            clears += self._cleared.get(table, [])
            held = (keys, partitions, clears)
        else:
            held = ({}, [], [])
            # the waits were for files of the table the reset replaced
            self._end_file_waits(table, "the table was reset or recreated: its full replaces it")
        self._pending[table].clear()
        self._epochs_of.get(table, {}).clear()
        self._partitions[table].clear()
        self._cleared.get(table, []).clear()
        self._heard_waiting[table] = held
        self._look_for_full(table, why)

    def _wait_for_recovery(self, table: str, reason: str, token: str) -> None:
        """the copy can no longer walk an opted-in table's epoch log (a :data:`_LOG_LOST` reason): never read it from
        L3, by the change notes or whole (decision 28 of the delta-files design: the writer recovers the table with
        one recovery delta folded into its full, and publishes it). The copy waits for a full holding the table under
        the incarnation of the log it reads now and past its mark, then reloads the table alone from it
        (:meth:`_reload_from_volume`) and walks the log from the full's version. Unlike a reset, nothing the copy heard
        is of a table that was replaced: what it heard and had not applied is held, and put back over the full, less
        what the full already holds. The copy keeps serving the table as it held it, and keeps following every other
        table.

        :param table: the table
        :ptype table: str
        :param reason: the registry's behind reason
        :ptype reason: str
        :param token: the table's generation as read now, under the log the copy reads
        :ptype token: str
        :return: nothing
        :rtype: None
        """
        incarnation = token.rpartition(":")[0]
        after = _FullAfter(
            why="lost epoch log",
            token=self._registry.generation_marks.recorded(table),
            inclusive=False,
            incarnation=incarnation,
        )
        log.warning(
            "the epoch log of a followed table that opts in to delta files cannot be walked from the copy's mark: it is "
            "reloaded from the writer's recovered full on the volume once published, never read from L3; the copy "
            "serves it as it held it meanwhile",
            extra={"extra_data": {"table": table, "reason": reason, "read": token, "full_after": after.token}},
        )
        self._full_after[table] = after
        # a stage is put back as the keys it held, which are held with the rest of what was heard
        self._forget_tables_now([table])
        self._whole.pop(table, None)
        epochs = self._epochs_of.get(table, {})
        keys = {key: set(epochs.pop(key, set())) for key in self._pending[table]}
        self._pending[table].clear()
        epochs.clear()
        partitions, self._partitions[table] = list(self._partitions[table]), []
        cleared = self._cleared.get(table, [])
        clears = list(cleared)
        cleared.clear()
        self._heard_waiting[table] = (keys, partitions, clears)
        self._look_for_full(table, after.why)

    def _look_for_full(self, table: str, why: str) -> None:
        """say in the status that ``table`` waits for a full, and look for one (:meth:`_reload_when_published`).

        :param table: the table
        :ptype table: str
        :param why: what it waits for, for the status
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        self._status.waiting_for_full[table] = f"{why}: looking for the writer's full of the table"
        task = self._reloads.get(table)
        if task is None or task.done():
            self._reloads[table] = asyncio.get_running_loop().create_task(
                self._reload_when_published(table), name=f"copy follower reload of {table}"
            )

    def _recovers_from_volume(self, table: str, reason: str, token: str | None) -> bool:
        """whether ``table``, whose log the registry cannot walk (``reason``), waits for a full rather than counting
        behind: a table that opts in to delta files and starts from the volume does (:meth:`_wait_for_recovery`), once
        a generation names the log it reads now; one already waiting goes on waiting.

        :param table: the table
        :ptype table: str
        :param reason: the registry's behind reason, one of :data:`_LOG_LOST`
        :ptype reason: str
        :param token: the table's generation as read now, None when its store holds none
        :ptype token: str | None
        :return: whether it waits for a full
        :rtype: bool
        """
        if self._fulls is None or table not in self._fulls.tables:
            return False
        if table in self._full_after:
            return True
        if token is None:
            return False  # no generation names the log read now: nothing to wait for yet; behind, as before
        self._wait_for_recovery(table, reason, token)
        return True

    async def _reload_when_published(self, table: str) -> None:
        """look for a full holding ``table`` at or past its reset every ``poll_seconds`` until one is published, then
        reload the table from it; the status says why each look found none. Nothing reads L3 meanwhile.

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        fulls = self._fulls
        if fulls is None:
            return  # only a follower starting from the volume waits for a full

        async def looked() -> bool:
            if table not in self._full_after or self._status.phase is FollowerPhase.STOPPED:
                return True
            short: str | None
            try:
                manifest = await asyncio.to_thread(current_full_manifest, fulls.root)
                short = self._full_after[table].short_of(table, manifest)
                if short is None:
                    if await self._reload_from_volume(table, fulls, manifest):
                        return True
                    short = "the table was reset again while its full loaded; looking again"
            except FileNotFoundError as exc:
                short = f"a part of the current full is not on the volume ({exc.filename}); looking again"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed look leaves the copy as it was; the next looks again
                contain(exc, log=log, what="a copy follower could not reload a waiting table from the volume's full")
                short = f"the reload failed ({type(exc).__name__}: {exc}); looking again"
            if table in self._full_after:
                self._status.waiting_for_full[table] = short
            return False

        # the platform's never-give-up loop (threetears.observe), at the volume's own pace: a look every poll_seconds
        await retry_until_done(looked, first_delay=fulls.poll_seconds, max_delay=fulls.poll_seconds)

    async def _reload_from_volume(self, table: str, fulls: VolumeFulls, manifest: FullManifest | None) -> bool:
        """reload ``table`` alone from ``manifest`` in its own change, under the pass lock: its parts staged and applied
        in place of the table the reset replaced, its mark taken as the full's oldest version (the next pass walks the
        log from there, applying each later epoch from its file), and what was heard of it while it waited put back to
        be applied over it.

        :param table: the table
        :ptype table: str
        :param fulls: the volume's fulls
        :ptype fulls: VolumeFulls
        :param manifest: the manifest found
        :ptype manifest: FullManifest | None
        :return: whether it was reloaded (False: reset again since the manifest was judged)
        :rtype: bool
        :raises FileNotFoundError: when a part the manifest names is gone (a newer full swept it)
        """
        async with self._lock:
            after = self._full_after.get(table)
            if after is None or after.short_of(table, manifest) is not None:
                return False
            await self._forget_tables([])  # the stage the reset forgot, among any others
            found = await VolumeFulls(fulls.root, [fulls.tables[table]], poll_seconds=fulls.poll_seconds).stage(
                self._backend, manifest=manifest
            )
            if isinstance(found, FullAbsent):
                return False
            self._mark += 1
            mark = self._mark
            rows = await asyncio.to_thread(self._backend.apply_staged, found.changes, mark=mark)
            self._applied_mark = mark
            floor = found.marks[table]
            self._registry.generation_marks.adopt_generation(table, floor)
            self._current_marks[table] = floor
            self._state_versions[table] = dict(found.versions[table])
            label = self._status.stamp
            self._stamps[mark] = (label, dict(self._current_marks))
            while len(self._stamps) > _STAMPS_KEPT:
                del self._stamps[min(self._stamps)]
            keys, partitions, clears = _past_floor(self._heard_waiting.pop(table), floor, after)
            held = split_generation_token(floor)
            for name, wait in list(self._file_waits.get(table, {}).items()):
                if _held_by_full(wait.epoch, held, after):
                    self._file_read(table, name, why=f"the full at {floor} holds epoch {wait.epoch}")
            del self._full_after[table]
            self._status.waiting_for_full.pop(table, None)
            self._pending[table] |= set(keys)
            epochs = self._epochs_of.setdefault(table, {})
            for key, of in keys.items():
                epochs.setdefault(key, set()).update(of)
            for partition in partitions:
                if partition not in self._partitions[table]:
                    self._partitions[table].append(partition)
            self._cleared.setdefault(table, []).extend(clears)
            self._rows_since_mark += rows
            self._status.applied += 1
            log.info(
                "a reset table was reloaded from the writer's full on the volume",
                extra={
                    "extra_data": {
                        "table": table,
                        "why": after.why,
                        "manifest": found.manifest,
                        "rows": found.rows,
                        "mark": floor,
                        "heard_while_waiting": len(keys),
                    }
                },
            )
            if label is not None:
                for listener in tuple(self._listeners):
                    listener(label)
        self._poked.set()
        return True

    def _on_file_wanted(self, table: str) -> Callable[[WantedFile], None]:
        def wanted(file: WantedFile) -> None:
            if file.keys_unknown and self._fulls is not None and table in self._fulls.tables:
                # its keys were only in the file, so the writer recovers the table (decision 28) and publishes a full
                # holding it; the copy waits for that full, keeping what it heard, and asks the writer once
                self._wait_for_reload(table, "file_lost", file.epoch, keep=True)
                self._wait_for_file(table, file.file, file.epoch, file.why, reach="keys_unknown", watch=False)
                return
            self._wait_for_file(
                table, file.file, file.epoch, file.why, reach="keys_unknown" if file.keys_unknown else "prefixes"
            )

        return wanted

    def _wait_for_file(self, table: str, file: str, epoch: str, why: str, *, reach: str, watch: bool = True) -> None:
        """hold ``table`` until ``file`` reads: nothing of it is staged or applied meanwhile, and nothing read from L3
        (decision 29). The writer is asked once to recreate it, and it is looked for again when the writer answers and
        every ``file_poll_seconds`` (:meth:`_await_files`); every other table is followed as before.

        :param table: the table
        :ptype table: str
        :param file: the file's name, as the epoch's record names it
        :ptype file: str
        :param epoch: the epoch
        :ptype epoch: str
        :param why: ``file_missing`` or ``file_unreadable``
        :ptype why: str
        :param reach: what the file brings once read (:class:`_FileWait`)
        :ptype reach: str
        :param watch: whether the file is looked for (False: the table waits for the writer's recovery full)
        :ptype watch: bool
        :return: nothing
        :rtype: None
        """
        waits = self._file_waits.setdefault(table, {})
        if file in waits:
            return
        waits[file] = _FileWait(epoch=epoch, file=file, why=why, reach=reach, watch=watch)
        _FILE_WAITS.labels(table=table, why=why).inc()
        log.warning(
            "a copy follower's table waits for a delta file: its changes are held, nothing is read from L3, and the "
            "writer is asked to recreate it; every other table is followed meanwhile",
            extra={"extra_data": {"table": table, "file": file, "epoch": epoch, "why": why, "reach": reach}},
        )
        self._say_file_waits(table)
        task = self._file_tasks.get(table)
        if task is None or task.done():
            self._file_tasks[table] = asyncio.get_running_loop().create_task(
                self._await_files(table), name=f"copy follower wait for {table}'s delta file"
            )

    def _say_file_waits(self, table: str) -> None:
        waits = self._file_waits.get(table)
        if waits:
            self._status.waiting_for_file[table] = "; ".join(wait.say() for wait in waits.values())
        else:
            self._status.waiting_for_file.pop(table, None)

    async def _await_files(self, table: str) -> None:
        """look for each file ``table`` waits for until every one reads, asking the writer once for each.

        :param table: the table
        :ptype table: str
        :return: nothing
        :rtype: None
        """

        async def looked() -> bool:
            # a look at each file; once the writer answered, the files are read again at once, else at the next poll
            while self._file_waits.get(table) and self._status.phase is not FollowerPhase.STOPPED:
                answered = False
                for name, wait in list(self._file_waits.get(table, {}).items()):
                    if wait.watch and (wait.why != "key_not_in_file" or wait.answer in _ANSWERED):
                        # a file that reads and lacks a key is read again only once the writer answered for it
                        reach = await self._look_for(table, wait)
                        if reach is not None:
                            await self._hear_file(table, name, wait, reach)
                            self._file_read(table, name, why="the file reads")
                            continue
                    if self._recreate is not None and wait.answer not in _ANSWERED:
                        wait.answer = await self._ask(table, wait)
                        answered = answered or (wait.watch and wait.answer in _ANSWERED)
                        recovering = wait.watch or wait.why == "record_missing"
                        if recovering and wait.answer == "recovering" and self._fulls is not None:
                            if table in self._fulls.tables:
                                # the writer recovers the table (decision 28: a commit too big to list, its keys only in
                                # the file, or a record the log lost): the file never comes, the recovery's full does
                                wait.watch = False
                                self._wait_for_reload(table, "file_lost", wait.epoch, keep=True)
                self._say_file_waits(table)
                if not answered:
                    return False
            return True

        # the platform's never-give-up loop (threetears.observe), at the file poll's own pace
        await retry_until_done(looked, first_delay=self._file_poll_seconds, max_delay=self._file_poll_seconds)

    async def _look_for(self, table: str, wait: _FileWait) -> _WaitedReach | None:
        """read a waited-for file off the volume on a thread: its checksum over every byte, its key columns decoded, and
        the keys it names as text, each once in the order first written.

        Why not the whole file: the wait needs only what the file reaches (its keys and the prefixes it clears); the
        pass that applies it reads it whole, validated, as Arrow (:meth:`_merge_from_files`). Turning its rows into
        Python here cost the loop a key at a time (337,000 keys for a commit too big to list, 2026-11-23 audit).

        :param table: the table
        :ptype table: str
        :param wait: the wait
        :ptype wait: _FileWait
        :return: what the file reaches, or None while it is still missing or unreadable (the wait's why says which)
        :rtype: _WaitedReach | None
        """
        root = delta_files_root()
        if root is None:
            return None
        try:
            return await asyncio.to_thread(_waited_reach, root / wait.file, table)
        except FileNotFoundError:
            wait.why = "file_missing"  # NOSILENT: the wait goes on, said in the status
        except DeltaFileUnreadableError:
            wait.why = "file_unreadable"  # NOSILENT: the wait goes on, said in the status
        except OSError as exc:
            # the volume itself failed the read (an errno): looked for again at the next poll
            contain(exc, log=log, what="a copy follower could not read a delta file it waits for; it looks again")
        return None

    async def _ask(self, table: str, wait: _FileWait) -> str:
        """ask the writer once to recreate a waited-for file; counted by outcome.

        :param table: the table
        :ptype table: str
        :param wait: the wait
        :ptype wait: _FileWait
        :return: the writer's answer (:class:`FileRecreate`)
        :rtype: str
        """
        assert self._recreate is not None  # noqa: S101 -- the caller's condition
        try:
            answer = await self._recreate(table, wait.file, wait.epoch)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a requester that raised is asked again at the next poll, said
            contain(exc, log=log, what="a copy follower's request to recreate a delta file failed; it asks again")
            answer = "failed"
        _RECREATE_ASKS.labels(table=table, outcome=answer).inc()
        log.warning(
            "a copy follower asked the writer to recreate a delta file",
            extra={"extra_data": {"table": table, "file": wait.file, "epoch": wait.epoch, "answer": answer}},
        )
        return answer

    async def _hear_file(self, table: str, name: str, wait: _FileWait, reach: _WaitedReach) -> None:
        """a waited-for load's, clear's or commit too big to list's file reads: its prefixes and keys are heard at its
        epoch, as the registry would have told them, while the table still waits (no pass stages any of it until all
        are heard). A keyed epoch's keys were held pending already.

        Why in runs: a commit too big to list names hundreds of thousands of keys (337,000 in the 2026-11-23 audit),
        each heard on the loop; a run of :data:`_KEYS_PER_TURN`, then the loop answers everything else. A wait ended
        meanwhile (a reset's full replaced the table) stops the hearing: what replaced the table holds the file.

        :param table: the table
        :ptype table: str
        :param name: the file
        :ptype name: str
        :param wait: its wait
        :ptype wait: _FileWait
        :param reach: what the file reaches (:meth:`_look_for`)
        :ptype reach: _WaitedReach
        :return: nothing
        :rtype: None
        """
        if wait.reach not in ("prefixes", "keys_unknown"):
            return
        if reach.prefixes:
            self._on_file_prefixes(table)([dict(prefix) for prefix in reach.prefixes], wait.epoch)
        # the file's own keys: what the wait was for, never counted as held past it
        heard = self._on_row(table, held=False)
        count = len(reach.keys)
        for start in range(0, count, _KEYS_PER_TURN):
            if start:
                await asyncio.sleep(0)
                if self._file_waits.get(table, {}).get(name) is not wait:
                    return
            for key in reach.keys[start : start + _KEYS_PER_TURN]:
                heard(CacheInvalidationMessage(table=table, ids=list(key), generation=wait.epoch, bump_rows=count))

    def _file_read(self, table: str, name: str, *, why: str) -> None:
        """a waited-for file reads (its reach heard first, :meth:`_hear_file`), or a full now holds its epoch: it is no
        longer waited for. Once the table waits for no file, its held changes are applied in order by the next pass.

        :param table: the table
        :ptype table: str
        :param name: the file
        :ptype name: str
        :param why: why the wait ends, for the log
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        waits = self._file_waits.get(table, {})
        wait = waits.pop(name, None)
        if wait is None:
            return
        if not waits:
            self._file_waits.pop(table, None)
            self._end_held_count(table)
        self._say_file_waits(table)
        log.warning(
            "a copy follower's table no longer waits for a delta file; its held changes apply",
            extra={"extra_data": {"table": table, "file": name, "epoch": wait.epoch, "why": why}},
        )
        self._poked.set()

    def _end_file_waits(self, table: str, why: str) -> None:
        """stop waiting for every file of ``table``: what replaces the table holds them (a reset's full, a whole read).

        :param table: the table
        :ptype table: str
        :param why: for the log
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        for name in list(self._file_waits.get(table, {})):
            self._file_read(table, name, why=why)

    def _on_caught_up(self) -> None:
        self._poked.set()

    def _on_stamp_moved(self, message: CacheInvalidationMessage) -> None:
        self._poked.set()

    # --- start -----------------------------------------------------------------------------------

    def register(self) -> None:
        """hear every table's changes through the registry; idempotent.

        The owner follows each table in the registry and schedules its catch-up pass (``CollectionRegistry
        .follow_generation`` and ``threetears.epoch.generation_catchup_tick``), as for any table it follows: the pass
        applies a missed epoch's rows, which reach this follower as heard keys, and pokes it.

        :return: nothing
        :rtype: None
        :raises ValueError: when a table is not followed in the registry
        """
        if self._removers:
            return
        unfollowed = sorted(name for name in self._tables if not self._registry.generation_marks.follows(name))
        if unfollowed:
            raise ValueError(
                f"follow {unfollowed} in the registry and schedule its catch-up pass before a copy follower hears them"
            )
        for name in self._tables:
            self._removers.append(
                self._registry.register_derived_cache(
                    name,
                    on_row=self._on_row(name),
                    on_table_dropped=self._on_dropped(name),
                    on_caught_up=self._on_caught_up,
                    on_partition=self._on_partition(name),
                    on_table_reset=self._on_reset(name),
                    on_file_prefixes=self._on_file_prefixes(name) if name in self._cleared else None,
                    on_file_wanted=self._on_file_wanted(name) if name in self._cleared else None,
                )
            )
        if self._stamp_table is not None:
            # the writer's stamp moved (a write begun, a commit on any replica): a pass reads it; nothing of the stamp
            # table itself is read or kept
            self._removers.append(
                self._registry.register_derived_cache(
                    self._stamp_table, on_row=self._on_stamp_moved, on_table_dropped=self.poke
                )
            )

    def adopt_marks(self, marks: Mapping[str, str | None]) -> None:
        """take each table's mark from a checkpoint: the catch-up walks the log from there.

        Call after :meth:`register` and before the first :meth:`catch_up`, once the checkpoint's rows are the copy's.

        :param marks: each table's generation the checkpoint holds it at
        :ptype marks: Mapping[str, str | None]
        :return: nothing
        :rtype: None
        :raises ValueError: when a table has no mark
        """
        missing = sorted(set(self._tables) - set(marks))
        if missing:
            raise ValueError(f"a checkpoint names no mark for {missing}")
        for name in self._tables:
            self._registry.generation_marks.adopt_generation(name, marks[name])

    async def start(self) -> bool:
        """register, and fill the copy: each table from the checkpoint that holds it (the newest, else the previous),
        whose marks the registry adopts so the next :meth:`catch_up` walks the log from them. A table no checkpoint
        holds at its schema is read whole, alone (why ``cold_start`` or ``schema_change``); with no checkpoint at
        all, every table is (:meth:`warm`).

        A table the checkpoint holds that this follower does not follow (one its owner derives from the followed
        ones) is never warmed here: when the checkpoint cannot give it, its owner derives it again (``settle_all``),
        which a start not wholly from a checkpoint asks for.

        :return: whether the start was wholly from a checkpoint (every table it holds loaded); False asks the owner to
            derive its derived tables again
        :rtype: bool
        """
        self.register()
        if self._fulls is not None:
            return await self._start_from_volume(self._fulls)
        if self._checkpoints is None:
            await self.warm()
            return False
        staged = await self._checkpoints.stage()
        if not staged.loaded:
            await self._warm_tables({name: staged.missing.get(name, "cold_start") for name in self._tables})
            return False
        followed_missing = {name: why for name, why in staged.missing.items() if name in self._tables}
        self._mark += 1
        await asyncio.to_thread(self._backend.apply_staged, staged.changes, mark=self._mark)
        self._applied_mark = self._mark
        # the loaded tables' marks alone: a table still to be read whole has no mark a checkpoint may record
        self._stamps[self._mark] = (staged.stamp, dict(staged.marks))
        self._current_marks = dict(staged.marks)
        for name, mark in staged.marks.items():
            self._registry.generation_marks.adopt_generation(name, mark)
        self._status.stamp = staged.stamp
        if followed_missing:
            await self._warm_tables(followed_missing)
        self._say(
            FollowerPhase.FOLLOWING,
            f"loaded a checkpoint of {staged.rows} rows ({len(followed_missing)} tables read whole, "
            f"{len(staged.missing) - len(followed_missing)} derived tables to derive again); catching up",
        )
        self._poked.set()
        return not staged.missing

    async def _start_from_volume(self, fulls: VolumeFulls) -> bool:
        """:meth:`start` for a follower whose opted-in tables start from the full on the delta-files volume: wait for a
        full holding every one of them (the status says so, and nothing reads L3 meanwhile), stage it, and stage every
        other table from the checkpoint as before; apply both in one transaction, take each table's mark (an opted-in
        table's oldest part version), and read whole only the other tables no checkpoint holds.

        The copy is not labelled with a stamp here: the full's parts are each at their own version, so the copy holds
        one state of every table only once the first catch-up applies what came after each.

        :param fulls: the volume's fulls
        :ptype fulls: VolumeFulls
        :return: False: the copy's derived tables are derived again, since none matches parts at their own versions (and
            False too when the follower was stopped while it waited)
        :rtype: bool
        """
        full = await self._wait_for_full(fulls)
        if full is None:
            return False
        staged = await self._checkpoints.stage() if self._checkpoints is not None else None
        checkpointed = staged is not None and staged.loaded
        marks: dict[str, str | None] = dict(full.marks)
        changes = list(full.changes)
        if staged is not None and checkpointed:
            marks.update(staged.marks)
            changes.extend(staged.changes)
        missing = {
            name: (staged.missing.get(name, "cold_start") if staged is not None else "cold_start")
            for name in self._tables
            if name not in marks
        }
        self._mark += 1
        await asyncio.to_thread(self._backend.apply_staged, changes, mark=self._mark)
        self._applied_mark = self._mark
        self._stamps[self._mark] = (None, dict(marks))
        self._current_marks = dict(marks)
        self._state_versions = {name: dict(held) for name, held in full.versions.items()}
        for name, mark in marks.items():
            self._registry.generation_marks.adopt_generation(name, mark)
        if missing:
            await self._warm_tables(missing)
        self._say(
            FollowerPhase.FOLLOWING,
            f"loaded full {full.manifest} from the volume ({full.rows} rows, {len(full.marks)} tables) and "
            f"{len(marks) - len(full.marks)} tables from a checkpoint ({len(missing)} read whole); catching up",
        )
        self._poked.set()
        return False

    async def _wait_for_full(self, fulls: VolumeFulls) -> StagedFull | None:
        """the current full staged, once the writer has published one holding every opted-in table: until then the
        status says the follower is waiting for the full, and it looks again every ``poll_seconds``. A copy never reads
        an opted-in table whole from L3 (decision 14 of the delta-files design: the writer rebuilds the full from L3
        once). A part the manifest names that is gone (a newer full was published and the old one swept) is looked
        for again the same way.

        :param fulls: the volume's fulls
        :ptype fulls: VolumeFulls
        :return: what was staged; None when the follower was stopped while it waited
        :rtype: StagedFull | None
        """
        staged: StagedFull | None = None

        async def looked() -> bool:
            nonlocal staged
            if self._status.phase is FollowerPhase.STOPPED:
                return True
            try:
                found = await fulls.stage(self._backend)
            except FileNotFoundError as exc:
                found = FullAbsent(f"a part of the current full is not on the volume ({exc.filename}); looking again")
            if isinstance(found, StagedFull):
                staged = found
                return True
            self._say(FollowerPhase.WAITING, f"waiting for the full: {found.reason}")
            return False

        # the platform's never-give-up loop (threetears.observe), at the volume's own pace: a look every poll_seconds
        await retry_until_done(looked, first_delay=fulls.poll_seconds, max_delay=fulls.poll_seconds)
        return staged

    async def _warm_tables(self, whys: Mapping[str, str]) -> None:
        """take these tables' marks at the head and read each whole at the next apply, counted by why.

        :param whys: each table and why (:data:`WHOLE_READ_WHYS`)
        :ptype whys: Mapping[str, str]
        :return: nothing
        :rtype: None
        """
        self._status.phase = FollowerPhase.WARMING
        for why in sorted(set(whys.values())):
            _warms().labels(why=why).inc()
        for name in whys:
            token = await self._generation(name)
            await self._registry.catch_up_generation(name, token, changes=self._changes)
        for name, why in whys.items():
            self._read_whole(name, why)

    async def warm(self) -> None:
        """the first start: take each table's mark at the head, then read every table whole beside the copy and swap
        it in, in one transaction. A change committed while it reads is heard (the marks were taken first) and applied
        by the next :meth:`catch_up`; the copy is labelled by that catch-up, not here.

        :return: nothing
        :rtype: None
        """
        await self._warm_tables(dict.fromkeys(self._tables, "cold_start"))

    def marks(self) -> dict[str, str | None]:
        """each table's mark in the registry now: what a checkpoint of the copy at its stamp records.

        :return: table -> its generation mark
        :rtype: dict[str, str | None]
        """
        return {name: self._registry.generation_marks.recorded(name) for name in self._tables}

    # --- the apply -------------------------------------------------------------------------------

    async def catch_up(self) -> bool:
        """one pass (:meth:`_catch_up`), counted: a failed pass raises as it did, and is counted in the status
        (``failures``, ``behind_since``, FAILING after :data:`FAILING_AFTER` in a row) and in
        ``copy_follower_pass_failures_total`` / ``copy_follower_behind_since_timestamp_seconds``; a good one clears
        them. The copy keeps serving its last change throughout.

        :return: as :meth:`_catch_up`
        :rtype: bool
        """
        try:
            done = await self._catch_up()
        except asyncio.CancelledError, KeyboardInterrupt:
            raise
        except _StagesLostError as exc:
            self._failed("stages_lost", exc)
            raise
        except Exception as exc:
            self._failed("pass", exc)
            raise
        if self._status.failures:
            log.info(
                "copy follower recovered",
                extra={"extra_data": {"failures": self._status.failures, "since": str(self._status.behind_since)}},
            )
        self._status.failures = 0
        self._status.behind_since = None
        _BEHIND_SINCE.set(0)
        return done

    def _failed(self, what: str, exc: BaseException) -> None:
        """count a failed pass: its counter, the status's run of failures and when it began, FAILING at the bound.

        :param what: ``stages_lost`` or ``pass``
        :ptype what: str
        :param exc: what it raised
        :ptype exc: BaseException
        :return: nothing
        :rtype: None
        """
        _PASS_FAILURES.labels(what=what).inc()
        self._status.failures += 1
        if self._status.behind_since is None:
            self._status.behind_since = datetime.now(UTC)
            _BEHIND_SINCE.set(self._status.behind_since.timestamp())
        line = f"a pass failed ({self._status.failures} in a row): {type(exc).__name__}: {exc}"
        if self._status.failures >= FAILING_AFTER:
            self._say(
                FollowerPhase.FAILING,
                f"{line}; serving the copy at {self._status.stamp} since {self._status.behind_since.isoformat()}",
            )
        else:
            self._say(FollowerPhase.WAITING, line)

    async def _catch_up(self) -> bool:
        """apply every key heard at one stamp of the writer's seqlock; label the copy with it.

        At a settled stamp every key heard is applied. While a write is in progress, and the writer names each scope's
        epoch and the scopes the write holds (:class:`Unsettled`), every change of a scope the write does not hold is
        applied and the copy labelled with a :class:`PartialStamp`: a held scope stays at the epoch the copy last
        applied it at, its keys pending until the write commits.

        What a pass reads is staged beside the copy and kept there until a pass ends at a stamp that held still for
        every scope it staged: a pass the stamp moved under keeps its stages, and the next pass reads only the keys
        heard since (a key named again leaves the stage and is read again). A stage holding a scope a write has since
        taken is forgotten (its keys read again once the write commits). A pass whose apply fails keeps every stage
        and tries only the apply again -- unless the backend says the stages are gone (``LookupError``: the worker
        holding them died): then every stage is forgotten and its keys, partitions and whole reads are put back, so the
        next pass reads the change again (never more than the change).

        :return: whether the copy is now labelled with this pass's stamp (False: a write in progress whose record
            names no scope, the stamp moved while it read, or the registry is behind on a table; the copy is as it was)
        :rtype: bool
        """
        async with self._lock:
            began = time.monotonic()
            self._poked.clear()
            first = await self._settled()
            scoped = _scoped(first)
            writing = isinstance(first, Unsettled)
            if writing and scoped is None:
                self._say(FollowerPhase.WAITING, f"a write is in progress: {first.reason}")
                return False
            held = scoped[1] if scoped is not None else frozenset()
            # taken before any generation is read: every epoch at or below a mark read after it is in the copy's reach
            now = datetime.now(UTC)
            behind: dict[str, str] = {}
            for name in self._tables:
                token = await self._generation(name)
                await self._registry.catch_up_generation(name, token, changes=self._changes)
                reason = self._registry.behind_reason(name)
                if reason in _LOG_LOST and self._recovers_from_volume(name, reason, token):
                    # decision 28: the table waits for the writer's recovered full, and holds no other table back
                    continue
                if (
                    reason is not None
                    and reason not in _NOTES_CANNOT
                    and token is not None
                    and name in self._current_at
                ):
                    # never a warm: the keys the change notes name since the table was last current are read instead
                    resolved = await self._registry.resolve_behind_from_notes(name, self._current_at[name], token)
                    reason = resolved if resolved is not None else self._registry.behind_reason(name)
                noted = self._registry.take_notes_resolved(name)
                if noted is not None:
                    self._noted[name] = min(self._noted.get(name, noted), noted)
                if reason is not None:
                    # the registry's reason (BEHIND_REASONS): the status and collection_epoch_behind_total say the same
                    behind[name] = reason
                else:
                    self._current_at[name] = now
            self._status.behind = behind
            if behind:
                self._say(FollowerPhase.BEHIND, f"behind on {sorted(behind)}; serving the copy at {self._status.stamp}")
                return False
            # a stage holding a scope the write now holds (or a table of no scope, while any write is) is read again;
            # a keyed stage loses only that scope's keys, once this pass has read (_still)
            await self._forget_tables(
                [
                    name
                    for name, scopes in self._staged_scopes.items()
                    if writing and self._clashes(scopes, held) and (None in scopes or not self._splittable(name))
                ]
            )
            label = self._label(first, held)
            # a table waiting for a delta file holds what was heard of it: it is not this pass's to apply
            waits = self._file_waits
            nothing = (
                not any(keys for name, keys in self._pending.items() if name not in waits)
                and not self._whole
                and not any(parts for name, parts in self._partitions.items() if name not in waits)
                and not any(queued for name, queued in self._cleared.items() if name not in waits)
            )
            if nothing and not self._staged and label == self._status.stamp:
                self._say(FollowerPhase.FOLLOWING, "current" if not writing else "current to every scope not held")
                return True
            if nothing and not self._staged and self._applied_mark in self._stamps:
                return await self._relabel(first, label, writing=writing, held=held)
            # the keys a cleared prefix reaches join the keys heard; none is left queued past this, so the marks below
            # cover them too
            await self._keys_of_cleared_prefixes()
            # taken with the keys, no await between: every key of an epoch at or below these marks is among them
            self._staging_marks = self.marks()
            pending, whole, partitions = self._take(held, writing=writing)
            try:
                await self._stage(pending, whole, held)
                await self._stage_partitions(partitions, held)
            except LookupError as exc:
                # the worker holding the stages died while this pass staged: what it had not staged is put back, and
                # every stage (this pass's and earlier ones') is forgotten, to be read again
                await self._unstaged(pending, whole, partitions)
                self._stages_lost()
                raise _StagesLostError(f"the stages were lost while staging: {exc}") from exc
            except BaseException:
                # the table being read when it failed is read again; every table staged before it keeps its stage
                await self._unstaged(pending, whole, partitions)
                raise
            second = await self._settled()
            if not self._still(first, second):
                log.info(
                    "copy follower kept its stages for the next pass: the stamp moved while the change was read",
                    extra={
                        "extra_data": {
                            "staged_whole": dict(sorted(self._staged_whole.items())),
                            "staged_tables": sorted(self._staged),
                            "partitions_waiting": {n: len(p) for n, p in sorted(self._partitions.items()) if p},
                            "held": sorted(held),
                        }
                    },
                )
                self._say(FollowerPhase.WAITING, "the stamp moved while the change was read; the next pass tops it up")
                return False
            try:
                await self._leave_out()
            except LookupError as exc:
                self._stages_lost()
                raise _StagesLostError(f"the stages were lost leaving scopes out of the change: {exc}") from exc
            label = self._label(first, held, left=self._left_scopes)
            self._mark += 1
            mark = self._mark
            # a failed apply keeps every stage: the next pass tries only the apply again, never a read -- unless the
            # stages died with the worker holding them: then the change is read again
            try:
                rows = await asyncio.to_thread(self._backend.apply_staged, list(self._staged.values()), mark=mark)
            except LookupError as exc:
                self._stages_lost()
                raise _StagesLostError(f"the stages were lost applying the change: {exc}") from exc
            self._applied_mark = mark
            self._hold_state_versions()
            self._epochs_applied()
            if self._staged_whole or any(self._partitions_staged.values()):
                log.info(
                    "copy follower applied a change",
                    extra={
                        "extra_data": {
                            "mark": mark,
                            "rows": rows,
                            "applied_whole": {
                                n: {"why": w, "left_out": sorted(self._whole_left_out.get(n, {}))}
                                for n, w in sorted(self._staged_whole.items())
                            },
                            "partitions_applied": {n: len(p) for n, p in sorted(self._partitions_staged.items()) if p},
                            "partitions_waiting": {n: len(p) for n, p in sorted(self._partitions.items()) if p},
                            "held": sorted(held),
                        }
                    },
                )
            # a table is current at the marks taken for this pass only when nothing of it is left to read: no key left
            # pending (a scope the write holds, a table of no scope while any write is), no whole read, no partition
            current = [
                name
                for name in self._tables
                if not self._pending[name]
                and name not in self._whole
                and not self._partitions[name]
                and name not in self._full_after
                and name not in self._file_waits
            ]
            for name in current:
                self._current_marks[name] = self._staging_marks.get(name)
            self._stamps[mark] = (label, dict(self._current_marks))
            # what this apply touched, for the checkpoint's incremental parts: a table read whole is every scope
            touched: dict[str, frozenset[str] | None] = {
                name: None
                if name in self._staged_whole
                else frozenset(scope for scope in self._staged_scopes.get(name, ()) if scope is not None)
                for name in self._staged
            }
            # why each table read whole was: the situation a checkpoint writing it whole again is in
            whole_whys = {name: why for name, why in self._staged_whole.items() if name in touched}
            # a table with no scope: the keys it replaced, which a checkpoint patches its one part by
            touched_keys = {
                name: frozenset(self._staged_touched.get(name, set()))
                for name, scopes in touched.items()
                if scopes is not None and self._tables[name].scope is None
            }
            while len(self._stamps) > _STAMPS_KEPT:
                del self._stamps[min(self._stamps)]
            self._staged = {}
            self._staged_keys = {}
            self._staged_touched = {}
            self._staged_whole = {}
            self._whole_left_out = {}
            self._staged_scopes = {}
            self._staged_key_scopes = {}
            self._partitions_staged = {}
            self._rows_since_mark += rows
            self._status.stamp = label
            self._status.applied += 1
            self._status.last_apply_seconds = time.monotonic() - began
            self._say(
                FollowerPhase.FOLLOWING,
                "current" if not writing else f"current to every scope not held; {len(held)} held by the write",
            )
            for listener in tuple(self._listeners):
                listener(label)
            self._checkpoint_when_due(rows, touched, touched_keys, whole_whys, self._noted_held(current))
            return True

    async def _relabel(self, first: Any, label: Any, *, writing: bool, held: frozenset[str]) -> bool:
        """a pass with nothing heard, staged, or to read whole, and a new stamp: the change the copy holds is relabelled
        in place. Its rows are those of the new stamp already, so the backend is asked for nothing: no apply, so on a
        pod's backend no worker round trip, no settle of an empty change, no publish of unchanged rows.

        Why in place: a reader learns its stamp from the mark its snapshot holds (:meth:`reading_marked`), so the new
        label goes on the mark the copy holds now, with each table's mark now (every table is current: nothing is left
        pending). A reader that opened before still reads the same rows; it reads them under the label it read first.
        Everything a labelling pass did besides the apply is kept: the stamp read again and the pass taken only when it
        held still (as :meth:`_still` judges a pass with nothing staged), the listeners told (a reader's epochs moved),
        the checkpoint counted (no rows; it may be due by time or a left-out table).

        Measured (E1, 2026-11-15, S/e1/measure_relabel.py, log S/e1/measure.log): every such pass before this sent
        the backend an empty change. Reports' pod relabels the copy around each write in progress that holds no state
        it is behind on (a PartialStamp in, the settled record out). On reports' shared copy at prod's size (9.52M
        lsf_projections rows, test_follower_change_cost's seed), followed as the pod follows it (settle in the apply
        worker), one empty apply took 181 ms median (144-241 ms, 10 runs; machine load 4.4 at start) for nothing
        changed; followed with no settle, 3.6 ms. Now such a pass costs one stamp read and a dict assignment.
        Counted by ``copy_follower_relabels_total``.

        :param first: what ``settled`` answered first
        :ptype first: Any
        :param label: the stamp the copy is labelled with
        :ptype label: Any
        :param writing: whether a write is in progress
        :ptype writing: bool
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :return: whether the copy is now labelled with the new stamp (False: the stamp moved while it was read)
        :rtype: bool
        """
        marks = self.marks()
        second = await self._settled()
        if not self._still(first, second):
            self._say(FollowerPhase.WAITING, "the stamp moved while the change was read; the next pass tops it up")
            return False
        assert self._applied_mark is not None  # noqa: S101 -- narrowed by the caller (a mark in self._stamps)
        # a table waiting for a delta file keeps the mark it was last current at: its held changes are not in the copy
        self._current_marks.update(
            {
                name: marks.get(name)
                for name in self._tables
                if name not in self._file_waits and name not in self._full_after
            }
        )
        # one assignment: a reader on another thread reads the old entry or the new, never half of one
        self._stamps[self._applied_mark] = (label, dict(self._current_marks))
        self._status.stamp = label
        self._status.relabelled += 1
        _RELABELS.inc()
        self._say(
            FollowerPhase.FOLLOWING,
            "current" if not writing else f"current to every scope not held; {len(held)} held by the write",
        )
        for listener in tuple(self._listeners):
            listener(label)
        self._checkpoint_when_due(0, {}, {}, {}, self._noted_held(self._tables))
        return True

    def _noted_held(self, current: Iterable[str]) -> datetime | None:
        """take the notes times of the tables the copy now holds current: their marks, which a checkpoint records, are
        past every epoch named from the notes so far. A table left pending keeps its own until it is current.

        :param current: the tables whose marks this change moved
        :ptype current: Iterable[str]
        :return: where the oldest of their notes brackets begins; None when none was named from the notes
        :rtype: datetime | None
        """
        held = [self._noted.pop(name) for name in current if name in self._noted]
        return min(held) if held else None

    def _checkpoint_when_due(
        self,
        rows: int,
        touched: Mapping[str, frozenset[str] | None],
        keys: Mapping[str, frozenset[tuple[str, ...]]],
        whys: Mapping[str, str],
        noted: datetime | None = None,
    ) -> None:
        """count a change's rows toward the next checkpoint, and write one beside the follower when it is due.

        :param rows: the rows the change applied
        :ptype rows: int
        :param touched: each table the change applied and the scopes it touched; None for a table read whole
        :ptype touched: Mapping[str, frozenset[str] | None]
        :param keys: each table with no scope the change applied (not whole), and the keys it replaced
        :ptype keys: Mapping[str, frozenset[tuple[str, ...]]]
        :param whys: each table read whole, onto why (``cold_start``, ``reset``, ``schema_change``,
            ``legacy_whole_record``)
        :ptype whys: Mapping[str, str]
        :param noted: where the oldest notes bracket begins of the epochs this change holds that were named from the
            change notes; None when none was
        :ptype noted: datetime | None
        :return: nothing
        :rtype: None
        """
        if self._checkpoints is None:
            return
        self._checkpoints.applied(rows, touched, keys, whys, noted=noted)
        self._checkpoint_if_due()

    def _checkpoint_if_due(self) -> None:
        """write a checkpoint beside the follower when one is due and none is being written; otherwise, when an epoch
        applied from the change notes will make one due, ask again then: after a large commit nothing else may be
        applied before its notes can no longer be read (:meth:`CopyCheckpoints.noted_due_in`).

        :return: nothing
        :rtype: None
        """
        assert self._checkpoints is not None  # noqa: S101 -- narrowed by the callers
        if self._checkpointing is not None and not self._checkpointing.done():
            return  # the write's end asks again
        if self._checkpoint_wake is not None:
            self._checkpoint_wake.cancel()
            self._checkpoint_wake = None
        if self._checkpoints.due():
            self._checkpointing = asyncio.create_task(self._write_checkpoint())
            return
        wait = self._checkpoints.noted_due_in()
        if wait is not None:
            self._checkpoint_wake = asyncio.get_running_loop().call_later(wait, self._checkpoint_if_due)

    async def _write_checkpoint(self) -> None:
        assert self._checkpoints is not None  # noqa: S101 -- narrowed by the caller
        try:
            written = await self._checkpoints.write(self.reading_marked)
        except asyncio.CancelledError, KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - a checkpoint is a start's shortcut; the copy is unaffected
            contain(
                exc,
                log=log,
                what="a copy checkpoint could not be written; the next change that makes one due tries again",
            )
            return
        if written is not None:
            # a change applied while it was written stays counted: asked again once this write is done, so an epoch
            # applied from the notes meanwhile is not left waiting for a change that may not come
            asyncio.get_running_loop().call_soon(self._checkpoint_if_due)

    async def checkpointed(self) -> None:
        """wait for a checkpoint being written, if one is.

        :return: nothing
        :rtype: None
        """
        if self._checkpointing is not None:
            await asyncio.gather(self._checkpointing, return_exceptions=True)

    @staticmethod
    def _clashes(scopes: set[str | None], held: frozenset[str]) -> bool:
        """whether a stage of these scopes may not be applied while a write holds ``held``.

        :param scopes: the scopes a stage holds rows of (None: a table of no scope)
        :ptype scopes: set[str | None]
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :return: whether it clashes
        :rtype: bool
        """
        return None in scopes or bool(scopes & held)

    def _label(self, first: Any, held: frozenset[str], *, left: frozenset[str] = frozenset()) -> Any:
        """the stamp the copy holds once this pass applies: ``first`` itself at a settled stamp with nothing left out;
        otherwise a :class:`PartialStamp` keeping each held scope, and each scope left out of the change
        (:meth:`_still`), at the epoch the copy holds it at now.

        :param first: what ``settled`` answered first
        :ptype first: Any
        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :param left: the scopes left out of this change, their keys read again by the next pass
        :ptype left: frozenset[str]
        :return: the label
        :rtype: Any
        """
        scoped = _scoped(first)
        waiting = self._waiting_scopes() if scoped is not None else frozenset()
        if not isinstance(first, Unsettled) and not left and not waiting:
            return first
        epochs = dict(scoped[0]) if scoped is not None else dict(first.epochs or {})
        before = _scoped(self._status.stamp)
        kept = before[0] if before is not None else {}
        if waiting is _EVERY_SCOPE:
            waiting = frozenset(epochs) | frozenset(kept)
        older = held | left | waiting
        for scope in older:
            epochs.pop(scope, None)
            if scope in kept:
                epochs[scope] = kept[scope]
        return PartialStamp(epochs=epochs, held=older, version=getattr(first, "version", None))

    def _waiting_scopes(self) -> frozenset[str]:
        """the scopes whose rows a waiting table holds back: a table waiting for a delta file (decision 29), for its
        full after a reset or a schema change, or for the writer's recovery of a lost log (decision 28). Its rows in
        those scopes are as the copy held them, older than the stamp, so the label keeps each such scope at the epoch
        the copy holds it at (:meth:`_label`), as a scope a write in progress holds: a reader is never told stale rows
        are current.

        Which scopes: each a held key, partition or cleared prefix names, for a file wait whose change is keyed. A
        wait whose reach is not known (a reset, a lost log, a load's or a big commit's file not yet read, a key whose
        scope is only in its row) holds every scope (:data:`_EVERY_SCOPE`). A table of no scope is labelled by its own
        mark, which a waiting table never moves (:meth:`_catch_up`, :meth:`_relabel`).

        :return: the scopes, or :data:`_EVERY_SCOPE`
        :rtype: frozenset[str]
        """
        scopes: set[str] = set()
        for name in sorted(set(self._file_waits) | set(self._full_after)):
            table = self._tables[name]
            if table.scope is None:
                continue
            if name in self._full_after:
                return _EVERY_SCOPE
            if any(wait.reach != "keys" for wait in self._file_waits[name].values()):
                return _EVERY_SCOPE
            at = table.scope_in_key
            column = table.scope[0]
            if self._pending[name] and at is None:
                return _EVERY_SCOPE
            scopes.update(table.scope_of(key[at]) for key in self._pending[name] if at is not None)
            reached = [*self._partitions[name], *(p for _, cleared in self._cleared.get(name, []) for p in cleared)]
            if any(column not in prefix for prefix in reached):
                return _EVERY_SCOPE
            scopes.update(table.scope_of(prefix[column]) for prefix in reached)
        return frozenset(scopes)

    def _splittable(self, name: str) -> bool:
        """whether a table's stage may lose some keys and keep the rest: a keyed stage, never a whole read's or a
        partition's.

        :param name: the table
        :ptype name: str
        :return: whether it may
        :rtype: bool
        """
        return (
            bool(self._staged_key_scopes.get(name))
            and name not in self._staged_whole
            and not self._partitions_staged.get(name)
        )

    def _take(
        self, held: frozenset[str], *, writing: bool
    ) -> tuple[dict[str, set[tuple[str, ...]]], dict[str, str], dict[str, list[dict[str, str]]]]:
        """take what was heard for this pass, leaving pending what a write in progress holds.

        A key whose scope its key names is left when the write holds that scope; a key whose scope only its row names
        is taken, and left after its row is read (:meth:`_stage_keyed`). A table of no scope is left whole while any
        write is in progress. A table to read whole is taken only at a settled stamp, or on the first start (nothing to
        keep: its held scopes are read once the write commits).

        :param held: the scopes the write holds
        :ptype held: frozenset[str]
        :param writing: whether a write is in progress
        :ptype writing: bool
        :return: the keys, the tables read whole and why, and the partitions taken
        :rtype: tuple[dict[str, set[tuple[str, ...]]], dict[str, str], dict[str, list[dict[str, str]]]]
        """
        pending: dict[str, set[tuple[str, ...]]] = {}
        partitions: dict[str, list[dict[str, str]]] = {}
        whole: dict[str, str] = {}
        for name, table in self._tables.items():
            keys = self._pending[name]
            if writing and table.scope is None:
                continue
            if name in self._whole:
                why = self._whole[name]
                if writing and why not in ("cold_start", "reset"):
                    continue
                whole[name] = self._whole.pop(name)
                self._pending[name] = set()
                self._partitions[name] = []
                # the whole read holds every file the table waited for
                self._end_file_waits(name, f"the table is read whole ({why})")
                continue
            if name in self._file_waits:
                # waiting for a delta file: its keys and partitions stay held, in order, until it reads
                continue
            at = table.scope_in_key
            if held and at is not None:
                taken = {key for key in keys if table.scope_of(key[at]) not in held}
            else:
                taken = set(keys)
            if taken:
                pending[name] = taken
                self._pending[name] = keys - taken
            parts = self._partitions[name]
            column = table.scope[0] if table.scope is not None else None
            if held:
                # a partition naming its scope is judged by it; one whose scope is in its rows only is judged as it is
                # read (:meth:`_stage_partitions`), and put back if a held scope turns up
                go = [p for p in parts if column not in p or table.scope_of(p[column]) not in held]
            else:
                go = list(parts)
            if go:
                partitions[name] = go
                self._partitions[name] = [p for p in parts if p not in go]
        return pending, whole, partitions

    def _still(self, first: Any, second: Any) -> bool:
        """whether a change is left to apply at ``first``: every stage, less the keys of each scope the stamp did not
        hold still for.

        A stamp that names no scope must be the same whole. Otherwise a staged scope that moved between the two reads,
        or that a write in progress now holds, is left out of this change: a keyed stage loses those scopes' keys alone
        (put back pending, read again by the next pass; :meth:`_leave_out` takes them out of the stage), the rest of it
        is applied, and the label keeps each scope left out at the epoch the copy holds it at (:meth:`_label`). A stage
        that cannot be split by key (a whole read's, a partition's), or that every key of would leave, is forgotten
        whole. A table of no scope needs the whole record unmoved and no write in progress: otherwise nothing is
        applied, and a write holding it forgets its stage (its keys read again once the write commits).

        Why per scope: a pass that read for minutes while an LSF load committed a state every ~13 s found some staged
        scope moved on every pass, and kept every stage for the next; the pages froze for the rest of the night
        (reports' burst replay, 2026-11-23: 21 passes in a row kept their stages, none applied after 10:42).

        :param first: what ``settled`` answered before the pass read
        :ptype first: Any
        :param second: what it answered after
        :ptype second: Any
        :return: whether to apply
        :rtype: bool
        """
        self._leaving = {}
        self._left_scopes = frozenset()
        a, b = _scoped(first), _scoped(second)
        if a is None or b is None:
            return bool(second == first)
        (epochs_a, _), (epochs_b, held_b) = a, b
        unsettled = isinstance(second, Unsettled)
        moved = {scope for scope in set(epochs_a) | set(epochs_b) if epochs_a.get(scope) != epochs_b.get(scope)}
        whole_moved = bool(moved) or (unsettled != isinstance(first, Unsettled))
        blocked = moved | (held_b if unsettled else frozenset())
        forget: list[str] = []
        left: set[str] = set()
        whole_rule = False
        for name, scopes in self._staged_scopes.items():
            if None in scopes:
                whole_rule = whole_rule or unsettled or whole_moved
                continue
            hit = {scope for scope in scopes if scope in blocked}
            if not hit:
                continue
            key_scopes = self._staged_key_scopes.get(name)
            if not key_scopes or not self._splittable(name):
                # a whole read's or a partition's stage is never split: the stamp is judged for it as a whole
                whole_rule = True
                continue
            left.update(scope for scope in hit if scope is not None)
            leaving = {key for key, of in key_scopes.items() if of & hit}
            if leaving >= self._staged_keys.get(name, set()):
                forget.append(name)
            elif leaving:
                self._leaving[name] = leaving
        if whole_rule:
            # nothing applies this pass: a stage holding a scope a write took is forgotten (its keys read again once
            # the write commits); one whose scope only moved is kept, its keys heard again topping it up
            self._leaving = {}
            if unsettled:
                self._forget_tables_now(
                    [name for name, scopes in self._staged_scopes.items() if self._clashes(scopes, held_b)]
                )
            return False
        self._forget_tables_now(forget)
        self._left_scopes = frozenset(left)
        # a pass whose every stage was forgotten has nothing to apply; one that staged nothing (a relabel) may
        return not forget or bool(self._staged)

    async def _leave_out(self) -> None:
        """take out of each stage the keys :meth:`_still` left out of the change, and put them back pending: the next
        pass reads them again, at a stamp that holds their scope still.

        :return: nothing
        :rtype: None
        """
        leaving, self._leaving = self._leaving, {}
        for name, keys in leaving.items():
            change = self._staged.get(name)
            if change is None:
                continue
            table = self._tables[name]
            ordered = sorted(keys)
            arrow_keys = pa.table(
                {column: pa.array([key[i] for key in ordered], pa.string()) for i, column in enumerate(table.key)}
            )
            stages = [stage for stage in (change.keys, change.rows) if stage is not None]
            await asyncio.to_thread(self._backend.unstage_keys, stages, name, table.key, arrow_keys)
            self._pending[name] |= keys
            self._staged_keys[name] -= keys
            key_scopes = self._staged_key_scopes[name]
            staged_epochs = self._staged_epochs.get(name, {})
            for key in keys:
                key_scopes.pop(key, None)
                # its epochs stay heard: the next pass stages it again from its files, never from L3
                staged_epochs.pop(key, None)
            self._staged_scopes[name] = {scope for of in key_scopes.values() for scope in of}
        if leaving:
            log.info(
                "copy follower left scopes out of a change: the stamp moved under them while the change was read",
                extra={
                    "extra_data": {
                        "left_keys": {name: len(keys) for name, keys in sorted(leaving.items())},
                        "left_scopes": len(self._left_scopes),
                    }
                },
            )

    def _hold_state_versions(self) -> None:
        """after an apply, before :meth:`_epochs_applied` takes the staged epochs: each state a staged key's row is of
        is held at the newest epoch the key was staged from, if later than the state's; a key staged with no epoch
        that orders it leaves its states' versions unknown. A table read whole or by partition (rows from L3, at no
        epoch a state names) holds no version at all until it is loaded from the volume again. What
        :meth:`check_drift` compares with a part's version (decision 32).

        :return: nothing
        :rtype: None
        """
        for name in self._staged_whole:
            self._state_versions.pop(name, None)
        for name, partitions in self._partitions_staged.items():
            if partitions:
                self._state_versions.pop(name, None)
        for name, of in self._staged_epochs.items():
            held = self._state_versions.get(name)
            if held is None:
                continue
            scopes_of = self._staged_key_scopes.get(name, {})
            for key, epochs in of.items():
                newest = _newest_epoch(epochs)
                for scope in scopes_of.get(key, frozenset({None})):
                    held[scope] = _VERSION_UNKNOWN if newest is None else _later_epoch(held.get(scope), newest)

    def _epochs_applied(self) -> None:
        """after an apply: each staged key's epochs it was staged from are held by the copy, so they are no longer
        heard. An epoch heard for the key since it was staged stays, to be applied by the next pass.

        :return: nothing
        :rtype: None
        """
        staged, self._staged_epochs = self._staged_epochs, {}
        for name, of in staged.items():
            heard = self._epochs_of[name]
            for key, epochs in of.items():
                left = heard.get(key)
                if left is None:
                    continue
                left -= epochs
                if not left:
                    del heard[key]

    def _forget_tables_now(self, names: Sequence[str]) -> None:
        """put back what these tables' stages hold, to be read again; their stages are dropped by the next pass.

        :param names: the tables
        :ptype names: Sequence[str]
        :return: nothing
        :rtype: None
        """
        for name in names:
            self._staged.pop(name, None)
            self._staged_scopes.pop(name, None)
            self._staged_key_scopes.pop(name, None)
            # its keys' epochs stay heard: each key is staged again from its files
            self._staged_epochs.pop(name, None)
            self._pending[name] |= self._staged_keys.pop(name, set())
            self._staged_touched.pop(name, None)
            self._partitions[name].extend(self._partitions_staged.pop(name, []))
            why = self._staged_whole.pop(name, None)
            if why is not None:
                self._whole.setdefault(name, why)
                self._whole_left_out.pop(name, None)
            self._dropping.update((f"{STAGE_PREFIX}{name}__rows", f"{STAGE_PREFIX}{name}__keys"))

    def _stages_lost(self) -> None:
        """the backend lost every stage (the worker holding them died): forget them all, putting back what each held
        to be read again; the next pass drops whatever names are left and reads the change again.

        :return: nothing
        :rtype: None
        """
        self._forget_tables_now(sorted(self._staged))

    async def _forget_tables(self, names: Sequence[str]) -> None:
        """:meth:`_forget_tables_now`, and drop the stages of every table forgotten so far.

        :param names: the tables
        :ptype names: Sequence[str]
        :return: nothing
        :rtype: None
        """
        self._forget_tables_now(names)
        if self._dropping:
            dropping, self._dropping = sorted(self._dropping), set()
            await asyncio.to_thread(self._backend.drop_stage, *dropping)

    async def _unstaged(
        self,
        pending: Mapping[str, set[tuple[str, ...]]],
        whole: Mapping[str, str],
        partitions: Mapping[str, list[dict[str, str]]],
    ) -> None:
        """after a pass failed while it read: put back what it had not staged, and forget the table it was reading.

        :param pending: the keys not yet staged (each table's entry leaves once its stage is complete)
        :ptype pending: Mapping[str, set[tuple[str, ...]]]
        :param whole: the tables not yet read whole
        :ptype whole: Mapping[str, str]
        :param partitions: the partitions not yet staged
        :ptype partitions: Mapping[str, list[dict[str, str]]]
        :return: nothing
        :rtype: None
        """
        for name, keys in pending.items():
            self._pending[name] |= keys
        for name, why in whole.items():
            self._whole.setdefault(name, why)
        for name, parts in partitions.items():
            self._partitions[name].extend(parts)
        reading, self._reading = self._reading, None
        await self._forget_tables([reading] if reading is not None else [])

    async def _stage(
        self, pending: dict[str, set[tuple[str, ...]]], whole: dict[str, str], held: frozenset[str]
    ) -> None:
        """stage each table's change beside the copy, a page at a time, onto what earlier passes staged. Each table's
        entry leaves ``pending`` or ``whole`` once its stage is complete.

        :param pending: the keys taken, by table
        :ptype pending: dict[str, set[tuple[str, ...]]]
        :param whole: the tables to read whole, and why
        :ptype whole: dict[str, str]
        :param held: the scopes a write in progress holds
        :ptype held: frozenset[str]
        :return: nothing
        :rtype: None
        """
        if not self._staged:
            # a stage no change holds is one an earlier process or a failed pass left: never applied
            left = await asyncio.to_thread(self._backend.staged_stages)
            if left:
                await asyncio.to_thread(self._backend.drop_stage, *left)
            self._dropping = set()
        await self._forget_tables([])
        for name, table in self._tables.items():
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            if name in whole:
                self._reading = name
                await self._forget_tables([name])
                self._pending[name] = set()  # a whole read covers every key heard before it
                self._epochs_of.get(name, {}).clear()
                self._staged_epochs.pop(name, None)
                self._whole.pop(name, None)
                self._partitions[name] = []
                why = whole[name]
                # the reason is known here: each is read by its own function, and declared as that situation
                kind: WholeKind
                if why == "schema_change":
                    kind = "schema_change"
                    scopes = await whole_read(
                        self._stage_schema_change(table, rows_stage, held),
                        kind="schema_change",
                        reason="a followed table was dropped or recreated by a migration: that table alone is read again",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                elif why == "legacy_whole_record":
                    kind = "rolling_upgrade"
                    scopes = await whole_read(
                        self._stage_legacy_whole_record(table, rows_stage, held),
                        kind="rolling_upgrade",
                        reason="a writer older than 0.95.0 recorded the whole table mid-roll: that table is read again",
                        until="0.96.0",
                        ruling="Pace 2026-11-12: rolling_upgrade until 0.96.0 (the older writers' whole record is read for one release)",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                else:
                    kind = "cold_start"
                    scopes = await whole_read(
                        self._stage_cold_start(table, rows_stage, why, held),
                        kind="cold_start",
                        reason="the copy holds nothing valid for the table: a first start with no checkpoint, or an operator's reset",
                        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
                        proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
                        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
                    )
                # the apply empties the copy's table under the same declared kind as this read (apply_staged)
                self._staged[name] = StagedChange(
                    table=name, key=table.key, rows=rows_stage, whole=True, whole_kind=kind
                )
                self._staged_whole[name] = why
                self._staged_scopes[name] = scopes
                del whole[name]
                self._reading = None
                continue
            keys = pending.get(name)
            if not keys:
                pending.pop(name, None)
                continue
            self._reading = name
            try:
                staged, scopes, key_scopes = await self._stage_keyed(table, sorted(keys), keys_stage, rows_stage, held)
            except _FileWaitError as wait:
                # nothing of the table is applied while it waits: its keys are put back, its stage (this pass's pages
                # and any earlier pass's) forgotten, every one of them held with its epochs until the file reads
                self._pending[name] |= keys
                self._forget_tables_now([name])
                del pending[name]
                self._reading = None
                self._wait_for_file(
                    name, wait.file, wait.epoch, wait.why, reach="keys", watch=wait.why != "record_missing"
                )
                continue
            if staged:
                was = self._staged.get(name)
                self._staged[name] = StagedChange(
                    table=name,
                    key=table.key,
                    keys=keys_stage,
                    rows=rows_stage,
                    whole=was is not None and was.whole,
                    whole_kind=None if was is None else was.whole_kind,
                )
                self._staged_keys.setdefault(name, set()).update(staged)
                if table.scope is None:
                    self._staged_touched.setdefault(name, set()).update(staged)
                self._staged_scopes.setdefault(name, set()).update(scopes)
                self._staged_key_scopes.setdefault(name, {}).update(key_scopes)
            del pending[name]
            self._reading = None

    def _scopes_of_rows(self, table: FollowedTable, rows: Any) -> list[str | None]:
        """each row's scope, in order (None for a table of no scope).

        :param table: the table
        :ptype table: FollowedTable
        :param rows: rows holding its scope column
        :ptype rows: pyarrow.Table
        :return: the scopes
        :rtype: list[str | None]
        """
        if table.scope is None:
            return [None] * rows.num_rows
        return [table.scope_of(value) for value in rows.column(table.scope[0]).to_pylist()]

    @staticmethod
    def _copy_columns(table: FollowedTable) -> str:
        """the copy's own key and scope columns of a table, selected over ``t``.

        :param table: the table
        :ptype table: FollowedTable
        :return: the select list
        :rtype: str
        """
        columns = list(table.key) + (
            [table.scope[0]] if table.scope is not None and table.scope[0] not in table.key else []
        )
        return ", ".join(f"t.{quote_identifier(c)}" for c in columns)

    def _copy_rows_of_keys(
        self,
        table: FollowedTable,
        columns: Annotated[Sequence[str], DeclaredKey],
        values: Sequence[Sequence[str]],
        *,
        selected: str | None = None,
    ) -> Any:
        """the copy's own key and scope columns of the rows some keys, or key prefixes, name: a local read
        semi-joined on a leading part of the key (the whole key, or a load's prefix), never L3; each row once.

        :param table: the table
        :ptype table: FollowedTable
        :param columns: the key columns named, a leading part of the key in key order
        :ptype columns: Sequence[str]
        :param values: each named column's values, as text, one list a column and one entry a key
        :ptype values: Sequence[Sequence[str]]
        :param selected: the select list over ``t``; None selects the key and scope columns
        :ptype selected: str | None
        :return: the rows
        :rtype: pyarrow.Table
        """
        selected = self._copy_columns(table) if selected is None else selected
        named = ", ".join(f"t.{quote_identifier(column)}" for column in columns)
        cast = ", ".join(f"CAST(k{at + 1} AS {table.key_cast(column)})" for at, column in enumerate(columns))
        statement = (
            f"SELECT {selected} FROM {quote_identifier(table.name)} AS t "  # noqa: S608 - trusted identifiers
            f"WHERE ({named}) IN (SELECT {cast} FROM ({KEYS}))"
        )
        with self._backend.read_snapshot() as cursor:
            # the keys one Arrow table the statement semi-joins (run_joined), as the held rows' read does: bound as
            # Python lists, DuckDB 1.5 looks for pandas once a value, 21.8 s for 50,000 keys of a 200,000-row copy
            # (2026-11-23); 5 ms so
            return run_joined(cursor, statement, [list(column) for column in values], lambda r: r.fetch_arrow_table())

    def _copy_rows_of_prefixes(
        self, table: FollowedTable, columns: Sequence[str], lead: Sequence[str], last: Sequence[str]
    ) -> Any:
        """the copy's own key and scope columns of some key prefixes' rows (one prefix shape: its leading columns'
        values, and the last column's values), read by that prefix of the key (:meth:`_copy_rows_of_keys`).

        :param table: the table
        :ptype table: FollowedTable
        :param columns: the prefix's key columns, in key order
        :ptype columns: Sequence[str]
        :param lead: the values of all but the last, as text
        :ptype lead: Sequence[str]
        :param last: the last column's values, as text
        :ptype last: Sequence[str]
        :return: the rows
        :rtype: pyarrow.Table
        """
        distinct = sorted(set(last))
        return self._copy_rows_of_keys(table, columns, [*([value] * len(distinct) for value in lead), distinct])

    async def _copy_scopes(
        self, table: FollowedTable, keys: Sequence[tuple[str, ...]]
    ) -> dict[tuple[str, ...], set[str]]:
        """the scope of each key's row the copy holds: a local read, never L3.

        :param table: a table whose scope is in its rows only
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :return: each held key's scopes
        :rtype: dict[tuple[str, ...], set[str]]
        """
        columns = [[key[i] for key in keys] for i in range(len(table.key))]
        mine = await asyncio.to_thread(self._copy_rows_of_keys, table, table.key, columns)
        of: dict[tuple[str, ...], set[str]] = {}
        if mine.num_rows:
            copy_keys = zip(*(mine.column(c).to_pylist() for c in table.key), strict=True)
            for key, scope in zip(copy_keys, self._scopes_of_rows(table, mine), strict=True):
                of.setdefault(tuple(str(v) for v in key), set()).add(str(scope))
        return of

    async def _keys_of_cleared_prefixes(self) -> None:
        """name, at its epoch, every key a cleared prefix reaches (:meth:`_on_file_prefixes`): each key of it the copy
        holds (a local read by the prefix, never L3) and each key heard and not yet applied that lies under it. Each
        is then staged from its epochs' files as any key heard (:meth:`_merge_from_files`), deleted at the clearing
        epoch, so a row L3 deleted under the prefix leaves the copy too. Runs until none is queued, so the caller's
        marks, taken with no await after, cover every one heard.

        :return: nothing
        :rtype: None
        """
        while any(queued for name, queued in self._cleared.items() if name not in self._file_waits):
            for name, queued in self._cleared.items():
                if not queued or name in self._file_waits:
                    # a table waiting for a delta file keeps its prefixes queued, in order, until it reads
                    continue
                taken = list(queued)
                queued.clear()
                if name in self._whole:
                    continue  # the whole read to come covers it
                table = self._tables[name]
                heard = self._epochs_of[name]
                for epoch, prefixes in taken:
                    reached: set[tuple[str, ...]] = set()
                    for columns, lead, batch in _prefix_batches(table.key, prefixes):
                        mine = await asyncio.to_thread(self._copy_rows_of_prefixes, table, columns, lead, batch)
                        reached.update(_text_keys(mine, table.key))
                    reached.update(
                        key for key in heard if under_prefix(dict(zip(table.key, key, strict=True)), prefixes)
                    )
                    for key in reached:
                        heard.setdefault(key, set()).add(epoch)
                    self._pending[name] |= reached

    async def _stage_keyed(
        self,
        table: FollowedTable,
        keys: Sequence[tuple[str, ...]],
        keys_stage: str,
        rows_stage: str,
        held: frozenset[str],
    ) -> tuple[set[tuple[str, ...]], set[str | None], dict[tuple[str, ...], frozenset[str | None]]]:
        """stage some keys' rows, read from L3 by key a page at a time -- or, for a table that opts in to delta files,
        merged from their epochs' files onto the copy's rows (:meth:`_merge_from_files`), L3 read only for a key whose
        epoch names no file (a missing or unreadable file raises, and the table waits for it); a key whose row (in L3 or its file, or in the copy) is of a scope a write holds is left
        pending, unstaged.

        :return: the keys staged, the scopes their rows are of, and each staged key's scopes
        :rtype: tuple[set[tuple[str, ...]], set[str | None], dict[tuple[str, ...], frozenset[str | None]]]
        :raises _FileWaitError: when a key's epoch file is missing or unreadable (the caller holds the table)
        """
        at = table.scope_in_key
        #: for a table that opts in to delta files: every key's row merged from its epochs' files, once this pass
        fallbacks: dict[str, dict[str, Any]] = {}
        from_files = await self._merge_from_files(table, keys, fallbacks) if table.delta_files else None
        epochs: dict[tuple[str, ...], frozenset[str | None]] = {}
        staged: set[tuple[str, ...]] = set()
        scopes: set[str | None] = set()
        key_scopes: dict[tuple[str, ...], frozenset[str | None]] = {}
        for start in range(0, len(keys), self._page_size):
            page = keys[start : start + self._page_size]
            chunk = page
            of: dict[tuple[str, ...], set[str]] = {}
            if table.scope is not None and at is None:
                # the scope is in the row only: first the copy's own rows (a local read), so a key whose row is of a
                # scope a write holds waits for it without an L3 read, and is read once, after the write commits
                of = await self._copy_scopes(table, chunk)
                if held:
                    late = {key for key in chunk if of.get(tuple(key), set()) & held}
                    if late:
                        self._pending[table.name] |= late
                        chunk = [key for key in chunk if key not in late]
                        if not chunk:
                            continue
            if from_files is not None:
                rows, epochs = await self._rows_of_page(table, from_files, start, page, chunk)
            else:
                (rows,) = await self._rows_from_l3(table, chunk)
            kept = list(chunk)
            if table.scope is None:
                scopes.add(None)
                key_scopes.update((tuple(key), frozenset({None})) for key in chunk)
            elif at is not None:
                for key in chunk:
                    in_key = table.scope_of(key[at])
                    scopes.add(in_key)
                    key_scopes[tuple(key)] = frozenset({in_key})
            else:
                # the row L3 holds now beside the one the copy holds (a delete's, a move's): a key whose new row is of a
                # held scope (a row new to the copy, or moved into it) waits too
                l3_keys = (
                    list(zip(*(rows.column(c).to_pylist() for c in table.key), strict=True)) if rows.num_rows else []
                )
                for key, scope in zip(l3_keys, self._scopes_of_rows(table, rows), strict=True):
                    of.setdefault(tuple(str(v) for v in key), set()).add(str(scope))
                if held:
                    left = {key for key in chunk if of.get(tuple(key), set()) & held}
                    if left:
                        self._pending[table.name] |= left
                        kept = [key for key in chunk if key not in left]
                        mask = [tuple(str(v) for v in key) not in left for key in l3_keys]
                        rows = rows.filter(pa.array(mask, pa.bool_())) if rows.num_rows else rows
                for key in kept:
                    of_key = of.get(tuple(key), set())
                    scopes.update(of_key)
                    key_scopes[tuple(key)] = frozenset(of_key)
            if not kept:
                continue
            arrow_keys = pa.table(
                {column: pa.array([key[i] for key in kept], pa.string()) for i, column in enumerate(table.key)}
            )
            await asyncio.to_thread(
                self._backend.stage_keys, keys_stage, table.name, table.key, arrow_keys, rows_stage=rows_stage
            )
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, rows, table.key)
            staged.update(kept)
            if table.delta_files:
                self._staged_epochs.setdefault(table.name, {}).update((tuple(key), epochs[key]) for key in kept)
        if fallbacks:
            self._fell_back(table, fallbacks)
        return staged, scopes, {key: key_scopes[key] for key in staged if key in key_scopes}

    async def _rows_from_l3(self, table: FollowedTable, keys: Sequence[tuple[str, ...]]) -> list[Any]:
        """some keys' rows as L3 holds them now, read by key in one statement.

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :return: one Arrow table of the rows that exist, typed as the rail answers them
        :rtype: list[pyarrow.Table]
        """
        read = self._reads[table.name]
        rows: list[Any] = await read_keyed_arrow(
            self._l3,
            table.name,
            keys=KeyBatch(table.key, keys, read.casts),
            types=read.types,
            batch=len(keys),
        )
        return rows

    async def _merge_from_files(
        self, table: FollowedTable, keys: Sequence[tuple[str, ...]], fallbacks: dict[str, dict[str, Any]]
    ) -> _FromFiles:
        """every key's row of a table that opts in to delta files, from its epochs' files: the row the copy holds, with
        every change of it in each epoch it was heard at applied in order, as L3 merged them
        (:class:`~threetears.core.collections.delta_files.HeldDeltaRows`). A key one of whose epochs names no file (an
        older writer's), or whose record is gone, is read from L3 by key instead, and counted in ``fallbacks``; a
        missing or unreadable file is never read around (decision 29): it raises, and the table waits for it.

        Why one merge a pass, on a thread: each file was read into Python rows (``to_pylist``, about ten times its
        Arrow: 291 MB for 200,000 rows of 15 text columns) and every epoch's kept through the pass, and each key's
        changes grouped and merged in a Python loop on the event loop (2026-11-23 audit, P0). Now a file stays Arrow
        and only one is held at a time; the loop's own part is the epochs' snapshot, in runs, and one record a epoch.

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the pass's keys, each its values as text; a key's position is its index here
        :ptype keys: Sequence[tuple[str, ...]]
        :param fallbacks: the keys read from L3 this pass, by why: their count and the files named; filled here
        :ptype fallbacks: dict[str, dict[str, Any]]
        :return: the merged rows
        :rtype: _FromFiles
        :raises _FileWaitError: when an epoch's file is missing or unreadable
        :raises GenerationUnavailableError: when the log cannot be read (the pass fails and is tried again)
        """
        heard_now = self._epochs_of[table.name]
        heard: list[frozenset[str | None]] = []
        alike: dict[frozenset[str | None], frozenset[str | None]] = {}
        for start in range(0, len(keys), _KEYS_PER_TURN):
            if start:
                # a key heard again meanwhile keeps the epoch for the next pass (an apply holds only these)
                await asyncio.sleep(0)
            for key in keys[start : start + _KEYS_PER_TURN]:
                epochs = frozenset(heard_now.get(key, ()))
                heard.append(alike.setdefault(epochs, epochs))
        plan = await asyncio.to_thread(_epoch_plan, heard)
        named: dict[str, str] = {}
        whys: dict[str, str] = {}
        for epoch in plan.order:
            record = None if self._changes is None else await self._changes.change(table.name, epoch)
            if record is None:
                # never read from L3 (decision 29): no record names a file, so the writer recovers the table
                raise _FileWaitError(table.name, _no_record(table.name, epoch), epoch, "record_missing")
            if record.delta_file is None:
                whys[epoch] = "no_file_named"
            else:
                named[epoch] = record.delta_file
        merged = await asyncio.to_thread(self._merged, table, keys, heard, plan, named, whys)
        for why, (count, files) in merged.fallbacks.items():
            fell = fallbacks.setdefault(why, {"keys": 0, "files": set()})
            fell["keys"] += count
            fell["files"] |= files
        return merged

    def _merged(
        self,
        table: FollowedTable,
        keys: Sequence[tuple[str, ...]],
        heard: list[frozenset[str | None]],
        plan: _EpochPlan,
        named: Mapping[str, str],
        whys: Mapping[str, str],
    ) -> _FromFiles:
        """the work of :meth:`_merge_from_files` off the loop: the copy's rows read, each epoch's file read in turn
        (the one before let go first) and applied to the keys heard at it. Blocking: run it on a thread.

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the pass's keys
        :ptype keys: Sequence[tuple[str, ...]]
        :param heard: each key's epochs, by position
        :ptype heard: list[frozenset[str | None]]
        :param plan: each epoch's keys
        :ptype plan: _EpochPlan
        :param named: each epoch's file, as its record names it
        :ptype named: Mapping[str, str]
        :param whys: each epoch whose file cannot be named, and why (a :data:`_DELTA_FILE_FALLBACKS` why)
        :ptype whys: Mapping[str, str]
        :return: the merged rows
        :rtype: _FromFiles
        :raises _FileWaitError: when an epoch's file is missing or unreadable
        """
        from_l3 = np.zeros(len(keys), dtype=bool)
        fallbacks: dict[str, tuple[int, set[str]]] = {}

        def fell(why: str, positions: Any, file: str | None) -> None:
            new = positions[~from_l3[positions]]
            if len(new):
                from_l3[new] = True
                count, files = fallbacks.get(why, (0, set()))
                fallbacks[why] = (count + int(len(new)), files | ({file} if file is not None else set()))

        fell("no_epoch", np.flatnonzero(plan.no_epoch), None)
        for epoch in plan.order:
            if epoch in whys:
                fell(whys[epoch], plan.keys_at[epoch], None)
        types = self._reads[table.name].types
        rows = HeldDeltaRows(table.key, keys, types)
        rows.hold(self._held_arrow(table, [keys[at] for at in np.flatnonzero(~from_l3).tolist()]))
        applied: dict[str, Any] = {}
        held: tuple[str, DeltaTable] | None = None
        for epoch in plan.order:
            if epoch in whys:
                continue
            name = named[epoch]
            # a commit of several records names one file: its changes of a key are applied once, at its first record
            done = applied.setdefault(name, np.zeros(len(keys), dtype=bool))
            positions = plan.keys_at[epoch]
            positions = positions[~from_l3[positions] & ~done[positions]]
            if not len(positions):
                continue
            if held is None or held[0] != name:
                held = None  # the file before is let go first: one file's Arrow held at a time
                held = (name, self._file_of(table, epoch, name))
            # the record named the key at this epoch and its file holds no change of it, nor clears a prefix of it:
            # never guessed, and never read from L3 (decision 29): the writer is asked to recreate it from the record
            if len(rows.apply(held[1], positions)):
                raise _FileWaitError(table.name, name, epoch, "key_not_in_file")
            done[positions] = True
        held = None
        merged, at = rows.rows()
        keep = ~from_l3[at]
        if not keep.all():
            merged, at = merged.filter(pa.array(keep)), at[keep]
        return _FromFiles(rows=merged, at=at, heard=heard, from_l3=from_l3, fallbacks=fallbacks, applied=rows.applied)

    def _file_of(self, table: FollowedTable, epoch: str, name: str) -> DeltaTable:
        """one epoch's delta file, read off the shared volume as Arrow. Blocking: run it on a thread.

        :param table: a table that opts in to delta files
        :ptype table: FollowedTable
        :param epoch: the epoch whose record names it
        :ptype epoch: str
        :param name: the file, as the record names it
        :ptype name: str
        :return: the file
        :rtype: DeltaTable
        :raises _FileWaitError: when it is missing or unreadable: the table waits for it, never read from L3
        """
        root = delta_files_root()
        if root is None:
            raise _FileWaitError(table.name, name, epoch, "file_missing")
        try:
            return read_delta_table(root / name, table=table.name)
        except FileNotFoundError:
            # NOSILENT: the table waits for the writer to recreate it (_FileWaitError), counted and logged at WARNING
            raise _FileWaitError(table.name, name, epoch, "file_missing") from None
        except DeltaFileUnreadableError as exc:
            log.warning(
                "an epoch's delta file is on the volume and cannot be read",
                extra={"extra_data": {"table": table.name, "epoch": epoch, "file": name, "error": str(exc)}},
            )
            raise _FileWaitError(table.name, name, epoch, "file_unreadable") from exc

    def _held_arrow(self, table: FollowedTable, keys: Sequence[tuple[str, ...]]) -> Any:
        """the rows the copy holds of some keys, typed as the L3 rail answers them: a local read, never L3. Blocking.

        Why the copy's: an upsert keeps every column its write left out, and the row's ``date_created``, from the row
        L3 held, which is the row the copy holds before the change (``design-pod-delta-files.md``, decision 19).

        :param table: the table
        :ptype table: FollowedTable
        :param keys: the keys, each its values as text
        :ptype keys: Sequence[tuple[str, ...]]
        :return: the held rows, every column typed as the rail answers it
        :rtype: pyarrow.Table
        """
        types = self._reads[table.name].types
        schema = pa.schema(list(types.items()))
        if not keys:
            return schema.empty_table()
        selected = ", ".join(quote_identifier(column.name) for column in table.columns)
        named = ", ".join(quote_identifier(column) for column in table.key)
        cast = ", ".join(f"CAST(k{at + 1} AS {table.key_cast(column)})" for at, column in enumerate(table.key))
        statement = (
            f"SELECT {selected} FROM {quote_identifier(table.name)} "  # noqa: S608 - trusted identifiers
            f"WHERE ({named}) IN (SELECT {cast} FROM ({KEYS}))"
        )
        with self._backend.read_snapshot() as cursor:
            # the keys one Arrow table the statement semi-joins (run_joined): bound as Python lists, DuckDB 1.5 looks
            # for pandas once a value, 49 s of the 50 s staging 50,000 keys took (2026-11-23); 40 ms so
            mine = run_joined(
                cursor,
                statement,
                [[key[at] for key in keys] for at in range(len(table.key))],
                lambda result: result.fetch_arrow_table(),
            )
        return pa.table([_rail_column(mine.column(name), kind) for name, kind in types.items()], schema=schema)

    async def _rows_of_page(
        self,
        table: FollowedTable,
        merged: _FromFiles,
        start: int,
        page: Sequence[tuple[str, ...]],
        chunk: Sequence[tuple[str, ...]],
    ) -> tuple[Any, dict[tuple[str, ...], frozenset[str | None]]]:
        """a page's rows of the pass's merge (:meth:`_merge_from_files`), and the keys it reads from L3 read by key.

        :param table: the table
        :ptype table: FollowedTable
        :param merged: the pass's merge
        :ptype merged: _FromFiles
        :param start: the page's first key's position
        :ptype start: int
        :param page: the page's keys
        :ptype page: Sequence[tuple[str, ...]]
        :param chunk: those of them staged now (a key of a held scope left out), in order
        :ptype chunk: Sequence[tuple[str, ...]]
        :return: the rows that exist after the change, typed as the rail answers them; and the epochs each key was
            staged from (an apply holds those)
        :rtype: tuple[pyarrow.Table, dict[tuple[str, ...], frozenset[str | None]]]
        """
        if len(chunk) == len(page):
            positions = list(range(start, start + len(page)))
        else:
            kept = set(chunk)
            positions = [start + offset for offset, key in enumerate(page) if key in kept]
        low, high = (int(edge) for edge in np.searchsorted(merged.at, [start, start + len(page)]))
        rows = merged.rows.slice(low, high - low)
        if len(chunk) != len(page):
            rows = rows.filter(pa.array(np.isin(merged.at[low:high], positions)))
        epochs = {key: merged.heard[position] for key, position in zip(chunk, positions, strict=True)}
        from_l3 = [key for key, position in zip(chunk, positions, strict=True) if merged.from_l3[position]]
        if from_l3:
            (read,) = await self._rows_from_l3(table, from_l3)
            rows = pa.concat_tables([rows, read.select(rows.column_names).cast(rows.schema)])
        return rows, epochs

    def _fell_back(self, table: FollowedTable, fallbacks: Mapping[str, Mapping[str, Any]]) -> None:
        """count and say the keys of a table that opts in to delta files a pass read from L3 instead of their files.

        :param table: the table
        :ptype table: FollowedTable
        :param fallbacks: by why, the keys' count and the files named
        :ptype fallbacks: Mapping[str, Mapping[str, Any]]
        :return: nothing
        :rtype: None
        """
        for why, fell in sorted(fallbacks.items()):
            _DELTA_FILE_FALLBACKS.labels(table=table.name, why=why).inc(fell["keys"])
            extra = {
                "extra_data": {
                    "table": table.name,
                    "why": why,
                    "keys": fell["keys"],
                    "files": sorted(fell["files"])[:_FILES_NAMED],
                }
            }
            log.info("copy follower read keys from L3 by key: an older writer's epochs name no delta file", extra=extra)

    async def _stage_partitions(self, partitions: dict[str, list[dict[str, str]]], held: frozenset[str]) -> None:
        """stage each partition's change beside the copy as keyed changes: the partition is replaced. The copy's own
        rows of it leave (a local read) and L3's rows of it go in, read a page at a time as a key range: the
        partitions of one shape are batched, every value of the prefix's last column bound as one array beside its
        leading columns' values, and paged by the key. A load from files and a clear of a key prefix are alike.

        While a write holds some scopes, a partition whose rows (the copy's, or L3's) turn out to be of a held scope
        is put back with everything the table staged, to be read once the write commits.

        :param partitions: each table's partitions taken; each table's entry leaves once staged
        :ptype partitions: dict[str, list[dict[str, str]]]
        :param held: the scopes a write in progress holds
        :ptype held: frozenset[str]
        :return: nothing
        :rtype: None
        """
        for name in list(partitions):
            parts = partitions[name]
            table = self._tables[name]
            held_stage = self._staged.get(name)
            if held_stage is not None and held_stage.whole and name in self._staged_whole:
                # a whole read beside covers the partitions of the scopes it read; never those it left out for a write
                # (queued by :meth:`_stage_whole`, read once the write commits), nor one whose scope its key does not
                # name. Dropped as covered, a left-out scope's rows were missing from the copy for as long as it ran:
                # the whole stage, kept across a pass whose stamp moved, empties the table on apply without them
                # (2026-11-22 replay: IA, SC, SD, TN and TX held none of their L3 rows)
                left_out = self._whole_left_out.get(name, {})
                column = table.scope[0] if table.scope is not None else None
                prefix = table.scope[1] if table.scope is not None else ""
                covered = [
                    p for p in parts if column is not None and column in p and f"{prefix}{p[column]}" not in left_out
                ]
                kept = [p for p in parts if p not in covered]
                log.info(
                    "copy follower dropped the partitions a staged whole read covers",
                    extra={
                        "extra_data": {
                            "table": name,
                            "dropped": sorted(f"{prefix}{p[column]}" for p in covered if column is not None),
                            "kept_to_read": len(kept),
                            "whole_why": self._staged_whole.get(name),
                            "whole_left_out": sorted(left_out),
                            "held": sorted(held),
                        }
                    },
                )
                if not kept:
                    del partitions[name]
                    continue
                # the rest are read as any partition is, into the same stage: the whole read's apply empties the
                # table and puts back its rows and theirs
                parts = kept
                partitions[name] = kept
            self._reading = name
            keys_stage = f"{STAGE_PREFIX}{name}__keys"
            rows_stage = f"{STAGE_PREFIX}{name}__rows"
            types = rail_arrow_types(table.columns)
            selected = ", ".join(quote_identifier(column) for column in types)
            scopes: set[str | None] = set()
            #: a table with no scope: the keys its partitions replace, for the checkpoint's patch
            replaced: set[tuple[str, ...]] = set()
            clashed = False
            #: whether this pass handed the backend a page of keys, of rows: a stage is made by its first page, so a
            #: partition that staged none must not name it (the apply failed on the missing table, every pass after)
            keys_made = rows_made = False
            #: prefixes the copy holds a row of a held scope in: they wait for the write, the rest are read now
            waiting: list[dict[str, str]] = []
            for columns, lead, batch in _prefix_batches(table.key, parts):
                last = columns[-1]
                mine = await asyncio.to_thread(self._copy_rows_of_prefixes, table, columns, lead, batch)
                of = self._scopes_of_rows(table, mine) if mine.num_rows else []
                if held and set(of) & held:
                    values = [str(v) for v in mine.column(last).to_pylist()]
                    late = {value for value, scope in zip(values, of, strict=True) if scope in held}
                    waiting += [
                        dict(p)
                        for p in parts
                        if tuple(sorted(p)) == tuple(sorted(columns))
                        and tuple(p[c] for c in columns[:-1]) == lead
                        and p[last] in late
                    ]
                    batch = [value for value in batch if value not in late]
                    mine = mine.filter(pa.array([value not in late for value in values], pa.bool_()))
                    of = [scope for value, scope in zip(values, of, strict=True) if value not in late]
                    if not batch:
                        continue
                scopes.update(of)
                if mine.num_rows and table.scope is None:
                    replaced.update(_text_keys(mine, table.key))
                if mine.num_rows:
                    await asyncio.to_thread(
                        self._backend.stage_keys,
                        keys_stage,
                        name,
                        table.key,
                        mine.select(list(table.key)),
                        rows_stage=rows_stage,
                    )
                    keys_made = True
                pages = keyset_arrow_pages(
                    self._l3,
                    f"SELECT {selected} FROM {quote_identifier(name)}",  # noqa: S608 - trusted identifiers
                    table.key,
                    types=types,
                    where={c: typed_text(table.kind(c), v) for c, v in zip(columns[:-1], lead, strict=True)} or None,
                    where_any={
                        last: (
                            f"{sql_cast(table.kind(last))}[]",
                            [typed_text(table.kind(last), v) for v in batch],
                        )
                    },
                    page_size=self._page_size,
                )
                async for page in pages:
                    if page.num_rows == 0:
                        continue
                    of = self._scopes_of_rows(table, page)
                    if held and set(of) & held:
                        clashed = True
                        break
                    scopes.update(of)
                    if table.scope is None:
                        replaced.update(_text_keys(page, table.key))
                    await asyncio.to_thread(
                        self._backend.stage_keys,
                        keys_stage,
                        name,
                        table.key,
                        page.select(list(table.key)),
                        rows_stage=rows_stage,
                    )
                    await asyncio.to_thread(self._backend.stage_rows, rows_stage, name, page, table.key)
                    keys_made = rows_made = True
                if clashed:
                    break
            if clashed:
                # a row of a scope the write holds: everything this table staged waits for the write to commit
                log.info(
                    "copy follower put partitions back: a row of a held scope turned up",
                    extra={"extra_data": {"table": name, "partitions": len(parts), "held": sorted(held)}},
                )
                self._partitions[name].extend(parts)
                self._forget_tables_now([name])
                del partitions[name]
                self._reading = None
                continue
            self._partitions[name].extend(waiting)
            self._staged_scopes.setdefault(name, set()).update(scopes)
            if replaced:
                self._staged_touched.setdefault(name, set()).update(replaced)
            self._partitions_staged.setdefault(name, []).extend(dict(p) for p in parts if dict(p) not in waiting)
            was = self._staged.get(name)
            # each stage named only once a page made it, here or by an earlier pass's change of the table: partitions
            # neither the copy nor L3 holds a row of change nothing, and stage nothing
            keys_named = keys_stage if keys_made or (was is not None and was.keys is not None) else None
            rows_named = rows_stage if rows_made or (was is not None and was.rows is not None) else None
            if was is not None or keys_named is not None or rows_named is not None:
                self._staged[name] = StagedChange(
                    table=name,
                    key=table.key,
                    keys=keys_named,
                    rows=rows_named,
                    whole=was is not None and was.whole,
                    whole_kind=None if was is None else was.whole_kind,
                )
            del partitions[name]
            self._reading = None

    @staticmethod
    def _whole_select(table: FollowedTable) -> tuple[str, dict[str, Any], str]:
        """a table's whole read beside the copy: its select of every column, the columns' arrow types, and its lead
        column's array cast (declared, so hashed on YugabyteDB: no declaration names a range-sharded lead).

        :param table: the table
        :ptype table: FollowedTable
        :return: the select, the types and the lead array cast
        :rtype: tuple[str, dict[str, Any], str]
        """
        types = rail_arrow_types(table.columns)
        selected = ", ".join(quote_identifier(column) for column in types)
        select = f"SELECT {selected} FROM {quote_identifier(table.name)}"  # noqa: S608 - trusted identifiers
        return select, types, f"{sql_cast(table.kind(table.read_order[0]))}[]"

    async def _stage_cold_start(
        self, table: FollowedTable, rows_stage: str, why: str, held: frozenset[str]
    ) -> set[str | None]:
        """read a table whole beside the copy because the copy holds nothing valid for it: a first start with no checkpoint
        (``cold_start``) or an operator's reset of the table (``reset``).

        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        select, types, lead = self._whole_select(table)
        pages = whole_read(
            keyset_arrow_pages(
                self._l3, select, table.read_order, types=types, page_size=self._page_size, lead_array=lead
            ),
            kind="cold_start",
            reason="the copy holds nothing valid for the table: a first start with no checkpoint, or an operator's reset",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
            proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
            speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
        )
        return await self._stage_whole(table, rows_stage, why, held, pages)

    async def _stage_schema_change(
        self, table: FollowedTable, rows_stage: str, held: frozenset[str]
    ) -> set[str | None]:
        """read a table whole beside the copy because a migration dropped or recreated it (``schema_change``).

        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        select, types, lead = self._whole_select(table)
        pages = whole_read(
            keyset_arrow_pages(
                self._l3, select, table.read_order, types=types, page_size=self._page_size, lead_array=lead
            ),
            kind="schema_change",
            reason="a followed table was dropped or recreated by a migration: that table alone is read again",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade",
            proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
            speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
        )
        return await self._stage_whole(table, rows_stage, "schema_change", held, pages)

    async def _stage_legacy_whole_record(
        self, table: FollowedTable, rows_stage: str, held: frozenset[str]
    ) -> set[str | None]:
        """read a table whole beside the copy because a writer older than 0.95.0 recorded the whole table mid-roll
        (``legacy_whole_record``; gone by 0.96.0).

        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        select, types, lead = self._whole_select(table)
        pages = whole_read(
            keyset_arrow_pages(
                self._l3, select, table.read_order, types=types, page_size=self._page_size, lead_array=lead
            ),
            kind="rolling_upgrade",
            reason="a writer older than 0.95.0 recorded the whole table mid-roll: that table is read again",
            until="0.96.0",
            ruling="Pace 2026-11-12: rolling_upgrade until 0.96.0 (the older writers' whole record is read for one release)",
            proof="packages/core/tests/unit/collections/test_copy_follower.py::test_stage_whole_is_reached_only_at_a_cold_start_a_reset_a_schema_change_or_a_legacy_whole_record",
            speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_stage_whole_speed_at_real_size",
        )
        return await self._stage_whole(table, rows_stage, "legacy_whole_record", held, pages)

    async def _stage_whole(
        self,
        table: FollowedTable,
        rows_stage: str,
        why: str,
        held: frozenset[str],
        pages: AsyncIterator[Any],
    ) -> set[str | None]:
        """stage a table's whole read (each situation's own function declares it) beside the copy, counted by why. While a write holds
        some scopes (a first start beside a long write) their rows are left out and each is read as a partition once
        the write commits.

        :param pages: the table's keyset pages, read whole
        :ptype pages: AsyncIterator[Any]
        :return: the scopes of the rows staged
        :rtype: set[str | None]
        """
        _WHOLE_READS.count(why, table=table.name)
        if why != "cold_start":
            _warms().labels(why=why).inc()
        scopes: set[str | None] = set()
        left_out: dict[str, int] = {}
        staged_rows = 0
        await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, pa.table({}), table.key)
        async for page in pages:
            of = self._scopes_of_rows(table, page)
            if held and table.scope is not None:
                for scope in of:
                    if scope is not None and scope in held:
                        left_out[scope] = left_out.get(scope, 0) + 1
                mask = [scope not in held for scope in of]
                page = page.filter(pa.array(mask, pa.bool_()))
                of = [scope for scope, keep in zip(of, mask, strict=True) if keep]
            scopes.update(of)
            staged_rows += page.num_rows
            await asyncio.to_thread(self._backend.stage_rows, rows_stage, table.name, page, table.key)
        queued: list[str] = []
        if held and table.scope is not None:
            column, prefix = table.scope
            for scope in held:
                if scope.startswith(prefix):
                    self._partitions[table.name].append({column: scope[len(prefix) :]})
                    queued.append(scope)
                    left_out.setdefault(scope, 0)
        self._whole_left_out[table.name] = left_out
        log.info(
            "copy follower staged a whole read",
            extra={
                "extra_data": {
                    "table": table.name,
                    "why": why,
                    "rows_staged": staged_rows,
                    "scopes_staged": len(scopes),
                    "held": sorted(held),
                    "left_out_rows": dict(sorted(left_out.items())),
                    "queued_partitions": sorted(queued),
                }
            },
        )
        return scopes or ({None} if table.scope is None else set())

    # --- running ---------------------------------------------------------------------------------

    # --- drift, against the full (decision 32) ------------------------------------------------------

    async def check_drift(self) -> dict[tuple[str, str | None], str]:
        """check each state of each opted-in table against the current full: a state the copy holds at exactly its
        part's version is read from the copy (a local read, never L3), fingerprinted in a worker process
        (:func:`~threetears.core.collections.volume_fulls.rows_fingerprint`) and compared with the fingerprint the
        part's manifest entry carries. Sets ``copy_follower_drift_parts{table,state}`` (1 drifted, 0 equal), counts
        each part's outcome (``copy_follower_drift_checks_total``) and logs a drift at WARNING. :meth:`run` calls it
        every ``drift_check_seconds``.

        :return: each part's outcome, by (table, state)
        :rtype: dict[tuple[str, str | None], str]
        """
        if self._fulls is None:
            return {}
        manifest = await asyncio.to_thread(current_full_manifest, self._fulls.root)
        if manifest is None:
            return {}
        outcomes: dict[tuple[str, str | None], str] = {}
        for name in sorted(self._fulls.tables):
            entry = manifest.tables.get(name)
            if entry is None:
                continue
            kinds = fingerprint_kinds(self._reads[name].types)
            for part in entry.parts:
                outcome = await self._check_part(self._tables[name], part, kinds, manifest.name)
                _DRIFT_CHECKS.labels(table=name, outcome=outcome).inc()
                outcomes[(name, part.scope)] = outcome
        return outcomes

    def _drift_current(self, name: str) -> bool:
        """whether a table holds no change it has heard and not applied: nothing pending, staged, queued, waiting or
        behind, so each state is exactly at the version :meth:`_hold_state_versions` says.

        :param name: the table
        :ptype name: str
        :return: whether it is current
        :rtype: bool
        """
        return (
            not self._pending[name]
            and not self._epochs_of.get(name)
            and name not in self._whole
            and not self._partitions[name]
            and not self._cleared.get(name)
            and name not in self._staged
            and name not in self._full_after
            and name not in self._file_waits
            and name not in self._status.behind
        )

    async def _check_part(
        self, table: FollowedTable, part: FullPart, kinds: tuple[tuple[str, str], ...], manifest: str
    ) -> str:
        """one part's drift check (:meth:`check_drift`).

        :param table: the table
        :ptype table: FollowedTable
        :param part: the current full's part
        :ptype part: FullPart
        :param kinds: the copy's columns' fingerprint kinds, by the types the rail answers them as
        :ptype kinds: tuple[tuple[str, str], ...]
        :param manifest: the full's name, for the log line
        :ptype manifest: str
        :return: the outcome (``copy_follower_drift_checks_total``)
        :rtype: str
        """
        if part.fingerprint is None:
            return "no_fingerprint"
        if not part.fingerprint.startswith(f"{fingerprint_columns(kinds)}-"):
            return "other_columns"
        keys = await asyncio.to_thread(self._part_keys, table, part)
        # under the pass lock: no apply moves the state between its version's check and the snapshot read
        async with self._lock:
            if not self._drift_current(table.name):
                return "not_current"
            held = self._state_versions.get(table.name, {}).get(part.scope)
            if held == _VERSION_UNKNOWN:
                return "version_unknown"
            if held != part.version:
                return "skipped"
            ipc = await asyncio.to_thread(self._state_ipc, table, part, keys)
        mine = await self._drift_workers.run(fingerprint_ipc, ipc, kinds)
        state = part.scope if part.scope is not None else ""
        if mine == part.fingerprint:
            _DRIFT_PARTS.labels(table=table.name, state=state).set(0)
            return "equal"
        _DRIFT_PARTS.labels(table=table.name, state=state).set(1)
        log.warning(
            "a copy's rows of a state differ from the full's at the same version",
            extra={
                "extra_data": {
                    "table": table.name,
                    "state": part.scope,
                    "version": part.version,
                    "manifest": manifest,
                    "full": part.fingerprint,
                    "copy": mine,
                }
            },
        )
        return "drift"

    def _part_keys(self, table: FollowedTable, part: FullPart) -> list[list[str]] | None:
        """the keys a part holds, each key column's values as text, for a state the copy cannot read by a prefix of
        its key; None for one it can (its scope column leads the key). Read off the volume (an immutable file), never
        L3. Blocking: run it on a thread.

        :param table: the table
        :ptype table: FollowedTable
        :param part: the part
        :ptype part: FullPart
        :return: one list a key column, one entry a key; None when the state is read by its key prefix
        :rtype: list[list[str]] | None
        """
        if part.scope is not None and table.scope is not None and table.scope_in_key == 0:
            return None
        assert self._fulls is not None  # noqa: S101 -- a drift check runs only for a follower started from the volume
        keys = read_full_part(self._fulls.root, part).select(list(table.key))
        return [[key_text(value) for value in keys.column(column).to_pylist()] for column in table.key]

    def _state_ipc(self, table: FollowedTable, part: FullPart, keys: list[list[str]] | None) -> bytes:
        """the copy's rows of one state, as an Arrow IPC stream for the worker that fingerprints them: a local read
        of the copy by key, never L3 and never a scan. A state whose scope column leads the key is read by that prefix
        (every row the copy holds of it); any other (a table keyed without its state, or one with no scope) by the keys
        its part holds, so a row the copy holds of the state and the part does not is not seen. Blocking: run it on a
        thread.

        :param table: the table
        :ptype table: FollowedTable
        :param part: the part
        :ptype part: FullPart
        :param keys: :meth:`_part_keys`
        :ptype keys: list[list[str]] | None
        :return: the rows
        :rtype: bytes
        """
        selected = ", ".join(f"t.{quote_identifier(column.name)}" for column in table.columns)
        if keys is None:
            assert part.scope is not None and table.scope is not None  # noqa: S101 -- _part_keys said so
            prefix = [[part.scope[len(table.scope[1]) :]]]
            rows = self._copy_rows_of_keys(table, table.key[:1], prefix, selected=selected)
        else:
            rows = self._copy_rows_of_keys(table, table.key, keys, selected=selected)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, rows.schema) as writer:
            writer.write_table(rows)
        return bytes(sink.getvalue().to_pybytes())

    def _drift_checks(self) -> PeriodicTask | None:
        """the periodic drift check :meth:`run` keeps, started; None when the copy starts from no full or the check
        is off. Its first check falls at a random point within one period, so the pods of a deployment, started
        together, do not all read and hash at once.

        :return: the task
        :rtype: PeriodicTask | None
        """
        if self._fulls is None or self._drift_seconds <= 0:
            return None
        task = PeriodicTask(
            self.check_drift,
            interval=self._drift_seconds,
            name="copy drift check",
            logger=log,
            first_delay=secrets.SystemRandom().uniform(0, self._drift_seconds),
            failure_message="a copy's drift check failed; the next period checks again",
        )
        task.start()
        return task

    async def run(self) -> None:
        """apply each change as it is heard, until cancelled: a pass whenever the registry hears a key, catches up, or
        :meth:`poke` is called (the writer's stamp moved). Never on a timer while passes succeed; after a failed pass,
        the next is tried after a wait doubling from the first of ``retry_waits`` to the longest (or sooner when
        poked), FAILING in the status after :data:`FAILING_AFTER` in a row, until one succeeds. Beside it, the drift
        check every ``drift_check_seconds`` (:meth:`check_drift`), ended with it.

        :return: nothing
        :rtype: None
        """
        drift = self._drift_checks()
        try:
            await self._passes()
        finally:
            if drift is not None:
                await drift.stop()
            if self._owns_drift_workers:
                await self._drift_workers.close()

    async def _passes(self) -> None:
        """:meth:`run`'s passes, until cancelled.

        :return: nothing
        :rtype: None
        """
        while True:
            if self._status.failures:
                # a failed pass is tried again after a wait (doubling to the longest), or sooner when poked: the keys
                # it left pending are never left waiting for a change that may not come
                first, longest = self._retry_waits
                wait = min(longest, first * 2 ** (self._status.failures - 1))
                try:
                    await asyncio.wait_for(self._poked.wait(), timeout=wait)
                except TimeoutError:  # NOSILENT: the wait ran out with no poke: that is the retry's cue, not a failure
                    pass
            else:
                await self._poked.wait()
            try:
                await self.catch_up()
            except asyncio.CancelledError, KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed pass leaves the copy as it was; the next tries again
                contain(
                    exc, log=log, what="a copy follower's pass failed; the copy is as it was and the keys stay pending"
                )

    def poke(self) -> None:
        """ask for a pass: the writer's stamp moved.

        :return: nothing
        :rtype: None
        """
        self._poked.set()

    def on_change(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        """call ``listener`` with the stamp after every change applied.

        :param listener: called with the stamp, on the loop; must not raise
        :ptype listener: Callable[[Any], None]
        :return: a call that removes it
        :rtype: Callable[[], None]
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def stop(self) -> None:
        """stop hearing the registry.

        :return: nothing
        :rtype: None
        """
        for remove in self._removers:
            remove()
        self._removers = []
        if self._checkpointing is not None:
            self._checkpointing.cancel()
        if self._checkpoint_wake is not None:
            self._checkpoint_wake.cancel()
            self._checkpoint_wake = None
        for task in self._reloads.values():
            task.cancel()
        self._reloads = {}
        for task in self._file_tasks.values():
            task.cancel()
        self._file_tasks = {}
        self._status.phase = FollowerPhase.STOPPED

    # --- reading ---------------------------------------------------------------------------------

    @property
    def stamp(self) -> Any:
        """the stamp the copy holds every table at; None until the first change is applied.

        :return: the stamp
        :rtype: Any
        """
        return self._status.stamp

    @contextmanager
    def reading(self) -> Iterator[tuple[Any, Any]]:
        """a cursor reading one state of every table, and the stamp it holds them at.

        The stamp is read from the snapshot itself (the change's mark, committed with it), so it is exactly the
        change the cursor reads, whatever is applied meanwhile.

        :return: ``(cursor, stamp)``; the stamp is None before the first change
        :rtype: Iterator[tuple[duckdb.DuckDBPyConnection, Any]]
        """
        with self.reading_marked() as (cursor, stamp, _marks):
            yield cursor, stamp

    @contextmanager
    def reading_marked(self) -> Iterator[tuple[Any, Any, Mapping[str, str | None]]]:
        """:meth:`reading`, with each table's epoch mark the read's change holds: what a checkpoint of it records.

        :return: ``(cursor, stamp, marks)``; the stamp None and the marks empty before the first change
        :rtype: Iterator[tuple[duckdb.DuckDBPyConnection, Any, Mapping[str, str | None]]]
        """
        with self._backend.read_snapshot() as cursor:
            mark = self._backend.staged_mark(cursor)
            stamp, marks = self._stamps.get(mark, (None, {})) if mark is not None else (None, {})
            yield cursor, stamp, marks

    def take_rows_since_mark(self) -> int:
        """the rows applied since the last call, and start counting again: what a checkpoint's trigger weighs.

        :return: the rows
        :rtype: int
        """
        rows, self._rows_since_mark = self._rows_since_mark, 0
        return rows

    def status(self) -> FollowerStatus:
        """what the follower is doing, with each table's pending keys counted now.

        :return: the status
        :rtype: FollowerStatus
        """
        self._status.pending = {name: len(keys) for name, keys in self._pending.items() if keys}
        return self._status

    def _say(self, phase: FollowerPhase, detail: str) -> None:
        if (self._status.phase, self._status.detail) != (phase, detail):
            log.info("copy follower", extra={"extra_data": {"phase": phase.value, "detail": detail}})
        self._status.phase = phase
        self._status.detail = detail


if TYPE_CHECKING:

    def _duckdb_is_a_follower_backend(backend: DuckDBBackend) -> FollowerBackend:
        """mypy's proof that the DuckDB L1 is a copy follower's backend: it fails here if they part."""
        return backend
