"""A copy follower's checkpoint: its copy at one stamp, in the pod's own Object Store (L2), so a start loads it in
seconds and walks the epoch change log from there. L3 is read whole only for a table no checkpoint holds (a true cold
start) or an operator's reset.

**Where.** The pod's own Object Store and pointer bucket (:func:`threetears.nats.object_store_requests.bind_pod_object_store`):
memory storage, three replicas, declared and swept by the hub. Nothing is on disk. A NATS restart that loses them is
a cold start: the next start reads L3 whole, and the first follower current again writes a new checkpoint.

**A manifest of parts.** The pointer (``{name}-ckpt``) holds a manifest: for each table its epoch mark, its schema
digest and its parts. A part is Arrow IPC (zstd) of at most ``part_rows`` rows of one table and one scope
(:attr:`~threetears.core.collections.copy_follower.FollowedTable.scope`; a table with no scope is one key), named
``{base}/{seq}/{owner}/{table}/{scope}/{n}``. Marks are per table, so each table loads on its own.

**Loading** (:meth:`CopyCheckpoints.stage`) stages each table on its own. A table whose parts do not read back loads
from the previous checkpoint, which the pointer keeps beside the newest. A table no checkpoint holds is
``cold_start``; one whose schema digest differs from the copy's is ``schema_change``. Either is left out, the rest
load, and the follower reads only those whole (:class:`StagedCheckpoint`).

**Writing** (:meth:`CopyCheckpoints.write`) rewrites only the (table, scope) parts the follower changed since this
replica's last checkpoint or load (:meth:`CopyCheckpoints.applied`) and names the unchanged parts' objects again: a
scope's rows are read from the copy by its scope column, and a table with no scope (one part, at most ``part_rows``
rows) is patched by key, its part read back from the Object Store, the touched keys dropped and the copy's rows of them
added. A table is read whole from the copy only when no checkpoint holds it (its cold start), its schema digest
changed, or the follower read it whole from L3 (a cold start, reset or schema change), each counted in
``copy_checkpoint_whole_writes_total{table, why}``. Every read of the copy and its parts' encoding is a job over the state the
write pins (``copy_reads.PinnedCopy``): over a copy published as shared Arrow, in a query worker, so a write opens no
DuckDB connection or cursor in the serving process. A replica whose last checkpoint's parts are no longer named by the
newest or the previous checkpoint (another replica wrote twice since) writes no checkpoint: it never rewrites whole
what the newest already holds. A write reads and encodes all its new parts before it puts the first, and asks the
Object Store for their room together: refused up front (``ObjectStoreFullError``, nothing written) when it has none,
never part way. A write runs under a lock and a write claim in the pointer bucket and moves the pointer
by compare-and-set. A failed or beaten write retires only its own new objects and keeps what it was asked to write for
the next one. When the pointer moves, the checkpoint before the previous one is retired, never an object the newest
or the previous names.

**Two code versions** share the one pointer during a roll. Each table's entry carries its schema digest, so a
version loads only the tables it reads the same way (from the newest, or the previous) and reads the rest whole. A
writer rewrites the tables it holds and keeps every other entry as it found it: a table it does not follow, or one it
has not read again yet.

**Forgetting tables** (:meth:`CopyCheckpoints.forget`) is an operator's act, never automatic: it drops the named
tables from the newest and the previous checkpoint and retires their parts, so the next start finds no checkpoint
holding them and reads each whole as its cold start, its mark taken at the log's head, while every other table loads.
It is how a follower stuck behind on a table (an epoch the log cannot give: ``keys_unknown``, ``aged_out``) whose
every replica restarts from a checkpoint older than that epoch is brought back: a start from the checkpoint walks the
log from the stale mark again, a start that sees the table for the first time does not.

**When.** When the rows applied since the last checkpoint reach ``ratio`` (default half) of the rows it holds, once
after the first warm, once a table left out of the load is read again, and once the oldest epoch the follower applied
from the change notes (a commit too large to record its keys) is a quarter note fence old (:meth:`CopyCheckpoints.due`):
a start from a checkpoint older than such an epoch needs its notes again, and they answer for at least half a fence.
Writes are counted in ``copy_checkpoint_writes_total{kind, outcome}``; how long such an epoch has waited for a
checkpoint, in ``copy_checkpoint_notes_unsaved_age_seconds{base}``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import weakref
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from threetears.interfaces.keyed.rows import KeyColumn
from threetears.observe import cancel_and_wait, contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.declared import whole_read
from threetears.core.operator import operator_reset
from threetears.core import keyed
from threetears.core.keyed import InlineDuckDB, WholeReads, run_inline
from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange
from threetears.core.cache.duckdb_binds import KEYS, run_joined
from threetears.core.frozen_state import frozen, with_entries, with_entry, without_entries, without_entry
from threetears.core.ddl.change_notes import DEFAULT_NOTE_FENCE
from threetears.core.sql_fragments import quote_identifier, unnest_key_join

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable
    from threetears.core.collections.copy_reads import PinnedCopy

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
except ImportError:  # NOSILENT: the snapshot extra is optional; a checkpoint says so when built
    pa = None
    pc = None
    ipc = None

__all__ = [
    "COLD_START",
    "SCHEMA_CHANGE",
    "Checkpoint",
    "CheckpointTable",
    "CopyCheckpoints",
    "StagedCheckpoint",
]

log = get_logger(__name__)

#: a checkpoint is due when the rows applied since the last reach this share of the rows it holds
_DEFAULT_RATIO: Final = 0.5

#: the most rows one part of a checkpoint holds: what a writer or a loader holds of it at a time
_DEFAULT_PART_ROWS: Final = 250_000

#: how long a writer's claim and lock last unrenewed
_CLAIM_TTL: Final = timedelta(minutes=2)

#: how often a write renews its claim and lock: a quarter of their life, so a stalled part still leaves three
#: quarters before either lapses (each renewal is two round trips to L2 a key; at every part a cold start's fifty
#: scopes cost two hundred)
_RENEW_EVERY_SECONDS: Final = _CLAIM_TTL.total_seconds() / 4

#: the manifest's format; a pointer of another is not read
_FORMAT: Final = 2

#: the object-name segment of a table with no scope
_NO_SCOPE: Final = "_"

#: why a table was left out of a load (:attr:`StagedCheckpoint.missing`): no checkpoint holds a part of it that reads
#: back; or every one holding it wrote it with another schema digest
COLD_START: Final = "cold_start"
SCHEMA_CHANGE: Final = "schema_change"

#: why a write read a table whole from the copy: the follower read it whole from L3 (a cold start, reset, schema
#: change or an older writer's whole record, ``copy_follower_whole_reads_total``)
_WHOLE_TOUCH: Final = "whole_touch"

#: the situation each why the follower reads a table whole for is (``copy_follower._stage``): what a checkpoint
#: writing that table whole again declares its read as
_FOLLOWER_KINDS: Final[Mapping[str, str]] = {
    "cold_start": "cold_start",
    "reset": "cold_start",
    "schema_change": "schema_change",
    "legacy_whole_record": "rolling_upgrade",
}

#: every why ``copy_checkpoint_whole_writes_total`` counts; nothing else reads a table whole from the copy. Only a
#: cold start is expected
_WHOLE_WRITES: Final = WholeReads(
    "copy_checkpoint_whole_writes_total",
    subject="tables a checkpoint write read whole from the copy",
    whys={
        COLD_START: "no checkpoint holds the table (the first, or one whose parts did not read back): its cold start",
        SCHEMA_CHANGE: "every checkpoint holding the table wrote it with another schema digest",
        _WHOLE_TOUCH: "the follower read the table whole from L3 (a cold start, reset, schema change or an older "
        "writer's whole record)",
    },
    expected={COLD_START},
)

#: the stage a part read back is patched in: a change set being applied, never a data table
_PART_STAGE: Final = f"{STAGE_PREFIX}ckpt_part__"

#: the touched keys a patched part drops from its old rows, staged beside it (:func:`_patched_part`)
_PART_KEYS_STAGE: Final = f"{STAGE_PREFIX}ckpt_part_keys__"

#: what a patch's job is handed (:func:`_patched_parts`): the touched keys, and the part's old bytes as the Object Store
#: holds them (one row), decoded where the job runs, never in the serving process
_PATCH_KEYS: Final = f"{STAGE_PREFIX}ckpt_touched__"
_PATCH_OLD: Final = f"{STAGE_PREFIX}ckpt_old__"

#: a parts job's answer (:func:`_encoded`): one row a part, its scope, its rows and its bytes (zstd Arrow IPC)
_PARTS_SCHEMA: Final = (
    pa.schema([("scope", pa.string()), ("rows", pa.int64()), ("data", pa.large_binary())]) if pa is not None else None
)


def _writes() -> Any:
    return counter(
        "copy_checkpoint_writes_total",
        description="copy checkpoint writes, by kind (full: every part written; incremental: unchanged parts named "
        "again) and outcome (written; failed; skipped: another replica holds the lock, the copy has no stamp yet, "
        "or a later checkpoint moved the pointer first)",
        label_names=("kind", "outcome"),
    )


def _notes_age() -> Any:
    return gauge(
        "copy_checkpoint_notes_unsaved_age_seconds",
        description="seconds since the oldest epoch this replica's copy took from the change notes (a commit too "
        "large to record its keys) that no checkpoint it wrote holds yet, 0 when none: a start from the checkpoint "
        "needs those notes, which answer for at least half a note fence (5 of 10 min). A checkpoint is due at a "
        "quarter fence; a value past it says the checkpoint is late, and past half a fence that a restart may be "
        "stuck behind",
        label_names=("base",),
    )


def _scope_segment(scope: str | None) -> str:
    """a scope as an object-name segment: a digest, so any scope's text makes a valid name.

    :param scope: the scope; None for a table with no scope
    :ptype scope: str | None
    :return: the segment
    :rtype: str
    """
    return _NO_SCOPE if scope is None else hashlib.sha256(scope.encode()).hexdigest()[:16]


def _encode_part(table: Any) -> bytes:
    """an Arrow table as one part: an Arrow IPC stream, zstd-compressed.

    :param table: the rows
    :ptype table: pyarrow.Table
    :return: the bytes
    :rtype: bytes
    """
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def _next_part(batches: Iterator[tuple[str | None, Any]]) -> tuple[str | None, int, bytes] | None:
    """the next (scope, batch) with rows, encoded as a part: its scope, its rows and its bytes; None when none is left.

    :param batches: each scope and its rows
    :ptype batches: Iterator[tuple[str | None, pyarrow.RecordBatch | pyarrow.Table]]
    :return: the part, or None
    :rtype: tuple[str | None, int, bytes] | None
    """
    for scope, batch in batches:
        if batch.num_rows:
            return (
                scope,
                batch.num_rows,
                _encode_part(batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])),
            )
    return None


def _decode_part(data: bytes) -> Any:
    """a part's rows.

    :param data: the bytes
    :ptype data: bytes
    :return: the rows
    :rtype: pyarrow.Table
    """
    return ipc.open_stream(data).read_all()


@dataclass(frozen=True, slots=True)
class CheckpointTable:
    """one table in a checkpoint: the mark its rows are at, the schema they were written with, and its parts.

    :ivar mark: the table's epoch mark
    :ivar digest: :meth:`DuckDBBackend.schema_digest` of the table that wrote it
    :ivar parts: each part's object name, scope (None for a table with no scope) and rows, in order
    :ivar derived: a table the copy derives from the followed ones (no epoch mark): it holds what they hold at the
        checkpoint's stamp, so it loads only with every followed table from the same checkpoint
    """

    mark: str | None
    digest: str
    parts: tuple[tuple[str, str | None, int], ...]
    derived: bool = False

    @property
    def rows(self) -> int:
        """the rows its parts hold.

        :return: the rows
        :rtype: int
        """
        return sum(rows for _, _, rows in self.parts)

    def encode(self) -> dict[str, Any]:
        """the entry as JSON-able data.

        :return: the data
        :rtype: dict[str, Any]
        """
        body: dict[str, Any] = {"mark": self.mark, "digest": self.digest, "parts": [list(part) for part in self.parts]}
        if self.derived:
            body["derived"] = True
        return body

    @classmethod
    def decode(cls, body: Mapping[str, Any]) -> CheckpointTable:
        """an entry read back.

        :param body: the data
        :ptype body: Mapping[str, Any]
        :return: the entry
        :rtype: CheckpointTable
        """
        mark = body["mark"]
        return cls(
            mark=None if mark is None else str(mark),
            digest=str(body["digest"]),
            parts=tuple((str(n), None if s is None else str(s), int(r)) for n, s, r in body["parts"]),
            derived=bool(body.get("derived", False)),
        )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """what a pointer names: each table's mark, digest and parts, the stamp it was written at, and the checkpoint the
    pointer named before (kept, so a table the newest cannot give loads from it).

    :ivar seq: its sequence, higher for a later one
    :ivar stamp: the writer's stamp, as ``encode_stamp`` wrote it
    :ivar tables: each table's entry
    :ivar previous: the checkpoint before it, without its own previous; None for the first
    """

    seq: int
    stamp: Any
    tables: Mapping[str, CheckpointTable]
    previous: Checkpoint | None = None

    @property
    def marks(self) -> dict[str, str | None]:
        """each table's mark.

        :return: table -> mark
        :rtype: dict[str, str | None]
        """
        return {name: entry.mark for name, entry in self.tables.items()}

    @property
    def rows(self) -> int:
        """the rows it holds.

        :return: the rows
        :rtype: int
        """
        return sum(entry.rows for entry in self.tables.values())

    def names(self) -> set[str]:
        """the objects this checkpoint names (not its previous).

        :return: the names
        :rtype: set[str]
        """
        return {name for entry in self.tables.values() for name, _, _ in entry.parts}

    def live(self) -> set[str]:
        """the objects this checkpoint and its previous name: what no retire may take.

        :return: the names
        :rtype: set[str]
        """
        return self.names() | (self.previous.names() if self.previous is not None else set())

    def without_previous(self) -> Checkpoint:
        """this checkpoint as a later one's previous.

        :return: the checkpoint
        :rtype: Checkpoint
        """
        return Checkpoint(seq=self.seq, stamp=self.stamp, tables=self.tables)

    def body(self) -> dict[str, Any]:
        """the checkpoint as JSON-able data, without its previous.

        :return: the data
        :rtype: dict[str, Any]
        """
        return {
            "format": _FORMAT,
            "seq": self.seq,
            "stamp": self.stamp,
            "tables": {name: entry.encode() for name, entry in self.tables.items()},
        }

    def encode(self) -> bytes:
        """the pointer's value: the checkpoint and its previous as JSON.

        :return: the bytes
        :rtype: bytes
        """
        body = self.body()
        body["previous"] = self.previous.body() if self.previous is not None else None
        return json.dumps(body, sort_keys=True).encode()

    @classmethod
    def _from_body(cls, body: Mapping[str, Any], previous: Checkpoint | None) -> Checkpoint:
        if body.get("format") != _FORMAT:
            raise ValueError(f"a checkpoint of format {body.get('format')!r}, not {_FORMAT}")
        return cls(
            seq=int(body["seq"]),
            stamp=body["stamp"],
            tables={str(name): CheckpointTable.decode(entry) for name, entry in body["tables"].items()},
            previous=previous,
        )

    @classmethod
    def decode(cls, raw: bytes) -> Checkpoint:
        """a pointer's value read back.

        :param raw: the bytes
        :ptype raw: bytes
        :return: the checkpoint
        :rtype: Checkpoint
        :raises ValueError: when it is not a checkpoint's JSON of this format
        :raises KeyError: when a field is missing
        """
        body = json.loads(raw)
        previous = body.get("previous")
        return cls._from_body(body, cls._from_body(previous, None) if previous is not None else None)


@dataclass(frozen=True, slots=True)
class StagedCheckpoint:
    """what :meth:`CopyCheckpoints.stage` staged beside the copy.

    :ivar stamp: the stamp of the oldest checkpoint a table loaded from; None when no table loaded
    :ivar marks: each LOADED table's mark (a table loaded from the previous checkpoint has that one's mark)
    :ivar changes: the loaded tables' stages, each whole, to apply in one transaction
    :ivar rows: the rows staged
    :ivar missing: each table not loaded, and why: :data:`COLD_START` or :data:`SCHEMA_CHANGE`
    """

    stamp: Any
    marks: Mapping[str, str | None]
    changes: list[StagedChange]
    rows: int
    missing: Mapping[str, str]

    @property
    def loaded(self) -> bool:
        """whether any table loaded.

        :return: whether any did
        :rtype: bool
        """
        return bool(self.marks)


@dataclass(slots=True)
class _Taken:
    """what a write took of the changes counted since the last checkpoint: given back if the write does not land."""

    since: int
    touched: Mapping[str, frozenset[str] | None]
    keys: Mapping[str, tuple[Any, ...]]
    kinds: Mapping[str, str]
    due: bool
    whole: frozenset[str]
    noted: datetime | None


@dataclass(frozen=True, slots=True)
class _Plan:
    """how a write writes one table: the entry it builds on and what of it to write again.

    :ivar table: the table
    :ivar prior: the entry this replica's copy of it last matched, its objects still named; None when written whole
    :ivar scopes: a scoped table's scopes to read again from the copy (empty: every part named again)
    :ivar keys: a table with no scope: the keys to patch its part by, an Arrow table of its key columns as text, a
        key perhaps more than once (None: its part named again)
    :ivar why: a :data:`_WHOLE_WRITES` why when the table is read whole from the copy; None otherwise
    :ivar kind: the situation a whole read is (``cold_start``, ``schema_change``, ``rolling_upgrade``), each read by its
        own declared function
    """

    table: FollowedTable
    prior: CheckpointTable | None
    scopes: frozenset[str] = frozenset()
    keys: Any = None
    why: str | None = None
    kind: str = ""


def _with_touched(
    held: Mapping[str, frozenset[str] | None], table: str, scopes: Iterable[str] | None
) -> Mapping[str, frozenset[str] | None]:
    """``held`` with a table's touched scopes counted in, as a new read-only snapshot; None (the table whole) wins over
    any scopes.

    :param held: each table touched, and its scopes
    :ptype held: Mapping[str, frozenset[str] | None]
    :param table: the table
    :ptype table: str
    :param scopes: the scopes touched; None: the table whole
    :ptype scopes: Iterable[str] | None
    :return: the new snapshot
    :rtype: Mapping[str, frozenset[str] | None]
    """
    if scopes is None:
        return with_entry(held, table, None)
    have = held.get(table, frozenset())
    if have is None:
        return held
    return with_entry(held, table, have | frozenset(scopes))


def _with_keys(held: Mapping[str, tuple[Any, ...]], table: str, keys: Iterable[Any]) -> Mapping[str, tuple[Any, ...]]:
    """``held`` with a table's touched keys counted in, as a new read-only snapshot: each batch kept as it is (immutable
    Arrow, never a Python object a key; decision 35), an empty one not at all.

    :param held: each table with no scope touched, and its key batches
    :ptype held: Mapping[str, tuple[Any, ...]]
    :param table: the table
    :ptype table: str
    :param keys: the key batches
    :ptype keys: Iterable[Any]
    :return: the new snapshot
    :rtype: Mapping[str, tuple[Any, ...]]
    """
    batches = tuple(batch for batch in keys if batch.num_rows)
    if not batches:
        return held
    return with_entry(held, table, (*held.get(table, ()), *batches))


def _touched_keys(batches: Sequence[Any], key: Sequence[str]) -> Any:
    """a table's touched keys as one Arrow table of its key columns, a key perhaps more than once (the patch's semi
    and anti steps each take a key once however often it is named); None when none was touched.

    :param batches: the batches counted
    :ptype batches: Sequence[pyarrow.Table]
    :param key: the table's key columns
    :ptype key: Sequence[str]
    :return: the keys, or None
    :rtype: pyarrow.Table | None
    """
    if not batches:
        return None
    return pa.concat_tables([batch.select(list(key)) for batch in batches])


class CopyCheckpoints:
    """a copy follower's checkpoints in the pod's Object Store and pointer bucket.

    :param name: the copy's name, a pointer-bucket token (letters, digits, ``-``, ``_``)
    :ptype name: str
    :param tables: the tables the copy holds
    :ptype tables: Sequence[FollowedTable]
    :param backend: the copy (each table's schema digest is recorded beside its parts)
    :ptype backend: DuckDBBackend
    :param store: the Object Store: ``put``, ``get``, ``list_names``, and ``require_room`` (a write asks it for all its
        new parts' room before it puts the first: :meth:`threetears.nats.object_store.NatsObjectStore.require_room`)
    :ptype store: Any
    :param pointers: the pointer bucket: ``get_entry``, ``create``, ``update``, ``delete``, ``list_keys``
    :ptype pointers: Any
    :param retire: asks the bucket's declarer to delete objects (``PodObjectStore.retire``)
    :ptype retire: Callable[[list[str]], Awaitable[Any]]
    :param owner: this replica, a token
    :ptype owner: str
    :param encode_stamp: the stamp as JSON-able data
    :ptype encode_stamp: Callable[[Any], Any]
    :param decode_stamp: the stamp back from it
    :ptype decode_stamp: Callable[[Any], Any]
    :param ratio: how much of the checkpoint's rows applied since make the next one due
    :ptype ratio: float
    :param part_rows: the most rows one part holds; a table with no scope must fit in one, patched by key
    :ptype part_rows: int
    """

    def __init__(
        self,
        *,
        name: str,
        tables: Sequence[FollowedTable],
        backend: DuckDBBackend,
        store: Any,
        pointers: Any,
        retire: Callable[[list[str]], Awaitable[Any]],
        owner: str,
        encode_stamp: Callable[[Any], Any] = lambda stamp: stamp,
        decode_stamp: Callable[[Any], Any] = lambda data: data,
        ratio: float = _DEFAULT_RATIO,
        part_rows: int = _DEFAULT_PART_ROWS,
        note_fence: timedelta = DEFAULT_NOTE_FENCE,
    ) -> None:
        if pa is None:
            raise ImportError("a checkpoint needs pyarrow: install 3tears[snapshot]")
        if not name or not owner or any(c in token for token in (name, owner) for c in "./"):
            raise ValueError("a checkpoint's name and owner are tokens with no '.' or '/'")
        self._tables = tuple(tables)
        self._digests = {t.name: backend.schema_digest(t.name) for t in self._tables}
        self.base: Final = f"{name}-ckpt"
        self._backend = backend
        self._store = store
        self._pointers = pointers
        self._retire = retire
        self._owner = owner
        self._encode_stamp = encode_stamp
        self._decode_stamp = decode_stamp
        self._ratio = ratio
        self._part_rows = part_rows
        #: the rows the last checkpoint this replica loaded or wrote holds; None before one
        self._held_rows: int | None = None
        #: the rows applied since it
        self._since = 0
        #: each table's entry this replica's copy of it matched at its last load or write, plus what changed since
        #: (``_touched``, ``_keys``): a write names its unchanged parts again while the newest or the previous
        #: checkpoint still names them
        self._priors: Mapping[str, CheckpointTable] = frozen()
        #: each table changed since, and the scopes of it changed (None: the table changed whole)
        self._touched: Mapping[str, frozenset[str] | None] = frozen()
        #: each table with no scope changed since, and the keys of it changed: immutable Arrow batches of its key
        #: columns as text, one an apply (never a Python object a key)
        self._keys: Mapping[str, tuple[Any, ...]] = frozen()
        #: a change was counted with no tables named: the next write writes every table
        #: each table the follower read whole since the last write, onto the situation it read it in
        self._whole_kinds: Mapping[str, str] = frozen()
        #: tables the next write writes whole or from the previous checkpoint: left out of the load, or loaded from
        #: the previous checkpoint
        self._whole: frozenset[str] = frozenset()
        #: each table a load left out, and why (:data:`COLD_START`, :data:`SCHEMA_CHANGE`): its cold start here
        self._missing: Mapping[str, str] = frozen()
        #: a checkpoint is due for a table of ``_whole``: the load left it out, or it was read again since
        self._whole_due = False
        #: the newest checkpoint holds tables this replica could only rewrite whole: it writes none until it loads again
        self._stale = False
        #: (seq, table) whose parts did not read back here: never kept as they are
        self._broken: frozenset[tuple[int, str]] = frozenset()
        #: the tables a follower keeps by their epochs (:meth:`follow`); every other is derived. None: every table
        self._followed: frozenset[str] | None = None
        #: where the oldest notes bracket begins of an epoch the follower applied from the change notes since the last
        #: checkpoint (database clock); None when none
        self._noted: datetime | None = None
        # Why a quarter fence: the notes keep a bracket answerable for at least half a fence from where it begins,
        # not a whole one. A reader refuses a bracket once its reach (its start less half the ring's wait, for a
        # writer's cached slot) is older than the switch before the current one (change_notes.py, the
        # not_covered checks), and switches are at least the wait (at least the fence) apart: a switch just after
        # the reach is followed by the next one wait later, half a wait after the bracket began. Due at half of
        # that, the follower has the other quarter fence to write the checkpoint, and under continuous large
        # commits checkpoints come at most about once a quarter fence (2.5 min at the default 10)
        self._noted_due_after = note_fence / 4
        held = weakref.ref(self)
        _notes_age().labels(base=self.base).set_function(
            lambda: checkpoints.notes_age_seconds() if (checkpoints := held()) is not None else 0.0
        )

    @property
    def table_names(self) -> frozenset[str]:
        """the tables this checkpoint holds: never one whose full is on the delta-files volume.

        :return: the names
        :rtype: frozenset[str]
        """
        return frozenset(t.name for t in self._tables)

    def follow(self, names: Iterable[str]) -> None:
        """name the tables a follower keeps by their epochs; every other table this checkpoint holds is derived from
        them (no epoch mark), loaded only with every followed table from the same checkpoint, since it holds what they
        hold at its stamp.

        A derived table's rows of a scope are derived from the followed tables' rows of that scope alone: a write
        rewrites a derived table's scopes the followed tables' changes touched since, and reads it whole only when a
        followed table was read whole. So every derived table declares a scope, and so does every followed table
        beside a derived one: a change of a followed table with no scope may reach every scope derived from it, and
        nothing finer than a whole read of the derived tables (a steady-state whole read) would answer it.

        :param names: the followed tables
        :ptype names: Iterable[str]
        :return: nothing
        :rtype: None
        :raises ValueError: when a derived table declares no scope, or a followed table declares none while a table
            is derived
        """
        followed = frozenset(names)
        derived = sorted(t.name for t in self._tables if t.name not in followed)
        unscoped = sorted(t.name for t in self._tables if t.name not in followed and t.scope is None)
        if unscoped:
            raise ValueError(
                f"derived tables {unscoped} declare no scope: a checkpoint could rewrite them only whole; give each "
                "the scope its rows are derived by (FollowedTable.scope)"
            )
        unscoped_followed = sorted(t.name for t in self._tables if t.name in followed and t.scope is None)
        if derived and unscoped_followed:
            raise ValueError(
                f"followed tables {unscoped_followed} declare no scope beside derived tables {derived}: a change of "
                "one may reach every scope derived from it, which a checkpoint could rewrite only by reading the "
                "derived tables whole; give each followed table the scope its rows belong to (FollowedTable.scope)"
            )
        self._followed = followed

    def _derived(self, name: str) -> bool:
        return self._followed is not None and name not in self._followed

    # --- reading ---------------------------------------------------------------------------------

    async def newest(self) -> tuple[Checkpoint, int] | None:
        """the checkpoint the pointer names, and the pointer's revision; None when there is none (or it is unreadable,
        said at WARNING: a start then reads L3).

        :return: the checkpoint and revision
        :rtype: tuple[Checkpoint, int] | None
        """
        entry = await self._pointers.get_entry(key=self.base)
        if entry is None:
            return None
        raw, revision = entry
        try:
            return Checkpoint.decode(raw), revision
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            contain(exc, log=log, what="a checkpoint pointer could not be read; the start reads L3 whole")
            return None

    async def stage(self) -> StagedCheckpoint:
        """stage each table beside the copy, a part at a time, from the newest checkpoint that holds it in parts that
        read back: the newest, or the previous. Called once, at a follower's start (:meth:`CopyFollower.start`): the
        one load of a whole table from L2, which then walks the change log from each table's mark.

        :return: the stamp, marks, stages and rows of the tables loaded, and each table left out and why
        :rtype: StagedCheckpoint
        """
        found = await self.newest()
        self._touched, self._keys, self._whole_kinds, self._since, self._stale = frozen(), frozen(), frozen(), 0, False
        self._noted = None
        if found is None:
            self._held_rows, self._priors, self._whole, self._whole_due, self._missing = (
                None,
                frozen(),
                frozenset(),
                False,
                frozen(),
            )
            return StagedCheckpoint(
                stamp=None, marks={}, changes=[], rows=0, missing={t.name: COLD_START for t in self._tables}
            )
        newest = found[0]
        sources = [newest] if newest.previous is None else [newest, newest.previous]
        loaded: dict[str, Checkpoint] = {}
        missing: dict[str, str] = {}
        changes: list[StagedChange] = []
        rows = 0
        for table in self._tables:
            if self._derived(table.name):
                continue
            held = [source for source in sources if table.name in source.tables]
            usable = [s for s in held if s.tables[table.name].digest == self._digests[table.name]]
            if not usable:
                missing[table.name] = SCHEMA_CHANGE if held else COLD_START
                continue
            stage = f"{STAGE_PREFIX}{table.name}__rows"
            for source in usable:
                if await self._stage_table(table, source, stage):
                    loaded[table.name] = source
                    rows += source.tables[table.name].rows
                    # a start from a checkpoint fills an empty copy: its apply empties the table as a cold start
                    changes.append(
                        StagedChange(table=table.name, key=table.key, rows=stage, whole=True, whole_kind="cold_start")
                    )
                    break
                self._broken = self._broken | {(source.seq, table.name)}
            else:
                missing[table.name] = COLD_START
        # a derived table holds what the followed ones hold at a stamp: it loads only when every one of them loaded
        # from the newest checkpoint, from that checkpoint; else the follower's owner derives it again
        whole_newest = not missing and all(source is newest for source in loaded.values())
        derived: dict[str, CheckpointTable] = {}
        for table in self._tables:
            if not self._derived(table.name):
                continue
            entry = newest.tables.get(table.name)
            if entry is None or not whole_newest:
                missing[table.name] = COLD_START
                continue
            if entry.digest != self._digests[table.name]:
                missing[table.name] = SCHEMA_CHANGE
                continue
            stage = f"{STAGE_PREFIX}{table.name}__rows"
            if await self._stage_table(table, newest, stage):
                rows += entry.rows
                derived[table.name] = entry
                changes.append(
                    StagedChange(table=table.name, key=table.key, rows=stage, whole=True, whole_kind="cold_start")
                )
            else:
                self._broken = self._broken | {(newest.seq, table.name)}
                missing[table.name] = COLD_START
        if missing:
            log.warning(
                "a checkpoint did not hold every table; the follower reads those whole from L3",
                extra={"extra_data": {"base": self.base, "missing": dict(missing), "loaded": sorted(loaded)}},
            )
        self._priors = frozen({name: source.tables[name] for name, source in loaded.items()} | derived)
        self._missing = frozen(missing)
        self._whole = frozenset(t.name for t in self._tables if loaded.get(t.name) is not newest)
        self._whole_due = bool(self._whole)
        self._held_rows = rows if loaded else None
        oldest = min(loaded.values(), key=lambda source: source.seq) if loaded else None
        return StagedCheckpoint(
            stamp=self._decode_stamp(oldest.stamp) if oldest is not None else None,
            # the followed tables' marks alone: a derived table has none
            marks={name: source.tables[name].mark for name, source in loaded.items()},
            changes=changes,
            rows=rows,
            missing=missing,
        )

    async def _stage_table(self, table: FollowedTable, source: Checkpoint, stage: str) -> bool:
        """stage one table's parts of one checkpoint; when a part does not read back, drop the stage.

        :return: whether every part staged
        :rtype: bool
        """
        try:
            await asyncio.to_thread(self._backend.drop_stage, stage)
            await asyncio.to_thread(self._backend.stage_rows, stage, table.name, pa.table({}), table.key)
            for object_name, _, rows in source.tables[table.name].parts:
                data = await self._store.get(object_name)
                await asyncio.to_thread(self._stage_part, stage, table, object_name, data, rows)
                del data
        except Exception as exc:  # noqa: BLE001 - the table loads from the previous checkpoint or is read whole, said
            contain(
                exc,
                log=log,
                what=f"checkpoint {source.seq}'s parts of {table.name} could not be loaded; "
                "it loads from the previous checkpoint, or is read whole",
            )
            await asyncio.to_thread(self._backend.drop_stage, stage)
            return False
        return True

    def _stage_part(self, stage: str, table: FollowedTable, object_name: str, data: bytes, rows: int) -> None:
        """one part decoded and staged, on the caller's worker thread: one hop a part.

        :raises ValueError: when the part holds other than the rows its manifest says
        """
        part = _decode_part(data)
        if part.num_rows != rows:
            raise ValueError(f"part {object_name} holds {part.num_rows} rows, its manifest says {rows}")
        # INVARIANT: a part holds each key once. Every part is written from the copy's own DuckDB table, which holds
        # the table's primary key (_next_part over a read of it, or _patched_part: the old part's rows of every other
        # key, keyed.anti, beside the copy's rows of the touched keys), so the last-row-per-key pass a stage makes of
        # rows read from L3 finds nothing to drop. A part that breaks it anyway is refused, never loaded twice: the
        # apply's insert into the copy's keyed table raises (test_copy_checkpoint.py::
        # test_a_part_holding_a_key_twice_is_refused_by_the_copys_key_when_it_is_applied). A new writer of parts must
        # keep it. Why skip the pass (unique_keys): it was most of what a start from a checkpoint cost over its bare
        # decode and copy-in. 1,000,000 rows in 52 parts, profiled under measure-lock (2026-11-15): stage() 0.222 s
        # against 0.125 s bare (decode 0.010 s, copy-in 0.115 s), the pass alone 0.087 s. The real-size budget
        # (test_copy_follower_whole_reads_live.py::test_checkpoint_stage_speed_at_real_size, 2x the bare decode and
        # copy-in) ran at 1.99x with it; without it 1.21x locally and 1.06x on YugabyteDB lane 1 (0.10 s against 0.10 s)
        self._backend.stage_rows(stage, table.name, part, table.key, unique_keys=True)

    # --- writing ---------------------------------------------------------------------------------

    def applied(
        self,
        rows: int,
        touched: Mapping[str, frozenset[str] | None],
        keys: Mapping[str, Any] | None = None,
        whys: Mapping[str, str] | None = None,
        noted: datetime | None = None,
    ) -> None:
        """count a change the follower applied: its rows make the next checkpoint due, and what it touched is what
        the next checkpoint writes. Kept until a write lands.

        :param rows: the rows the change applied
        :ptype rows: int
        :param touched: each table the change touched, and the scopes of the rows it touched (for a table with no
            scope, any set); None for a table read whole
        :ptype touched: Mapping[str, frozenset[str] | None]
        :param keys: each table with no scope the change touched (not whole), and the keys it touched as one Arrow
            table of its key columns as text (none, an empty one): what the next checkpoint patches its part by. Kept as
            it is, never edited
        :ptype keys: Mapping[str, pyarrow.Table] | None
        :param whys: each table read whole, onto why the follower read it (``cold_start``, ``reset``,
            ``schema_change``, ``legacy_whole_record``): the situation the checkpoint's whole read of it is
        :ptype whys: Mapping[str, str] | None
        :param noted: when the change holds keys the follower named from the change notes, where the oldest of their
            notes brackets begins (database clock): the next checkpoint is due a quarter note fence after it
        :ptype noted: datetime | None
        :return: nothing
        :rtype: None
        :raises ValueError: when a table with no scope is touched, not whole, and no keys are named for it, or a table
            read whole has no known why
        """
        whole = sorted(name for name, scopes in touched.items() if scopes is None)
        unknown = [name for name in whole if (whys or {}).get(name) not in _FOLLOWER_KINDS]
        if unknown:
            raise ValueError(f"{unknown} were read whole: name why, one of {sorted(_FOLLOWER_KINDS)}")
        self._since += rows
        self._note(noted)
        self._whole_kinds = with_entries(
            self._whole_kinds, {name: _FOLLOWER_KINDS[(whys or {})[name]] for name in whole}
        )
        scopeless = {t.name for t in self._tables if t.scope is None}
        unkeyed = sorted(
            name
            for name, scopes in touched.items()
            if scopes is not None and name in scopeless and (keys is None or name not in keys)
        )
        if unkeyed:
            raise ValueError(
                f"{unkeyed} have no scope: name the keys the change touched, which their part is patched by"
            )
        for table, scopes in touched.items():
            self._whole_due = self._whole_due or table in self._whole
            self._touched = _with_touched(self._touched, table, scopes)
            if scopes is not None and keys is not None and table in keys:
                self._keys = _with_keys(self._keys, table, (keys[table],))

    def due(self) -> bool:
        """whether a checkpoint is due: none is known, a table the load left out is due, the rows applied since
        reach the ratio of its rows, or an epoch applied from the change notes since is a quarter note fence old; never
        while this replica is stale (the newest holds tables it could only rewrite whole, until it loads again).

        :return: whether it is due
        :rtype: bool
        """
        if self._stale:
            return False
        if self._held_rows is None or self._whole_due:
            return True
        if self.noted_due_in() == 0.0:
            return True
        return self._since >= max(1, int(self._ratio * self._held_rows))

    def noted_due_in(self) -> float | None:
        """the seconds until an epoch applied from the change notes since the last checkpoint makes one due: 0 once
        it is due; None when there is none. A follower with no change to apply asks again then.

        Compared on this process's clock against the notes' database clock: a skew is taken from the quarter fence the
        rule leaves between a checkpoint due and the notes it saves being refused.

        :return: the seconds; None when nothing was applied from the notes since
        :rtype: float | None
        """
        if self._noted is None:
            return None
        waited = datetime.now(tz=self._noted.tzinfo or UTC) - self._noted
        return max(0.0, (self._noted_due_after - waited).total_seconds())

    def notes_age_seconds(self) -> float:
        """how long the oldest epoch applied from the change notes has waited for a checkpoint holding it; 0 when
        none waits (``copy_checkpoint_notes_unsaved_age_seconds``).

        :return: the seconds
        :rtype: float
        """
        if self._noted is None:
            return 0.0
        return max(0.0, (datetime.now(tz=self._noted.tzinfo or UTC) - self._noted).total_seconds())

    def _note(self, noted: datetime | None) -> None:
        if noted is not None:
            self._noted = noted if self._noted is None else min(self._noted, noted)

    def _incremental(self) -> bool:
        return bool(self._priors)

    async def write(
        self, read: Callable[[], AbstractAsyncContextManager[tuple[PinnedCopy, Any, Mapping[str, str | None]]]]
    ) -> Checkpoint | None:
        """write a checkpoint of the copy as one read holds it, and move the pointer to it.

        :param read: pins the state the checkpoint writes: an async context manager giving ``(state, stamp, marks)``,
            the state a :class:`~threetears.core.collections.copy_reads.PinnedCopy` its parts are read through in jobs
            (:meth:`~threetears.core.collections.copy_follower.CopyFollower.current`)
        :ptype read: Callable[[], AbstractAsyncContextManager[tuple[PinnedCopy, Any, Mapping[str, str | None]]]]
        :return: the checkpoint written; None when another replica holds the lock, the copy has no stamp yet, a later
            checkpoint got there first, or this replica could only rewrite whole what the newest holds
        :rtype: Checkpoint | None
        """
        lock = f"{self.base}-lock"
        if await self._pointers.create(key=lock, value=self._owner.encode(), ttl=_CLAIM_TTL) is None:
            _writes().labels(kind="incremental" if self._incremental() else "full", outcome="skipped").inc()
            return None
        try:
            return await self._write_locked(read, lock)
        finally:
            await self._pointers.delete(key=lock)

    def _take(self) -> _Taken:
        """take what was counted since the last checkpoint, at the read the write holds: a change applied after the
        read began is not in it, and stays counted for the next.

        :return: what was taken
        :rtype: _Taken
        """
        taken = _Taken(
            since=self._since,
            touched=self._touched,
            keys=self._keys,
            kinds=self._whole_kinds,
            due=self._whole_due,
            whole=frozenset(self._whole),
            noted=self._noted,
        )
        self._touched, self._keys, self._whole_kinds, self._whole_due = frozen(), frozen(), frozen(), False
        self._noted = None
        return taken

    def _give_back(self, taken: _Taken) -> None:
        """a write that did not land: what it took is counted again, beside what came since.

        :param taken: what it took
        :ptype taken: _Taken
        :return: nothing
        :rtype: None
        """
        for table, scopes in taken.touched.items():
            self._touched = _with_touched(self._touched, table, scopes)
        for table, keys in taken.keys.items():
            self._keys = _with_keys(self._keys, table, keys)
        self._whole_kinds = frozen({**taken.kinds, **self._whole_kinds})
        self._whole_due = self._whole_due or taken.due
        self._note(taken.noted)

    def _plan(self, table: FollowedTable, taken: _Taken, newest: Checkpoint | None) -> _Plan | None:
        """how a write writes one table; None when this replica could only rewrite whole what the newest holds.

        :return: the plan
        :rtype: _Plan | None
        """
        name = table.name
        live = newest.live() if newest is not None else set()
        prior = self._priors.get(name)
        if prior is not None and (prior.digest != self._digests[name] or not {n for n, _, _ in prior.parts} <= live):
            prior = None  # another replica wrote twice since: its objects may be retired
        if prior is None:
            if name in self._missing:
                return _Plan(table, None, why=self._missing[name], kind=self._missing[name])
            held = newest.tables.get(name) if newest is not None else None
            if newest is None or held is None or (newest.seq, name) in self._broken:
                return _Plan(table, None, why=COLD_START, kind=COLD_START)
            if held.digest != self._digests[name]:
                return _Plan(table, None, why=SCHEMA_CHANGE, kind=SCHEMA_CHANGE)
            return None
        if self._derived(name):
            return self._plan_derived(table, taken, prior)
        if name in taken.touched and taken.touched[name] is None:
            return _Plan(table, None, why=_WHOLE_TOUCH, kind=taken.kinds[name])
        if name not in taken.touched:
            return _Plan(table, prior)
        if table.scope is None:
            return _Plan(table, prior, keys=_touched_keys(taken.keys.get(name, ()), table.key))
        return _Plan(table, prior, scopes=frozenset(taken.touched[name] or ()))

    def _plan_derived(self, table: FollowedTable, taken: _Taken, prior: CheckpointTable) -> _Plan:
        """a derived table's plan: the scopes the followed tables' changes touched since, its rows of a scope being
        derived from theirs alone; whole when a followed table was read whole.

        :return: the plan
        :rtype: _Plan
        """
        followed = {name: scopes for name, scopes in taken.touched.items() if not self._derived(name)}
        whole = sorted(name for name, scopes in followed.items() if scopes is None)
        if whole:
            # read whole after what it derives from was: the situation that table was read in (a cold start first)
            kinds = {taken.kinds[name] for name in whole}
            kind = next(k for k in ("cold_start", "schema_change", "rolling_upgrade") if k in kinds)
            return _Plan(table, None, why=_WHOLE_TOUCH, kind=kind)
        return _Plan(table, prior, scopes=frozenset(s for scopes in followed.values() for s in scopes or ()))

    async def _write_locked(
        self,
        read: Callable[[], AbstractAsyncContextManager[tuple[PinnedCopy, Any, Mapping[str, str | None]]]],
        lock: str,
    ) -> Checkpoint | None:
        found = await self.newest()
        newest, revision = (found[0], found[1]) if found is not None else (None, None)
        seq = (newest.seq + 1) if newest is not None else 1
        claim = f"{self.base}.w.{seq}.{self._owner}"
        await self._pointers.create(key=claim, value=b"", ttl=_CLAIM_TTL)
        written: list[str] = []
        taken: _Taken | None = None
        kind = "incremental" if self._incremental() else "full"
        try:
            async with read() as (state, stamp, marks):
                if stamp is None:
                    _writes().labels(kind=kind, outcome="skipped").inc()
                    return None
                # nothing awaits between the read beginning and this: every change it holds is counted in it
                taken = self._take()
                tables: dict[str, CheckpointTable] = {}
                plans: list[_Plan] = []
                for table in self._tables:
                    if not self._derived(table.name) and table.name not in marks:
                        kept = self._kept(newest, table.name)
                        if kept is not None:
                            tables[table.name] = kept
                        continue
                    plan = self._plan(table, taken, newest)
                    if plan is None:
                        # the newest holds this table and this replica's parts of it are gone: only a whole read could
                        # write it, which the newest already holds; this replica writes none until it loads again
                        self._give_back(taken)
                        taken = None
                        self._stale = True
                        log.info(
                            "a newer checkpoint holds what this replica could only rewrite whole; it writes none",
                            extra={"extra_data": {"base": self.base, "table": table.name, "newest": seq - 1}},
                        )
                        _writes().labels(kind=kind, outcome="skipped").inc()
                        return None
                    plans.append(plan)
                kind = "incremental" if any(plan.prior is not None for plan in plans) else "full"
                # every new part read and encoded before the first is put, and their room asked for together: put as
                # each was encoded, a checkpoint the store had no room for was refused part way, after some of its
                # parts were written (then retired), and the next write met the same. The encoded parts are held
                # until put (zstd Arrow IPC, the copy's rows compressed), so the copy is read once, as before: read
                # twice instead (once to measure, once to put), every write would read and encode its rows twice.
                # What holding them costs (2026-11-18, a cold write of a copy of the reports pod's shape at its size,
                # 2.89 GiB in the process, 15 M rows of 7 columns in 51 scopes): 3,060 parts, 0.308 GiB encoded, the
                # write's peak 0.115 GiB over the copy's own, 1.9 s. Even all 0.31 GiB held at once is under 2% of the
                # pod's 16 GiB limit beside its copy: no bound per batch of parts is needed at that size
                pending: list[tuple[str, bytes]] = []
                for plan in plans:
                    parts = await self._write_table(state, plan, seq, pending, (lock, claim))
                    derived = self._derived(plan.table.name)
                    tables[plan.table.name] = CheckpointTable(
                        mark=None if derived else marks[plan.table.name],
                        digest=self._digests[plan.table.name],
                        parts=parts,
                        derived=derived,
                    )
                await self._store.require_room([(name, len(data)) for name, data in pending])
                await self._put_pending(pending, written, (lock, claim))
                followed = {t.name for t in self._tables}
                for name, entry in (newest.tables if newest is not None else {}).items():
                    if name not in followed:
                        tables[name] = entry  # another code version's table: kept as it is
                checkpoint = Checkpoint(
                    seq=seq,
                    stamp=self._encode_stamp(stamp),
                    tables=tables,
                    previous=newest.without_previous() if newest is not None else None,
                )
                written_whole = {name for name in taken.whole if name in marks}
            raw = checkpoint.encode()
            moved = (
                await self._pointers.create(key=self.base, value=raw)
                if revision is None
                else await self._pointers.update(key=self.base, value=raw, revision=revision)
            )
            if moved is None:
                self._give_back(taken)
                await self._retire_quietly(written, "a later checkpoint moved the pointer first")
                _writes().labels(kind=kind, outcome="skipped").inc()
                return None
        except BaseException:
            if taken is not None:
                self._give_back(taken)
            await self._retire_quietly(written, "a checkpoint write failed part way")
            _writes().labels(kind=kind, outcome="failed").inc()
            raise
        finally:
            await self._pointers.delete(key=claim)
        assert taken is not None  # noqa: S101 -- taken before any checkpoint is built
        self._since = max(0, self._since - taken.since)
        self._held_rows = checkpoint.rows
        self._whole = self._whole - written_whole
        for plan in plans:
            self._priors = with_entry(self._priors, plan.table.name, tables[plan.table.name])
            self._missing = without_entry(self._missing, plan.table.name)
            if plan.why is not None:
                _WHOLE_WRITES.count(plan.why, table=plan.table.name)
        _writes().labels(kind=kind, outcome="written").inc()
        if newest is not None and newest.previous is not None:
            gone = newest.previous.names() - checkpoint.live()
            await self._retire_quietly(sorted(gone), "older than the previous checkpoint")
        log.info(
            "copy checkpoint written",
            extra={
                "extra_data": {
                    "base": self.base,
                    "seq": seq,
                    "kind": kind,
                    "rows": checkpoint.rows,
                    "parts_written": len(written),
                    "parts": len(checkpoint.names()),
                    "whole": {plan.table.name: plan.why for plan in plans if plan.why is not None},
                }
            },
        )
        return checkpoint

    def _kept(self, newest: Checkpoint | None, table: str) -> CheckpointTable | None:
        """a table the read holds no mark of (not read again yet): its entry as the newest, or else the previous,
        checkpoint holds it (each table's mark is its own); None when neither holds it in parts that read back.

        :return: the entry
        :rtype: CheckpointTable | None
        """
        sources = [] if newest is None else [newest] if newest.previous is None else [newest, newest.previous]
        for source in sources:
            if table in source.tables and (source.seq, table) not in self._broken:
                return source.tables[table]
        return None

    async def _write_table(
        self, state: PinnedCopy, plan: _Plan, seq: int, pending: list[tuple[str, bytes]], keys: tuple[str, str]
    ) -> tuple[tuple[str, str | None, int], ...]:
        """one table's parts: read whole from the state (its cold start, a schema change, or a whole read of it), or
        its touched scopes read again by scope, or its one part patched by the touched keys; every other part named
        again as the plan's prior holds it. Each read and its parts' encoding is a job over the state
        (:meth:`~threetears.core.collections.copy_reads.PinnedCopy.run_job`: a query worker over a shared copy).

        :return: the parts, in order
        :rtype: tuple[tuple[str, str | None, int], ...]
        :raises ValueError: when a table with no scope holds more than one part's rows
        """
        table, prior = plan.table, plan.prior
        if prior is None:
            encoded = await self._read_parts(state, _whole_parts, (plan.kind, table, self._part_rows), None, keys)
            parts = self._named(encoded, table.name, seq, pending)
            self._refuse_unscoped(table, parts)
            return tuple(parts)
        if table.scope is None:
            if plan.keys is None:
                return prior.parts
            encoded = await self._patch(state, table, prior, plan.keys, keys)
            parts = self._named(encoded, table.name, seq, pending)
            self._refuse_unscoped(table, parts)
            return tuple(parts)
        kept = [part for part in prior.parts if part[1] not in plan.scopes]
        if not plan.scopes:
            return prior.parts
        column, prefix = table.scope
        cast = table.key_cast(column)
        scopes = tuple((scope, scope[len(prefix) :]) for scope in sorted(plan.scopes) if scope.startswith(prefix))
        fresh: list[tuple[str, str | None, int]] = []
        if scopes:
            encoded = await self._read_parts(
                state, _scope_parts, (table, column, cast, self._part_rows, scopes), None, keys
            )
            fresh = self._named(encoded, table.name, seq, pending)
        order = sorted({part[1] or "" for part in [*kept, *fresh]})
        return tuple(part for scope in order for part in [*kept, *fresh] if (part[1] or "") == scope)

    def _refuse_unscoped(self, table: FollowedTable, parts: Sequence[tuple[str, str | None, int]]) -> None:
        """a table with no scope must fit one part, which a write patches by key.

        :raises ValueError: when it does not
        """
        if table.scope is None and len(parts) > 1:
            raise ValueError(
                f"{table.name} holds {sum(p[2] for p in parts)} rows, over the {self._part_rows} one part holds, and "
                "declares no scope: give it the scope its rows belong to (FollowedTable.scope)"
            )

    async def _patch(
        self, state: PinnedCopy, table: FollowedTable, prior: CheckpointTable, keys: Any, held: tuple[str, str]
    ) -> list[tuple[str | None, int, bytes]]:
        """a table with no scope's one part, patched by key: its bytes read back from the Object Store and handed to
        the patch's job (:func:`_patched_parts`), which drops the touched keys' rows and adds the copy's rows of them
        (a keyed read), and encodes the part again.

        :return: the part, encoded (none when it holds no rows)
        :rtype: list[tuple[str | None, int, bytes]]
        """
        tables: dict[str, Any] = {_PATCH_KEYS: keys}
        expected, object_name = -1, ""
        if prior.parts:
            (object_name, _, rows), *more = prior.parts
            if more:
                raise ValueError(f"{table.name} has {len(prior.parts)} parts and no scope: it cannot be patched by key")
            data = await self._store.get(object_name)
            tables[_PATCH_OLD] = await asyncio.to_thread(_old_part, data)
            expected = rows
        return await self._read_parts(state, _patched_parts, (table, expected, object_name), tables, held)

    async def _read_parts(
        self,
        state: PinnedCopy,
        job: Callable[[Any, Sequence[Any]], Any],
        args: tuple[Any, ...],
        tables: Mapping[str, Any] | None,
        held: tuple[str, str],
    ) -> list[tuple[str | None, int, bytes]]:
        """a parts job's answer (:func:`_encoded`): each part's scope, rows and bytes, the lock and claim renewed every
        :data:`_RENEW_EVERY_SECONDS` while the job runs.

        :return: the parts, in order
        :rtype: list[tuple[str | None, int, bytes]]
        """
        renewing = asyncio.ensure_future(self._renew_every(held))
        try:
            answer = await state.run_job(job, args, tables=tables)
        finally:
            await cancel_and_wait(renewing)
        return await asyncio.to_thread(_parts_of, answer)

    async def _renew_every(self, held: tuple[str, str]) -> None:
        """renew the lock and claim every :data:`_RENEW_EVERY_SECONDS`, until cancelled.

        :param held: the lock and claim
        :ptype held: tuple[str, str]
        :return: nothing
        :rtype: None
        """
        while True:
            await asyncio.sleep(_RENEW_EVERY_SECONDS)
            for key in held:
                await self._renew(key)

    def _named(
        self, encoded: Sequence[tuple[str | None, int, bytes]], table: str, seq: int, pending: list[tuple[str, bytes]]
    ) -> list[tuple[str, str | None, int]]:
        """name each encoded part of its scope, held in ``pending`` until the write puts them all
        (:meth:`_put_pending`).

        :return: the parts
        :rtype: list[tuple[str, str | None, int]]
        """
        parts: list[tuple[str, str | None, int]] = []
        counts: dict[str | None, int] = {}
        for scope, rows, data in encoded:
            n = counts.get(scope, 0)
            counts[scope] = n + 1
            name = f"{self.base}/{seq}/{self._owner}/{table}/{_scope_segment(scope)}/{n}"
            pending.append((name, data))
            parts.append((name, scope, rows))
        return parts

    async def _put_pending(self, pending: list[tuple[str, bytes]], written: list[str], keys: tuple[str, str]) -> None:
        """put each encoded part, in order, each named in ``written`` once put (a failed write retires those), renewing
        the lock and claim every :data:`_RENEW_EVERY_SECONDS`.

        :param pending: each part's name and bytes
        :ptype pending: list[tuple[str, bytes]]
        :param written: the parts put so far
        :ptype written: list[str]
        :param keys: the lock and claim
        :ptype keys: tuple[str, str]
        :return: nothing
        :rtype: None
        """
        renewed = time.monotonic()
        # taken from the end of the list reversed: each part's bytes let go as it is put
        pending.reverse()
        while pending:
            name, data = pending.pop()
            await self._store.put(name, data)
            written.append(name)
            if time.monotonic() - renewed >= _RENEW_EVERY_SECONDS:
                for key in keys:
                    await self._renew(key)
                renewed = time.monotonic()

    async def _renew(self, key: str) -> None:
        """keep a lock or claim alive while the write goes on.

        :param key: the key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        entry = await self._pointers.get_entry(key=key)
        if entry is not None:
            await self._pointers.update(key=key, value=entry[0], revision=entry[1], ttl=_CLAIM_TTL)

    async def _retire_quietly(self, names: list[str], why: str) -> None:
        for start in range(0, len(names), 1000):
            try:
                await self._retire(names[start : start + 1000])
            except Exception as exc:  # noqa: BLE001 - what is left the next writer's sweep retires
                contain(
                    exc, log=log, what=f"checkpoint objects could not be retired ({why}); the next write retires them"
                )
                return

    @operator_reset(
        "an operator's reset of a copy follower's tables in its checkpoint: the named tables leave the newest and the "
        "previous checkpoint, so the next start reads each whole as its cold start at the log's head",
        proof="packages/core/tests/unit/collections/test_copy_checkpoint.py::"
        "test_forgotten_tables_load_as_a_cold_start_and_every_other_table_from_the_checkpoint",
    )
    async def forget(self, names: Iterable[str]) -> Checkpoint | None:
        """drop the named tables from the newest and the previous checkpoint, and retire the objects only they named:
        the next start reads each of them whole as its cold start (its mark taken at the log's head), and loads every
        other table from the checkpoint. An operator's act (``@operator_reset``), never a load's or a start's.

        A table derived from the followed ones stays in the manifest; a start that reads a followed table whole loads
        no derived table, which its owner derives again. A replica running now whose parts of a forgotten table were
        retired writes no checkpoint until it starts again (:meth:`write`): forget, then roll the pod.

        :param names: the tables to forget, each one this copy holds
        :ptype names: Iterable[str]
        :return: the checkpoint the pointer names now; None when there is none, or another replica holds the
            writer's lock or moved the pointer meanwhile (ask again)
        :rtype: Checkpoint | None
        :raises ValueError: when a name is not a table this copy holds
        """
        forgotten = frozenset(names)
        unknown = sorted(forgotten - {t.name for t in self._tables})
        if unknown:
            raise ValueError(f"the copy holds no table {unknown}: nothing to forget")
        lock = f"{self.base}-lock"
        if await self._pointers.create(key=lock, value=self._owner.encode(), ttl=_CLAIM_TTL) is None:
            return None
        try:
            found = await self.newest()
            if found is None:
                return None
            newest, revision = found
            previous = newest.previous
            kept = Checkpoint(
                seq=newest.seq + 1,
                stamp=newest.stamp,
                tables={n: e for n, e in newest.tables.items() if n not in forgotten},
                previous=None
                if previous is None
                else Checkpoint(
                    seq=previous.seq,
                    stamp=previous.stamp,
                    tables={n: e for n, e in previous.tables.items() if n not in forgotten},
                ),
            )
            if await self._pointers.update(key=self.base, value=kept.encode(), revision=revision) is None:
                return None
        finally:
            await self._pointers.delete(key=lock)
        gone = newest.live() - kept.live()
        await self._retire_quietly(sorted(gone), "forgotten by an operator")
        self._priors = without_entries(self._priors, forgotten)
        log.warning(
            "an operator forgot tables in the copy checkpoint; the next start reads each whole as its cold start",
            extra={
                "extra_data": {"base": self.base, "tables": sorted(forgotten), "seq": kept.seq, "retired": len(gone)}
            },
        )
        return kept

    async def sweep(self) -> int:
        """retire every object of this base that neither the newest nor the previous checkpoint names and no live
        claim is writing.

        :return: the objects asked to be retired
        :rtype: int
        """
        found = await self.newest()
        named = found[0].live() if found is not None else set()
        listed = await self._store.list_names(prefix=f"{self.base}/")
        stray = [name for name in listed if name not in named and not await self._claimed(name)]
        await self._retire_quietly(stray, "named by no pointer")
        return len(stray)

    async def _claimed(self, object_name: str) -> bool:
        seq = object_name.split("/")[1]
        keys = await self._pointers.list_keys(prefix=f"{self.base}.w.{seq}.")
        return bool(keys)


def _batches_cold_start(cursor: Any, table: str, rows: int) -> Any:
    """a table's rows on a read's cursor, as a reader of Arrow batches of at most ``rows`` rows, because no checkpoint
    holds it (the first write, one whose parts did not read back) or the follower read it whole at its cold start or an
    operator's reset (``copy_checkpoint_whole_writes_total``).

    :param cursor: the read's cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param table: the table, a trusted identifier
    :ptype table: str
    :param rows: the most rows a batch holds
    :ptype rows: int
    :return: the reader
    :rtype: pyarrow.RecordBatchReader
    """
    return whole_read(
        cursor.execute(f"SELECT * FROM {quote_identifier(table)}").fetch_record_batch(rows),  # noqa: S608
        kind="cold_start",
        reason="no checkpoint holds the table, or the follower read it whole at a cold start or reset: written whole",
        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade; copy_checkpoint split per reason",
        proof="packages/core/tests/unit/collections/test_copy_checkpoint.py::test_a_whole_write_is_reached_only_at_a_cold_start_a_schema_change_or_a_whole_read",
        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_checkpoint_whole_write_speed_at_real_size",
    )


def _batches_schema_change(cursor: Any, table: str, rows: int) -> Any:
    """a table's rows on a read's cursor, as :func:`_batches_cold_start` reads them, because its schema changed (every
    checkpoint holding it wrote another schema digest, or the follower read it whole after a migration).

    :return: the reader
    :rtype: pyarrow.RecordBatchReader
    """
    return whole_read(
        cursor.execute(f"SELECT * FROM {quote_identifier(table)}").fetch_record_batch(rows),  # noqa: S608
        kind="schema_change",
        reason="a migration changed the table: every checkpoint holding it wrote another schema; written whole again",
        ruling="Pace 2026-11-12: the one full load is a true no-data cold start; coordinator 2026-11-12: reset, schema change, rolling upgrade; copy_checkpoint split per reason",
        proof="packages/core/tests/unit/collections/test_copy_checkpoint.py::test_a_whole_write_is_reached_only_at_a_cold_start_a_schema_change_or_a_whole_read",
        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_checkpoint_whole_write_speed_at_real_size",
    )


def _batches_rolling_upgrade(cursor: Any, table: str, rows: int) -> Any:
    """a table's rows on a read's cursor, as :func:`_batches_cold_start` reads them, because the follower read it whole
    for a writer older than 0.95.0 that recorded the whole table mid-roll (gone by 0.96.0).

    :return: the reader
    :rtype: pyarrow.RecordBatchReader
    """
    return whole_read(
        cursor.execute(f"SELECT * FROM {quote_identifier(table)}").fetch_record_batch(rows),  # noqa: S608
        kind="rolling_upgrade",
        reason="a writer older than 0.95.0 recorded the whole table mid-roll: the checkpoint writes it whole again",
        until="0.96.0",
        ruling="Pace 2026-11-12: rolling_upgrade until 0.96.0 (the older writers' whole record is read for one release)",
        proof="packages/core/tests/unit/collections/test_copy_checkpoint.py::test_a_whole_write_is_reached_only_at_a_cold_start_a_schema_change_or_a_whole_read",
        speed="packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_checkpoint_whole_write_speed_at_real_size",
    )


#: each situation a checkpoint write reads a table whole in, onto its own declared read
WHOLE_WRITE_READS: Final[Mapping[str, Callable[[Any, str, int], Any]]] = {
    COLD_START: _batches_cold_start,
    SCHEMA_CHANGE: _batches_schema_change,
    "rolling_upgrade": _batches_rolling_upgrade,
}


def _reader_batches(reader: Any) -> Iterator[Any]:
    """each batch a reader gives, in order.

    :param reader: the reader
    :ptype reader: pyarrow.RecordBatchReader
    :return: the batches
    :rtype: Iterator[pyarrow.RecordBatch]
    """
    while (batch := _next_batch(reader)) is not None:
        yield batch


def _split_by_scope(reader: Any, table: FollowedTable) -> Iterator[tuple[str | None, Any]]:
    """each batch a reader gives, cut by the scope of its rows (None for a table with no scope).

    :param reader: the reader
    :ptype reader: pyarrow.RecordBatchReader
    :param table: the table
    :ptype table: FollowedTable
    :return: each scope and its rows of a batch, in order
    :rtype: Iterator[tuple[str | None, pyarrow.RecordBatch]]
    """
    for batch in _reader_batches(reader):
        if table.scope is None:
            yield None, batch
            continue
        for value, rows in _scope_runs(batch, table.scope[0]):
            yield table.scope_of(value), rows


def _scope_runs(batch: Any, column: str) -> Iterator[tuple[Any, Any]]:
    """each value of a batch's scope column with the rows holding it, in the order the values first appear: the batch
    cut where the value changes (slices, no copy) when each value's rows run together, as a copy's rows of one scope
    do; else cut the same way after one stable sort by value. One pass over the column, never one per value.

    :param batch: the rows
    :ptype batch: pyarrow.RecordBatch
    :param column: the scope column
    :ptype column: str
    :return: each value and its rows
    :rtype: Iterator[tuple[Any, pyarrow.RecordBatch]]
    """
    if batch.num_rows == 0:
        return
    encoded = batch.column(batch.schema.get_field_index(column)).dictionary_encode(null_encoding="encode")
    values = encoded.dictionary.to_pylist()
    codes = encoded.indices
    starts = _run_starts(codes)
    if len(starts) != len(values):
        order = pc.sort_indices(codes)
        batch, codes = batch.take(order), codes.take(order)
        starts = _run_starts(codes)
    for start, end in zip(starts, [*starts[1:], batch.num_rows], strict=True):
        yield values[codes[start].as_py()], batch.slice(start, end - start)


def _run_starts(codes: Any) -> list[int]:
    """where each run of equal codes starts.

    :param codes: the codes, none null
    :ptype codes: pyarrow.Array
    :return: the first row of each run, in order
    :rtype: list[int]
    """
    rows = len(codes)
    changed = pc.not_equal(codes.slice(1), codes.slice(0, rows - 1))
    return [0, *(int(index) + 1 for index in pc.indices_nonzero(changed).to_pylist())]


def _scope_batches(cursor: Any, table: str, column: KeyColumn, cast: str, value: str, rows: int) -> Any:
    """the rows of one scope on a read's cursor, as a reader of Arrow batches of at most ``rows``: those whose scope
    column holds ``value``, and for ``prefix + "None"`` those holding none too (FollowedTable.scope_of names both
    so). Read by the keyed batch join on the scope column, which keys the checkpoint's parts of the copy; its NULL
    bucket matched null-safe.

    :param cursor: the read's cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param table: the table, a trusted identifier
    :ptype table: str
    :param column: its scope column, a trusted identifier
    :ptype column: str
    :param cast: the column's SQL type, which the value's text is cast to
    :ptype cast: str
    :param value: the value, as text
    :ptype value: str
    :param rows: the most rows a batch holds
    :ptype rows: int
    :return: the reader
    :rtype: pyarrow.RecordBatchReader
    """
    values: list[str | None] = [None] if value == "None" else []
    if value != "None" or cast == "text":
        values.append(value)  # "None" is a value of a text column only
    pk_unnest, pk_join = unnest_key_join([column], [cast], null_safe=True)
    return cursor.execute(
        f"SELECT t.* FROM {quote_identifier(table)} AS t JOIN (SELECT {pk_unnest}) AS k ON {pk_join}",  # noqa: S608
        [values],
    ).fetch_record_batch(rows)


def _scope_rows(
    cursor: Any, table: str, column: str, cast: str, scope: str, value: str, rows: int
) -> Iterator[tuple[str, Any]]:
    """one scope's rows on a read's cursor, a batch at a time (:func:`_scope_batches`).

    :return: the scope and each batch
    :rtype: Iterator[tuple[str, pyarrow.RecordBatch]]
    """
    for batch in _reader_batches(_scope_batches(cursor, table, column, cast, value, rows)):
        yield scope, batch


def _patched_part(cursor: Any, table: FollowedTable, old: Any, keys: Any) -> Any:
    """a table's one part with the touched keys' rows replaced: the old part's rows of every other key, and the copy's
    rows of those keys (a read by key on a read's cursor).

    :param cursor: the read's cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param table: a table with no scope
    :ptype table: FollowedTable
    :param old: the part's rows as the checkpoint holds them; None for a table that held none
    :ptype old: pyarrow.Table | None
    :param keys: the touched keys, the table's key columns as text (a key perhaps more than once)
    :ptype keys: pyarrow.Table
    :return: the rows
    :rtype: pyarrow.Table
    """
    # the copy's rows of the keys: the keys one Arrow table the statement semi-joins (run_joined), as the copy's own
    # keyed reads take them (CopyFollower's read by key). Bound as Python lists, DuckDB probes each value: 25,000
    # touched keys of a 250,000-row part took 22.7 s to read against 0.17 s through Arrow (DuckDB 1.5.6, 2026-11-15;
    # the patch's real-size budget: test_copy_follower_whole_reads_live.py::test_checkpoint_patch_speed_at_real_size)
    named = ", ".join(table.held_key(column, "t") for column in table.key)
    cast = ", ".join(f"CAST(k{at + 1} AS {table.key_cast(column)})" for at, column in enumerate(table.key))
    fresh = run_joined(
        cursor,
        f"SELECT t.* FROM {quote_identifier(table.name)} AS t WHERE ({named}) IN (SELECT {cast} FROM ({KEYS}))",  # noqa: S608
        [keys.column(column) for column in table.key],
        lambda result: result.fetch_arrow_table(),
    )
    if old is None:
        return fresh
    # the old part's rows no touched key names: keyed.anti over the part and the touched keys, each staged on the
    # cursor (the keys as text, cast to each key column's type as the read above casts them)
    named_keys = {f"k{place}": keys.column(column) for place, column in enumerate(table.key)}
    typed = [
        keyed.Selected(f"CAST(t.k{place} AS {table.key_cast(column)})", column)
        for place, column in enumerate(table.key)
    ]
    cursor.register(_PART_STAGE, old)
    cursor.register(_PART_KEYS_STAGE, pa.table(named_keys))
    try:
        step = keyed.anti(
            _PART_STAGE,
            keyed.scan(_PART_KEYS_STAGE, typed),
            on=[keyed.Match(column) for column in table.key],
        )
        kept = run_inline(keyed.read_step_arrow(InlineDuckDB(cursor), step))
    finally:
        cursor.unregister(_PART_KEYS_STAGE)
        cursor.unregister(_PART_STAGE)
    return pa.concat_tables([kept, fresh.cast(kept.schema)])


def _next_batch(reader: Any) -> Any:
    try:
        return reader.read_next_batch()
    except StopIteration:  # NOSILENT: the reader's end is its answer: no more batches
        return None


# --- the jobs a write reads its parts in (copy_reads: a query worker over a shared copy) ---------------------------


def _encoded(batches: Iterator[tuple[str | None, Any]]) -> Any:
    """each (scope, batch) with rows encoded as one part (:func:`_next_part`), as a reader of one row a part
    (:data:`_PARTS_SCHEMA`): read as it is written, so the job holds one answer file's parts at a time, never every
    part of a table.

    :param batches: each scope and its rows
    :ptype batches: Iterator[tuple[str | None, pyarrow.RecordBatch | pyarrow.Table]]
    :return: the parts
    :rtype: pyarrow.RecordBatchReader
    """

    def parts() -> Iterator[Any]:
        while (item := _next_part(batches)) is not None:
            scope, rows, data = item
            yield pa.record_batch(
                [pa.array([scope], pa.string()), pa.array([rows], pa.int64()), pa.array([data], pa.large_binary())],
                schema=_PARTS_SCHEMA,
            )

    return pa.RecordBatchReader.from_batches(_PARTS_SCHEMA, parts())


def _whole_parts(connection: Any, args: Sequence[Any]) -> Any:
    """a job: a table's parts, read whole from the state by the situation's own declared read
    (:data:`WHOLE_WRITE_READS`), cut by scope and encoded.

    :param connection: the state's connection or cursor
    :ptype connection: duckdb.DuckDBPyConnection
    :param args: the situation (``cold_start``, ``schema_change``, ``rolling_upgrade``), the table, its part's rows
    :ptype args: Sequence[Any]
    :return: the parts (:func:`_encoded`)
    :rtype: pyarrow.RecordBatchReader
    """
    kind, table, rows = args
    return _encoded(_split_by_scope(WHOLE_WRITE_READS[kind](connection, table.name, rows), table))


def _scope_parts(connection: Any, args: Sequence[Any]) -> Any:
    """a job: a table's parts of some scopes, each read by its scope column (:func:`_scope_rows`) and encoded.

    :param connection: the state's connection or cursor
    :ptype connection: duckdb.DuckDBPyConnection
    :param args: the table, its scope column, that column's SQL type, its part's rows, and each scope with its value
    :ptype args: Sequence[Any]
    :return: the parts, scope by scope in the order given (:func:`_encoded`)
    :rtype: pyarrow.RecordBatchReader
    """
    table, column, cast, rows, scopes = args
    return _encoded(
        item
        for scope, value in scopes
        for item in _scope_rows(connection, table.name, column, cast, scope, value, rows)
    )


def _patched_parts(connection: Any, args: Sequence[Any]) -> Any:
    """a job: a table with no scope's one part patched by the touched keys (:func:`_patched_part`), its old bytes
    (handed as :data:`_PATCH_OLD`, when it held a part) decoded here and the result encoded.

    :param connection: the state's connection or cursor, the touched keys registered as :data:`_PATCH_KEYS`
    :ptype connection: duckdb.DuckDBPyConnection
    :param args: the table, the rows its old part's manifest entry says (-1 for no old part), and that part's name
    :ptype args: Sequence[Any]
    :return: the part (:func:`_encoded`; none when it holds no rows)
    :rtype: pyarrow.RecordBatchReader
    :raises ValueError: when the old part holds other than the rows its manifest says
    """
    table, expected, object_name = args
    keys = connection.table(_PATCH_KEYS).to_arrow_table()
    old = None
    if expected >= 0:
        old = _decode_part(connection.table(_PATCH_OLD).to_arrow_table().column("data")[0].as_py())
        if old.num_rows != expected:
            raise ValueError(f"part {object_name} holds {old.num_rows} rows, its manifest says {expected}")
    return _encoded(iter([(None, _patched_part(connection, table, old, keys))]))


def _old_part(data: bytes) -> Any:
    """a part's bytes as the one-row table a patch's job is handed (:data:`_PATCH_OLD`).

    :param data: the bytes
    :ptype data: bytes
    :return: the table
    :rtype: pyarrow.Table
    """
    return pa.table({"data": pa.array([data], pa.large_binary())})


def _parts_of(answer: Any) -> list[tuple[str | None, int, bytes]]:
    """a parts job's answer as each part's scope, rows and bytes (a Python object a part, never a row). Blocking:
    run it on a thread.

    :param answer: the answer (:data:`_PARTS_SCHEMA`)
    :ptype answer: pyarrow.Table
    :return: the parts, in order
    :rtype: list[tuple[str | None, int, bytes]]
    """
    found: list[tuple[str | None, int, bytes]] = []
    for batch in answer.to_batches():
        scopes, rows, datas = batch.column(0), batch.column(1), batch.column(2)
        for at in range(batch.num_rows):
            found.append((scopes[at].as_py(), int(rows[at].as_py()), datas[at].as_py()))
    return found
