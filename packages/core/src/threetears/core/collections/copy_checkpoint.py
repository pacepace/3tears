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
replica's last checkpoint (:meth:`CopyCheckpoints.applied`) and names the unchanged parts' objects again. It runs
under a lock and a write claim in the pointer bucket and moves the pointer by compare-and-set. A failed or beaten
write retires only its own new objects and keeps what it was asked to write for the next one. When the pointer moves,
the checkpoint before the previous one is retired, never an object the newest or the previous names.

**Two code versions** share the one pointer during a roll. Each table's entry carries its schema digest, so a
version loads only the tables it reads the same way (from the newest, or the previous) and reads the rest whole. A
writer rewrites the tables it holds and keeps every other entry as it found it: a table it does not follow, or one it
has not read again yet.

**When.** Not on a timer: when the rows applied since the last checkpoint reach ``ratio`` (default half) of the rows it
holds, once after the first warm, and once a table left out of the load is read again (:meth:`CopyCheckpoints.due`).
Writes are counted in ``copy_checkpoint_writes_total{kind, outcome}``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except ImportError:  # NOSILENT: the snapshot extra is optional; a checkpoint says so when built
    pa = None
    ipc = None

__all__ = ["COLD_START", "SCHEMA_CHANGE", "Checkpoint", "CheckpointTable", "CopyCheckpoints", "StagedCheckpoint"]

log = get_logger(__name__)

#: a checkpoint is due when the rows applied since the last reach this share of the rows it holds
_DEFAULT_RATIO: Final = 0.5

#: the most rows one part of a checkpoint holds: what a writer or a loader holds of it at a time
_DEFAULT_PART_ROWS: Final = 250_000

#: how long a writer's claim and lock last unrenewed; renewed at every part
_CLAIM_TTL: Final = timedelta(minutes=2)

#: the manifest's format; a pointer of another is not read
_FORMAT: Final = 2

#: the object-name segment of a table with no scope
_NO_SCOPE: Final = "_"

#: why a table was left out of a load (:attr:`StagedCheckpoint.missing`): no checkpoint holds a part of it that reads
#: back; or every one holding it wrote it with another schema digest
COLD_START: Final = "cold_start"
SCHEMA_CHANGE: Final = "schema_change"


def _writes() -> Any:
    return counter(
        "copy_checkpoint_writes_total",
        description="copy checkpoint writes, by kind (full: every part written; incremental: unchanged parts named "
        "again) and outcome (written; failed; skipped: another replica holds the lock, the copy has no stamp yet, "
        "or a later checkpoint moved the pointer first)",
        label_names=("kind", "outcome"),
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
    """

    mark: str | None
    digest: str
    parts: tuple[tuple[str, str | None, int], ...]

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
        return {"mark": self.mark, "digest": self.digest, "parts": [list(part) for part in self.parts]}

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
    touched: dict[str, set[str] | None]
    everything: bool
    due: bool
    whole: frozenset[str]


def _add_touched(into: dict[str, set[str] | None], table: str, scopes: Iterable[str] | None) -> None:
    """count a table's touched scopes into ``into``; None (the table whole) wins over any scopes.

    :return: nothing
    :rtype: None
    """
    if scopes is None:
        into[table] = None
        return
    held = into.get(table, set())
    if held is not None:
        held.update(scopes)
        into[table] = held


class CopyCheckpoints:
    """a copy follower's checkpoints in the pod's Object Store and pointer bucket.

    :param name: the copy's name, a pointer-bucket token (letters, digits, ``-``, ``_``)
    :ptype name: str
    :param tables: the tables the copy holds
    :ptype tables: Sequence[FollowedTable]
    :param backend: the copy (each table's schema digest is recorded beside its parts)
    :ptype backend: DuckDBBackend
    :param store: the Object Store: ``put``, ``get``, ``list_names``
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
    :param part_rows: the most rows one part holds
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
        #: the checkpoint the changes counted here are since: a write after it names its unchanged parts again
        self._base_seq: int | None = None
        #: each table changed since, and the scopes of it changed (None: the table changed whole)
        self._touched: dict[str, set[str] | None] = {}
        #: a change was counted with no tables named: the next write writes every table
        self._everything = False
        #: tables the next write writes whole: left out of the load, or loaded from the previous checkpoint
        self._whole: set[str] = set()
        #: a checkpoint is due for a table of ``_whole``: the load left it out, or it was read again since
        self._whole_due = False
        #: (seq, table) whose parts did not read back here: never kept as they are
        self._broken: set[tuple[int, str]] = set()

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
        read back: the newest, or the previous.

        :return: the stamp, marks, stages and rows of the tables loaded, and each table left out and why
        :rtype: StagedCheckpoint
        """
        found = await self.newest()
        self._touched, self._everything, self._since = {}, False, 0
        if found is None:
            self._held_rows, self._base_seq, self._whole, self._whole_due = None, None, set(), False
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
                    changes.append(StagedChange(table=table.name, key=table.key, rows=stage, whole=True))
                    break
                self._broken.add((source.seq, table.name))
            else:
                missing[table.name] = COLD_START
        if missing:
            log.warning(
                "a checkpoint did not hold every table; the follower reads those whole from L3",
                extra={"extra_data": {"base": self.base, "missing": dict(missing), "loaded": sorted(loaded)}},
            )
        self._base_seq = newest.seq
        self._whole = {t.name for t in self._tables if loaded.get(t.name) is not newest}
        self._whole_due = bool(self._whole)
        self._held_rows = rows if loaded else None
        oldest = min(loaded.values(), key=lambda source: source.seq) if loaded else None
        return StagedCheckpoint(
            stamp=self._decode_stamp(oldest.stamp) if oldest is not None else None,
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
                part = await asyncio.to_thread(_decode_part, data)
                if part.num_rows != rows:
                    raise ValueError(f"part {object_name} holds {part.num_rows} rows, its manifest says {rows}")
                await asyncio.to_thread(self._backend.stage_rows, stage, table.name, part, table.key)
                del part, data
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

    # --- writing ---------------------------------------------------------------------------------

    def applied(self, rows: int, touched: Mapping[str, frozenset[str] | None] | None = None) -> None:
        """count a change the follower applied: its rows make the next checkpoint due, and what it touched is what
        the next checkpoint writes. Kept until a write lands.

        :param rows: the rows the change applied
        :ptype rows: int
        :param touched: each table the change touched, and the scopes of the rows it touched (for a table with no
            scope, any set); None for a table read whole. None for the whole argument: not known, and the next
            checkpoint writes every table
        :ptype touched: Mapping[str, frozenset[str] | None] | None
        :return: nothing
        :rtype: None
        """
        self._since += rows
        if touched is None:
            self._everything = True
            self._whole_due = self._whole_due or bool(self._whole)
            return
        for table, scopes in touched.items():
            self._whole_due = self._whole_due or table in self._whole
            _add_touched(self._touched, table, scopes)

    def due(self) -> bool:
        """whether a checkpoint is due: none is known, a table the load left out is due, or the rows applied since
        reach the ratio of its rows.

        :return: whether it is due
        :rtype: bool
        """
        if self._held_rows is None or self._whole_due:
            return True
        return self._since >= max(1, int(self._ratio * self._held_rows))

    def _incremental(self) -> bool:
        return self._base_seq is not None and not self._everything

    async def write(self, read: Callable[[], Any]) -> Checkpoint | None:
        """write a checkpoint of the copy as one read holds it, and move the pointer to it.

        :param read: opens the read: a context manager giving ``(cursor, stamp, marks)``
            (:meth:`~threetears.core.collections.copy_follower.CopyFollower.reading_marked`)
        :ptype read: Callable[[], Any]
        :return: the checkpoint written; None when another replica holds the lock, the copy has no stamp yet, or a
            later checkpoint got there first
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
            everything=self._everything,
            due=self._whole_due,
            whole=frozenset(self._whole),
        )
        self._touched, self._everything, self._whole_due = {}, False, False
        return taken

    def _give_back(self, taken: _Taken) -> None:
        """a write that did not land: what it took is counted again, beside what came since.

        :param taken: what it took
        :ptype taken: _Taken
        :return: nothing
        :rtype: None
        """
        for table, scopes in taken.touched.items():
            _add_touched(self._touched, table, scopes)
        self._everything = self._everything or taken.everything
        self._whole_due = self._whole_due or taken.due

    def _rewrite(self, table: FollowedTable, taken: _Taken, prior: CheckpointTable | None) -> set[str] | None:
        """the scopes of a table a write writes again (an empty set: none, every part named again); None: every one.

        :return: the scopes
        :rtype: set[str] | None
        """
        if prior is None or table.name in taken.whole:
            return None
        if table.name not in taken.touched:
            return set()
        changed = taken.touched[table.name]
        return None if changed is None or table.scope is None else changed

    async def _write_locked(self, read: Callable[[], Any], lock: str) -> Checkpoint | None:
        found = await self.newest()
        newest, revision = (found[0], found[1]) if found is not None else (None, None)
        seq = (newest.seq + 1) if newest is not None else 1
        claim = f"{self.base}.w.{seq}.{self._owner}"
        await self._pointers.create(key=claim, value=b"", ttl=_CLAIM_TTL)
        written: list[str] = []
        taken: _Taken | None = None
        kind = "incremental" if self._incremental() else "full"
        try:
            with read() as (cursor, stamp, marks):
                if stamp is None:
                    _writes().labels(kind=kind, outcome="skipped").inc()
                    return None
                # nothing awaits between the read beginning and this: every change it holds is counted in it
                taken = self._take()
                base = newest if newest is not None and newest.seq == self._base_seq and not taken.everything else None
                tables: dict[str, CheckpointTable] = {}
                plans: list[tuple[FollowedTable, CheckpointTable | None, set[str] | None]] = []
                for table in self._tables:
                    if table.name not in marks:
                        kept = self._kept(newest, table.name)
                        if kept is not None:
                            tables[table.name] = kept
                        continue
                    prior = base.tables.get(table.name) if base is not None else None
                    if prior is not None and prior.digest != self._digests[table.name]:
                        prior = None
                    plans.append((table, prior, self._rewrite(table, taken, prior)))
                kind = "incremental" if any(rewrite is not None for _, _, rewrite in plans) else "full"
                for table, prior, rewrite in plans:
                    parts = await self._write_table(cursor, table, seq, rewrite, prior, written, (lock, claim))
                    tables[table.name] = CheckpointTable(
                        mark=marks[table.name], digest=self._digests[table.name], parts=parts
                    )
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
        self._base_seq = seq
        self._whole -= written_whole
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
        self,
        cursor: Any,
        table: FollowedTable,
        seq: int,
        rewrite: set[str] | None,
        prior: CheckpointTable | None,
        written: list[str],
        keys: tuple[str, str],
    ) -> tuple[tuple[str, str | None, int], ...]:
        """one table's parts: the scopes in ``rewrite`` (every scope when None) written from the read, every other
        named again as ``prior`` holds it.

        :return: the parts, in order
        :rtype: tuple[tuple[str, str | None, int], ...]
        """
        if table.scope is None:
            if rewrite is not None and prior is not None:
                return prior.parts
            reader = await asyncio.to_thread(_batches, cursor, table.name, self._part_rows)
            return tuple(await self._put_parts([reader], table.name, None, seq, written, keys))
        column = table.scope[0]
        kept: dict[str, list[tuple[str, str | None, int]]] = {}
        if rewrite is not None and prior is not None:
            for part in prior.parts:
                if part[1] is not None and part[1] not in rewrite:
                    kept.setdefault(part[1], []).append(part)
            if not rewrite:
                return tuple(part for scope in sorted(kept) for part in kept[scope])
        values = await asyncio.to_thread(_scope_values, cursor, table.name, column)
        by_scope: dict[str, list[Any]] = {}
        for value in values:
            by_scope.setdefault(table.scope_of(value), []).append(value)
        parts: list[tuple[str, str | None, int]] = []
        for scope in sorted(set(by_scope) | set(kept)):
            if scope in kept:
                parts.extend(kept[scope])
                continue
            readers = [
                await asyncio.to_thread(_scope_batches, cursor, table.name, column, value, self._part_rows)
                for value in by_scope[scope]
            ]
            parts.extend(await self._put_parts(readers, table.name, scope, seq, written, keys))
        return tuple(parts)

    async def _put_parts(
        self,
        readers: Iterable[Any],
        table: str,
        scope: str | None,
        seq: int,
        written: list[str],
        keys: tuple[str, str],
    ) -> list[tuple[str, str | None, int]]:
        """store each batch the readers give as one part, renewing the lock and claim at each.

        :return: the parts
        :rtype: list[tuple[str, str | None, int]]
        """
        parts: list[tuple[str, str | None, int]] = []
        prefix = f"{self.base}/{seq}/{self._owner}/{table}/{_scope_segment(scope)}"
        for reader in readers:
            while True:
                batch = await asyncio.to_thread(_next_batch, reader)
                if batch is None:
                    break
                data = await asyncio.to_thread(_encode_part, pa.Table.from_batches([batch]))
                name = f"{prefix}/{len(parts)}"
                await self._store.put(name, data)
                written.append(name)
                parts.append((name, scope, batch.num_rows))
                for key in keys:
                    await self._renew(key)
        return parts

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


def _batches(cursor: Any, table: str, rows: int) -> Any:
    """a table's rows on a read's cursor, as a reader of Arrow batches of at most ``rows`` rows.

    :param cursor: the read's cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param table: the table, a trusted identifier
    :ptype table: str
    :param rows: the most rows a batch holds
    :ptype rows: int
    :return: the reader
    :rtype: pyarrow.RecordBatchReader
    """
    return cursor.execute(f"SELECT * FROM {quote_identifier(table)}").fetch_record_batch(rows)  # noqa: S608


def _scope_values(cursor: Any, table: str, column: str) -> list[Any]:
    """the values a table's scope column holds on a read's cursor, as Arrow gives them (what the follower names a
    row's scope from).

    :param cursor: the read's cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param table: the table, a trusted identifier
    :ptype table: str
    :param column: its scope column, a trusted identifier
    :ptype column: str
    :return: the values
    :rtype: list[Any]
    """
    found = cursor.execute(
        f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table)}"  # noqa: S608
    ).fetch_arrow_table()
    return list(found.column(0).to_pylist())


def _scope_batches(cursor: Any, table: str, column: str, value: Any, rows: int) -> Any:
    """the rows holding one scope-column value on a read's cursor, as a reader of Arrow batches of at most ``rows``.

    :param cursor: the read's cursor
    :ptype cursor: duckdb.DuckDBPyConnection
    :param table: the table, a trusted identifier
    :ptype table: str
    :param column: its scope column, a trusted identifier
    :ptype column: str
    :param value: the value; None for the rows that hold none
    :ptype value: Any
    :param rows: the most rows a batch holds
    :ptype rows: int
    :return: the reader
    :rtype: pyarrow.RecordBatchReader
    """
    quoted = quote_identifier(column)
    if value is None:
        return cursor.execute(
            f"SELECT * FROM {quote_identifier(table)} WHERE {quoted} IS NULL"  # noqa: S608
        ).fetch_record_batch(rows)
    return cursor.execute(
        f"SELECT * FROM {quote_identifier(table)} WHERE {quoted} = ?",  # noqa: S608
        [value],
    ).fetch_record_batch(rows)


def _next_batch(reader: Any) -> Any:
    try:
        return reader.read_next_batch()
    except StopIteration:  # NOSILENT: the reader's end is its answer: no more batches
        return None
