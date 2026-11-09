"""A copy follower's checkpoint: its whole copy at one stamp, in the pod's own Object Store (L2), so a start loads it in
seconds and walks the epoch change log from there.

**Why.** A :class:`~threetears.core.collections.copy_follower.CopyFollower` with no checkpoint starts by reading every
table whole from L3, a page a statement: minutes for an election night's tables. A checkpoint is the copy as Arrow,
written by a follower that is current, and a starting follower loads it straight into DuckDB, adopts the epoch marks
it was taken at, and applies only the epochs after them, through the standard catch-up. L3 is read whole only on the
very first start, when no checkpoint exists.

**Where.** The pod's own Object Store and pointer bucket (:func:`threetears.nats.object_store_requests.bind_pod_object_store`):
memory storage, three replicas, declared and swept by the hub. Nothing is on disk. A NATS restart that loses them
loses only the checkpoint: the next start reads L3 whole, and the first follower current again writes a new one.

**Immutable, swapped whole.** A checkpoint is written as new objects named by its sequence
(``{base}/{seq}/{table}/{part}``), each table cut into parts of at most ``part_rows`` rows (Arrow IPC, zstd), under a
write claim in the pointer bucket (``{base}.w.{seq}.{owner}``, the hub's sweep spares an object a claim may be
writing). Only when every part is stored does the pointer (``{base}``) move to it, by compare-and-set; the claim is
then let go and the checkpoint the pointer named before is retired. A writer that dies part way leaves objects no
pointer names, which the next writer retires; a reader only ever loads what a pointer names.

**One per schema.** ``base`` carries a digest of the copy's columns, so two code versions in a rolling upgrade each
keep their own checkpoint and never load the other's.

**When.** Not on a timer: when the rows applied since the last checkpoint reach ``ratio`` (default half) of the rows it
holds (:meth:`CopyCheckpoints.due`), and once after the first warm. One replica writes at a time (a lock key in the
pointer bucket, with a TTL renewed while it writes); the others skip.

**Loading** stages every part beside the copy, a part at a time, and the follower applies them in one transaction
with the marks the checkpoint names (:meth:`~threetears.core.collections.copy_follower.CopyFollower.start`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import contain, get_logger

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

__all__ = ["Checkpoint", "CopyCheckpoints"]

log = get_logger(__name__)

#: a checkpoint is due when the rows applied since the last reach this share of the rows it holds
_DEFAULT_RATIO: Final = 0.5

#: the most rows one part of a checkpoint holds: what a writer or a loader holds of it at a time
_DEFAULT_PART_ROWS: Final = 250_000

#: how long a writer's claim and lock last unrenewed; renewed at every part
_CLAIM_TTL: Final = timedelta(minutes=2)


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
class Checkpoint:
    """what a pointer names: a checkpoint's parts, the stamp and marks it holds the copy at.

    :ivar seq: its sequence, higher for a later one
    :ivar stamp: the writer's stamp, as ``encode_stamp`` wrote it
    :ivar marks: each table's epoch mark
    :ivar parts: each part's object name, table and rows, in order
    :ivar rows: the rows it holds
    """

    seq: int
    stamp: Any
    marks: Mapping[str, str | None]
    parts: tuple[tuple[str, str, int], ...]
    rows: int

    def encode(self) -> bytes:
        """the pointer's value: the checkpoint as JSON.

        :return: the bytes
        :rtype: bytes
        """
        return json.dumps(
            {
                "seq": self.seq,
                "stamp": self.stamp,
                "marks": dict(self.marks),
                "parts": [list(part) for part in self.parts],
                "rows": self.rows,
            },
            sort_keys=True,
        ).encode()

    @classmethod
    def decode(cls, raw: bytes) -> Checkpoint:
        """a pointer's value read back.

        :param raw: the bytes
        :ptype raw: bytes
        :return: the checkpoint
        :rtype: Checkpoint
        :raises ValueError: when it is not a checkpoint's JSON
        :raises KeyError: when a field is missing
        """
        body = json.loads(raw)
        return cls(
            seq=int(body["seq"]),
            stamp=body["stamp"],
            marks=dict(body["marks"]),
            parts=tuple((str(n), str(t), int(r)) for n, t, r in body["parts"]),
            rows=int(body["rows"]),
        )


class CopyCheckpoints:
    """a copy follower's checkpoints in the pod's Object Store and pointer bucket.

    :param name: the copy's name, a pointer-bucket token (letters, digits, ``-``, ``_``)
    :ptype name: str
    :param tables: the tables the copy holds
    :ptype tables: Sequence[FollowedTable]
    :param backend: the copy (its tables' columns digest the checkpoint's name)
    :ptype backend: DuckDBBackend
    :param store: the Object Store: ``put``, ``get``, ``list_names``
    :ptype store: Any
    :param pointers: the pointer bucket: ``get_entry``, ``create``, ``update``, ``delete``
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
        if "." in name or "." in owner or not name or not owner:
            raise ValueError("a checkpoint's name and owner are tokens with no '.'")
        self._tables = tuple(tables)
        digest = hashlib.sha256(
            "\n".join(f"{t.name}={backend.schema_digest(t.name)}" for t in self._tables).encode()
        ).hexdigest()[:12]
        self.base: Final = f"{name}-ckpt-{digest}"
        self._backend = backend
        self._store = store
        self._pointers = pointers
        self._retire = retire
        self._owner = owner
        self._encode_stamp = encode_stamp
        self._decode_stamp = decode_stamp
        self._ratio = ratio
        self._part_rows = part_rows
        self._held_rows: int | None = None
        self._since = 0

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
        except (ValueError, KeyError, TypeError) as exc:
            contain(exc, log=log, what="a checkpoint pointer could not be read; the start reads L3 whole")
            return None

    async def stage(self) -> tuple[Any, Mapping[str, str | None], list[StagedChange], int] | None:
        """stage the newest checkpoint beside the copy, a part at a time.

        :return: its stamp, its marks, the changes to apply (every table whole) and its rows; None when there is none,
            or a part could not be read (the stages are dropped and the start reads L3)
        :rtype: tuple[Any, Mapping[str, str | None], list[StagedChange], int] | None
        """
        found = await self.newest()
        if found is None:
            return None
        checkpoint, _ = found
        names = {t.name for t in self._tables}
        if set(checkpoint.marks) != names:
            log.warning(
                "a checkpoint names other tables than the copy holds; the start reads L3 whole",
                extra={"extra_data": {"base": self.base, "named": sorted(checkpoint.marks), "held": sorted(names)}},
            )
            return None
        stages = {t.name: f"{STAGE_PREFIX}{t.name}__rows" for t in self._tables}
        try:
            for table in self._tables:
                await asyncio.to_thread(
                    self._backend.stage_rows, stages[table.name], table.name, pa.table({}), table.key
                )
            for object_name, table_name, rows in checkpoint.parts:
                data = await self._store.get(object_name)
                part = await asyncio.to_thread(_decode_part, data)
                if part.num_rows != rows:
                    raise ValueError(f"part {object_name} holds {part.num_rows} rows, its pointer says {rows}")
                table = next(t for t in self._tables if t.name == table_name)
                await asyncio.to_thread(self._backend.stage_rows, stages[table_name], table_name, part, table.key)
                del part, data
        except Exception as exc:  # noqa: BLE001 - any part unread: the start reads L3, said
            contain(exc, log=log, what="a checkpoint could not be loaded whole; the start reads L3 whole")
            await asyncio.to_thread(self._backend.drop_stage, *stages.values())
            return None
        self._held_rows = checkpoint.rows
        self._since = 0
        changes = [StagedChange(table=t.name, key=t.key, rows=stages[t.name], whole=True) for t in self._tables]
        return self._decode_stamp(checkpoint.stamp), checkpoint.marks, changes, checkpoint.rows

    # --- writing ---------------------------------------------------------------------------------

    def applied(self, rows: int) -> None:
        """count rows a change applied: what makes the next checkpoint due.

        :param rows: the rows
        :ptype rows: int
        :return: nothing
        :rtype: None
        """
        self._since += rows

    def due(self) -> bool:
        """whether a checkpoint is due: none is known, or the rows applied since reach the ratio of its rows.

        :return: whether it is due
        :rtype: bool
        """
        if self._held_rows is None:
            return True
        return self._since >= max(1, int(self._ratio * self._held_rows))

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
            return None
        try:
            return await self._write_locked(read, lock)
        finally:
            await self._pointers.delete(key=lock)

    async def _write_locked(self, read: Callable[[], Any], lock: str) -> Checkpoint | None:
        found = await self.newest()
        previous, revision = (found[0], found[1]) if found is not None else (None, None)
        seq = (previous.seq + 1) if previous is not None else 1
        claim = f"{self.base}.w.{seq}.{self._owner}"
        await self._pointers.create(key=claim, value=b"", ttl=_CLAIM_TTL)
        parts: list[tuple[str, str, int]] = []
        written: list[str] = []
        try:
            with read() as (cursor, stamp, marks):
                if stamp is None:
                    return None
                for table in self._tables:
                    batches = await asyncio.to_thread(_batches, cursor, table.name, self._part_rows)
                    index = 0
                    while True:
                        batch = await asyncio.to_thread(_next_batch, batches)
                        if batch is None:
                            break
                        data = await asyncio.to_thread(_encode_part, pa.Table.from_batches([batch]))
                        name = f"{self.base}/{seq}/{table.name}/{index}"
                        await self._store.put(name, data)
                        written.append(name)
                        parts.append((name, table.name, batch.num_rows))
                        index += 1
                        await self._renew(lock)
                        await self._renew(claim)
                checkpoint = Checkpoint(
                    seq=seq,
                    stamp=self._encode_stamp(stamp),
                    marks=dict(marks),
                    parts=tuple(parts),
                    rows=sum(rows for _, _, rows in parts),
                )
            raw = checkpoint.encode()
            moved = (
                await self._pointers.create(key=self.base, value=raw)
                if revision is None
                else await self._pointers.update(key=self.base, value=raw, revision=revision)
            )
            if moved is None:
                await self._retire_quietly(written, "a later checkpoint moved the pointer first")
                return None
        except BaseException:
            await self._retire_quietly(written, "a checkpoint write failed part way")
            raise
        finally:
            await self._pointers.delete(key=claim)
        self._held_rows = checkpoint.rows
        self._since = 0
        if previous is not None:
            await self._retire_quietly([name for name, _, _ in previous.parts], "superseded by a newer checkpoint")
        log.info(
            "copy checkpoint written",
            extra={"extra_data": {"base": self.base, "seq": seq, "rows": checkpoint.rows, "parts": len(parts)}},
        )
        return checkpoint

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
        """retire every object of this base the pointer does not name and no live claim is writing.

        :return: the objects asked to be retired
        :rtype: int
        """
        found = await self.newest()
        named = {name for name, _, _ in found[0].parts} if found is not None else set()
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


def _next_batch(reader: Any) -> Any:
    try:
        return reader.read_next_batch()
    except StopIteration:  # NOSILENT: the reader's end is its answer: no more batches
        return None
