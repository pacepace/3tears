"""the writer recreates a missing or unreadable delta file, once, from L3 by key; copies wait and ask it over NATS.

**Why.** A copy of a table that opts in to delta files applies every change from the file its epoch's record names
(:mod:`~threetears.core.collections.delta_files`). Until now a copy that found that file missing read the change's
keys from L3 itself, so a lost or damaged file sent every pod to L3 at once: the load that failed election night. Now
the copy reads nothing from L3, holds the table and asks the writer; the writer reads that one change's rows from L3 by
key, once, and writes the file back under the same name (``design-pod-delta-files.md``, decision 29, which amends 3).

**The request** rides 3tears' owner-routed forward (:func:`threetears.nats.forward.forward` /
:func:`~threetears.nats.forward.serve_owner`) in the ``delta-files`` family (:data:`_FAMILY`), keyed by the
deployment's own key (the write lease's): ``{ns}.forward.{sha256(delta-files)}.{sha256(key)}``. Its body is JSON
``{"table": ..., "file": ..., "epoch": ...}`` (:class:`_Request`); the reply is one word
(:class:`_Outcome`). Only the lease holder serves the subject (:func:`serving_recreates`, entered under the
lease beside :class:`~threetears.core.collections.delta_fold.DeltaFolds`), so a replica that is not the writer never
answers; with no writer, the copy hears ``no_writer`` and asks again later (:class:`DeltaRecreateRequester`).

**The recreation** (:class:`DeltaRecreator`). The file's name and keys come from the epoch's own record in the NATS
epoch log, never from the request: a record that does not name the file asked for is refused, so a request cannot
write anywhere else. The record's keys are read from L3 by key (:func:`~threetears.core.keyed.read_keyed_arrow`), its
key prefixes (a load's or a clear's) a page at a time by the prefix (:func:`~threetears.core.keyset.keyset_arrow_pages`);
a key L3 no longer holds becomes a delete marker. The rows are written as every column set (the row as L3 holds it now,
so it may be newer than the original; later changes apply over it, newest per key), the prefixes cleared first, through
:func:`~threetears.core.collections.delta_files.write_delta_file`: a temporary name, flushed, renamed over the name, the
directory flushed, so a reader sees the old file or the whole new one. A file that reads whole when asked is answered
``present`` and left alone. Requests for one file while it is being recreated wait for that one recreation.

**A commit too big to list its rows** (a keys-unknown record, decision 21) had its keys only in the file, so it cannot
be recreated by key: the table is handed to decision 28's recovery (:class:`LostDeltaRecovery`), which reads the keys
the leftover files name from L3 and publishes a full; the copy waits for that full (``recovering``).

Counted (``delta_file_recreations_total`` by outcome, ``delta_file_recreate_keys_read_total``) and logged at WARNING.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from threetears.nats.errors import RequestError
from threetears.nats.forward import ForwardedHandlerError, NoOwnerError, forward, serve_owner
from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE
from threetears.core.collections.copy_follower import FollowedTable, typed_text
from threetears.core.collections.delta_files import (
    DeltaFileUnreadableError,
    TableDelta,
    delta_files_root,
    read_delta_file,
    write_delta_file,
)
from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.schema_backed import rail_arrow_types, sql_cast
from threetears.core.keyed import KeyBatch, read_keyed_arrow
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.sql_fragments import quote_identifier

if TYPE_CHECKING:
    from threetears.nats.capabilities import RawRequester
    from threetears.nats.forward import ForwardTransport

    from threetears.core.collections.epoch_change import EpochChange, EpochChangeReader

__all__ = [
    "DeltaRecreateRequester",
    "DeltaRecreator",
    "LostDeltaRecovery",
    "serving_recreates",
]

log = get_logger(__name__)

#: the forward family recreate requests ride, so a NATS grant can name them
#: (``Subjects.forward_scoped_wildcard("delta-files")``) without granting every forward subject
_FAMILY: Final = "delta-files"

#: how long a copy waits for the writer's answer: a recreation reads one change's rows from L3 and writes them, seconds
#: for the largest listed change. A copy that times out hears ``no_writer`` and asks again at its next look, which the
#: writer answers ``present`` once the first recreation is done
_TIMEOUT: Final = timedelta(seconds=60)


class _Outcome(StrEnum):
    """the writer's answer to a recreate request (the copy's :class:`~threetears.core.collections.copy_follower.
    FileRecreate` answer).
    """

    #: the file was written again under its name, from L3
    RECREATED = "recreated"
    #: the file reads whole: nothing written
    PRESENT = "present"
    #: a commit too big to list: the table is handed to decision 28's recovery, whose full the copy waits for
    RECOVERING = "recovering"
    #: no lease holder answered (copy side only): asked again
    NO_WRITER = "no_writer"
    #: the writer's recreation raised, or the request was refused (copy side only): asked again
    FAILED = "failed"


#: recreations by the writer, by table and outcome (``recreated``, ``present``, ``recovering``, ``refused``, ``failed``)
_RECREATIONS = counter(
    "delta_file_recreations_total",
    description="the writer's answers to copies' requests to recreate a missing or unreadable delta file, by table "
    "and outcome",
    label_names=("table", "outcome"),
)

#: the keys the writer read from L3 to recreate files, by table: each a key one change's record names, read by key
_KEYS_READ = counter(
    "delta_file_recreate_keys_read_total",
    description="keys the writer read from L3 by key to recreate a delta file, by table; the only L3 read a lost "
    "file costs, once, never by every copy",
    label_names=("table",),
)


class _RefusedError(ValueError):
    """a recreate request the writer refuses: a table that does not opt in, a record that does not name the file asked
    for, no delta files' directory on the writer. Framed back to the copy, which counts it ``failed`` and asks again.
    """


@dataclass(frozen=True, slots=True)
class _Request:
    """a copy's request that the writer recreate one epoch's delta file.

    :ivar table: the table
    :ivar file: the file's name under the delta files' directory, as the epoch's record names it
    :ivar epoch: the epoch whose record names it
    """

    table: str
    file: str
    epoch: str

    def encode(self) -> bytes:
        """the request's body: JSON.

        :return: the bytes
        :rtype: bytes
        """
        return json.dumps({"table": self.table, "file": self.file, "epoch": self.epoch}).encode()

    @classmethod
    def decode(cls, payload: bytes) -> _Request:
        """a request's body read back.

        :param payload: the body
        :ptype payload: bytes
        :return: the request
        :rtype: _Request
        :raises ValueError: when the body is not a request: not JSON, or a field missing or not text
        """
        try:
            body = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"a recreate request is not JSON: {exc}") from exc
        fields = [body.get(name) if isinstance(body, dict) else None for name in ("table", "file", "epoch")]
        if not all(isinstance(value, str) and value for value in fields):
            raise ValueError(f"a recreate request names a table, a file and an epoch, as text: got {body!r}")
        table, file, epoch = fields
        return cls(table=str(table), file=str(file), epoch=str(epoch))


class LostDeltaRecovery(Protocol):
    """decision 28's recovery, for one table whose delta file cannot be recreated by key.

    The writer hands it a table when a commit too big to list its rows (its record lists none) lost its file: the keys
    were only in the file. The recovery reads, from L3 by key, every key the table's leftover delta files name, writes
    them as one recovery delta, folds it and publishes a full holding the table current to at least ``epoch`` in every
    part (a copy waiting for the file reloads the table from a full whose floor for it is at or past ``epoch``:
    :class:`~threetears.core.collections.copy_follower.CopyFollower`), and deletes the leftover files. It must coalesce:
    copies' requests for several such files of one table may each hand the table over. It runs only under the write
    lease, as the fold does. :meth:`~threetears.core.collections.delta_fold.DeltaFolds.recover_table` is the one the
    writing load hands over.
    """

    async def recover_table(self, table: str, *, epoch: str, file: str) -> None:
        """start (or join) the table's recovery; return once it is started, not when it is published.

        :param table: the table
        :ptype table: str
        :param epoch: the epoch whose file was lost
        :ptype epoch: str
        :param file: the lost file's name
        :ptype file: str
        :return: nothing
        :rtype: None
        """
        ...


class DeltaRecreator:
    """the writer's side: recreates one epoch's delta file from L3 by key, once however many copies ask.

    :param tables: the tables that opt in to delta files, each with every column its files hold
        (:meth:`~threetears.core.collections.copy_follower.FollowedTable.of` its schema)
    :ptype tables: Sequence[FollowedTable]
    :param l3: L3, the rail's proxy or anything with an asyncpg-shaped ``fetch``
    :ptype l3: Any
    :param changes: the epoch change log, which names each change's keys and file
    :ptype changes: EpochChangeReader
    :param recovery: decision 28's recovery, for a commit too big to list; None answers such a request ``failed``,
        logged
    :ptype recovery: LostDeltaRecovery | None
    :param page_size: keys a statement names, and rows a page of a prefix's read holds
    :ptype page_size: int
    :param root: the delta files' directory; None takes :func:`~threetears.core.collections.delta_files.delta_files_root`
        at each request
    :ptype root: Path | None
    """

    def __init__(
        self,
        tables: Sequence[FollowedTable],
        l3: Any,
        changes: EpochChangeReader,
        *,
        recovery: LostDeltaRecovery | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        root: Path | None = None,
    ) -> None:
        self._tables = {table.name: table for table in tables}
        self._l3 = l3
        self._changes = changes
        self._recovery = recovery
        self._page_size = page_size
        self._root = root
        #: each file being recreated, onto its one recreation: a request for it meanwhile waits for that one
        self._flight: dict[str, asyncio.Task[_Outcome]] = {}

    async def answer(self, payload: bytes) -> bytes:
        """the forward handler: a request's body in, the outcome's word out.

        :param payload: the request's body
        :ptype payload: bytes
        :return: the outcome
        :rtype: bytes
        :raises ValueError: when the request does not decode, or is refused (framed back to the copy, which asks again)
        :raises Exception: what the recreation raised (framed back likewise)
        """
        request = _Request.decode(payload)
        return (await self.recreate(request.table, request.file, request.epoch)).encode()

    async def recreate(self, table: str, file: str, epoch: str) -> str:
        """recreate one epoch's file once: a request for it while it is being recreated waits for that recreation.

        :param table: the table
        :ptype table: str
        :param file: the file's name, as the copy read it in the epoch's record
        :ptype file: str
        :param epoch: the epoch
        :ptype epoch: str
        :return: ``recreated``, ``present`` or ``recovering``
        :rtype: str
        :raises ValueError: when the request is refused: a table that does not opt in, or a record that does
            not name the file
        :raises LookupError: when the epoch log holds no record of the epoch, or a commit too big to list has no
            recovery wired
        """
        request = _Request(table=table, file=file, epoch=epoch)
        flight = self._flight.get(request.file)
        if flight is None:
            flight = asyncio.get_running_loop().create_task(
                self._recreate_once(request), name=f"recreate delta file {request.file}"
            )
            self._flight[request.file] = flight

            def landed(done: asyncio.Task[_Outcome]) -> None:
                if self._flight.get(request.file) is done:
                    del self._flight[request.file]

            flight.add_done_callback(landed)
        # shielded: a copy whose request is cancelled (its timeout) does not cancel the recreation the others wait for
        return (await asyncio.shield(flight)).value

    async def _recreate_once(self, request: _Request) -> _Outcome:
        try:
            outcome = await self._recreate(request)
        except _RefusedError:
            _RECREATIONS.labels(table=request.table, outcome="refused").inc()
            raise
        except Exception:
            # counted, and raised on to every copy waiting for it (framed back by serve_owner, logged there)
            _RECREATIONS.labels(table=request.table, outcome="failed").inc()
            raise
        _RECREATIONS.labels(table=request.table, outcome=outcome.value).inc()
        return outcome

    async def _recreate(self, request: _Request) -> _Outcome:
        table = self._tables.get(request.table)
        if table is None:
            raise _RefusedError(f"{request.table}: not a table whose delta files this writer recreates")
        root = self._root if self._root is not None else delta_files_root()
        if root is None:
            raise _RefusedError("no delta files' directory is mounted on the writer")
        record = await self._changes.change(table.name, request.epoch)
        if record is None:
            raise LookupError(f"{table.name}: the epoch log holds no record of {request.epoch}")
        if record.delta_file != request.file:
            # the file's name and keys come from the record alone: a request never names where the writer writes
            raise _RefusedError(
                f"{table.name}: the record of {request.epoch} names {record.delta_file!r}, not {request.file!r}"
            )
        path = root / request.file
        try:
            await asyncio.to_thread(read_delta_file, path, table=table.name)
        except FileNotFoundError:
            why = "file_missing"  # NOSILENT: recreated below, logged at WARNING
        except DeltaFileUnreadableError as exc:
            why = f"file_unreadable: {exc}"
        else:
            return _Outcome.PRESENT
        if record.keys_unknown:
            return await self._recover(table, request, why)
        delta, keys = await self._delta_from_l3(table, record)
        write_id = Path(request.file).stem
        written = await asyncio.to_thread(write_delta_file, root, delta, write_id)
        if written != request.file:
            raise ValueError(f"{table.name}: the recreated file was named {written!r}, not {request.file!r}")
        log.warning(
            "the writer recreated a delta file from L3 by key: a copy found it missing or unreadable",
            extra={
                "extra_data": {
                    "table": table.name,
                    "epoch": request.epoch,
                    "file": request.file,
                    "why": why,
                    "keys": keys,
                    "prefixes": len(delta.prefixes),
                    "entries": delta.entries,
                }
            },
        )
        return _Outcome.RECREATED

    async def _recover(self, table: FollowedTable, request: _Request, why: str) -> _Outcome:
        log.warning(
            "a commit too big to list lost its delta file: its keys were only in the file, so the table goes to the "
            "recovery delta (decision 28), whose full the copies wait for",
            extra={"extra_data": {"table": table.name, "epoch": request.epoch, "file": request.file, "why": why}},
        )
        if self._recovery is None:
            raise LookupError(
                f"{table.name}: {request.file} lists no keys and no recovery is wired to recover the table"
            )
        await self._recovery.recover_table(table.name, epoch=request.epoch, file=request.file)
        return _Outcome.RECOVERING

    async def _delta_from_l3(self, table: FollowedTable, record: EpochChange) -> tuple[TableDelta, int]:
        """the change's rows as L3 holds them now: its prefixes' rows a page at a time, then its keys by key; a key L3
        no longer holds a delete marker.

        :param table: the table
        :ptype table: FollowedTable
        :param record: the change's record
        :ptype record: EpochChange
        :return: the delta, and the keys read by key
        :rtype: tuple[TableDelta, int]
        """
        types = rail_arrow_types(table.columns)
        every = [column.name for column in table.columns]
        delta = TableDelta(name=table.name, types=types, key=tuple(table.key))
        prefixes = [dict(prefix) for prefix in record.prefixes or ()]
        if prefixes:
            # cleared before its rows, as the original file cleared them (decision 23)
            delta.clears(prefixes)
        selected = ", ".join(quote_identifier(name) for name in types)
        for prefix in prefixes:
            # the rows L3 holds under the prefix, a page at a time by the key: an index of the key reads it, a range
            columns = [column for column in table.key if column in prefix]
            last = columns[-1]
            pages = keyset_arrow_pages(
                self._l3,
                f"SELECT {selected} FROM {quote_identifier(table.name)}",  # noqa: S608 - trusted identifiers
                table.key,
                types=types,
                where={column: typed_text(table.kind(column), prefix[column]) for column in columns[:-1]} or None,
                where_any={last: (f"{sql_cast(table.kind(last))}[]", [typed_text(table.kind(last), prefix[last])])},
                page_size=self._page_size,
            )
            async for page in pages:
                delta.arrow(page, every)
        keys = list(dict.fromkeys(tuple(row.ids) for row in record.rows))
        casts = tuple(table.key_cast(column) for column in table.key)
        found: set[tuple[str, ...]] = set()
        for start in range(0, len(keys), self._page_size):
            chunk = keys[start : start + self._page_size]
            keyed = await read_keyed_arrow(
                self._l3, table.name, keys=KeyBatch(tuple(table.key), chunk, casts), types=types, batch=len(chunk)
            )
            _KEYS_READ.labels(table=table.name).inc(len(chunk))
            for page in keyed:
                if page.num_rows:
                    delta.arrow(page, every)
                    found.update(
                        zip(*(map(key_text, page.column(column).to_pylist()) for column in table.key), strict=True)
                    )
        gone = [key for key in keys if key not in found]
        if gone:
            delta.deletes([self._typed_key(table, key) for key in gone])
        return delta, len(keys)

    @staticmethod
    def _typed_key(table: FollowedTable, key: Sequence[str]) -> tuple[Any, ...]:
        return tuple(typed_text(table.kind(column), text) for column, text in zip(table.key, key, strict=True))


@asynccontextmanager
async def serving_recreates(nats: ForwardTransport, key: str, recreator: DeltaRecreator) -> AsyncIterator[None]:
    """answer copies' recreate requests while this context is active: enter it once the write lease is held (beside
    :class:`~threetears.core.collections.delta_fold.DeltaFolds`), and leave it the moment the lease is lost, so only
    the lease holder ever answers. A replica that is not the writer never subscribes, so a request never reaches it.

    :param nats: a connected client, or any :class:`~threetears.nats.forward.ForwardTransport`
    :ptype nats: ForwardTransport
    :param key: the deployment's key, the same on every replica (the write lease's): copies ask under it
    :ptype key: str
    :param recreator: the writer's recreations
    :ptype recreator: DeltaRecreator
    :return: a context serving while active
    :rtype: AsyncIterator[None]
    :raises SubscribeError: when the subscription is refused (the pod's grant lacks the family)
    """
    # sender-posture: trigger-only -- the body only names which epoch's record to read; the file's name and keys are
    # taken from that record in the epoch log, never from the body, so a request can at most recreate a real file
    async with serve_owner(nats, key, recreator.answer, family=_FAMILY):
        yield


class DeltaRecreateRequester:
    """the copy's side: asks whichever replica holds the write lease to recreate a delta file
    (:class:`~threetears.core.collections.copy_follower.FileRecreate`).

    :param nats: a connected client, or any :class:`~threetears.nats.capabilities.RawRequester`
    :ptype nats: RawRequester
    :param key: the deployment's key the writer serves under (:func:`serving_recreates`)
    :ptype key: str
    :param timeout: how long to wait for the writer's answer
    :ptype timeout: timedelta
    """

    def __init__(self, nats: RawRequester, key: str, *, timeout: timedelta = _TIMEOUT) -> None:
        self._nats = nats
        self._key = key
        self._timeout = timeout

    async def __call__(self, table: str, file: str, epoch: str) -> str:
        """ask once.

        :param table: the table
        :ptype table: str
        :param file: the file's name, as the epoch's record names it
        :ptype file: str
        :param epoch: the epoch
        :ptype epoch: str
        :return: the writer's :class:`_Outcome`, ``no_writer`` when none answered, ``failed`` when its
            recreation raised or the request did not cross
        :rtype: str
        """
        payload = _Request(table=table, file=file, epoch=epoch).encode()
        try:
            reply = await forward(self._nats, self._key, payload, timeout=self._timeout, family=_FAMILY)
        except NoOwnerError:
            return _Outcome.NO_WRITER.value  # NOSILENT: the copy counts and logs the answer, and asks again
        except (ForwardedHandlerError, RequestError) as exc:
            log.warning(
                "the writer could not recreate a delta file a copy asked for; it asks again",
                extra={"extra_data": {"table": table, "file": file, "epoch": epoch, "error": str(exc)}},
            )
            return _Outcome.FAILED.value
        try:
            return _Outcome(reply.decode()).value
        except (UnicodeDecodeError, ValueError) as exc:
            log.warning(
                "the writer answered a recreate request with no outcome this copy knows; it asks again",
                extra={"extra_data": {"table": table, "file": file, "reply": repr(reply[:64]), "error": str(exc)}},
            )
            return _Outcome.FAILED.value
