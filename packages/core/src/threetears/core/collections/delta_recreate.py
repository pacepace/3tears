"""the writer recreates a missing or unreadable delta file, once, from L3 by key; copies wait and ask it over NATS.

**Why.** A copy of a table that opts in to delta files applies every change from the file its epoch's record names
(:mod:`~threetears.core.collections.delta_files`). Until now a copy that found that file missing read the change's
keys from L3 itself, so a lost or damaged file sent every pod to L3 at once: the load that failed election night. Now
the copy reads nothing from L3, holds the table and asks the writer; the writer reads that one change's rows from L3 by
key, once, and writes the file back under the same name (``docs/design-pod-delta-files.md``, decision 29, which amends 3).

**The request** rides 3tears' owner-routed forward (:func:`threetears.nats.forward.forward` /
:func:`~threetears.nats.forward.serve_owner`) in the ``delta-files`` family
(:data:`~threetears.nats.DELTA_FILES_FORWARD_FAMILY`, the one spelling the NATS grant names too), keyed by the
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

**A failed load request's file, named unconfirmed** (``EpochChange.delta_file_unconfirmed``, ruling C of 2026-11-23).
A load commits a transaction per 50k rows, so a request that fails part way leaves a leading part of its rows in L3,
and its record names its file unconfirmed: the file holds every row the request sent, some perhaps never taken. Copies
wait and ask, as for a lost file. The writer reads that request's keys from the file itself (Arrow, on a thread; the
record lists none), reads those keys' rows from L3 by key, once (:func:`~threetears.core.keyed.read_keyed_arrow`), a key
L3 does not hold becoming a delete marker, and writes the file again under its name holding exactly that, marked
confirmed (:attr:`~threetears.core.collections.delta_files.DeltaKeys.confirmed`); the copies then apply it, and the
writer's fold is told it (:class:`_ConfirmedFolds`), so the fold holds the rows that did commit. A file already confirmed
is answered ``present``; one lost or unreadable (its keys were only in it) goes to decision 28's recovery.

**A load's intent record** (``EpochChange.unconfirmed_writer``, Pace's ruling of 2026-11-23). The hub names a load's
file unconfirmed BEFORE its first commit, beside its own writer beat's key, so a crash after a commit leaves a record
naming the rows. While the hub may still be loading, the file is never confirmed over it: a later record of the table
naming the same file supersedes the intent (the hub's confirmed record on success, its unconfirmed record naming no
writer on a failure it answered), and the copy is answered ``superseded`` and applies that record alone, once; else,
while the named beat is kept (:class:`~threetears.core.collections.lost_writes.WriterBeats`), the copy is answered
``pending`` and asks again; once the beat is gone (the hub stopped, or the platform's recovery declared it dead and
deleted it) the file is confirmed from L3 as above. Never by elapsed time. A writer given no beats to read never
confirms an intent the hub has not superseded (logged).

**No full at all** (Pace's ruling of 2026-11-24: "We have a mechanism to recover bad files; it should also detect no
files"). A copy starting from the volume whose current full lacks a table (none published: a new or emptied volume)
asks the writer for that table's full through the same request, ``{"table": ..., "want": "full"}``
(:meth:`DeltaRecreateRequester.full`). The writer answers through its fold (:class:`_MissingFulls`,
:meth:`~threetears.core.collections.delta_fold.DeltaFolds.full_owed`): ``present`` when the full holds it, else
``recovering`` once the fold is set to rebuild it from L3 (decision 14's cold start, its one whole read) and publish
it, with no write begun; the copy reads nothing from L3 and waits for that full. A writer given no fold for it refuses
(the copy asks again). An older writer refuses the request too (it names no file): the copy asks again, and the newer
writer answers it.

Counted (``delta_file_recreations_total`` by outcome, ``delta_file_recreate_keys_read_total``,
``delta_full_requests_total``) and logged at WARNING.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.nats import DELTA_FILES_FORWARD_FAMILY
from threetears.nats.errors import RequestError
from threetears.nats.forward import ForwardedHandlerError, NoOwnerError, forward, serve_owner
from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE
from threetears.core.collections.copy_follower import FollowedTable, typed_text
from threetears.core.collections.delta_files import (
    DeltaFileUnreadableError,
    TableDelta,
    _first_written,
    _key_texts,
    _text_column,
    _under_texts,
    delta_files_root,
    joined_key_texts,
    key_text_table,
    read_delta_file_keys,
    read_delta_table,
    write_delta_file,
)
from threetears.core.collections.delta_fold import _typed_texts
from threetears.core.collections.schema_backed import rail_arrow_types, sql_cast
from threetears.core.collections.volume_fulls import current_full_manifest
from threetears.core.keyed import KeyBatch, read_keyed_arrow
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.sql_fragments import quote_identifier

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from threetears.nats.capabilities import RawRequester

    from threetears.core.collections.generation import GenerationSource
    from threetears.nats.forward import ForwardTransport

    from threetears.core.collections.epoch_change import EpochChange, EpochChangeReader

__all__ = [
    "DeltaRecreateRequester",
    "DeltaRecreator",
    "LostDeltaRecovery",
    "serving_recreates",
]

log = get_logger(__name__)

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
    #: a load's intent record whose loading writer may still be loading: nothing written, asked again
    PENDING = "pending"
    #: a load's intent record a later record of the table names the same file in: that record alone applies it
    SUPERSEDED = "superseded"
    #: the file is gone and the current full covers it by its name (decisions 53 and 54: a fold that deletes took it
    #: once every copy known was past it): never written again from L3; the copy takes that change from the full
    COVERED = "covered"


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


#: copies' asks for a table's full the current one lacks, answered by the writer, by table and outcome: ``present``
#: (the full holds it), ``recovering`` (its fold rebuilds it from L3 and publishes it), ``refused`` (a table it does not
#: fold, or no fold wired)
_FULL_ASKS = counter(
    "delta_full_requests_total",
    description="the writer's answers to copies' asks for a table's full the volume's current full lacks, by table and "
    "outcome; recovering: its fold rebuilds the full from L3 once, with no write begun",
    label_names=("table", "outcome"),
)

#: between a reply's outcome word and the writer's progress after it (a rebuild's step: ``DeltaRecreator.full``)
_PROGRESS_SEPARATOR: Final = "\n"

#: the body's ``want`` naming a request for a table's full, not a file
_WANT_FULL: Final = "full"


class _RefusedError(ValueError):
    """a recreate request the writer refuses: a table that does not opt in, a record that does not name the file asked
    for, no delta files' directory on the writer. Framed back to the copy, which counts it ``failed`` and asks again.
    """


@dataclass(frozen=True, slots=True)
class _Request:
    """a copy's request that the writer recreate one epoch's delta file, or (``full``) publish a table's full.

    :ivar table: the table
    :ivar file: the file's name under the delta files' directory, as the epoch's record names it; empty for a full
    :ivar epoch: the epoch whose record names it; empty for a full
    :ivar full: whether the copy asks for the table's full (module docstring, "No full at all")
    """

    table: str
    file: str
    epoch: str
    full: bool = False

    def encode(self) -> bytes:
        """the request's body: JSON.

        :return: the bytes
        :rtype: bytes
        """
        if self.full:
            return json.dumps({"table": self.table, "want": _WANT_FULL}).encode()
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
        if isinstance(body, dict) and body.get("want") == _WANT_FULL:
            table = body.get("table")
            if not isinstance(table, str) or not table:
                raise ValueError(f"a request for a full names a table, as text: got {body!r}")
            return cls(table=table, file="", epoch="", full=True)
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


class _MissingFulls(Protocol):
    """the writer's fold, asked for a table's full a copy found the volume's current full lacking (module docstring,
    "No full at all"). :meth:`~threetears.core.collections.delta_fold.DeltaFolds.full_owed` is the fold's own; a writer
    whose fold starts only with a write wraps it to start the fold first (the reports pod's store).
    """

    async def full_owed(self, table: str) -> bool:
        """whether the current full lacks ``table``; when it does, the fold is set to rebuild it from L3 and publish
        it, and this returns without waiting for that.

        :param table: the table
        :ptype table: str
        :return: True when a full is owed (the copy waits for it), False when the current full holds the table
        :rtype: bool
        """
        ...


@runtime_checkable
class _RebuildProgress(Protocol):
    """a writer's fold that says what its rebuild of a missing full is doing now
    (:attr:`~threetears.core.collections.delta_fold.DeltaFolds.rebuilding`): sent to the copy with ``recovering``, so a
    copy waiting minutes for a cold rebuild says which table is being read, never a silent wait.
    """

    @property
    def rebuilding(self) -> str | None:
        """``rebuilding from L3: n of m tables read`` or ``publishing ...`` while a rebuild runs; None while none does.

        :return: the step, or None
        :rtype: str | None
        """
        ...


@runtime_checkable
class _ConfirmedFolds(Protocol):
    """the writer's fold, told each failed load request's file the writer confirmed from L3: the rows that committed
    are folded as any commit's. :class:`~threetears.core.collections.delta_fold.DeltaFolds` is the one the writing load
    hands over (it is also the recovery, so a recreator given it as ``recovery`` tells it without more wiring).
    """

    def confirmed(self, table: str, *, file: str, epoch: str) -> None:
        """note a confirmed file, at its record's epoch; never waits, never raises for a fold.

        :param table: the table
        :ptype table: str
        :param file: the file, under the root
        :ptype file: str
        :param epoch: the epoch whose record names it
        :ptype epoch: str
        :return: nothing
        :rtype: None
        """
        ...


@runtime_checkable
class _WriterLiveness(Protocol):
    """whether a loading writer is gone, by its beat's key: :class:`~threetears.core.collections.lost_writes
    .WriterBeats` is the one the platform keeps.
    """

    async def gone(self, key: str) -> bool:
        """whether the writer whose beat is kept under ``key`` is gone.

        :param key: the beat's key
        :ptype key: str
        :return: whether it is gone
        :rtype: bool
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
    :param folds: the writer's fold, told each failed load request's file once confirmed; None takes ``recovery`` when
        it is one (:class:`_ConfirmedFolds`), else tells none
    :ptype folds: _ConfirmedFolds | None
    :param page_size: keys a statement names, and rows a page of a prefix's read holds
    :ptype page_size: int
    :param root: the delta files' directory; None takes :func:`~threetears.core.collections.delta_files.delta_files_root`
        at each request
    :ptype root: Path | None
    :param writers: the loading writers' liveness, for a load's intent record (module docstring); None confirms no
        intent the hub has not superseded
    :ptype writers: _WriterLiveness | None
    :param generations: the tables' current generations, to walk the log past an intent record for the record that
        supersedes it; None reads no such record
    :ptype generations: GenerationSource | None
    :param fulls: the writer's fold, asked for a table's full a copy found lacking (:class:`_MissingFulls`); None
        refuses such an ask (the copy asks again)
    :ptype fulls: _MissingFulls | None
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
        folds: _ConfirmedFolds | None = None,
        writers: _WriterLiveness | None = None,
        generations: GenerationSource | None = None,
        fulls: _MissingFulls | None = None,
    ) -> None:
        self._fulls = fulls
        self._writers = writers
        self._generations = generations
        self._tables = {table.name: table for table in tables}
        self._l3 = l3
        self._changes = changes
        self._recovery = recovery
        self._folds = folds if folds is not None else recovery if isinstance(recovery, _ConfirmedFolds) else None
        self._page_size = page_size
        self._root = root
        #: each file being recreated, onto its one recreation: a request for it meanwhile waits for that one. A
        #: read-only snapshot replaced whole on the loop: a request adds its file's flight, and the flight removes
        #: itself as it ends (:meth:`_recreate_once`), so each entry has one holder
        self._flight: Mapping[str, asyncio.Task[_Outcome]] = MappingProxyType({})

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
        if request.full:
            return (await self.full(request.table)).encode()
        return (await self.recreate(request.table, request.file, request.epoch)).encode()

    async def full(self, table: str) -> str:
        """a copy's ask for ``table``'s full (module docstring, "No full at all"): the fold says whether the current full
        holds it, and when not is set to rebuild it from L3 and publish it; never waits for that publish.

        :param table: the table
        :ptype table: str
        :return: ``present`` (the full holds it) or ``recovering`` (the fold publishes it; the copy waits for it)
        :rtype: str
        :raises ValueError: when the ask is refused: a table that does not opt in, or no fold wired to rebuild it
        """
        if table not in self._tables or self._fulls is None:
            _FULL_ASKS.labels(table=table, outcome="refused").inc()
            raise _RefusedError(
                f"{table}: not a table whose full this writer publishes"
                if table not in self._tables
                else f"{table}: no fold is wired to rebuild a missing full"
            )
        owed = await self._fulls.full_owed(table)
        outcome = _Outcome.RECOVERING if owed else _Outcome.PRESENT
        _FULL_ASKS.labels(table=table, outcome=outcome.value).inc()
        progress = self._fulls.rebuilding if owed and isinstance(self._fulls, _RebuildProgress) else None
        if owed:
            log.warning(
                "a copy found no full of a table on the volume: the writer's fold rebuilds it from L3 and publishes it, "
                "with no write begun",
                extra={"extra_data": {"table": table, "progress": progress}},
            )
        # the outcome word, then the rebuild's step on a line of its own (_PROGRESS_SEPARATOR): the copy says it in its
        # waiting status (DeltaRecreateRequester._asked)
        return outcome.value if progress is None else f"{outcome.value}{_PROGRESS_SEPARATOR}{progress}"

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
        if flight is None or flight.done():
            flight = asyncio.get_running_loop().create_task(
                self._recreate_once(request), name=f"recreate delta file {request.file}"
            )
            # WHY (2026-11-24): a copy whose wait timed out leaves the shielded flight with no waiter, and its failure
            # went unretrieved ("Task exception was never retrieved", a local replay's writer log); the flight's own
            # callback retrieves and logs it whether or not anyone still waits
            flight.add_done_callback(_flight_ended)
            self._flight = MappingProxyType({**self._flight, request.file: flight})
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
        finally:
            # the flight lets its own entry go as it ends: a request after it starts a recreation of its own
            if self._flight.get(request.file) is asyncio.current_task():
                self._flight = MappingProxyType(
                    {file: flight for file, flight in self._flight.items() if file != request.file}
                )
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
            # no record names the epoch's file, or its keys: a copy waiting on it is never read around (decision 29),
            # so the table is recovered from L3 once and published as a full (decision 28), which the copies wait for
            return await self._recover(table, request, "record_missing")
        if record.delta_file != request.file:
            # the file's name and keys come from the record alone: a request never names where the writer writes
            raise _RefusedError(
                f"{table.name}: the record of {request.epoch} names {record.delta_file!r}, not {request.file!r}"
            )
        if record.delta_file_unconfirmed:
            if record.unconfirmed_writer is not None:
                held = await self._intent_held(table, request, record.unconfirmed_writer)
                if held is not None:
                    return held
            return await self._confirm(table, request, root)
        path = root / request.file
        try:
            lacking = await asyncio.to_thread(_keys_lacking, path, table, record)
        except FileNotFoundError:
            if await asyncio.to_thread(_covered_by_full, root, table.name, request.file):
                # deleted by a fold that deletes once its full covered it (decision 54): the full holds the change, so
                # nothing is read from L3 and nothing written; the copy takes it from the full
                log.warning(
                    "a copy asked for a delta file the current full covers: answered covered, nothing read from L3",
                    extra={"extra_data": {"table": table.name, "epoch": request.epoch, "file": request.file}},
                )
                return _Outcome.COVERED
            why = "file_missing"  # NOSILENT: recreated below, logged at WARNING
        except DeltaFileUnreadableError as exc:
            why = f"file_unreadable: {exc}"
        else:
            if not lacking:
                return _Outcome.PRESENT
            # it reads, and holds no change of a key its record names: a copy cannot apply that key from it
            why = f"key_not_in_file: {lacking} of the record's keys"
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

    async def _intent_held(self, table: FollowedTable, request: _Request, writer: str) -> _Outcome | None:
        """a load's intent record: ``superseded`` once a later record of the table names its file, ``pending`` while its
        loading writer's beat is kept, ``None`` (confirm it) once that beat is gone (module docstring).

        :param table: the table
        :ptype table: FollowedTable
        :param request: the copy's request, its epoch the intent's
        :ptype request: _Request
        :param writer: the loading writer's beat key, as the record names it
        :ptype writer: str
        :return: the answer, or None to confirm the file now
        :rtype: _Outcome | None
        """
        if await self._superseded(table.name, request):
            log.info(
                "a load's intent record is superseded by a later record naming its file: that record alone applies it",
                extra={"extra_data": {"table": table.name, "epoch": request.epoch, "file": request.file}},
            )
            return _Outcome.SUPERSEDED
        if self._writers is None:
            log.warning(
                "a load's intent record names its loading writer, and this writer reads no beats: it is never "
                "confirmed until a later record supersedes it",
                extra={"extra_data": {"table": table.name, "epoch": request.epoch, "writer": writer}},
            )
            return _Outcome.PENDING
        if not await self._writers.gone(writer):
            return _Outcome.PENDING
        # gone: no record of its own comes now. Looked for once more, so a record written just before it went is the
        # one applied, never the file confirmed over it as well
        if await self._superseded(table.name, request):
            return _Outcome.SUPERSEDED
        log.warning(
            "a load's loading writer is gone and no record superseded its intent: the file is confirmed from L3",
            extra={"extra_data": {"table": table.name, "epoch": request.epoch, "writer": writer}},
        )
        return None

    async def _superseded(self, table: str, request: _Request) -> bool:
        """whether a record of ``table`` after the request's epoch names the request's file (the epoch log, NATS: never
        L3).

        :param table: the table
        :ptype table: str
        :param request: the request
        :ptype request: _Request
        :return: whether one does
        :rtype: bool
        """
        if self._generations is None:
            return False
        now = await self._generations.current(table)
        if now == request.epoch:
            return False
        walk = await self._changes.changes_after(table, request.epoch, now)
        return any(change.delta_file == request.file and change.token != request.epoch for change in walk.changes)

    async def _confirm(self, table: FollowedTable, request: _Request, root: Path) -> _Outcome:
        """write a failed load request's file again from L3 (module docstring): its keys, read from the file, read from
        L3 by key once; written under its name marked confirmed; the fold told.

        :param table: the table
        :ptype table: FollowedTable
        :param request: the copy's request, its file the one the record names unconfirmed
        :ptype request: _Request
        :param root: the delta files' directory
        :ptype root: Path
        :return: ``recreated`` once confirmed, ``present`` when it was already, ``recovering`` when the file is lost
        :rtype: _Outcome
        """
        try:
            held = await asyncio.to_thread(read_delta_file_keys, root / request.file, table=table.name)
        except FileNotFoundError:
            # the request's keys were only in its file: decision 28's recovery reads what the leftover files name
            return await self._recover(table, request, "unconfirmed_file_missing")
        except DeltaFileUnreadableError as exc:
            return await self._recover(table, request, f"unconfirmed_file_unreadable: {exc}")
        if held.confirmed:
            return _Outcome.PRESENT
        delta = TableDelta(name=table.name, types=rail_arrow_types(table.columns), key=tuple(table.key))
        keys = await self._keys_from_l3(table, delta, held.keys)
        write_id = Path(request.file).stem
        written = await asyncio.to_thread(write_delta_file, root, delta, write_id, confirmed=True)
        if written != request.file:
            raise ValueError(f"{table.name}: the confirmed file was named {written!r}, not {request.file!r}")
        log.warning(
            "the writer confirmed a failed load request's delta file from L3 by key: the rows that committed, as L3 "
            "holds them, the rest delete markers",
            extra={
                "extra_data": {
                    "table": table.name,
                    "epoch": request.epoch,
                    "file": request.file,
                    "keys": keys,
                    "entries": delta.entries,
                }
            },
        )
        if self._folds is not None:
            self._folds.confirmed(table.name, file=request.file, epoch=request.epoch)
        return _Outcome.RECREATED

    async def _recover(self, table: FollowedTable, request: _Request, why: str) -> _Outcome:
        log.warning(
            "a commit too big to list lost its delta file (its keys were only in the file), or the log lost an epoch's "
            "record: the table goes to the recovery delta (decision 28), whose full the copies wait for",
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
                table=table.name,
            )
            async for page in pages:
                delta.arrow(page, every)
        # the record's keys (their text, ``key_text``'s spelling) as Arrow key columns, each once in the order named;
        # what L3 answers is matched against them in Arrow, never a key held as a Python object
        # an instant key in its one canonical text (key_text with ``instant``), as L3's rows are matched below
        keys = (
            _first_written(key_text_table(_record_keys(record, table.key), table.key, table.instant_keys), table.key)
            if record.rows
            else None
        )
        if keys is None or not keys.num_rows:
            return delta, 0
        return delta, await self._keys_from_l3(table, delta, keys)

    async def _keys_from_l3(self, table: FollowedTable, delta: TableDelta, keys: Any) -> int:
        """add to ``delta`` each key's row as L3 holds it now, read by key a page at a time, every column set; a key L3
        no longer holds a delete marker.

        :param table: the table
        :ptype table: FollowedTable
        :param delta: the file being made
        :ptype delta: TableDelta
        :param keys: the keys, text key columns named as the key's, each once
        :ptype keys: pyarrow.Table
        :return: the keys read by key
        :rtype: int
        """
        if not keys.num_rows:
            return 0
        types = rail_arrow_types(table.columns)
        every = [column.name for column in table.columns]
        casts = tuple(table.key_cast(column) for column in table.key)
        found: list[Any] = []
        for start in range(0, keys.num_rows, self._page_size):
            chunk = keys.slice(start, self._page_size)
            columns = [chunk.column(column) for column in table.key]
            keyed = await read_keyed_arrow(
                self._l3, table.name, keys=KeyBatch(tuple(table.key), columns, casts), types=types, batch=chunk.num_rows
            )
            _KEYS_READ.labels(table=table.name).inc(chunk.num_rows)
            for page in keyed:
                if page.num_rows:
                    delta.arrow(page, every)
                    # the rail's ``...Z`` of an instant key matched in the record's canonical text, never as given
                    found.append(joined_key_texts(key_text_table(page, table.key, table.instant_keys), table.key))
        held = pa.concat_arrays(found) if found else pa.array([], pa.string())
        gone = keys.filter(pc.invert(pc.is_in(joined_key_texts(keys, table.key), value_set=held)))
        if gone.num_rows:
            # a key L3 no longer holds: a delete marker, its values typed as the column binds them
            delta.deletes(
                pa.table(
                    {
                        column: _typed_texts(gone.column(column), table.kind(column), types[column])
                        for column in table.key
                    }
                )
            )
        return int(keys.num_rows)


def _covered_by_full(root: Path, table: str, file: str) -> bool:
    """whether the current full on the volume covers ``file`` of ``table`` by its name (decision 53); blocking.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: str
    :param file: the delta file, under the root
    :ptype file: str
    :return: whether it is covered
    :rtype: bool
    """
    current = current_full_manifest(root)
    entry = current.tables.get(table) if current is not None else None
    return entry is not None and entry.covers(file)


def _flight_ended(flight: asyncio.Task[_Outcome]) -> None:
    """retrieve a recreation's failure and log it, so none goes unretrieved when every copy waiting on it gave up.

    :param flight: the recreation, done
    :ptype flight: asyncio.Task[_Outcome]
    :return: None
    :rtype: None
    """
    if flight.cancelled():
        return
    failure = flight.exception()
    if failure is not None:
        log.warning(
            "a delta file's recreation failed; a copy asking again starts another",
            extra={"extra_data": {"task": flight.get_name(), "error": repr(failure)}},
        )


def _record_keys(record: EpochChange, key: Sequence[str]) -> Any:
    """the keys a record names, as text key columns: one row a key, in the order named (the record holds each key's
    text already, ``key_text``'s spelling).

    :param record: the record
    :ptype record: EpochChange
    :param key: the key columns, in key order
    :ptype key: Sequence[str]
    :return: the keys
    :rtype: pyarrow.Table
    """
    return pa.table(
        {column: pa.array([row.ids[index] for row in record.rows], pa.string()) for index, column in enumerate(key)}
    )


def _keys_lacking(path: Path, table: FollowedTable, record: EpochChange) -> int:
    """how many of the keys a record names its file holds no change of, nor clears a prefix of: the file's checksum
    verified and every array validated in full (:func:`~threetears.core.collections.delta_files.read_delta_table`),
    then its keys matched against the record's in Arrow, never a row of the file decoded to Python. Blocking: run it on
    a thread.

    Why: a copy that finds a key its record named missing from the file waits and asks (decision 29); a file that only
    reads whole would be answered ``present`` and the copy would ask for ever. Why Arrow: the old check built every
    key of the file as Python text (``to_pylist``), a per-row loop on the writer's serving process for a large file.

    :param path: the file
    :ptype path: Path
    :param table: the table
    :ptype table: FollowedTable
    :param record: the epoch's record
    :ptype record: EpochChange
    :return: how many keys it lacks; 0 for a record that lists none
    :rtype: int
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when it is there and is not the table's delta file
    """
    delta = read_delta_table(path, table=table.name)
    if not record.rows:
        return 0
    instants = table.instant_keys
    # every side in key_text's one canonical spelling (an instant's with ``instant``): a file's ``...Z`` and the
    # record's ``...+00:00`` of one instant are one key; matched as given, every such key read as lacking
    held = _key_texts([_text_column(delta.rows.column(column), instant=column in instants) for column in table.key])
    named = key_text_table(_record_keys(record, table.key), table.key, instants)
    lacking = np.asarray(
        pc.invert(pc.is_in(joined_key_texts(named, table.key), value_set=held)).to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    under = _under_texts({column: named.column(column) for column in table.key}, len(lacking), delta.prefixes, instants)
    return int((lacking & ~under).sum())


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
    async with serve_owner(nats, key, recreator.answer, family=DELTA_FILES_FORWARD_FAMILY):
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
        return await self._asked(_Request(table=table, file=file, epoch=epoch))

    async def full(self, table: str) -> str:
        """ask once for ``table``'s full, which the volume's current full lacks (module docstring, "No full at all").

        :param table: the table
        :ptype table: str
        :return: ``present``, ``recovering`` (``recovering: <the rebuild's step>`` when the writer says it, as
            ``rebuilding from L3: n of m tables read``), ``no_writer`` when none answered, ``failed`` when the writer refused or
            the request did not cross
        :rtype: str
        """
        return await self._asked(_Request(table=table, file="", epoch="", full=True))

    async def _asked(self, request: _Request) -> str:
        """send one request and read the writer's answer.

        :param request: the request
        :ptype request: _Request
        :return: the writer's :class:`_Outcome`, ``no_writer`` or ``failed``
        :rtype: str
        """
        table, file, epoch = request.table, request.file, request.epoch
        payload = request.encode()
        try:
            reply = await forward(
                self._nats, self._key, payload, timeout=self._timeout, family=DELTA_FILES_FORWARD_FAMILY
            )
        except NoOwnerError:
            return _Outcome.NO_WRITER.value  # NOSILENT: the copy counts and logs the answer, and asks again
        except (ForwardedHandlerError, RequestError) as exc:
            log.warning(
                "the writer could not recreate a delta file a copy asked for; it asks again",
                extra={"extra_data": {"table": table, "file": file, "epoch": epoch, "error": str(exc)}},
            )
            return _Outcome.FAILED.value
        try:
            word, _, progress = reply.decode().partition(_PROGRESS_SEPARATOR)
            outcome = _Outcome(word).value
            return outcome if not progress else f"{outcome}: {progress}"
        except (UnicodeDecodeError, ValueError) as exc:
            log.warning(
                "the writer answered a recreate request with no outcome this copy knows; it asks again",
                extra={"extra_data": {"table": table, "file": file, "reply": repr(reply[:64]), "error": str(exc)}},
            )
            return _Outcome.FAILED.value
