"""collection write generations over the epoch change log: an epoch is its change, numbered by the stream.

:class:`EpochGenerationSource` implements
:class:`threetears.core.collections.generation.GenerationSource`: a collection that caches absences
stamps every recorded absence with the table's generation, and a collection switched on to carry a
write generation advances it once per commit and stamps the token on that commit's row broadcasts.
It declares and writes the log, so it is for a principal granted both -- the hub, the gateway.

:class:`EpochGenerationReader` is the other half, for a pod that only FOLLOWS: it binds the log,
reads a table's generation, an epoch's change and the epochs since a mark, and watches a table's
newest epoch. It never writes, so a pod cannot fake an advance the fleet acts on.

**The epoch is the stream's sequence.** The log (``{ns}-epoch-changes``, a
:class:`~threetears.nats.kv.NatsChangeLog`) is one memory stream, shaped as a KV bucket so a pod's
ordinary bucket grant reads it, whose every key keeps its history. An advance appends ONE record -- the
commit's changed rows, the whole table, or keys unknown -- under the table's key, and the sequence the
stream gives it is the epoch: no per-table counter, so concurrent writers never contend, and the number
and the change can never disagree. An append carries ``Nats-Msg-Id``, so one retried inside the dedup
window records once.

**A token is ``"{incarnation}:{sequence}"``.** The incarnation is the log's own, carried in its stream
metadata, minted when the log is declared. A log put back after the broker lost it is a new incarnation
that, once its last restored record is appended and not before, names the lost one and the sequence from
which the L3 change notes cover it (``continues``, ``covered_from``;
:mod:`threetears.core.collections.change_notes`): a pod whose mark is at or past it walks every record and
catches up, and one older is behind. While the restore is under way the log is marked ``restoring`` and
continues nothing, so every pod of the lost log is behind until it is whole. A table's generation is the sequence of
its newest record (``:0`` for a table with none): it moves with every write to the table, which is all
an absence's stamp needs. A table's sequences are not consecutive -- other tables' records sit between
them -- so a follower catches up by walking the table's records after its mark
(:meth:`EpochGenerationReader.changes_after`), never by counting.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.core.collections.change_notes import RESTORE_BUDGET_RATIO, NotedChange
from threetears.core.collections.epoch_change import (
    EPOCH_CHANGE_MAX_AGE,
    ChangedRow,
    EpochChange,
    EpochQuiet,
    EpochWalk,
    changed_row_runs,
    check_change_form,
    epoch_change_body,
    read_epoch_change_body,
    read_logged_epoch_change,
)
from threetears.core.collections.generation import split_generation_token
from threetears.core.exceptions import GenerationUnavailableError, L2LostError, RestoreTooLargeError
from threetears.nats.change_log import ChangeLogEntry, ChangeLogState
from threetears.nats.errors import KvBucketNotFoundError, KvError, KvFullError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge
from uuid_utils import uuid7

from threetears.epoch.client import _key_for

if TYPE_CHECKING:
    from threetears.nats.kv_watch import KvKeyUpdate

__all__ = [
    "EPOCH_CHANGES_BUCKET",
    "EPOCH_CHANGES_FILL_WARN_RATIO",
    "EpochChangeRetention",
    "EpochGenerationReader",
    "EpochGenerationSource",
    "generation_kv_key",
]

log = get_logger(__name__)

#: the change log every table's epochs are appended to. Memory storage, like every L2 bucket: a broker
#: restart empties it, and the change-note keeper puts it back from the L3 change notes under a new
#: incarnation; until then a follower whose epoch it cannot read is behind.
EPOCH_CHANGES_BUCKET: Final = "epoch-changes"

#: the fraction of its byte bound the change log may hold before it is warned on. Past its bound it refuses
#: every append, failing the write that made it, so the warning comes while a quarter of the bound is left
#: to raise it or find what fills it. Alert on ``threetears_epoch_changes_fill_ratio`` above this.
EPOCH_CHANGES_FILL_WARN_RATIO: Final = 0.75
_FILL_WARN_RATIO: Final = EPOCH_CHANGES_FILL_WARN_RATIO

#: the bytes a restored record takes in the log beyond its key and body, as a sizing estimate: subject
#: prefix, the dedup header and the stream's own per-message overhead, rounded up
_RESTORED_RECORD_OVERHEAD: Final = 160

#: how long a write waits for the change-note keeper to put a lost log back, where the keeper owns it
_RESTORE_WAIT: Final = timedelta(seconds=10)
_RESTORE_POLL_SECONDS: Final = 0.1


def _changes_refused() -> Any:
    """the count of change records the log refused, fetched where it is moved, never held at import."""
    return counter(
        "collection_epoch_change_refused_total",
        description="epoch change records the change log refused (full, absent, unreachable) after a committed write",
        label_names=("table",),
    )


def _fill_ratio() -> Any:
    """the change log's fill gauge, fetched where it is set."""
    return gauge(
        "threetears_epoch_changes_fill_ratio",
        description="bytes the epoch change log holds over its max_bytes; it refuses appends at 1",
    )


@dataclass(frozen=True, slots=True)
class EpochChangeRetention:
    """how long, and how much of, the epoch change log L2 keeps.

    A follower reads only the records of epochs it did not hear, which on a live pod are seconds
    old; ``max_age`` bounds how long a partitioned pod can be away and still catch up from L2. The
    log refuses an append past ``max_bytes`` rather than evicting a record a pod may still need, so
    a full log is a loud write failure, never a silent loss.

    :ivar max_age: how long a record is kept
    :ivar max_bytes: the most the log holds, per replica
    :ivar replicas: how many servers hold it; ``None`` creates on one (a local single-node broker). A
        platform declares three: durable state is never in a one-replica bucket, and the hub's source
        passes them (its start and reconnect sweep raises any stream it finds below three as well).
    """

    max_age: timedelta = EPOCH_CHANGE_MAX_AGE
    max_bytes: int = 64 * 1024 * 1024
    replicas: int | None = None


#: what a source keeps unless told otherwise.
_DEFAULT_EPOCH_CHANGE_RETENTION: Final = EpochChangeRetention()


def generation_kv_key(table_name: str) -> str:
    """the key under which ``table_name``'s epochs are appended: the table's subject in the epoch family.

    The one derivation the source, the reader and a grant for either use.

    :param table_name: the collection's table
    :ptype table_name: str
    :return: the key
    :rtype: str
    """
    return _key_for(Subjects.collection_generation_epoch(table_name))


@runtime_checkable
class _ChangeLogLike(Protocol):
    """the change log a source and a reader use: :class:`threetears.nats.NatsChangeLog`, or its test double."""

    async def declare(
        self,
        *,
        max_age: timedelta,
        max_bytes: int,
        replicas: int | None,
        incarnation: str | None = None,
        first_seq: int | None = None,
        continues: tuple[str, int | None] | None = None,
        restoring: str | None = None,
    ) -> ChangeLogState: ...

    async def state(self) -> ChangeLogState | None: ...

    async def label_restored(self, *, continues: tuple[str, int | None]) -> ChangeLogState: ...

    async def append(self, key: str, value: bytes, *, msg_id: str) -> int: ...

    async def last(self, key: str) -> ChangeLogEntry | None: ...

    async def at(self, seq: int) -> ChangeLogEntry | None: ...

    async def next_after(self, key: str, seq: int) -> ChangeLogEntry | None: ...

    def watch(self, key: str) -> AsyncGenerator[KvKeyUpdate]: ...


@runtime_checkable
class _ChangeLogging(Protocol):
    """a client that binds change logs."""

    def change_log(self, name: str) -> _ChangeLogLike: ...


def _token(incarnation: str, seq: int) -> str:
    """the generation token of the epoch at ``seq`` under ``incarnation``.

    :param incarnation: the log's incarnation
    :ptype incarnation: str
    :param seq: the sequence
    :ptype seq: int
    :return: ``{incarnation}:{seq}``
    :rtype: str
    """
    return f"{incarnation}:{seq}"


class _ChangeLogReading:
    """what a source and a reader share: reading the change log, never declaring it."""

    def __init__(self, nats_client: Any) -> None:
        """capture the client; no I/O.

        :param nats_client: a connected client that binds change logs
        :ptype nats_client: Any
        :return: nothing
        :rtype: None
        """
        self._nats = nats_client

    def _log(self) -> _ChangeLogLike:
        """the change log, bound through the client.

        :return: the log
        :rtype: _ChangeLogLike
        :raises GenerationUnavailableError: when the client binds no change log
        """
        if not isinstance(self._nats, _ChangeLogging):
            raise GenerationUnavailableError(
                f"{type(self._nats).__name__} binds no change log, so no epoch can be read"
            )
        return self._nats.change_log(EPOCH_CHANGES_BUCKET)

    @property
    def closed(self) -> bool:
        """whether the client's connection is closed, so no watch through it can deliver again.

        :return: ``True`` once the client reports itself closed
        :rtype: bool
        """
        return getattr(self._nats, "is_closed", False) is True

    async def _state(self) -> ChangeLogState | None:
        """the log's state, or ``None`` when the broker holds no log.

        :return: the state
        :rtype: ChangeLogState | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        try:
            return await self._log().state()
        except KvError as exc:
            raise GenerationUnavailableError(f"the epoch change log could not be read: {exc}") from exc

    async def log_state(self) -> ChangeLogState | None:
        """what the change log holds, or ``None`` when the broker holds none.

        :return: the state
        :rtype: ChangeLogState | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        return await self._state()

    async def newest_epochs(self, tables: Sequence[str]) -> tuple[ChangeLogState | None, dict[str, int]]:
        """the log's state, then the sequence of each table's newest epoch (``0`` for none), in one pass.

        The state is read first, so an epoch appended after it is seen in the table's newest, never
        hidden behind the state's head: what the follower's quiet pass needs to move a mark safely.

        :param tables: the tables
        :ptype tables: Sequence[str]
        :return: the state, ``None`` when the broker holds no log (and then no tables), and each table's newest
        :rtype: tuple[ChangeLogState | None, dict[str, int]]
        :raises GenerationUnavailableError: when the log cannot be read
        """
        state = await self._state()
        if state is None or not tables:
            return state, {}
        try:
            newest = await asyncio.gather(*(self._log().last(generation_kv_key(table)) for table in tables))
        except KvBucketNotFoundError:
            # NOSILENT: the log went between the reads: as absent as before the first
            return None, {}
        except KvError as exc:
            raise GenerationUnavailableError(f"the newest epochs could not be read: {exc}") from exc
        return state, {
            table: entry.seq if entry is not None else 0 for table, entry in zip(tables, newest, strict=True)
        }

    async def read(self, table_name: str) -> str | None:
        """the table's current generation: the sequence of its newest epoch, or ``:0`` when it has none.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the token ``incarnation:sequence``, or ``None`` when the broker holds no log (lost,
            or never declared)
        :rtype: str | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        state = await self._state()
        if state is None:
            return None
        try:
            newest = await self._log().last(generation_kv_key(table_name))
        except KvBucketNotFoundError:
            # NOSILENT: the log went between the two reads: as absent as before the first
            return None
        except KvError as exc:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be read: {exc}") from exc
        return _token(state.incarnation, newest.seq if newest is not None else 0)

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the change of the epoch ``token``, or ``None`` when the log does not hold it.

        :param table_name: the table
        :ptype table_name: str
        :param token: the epoch
        :ptype token: str
        :return: the record; ``None`` when the log is absent, of another incarnation, or no longer
            keeps it -- the follower is then behind
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the log cannot be read, or holds another table's
            record at that sequence
        """
        parts = split_generation_token(token)
        state = await self._state()
        if parts is None or state is None or parts[0] != state.incarnation:
            return None
        try:
            entry = await self._log().at(parts[1])
        except KvBucketNotFoundError:
            # NOSILENT: the log went between the two reads
            return None
        except KvError as exc:
            raise GenerationUnavailableError(
                f"the change of {table_name!r} at {token!r} could not be read: {exc}"
            ) from exc
        if entry is None:
            return None
        if entry.key != generation_kv_key(table_name):
            raise GenerationUnavailableError(
                f"the epoch {token!r} is {entry.key!r}'s, not {table_name!r}'s: a token from another table"
            )
        return read_epoch_change_body(entry.value, table_name, token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """every epoch of ``table_name`` after ``after`` and up to ``upto``, as the log holds them, in order.

        Walks the table's records, never counts: a table's sequences have other tables' between them.
        The walk is incomplete when the log may have lost one of them: it is absent or of another
        incarnation, its oldest record is newer than the mark's next sequence (retention), or the walk
        did not reach the generation read.

        :param table_name: the table
        :ptype table_name: str
        :param after: the follower's mark
        :ptype after: str
        :param upto: the generation read
        :ptype upto: str
        :return: the walk
        :rtype: EpochWalk
        :raises GenerationUnavailableError: when the log cannot be read
        """
        low = split_generation_token(after)
        high = split_generation_token(upto)
        state = await self._state()
        if low is None or high is None or state is None or low[0] != state.incarnation or high[0] != low[0]:
            return EpochWalk(changes=(), complete=False, first_missing=upto)
        key = generation_kv_key(table_name)
        complete = state.first_seq <= low[1] + 1
        # the epochs after the mark are gone from the front of the log: retention took them
        reason = "record_missing" if complete else "aged_out"
        changes: list[EpochChange] = []
        seq = low[1]
        try:
            while True:
                entry = await self._log().next_after(key, seq)
                if entry is None or entry.seq > high[1]:
                    break
                seq = entry.seq
                changes.append(read_epoch_change_body(entry.value, table_name, _token(state.incarnation, seq)))
        except KvBucketNotFoundError:
            # NOSILENT: the log went mid-walk; what was read is applied, and the follower is behind
            return EpochWalk(changes=tuple(changes), complete=False, first_missing=_token(state.incarnation, seq + 1))
        except KvError as exc:
            raise GenerationUnavailableError(f"the epochs of {table_name!r} could not be read: {exc}") from exc
        if seq < high[1]:
            # the generation read is past the newest record the walk found: that record is gone
            complete = False
        first_missing = None if complete else _token(state.incarnation, low[1] + 1)
        return EpochWalk(changes=tuple(changes), complete=complete, first_missing=first_missing, reason=reason)

    async def continues(self, incarnation: str) -> tuple[str, int | None] | None:
        """the lost log's incarnation the log ``incarnation`` continues, and the sequence its notes cover it from.

        :param incarnation: the log's incarnation, as a generation read names it
        :ptype incarnation: str
        :return: ``(lost incarnation, covered from)``; the second is ``None`` when the notes vouched for none
            of the lost log. ``None`` when the log was not restored, or is no longer ``incarnation``
        :rtype: tuple[str, int | None] | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        state = await self._state()
        if state is None or state.incarnation != incarnation or state.continues is None:
            return None
        return state.continues, state.covered_from

    async def quiet_after(self, table_name: str, mark: str) -> EpochQuiet:
        """what the log can vouch for of ``table_name`` after ``mark``: how far the mark may move, or that it aged out.

        The log's state is read before the table's newest epoch, so an epoch of the table appended
        between the two is seen, and never skipped.

        :param table_name: the table
        :ptype table_name: str
        :param mark: the follower's mark
        :ptype mark: str
        :return: the log's answer; nothing to vouch for when the mark is of another incarnation
        :rtype: EpochQuiet
        :raises GenerationUnavailableError: when the log cannot be read
        """
        parts = split_generation_token(mark)
        state = await self._state()
        if parts is None or state is None or parts[0] != state.incarnation:
            return EpochQuiet(through=None)
        aged_out = state.first_seq > parts[1] + 1 and state.last_seq > parts[1]
        try:
            newest = await self._log().last(generation_kv_key(table_name))
        except KvBucketNotFoundError:
            # NOSILENT: the log went between the reads; nothing is vouched for, and the next judgement reads again
            return EpochQuiet(through=None)
        except KvError as exc:
            raise GenerationUnavailableError(f"the epochs of {table_name!r} could not be read: {exc}") from exc
        written = newest.seq if newest is not None else 0
        quiet = written <= parts[1] and state.last_seq > parts[1]
        return EpochQuiet(through=_token(state.incarnation, state.last_seq) if quiet else None, aged_out=aged_out)

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """the table's newest epoch, then each later one, until the caller stops.

        Pushed by the broker through one named consumer on the table's key in the log, which the
        bucket grant a pod holds admits. A table with no epoch yields nothing until it has one.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation tokens, in order
        :rtype: AsyncGenerator[str | None]
        :raises GenerationUnavailableError: when the log is absent, or cannot be read or watched
        :raises KvError: when the connection closes, so nothing can ever be delivered again
        """
        state = await self._state()
        if state is None:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be watched: no log")
        async with aclosing(self._log().watch(generation_kv_key(table_name))) as updates:
            async for update in updates:
                # the incarnation is read with each push: a log lost and declared again numbers from
                # one under a new one, and a push read against the old would look already applied
                now = await self._state()
                yield None if now is None else _token(now.incarnation, update.revision)


def _restored_records(changes: Sequence[NotedChange]) -> list[tuple[str, bytes]]:
    """the records a restore of ``changes`` appends: per table its row runs within the record bound, or the whole table.

    :param changes: what the notes say changed, per table
    :ptype changes: Sequence[NotedChange]
    :return: ``(key, body)`` per record
    :rtype: list[tuple[str, bytes]]
    """
    records: list[tuple[str, bytes]] = []
    for change in changes:
        if change.whole_table:
            bodies = [epoch_change_body(change.table, None)]
        elif change.rows:
            bodies = [epoch_change_body(change.table, run) for run in changed_row_runs(change.table, change.rows)]
        else:
            continue
        records.extend((generation_kv_key(change.table), body) for body in bodies)
    return records


def _restored_bytes(records: Sequence[tuple[str, bytes]]) -> int:
    """the bytes ``records`` take in the log, by the restore's estimate.

    :param records: ``(key, body)`` per record
    :ptype records: Sequence[tuple[str, bytes]]
    :return: the estimate
    :rtype: int
    """
    return sum(len(key) + len(body) + _RESTORED_RECORD_OVERHEAD for key, body in records)


async def _each(changes: Sequence[NotedChange] | AsyncIterable[NotedChange]) -> AsyncIterator[NotedChange]:
    """the changes one at a time, whether held or streamed.

    :param changes: the changes
    :ptype changes: Sequence[NotedChange] | AsyncIterable[NotedChange]
    :return: each change
    :rtype: AsyncIterator[NotedChange]
    """
    if isinstance(changes, Sequence):
        for change in changes:
            yield change
        return
    async for change in changes:
        yield change


class EpochGenerationSource(_ChangeLogReading):
    """advances collection write generations by appending each epoch's change to the change log.

    **A lost log has one owner.** In a deployment that runs the change-note keeper
    (:class:`~threetears.core.collections.change_notes.ChangeNoteKeeper`), the keeper puts a lost log back
    from the L3 change notes, so every process's source there is built with ``restored_from_l3=True``: it
    never declares a log. While the log is absent a write waits up to ``restore_wait`` for the keeper, then
    fails (its rows and their notes committed, its epoch not appended, logged and counted as any refused
    append); a read of the generation fails at once, so its caller asks L3. The keeper alone starts the log
    (:meth:`begin_restore` and :meth:`finish_restore`, or :meth:`start_log` when its ring has never seen one). Elsewhere a lost
    log is declared again under a new incarnation, and every follower is behind (``replaced``) until it
    restarts.
    """

    def __init__(
        self,
        nats_client: Any,
        *,
        change_retention: EpochChangeRetention = _DEFAULT_EPOCH_CHANGE_RETENTION,
        restored_from_l3: bool = False,
        restore_wait: timedelta = _RESTORE_WAIT,
    ) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to declare and write the change log
        :ptype nats_client: Any
        :param change_retention: how the log is kept, declared on its first use
        :ptype change_retention: EpochChangeRetention
        :param restored_from_l3: a lost log is put back by the change-note keeper, so this source never
            declares one; set in every process of a deployment that runs the keeper
        :ptype restored_from_l3: bool
        :param restore_wait: how long a write waits for the keeper to put a lost log back
        :ptype restore_wait: timedelta
        :return: nothing
        :rtype: None
        """
        super().__init__(nats_client)
        self._change_retention = change_retention
        self._restored_from_l3 = restored_from_l3
        self._restore_wait = restore_wait
        self._declared: ChangeLogState | None = None

    async def start_log(self) -> str:
        """declare the log where the broker holds none, for the keeper when its ring has never seen one.

        :return: the log's incarnation
        :rtype: str
        :raises GenerationUnavailableError: when it can be neither declared nor read
        """
        return (await self._declare_now()).incarnation

    async def _declare(self) -> ChangeLogState:
        """the log to append to: declared, or -- where the keeper owns a lost log -- waited for.

        :return: the log's state
        :rtype: ChangeLogState
        :raises GenerationUnavailableError: when it can be neither declared nor read, or was not put back
            within the wait
        """
        if not self._restored_from_l3:
            return await self._declare_now()
        deadline = time.monotonic() + self._restore_wait.total_seconds()
        while True:
            state = await self._state()
            if state is not None:
                self._declared = state
                return state
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(_RESTORE_POLL_SECONDS)
        log.error(
            "the epoch change log is lost and was not put back from L3 within the wait; this write's epoch is not "
            "appended. Check that the change-note keeper is running",
            extra={"extra_data": {"waited_seconds": self._restore_wait.total_seconds()}},
        )
        raise GenerationUnavailableError("the epoch change log is lost and waits for its restore from L3")

    async def _declare_now(self) -> ChangeLogState:
        """declare the log; a log the broker lost since is declared again, under a new incarnation.

        :return: the log's state
        :rtype: ChangeLogState
        :raises GenerationUnavailableError: when it can be neither declared nor read
        """
        retention = self._change_retention
        try:
            state = await self._log().declare(
                max_age=retention.max_age, max_bytes=retention.max_bytes, replicas=retention.replicas
            )
        except KvError as exc:
            raise GenerationUnavailableError(f"the epoch change log could not be declared: {exc}") from exc
        if self._declared is not None and self._declared.incarnation != state.incarnation:
            log.warning(
                "the epoch change log was lost and declared again under a new incarnation, not put back from the "
                "change notes; every follower is behind (replaced) until it restarts. A deployment that runs the "
                "change-note keeper builds every source with restored_from_l3=True, so only the keeper declares it",
                extra={"extra_data": {"was": self._declared.incarnation, "now": state.incarnation}},
            )
        self._declared = state
        return state

    async def current(self, table_name: str) -> str:
        """the table's current generation, declaring the log when there is none and this source may.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token, ``incarnation:sequence``
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be read or the log declared, or the
            log is lost and the keeper owns putting it back
        """
        token = await self.read(table_name)
        if token is not None:
            return token
        if self._restored_from_l3:
            raise GenerationUnavailableError("the epoch change log is lost and waits for its restore from L3")
        state = await self._declare_now()
        return _token(state.incarnation, 0)

    async def advance(self, table_name: str) -> str:
        """move the table's generation on for a write whose rows nobody named: its keys are unknown.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token this advance wrote
        :rtype: str
        :raises GenerationUnavailableError: when the epoch could not be appended
        """
        return await self.advance_with_change(table_name, None, keys_unknown=True)

    async def advance_with_change(
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str:
        """append the epochs of one committed write: the rows it changed, the whole table, or keys unknown.

        The sequence the log gives the append is the epoch. An append never contends with another, so
        concurrent writers to one table each get theirs. Rows too many for one record are appended as
        one record per run (:func:`~threetears.core.collections.epoch_change.changed_row_runs`), in
        order, and the write's epoch is the last: a follower that missed it applies each run by key.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the committed write changed; ``None`` when it changed the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :param keys_unknown: the writer named no rows and did not change the whole table (a broker
            advancing for a commit that sent no ``changed_rows``); requires ``rows=None``
        :ptype keys_unknown: bool
        :return: the generation token the advance wrote
        :rtype: str
        :raises GenerationUnavailableError: when the epoch could not be appended
        :raises ValueError: when ``keys_unknown`` is set with rows, before anything is appended
        """
        check_change_form(rows, keys_unknown=keys_unknown)
        if rows is None or not rows:
            bodies = [epoch_change_body(table_name, rows, keys_unknown=keys_unknown)]
        else:
            bodies = [epoch_change_body(table_name, run) for run in changed_row_runs(table_name, rows)]
        state = self._declared or await self._declare()
        seq = 0
        for body in bodies:
            try:
                seq = await self._append(table_name, body)
            except KvBucketNotFoundError:
                state = await self._declare()
                try:
                    seq = await self._append(table_name, body)
                except KvBucketNotFoundError as exc:
                    raise GenerationUnavailableError(
                        f"the epoch change log vanished as {table_name!r} advanced"
                    ) from exc
        # the log this source last declared may have been put back since by another process (the keeper's
        # restore, under a new incarnation): the append landed in whichever log is there now, so its token
        # names that log's incarnation, read after the append
        after = await self._state()
        if after is None:
            raise GenerationUnavailableError(f"the epoch change log vanished as {table_name!r} advanced")
        if after.incarnation != state.incarnation:
            log.info(
                "the epoch change log is under a new incarnation since this source last read it",
                extra={"extra_data": {"was": state.incarnation, "now": after.incarnation}},
            )
        self._declared = after
        return _token(after.incarnation, seq)

    async def advance_continuing(self, table_name: str, rows: Sequence[ChangedRow]) -> str:
        """append one epoch that IS a write's commit, never declaring a log: an L2-first collection's commit.

        A log the broker lost is not started again here: an epoch started in a new log is one no pod
        could follow from its mark, and the change-note keeper puts the lost one back.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the write changed, with their values
        :ptype rows: Sequence[ChangedRow]
        :return: the generation token the append wrote
        :rtype: str
        :raises L2LostError: when the log is absent
        :raises GenerationUnavailableError: when the append failed otherwise
        """
        state = await self._state()
        if state is None:
            raise L2LostError(f"the epoch change log is absent; {table_name!r}'s write does not commit")
        try:
            seq = await self._append(table_name, epoch_change_body(table_name, list(rows)))
        except KvBucketNotFoundError as exc:
            raise L2LostError(f"the epoch change log went; {table_name!r}'s write does not commit") from exc
        return _token(state.incarnation, seq)

    async def _append(self, table_name: str, body: bytes) -> int:
        """append one epoch's body under the table's key, and say its sequence.

        :param table_name: the table
        :ptype table_name: str
        :param body: the stored change
        :ptype body: bytes
        :return: the sequence
        :rtype: int
        :raises KvBucketNotFoundError: when the log is absent
        :raises GenerationUnavailableError: when the log refused it or could not be reached
        """
        try:
            return await self._log().append(generation_kv_key(table_name), body, msg_id=str(uuid7()))
        except KvBucketNotFoundError:
            raise
        except KvError as exc:
            _changes_refused().labels(table=table_name).inc()
            full = isinstance(exc, KvFullError)
            log.error(
                "an epoch was not appended after a committed write; a follower that missed its broadcasts cannot "
                "apply it" + (": the change log is FULL" if full else ""),
                extra={"extra_data": {"table": table_name, "error": str(exc), "full": full}},
            )
            raise GenerationUnavailableError(f"the epoch of {table_name!r} could not be appended: {exc}") from exc

    async def begin_restore(self, lost: str) -> str | None:
        """declare the log a restore of ``lost`` fills, before the notes are read, where the broker holds none.

        Declared at once so a write waiting for the log (``restored_from_l3``, up to ``restore_wait``) appends
        to it rather than timing out while the notes are read. The log is marked as being restored from
        ``lost`` and continues nothing yet: a follower of ``lost`` stays behind (``replaced``) until
        :meth:`finish_restore` labels it, and then walks every record from the log's start, the writes that
        landed meanwhile included.

        :param lost: the lost log's incarnation
        :ptype lost: str
        :return: the new log's incarnation; ``None`` when a log was already there
        :rtype: str | None
        :raises GenerationUnavailableError: when the log cannot be created or read
        """
        if await self._state() is not None:
            return None
        retention = self._change_retention
        restored = str(uuid7())
        try:
            state = await self._log().declare(
                max_age=retention.max_age,
                max_bytes=retention.max_bytes,
                replicas=retention.replicas,
                incarnation=restored,
                restoring=lost,
            )
        except KvError as exc:
            raise GenerationUnavailableError(
                f"the epoch change log could not be declared for its restore: {exc}"
            ) from exc
        if state.incarnation != restored:
            return None
        self._declared = state
        return restored

    async def finish_restore(
        self,
        changes: Sequence[NotedChange] | AsyncIterable[NotedChange],
        *,
        incarnation: str,
        lost: str,
        covered_from: int | None,
    ) -> int:
        """append one record per change into the log :meth:`begin_restore` declared, then label it restored.

        Each change is appended as the record a write would append: its distinct rows, split into runs
        within the record bound, or the whole table. ``changes`` may be streamed (the keeper reads the notes
        a page at a time), so memory stays bounded by one change. Only after the last append is the log
        labelled as continuing ``lost`` from ``covered_from``, so a follower never carries its mark over to a
        partial restore. Safe to run again after an interrupted run: every append carries a fresh id (the
        broker would drop a repeated one inside its dedup window), and a record appended twice is evicted
        twice.

        The restore may fill at most :data:`RESTORE_BUDGET_RATIO` of ``max_bytes``, what the log already holds
        included. A sequence is sized first and nothing is appended when it would pass; a stream is counted as
        it goes, and the restore stops, unlabelled, before the append that would pass it (the keeper's next
        pass sizes again, from the log as it then is).

        :param changes: what the notes say changed, per table (:mod:`threetears.core.collections.change_notes`)
        :ptype changes: Sequence[NotedChange] | AsyncIterable[NotedChange]
        :param incarnation: the log :meth:`begin_restore` declared
        :ptype incarnation: str
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param covered_from: the lost log's sequence from which the notes cover every epoch; ``None`` when
            they vouch for none of it, so every follower of the lost log is behind
        :ptype covered_from: int | None
        :return: the records appended
        :rtype: int
        :raises RestoreTooLargeError: when the restore would pass its budget; the log is left unlabelled
        :raises GenerationUnavailableError: when the log is not the one being restored, or cannot be written
        """
        state = await self._state()
        if state is None or state.incarnation != incarnation or state.restoring != lost:
            raise GenerationUnavailableError(
                f"the epoch change log is not the restore of {lost!r} under {incarnation!r}: {state}"
            )
        budget = None if not state.max_bytes else state.max_bytes * RESTORE_BUDGET_RATIO - state.bytes
        if isinstance(changes, Sequence):
            needed = _restored_bytes(_restored_records(changes))
            if budget is not None and needed > budget:
                raise RestoreTooLargeError(
                    f"the restore of {lost!r} needs {needed} bytes and the log holds {state.bytes} of "
                    f"{state.max_bytes}; it may fill at most {RESTORE_BUDGET_RATIO:.0%}"
                )
        attempt = uuid7().hex
        appended = 0
        used = 0
        try:
            async for change in _each(changes):
                for key, body in _restored_records([change]):
                    used += _restored_bytes([(key, body)])
                    if budget is not None and used > budget:
                        raise RestoreTooLargeError(
                            f"the restore of {lost!r} passed its budget of {int(budget)} bytes after {appended} "
                            "records; left unlabelled"
                        )
                    await self._log().append(key, body, msg_id=f"restore-{incarnation}-{attempt}-{appended}")
                    appended += 1
            labelled = await self._log().label_restored(continues=(lost, covered_from))
        except KvError as exc:
            raise GenerationUnavailableError(f"the epoch change log could not be restored: {exc}") from exc
        self._declared = labelled
        return appended

    async def epoch_at(self, seq: int) -> EpochChange | None:
        """the record the log holds at ``seq`` under its current incarnation, of whichever table.

        :param seq: the sequence
        :ptype seq: int
        :return: the record; ``None`` when the log is absent or no longer holds it
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the log cannot be read, or the record is malformed
        """
        state = await self._state()
        if state is None:
            return None
        try:
            entry = await self._log().at(seq)
        except KvBucketNotFoundError:
            # NOSILENT: the log went between the two reads; the scrub reads it again next pass
            return None
        except KvError as exc:
            raise GenerationUnavailableError(f"the epoch at {seq} could not be read: {exc}") from exc
        if entry is None:
            return None
        return read_logged_epoch_change(entry.value, _token(state.incarnation, seq))

    async def observe_change_log_fill(self) -> float | None:
        """measure how full the change log is, set ``threetears_epoch_changes_fill_ratio``, and warn near full.

        :return: bytes held over ``max_bytes``; ``None`` when the log is absent, unbounded or unreadable
        :rtype: float | None
        """
        try:
            state = await self._state()
        except GenerationUnavailableError as exc:
            # NOSILENT: an unreadable log has no fill to report; the next observation reads again
            log.warning("the epoch change log's fill could not be read", extra={"extra_data": {"error": str(exc)}})
            return None
        if state is None or not state.max_bytes:
            return None
        ratio = state.bytes / state.max_bytes
        _fill_ratio().set(ratio)
        if ratio >= _FILL_WARN_RATIO:
            log.warning(
                "the epoch change log is nearly full; at its bound it refuses every epoch after a committed write",
                extra={"extra_data": {"bytes": state.bytes, "max_bytes": state.max_bytes, "ratio": round(ratio, 3)}},
            )
        return ratio


class EpochGenerationReader(_ChangeLogReading):
    """reads and watches collection write generations in the change log, and never writes them.

    What a pod that follows a table holds: it binds the log the hub declared rather than creating it,
    and reports a table with no log as having no generation -- a follower has no write to declare one
    with, and "no generation yet" is itself something its mark records.
    """

    def __init__(self, nats_client: Any, *, create_if_missing: bool = False) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to bind the change log
        :ptype nats_client: Any
        :param create_if_missing: kept for the callers that pass it; a reader never creates the log
        :ptype create_if_missing: bool
        :return: nothing
        :rtype: None
        """
        super().__init__(nats_client)
        self._create_if_missing = create_if_missing
