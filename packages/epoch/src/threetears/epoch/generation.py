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
metadata: minted when the log is declared, and kept when the refiller restores it at its old
sequences, so a pod's mark survives the broker losing the log. A table's generation is the sequence of
its newest record (``:0`` for a table with none): it moves with every write to the table, which is all
an absence's stamp needs. A table's sequences are not consecutive -- other tables' records sit between
them -- so a follower catches up by walking the table's records after its mark
(:meth:`EpochGenerationReader.changes_after`), never by counting.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.core.collections.epoch_change import (
    EPOCH_CHANGE_MAX_AGE,
    ChangedRow,
    EpochChange,
    EpochWalk,
    check_change_form,
    epoch_change_body,
    read_epoch_change_body,
)
from threetears.core.collections.generation import split_generation_token
from threetears.core.exceptions import GenerationUnavailableError, L2LostError
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
#: restart empties it, and the refiller puts it back from L3 at its old sequences; until then a follower
#: whose epoch it cannot read is behind.
EPOCH_CHANGES_BUCKET: Final = "epoch-changes"

#: the fraction of its byte bound the change log may hold before it is warned on. Past its bound it refuses
#: every append, failing the write that made it, so the warning comes while a quarter of the bound is left
#: to raise it or find what fills it. Alert on ``threetears_epoch_changes_fill_ratio`` above this.
EPOCH_CHANGES_FILL_WARN_RATIO: Final = 0.75
_FILL_WARN_RATIO: Final = EPOCH_CHANGES_FILL_WARN_RATIO

#: how often a watch on a log not yet declared looks again. A follower starts before the first write
#: declares the log, and an absent log is a table with no epoch, not a failure: the watch waits for it.
#: Nothing is missed meanwhile -- a key watch pushes the key's newest record first -- but an epoch whose
#: broadcasts were lost is caught only once the watch binds, so this bounds how late that is: one
#: second, a follower's first restart delay, never backed off.
_ABSENT_LOG_POLL_SECONDS: Final = 1.0

#: the key restore fills a sequence the L3 log holds nothing for, so every restored record keeps its number
_RESTORE_FILLER_KEY: Final = "_restored-gap"


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
    ) -> ChangeLogState: ...

    async def state(self) -> ChangeLogState | None: ...

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
        incarnation, its oldest record is newer than the mark's next sequence (retention, or a restore
        that did not reach back that far), or the walk did not reach the generation read.

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
        return EpochWalk(changes=tuple(changes), complete=complete, first_missing=first_missing)

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """the table's newest epoch, then each later one, until the caller stops.

        Pushed by the broker through one named consumer on the table's key in the log, which the
        bucket grant a pod holds admits. A table with no epoch yields nothing until it has one.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation tokens, in order
        :rtype: AsyncGenerator[str | None]
        :raises GenerationUnavailableError: when the log cannot be read or watched
        :raises KvError: when the connection closes, so nothing can ever be delivered again
        """
        state = await self._state()
        if state is None:
            log.info(
                "the epoch change log is not declared yet; the table is watched once its first write declares it",
                extra={"extra_data": {"table": table_name, "poll_seconds": _ABSENT_LOG_POLL_SECONDS}},
            )
        while state is None:
            await asyncio.sleep(_ABSENT_LOG_POLL_SECONDS)
            if self.closed:
                # the close the docstring promises as KvError: a read through a closed client would
                # surface as GenerationUnavailableError, which a caller takes for a log it may retry
                raise KvError(f"the connection closed while {table_name!r} waited for the change log")
            state = await self._state()
        async with aclosing(self._log().watch(generation_kv_key(table_name))) as updates:
            async for update in updates:
                # the incarnation is read with each push: a log lost and declared again numbers from
                # one under a new one, and a push read against the old would look already applied
                now = await self._state()
                yield None if now is None else _token(now.incarnation, update.revision)


class EpochGenerationSource(_ChangeLogReading):
    """advances collection write generations by appending each epoch's change to the change log."""

    def __init__(
        self, nats_client: Any, *, change_retention: EpochChangeRetention = _DEFAULT_EPOCH_CHANGE_RETENTION
    ) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to declare and write the change log
        :ptype nats_client: Any
        :param change_retention: how the log is kept, declared on its first use
        :ptype change_retention: EpochChangeRetention
        :return: nothing
        :rtype: None
        """
        super().__init__(nats_client)
        self._change_retention = change_retention
        self._declared: ChangeLogState | None = None

    async def _declare(self) -> ChangeLogState:
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
                "the epoch change log was lost and declared again under a new incarnation; a follower that missed "
                "an epoch before it is behind until it restarts or the log is restored",
                extra={"extra_data": {"was": self._declared.incarnation, "now": state.incarnation}},
            )
        self._declared = state
        return state

    async def current(self, table_name: str) -> str:
        """the table's current generation, declaring the log when there is none.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token, ``incarnation:sequence``
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be read or the log declared
        """
        token = await self.read(table_name)
        if token is not None:
            return token
        state = await self._declare()
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
        """append one epoch: the rows a committed write changed, the whole table, or keys unknown.

        The sequence the log gives the append is the epoch. An append never contends with another, so
        concurrent writers to one table each get theirs. A log the broker lost is declared again.

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
        body = epoch_change_body(table_name, rows, keys_unknown=keys_unknown)
        state = self._declared or await self._declare()
        try:
            seq = await self._append(table_name, body)
        except KvBucketNotFoundError:
            state = await self._declare()
            try:
                seq = await self._append(table_name, body)
            except KvBucketNotFoundError as exc:
                raise GenerationUnavailableError(f"the epoch change log vanished as {table_name!r} advanced") from exc
        return _token(state.incarnation, seq)

    async def advance_continuing(self, table_name: str, rows: Sequence[ChangedRow]) -> str:
        """append one epoch that IS a write's commit, never declaring a log: an L2-first collection's commit.

        A log the broker lost is not started again here: an epoch started in a new log is one no pod
        could follow from its mark, and the refiller puts the old one back at its old sequences.

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

    async def restore_epochs(self, changes: Sequence[EpochChange]) -> bool:
        """put a lost log back from the L3 epoch log, every record at its old sequence.

        Only where the broker holds no log: the log is created under the records' incarnation, starting
        at the oldest record's sequence, and every sequence from there to the newest is appended in
        order -- a record where L3 holds one, a filler where it holds none -- so each record keeps the
        number every follower's mark was taken against.

        :param changes: every epoch record to put back, of any tables, under one incarnation
        :ptype changes: Sequence[EpochChange]
        :return: ``True`` when this call created the log and put every record back; ``False`` when a
            log was already there, or another writer's append took a sequence a record needed (the
            records after it are then missing to their followers, who are behind)
        :rtype: bool
        :raises GenerationUnavailableError: when the log cannot be created or written
        :raises ValueError: when the records are of more than one incarnation, or are no followable tokens
        """
        numbered: dict[int, EpochChange] = {}
        incarnations: set[str] = set()
        for change in changes:
            parts = split_generation_token(change.token)
            if parts is None:
                raise ValueError(f"epoch {change.token!r} of {change.table!r} is no followable token")
            incarnations.add(parts[0])
            numbered[parts[1]] = change
        if len(incarnations) != 1:
            raise ValueError(f"a restore puts back one incarnation's epochs, not {sorted(incarnations)}")
        if await self._state() is not None:
            return False
        (incarnation,) = incarnations
        first, last = min(numbered), max(numbered)
        retention = self._change_retention
        try:
            await self._log().declare(
                max_age=retention.max_age,
                max_bytes=retention.max_bytes,
                replicas=retention.replicas,
                incarnation=incarnation,
                first_seq=first,
            )
            for seq in range(first, last + 1):
                held = numbered.get(seq)
                if held is None:
                    key, body = _RESTORE_FILLER_KEY, b"{}"
                else:
                    key = generation_kv_key(held.table)
                    body = epoch_change_body(
                        held.table,
                        None if held.whole_table or held.keys_unknown else held.rows,
                        keys_unknown=held.keys_unknown,
                    )
                took = await self._log().append(key, body, msg_id=f"restore-{incarnation}-{seq}")
                if took != seq:
                    log.error(
                        "a restore of the epoch change log lost a sequence to another writer; the log keeps what it has",
                        extra={"extra_data": {"wanted": seq, "took": took}},
                    )
                    return False
        except KvError as exc:
            raise GenerationUnavailableError(f"the epoch change log could not be restored: {exc}") from exc
        self._declared = await self._state()
        return True

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
