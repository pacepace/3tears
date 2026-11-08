"""collection write generations over the epoch bucket.

:class:`EpochGenerationSource` implements
:class:`threetears.core.collections.generation.GenerationSource`: a collection that caches absences
stamps every recorded absence with the table's generation, and a collection switched on to carry a
write generation advances it once per commit and stamps the token on that commit's row broadcasts.
It reads and writes the bucket, so it is for a principal granted both -- the hub, the gateway.

:class:`EpochGenerationReader` is the other half, for a pod that only FOLLOWS: it binds the bucket,
reads a table's generation or watches its key, and never writes. A pod holds a read on the bucket
and no write, so that it cannot fake an advance the fleet acts on.

**One value per table, holding both halves of the answer: ``"{incarnation}:{count}"``.** The count
moves on with every committed write. The incarnation is minted when the value is first created, so
a broker restart that empties the memory-backed bucket forces a new one on the next read. Keeping
both in one value is what makes a single read trustworthy: reading an incarnation and a counter as
two keys can straddle a wipe and yield an old incarnation beside a reset count, or a new
incarnation beside an old count -- and the second can later equal a genuine token, revalidating an
absence that writes had already superseded.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.core.collections.epoch_change import (
    EPOCH_CHANGE_MAX_AGE,
    ChangedRow,
    EpochChange,
    check_change_form,
    epoch_change_record,
    read_epoch_change,
)
from threetears.core.collections.generation import split_generation_token
from threetears.core.collections.l2_first import L2LostError
from threetears.core.exceptions import GenerationUnavailableError
from threetears.nats.errors import KvBucketNotFoundError, KvError
from threetears.nats.kv import KvDeclaring
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge
from uuid_utils import uuid7

from threetears.epoch.client import _EPOCH_BUCKET, _key_for

if TYPE_CHECKING:
    from threetears.nats.kv import KvBucketLike, KvCapable

__all__ = [
    "EPOCH_CHANGES_BUCKET",
    "EPOCH_CHANGES_FILL_WARN_RATIO",
    "EpochChangeRetention",
    "EpochGenerationReader",
    "EpochGenerationSource",
    "generation_kv_key",
]

log = get_logger(__name__)

#: matched to the coordination counters' budget: contention on one table's generation is a burst of
#: concurrent writes to that table, which is exactly when a small budget would give up.
_MAX_CAS_ATTEMPTS: Final = 30

#: full-jitter backoff bound between compare-and-swap retries, seconds.
_CAS_RETRY_BACKOFF_SECONDS: Final = 0.02


#: the bucket holding each epoch's change record, beside the epoch bucket holding each table's head.
#: Memory storage, like every L2 bucket: a broker restart empties it, and a follower whose epoch has
#: no record is behind until it is written again.
EPOCH_CHANGES_BUCKET: Final = "epoch-changes"

_LEGAL_KEY_TOKEN: Final = re.compile(r"^[A-Za-z0-9_-]+$")

#: the fraction of its byte bound the epoch change log may hold before it is warned on. The log refuses
#: a record past its bound, and a refused record fails the write that made it, so the warning has to
#: come while there is still room: at three quarters, an operator has a quarter of the bound's worth
#: of writes to raise it or find what is filling it. Alert on ``threetears_epoch_changes_fill_ratio``
#: above this.
EPOCH_CHANGES_FILL_WARN_RATIO: Final = 0.75

#: the fill gauge and its warning counter, fetched where they are set rather than at import
_CHANGES_FILL_NAME: Final = "threetears_epoch_changes_fill_ratio"
_CHANGES_FILL_HELP: Final = (
    "how full the epoch change log is, as a fraction of its byte bound; it refuses records past 1.0, "
    "failing their writes, so alert above 0.75"
)
_CHANGES_FILL_WARNED_NAME: Final = "threetears_epoch_changes_fill_warnings_total"
_CHANGES_FILL_WARNED_HELP: Final = "measurements of the epoch change log above its warn ratio"


@runtime_checkable
class _Measurable(Protocol):
    """a bucket that can say how many bytes it holds; the real bucket and the shipped in-memory one can."""

    async def held_bytes(self) -> int: ...


def _changes_refused() -> Any:
    """the count of change records the log refused, fetched where it is moved, never held at import."""
    return counter(
        "collection_epoch_change_refused_total",
        description="epoch change records the bucket refused (full, or unreachable) after a committed write",
        label_names=("table",),
    )


@dataclass(frozen=True, slots=True)
class EpochChangeRetention:
    """how long, and how much of, the epoch change log L2 keeps.

    A follower reads only the records of epochs it did not hear, which on a live pod are seconds
    old; ``max_age`` bounds how long a partitioned pod can be away and still catch up from L2. The
    bucket refuses a write past ``max_bytes`` rather than evicting a record a pod may still need, so
    a full log is a loud write failure, never a silent loss.

    :ivar max_age: how long a record is kept
    :ivar max_bytes: the most the bucket holds, per replica
    :ivar replicas: how many servers hold it; ``None`` keeps a live bucket's count and creates on
        one (a local single-node broker). A platform declares three: durable state is never in a
        one-replica bucket, and the hub's source passes them (its start and reconnect sweep raises
        any stream it finds below three as well).
    """

    max_age: timedelta = EPOCH_CHANGE_MAX_AGE
    max_bytes: int = 64 * 1024 * 1024
    replicas: int | None = None


#: what a source keeps unless told otherwise.
_DEFAULT_EPOCH_CHANGE_RETENTION: Final = EpochChangeRetention()


def _epoch_change_kv_key(table_name: str, token: str) -> str:
    """the key in :data:`EPOCH_CHANGES_BUCKET` holding the change record of the advance that wrote ``token``.

    ``{table}.{incarnation}.{count}``. A table name that is not a legal key token is digested.

    :param table_name: the table
    :ptype table_name: str
    :param token: a followable generation token, ``{incarnation}:{count}``
    :ptype token: str
    :return: the KV key
    :rtype: str
    :raises GenerationUnavailableError: when ``token`` is not followable, so no record can be named
    """
    parts = split_generation_token(token)
    if parts is None or not _LEGAL_KEY_TOKEN.match(parts[0]):
        raise GenerationUnavailableError(f"generation {token!r} of {table_name!r} names no epoch change record")
    table = (
        table_name
        if _LEGAL_KEY_TOKEN.match(table_name)
        else "t-" + hashlib.sha256(table_name.encode()).hexdigest()[:32]  # convert at border: KV key token
    )
    return f"{table}.{parts[0]}.{parts[1]}"


def generation_kv_key(table_name: str) -> str:
    """the key in the epoch bucket holding ``table_name``'s write generation.

    The one derivation the source, the reader and a grant for either all use: a principal that
    follows a table is granted a read of exactly this key.

    :param table_name: the collection's table
    :ptype table_name: str
    :return: the KV key
    :rtype: str
    """
    return _key_for(Subjects.collection_generation_epoch(table_name))


@runtime_checkable
class _KeyWatchable(Protocol):
    """a bucket whose single keys can be watched; the real bucket and the shipped in-memory one are."""

    def watch_key(self, *, key: str) -> AsyncGenerator[Any]: ...


def _parse(table_name: str, raw: bytes) -> tuple[str, int]:
    """split a stored generation into its incarnation and count.

    :param table_name: the table, for the error
    :ptype table_name: str
    :param raw: the stored value
    :ptype raw: bytes
    :return: ``(incarnation, count)``
    :rtype: tuple[str, int]
    :raises GenerationUnavailableError: when the value is not ``incarnation:count``
    """
    incarnation, separator, count = raw.decode("utf-8", errors="replace").rpartition(":")
    if not separator or not incarnation or not count.isdigit():
        raise GenerationUnavailableError(f"write generation for {table_name!r} is malformed: {raw!r}")
    return incarnation, int(count)


class EpochGenerationSource:
    """reads and advances collection write generations in the epoch KV bucket."""

    def __init__(
        self, nats_client: KvCapable, *, change_retention: EpochChangeRetention = _DEFAULT_EPOCH_CHANGE_RETENTION
    ) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to open the epoch bucket
        :ptype nats_client: KvCapable
        :param change_retention: how the epoch change log is kept, declared on its first write
        :ptype change_retention: EpochChangeRetention
        :return: nothing
        :rtype: None
        """
        self._nats = nats_client
        self._change_retention = change_retention
        self._changes_bucket: KvBucketLike | None = None

    async def _changes(self) -> KvBucketLike:
        """the epoch change log, declared with its retention once per source by the principal that writes it.

        Every writer of the log holds the platform's one retention (the hub and the gateway build
        their sources from the same declaration), so a second writer's declaration reconciles
        nothing; the handle is kept so a write makes no declaration round trip.

        :return: the bucket
        :rtype: KvBucketLike
        :raises GenerationUnavailableError: when the client cannot declare a bucket
        """
        if self._changes_bucket is not None:
            return self._changes_bucket
        if not isinstance(self._nats, KvDeclaring):
            raise GenerationUnavailableError(
                f"{type(self._nats).__name__} cannot declare the epoch change log, so no epoch's change can be stored"
            )
        retention = self._change_retention
        self._changes_bucket = await self._nats.ensure_kv_bucket(
            name=EPOCH_CHANGES_BUCKET,
            ttl=retention.max_age,
            storage="memory",
            max_bytes=retention.max_bytes,
            replicas=retention.replicas,
            owns_bucket=True,
        )
        return self._changes_bucket

    async def advance_with_change(
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str:
        """advance the table's generation once and store the rows the advance covered as its change record.

        The record is written after the compare-and-swap that numbered it, before the caller sends
        the commit's row broadcasts, so a follower that misses one finds the record when it judges the
        advance. A record the log refuses -- full, or unreachable -- leaves an epoch no follower that
        missed it can apply: it is counted, logged as an error and raised, like a failed advance.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the committed write changed; ``None`` when it changed the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :param keys_unknown: the writer named no rows and did not change the whole table (a broker
            advancing for a commit that sent no ``changed_rows``); requires ``rows=None``
        :ptype keys_unknown: bool
        :return: the generation token the advance wrote
        :rtype: str
        :raises GenerationUnavailableError: when the generation could not be advanced or its change
            could not be stored
        :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
        """
        check_change_form(rows, keys_unknown=keys_unknown)
        token = await self.advance(table_name)
        await self._store_change(table_name, token, rows, keys_unknown=keys_unknown)
        return token

    async def _store_change(
        self, table_name: str, token: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> None:
        """store the change record of the advance that wrote ``token``.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation the advance wrote
        :ptype token: str
        :param rows: the rows it covered; ``None`` for the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :param keys_unknown: the advance's reach is the whole table because nobody listed its rows
        :ptype keys_unknown: bool
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the log refused it
        """
        key = _epoch_change_kv_key(table_name, token)
        try:
            bucket = await self._changes()
            await bucket.put(key=key, value=epoch_change_record(table_name, token, rows, keys_unknown=keys_unknown))
        except KvError as exc:
            _changes_refused().labels(table=table_name).inc()
            log.error(
                "an epoch's change record was not stored after a committed write; a follower that missed its "
                "broadcasts is behind on the table until the record is written",
                extra={"extra_data": {"table": table_name, "epoch": token, "error": str(exc)}},
            )
            raise GenerationUnavailableError(
                f"the change of {table_name!r} at {token!r} could not be stored: {exc}"
            ) from exc

    async def _bucket(self) -> KvBucketLike:
        """the epoch bucket, opened with no bucket-wide expiry so generations never lapse on a timer.

        :return: the bucket
        :rtype: KvBucketLike
        """
        return await self._nats.kv_bucket(name=_EPOCH_BUCKET, ttl=None)

    async def read(self, table_name: str) -> str | None:
        """the table's current write generation, or ``None`` when the bucket holds none; never mints one.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token, or ``None``
        :rtype: str | None
        :raises GenerationUnavailableError: when the bucket cannot be read, or holds no generation
        """
        key = generation_kv_key(table_name)
        try:
            raw = await (await self._bucket()).get(key=key)
        except KvError as exc:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be read: {exc}") from exc
        if raw is None:
            return None
        _parse(table_name, raw)
        return raw.decode("utf-8")

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the change record of the advance that wrote ``token``, or ``None`` when the log does not hold it.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that advance wrote
        :ptype token: str
        :return: the record, or ``None``
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the log cannot be read, or holds another epoch's record
        """
        key = _epoch_change_kv_key(table_name, token)
        try:
            raw = await (await self._changes()).get(key=key)
        except KvError as exc:
            raise GenerationUnavailableError(
                f"the change of {table_name!r} at {token!r} could not be read: {exc}"
            ) from exc
        return None if raw is None else read_epoch_change(raw, table_name, token)

    async def advance_continuing(self, table_name: str, rows: Sequence[ChangedRow]) -> str:
        """advance an existing generation once and store ``rows`` as its change: an L2-first commit.

        :meth:`advance_with_change`, except that a table with no generation is refused rather than given
        a new incarnation: an L2-first table has no generation only when NATS lost it, and an epoch begun
        under a new incarnation is one no pod could follow from its mark, nor the refiller continue.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the write changed
        :ptype rows: Sequence[ChangedRow]
        :return: the generation token the advance wrote
        :rtype: str
        :raises L2LostError: when the table has no generation
        :raises GenerationUnavailableError: when the generation could not be advanced or its change stored
        """
        token = await self._advance(table_name, mint=False)
        await self._store_change(table_name, token, rows)
        return token

    async def restore_epochs(self, table_name: str, head: str | None, changes: Sequence[EpochChange]) -> bool:
        """put a table's epoch history back after NATS lost it, only where the bucket holds no generation.

        Each record is created where the log holds none for its epoch, then the generation itself, last,
        so a writer (which refuses a table with no generation) commits nothing before the history is back.

        :param table_name: the table
        :ptype table_name: str
        :param head: the generation to resume at; ``None`` mints a fresh incarnation at count zero
        :ptype head: str | None
        :param changes: the records to put back, oldest first
        :ptype changes: Sequence[EpochChange]
        :return: whether this call created the generation; ``False`` when one was already there
        :rtype: bool
        :raises GenerationUnavailableError: when the bucket or the log cannot be written
        """
        try:
            log_bucket = await self._changes()
            for change in changes:
                key = _epoch_change_kv_key(change.table, change.token)
                await log_bucket.create(key=key, value=change.model_dump_json().encode("utf-8"))
            value = head if head is not None else f"{uuid7()}:0"
            created = await (await self._bucket()).create(key=generation_kv_key(table_name), value=value.encode())
        except KvError as exc:
            raise GenerationUnavailableError(f"the epochs of {table_name!r} could not be restored: {exc}") from exc
        return created is not None

    async def observe_change_log_fill(self) -> float | None:
        """measure how full the epoch change log is, set the gauge, and warn above the warn ratio.

        :return: the fraction of its byte bound the log holds; ``None`` when the log has no bound, or
            the bucket cannot say what it holds
        :rtype: float | None
        """
        bound = self._change_retention.max_bytes
        try:
            bucket = await self._changes()
            if bound <= 0 or not isinstance(bucket, _Measurable):
                return None
            held = await bucket.held_bytes()
        except (KvError, GenerationUnavailableError) as exc:
            log.warning(
                "the epoch change log could not be measured",
                extra={"extra_data": {"error": f"{type(exc).__name__}: {exc}"}},
            )
            return None
        ratio = held / bound
        gauge(_CHANGES_FILL_NAME, description=_CHANGES_FILL_HELP).set(ratio)
        if ratio >= EPOCH_CHANGES_FILL_WARN_RATIO:
            counter(_CHANGES_FILL_WARNED_NAME, description=_CHANGES_FILL_WARNED_HELP).inc()
            log.warning(
                "the epoch change log is nearly full; past its bound it refuses records and fails their writes. "
                "Raise its max_bytes, or find what writes so many epochs",
                extra={"extra_data": {"held_bytes": held, "max_bytes": bound, "fill_ratio": round(ratio, 3)}},
            )
        return ratio

    async def current(self, table_name: str) -> str:
        """the table's current write generation, minting its first incarnation when there is none.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token, ``incarnation:count``
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be read or established
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
            raw = await bucket.get(key=key)
            if raw is None:
                seeded = f"{uuid7()}:0".encode()  # convert at border: opaque KV value
                if await bucket.create(key=key, value=seeded) is not None:
                    raw = seeded
                else:
                    raw = await bucket.get(key=key)
        except KvError as exc:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be read: {exc}") from exc
        if raw is None:
            # lost the create and read nothing: the bucket was emptied between the two.
            raise GenerationUnavailableError(f"write generation for {table_name!r} vanished while being established")
        _parse(table_name, raw)
        return raw.decode("utf-8")

    async def advance(self, table_name: str) -> str:
        """move the table's write generation on by one, and say what it became.

        The token returned is the value this call's own compare-and-swap wrote, never one a
        concurrent advance wrote after it: a writer stamps it on the row broadcasts of the commit
        it advanced for, and a follower counts those rows against exactly this advance.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token this advance wrote, ``incarnation:count``
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be advanced within the retry
            budget, or the store is unreachable
        """
        return await self._advance(table_name, mint=True)

    async def _advance(self, table_name: str, *, mint: bool) -> str:
        """move the table's write generation on by one, minting a first incarnation only when ``mint``.

        :param table_name: the collection's table
        :ptype table_name: str
        :param mint: whether a table with no generation starts a new incarnation
        :ptype mint: bool
        :return: the generation token this advance wrote
        :rtype: str
        :raises L2LostError: when the table has no generation and ``mint`` is ``False``
        :raises GenerationUnavailableError: when the generation cannot be advanced
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
            for _ in range(_MAX_CAS_ATTEMPTS):
                entry = await bucket.get_entry(key=key)
                if entry is None and not mint:
                    raise L2LostError(
                        f"write generation for {table_name!r} is not in L2 (lost with NATS, or not yet put back "
                        f"by the refiller); an L2-first write cannot commit until it is"
                    )
                if entry is None:
                    # no generation yet (never read, or emptied): a new incarnation already
                    # invalidates every absence recorded under an old one, so starting at 1 is enough.
                    written = f"{uuid7()}:1"
                    if await bucket.create(key=key, value=written.encode()) is not None:
                        return written
                else:
                    raw, revision = entry
                    incarnation, count = _parse(table_name, raw)
                    written = f"{incarnation}:{count + 1}"
                    if await bucket.update(key=key, value=written.encode(), revision=revision) is not None:
                        return written
                await asyncio.sleep(random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS))  # noqa: S311 - jitter, not security
        except KvError as exc:
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be advanced: {exc}"
            ) from exc
        raise GenerationUnavailableError(
            f"write generation for {table_name!r} lost {_MAX_CAS_ATTEMPTS} compare-and-swap rounds to concurrent writes"
        )


class EpochGenerationReader:
    """reads and watches collection write generations in the epoch bucket, and never writes them.

    What a pod that follows a table holds. It binds the bucket the hub declared rather than
    creating it, and where :meth:`EpochGenerationSource.current` mints a generation for a table
    that has none, this reports that there is none: a follower has no write to mint one with, and
    "no generation yet" is itself something its mark records.
    """

    def __init__(self, nats_client: KvCapable, *, create_if_missing: bool = False) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to open the epoch bucket
        :ptype nats_client: KvCapable
        :param create_if_missing: whether opening the bucket may create it. ``False``, the default,
            binds only, which is all a pod's grant allows; ``True`` is for a principal that also
            declares the bucket and follows through the same reader
        :ptype create_if_missing: bool
        :return: nothing
        :rtype: None
        """
        self._nats = nats_client
        self._create_if_missing = create_if_missing

    @property
    def closed(self) -> bool:
        """whether the client's connection is closed, so no watch through it can deliver again.

        :return: ``True`` once the client reports itself closed
        :rtype: bool
        """
        return getattr(self._nats, "is_closed", False) is True

    async def _bucket(self) -> KvBucketLike:
        """the epoch bucket, opened as :class:`EpochGenerationSource` opens it but bound, not created.

        :return: the bucket
        :rtype: KvBucketLike
        """
        return await self._nats.kv_bucket(name=_EPOCH_BUCKET, ttl=None, create_if_missing=self._create_if_missing)

    async def read(self, table_name: str) -> str | None:
        """the table's current write generation, or ``None`` when the bucket holds none for it.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token ``incarnation:count``, or ``None``
        :rtype: str | None
        :raises GenerationUnavailableError: when the bucket cannot be read, or holds a value that
            is not a generation

        A read is a get of the key, which the whole-bucket read an agent pod holds admits. A
        principal granted single keys only (``JsCapability.KV_KEY_READ``) reaches them through
        :meth:`watch` instead: that grant's get is the direct one, which the epoch bucket is not
        declared for.
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
            raw = await bucket.get(key=key)
        except KvError as exc:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be read: {exc}") from exc
        if raw is None:
            return None
        _parse(table_name, raw)
        return raw.decode("utf-8")

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the change record of the advance that wrote ``token``, or ``None`` when the log does not hold it.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that advance wrote
        :ptype token: str
        :return: the record; ``None`` when the log was never written for it, lost it, or no longer
            keeps it -- the follower is then behind on the table
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the log cannot be read, or holds something that is
            not this epoch's record
        """
        key = _epoch_change_kv_key(table_name, token)
        try:
            bucket = await self._nats.kv_bucket(
                name=EPOCH_CHANGES_BUCKET, ttl=None, create_if_missing=self._create_if_missing
            )
            raw = await bucket.get(key=key)
        except KvBucketNotFoundError:
            # NOSILENT: no log was ever declared, or the broker lost it: no record, so the follower is behind
            return None
        except KvError as exc:
            raise GenerationUnavailableError(
                f"the change of {table_name!r} at {token!r} could not be read: {exc}"
            ) from exc
        if raw is None:
            return None
        return read_epoch_change(raw, table_name, token)

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """the table's write generation as it stands, then each value it takes, until the caller stops.

        Pushed by the broker through one named consumer on the table's key
        (:meth:`threetears.nats.NatsKvBucket.watch_key`), which a grant narrowed to that one key
        admits. The first value yielded is the key's latest, so a watcher that starts, or restarts
        after the broker did, needs no separate read; a key that has never been written yields
        nothing until it is. A value that is not a generation is yielded as it is: it compares
        unequal to every mark, which leaves a follower behind on the table.

        Close it by stopping iteration.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation tokens, in order; ``None`` when the key was deleted
        :rtype: AsyncGenerator[str | None]
        :raises GenerationUnavailableError: when the bucket cannot be opened or cannot watch a key
        :raises KvError: when the connection closes, so nothing can ever be delivered again
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
        except KvError as exc:
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be watched: {exc}"
            ) from exc
        if not isinstance(bucket, _KeyWatchable):
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be watched: {type(bucket).__name__} has no watch_key"
            )
        # sender-posture: own-consumer -- a generation counts only through the KV key watch, which takes an update only with its own consumer's ack reply
        async with aclosing(bucket.watch_key(key=key)) as updates:
            async for update in updates:
                value: bytes | None = update.value
                yield None if value is None else value.decode("utf-8", errors="replace")
