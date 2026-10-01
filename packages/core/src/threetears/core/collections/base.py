"""base collection with three-tier caching.

composite primary keys are first-class. a subclass declares
``primary_key_column = "memory_id"`` for single-pk tables (the default
shape) or ``primary_key_column = ("conversation_id", "item_id")`` for
composite-pk tables. internally, every cache-keying path normalizes
the declared pk and caller-supplied id into a tuple via
:meth:`BaseCollection.normalize_pk`; the L1 (SQLite / DuckDB), L2
(NATS KV), and L3 (pluggable durable backend — SQL, git, …) tiers all
accept the tuple uniformly.
the invalidation wire envelope carries ``ids`` (plural, always an
array) matching the pk column order, so single-pk emits a length-1
array and composite-pk emits a length-N array.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, Literal, TypeVar

from sqlalchemy import Column, Float, MetaData, String, Table, Text

from threetears.core._bridge import fire_and_forget, sync_await
from threetears.core.backends.protocol import L3Backend
from threetears.core.cache import MISSING
from threetears.core.cache.base import _CACHED_AT_COLUMN
from threetears.core.collections.bypassing_write import BypassingWrite
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.flush import FlushStrategy, WriteBuffer
from threetears.core.collections.l2_order import (
    L2_ORDER_COLUMNS,
    L2Order,
    l2_order_of,
    with_l2_order,
    without_l2_order,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity, derive_addressing_id
from threetears.core.exceptions import (
    ConcurrentModificationError,
    CorruptCacheEntry,
    GenerationUnavailableError,
    L2EpochRegressedError,
    L2ScopeNotConfiguredError,
)
from threetears.nats.errors import KvError
from threetears.observe import get_logger, traced

if TYPE_CHECKING:
    # annotation-only: importing these eagerly would load the NATS client into
    # every L1-only consumer of this module. the `isinstance` below tests the
    # local `_NatsClientFromRegistry` sentinel, not `NatsClient`.
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = ["NATS_CLIENT_FROM_REGISTRY", "BaseCollection", "CasMutation", "EntityT"]

log = get_logger(__name__)

EntityT = TypeVar("EntityT", bound=BaseEntity)

#: the JetStream KV key grammar enforced by ``nats-server`` (``kv.go``).
#: a key (or, here, a key *body*) that falls outside this character set
#: is rejected with ``nats: JetStream.InvalidKeyError`` at runtime, so a
#: pk value carrying a colon / space / other out-of-grammar character
#: cannot be interpolated raw into a KV key. matched as a whole string.
_KV_KEY_GRAMMAR: Final = re.compile(r"^[-/_=.a-zA-Z0-9]+$")

#: the L2 value recording that a key is absent from every tier, followed by the write generation it
#: was recorded under. It opens with a NUL byte so no JSON-serialised row can equal or start with it.
_ABSENT_MARKER_PREFIX: Final = b"\x00threetears.collections.absent\x00"

#: the framework-owned L1 table holding negative-cache markers for every collection on a backend,
#: beside the collections' own tables -- the same arrangement as the write buffer's table.
_ABSENT_MARKER_TABLE: Final = "collection_absent_markers"
_ABSENT_MARKER_METADATA: Final = MetaData()
Table(
    _ABSENT_MARKER_TABLE,
    _ABSENT_MARKER_METADATA,
    Column("key", String, primary_key=True),
    Column("generation", Text, nullable=False),
    # monotonic seconds: this row is never shared across processes, so the local clock is the
    # only one that ever reads it.
    Column("deadline", Float, nullable=False),
)


#: how often a collection sweeps expired absent-markers from its pod's L1, and how many rows one
#: sweep removes. Bounded both ways so the sweep never becomes a cost a lookup notices; a backlog
#: larger than one batch drains over successive intervals.
_ABSENT_MARKER_SWEEP_INTERVAL_SECONDS: Final = 60.0
_ABSENT_MARKER_SWEEP_BATCH: Final = 500
_GENERATION_WARNING_INTERVAL_SECONDS: Final = 60.0

#: full-jitter bound between compare-and-swap rounds, seconds. Without it the losers of a round
#: retry in lockstep and spend the whole budget on the same instant, which is exactly what a burst
#: against ONE key produces -- a credential-stuffing run against one account, or every replica
#: incrementing one counter. Matches the value the coordination primitives used.
_CAS_RETRY_BACKOFF_SECONDS: Final = 0.02


@dataclass(frozen=True, slots=True)
class CasMutation:
    """what :meth:`BaseCollection.l2_cas_mutate` did.

    :ivar action: ``"created"`` when no prior row existed in any tier, ``"updated"`` when one did,
        ``"deleted"``, or ``"noop"`` when the callback declined to write
    :ivar row: the row as written, or ``None`` for a delete or a noop
    """

    action: Literal["created", "updated", "deleted", "noop"]
    row: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _AbsentMarker:
    """a decoded negative-cache marker.

    :ivar generation: the table's write generation when the absence was recorded
    """

    generation: str


@dataclass(frozen=True, slots=True)
class _L2Lookup:
    """what one L2 read found.

    :ivar row: a live row, or ``None``
    :ivar marker: an absent-marker, whatever its generation, or ``None``
    :ivar revision: whenever the key held no live row, the revision of its latest message -- a
        marker, an expired row, an undecodable entry, a deletion, or ``0`` for a key with no
        message at all -- so a replacement written at it lands only if nothing has happened to the
        key since this read; ``None`` for a live row, or when L2 could not be read
    """

    row: dict[str, Any] | None
    marker: _AbsentMarker | None
    revision: int | None


@dataclass(frozen=True, slots=True)
class _L2BeforeWrite:
    """L2's state for one key as a save read it, before its L3 write.

    :ivar fenced: whether the save's L2 write is conditional -- ``False`` when there is no L2
        bucket, or no L3 pool (L2 is then the source of truth, and its writes are last-writer-wins)
    :ivar revision: the revision of the key's latest message, a deletion included, or ``0`` for a
        key with no message; ``None`` when the save is unfenced or L2 could not be read
    """

    fenced: bool
    revision: int | None


@dataclass(slots=True)
class _KeyActivity:
    """what this process is doing to one key, tracked only while something is watching it.

    :ivar changes: how many writes have begun on the key, and evictions of it have landed, since
        it was first watched
    :ivar writers: writes of the key in flight
    :ivar holders: tickets outstanding on the key, reads and writes both; the entry is dropped when
        the last is released, so the map holds only keys with work in flight
    """

    changes: int = 0
    writers: int = 0
    holders: int = 0


@dataclass(frozen=True, slots=True)
class _KeyTicket:
    """one read's or one write's claim on a key, taken before its first await.

    :ivar key: the key, as the pk values' string forms
    :ivar activity: the key's shared record
    :ivar seen: ``activity.changes`` as this ticket left it
    :ivar contended: whether a write of the key was already in flight when this ticket was taken
    :ivar writing: whether the ticket is a write's
    """

    key: tuple[str, ...]
    activity: _KeyActivity
    seen: int
    contended: bool
    writing: bool


class _L1Fence:
    """orders this process's L1 writes of a key against every other write and eviction of it.

    Every L1 write that follows an await -- a pull-through caching what L2 or L3 returned, a save
    caching the row it committed -- is a claim that the row is still the newest, and the await is
    where something newer can land: this process's own write of the key, or a peer's broadcast
    evicting it. Caching after that leaves L1 behind the tier it was read from, with nothing left to
    evict it. The L2 revision fences this between processes; this is the same fence inside one,
    and the only one a collection with no L2 has.

    A read or write takes a ticket before its first await and may cache only while no write of the
    key began, and no eviction of it landed, since -- and, for a write, while no other write of the
    key was in flight when it began. Two overlapping writes can reach L3 in either order, so neither
    knows it is the later one; both drop the key, and the next read takes whichever row L3 kept.
    The check and the L1 write run with no await between them.
    """

    __slots__ = ("_keys",)

    def __init__(self) -> None:
        """start with no key watched.

        :return: nothing
        :rtype: None
        """
        self._keys: dict[tuple[str, ...], _KeyActivity] = {}

    def begin(self, key: tuple[str, ...], *, writing: bool) -> _KeyTicket:
        """take a ticket on ``key`` before the first await of a read or write of it.

        :param key: the key, as the pk values' string forms
        :ptype key: tuple[str, ...]
        :param writing: whether the ticket is a write's, which every other ticket on the key sees
        :ptype writing: bool
        :return: the ticket
        :rtype: _KeyTicket
        """
        activity = self._keys.get(key)
        if activity is None:
            activity = _KeyActivity()
            self._keys[key] = activity
        contended = activity.writers > 0
        if writing:
            activity.writers += 1
            activity.changes += 1
        activity.holders += 1
        return _KeyTicket(key=key, activity=activity, seen=activity.changes, contended=contended, writing=writing)

    @staticmethod
    def still_newest(ticket: _KeyTicket) -> bool:
        """whether the row ``ticket``'s holder is about to cache is still the newest this process knows.

        :param ticket: the holder's ticket
        :ptype ticket: _KeyTicket
        :return: ``True`` when no write overlapped the ticket and no eviction landed since it was taken
        :rtype: bool
        """
        return not ticket.contended and ticket.activity.changes == ticket.seen

    def end(self, ticket: _KeyTicket) -> None:
        """release ``ticket``, dropping the key's record once nothing holds it.

        :param ticket: the ticket to release
        :ptype ticket: _KeyTicket
        :return: nothing
        :rtype: None
        """
        activity = ticket.activity
        if ticket.writing:
            activity.writers -= 1
        activity.holders -= 1
        if activity.holders == 0 and self._keys.get(ticket.key) is activity:
            del self._keys[ticket.key]

    def changed(self, key: tuple[str, ...]) -> None:
        """record that ``key`` was evicted or rewritten, so no ticket taken before now may cache it.

        :param key: the key, as the pk values' string forms
        :ptype key: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        activity = self._keys.get(key)
        if activity is not None:
            activity.changes += 1

    @contextmanager
    def watching(self, key: tuple[str, ...], *, writing: bool) -> Iterator[_KeyTicket]:
        """hold a ticket on ``key`` for the body, released however the body ends.

        :param key: the key, as the pk values' string forms
        :ptype key: tuple[str, ...]
        :param writing: whether the ticket is a write's
        :ptype writing: bool
        :return: the ticket, for :meth:`still_newest`
        :rtype: Iterator[_KeyTicket]
        """
        ticket = self.begin(key, writing=writing)
        try:
            yield ticket
        finally:
            self.end(ticket)


class _NatsClientFromRegistry:
    """sentinel type for :data:`NATS_CLIENT_FROM_REGISTRY`."""

    __slots__ = ()


# default for the ``nats_client`` constructor argument: resolve the L2
# client from the registry (``CollectionRegistry.get_l2_client``), so
# ``registry.configure(l2_client=...)`` / ``bind_table(..., l2_client=...)``
# are effective wiring paths. distinguishes "argument omitted" from an
# explicit ``None`` (which keeps its historical meaning: L2 disabled for
# this collection regardless of registry state).
NATS_CLIENT_FROM_REGISTRY: Final = _NatsClientFromRegistry()


class BaseCollection(ABC, Generic[EntityT]):
    """abstract base collection with three-tier caching (L1 -> L2 -> L3).

    :cvar primary_key_column: name of primary-key column (single-pk
        shape, ``str``) or tuple of column names in declared order
        (composite-pk shape, ``tuple[str, ...]``). part of the
        collection-entity contract (siblings read it during CAS and
        cache writes). subclasses override with their table's pk.
        :attr:`primary_key_columns` is the internal-use normalized
        tuple form; callers iterating pk columns MUST read that
        property rather than inspecting the attribute directly.
    :ivar l3_pool: the L3 durable-store handle for this collection. its
        concrete type depends on the configured backend (an **asyncpg pool**
        for the SQL backend; a git store for a git backend; ``None`` when the
        collection is configured without L3, e.g. unit tests using only L1+L2).
        for the **SQL backend** this is the public extension seam for ad-hoc
        SQL: subclasses and external callers (hub endpoints implementing keyset
        pagination, JOINs, bulk queries) may invoke ``await self.l3_pool.fetch(...)``
        / ``execute(...)`` / ``fetchrow(...)`` directly when the query cannot be
        expressed through the Collection API. prefer the collection methods
        (``get``, ``save_entity``, ``delete``, ``__getitem__``, ``__setitem__``)
        for standard CRUD; drop to a backend-specific handle only when no
        Collection method fits. the handle is shared across every collection
        bound to the same backend (resolved through :class:`CollectionRegistry`);
        callers MUST NOT call ``close()`` on it from a collection method or in
        any per-request flow -- its lifecycle is owned by the process that
        constructed the registry. ``None`` is a valid value: callers that need
        to operate without L3 must guard with ``if self.l3_pool is not None``
        rather than assuming presence.
    """

    primary_key_column: str | tuple[str, ...] = "id"

    #: Columns holding timestamps, declared rather than hand-rehydrated.
    #:
    #: The L2 JSON codec renders a ``datetime`` as an ISO string, so a row read through L2
    #: differs in TYPE from the identical row read through L1 or L3 unless something restores
    #: it. That asymmetry is cosmetic while the row is only read; it becomes a fault the moment
    #: one is written BACK, because an update fences on ``date_updated`` as an optimistic lock
    #: and a string bound against ``TIMESTAMPTZ`` fails at the asyncpg border.
    #:
    #: Declaring the columns rather than overriding :meth:`deserialize` is what stops this being
    #: solved once per collection. It had been solved three times, in three packages, with three
    #: different answers to what "rehydrate" means -- tuple versus frozenset, coerce versus pass
    #: through, raise versus preserve the string. Every collection now gets one answer, and gets
    #: it by naming its columns.
    #:
    #: A ``frozenset`` so a subclass can extend its parent's set with ``|`` rather than
    #: restating it, which is how a column gets silently dropped.
    datetime_columns: ClassVar[frozenset[str]] = frozenset()

    #: How long a recorded absence may live, or ``None`` to record none. At least one second.
    #:
    #: Without it, a lookup of a key nobody ever wrote misses L1 and L2 and reaches L3 every
    #: time -- which, for a denylist checked on every request ("is this token revoked?"), puts a
    #: database query behind nearly every call. Setting it records a full miss as an absent-marker
    #: in L1 and in L2, stamped with the table's write generation
    #: (:meth:`CollectionRegistry.set_generation_source`), read BEFORE the L3 lookup that found
    #: nothing. A marker answers only while its stamp is the table's current generation, and every
    #: committed write advances the generation, so a marker cannot outlive a write that landed
    #: after its L3 read -- whichever pod, principal, broadcast or clock was involved.
    #:
    #: The max age is the backstop, not the mechanism: it bounds a marker only when a write
    #: commits and then fails to advance the generation, and it is the server-side lifetime that
    #: keeps markers from filling the shared L2 bucket. Opting in requires a generation source on
    #: the registry, refuses deferred L3 flushes (a write visible before its row lands would be
    #: hidden by a marker recorded in between), and refuses subscript writes (fire-and-forget
    #: cannot report a generation it failed to advance).
    #:
    #: A failed L2 write degrades exactly as it does on any collection: the marker it did not
    #: replace was stamped before the commit advanced the generation, so it has already stopped
    #: answering.
    #:
    #: **Wiring.** Without an L3 pool the opt-in is inert -- there is no durable tier for an
    #: absence to be an absence OF. With an L3 pool it is live whether or not an L2 client is
    #: wired, deliberately: a writer that has L3 and no L2 must still advance the generation,
    #: because the absences it has to invalidate were recorded by OTHER pods that do have L2. So
    #: construction still requires a generation source there, and still refuses the write shapes
    #: above.
    negative_cache_max_age: ClassVar[timedelta | None] = None

    #: The column holding each row's expiry time, or ``None`` for rows that never expire.
    #:
    #: A row whose expiry has passed is absent to every read that answers "does this exist" --
    #: ``get``, ``ensure``, ``collection[id]`` -- at L1, L2 and L3, so correctness never depends on
    #: anything sweeping it; deleting expired rows is table-size hygiene only. A ``None`` value in
    #: the column means that row does not expire. Must be one of :attr:`datetime_columns`, so an
    #: L2 read decodes it or reports the entry corrupt rather than failing mid-comparison.
    expires_at_column: ClassVar[str | None] = None

    #: How this collection's L3 writes land: ``"synchronous"`` before the write returns,
    #: ``"write_behind"`` through the write buffer, or ``None`` to follow the process-wide
    #: ``collection_flush`` strategy and table list.
    #:
    #: Declared on the collection because the right answer is a property of the data, not of the
    #: deployment: an attempt counter can lose one flush interval of increments to a broker wipe
    #: coinciding with a writer crash and nobody is harmed, while a revocation cannot lose one.
    #: ``"write_behind"`` requires a write buffer at construction.
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """refuse, at class definition, an expiry or negative-cache setting that cannot work.

        :param kwargs: forwarded to :func:`object.__init_subclass__`
        :ptype kwargs: Any
        :return: None
        :rtype: None
        :raises TypeError: when :attr:`expires_at_column` is not a declared datetime column, or
            :attr:`negative_cache_max_age` is under one second
        """
        super().__init_subclass__(**kwargs)
        if cls.expires_at_column is not None and cls.expires_at_column not in cls.datetime_columns:
            raise TypeError(
                f"{cls.__name__}.expires_at_column {cls.expires_at_column!r} must be one of its "
                f"datetime_columns, so every tier reads it back as a time"
            )
        if cls.negative_cache_max_age is not None and cls.negative_cache_max_age < timedelta(seconds=1):
            raise TypeError(
                f"{cls.__name__}.negative_cache_max_age must be at least one second, the finest "
                f"server-side lifetime an L2 entry can carry; got {cls.negative_cache_max_age}"
            )

    # datasource-task-06 DS-06-04: per-concrete-class memo of table
    # names that have already emitted the "nats_client missing"
    # warning, so a busy write path logs the wiring gap once rather
    # than per-write. each subclass gets its own set via the
    # ``type(self)._missing_nats_warned_tables = ...`` assignment in
    # :meth:`_warn_missing_nats_client_once`; declared here so the
    # attribute is typed + present on the base.
    _missing_nats_warned_tables: ClassVar[set[str]] = set()

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: NatsClient | _NatsClientFromRegistry | None = NATS_CLIENT_FROM_REGISTRY,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
        # L2 resolution mirrors L1/L3: when the argument is omitted, the
        # registry is the wiring path (``configure(l2_client=...)`` /
        # ``bind_table``). an explicit client always wins; an explicit
        # ``None`` disables L2 for this collection.
        if isinstance(nats_client, _NatsClientFromRegistry):
            self._nats_client: NatsClient | None = registry.get_l2_client(self.table_name)
        else:
            self._nats_client = nats_client
        self._kv: NatsKvBucket | None = None
        self._write_buffer = write_buffer
        self._flush_strategy = FlushStrategy(config.collection_flush)
        self._flush_tables = frozenset(t.strip() for t in config.collection_flush_tables.split(",") if t.strip())
        # Resolve L1 and L3 from registry
        self._l1 = registry.get_l1_backend(self.table_name)
        self.l3_pool = registry.get_l3_pool(self.table_name)
        self._next_absent_marker_sweep = 0.0
        self._last_generation_warning: float | None = None
        self._refuse_unsound_negative_cache()
        if self._negative_cache_active and self._l1 is not None:
            self._l1.initialize(_ABSENT_MARKER_METADATA)
        # Auto-register
        registry.register(self)

    def _refuse_unsound_negative_cache(self) -> None:
        """refuse, at construction, a negative-caching collection whose wiring would let a marker lie.

        :return: None
        :rtype: None
        :raises ValueError: when this collection opts into negative caching with an L3 pool but the
            registry carries no generation source, or the table's L3 writes are deferred -- with
            or without an L2 client, because a writer without one must still advance the generation
            other pods' absences are stamped with
        """
        if self.l3_write_policy == "write_behind" and self.l3_pool is not None and self._write_buffer is None:
            raise ValueError(
                f"{type(self).__name__} declares l3_write_policy='write_behind' but was constructed with "
                f"no write buffer, so its L3 writes would have nowhere to wait"
            )
        if not self._negative_cache_writes_advance:
            return
        if self._registry.generation_source is None:
            raise ValueError(
                f"{type(self).__name__} opts into negative caching but its registry has no generation "
                f"source: nothing could invalidate a recorded absence when a write lands. wire "
                f"registry.set_generation_source(...) before constructing it"
            )
        if self._declares_deferred_l3_writes:
            raise ValueError(
                f"{type(self).__name__} opts into negative caching but {self.table_name!r} defers its L3 "
                f"writes: a reader between the L2 write and the buffered flush would record the row "
                f"absent under the generation that write already advanced"
            )

    @property
    def registry(self) -> CollectionRegistry | None:
        """the registry this collection was constructed with, or ``None`` for one built without one.

        public so a subclass in another package can reach the registry's shared services -- the
        pod's scan cache, above all -- without binding to this class's private slot. ``None`` for an
        instance that never ran :meth:`__init__` (a bare one a test builds to drive SQL alone):
        callers treat the registry's services as an optimisation and still serve without it.

        :return: the registry, or ``None``
        :rtype: CollectionRegistry | None
        """
        result: CollectionRegistry | None = getattr(self, "_registry", None)
        return result

    @property
    def broadcasts_invalidations(self) -> bool:
        """whether this collection's evictions reach other replicas.

        ``False`` when it was built with no NATS client: a write or eviction then drops the key on
        this replica only, and every other replica keeps serving the row it cached. A host that
        runs shared work over a collection -- a tick under a cross-pod lock -- checks this to refuse
        a collection wired without the bus rather than let it evict locally in silence.

        :return: ``True`` when the collection holds a NATS client to broadcast through
        :rtype: bool
        """
        return self._nats_client is not None

    @property
    def required_l3_pool(self) -> L3Backend:
        """:attr:`l3_pool`, or a clear failure saying why it had to be there.

        For the raw-SQL escape hatch. :attr:`l3_pool` is legitimately ``None`` -- a
        collection can be configured on L1+L2 alone -- so every ad-hoc query has to
        establish that it is not, and the documented instruction is to guard rather than
        assume. In practice the guard was routinely skipped, because ``await
        self.l3_pool.fetch(...)`` reads fine and only fails when someone actually
        constructs the collection without L3. What they then get is
        ``AttributeError: 'NoneType' object has no attribute 'fetch'`` from inside a query
        method, which says nothing about the real mistake.

        A query written in SQL cannot degrade to "no L3" in any meaningful way, so the
        honest behaviour is to fail immediately and say what is missing. Use this wherever
        the query genuinely requires the backend; keep ``if self.l3_pool is not None`` for
        the callers that have a real fallback.

        :return: the L3 backend handle, guaranteed non-``None``
        :rtype: L3Backend
        :raises RuntimeError: when this collection has no L3 backend configured
        """
        if self.l3_pool is None:
            raise RuntimeError(
                f"{type(self).__name__} (table {self.table_name!r}) needs an L3 backend for this "
                "query, but none is configured. Raw SQL has no meaningful L1/L2-only fallback -- "
                "either configure an L3 pool on the CollectionRegistry, or call a Collection API "
                "method that can serve from cache instead."
            )
        return self.l3_pool

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the database table name for this collection."""
        ...

    @property
    @abstractmethod
    def entity_class(self) -> type[EntityT]:
        """Return the entity class for this collection."""
        ...

    @property
    def primary_key_columns(self) -> tuple[str, ...]:
        """normalize :attr:`primary_key_column` to tuple form.

        single-pk subclasses declare ``primary_key_column = "foo"`` and
        read ``("foo",)`` here; composite-pk subclasses declare
        ``primary_key_column = ("a", "b")`` and read the same tuple.
        every internal caller iterating pk columns uses this property.

        :return: tuple of pk column names in declared order
        :rtype: tuple[str, ...]
        """
        if isinstance(self.primary_key_column, tuple):
            return self.primary_key_column
        return (self.primary_key_column,)

    def normalize_pk(self, entity_id: Any) -> tuple[Any, ...]:
        """normalize caller-supplied id to tuple of pk values.

        single-pk collections accept either ``value`` or ``(value,)``
        and return ``(value,)``; composite-pk collections MUST receive
        a tuple of length matching :attr:`primary_key_columns`. a
        non-tuple input is wrapped in a 1-tuple.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: tuple of pk values matching
            :attr:`primary_key_columns` length
        :rtype: tuple[Any, ...]
        :raises ValueError: if tuple length does not match
            :attr:`primary_key_columns`
        """
        if isinstance(entity_id, tuple):
            values = entity_id
        else:
            values = (entity_id,)
        pk_cols = self.primary_key_columns
        if len(values) != len(pk_cols):
            raise ValueError(
                f"{self.table_name}: primary key arity mismatch: "
                f"got {len(values)} value(s) for {len(pk_cols)} column(s) {pk_cols}"
            )
        return values

    @abstractmethod
    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one entity's row from the L3 durable tier, keyed by pk.

        public extension point — the L3 backend is **pluggable** (storage-
        agnostic): the SQL backend issues a SELECT, a git backend reads the
        entity's file, an in-memory backend reads a dict — the framework only
        requires a row dict on hit / ``None`` on miss. invoked on an L1+L2 miss
        via :meth:`_pull_through` and on :meth:`reload_entity`. callers needing
        a direct-to-L3 read without cache side-effects may invoke it directly;
        prefer :meth:`ensure` or :meth:`get` for the normal three-tier path.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk). composite-pk backends MUST accept the tuple shape;
            single-pk backends accept the scalar shape.
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        ...

    @abstractmethod
    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """persist one entity's row to the L3 durable tier.

        public extension point — the L3 backend is **pluggable** (storage-
        agnostic): the SQL backend issues an upsert, a git backend writes +
        stages the entity's file, an in-memory backend writes a dict. invoked
        on every non-deferred :meth:`save_entity` and from :meth:`persist_to_store`
        during write-buffer flush.

        :param data: row data keyed by column name; pk columns named in
            :attr:`primary_key_columns` MUST be present
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-modification ``date_updated``
            for optimistic-lock validation, ``None`` for inserts. a
            non-``None`` value means the row was read as existing, so a
            backend MUST write it update-only: a row deleted since the read
            answers 0, never a re-insert
        :ptype original_timestamp: datetime | None
        :param conn: optional **backend-specific transaction handle** (e.g. an
            asyncpg connection for the SQL backend) that overrides
            :attr:`l3_pool` for this single write, so it commits atomically
            with whatever other operations the caller already issued on the
            same transaction. ``None`` uses the collection's own L3 store.
            backends MUST honor this so the framework's transactional
            save_entity path stays atomic.
        :ptype conn: Any
        :return: rows affected (0 on optimistic-lock failure or a row deleted
            since it was read, 1 on success)
        :rtype: int
        """
        ...

    @property
    def emits_cas_fence(self) -> bool:
        """whether EVERY L3 write this collection generates carries a CAS fence.

        ``False`` by default, and for the ordinary collection it is the truth:
        a save with ``original_timestamp=None`` is written unfenced, so a
        ``save_to_store`` returning 0 means the backend chose not to write
        (``ON CONFLICT DO NOTHING``, a no-op update) rather than "another
        writer beat me".

        a collection that fences unconditionally --
        :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`
        over a ``TableSchema(cas_null_safe=True)`` -- overrides this to ``True``.
        the framework then knows a 0 rowcount is a LOST RACE on every write
        path, including the fire-and-forget ones that have no caller left to
        return it to, and says so in the log instead of dropping it silently.

        :return: whether a 0 rowcount from :meth:`save_to_store` always means
            "lost the race"
        :rtype: bool
        """
        return False

    def complete_written_row(self, data: dict[str, Any]) -> dict[str, Any]:
        """``data`` with every column the write stores as a value known before it runs, filled in.

        Public extension point, paired with :meth:`columns_decided_by_store`: a column a write
        leaves out is not always one the database decides. When the write is certain to store a
        known value for it -- ``NULL``, for a nullable column with no default that the statement
        writes either way -- the row cached is completed with that value instead of being read
        back. The framework completes the row before asking which columns the database decides,
        and caches the completed row.

        The default completes nothing. :class:`~threetears.core.collections.schema_backed
        .SchemaBackedCollection` completes from its declared columns when it writes through its
        generated SQL.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the row with every column of known stored value present; ``data`` itself when
            there is none to add
        :rtype: dict[str, Any]
        """
        return data

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the columns whose stored value writing ``data`` leaves to the database, not to ``data``.

        Public extension point. A write may leave columns for the database to decide: a server
        default for a column it does not name, the stored value an update keeps for a column it
        leaves out. The row the write sent is then not the row L3 holds, and a tier that cached
        it would serve a row missing those columns -- on every replica reading L2, for as long as
        the entry lives. A non-empty answer says so, and the framework acts on it at every write:

        - a synchronous write (:meth:`save_entity`, an assignment) reads the row back from L3 once
          it commits, and caches that;
        - a write that reaches L1 and L2 before L3 (write-behind, :meth:`l2_cas_mutate` on a
          collection with an L3 pool) has nothing to read back yet, so it is refused before any
          tier takes it.

        The default answers none: a collection that declares no columns writes the row it is
        given. :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection` answers
        from its declared columns. Never consulted on a collection with no L3 pool, whose L1 and
        L2 are the record and fill nothing in.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the names of the columns the database decides, in declared order; empty when
            ``data`` is the row L3 holds after the write
        :rtype: tuple[str, ...]
        """
        return ()

    def _write_ahead_row(self, data: dict[str, Any]) -> dict[str, Any]:
        """the row a write that reaches L1 and L2 before L3 caches, refused when the database decides any of it.

        Such a write has no committed row to read back when it is cached, so the tiers would hold
        the row as sent until it expired: missing the columns the database filled in, on every
        replica reading L2. A collection with no L3 pool is the record itself and caches ``data``.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the row completed by :meth:`complete_written_row`, or ``data`` with no L3 pool
        :rtype: dict[str, Any]
        :raises ValueError: when this collection has an L3 pool and :meth:`columns_decided_by_store`
            names any column of the completed row
        """
        if self.l3_pool is None:
            return data
        completed = self.complete_written_row(data)
        decided = self.columns_decided_by_store(completed)
        if decided:
            raise ValueError(
                f"{type(self).__name__}: a write to {self.table_name!r} reaches L1 and L2 before L3, so "
                f"its row must be the row L3 will hold, but the database decides {list(decided)!r}. "
                f"name every one of them in the row, with the value it should store"
            )
        return completed

    async def _stored_row(self, entity_id: Any, data: dict[str, Any]) -> dict[str, Any] | None:
        """the whole row L3 holds after a committed write of ``data``: ``data`` itself, or a read of it.

        The row is first completed by :meth:`complete_written_row`, and read from L3 only when
        :meth:`columns_decided_by_store` names a column of it; never on a collection with no L3
        pool. The read runs after the commit, so it sees this write or a later one; either is what
        L3 holds, and the caller's fences decide whether it is cached.

        A read that fails answers ``None``, which caches nothing: the write has committed, so the
        save still succeeds, and the next read of the key goes to L3. A read that finds no row
        (deleted since the commit) answers ``None`` too. A cancelled read drops the key from this
        process's L1, which may hold the row as sent, and propagates.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param data: the row as written, stamped
        :ptype data: dict[str, Any]
        :return: the row L3 holds, or ``None`` when it could not be read
        :rtype: dict[str, Any] | None
        """
        completed = data if self.l3_pool is None else self.complete_written_row(data)
        decided = () if self.l3_pool is None else self.columns_decided_by_store(completed)
        stored: dict[str, Any] | None = completed
        if decided:
            stored = None
            try:
                stored = await self.fetch_from_store(entity_id)
            except Exception as exc:
                log.warning(
                    "reading back a row the database completed failed; nothing caches it and the next read goes to L3",
                    extra={
                        "extra_data": {
                            "entity_id": str(entity_id),  # convert at border: log extra_data field
                            "table": self.table_name,
                            "decided_by_store": list(decided),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    },
                )
            # BaseException after Exception: a cancellation is not an Exception, and leaves L1
            # holding the row as sent unless it is dropped here.
            except BaseException:
                self._evict_l1(entity_id)
                raise
            else:
                if stored is None:
                    log.info(
                        "a row the database completed was deleted before it was read back; it is cached nowhere",
                        extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name}},
                    )
        return stored

    async def _drop_unread_row(self, entity_id: Any) -> None:
        """drop a committed row that could not be read back from this replica's L1 and from L2.

        L2 may hold the row this write replaced, so its key is deleted, which is always correct:
        the next read seeds it from L3. Peers are told by the write's own broadcast.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._clear_l1_marker(entity_id)
        self._evict_l1(entity_id)
        await self._delete_from_l2(entity_id)

    @property
    def persists_l2_order(self) -> bool:
        """whether this collection's L3 stores the order a compare-and-swap won, and fences on it.

        ``False`` by default. :meth:`l2_cas_mutate` on a collection with an L3 pool requires
        ``True``: its winners persist independently, and only a write conditional on the stored
        order stops an earlier winner that lands last from overwriting a later one (see
        :mod:`threetears.core.collections.l2_order`). A collection answering ``True`` declares
        the ``l2_epoch`` / ``l2_revision`` columns and implements :meth:`save_ordered_to_store`;
        :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection` does both from
        its declared schema.

        :return: whether :meth:`save_ordered_to_store` is implemented and the order columns exist
        :rtype: bool
        """
        return False

    async def save_ordered_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        """persist a compare-and-swap row to L3 only over a row whose stored order is older.

        ``data`` carries the order its swap won in ``l2_epoch`` / ``l2_revision``. The write lands
        when no row exists, or when the stored row's order is strictly older -- ``NULL`` being
        older than every order -- and otherwise leaves the stored row alone and reports 0 rows.
        That 0 is not a failure: the stored row came from a later swap, which built on this one.

        Public extension point, like :meth:`save_to_store`; only reached when
        :attr:`persists_l2_order` is ``True``.

        :param data: row payload, order columns included
        :ptype data: dict[str, Any]
        :param conn: optional backend-specific transaction handle the write joins
        :ptype conn: Any
        :return: rows affected: 1 when written, 0 when the stored order is newer or equal
        :rtype: int
        :raises NotImplementedError: on a collection that does not persist the order
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not persist the L2 order (persists_l2_order is False), so it "
            f"cannot write a compare-and-swap row fenced on it"
        )

    @abstractmethod
    async def delete_from_store(self, entity_id: Any) -> None:
        """delete one entity's row from the L3 durable tier, keyed by pk.

        public extension point — the L3 backend is **pluggable** (storage-
        agnostic): the SQL backend issues a DELETE, a git backend removes the
        entity's file, an in-memory backend drops the dict entry. invoked from
        :meth:`delete`.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        ...

    @abstractmethod
    def serialize(self, data: dict[str, Any]) -> bytes:
        """encode row dict to bytes for the L2 (NATS KV) tier.

        public extension point. subclasses override to apply their
        JSON codec (typically :func:`threetears.core.serialization.serialize_to_json`)
        plus any domain-specific pre-encoding.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: serialized bytes ready for L2 write
        :rtype: bytes
        """
        ...

    @abstractmethod
    def deserialize(self, data: bytes) -> dict[str, Any]:
        """decode bytes from the L2 tier back to a row dict.

        public extension point. subclasses override to reverse
        :meth:`serialize`, rehydrating typed fields (UUID, Decimal,
        datetime) from their JSON representations.

        :param data: serialized bytes previously produced by
            :meth:`serialize`
        :ptype data: bytes
        :return: row dict keyed by column name
        :rtype: dict[str, Any]
        """
        ...

    # --- timestamp discipline across the L2 boundary ---
    #
    # One rule, enforced at both ends: every datetime in this system is timezone-aware UTC.
    # Write-side normalisation is the half that matters, because it is the only one that
    # PREVENTS anything. Read-side coercion is a legacy tail: it repairs values written before
    # the rule existed, says so in the log, and can be deleted once those age out.
    #
    # Coercing a naive value to UTC on read, alone, is the mistake rather than the fix. If the
    # value was local time, coercion shifts it silently by hours and stamps the result
    # authoritative -- worse than leaving a string, because it now looks correct. Normalising on
    # write means the guess has a shrinking, observable lifetime instead of a permanent one.

    def _normalise_datetimes_for_write(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return *data* with every :attr:`datetime_columns` value aware-UTC.

        A naive value is assumed UTC and stamped, because by the time a row reaches here the
        information needed to interpret it any other way is gone. That assumption is recorded
        at WARNING: it is a guess, and a caller producing naive timestamps has a bug upstream
        that will keep producing them until somebody sees this line.

        Copies rather than mutating: the caller's dict is often an entity's live row, and a
        serialization step that edits its input is a surprise nobody reads the code to find.
        """
        if not self.datetime_columns:
            return data
        out = dict(data)
        for column in self.datetime_columns:
            value = out.get(column)
            if isinstance(value, datetime) and value.tzinfo is None:
                log.warning(
                    "naive datetime normalised to UTC on write; the value's real offset is "
                    "unknowable here, so this is an assumption and the producer should be fixed",
                    extra={"extra_data": {"table": self.table_name, "column": column}},
                )
                out[column] = value.replace(tzinfo=UTC)
        return out

    def _rehydrate_datetimes(self, row: dict[str, Any]) -> dict[str, Any]:
        """Restore :attr:`datetime_columns` from the ISO strings the L2 codec produced.

        :raises CorruptCacheEntry: when a value will not parse. The caller treats that as a
            cache miss and falls through to L3 rather than failing the read -- see the
            exception's own docstring for why that beats propagating or papering over it.
        """
        if not self.datetime_columns:
            return row
        for column in self.datetime_columns:
            value = row.get(column)
            if not isinstance(value, str):
                continue
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise CorruptCacheEntry(self.table_name, column, value) from exc
            if parsed.tzinfo is None:
                log.warning(
                    "naive datetime read from L2 and assumed UTC; written before write-side "
                    "normalisation, or by something that bypasses it",
                    extra={"extra_data": {"table": self.table_name, "column": column}},
                )
                parsed = parsed.replace(tzinfo=UTC)
            row[column] = parsed
        return row

    # --- L1 cache (sync, for BaseEntity) ---
    #
    # these five methods are the synchronous cache-access API shared
    # with BaseEntity (and, transitively, the subclasses' __getitem__
    # and __setitem__ paths). they are public because BaseEntity and
    # some subclass-level collections (agent-tools ContextItems,
    # agent-memory MemoriesCollection) call them across the class
    # boundary -- the contract is "if you hold a collection reference
    # you may read/write its L1 through these five methods". mutations
    # always return a bool so the entity can fall back to in-memory
    # ``_changes`` when L1 is absent.

    def get_field_sync(self, entity_id: Any, field: str) -> Any:
        """read one column synchronously from the L1 cache.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :param field: column name to read
        :ptype field: str
        :return: column value, or ``MISSING`` sentinel when L1 is
            absent, the row is not cached, or the column is absent
            from the cached row
        :rtype: Any
        """
        if self._l1 is None:
            return MISSING
        row = self._select_from_l1(entity_id)
        if row is None:
            return MISSING
        return row.get(field, MISSING)

    def set_field_sync(self, entity_id: Any, field: str, value: Any) -> bool:
        """write one column synchronously into the L1 cache.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :param field: column name to write
        :ptype field: str
        :param value: new value for the column
        :ptype value: Any
        :return: true on successful write, false when L1 is absent or
            the row is not yet cached (caller must fall back to the
            in-memory change buffer)
        :rtype: bool
        """
        if self._l1 is None:
            return False
        row = self._select_from_l1(entity_id)
        if row is None:
            return False
        row[field] = value
        self._l1.upsert(self.table_name, row, self.primary_key_columns)
        return True

    def get_row_sync(self, entity_id: Any) -> dict[str, Any] | None:
        """read the full cached row for an entity, synchronously.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :return: row dict, or ``None`` when L1 is absent or the row
            is not cached
        :rtype: dict[str, Any] | None
        """
        return self._select_from_l1(entity_id)

    @property
    def l1_max_age_seconds(self) -> float | None:
        """How long an L1 row cached from a lower tier may be served, or ``None``.

        **``None`` unless a collection opts in, and structurally ``None`` when
        there is no L3.** The second half is the load-bearing one. A collection
        with no L3 pool does not fall back to a slower tier on a miss: the
        L1+L2-only collections override :meth:`get` to return ``None`` on a
        total miss, so an expired row does not become a pull-through, it becomes
        "this row does not exist". Downstream, a compare-and-set that reads
        absence writes a fresh row over a live one -- a presence room with ten
        members replaced by a room with one, and no error anywhere. Expiry is a
        cache mechanism, and a tier that is the source of truth is not a cache.

        :return: the configured bound, or ``None`` when expiry is off
        :rtype: float | None
        """
        if self.l3_pool is None:
            return None
        return self._registry.get_l1_max_age(self.table_name)

    def _select_from_l1(self, entity_id: Any, *, expiring: bool = False) -> dict[str, Any] | None:
        """The one L1 read, with expiry applied only where a miss is repairable.

        Every reader **in this class** routes through here rather than calling
        the backend itself, so the policy has one home. But the callers do not
        share a contract, and that is why ``expiring`` is a parameter rather
        than always-on:

        - **Repairing callers** (:meth:`ensure`, :meth:`_resolve_row`,
          :meth:`_ensure_in_l1`) treat a
          miss as "go to the lower tier", so expiring a row makes it reload.
          That is the whole mechanism, and they pass ``expiring=True``.

          :meth:`_resolve_row` is on this list for a reason worth stating: it
          backs ``collection[id]``, the primary read path, and it reaches L1
          directly rather than through :meth:`get_row_sync`. Routing it through
          that reporting method instead makes the bound unreachable for
          subscript reads -- the non-expiring read returns the stale row and
          nothing further runs -- which is inert, not conservative.
        - **Reporting callers** (:meth:`get_row_sync`, :meth:`get_field_sync`,
          :meth:`set_field_sync`, :meth:`exists_in_cache_sync`) treat a miss as
          "not cached" and return it to a caller that will not fall back.
          Expiring for them turns a stale row into a *deleted* one and reports
          absence, which is worse than the staleness it was bounding:
          ``__setitem__`` reads a field write back through
          :meth:`get_row_sync` and skips propagation when it sees ``None``, so
          the write is silently dropped, and an entity handle held across the
          bound starts answering ``None`` for fields it has.

        The distinction is a miss's *meaning*, not its value. Expiry converts a
        stale hit into a miss, which is only an improvement where a miss is
        cheap and self-correcting.

        The class scoping is deliberate. A subclass in another package can
        still reach ``self._l1`` directly, and one does --
        ``ContextItemCollection.touch`` reads L1 to stamp ``date_accessed``.
        That read is outside the bound, which is harmless there because it
        neither serves the row to a caller nor clears the stamp on write-back,
        but it is not covered by this funnel and should not be assumed to be.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :param expiring: whether to apply the collection's max-age bound; only
            a caller that repairs a miss by pulling through may pass ``True``
        :ptype expiring: bool
        :return: row dict, or ``None`` when L1 is absent, the row is not
            cached, or (when ``expiring``) the row was past its max age
        :rtype: dict[str, Any] | None
        """
        if self._l1 is None:
            return None
        max_age = self.l1_max_age_seconds if expiring else None
        row: dict[str, Any] | None
        if max_age is None:
            # The kwarg is omitted, not passed as None, when expiry is off.
            # ``L1Backend`` is a published Protocol, so an out-of-repo
            # implementation predating this parameter would raise TypeError on
            # EVERY cached read otherwise -- and a bound nobody configured is
            # the overwhelmingly common case, so the whole platform would break
            # for a feature it had not opted into.
            row = self._l1.select_by_id(
                self.table_name,
                self.normalize_pk(entity_id),
                self.primary_key_columns,
            )
        else:
            row = self._l1.select_by_id(
                self.table_name,
                self.normalize_pk(entity_id),
                self.primary_key_columns,
                max_age_seconds=max_age,
            )
        # Row expiry follows the same split as the max-age bound, for the same reason: the reads
        # that answer "does this exist" (get, ensure, collection[id]) are the repairing ones, and
        # an expired row is absent to them. A reporting read serves an entity's own internals --
        # to_dict() during a save reads its row through here -- and hiding the row there turns
        # updating an entity past its expiry into a crash rather than a write.
        if expiring and row is not None and self._row_is_expired(row):
            self._evict_l1(entity_id)
            row = None
        return row

    def _row_is_expired(self, row: dict[str, Any]) -> bool:
        """whether ``row``'s declared expiry time has passed.

        :param row: a row from any tier
        :ptype row: dict[str, Any]
        :return: ``True`` when the collection declares :attr:`expires_at_column` and the row's
            value there is at or before now; ``False`` otherwise, including for a ``None`` value
        :rtype: bool
        :raises TypeError: when the expiry column holds something that is not a time
        """
        column = self.expires_at_column
        if column is None:
            return False
        value = row.get(column)
        if value is None:
            return False
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise TypeError(f"{self.table_name}.{column} must hold a datetime, got {type(value).__name__}")
        if value.tzinfo is None:
            # the platform writes aware UTC; a naive value here is an L1 SQLite read, which drops
            # the zone on the way back out, not a local time.
            value = value.replace(tzinfo=UTC)
        return value <= datetime.now(UTC)

    @property
    def _declares_deferred_l3_writes(self) -> bool:
        """whether this collection's L3 writes are MEANT to wait, whatever it was wired with.

        :return: ``True`` for ``l3_write_policy="write_behind"``, or for no declared policy when
            the process-wide strategy defers this table
        :rtype: bool
        """
        if self.l3_write_policy is not None:
            return self.l3_write_policy == "write_behind"
        return self._flush_strategy != FlushStrategy.ALWAYS and self.table_name in self._flush_tables

    @property
    def _defers_l3_writes(self) -> bool:
        """whether this collection's L3 writes actually go through its write buffer.

        :return: ``True`` when writes are meant to wait and a write buffer exists
        :rtype: bool
        """
        return self._declares_deferred_l3_writes and self._write_buffer is not None

    @property
    def _negative_cache_writes_advance(self) -> bool:
        """whether this collection's committed writes must advance the table's write generation.

        Deliberately independent of L2. Absences are recorded by READERS, which may be other pods
        with L2 while this one has none; a writer that skipped the advance because it has no L2
        client of its own would leave their absences answering over its commit.

        :return: ``True`` when :attr:`negative_cache_max_age` is set and an L3 pool exists
        :rtype: bool
        """
        return self.negative_cache_max_age is not None and self.l3_pool is not None

    @property
    def _negative_cache_active(self) -> bool:
        """whether this collection records and trusts absences, which also makes L2 writes strict.

        :return: ``True`` when writes advance the generation and L2 and a generation source exist
        :rtype: bool
        """
        return (
            self._negative_cache_writes_advance
            and self._nats_client is not None
            and self._registry.generation_source is not None
        )

    async def _current_generation(self) -> str | None:
        """this table's write generation, or ``None`` when it cannot be read.

        ``None`` means no absence may be trusted or recorded this time; the caller asks L3.

        :return: the generation token, or ``None``
        :rtype: str | None
        """
        source = self._registry.generation_source
        if source is None:
            return None
        try:
            return await source.current(self.table_name)
        except GenerationUnavailableError as exc:
            # every lookup lands here while the source is down; one warning per interval says so.
            # absence of a previous warning means never warned, not a warning at time zero.
            now = time.monotonic()
            last = self._last_generation_warning
            if last is None or now - last >= _GENERATION_WARNING_INTERVAL_SECONDS:
                self._last_generation_warning = now
                log.warning(
                    "write generation unavailable; asking L3 rather than trusting a recorded absence",
                    extra={"extra_data": {"table": self.table_name, "error": str(exc)}},
                )
            return None

    async def _advance_generation(self) -> GenerationUnavailableError | None:
        """advance this table's write generation after a committed write, when negative caching is on.

        :return: the failure, for the caller to raise once the rest of the write path has run, or
            ``None`` when the generation advanced or nothing needed advancing
        :rtype: GenerationUnavailableError | None
        """
        # the opt-in is checked before anything else is touched, so a collection that never opted
        # in runs none of this path however it was assembled. L2 is deliberately not part of it.
        if not self._negative_cache_writes_advance:
            return None
        source = self._registry.generation_source
        if source is None:
            return None
        try:
            await source.advance(self.table_name)
        except GenerationUnavailableError as exc:
            log.error(
                "write generation could not be advanced after a committed write; absences recorded "
                "before it stay trusted until they expire",
                extra={"extra_data": {"table": self.table_name, "error": str(exc)}},
            )
            return exc
        return None

    def _absent_marker_key(self, entity_id: Any) -> str:
        """the L1 marker key for one pk: table-qualified, digested so any pk shape fits.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: the key
        :rtype: str
        """
        body = "\x1f".join(str(v) for v in self.normalize_pk(entity_id))
        return f"{self.table_name}.{hashlib.sha256(body.encode('utf-8')).hexdigest()}"

    def _l1_marker_matches(self, entity_id: Any, generation: str) -> bool:
        """whether this pod's L1 holds a live absent-marker for ``entity_id`` under ``generation``.

        A marker under any other generation, or past its deadline, is removed.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param generation: the table's current write generation
        :ptype generation: str
        :return: ``True`` when the marker may answer
        :rtype: bool
        """
        if self._l1 is None:
            return False
        key = (self._absent_marker_key(entity_id),)
        row = self._l1.select_by_id(_ABSENT_MARKER_TABLE, key, ("key",))
        if row is None:
            return False
        if row["generation"] == generation and time.monotonic() < float(row["deadline"]):
            return True
        self._l1.delete_by_id(_ABSENT_MARKER_TABLE, key, ("key",))
        return False

    def _write_l1_marker(self, entity_id: Any, generation: str) -> None:
        """record in this pod's L1 that ``entity_id`` is absent under ``generation``.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param generation: the generation read before the L3 lookup that found nothing
        :ptype generation: str
        :return: None
        :rtype: None
        """
        if self._l1 is None or self.negative_cache_max_age is None:
            return
        now = time.monotonic()
        self._l1.upsert(
            _ABSENT_MARKER_TABLE,
            {
                "key": self._absent_marker_key(entity_id),
                "generation": generation,
                "deadline": now + self.negative_cache_max_age.total_seconds(),
            },
            "key",
        )
        if now >= self._next_absent_marker_sweep:
            self._next_absent_marker_sweep = now + _ABSENT_MARKER_SWEEP_INTERVAL_SECONDS
            self._sweep_expired_l1_markers(now)

    def _sweep_expired_l1_markers(self, now: float) -> None:
        """delete every absent-marker past its deadline from this pod's L1.

        A marker is otherwise removed only when its own key is read or written again, and the keys
        a denylist checks -- one per token -- are rarely seen twice, so without this the table grows
        with every distinct key ever looked up. Runs from the marker write path at most once per
        :data:`_ABSENT_MARKER_SWEEP_INTERVAL_SECONDS` per collection and drains the backlog in
        batches of :data:`_ABSENT_MARKER_SWEEP_BATCH`, so the table never holds more than the
        markers written within one max age plus one interval, whatever the miss rate.

        :param now: the monotonic time the caller already read
        :ptype now: float
        :return: None
        :rtype: None
        """
        if self._l1 is None:
            return
        while True:
            expired = self._l1.execute_query(
                f"SELECT key FROM {_ABSENT_MARKER_TABLE} WHERE deadline <= ? LIMIT {_ABSENT_MARKER_SWEEP_BATCH}",
                (now,),
            )
            for row in expired:
                self._l1.delete_by_id(_ABSENT_MARKER_TABLE, (row["key"],), ("key",))
            if len(expired) < _ABSENT_MARKER_SWEEP_BATCH:
                return

    def _clear_l1_marker(self, entity_id: Any) -> None:
        """drop this pod's L1 absent-marker for ``entity_id``, if it holds one.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: None
        :rtype: None
        """
        if self._l1 is None or not self._negative_cache_active:
            return
        self._l1.delete_by_id(_ABSENT_MARKER_TABLE, (self._absent_marker_key(entity_id),), ("key",))

    def write_to_cache_sync(
        self,
        data: dict[str, Any],
        primary_key: str | tuple[str, ...] | None = None,
        *,
        from_lower_tier: bool = False,
    ) -> bool:
        """upsert full row into L1 cache, synchronously.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param primary_key: override collection's pk column(s) for this
            write; ``None`` defaults to :attr:`primary_key_columns`.
            accepts either single column name (str) or tuple of column
            names (composite-pk override).
        :ptype primary_key: str | tuple[str, ...] | None
        :param from_lower_tier: whether ``data`` was just read from L2 or
            L3 rather than authored here. Stamps the row's provenance, which
            is what makes it eligible for max-age expiry. **A subclass
            accessor that reads L3 and caches the result must pass this**:
            without it the row is indistinguishable from a local write and
            never expires, so the rows most likely to go stale are exactly
            the ones exempt. Defaults to ``False`` because the unstamped
            reading is the safe one -- it can only under-expire, never
            revert a local write.
        :ptype from_lower_tier: bool
        :return: ``True`` on successful write, ``False`` when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        pk: str | tuple[str, ...] = primary_key if primary_key is not None else self.primary_key_columns
        self._l1.upsert(self.table_name, self._stamped(data) if from_lower_tier else data, pk)
        return True

    def exists_in_cache_sync(self, entity_id: Any) -> bool:
        """true iff the given entity is present in the L1 cache.

        :param entity_id: primary-key value identifying the row
        :ptype entity_id: Any
        :return: presence flag; false when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        row = self._select_from_l1(entity_id)
        return row is not None

    def evict_from_cache_sync(self, entity_id: Any) -> bool:
        """remove a row from the L1 cache only, synchronously.

        narrower than :meth:`invalidate_cache`: drops the L1 slot for
        this pod without touching L2, without publishing a cross-pod
        invalidation, and without awaiting anything. the receiving half
        of a peer's invalidation broadcast
        (:meth:`CollectionRegistry.start_invalidation_listener`) evicts
        through here, so a read of the key in flight on this process
        does not cache what it read before the eviction. also for test
        harnesses that want to simulate an L1 eviction and exercise
        the L2 / L3 fall-through path, and for single-pod cache-
        management flows where L2 coherence is driven separately.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk) identifying the row
        :ptype entity_id: Any
        :return: ``True`` when L1 was present and the row (if any) was
            deleted; ``False`` when L1 is absent
        :rtype: bool
        """
        if self._l1 is None:
            return False
        self._evict_l1(entity_id)
        return True

    @property
    def _l1_fence(self) -> _L1Fence:
        """this collection's in-process ordering of L1 writes per key (:class:`_L1Fence`).

        Made on first use rather than in ``__init__``: a harness may assemble an instance without
        running ``__init__`` and still drive the write paths, and they must order their L1 writes
        all the same.

        :return: the fence
        :rtype: _L1Fence
        """
        fence = self.__dict__.get("_l1_fence_state")
        if fence is None:
            fence = _L1Fence()
            self.__dict__["_l1_fence_state"] = fence
        result: _L1Fence = fence
        return result

    def _fence_key(self, entity_id: Any) -> tuple[str, ...]:
        """the key :attr:`_l1_fence` tracks ``entity_id`` under.

        String forms, because an invalidation arrives with the pk values as strings and must land
        on the same key a local write of the typed values took.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: the pk values' string forms, in declared order
        :rtype: tuple[str, ...]
        """
        return tuple(str(value) for value in self.normalize_pk(entity_id))

    def _evict_l1(self, entity_id: Any) -> None:
        """drop ``entity_id``'s row from this process's L1, and stop any read or write in flight caching it.

        The one way a row leaves this collection's L1, so every eviction also reaches the fence: a
        pull-through that read the row before the eviction would otherwise cache it after.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._l1_fence.changed(self._fence_key(entity_id))
        if self._l1 is not None:
            self._l1.delete_by_id(self.table_name, self.normalize_pk(entity_id), self.primary_key_columns)

    # --- L2 cache (NATS KV, async) ---

    L2_BUCKET_SUFFIX = "collections"

    async def _ensure_kv(self) -> NatsKvBucket | None:
        """lazily resolve and cache the JetStream KV bucket for L2.

        the new wrapper KV API lives on :class:`NatsKvBucket`, not on
        the bare client, so the collection holds a bucket handle
        rather than a raw client. resolution is lazy so collections
        constructed without a connected client (unit tests, L1+L3
        configurations) never touch JetStream.

        whether a collection may CREATE the shared bucket is the registry's
        :attr:`CollectionRegistry.l2_create_if_missing` decision, not this
        method's: baking a literal in here would put one deployment's
        bucket-ownership policy in the library, and ``False`` is only safe
        once the declaring identity re-declares the bucket on every NATS
        reconnect.

        :return: ready bucket handle, or ``None`` when no NATS client
            was supplied
        :rtype: NatsKvBucket | None
        :raises KvError: when bucket binding fails (does NOT swallow
            -- API drift / config errors must surface, not warn)
        """
        if self._nats_client is None:
            return None
        if self._kv is None:
            self._kv = await self._nats_client.kv_bucket(
                name=self.L2_BUCKET_SUFFIX,
                create_if_missing=self._registry.l2_create_if_missing,
            )
        return self._kv

    def l2_key(self, entity_id: Any) -> str:
        """build a grammar-safe, principal-scoped NATS KV key for given pk.

        single-pk shape: ``{scope}.{table_name}.{value}``. composite-pk
        shape: ``{scope}.{table_name}.{v1}_{v2}_...`` -- pk values
        stringified at the NATS boundary (per CLAUDE.md UUID/datetime
        border-conversion rule) and joined with ``"_"`` to form the
        **key body**.

        ``{scope}`` is the registry's :attr:`CollectionRegistry.kv_key_scope`,
        the principal this process authenticates as. Every principal
        shares one ``{ns}-collections`` bucket, so without a leading
        per-principal token the only grant expressible is
        ``$KV.{bucket}.>`` -- every key in the platform, to everyone. The
        scope is what lets the minted grant narrow to
        ``$KV.{bucket}.{scope}.>``. There is ONE tier: every key is
        scoped, always; no key is shared.

        The scope is never hashed and never derived from a sanitized
        display name. Not hashed, because an operator reading a subject
        in a grant has to be able to tell whose it is. Not
        name-derived, because the dots-to-dash sanitizer is
        non-injective, and two principals landing on one scope is
        precisely the outcome scoping exists to prevent.

        the body is constrained by the JetStream KV grammar
        (``^[-/_=.a-zA-Z0-9]+$`` per ``nats-server`` ``kv.go``): a body
        carrying a colon ``:``, a space, or any other out-of-grammar
        character is rejected with ``nats: JetStream.InvalidKeyError``.
        so the body is checked against the grammar:

        - **grammar-safe** (the common case -- UUIDs use ``-``,
          integers/postgres oids are pure digits, slugs stay in
          ``[-_a-zA-Z0-9]``): the readable body is kept verbatim, so
          every existing pk in the codebase keeps its current,
          human-readable key (backward-compatible).
        - **out-of-grammar**: the body is replaced by a **SHA-256 hex
          digest** of it -- always grammar-valid (hex is in-grammar) and
          collision-resistant (distinct bodies map to distinct digests,
          unlike a naive ``:``->``=`` replace which silently collides).

        the ``{scope}.{table_name}.`` prefix is always grammar-safe (the
        scope is checked against the stricter scope grammar at wiring
        time; table names are ``[a-z_]`` identifiers) and is never
        hashed, so it stays a readable namespace. only the body is
        conditionally hashed. the raw pk continues to round-trip through
        L1, the stored value, and the invalidation envelope, so
        reversibility is not needed.

        deterministic: the same pk always yields the same key (the
        grammar check and the digest are both pure functions of the
        stringified pk values).

        **edge case**: if a *grammar-safe* pk value naturally contains
        ``"_"`` the composite form is ambiguous with a pk value that
        contains the resulting joined substring. callers that introduce
        underscore-bearing grammar-safe pk values MUST either escape
        them before passing or override this method.

        :param entity_id: pk value (single-pk) or tuple of pk values
            in declared order (composite-pk)
        :ptype entity_id: Any
        :return: grammar-safe nats KV key, scoped by principal and table name
        :rtype: str
        :raises L2ScopeNotConfiguredError: if this collection's registry
            carries no ``kv_key_scope``. the BACKSTOP raise, not the
            primary one: :meth:`CollectionRegistry.configure` and
            :meth:`CollectionRegistry.bind_table` both refuse an L2
            client with no scope at wiring time, which is where a
            process can still fail its startup rather than dying on the
            first cache access under load. this covers the ONE path
            neither of them sees -- ``nats_client=`` passed straight to
            the constructor, which wins over the registry default and
            never calls either. deliberately NOT a :class:`KvError` -- four
            of this method's five call sites sit inside ``except
            KvError`` handlers that degrade to a warning, so a
            ``KvError`` here would leave the fleet running with L2
            silently off
        """
        scope = self._registry.kv_key_scope
        if scope is None:
            raise L2ScopeNotConfiguredError(
                f"{self.table_name}: no kv_key_scope on this collection's registry, so its L2 "
                f"keys would carry no principal segment. wire it with "
                f"registry.configure(kv_key_scope=threetears.nats.kv_key_scope_for(...))"
            )
        pk_values = self.normalize_pk(entity_id)
        body = "_".join(str(v) for v in pk_values)
        if not _KV_KEY_GRAMMAR.match(body):
            body = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return f"{scope}.{self.table_name}.{body}"

    async def _get_from_l2(self, entity_id: Any) -> dict[str, Any] | None:
        """read entity payload from the L2 NATS KV bucket.

        narrow exception scope: only :class:`KvError` (real transport
        failure) degrades to ``None``. programming errors
        (``AttributeError``, ``TypeError``, etc.) propagate so wrapper
        API drift surfaces loudly instead of silently warning. bucket
        *resolution* (:meth:`_ensure_kv`) is covered by this same catch,
        not just the subsequent op -- an L2 outage during first-open
        (bucket never resolved yet, e.g. right after NATS drops) is
        exactly as much a transport failure as one during an already-open
        bucket's get/put/delete, and must degrade the same way.

        An absent-marker and an expired row both read as ``None`` here: this method answers "is
        there a live row", and neither is one.
        """
        lookup = await self._l2_lookup(entity_id)
        return lookup.row

    async def _l2_lookup(self, entity_id: Any) -> _L2Lookup:
        """read one L2 entry and classify it: live row, fresh absent-marker, or neither.

        Same narrow exception scope as :meth:`_get_from_l2`: a :class:`KvError` degrades to a
        plain miss, which sends the caller to L3.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: the classified entry
        :rtype: _L2Lookup
        """
        empty = _L2Lookup(row=None, marker=None, revision=None)
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return empty
            key = self.l2_key(entity_id)
            # the key's LATEST message, a deletion marker included: its revision is what lets a
            # replacement -- an absent-marker, or a row seeded from L3 -- land only if nothing has
            # happened to the key since this read, where a create would also land over a write
            # that was made and then evicted in between.
            raw, latest = await kv.get_latest(key=key)
        except KvError as exc:
            log.warning(
                "L2 cache read failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return empty
        revision = latest
        if raw is None:
            return _L2Lookup(row=None, marker=None, revision=revision)
        try:
            decoded = self._decode_l2_value(raw)
        except CorruptCacheEntry as exc:
            # A cache miss, not a failure. Returning None sends the caller to L3, which is
            # authoritative -- the same path a cold key takes. Failing the read instead would
            # let one poisoned key break a lookup that L3 could have answered, and returning
            # the row undecoded would hand back a string where the caller declared a datetime.
            log.warning(
                "L2 entry could not be decoded; falling through to L3",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "column": exc.column,
                    },
                },
            )
            # keep the revision: a negative-caching collection replaces the poisoned entry rather
            # than creating over it, which would fail and send every later read to L3 too.
            return _L2Lookup(row=None, marker=None, revision=revision)
        if isinstance(decoded, _AbsentMarker):
            return _L2Lookup(row=None, marker=decoded, revision=revision)
        if self._row_is_expired(decoded):
            return _L2Lookup(row=None, marker=None, revision=revision)
        return _L2Lookup(row=decoded, marker=None, revision=None)

    def _decode_l2_value(self, raw: bytes) -> dict[str, Any] | _AbsentMarker:
        """decode one L2 value of this collection: a row, or an absent-marker.

        The one decoder every read of a collection key goes through, so an absent-marker is never
        handed to a subclass :meth:`deserialize` that has no idea what it is.

        :param raw: the stored bytes
        :ptype raw: bytes
        :return: the rehydrated row, or the marker
        :rtype: dict[str, Any] | _AbsentMarker
        :raises CorruptCacheEntry: when a row, or a marker's generation, cannot be decoded
        """
        if raw.startswith(_ABSENT_MARKER_PREFIX):
            try:
                return _AbsentMarker(generation=raw[len(_ABSENT_MARKER_PREFIX) :].decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise CorruptCacheEntry(self.table_name, "absent-marker generation", raw) from exc
        return self._rehydrate_datetimes(self.deserialize(raw))

    async def _write_l2_marker(self, entity_id: Any, generation: str, revision: int | None) -> None:
        """record in L2 that ``entity_id`` is absent under ``generation``, never over a writer's value.

        Creates the marker when the key held nothing, or compare-and-swaps it over the stale
        marker, expired row or undecodable entry the lookup found; a value a writer put since then
        makes either write fail, and the writer's value stands. The entry carries a server-side
        lifetime of :attr:`negative_cache_max_age`, so markers nobody reads again leave the bucket.
        A failure here costs one more L3 read later, never correctness, so it degrades to a
        warning -- including the refusal of a bucket its declarer has not yet let carry lifetimes.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param generation: the generation read before the L3 lookup that found nothing
        :ptype generation: str
        :param revision: revision of the entry to replace, or ``None`` to create
        :ptype revision: int | None
        :return: None
        :rtype: None
        """
        marker = _ABSENT_MARKER_PREFIX + generation.encode("utf-8")
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return
            key = self.l2_key(entity_id)
            if revision is None:
                await kv.create(key=key, value=marker, ttl=self.negative_cache_max_age)
            else:
                await kv.update(key=key, value=marker, revision=revision, ttl=self.negative_cache_max_age)
        except KvError as exc:
            log.warning(
                "L2 absent-marker write failed; the next lookup will ask L3 again",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name, "error": str(exc)}},
            )

    async def _save_to_l2(self, entity_id: Any, data: dict[str, Any]) -> bool:
        """write entity payload to the L2 NATS KV bucket.

        narrow exception scope: only :class:`KvError` degrades to
        ``False``. programming errors propagate. bucket resolution
        (:meth:`_ensure_kv`) is covered by this same catch -- see
        :meth:`_get_from_l2`'s docstring for why.
        """
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return False
            key = self.l2_key(entity_id)
            value = self.serialize(self._normalise_datetimes_for_write(data))
            lifetime = self._l2_entry_lifetime(data)
            # the keyword is sent only when this table declares an expiry, so the call a
            # non-expiring collection makes is exactly the call it has always made -- a bucket
            # shim or a test double that predates per-entry lifetimes still satisfies it.
            if lifetime is None:
                await kv.put(key=key, value=value)
            else:
                await kv.put(key=key, value=value, ttl=lifetime)
        except KvError as exc:
            log.warning(
                "L2 cache write failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return False
        return True

    def _l2_entry_lifetime(self, row: dict[str, Any]) -> timedelta | None:
        """the server-side lifetime this row's L2 entry should carry, from its declared expiry.

        L1 drops an expired row when it reads it and the owning collection sweeps L3, which left
        L2 as the one tier with no reclamation: the shared collections bucket is opened with no
        expiry, so an expiring table's keys -- one per request for a claim, one per account or IP
        for a counter -- stayed resident in a memory-backed broker until it restarted. A row that
        declares when it expires can say so to the server, which is what this does.

        A row already past its expiry still gets the floor of one second (the finest the header
        can express) rather than no lifetime at all: it reads as absent everywhere either way, and
        the point is that it leaves.

        :param row: the row about to be written
        :ptype row: dict[str, Any]
        :return: the lifetime, or ``None`` when this table declares no expiry
        :rtype: timedelta | None
        """
        column = self.expires_at_column
        if column is None:
            return None
        expires_at = row.get(column)
        if not isinstance(expires_at, datetime):
            return None
        remaining = expires_at - datetime.now(UTC)
        return remaining if remaining >= timedelta(seconds=1) else timedelta(seconds=1)

    async def _delete_from_l2(self, entity_id: Any) -> bool:
        """delete entity payload from the L2 NATS KV bucket.

        narrow exception scope: only :class:`KvError` degrades to
        ``False``. programming errors propagate. bucket resolution
        (:meth:`_ensure_kv`) is covered by this same catch -- see
        :meth:`_get_from_l2`'s docstring for why.
        """
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return False
            return await kv.delete(key=self.l2_key(entity_id))
        except KvError as exc:
            log.warning(
                "L2 cache delete failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return False

    async def delete_l2_entry(self, entity_id: Any) -> bool:
        """evict this pod's OWN scoped L2 entry for one pk, when one exists.

        The receiver-side half of cross-pod cache coherence, called by
        :class:`CollectionRegistry`'s invalidation listener. Public because that caller lives in
        another class: reaching :meth:`_delete_from_l2` from there is a cross-class private
        access, which ``SLF`` forbids repo-wide and which the underscore contract says to fix by
        promoting the api rather than exempting the caller.

        **Presence-gated, and that gate is load-bearing.** A JetStream KV delete is a publish of
        a delete-marker message and happens UNCONDITIONALLY -- deleting an absent key still
        writes a marker. Ungated, every broadcast would write one marker per receiver for every
        entity that receiver never cached, into a memory-storage bucket with ``history=1``,
        unlimited ``max_age`` and no ``max_bytes``.

        Distinct from :meth:`_delete_from_l2`, which is the unconditional delete the write path
        uses: there the entity is being removed and a delete that raced a concurrent write must
        still land, so paying a probe first would open a window in which the racing value
        survives. Here the value is already known stale and skipping an absent key costs
        nothing.

        **Structurally a no-op when there is no L3**, on exactly the reasoning
        :attr:`l1_max_age_seconds` already applies to expiry: a tier that is the source of
        truth is not a cache, and evicting from it is not eviction, it is deletion. An
        L1+L2-only collection has nothing to pull through from, so a key this method removed
        is a row that no longer exists -- ``HeartbeatCollection``'s pod row, a presence room's
        membership, the identity fence's generation, which fails OPEN on a missing key and
        would admit the superseded connection it exists to refuse. The staleness this eviction
        exists to prevent cannot arise there either: with no L3 there is no ``_pull_through``
        re-caching anything, and a peer principal's copy is its own truth rather than a stale
        view of somebody else's.

        Same narrow exception scope as its siblings: only :class:`KvError` (real transport
        failure, including during bucket resolution) degrades to ``False``. A
        :class:`L2ScopeNotConfiguredError` from :meth:`l2_key` is NOT a transport failure and
        propagates.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
            (composite-pk)
        :ptype entity_id: Any
        :return: whether an entry was found and deleted
        :rtype: bool
        :raises L2ScopeNotConfiguredError: if this collection's registry carries no scope
        """
        if self.l3_pool is None:
            return False
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return False
            key = self.l2_key(entity_id)
            if await kv.get(key=key) is None:
                return False
            return await kv.delete(key=key)
        except KvError as exc:
            log.warning(
                "L2 cache eviction failed",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),
                        "table": self.table_name,
                        "error": str(exc),
                    },
                },
            )
            return False

    # --- Subscript access (sync, transparent pull-through) ---

    def _ensure_in_l1(self, entity_id: Any) -> dict[str, Any] | None:
        """Pull entity into L1 via L2/L3 if not already cached. Sync.

        Returns the row data if found, None if not found in any tier.
        """
        row = self._select_from_l1(entity_id, expiring=True)
        if row is not None:
            return row
        return sync_await(self._pull_through(entity_id))

    async def _pull_through(self, entity_id: Any) -> dict[str, Any] | None:
        """Async pull-through: L2 -> L1, then L3 -> L1+L2. Returns the data or None.

        An expired row is absent at every tier, L3's included. When negative caching is on, the
        table's write generation is read FIRST -- before L2 and before L3 -- and an absent-marker
        in L1 or L2 answers ``None`` only when stamped with that generation. A full miss is then
        recorded under it in both tiers. Because every committed write advances the generation, a
        marker recorded from an L3 read that predated a write carries a generation that write
        already moved past, and never answers again.

        The row is cached in L1 only while no write of the key began, and no eviction of it
        landed, in this process since the read started (:class:`_L1Fence`): either means the row
        read may already be older than what L2 or L3 holds.
        """
        with self._l1_fence.watching(self._fence_key(entity_id), writing=False) as ticket:
            found = await self._pull_through_watched(entity_id, ticket)
        return found

    async def _pull_through_watched(self, entity_id: Any, ticket: _KeyTicket) -> dict[str, Any] | None:
        """the pull-through itself, under the read's ticket on the key.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param ticket: the read's ticket, taken before its first await
        :ptype ticket: _KeyTicket
        :return: the row, or ``None`` when no tier holds a live one
        :rtype: dict[str, Any] | None
        """
        generation: str | None = None
        if self._negative_cache_active:
            generation = await self._current_generation()
            if generation is not None and self._l1_marker_matches(entity_id, generation):
                log.debug(
                    "absence served from an L1 marker",
                    extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name}},
                )
                return None
        lookup = await self._l2_lookup(entity_id)
        if lookup.row is not None:
            if self._l1 is not None and self._l1_fence.still_newest(ticket):
                self._l1.upsert(self.table_name, self._stamped(lookup.row), self.primary_key_columns)
            return lookup.row
        if generation is not None and lookup.marker is not None and lookup.marker.generation == generation:
            self._write_l1_marker(entity_id, generation)
            log.debug(
                "absence served from an L2 marker",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name}},
            )
            return None
        pg_data = await self.fetch_from_store(entity_id)
        if pg_data is not None and self._row_is_expired(pg_data):
            pg_data = None
        if pg_data is not None:
            pg_data = await self._seed_l2_from_store(entity_id, pg_data, lookup.revision, ticket)
        elif generation is not None:
            self._write_l1_marker(entity_id, generation)
            await self._write_l2_marker(entity_id, generation, lookup.revision)
        return pg_data

    async def _seed_l2(self, entity_id: Any, stored: dict[str, Any], revision: int) -> bool:
        """put a row read from L3 into L2 only if nothing has happened to the key since it was read.

        The one way a read path moves a row from L3 into L2. A row read from L3 is only as new as
        the moment the query ran. A writer whose ``save_entity`` committed and put its row into L2
        after that moment holds a NEWER value, and an unconditional put from the read would land
        the older row over it: every reader on every replica is then served the older value until
        the next write or the entry's lifetime -- the write was correct and every reader wrong. On
        a collection whose rows :meth:`l2_cas_mutate` orders it is worse: the next swap builds on
        the older row and persists over the newer one.

        A create-if-absent is not enough. The save's broadcast makes every peer in its scope delete
        the key it just wrote, so the key can be empty again by the time the read seeds, and a
        create lands there. So the seed is written at ``revision`` -- the revision of the key's
        latest message, a deletion marker included, read BEFORE the L3 query
        (:meth:`~threetears.nats.NatsKvBucket.get_latest`) -- and lands only while the key's
        history is exactly as the read found it. Any write or deletion since refuses it.

        A transport failure writes nothing, so nothing is out of order; it degrades to a warning
        as every L2 write on a three-tier read path does.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param stored: the live row read from L3
        :ptype stored: dict[str, Any]
        :param revision: the key's latest revision as read before the L3 query; ``0`` when the key
            had no message at all
        :ptype revision: int
        :return: whether the key changed since it was read, so this row was not written
        :rtype: bool
        """
        superseded = False
        try:
            kv = await self._ensure_kv()
            if kv is not None:
                key = self.l2_key(entity_id)
                payload = self.serialize(self._normalise_datetimes_for_write(stored))
                lifetime = self._l2_entry_lifetime(stored)
                timed = {} if lifetime is None else {"ttl": lifetime}
                superseded = await kv.update(key=key, value=payload, revision=revision, **timed) is None
        except KvError as exc:
            log.warning(
                "L2 seed from L3 failed; the next read asks L3 again",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name, "error": str(exc)}},
            )
        if superseded:
            log.debug(
                "L2 seed from L3 lost to a write made since the read; the newer state stands",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name}},
            )
        return superseded

    async def _seed_l2_from_store(
        self, entity_id: Any, stored: dict[str, Any], revision: int | None, ticket: _KeyTicket
    ) -> dict[str, Any]:
        """seed L2 and L1 with a row a pull-through read from L3, answering with whatever is newest.

        :meth:`_seed_l2` decides whether the row reaches L2. When the key changed since the read
        and holds a live value, that value is newer than anything L3 returned, so it is what this
        read answers and what L1 keeps; answering with the L3 row would hand this caller the value
        every other reader has already moved past.

        When the key changed and holds no live value, a writer deleted it -- a save whose own L2
        write was refused, or an invalidation -- so the L3 row may predate that write. It is
        answered, as the value at the moment of the read, and not cached anywhere.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param stored: the live row read from L3
        :ptype stored: dict[str, Any]
        :param revision: the key's latest revision as the lookup found it, or ``None`` when L2
            could not be read -- in which case L2 is left alone
        :ptype revision: int | None
        :param ticket: the read's ticket on the key; L1 takes the row only while it is still newest
        :ptype ticket: _KeyTicket
        :return: the row this read answers with -- the seeded one, or the newer one that beat it
        :rtype: dict[str, Any]
        """
        answer = stored
        cacheable = True
        if revision is not None and await self._seed_l2(entity_id, stored, revision):
            current = await self._l2_lookup(entity_id)
            if current.row is not None:
                answer = current.row
            else:
                cacheable = False
        if cacheable and self._l1 is not None and self._l1_fence.still_newest(ticket):
            self._l1.upsert(self.table_name, self._stamped(answer), self.primary_key_columns)
        return answer

    @staticmethod
    def _stamped(data: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``data`` carrying the cache-age stamp for this instant.

        Stamped wherever a row arrives from a LOWER tier -- both pull-through
        sites and :meth:`reload_entity` -- and nowhere else. The stamp records
        provenance, not local activity: stamping on every write would let
        ``set_field_sync`` renew a stale row's lifetime by editing one field,
        which makes exactly the rows this bounds immortal.

        A copy, because the caller's dict is returned to the caller and becomes
        entity data. The stamp is storage bookkeeping and must not ride along.
        """
        return {**data, _CACHED_AT_COLUMN: time.monotonic()}

    def _resolve_row(self, entity_id: Any) -> dict[str, Any]:
        """Get row from L1, pulling through L2/L3 on miss. Raises KeyError if not found.

        The first read expires, and goes to :meth:`_select_from_l1` directly rather
        than through :meth:`get_row_sync`. This method repairs a miss by pulling
        through, which is what earns it the bound; ``get_row_sync`` is a reporting
        read and does not expire, so borrowing it here would return the stale row
        and return it forever -- ``collection[id]`` never reaches the pull-through
        below, whatever max age the collection configured, while ``get()`` beside
        it refreshes. The re-read after the pull-through does NOT expire: it was
        just written.
        """
        row = self._select_from_l1(entity_id, expiring=True)
        if row is not None:
            return row
        data = self._ensure_in_l1(entity_id)
        if data is None:
            raise KeyError(f"{self.table_name}[{entity_id!r}]: entity not found")
        # If L1 exists, re-read from it (ensure_in_l1 populated it)
        if self._l1 is not None:
            row = self.get_row_sync(entity_id)
            if row is not None:
                return row
        # No L1 — return the data directly from pull-through
        return data

    def __getitem__(self, key: Any) -> Any:
        """Subscript read with transparent three-tier pull-through.

        collection[entity_id]          -> EntityT
        collection[entity_id, "field"] -> field value

        On L1 miss, transparently pulls data through L2/L3 into L1
        via a background event loop. Raises KeyError only if the entity
        doesn't exist in any tier.
        """
        if isinstance(key, tuple):
            entity_id, field = key
            # Straight to the repairing read, not through ``get_field_sync`` first.
            # That is a reporting read and does not expire, so a stale row still
            # holding the field answered here and the bound never applied to
            # ``collection[id, "field"]``. Resolving first is also one L1 read
            # rather than two, and the outcomes are otherwise identical -- a row
            # without the field raises below either way.
            row = self._resolve_row(entity_id)
            result = row.get(field, MISSING)
            if result is MISSING:
                raise KeyError(f"{self.table_name}[{entity_id!r}, {field!r}]: field not found")
            return result
        entity_id = key
        row = self._resolve_row(entity_id)
        entity = self._entity_from_read(entity_id, row)
        entity.original_date_updated = row.get("date_updated")
        return entity

    def _entity_from_read(self, entity_id: Any, row: dict[str, Any]) -> EntityT:
        """build the entity a read answers with, from L1's copy when the read cached one.

        A loaded entity holds its own row and writes no tier (:class:`BaseEntity`), so the read
        alone decides what L1 holds: when it declined to cache (:class:`_L1Fence`: a write or
        eviction of the key overlapped it), L1 is left without the row. When L1 holds the row
        after the read, the entity is built from that copy.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param row: the row the read returned
        :ptype row: dict[str, Any]
        :return: the entity
        :rtype: EntityT
        """
        cached = self._select_from_l1(entity_id)
        entity: EntityT = self.entity_class(cached if cached is not None else row, is_new=False, collection=self)
        return entity

    def __setitem__(self, key: Any, value: Any) -> None:
        """Subscript write with three-tier propagation.

        collection[entity_id] = data_dict       -> write full entity
        collection[entity_id, "field"] = value   -> write single field

        Writes to L1 synchronously. The rest is non-blocking (fire-and-forget
        on the background event loop) and follows :meth:`save_entity`'s
        contract: L3 first, then L2 as a compare-and-swap at the revision read
        before the L3 write, L1 kept only when L2 took the row, then the
        broadcast (:meth:`_async_propagate_write`). A collection that defers
        its L3 writes writes L2 first and buffers the L3 write for a later
        flush.

        A row the database completes (:meth:`columns_decided_by_store`) is read back from L3 once
        the write commits, and that is what L2 and L1 keep. On a write-behind collection there is
        nothing to read back before L2 takes the row, so such an assignment raises ``ValueError``
        before L1 takes it.

        Refused on a collection that caches absences: the fire-and-forget write has no caller to
        tell when its write generation failed to advance, and an unadvanced generation keeps an
        absence recorded before the write answering. Use :meth:`save_entity`.
        """
        if self._negative_cache_writes_advance:
            raise TypeError(
                f"{type(self).__name__} caches absences; subscript writes cannot report a write "
                f"generation they failed to advance. use save_entity()"
            )
        if isinstance(key, tuple):
            entity_id, field = key
            current = self.get_row_sync(entity_id)
            if current is not None and self._defers_l3_writes:
                self._refuse_store_decided_assignment({**current, field: value})
            self.set_field_sync(entity_id, field, value)
            row = self.get_row_sync(entity_id)
            if row is not None:
                self._propagate_write(entity_id, row)
        else:
            entity_id = key
            if not isinstance(value, dict):
                raise TypeError(f"collection[id] = value requires a dict, got {type(value).__name__}")
            if self._defers_l3_writes:
                self._refuse_store_decided_assignment(value)
            self.write_to_cache_sync(value)
            self._propagate_write(entity_id, value)

    def _refuse_store_decided_assignment(self, row: dict[str, Any]) -> None:
        """refuse a write-behind assignment whose row, as propagation sends it, the database completes.

        Checked before L1 takes the row, on the row :meth:`_async_propagate_write` will send:
        stamped with ``date_updated``, and carrying no order where this collection persists one.
        The propagation completes the row it caches itself (:meth:`_write_ahead_row`).

        :param row: the row the assignment writes
        :ptype row: dict[str, Any]
        :return: nothing
        :rtype: None
        :raises ValueError: when the database would decide any column of it
        """
        sent = {**row, "date_updated": datetime.now(UTC)}
        self._write_ahead_row(without_l2_order(sent) if self.persists_l2_order else sent)

    def _propagate_write(self, entity_id: Any, data: dict[str, Any]) -> None:
        """Non-blocking propagation of a write to L3, L2 and the broadcast (:meth:`_async_propagate_write`)."""
        fire_and_forget(self._async_propagate_write(entity_id, dict(data)))

    async def _async_propagate_write(self, entity_id: Any, data: dict[str, Any]) -> None:
        """carry a subscript write to L3, L2 and the broadcast, under the same contract as :meth:`save_entity`.

        L3 first, unless the collection defers its L3 writes; then L2 as a compare-and-swap at the
        revision read before the L3 write; L1 keeps the row only when L2 took it and no other write
        of the key overlapped this one; then the broadcast. Writing L2 before L3, as this once did,
        let two writers leave L2 holding one row and L3 the other, with nothing left to evict it.

        A write-behind collection writes L1 and L2 first and buffers the L3 write, as
        :meth:`save_entity` does there: its L2 is ahead of L3 by design.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param data: the row, a copy the caller no longer holds
        :ptype data: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        now = datetime.now(UTC)
        data["date_updated"] = now
        if self.persists_l2_order:
            # as in save_entity: a write that won no compare-and-swap stores no order.
            data.update(without_l2_order(data))

        if self._defers_l3_writes:
            assert self._write_buffer is not None
            # refused at the assignment when the database would decide any of it; completed here
            data = self._write_ahead_row(data)
            self._l1_fence.changed(self._fence_key(entity_id))
            if self._l1 is not None:
                self._l1.upsert(self.table_name, data, self.primary_key_columns)
            await self._save_to_l2(entity_id, data)
            await self._publish_invalidation(entity_id)
            await self._write_buffer.add(self.table_name, entity_id, data)
            return

        with self._l1_fence.watching(self._fence_key(entity_id), writing=True) as ticket:
            before = await self._l2_revision_before_write(entity_id)
            try:
                rows_affected = await self.save_to_store(data)
            # BaseException, not Exception: a cancelled write's outcome is as unknown as a failed
            # one's, and CancelledError is not an Exception. Only an Exception is swallowed: this
            # path is fire-and-forget, so there is no caller to hand it to.
            except BaseException as exc:
                log.error(
                    "Background L3 write failed; withdrawing it from L1 and L2",
                    extra={
                        "extra_data": {
                            "entity_id": str(entity_id),
                            "table": self.table_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    },
                )
                # L1 took this row synchronously at the assignment; withdraw it from every tier
                # so every reader goes to L3.
                await self.invalidate_cache(entity_id)
                if not isinstance(exc, Exception):
                    raise
                return
            if rows_affected == 0:
                # the store kept a different row than the one L1 holds -- a lost race on a fenced
                # table, the DO NOTHING outcome on any other. either way L1 disagrees with L3
                # until withdrawn.
                await self.invalidate_cache(entity_id)
                # This path is fire-and-forget: there is no caller left to hand a rowcount back
                # to, and no exception is raised when a CAS fence rejects the write. On an
                # unconditionally fenced collection a 0 here is a LOST WRITE -- the caller's value
                # never reached L3 -- so say so. Left silent for unfenced collections, where 0 is
                # the ordinary "DO NOTHING matched" outcome.
                if self.emits_cas_fence:
                    log.error(
                        "Background L3 write lost its CAS race and was dropped; "
                        "the sync item-assignment write path cannot retry -- use "
                        "save_entity() with a retry loop on a cas_null_safe table",
                        extra={
                            "extra_data": {
                                "entity_id": str(entity_id),
                                "table": self.table_name,
                            }
                        },
                    )
                return
            # the database may have filled in columns the assignment did not name; what is cached
            # is the row L3 holds, never the row as sent.
            stored = await self._stored_row(entity_id, data)
            if stored is None:
                await self.invalidate_cache(entity_id)
                return
            await self._cache_committed_row(entity_id, stored, before, ticket)

        # Signal other pods to evict stale L1
        await self._publish_invalidation(entity_id)

    def __contains__(self, entity_id: Any) -> bool:
        """Check if entity is in L1 cache."""
        return self.exists_in_cache_sync(entity_id)

    # --- Cache coherence signaling ---

    async def aclose(self) -> None:
        """stop whatever background work this collection started. A no-op by default.

        The teardown seam :meth:`CollectionRegistry.close_collections` calls on every registered
        collection. Declared here rather than discovered with ``getattr`` so a subclass's typo or
        changed signature fails as a type error rather than as a logged teardown failure, and so
        an unrelated ``aclose`` on somebody else's object is never called at registry teardown.

        Most collections start nothing and inherit this. A collection that does -- a coordination
        table's periodic flusher -- overrides it and owes its last flush here.

        :return: nothing
        :rtype: None
        """
        return None

    async def _publish_invalidation(self, entity_id: Any, *, l2_key_current: bool = False) -> None:
        """Signal other pods to evict this entity from their L1 caches.

        ``l2_key_current`` is for revision-fenced writes only: peers sharing this registry's L2
        scope then keep the key rather than evicting it (see
        :attr:`CacheInvalidationMessage.l2_current_scope`).

        datasource-task-06 DS-06-04: when ``nats_client`` is missing
        the publish is a no-op -- consumer pods serve stale L1
        entries indefinitely. log a one-shot WARNING the first
        time this happens for a given table so the wiring gap is
        visible in the operator's log. the warning is per-table to
        keep the log scannable across a fleet with many collections.

        design choice: WARNING vs hard-fail-at-construction.
        the spec (DS-06-04) accepted either. WARNING is the
        chosen shape because:

        - the underlying wiring gap (Hub's datasource collections
          constructed without ``nats_client``) is FIXED in
          ``3tears.hub.app.lifespan``; the WARNING is the
          regression net, not the primary fix.
        - several legitimate test scaffolds construct collections
          without a NATS client (SQLite-only L1 tests in
          ``threetears.core``). hard-failing at construction would
          force every such test to stand up a NATS mock for a
          feature it does not exercise.
        - the WARNING fires at the first attempted publish, NOT at
          construction, so collections that never publish (read-
          only utility flows) stay quiet.
        - one-shot dedup means a busy write path doesn't drown
          the log.
        """
        if self._nats_client is None:
            self._warn_missing_nats_client_once()
        await self._registry.publish_invalidation(
            self._nats_client,
            self.table_name,
            entity_id,
            l2_key_current=l2_key_current,
        )

    def _warn_missing_nats_client_once(self) -> None:
        """emit a one-shot WARNING when invalidation cannot publish.

        guards against log spam under workloads that issue many
        writes per second -- the wiring gap is process-wide and
        worth surfacing once. uses a class-level set keyed on
        table_name so collections of distinct types each get one
        warning.

        :return: nothing
        :rtype: None
        """
        cls = type(self)
        warned: set[str] = getattr(cls, "_missing_nats_warned_tables", None) or set()
        if self.table_name in warned:
            return
        warned.add(self.table_name)
        cls._missing_nats_warned_tables = warned
        log.warning(
            "collection invalidation is silently disabled: table=%s -- "
            "consumer pods will serve stale L1 entries until the "
            "collection is reconstructed with a non-None nats_client. "
            "wiring gap: datasource-task-06 DS-06-04.",
            self.table_name,
        )

    # --- Span attribute helpers (no-op when OTel unavailable) ---

    @staticmethod
    def _set_span_attr(key: str, value: Any) -> None:
        """Set an attribute on the current OTel span, if available."""
        try:
            from opentelemetry import trace as _trace

            span = _trace.get_current_span()
            span.set_attribute(key, value)
        # NOSILENT: optional dependency probe; absence is a supported configuration
        except ImportError:
            pass

    def _set_span_table(self) -> None:
        """Set ``cache.table`` on the current span."""
        self._set_span_attr("cache.table", self.table_name)

    # --- Three-tier operations ---

    async def ensure(self, entity_id: Any) -> dict[str, Any] | None:
        """pull entity into L1 cache through L2/L3 if not already present.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: entity data dict if found in any tier, ``None`` if not
            found anywhere. after ``ensure()`` returns data, subscript
            access is guaranteed to hit L1 (when L1 is available).
        :rtype: dict[str, Any] | None
        """
        row = self._select_from_l1(entity_id, expiring=True)
        if row is not None:
            return row
        data = await self._pull_through(entity_id)
        return data

    @traced()
    async def get(self, entity_id: Any) -> EntityT | None:
        """three-tier read: L1 -> L2 -> L3, promote on miss.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: entity instance on hit in any tier, ``None`` on
            total-miss
        :rtype: EntityT | None
        """
        self._set_span_table()
        data = await self.ensure(entity_id)
        if data is None:
            self._set_span_attr("cache.hit_tier", "miss")
            return None
        self._set_span_attr("cache.hit_tier", "L1+")
        return self._entity_from_read(entity_id, data)

    @traced()
    async def save_entity(
        self,
        entity: BaseEntity,
        *,
        conn: Any = None,
    ) -> None:
        """save entity through the three-tier write path.

        **Cached only while still the newest write.** On a collection with an L3 pool
        and an L2 bucket, the key's latest L2 revision is read before the L3 write, and the
        committed row is written to L2 as a compare-and-swap at that revision rather than as an
        unconditional put. L3 is a round trip, and a later save of the same row -- this
        replica's own, or a peer's -- can complete inside it; when it has, the swap is refused,
        the key is deleted from L2 and from this replica's L1, and the next read takes whichever
        row L3 committed last. L1 caches the row only when L2 took it. The ordering needs no
        order columns: the L2 revision read before the write is the fence, on every collection.
        The save itself never fails for it; an unreadable L2 caches nothing and the save still
        succeeds.

        **The handle holds what it saved.** Whether or not any tier took the row, the saved entity
        holds the row as stored (:meth:`BaseEntity.hold_row`) and answers from it, never from L1's
        copy of the key: that copy is a cache a caller's transaction settling the key, a peer's
        broadcast or expiry may drop at any time, and a later write may replace.

        **Cached only while no other write of the key overlapped it, in this process.** L1 takes
        the row only when no other save of the same key was in flight while this one was, and no
        eviction of the key landed meanwhile. Overlapping saves can reach L3 in an order neither
        sees; both drop the key from L1 and the next read takes whichever row L3 kept. On a
        collection with an L3 pool and no L2 this is the only ordering there is.

        **Joining a caller's transaction caches nothing until it ends.** With ``conn``, the row is
        written to L3 inside the caller's transaction and is not final when this returns, so no
        tier takes it and nothing is broadcast: the entity's working copy leaves L1 for its own
        change buffer. The connection's transaction must have been opened by
        :class:`~threetears.core.collections.caller_transaction.CallerTransaction`, which evicts the
        key from L1 and L2 and broadcasts the eviction once the transaction has committed or
        rolled back; the next read takes whichever row L3 ended with.

        **Cached as L3 holds it.** A row that leaves columns for the database to decide -- a
        server default for a column it does not name, the stored value an update keeps
        (:meth:`columns_decided_by_store`) -- is read back from L3 once the write commits, and the
        read is what L2, L1 and the handle keep. A read back that fails caches nothing; the save
        still succeeds and the next read goes to L3. A write-behind collection, whose L2 takes the
        row before L3 does, refuses such a row with ``ValueError`` before any tier takes it.

        Unchanged elsewhere: a collection with no L3 pool (L2 is its source of truth) and a
        write-behind collection keep the unconditional put, and a collection with no L2 caches in
        L1 as before, subject to the in-process ordering above.

        :param entity: entity instance to persist
        :ptype entity: BaseEntity
        :param conn: optional **backend-specific connection** (e.g. an asyncpg
            connection for the SQL backend) whose transaction the L3 write
            joins, in place of :attr:`l3_pool`, so it commits or rolls back
            with whatever else the caller issued on it. the transaction must
            be open through ``CallerTransaction(conn)``. ``None`` lets the
            collection's own L3 store service the write. refused on a
            collection that caches absences, whose write generation must
            advance after the commit, and on one that defers its L3 writes,
            whose row would reach L3 outside the transaction
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: on optimistic-lock fence
            mismatch when the entity carries an
            ``original_date_updated`` value
        :raises ValueError: when ``conn`` is passed to a collection that caches absences or defers
            its L3 writes, or its transaction was not opened by ``CallerTransaction``; and when a
            collection that defers its L3 writes is given a row the database would complete
        :raises GenerationUnavailableError: when a collection that caches absences committed the
            write but could not advance its write generation; retry the save
        """
        self._set_span_table()
        data = entity.to_dict()
        # The key is derived from the PAYLOAD, not from the entity's
        # construction-time ``addressing_id``. Those agree whenever
        # ``to_dict()`` reads the L1 row (that row is fetched BY the
        # construction-time key), but diverge when ``to_dict()`` falls
        # back to ``_changes`` -- no L1 backend wired, or the L1 row
        # evicted -- and a pk column has since been written. Keying L2
        # by the stale value while L3 takes the payload caches one
        # partition's row under another's key, which is the exact
        # cross-partition bleed the composite key exists to prevent.
        # ``strict`` makes a payload missing a pk column raise here
        # rather than silently addressing the scalar.
        entity_id: Any = derive_addressing_id(entity.id, data, self, strict=True)
        original_timestamp = getattr(entity, "original_date_updated", None)
        # the entity's working copy as it stood before this save stamped it: what the handle
        # keeps if the L3 write does not land (see ``_withdraw_unstored``).
        working = dict(data)
        if self.persists_l2_order:
            # only a won compare-and-swap stores an order. A save that won none must not carry
            # one it copied from the row it read: at its own L3 write it would pose as that swap,
            # and a buffered flush of it would be refused as the row it already is.
            data = without_l2_order(data)

        now = datetime.now(UTC)
        if entity.is_new:
            data["date_created"] = now
        if "date_updated" in data or not entity.is_new:
            data["date_updated"] = now

        # Datetime values flow through aware-UTC end to end. Per-column
        # coercion at the L3 backend border is the responsibility of
        # subclasses (SchemaBackedCollection drives this from declared
        # column types). collections-task-05 eliminated DATETIME_TYPE /
        # TIMESTAMP from the platform; the unconditional tzinfo strip
        # that used to live here was load-bearing only while TIMESTAMP
        # columns existed and is gone with them.

        defer = self._defers_l3_writes
        if conn is not None and self._negative_cache_writes_advance:
            # the write joins a transaction the caller commits later; the generation would advance
            # before the row is visible, and a reader in between would record it absent under the
            # new generation, where no later advance reaches it.
            raise ValueError(
                f"{type(self).__name__} caches absences and cannot join a caller's transaction: its "
                f"write generation must advance after the commit, which only the collection's own "
                f"write can guarantee"
            )
        if conn is not None and defer:
            raise ValueError(
                f"{type(self).__name__} defers its L3 writes to a write buffer, so a save cannot join a "
                f"caller's transaction: the row would reach L3 at the next flush, outside it"
            )
        caller_transaction: CallerTransaction | None = None
        if conn is not None:
            caller_transaction = CallerTransaction.join(conn, writer=f"{type(self).__name__}.save_entity")
        generation_failure: GenerationUnavailableError | None = None

        if defer:
            # the row is visible in L1 and L2 before L3 by design: L2 is ahead of L3 for up to one
            # flush interval, so it must already be the row L3 will hold. A read of the key in
            # flight read it before this write.
            try:
                data = self._write_ahead_row(data)
            except ValueError:
                self._withdraw_unstored(entity, entity_id, working)
                raise
            self._l1_fence.changed(self._fence_key(entity_id))
            if self._l1 is not None:
                self._l1.upsert(self.table_name, data, self.primary_key_columns)
            await self._save_to_l2(entity_id, data)
            assert self._write_buffer is not None
            await self._write_buffer.add(self.table_name, entity_id, data)
            entity.mark_clean()
            entity.original_date_updated = data.get("date_updated")
            # a saved handle answers from the row it saved, never from L1's copy of the key, which
            # any eviction may drop and any later write may replace.
            entity.hold_row(data)
        elif caller_transaction is not None:
            # enrolled before the write, so a write whose outcome is unknown -- it raised, but the
            # caller may still commit what reached L3 -- is settled with the rest.
            caller_transaction.enroll(self, entity_id)
            await self._store_uncommitted(entity, entity_id, data, original_timestamp, working, conn)
            # nothing is cached and nothing is broadcast until the caller's transaction ends
            # (CallerTransaction settles every key its writes touched).
            return
        else:
            with self._l1_fence.watching(self._fence_key(entity_id), writing=True) as ticket:
                # L2's state BEFORE the L3 write: the row reaches L2 below only if nothing has
                # touched the key since (see _cache_committed_row).
                before = await self._l2_revision_before_write(entity_id)
                try:
                    rows_affected = await self.save_to_store(data, original_timestamp)
                # BaseException, not Exception: a cancellation mid-write leaves the outcome as
                # unknown as any failure does, and CancelledError is not an Exception.
                except BaseException:
                    self._withdraw_unstored(entity, entity_id, working)
                    raise
                if rows_affected == 0:
                    self._withdraw_unstored(entity, entity_id, working)
                    if entity.is_new:
                        raise RuntimeError(
                            f"L3 insert failed for {self.table_name} entity {entity_id}: 0 rows affected"
                        )
                    raise ConcurrentModificationError(self.table_name, entity_id, original_timestamp or datetime.min)
                # the row is committed: advance the generation before anything else, so an absence
                # a reader recorded from an L3 read that predated this commit stops answering as
                # soon as possible. a failure is raised only once L1, L2 and the broadcast have run.
                generation_failure = await self._advance_generation()
                # the database may have filled in columns the row did not name; what every tier
                # and the handle keep is the row L3 holds, never the row as sent.
                stored = await self._stored_row(entity_id, data)
                entity.mark_clean()
                if stored is None:
                    # committed, but not readable back: nothing caches it, and the handle keeps
                    # what it sent until it is reloaded.
                    entity.original_date_updated = data.get("date_updated")
                    await self._drop_unread_row(entity_id)
                    entity.hold_row(data)
                else:
                    entity.original_date_updated = stored.get("date_updated")
                    await self._cache_committed_row(entity_id, stored, before, ticket)
                    # a key the stored row does not carry is one this table does not hold: the
                    # caller's own, which a read back cannot speak to. the handle keeps it; the
                    # tiers do not, since they hold what an L3 read gives.
                    carried = {key: value for key, value in data.items() if key not in stored}
                    # the handle holds the row as stored whether or not L1 took it: L1's copy of
                    # the key is a cache any eviction may drop (a caller's transaction settling
                    # the key, a peer's broadcast, expiry) and any later write may replace, and
                    # a handle reading through it would answer None, or another version.
                    entity.hold_row({**carried, **stored})

        await self._publish_invalidation(entity_id)
        if generation_failure is not None:
            raise generation_failure

    async def _store_uncommitted(
        self,
        entity: BaseEntity,
        entity_id: Any,
        data: dict[str, Any],
        original_timestamp: datetime | None,
        working: dict[str, Any],
        conn: Any,
    ) -> None:
        """write a row to L3 inside a caller's transaction, leaving no tier holding it.

        The row is not final until the caller's transaction ends, so no cache may take it: the
        entity's working copy leaves L1 for the entity's own change buffer, and nothing reaches L2
        or the broadcast. The :class:`CallerTransaction` evicts the key from every tier once the
        transaction has ended.

        :param entity: the entity being saved
        :ptype entity: BaseEntity
        :param entity_id: the key its row is cached under
        :ptype entity_id: Any
        :param data: the row, stamped
        :ptype data: dict[str, Any]
        :param original_timestamp: the optimistic-lock fence, or ``None`` for an insert
        :ptype original_timestamp: datetime | None
        :param working: the entity's data as it stood before the save stamped it
        :ptype working: dict[str, Any]
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ConcurrentModificationError: on an optimistic-lock fence mismatch
        :raises RuntimeError: when an insert affects no row
        """
        try:
            rows_affected = await self.save_to_store(data, original_timestamp, conn=conn)
        # BaseException, not Exception: CancelledError is not an Exception, and a cancelled write
        # leaves the outcome as unknown as any failure does.
        except BaseException:
            self._withdraw_unstored(entity, entity_id, working)
            raise
        if rows_affected == 0:
            self._withdraw_unstored(entity, entity_id, working)
            if entity.is_new:
                raise RuntimeError(f"L3 insert failed for {self.table_name} entity {entity_id}: 0 rows affected")
            raise ConcurrentModificationError(self.table_name, entity_id, original_timestamp or datetime.min)
        entity.mark_clean()
        entity.original_date_updated = data.get("date_updated")
        self._evict_l1(entity_id)
        entity.hold_row(data)

    def _withdraw_unstored(self, entity: BaseEntity, entity_id: Any, working: dict[str, Any]) -> None:
        """take an entity's working copy out of L1 after its L3 write did not land.

        an entity is a proxy onto its L1 row: construction writes the row and every attribute
        set writes through, so by the time :meth:`save_entity` reaches L3 the working copy is
        already this pod's cached answer for that key. when the write is refused (a lost CAS
        race, an insert that found the row taken) or fails, that copy is state L3 never took.
        left in L1 it is served as stored: a writer that lost retries through :meth:`ensure`,
        finds its own change "present", and stops without it ever reaching L3.

        the row is evicted rather than repaired: this pod cannot know the stored row without
        reading it, and a miss is exactly that read, taken by the next reader. nothing else is
        touched. L2 and the peers' L1 never held the working copy -- only a write that landed
        publishes to L2 or broadcasts -- and the winner's own write already invalidated them.

        the working copy moves into the entity's own change buffer, so the caller's handle
        still reads what it was trying to save, and a retry through the same handle writes it.

        :param entity: the entity whose save did not land
        :ptype entity: BaseEntity
        :param entity_id: the key its row is cached under
        :ptype entity_id: Any
        :param working: its data as it stood before the save stamped it
        :ptype working: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._evict_l1(entity_id)
        entity.hold_row(working)
        log.info(
            "L3 write did not land; its working copy was withdrawn from L1",
            extra={"extra_data": {"table": self.table_name, "entity_id": str(entity_id)}},
        )

    async def _l2_revision_before_write(self, entity_id: Any) -> _L2BeforeWrite:
        """read the revision of the key's latest L2 message before a save writes L3.

        The fence :meth:`_cache_committed_row` writes L2 at. Only a collection with an L3 pool and
        an L2 bucket is fenced: without L3, L2 is the source of truth and a save's put is
        last-writer-wins by contract; without L2 there is nothing to write.

        A failed read leaves the save fenced with no revision, so the committed row is never
        cached: the save itself still succeeds, since L3 is the durable record and a miss is
        always correct. It degrades to a warning as every L2 access on a three-tier path does.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: whether the save's L2 write is fenced, and the revision to fence it at
        :rtype: _L2BeforeWrite
        """
        before = _L2BeforeWrite(fenced=False, revision=None)
        if self.l3_pool is not None:
            try:
                kv = await self._ensure_kv()
                if kv is not None:
                    _, revision = await kv.get_latest(key=self.l2_key(entity_id))
                    before = _L2BeforeWrite(fenced=True, revision=revision)
            except KvError as exc:
                before = _L2BeforeWrite(fenced=True, revision=None)
                log.warning(
                    "L2 read before a save failed; the saved row is not cached and the next read goes to L3",
                    extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name, "error": str(exc)}},
                )
        return before

    async def _cache_committed_row(
        self, entity_id: Any, data: dict[str, Any], before: _L2BeforeWrite, ticket: _KeyTicket
    ) -> None:
        """cache a row whose L3 write committed, in L2 and L1, only while no later write has touched the key.

        The L3 write is a round trip, and a later save of the same row can complete inside it:
        this replica's own, or a peer's, whose broadcast has already evicted this replica's L1 and
        deleted the shared key. An unconditional put of this row after that would leave it in L2,
        and in L1, behind L3 with nothing left to evict it.

        So on a fenced collection the row is written to L2 at ``before.revision`` -- the key's
        latest message as read before the L3 write -- and lands only while the key's history is
        exactly as that read found it. A write that lands was the newest when it landed: any save
        that committed after this one read the key after this one did, so it either finds this
        write and replaces it, or finds the key changed and evicts it. A refused write does not
        know whether the key now holds a newer row or an older one, so it deletes the key and
        every reader goes to L3, which holds whichever write committed last.

        L1 takes the row only when L2 took it (or there is no fenced L2), and only while this
        process's own fence still names this write the newest (:class:`_L1Fence`): no other write
        of the key overlapped it and no eviction landed since it began. That second check is the
        whole fence for a collection with an L3 pool and no L2, whose overlapping saves reach L3
        in an order neither can see. The check and the L1 write run with no await between them;
        otherwise the key is dropped from L1.

        A collection with no L3 pool is left as it was: its L1 or L2 is the source of truth, so
        there is nothing to drop the row back to, and its writes are last-writer-wins -- L1 takes
        the row, then L2 takes it with an unconditional put.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param data: the row as committed
        :ptype data: dict[str, Any]
        :param before: L2's state for the key as read before the L3 write
        :ptype before: _L2BeforeWrite
        :param ticket: the write's ticket on the key, taken before its first await
        :ptype ticket: _KeyTicket
        :return: nothing
        :rtype: None
        """
        self._clear_l1_marker(entity_id)
        landed = True
        if before.fenced:
            landed = await self._write_l2_at(entity_id, data, before.revision)
        newest = self.l3_pool is None or self._l1_fence.still_newest(ticket)
        cached = self._l1 is not None and landed and newest
        if cached:
            assert self._l1 is not None  # narrow: cached implies an L1 backend
            self._l1.upsert(self.table_name, data, self.primary_key_columns)
        else:
            self._evict_l1(entity_id)
        if not before.fenced:
            await self._save_to_l2(entity_id, data)

    async def _write_l2_at(self, entity_id: Any, data: dict[str, Any], revision: int | None) -> bool:
        """write a committed row to L2 at ``revision``, or delete the key when that is refused.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param data: the row as committed
        :ptype data: dict[str, Any]
        :param revision: the key's latest revision as read before the L3 write, ``0`` when it had
            no message; ``None`` when that read failed, which writes nothing and deletes the key
        :ptype revision: int | None
        :return: whether L2 now holds this row
        :rtype: bool
        """
        written = False
        try:
            kv = await self._ensure_kv()
            if kv is not None and revision is not None:
                payload = self.serialize(self._normalise_datetimes_for_write(data))
                lifetime = self._l2_entry_lifetime(data)
                timed = {} if lifetime is None else {"ttl": lifetime}
                written = (
                    await kv.update(key=self.l2_key(entity_id), value=payload, revision=revision, **timed) is not None
                )
        except KvError as exc:
            log.warning(
                "L2 write after a save failed; the key is dropped and the next read goes to L3",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name, "error": str(exc)}},
            )
        if not written:
            # the key may hold a row older than the one L3 now has; a delete is always correct,
            # since the next read seeds L2 from L3 at the deletion's revision.
            await self._delete_from_l2(entity_id)
            log.debug(
                "a saved row did not reach L2 at the revision read before its L3 write; the key is dropped",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name}},
            )
        return written

    async def persist_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        """Persist a write-buffer entry to L3. Used by ``flush_pending``.

        A buffered compare-and-swap row -- one carrying the order its swap won, on a collection
        that persists it -- is written through :meth:`save_ordered_to_store`, so a flush that
        arrives after a newer winner's leaves the newer row in place.

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param conn: optional **backend-specific transaction handle** that overrides
            :attr:`l3_pool` for this write, so the flush can persist a toposorted batch
            inside ONE backend transaction (``flush_pending`` atomic-batch path).
            ``None`` uses the collection's own L3 store (the per-entity fallback).
        :ptype conn: Any
        :return: rows affected reported by the backend; for an ordered row, 0 when a row with a
            newer-or-equal order is already stored
        :rtype: int
        """
        rows: int
        if self.persists_l2_order and l2_order_of(data) is not None:
            rows = await self.save_ordered_to_store(data, conn=conn)
        else:
            rows = await self.save_to_store(data, conn=conn)
        return rows

    @traced()
    async def reload_entity(self, entity: BaseEntity) -> None:
        """Reload entity from L3."""
        self._set_span_table()
        # addressing_id, not id: on a composite-pk table the bare row id
        # addresses nothing, and every tier below this line keys off the
        # full tuple.
        entity_id = entity.addressing_id
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        with self._l1_fence.watching(self._fence_key(entity_id), writing=False) as ticket:
            # L2's state BEFORE the L3 read, so the refresh below lands only if nothing was written
            # to the key in between (see _seed_l2).
            before = await self._l2_latest_before_refresh(entity_id)
            data = await self.fetch_from_store(entity_id)
            if data is None:
                raise ValueError(f"Entity {entity_id} not found in storage")
            # the handle holds the row it reloaded, as a loaded handle holds the row it read: L1's
            # copy is the fenced write below, never set_data's, and the handle never answers from it.
            entity.hold_row(data)
            entity.set_data(data)
            entity.original_date_updated = data.get("date_updated")
            if self._l1 is not None:
                if self._l1_fence.still_newest(ticket):
                    # Stamped: this row came from L3, so its provenance is a lower tier
                    # even though no pull-through ran. Leaving it unstamped would make a
                    # freshly-reloaded row read as locally authored, and locally
                    # authored rows never expire.
                    self._l1.upsert(self.table_name, self._stamped(data), self.primary_key_columns)
                else:
                    # a write or eviction of the key overlapped the read, so the row may already be
                    # older than L3's: the handle keeps it, L1 does not.
                    self._evict_l1(entity_id)
        if before is not None:
            live, revision = before
            # a live L2 value is refreshed only where L3 is never behind L2: a collection whose L3
            # writes are deferred, or whose rows l2_cas_mutate orders, can hold a newer value in L2
            # than in L3 (a persist not yet landed), and replacing it would move every reader back.
            if not live or not (self._defers_l3_writes or self.persists_l2_order):
                await self._seed_l2(entity_id, data, revision)
        await self._publish_invalidation(entity_id)

    async def _l2_latest_before_refresh(self, entity_id: Any) -> tuple[bool, int] | None:
        """read whether L2 holds a live value for ``entity_id``, and its latest revision.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: ``(holds a live value, latest revision)``, or ``None`` when there is no L2 or it
            could not be read -- in which case a refresh leaves L2 alone
        :rtype: tuple[bool, int] | None
        """
        result: tuple[bool, int] | None = None
        try:
            kv = await self._ensure_kv()
            if kv is not None:
                raw, revision = await kv.get_latest(key=self.l2_key(entity_id))
                result = (raw is not None, revision)
        except KvError as exc:
            log.warning(
                "L2 read before a reload failed; the reload leaves L2 as it is",
                extra={"extra_data": {"entity_id": str(entity_id), "table": self.table_name, "error": str(exc)}},
            )
        return result

    @traced()
    async def delete(self, entity_id: Any) -> bool:
        """delete entity from all tiers.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: always ``True`` (delete is idempotent across tiers)
        :rtype: bool
        """
        self._set_span_table()
        if self._write_buffer is not None:
            await self._write_buffer.remove(self.table_name, entity_id)
        await self.delete_from_store(entity_id)
        self._evict_l1(entity_id)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)
        return True

    @traced()
    async def l2_cas_mutate(
        self,
        entity_id: Any,
        mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]],
        *,
        max_retries: int = 8,
    ) -> CasMutation:
        """atomically read-modify-write one entity's L2 value under revision CAS.

        the framework's optimistic-lock fence lives in the L3 write
        path (:meth:`save_to_store` returning ``0`` rows), so an
        L1+L2-only collection (no L3 pool) has no ``save_entity``-level
        CAS for atomically mutating a single value. this primitive
        fills that gap: it compare-and-swaps the entity's single L2
        (NATS-KV) value directly, using the wrapper's documented
        revision primitives, then reconciles L1 and fires the cross-pod
        invalidation. it operates on exactly one pk-keyed value (not a
        key-listing), so it stays within the pk-keyed collection
        contract.

        the ``mutate`` callback receives the current row dict (or
        ``None`` when the value is absent) and returns
        ``(action, new_row)``:

        - ``("noop", None)`` -- nothing to do; returns without writing.
        - ``("delete", *)`` -- CAS-delete the value (e.g. the last
          member left). ``new_row`` is ignored. deleting an
          already-absent value is a successful idempotent no-op, so a
          callback may return ``"delete"`` on a ``None`` row without
          special-casing. **Refused on a collection with an L3 pool**,
          see below.
        - ``("upsert", new_row)`` -- create-if-absent (so a racing
          creator loses the row) when the value was absent, else
          CAS-update against the read revision. ``new_row`` MUST be
          non-``None``.

        ``date_created`` / ``date_updated`` are stamped on an upsert
        consistently with :meth:`save_entity` (``date_created`` only
        when the value is newly created; ``date_updated`` on every
        upsert). the callback MUST be **idempotent and side-effect
        free** -- it is re-invoked from the freshly-read state on every
        retry.

        **On a collection with an L3 pool**, the same compare-and-swap is
        the concurrency fence and L3 is the durable record behind it:

        - when L2 holds no live row -- a wiped broker, a cold key, an
          absent-marker -- the callback is shown L3's row, so a counter
          continues from its durable value rather than restarting at zero;
        - the won result is persisted per :attr:`l3_write_policy`:
          synchronously before this returns, or through the write buffer;
        - every persist carries the ORDER its swap won -- the L2 revision,
          and the creation time of the L2 stream that revision belongs to
          (:mod:`threetears.core.collections.l2_order`) -- and lands in L3
          only over a row with an older stored order. Two winners of
          consecutive revisions persist independently and can reach L3 in
          either order; the fence is what makes L3 end on the later one
          either way. A persist refused because a newer order is already
          stored is not an error: the later swap built on this one, so its
          row already carries this change. The creation time is what keeps
          the order monotonic across a broker restart, which recreates the
          bucket with its revisions back at 1;
        - so the collection must persist the order
          (:attr:`persists_l2_order`: the ``l2_epoch`` and ``l2_revision``
          columns, and :meth:`save_ordered_to_store`). One that does not is
          refused before L2 is touched: an unfenced persist is the data loss
          this exists to prevent, not a degraded mode of it;
        - a ``"delete"`` is refused, before L2 is touched. Removing the row
          from L3 leaves nothing to carry an order, so a persist of an earlier
          winner still in flight -- a held connection, another replica's
          write buffer -- would land afterwards and resurrect the row, and the
          next mutation would seed from it. Express removal as an upsert the
          reads treat as absent instead: an expiry in the past on a collection
          that declares :attr:`expires_at_column`, or an empty state. That row
          keeps its order, so a late persist is refused;
        - a collection that caches absences advances its write generation
          after a synchronous persist, as :meth:`save_entity` does, and
          raises if it cannot once L1 and the broadcast have run;
        - a persist that fails withdraws the won L2 value (deleted at the
          revision it won) before the error propagates, so a retry does not
          see a write it was told failed;
        - the invalidation broadcast tells peers sharing this registry's L2
          scope to keep the key, which already holds the newest value.

        **What L3 guarantees, exactly.** L3 holds, for each row, the result of
        the latest swap whose persist has landed, and never goes back to an
        earlier one. When L2 then loses the key, the next mutation continues
        from that row, so nothing that reached L3 is lost. What had not yet
        reached L3 when L2 lost the key is lost: under ``"write_behind"`` that
        is up to one flush interval of changes, the trade
        :attr:`l3_write_policy` states for data like attempt counters; under
        ``"synchronous"`` it narrows to a swap whose persist was still in
        flight at the instant the broker restarted.

        The fence is one scoped L2 key, so a three-tier compare-and-swap row
        is mutated by one principal's replicas; two principals would each
        hold their own key and overwrite each other in L3. Mutate such a row
        only through this method: an unfenced :meth:`save_entity` on the
        same row invalidates the key for every peer, and stores no order.

        **The one sanctioned second writer shares the owner's scope.** A
        principal granted write on another principal's rows -- a tool pod
        granted write on an agent's tables -- does not build its registry
        from its own scope. It builds it from the OWNER's id:
        ``kv_key_scope_for(Principal.AGENT_POD, agent_id=owner)`` for the L2
        scope, and the owner's namespace for its L3 backend, both derived
        from that one id. Its swaps then compare against the owner's key, so
        the two writers share one revision fence and neither overwrites the
        other; the invalidation it publishes names that shared scope, so the
        owner's replicas keep the key rather than evicting it. It still
        authenticates as itself, so L3 refuses it without the grant and the
        write audit names it, not the owner. A writer that keys on its own
        scope instead is the two-principal case above, so derive both halves
        in one place rather than at two call sites that can name two owners.

        **L1-only fallback**: when no NATS client is wired
        (:meth:`_ensure_kv` is ``None`` -- unit / single-pod), there is
        no cross-pod contention to CAS against, so the method degrades
        to a single uncontended read-modify-write via :meth:`get` /
        :meth:`save_entity` / :meth:`delete`.

        **deliberately NOT degrade-on-KvError, unlike its L1+L2+L3
        siblings**: :meth:`_get_from_l2`/:meth:`_save_to_l2`/
        :meth:`_delete_from_l2` catch :class:`KvError` from
        :meth:`_ensure_kv` (bucket resolution) the same as from the
        op itself, because L3 remains the durable source of truth
        behind them -- an L2 outage there is a cache miss, not data
        loss. this method has no such backstop: for the L1+L2-only
        collections it targets (e.g. ``channels/presence``'s room
        membership), L2 **is** the source of truth, so a bucket-open
        failure here is a real write/read failure, not best-effort
        cache noise, and MUST propagate rather than silently no-op a
        membership change that never actually happened.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :param mutate: pure callback computing the next value from the
            current row (or ``None`` when absent)
        :ptype mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]]
        :param max_retries: how many times to retry on a CAS conflict
            before surfacing the error
        :ptype max_retries: int
        :return: what was done, and the row written -- on a collection with
            an L3 pool, carrying the order its swap won
        :rtype: CasMutation
        :raises ConcurrentModificationError: when the retry budget is
            exhausted (a genuine livelock, never the common case)
        :raises KvError: on any L2 transport failure, including
            during bucket resolution -- intentionally not swallowed,
            see the note above
        :raises RuntimeError: when a synchronous L3 persist affects no row
            and no newer order is stored in its place; L2 is withdrawn first,
            as for any persist failure
        :raises GenerationUnavailableError: when a collection that caches
            absences persisted the result but could not advance its write
            generation
        :raises ValueError: on a collection with an L3 pool whose every L3
            write is fenced (``cas_null_safe``), or that does not persist the
            L2 order, before L2 is touched; on a ``"delete"`` there, before L2
            is touched; and there on an ``"upsert"`` whose row leaves any column
            for the database to decide (:meth:`columns_decided_by_store`), before
            L2 is touched
        :raises L2EpochRegressedError: when L3 holds the row under an order
            later than anything the current bucket can write, before L2 is
            touched
        """
        self._set_span_table()
        kv = await self._ensure_kv()
        if kv is None:
            return await self._l1_only_cas_mutate(entity_id, mutate)
        ordered = self.l3_pool is not None
        if ordered:
            self._refuse_unfenceable_cas()

        key = self.l2_key(entity_id)
        # read once, BEFORE any write: a bucket recreated after this read can only make the order
        # computed from it too low, never too high -- and the won write re-reads it to catch that.
        epoch = await kv.date_created() if ordered else None
        for attempt in range(max_retries):
            entry = await kv.get_entry(key=key)
            if entry is None:
                row: dict[str, Any] | None = None
                revision: int | None = None
            else:
                raw_bytes, revision = entry
                try:
                    decoded = self._decode_l2_value(raw_bytes)
                    # an absent-marker is no prior row; its revision is kept so the write below
                    # compare-and-swaps the marker away rather than creating over it.
                    row = None if isinstance(decoded, _AbsentMarker) else decoded
                except CorruptCacheEntry as exc:
                    # Treated as no prior row, but the REVISION is kept deliberately. The write
                    # below then takes the `update` branch and compare-and-swaps the corrupt
                    # entry away at the revision that held it, which self-heals the key.
                    # Dropping the revision too would take the `create` branch against a key
                    # that exists, fail, and retry until the attempt budget ran out.
                    log.warning(
                        "L2 entry could not be decoded; replacing it in this CAS round",
                        extra={
                            "extra_data": {
                                "entity_id": str(entity_id),
                                "table": self.table_name,
                                "column": exc.column,
                            },
                        },
                    )
                    row = None

            if row is not None and self._row_is_expired(row):
                row = None
            if row is None and ordered:
                # L2 holds no live row, but L3 is the source of truth: a wiped broker must not reset
                # a counter to zero. no separate seeding write -- the create below converges, since
                # a racing seeder that lands first makes it fail and the next round reads L2.
                seeded = await self.fetch_from_store(entity_id)
                row = None if seeded is None or self._row_is_expired(seeded) else seeded
                if seeded is not None:
                    assert epoch is not None  # narrow: read above whenever ordered
                    self._refuse_regressed_epoch(entity_id, seeded, epoch)

            action, new_row = mutate(row)
            if action == "noop":
                return CasMutation(action="noop", row=None)
            if action == "delete" and ordered:
                raise ValueError(
                    f"{type(self).__name__} persists compare-and-swap rows to L3, where a delete cannot be "
                    f"fenced: an earlier winner's persist still in flight would land after it and resurrect "
                    f"the row. Upsert a row the reads treat as absent instead -- an expiry in the past, or "
                    f"an empty state -- which keeps its order in L3"
                )

            now = datetime.now(UTC)
            ok: bool
            won_revision: int | None = None
            payload = b""
            if action == "delete":
                ok = await kv.delete(key=key, revision=revision)
            else:
                assert new_row is not None, "'upsert' action must carry a non-None new_row"
                new_row["date_updated"] = now
                # a row new to every tier is stamped now; one seeded from L3, or rewritten by a
                # callback that dropped the field, keeps the creation time it already had.
                new_row.setdefault("date_created", now if row is None else row.get("date_created", now))
                if ordered:
                    # the L2 value is written before its own revision exists, so it carries no
                    # order; the order is stamped on the row once the swap has won.
                    new_row = without_l2_order(new_row)
                # L2 takes the row before L3 does, so it must already be the row L3 will hold.
                new_row = self._write_ahead_row(new_row)
                lifetime = self._l2_entry_lifetime(new_row)
                # as in _save_to_l2: the ttl keyword is sent only when this table declares an
                # expiry, so a bucket shim that predates per-entry lifetimes still satisfies the
                # call a non-expiring collection makes.
                payload = self.serialize(self._normalise_datetimes_for_write(new_row))
                timed = {} if lifetime is None else {"ttl": lifetime}
                if revision is None:
                    # value absent: create-if-absent so a racing creator loses.
                    won_revision = await kv.create(key=key, value=payload, **timed)
                else:
                    won_revision = await kv.update(key=key, value=payload, revision=revision, **timed)
                ok = won_revision is not None

            if not ok:
                if attempt == max_retries - 1:
                    raise ConcurrentModificationError(self.table_name, entity_id, datetime.min)
                log.info(
                    "L2 CAS conflict; retrying",
                    extra={
                        "extra_data": {
                            "entity_id": str(entity_id),
                            "table": self.table_name,
                            "attempt": attempt + 1,
                            "action": action,
                        }
                    },
                )
                await asyncio.sleep(random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS))  # noqa: S311 - jitter, not security
                continue

            # CAS won: the L2 revision was the fence. persist to L3 (when there is one) only now,
            # so a write that lost the race is never persisted; then reconcile L1 and notify peers.
            if ordered and new_row is not None:
                assert epoch is not None and won_revision is not None  # narrow: an ordered upsert that won
                order = await self._won_order(entity_id, key, epoch, won_revision, payload)
                new_row = with_l2_order(new_row, order)
            try:
                generation_failure = await self._persist_cas_result(entity_id, action, new_row)
            # BaseException, not Exception: a persist cancelled mid-write leaves L2 holding a value
            # the caller is told did not complete, as any failure does, and CancelledError is not an
            # Exception.
            except BaseException as exc:  # prawduct:allow prawduct/broad-except -- compensates L2 for any persist failure, then re-raises it unchanged
                await self._withdraw_unpersisted_cas(entity_id, key, action, won_revision, exc)
                raise
            if action == "delete":
                self._evict_l1(entity_id)
            else:
                assert new_row is not None  # narrow: "upsert" always carries a row
                if self._l1 is not None:
                    # an ordered swap reaches this line only after the persist's round trip, and a
                    # later swap can complete inside it: this replica's own, or a peer's whose
                    # broadcast already evicted this L1. Caching this row then would leave L1 behind
                    # L2 with nothing left to evict it, so it is cached only while L2 still holds
                    # it; otherwise the key is dropped and the next read takes the newer value from
                    # L2. No await separates the check from the write, so nothing lands between.
                    assert won_revision is not None  # narrow: an upsert that won
                    # a read of the key in flight here read it before this swap, so it must not
                    # cache what it read: the swap is recorded on the fence as it lands in L1.
                    if not ordered or await self._swap_still_current(entity_id, key, won_revision, payload):
                        self._l1_fence.changed(self._fence_key(entity_id))
                        self._l1.upsert(self.table_name, new_row, self.primary_key_columns)
                    else:
                        self._evict_l1(entity_id)
                self._clear_l1_marker(entity_id)
            await self._publish_invalidation(entity_id, l2_key_current=True)
            if generation_failure is not None:
                raise generation_failure
            outcome: CasMutation
            if action == "delete":
                outcome = CasMutation(action="deleted", row=None)
            else:
                outcome = CasMutation(action="created" if row is None else "updated", row=new_row)
            return outcome
        raise AssertionError("unreachable: every CAS round either returns or retries within the budget")

    def _refuse_unfenceable_cas(self) -> None:
        """refuse, before L2 is touched, a compare-and-swap whose L3 persist cannot be ordered.

        :return: nothing
        :rtype: None
        :raises ValueError: when every L3 write is fenced (``cas_null_safe``), or the collection
            does not persist the L2 order
        """
        if self.emits_cas_fence:
            # the persist carries no fence value, so a fenced table would refuse every update
            # after the first -- having already let L2 advance.
            raise ValueError(
                f"{type(self).__name__} fences every L3 write (cas_null_safe), so l2_cas_mutate cannot "
                f"persist to it: the L2 revision is this method's fence and the persist is unfenced"
            )
        if not self.persists_l2_order:
            raise ValueError(
                f"{type(self).__name__} has an L3 pool but does not persist the L2 order its swaps win "
                f"({L2_ORDER_COLUMNS[0]} TIMESTAMPTZ and {L2_ORDER_COLUMNS[1]} BIGINT, written through "
                f"save_ordered_to_store). Without them two winners persisting in reverse leave L3 on the "
                f"earlier row, and a broker restart then loses the later change. Declare both columns "
                f"(threetears.core.collections.schema_backed.l2_order_columns()) and migrate the table "
                f"(threetears.core.collections.l2_order.l2_order_migration_statements())"
            )

    def _refuse_regressed_epoch(self, entity_id: Any, stored: dict[str, Any], epoch: datetime) -> None:
        """refuse a swap whose every possible order is older than the one L3 already holds.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param stored: the row L3 holds
        :ptype stored: dict[str, Any]
        :param epoch: the current bucket's creation time
        :ptype epoch: datetime
        :return: nothing
        :rtype: None
        :raises L2EpochRegressedError: when the stored order's epoch is later than ``epoch``
        """
        stored_order = l2_order_of(stored)
        if stored_order is not None and stored_order.epoch > epoch:
            raise L2EpochRegressedError(self.table_name, entity_id, stored_order.epoch, epoch)

    async def _won_order(self, entity_id: Any, key: str, epoch: datetime, won_revision: int, payload: bytes) -> L2Order:
        """the order a won swap holds: its revision, and the creation time of the stream it landed in.

        ``epoch`` was read before the swap. When the bucket's creation time is unchanged now, the
        swap landed in that stream. When it moved, the bucket was recreated around the swap and
        either stream could hold it; the key is read back, and when it still holds this swap's
        value at this swap's revision the swap is in the current stream. Otherwise the earlier time
        is kept: either the swap's stream died with the swap in it, and that time is the true one,
        or a later swap in the new stream has already built on this one and carries it into L3.
        Never guessing the later time is what matters: an order too high would outrank the new
        stream's writes, which restart at revision 1, and refuse them in L3.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param key: the scoped L2 key the swap won
        :ptype key: str
        :param epoch: the bucket's creation time read before the swap
        :ptype epoch: datetime
        :param won_revision: the revision the swap produced
        :ptype won_revision: int
        :param payload: the bytes the swap wrote
        :ptype payload: bytes
        :return: the swap's order
        :rtype: L2Order
        :raises KvError: when the bucket cannot be read
        """
        kv = await self._ensure_kv()
        assert kv is not None  # narrow: only reached after a swap won on this bucket
        current = await kv.date_created()
        chosen = epoch
        if current != epoch:
            entry = await kv.get_entry(key=key)
            if entry is not None and entry == (payload, won_revision):
                chosen = current
            log.info(
                "L2 bucket was recreated around a compare-and-swap; its order uses the stream it landed in",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),  # convert at border: log extra_data field
                        "table": self.table_name,
                        "epoch_before": epoch.isoformat(),
                        "epoch_after": current.isoformat(),
                        "chosen": chosen.isoformat(),
                    }
                },
            )
        return L2Order(epoch=chosen, revision=won_revision)

    async def _swap_still_current(self, entity_id: Any, key: str, won_revision: int, payload: bytes) -> bool:
        """whether L2 still holds exactly the value a won swap wrote, at the revision it won.

        Asked after an ordered swap's persist, before its row is cached in L1: anything else in the
        key -- a later swap's value, or no value -- means the row is no longer the newest, and
        caching it would serve the older value on this replica until something evicted it. A
        failed read answers ``False``: the swap has already won and persisted, so the mutation must
        not be reported failed, and leaving L1 empty is always correct because the next read goes
        to L2.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order, for the log
        :ptype entity_id: Any
        :param key: the scoped L2 key the swap won
        :ptype key: str
        :param won_revision: the revision the swap produced
        :ptype won_revision: int
        :param payload: the bytes the swap wrote
        :ptype payload: bytes
        :return: ``True`` only when the key still holds this swap's value at this swap's revision
        :rtype: bool
        """
        current = False
        try:
            kv = await self._ensure_kv()
            if kv is not None:
                current = await kv.get_entry(key=key) == (payload, won_revision)
        except KvError as exc:
            log.warning(
                "L2 read after a compare-and-swap failed; its row is not cached in L1 and the next read goes to L2",
                extra={
                    "extra_data": {
                        "entity_id": str(entity_id),  # convert at border: log extra_data field
                        "table": self.table_name,
                        "error": str(exc),
                    }
                },
            )
        return current

    async def _withdraw_unpersisted_cas(
        self,
        entity_id: Any,
        key: str,
        action: Literal["upsert", "delete"],
        won_revision: int | None,
        persist_error: BaseException,
    ) -> None:
        """take back a won compare-and-swap whose L3 persist failed.

        The caller is about to be told the write failed, so L2 must not keep answering with it: a
        retried claim would read "already claimed" and skip its work, and a retried increment
        would count twice. The upsert is withdrawn by deleting the key at the revision it won,
        not by restoring the prior bytes: with an L3 pool behind it an absent L2 entry is always
        a correct state, since the next read or mutation starts from L3, which never took the
        write. A failed delete needs nothing: its L2 entry is already absent and L3 still holds
        the row.

        When the delete at that revision fails, another writer has already compare-and-swapped
        on top of the unpersisted value and its own persist carries it forward; that is logged,
        not repaired.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param key: the scoped L2 key the mutation won
        :ptype key: str
        :param action: the action that won in L2
        :ptype action: str
        :param won_revision: the L2 revision the upsert produced, ``None`` for a delete
        :ptype won_revision: int | None
        :param persist_error: why the persist failed, for the log
        :ptype persist_error: BaseException
        :return: nothing
        :rtype: None
        """
        withdrawn = False
        withdraw_error: str | None = None
        remedy = "retry the mutation once L3 accepts writes"
        if action == "upsert" and won_revision is not None:
            try:
                kv = await self._ensure_kv()
                withdrawn = kv is not None and await kv.delete(key=key, revision=won_revision)
            except KvError as exc:
                withdraw_error = str(exc)
            if withdraw_error is not None:
                remedy = "L2 still holds the unpersisted value: delete the key, or retry once L2 and L3 are reachable"
            elif not withdrawn:
                remedy = "a later compare-and-swap already built on this value in L2, and its persist carries it"
        log.error(
            "L3 persist of a won L2 compare-and-swap failed; the write is reported failed",
            extra={
                "extra_data": {
                    "entity_id": str(entity_id),  # convert at border: log extra_data field
                    "table": self.table_name,
                    "action": action,
                    "error": f"{type(persist_error).__name__}: {persist_error}",
                    "l2_withdrawn": withdrawn,
                    "l2_withdraw_error": withdraw_error,
                    "remedy": remedy,
                }
            },
        )

    async def _persist_cas_result(
        self, entity_id: Any, action: Literal["upsert", "delete"], new_row: dict[str, Any] | None
    ) -> GenerationUnavailableError | None:
        """land a won compare-and-swap in L3, per this collection's L3 write policy.

        Nothing to do without an L3 pool: there, L2 is the source of truth, and a delete reaches
        this only there (:meth:`l2_cas_mutate` refuses one on a collection with an L3 pool). A
        write-behind upsert joins the write buffer, which keeps the newer order when two land
        on one row, and is flushed later; a synchronous one is written before this returns. Both
        are written fenced on the order the row carries.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param action: the action that won
        :ptype action: str
        :param new_row: the row written, for an upsert, carrying the order its swap won
        :ptype new_row: dict[str, Any] | None
        :return: a write-generation failure for the caller to raise once L1 and the broadcast have
            run, or ``None``
        :rtype: GenerationUnavailableError | None
        :raises RuntimeError: when a synchronous L3 write affects no row and no newer order is
            stored in its place
        """
        if self.l3_pool is None or action == "delete":
            return None
        assert new_row is not None  # narrow: "upsert" always carries a row
        if self._defers_l3_writes:
            assert self._write_buffer is not None  # narrow: deferral requires one
            await self._write_buffer.add(self.table_name, entity_id, new_row)
            return None
        if await self.save_ordered_to_store(new_row) == 0:
            await self._confirm_superseded(entity_id, new_row)
            return None
        return await self._advance_generation()

    async def _confirm_superseded(self, entity_id: Any, new_row: dict[str, Any]) -> None:
        """confirm that an ordered persist which wrote nothing was refused for a newer stored order.

        The ordered write affects no row in exactly one expected case: L3 already holds this row
        under a newer-or-equal order, because a later swap -- which built on this one -- persisted
        first. That is success, and it needs no write-generation advance: the later writer's
        commit advanced it. Anything else that affected no row is a write that went nowhere, and
        is raised as such.

        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param new_row: the row whose persist affected nothing, carrying its order
        :ptype new_row: dict[str, Any]
        :return: nothing
        :rtype: None
        :raises RuntimeError: when L3 holds no row, or holds one under an older order
        """
        ours = l2_order_of(new_row)
        stored = await self.fetch_from_store(entity_id)
        theirs = None if stored is None else l2_order_of(stored)
        if ours is None or theirs is None or theirs < ours:
            raise RuntimeError(
                f"{self.table_name}: persisting a won compare-and-swap for {entity_id!r} affected no L3 row, "
                f"and L3 holds no newer order in its place (stored order {theirs}, ours {ours})"
            )
        log.debug(
            "compare-and-swap persist superseded by a newer stored order; the later swap carries it",
            extra={
                "extra_data": {
                    "entity_id": str(entity_id),  # convert at border: log extra_data field
                    "table": self.table_name,
                    "ours": str(ours),
                    "stored": str(theirs),
                }
            },
        )

    async def _l1_only_cas_mutate(
        self,
        entity_id: Any,
        mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]],
    ) -> CasMutation:
        """uncontended read-modify-write fallback for :meth:`l2_cas_mutate`.

        used when no NATS client is wired (L1-only unit / single-pod
        mode): there is no cross-pod contention to CAS against, so a
        plain read-modify-write via :meth:`get` + :meth:`save_entity` /
        :meth:`delete` is correct, and those carry the L3 write and the
        write-generation advance with them. ``date_created`` /
        ``date_updated`` stamping is owned by :meth:`save_entity`, so this
        path does not re-stamp.

        :param entity_id: pk value (single-pk) or tuple of pk values
        :ptype entity_id: Any
        :param mutate: same callback contract as :meth:`l2_cas_mutate`
        :ptype mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]]
        :return: what was done
        :rtype: CasMutation
        """
        entity = await self.get(entity_id)
        row = entity.to_dict() if entity is not None else None
        action, new_row = mutate(row)
        outcome: CasMutation
        if action == "noop":
            outcome = CasMutation(action="noop", row=None)
        elif action == "delete":
            await self.delete(entity_id)
            outcome = CasMutation(action="deleted", row=None)
        else:
            assert new_row is not None, "'upsert' action must carry a non-None new_row"
            if entity is None:
                await self.save_entity(self.create(new_row))
            else:
                entity.set_data(new_row)
                await self.save_entity(entity)
            outcome = CasMutation(action="created" if entity is None else "updated", row=new_row)
        return outcome

    @traced()
    async def invalidate_cache(self, entity_id: Any) -> None:
        """delete from L1 and L2, signal other pods.

        :param entity_id: pk value (single-pk) or tuple of pk values in
            declared column order (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._set_span_table()
        self._evict_l1(entity_id)
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)

    @asynccontextmanager
    async def bypassing_write(self, *entity_ids: Any, conn: Any = None) -> AsyncIterator[BypassingWrite]:
        """run a write that bypasses :meth:`save_entity`, then settle every cache of the rows it touched.

        The one owner of the rule for a targeted UPDATE or DELETE on a table :meth:`get` serves: the
        statement leaves L1 and L2 on every replica holding the row it replaced, so every row it may
        have changed is evicted from both and the eviction broadcast.

        - **On the collection's own pool** (``conn`` is ``None``): the rows are evicted once the
          body ends, however it ended -- a statement that raised may still have reached L3, and a
          cancelled one's outcome is as unknown. The eviction is shielded, so a cancellation
          arriving meanwhile cannot stop it partway. A body that knows it changed nothing (a
          compare-and-swap that lost) calls :meth:`BypassingWrite.unchanged` and evicts nothing,
          unless it then raises.
        - **Joined to a caller's transaction** (``conn`` given): the rows are not final until that
          transaction ends, so they are enrolled in the enclosing
          :class:`~threetears.core.collections.caller_transaction.CallerTransaction` and settled
          there. A connection no ``CallerTransaction`` opened is refused before the body runs.

        Usage::

            async with self.bypassing_write((conversation_id, schedule_id)):
                # cache-bypass: targeted UPDATE; the row is evicted from every tier once it lands.
                await self.l3_pool.execute("UPDATE ...", ...)

        :param entity_ids: the rows the write may change, when known before it -- pk values
            (single-pk) or tuples of pk values in declared column order; more can be named during
            the write with :meth:`BypassingWrite.touches`
        :ptype entity_ids: Any
        :param conn: the caller's connection the write joins, or ``None`` for the collection's pool
        :ptype conn: Any
        :return: the write's handle
        :rtype: AsyncIterator[BypassingWrite]
        :raises ValueError: when ``conn`` is given and its transaction was not opened by
            ``CallerTransaction``
        """
        transaction = (
            None if conn is None else CallerTransaction.join(conn, writer=f"{type(self).__name__}.bypassing_write")
        )
        write = BypassingWrite(self, transaction)
        write.touches(*entity_ids)
        completed = False
        try:
            yield write
            completed = True
        finally:
            if transaction is None and not (completed and write.is_unchanged):
                await asyncio.shield(self._evict_every(write.keys))

    async def _evict_every(self, entity_ids: tuple[Any, ...]) -> None:
        """evict each row in ``entity_ids`` from L1 and L2 and broadcast it (:meth:`invalidate_cache`).

        :param entity_ids: pk values (single-pk) or tuples of pk values in declared column order
        :ptype entity_ids: tuple[Any, ...]
        :return: nothing
        :rtype: None
        """
        for entity_id in entity_ids:
            await self.invalidate_cache(entity_id)

    def create(self, data: dict[str, Any]) -> EntityT:
        """Create new entity (not persisted until save)."""
        return self.entity_class(data, is_new=True, collection=self)
