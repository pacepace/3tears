"""Collection registry — DI container + table_name lookup + cache coherence."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid7

from pydantic import BaseModel, SerializerFunctionWrapHandler, ValidationError, model_serializer
from threetears.nats import KV_KEY_SCOPE_GRAMMAR, Subjects
from threetears.nats.errors import PublishError
from threetears.core.keyed import WholeReads
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage, NotedKeysReader
from threetears.core.collections.epoch_change import (
    RESET_REASON,
    EpochChange,
    EpochChangeReader,
    EpochLogExtent,
    EpochWalk,
    key_text,
    walk_by_count,
)
from threetears.core.collections.generation import (
    GenerationMarks,
    GenerationVerdict,
    L2WriteGeneration,
    NoWriteGeneration,
    WriteGeneration,
    source_reads,
)
from threetears.core.backends.protocol import DurableStore
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.scan_cache import ScanCache
from threetears.core.exceptions import GenerationUnavailableError, InvalidL2ScopeError, L2ScopeNotConfiguredError
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

if TYPE_CHECKING:
    from threetears.core.backends import L3Backend
    from threetears.core.collections.generation import GenerationSource

    # annotation-only. `Subjects` above is a genuine runtime use, but it lives
    # in a nats-py-free submodule, so importing it eagerly costs nothing.
    from threetears.nats import NatsClient, Subscription

__all__ = [
    "CacheInvalidationMessage",
    "CollectionRegistry",
    "DerivedCacheRegistration",
]

log = get_logger(__name__)

# the epoch catch-up's instruments, fetched where they are moved, never held at import: a module that a
# fresh import of the package runs again would register them twice


def _epochs_applied() -> Any:
    return counter(
        "collection_epoch_changes_applied_total",
        description="epoch change records a follower fetched and applied because it had not heard every row of them",
        label_names=("table",),
    )


#: every reason a followed table falls behind, and what ends it: the one list the metric's description and the
#: ERROR are built from, so neither can leave a reason out
BEHIND_REASONS: Final[dict[str, str]] = {
    "record_missing": (
        "the change log holds no record for an epoch after the mark; ends on its own once the record can be read "
        "(after a NATS loss, when the change-note keeper's restore is labelled), otherwise at a restart"
    ),
    "record_unreadable": "a record could not be read; the next catch-up pass reads it again",
    "keys_unknown": (
        "an epoch's writer named no rows, and its record carries no notes bracket (a broker built before it) or "
        "this registry has no noted-keys reader; ends at a restart (a first sight of the table)"
    ),
    "notes_not_covered": (
        "an epoch's writer named too many rows to record, and the change notes no longer hold its bracket (or "
        "the broker has no l3.noted_keys door, or did not answer); every catch-up pass asks again, and it ends "
        "on its own when the change-note keeper's restore of a lost log is labelled and the mark is at or past "
        "covered_from, otherwise at a restart"
    ),
    "notes_unavailable": (
        "an epoch's writer named too many rows to record, and no change-note ring is installed to name them; "
        "every catch-up pass asks again, otherwise it ends at a restart"
    ),
    "replaced": (
        "the log is under another incarnation that does not continue the mark's; ends on its own when the "
        "change-note keeper labels its restore of the lost log and the mark is at or past covered_from, "
        "otherwise at a restart"
    ),
    "record_lost": (
        "the mark is older than the restored log's covered_from, so an epoch it missed may be one no change note "
        "names; ends at a restart"
    ),
    "aged_out": "the log's retention passed the mark; ends at a restart",
    "notes_whole": (
        "the change notes since the table was last current name a whole-table write (a reset, a file load, a large "
        "copy) whose reach they cannot say; ends when the change-note keeper's restore of the log is labelled, "
        "otherwise at a restart. Never a read of the table"
    ),
}


def _epochs_behind() -> Any:
    return counter(
        "collection_epoch_behind_total",
        description=(
            "times a followed table fell behind, by why ("
            + "; ".join(f"{reason}: {what}" for reason, what in BEHIND_REASONS.items())
            + "); the pod serves what it holds"
        ),
        label_names=("table", "reason"),
    )


#: why a process drops a table whole from its caches, counted by table and why (``collection_whole_table_drops_total``);
#: a load or a clear names its key prefixes and drops nothing whole
WHOLE_TABLE_DROPS: Final = WholeReads(
    "collection_whole_table_drops_total",
    subject="tables a process dropped whole from its caches",
    whys={
        "reset": "an operator's reset (SchemaBackedCollection.reset), here or in an epoch record naming reason='reset'",
        # TODO(3tears 0.96.0 contract step, dated 2026-11-09): zero once no 0.94-or-older writer is left to record one
        "whole_record": "an epoch record naming the whole table with no key prefix and no reason: an older writer's",
        "local": "this process dropped it itself, for a change it made (a schema migration, a failed reset)",
    },
    expected={"reset", "local"},
)

# TODO(3tears 0.96.0 contract step, dated 2026-11-10): WHOLE_TABLE_DROPS.whys, kept one release for its importers
WHOLE_TABLE_DROP_WHYS: Final[Mapping[str, str]] = WHOLE_TABLE_DROPS.whys


def _tables_behind() -> Any:
    return gauge(
        "collection_epoch_tables_behind",
        description="followed tables this registry is behind on: an epoch it could not apply, awaiting its record in L2",
    )


def _as_l3_backend(l3_pool: Any) -> Any:
    """normalize a raw L3 transport to a :class:`DurableStore`-capable backend.

    The collection CRUD lifecycle routes through the structured
    :class:`~threetears.core.backends.protocol.DurableStore` ops, so the resolved
    ``l3_pool`` must expose them. A backend that already satisfies ``DurableStore``
    (the named :class:`~threetears.core.backends.sql.SqlL3Backend`, or a non-SQL
    backend such as scriob's ``GitL3Backend``) is returned **unchanged**. A raw
    transport that only speaks raw SQL (a bare asyncpg ``Pool`` or the
    ``NatsProxyL3Backend``) is wrapped in a ``SqlL3Backend`` so it gains the
    structured ops (which generate SQL). ``None`` passes through.

    :param l3_pool: a raw transport, a ``DurableStore`` backend, or ``None``.
    :ptype l3_pool: Any
    :return: a ``DurableStore``-capable backend, or ``None``.
    :rtype: Any
    """
    if l3_pool is None:
        return None
    if isinstance(l3_pool, DurableStore):
        return l3_pool
    return SqlL3Backend(l3_pool)


#: the most notes one page of a keys-unknown epoch's keys reads (:meth:`CollectionRegistry.catch_up_generation`)
_NOTED_KEYS_PAGE_ROWS: Final = 10_000

#: the most bytes of keys one such page holds; a reader over the bus holds it under the bus's announced payload too
_NOTED_KEYS_PAGE_BYTES: Final = 4 * 1024 * 1024

#: the rows of a page applied between yields to the event loop
_NOTED_ROWS_PER_YIELD: Final = 500

#: why the notes could not name an epoch's keys, as a reader answers it -> the reason this registry is behind
#: how far a follower's clock may run ahead of the database's: a notes bracket from it starts this much earlier
_NOTES_CLOCK_SKEW: Final = timedelta(seconds=60)

_NOTES_BEHIND: Final[dict[str, str]] = {"not_covered": "notes_not_covered", "notes_unavailable": "notes_unavailable"}


async def _noted_pages(
    reader: NotedKeysReader, table: str, noted_between: tuple[datetime, datetime]
) -> AsyncGenerator[NotedKeysPage | str]:
    """a keys-unknown commit's noted keys, a page at a time, each read only once the one before is consumed.

    :param reader: what reads the change notes
    :ptype reader: NotedKeysReader
    :param table: the table
    :ptype table: str
    :param noted_between: the commit's notes bracket
    :ptype noted_between: tuple[datetime, datetime]
    :return: each page; or, last, the :data:`BEHIND_REASONS` key of why the notes cannot name the keys
    :rtype: AsyncGenerator[NotedKeysPage | str]
    """
    cursor: NotedKeysCursor | None = None
    while True:
        page = await reader.noted_keys(
            table,
            noted_between=noted_between,
            cursor=cursor,
            limit=_NOTED_KEYS_PAGE_ROWS,
            max_bytes=_NOTED_KEYS_PAGE_BYTES,
        )
        if not isinstance(page, NotedKeysPage):
            yield _NOTES_BEHIND.get(page, "notes_unavailable")
            return
        yield page
        if page.next is None:
            return
        cursor = page.next


#: the most rows one batched invalidation message names: a few hundred bytes each, far under a NATS payload
_ROWS_A_MESSAGE: Final = 500


class InvalidatedRow(BaseModel):
    """one more row a :class:`CacheInvalidationMessage` names, beside the one its own fields name.

    :ivar ids: stringified pk values in declared column order
    :ivar l2_current_scope: as :attr:`CacheInvalidationMessage.l2_current_scope`, for this row
    :ivar columns: as :attr:`CacheInvalidationMessage.columns`, for this row
    """

    ids: list[str]
    l2_current_scope: str | None = None
    columns: dict[str, str | None] | None = None


class CacheInvalidationMessage(BaseModel):
    """typed wire envelope for cross-pod cache invalidation broadcasts.

    publishers emit one message per write op; every other pod
    subscribed to :func:`Subjects.cache_invalidate` evicts the named
    entity from its local L1 cache. ``ids`` is always an array whose
    length matches the target collection's
    :attr:`BaseCollection.primary_key_columns` declaration -- single-pk
    collections emit length-1 arrays; composite-pk collections emit
    length-N arrays in declared column order.

    typed Pydantic envelope replaces the previous raw-JSON-bytes
    publish; the typed wrapper enforces wire-format consistency and
    surfaces drift at parse time rather than via opaque key errors
    deep in the listener.

    :ivar table: target table name (matches :attr:`BaseCollection.table_name`)
    :ivar ids: stringified pk values in declared column order
    :ivar origin: opaque per-registry id of the publisher. a receiving
        registry skips any message carrying its OWN origin -- a single
        pod both publishes and subscribes on this subject, and acting on
        its own invalidation would evict the L1 row it just wrote (the
        fresh value is already local; the broadcast is for OTHER pods).
        ``None`` for messages from a pre-origin publisher -> the receiver
        cannot prove self-origin and evicts (the historical behaviour),
        which is safe: a redundant local eviction only forces a
        pull-through, never stale data.
    :ivar l2_current_scope: the L2 key scope whose entry for this pk the
        publisher left current, set only by a revision-fenced write
        (:meth:`BaseCollection.l2_cas_mutate`). A receiver sharing that
        scope shares that key and skips its L2 eviction: the key already
        holds the newest value, and for a compare-and-swap counter it is
        the only copy newer than L3, so deleting it would move the counter
        backwards. ``None`` for every unfenced write, whose L2 put can land
        out of order and still needs the eviction to heal.
    :ivar generation: the table's write generation as the commit this row
        belongs to advanced it (:meth:`GenerationSource.advance`'s return).
        ``None`` from a publisher that advanced nothing: a table not switched
        on, or a release that predates the field. A receiver then evicts the
        row exactly as it always did and counts nothing.
    :ivar bump_rows: how many row messages the advance named in
        ``generation`` covers, this one included. One commit, flush or
        settled transaction advances a table once however many rows it
        wrote, so a receiver that has counted this many messages for the
        generation has heard all of that advance. Set whenever ``generation``
        is.
    :ivar columns: the values, as strings, of the columns the publishing
        collection declares in :attr:`BaseCollection.invalidation_columns`,
        taken from the row the write saw -- the row being deleted, for a
        delete -- so a receiver whose own entries are keyed by something
        other than the primary key needs no read to know which row this was.
        ``None`` when the collection declares none, or the write saw no row
        (an eviction that names only a key).

    :ivar more_rows: further rows of the same table, the same origin and the same advance, each handled
        exactly as a message of its own would be (and each counted toward ``bump_rows``): one message for a
        write of many rows, not one a row. ``None`` for a message naming one row.

    Every field after ``ids`` is optional and was added after the envelope first
    shipped; a receiver that predates one ignores it, and a message without it is
    handled as it was before the field existed. ``more_rows`` is the exception: a
    receiver that predates it would evict the first row and not the rest, so a
    registry sends it only when switched on (:attr:`CollectionRegistry.batch_invalidations`),
    once every process on the bus reads it (expand, then migrate, then contract).
    """

    table: str
    ids: list[str]
    origin: str | None = None
    l2_current_scope: str | None = None
    generation: str | None = None
    bump_rows: int | None = None
    columns: dict[str, str | None] | None = None
    more_rows: list[InvalidatedRow] | None = None

    @model_serializer(mode="wrap")
    def _leave_out_more_rows_when_none(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """the message as serialized, without ``more_rows`` when it names no further row.

        A message of one row is then the message every receiver read before the field existed, byte for byte:
        the expand step adds nothing to what an older process is sent.

        :param handler: pydantic's serializer
        :ptype handler: SerializerFunctionWrapHandler
        :return: the fields
        :rtype: dict[str, Any]
        """
        fields: dict[str, Any] = handler(self)
        if fields.get("more_rows") is None:
            fields.pop("more_rows", None)
        return fields

    def each_row(self) -> list[CacheInvalidationMessage]:
        """this message as one message a row: itself, then one for each of :attr:`more_rows`.

        :return: the messages, in order
        :rtype: list[CacheInvalidationMessage]
        """
        rows = [self.model_copy(update={"more_rows": None})]
        for row in self.more_rows or ():
            rows.append(
                CacheInvalidationMessage(
                    table=self.table,
                    ids=row.ids,
                    origin=self.origin,
                    l2_current_scope=row.l2_current_scope,
                    generation=self.generation,
                    bump_rows=self.bump_rows,
                    columns=row.columns,
                )
            )
        return rows


@dataclass(frozen=True, slots=True)
class _L2KeptChanges:
    """the epoch change log of a table kept in L2, read through its collection."""

    collection: Any

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the record of the advance that wrote ``token``, from beside the table's generation in L2.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that advance wrote
        :ptype token: str
        :return: the record, or ``None`` when L2 does not hold it
        :rtype: EpochChange | None
        """
        record: EpochChange | None = await self.collection.l2_epoch_change(table_name, token)
        return record

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """the table's epochs after ``after`` up to ``upto``: an L2-kept table numbers them one apart.

        :param table_name: the table
        :ptype table_name: str
        :param after: the mark
        :ptype after: str
        :param upto: the generation read
        :ptype upto: str
        :return: the walk
        :rtype: EpochWalk
        """
        return await walk_by_count(self, table_name, after, upto)


@dataclass(frozen=True, slots=True)
class DerivedCacheRegistration:
    """how one cache derived from a collection table hears that table change.

    :ivar on_row: called with each row message for the table -- a peer's broadcast, and this
        process's own writes -- to evict exactly the entries that row reaches
    :ivar on_table_dropped: called when the table is dropped in this process because the reach of
        a change is unknown; the cache drops every entry derived from the table
    :ivar on_caught_up: called when this registry's mark for the table moves after a catch-up: every
        epoch up to it is applied here, so a cache that stamps what it derives with the mark may take
        it; ``None`` for a cache that keeps no stamp
    :ivar on_first_sight: called when this registry first sights the table's generation and takes it as
        its mark: a cache that keeps its own marks starts them from it. Nothing is dropped: what the
        cache holds was derived after following began. ``None`` falls back to :attr:`on_caught_up`
    :ivar on_partition: called, in place of :attr:`on_table_dropped`, when a whole-table change is a load or a clear
        whose reach the record names (:attr:`~threetears.core.collections.epoch_change.EpochChange.prefixes`, or an
        older writer's :attr:`~threetears.core.collections.epoch_change.EpochChange.partition`): once per key prefix,
        the columns its rows hold and their values. ``None`` drops the table as for any whole-table change
    :ivar on_table_reset: called, in place of :attr:`on_table_dropped`, when the table is reset by an operator (a
        record naming :data:`~threetears.core.collections.epoch_change.RESET_REASON`): a copy reads it again as on a
        first start. ``None`` drops the table
    """

    on_row: Callable[[CacheInvalidationMessage], None]
    on_table_dropped: Callable[[], None]
    on_caught_up: Callable[[], None] | None = None
    on_first_sight: Callable[[], None] | None = None
    on_partition: Callable[[Mapping[str, str]], None] | None = None
    on_table_reset: Callable[[], None] | None = None


def _write_generation_kind(collection: Any) -> str:
    """which of the write-generation declarations a collection carries.

    Read off the instance rather than assumed, because a registry holds whatever was registered
    with it: a stand-in with no declaration is undeclared.

    :param collection: a registered collection, or a stand-in for one
    :ptype collection: Any
    :return: ``"on"``, ``"on_in_l2"``, ``"opted_out"`` or ``"undeclared"``
    :rtype: str
    """
    declared = getattr(collection, "write_generation", None)
    if isinstance(declared, WriteGeneration):
        return "on"
    if isinstance(declared, L2WriteGeneration):
        return "on_in_l2"
    if isinstance(declared, NoWriteGeneration):
        return "opted_out"
    return "undeclared"


class CollectionRegistry:
    """Registry for collection instances with dependency injection.

    Holds default L1/L2/L3 dependencies and per-collection overrides.
    Collections register themselves and resolve their dependencies through
    the registry.
    """

    def __init__(self) -> None:
        self._l1_backend: Any = None
        self._l2_client: Any = None
        self._l3_pool: L3Backend | None = None
        # The leading segment of every L2 key this registry's collections write
        # (``{scope}.{table}.{body}``). One per registry, because a registry is
        # the unit a principal wires: the scope is the sharing boundary, so every
        # collection on one registry must land in one place or replicas of the
        # same principal stop seeing each other's writes.
        self._kv_key_scope: str | None = None
        # Whether a collection on this registry may CREATE the shared collections
        # bucket, or must bind to one somebody else declared.
        #
        # ``True`` is the default and stays the default. ``NatsKvBucket._reopen``
        # re-runs the opener with this flag, and it exists because a single-node
        # NATS restart on ephemeral JetStream storage wiped every bucket and
        # silenced the wake scheduler in production -- a fleet that cannot
        # recreate the bucket runs with L2 off, at WARNING, until whoever
        # declares it restarts. Readers-not-writers is enforced by the NATS
        # GRANT, where a denied create is fail-closed by construction.
        #
        # It lives here rather than as a literal in ``base.py`` so one
        # deployment's bucket-ownership policy is not baked into the library.
        self._l2_create_if_missing: bool = True
        self._collections: dict[str, Any] = {}  # table_name -> collection instance
        self._overrides: dict[str, dict[str, Any]] = {}  # table_name -> {l1_backend, l2_client, l3_pool}
        # The write generation a negative-caching collection stamps its absences with. One per
        # registry: every collection on it answers to the same principal's view of the world.
        self._generation_source: GenerationSource | None = None
        self._write_marks_schema: str | None = None
        self._l3_schema: str | None = None
        # Per followed table, the last write generation whose writes this registry has accounted
        # for. Empty until a table is followed (:meth:`follow_generation`).
        self._generation_marks = GenerationMarks()
        # table -> every follower's answer to whether its generation watch is running now
        self._watching: dict[str, list[Callable[[], bool]]] = {}
        # table -> why this registry is behind on it: an epoch whose change it could not apply. It keeps
        # serving what it holds, its derived caches included, and says so (log, counter, gauge).
        self._behind: dict[str, str] = {}
        # names the keys of an epoch whose writer named too many rows to record, from the change notes: the bus's
        # l3.noted_keys door, taken from an L3 transport that has one (:meth:`configure`), or set
        self._noted_keys: NotedKeysReader | None = None
        # table_name -> the caches derived from it, in registration order.
        self._derived_caches: dict[str, list[DerivedCacheRegistration]] = {}
        #: when the last catch_up_l2_generations pass ran, by time.monotonic(); None while none has
        self._l2_caught_up_at: float | None = None
        # Per-registry (effectively per-pod) identity stamped on every
        # invalidation this registry publishes, so its own listener can
        # skip self-published messages and avoid evicting rows it just
        # wrote. An opaque token, never used as a UUID.
        self._origin_id: str = str(uuid7())  # convert at border: invalidation wire-envelope origin token
        #: whether a write of many rows is broadcast as one message naming them all
        #: (:attr:`CacheInvalidationMessage.more_rows`) rather than one a row. Off until every process on
        #: the bus reads that field: a reader from before it would evict a message's first row only
        self.batch_invalidations = False
        self._scan_cache: ScanCache | None = None
        # The invalidation listener's lifecycle state, set together by
        # :meth:`start_invalidation_listener` and cleared together by
        # :meth:`stop_invalidation_listener`. The subscription being non-``None``
        # IS the "a listener is live" answer -- deliberately not a separate
        # boolean, which would be one more thing to drift out of step with the
        # handle it describes. The client is retained solely so ``stop`` can
        # route the unsubscribe back through it, keeping the client's own
        # subscription bookkeeping correct.
        self._nats_client: NatsClient | None = None
        self._invalidation_subscription: Subscription | None = None

    def configure(
        self,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
        kv_key_scope: str | None = None,
        l2_create_if_missing: bool | None = None,
    ) -> None:
        """Set default dependencies for all collections.

        **Every argument merges into existing state rather than replacing it** -- a ``None``
        leaves whatever was configured before untouched. That is what makes two-pass wiring
        (the scope in one call, the client in a later one) the normal shape it already is at
        several call sites.

        The L2-scope refusal is evaluated over the MERGED state at the end of the call,
        never over this call's arguments. A per-call check would refuse the first pass of
        every two-pass site.

        **The two passes are not interchangeable.** Scope first, then client, works because
        the first pass ends with no client and the second ends with both. CLIENT first
        RAISES: that call ends with a client and no scope, which is exactly the state the
        refusal exists to catch, and it cannot know a later call intends to supply one.
        Either order the scope before the client, or pass both together.

        :param l1_backend: default pod-local L1 backend, or ``None`` to leave it unchanged
        :ptype l1_backend: Any
        :param l2_client: default NATS client backing L2, or ``None`` to leave it unchanged
        :ptype l2_client: Any
        :param l3_pool: default durable L3 backend, or ``None`` to leave it unchanged
        :ptype l3_pool: L3Backend | None
        :param kv_key_scope: the leading segment of every L2 key this registry's collections
            write, from :func:`threetears.nats.kv_key_scope_for`; ``None`` leaves it unchanged
        :ptype kv_key_scope: str | None
        :param l2_create_if_missing: whether this registry's collections may CREATE the
            shared collections bucket. defaults to ``True`` and should be left there until
            the declaring identity re-declares the bucket on every NATS reconnect --
            otherwise a NATS restart that wipes JetStream leaves L2 off fleet-wide, at
            WARNING, until that identity restarts. ``None`` leaves it unchanged
        :ptype l2_create_if_missing: bool | None
        :return: nothing
        :rtype: None
        :raises InvalidL2ScopeError: if ``kv_key_scope`` falls outside the scope grammar
        :raises L2ScopeNotConfiguredError: if the merged state holds an L2 client and no scope
        """
        if l2_create_if_missing is not None:
            self._l2_create_if_missing = l2_create_if_missing
        if kv_key_scope is not None:
            if not KV_KEY_SCOPE_GRAMMAR.match(kv_key_scope):
                raise InvalidL2ScopeError(
                    f"kv_key_scope {kv_key_scope!r} does not match "
                    f"{KV_KEY_SCOPE_GRAMMAR.pattern}; the scope is one NATS subject token, so a "
                    f"'.' or '/' in it renders two and the per-principal $KV grant stops matching"
                )
            self._kv_key_scope = kv_key_scope
        if l1_backend is not None:
            self._l1_backend = l1_backend
            # a cache built over the previous backend would keep answering from
            # it; drop it so the next access rebuilds over the new one.
            self._scan_cache = None
        if l2_client is not None:
            self._l2_client = l2_client
        if l3_pool is not None:
            self._l3_pool = _as_l3_backend(l3_pool)
            if isinstance(l3_pool, NotedKeysReader):
                # the pod's broker transport names a keys-unknown epoch's keys from the notes, on its own tenant
                self._noted_keys = l3_pool
        if self._l2_client is not None and self._kv_key_scope is None:
            raise L2ScopeNotConfiguredError(
                "an L2 client is wired with no kv_key_scope: every collection on this registry "
                "would write into the shared collections bucket under a key no per-principal "
                "grant can name. pass kv_key_scope=threetears.nats.kv_key_scope_for(...)"
            )

    @property
    def kv_key_scope(self) -> str | None:
        """the leading segment of every L2 key this registry's collections write.

        Read by :meth:`BaseCollection.l2_key` on every L2 access. ``None`` means unwired, which
        is legitimate for an L1-only or L3-only registry and is a hard error for one holding an
        L2 client. All three wiring paths are gated: :meth:`configure` and :meth:`bind_table`
        each refuse an L2 client with no scope, and ``l2_key`` backstops the
        ``nats_client=``-direct construction path, which reaches neither.

        :return: the scope segment, or ``None`` when no scope is wired
        :rtype: str | None
        """
        return self._kv_key_scope

    @property
    def l2_create_if_missing(self) -> bool:
        """whether this registry's collections may create the shared collections bucket.

        Read by :meth:`BaseCollection._ensure_kv` when it resolves the bucket. ``True``
        remains the default; ``False`` says this process BINDS the bucket and never
        declares it, which is what a principal whose grant carries ``STREAM.INFO`` and no
        ``STREAM.CREATE`` must say -- a refused create is never answered, so leaving the
        default would spend a JetStream deadline at every startup and then bind anyway.
        Three processes in THIS repo set it -- the registry server's heartbeat registry,
        its rbac registry, and the tool pod -- and the agent pod does too, from the SDK.
        Each pairs it with an eager
        :func:`~threetears.core.collections.bind_collections_bucket`. It also turns off
        :meth:`threetears.nats.NatsKvBucket._reopen`'s restart self-heal, which is safe
        only because the DECLARING IDENTITY -- the hub, in its own lifespan, via
        ``ensure_kv_bucket`` registered on ``NatsClient.add_reconnect_callback`` --
        re-declares the bucket on every NATS reconnect. Nothing else may.

        :return: ``True`` when a collection may create the bucket
        :rtype: bool
        """
        return self._l2_create_if_missing

    def register(
        self,
        collection: Any,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """Register a collection instance with optional per-collection overrides.

        :param collection: the collection, keyed by its ``table_name``; a later one for the same
            table replaces the earlier
        :ptype collection: Any
        :param l1_backend: L1 backend override for this table
        :ptype l1_backend: Any
        :param l2_client: L2 client override for this table
        :ptype l2_client: Any
        :param l3_pool: L3 pool override for this table
        :ptype l3_pool: L3Backend | None
        :return: nothing
        :rtype: None
        :raises ValueError: when the registry already holds a collection for the table that
            declares its write generation differently. One table has one generation, so two
            classes for it that disagree would leave some of its writes advancing it and others
            not, and a follower reading "the generation did not move" as "nothing changed" wrong
        """
        table = collection.table_name
        held = self._collections.get(table)
        if held is not None and held is not collection:
            ours, theirs = _write_generation_kind(collection), _write_generation_kind(held)
            if ours != theirs:
                raise ValueError(
                    f"two collections for table {table!r} disagree on write_generation: "
                    f"{type(held).__name__} is {theirs} and {type(collection).__name__} is {ours}. "
                    f"a table has one write generation; give it one collection class, or declare "
                    f"the same write_generation on both"
                )
        self._collections[table] = collection
        if getattr(collection, "keeps_write_generation_in_l2", False) is True:
            # its generation is in L2 beside its rows, watched (threetears.epoch.follow_l2_kept_tables) or
            # read on demand (catch_up_l2_generations): following it is what makes this registry count
            # the table's broadcasts and its own advances
            self.follow_generation(table)
        if l1_backend or l2_client or l3_pool:
            self._overrides[table] = {}
            if l1_backend:
                self._overrides[table]["l1_backend"] = l1_backend
            if l2_client:
                self._overrides[table]["l2_client"] = l2_client
            if l3_pool:
                self._overrides[table]["l3_pool"] = _as_l3_backend(l3_pool)

    def bind_table(
        self,
        table_name: str,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """pin per-table backend overrides BEFORE the Collection is constructed.

        :class:`BaseCollection.__init__` reads ``l3_pool`` (and friends)
        from the registry via :meth:`get_l3_pool` immediately and then
        auto-registers. :meth:`register`'s ``l3_pool=`` kwarg records
        an override but fires too late -- the Collection has already
        snapped its pool from the registry default. ``bind_table``
        records the override under the table name so the subsequent
        Collection construction reads the intended backend on its
        first :meth:`get_l3_pool` call.

        used by multi-pool agent-side bootstraps (three-tier-task-01
        Phase C2: the rbac metadata Collections need a separate
        :class:`NatsProxyL3Backend` pool bound to ``system.platform.rbac``
        because the broker route different namespaces to different
        schemas). every key left as ``None`` is ignored so callers can
        layer l1 / l2 / l3 bindings independently.

        :param table_name: target table name (matches
            :attr:`BaseCollection.table_name` on the Collection that
            will later be constructed)
        :ptype table_name: str
        :param l1_backend: L1 backend override for this table, or
            ``None`` to leave any existing binding untouched
        :ptype l1_backend: Any
        :param l2_client: L2 client override for this table. requires the registry to
            already carry a ``kv_key_scope`` -- see the raise below
        :ptype l2_client: Any
        :param l3_pool: L3 pool override for this table
        :ptype l3_pool: Any
        :return: nothing
        :rtype: None
        :raises L2ScopeNotConfiguredError: if ``l2_client`` is given while this registry
            has no ``kv_key_scope``. this is the THIRD way an L2 client reaches a
            collection (after :meth:`configure` and ``BaseCollection(nats_client=)``) and
            it was the one with no gate, so a registry wired only through here reached
            production unscoped and failed later, from ``l2_key``, in a request path.
            there is deliberately no per-table ``kv_key_scope`` parameter: the scope
            identifies the PRINCIPAL, and one process is one principal, so a per-table
            value would let a single process write under two identities' scopes -- the
            precise thing scoping exists to prevent. set it on the registry
        """
        if l1_backend is None and l2_client is None and l3_pool is None:
            return
        if l2_client is not None and self._kv_key_scope is None:
            raise L2ScopeNotConfiguredError(
                f"bind_table({table_name!r}, l2_client=...) with no kv_key_scope on this "
                f"registry: that collection would write into the shared collections bucket "
                f"under a key no per-principal grant can name. configure the registry first "
                f"-- registry.configure(kv_key_scope=threetears.nats.kv_key_scope_for(...))"
            )
        existing = self._overrides.setdefault(table_name, {})
        if l1_backend is not None:
            existing["l1_backend"] = l1_backend
        if l2_client is not None:
            existing["l2_client"] = l2_client
        if l3_pool is not None:
            existing["l3_pool"] = _as_l3_backend(l3_pool)

    def get_collection(self, table_name: str) -> Any | None:
        """Look up a registered collection by table name."""
        return self._collections.get(table_name)

    def registered_tables(self) -> tuple[str, ...]:
        """the table of every collection registered here.

        :return: the table names, in registration order
        :rtype: tuple[str, ...]
        """
        return tuple(self._collections)

    def get_l1_backend(self, table_name: str) -> Any:
        """Get L1 backend for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l1_backend", self._l1_backend)

    def set_generation_source(self, source: GenerationSource) -> None:
        """wire the write generation negative-caching collections on this registry stamp absences with.

        Required before constructing any collection that sets
        :attr:`BaseCollection.negative_cache_max_age` and has an L3 pool: without it an absence
        could only be invalidated by timing, which races.

        :param source: the generation source, normally ``threetears.epoch``'s
        :ptype source: GenerationSource
        :return: nothing
        :rtype: None
        """
        self._generation_source = source

    def set_noted_keys_reader(self, reader: NotedKeysReader | None) -> None:
        """name the reader that resolves a keys-unknown epoch's keys from the change notes; ``None`` for none.

        :meth:`configure` takes an L3 transport that is one (the pod's broker transport); this sets another, or
        none, in which case such an epoch leaves this registry behind on its table (``keys_unknown``).

        :param reader: the reader, or ``None``
        :ptype reader: NotedKeysReader | None
        :return: nothing
        :rtype: None
        """
        self._noted_keys = reader

    def behind_reason(self, table_name: str) -> str | None:
        """why this registry is behind on ``table_name`` (a :data:`BEHIND_REASONS` key), ``None`` when it is not.

        :param table_name: the table
        :ptype table_name: str
        :return: the reason, or ``None``
        :rtype: str | None
        """
        return self._behind.get(table_name)

    def set_write_marks_schema(self, schema: str | None) -> None:
        """name the schema holding the database's write-mark queue, so a writer clears the marks its advance covers.

        A switched-on table's trigger marks each write in the write's transaction, and the reconciler
        advances the table's generation from each mark (:mod:`threetears.core.collections.write_marks`).
        A writer that advances after its own commit, and broadcasts, was followed by a second advance from
        its mark that no follower heard, so every follower dropped the table. With the queue named, the
        writer reads the table's marks after its commit, advances, and deletes exactly the marks it read:
        each committed before its advance, which covers it. A writer that dies before the delete leaves
        them to the reconciler. Unset, as before: the reconciler drains every mark. Only a process whose
        L3 holds the queue (the hub over its own database) sets it.

        :param schema: the schema, or ``None`` to leave every mark to the reconciler
        :ptype schema: str | None
        :return: nothing
        :rtype: None
        """
        self._write_marks_schema = schema

    def set_l3_schema(self, schema: str | None) -> None:
        """name the schema this registry's L3 writes land in, so an agent-schema table's records name their schema.

        A process that writes L3 itself and records each write's change (an epoch source that keeps the change
        log) keys a ``TENANT_BY_SCHEMA`` table's records by the schema the write ran in
        (:func:`~threetears.core.collections.generation.advance_for_write`). Unset, such a collection refuses to
        be built, and an advance of one is refused: its records would go on ``_all``, where none of that schema's
        pods read the table. A process whose writes go through the broker (a pod) never needs it: the broker names
        the schema.

        :param schema: the schema, or ``None`` to leave it unnamed
        :ptype schema: str | None
        :return: nothing
        :rtype: None
        """
        self._l3_schema = schema

    @property
    def l3_schema(self) -> str | None:
        """the schema this registry's L3 writes land in (:meth:`set_l3_schema`), or ``None``.

        :return: the schema, or ``None`` when unset
        :rtype: str | None
        """
        return self._l3_schema

    @property
    def write_marks_schema(self) -> str | None:
        """the schema holding the write-mark queue a writer clears its own marks from, or ``None``.

        :return: the schema, or ``None`` when unset
        :rtype: str | None
        """
        return self._write_marks_schema

    @property
    def generation_source(self) -> GenerationSource | None:
        """the write generation source wired on this registry, or ``None``.

        :return: the source, or ``None`` when none is wired
        :rtype: GenerationSource | None
        """
        return self._generation_source

    @property
    def readable_generation_source(self) -> GenerationSource | None:
        """the wired source when it can read a table's generation, for absence caching; else ``None``.

        :return: the source, or ``None`` when none is wired or it can only advance
            (:func:`~threetears.core.collections.generation.source_reads`)
        :rtype: GenerationSource | None
        """
        return self._generation_source if source_reads(self._generation_source) else None

    # ------------------------------------------------------------------
    # Following a table's write generation
    # ------------------------------------------------------------------

    def follow_generation(self, table_name: str) -> None:
        """start keeping this registry's mark for ``table_name``'s write generation.

        A followed table's row broadcasts are counted against the generation each names, and a
        catch-up pass (``threetears.epoch.generation_catchup_tick``, or the key watcher beside it)
        judges the mark against the table's current generation and, when a broadcast was missed,
        applies the missed epochs' rows (:meth:`catch_up_generation`); never a drop. Whoever follows a
        table also schedules the pass. The table need not be registered as a collection -- an agent pod
        follows the access tables it holds only derived caches of.

        :param table_name: the table to follow
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        self._generation_marks.follow(table_name)

    def generation_kept_in_l2(self, table_name: str) -> bool:
        """whether ``table_name``'s write generation is kept in L2 by the collection registered for it.

        Such a table's generation is not in the epoch bucket: :meth:`catch_up_l2_generations` judges it,
        and a pass over the epoch bucket leaves it alone.

        :param table_name: the table
        :ptype table_name: str
        :return: ``True`` when the registered collection keeps its generation in L2
        :rtype: bool
        """
        collection = self._collections.get(table_name)
        return getattr(collection, "keeps_write_generation_in_l2", False) is True

    async def catch_up_l2_generations(self) -> int:
        """one catch-up pass over the followed tables whose write generation is kept in L2, returning how many it caught up.

        Each table's generation is read from L2 (``BaseCollection.l2_write_generation``) and caught up
        (:meth:`catch_up_generation`) from the change records its writers keep beside it in L2
        (``BaseCollection.l2_epoch_change``): each epoch this registry did not hear every row of is
        applied row by row. A record L2 does not hold leaves it behind on the table -- serving what it
        holds, logged and counted -- never a drop. A table seen for the first time takes its mark. A generation that cannot be read is logged and that table is left as it was; the
        next pass reads again. The caller's scheduler drives the cadence, as for
        ``threetears.epoch.generation_catchup_tick``, which runs this pass too.

        :return: how many tables this pass applied missed epochs to
        :rtype: int
        """
        caught_up = 0
        judged = False
        for table_name in self._generation_marks.followed:
            collection = self._collections.get(table_name)
            if not self.generation_kept_in_l2(table_name) or collection is None:
                continue
            try:
                token = await collection.l2_write_generation()
            except GenerationUnavailableError as exc:
                # NOSILENT: nothing is applied and the mark does not move; the next pass reads again
                log.warning(
                    "write generation unavailable in L2; the table is left as it is and the next pass reads again",
                    extra={"extra_data": {"table": table_name, "error": str(exc)}},
                )
                continue
            judged = True
            # the records are beside the generation, in L2, and the collection reads them
            verdict = await self.catch_up_generation(table_name, token, changes=_L2KeptChanges(collection))
            if verdict is GenerationVerdict.MISSED and not self.behind_on(table_name):
                caught_up += 1
        if judged:
            # a pass that could read no generation judged nothing, and does not count as run
            self._l2_caught_up_at = time.monotonic()
        return caught_up

    @property
    def l2_catch_up_age(self) -> timedelta | None:
        """how long since the last :meth:`catch_up_l2_generations` pass ran here, for a health surface.

        :return: the time since it ran; ``None`` when no pass has run
        :rtype: timedelta | None
        """
        if self._l2_caught_up_at is None:
            return None
        return timedelta(seconds=time.monotonic() - self._l2_caught_up_at)

    def l2_kept_tables(self) -> tuple[str, ...]:
        """the followed tables whose generation is kept in L2, for a host that watches each one.

        :return: their names
        :rtype: tuple[str, ...]
        """
        return tuple(table for table in self._generation_marks.followed if self.generation_kept_in_l2(table))

    def watched_by(self, table_name: str, watching: Callable[[], bool]) -> None:
        """say that a follower watches ``table_name``'s generation, and how to ask whether it still does.

        Called by each follower that runs a watch on the table (``threetears.epoch.TableFollower``); more than
        one may. A cache derived from the table trusts what it holds only while every one of them
        answers ``True`` (:meth:`tables_trusted`). The follower withdraws with :meth:`not_watched_by`.

        :param table_name: the table
        :ptype table_name: str
        :param watching: answers whether this follower's watch is running now
        :ptype watching: Callable[[], bool]
        :return: nothing
        :rtype: None
        """
        self._watching.setdefault(table_name, []).append(watching)

    def not_watched_by(self, table_name: str, watching: Callable[[], bool]) -> None:
        """withdraw one follower's watch on ``table_name``, as given to :meth:`watched_by`. Idempotent.

        :param table_name: the table
        :ptype table_name: str
        :param watching: the callable the follower registered
        :ptype watching: Callable[[], bool]
        :return: nothing
        :rtype: None
        """
        followers = self._watching.get(table_name)
        if followers is None:
            return
        remaining = [each for each in followers if each is not watching]
        if remaining:
            self._watching[table_name] = remaining
        else:
            del self._watching[table_name]

    def tables_trusted(self, table_names: Sequence[str]) -> bool:
        """whether every one of ``table_names`` is followed here and every watch on it is running.

        A cache derived from them may serve and store only while this holds: a broadcast that
        never arrives is caught by the table's generation only while it is followed and watched,
        and a table one follower claims is not trusted while that follower's watch is down, even
        when another's runs.

        :param table_names: the tables
        :ptype table_names: Sequence[str]
        :return: ``True`` when every table is followed and all its watches run now
        :rtype: bool
        """
        for table in table_names:
            followers = self._watching.get(table)
            if not self._generation_marks.follows(table) or not followers:
                return False
            if not all(watching() for watching in followers):
                return False
        return True

    @property
    def generation_marks(self) -> GenerationMarks:
        """this registry's per-table generation marks, for the pass that judges them.

        :return: the marks
        :rtype: GenerationMarks
        """
        return self._generation_marks

    def account_generation(self, table_name: str, token: str) -> None:
        """record an advance one of this registry's own writes made.

        A registry skips its own broadcasts, so the collection that advanced a generation says so
        here; otherwise a registry that both writes and follows a table would find every one of
        its own writes unheard.

        :param table_name: the table advanced
        :ptype table_name: str
        :param token: the generation the advance wrote
        :ptype token: str
        :return: nothing
        :rtype: None
        """
        self._generation_marks.account(table_name, token)

    async def catch_up_generation(
        self, table_name: str, token: str | None, *, changes: EpochChangeReader | None
    ) -> GenerationVerdict:
        """judge ``table_name``'s generation as just read, and apply every epoch this registry has not heard.

        The one operation a follower runs, at startup, on every generation pushed or read, and after
        a gap: fetch the change record of each advance after this registry's mark that it did not hear
        every row of, in order, and apply exactly its rows (:meth:`apply_epoch_change`). A gap never
        drops the table.

        - :attr:`~GenerationVerdict.CURRENT`: nothing to apply.
        - :attr:`~GenerationVerdict.FIRST_SIGHT`: the mark is taken at the generation read, the head.
          A follower starts before it serves the table, so it holds nothing older; nothing is re-read.
        - :attr:`~GenerationVerdict.MISSED`: each unheard epoch's record is applied and the mark moves.
          The epochs are read by walking the table's records after the mark (``changes_after``). A
          record L2 does not hold holds the mark there, while the epochs after it are still applied:
          the registry is behind on the table, keeps serving what it holds, logs and counts it once,
          and the next judgement reads the missing record again.
        - :attr:`~GenerationVerdict.REPLACED`: the generation's store was replaced, so the epochs
          between the mark and the replacement cannot be named. The registry is behind, as above,
          until the mark's own epochs can be read again.

        :param table_name: a followed table
        :ptype table_name: str
        :param token: the table's current generation, or ``None`` when its store holds none
        :ptype token: str | None
        :param changes: reads each epoch's change record from L2; ``None`` when this registry has no
            reader, so every unheard epoch leaves it behind
        :ptype changes: EpochChangeReader | None
        :return: the verdict before catching up; :attr:`~GenerationVerdict.CURRENT` too when every epoch
            since the mark had been heard in full, so nothing was applied
        :rtype: GenerationVerdict
        :raises KeyError: when ``table_name`` is not followed
        :raises GenerationUnavailableError: when a record cannot be read; nothing after it is applied
        """
        marks = self._generation_marks
        verdict = marks.judge(table_name, token)
        extent = changes if isinstance(changes, EpochLogExtent) else None
        if verdict is GenerationVerdict.REPLACED and token is not None and extent is not None:
            # a restored log holds what the change notes say changed after their floor: a mark at or past it
            # is carried to the restored log's start and walks every record; an older one is behind
            carried = await self._carry_over_restore(table_name, token, extent)
            if carried is None:
                return verdict
            verdict = carried
        if verdict is GenerationVerdict.CURRENT:
            recorded = marks.recorded(table_name)
            if extent is not None and recorded is not None:
                quiet = await extent.quiet_after(table_name, recorded)
                if quiet.aged_out:
                    # retention took the epochs after the mark: whether the table was written since, nobody
                    # can say, so this registry is behind on it -- serving, said and counted, never current
                    self._fall_behind(table_name, reason="aged_out", detail={"read": token, "recorded": recorded})
                    return verdict
                if quiet.through is not None:
                    # the log holds nothing of the table after the mark: the mark moves with the log, so a
                    # quiet table's follower stays inside its retention
                    marks.advance_to(table_name, quiet.through)
                    self._tell_caught_up(table_name)
            self._caught_up(table_name)
            return verdict
        if verdict is GenerationVerdict.FIRST_SIGHT:
            # the head is the mark: a process starts following before it serves, so it holds nothing
            # older to bring forward, and nothing is re-read from any tier
            marks.adopt_generation(table_name, token)
            if extent is not None and token is not None:
                quiet = await extent.quiet_after(table_name, token)
                if quiet.through is not None:
                    marks.advance_to(table_name, quiet.through)
            # a cache derived from the table keeps its own marks, and starts them from this one; like the
            # table's own rows, what it holds was derived after following began, so nothing is dropped
            for registration in tuple(self._derived_caches.get(table_name, ())):
                first_sight = registration.on_first_sight or registration.on_caught_up
                if first_sight is not None:
                    first_sight()
            self._caught_up(table_name)
            return verdict
        if verdict is GenerationVerdict.REPLACED or token is None:
            self._fall_behind(
                table_name,
                reason="replaced",
                detail={"read": token, "recorded": marks.recorded(table_name)},
            )
            return verdict
        assert token is not None  # narrow: a MISSED verdict reads a generation
        recorded = marks.recorded(table_name)
        after = recorded if recorded is not None else f"{token.rpartition(':')[0]}:0"
        if changes is None:
            incarnation, _, count = after.rpartition(":")
            walk = EpochWalk(changes=(), complete=False, first_missing=f"{incarnation}:{int(count) + 1}")
        else:
            try:
                walk = await changes.changes_after(table_name, after, token)
            except GenerationUnavailableError as exc:
                self._fall_behind(table_name, reason="record_unreadable", detail={"after": after, "error": str(exc)})
                raise
        # one hole holds the mark, never the epochs after it: they are applied, and the next walk skips
        # them as heard. A record whose writer named no rows is a hole no later read fills; it is told
        # apart in the log and the counter from a record the log does not hold.
        hole: tuple[str, str] | None = None if walk.complete else (walk.first_missing or token, walk.reason)
        reached: str | None = None
        applied_any = False
        for change in walk.changes:
            if change.keys_unknown and not marks.heard_in_full(table_name, change.token):
                # a commit too large to record names its keys in the change notes: read them, a page at a time
                why = await self._apply_noted_keys(change)
                if why is not None:
                    hole = hole or (change.token, why)
                    continue
                if hole is None:
                    reached = change.token
                marks.applied(table_name, change.token)
                applied_any = True
                _epochs_applied().labels(table=table_name).inc()
                continue
            if hole is None:
                reached = change.token
            if marks.heard_in_full(table_name, change.token):
                continue
            if change.whole_table:
                # the mark moves first, so a cache derived from the table that re-bases on the drop
                # starts from this epoch rather than distrusting the one it is already at
                marks.applied(table_name, change.token)
                await self.apply_epoch_change(change)
            else:
                await self.apply_epoch_change(change)
                marks.applied(table_name, change.token)
            applied_any = True
            _epochs_applied().labels(table=table_name).inc()
        if hole is not None:
            if reached is not None:
                marks.advance_to(table_name, reached)
                self._tell_caught_up(table_name)
            self._fall_behind(
                table_name,
                reason=hole[1],
                detail={"epoch": hole[0], "read": token, "recorded": marks.recorded(table_name)},
            )
            return verdict
        marks.advance_to(table_name, token)
        self._tell_caught_up(table_name)
        self._caught_up(table_name)
        # every epoch since the mark was heard in full: the broadcasts had already done the catching up
        return verdict if applied_any else GenerationVerdict.CURRENT

    async def apply_epoch_change(self, change: EpochChange) -> None:
        """apply one epoch's change here, exactly as hearing each of its row broadcasts would.

        A record that names the whole table invalidates the table, because that is what its write did.

        :param change: the epoch's change record
        :ptype change: EpochChange
        :return: nothing
        :rtype: None
        """
        if change.whole_table:
            if change.reason == RESET_REASON:
                self.drop_table(change.table, reason=f"epoch {change.token} reset the table", why="reset")
            elif change.prefixes:
                self.drop_partitions(
                    change.table, change.prefixes, reason=f"epoch {change.token} loaded or cleared key prefixes"
                )
            elif change.partition:
                self.drop_partition(change.table, change.partition, reason=f"epoch {change.token} loaded a partition")
            else:
                self.drop_table(
                    change.table, reason=f"epoch {change.token} changed the whole table", why="whole_record"
                )
            return
        for row in change.rows:
            # stamped with the epoch, as its broadcast was: a derived cache counts it toward the advance
            await self._apply_row_message(
                CacheInvalidationMessage(
                    table=change.table,
                    ids=list(row.ids),
                    l2_current_scope=row.l2_current_scope,
                    generation=change.token,
                    bump_rows=len(change.rows),
                    columns=dict(row.columns) if row.columns is not None else None,
                )
            )

    async def _apply_noted_keys(self, change: EpochChange) -> str | None:
        """apply a keys-unknown epoch's rows as the change notes name them, each as its row message would be.

        The record's bracket (:attr:`~threetears.core.collections.epoch_change.EpochChange.noted_between`) holds
        every note its commit wrote; the reader names them a page at a time, each page applied before the next is
        read, so memory holds one page. Keys another commit wrote in the bracket are evicted too, which costs a
        read again and is never wrong. A whole-table note in the bracket is another commit's truncate or load,
        recorded as its own whole-table epoch.

        :param change: a keys-unknown record
        :ptype change: EpochChange
        :return: ``None`` once every page is applied; else the :data:`BEHIND_REASONS` key of why the keys cannot
            be named (no bracket or no reader: ``keys_unknown``)
        :rtype: str | None
        """
        reader = self._noted_keys
        if change.noted_between is None or reader is None:
            return "keys_unknown"
        async with aclosing(_noted_pages(reader, change.table, change.noted_between)) as pages:
            async for page in pages:
                if isinstance(page, str):
                    return page
                columns = page.columns or (None,) * len(page.keys)
                for index, (ids, held) in enumerate(zip(page.keys, columns, strict=True)):
                    if index and not index % _NOTED_ROWS_PER_YIELD:
                        # a page of many rows yields between runs of them, so the loop answers everything else
                        # between
                        await asyncio.sleep(0)
                    await self._apply_row_message(
                        CacheInvalidationMessage(
                            table=change.table, ids=list(ids), columns=None if held is None else dict(held)
                        )
                    )
        return None

    async def resolve_behind_from_notes(self, table_name: str, since: datetime, token: str) -> str | None:
        """end being behind on ``table_name`` from its change notes, never by reading the table: every key its notes
        name since ``since`` is applied as its row message would be, and ``token`` becomes its mark.

        For a follower that knows when it last held the table current (``since``, on its own clock; the bracket
        widens by :data:`_NOTES_CLOCK_SKEW` each way): every commit after its mark wrote notes in it, and every commit
        up to ``token`` (read before the notes) is among them. Keys another commit wrote in the bracket are read again
        harmlessly. A whole-table note in it (a reset, a file load, a large copy) names no reach: the table stays
        behind (``notes_whole``) and is never read whole for it.

        :param table_name: a table this registry is behind on
        :ptype table_name: str
        :param since: when the follower last held it current, timezone-aware
        :ptype since: datetime
        :param token: the table's generation, read before this call
        :ptype token: str
        :return: ``None`` once resolved; else the :data:`BEHIND_REASONS` key of why the notes cannot say
        :rtype: str | None
        """
        reader = self._noted_keys
        if reader is None:
            return "notes_unavailable"
        bracket = (since - _NOTES_CLOCK_SKEW, datetime.now(tz=since.tzinfo) + _NOTES_CLOCK_SKEW)
        keys = 0
        async with aclosing(_noted_pages(reader, table_name, bracket)) as pages:
            async for page in pages:
                if isinstance(page, str):
                    return page
                if page.whole_table:
                    return "notes_whole"
                columns = page.columns or (None,) * len(page.keys)
                for index, (ids, held) in enumerate(zip(page.keys, columns, strict=True)):
                    if index and not index % _NOTED_ROWS_PER_YIELD:
                        await asyncio.sleep(0)
                    await self._apply_row_message(
                        CacheInvalidationMessage(
                            table=table_name, ids=list(ids), columns=None if held is None else dict(held)
                        )
                    )
                    keys += 1
        self._generation_marks.adopt_generation(table_name, token)
        self._caught_up(table_name)
        self._tell_caught_up(table_name)
        log.info(
            "caught up on a table from its change notes, without reading it",
            extra={"extra_data": {"table": table_name, "since": since.isoformat(), "keys": keys, "token": token}},
        )
        return None

    def behind_on(self, table_name: str) -> bool:
        """whether this registry is behind on ``table_name``: an epoch it has not been able to apply.

        :param table_name: the table
        :ptype table_name: str
        :return: ``True`` while an epoch of the table awaits its change record
        :rtype: bool
        """
        return table_name in self._behind

    async def _carry_over_restore(
        self, table_name: str, token: str, extent: EpochLogExtent
    ) -> GenerationVerdict | None:
        """carry the mark over to a restored log that continues the mark's own, or say this registry is behind.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation read, under the restored log's incarnation
        :ptype token: str
        :param extent: what the log can say of its own reach
        :ptype extent: EpochLogExtent
        :return: the verdict against the carried mark; :attr:`~GenerationVerdict.REPLACED` when the log
            continues another incarnation than the mark's; ``None`` when this registry fell behind
        :rtype: GenerationVerdict | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        marks = self._generation_marks
        restored = token.rpartition(":")[0]
        lineage = await extent.continues(restored)
        recorded = marks.recorded(table_name)
        if lineage is None or recorded is None or recorded.rpartition(":")[0] != lineage[0]:
            return GenerationVerdict.REPLACED
        lost, covered = lineage
        if marks.continue_under(table_name, lost, restored, covered):
            return marks.judge(table_name, token)
        # the mark is older than the notes reach: an epoch it missed may be one no note names
        self._fall_behind(
            table_name,
            reason="record_lost",
            detail={
                "read": token,
                "recorded": recorded,
                "covered_from": None if covered is None else f"{lost}:{covered}",
            },
        )
        return None

    def _fall_behind(self, table_name: str, *, reason: str, detail: Mapping[str, Any]) -> None:
        """record that the catch-up of ``table_name`` stopped short, and say so where an operator will see it.

        :param table_name: the table
        :ptype table_name: str
        :param reason: why
        :ptype reason: str
        :param detail: what the judgement saw, for the log
        :ptype detail: Mapping[str, Any]
        :return: nothing
        :rtype: None
        """
        if self._behind.get(table_name) == reason:
            # said once when it began; the gauge says how long it lasts
            return
        self._behind[table_name] = reason
        _epochs_behind().labels(table=table_name, reason=reason).inc()
        _tables_behind().set(len(self._behind))
        log.error(
            "behind on a table: an epoch's change could not be applied; serving what this process holds. "
            f"{reason}: {BEHIND_REASONS.get(reason, 'see collection_epoch_behind_total')}",
            extra={"extra_data": {"table": table_name, "reason": reason, **detail}},
        )

    def _tell_caught_up(self, table_name: str) -> None:
        """tell every cache derived from ``table_name`` that this registry's mark for it moved.

        :param table_name: the table
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        for registration in tuple(self._derived_caches.get(table_name, ())):
            if registration.on_caught_up is not None:
                registration.on_caught_up()

    def _caught_up(self, table_name: str) -> None:
        """record that ``table_name`` is caught up to the generation last judged.

        :param table_name: the table
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        if self._behind.pop(table_name, None) is not None:
            _tables_behind().set(len(self._behind))
            log.info("caught up on a table after being behind", extra={"extra_data": {"table": table_name}})

    def drop_table(self, table_name: str, *, reason: str, why: str = "local") -> None:
        """drop everything this process holds of ``table_name``, because a write changed the whole table.

        For a write whose reach is the table itself: a truncate, a file load, an announced migration,
        here or (through its epoch's change record) in another process. A missed broadcast never
        comes here; it is caught up epoch by epoch (:meth:`catch_up_generation`). Dropped, in this
        process only:

        - every cached scan that depends on the table;
        - every row of it in L1, through its collection, which also stops trusting this
          process's own L2 entries for the table (:meth:`BaseCollection.drop_cached_table`);
        - every entry of every cache derived from it (:meth:`register_derived_cache`).

        Always safe: the next read of anything dropped goes back to the durable tier.

        :param table_name: the table to drop
        :ptype table_name: str
        :param reason: why, for the log
        :ptype reason: str
        :param why: a :data:`WHOLE_TABLE_DROP_WHYS` key, as ``collection_whole_table_drops_total`` counts it. A reset
            tells each derived cache that takes one (``on_table_reset``) in place of dropping it
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        WHOLE_TABLE_DROPS.count(why, table=table_name)
        self.drop_local_scans(table_name)
        collection = self._collections.get(table_name)
        rows = collection.drop_cached_table() if collection is not None else 0
        derived = tuple(self._derived_caches.get(table_name, ()))
        for registration in derived:
            if why == "reset" and registration.on_table_reset is not None:
                registration.on_table_reset()
            else:
                registration.on_table_dropped()
        log.warning(
            "dropped a table from this process's caches; the next read of it goes to the durable tier",
            extra={
                "extra_data": {
                    "table": table_name,
                    "reason": reason,
                    "why": why,
                    "l1_rows": rows,
                    "derived_caches": len(derived),
                }
            },
        )

    def drop_partitions(self, table_name: str, prefixes: Sequence[Mapping[str, str]], *, reason: str) -> None:
        """:meth:`drop_partition` for a load or a clear that reached several key prefixes: the table's own caches drop
        it once, and each derived cache that takes a partition is told each prefix.

        :param table_name: the table
        :ptype table_name: str
        :param prefixes: each key prefix, the leading key columns and their values as text
        :ptype prefixes: Sequence[Mapping[str, str]]
        :param reason: why, for the log
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self._drop_reach(table_name, prefixes, reason=reason)

    def drop_partition(self, table_name: str, partition: Mapping[str, str], *, reason: str) -> None:
        """what a load from files whose reach is known does here: the table's own caches drop it whole, and each
        derived cache that takes a partition is told it in place of dropping everything.

        A load's rows all hold its constants (:attr:`~threetears.core.collections.epoch_change.EpochChange.partition`),
        so a cache that keeps the table whole (a copy follower) reads those rows again rather than the table. The
        collection's own L1, its cached scans and every derived cache without ``on_partition`` drop as
        :meth:`drop_table` drops them.

        :param table_name: the table
        :ptype table_name: str
        :param partition: the columns the load's rows hold, and their values as text
        :ptype partition: Mapping[str, str]
        :param reason: why, for the log
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self._drop_reach(table_name, [partition], reason=reason)

    def _drop_reach(self, table_name: str, prefixes: Sequence[Mapping[str, str]], *, reason: str) -> None:
        """drop the table's own caches once and tell each derived cache each prefix (or drop it, when it takes none).

        :param table_name: the table
        :ptype table_name: str
        :param prefixes: the reach
        :ptype prefixes: Sequence[Mapping[str, str]]
        :param reason: why, for the log
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self.drop_local_scans(table_name)
        collection = self._collections.get(table_name)
        rows = collection.drop_cached_table() if collection is not None else 0
        derived = tuple(self._derived_caches.get(table_name, ()))
        for registration in derived:
            if registration.on_partition is not None:
                for prefix in prefixes:
                    registration.on_partition(dict(prefix))
            else:
                registration.on_table_dropped()
        log.info(
            "a load's partition reached this process: derived copies read it again, every other cache dropped the table",
            extra={
                "extra_data": {
                    "table": table_name,
                    "reason": reason,
                    "prefixes": [dict(prefix) for prefix in prefixes[:20]],
                    "prefix_count": len(prefixes),
                    "l1_rows": rows,
                    "derived_caches": len(derived),
                }
            },
        )

    def register_derived_cache(
        self,
        table_name: str,
        *,
        on_row: Callable[[CacheInvalidationMessage], None],
        on_table_dropped: Callable[[], None],
        on_caught_up: Callable[[], None] | None = None,
        on_first_sight: Callable[[], None] | None = None,
        on_partition: Callable[[Mapping[str, str]], None] | None = None,
        on_table_reset: Callable[[], None] | None = None,
    ) -> Callable[[], None]:
        """tell a cache derived from ``table_name`` about each row that changes, and when to drop it all.

        A derived cache holds answers computed from a collection table's rows -- who may see
        what, resolved from the access tables -- under keys that are not the table's primary
        key. It registers, per table, how one row message reaches its own entries:

        - ``on_row`` gets every row message for the table: each peer's broadcast, and each write
          this process makes itself (a registry skips its own broadcasts, and the writer's cache
          is as stale as anyone's). It evicts the entries that row reaches and nothing else. The
          message's ``columns`` carry what the publishing collection declared in
          :attr:`BaseCollection.invalidation_columns`; a message without them names only the
          primary key.
        - ``on_table_dropped`` is called only when the table is dropped here
          (:meth:`drop_table`). That is the one time the cache empties everything derived from
          the table.

        Both run synchronously inside the handling of the change and must not block. One that
        raises fails that handling: a peer's row is then not counted as heard, and the next
        catch-up pass drops the table.

        :param table_name: the table the cache is derived from
        :ptype table_name: str
        :param on_row: evicts the entries one row message reaches
        :ptype on_row: Callable[[CacheInvalidationMessage], None]
        :param on_table_dropped: drops every entry derived from the table
        :ptype on_table_dropped: Callable[[], None]
        :param on_caught_up: told when the registry's mark for the table moves after a catch-up, so a
            cache stamping what it derives with the mark takes the new one; ``None`` for none
        :ptype on_caught_up: Callable[[], None] | None
        :param on_first_sight: told when the registry first sights the table and takes its mark, so a cache
            keeping its own marks starts them there; ``None`` uses ``on_caught_up``. A first sight drops nothing
        :ptype on_first_sight: Callable[[], None] | None
        :param on_partition: told, in place of ``on_table_dropped``, the reach of a load from files the record names
            (the columns its rows hold and their values), so the cache drops only those rows; ``None`` drops it all
        :ptype on_partition: Callable[[Mapping[str, str]], None] | None
        :param on_table_reset: told, in place of ``on_table_dropped``, an operator's reset of the table, so a copy
            reads it again as on a first start; ``None`` drops it all
        :ptype on_table_reset: Callable[[], None] | None
        :return: a call that removes the registration
        :rtype: Callable[[], None]
        """
        registration = DerivedCacheRegistration(
            on_row=on_row,
            on_table_dropped=on_table_dropped,
            on_caught_up=on_caught_up,
            on_first_sight=on_first_sight,
            on_partition=on_partition,
            on_table_reset=on_table_reset,
        )
        registrations = self._derived_caches.setdefault(table_name, [])
        registrations.append(registration)

        def remove() -> None:
            if registration in registrations:
                registrations.remove(registration)

        return remove

    def has_derived_caches(self, table_name: str) -> bool:
        """whether any cache derived from ``table_name`` is registered here.

        For a caller about to build row messages only to hand them to derived caches: a bulk
        load on a registry with none builds nothing.

        :param table_name: the table
        :ptype table_name: str
        :return: ``True`` when at least one derived cache is registered for the table
        :rtype: bool
        """
        return bool(self._derived_caches.get(table_name))

    def tell_derived_caches(self, message: CacheInvalidationMessage) -> None:
        """hand one row message to every cache derived from its table.

        :param message: the row message
        :ptype message: CacheInvalidationMessage
        :return: nothing
        :rtype: None
        """
        for registration in tuple(self._derived_caches.get(message.table, ())):
            registration.on_row(message)

    def row_message(
        self,
        table_name: str,
        entity_id: Any,
        *,
        l2_key_current: bool = False,
        generation: str | None = None,
        bump_rows: int | None = None,
        columns: Mapping[str, str | None] | None = None,
    ) -> CacheInvalidationMessage:
        """build the row message this registry publishes for one changed row.

        :param table_name: the row's table
        :ptype table_name: str
        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param l2_key_current: whether a revision-fenced write left this registry's scoped L2 key
            holding the newest value
        :ptype l2_key_current: bool
        :param generation: the write generation the row's commit advanced to, or ``None``
        :ptype generation: str | None
        :param bump_rows: how many row messages that advance covers, or ``None``
        :ptype bump_rows: int | None
        :param columns: the declared invalidation columns' values, or ``None``
        :ptype columns: Mapping[str, str | None] | None
        :return: the message, stamped with this registry's origin
        :rtype: CacheInvalidationMessage
        """
        values = entity_id if isinstance(entity_id, tuple) else (entity_id,)
        return CacheInvalidationMessage(
            table=table_name,
            ids=[key_text(v) for v in values],  # convert at border: invalidation wire-envelope pk values
            origin=self._origin_id,
            l2_current_scope=self._kv_key_scope if l2_key_current else None,
            generation=generation,
            bump_rows=bump_rows if generation is not None else None,
            columns=dict(columns) if columns is not None else None,
        )

    @property
    def scan_cache(self) -> ScanCache:
        """the pod's visibility-scan cache, created on first use.

        Built over the DEFAULT L1 backend rather than a per-table override: a
        scan spans tables (its visibility predicate reads the RBAC tables), so
        it has no single owning table's backend to live in. It caches a scan only while every
        table the scan depends on is followed with its watch running (:meth:`tables_trusted`).

        :return: the scan cache
        :rtype: ScanCache
        """
        if self._scan_cache is None:
            self._scan_cache = ScanCache(self._l1_backend, trusted=self.tables_trusted)
        return self._scan_cache

    def get_l2_client(self, table_name: str) -> Any:
        """Get L2 client for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l2_client", self._l2_client)

    def get_l3_pool(self, table_name: str) -> L3Backend | None:
        """Get L3 pool for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return cast("L3Backend | None", overrides.get("l3_pool", self._l3_pool))

    # ------------------------------------------------------------------
    # Cache coherence -- cross-pod L1 invalidation via typed NATS pub/sub
    # ------------------------------------------------------------------

    async def start_invalidation_listener(self, nats_client: NatsClient) -> None:
        """subscribe to cache invalidation signals from other pods.

        wire envelope is :class:`CacheInvalidationMessage` -- typed
        Pydantic, serialized via ``model_dump_json()``. on receipt the
        listener evicts the named entity from local L1 cache for the
        matching collection. callers must invoke after every
        Collection has registered.

        narrow exception scope: malformed payloads (Pydantic
        :class:`ValidationError`) and unknown-table receipts log and
        skip; programming errors (``AttributeError`` / ``TypeError``)
        propagate so wrapper drift / collection-misregistration
        surfaces immediately rather than as silent
        invalidation-skips.

        **idempotent while a listener is live.** a second call is a
        no-op rather than a second consumer: two subscriptions on one
        subject in one process means every broadcast is handled twice,
        and a teardown that releases one of them leaves the other
        running. call :meth:`stop_invalidation_listener` first to
        rebind to a different client.

        :param nats_client: connected typed NATS wrapper client
        :ptype nats_client: NatsClient
        :return: nothing
        :rtype: None
        :raises SubscribeError: if the underlying subscribe fails to
            register (transport / config error)
        """
        if self._invalidation_subscription is not None:
            log.debug(
                "Invalidation listener already running",
                extra={"extra_data": {"subject": Subjects.cache_invalidate().path}},
            )
            return

        async def _on_invalidation(message: CacheInvalidationMessage) -> None:
            # Skip invalidations this registry published itself. A single
            # pod both publishes and subscribes on this subject; acting on
            # our own broadcast would evict the L1 row save_entity just
            # wrote, so a subsequent L1-only read of that entity (e.g. an
            # entity field accessor) would miss. The broadcast exists to
            # evict OTHER pods, which carry a different origin.
            if message.origin is not None and message.origin == self._origin_id:
                return
            # a message naming many rows is heard as one message a row, in order
            for row in message.each_row() if message.more_rows else (message,):
                await self._apply_row_message(row)
                # Counted as heard only once everything above has run to the end: a row whose
                # eviction raised part-way is one this registry cannot vouch for, and leaving it
                # uncounted is what makes the next catch-up apply its epoch's change record. A message naming no
                # generation -- a table not switched on, an older publisher -- counts nothing.
                if row.generation is not None and row.bump_rows is not None:
                    self._generation_marks.hear(row.table, row.generation, row.bump_rows)

        # SubscribeError propagates deliberately: cache coherence is not
        # optional, so a process that cannot subscribe must fail its startup
        # rather than run on silently as an island. Both fields stay unset on
        # that path, so a failed start leaves nothing behind for ``stop`` to
        # find and a retry is a clean first start.
        # sender-posture: unauthenticated open S3 -- the body names what to evict and the L2 scope
        subscription = await nats_client.subscribe_typed(
            subject=Subjects.cache_invalidate(),
            message_type=CacheInvalidationMessage,
            cb=_on_invalidation,
        )
        self._nats_client = nats_client
        self._invalidation_subscription = subscription

    async def _apply_row_message(self, message: CacheInvalidationMessage) -> None:
        """evict one changed row from everything this process holds of it: scans, derived caches, L2, L1.

        What hearing a peer's row broadcast does, and what applying an epoch's change record does for
        each of its rows.

        :param message: the row message
        :ptype message: CacheInvalidationMessage
        :return: nothing
        :rtype: None
        """
        # Evict dependent SCANS first, and unconditionally. Every check
        # below returns early for a table this pod does not hold as a
        # collection -- and `role_assignments` / `group_members` are exactly
        # that on an agent pod, while being precisely the tables whose write
        # must drop a cached visibility scan. Ordering this after those
        # guards would leave a revoked grant readable until the TTL lapsed.
        self.scan_cache.drop_for_table(message.table)
        # Derived caches next, and for the same reason ahead of every guard below: a cache
        # derived from the access tables lives on pods that hold none of them as collections.
        # One that raises must not cost the row its eviction, so the row is evicted either
        # way and the failure then surfaces, leaving the message uncounted.
        try:
            self.tell_derived_caches(message)
        finally:
            await self._evict_collection_row(message)

    async def _evict_collection_row(self, message: CacheInvalidationMessage) -> None:
        """evict one changed row from its collection's L2 entry and L1, when this process holds the table.

        :param message: the row message
        :ptype message: CacheInvalidationMessage
        :return: nothing
        :rtype: None
        """
        collection = self._collections.get(message.table)
        if collection is None:
            # unknown-table receipts are expected during partial
            # rollouts (sender has a Collection the receiver does
            # not). log + skip without warning.
            return

        pk_cols = collection.primary_key_columns
        if len(message.ids) != len(pk_cols):
            log.warning(
                "Invalidation pk arity mismatch",
                extra={
                    "extra_data": {
                        "table": message.table,
                        "expected_columns": list(pk_cols),
                        "received_values": len(message.ids),
                    },
                },
            )
            return
        entity_id = tuple(message.ids)

        # EVICT L2 HERE, ahead of the L1 guards below, and the placement is the
        # requirement rather than a preference.
        #
        # Keys are per-principal (``{scope}.{table}.{body}``), so the writer updates
        # only ITS OWN key. A peer that drops L1 and pulls through reads L2 first --
        # its own, still-stale key -- and re-caches the value the broadcast was sent
        # to retract. ``max_age`` on the bucket is unlimited and no collection sets an
        # L1 bound, so nothing heals it: a revoked grant would be enforced forever,
        # which is worse than the exposure per-principal keys exist to close. Under
        # the old single shared key this could not happen (the peer read the writer's
        # fresh value), so the eviction is a scoping-induced obligation and lands with
        # it.
        #
        # Behind ``l1 is None`` / ``not l1.has_table(...)`` it would skip exactly the
        # collections whose L1 schema was never initialised -- and those are precisely
        # the ones that would then keep the stale L2 value forever. L2 presence is
        # independent of L1 presence.
        #
        # NOT ``invalidate_cache``, which looks right and RE-PUBLISHES: the origin
        # filter above only skips SELF, so every receiver would rebroadcast under its
        # own origin, unbounded.
        #
        # The arity check is hoisted above this rather than left where it was, because
        # ``l2_key`` normalises the pk and raises on a mismatch; it touches no L1.
        #
        # Skipped only when a revision-fenced writer says it left THIS scope's key current:
        # the key is shared, already newest, and deleting it would discard the only copy of a
        # compare-and-swap value that L3 may not hold yet.
        if message.l2_current_scope is None or message.l2_current_scope != self._kv_key_scope:
            await collection.delete_l2_entry(entity_id)

        l1 = self.get_l1_backend(message.table)
        if l1 is None:
            return
        if hasattr(l1, "has_table") and not l1.has_table(message.table):
            # this pod's L1 backend was never initialize()'d with this
            # table's schema -- its OWN collections never touch it
            # locally, so there is nothing to evict. Same "unknown
            # receipts are expected during partial rollouts" treatment
            # as the `collection is None` case above: without this
            # check, `l1.delete_by_id` below raises straight through
            # (sqlite3.OperationalError: no such table / DuckDB's
            # equivalent CatalogException) on every single broadcast
            # for a table this pod doesn't cache, which is the common
            # case for any agent that doesn't use every framework
            # feature (e.g. a security-scanning agent hearing knowledge-
            # subsystem invalidations for `concepts`/`playbook_entries`
            # it never reads). `hasattr` guards backends (or test
            # doubles) that predate this method.
            return

        # through the collection, not the backend: a read of the key in flight on this pod read
        # it before the eviction, and the collection's fence is what stops it caching that.
        collection.evict_from_cache_sync(entity_id)

    async def close_collections(self) -> None:
        """close every registered collection that has something to shut down.

        The teardown owner for a collection's own background work: a write-behind coordination
        collection starts a periodic flusher from its write path, and the final flush is the
        difference between a clean shutdown losing nothing and losing one flush interval. Without
        an owner here, the only caller of that ``aclose`` was a test, and the task leaked at loop
        close.

        Runs beside :meth:`stop_invalidation_listener` in a process's shutdown path and follows
        the same rules: a no-op when nothing needs closing, and one collection's failure does not
        abandon the rest, because a teardown that stops halfway leaves the task it was there to
        stop still running.

        :return: nothing
        :rtype: None
        """
        for table, collection in list(self._collections.items()):
            try:
                # a declared seam on BaseCollection, not a getattr probe: the default is a no-op,
                # so a collection that starts nothing costs nothing, and a typo in an override is
                # a type error rather than a teardown that quietly skipped it.
                await collection.aclose()
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- one table must not abandon the rest
                contain(
                    exc,
                    log=log,
                    what="closing a collection failed; continuing with the rest of the teardown",
                    extra={"table": table, "error": f"{type(exc).__name__}: {exc}"},
                    level="error",
                )

    @property
    def invalidation_listener_running(self) -> bool:
        """whether this registry's cross-pod invalidation listener is running.

        What hears a peer's row broadcasts, so whatever judges this registry's followed tables, or
        evicts a cache derived from them, needs it running first.

        :return: ``True`` between :meth:`start_invalidation_listener` and
            :meth:`stop_invalidation_listener`
        :rtype: bool
        """
        return self._invalidation_subscription is not None

    async def stop_invalidation_listener(self) -> None:
        """drop this registry's cache-invalidation subscription.

        the teardown half of :meth:`start_invalidation_listener`, for a
        process's shutdown path. after it returns the registry holds no
        subscription and no client, and a later ``start`` subscribes
        again -- the registry is reusable, not one-shot.

        **a no-op when no listener is live**, whether one was never
        started or was already stopped. callers run this from a
        ``finally`` / ``close()`` alongside other teardown, where a
        raise would abandon the teardown steps after it.

        **no exception handling here, deliberately.**
        :meth:`NatsClient.unsubscribe` already absorbs the transport
        failures a shutdown produces: it returns early on an
        already-closed subscription, and it wraps both the raw
        unsubscribe and the dispatch-task join in its own handlers that
        log at WARNING. so unsubscribing on a draining connection does
        not raise, and a catch here would be code that never runs while
        looking like it guards something. what WOULD reach a caller is a
        genuine programming error, which must surface.

        the handle is released before the unsubscribe is awaited, so the
        registry is restartable even on the paths that unwind slowly.

        :return: nothing
        :rtype: None
        """
        subscription = self._invalidation_subscription
        nats_client = self._nats_client
        if subscription is None or nats_client is None:
            return
        self._invalidation_subscription = None
        self._nats_client = None
        await nats_client.unsubscribe(subscription)
        log.info(
            "Invalidation listener stopped",
            extra={"extra_data": {"subject": Subjects.cache_invalidate().path}},
        )

    async def publish_invalidation(
        self,
        nats_client: NatsClient | None,
        table_name: str,
        entity_id: Any,
        *,
        l2_key_current: bool = False,
        generation: str | None = None,
        bump_rows: int | None = None,
        columns: Mapping[str, str | None] | None = None,
    ) -> None:
        """publish cache invalidation signal for an entity.

        called by :class:`BaseCollection` after any write operation.
        emits :class:`CacheInvalidationMessage` via the typed wrapper.
        ``ids`` is the stringified pk-value tuple in declared column
        order.

        narrow exception scope: only :class:`PublishError` is logged
        and swallowed. The write has already succeeded, and nats-py
        buffers a publish issued while disconnected and flushes it on
        reconnect, so an ordinary outage loses nothing.

        The older justification here -- that a peer would "pull a
        stale-but-still-correct row from L3" -- does not hold and is
        recorded as wrong rather than deleted: a peer that missed the
        eviction keeps serving its own L1 and never re-reads. What
        actually bounds that staleness is the L1 max-age, and only for
        a collection that has opted into one. programming errors
        propagate.

        :param nats_client: connected typed NATS wrapper client;
            ``None`` short-circuits (no-op)
        :ptype nats_client: NatsClient | None
        :param table_name: target table name
        :ptype table_name: str
        :param entity_id: pk value (single-pk) or tuple of pk values
            in declared order (composite-pk)
        :ptype entity_id: Any
        :param l2_key_current: the write was revision-fenced and left this
            registry's scoped L2 key holding the newest value, so receivers
            sharing the scope keep it (see
            :attr:`CacheInvalidationMessage.l2_current_scope`)
        :ptype l2_key_current: bool
        :param generation: the write generation the row's commit advanced the table to, when it
            advanced one (:attr:`CacheInvalidationMessage.generation`)
        :ptype generation: str | None
        :param bump_rows: how many row messages that advance covers
            (:attr:`CacheInvalidationMessage.bump_rows`); ignored without ``generation``
        :ptype bump_rows: int | None
        :param columns: the publishing collection's declared invalidation columns' values, from
            the row the write saw (:attr:`CacheInvalidationMessage.columns`)
        :ptype columns: Mapping[str, str | None] | None
        :return: nothing
        :rtype: None
        """
        # A LOCAL write evicts LOCAL scans, before anything touches the bus.
        #
        # The listener below deliberately ignores this registry's own
        # broadcasts: for a by-pk row that is right, because `save_entity` just
        # wrote the freshest copy into L1 and evicting it would force a needless
        # re-read. A SCAN is the opposite. The write changed WHICH ROWS MATCH,
        # so the result this pod has cached is stale the instant it commits --
        # and it is the one pod guaranteed never to hear about it.
        #
        # Shipped without this in 0.23.6: a hub that imported knowledge kept
        # serving the pre-import concept set until the TTL lapsed. Deploy
        # content, ask a question, get the old answer, with nothing to indicate
        # why. Caught by the concept-visibility tests, which write and then read
        # back through the same registry.
        #
        # Deliberately ahead of the `nats_client is None` return: local eviction
        # is not a broadcast and must not be skipped when there is no bus (devx,
        # tests, a pod whose NATS is down).
        self.drop_local_scans(table_name)
        if nats_client is None and not self.has_derived_caches(table_name):
            return
        message = self.row_message(
            table_name,
            entity_id,
            l2_key_current=l2_key_current,
            generation=generation,
            bump_rows=bump_rows,
            columns=columns,
        )
        # A LOCAL write also reaches LOCAL derived caches, for the reason it reaches local scans:
        # the listener skips this registry's own broadcasts, and a cache derived from the row is
        # stale here the instant the write commits. Ahead of the `nats_client is None` return too.
        self.tell_derived_caches(message)
        if nats_client is None:
            return
        try:
            await nats_client.publish(
                subject=Subjects.cache_invalidate(),
                message=message,
            )
        except PublishError as exc:
            log.warning(
                "Failed to publish invalidation signal",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(
                            entity_id
                        ),  # convert at border: invalidation-publish-failed log extra_data field
                        "error": str(exc),
                    },
                },
            )
        except ValidationError as exc:
            # wire envelope failed validation -- programming error,
            # but propagating would mask the calling write op. log
            # loud and continue; surfaces as a real failure in CI
            # (there is no such counter, and nothing in
            # test_cache_coherence.py asserts on one -- the claim this
            # comment used to make was simply untrue).
            log.error(
                "Invalidation envelope validation failed",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(
                            entity_id
                        ),  # convert at border: invalidation-envelope-invalid log extra_data field
                        "error": str(exc),
                    },
                },
            )

    async def publish_invalidations(
        self, nats_client: NatsClient | None, table_name: str, messages: Sequence[CacheInvalidationMessage]
    ) -> None:
        """publish the row messages of one write of many rows: one message for them all when
        :attr:`batch_invalidations` is on (up to :data:`_ROWS_A_MESSAGE` rows each), else one a row.

        Locally exactly as :meth:`publish_invalidation` a row at a time: this process's scans of the table are
        dropped, and each row reaches its derived caches. A failed publish is logged as one is there.

        :param nats_client: connected typed NATS wrapper client; ``None`` publishes nothing
        :ptype nats_client: NatsClient | None
        :param table_name: the table every message names
        :ptype table_name: str
        :param messages: the row messages, each built by :meth:`row_message`
        :ptype messages: Sequence[CacheInvalidationMessage]
        :return: nothing
        :rtype: None
        """
        self.drop_local_scans(table_name)
        for message in messages:
            self.tell_derived_caches(message)
        if nats_client is None or not messages:
            return
        envelopes: list[CacheInvalidationMessage] = list(messages)
        if self.batch_invalidations:
            envelopes = [
                chunk[0].model_copy(
                    update={
                        "more_rows": [
                            InvalidatedRow(ids=row.ids, l2_current_scope=row.l2_current_scope, columns=row.columns)
                            for row in chunk[1:]
                        ]
                        or None
                    }
                )
                for chunk in (
                    messages[start : start + _ROWS_A_MESSAGE] for start in range(0, len(messages), _ROWS_A_MESSAGE)
                )
            ]
        for envelope in envelopes:
            try:
                await nats_client.publish(subject=Subjects.cache_invalidate(), message=envelope)
            except PublishError as exc:
                log.warning(
                    "Failed to publish invalidation signal",
                    extra={
                        "extra_data": {
                            "table": table_name,
                            "rows": 1 + len(envelope.more_rows or ()),
                            "error": str(exc),
                        }
                    },
                )

    def drop_local_scans(self, table_name: str) -> None:
        """the local half of an invalidation: this process's cached scans of a table it just wrote.

        A write changes which rows match a scan, so every cached scan of the table is stale the
        moment it commits, and this process is the one never to hear its own broadcast. Every local
        write path drops them through here, with a bus or without one.

        :param table_name: the table written
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        self.scan_cache.drop_for_table(table_name)

    def clear(self) -> None:
        """Remove all registered collections, overrides and L1 bounds (for tests).

        **Does NOT stop the invalidation listener**, and cannot: unsubscribing
        is awaitable and this method is synchronous. A registry cleared while a
        listener is live keeps that subscription and the callback goes on
        firing. The by-pk eviction below it is inert -- the collection lookup
        misses and returns -- but the scan-cache drop sits ABOVE that guard and
        runs on every broadcast regardless, so a cleared registry is not an
        idle one. Call :meth:`stop_invalidation_listener` alongside this in any
        teardown that started a listener.
        """
        self._collections.clear()
        self._overrides.clear()
        self._derived_caches.clear()
        self._generation_marks = GenerationMarks()
        self._watching = {}
