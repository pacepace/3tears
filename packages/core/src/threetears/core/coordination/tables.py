"""the L3 tables the durable coordination primitives keep their state in.

One table per kind of state, each keyed ``(purpose, key)``:

- ``coordination_counters`` -- windowed attempt counts (write-behind);
- ``coordination_claims`` -- idempotency claims and their stored outcome (write-behind);
- ``coordination_revocations`` -- standing revocations, with the moment revoked from
  (synchronous, negative-cached: nearly every check is of a key nobody revoked);
- ``coordination_redemptions`` -- single-use redemptions, the durable ledger shape
  (synchronous).

**Why ``purpose`` is in the key rather than the table name.** Each primitive used to name a KV
bucket, and one process builds many of them: identity-core holds three attempt counters and
identity-edge seven route throttles, each formerly its own bucket. A collection is registered per
table, so a table per bucket would mean seven collections over seven tables for one process, and
seven migrations for a consumer to apply. ``purpose`` carries what the bucket name carried, the
rows stay separated by it, and one shared collection serves every instance
(:func:`coordination_collection`).

**Which tiers are optional depends on what the primitive promises.** identity-edge has no L3 by
design and no database credential; a counter there runs L1+L2 and still throttles across
replicas, because a counter's worst case without durability is a lost increment. A registry with
no L2 (scriob's control plane today) runs L1+L3, correct within one process but counting per
replica. Neither is refused for a counter -- refusing would take a degraded throttle offline
rather than leaving it weaker -- though a missing L3 is logged once per table, since the
deliberate case and a wiring gap look identical from outside and only the log tells them apart.

**A primitive whose contract is exactly-once refuses a registry with no L2**
(:meth:`CoordinationCollection.require_l2_fence`): the compare-and-swap IS that guarantee, and
without L2 two replicas can both be told they were first. That is ``RedemptionLedger`` and
``IdempotencyKeyStore`` today.
"""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from collections.abc import Callable
from typing import Any, ClassVar, Final, Literal, TypeVar

from asyncpg import PostgresError
from sqlalchemy import MetaData

from threetears.core.collections.base import CasMutation
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    Index as SchemaIndex,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import CoreConfig, DefaultCoreConfig
from threetears.core.coordination.flusher import PeriodicFlusher
from threetears.core.data.schema import ColumnDef, IndexDef as DdlIndexDef, TableDef
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import DataLayerUnavailableError, L2ScopeNotConfiguredError
from threetears.nats.errors import KvError
from threetears.observe import get_logger

__all__ = [
    "COORDINATION_TABLE_SCHEMAS",
    "STORAGE_FAILURES",
    "CoordinationClaimsCollection",
    "CoordinationCollection",
    "CoordinationCountersCollection",
    "CoordinationRedemptionsCollection",
    "CoordinationRevocationsCollection",
    "CoordinationRow",
    "coordination_collection",
    "table_def_for",
]

log = get_logger(__name__)

#: what a coordination primitive means by "storage failed", and so what a ``fail_open`` primitive
#: may degrade on. Deliberately a named set rather than ``Exception``: an outage is a reason to
#: stop throttling for a moment, and a wiring or programming error is not, so the second kind
#: still propagates rather than silently admitting every caller.
#:
#: ``KvError`` is L2; ``DataLayerUnavailableError`` is the framework's L3-down signal;
#: ``PostgresError`` covers a direct asyncpg pool; the builtins cover a socket, a DNS failure and
#: a timeout wherever the backend does not wrap them.
STORAGE_FAILURES: Final[tuple[type[BaseException], ...]] = (
    KvError,
    DataLayerUnavailableError,
    PostgresError,
    ConnectionError,
    TimeoutError,
    OSError,
)

#: how long a recorded absence may answer on the revocation table. The generation stamp is what
#: makes an absence correct; this only bounds the window left by a write that commits and then
#: cannot advance the generation.
_REVOCATION_NEGATIVE_CACHE_SECONDS: Final = 60.0

#: how often one process sweeps a table's expired rows, and how many it removes per statement.
#: The sweep is table-size hygiene: every read already treats an expired row as absent.
_SWEEP_INTERVAL_SECONDS: Final = 300.0
_SWEEP_BATCH: Final = 1_000


#: so :func:`coordination_collection` returns the class it was asked for, and a consumer's
#: lifecycle calls on it stay checked.
_CollectionT = TypeVar("_CollectionT", bound="CoordinationCollection")


class CoordinationRow(BaseEntity):
    """one coordination row; addressed by its collection's ``(purpose, key)``."""

    primary_key_field: str = "key"


def _common_columns() -> list[Column]:
    """the columns every coordination table carries.

    :return: purpose, key, expiry and the two timestamps
    :rtype: list[Column]
    """
    return [
        Column("purpose", STRING_TYPE, immutable=True),
        Column("key", STRING_TYPE, immutable=True),
        Column("expires_at", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE, nullable=True),
    ]


class CoordinationCollection(SchemaBackedCollection[CoordinationRow]):
    """shared behaviour of the coordination tables: the composite key, the flusher, the sweep.

    Public because it is the type a consumer holds: the lifecycle a process must drive
    (:meth:`ensure_flushing`, :meth:`aclose`) and the hygiene it may drive (:meth:`sweep_expired`)
    live here, so a wave-2 consumer annotating the collection it got from
    :func:`coordination_collection` would otherwise have to import a private name across packages
    or fall back to ``Any`` and lose checking on exactly those calls.
    """

    primary_key_column: str | tuple[str, ...] = ("purpose", "key")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """build the collection and arm its first sweep.

        :param args: positional args forwarded to :class:`SchemaBackedCollection`
        :ptype args: Any
        :param kwargs: keyword args forwarded to :class:`SchemaBackedCollection`
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        self._next_expiry_sweep = 0.0
        self._flusher: PeriodicFlusher | None = None
        self._warned_no_durable_tier = False
        if self._l1 is not None:
            # These tables are the framework's, not the consumer's, so nothing else declares them
            # to L1. Initialising here is the same move the absent-marker table makes, and it is
            # what lets a consumer wire a coordination primitive without knowing our schema.
            self._l1.initialize(self.schema.to_sqlalchemy_table(MetaData()).metadata)

    @property
    def write_buffer(self) -> WriteBuffer | None:
        """the buffer this collection's deferred L3 writes wait in, when it defers them.

        :return: the write buffer, or ``None``
        :rtype: WriteBuffer | None
        """
        return self._write_buffer

    async def l2_cas_mutate(
        self,
        entity_id: Any,
        mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]],
        *,
        max_retries: int = 8,
    ) -> CasMutation:
        """compare-and-swap, with this table's flusher armed and its sweep driven by the write.

        Both live here rather than in each primitive: they are the collection's own invariants,
        and a wave-2 primitive that forgot either would buffer rows nothing ever flushed -- the
        silent durability loss write-behind exists to bound -- or grow a table for a 400-day ttl
        with nothing saying so. The sweep is self-throttled and failure-contained
        (:meth:`sweep_expired_if_due`), so driving it from every write path costs a monotonic
        clock read per write outside its interval.

        :param entity_id: pk value or tuple of pk values in declared order
        :ptype entity_id: Any
        :param mutate: the mutation callback, as :meth:`BaseCollection.l2_cas_mutate` documents
        :ptype mutate: Callable[[dict[str, Any] | None], tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]]
        :param max_retries: how many compare-and-swap rounds to allow
        :ptype max_retries: int
        :return: what the mutation did
        :rtype: CasMutation
        """
        outcome = await super().l2_cas_mutate(entity_id, mutate, max_retries=max_retries)
        self.ensure_flushing()
        await self.sweep_expired_if_due()
        return outcome

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        """save, with this table's flusher armed and its sweep driven by the write.

        :param entity: the entity to persist
        :ptype entity: BaseEntity
        :param conn: optional backend connection the L3 write joins
        :ptype conn: Any
        :return: nothing
        :rtype: None
        """
        await super().save_entity(entity, conn=conn)
        self.ensure_flushing()
        await self.sweep_expired_if_due()

    def ensure_flushing(self) -> None:
        """start this collection's periodic flusher, once, if it defers L3 writes.

        Armed by this collection's own write paths above, so a primitive cannot forget it.
        Public because a consumer wiring a collection outside those paths may still need it.

        :return: nothing
        :rtype: None
        """
        if self.write_buffer is None or self.l3_pool is None:
            return
        if self._flusher is None:
            self._flusher = PeriodicFlusher(self.write_buffer, self._registry)
        self._flusher.ensure_running()

    async def aclose(self) -> None:
        """stop the flusher and flush what is still buffered.

        :return: nothing
        :rtype: None
        """
        flusher = self._flusher
        self._flusher = None
        if flusher is not None:
            await flusher.aclose()

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        """persist to L3, or report success when this deployment has no L3.

        A coordination collection with no L3 is a supported wiring, not a broken one:
        identity-edge holds no database by design and still throttles across its replicas on L2.
        The generated path returns 0 rows when there is no durable store, and ``save_entity``
        reads 0 as a failed insert -- so a row written by an edge replica raised rather than
        landing. Reporting the write as done is the truth here: there was nothing to write to.

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value
        :ptype original_timestamp: datetime | None
        :param conn: optional backend connection the write joins
        :ptype conn: Any
        :return: rows affected, or 1 when this collection has no L3
        :rtype: int
        """
        if self.l3_pool is None:
            self._warn_no_durable_tier_once()
            return 1
        return await super().save_to_store(data, original_timestamp, conn=conn)

    def _warn_no_durable_tier_once(self) -> None:
        """say once per table that this process is keeping coordination state without L3.

        Running without L3 is deliberate in one deployment (identity-edge holds no database) and
        a wiring mistake everywhere else, and the two are indistinguishable from the outside: the
        writes succeed either way and the state simply does not survive a broker restart. One line
        per table per process is what tells an operator which one they are looking at. Once,
        because this sits on the write path of a throttle.

        :return: nothing
        :rtype: None
        """
        if self._warned_no_durable_tier:
            return
        self._warned_no_durable_tier = True
        log.warning(
            "coordination state has no durable tier on this registry; it lives in L1 and L2 only "
            "and a broker restart loses it. Deliberate for a process with no database "
            "(identity-edge); a wiring gap anywhere else",
            extra={"extra_data": {"table": self.table_name}},
        )

    def l2_key(self, entity_id: Any) -> str:
        """the L2 key for one ``(purpose, key)`` pair, digested so neither half can bleed.

        :meth:`BaseCollection.l2_key` joins pk values with ``"_"`` and keeps the result verbatim
        when it is grammar-safe, and its docstring names the precondition that makes that sound:
        a caller introducing underscore-bearing grammar-safe pk values must escape or override.
        These tables break it. ``purpose`` is caller-chosen and ``key`` is caller-supplied, and
        :class:`~threetears.core.coordination.idempotency.IdempotencyKeyStore` deliberately does
        not digest its key, so ``("jobs", "user_42")`` and ``("jobs_user", "42")`` would join to
        one L2 key while staying distinct rows in L1 and L3. Two stores whose purposes are
        prefix-related would then share an L2 entry and answer each other's callers.

        Digesting unconditionally, over the same ``\\x1f`` join
        :meth:`BaseCollection._absent_marker_key` uses, removes the ambiguity rather than
        documenting it: the separator cannot occur in either half.

        :param entity_id: pk value or tuple of pk values in declared order
        :ptype entity_id: Any
        :return: grammar-safe nats KV key, scoped by principal and table name
        :rtype: str
        :raises L2ScopeNotConfiguredError: when the registry carries no ``kv_key_scope``
        """
        scope = self._registry.kv_key_scope
        if scope is None:
            raise L2ScopeNotConfiguredError(
                f"{self.table_name}: no kv_key_scope on this collection's registry, so its L2 "
                f"keys would carry no principal segment. wire it with "
                f"registry.configure(kv_key_scope=threetears.nats.kv_key_scope_for(...))"
            )
        body = "\x1f".join(str(v) for v in self.normalize_pk(entity_id))
        return f"{scope}.{self.table_name}.{hashlib.sha256(body.encode('utf-8')).hexdigest()}"

    @property
    def entity_class(self) -> type[CoordinationRow]:
        """the entity every coordination table holds.

        :return: the row type
        :rtype: type[CoordinationRow]
        """
        return CoordinationRow

    @property
    def table_name(self) -> str:
        """the table this collection reads and writes.

        :return: the declared schema's name
        :rtype: str
        """
        return self.schema.name

    def require_l2_fence(self, primitive: str) -> None:
        """refuse a primitive whose contract is exactly-once when nothing fences it.

        ``l2_cas_mutate`` is a compare-and-swap against L2, and with no L2 client it degrades to a
        plain read-modify-write through L1 and L3. That is fine for a counter, which loses at worst
        an increment -- and wrong for a claim or a redemption, whose whole contract is that exactly
        one caller across every replica is told it was first: two replicas that both read absent
        would both be told "created", because the L3 upsert resolves the conflict instead of
        refusing it.

        Refused at construction rather than logged, on the same reasoning
        ``BaseCollection._refuse_unsound_negative_cache`` uses: a guarantee that quietly does not
        hold is worse than a process that will not start.

        :param primitive: the primitive's name, for the message
        :ptype primitive: str
        :return: nothing
        :rtype: None
        :raises ValueError: when this collection has no L2 client
        """
        if self._nats_client is None:
            raise ValueError(
                f"{primitive} needs an L2 client: its exactly-once guarantee is a compare-and-swap "
                f"against L2, and without one two replicas can both be told they were first. Pass a "
                f"registry configured with l2_client= (and a kv_key_scope), or use a primitive whose "
                f"contract survives a single-process fence"
            )

    async def sweep_expired(self, *, now: datetime | None = None, batch: int = _SWEEP_BATCH) -> int:
        """delete rows whose expiry has passed from L3, in one bounded statement.

        Hygiene only, and deliberately not a scheduled job: identity runs no scheduler and survey
        pods take none by design, so a scheduled sweep would have no home in half the consumers.
        Correctness never waits on it -- an expired row is already absent to every read at every
        tier -- so concurrent sweepers just delete rows the other already deleted, and no
        singleton is needed.

        **L1 and L2 reclaim themselves, by different means.** L1 drops an expired row when it is
        read. L2 carries the row's own expiry as a server-side lifetime on every write
        (:meth:`BaseCollection._l2_entry_lifetime`), so the broker removes the entry without
        being asked. This statement is the third tier only.

        **This is the one direct L3 write a negative-caching collection may make**, and it needs
        no write-generation advance: deleting a row that every tier already reads as absent cannot
        change any read's answer, and an absent-marker only ever claims absence.

        Raises on a storage failure, so a caller sweeping deliberately sees it;
        :meth:`sweep_expired_if_due`, which this collection's write paths drive, contains it.

        :param now: the moment to compare against, for tests; defaults to now
        :ptype now: datetime | None
        :param batch: the most rows to delete in one statement
        :ptype batch: int
        :return: rows deleted, or 0 when this collection has no L3
        :rtype: int
        :raises threetears.core.exceptions.DataLayerUnavailableError: on an L3 failure, and
            whatever else the backend raises
        """
        store = self.l3_pool
        if store is None:
            return 0
        cutoff = now or datetime.now(UTC)
        table = self.table_name
        deleted = await store.execute(
            f"DELETE FROM {table} WHERE (purpose, key) IN ("  # noqa: S608 - table name is a declared literal
            f"SELECT purpose, key FROM {table} WHERE expires_at IS NOT NULL AND expires_at < $1 LIMIT {int(batch)})",
            cutoff,
        )
        rows = _rows_affected(deleted)
        # Logged even at zero: an operator looking at a growing table needs to tell "sweeping,
        # nothing expired" from "never sweeping" and from "sweeping and failing".
        log.info(
            "coordination expiry sweep",
            extra={"extra_data": {"table": table, "rows_deleted": rows, "cutoff": cutoff.isoformat()}},
        )
        return rows

    async def sweep_expired_if_due(self) -> int:
        """run :meth:`sweep_expired` at most once per interval in this process, failures contained.

        Driven by this collection's own write paths (:meth:`l2_cas_mutate`, :meth:`save_entity`),
        so a failed sweep must not fail the throttle or claim that triggered it: table-size
        hygiene the docstring above says correctness never waits on cannot be what denies a
        login. The interval stamp is advanced before the statement runs, so a failing sweep
        self-throttles to one attempt per interval.

        :return: rows deleted, 0 when the interval has not elapsed or the sweep failed
        :rtype: int
        """
        now = time.monotonic()
        if now < self._next_expiry_sweep:
            return 0
        self._next_expiry_sweep = now + _SWEEP_INTERVAL_SECONDS
        try:
            return await self.sweep_expired()
        except STORAGE_FAILURES as exc:
            log.warning(
                "coordination expiry sweep failed; retrying after the interval",
                extra={
                    "extra_data": {
                        "table": self.table_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "interval_seconds": _SWEEP_INTERVAL_SECONDS,
                    }
                },
            )
            return 0


def _rows_affected(result: Any) -> int:
    """read a rowcount out of whatever the L3 backend returned.

    asyncpg reports a command tag (``"DELETE 12"``); other backends report an int or nothing.

    :param result: the backend's return value
    :ptype result: Any
    :return: rows affected, 0 when it cannot be read
    :rtype: int
    """
    if isinstance(result, int):
        return result
    if isinstance(result, str):
        tail = result.rsplit(" ", 1)[-1]
        return int(tail) if tail.isdigit() else 0
    return 0


class CoordinationCountersCollection(CoordinationCollection):
    """windowed attempt counts: many increments, each cheap, none worth an L3 write on its own."""

    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"
    expires_at_column: ClassVar[str | None] = "expires_at"
    datetime_columns: ClassVar[frozenset[str]] = frozenset(
        {"window_start", "expires_at", "date_created", "date_updated"}
    )
    schema = TableSchema(
        name="coordination_counters",
        primary_key=("purpose", "key"),
        columns=[
            *_common_columns(),
            # the count within the live window, and when that window opened. The window is
            # anchored at its first attempt, not refreshed per write, so a steady stream of
            # attempts cannot extend it.
            Column("count", INT_TYPE),
            Column("window_start", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
        indexes=(SchemaIndex("idx_coordination_counters_expiry", "expires_at"),),
    )


class CoordinationClaimsCollection(CoordinationCollection):
    """idempotency claims: the claim itself, and the outcome a retry must be given back."""

    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"
    expires_at_column: ClassVar[str | None] = "expires_at"
    datetime_columns: ClassVar[frozenset[str]] = frozenset(
        {"date_claimed", "date_completed", "expires_at", "date_created", "date_updated"}
    )
    schema = TableSchema(
        name="coordination_claims",
        primary_key=("purpose", "key"),
        columns=[
            *_common_columns(),
            Column("status", STRING_TYPE),
            Column("result", BYTES_TYPE, nullable=True),
            Column("error", STRING_TYPE, nullable=True),
            # opaque caller bytes attached at claim time (a request-body hash, typically), kept
            # for the record's whole life so a caller told "exists" can tell the same request
            # from a different one reusing the key.
            Column("claim_metadata", BYTES_TYPE, nullable=True),
            Column("date_claimed", DATETIMETZ_TYPE, immutable=True),
            Column("date_completed", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        indexes=(SchemaIndex("idx_coordination_claims_expiry", "expires_at"),),
    )


class CoordinationRevocationsCollection(CoordinationCollection):
    """standing revocations: read on nearly every request, written rarely, and absence is the
    common answer -- which is what negative caching is for."""

    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "synchronous"
    expires_at_column: ClassVar[str | None] = "expires_at"
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=_REVOCATION_NEGATIVE_CACHE_SECONDS)
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"revoked_at", "expires_at", "date_created", "date_updated"})
    schema = TableSchema(
        name="coordination_revocations",
        primary_key=("purpose", "key"),
        columns=[
            *_common_columns(),
            # the moment the key is revoked FROM: a session that started before it is denied, one
            # that starts after it is a legitimate new session. So the check is a comparison, not
            # a membership test.
            Column("revoked_at", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
        indexes=(SchemaIndex("idx_coordination_revocations_expiry", "expires_at"),),
    )


class CoordinationRedemptionsCollection(CoordinationCollection):
    """single-use redemptions: the durable ledger shape, where a second sighting is the answer."""

    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "synchronous"
    expires_at_column: ClassVar[str | None] = "expires_at"
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"expires_at", "date_created", "date_updated"})
    schema = TableSchema(
        name="coordination_redemptions",
        primary_key=("purpose", "key"),
        columns=[*_common_columns()],
        cas_column="date_updated",
        indexes=(SchemaIndex("idx_coordination_redemptions_expiry", "expires_at"),),
    )


#: every coordination table, in migration order.
COORDINATION_TABLE_SCHEMAS: Final[tuple[TableSchema, ...]] = (
    CoordinationCountersCollection.schema,
    CoordinationClaimsCollection.schema,
    CoordinationRevocationsCollection.schema,
    CoordinationRedemptionsCollection.schema,
)


#: collection column-type tag -> the DDL type the SQL builder names. Only the tags these tables
#: use; an unmapped tag raises rather than guessing a column type into a security table.
_DDL_TYPES: Final[dict[str, str]] = {
    STRING_TYPE: "text",
    INT_TYPE: "integer",
    BYTES_TYPE: "bytea",
    DATETIMETZ_TYPE: "timestamptz",
}


def table_def_for(schema: TableSchema) -> TableDef:
    """render one collection's declared schema as the DDL definition that creates it.

    One declaration, two readers: the collection reads the :class:`TableSchema` and the migration
    creates what this returns, so the table a consumer migrates cannot drift from the table the
    collection reads.

    :param schema: the collection's declared schema
    :ptype schema: TableSchema
    :return: the DDL definition
    :rtype: TableDef
    :raises KeyError: when a column carries a type tag this renderer does not map
    """
    pk = schema.primary_key
    pk_columns = {pk} if isinstance(pk, str) else set(pk)
    return TableDef(
        name=schema.name,
        columns=[
            ColumnDef(
                name=column.name,
                column_type=_DDL_TYPES[column.column_type],
                nullable=column.nullable and column.name not in pk_columns,
                default=column.server_default,
                primary_key=column.name in pk_columns,
            )
            for column in schema.columns
        ],
        indexes=[
            DdlIndexDef(name=index.name, columns=list(index.columns), unique=index.unique) for index in schema.indexes
        ],
    )


def coordination_collection(
    registry: CollectionRegistry,
    collection_class: type[_CollectionT],
    config: CoreConfig | None = None,
    **kwargs: Any,
) -> _CollectionT:
    """return the one collection of its table on this registry, building it the first time.

    Every primitive over a table shares it. A process builds many primitives over one table (seven
    route throttles, seven revocation denylists), and the registry keys collections by table name,
    so building one per primitive would leave the registry holding only the last -- and the
    invalidation listener would then evict through a collection nobody else uses.

    :param registry: the registry to look in and register with
    :ptype registry: CollectionRegistry
    :param collection_class: which coordination collection is wanted; the return is typed as it
    :ptype collection_class: type[CoordinationCollection]
    :param config: core config forwarded on first construction; ``None`` uses the framework
        defaults, which each table's declared ``l3_write_policy`` overrides anyway
    :ptype config: CoreConfig | None
    :param kwargs: further keyword args forwarded on first construction (e.g. ``write_buffer``)
    :ptype kwargs: Any
    :return: the shared collection
    :rtype: CoordinationCollection
    :raises TypeError: when the registry already holds a different collection for that table
    """
    table = collection_class.schema.name
    existing = registry.get_collection(table)
    if existing is not None:
        if not isinstance(existing, collection_class):
            raise TypeError(
                f"table {table!r} is already registered on this registry as "
                f"{type(existing).__name__}, not {collection_class.__name__}"
            )
        return existing
    if (
        collection_class.l3_write_policy == "write_behind"
        and registry.get_l3_pool(table) is not None
        and "write_buffer" not in kwargs
    ):
        # a declared write-behind policy needs somewhere to wait, and the caller of a coordination
        # primitive has no reason to know that. The flusher that drains it is started by the
        # primitive's first write (see ``ensure_flushing``).
        kwargs["write_buffer"] = WriteBuffer()
    # the framework defaults are the right fallback: each coordination table declares its own
    # l3_write_policy, which overrides whatever flush strategy a consumer's config carries.
    # BaseCollection registers itself on construction, so nothing registers it here.
    return collection_class(registry, config or DefaultCoreConfig(), **kwargs)
