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

**Every tier is optional, and that is deliberate.** identity-edge has no L3 by design and no
database credential; a counter there runs L1+L2 and still throttles across replicas. A registry
with no L2 (scriob's control plane today) runs L1+L3, which is correct within one process but
counts per replica; the primitives log that rather than refusing, because refusing would take a
degraded throttle offline instead of leaving it weaker.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Literal

from asyncpg import PostgresError
from sqlalchemy import MetaData

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
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.flusher import PeriodicFlusher
from threetears.core.data.schema import ColumnDef, IndexDef as DdlIndexDef, TableDef
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.nats.errors import KvError
from threetears.observe import get_logger

__all__ = [
    "COORDINATION_TABLE_SCHEMAS",
    "STORAGE_FAILURES",
    "CoordinationClaimsCollection",
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


class _CoordinationCollection(SchemaBackedCollection[CoordinationRow]):
    """shared behaviour of the coordination tables: the composite key, and the expiry sweep."""

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

    def ensure_flushing(self) -> None:
        """start this collection's periodic flusher, once, if it defers L3 writes.

        Called from the write path of every primitive over the table: nothing else in a consumer
        drains a write buffer, so a write-behind counter whose flusher was never started would
        keep its increments until the process happened to call ``flush_pending``, which no
        consumer does.

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

    async def sweep_expired(self, *, now: datetime | None = None, batch: int = _SWEEP_BATCH) -> int:
        """delete rows whose expiry has passed, in one bounded statement.

        Hygiene only, and deliberately not a scheduled job: identity runs no scheduler and survey
        pods take none by design, so a scheduled sweep would have no home in half the consumers.
        Correctness never waits on it -- an expired row is already absent to every read at every
        tier -- so concurrent sweepers just delete rows the other already deleted, and no singleton
        is needed. L2 is left alone for the same reason: an expired L2 row answers as absent, and
        its own lifetime removes it.

        :param now: the moment to compare against, for tests; defaults to now
        :ptype now: datetime | None
        :param batch: the most rows to delete in one statement
        :ptype batch: int
        :return: rows deleted, or 0 when this collection has no L3
        :rtype: int
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
        return _rows_affected(deleted)

    async def sweep_expired_if_due(self) -> int:
        """run :meth:`sweep_expired` at most once per interval in this process.

        :return: rows deleted, or 0 when the interval has not elapsed
        :rtype: int
        """
        now = time.monotonic()
        if now < self._next_expiry_sweep:
            return 0
        self._next_expiry_sweep = now + _SWEEP_INTERVAL_SECONDS
        return await self.sweep_expired()


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


class CoordinationCountersCollection(_CoordinationCollection):
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


class CoordinationClaimsCollection(_CoordinationCollection):
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


class CoordinationRevocationsCollection(_CoordinationCollection):
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


class CoordinationRedemptionsCollection(_CoordinationCollection):
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
    collection_class: type[_CoordinationCollection],
    config: Any = None,
    **kwargs: Any,
) -> _CoordinationCollection:
    """return the one collection of its table on this registry, building it the first time.

    Every primitive over a table shares it. A process builds many primitives over one table (seven
    route throttles, seven revocation denylists), and the registry keys collections by table name,
    so building one per primitive would leave the registry holding only the last -- and the
    invalidation listener would then evict through a collection nobody else uses.

    :param registry: the registry to look in and register with
    :ptype registry: CollectionRegistry
    :param collection_class: which coordination collection is wanted
    :ptype collection_class: type[_CoordinationCollection]
    :param config: core config forwarded on first construction
    :ptype config: Any
    :param kwargs: further keyword args forwarded on first construction (e.g. ``write_buffer``)
    :ptype kwargs: Any
    :return: the shared collection
    :rtype: _CoordinationCollection
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
    built = collection_class(registry, config or DefaultCoreConfig(), **kwargs)
    registry.register(built)
    return built
