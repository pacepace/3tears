"""unit tests for :class:`SchemaBackedCollection` + :class:`TableSchema`.

exercises the SQL generator, write/read coercion, CAS / upsert /
append-only branches, composite-pk pk handling, L2 serialize/deserialize
round-trip, and the required-column guard. the pool is a recording
mock so the tests can inspect (sql, args) pairs without spinning up
testcontainers -- those integration tests live per-collection and are
covered by the downstream packages.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    ForeignKey,
    ForeignKeyDef,
    Index,
    IndexDef,
    PartitionEnforcementError,
    SchemaBackedCollection,
    TableSchema,
    spans_partitions,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


# ---------------------------------------------------------------------------
# stubs / fixtures
# ---------------------------------------------------------------------------


class _StubEntity(BaseEntity):
    primary_key_field = "id"


class _ItemCollection(SchemaBackedCollection[_StubEntity]):
    """single-pk, JSONB + vector + CAS, for most unit tests."""

    primary_key_column: str = "id"
    schema = TableSchema(
        name="items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("owner_id", UUID_TYPE, immutable=True),
            Column("label", STRING_TYPE),
            Column("payload", JSONB_TYPE, nullable=True),
            Column("vec", VECTOR_TYPE, nullable=True, vector_dim=4),
            Column("blob", BYTES_TYPE, nullable=True),
            Column("counter", INT_TYPE),
            Column("flag", BOOL_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class _JournalCollection(SchemaBackedCollection[_StubEntity]):
    """append-only variant for the insert-only branch test."""

    primary_key_column: str = "id"
    schema = TableSchema(
        name="journal",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("event", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
        on_conflict="raise",
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "journal"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class _CompositeCollection(SchemaBackedCollection[_StubEntity]):
    """composite-pk variant for pk-arity / fetch tests."""

    primary_key_column: str | tuple[str, ...] = ("left_id", "right_id")
    schema = TableSchema(
        name="pairs",
        primary_key=("left_id", "right_id"),
        columns=[
            Column("left_id", UUID_TYPE),
            Column("right_id", UUID_TYPE),
            Column("weight", INT_TYPE),
            Column("date_added", DATETIMETZ_TYPE, immutable=True),
        ],
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "pairs"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class _TzCollection(SchemaBackedCollection[_StubEntity]):
    """TIMESTAMPTZ-bearing collection for the DATETIMETZ_TYPE coverage.

    mirrors the shape of the hub's RBAC collections (groups / roles)
    where ``date_created`` / ``date_updated`` are TIMESTAMPTZ on the
    L3 side and CAS rides ``date_updated``. exists so the unit tests
    can exercise the aware-UTC write coercion (the path that
    collections-task-03c proved was broken when DATETIMETZ_TYPE
    columns were declared as DATETIME_TYPE -- the codec then silently
    shifted writes by the host's local TZ offset).
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="tzitems",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "tzitems"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class _RecordingPool:
    """minimal asyncpg.Pool shape that records every call.

    every invocation of :meth:`execute` / :meth:`fetchrow` /
    :meth:`fetch` is appended to :attr:`calls` as a ``(method, sql,
    args)`` tuple. return values come from the corresponding queue.
    """

    def __init__(self) -> None:
        """initialize empty recording state."""
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.execute_status: str = "UPDATE 1"
        self.fetchrow_row: dict[str, Any] | None = None

    async def execute(self, sql: str, *args: Any) -> str:
        """record the call and return :attr:`execute_status`."""
        self.calls.append(("execute", sql, args))
        return self.execute_status

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        """record the call and return :attr:`fetchrow_row`."""
        self.calls.append(("fetchrow", sql, args))
        return self.fetchrow_row

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """record the call and return an empty list."""
        self.calls.append(("fetch", sql, args))
        return []


def _registry(pool: Any) -> CollectionRegistry:
    """build a registry wired with a single pool for all tables."""
    reg = CollectionRegistry()
    reg.configure(l3_pool=pool)
    return reg


def _config() -> DefaultCoreConfig:
    """build an always-flush config."""
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _nats() -> AsyncMock:
    """build a no-op NATS wrapper mock matching the typed-wrapper api.

    schema-backed tests verify SQL generation + asyncpg parameter
    coercion; they do NOT exercise the L2 KV path. but
    :class:`BaseCollection` may call into the wrapper during
    auto-registration; the mock returns sensible no-op values for
    every method the wrapper exposes.
    """
    bucket = AsyncMock()
    bucket.get = AsyncMock(return_value=None)
    bucket.put = AsyncMock(return_value=1)
    bucket.delete = AsyncMock(return_value=True)

    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    return nats


# ---------------------------------------------------------------------------
# TableSchema validation
# ---------------------------------------------------------------------------


class TestTableSchemaValidation:
    """constructor-time validation on :class:`TableSchema`."""

    def test_rejects_missing_pk_column(self) -> None:
        """pk name must exist in columns."""
        with pytest.raises(ValueError, match="primary_key column"):
            TableSchema(
                name="t",
                primary_key="id",
                columns=[Column("name", STRING_TYPE)],
            )

    def test_rejects_missing_cas_column(self) -> None:
        """cas_column must exist in columns."""
        with pytest.raises(ValueError, match="cas_column"):
            TableSchema(
                name="t",
                primary_key="id",
                columns=[Column("id", UUID_TYPE)],
                cas_column="date_updated",
            )

    def test_composite_pk_normalizes_to_tuple(self) -> None:
        """composite pk accepts tuple and exposes tuple form."""
        schema = TableSchema(
            name="t",
            primary_key=("a", "b"),
            columns=[Column("a", UUID_TYPE), Column("b", UUID_TYPE)],
        )
        assert schema.pk_columns == ("a", "b")

    def test_single_pk_normalizes_to_1tuple(self) -> None:
        """scalar pk exposes length-1 tuple via pk_columns."""
        schema = TableSchema(
            name="t",
            primary_key="id",
            columns=[Column("id", UUID_TYPE)],
        )
        assert schema.pk_columns == ("id",)

    def test_mutable_columns_excludes_pk_and_immutable(self) -> None:
        """mutable_columns returns only non-pk, non-immutable columns."""
        schema = TableSchema(
            name="t",
            primary_key="id",
            columns=[
                Column("id", UUID_TYPE),
                Column("tag", STRING_TYPE, immutable=True),
                Column("value", STRING_TYPE),
            ],
        )
        mutable = [c.name for c in schema.mutable_columns()]
        assert mutable == ["value"]


# ---------------------------------------------------------------------------
# partition column primitive (collections-task-02)
# ---------------------------------------------------------------------------


class TestPartitionColumnSchema:
    """validation of the ``partition=True`` flag on :class:`Column`."""

    def test_partition_column_exposed_via_property(self) -> None:
        """schema.partition_column returns the flagged column name."""
        schema = TableSchema(
            name="t",
            primary_key=("a", "b"),
            columns=[
                Column("a", UUID_TYPE, partition=True),
                Column("b", UUID_TYPE),
            ],
        )
        assert schema.partition_column == "a"

    def test_no_partition_returns_none(self) -> None:
        """tables without a partition flag report None."""
        schema = TableSchema(
            name="t",
            primary_key="id",
            columns=[Column("id", UUID_TYPE)],
        )
        assert schema.partition_column is None

    def test_partition_must_be_part_of_primary_key(self) -> None:
        """partition column must appear in primary_key."""
        with pytest.raises(ValueError, match="must be part of primary_key"):
            TableSchema(
                name="t",
                primary_key="id",
                columns=[
                    Column("id", UUID_TYPE),
                    Column("agent_id", UUID_TYPE, partition=True),
                ],
            )

    def test_only_one_partition_per_table(self) -> None:
        """multiple partition columns are rejected."""
        with pytest.raises(ValueError, match="only one partition"):
            TableSchema(
                name="t",
                primary_key=("a", "b", "c"),
                columns=[
                    Column("a", UUID_TYPE, partition=True),
                    Column("b", UUID_TYPE, partition=True),
                    Column("c", UUID_TYPE),
                ],
            )

    def test_partition_implies_immutable(self) -> None:
        """partition=True coerces immutable=True automatically."""
        schema = TableSchema(
            name="t",
            primary_key=("a", "b"),
            columns=[
                Column("a", UUID_TYPE, partition=True),
                Column("b", UUID_TYPE),
                Column("name", STRING_TYPE),
            ],
        )
        assert schema.column("a").immutable is True
        # mutable_columns excludes pk; partition should not appear
        mutable_names = [c.name for c in schema.mutable_columns()]
        assert "a" not in mutable_names
        assert "name" in mutable_names


class _PartitionedEntity(BaseEntity):
    primary_key_field = "id"


class TestPartitionEnforcementSubclass:
    """``__init_subclass__`` blocks classes that violate the partition guard."""

    def test_subclass_with_partition_aware_methods_loads(self) -> None:
        """method that accepts the partition column passes the guard."""

        class _GoodCollection(SchemaBackedCollection[_PartitionedEntity]):
            primary_key_column: str | tuple[str, ...] = ("conversation_id", "id")
            schema = TableSchema(
                name="ctx",
                primary_key=("conversation_id", "id"),
                columns=[
                    Column("conversation_id", UUID_TYPE, partition=True),
                    Column("id", UUID_TYPE),
                    Column("payload", STRING_TYPE),
                ],
            )

            @property
            def table_name(self) -> str:
                return "ctx"

            @property
            def entity_class(self) -> type[_PartitionedEntity]:
                return _PartitionedEntity

            async def find_by_conversation(
                self,
                conversation_id: uuid.UUID,
            ) -> list[_PartitionedEntity]:
                _ = conversation_id
                return []

        # construction succeeds; class is well-formed
        assert _GoodCollection.schema.partition_column == "conversation_id"

    def test_subclass_with_unscoped_method_fails(self) -> None:
        """method missing partition column triggers PartitionEnforcementError."""
        with pytest.raises(PartitionEnforcementError, match="find_all"):

            class _BadCollection(SchemaBackedCollection[_PartitionedEntity]):
                primary_key_column: str | tuple[str, ...] = ("conversation_id", "id")
                schema = TableSchema(
                    name="ctx",
                    primary_key=("conversation_id", "id"),
                    columns=[
                        Column("conversation_id", UUID_TYPE, partition=True),
                        Column("id", UUID_TYPE),
                        Column("payload", STRING_TYPE),
                    ],
                )

                @property
                def table_name(self) -> str:
                    return "ctx"

                @property
                def entity_class(self) -> type[_PartitionedEntity]:
                    return _PartitionedEntity

                async def find_all(self) -> list[_PartitionedEntity]:
                    return []

    def test_spans_partitions_decorator_passes_guard(self) -> None:
        """``@spans_partitions`` opt-in is accepted by the guard."""

        class _SpansCollection(SchemaBackedCollection[_PartitionedEntity]):
            primary_key_column: str | tuple[str, ...] = ("agent_id", "id")
            schema = TableSchema(
                name="memories",
                primary_key=("agent_id", "id"),
                columns=[
                    Column("agent_id", UUID_TYPE, partition=True),
                    Column("id", UUID_TYPE),
                    Column("payload", STRING_TYPE),
                ],
            )

            @property
            def table_name(self) -> str:
                return "memories"

            @property
            def entity_class(self) -> type[_PartitionedEntity]:
                return _PartitionedEntity

            @spans_partitions
            async def find_for_user_in_agents(
                self,
                *,
                user_id: uuid.UUID,
                agent_ids: tuple[uuid.UUID, ...],
            ) -> list[_PartitionedEntity]:
                _ = user_id
                _ = agent_ids
                return []

        assert _SpansCollection.schema.partition_column == "agent_id"

    def test_partition_exempt_methods_allowlist(self) -> None:
        """``_partition_exempt_methods`` allowlist is honored."""

        class _ExemptCollection(SchemaBackedCollection[_PartitionedEntity]):
            primary_key_column: str | tuple[str, ...] = ("agent_id", "id")
            schema = TableSchema(
                name="t",
                primary_key=("agent_id", "id"),
                columns=[
                    Column("agent_id", UUID_TYPE, partition=True),
                    Column("id", UUID_TYPE),
                ],
            )
            # rationale: per-pod operational summary for monitoring
            _partition_exempt_methods = frozenset({"count_total"})

            @property
            def table_name(self) -> str:
                return "t"

            @property
            def entity_class(self) -> type[_PartitionedEntity]:
                return _PartitionedEntity

            async def count_total(self) -> int:
                return 0

        assert _ExemptCollection.schema.partition_column == "agent_id"


class TestSpansPartitionsDecorator:
    """call-time validation of ``@spans_partitions``-decorated methods."""

    @pytest.mark.asyncio
    async def test_tuple_argument_passes(self) -> None:
        """tuple of partition values is accepted."""

        @spans_partitions
        async def find_in(*, agent_ids: tuple[uuid.UUID, ...]) -> int:
            return len(agent_ids)

        result = await find_in(agent_ids=(uuid.uuid4(), uuid.uuid4()))
        assert result == 2

    @pytest.mark.asyncio
    async def test_list_argument_rejected(self) -> None:
        """list (rather than tuple) is rejected as ambiguous."""

        @spans_partitions
        async def find_in(*, agent_ids: tuple[uuid.UUID, ...]) -> int:
            return len(agent_ids)

        with pytest.raises(TypeError, match="must be a tuple"):
            await find_in(agent_ids=[uuid.uuid4()])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_empty_tuple_rejected(self) -> None:
        """empty tuple refuses to emit a zero-partition query."""

        @spans_partitions
        async def find_in(*, agent_ids: tuple[uuid.UUID, ...]) -> int:
            return len(agent_ids)

        with pytest.raises(TypeError, match="empty tuple"):
            await find_in(agent_ids=())

    def test_spans_partitions_raises_when_no_ids_param_and_not_marker_only(
        self,
    ) -> None:
        """decoration fails fast when no ``_ids`` param is found.

        review-task-01 finding D-2: the original decorator silently
        degraded to a marker-only role when no ``_ids``-suffix
        parameter was found. partition-hardening-task-01 sub-task 2
        flips this from silent to fail-fast at decoration time --
        callers must either rename the parameter to end in ``_ids``
        or opt into the marker-only path explicitly via
        ``marker_only=True``.
        """
        with pytest.raises(PartitionEnforcementError, match="marker_only"):

            @spans_partitions
            async def list_with_filters(
                self: Any,
                sql: str,
                params: list[Any],
            ) -> list[Any]:
                _ = self
                _ = sql
                _ = params
                return []

    def test_spans_partitions_marker_only_skips_ids_check(self) -> None:
        """``marker_only=True`` opt-in skips the ``_ids`` parameter scan.

        a method that is structurally cross-partition by other means
        (e.g. an opaque pre-built SQL string where ACL is enforced
        upstream) opts into the marker-only path via
        ``@spans_partitions(marker_only=True)``. decoration succeeds
        even though the signature has no ``_ids`` parameter.
        """

        @spans_partitions(marker_only=True)
        async def list_with_filters(
            self: Any,
            sql: str,
            params: list[Any],
        ) -> list[Any]:
            _ = self
            _ = sql
            _ = params
            return []

        assert getattr(list_with_filters, "_spans_partitions", False) is True

    @pytest.mark.asyncio
    async def test_spans_partitions_marker_only_skips_runtime_guard(
        self,
    ) -> None:
        """``marker_only=True`` callable skips the tuple-shape runtime guard.

        the runtime guard is the ``_ids``-tuple validator that fires
        on the non-marker-only path. a marker-only method may be
        called with arbitrary arg shapes (string / list / dict) --
        the partition discipline is satisfied by the structural
        upstream guarantee, not by the call-site argument shape.
        """

        @spans_partitions(marker_only=True)
        async def list_with_filters(
            sql: str,
            params: list[Any],
        ) -> int:
            _ = sql
            return len(params)

        # arbitrary arg shapes pass without TypeError on the guard
        result = await list_with_filters(
            "SELECT 1",
            ["a", "b", "c"],
        )
        assert result == 3


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------


class TestInsertSqlShape:
    """SQL-shape checks on the INSERT path, exercised via save_to_postgres."""

    @pytest.mark.asyncio
    async def test_upsert_has_on_conflict_and_do_update_set(self) -> None:
        """ON CONFLICT (pk) DO UPDATE SET lists every mutable column."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(_sample_item())
        sql = pool.calls[0][1]
        assert "INSERT INTO items" in sql
        assert "ON CONFLICT (id) DO UPDATE SET" in sql
        assert "label = EXCLUDED.label" in sql
        assert "payload = EXCLUDED.payload" in sql
        assert "date_updated = EXCLUDED.date_updated" in sql
        assert "owner_id = EXCLUDED.owner_id" not in sql
        assert "date_created = EXCLUDED.date_created" not in sql

    @pytest.mark.asyncio
    async def test_jsonb_column_has_jsonb_cast(self) -> None:
        """JSONB columns render as ``$N::jsonb`` in INSERT VALUES."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(_sample_item())
        sql = pool.calls[0][1]
        assert "::jsonb" in sql

    @pytest.mark.asyncio
    async def test_vector_column_has_vector_cast(self) -> None:
        """VECTOR columns render as ``$N::vector`` in INSERT VALUES."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(_sample_item())
        sql = pool.calls[0][1]
        assert "::vector" in sql

    @pytest.mark.asyncio
    async def test_on_conflict_raise_has_no_on_conflict(self) -> None:
        """on_conflict='raise' tables get plain INSERT only."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _JournalCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "event": "hello",
                "date_created": datetime.now(UTC),
            },
        )
        sql = pool.calls[0][1]
        assert "INSERT INTO journal" in sql
        assert "ON CONFLICT" not in sql

    @pytest.mark.asyncio
    async def test_on_conflict_ignore_emits_do_nothing(self) -> None:
        """on_conflict='ignore' tables emit ON CONFLICT (pk) DO NOTHING."""

        class _DedupCollection(SchemaBackedCollection[_StubEntity]):
            """dedup-on-redelivery variant."""

            primary_key_column: str = "id"
            schema = TableSchema(
                name="dedup",
                primary_key="id",
                columns=[
                    Column("id", UUID_TYPE),
                    Column("event", STRING_TYPE),
                    Column("date_created", DATETIMETZ_TYPE, immutable=True),
                ],
                on_conflict="ignore",
            )

            @property
            def table_name(self) -> str:
                return "dedup"

            @property
            def entity_class(self) -> type[_StubEntity]:
                return _StubEntity

        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _DedupCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "event": "hello",
                "date_created": datetime.now(UTC),
            },
        )
        sql = pool.calls[0][1]
        assert "INSERT INTO dedup" in sql
        assert "ON CONFLICT (id) DO NOTHING" in sql

    @pytest.mark.asyncio
    async def test_composite_pk_on_conflict_uses_both_columns(self) -> None:
        """composite pk emits ON CONFLICT (a, b)."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _CompositeCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(
            {
                "left_id": uuid.uuid4(),
                "right_id": uuid.uuid4(),
                "weight": 3,
                "date_added": datetime.now(UTC),
            },
        )
        sql = pool.calls[0][1]
        assert "ON CONFLICT (left_id, right_id)" in sql


class TestCasUpdateSqlShape:
    """SQL-shape checks on the CAS UPDATE path, via save_to_postgres."""

    @pytest.mark.asyncio
    async def test_fences_on_cas_column(self) -> None:
        """UPDATE ... WHERE id = $1 AND date_updated = $N."""
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(_sample_item(), original_timestamp=datetime.now(UTC))
        sql = pool.calls[0][1]
        assert "UPDATE items SET" in sql
        assert "WHERE id = $1" in sql
        assert "AND date_updated = $" in sql

    @pytest.mark.asyncio
    async def test_cas_path_unreachable_without_cas_column(self) -> None:
        """composite collection has no cas_column -- original_timestamp is ignored."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _CompositeCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(
            {
                "left_id": uuid.uuid4(),
                "right_id": uuid.uuid4(),
                "weight": 1,
                "date_added": datetime.now(UTC),
            },
            original_timestamp=datetime.now(UTC),
        )
        sql = pool.calls[0][1]
        # no UPDATE path, fell through to INSERT upsert instead
        assert "INSERT INTO pairs" in sql

    @pytest.mark.asyncio
    async def test_cas_columns_in_set_clause_are_mutable_only(self) -> None:
        """SET clause lists only mutable columns in declared order."""
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(_sample_item(), original_timestamp=datetime.now(UTC))
        sql = pool.calls[0][1]
        # mutable columns for _ItemCollection in declared order:
        # label, payload, vec, blob, counter, flag, date_updated
        for keyword in ("label =", "payload =", "vec =", "blob =", "counter =", "flag =", "date_updated ="):
            assert keyword in sql
        # owner_id / date_created are immutable; must not appear in SET
        assert "owner_id =" not in sql
        assert "date_created =" not in sql


class TestFetchAndDeleteSql:
    """SQL-shape checks on SELECT / DELETE paths, via public methods."""

    @pytest.mark.asyncio
    async def test_single_pk_select(self) -> None:
        """by-pk fetch projects DECLARED columns, vector cast to ::text.

        never ``SELECT *``: an undeclared table column is never read, and
        the declared ``vec`` VECTOR column is rendered ``vec::text AS vec``
        so a real pool returns the bracketed string form the read coercion
        parses (no pgvector binary codec is registered on any 3tears pool).
        """
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.fetch_from_postgres(uuid.uuid4())
        sql = pool.calls[0][1]
        assert sql == (
            "SELECT id, owner_id, label, payload, vec::text AS vec, blob, "
            "counter, flag, date_created, date_updated FROM items WHERE id = $1"
        )

    @pytest.mark.asyncio
    async def test_composite_pk_select(self) -> None:
        """by-pk fetch projects declared columns over a composite pk."""
        pool = _RecordingPool()
        coll = _CompositeCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.fetch_from_postgres((uuid.uuid4(), uuid.uuid4()))
        sql = pool.calls[0][1]
        assert sql == ("SELECT left_id, right_id, weight, date_added FROM pairs WHERE left_id = $1 AND right_id = $2")

    @pytest.mark.asyncio
    async def test_single_pk_delete(self) -> None:
        """DELETE FROM items WHERE id = $1."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.delete_from_postgres(uuid.uuid4())
        sql = pool.calls[0][1]
        assert sql == "DELETE FROM items WHERE id = $1"


# ---------------------------------------------------------------------------
# write coercion (asyncpg parameter shape)
# ---------------------------------------------------------------------------


class TestWriteCoercion:
    """coercion of caller-supplied values into asyncpg-ready shapes."""

    @pytest.mark.asyncio
    async def test_str_uuid_becomes_uuid(self) -> None:
        """string UUIDs are promoted to stdlib UUID at the WRITE boundary."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        item_id = uuid.uuid4()
        data = _sample_item(item_id=item_id)
        data["id"] = str(item_id)
        await coll.save_to_postgres(data)
        assert pool.calls[0][0] == "execute"
        args = pool.calls[0][2]
        # id is the first positional arg per column order
        assert args[0] == item_id
        assert isinstance(args[0], uuid.UUID)

    @pytest.mark.asyncio
    async def test_jsonb_dict_passes_through_to_asyncpg(self) -> None:
        """JSONB columns hand the raw dict to asyncpg.

        asyncpg's registered ``jsonb`` text codec is the canonical
        Python -> Postgres encoder; ``_encode_jsonb`` must NOT
        json.dumps the value or the codec runs a second encoding step
        and the column ends up storing a JSON-encoded string of the
        intended object (regression: bootstrap admin user persisted
        ``auth_identity`` as a quoted string, breaking
        ``->>'username'`` lookups during login).
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        payload = {"k": "v"}
        data = _sample_item(payload=payload)
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        # payload is at column index 3 (id, owner_id, label, payload)
        assert args[3] == payload
        assert isinstance(args[3], dict)

    @pytest.mark.asyncio
    async def test_jsonb_pre_encoded_string_is_decoded(self) -> None:
        """legacy callers handing a JSON string get decoded before asyncpg.

        a pre-encoded JSON string would also double-encode through the
        codec; ``_encode_jsonb`` decodes it back to a structure so the
        codec applies exactly one encoding step.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item(payload=json.dumps({"k": "v"}))
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        assert args[3] == {"k": "v"}
        assert isinstance(args[3], dict)

    @pytest.mark.asyncio
    async def test_list_vec_becomes_bracketed_string(self) -> None:
        """VECTOR columns receive bracketed textual form for ``::vector`` cast."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item(vec=[1.0, 2.0, 3.0])
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        # vec is at column index 4
        assert args[4] == "[1.0, 2.0, 3.0]"

    @pytest.mark.asyncio
    async def test_aware_datetime_round_trips_aware_utc(self) -> None:
        """aware-UTC datetimes round-trip aware through the asyncpg bind.

        collections-task-05 eliminated DATETIME_TYPE; every datetime
        column is now DATETIMETZ_TYPE. The write coercion ensures the
        asyncpg-bound value carries ``tzinfo=UTC`` so the codec's
        ``astimezone(UTC)`` is a no-op. This test pins that behavior on
        the general-purpose ``_ItemCollection`` (the dedicated
        ``_TzCollection`` covers the same path with explicit
        DATETIMETZ-only intent in the tests below).
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        data = _sample_item(date_created=aware, date_updated=aware)
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        # date_created is at column index 8
        assert args[8].tzinfo is UTC
        assert args[8] == aware

    @pytest.mark.asyncio
    async def test_datetimetz_keeps_aware_utc_on_insert(self) -> None:
        """DATETIMETZ_TYPE columns bind aware-UTC, never naive.

        regression coverage for collections-task-03c: asyncpg's
        TIMESTAMPTZ codec runs ``obj.astimezone(utc)`` on every bound
        value, which interprets a naive datetime as the host's local
        timezone and silently shifts the wire value by the local
        offset. binding aware-UTC makes the codec's astimezone a
        no-op and keeps the stored instant byte-stable across hosts.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _TzCollection(_registry(pool), _config(), nats_client=_nats())
        aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "name": "x",
                "date_created": aware,
                "date_updated": aware,
            }
        )
        args = pool.calls[0][2]
        # date_created at column index 2, date_updated at index 3
        assert args[2].tzinfo is UTC
        assert args[2] == aware
        assert args[3].tzinfo is UTC
        assert args[3] == aware

    @pytest.mark.asyncio
    async def test_datetimetz_naive_input_wrapped_with_utc(self) -> None:
        """naive datetime bound to a DATETIMETZ_TYPE column is interpreted as UTC.

        :class:`BaseCollection.save_entity` strips ``tzinfo`` from
        every datetime before invoking ``save_to_postgres`` so the
        TIMESTAMP-column path round-trips cleanly. for TIMESTAMPTZ
        columns the per-column write coercion has to UN-strip:
        re-wrap a naive value with ``UTC`` so asyncpg's TIMESTAMPTZ
        codec sees an aware value and runs its ``astimezone(utc)`` as
        a no-op. without this defensive re-wrap the strip-then-bind
        sequence shifts the wire value by the local TZ offset.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _TzCollection(_registry(pool), _config(), nats_client=_nats())
        naive = datetime(2026, 1, 2, 3, 4, 5)  # naive but logically UTC
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "name": "x",
                "date_created": naive,
                "date_updated": naive,
            }
        )
        args = pool.calls[0][2]
        assert args[2].tzinfo is UTC
        assert args[2] == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert args[3].tzinfo is UTC

    @pytest.mark.asyncio
    async def test_datetimetz_cas_fence_keeps_aware_utc(self) -> None:
        """CAS fence bound on a DATETIMETZ_TYPE column is aware-UTC.

        the bug surfaced because the CAS predicate value flowed
        through the same naive coercion as the column write -- the
        predicate then mismatched the stored aware-UTC instant on
        non-UTC hosts. this test pins the fix: the fence value is
        aware-UTC even when the entity passes naive (the realistic
        path because save_entity strips tzinfo before invoking
        save_to_postgres).
        """
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 1"
        coll = _TzCollection(_registry(pool), _config(), nats_client=_nats())
        original = datetime(2026, 1, 1, tzinfo=UTC)
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "name": "x",
                "date_created": datetime(2026, 1, 1, tzinfo=UTC),
                "date_updated": datetime(2026, 1, 2, tzinfo=UTC),
            },
            original_timestamp=original,
        )
        args = pool.calls[0][2]
        # CAS fence value is the last positional arg (per
        # _build_cas_update_sql). it must be aware-UTC so the
        # TIMESTAMPTZ codec's astimezone(utc) is a no-op.
        assert args[-1].tzinfo is UTC
        assert args[-1] == original

    @pytest.mark.asyncio
    async def test_missing_required_column_raises(self) -> None:
        """non-nullable columns without a data-dict entry raise KeyError."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item()
        del data["label"]
        with pytest.raises(KeyError, match="label"):
            await coll.save_to_postgres(data)

    @pytest.mark.asyncio
    async def test_missing_nullable_defaults_to_none(self) -> None:
        """nullable columns without a data-dict entry bind NULL."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item()
        del data["payload"]  # nullable
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        # payload is at column index 3
        assert args[3] is None


# ---------------------------------------------------------------------------
# CAS / upsert branching
# ---------------------------------------------------------------------------


class TestCasBranching:
    """routing between UPDATE and INSERT paths on save_to_postgres."""

    @pytest.mark.asyncio
    async def test_none_timestamp_goes_through_upsert(self) -> None:
        """original_timestamp=None -> INSERT ... ON CONFLICT DO UPDATE."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_postgres(_sample_item())
        sql = pool.calls[0][1]
        assert "INSERT INTO items" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_set_timestamp_goes_through_cas_update(self) -> None:
        """non-None original_timestamp -> UPDATE fenced on cas_column."""
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        old_ts = datetime(2026, 1, 1, tzinfo=UTC)
        await coll.save_to_postgres(_sample_item(), original_timestamp=old_ts)
        sql = pool.calls[0][1]
        assert "UPDATE items SET" in sql
        assert "WHERE id = $1" in sql
        assert "AND date_updated = $" in sql
        # CAS fence value is the last positional arg. The collection's
        # cas_column is now DATETIMETZ_TYPE so the fence value carries
        # tzinfo=UTC; before collections-task-05 the unconditional
        # base.py strip projected it to naive.
        args = pool.calls[0][2]
        assert args[-1] == old_ts
        assert args[-1].tzinfo is UTC

    @pytest.mark.asyncio
    async def test_cas_zero_rowcount_on_stale(self) -> None:
        """mock pool returns ``UPDATE 0``; save_to_postgres returns 0."""
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 0"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        result = await coll.save_to_postgres(_sample_item(), original_timestamp=datetime.now(UTC))
        assert result == 0

    @pytest.mark.asyncio
    async def test_no_cas_ignores_original_timestamp(self) -> None:
        """schemas without cas_column silently ignore the CAS arg."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _CompositeCollection(_registry(pool), _config(), nats_client=_nats())
        data = {
            "left_id": uuid.uuid4(),
            "right_id": uuid.uuid4(),
            "weight": 7,
            "date_added": datetime.now(UTC),
        }
        await coll.save_to_postgres(data, original_timestamp=datetime.now(UTC))
        sql = pool.calls[0][1]
        assert "INSERT INTO pairs" in sql
        assert "ON CONFLICT" in sql


# ---------------------------------------------------------------------------
# read coercion
# ---------------------------------------------------------------------------


class TestReadCoercion:
    """normalization of asyncpg's return shape on fetch."""

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_miss(self) -> None:
        """fetchrow returning None propagates as None."""
        pool = _RecordingPool()
        pool.fetchrow_row = None
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        result = await coll.fetch_from_postgres(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_decodes_jsonb_string(self) -> None:
        """JSONB columns arriving as strings round-trip to dict."""
        pool = _RecordingPool()
        item_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        pool.fetchrow_row = {
            "id": item_id,
            "owner_id": owner_id,
            "label": "hi",
            "payload": '{"k":"v"}',
            "vec": None,
            "blob": None,
            "counter": 1,
            "flag": True,
            "date_created": datetime(2026, 1, 1),
            "date_updated": datetime(2026, 1, 2),
        }
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        row = await coll.fetch_from_postgres(item_id)
        assert row is not None
        assert row["payload"] == {"k": "v"}
        assert row["id"] == item_id
        assert row["owner_id"] == owner_id

    @pytest.mark.asyncio
    async def test_fetch_decodes_vector_string(self) -> None:
        """VECTOR columns arriving as bracketed strings round-trip to lists."""
        pool = _RecordingPool()
        item_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        pool.fetchrow_row = {
            "id": item_id,
            "owner_id": owner_id,
            "label": "hi",
            "payload": None,
            "vec": "[1.5, 2.5, 3.5]",
            "blob": None,
            "counter": 1,
            "flag": True,
            "date_created": datetime(2026, 1, 1),
            "date_updated": datetime(2026, 1, 2),
        }
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        row = await coll.fetch_from_postgres(item_id)
        assert row is not None
        assert row["vec"] == [1.5, 2.5, 3.5]

    @pytest.mark.asyncio
    async def test_fetch_normalizes_datetimetz_to_aware_utc(self) -> None:
        """DATETIMETZ_TYPE columns arriving aware (asyncpg's typical
        TIMESTAMPTZ shape) pass through with ``tzinfo`` retained.

        also covers the defensive path where a value flows in naive
        from a non-asyncpg source (L2 JSON rehydration through a
        bespoke path, hand-rolled proxy pool round-tripping through
        string isoformat) -- the read coercion wraps with ``UTC`` so
        downstream callers always see a stable aware-UTC shape.
        """
        pool = _RecordingPool()
        item_id = uuid.uuid4()
        aware = datetime(2026, 1, 1, tzinfo=UTC)
        pool.fetchrow_row = {
            "id": item_id,
            "name": "x",
            "date_created": aware,
            "date_updated": aware,
        }
        coll = _TzCollection(_registry(pool), _config(), nats_client=_nats())
        row = await coll.fetch_from_postgres(item_id)
        assert row is not None
        assert row["date_created"].tzinfo is UTC
        assert row["date_updated"].tzinfo is UTC

        # naive arrival path -- defensive normalization
        pool.fetchrow_row = {
            "id": item_id,
            "name": "x",
            "date_created": datetime(2026, 1, 1),
            "date_updated": datetime(2026, 1, 1),
        }
        row = await coll.fetch_from_postgres(item_id)
        assert row is not None
        assert row["date_created"].tzinfo is UTC
        assert row["date_updated"].tzinfo is UTC


# ---------------------------------------------------------------------------
# L2 serialize / deserialize round-trip
# ---------------------------------------------------------------------------


class TestL2Roundtrip:
    """JSON payload produced by serialize + deserialize round-trips cleanly."""

    def test_uuid_datetime_roundtrip(self) -> None:
        """UUIDs and datetimes survive the JSON round-trip."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item()
        payload = coll.serialize(data)
        restored = coll.deserialize(payload)
        assert restored["id"] == data["id"]
        assert restored["owner_id"] == data["owner_id"]
        assert restored["date_created"] == data["date_created"]

    def test_bytes_roundtrip_via_base64(self) -> None:
        """bytes columns round-trip through base64 on JSON."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item(blob=b"\x00\x01\x02hello")
        payload = coll.serialize(data)
        parsed = json.loads(payload)
        assert isinstance(parsed["blob"], str)
        restored = coll.deserialize(payload)
        assert restored["blob"] == b"\x00\x01\x02hello"

    def test_unknown_columns_pass_through(self) -> None:
        """columns not declared in schema pass through untouched."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item()
        data["ad_hoc"] = "extra"
        payload = coll.serialize(data)
        restored = coll.deserialize(payload)
        assert restored["ad_hoc"] == "extra"


# ---------------------------------------------------------------------------
# delete_from_postgres
# ---------------------------------------------------------------------------


class TestDelete:
    """DELETE pathway issues the right SQL + args."""

    @pytest.mark.asyncio
    async def test_single_pk_delete(self) -> None:
        """DELETE uses one positional arg."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        target = uuid.uuid4()
        await coll.delete_from_postgres(target)
        method, sql, args = pool.calls[0]
        assert method == "execute"
        assert "DELETE FROM items" in sql
        assert args == (target,)

    @pytest.mark.asyncio
    async def test_composite_pk_delete_takes_tuple(self) -> None:
        """composite DELETE unpacks a tuple into two positional args."""
        pool = _RecordingPool()
        coll = _CompositeCollection(_registry(pool), _config(), nats_client=_nats())
        left = uuid.uuid4()
        right = uuid.uuid4()
        await coll.delete_from_postgres((left, right))
        method, sql, args = pool.calls[0]
        assert method == "execute"
        assert "DELETE FROM pairs" in sql
        assert args == (left, right)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sample_item(
    item_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
    vec: list[float] | None = None,
    blob: bytes | None = None,
    date_created: datetime | None = None,
    date_updated: datetime | None = None,
) -> dict[str, Any]:
    """build a sample row dict for :class:`_ItemCollection`."""
    now = datetime.now(UTC)
    return {
        "id": item_id or uuid.uuid4(),
        "owner_id": uuid.uuid4(),
        "label": "a label",
        "payload": payload if payload is not None else {"k": "v"},
        "vec": vec,
        "blob": blob,
        "counter": 1,
        "flag": True,
        "date_created": date_created or now,
        "date_updated": date_updated or now,
    }


# ---------------------------------------------------------------------------
# v0.8.0: Column validators for new fields
# ---------------------------------------------------------------------------


class TestColumnV080Validators:
    """cross-field validators introduced by v0.8.0 shard 01."""

    def test_column_enum_type_requires_enum_column_type(self) -> None:
        """enum_type set on non-ENUM_TYPE column is rejected."""
        with pytest.raises(ValueError, match="enum_type only valid"):
            Column("x", STRING_TYPE, enum_type=("a", "b"), enum_name="foo")

    def test_column_enum_column_type_requires_enum_values(self) -> None:
        """ENUM_TYPE without enum_type tuple is rejected."""
        with pytest.raises(ValueError, match="ENUM_TYPE requires enum_type"):
            Column("x", ENUM_TYPE)

    def test_column_enum_column_type_requires_enum_name(self) -> None:
        """ENUM_TYPE with enum_type but no enum_name is rejected."""
        with pytest.raises(ValueError, match="enum_type requires enum_name"):
            Column("x", ENUM_TYPE, enum_type=("a",))

    def test_column_enum_type_empty_tuple_rejected(self) -> None:
        """ENUM_TYPE with empty enum_type tuple is rejected."""
        with pytest.raises(ValueError, match="must be a non-empty tuple"):
            Column("x", ENUM_TYPE, enum_type=(), enum_name="foo")

    def test_column_enum_full_declaration_constructs(self) -> None:
        """positive: ENUM_TYPE with all required fields constructs cleanly."""
        col = Column(
            "memory_type",
            ENUM_TYPE,
            enum_type=("preference", "fact"),
            enum_name="memory_type",
        )
        assert col.column_type == ENUM_TYPE
        assert col.enum_type == ("preference", "fact")
        assert col.enum_name == "memory_type"

    def test_column_vector_dim_requires_vector_column_type(self) -> None:
        """vector_dim set on non-VECTOR_TYPE column is rejected."""
        with pytest.raises(ValueError, match="vector_dim only valid"):
            Column("x", STRING_TYPE, vector_dim=1024)

    def test_column_vector_column_type_requires_dim(self) -> None:
        """VECTOR_TYPE without vector_dim is rejected."""
        with pytest.raises(ValueError, match="VECTOR_TYPE requires vector_dim"):
            Column("x", VECTOR_TYPE)

    def test_column_numeric_requires_precision_and_scale(self) -> None:
        """NUMERIC_TYPE missing either precision or scale is rejected."""
        with pytest.raises(ValueError, match="NUMERIC_TYPE requires both"):
            Column("x", NUMERIC_TYPE)
        with pytest.raises(ValueError, match="NUMERIC_TYPE requires both"):
            Column("x", NUMERIC_TYPE, precision=12)
        with pytest.raises(ValueError, match="NUMERIC_TYPE requires both"):
            Column("x", NUMERIC_TYPE, scale=8)

    def test_column_precision_scale_require_numeric_type(self) -> None:
        """precision / scale on non-NUMERIC_TYPE column is rejected."""
        with pytest.raises(ValueError, match="precision/scale only valid"):
            Column("x", INT_TYPE, precision=12, scale=8)

    def test_column_numeric_full_declaration_constructs(self) -> None:
        """positive: NUMERIC_TYPE with both precision and scale constructs."""
        col = Column("cost", NUMERIC_TYPE, precision=12, scale=8, nullable=True)
        assert col.column_type == NUMERIC_TYPE
        assert col.precision == 12
        assert col.scale == 8

    def test_column_tsvector_requires_immutable(self) -> None:
        """TSVECTOR_TYPE without immutable=True is rejected."""
        with pytest.raises(ValueError, match="TSVECTOR_TYPE"):
            Column("x", TSVECTOR_TYPE, nullable=True)

    def test_column_tsvector_with_immutable_constructs(self) -> None:
        """positive: TSVECTOR_TYPE with immutable=True constructs cleanly."""
        col = Column("search_vector", TSVECTOR_TYPE, nullable=True, immutable=True)
        assert col.column_type == TSVECTOR_TYPE
        assert col.immutable is True
        assert col.nullable is True

    def test_column_foreign_key_passes_through(self) -> None:
        """foreign_key 2-tuple is retained on the dataclass."""
        col = Column("user_id", UUID_TYPE, foreign_key=("users", "user_id"))
        assert col.foreign_key == ("users", "user_id")

    def test_column_foreign_key_bad_shape_rejected(self) -> None:
        """foreign_key with wrong arity is rejected."""
        with pytest.raises(ValueError, match="foreign_key must be a 2-tuple"):
            Column("x", UUID_TYPE, foreign_key=("only-one",))  # type: ignore[arg-type]

    def test_column_server_default_passes_through(self) -> None:
        """server_default is retained on the dataclass; no cross-field check."""
        col = Column("kind", STRING_TYPE, server_default="'image'")
        assert col.server_default == "'image'"


# ---------------------------------------------------------------------------
# v0.8.0: ForeignKey factory
# ---------------------------------------------------------------------------


class TestForeignKeyFactory:
    """:func:`ForeignKey` factory + :class:`ForeignKeyDef` validation."""

    def test_foreign_key_single_column_via_strings(self) -> None:
        """bare-string args coerce to 1-tuples for the common case."""
        fk = ForeignKey("user_id", "users", "user_id")
        assert isinstance(fk, ForeignKeyDef)
        assert fk.local_cols == ("user_id",)
        assert fk.ref_table == "users"
        assert fk.ref_cols == ("user_id",)
        assert fk.on_delete == "NO ACTION"

    def test_foreign_key_composite_via_tuples(self) -> None:
        """composite FK preserves tuple shape and on_delete clause."""
        fk = ForeignKey(
            ("agent_id", "memory_id"),
            "memories",
            ("agent_id", "memory_id"),
            on_delete="CASCADE",
        )
        assert fk.local_cols == ("agent_id", "memory_id")
        assert fk.ref_cols == ("agent_id", "memory_id")
        assert fk.on_delete == "CASCADE"

    def test_foreign_key_mismatched_lengths_raises(self) -> None:
        """local_cols and ref_cols must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            ForeignKey(("a",), "t", ("a", "b"))

    def test_foreign_key_invalid_on_delete_raises(self) -> None:
        """on_delete outside the Literal set is rejected at runtime."""
        with pytest.raises(ValueError, match="on_delete must be one of"):
            ForeignKey("user_id", "users", "user_id", on_delete="BOGUS")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# v0.8.0: Index factory
# ---------------------------------------------------------------------------


class TestIndexFactory:
    """:func:`Index` factory + :class:`IndexDef` validation."""

    def test_index_basic(self) -> None:
        """varargs columns produce the expected tuple."""
        idx = Index("ix_x", "a", "b")
        assert isinstance(idx, IndexDef)
        assert idx.name == "ix_x"
        assert idx.columns == ("a", "b")
        assert idx.unique is False
        assert idx.where is None

    def test_index_unique_partial(self) -> None:
        """unique + where flow through to the dataclass."""
        idx = Index("ix_x", "a", unique=True, where="a IS NOT NULL")
        assert idx.unique is True
        assert idx.where == "a IS NOT NULL"
        assert idx.columns == ("a",)

    def test_index_no_columns_raises(self) -> None:
        """no-columns Index is rejected."""
        with pytest.raises(ValueError, match="must be a non-empty tuple"):
            Index("ix_x")


# ---------------------------------------------------------------------------
# v0.8.0: TableSchema validators
# ---------------------------------------------------------------------------


class TestTableSchemaV080Validators:
    """new TableSchema-level cross-field validators."""

    def test_table_schema_foreign_keys_must_reference_declared_columns(self) -> None:
        """FK whose local_cols are not declared in columns is rejected."""
        with pytest.raises(ValueError, match="not declared in columns"):
            TableSchema(
                name="t",
                primary_key="id",
                columns=[Column("id", UUID_TYPE)],
                foreign_keys=(ForeignKey("missing_col", "other", "other_id"),),
            )

    def test_table_schema_indexes_must_reference_declared_columns(self) -> None:
        """Index columns must resolve against the schema's columns."""
        with pytest.raises(ValueError, match="not declared in columns"):
            TableSchema(
                name="t",
                primary_key="id",
                columns=[Column("id", UUID_TYPE)],
                indexes=(Index("ix_bad", "missing_col"),),
            )

    def test_table_schema_duplicate_index_names_rejected(self) -> None:
        """two indexes with the same name on one TableSchema are rejected."""
        with pytest.raises(ValueError, match="duplicate index names"):
            TableSchema(
                name="t",
                primary_key="id",
                columns=[
                    Column("id", UUID_TYPE),
                    Column("a", STRING_TYPE),
                    Column("b", STRING_TYPE),
                ],
                indexes=(
                    Index("dup", "a"),
                    Index("dup", "b"),
                ),
            )

    def test_table_schema_default_values_for_v080_fields(self) -> None:
        """foreign_keys / indexes default to empty tuples."""
        schema = TableSchema(
            name="t",
            primary_key="id",
            columns=[Column("id", UUID_TYPE)],
        )
        assert schema.foreign_keys == ()
        assert schema.indexes == ()


# ---------------------------------------------------------------------------
# v0.8.0: partition-coercion regression — preserve new fields
# ---------------------------------------------------------------------------


class TestPartitionColumnCoercionV080:
    """regression coverage for the partition-coercion field-passthrough fix."""

    def test_partition_column_preserves_v080_fields_through_coercion(self) -> None:
        """partition columns retain foreign_key / server_default after the
        coerce-to-immutable rebuild in :meth:`TableSchema.__post_init__`.

        without the fix, the coercion only carried the v0.7.x fields
        (name, column_type, immutable, nullable, partition) and silently
        dropped any v0.8.0 field set on the source declaration. this
        test fails fast if a future edit forgets to extend the
        passthrough.
        """
        schema = TableSchema(
            name="memories",
            primary_key=("agent_id", "memory_id"),
            columns=[
                Column("memory_id", UUID_TYPE),
                # NOTE: deliberately NOT setting immutable=True so the
                # coercion path runs (it short-circuits when immutable
                # is already True).
                Column(
                    "agent_id",
                    UUID_TYPE,
                    partition=True,
                    foreign_key=("agents", "agent_id"),
                    server_default="'00000000-0000-0000-0000-000000000000'",
                ),
            ],
        )
        coerced = schema.column("agent_id")
        assert coerced.partition is True
        # the coerce path forces immutable=True regardless of the source value
        assert coerced.immutable is True
        # v0.8.0 fields must survive the rebuild
        assert coerced.foreign_key == ("agents", "agent_id")
        assert coerced.server_default == "'00000000-0000-0000-0000-000000000000'"


# ---------------------------------------------------------------------------
# v0.8.0: TSVECTOR write-path audit
# ---------------------------------------------------------------------------
#
# Per shard 03 §"TSVECTOR write-path audit": declaring
# ``Column("search_vector", TSVECTOR_TYPE, nullable=True, immutable=True)``
# changes Collection write behaviour. These tests pin the SQL
# generators' current behaviour so any future regression that
# accidentally adds TSVECTOR to ``DO UPDATE SET`` / ``CAS UPDATE SET``
# (would silently overwrite the trigger-maintained value) breaks
# loudly.


class _TsvectorCollection(SchemaBackedCollection[_StubEntity]):
    """schema-backed collection carrying a trigger-maintained TSVECTOR.

    used by the TSVECTOR audit tests to drive ``_build_insert_sql`` /
    ``_build_cas_update_sql`` against a TableSchema that mirrors the
    v0.8.0 ``memories.search_vector`` shape (``TSVECTOR_TYPE``,
    ``nullable=True``, ``immutable=True``).
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="fts_items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("content", STRING_TYPE),
            # immutable=True is required by the TSVECTOR_TYPE validator
            # (the trigger maintains the value server-side; UPDATE
            # generators must skip the column).
            Column("search_vector", TSVECTOR_TYPE, nullable=True, immutable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "fts_items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class TestTsvectorWritePathAudit:
    """TSVECTOR write-path audit (shard 03 §"TSVECTOR write-path audit").

    The :attr:`Column.immutable` flag must keep TSVECTOR_TYPE columns
    out of every UPDATE ``SET`` clause emitted by the SQL generators.
    The INSERT path is allowed to list the column (with a NULL
    parameter) because the Postgres ``BEFORE INSERT`` trigger overrides
    any caller-supplied value -- but if a future generator change
    skipped immutable columns from INSERT, that would also be safe
    (the column default + trigger would still produce the right
    value). This test pins the CURRENT behaviour so the choice is
    deliberate.
    """

    @pytest.mark.asyncio
    async def test_immutable_tsvector_excluded_from_upsert_set(self) -> None:
        """``ON CONFLICT DO UPDATE SET`` MUST NOT include ``search_vector``.

        an UPDATE SET containing the trigger-maintained tsvector
        column would either (a) overwrite the trigger's computed value
        with whatever the caller passed (NULL) on every upsert, or
        (b) require the caller to supply a hand-computed tsvector
        from python -- both regressions. ``Column.immutable=True``
        excludes the column from ``mutable_columns()`` and therefore
        from the SET clause.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _TsvectorCollection(_registry(pool), _config(), nats_client=_nats())
        now = datetime.now(UTC)
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "content": "hello world",
                "search_vector": None,
                "date_created": now,
                "date_updated": now,
            },
        )
        sql = pool.calls[0][1]
        assert "INSERT INTO fts_items" in sql
        assert "ON CONFLICT (id) DO UPDATE SET" in sql
        # content + date_updated are mutable -- they ARE in SET
        assert "content = EXCLUDED.content" in sql
        assert "date_updated = EXCLUDED.date_updated" in sql
        # search_vector is immutable + trigger-maintained -- it MUST
        # NOT appear in the SET clause
        assert "search_vector = EXCLUDED.search_vector" not in sql
        assert "search_vector =" not in sql.split("DO UPDATE SET", 1)[1]

    @pytest.mark.asyncio
    async def test_tsvector_fetch_casts_search_vector_to_text(self) -> None:
        """by-pk fetch renders the codec-less TSVECTOR column ``::text``.

        ``tsvector`` (like ``vector``) has no asyncpg codec on any 3tears
        pool, so a raw ``search_vector`` projection would raise
        ``UnsupportedClientFeatureError`` on a real pool / through the L3
        broker the moment a by-pk get/update/delete touched a memory row.
        the by-pk projection casts it ``::text`` so asyncpg returns the
        string form (a trigger-maintained full-text column, never
        consumed as data).
        """
        pool = _RecordingPool()
        coll = _TsvectorCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.fetch_from_postgres(uuid.uuid4())
        sql = pool.calls[0][1]
        assert sql == (
            "SELECT id, content, search_vector::text AS search_vector, "
            "date_created, date_updated FROM fts_items WHERE id = $1"
        )

    @pytest.mark.asyncio
    async def test_immutable_tsvector_excluded_from_cas_update_set(self) -> None:
        """CAS ``UPDATE SET`` MUST NOT include ``search_vector`` either.

        the CAS path mirrors the upsert path -- :meth:`mutable_columns`
        is the single source of truth for which columns reach a SET
        clause. covered explicitly so a future divergence between the
        upsert and CAS generators is caught by the audit suite.
        """
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 1"
        coll = _TsvectorCollection(_registry(pool), _config(), nats_client=_nats())
        now = datetime.now(UTC)
        await coll.save_to_postgres(
            {
                "id": uuid.uuid4(),
                "content": "updated content",
                "search_vector": None,
                "date_created": now,
                "date_updated": now,
            },
            original_timestamp=now,
        )
        sql = pool.calls[0][1]
        assert "UPDATE fts_items SET" in sql
        # content + date_updated mutable -> in SET
        assert "content =" in sql
        assert "date_updated =" in sql
        # search_vector immutable -> NOT in SET
        # CAS UPDATE SQL shape: "UPDATE <table> SET <set> WHERE <where>"
        set_section = sql.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
        assert "search_vector" not in set_section

    @pytest.mark.asyncio
    async def test_immutable_tsvector_in_insert_column_list_with_null_param(self) -> None:
        """INSERT column list includes ``search_vector`` with a NULL bind value.

        The current generator behaviour: ``_build_insert_sql`` iterates
        ALL schema columns (including immutable ones), so the column
        list contains ``search_vector``; ``_build_insert_params`` reads
        the column from the data dict and the nullable=True column
        defaults missing keys to ``None`` so the parameter list carries
        ``None`` for ``search_vector``.

        Why this is safe: the Postgres ``BEFORE INSERT OR UPDATE OF
        content, summary`` trigger fires on every INSERT and overrides
        whatever value the row carries -- so the explicit ``NULL`` is
        replaced by the trigger-computed tsvector before the row is
        committed.

        Why we don't skip it from INSERT: the generator's
        ``mutable_columns()`` filter is reserved for UPDATE-side
        guards. Mirroring that filter into ``_build_insert_sql`` would
        widen the scope of ``immutable=True`` beyond its stated
        semantic ("excluded from UPDATE SET"). The current behaviour
        keeps ``immutable=True`` UPDATE-only and relies on the
        Postgres trigger for INSERT-side semantics; both reach the
        same outcome.

        If a future refactor decides to exclude immutable columns from
        INSERT too (also safe given the trigger), this test must be
        updated, not the generator (the behaviour choice is deliberate).
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _TsvectorCollection(_registry(pool), _config(), nats_client=_nats())
        now = datetime.now(UTC)
        item_id = uuid.uuid4()
        # caller deliberately does NOT pass search_vector -- it's
        # nullable so the generator should default to None
        await coll.save_to_postgres(
            {
                "id": item_id,
                "content": "hello world",
                "date_created": now,
                "date_updated": now,
            },
        )
        sql = pool.calls[0][1]
        args = pool.calls[0][2]
        # column list contains search_vector
        insert_cols_section = sql.split("(", 1)[1].split(")", 1)[0]
        assert "search_vector" in insert_cols_section
        # parameter list (positional asyncpg args) matches column order;
        # find search_vector's position and assert the corresponding
        # arg is None
        col_names = [c.strip() for c in insert_cols_section.split(",")]
        sv_index = col_names.index("search_vector")
        assert args[sv_index] is None, (
            f"search_vector INSERT param at index {sv_index} should be None "
            f"when caller omits the key; got {args[sv_index]!r}"
        )


# ---------------------------------------------------------------------------
# v0.8.0: backward compatibility — existing TableSchema declarations work
# ---------------------------------------------------------------------------


class TestExistingTableSchemaBackwardCompat:
    """backward-compatibility guard for downstream collections."""

    def test_existing_table_schema_declaration_works_unchanged(self) -> None:
        """A representative real-world TableSchema (MemoriesCollection.schema)
        constructs without errors after the v0.8.0 API extension and
        carries the v0.8.0 enrichments.

        Guards against accidentally making any new field non-optional
        and against making any new validator reject the declaration.
        Note: ``MemoriesCollection.schema`` was enriched in shard 03
        (v0.8.0) with single-column FKs, ENUM type_memory, vector_dim
        on embedding, TSVECTOR search_vector, and named indexes.
        """
        # Import inside the test so workspaces with the agent-memory
        # package missing still collect the rest of this module.
        from threetears.agent.memory.collections import MemoriesCollection

        schema = MemoriesCollection.schema
        assert isinstance(schema, TableSchema)
        assert schema.name == "memories"
        assert schema.pk_columns == ("agent_id", "memory_id")
        # the partition column survives coercion + v0.8.0 validation
        assert schema.partition_column == "agent_id"
        # v0.8.0: indexes declared (ix_memories_user_date from v0.7.5
        # factory + ix_memories_user_alias relocated from metallm
        # alembic 088).
        index_names = {idx.name for idx in schema.indexes}
        assert "ix_memories_user_date" in index_names
        assert "ix_memories_user_alias" in index_names
        # v0.8.0: every Column with ``foreign_key=`` survives partition
        # coercion. user_id has an inline FK to users.user_id.
        user_id_col = schema.column("user_id")
        assert user_id_col.foreign_key == ("users", "user_id")


# ---------------------------------------------------------------------------
# v0.8.0 shard 03: INSERT generators honor Column.server_default
# ---------------------------------------------------------------------------
#
# Without this gate, ``nullable=False`` columns that declare a
# ``server_default`` (e.g. ``media.metadata_json`` /
# ``media.media_category`` / ``media.extraction_status`` /
# ``mcp_tool_grants.date_created``) force every Python writer to
# duplicate the default value or hit a ``KeyError`` from
# ``_pull_value``. The post-shard-03 behaviour: when the caller omits
# such a column from the data dict, the INSERT generator drops the
# column from BOTH the SQL column list AND the parameter list, letting
# Postgres apply the server-side default.


class _ServerDefaultCollection(SchemaBackedCollection[_StubEntity]):
    """schema with a ``nullable=False, server_default=...`` column.

    used by the server-default gate tests below to drive
    ``_build_insert_sql`` / ``_build_insert_params`` against the
    canonical "required + has server default" shape (the shape every
    ``date_created TIMESTAMPTZ NOT NULL DEFAULT now()`` column in the
    package wears).
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="sd_items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            # NOT NULL + server_default: the shape of
            # mcp_tool_grants.date_created, media.metadata_json,
            # conversation_memory_refs.date_created, etc.
            Column(
                "kind",
                STRING_TYPE,
                server_default="'image'::text",
            ),
        ],
        on_conflict="raise",
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "sd_items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class _NoDefaultCollection(SchemaBackedCollection[_StubEntity]):
    """schema with a ``nullable=False`` column that has NO server_default.

    used to confirm the v0.8.0 gate does NOT mask the required-column
    KeyError for columns without a default: those must still raise so
    the caller learns immediately that a required column was missed.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="nd_items",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            # NOT NULL, no server_default — every writer MUST supply
            # this value or hit KeyError.
            Column("kind", STRING_TYPE),
        ],
        on_conflict="raise",
    )

    @property
    def table_name(self) -> str:
        """return table name."""
        return "nd_items"

    @property
    def entity_class(self) -> type[_StubEntity]:
        """return entity class."""
        return _StubEntity


class TestInsertServerDefaultGate:
    """v0.8.0: server_default columns drop out of INSERT when omitted."""

    @pytest.mark.asyncio
    async def test_insert_omits_server_default_column_when_caller_omits(self) -> None:
        """``kind`` column with ``server_default`` and absent from
        the data dict MUST NOT appear in the INSERT SQL or params.

        Without this gate, ``_pull_value`` raised KeyError for
        ``nullable=False`` columns regardless of server_default,
        forcing every caller to supply a value. After v0.8.0 the gate
        drops the column so Postgres applies the server-side default
        verbatim.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ServerDefaultCollection(_registry(pool), _config(), nats_client=_nats())
        item_id = uuid.uuid4()
        # caller deliberately omits ``kind`` — server default should
        # apply
        await coll.save_to_postgres({"id": item_id, "name": "x"})
        sql = pool.calls[0][1]
        args = pool.calls[0][2]
        # column list MUST NOT include ``kind``
        insert_cols_section = sql.split("(", 1)[1].split(")", 1)[0]
        col_names = [c.strip() for c in insert_cols_section.split(",")]
        assert "kind" not in col_names, (
            f"server_default column 'kind' should be omitted from INSERT when caller omits it; got {col_names!r}"
        )
        # params list MUST be the same length as the emitted column
        # list — i.e. no stray NULL for ``kind``
        assert len(args) == len(col_names), (
            f"INSERT params count {len(args)} should match emitted column count {len(col_names)}"
        )

    @pytest.mark.asyncio
    async def test_insert_includes_server_default_column_when_caller_supplies(self) -> None:
        """when the caller DOES supply a value, the column IS in the
        SQL (caller-supplied value wins over server default).
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ServerDefaultCollection(_registry(pool), _config(), nats_client=_nats())
        item_id = uuid.uuid4()
        await coll.save_to_postgres({"id": item_id, "name": "x", "kind": "document"})
        sql = pool.calls[0][1]
        args = pool.calls[0][2]
        insert_cols_section = sql.split("(", 1)[1].split(")", 1)[0]
        col_names = [c.strip() for c in insert_cols_section.split(",")]
        assert "kind" in col_names, f"caller-supplied 'kind' must appear in INSERT cols; got {col_names!r}"
        # the supplied value is in args at the index matching ``kind``
        idx = col_names.index("kind")
        assert args[idx] == "document"

    @pytest.mark.asyncio
    async def test_insert_still_raises_on_missing_required_column_without_default(self) -> None:
        """required column WITHOUT server_default still raises KeyError
        when omitted — the v0.8.0 gate only applies to columns that
        DECLARE a server default.
        """
        pool = _RecordingPool()
        coll = _NoDefaultCollection(_registry(pool), _config(), nats_client=_nats())
        with pytest.raises(KeyError, match="kind"):
            # ``kind`` is non-nullable with no server_default — must
            # raise, not silently bind NULL
            await coll.save_to_postgres({"id": uuid.uuid4(), "name": "x"})


# ---------------------------------------------------------------------------
# v0.8.0 shard 03: mutable_columns() pins for MediaContent / MemoryChunk
# ---------------------------------------------------------------------------


class TestMediaContentImmutabilityPin:
    """v0.8.0: customer_id / media_id / user_id silently flipped to
    immutable; pin so a regression flagging them mutable surfaces in
    unit tests rather than in the Alembic auto-gen diff.
    """

    def test_media_content_collection_immutable_columns_pin(self) -> None:
        """v0.8.0: customer_id / media_id / user_id are immutable;
        UPDATE generators must not include them in SET clause.
        """
        from threetears.agent.memory.collections import MediaContentCollection

        mutable = {c.name for c in MediaContentCollection.schema.mutable_columns()}
        for col in ("customer_id", "media_id", "user_id"):
            assert col not in mutable, f"{col} must not be in mutable columns for media_content"

    def test_memory_chunk_collection_immutable_columns_pin(self) -> None:
        """v0.8.0: customer_id / memory_id / user_id are immutable on
        memory_chunks; UPDATE generators must not include them in
        SET clause.
        """
        from threetears.agent.memory.collections import MemoryChunkCollection

        mutable = {c.name for c in MemoryChunkCollection.schema.mutable_columns()}
        for col in ("customer_id", "memory_id", "user_id"):
            assert col not in mutable, f"{col} must not be in mutable columns for memory_chunks"
