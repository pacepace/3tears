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
    DATETIME_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
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
            Column("vec", VECTOR_TYPE, nullable=True),
            Column("blob", BYTES_TYPE, nullable=True),
            Column("counter", INT_TYPE),
            Column("flag", BOOL_TYPE),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_updated", DATETIME_TYPE),
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
            Column("date_created", DATETIME_TYPE, immutable=True),
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
            Column("date_added", DATETIME_TYPE, immutable=True),
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
    """build a no-op NATS mock."""
    nats = AsyncMock()
    nats.bucket_name = MagicMock(return_value="test")
    nats.get = AsyncMock(return_value=None)
    nats.put = AsyncMock(return_value=True)
    nats.delete = AsyncMock(return_value=True)
    nats.publish = AsyncMock()
    nats.subscribe = AsyncMock()
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
                self, conversation_id: uuid.UUID,
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
                self, *, user_id: uuid.UUID, agent_ids: tuple[uuid.UUID, ...],
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
                    Column("date_created", DATETIME_TYPE, immutable=True),
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
        """SELECT * FROM items WHERE id = $1."""
        pool = _RecordingPool()
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.fetch_from_postgres(uuid.uuid4())
        sql = pool.calls[0][1]
        assert sql == "SELECT * FROM items WHERE id = $1"

    @pytest.mark.asyncio
    async def test_composite_pk_select(self) -> None:
        """SELECT ... WHERE left_id = $1 AND right_id = $2."""
        pool = _RecordingPool()
        coll = _CompositeCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.fetch_from_postgres((uuid.uuid4(), uuid.uuid4()))
        sql = pool.calls[0][1]
        assert sql == "SELECT * FROM pairs WHERE left_id = $1 AND right_id = $2"

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
    async def test_dict_payload_becomes_json_string(self) -> None:
        """JSONB columns receive a JSON-encoded string for ``::jsonb`` cast."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        data = _sample_item(payload={"k": "v"})
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        # payload is at column index 3 (id, owner_id, label, payload)
        assert args[3] == json.dumps({"k": "v"})

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
    async def test_aware_datetime_becomes_naive_utc(self) -> None:
        """aware datetimes are projected to naive-UTC before asyncpg bind."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _ItemCollection(_registry(pool), _config(), nats_client=_nats())
        aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        data = _sample_item(date_created=aware, date_updated=aware)
        await coll.save_to_postgres(data)
        args = pool.calls[0][2]
        # date_created is at column index 8
        assert args[8].tzinfo is None
        assert args[8] == datetime(2026, 1, 2, 3, 4, 5)

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
        await coll.save_to_postgres({
            "id": uuid.uuid4(),
            "name": "x",
            "date_created": aware,
            "date_updated": aware,
        })
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
        await coll.save_to_postgres({
            "id": uuid.uuid4(),
            "name": "x",
            "date_created": naive,
            "date_updated": naive,
        })
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
        # CAS fence value is the last positional arg
        args = pool.calls[0][2]
        assert args[-1] == old_ts.replace(tzinfo=None)

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
