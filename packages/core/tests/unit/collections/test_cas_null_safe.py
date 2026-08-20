"""unit tests for ``TableSchema(cas_null_safe=True)`` -- the NULL-safe CAS fence.

the defect these cover: ``SqlL3Backend._upsert_schema`` treated compare-and-swap
as eligible only when the expected value was non-``None``, so the FIRST write of
a row -- ``original_timestamp=None`` -- fell through to an UNFENCED
``INSERT ... ON CONFLICT (pk) DO UPDATE SET ...``. for a table whose primary key
is DERIVED from the business key (a ``uuid5``, a hash) two concurrent
first-writers compute the same id, both statements report one row affected, and
the second silently overwrites the first. a counter increment or a set-member
append loses the first writer's work with nothing raised anywhere.

the shape asserted here is the one three collections in ``14-eng-ai-survey``
hand-write today precisely because the generator could not express it
(``split_assignments_data.py:188``, ``indexes_data.py:187``,
``sessions_data.py:174``).

these are SQL-SHAPE tests, and they run in the default gate. the test that
proves the LOST WRITE itself needs a real Postgres and lives in
``tests/integration/test_cas_null_safe_lost_write.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from threetears.core.backends import schema_sql
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError

# ---------------------------------------------------------------------------
# fixtures / stubs
# ---------------------------------------------------------------------------


class _CounterEntity(BaseEntity):
    primary_key_field = "id"


def _counter_schema(**overrides: Any) -> TableSchema:
    """build the derived-id counter schema the dependants model.

    :param overrides: TableSchema kwargs to replace
    :ptype overrides: Any
    :return: table schema
    :rtype: TableSchema
    """
    kwargs: dict[str, Any] = {
        "name": "counters",
        "primary_key": "id",
        "columns": [
            Column("id", UUID_TYPE),
            Column("label", STRING_TYPE),
            Column("count", INT_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        "cas_column": "date_updated",
        "cas_null_safe": True,
    }
    kwargs.update(overrides)
    return TableSchema(**kwargs)


class _CounterCollection(SchemaBackedCollection[_CounterEntity]):
    """derived-id counter with the NULL-safe fence switched on."""

    primary_key_column: str = "id"
    schema = _counter_schema()

    @property
    def table_name(self) -> str:
        """return table name."""
        return "counters"

    @property
    def entity_class(self) -> type[_CounterEntity]:
        """return entity class."""
        return _CounterEntity


class _UnfencedCounterCollection(SchemaBackedCollection[_CounterEntity]):
    """same shape with the flag left at its default -- the regression control."""

    primary_key_column: str = "id"
    schema = _counter_schema(name="plain_counters", cas_null_safe=False)

    @property
    def table_name(self) -> str:
        """return table name."""
        return "plain_counters"

    @property
    def entity_class(self) -> type[_CounterEntity]:
        """return entity class."""
        return _CounterEntity


class _RecordingPool:
    """minimal asyncpg.Pool shape recording ``(method, sql, args)``."""

    def __init__(self) -> None:
        """initialize empty recording state."""
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.execute_status: str = "INSERT 0 1"
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
    """build a registry wired with a single pool for all tables.

    :param pool: recording pool
    :ptype pool: Any
    :return: configured registry
    :rtype: CollectionRegistry
    """
    reg = CollectionRegistry()
    reg.configure(l3_pool=pool)
    return reg


def _config(**overrides: Any) -> DefaultCoreConfig:
    """build an always-flush config unless overridden.

    :param overrides: config kwargs to replace
    :ptype overrides: Any
    :return: core config
    :rtype: DefaultCoreConfig
    """
    kwargs: dict[str, Any] = {"collection_flush": "ALWAYS", "collection_flush_tables": ""}
    kwargs.update(overrides)
    return DefaultCoreConfig(**kwargs)


def _nats() -> AsyncMock:
    """build a no-op NATS wrapper mock.

    :return: mock nats wrapper
    :rtype: AsyncMock
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


def _row(**overrides: Any) -> dict[str, Any]:
    """build a counter row payload.

    :param overrides: column values to replace
    :ptype overrides: Any
    :return: row dict
    :rtype: dict[str, Any]
    """
    data: dict[str, Any] = {
        "id": uuid.uuid4(),
        "label": "alpha",
        "count": 1,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# the default is off, and nothing about it moved
# ---------------------------------------------------------------------------


class TestDefaultIsUnchanged:
    """the flag defaults off and leaves every existing schema's SQL alone."""

    def test_flag_defaults_to_false(self) -> None:
        """a schema that says nothing about it is not fenced."""
        schema = TableSchema(
            name="plain",
            primary_key="id",
            columns=[Column("id", UUID_TYPE), Column("date_updated", DATETIMETZ_TYPE)],
            cas_column="date_updated",
        )
        assert schema.cas_null_safe is False

    @pytest.mark.asyncio
    async def test_unflagged_schema_still_emits_unfenced_upsert_on_none_cas(self) -> None:
        """the pre-existing behaviour for ``cas=None`` is preserved verbatim."""
        pool = _RecordingPool()
        coll = _UnfencedCounterCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_store(_row())
        sql = pool.calls[0][1]
        assert sql.startswith("INSERT INTO plain_counters")
        assert "ON CONFLICT (id) DO UPDATE SET" in sql
        assert "IS NOT DISTINCT FROM" not in sql
        assert "WHERE" not in sql

    @pytest.mark.asyncio
    async def test_unflagged_schema_still_emits_bare_cas_update_on_set_cas(self) -> None:
        """the pre-existing ``UPDATE ... WHERE cas = $n`` shape is preserved verbatim."""
        pool = _RecordingPool()
        pool.execute_status = "UPDATE 1"
        coll = _UnfencedCounterCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_store(_row(), original_timestamp=datetime(2026, 1, 1, tzinfo=UTC))
        sql = pool.calls[0][1]
        assert sql.startswith("UPDATE plain_counters SET")
        assert "AND date_updated = $" in sql
        assert "IS NOT DISTINCT FROM" not in sql

    def test_unflagged_collection_does_not_claim_a_fence(self) -> None:
        """``emits_cas_fence`` tracks the flag, and is False by default."""
        pool = _RecordingPool()
        coll = _UnfencedCounterCollection(_registry(pool), _config(), nats_client=_nats())
        assert coll.emits_cas_fence is False


# ---------------------------------------------------------------------------
# the fence itself
# ---------------------------------------------------------------------------


class TestNullSafeFenceShape:
    """what ``cas_null_safe=True`` actually emits."""

    @pytest.mark.asyncio
    async def test_first_write_is_fenced_not_unfenced(self) -> None:
        """THE DEFECT: ``cas=None`` used to fall through to an unfenced upsert.

        with the flag on it must stay fenced, and fence NULL-safely -- plain
        ``=`` can never match the NULL a first write expects.
        """
        pool = _RecordingPool()
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_store(_row(), original_timestamp=None)
        sql = pool.calls[0][1]
        assert sql.startswith("INSERT INTO counters")
        assert "ON CONFLICT (id) DO UPDATE SET" in sql
        assert "WHERE counters.date_updated IS NOT DISTINCT FROM $6" in sql
        # the fence value is the last positional arg, and it is NULL
        assert pool.calls[0][2][-1] is None

    @pytest.mark.asyncio
    async def test_subsequent_write_uses_the_same_statement_shape(self) -> None:
        """a non-NULL expected value rides the same upsert, not a bare UPDATE.

        one shape for create and update alike is what makes this a drop-in for
        the hand-written ``save_to_store`` in the dependants, whose single
        statement serves both.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 1"
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        expected = datetime(2026, 1, 1, tzinfo=UTC)
        await coll.save_to_store(_row(), original_timestamp=expected)
        sql = pool.calls[0][1]
        assert sql.startswith("INSERT INTO counters")
        assert "WHERE counters.date_updated IS NOT DISTINCT FROM $6" in sql
        assert pool.calls[0][2][-1] == expected

    @pytest.mark.asyncio
    async def test_fence_column_is_advanced_by_the_update(self) -> None:
        """``DO UPDATE SET`` must write the fence column, or it never advances."""
        pool = _RecordingPool()
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_store(_row())
        sql = pool.calls[0][1]
        assert "date_updated = EXCLUDED.date_updated" in sql

    @pytest.mark.asyncio
    async def test_immutable_columns_stay_out_of_the_set_clause(self) -> None:
        """the fence does not smuggle immutable columns into DO UPDATE SET."""
        pool = _RecordingPool()
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        await coll.save_to_store(_row())
        sql = pool.calls[0][1]
        assert "date_created = EXCLUDED.date_created" not in sql

    @pytest.mark.asyncio
    async def test_zero_rowcount_reaches_the_caller_as_lost_the_race(self) -> None:
        """0 affected rows is the loser's signal, and it is not swallowed."""
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 0"
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        assert await coll.save_to_store(_row(), original_timestamp=None) == 0

    @pytest.mark.asyncio
    async def test_losing_first_writer_gets_a_retryable_error(self) -> None:
        """0 rows on an ``is_new=False`` entity raises ``ConcurrentModificationError``.

        that is the exception the dependants' retry loops catch. an
        ``is_new=True`` entity would get an unretryable ``RuntimeError``
        instead, which is why they construct first writes ``is_new=False``.
        """
        pool = _RecordingPool()
        pool.execute_status = "INSERT 0 0"
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        entity = _CounterEntity(_row(date_updated=None), is_new=False, collection=coll)
        with pytest.raises(ConcurrentModificationError):
            await coll.save_entity(entity)

    def test_collection_reports_that_it_fences(self) -> None:
        """``emits_cas_fence`` is True so the framework can treat 0 as a loss."""
        pool = _RecordingPool()
        coll = _CounterCollection(_registry(pool), _config(), nats_client=_nats())
        assert coll.emits_cas_fence is True


class TestNullSafeFenceGenerator:
    """direct coverage of the pure SQL builders."""

    def test_qualified_table_name_fences_on_the_unqualified_relation(self) -> None:
        """``ON CONFLICT DO UPDATE ... WHERE`` addresses the target by relation name."""
        schema = _counter_schema(name="analytics.counters")
        sql = schema_sql.build_cas_upsert_sql(schema, _row())
        assert "INSERT INTO analytics.counters" in sql
        assert "WHERE counters.date_updated IS NOT DISTINCT FROM" in sql

    def test_fence_param_is_appended_last_and_may_be_none(self) -> None:
        """params are the INSERT's own, plus the fence value at the end."""
        schema = _counter_schema()
        row = _row()
        insert_params = schema_sql.build_insert_params(schema, row)
        upsert_params = schema_sql.build_cas_upsert_params(schema, row, None)
        assert upsert_params[:-1] == insert_params
        assert upsert_params[-1] is None

    def test_builder_refuses_a_schema_without_a_cas_column(self) -> None:
        """defensive: the builder never emits a fence it cannot name."""
        schema = TableSchema(
            name="counters",
            primary_key="id",
            columns=[Column("id", UUID_TYPE), Column("count", INT_TYPE)],
        )
        with pytest.raises(RuntimeError, match="cas_column"):
            schema_sql.build_cas_upsert_sql(schema, {"id": uuid.uuid4(), "count": 1})

    def test_builder_refuses_a_non_update_conflict_mode(self) -> None:
        """the fence lives on the DO UPDATE branch, which other modes never emit."""
        schema = TableSchema(
            name="counters",
            primary_key="id",
            columns=[Column("id", UUID_TYPE), Column("date_updated", DATETIMETZ_TYPE)],
            cas_column="date_updated",
            on_conflict="ignore",
        )
        with pytest.raises(RuntimeError, match="on_conflict"):
            schema_sql.build_cas_upsert_sql(schema, {"id": uuid.uuid4(), "date_updated": None})


# ---------------------------------------------------------------------------
# preconditions -- every one of these degrades silently if unchecked
# ---------------------------------------------------------------------------


class TestOptInPreconditions:
    """``TableSchema`` rejects a fence that could never work."""

    def test_requires_a_cas_column(self) -> None:
        """there is nothing to fence on without one."""
        with pytest.raises(ValueError, match="requires cas_column to be set"):
            TableSchema(
                name="counters",
                primary_key="id",
                columns=[Column("id", UUID_TYPE), Column("count", INT_TYPE)],
                cas_null_safe=True,
            )

    def test_requires_on_conflict_update(self) -> None:
        """``raise`` / ``ignore`` never emit the DO UPDATE branch the fence rides."""
        with pytest.raises(ValueError, match="requires on_conflict='update'"):
            _counter_schema(on_conflict="ignore")

    def test_rejects_an_immutable_cas_column(self) -> None:
        """an immutable fence column is dropped from SET, so it never advances."""
        with pytest.raises(ValueError, match="mutable non-pk column"):
            _counter_schema(
                columns=[
                    Column("id", UUID_TYPE),
                    Column("count", INT_TYPE),
                    Column("date_updated", DATETIMETZ_TYPE, immutable=True),
                ],
            )

    def test_rejects_a_pk_cas_column(self) -> None:
        """same reasoning: pk columns are excluded from DO UPDATE SET."""
        with pytest.raises(ValueError, match="mutable non-pk column"):
            TableSchema(
                name="counters",
                primary_key="date_updated",
                columns=[Column("date_updated", DATETIMETZ_TYPE), Column("count", INT_TYPE)],
                cas_column="date_updated",
                cas_null_safe=True,
            )

    def test_rejects_a_server_default_cas_column(self) -> None:
        """an omitted server-default column is dropped from the INSERT entirely."""
        with pytest.raises(ValueError, match="no server_default"):
            _counter_schema(
                columns=[
                    Column("id", UUID_TYPE),
                    Column("count", INT_TYPE),
                    Column("date_updated", DATETIMETZ_TYPE, server_default="now()"),
                ],
            )

    def test_deferred_flush_is_refused_at_construction(self) -> None:
        """a buffered write is replayed with no fence value, so it would be dropped."""
        pool = _RecordingPool()
        config = _config(collection_flush="ON_SCHEDULE", collection_flush_tables="counters")
        with pytest.raises(ValueError, match="cas_null_safe=True but is also listed"):
            _CounterCollection(_registry(pool), config, nats_client=_nats(), write_buffer=AsyncMock())

    def test_deferred_flush_of_an_unflagged_table_is_still_allowed(self) -> None:
        """the guard is scoped to the opt-in and changes nothing else."""
        pool = _RecordingPool()
        config = _config(collection_flush="ON_SCHEDULE", collection_flush_tables="plain_counters")
        coll = _UnfencedCounterCollection(_registry(pool), config, nats_client=_nats(), write_buffer=AsyncMock())
        assert coll.emits_cas_fence is False

    def test_no_write_buffer_means_no_deferral_and_no_complaint(self) -> None:
        """the config alone is harmless: with no buffer wired, nothing defers."""
        pool = _RecordingPool()
        config = _config(collection_flush="ON_SCHEDULE", collection_flush_tables="counters")
        coll = _CounterCollection(_registry(pool), config, nats_client=_nats(), write_buffer=None)
        assert coll.emits_cas_fence is True
