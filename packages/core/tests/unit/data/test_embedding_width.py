"""the configured embedding width: its range, the backend ceiling, the derived schema, the write refusal.

The DDL that reads and changes a stored width runs against real PostgreSQL in the agent-memory and
agent-intention integration suites; this module pins what needs no database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    Index,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.embedding_width import (
    DEFAULT_EMBEDDING_DIM,
    MAX_EMBEDDING_DIM,
    POSTGRES_MAX_INDEXED_EMBEDDING_DIM,
    EmbeddingColumn,
    EmbeddingWidthMismatch,
    check_embedding_dim,
    embedding_columns,
    require_indexable_embedding_dim,
    resize_embedding_columns,
    schema_with_embedding_dim,
)
from threetears.core.entities.base import BaseEntity

_POSTGRES_VERSION = "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+2) on x86_64-pc-linux-gnu"
_YUGABYTE_VERSION = "PostgreSQL 15.12-YB-2025.2.0.0-b0 on x86_64-pc-linux-gnu"


# parity-with: threetears.core.data.embedding_width.EmbeddingWidthStore
class _FakeStore:
    """answers ``SELECT version()`` as the backend it is told to be, and records every statement."""

    def __init__(self, version: str) -> None:
        """record the version string ``SELECT version()`` answers.

        :param version: the server's version string
        :ptype version: str
        """
        self.statements: list[str] = []
        self._version = version

    async def execute(self, sql: str, *params: Any) -> str:
        """record ``sql`` and report it done.

        :param sql: statement text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: a status tag
        :rtype: str
        """
        self.statements.append(sql)
        return "OK"

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """record ``sql`` and answer the version query.

        :param sql: query text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: the version row
        :rtype: list[dict[str, Any]]
        """
        self.statements.append(sql)
        return [{"version": self._version}]


_SCHEMA = TableSchema(
    name="notes",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("body", STRING_TYPE),
        Column("embedding", VECTOR_TYPE, nullable=True, vector_dim=4),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    indexes=(
        Index("ix_notes_body", "body"),
        Index("ix_notes_embedding", "embedding", using="hnsw", ops={"embedding": "vector_cosine_ops"}),
    ),
)


class _Note(BaseEntity):
    primary_key_field = "id"


class _NotesCollection(SchemaBackedCollection[_Note]):
    """a table with a four-dimensional embedding column."""

    primary_key_column: str = "id"
    schema = _SCHEMA

    @property
    def table_name(self) -> str:
        """the table name."""
        return "notes"

    @property
    def entity_class(self) -> type[_Note]:
        """the entity class."""
        return _Note


class _RecordingPool:
    """an asyncpg-pool shape that records every call and reports one row written."""

    def __init__(self) -> None:
        """start with no calls."""
        self.calls: list[str] = []

    async def execute(self, sql: str, *args: Any) -> str:
        """record the call."""
        self.calls.append(sql)
        return "INSERT 0 1"

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        """record the call; nothing is held."""
        self.calls.append(sql)
        return None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """record the call; nothing is held."""
        self.calls.append(sql)
        return []


def _collection(pool: _RecordingPool) -> _NotesCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return _NotesCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


def _row(embedding: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {"id": uuid.uuid4(), "body": "a note", "embedding": embedding, "date_created": now, "date_updated": now}


class TestTheConfiguredWidth:
    @pytest.mark.parametrize("width", [1, DEFAULT_EMBEDDING_DIM, 1536, 3072, MAX_EMBEDDING_DIM])
    def test_a_width_any_backend_can_index_is_kept(self, width: int) -> None:
        assert check_embedding_dim(width) == width

    @pytest.mark.parametrize("width", [0, -1, MAX_EMBEDDING_DIM + 1])
    def test_a_width_outside_one_to_sixteen_thousand_is_refused(self, width: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 16,000"):
            check_embedding_dim(width)

    @pytest.mark.parametrize("width", [True, 1024.0, "1024"])
    def test_a_width_that_is_not_an_int_is_refused(self, width: Any) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            check_embedding_dim(width)

    def test_the_default_is_the_width_the_tables_were_built_with(self) -> None:
        assert DEFAULT_EMBEDDING_DIM == 1024


class TestTheBackendCeiling:
    async def test_a_width_postgres_indexes_asks_the_database_nothing(self) -> None:
        store = _FakeStore(_POSTGRES_VERSION)
        await require_indexable_embedding_dim(store, POSTGRES_MAX_INDEXED_EMBEDDING_DIM)
        assert store.statements == []

    async def test_postgres_refuses_a_width_above_two_thousand_naming_both_ceilings(self) -> None:
        store = _FakeStore(_POSTGRES_VERSION)
        with pytest.raises(ValueError, match="at most 2,000") as raised:
            await require_indexable_embedding_dim(store, POSTGRES_MAX_INDEXED_EMBEDDING_DIM + 1)
        assert "16,000" in str(raised.value)
        assert "YugabyteDB" in str(raised.value)

    @pytest.mark.parametrize("width", [POSTGRES_MAX_INDEXED_EMBEDDING_DIM + 1, 3072, MAX_EMBEDDING_DIM])
    async def test_yugabyte_indexes_up_to_sixteen_thousand(self, width: int) -> None:
        store = _FakeStore(_YUGABYTE_VERSION)
        await require_indexable_embedding_dim(store, width)
        assert store.statements == ["SELECT version() AS version"]

    async def test_a_width_above_sixteen_thousand_is_refused_before_asking(self) -> None:
        store = _FakeStore(_YUGABYTE_VERSION)
        with pytest.raises(ValueError, match="between 1 and 16,000"):
            await require_indexable_embedding_dim(store, MAX_EMBEDDING_DIM + 1)
        assert store.statements == []


class TestTheDerivedSchema:
    def test_every_vector_column_takes_the_width_and_nothing_else_changes(self) -> None:
        derived = schema_with_embedding_dim(_SCHEMA, 1536)
        assert derived.column("embedding").vector_dim == 1536
        assert [c.name for c in derived.columns] == [c.name for c in _SCHEMA.columns]
        unchanged = [c for c in derived.columns if c.name != "embedding"]
        assert unchanged == [c for c in _SCHEMA.columns if c.name != "embedding"]
        assert (derived.name, derived.pk_columns, derived.indexes, derived.cas_column) == (
            _SCHEMA.name,
            _SCHEMA.pk_columns,
            _SCHEMA.indexes,
            _SCHEMA.cas_column,
        )

    def test_the_declared_schema_is_left_alone(self) -> None:
        schema_with_embedding_dim(_SCHEMA, 1536)
        assert _SCHEMA.column("embedding").vector_dim == 4

    def test_the_declared_width_returns_the_declared_schema(self) -> None:
        assert schema_with_embedding_dim(_SCHEMA, 4) is _SCHEMA

    def test_a_schema_with_no_vector_column_is_refused(self) -> None:
        plain = TableSchema(name="plain", primary_key="id", columns=[Column("id", UUID_TYPE)])
        with pytest.raises(ValueError, match="no vector column"):
            schema_with_embedding_dim(plain, 1536)

    def test_the_sqlalchemy_table_carries_the_width(self) -> None:
        from sqlalchemy import MetaData

        table = schema_with_embedding_dim(_SCHEMA, 1536).to_sqlalchemy_table(MetaData())
        assert table.c.embedding.type.dim == 1536

    def test_the_embedding_columns_carry_the_indexes_over_them(self) -> None:
        assert embedding_columns(_SCHEMA) == (
            EmbeddingColumn(table="notes", column="embedding", indexes=(_SCHEMA.indexes[1],)),
        )


class TestAMisfitVectorIsRefusedBeforeAnyStatement:
    async def test_save_entity_refuses_a_vector_of_another_width(self) -> None:
        pool = _RecordingPool()
        notes = _collection(pool)
        with pytest.raises(ValueError, match=r"expected 4, got 3"):
            await notes.save_entity(notes.create(_row([0.1, 0.2, 0.3])))
        assert pool.calls == []

    async def test_save_entity_refuses_bracketed_text_of_another_width(self) -> None:
        pool = _RecordingPool()
        notes = _collection(pool)
        with pytest.raises(ValueError, match=r"expected 4, got 5"):
            await notes.save_entity(notes.create(_row("[1,2,3,4,5]")))
        assert pool.calls == []

    async def test_save_entity_writes_a_vector_of_the_declared_width(self) -> None:
        pool = _RecordingPool()
        notes = _collection(pool)
        await notes.save_entity(notes.create(_row([0.1, 0.2, 0.3, 0.4])))
        assert any(sql.lstrip().upper().startswith("INSERT") for sql in pool.calls)

    async def test_save_entity_writes_a_row_with_no_embedding(self) -> None:
        pool = _RecordingPool()
        notes = _collection(pool)
        await notes.save_entity(notes.create(_row(None)))
        assert any(sql.lstrip().upper().startswith("INSERT") for sql in pool.calls)

    async def test_a_derived_width_is_the_one_enforced(self) -> None:
        pool = _RecordingPool()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)

        class _WideNotes(_NotesCollection):
            schema = schema_with_embedding_dim(_SCHEMA, 6)

        notes = _WideNotes(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))
        with pytest.raises(ValueError, match=r"expected 6, got 4"):
            await notes.save_entity(notes.create(_row([0.1, 0.2, 0.3, 0.4])))
        assert pool.calls == []


class _Conn:
    """an asyncpg-shaped connection with a transaction, recording every statement run on it."""

    def __init__(self) -> None:
        """start with nothing run."""
        self.executed: list[str] = []

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        """open a transaction (statements run immediately; nothing here rolls back)."""
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        """record the statement."""
        self.executed.append(sql)
        return f"INSERT 0 {sql.count('),(') + 1}"


class TestABatchWithAMisfitVectorIsRefusedBeforeAnyStatement:
    async def test_save_rows_refuses_a_batch_holding_one_vector_of_another_width(self) -> None:
        from threetears.core.collections.caller_transaction import CallerTransaction

        notes, conn = _collection(_RecordingPool()), _Conn()
        with pytest.raises(ValueError, match=r"expected 4, got 3"):
            async with CallerTransaction(conn):
                await notes.save_rows([_row([0.1, 0.2, 0.3, 0.4]), _row([0.1, 0.2, 0.3])], conn=conn)
        assert conn.executed == [], "the whole batch is refused, the fitting row included"

    async def test_save_rows_writes_a_batch_of_the_declared_width(self) -> None:
        from threetears.core.collections.caller_transaction import CallerTransaction

        notes, conn = _collection(_RecordingPool()), _Conn()
        async with CallerTransaction(conn):
            written = await notes.save_rows([_row([0.1, 0.2, 0.3, 0.4]), _row(None)], conn=conn)
        assert written == 2 and len(conn.executed) == 1


class TestTextThatIsNotAVector:
    @pytest.mark.parametrize("text", ["", "abc", "[1,2"])
    async def test_it_is_left_for_the_database_to_judge_not_refused_with_an_unrelated_error(self, text: str) -> None:
        """the width check measures what it can read; unreadable text is not its to refuse."""
        pool = _RecordingPool()
        notes = _collection(pool)
        await notes.save_entity(notes.create(_row(text)))
        assert any(sql.lstrip().upper().startswith("INSERT") for sql in pool.calls)


class TestTheMismatchError:
    def test_it_names_the_table_both_widths_and_the_resize(self) -> None:
        error = EmbeddingWidthMismatch("memories", "embedding", 1024, 1536)
        message = str(error)
        assert "memories.embedding" in message
        assert "vector(1024)" in message
        assert "1536" in message
        assert "resize_embedding_width" in message
        assert "re-embedded" in message
        assert (error.table, error.column, error.stored_width, error.configured_width) == (
            "memories",
            "embedding",
            1024,
            1536,
        )


_YB_INDEXDEF = 'CREATE INDEX ix_notes_embedding ON "zz".notes USING ybhnsw (embedding vector_cosine_ops)'
_PG_INDEXDEF = (
    "CREATE INDEX ix_notes_embedding ON zz.notes USING hnsw (embedding vector_cosine_ops) "
    "WITH (m='16', ef_construction='64')"
)


# parity-with: threetears.core.data.embedding_width.EmbeddingWidthStore
class _FakeCatalogueStore:
    """a schema holding ``notes.embedding`` at ``vector(4)`` with one index, as the catalogue describes it."""

    def __init__(self, index_definition: str, reloptions: list[str] | None) -> None:
        """record what the catalogue answers for the column's one index.

        :param index_definition: what ``pg_get_indexdef`` answers
        :ptype index_definition: str
        :param reloptions: what ``pg_class.reloptions`` answers for the index
        :ptype reloptions: list[str] | None
        """
        self.executed: list[str] = []
        self._definition = index_definition
        self._reloptions = reloptions
        self._width = 4

    async def execute(self, sql: str, *params: Any) -> str:
        """record ``sql``; an ``ALTER`` moves the column to its new width.

        :param sql: statement text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: a status tag
        :rtype: str
        """
        self.executed.append(sql)
        if sql.startswith("ALTER TABLE"):
            self._width = int(sql.rsplit("vector(", 1)[1].split(")", 1)[0])
        return "OK"

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """answer the catalogue reads a resize makes.

        :param sql: query text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        rows: list[dict[str, Any]]
        if "current_schema() AS schema_name" in sql:
            rows = [{"schema_name": "zz"}]
        elif "format_type" in sql:
            rows = [{"column_type": f"vector({self._width})"}]
        elif "pg_get_indexdef" in sql:
            rows = [
                {
                    "index_name": "ix_notes_embedding",
                    "index_definition": self._definition,
                    "index_options": self._reloptions,
                }
            ]
        elif "count(*)" in sql:
            rows = [{"held": 0}]
        else:
            rows = [{"version": _POSTGRES_VERSION}]
        return rows


def _created(store: _FakeCatalogueStore) -> list[str]:
    return [sql for sql in store.executed if sql.startswith("CREATE") and "ix_notes_embedding" in sql]


class TestAResizeKeepsTheIndexOptions:
    async def test_a_definition_without_with_takes_the_stored_options(self) -> None:
        """YugabyteDB's ``pg_get_indexdef`` leaves out ``WITH (...)``; the options live in ``reloptions``."""
        store = _FakeCatalogueStore(_YB_INDEXDEF, ["m=16", "ef_construction=64"])
        await resize_embedding_columns(store, embedding_columns(_SCHEMA), 6)
        assert _created(store)[0] == (
            'CREATE INDEX IF NOT EXISTS ix_notes_embedding ON "zz".notes USING ybhnsw (embedding vector_cosine_ops) '
            "WITH (m=16, ef_construction=64)"
        )

    async def test_a_definition_carrying_with_is_left_as_it_is(self) -> None:
        store = _FakeCatalogueStore(_PG_INDEXDEF, ["m=16", "ef_construction=64"])
        await resize_embedding_columns(store, embedding_columns(_SCHEMA), 6)
        assert _created(store)[0] == _PG_INDEXDEF.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)

    async def test_an_index_with_no_options_is_recreated_as_defined(self) -> None:
        store = _FakeCatalogueStore(_YB_INDEXDEF, None)
        await resize_embedding_columns(store, embedding_columns(_SCHEMA), 6)
        assert _created(store)[0] == _YB_INDEXDEF.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)

    async def test_the_options_go_before_a_partial_index_predicate(self) -> None:
        store = _FakeCatalogueStore(_YB_INDEXDEF + " WHERE (body IS NOT NULL)", ["m=16"])
        await resize_embedding_columns(store, embedding_columns(_SCHEMA), 6)
        assert _created(store)[0].endswith("(embedding vector_cosine_ops) WITH (m=16) WHERE (body IS NOT NULL)")


def test_a_declared_desc_column_is_recreated_descending() -> None:
    """an index rebuilt from its declaration keeps a ``desc`` column descending."""
    from threetears.core.data.embedding_width import _declared_index_sql

    index = Index("ix_t_recent", "agent_id", "date_created", desc="date_created")
    assert _declared_index_sql("public.t", index) == (
        'CREATE INDEX IF NOT EXISTS "ix_t_recent" ON public.t ("agent_id", "date_created" DESC)'
    )
