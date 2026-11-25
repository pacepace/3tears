"""the SQLite store keeps the L3 contract (``tests/l3_contract.py``), and holds its file as a store must.

The contract is the one ``SqlL3Backend`` keeps over Postgres in the integration suite. Below it: what the file
itself owes -- one process, a format it can read, tables it can read -- and the collection a deployment builds over
it, with no L2 and the ordinary L1 cache in front.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest

from packages.core.tests.l3_contract import CONTRACT_SCHEMAS, CollectionOverStoreContract, SqlStoreContract
from threetears.core.backends import SqliteL3Backend, l3_store
from threetears.core.backends.sqlite_file import SqliteStoreFormatError, SqliteStoreLockedError
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    Index,
    TableSchema,
    UniqueConstraintDef,
)
from threetears.core.testing.l1_tables import TEST_L1_TABLE


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteL3Backend]:
    """an open store over a fresh file, holding the contract's tables."""
    # the configuration's one setting, as a deployment names its L3
    async with l3_store(f"sqlite:///{tmp_path / 'state.sqlite'}") as opened:
        assert isinstance(opened, SqliteL3Backend)
        for schema in CONTRACT_SCHEMAS:
            opened.register_schema(schema.name, schema)
        yield opened


class TestTheSqliteStoreKeepsTheL3Contract(SqlStoreContract):
    """the schema tier, on SQLite: the same statements, the same answers, the same exceptions."""


def _schema(*extra: Column, name: str = "things", unique: UniqueConstraintDef | None = None) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("value", STRING_TYPE), *extra],
        unique_constraints=() if unique is None else (unique,),
        l1_eviction=TEST_L1_TABLE,
    )


class TestCollectionsOverTheSqliteStore(CollectionOverStoreContract):
    """a collection over this store, with no L2 and with L2."""


class TestTheFile:
    async def test_rows_written_are_read_by_the_next_opening(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "state.sqlite"
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
            await first.upsert("things", {"id": "a", "value": "kept"})
        async with SqliteL3Backend(path) as second:
            second.register_schema("things", _schema())
            assert await second.fetch_one("things", {"id": "a"}) == {"id": "a", "value": "kept"}

    async def test_a_second_holder_is_refused_until_the_first_closes(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        first = SqliteL3Backend(path)
        first.open()
        try:
            with pytest.raises(SqliteStoreLockedError, match="another process"):
                SqliteL3Backend(path).open()
        finally:
            first.close()
        again = SqliteL3Backend(path)
        again.open()
        again.close()

    async def test_the_file_is_in_wal_mode_and_syncs_every_commit(self, store: SqliteL3Backend) -> None:
        assert await store.fetchval("PRAGMA journal_mode") == "wal"
        # 2 is FULL; the pragma reads the connection it runs on, and every connection is set alike
        assert await store.fetchval("PRAGMA synchronous") == 2

    async def test_a_file_of_another_format_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE things (id TEXT)")
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        with pytest.raises(SqliteStoreFormatError, match="format 99"):
            SqliteL3Backend(path).open()

    async def test_a_file_some_other_program_wrote_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE things (id TEXT)")
        conn.commit()
        conn.close()
        with pytest.raises(SqliteStoreFormatError, match="format 0"):
            SqliteL3Backend(path).open()

    async def test_a_new_nullable_column_is_added_to_a_stored_table(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
            await first.upsert("things", {"id": "a", "value": "old"})
        async with SqliteL3Backend(path) as second:
            second.register_schema("things", _schema(Column("note", STRING_TYPE, nullable=True)))
            assert await second.fetch_one("things", {"id": "a"}) == {"id": "a", "value": "old", "note": None}

    async def test_a_new_column_with_a_default_takes_it_in_stored_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
            await first.upsert("things", {"id": "a", "value": "old"})
        async with SqliteL3Backend(path) as second:
            second.register_schema("things", _schema(Column("tries", INT_TYPE, server_default="0")))
            assert await second.fetch_one("things", {"id": "a"}) == {"id": "a", "value": "old", "tries": 0}

    @pytest.mark.parametrize(
        "changed",
        [
            pytest.param(TableSchema(name="things", primary_key="id", columns=[Column("id", STRING_TYPE)]), id="drop"),
            pytest.param(
                TableSchema(
                    name="things", primary_key="id", columns=[Column("id", STRING_TYPE), Column("value", INT_TYPE)]
                ),
                id="retype",
            ),
            pytest.param(_schema(Column("required", STRING_TYPE)), id="required-new"),
        ],
    )
    async def test_a_table_declared_in_another_shape_is_refused(self, tmp_path: Path, changed: TableSchema) -> None:
        path = tmp_path / "state.sqlite"
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
        async with SqliteL3Backend(path) as second:
            with pytest.raises(SqliteStoreFormatError, match="another shape"):
                second.register_schema("things", changed)

    async def test_a_new_column_defaulted_to_now_is_refused_naming_it(self, tmp_path: Path) -> None:
        # SQLite adds no column whose default is an expression; rebuilding the table would read every row
        path = tmp_path / "state.sqlite"
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
            await first.upsert("things", {"id": "a", "value": "old"})
        stamped = _schema(Column("seen", DATETIMETZ_TYPE, server_default="now()"))
        async with SqliteL3Backend(path) as second:
            with pytest.raises(SqliteStoreFormatError, match=r"now\(\)\): \['seen'\]"):
                second.register_schema("things", stamped)
            # nothing changed: the stored shape still registers, and its row reads
            second.register_schema("things", _schema())
            assert await second.fetch_one("things", {"id": "a"}) == {"id": "a", "value": "old"}

    async def test_a_unique_constraint_is_added_to_a_stored_table(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        unique = UniqueConstraintDef("things_value_key", ("value",))
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
            await first.upsert("things", {"id": "a", "value": "old"})
        async with SqliteL3Backend(path) as second:
            second.register_schema("things", _schema(unique=unique))
            with pytest.raises(asyncpg.UniqueViolationError):
                await second.upsert("things", {"id": "b", "value": "old"})

    async def test_a_unique_index_over_stored_duplicates_is_refused_naming_it(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        async with SqliteL3Backend(path) as first:
            first.register_schema("things", _schema())
            await first.upsert("things", {"id": "a", "value": "same"})
            await first.upsert("things", {"id": "b", "value": "same"})
        async with SqliteL3Backend(path) as second:
            with pytest.raises(SqliteStoreFormatError, match="things_value_key"):
                second.register_schema("things", _schema(unique=UniqueConstraintDef("things_value_key", ("value",))))

    async def test_a_unique_index_declared_nulls_not_distinct_holds_one_null(self, store: SqliteL3Backend) -> None:
        schema = TableSchema(
            name="owned",
            primary_key="id",
            columns=[Column("id", STRING_TYPE), Column("owner", STRING_TYPE, nullable=True)],
            indexes=(Index("owned_owner_key", "owner", unique=True, nulls_not_distinct=True),),
            l1_eviction=TEST_L1_TABLE,
        )
        store.register_schema("owned", schema)
        await store.upsert("owned", {"id": "a", "owner": None})
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.upsert("owned", {"id": "b", "owner": None})

    async def test_a_foreign_file_is_refused_as_found_not_turned_to_wal(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE things (id TEXT)")
        conn.commit()
        conn.close()
        with pytest.raises(SqliteStoreFormatError):
            SqliteL3Backend(path).open()
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        finally:
            conn.close()

    async def test_now_as_a_default_writes_the_instant(self, store: SqliteL3Backend) -> None:
        schema = _schema(Column("seen", DATETIMETZ_TYPE, server_default="now()"), name="stamped")
        store.register_schema("stamped", schema)
        await store.upsert("stamped", {"id": "a", "value": "v"})
        row = await store.fetch_one("stamped", {"id": "a"})
        assert row is not None
        assert row["seen"].utcoffset() is not None

    async def test_a_default_it_cannot_spell_is_refused(self, store: SqliteL3Backend) -> None:
        schema = _schema(Column("token", STRING_TYPE, server_default="gen_random_uuid()::text"), name="tokens")
        with pytest.raises(SqliteStoreFormatError, match="cannot spell"):
            store.register_schema("tokens", schema)

    async def test_a_schema_qualified_table_is_refused(self, store: SqliteL3Backend) -> None:
        with pytest.raises(ValueError, match="qualified"):
            store.register_schema("analytics.things", _schema(name="analytics.things"))

    async def test_a_store_not_opened_says_so(self, tmp_path: Path) -> None:
        closed = SqliteL3Backend(tmp_path / "state.sqlite")
        with pytest.raises(RuntimeError, match="not open"):
            closed.register_schema("things", _schema())
        with pytest.raises(RuntimeError, match="not open"):
            await closed.health()

    async def test_health_answers_while_the_file_reads(self, store: SqliteL3Backend) -> None:
        await store.health()


class TestOneWriterProcess:
    """many tasks of the one process writing at once: every write lands, and a transaction is never interleaved."""

    async def test_concurrent_writes_of_distinct_keys_all_land(self, store: SqliteL3Backend) -> None:
        await asyncio.gather(
            *(store.upsert("l3c_ignoring", {"id": f"k{i}", "value": str(i)}) for i in range(200)),
        )
        assert len(await store.scan("l3c_ignoring")) == 200

    async def test_read_modify_write_under_the_fence_loses_no_increment(self, store: SqliteL3Backend) -> None:
        await store.upsert("l3c_null_safe", {"id": "n", "value": 0, "date_updated": datetime.now(UTC)}, cas=None)

        async def increment() -> None:
            while True:
                row = await store.fetch_one("l3c_null_safe", {"id": "n"})
                assert row is not None
                await asyncio.sleep(0)
                written = await store.upsert(
                    "l3c_null_safe",
                    {"id": "n", "value": row["value"] + 1, "date_updated": datetime.now(UTC)},
                    cas=row["date_updated"],
                )
                if written == 1:
                    return

        await asyncio.gather(*(increment() for _ in range(40)))
        row = await store.fetch_one("l3c_null_safe", {"id": "n"})
        assert row is not None
        assert row["value"] == 40

    async def test_a_transaction_is_not_interleaved_with_another_tasks_write(self, store: SqliteL3Backend) -> None:
        entered = asyncio.Event()
        order: list[str] = []

        async def transaction() -> None:
            async with store.transaction() as conn:
                entered.set()
                await store.upsert("l3c_ignoring", {"id": "t1", "value": "tx"}, conn=conn)
                await asyncio.sleep(0.05)
                order.append("transaction committed")

        async def other() -> None:
            await entered.wait()
            await store.upsert("l3c_ignoring", {"id": "o", "value": "other"})
            order.append("other written")

        await asyncio.gather(transaction(), other())
        assert order == ["transaction committed", "other written"]

    async def test_reads_see_the_last_commit_while_a_transaction_is_open(self, store: SqliteL3Backend) -> None:
        await store.upsert("l3c_ignoring", {"id": "a", "value": "committed"})
        async with store.transaction() as conn:
            await store.delete("l3c_ignoring", {"id": "a"}, conn=conn)
            # another reader sees the last commit, not the transaction's delete, and does not wait for it
            seen = await asyncio.wait_for(store.fetch_one("l3c_ignoring", {"id": "a"}), timeout=2)
            assert seen == {"id": "a", "value": "committed"}
        assert await store.fetch_one("l3c_ignoring", {"id": "a"}) is None

    async def test_a_write_outside_the_transaction_its_task_holds_is_refused(self, store: SqliteL3Backend) -> None:
        async with store.transaction():
            with pytest.raises(RuntimeError, match="pass conn="):
                await store.upsert("l3c_ignoring", {"id": "x", "value": "v"})

    async def test_a_nested_transaction_rolls_back_alone(self, store: SqliteL3Backend) -> None:
        async with store.transaction() as conn:
            await store.upsert("l3c_ignoring", {"id": "kept", "value": "v"}, conn=conn)
            with pytest.raises(RuntimeError, match="inner"):
                async with conn.transaction():
                    await store.upsert("l3c_ignoring", {"id": "dropped", "value": "v"}, conn=conn)
                    raise RuntimeError("inner")
        assert [r["id"] for r in await store.scan("l3c_ignoring")] == ["kept"]

    async def test_another_stores_connection_is_refused(self, store: SqliteL3Backend, tmp_path: Path) -> None:
        async with SqliteL3Backend(tmp_path / "other.sqlite") as other, other.transaction() as foreign:
            with pytest.raises(RuntimeError, match="another store"):
                await store.upsert("l3c_ignoring", {"id": "x", "value": "v"}, conn=foreign)


_INSERT = "INSERT INTO l3c_ignoring (id, value) VALUES ($1, $2)"
_VALUE = "SELECT value FROM l3c_ignoring WHERE id = $1::text"


async def _ids(store: SqliteL3Backend) -> list[str]:
    return sorted(row["id"] for row in await store.scan("l3c_ignoring"))


class TestRawStatements:
    """the raw-SQL surface beyond ``store.execute``: batches, and statements on a held connection."""

    async def test_a_batch_in_one_transaction_commits_whole_and_answers_its_tags(self, store: SqliteL3Backend) -> None:
        tags = await store.execute_batch(
            [{"query": _INSERT, "params": ["a", "1"]}, {"query": _INSERT, "params": ["b", "2"]}],
        )
        assert tags == ["INSERT 0 1", "INSERT 0 1"]
        assert await _ids(store) == ["a", "b"]

    async def test_a_batch_in_one_transaction_writes_nothing_when_a_statement_fails(
        self, store: SqliteL3Backend
    ) -> None:
        raising = "INSERT INTO l3c_raising (id, value) VALUES ($1, $2)"
        await store.execute(raising, "taken", "v")
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.execute_batch(
                [{"query": _INSERT, "params": ["a", "1"]}, {"query": raising, "params": ["taken", "again"]}],
            )
        assert await _ids(store) == []

    async def test_a_batch_outside_a_transaction_keeps_the_statements_before_a_failure(
        self, store: SqliteL3Backend
    ) -> None:
        raising = "INSERT INTO l3c_raising (id, value) VALUES ($1, $2)"
        await store.execute(raising, "taken", "v")
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.execute_batch(
                [{"query": _INSERT, "params": ["a", "1"]}, {"query": raising, "params": ["taken", "again"]}],
                transaction=False,
            )
        assert await _ids(store) == ["a"]

    async def test_statements_on_a_transactions_connection_roll_back_with_it(self, store: SqliteL3Backend) -> None:
        with pytest.raises(RuntimeError, match="abandoned"):
            async with store.transaction() as conn:
                assert await conn.execute(_INSERT, "a", "1") == "INSERT 0 1"
                assert await conn.execute("UPDATE l3c_ignoring SET value = $1 WHERE id = $2", "2", "a") == "UPDATE 1"
                # the connection reads its own uncommitted writes, with the Postgres cast dropped
                assert await conn.fetchval(_VALUE, "a") == "2"
                assert await conn.fetchrow(_VALUE, "a") == {"value": "2"}
                assert await conn.fetch("SELECT id FROM l3c_ignoring") == [{"id": "a"}]
                raise RuntimeError("abandoned")
        assert await _ids(store) == []

    async def test_statements_on_an_acquired_connection_commit_each_on_its_own(self, store: SqliteL3Backend) -> None:
        with pytest.raises(RuntimeError, match="abandoned"):
            async with store.acquire() as conn:
                await conn.execute(_INSERT, "a", "1")
                # committed already: the reading connection sees it while the block still holds the writer
                assert await store.fetchval(_VALUE, "a") == "1"
                raise RuntimeError("abandoned")
        assert await _ids(store) == ["a"]

    async def test_a_transaction_opened_on_an_acquired_connection_rolls_back_whole(
        self, store: SqliteL3Backend
    ) -> None:
        async with store.acquire() as conn:
            with pytest.raises(RuntimeError, match="abandoned"):
                async with conn.transaction():
                    await conn.execute(_INSERT, "a", "1")
                    raise RuntimeError("abandoned")
            await conn.execute(_INSERT, "b", "2")
        assert await _ids(store) == ["b"]

    @pytest.mark.parametrize("opened", ["transaction", "acquire"])
    async def test_a_connection_refuses_every_statement_once_its_block_has_ended(
        self, store: SqliteL3Backend, opened: str
    ) -> None:
        async with store.transaction() if opened == "transaction" else store.acquire() as conn:
            pass
        with pytest.raises(RuntimeError, match="after the block"):
            await conn.execute(_INSERT, "a", "1")
        with pytest.raises(RuntimeError, match="after the block"):
            await conn.fetch(_VALUE, "a")
        with pytest.raises(RuntimeError, match="after the block"):
            conn.transaction()
        with pytest.raises(RuntimeError, match="after the block"):
            await store.upsert("l3c_ignoring", {"id": "a", "value": "1"}, conn=conn)
        assert await _ids(store) == []


#: a statement that keeps its worker thread busy for a second or more, so a cancellation lands while it runs
_SLOW = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 10000000) SELECT count(*) FROM c"


class TestACancelledWrite:
    """a write whose task is cancelled while its statement runs: it ends whole or not at all, and the next write lands."""

    async def test_a_cancelled_batch_ends_whole_and_the_next_write_lands(self, store: SqliteL3Backend) -> None:
        # another connection holds the file's write lock, so the batch's BEGIN waits on its worker thread
        other = sqlite3.connect(store.path, isolation_level=None)
        other.execute("BEGIN IMMEDIATE")
        try:
            batch = asyncio.create_task(
                store.execute_batch(
                    [{"query": _INSERT, "params": ["a", "1"]}, {"query": _INSERT, "params": ["b", "2"]}],
                ),
            )
            await asyncio.sleep(0.2)
            batch.cancel()
            following = asyncio.create_task(store.upsert("l3c_ignoring", {"id": "c", "value": "3"}))
            await asyncio.sleep(0.05)
            # the batch is still on the writing connection: its task waits for it, holding the lock
            assert not batch.done()
            assert not following.done()
        finally:
            other.execute("COMMIT")
            other.close()
        with pytest.raises(asyncio.CancelledError):
            await batch
        assert await following == 1
        assert await _ids(store) == ["a", "b", "c"]

    async def test_a_transaction_cancelled_mid_statement_rolls_back_whole(self, store: SqliteL3Backend) -> None:
        async def transaction() -> None:
            async with store.transaction() as conn:
                await conn.execute(_INSERT, "a", "1")
                await conn.fetchval(_SLOW)
                await conn.execute(_INSERT, "b", "2")

        running = asyncio.create_task(transaction())
        await asyncio.sleep(0.1)
        running.cancel()
        await asyncio.sleep(0.05)
        # the slow statement is still on the writing connection: the rollback waits for it
        assert not running.done()
        with pytest.raises(asyncio.CancelledError):
            await running
        await store.upsert("l3c_ignoring", {"id": "c", "value": "3"})
        assert await _ids(store) == ["c"]

    async def test_a_transaction_cancelled_while_it_begins_leaves_none_open(self, store: SqliteL3Backend) -> None:
        # another connection holds the file's write lock, so the BEGIN waits on its worker thread
        other = sqlite3.connect(store.path, isolation_level=None)
        other.execute("BEGIN IMMEDIATE")
        try:

            async def transaction() -> None:
                async with store.transaction() as conn:
                    await conn.execute(_INSERT, "a", "1")

            running = asyncio.create_task(transaction())
            await asyncio.sleep(0.2)
            running.cancel()
            await asyncio.sleep(0.05)
            assert not running.done()
        finally:
            other.execute("COMMIT")
            other.close()
        with pytest.raises(asyncio.CancelledError):
            await running
        await store.upsert("l3c_ignoring", {"id": "c", "value": "3"})
        async with store.transaction() as conn:
            await conn.execute(_INSERT, "d", "4")
        assert await _ids(store) == ["c", "d"]


class TestBuildingACollectionDuringATransaction:
    async def test_a_new_table_is_refused_while_a_transaction_is_open(self, store: SqliteL3Backend) -> None:
        async with store.transaction() as conn:
            await conn.execute(_INSERT, "a", "1")
            with pytest.raises(RuntimeError, match="while a transaction is"):
                store.register_schema("later", _schema(name="later"))
        assert await _ids(store) == ["a"]
        store.register_schema("later", _schema(name="later"))

    async def test_a_table_already_as_declared_registers_while_a_transaction_is_open(
        self, store: SqliteL3Backend
    ) -> None:
        store.register_schema("things", _schema())
        async with store.transaction():
            store.register_schema("things", _schema())


class TestTheL3Url:
    async def test_three_slashes_then_the_path_as_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with l3_store(f"sqlite:///{tmp_path / 'abs.sqlite'}") as absolute:
            assert isinstance(absolute, SqliteL3Backend)
            assert absolute.path == tmp_path / "abs.sqlite"
        monkeypatch.chdir(tmp_path)
        async with l3_store("sqlite:///rel.sqlite") as relative:
            assert isinstance(relative, SqliteL3Backend)
            assert relative.path == Path("rel.sqlite")
        assert (tmp_path / "rel.sqlite").exists()

    @pytest.mark.parametrize("url", ["sqlite://", "sqlite:///", "sqlite://host/x.sqlite"])
    async def test_a_url_naming_no_file_is_refused(self, url: str) -> None:
        with pytest.raises(ValueError, match="sqlite:///<path>"):
            async with l3_store(url):
                pass

    async def test_another_scheme_is_refused(self) -> None:
        with pytest.raises(ValueError, match="scheme 'mysql'"):
            async with l3_store("mysql://h/db"):
                pass

    async def test_pool_options_are_refused_for_a_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no pool"):
            async with l3_store(f"sqlite:///{tmp_path / 'x.sqlite'}", min_size=2):
                pass

    async def test_leaving_the_block_gives_the_file_back(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'x.sqlite'}"
        async with l3_store(url) as store:
            assert isinstance(store, SqliteL3Backend)
            assert store.is_open
        assert not store.is_open
        async with l3_store(url):
            pass
