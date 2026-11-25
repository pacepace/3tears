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

import pytest

from packages.core.tests.l3_contract import CONTRACT_SCHEMAS, CollectionOverStoreContract, SqlStoreContract
from threetears.core.backends import SqliteL3Backend, l3_store
from threetears.core.backends.sqlite_file import SqliteStoreFormatError, SqliteStoreLockedError
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
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
    duplicate_error = sqlite3.IntegrityError


def _schema(*extra: Column, name: str = "things") -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("value", STRING_TYPE), *extra],
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


class TestTheL3Url:
    async def test_three_slashes_then_the_path_as_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with l3_store(f"sqlite:///{tmp_path / 'abs.sqlite'}") as absolute:
            assert absolute.path == tmp_path / "abs.sqlite"
        monkeypatch.chdir(tmp_path)
        async with l3_store("sqlite:///rel.sqlite") as relative:
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
            assert store.is_open
        assert not store.is_open
        async with l3_store(url):
            pass
