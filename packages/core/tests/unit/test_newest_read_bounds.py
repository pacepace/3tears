"""a newest read of a versioned table runs under its bounds on every path, and reads only its own range.

The bounds (``keys.newest_settings``: ``work_mem``, and for a YugabyteDB keyed batch ``yb_fetch_row_limit``) are set
for the read's own transaction:

- over the L3 rail, named on the request (``read_bounds.READ_SETTINGS_FIELD``) of ``l3.query`` and each ``l3.tx``
  read, for the hub's broker to set (a pod cannot: the broker refuses ``SET`` at the door), whether the read goes
  through the proxy, the store over it, or a transaction of either; a read of a plain table names none;
- on the store's direct connection (``SqlL3Backend`` over asyncpg), ``SET LOCAL`` in a transaction around the read.

A key-ordered listing page (``keyset_page``, ``keyset_rows``) names its cursor and its ``LIMIT`` inside the newest
relation's ``DISTINCT ON``, and a YugabyteDB batch of key prefixes reads one prefix a range at a time: neither sorts a
scope's every version in one statement. Their answers are each key's newest live version, on the in-process DuckDB
``test_keyed_newest_versions`` seeds.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from threetears.nats import Subject

from threetears.core.backends import dialect
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.read_bounds import READ_SETTINGS_FIELD, bounded_read, read_setting_fields
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.keyed import KeyBatch, Sort, ThreadedDuckDB, keyset_page, keyset_rows, read_keyed, read_keyed_arrow
from threetears.core.keyset import read_keyset_pages
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.interfaces.keyed.keys import NEWEST_FETCH_ROW_LIMIT, NEWEST_WORK_MEM

_TX_ID = str(uuid.UUID("00000000-0000-4000-8000-000000000905"))

_TABLE = "nb_results"
_PLAIN = "nb_plain"

#: registers the table as versioned: every keyed read of it reads its newest live versions
_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=("state", "id"),
    columns=[Column("state", STRING_TYPE), Column("id", INT_TYPE), Column("v", INT_TYPE)],
    row_epoch=row_versions("state"),
    l1_eviction=TEST_L1_TABLE,
)

_PLAIN_SCHEMA = TableSchema(
    name=_PLAIN,
    primary_key=("state", "id"),
    columns=[Column("state", STRING_TYPE), Column("id", INT_TYPE), Column("v", INT_TYPE)],
    l1_eviction=TEST_L1_TABLE,
)

_BATCH = {READ_SETTINGS_FIELD: {"work_mem": NEWEST_WORK_MEM, "yb_fetch_row_limit": NEWEST_FETCH_ROW_LIMIT}}
_FILL = {READ_SETTINGS_FIELD: {"work_mem": NEWEST_WORK_MEM}}
_SET_BATCH = f"SET LOCAL work_mem = {NEWEST_WORK_MEM}; SET LOCAL yb_fetch_row_limit = {NEWEST_FETCH_ROW_LIMIT}"


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, bool]]:
    """the engine a reader that is no asyncpg connection says it answers from (asked once a reader, as the rail's
    proxy is asked): YugabyteDB unless a test says otherwise."""
    said = {"yugabyte": True}

    async def answers(l3: Any) -> bool:
        return said["yugabyte"]

    monkeypatch.setattr(dialect, "_l3_on_yugabyte", answers)
    yield said


class _Broker:
    """answers each request with rows naming one key's newest version, and keeps each request's body."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        body = json.loads(payload)
        path = str(subject)
        self.sent.append((path, body))
        if path.endswith("l3.tx.begin"):
            return json.dumps({"success": True, "tx_id": _TX_ID}).encode()
        if path.endswith(("l3.tx.commit", "l3.tx.rollback")):
            return json.dumps({"success": True}).encode()
        row = {"state": "CA", "id": 1, "v": 3, "_newest_versions": 3, "_newest_kept": True, "_newest_live": True}
        return json.dumps({"success": True, "rows": [row]}).encode()

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
        )

    def reads(self) -> list[dict[str, Any]]:
        """each read's body, as the broker receives it."""
        return [body for path, body in self.sent if path.endswith(("l3.query", "l3.tx.fetch", "l3.tx.fetchrow"))]


def _keys() -> KeyBatch:
    return KeyBatch(("state", "id"), [["CA"], [1]], ("text", "integer"))


class TestTheField:
    def test_a_read_names_its_bounds_inside_the_block_and_nothing_outside(self) -> None:
        """Flips if a request outside a bounded read carries the field (an older broker's requests change), or one
        inside it does not."""
        assert read_setting_fields() == {}
        with bounded_read((("work_mem", 65_536),)):
            assert read_setting_fields() == {READ_SETTINGS_FIELD: {"work_mem": 65_536}}
        assert read_setting_fields() == {}

    @pytest.mark.parametrize("named", [(("search_path", 1),), (("work_mem", 0),), (("work_mem", True),)])
    def test_a_read_names_no_other_setting_and_no_value_but_a_positive_integer(self, named: Any) -> None:
        """Flips if a read may name a setting past its bounds, or a value the broker could not compare."""
        with pytest.raises(ValueError), bounded_read(named):
            pass


class TestOnTheRail:
    @pytest.mark.parametrize("yugabyte", [True, False])
    async def test_a_keyed_batch_through_the_store_names_its_bounds(
        self, engine: dict[str, bool], yugabyte: bool
    ) -> None:
        """Flips if a keyed batch of a versioned table over the rail reaches the broker without its bounds (the
        broker refuses SET, so they are the request's or none), or names YugabyteDB's fetch limit on PostgreSQL."""
        engine["yugabyte"] = yugabyte
        broker = _Broker()
        store = SqlL3Backend(broker.proxy())
        await read_keyed(store, _TABLE, keys=_keys(), columns=["state", "id", "v"])
        [body] = broker.reads()
        assert {READ_SETTINGS_FIELD: body.get(READ_SETTINGS_FIELD)} == (_BATCH if yugabyte else _FILL)

    async def test_an_arrow_batch_through_the_proxy_names_its_bounds(self, engine: dict[str, bool]) -> None:
        """Flips if a keyed batch read as Arrow over the rail drops its bounds."""
        broker = _Broker()
        types = {"state": pa.string(), "id": pa.int64(), "v": pa.int64()}
        await read_keyed_arrow(broker.proxy(), _TABLE, keys=_keys(), types=types)
        [body] = broker.reads()
        assert body[READ_SETTINGS_FIELD] == _BATCH[READ_SETTINGS_FIELD]

    async def test_a_read_in_a_rail_transaction_names_its_bounds(self, engine: dict[str, bool]) -> None:
        """Flips if a newest read sent as an l3.tx statement drops its bounds."""
        broker = _Broker()
        proxy = broker.proxy()
        async with proxy.transaction() as conn:
            await read_keyed(conn, _TABLE, keys=_keys(), columns=["state", "id", "v"])
        [body] = broker.reads()
        assert broker.sent[1][0].endswith("l3.tx.fetch")
        assert body[READ_SETTINGS_FIELD] == _BATCH[READ_SETTINGS_FIELD]

    async def test_a_fill_page_through_the_paging_proxy_names_its_bounds(self, engine: dict[str, bool]) -> None:
        """Flips if a fill page sent through a transport that answers pages (fetch_page) drops its bounds."""
        broker = _Broker()
        await read_keyset_pages(
            SqlL3Backend(broker.proxy()),
            f"SELECT state, id, v FROM {_TABLE}",
            ["state", "id"],
            where={"state": "CA"},
            page_size=10,
            table=_TABLE,
        )
        assert broker.reads()
        assert all(body[READ_SETTINGS_FIELD] == _FILL[READ_SETTINGS_FIELD] for body in broker.reads())

    async def test_a_read_of_a_plain_table_names_none(self, engine: dict[str, bool]) -> None:
        """Flips if a read of a table that keeps one row a key carries settings (every request but a newest read's
        is as it was)."""
        assert _PLAIN_SCHEMA.name == _PLAIN
        broker = _Broker()
        await read_keyed(SqlL3Backend(broker.proxy()), _PLAIN, keys=_keys(), columns=["state", "id", "v"])
        [body] = broker.reads()
        assert READ_SETTINGS_FIELD not in body


class _DirectConnection(asyncpg.Connection):  # type: ignore[misc]  # asyncpg is untyped
    """an asyncpg connection (as the store and the helpers judge one) that keeps the statements and transactions it is
    sent; never connected."""

    calls: list[str]

    def __del__(self) -> None:
        pass

    def is_closed(self) -> bool:
        return False

    def is_in_transaction(self) -> bool:
        return False

    def transaction(self, **_: Any) -> Any:
        calls = self.calls

        class _Transaction:
            async def __aenter__(self) -> None:
                calls.append("BEGIN")

            async def __aexit__(self, *exc: object) -> None:
                calls.append("COMMIT")

        return _Transaction()

    async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
        self.calls.append(query)
        return "SET"

    async def fetch(self, query: str, *args: Any, timeout: float | None = None, record_class: Any = None) -> list[Any]:
        self.calls.append("fetch")
        return [{"state": "CA", "id": 1, "v": 3, "_newest_versions": 3}]


class TestOnTheStoresDirectConnection:
    async def test_a_keyed_batch_through_the_store_sets_its_bounds_in_its_own_transaction(
        self, engine: dict[str, bool]
    ) -> None:
        """Flips if the store over a direct connection sends a newest read with no SET LOCAL (its reads were never
        wrapped), or sets them outside the read's own transaction."""
        conn = _DirectConnection.__new__(_DirectConnection)
        conn.calls = []
        calls = conn.calls
        rows = await read_keyed(SqlL3Backend(conn), _TABLE, keys=_keys(), columns=["state", "id", "v"])
        assert rows == [{"state": "CA", "id": 1, "v": 3}]
        assert calls == ["BEGIN", _SET_BATCH, "fetch", "COMMIT"]


_DDL = "(state VARCHAR, id INTEGER, v INTEGER, epoch BIGINT, gone BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (state, id, epoch))"


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """the versioned table: ``CA`` 1..6 (three versions of 1, a gone marker on 2), ``NV`` 1..2."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {_TABLE} {_DDL}")
    conn.execute(
        "CREATE TABLE threetears_run_markers (table_name VARCHAR, scope VARCHAR, prefix VARCHAR, epoch BIGINT)"
    )
    conn.execute(
        f"INSERT INTO {_TABLE} (state, id, v, epoch, gone) VALUES "
        "('CA', 1, 1, 10, false), ('CA', 1, 2, 20, false), ('CA', 1, 3, 30, false), "
        "('CA', 2, 10, 10, false), ('CA', 2, NULL, 20, true), "
        "('CA', 3, 30, 10, false), ('CA', 4, 40, 10, false), ('CA', 5, 50, 10, false), ('CA', 6, 60, 10, false), "
        "('NV', 1, 70, 10, false), ('NV', 2, 80, 10, false)"
    )
    yield conn
    conn.close()


class _L3:
    """an L3 reader answering from DuckDB (it names no dialect, so the helpers take it for L3), keeping every
    statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._inner.fetch(query, *params)


_NEWEST_CA = {1: 3, 3: 30, 4: 40, 5: 50, 6: 60}

#: a versioned page's LIMIT and its cursor inside the DISTINCT ON: ``... WHERE <cursor> ORDER BY ..., epoch DESC
#: LIMIT n) AS n``
_INSIDE = re.compile(r"SELECT DISTINCT ON \(.*WHERE .*t\.\"id\"\)? > \(?\$\d+.* DESC LIMIT \d+\) AS n")


class TestKeyOrderedListings:
    async def test_a_page_past_its_cursor_reads_its_range_inside_the_relation(
        self, db: duckdb.DuckDBPyConnection, engine: dict[str, bool]
    ) -> None:
        """Flips if a key-ordered page of a versioned table leaves its LIMIT or cursor outside the newest relation
        (the planner then bounds it, or does not: a big scope sorted a page), or answers an older or gone version."""
        engine["yugabyte"] = False
        reader = _L3(db)
        order = [Sort("id", cast="integer")]
        read: dict[int, int] = {}
        cursor = None
        while True:
            page = await keyset_page(
                reader, _TABLE, columns=["state", "id", "v"], by={"state": "CA"}, order=order, limit=2, cursor=cursor
            )
            assert len(page.rows) <= 2
            read.update({int(row["id"]): int(row["v"]) for row in page.rows})
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert read == _NEWEST_CA
        past = [statement for statement in reader.statements if '"id" >' in statement]
        assert past and all(_INSIDE.search(statement) for statement in past)
        # nothing bounds the page outside the relation: its outer statement ends at its order
        assert all(re.search(r"\) AS t ORDER BY t\.\"id\"$", statement) for statement in reader.statements)

    async def test_keyset_rows_reads_a_range_a_statement(
        self, db: duckdb.DuckDBPyConnection, engine: dict[str, bool]
    ) -> None:
        """Flips if a listing read to its end reads its scope in a statement whose LIMIT is outside the relation."""
        engine["yugabyte"] = False
        reader = _L3(db)
        rows = await keyset_rows(
            reader, _TABLE, columns=["state", "id", "v"], by={"state": "CA"}, order=[Sort("id", cast="integer")], page=1
        )
        assert {int(row["id"]): int(row["v"]) for row in rows} == _NEWEST_CA
        assert len(reader.statements) >= 3
        assert all("DESC LIMIT 2) AS n" in statement for statement in reader.statements)


class TestPrefixBatches:
    async def test_a_yugabyte_prefix_batch_reads_one_prefix_a_range_at_a_time(
        self, db: duckdb.DuckDBPyConnection, engine: dict[str, bool]
    ) -> None:
        """Flips if a batch of key prefixes on YugabyteDB names several prefixes' whole scopes in one sorted
        statement (its first named the batch's first 2,000), rather than one prefix pinned, its keys a range a
        statement under the read's LIMIT; or answers other than each key's newest live version."""
        engine["yugabyte"] = True
        reader = _L3(db)
        keys = KeyBatch(("state",), [["CA", "NV"]], ("text",))
        rows = await read_keyed(reader, _TABLE, keys=keys, columns=["state", "id", "v"], batch=2, row_cap=3)
        assert {(row["state"], int(row["id"])): int(row["v"]) for row in rows} == {
            **{("CA", key): value for key, value in _NEWEST_CA.items()},
            ("NV", 1): 70,
            ("NV", 2): 80,
        }
        assert all(set(row) == {"state", "id", "v"} for row in rows)
        first = reader.statements[0]
        assert 'WHERE t."state" = $1' in first and "ANY" not in first and re.search(r"DESC LIMIT \d+\) AS n", first)
        assert all(re.search(r"DESC LIMIT [12]\) AS n", statement) for statement in reader.statements)

    async def test_an_arrow_prefix_batch_reads_one_prefix_a_range_at_a_time(
        self, db: duckdb.DuckDBPyConnection, engine: dict[str, bool]
    ) -> None:
        """Flips as the dict read does, for the Arrow read: no row a Python object, the same rows."""
        engine["yugabyte"] = True
        reader = _L3(db)
        keys = KeyBatch(("state",), [["CA"]], ("text",))
        types = {"state": pa.string(), "id": pa.int32(), "v": pa.int32()}
        tables = await read_keyed_arrow(reader, _TABLE, keys=keys, types=types, row_cap=3)
        read = pa.concat_tables(tables)
        assert read.column_names == ["state", "id", "v"]
        assert dict(zip(read.column("id").to_pylist(), read.column("v").to_pylist(), strict=True)) == _NEWEST_CA
        assert len(reader.statements) >= 3
        assert all(re.search(r"DESC LIMIT [12]\) AS n", statement) for statement in reader.statements)
