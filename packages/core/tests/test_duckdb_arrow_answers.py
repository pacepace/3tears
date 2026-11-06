"""DuckDB answers as Arrow: ``execute_arrow`` writes a query's rows without the interpreter lock, and ``arrow_rows``
makes the rows ``execute_query`` answers of them a piece at a time.

``execute_query`` makes a Python object of every value in one ``fetchall``, holding the interpreter lock (the event
loop's too) for as long as the answer is large; ``execute_arrow`` answers the same rows as Arrow, which DuckDB writes
with the lock let go. A caller that needs dict rows takes them through ``arrow_rows``: each value is the one
``fetchall`` answers (a ``SUM`` an ``int``, a UUID a ``uuid.UUID``, an interval a ``timedelta``, a map a ``dict``), and
no call makes more than a few hundred rows. A text ``NUMERIC`` (the rail's form) is typed in Arrow where that is exact,
storing what its dict row stores.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import Column, MetaData, Numeric, String, Table

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.base import stored_decimal  # noqa: E402
from threetears.core.cache.duckdb import DuckDBBackend, arrow_rows, query_arrow  # noqa: E402


@pytest.fixture()
def backend() -> Any:
    metadata = MetaData()
    Table("shares", metadata, Column("id", String, primary_key=True), Column("share", Numeric(12, 4)))
    made = DuckDBBackend()
    made.initialize(metadata)
    connection = made.get_connection()
    connection.execute("CREATE TYPE mood AS ENUM ('calm', 'busy')")
    connection.execute(
        "CREATE TABLE every_type (i BIGINT, d DECIMAL(18,3), ts TIMESTAMPTZ, tn TIMESTAMP, dt DATE, u UUID, s VARCHAR, "
        "b BLOB, iv INTERVAL, m MAP(VARCHAR, INTEGER), l INTEGER[], st STRUCT(a INTEGER), e mood, h HUGEINT, f DOUBLE)"
    )
    connection.execute(
        "INSERT INTO every_type VALUES (1, 1.5, '2026-11-03 20:00:00+00', '2026-11-03 20:00:00', '2026-11-03', "
        "'3b62f72b-81ea-4142-aeb2-33ddcae51b69', 'x', 'ab'::BLOB, INTERVAL '1 month 2 days 3 seconds', MAP {'k': 1}, "
        "[1, 2], {'a': 1}, 'calm', 5, 0.1), (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
        "NULL, NULL, NULL)"
    )
    yield made
    made.reset()


def _alike(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    assert len(left) == len(right)
    for one, other in zip(left, right, strict=True):
        assert list(one) == list(other)
        for name in one:
            assert (one[name], type(one[name])) == (other[name], type(other[name])), name


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM every_type",
        # a SUM of integers is DuckDB's HUGEINT, which its Arrow writes as decimal128(38, 0)
        "SELECT s, SUM(i) AS total, COUNT(*) AS n, AVG(i) AS mean, LIST(h) AS hs FROM every_type GROUP BY s ORDER BY s",
        "SELECT 170141183460469231731687303715884105727::HUGEINT AS widest",
    ],
)
def test_arrow_rows_are_the_rows_execute_query_answers(backend: Any, sql: str) -> None:
    _alike(arrow_rows(backend.execute_arrow(sql)), backend.execute_query(sql))


def test_an_answer_names_each_columns_duckdb_type_and_a_sum_is_an_integer(backend: Any) -> None:
    table = backend.execute_arrow("SELECT SUM(i) AS total, s FROM every_type GROUP BY s")
    assert table.schema.field("total").type == pa.int64()
    assert table.schema.field("total").metadata[b"duckdb_type"] == b"HUGEINT"
    assert table.schema.field("s").metadata[b"duckdb_type"] == b"VARCHAR"


def test_a_nested_value_arrow_holds_otherwise_than_fetchall_is_refused(backend: Any) -> None:
    with pytest.raises(ValueError, match="execute_query"):
        arrow_rows(backend.execute_arrow("SELECT LIST(u) AS us FROM every_type"))


class _Pieces:
    """an answer that records the pieces it is asked for."""

    def __init__(self, table: Any) -> None:
        self.table = table
        self.schema = table.schema
        self.asked: list[int] = []

    def to_batches(self, max_chunksize: int) -> Any:
        self.asked.append(max_chunksize)
        return self.table.to_batches(max_chunksize=max_chunksize)


def test_arrow_rows_are_made_a_piece_at_a_time(backend: Any) -> None:
    answer = _Pieces(backend.execute_arrow("SELECT range AS i FROM range(1000)"))
    rows = arrow_rows(answer)
    assert [row["i"] for row in rows] == list(range(1000))
    # a few hundred rows a call, never the whole answer in one
    assert len(answer.asked) == 1 and answer.asked[0] < 1000


def test_execute_arrow_lets_go_of_the_interpreter_lock() -> None:
    """a large answer written as Arrow holds the lock briefly; made Python rows, for as long as it is large."""
    made = DuckDBBackend()
    made.initialize(MetaData())
    sql = "SELECT range AS i, range * 1.5 AS f, 'name' || range AS s FROM range(300000)"
    started = time.thread_time()
    made.execute_arrow(sql)
    arrow_cpu = time.thread_time() - started
    started = time.thread_time()
    made.execute_query(sql)
    rows_cpu = time.thread_time() - started
    made.reset()
    # this thread's own CPU: DuckDB's work is on its threads, and the Arrow is written there
    assert arrow_cpu < rows_cpu / 10, (arrow_cpu, rows_cpu)


def test_a_text_numeric_is_stored_as_its_dict_row_is(backend: Any) -> None:
    """the rail answers a NUMERIC as its digits; typed in Arrow, it stores what the same row as a dict stores."""
    rows = [
        {"id": "a", "share": "0.5000"},
        {"id": "b", "share": "1.5"},
        {"id": "c", "share": "2.12345"},
        {"id": "d", "share": None},
    ]
    _, stored = backend.storage_arrow("shares", pa.Table.from_pylist(rows), "id")
    as_dicts = [None if row["share"] is None else stored_decimal(row["share"], 4) for row in rows]
    assert stored.column("share").to_pylist() == as_dicts
    assert stored.column("share").to_pylist() == [Decimal("0.5000"), Decimal("1.5000"), Decimal("2.1235"), None]


def test_text_numerics_the_cast_takes_are_typed_in_arrow(backend: Any) -> None:
    rows = pa.Table.from_pylist([{"id": "a", "share": "0.5000"}, {"id": "b", "share": "1.5"}])
    _, stored = backend.storage_arrow("shares", rows, "id")
    assert stored.schema.field("share").type == pa.decimal128(38, 4)


def test_a_read_snapshots_cursor_answers_arrow_alike(backend: Any) -> None:
    sql = "SELECT * FROM every_type"
    with backend.read_snapshot() as cursor:
        answer = query_arrow(cursor, sql)
    assert answer.equals(backend.execute_arrow(sql))
