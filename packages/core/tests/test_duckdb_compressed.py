"""A compressed DuckDB L1: the tables held compressed, checkpointed off the write's path, read as any L1's.

The contracts: every way into the copy (a write, a read cursor, a thread's connection) reaches the compressed
tables; a write asks for one checkpoint on the backend's own thread and returns without waiting for it; the
checkpoint makes the tables smaller than the same rows uncompressed; a reset stops the thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")

from threetears.core.cache.duckdb import DuckDBBackend, PartitionReplacement  # noqa: E402

ROWS = 400_000


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "results",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("state", String(8)),
        Column("name", String(64)),
        Column("votes", Integer),
    )
    return metadata


def _rows(state: str, start: int, rows: int) -> pa.Table:
    """rows of one state: a text column of a few thousand distinct values, as election tables have.

    :return: the rows
    :rtype: pa.Table
    """
    ids = range(start, start + rows)
    return pa.table(
        {
            "id": pa.array(ids, pa.int32()),
            "state": pa.array([state] * rows),
            "name": pa.array([f"candidate-{(i * 7919) % 4093}" for i in ids]),
            "votes": pa.array([(i * 31) % 100_003 for i in ids], pa.int32()),
        }
    )


def _memory(backend: DuckDBBackend) -> int:
    with backend.read_snapshot() as cursor:
        found = cursor.execute("SELECT sum(memory_usage_bytes) FROM duckdb_memory()").fetchone()
    assert found is not None
    return int(found[0])


def _until(condition: Callable[[], bool], what: str, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        time.sleep(0.01)


@pytest.fixture
def compressed() -> Iterator[DuckDBBackend]:
    backend = DuckDBBackend(compressed=True)
    backend.initialize(_metadata())
    yield backend
    backend.reset()


def _fill(backend: DuckDBBackend) -> None:
    backend.replace_partitions(
        [
            PartitionReplacement(table="results", column="state", value=state, arrow=_rows(state, index * ROWS, ROWS))
            for index, state in enumerate(("VA", "GA"))
        ]
    )


class TestCompressedCopy:
    def test_every_way_in_reaches_the_compressed_tables(self, compressed: DuckDBBackend) -> None:
        assert compressed.compressed
        compressed.upsert_many("results", [{"id": 1, "state": "VA", "name": "a", "votes": 5}], "id")
        with compressed.read_snapshot() as cursor:
            assert cursor.execute("SELECT current_database()").fetchone() == ("l1",)
            # attached with COMPRESS: an in-memory catalog attached without it holds real tables uncompressed (the
            # reports' extracts: 513 MiB without, 116 MiB with), however often checkpointed
            options = cursor.execute("SELECT options FROM duckdb_databases() WHERE database_name = 'l1'").fetchone()
            assert options == ({"compress": "true"},)
            assert cursor.execute("SELECT votes FROM results").fetchall() == [(5,)]
        assert compressed.get_connection().execute("SELECT count(*) FROM results").fetchone() == (1,)
        assert compressed.execute_query("SELECT name FROM results") == [{"name": "a"}]

    def test_a_write_checkpoints_off_its_path_and_the_copy_shrinks(self, compressed: DuckDBBackend) -> None:
        plain = DuckDBBackend()
        plain.initialize(_metadata())
        try:
            _fill(plain)
            _fill(compressed)
            _until(lambda: compressed.background_checkpoints >= 1, "the checkpointer ran")
            _until(lambda: _memory(compressed) < _memory(plain) / 2, "the compressed copy is under half the plain one")
            assert compressed.background_checkpoints_failed == 0
            with compressed.read_snapshot() as cursor:
                assert cursor.execute("SELECT count(*), sum(votes) FROM results").fetchone() == (
                    2 * ROWS,
                    sum((i * 31) % 100_003 for i in range(2 * ROWS)),
                )
        finally:
            plain.reset()

    def test_a_reader_keeps_its_state_across_a_write_and_its_checkpoint(self, compressed: DuckDBBackend) -> None:
        _fill(compressed)
        _until(lambda: compressed.background_checkpoints >= 1, "the first checkpoint ran")
        with compressed.read_snapshot() as cursor:
            compressed.replace_partitions(
                [PartitionReplacement(table="results", column="state", value="VA", arrow=_rows("VA", 10 * ROWS, 10))]
            )
            _until(lambda: compressed.background_checkpoints >= 2, "the second checkpoint ran")
            assert cursor.execute("SELECT count(*) FROM results").fetchone() == (2 * ROWS,)
        with compressed.read_snapshot() as cursor:
            assert cursor.execute("SELECT count(*) FROM results").fetchone() == (ROWS + 10,)

    def test_a_plain_backend_asks_for_no_checkpoint(self) -> None:
        plain = DuckDBBackend()
        plain.initialize(_metadata())
        try:
            _fill(plain)
            plain.request_checkpoint()
            assert plain.background_checkpoints == 0
            assert plain._checkpointer is None  # noqa: SLF001 - no thread was started
        finally:
            plain.reset()

    def test_a_reset_stops_the_checkpointer(self, compressed: DuckDBBackend) -> None:
        _fill(compressed)
        _until(lambda: compressed.background_checkpoints >= 1, "the checkpointer ran")
        thread = compressed._checkpointer  # noqa: SLF001 - the thread the reset must stop
        assert thread is not None
        compressed.reset()
        assert not thread.is_alive()
