"""read_state_arrow: one state's rows of a LOCAL copy table, pinned by its declared state column; never L3.

Decision 36 of docs/design-pod-delta-files.md: a table whose key does not name its state (AP results) has its state's rows
read by that column only on the pod's own copy. The helper refuses any connection that reads L3.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from threetears.core.keyed import InlineDuckDB, ThreadedDuckDB, read_state_arrow


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE results (office_key VARCHAR, join_state_code VARCHAR, votes BIGINT)")
    conn.execute("INSERT INTO results VALUES ('hs_VA_1', 'VA', 1), ('hs_VA_2', 'VA', 2), ('hs_IL_1', 'IL', 3)")
    yield conn
    conn.close()


class _L3Pool:
    """a connection that reads L3: no dialect named, no DuckDB or SQLite reader."""


async def test_it_reads_exactly_one_states_rows_of_a_local_copy(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_state_arrow returns another state's rows, or misses one of the state's."""
    rows = await read_state_arrow(
        InlineDuckDB(db), "results", scope_column="join_state_code", state="VA", columns=["office_key", "votes"]
    )
    assert sorted(rows.to_pylist(), key=lambda row: row["office_key"]) == [
        {"office_key": "hs_VA_1", "votes": 1},
        {"office_key": "hs_VA_2", "votes": 2},
    ]


async def test_it_reads_through_a_threaded_copy_too(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a ThreadedDuckDB copy is refused: the copy's reads run off the loop through it."""
    rows = await read_state_arrow(ThreadedDuckDB(db), "results", scope_column="join_state_code", state="IL")
    assert rows.num_rows == 1


async def test_it_refuses_a_connection_that_reads_l3() -> None:
    """Flips if read_state_arrow runs on an L3 connection: a state's rows there are read by key, never by a column
    the key does not lead (that would be the scan election night died of)."""
    with pytest.raises(ValueError, match="never L3"):
        await read_state_arrow(_L3Pool(), "results", scope_column="join_state_code", state="VA")


async def test_it_refuses_anything_but_a_duckdb_copy() -> None:
    """Flips if a reader naming its own dialect (not L3, but no local copy either) is read: the helper is the copy's."""

    class _Warehouse:
        sql_dialect = "redshift"

    with pytest.raises(TypeError, match="DuckDB copy"):
        await read_state_arrow(_Warehouse(), "results", scope_column="join_state_code", state="VA")
