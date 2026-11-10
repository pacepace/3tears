"""the keyed batch join (:func:`threetears.core.sql_fragments.unnest_key_join`) reads exactly the rows its keys name.

Checked on DuckDB, the copy's own engine (PostgreSQL and YugabyteDB run the same text in the copy follower's live
tests): a key batch, a leading prefix of the key, a NULL bucket matched null-safe, and its refusal of no columns.
"""

from __future__ import annotations

import pytest

from threetears.core.sql_fragments import unnest_key_join

duckdb = pytest.importorskip("duckdb")


def _db() -> object:
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE votes (state VARCHAR, geo_id BIGINT, votes BIGINT)")
    db.execute("INSERT INTO votes VALUES ('VA', 1, 10), ('VA', 2, 20), ('OH', 1, 30), (NULL, 9, 90)")
    return db


def _read(db: object, columns: list[str], casts: list[str], values: list[list[str | None]], **kw: bool) -> list[tuple]:
    unnested, joined = unnest_key_join(columns, casts, **kw)
    sql = f"SELECT t.* FROM votes AS t JOIN (SELECT {unnested}) AS k ON {joined}"  # noqa: S608
    return sorted(db.execute(sql, values).fetchall(), key=str)  # type: ignore[attr-defined]


def test_a_key_batch_reads_exactly_its_rows() -> None:
    rows = _read(_db(), ["state", "geo_id"], ["text", "bigint"], [["VA", "OH"], ["2", "1"]])
    assert rows == [("OH", 1, 30), ("VA", 2, 20)]


def test_a_leading_prefix_reads_every_row_under_it() -> None:
    assert _read(_db(), ["state"], ["text"], [["VA"]]) == [("VA", 1, 10), ("VA", 2, 20)]


def test_a_null_reads_the_null_bucket_only_when_null_safe() -> None:
    assert _read(_db(), ["state"], ["text"], [[None]]) == []
    assert _read(_db(), ["state"], ["text"], [[None]], null_safe=True) == [(None, 9, 90)]


def test_no_column_or_a_missing_cast_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one key column"):
        unnest_key_join([], [])
    with pytest.raises(ValueError, match="at least one key column"):
        unnest_key_join(["state", "geo_id"], ["text"])
