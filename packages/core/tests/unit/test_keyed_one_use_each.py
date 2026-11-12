"""a keyed read through a reader naming its engine's dialect is sent one placeholder a use (``keyed._filters.sent_by``).

Redshift, through a datasource reader, binds each placeholder once: it refuses a statement naming ``$1`` twice,
where PostgreSQL (and the stand-ins the units run on) take it. Reports #37 met it on ``counts_by_relations(...,
fresh=)``, whose fresh flag named the mark its ``WHERE`` binds; a keyset page past a cursor names each sort value
two or three times. Every statement such a reader is sent is checked here, from every builder that reads a warehouse.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import Dialect, Sort, ThreadedDuckDB, counts_by_relations, keyset_rows
from threetears.interfaces.keyed.filters import one_use_each


class _Redshift:
    """a DuckDB standing in for a Redshift datasource reader: it names its dialect, and records what it is sent."""

    sql_dialect = Dialect.REDSHIFT

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(db)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((query, params))
        return [dict(row) for row in await self._inner.fetch(query, *params)]


def _each_once(statements: list[tuple[str, tuple[Any, ...]]]) -> None:
    for sql, params in statements:
        numbers = [int(found) for found in re.findall(r"\$(\d+)", sql)]
        assert sorted(numbers) == list(range(1, len(numbers) + 1)), sql
        assert len(params) == len(numbers), (sql, params)


@pytest.fixture
def db() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    for name, floor in (("ra", 0), ("rb", 30)):
        conn.execute(
            f"CREATE TABLE {name} AS SELECT 'S' || (i % 3) AS state, 'g' || (i % 4) AS geo, i AS id, "
            "CAST(TIMESTAMP '2026-11-03 00:00:00' + to_minutes(i) AS VARCHAR) AS up, "
            "CASE WHEN i % 5 = 0 THEN TIMESTAMP '2026-11-04 00:00:00' + to_minutes(i) END AS called "
            f"FROM range({floor}, 60) r(i)"
        )
    return conn


_MARKS: dict[str, dict[str, str | None]] = {
    "ra": {"up": "2026-11-03 00:40:00", "called": None},
    "rb": {"up": "2026-11-03 00:50:00", "called": "2026-11-04 00:45:00"},
}


def _groups(answer: Any) -> set[tuple[Any, ...]]:
    return {(label, found.value, found.rows, found.fresh) for label, groups in answer.items() for found in groups}


async def test_a_count_past_marks_with_its_fresh_flag_names_each_placeholder_once(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if ``counts_by_relations(..., fresh=)`` sends a Redshift reader a statement naming a placeholder twice
    (the fresh flag reusing its mark's ``$n``), or answers it other groups than a reader naming no dialect."""
    for packed_rows in (None, 3):
        options: dict[str, Any] = {
            "column": "state",
            "since": _MARKS,
            "fresh": "up",
            "packed_rows": packed_rows,
            "groups": 40,
        }
        redshift = _Redshift(db)
        sent = await counts_by_relations(redshift, {"ra": "ra", "rb": "rb"}, **options)
        plain = await counts_by_relations(ThreadedDuckDB(db), {"ra": "ra", "rb": "rb"}, **options)
        assert _groups(sent) == _groups(plain) and any(found[-1] for found in _groups(plain)), packed_rows
        _each_once(redshift.statements)


async def test_a_keyset_page_past_a_cursor_names_each_placeholder_once(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a keyset read past a cursor (the expansion a warehouse is read by, each sort value named in its bound
    and its ties) sends a Redshift reader a statement naming a placeholder twice, or reads other rows."""
    order = [Sort("geo"), Sort("id", cast="bigint")]
    redshift = _Redshift(db)
    sent = await keyset_rows(redshift, "ra", columns=["geo", "id"], by={"state": "S1"}, order=order, page=3)
    plain = await keyset_rows(ThreadedDuckDB(db), "ra", columns=["geo", "id"], by={"state": "S1"}, order=order, page=3)
    assert sent == plain and len(redshift.statements) > 2
    _each_once(redshift.statements)


def test_one_use_each_renumbers_every_use_and_leaves_quoted_text() -> None:
    """Flips if a placeholder used twice keeps one number, a value lands at the wrong place, a cast or a quoted
    ``$1`` is touched, or a placeholder past the parameters is sent."""
    sql = "SELECT '$1' AS \"$2\" FROM t WHERE a = $2::text AND (b > $1 OR (b = $1 AND c > $2::text::int))"
    assert one_use_each(sql, ["one", "two"]) == (
        "SELECT '$1' AS \"$2\" FROM t WHERE a = $1::text AND (b > $2 OR (b = $3 AND c > $4::text::int))",
        ["two", "one", "one", "two"],
    )
    with pytest.raises(ValueError, match=r"\$3"):
        one_use_each("SELECT 1 WHERE a = $3", [1])
