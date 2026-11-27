"""a walk in one statement checks its tables (versioned: refused; stamped: guarded in place) once a shape for each
generation of the registrations, and a registration made after a shape was checked is always seen.

Why (2026-11-26): checked every call, the tables cost the visible-customer set 6.7 us of its ~24 us of Python
(``threetears.core.keyed._filters.one_statement_guards``); their answer changes only when a registration does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import Joined, ThreadedDuckDB, read_reached
from threetears.interfaces.keyed.keys import (
    StampedTable,
    VersionedTable,
    register_stamped_table,
    register_versioned_table,
    registered_generation,
)


class _L3:
    """an L3 reader answering from DuckDB (it names no dialect, so every helper takes it for L3), keeping each
    statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._inner.fetch(query, *params)


@pytest.fixture
def tables(request: pytest.FixtureRequest) -> Iterator[tuple[duckdb.DuckDBPyConnection, str, str]]:
    """a link table and a joined one, each named for this test alone (registrations are process-wide)."""
    own = request.node.originalname.removeprefix("test_")
    links, grants = f"links_{own}", f"grants_{own}"
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {links} (member_id VARCHAR, group_id VARCHAR, gone BOOLEAN NOT NULL DEFAULT false)")
    conn.executemany(
        f"INSERT INTO {links} VALUES (?, ?, ?)", [("u1", "g1", False), ("g1", "g2", False), ("g1", "g3", True)]
    )
    conn.execute(f"CREATE TABLE {grants} (group_id VARCHAR, grant_id VARCHAR, epoch BIGINT, gone BOOLEAN)")
    conn.executemany(f"INSERT INTO {grants} VALUES (?, ?, ?, ?)", [("g1", "x1", 1, False), ("g2", "x2", 1, False)])
    yield conn, links, grants
    conn.close()


async def _walk(reader: Any, links: str, grants: str) -> list[tuple[str, str | None]]:
    rows = await read_reached(
        reader,
        links,
        start={"member_id": "u1"},
        on="member_id",
        to="group_id",
        columns=("group_id",),
        depth=2,
        joins=[Joined(grants, on={"group_id": "group_id"}, columns={"grant_id": "grant_id"}, left=True)],
    )
    return sorted((row["group_id"], row["grant_id"]) for row in rows)


async def test_a_stamped_registration_after_a_shape_was_checked_guards_the_next_call(
    tables: tuple[duckdb.DuckDBPyConnection, str, str],
) -> None:
    """Flips if a shape checked before its table was registered stamped keeps its unguarded answer: the next walk
    would read the row a delete marked gone (g1 -> g3 here)."""
    conn, links, grants = tables
    reader = _L3(conn)
    assert await _walk(reader, links, grants) == [("g1", "x1"), ("g2", "x2"), ("g3", None)]
    assert await _walk(reader, links, grants) == [("g1", "x1"), ("g2", "x2"), ("g3", None)]
    assert all('"gone"' not in sql for sql in reader.statements)
    before = registered_generation()
    register_stamped_table(StampedTable(links, "gone", ("member_id", "group_id")))
    assert registered_generation() > before
    assert await _walk(reader, links, grants) == [("g1", "x1"), ("g2", "x2")]
    assert 'NOT t."gone"' in reader.statements[-1], reader.statements[-1]


async def test_a_versioned_registration_after_a_shape_was_checked_refuses_the_next_call(
    tables: tuple[duckdb.DuckDBPyConnection, str, str],
) -> None:
    """Flips if a shape checked before a table it joins was registered versioned keeps running (it would read every
    version of the table), or a refusal is answered only once: each call is refused, and nothing is sent."""
    conn, links, grants = tables
    reader = _L3(conn)
    assert await _walk(reader, links, grants) == [("g1", "x1"), ("g2", "x2"), ("g3", None)]
    register_versioned_table(VersionedTable(grants, ("group_id", "grant_id"), "epoch", "gone"))
    sent = len(reader.statements)
    for _ in range(2):
        with pytest.raises(ValueError, match="versioned"):
            await _walk(reader, links, grants)
    assert len(reader.statements) == sent


async def test_a_local_copy_and_l3_are_checked_apart(tables: tuple[duckdb.DuckDBPyConnection, str, str]) -> None:
    """Flips if a shape's answer on L3 is taken for a local copy's walk of the same tables (or the other way): a
    local copy holds live rows alone and no gone flag, so its walk is never guarded nor refused."""
    conn, links, grants = tables
    register_stamped_table(StampedTable(links, "gone", ("member_id", "group_id")))
    reader = _L3(conn)
    assert await _walk(reader, links, grants) == [("g1", "x1"), ("g2", "x2")]
    assert 'NOT t."gone"' in reader.statements[-1]
    # the local copy here still holds the gone row, so a walk unguarded reaches it: proof no guard was written
    assert await _walk(ThreadedDuckDB(conn), links, grants) == [("g1", "x1"), ("g2", "x2"), ("g3", None)]
