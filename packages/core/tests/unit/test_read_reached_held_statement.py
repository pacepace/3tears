"""a capped walk's statement is written once a call shape (:func:`read_reached` with ``depth``), each call binding its
own pins.

Why (2026-11-21): writing the visible-customer walk's statement cost 16.4 us of a 149 us call on the exclusive runner
and took the set over its bar of twice the statement it replaced; the text is the same every call of one shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import AnyOf, Const, InlineDuckDB, Joined, read_reached


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE members (member_type VARCHAR, member_id VARCHAR, group_id VARCHAR)")
    conn.executemany(
        "INSERT INTO members VALUES (?, ?, ?)",
        [("user", "u1", "g1"), ("group", "g1", "g2"), ("user", "u2", "g9"), ("user", None, "g0")],
    )
    conn.execute("CREATE TABLE grants (group_id VARCHAR, scope VARCHAR, grant_id VARCHAR)")
    conn.executemany(
        "INSERT INTO grants VALUES (?, ?, ?)",
        [("g1", "a", "x1"), ("g2", "b", "x2"), ("g9", "a", "x9"), ("g0", "a", "x0")],
    )
    yield conn
    conn.close()


def _recorded(db: duckdb.DuckDBPyConnection) -> tuple[InlineDuckDB, list[tuple[str, tuple[Any, ...]]]]:
    sent: list[tuple[str, tuple[Any, ...]]] = []
    inline = InlineDuckDB(db)
    fetch = inline.fetch

    async def recorded(sql: str, *args: Any) -> Any:
        sent.append((sql, args))
        return await fetch(sql, *args)

    inline.fetch = recorded  # type: ignore[method-assign]
    return inline, sent


async def _walk(conn: InlineDuckDB, member: Any, scopes: tuple[str, ...] = ("a", "b")) -> list[tuple[str, str]]:
    rows = await read_reached(
        conn,
        "members",
        start={"member_type": Const("user"), "member_id": member},
        on="member_id",
        to="group_id",
        by={"member_type": Const("group")},
        columns=("group_id",),
        depth=2,
        joins=[
            Joined(
                "grants",
                on={"group_id": "group_id"},
                by={"scope": AnyOf(scopes)},
                columns={"grant_id": "grant_id"},
                left=True,
            )
        ],
    )
    return sorted((row["group_id"], row["grant_id"]) for row in rows)


async def test_each_call_of_a_shape_binds_its_own_pins_into_one_text(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a held statement answers a later caller's walk with an earlier caller's pins, or a call of the same
    shape writes another text."""
    conn, sent = _recorded(db)
    assert await _walk(conn, "u1") == [("g1", "x1"), ("g2", "x2")]
    assert await _walk(conn, "u2") == [("g9", "x9")]
    (first_sql, first_args), (second_sql, second_args) = sent
    assert first_sql == second_sql
    assert first_args[0] == "u1" and second_args[0] == "u2"
    assert first_args[1:] == second_args[1:]


async def test_a_pin_written_otherwise_is_another_shape(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a first-row pin written as ``IS NULL`` takes the text of one bound as ``$n``, or a join's pin values
    are held from an earlier call (they are bound after the first rows' pins)."""
    conn, sent = _recorded(db)
    assert await _walk(conn, "u1") == [("g1", "x1"), ("g2", "x2")]
    assert await _walk(conn, None) == [("g0", "x0")]
    assert await _walk(conn, "u1", scopes=("b",)) == [("g1", None), ("g2", "x2")]
    bound, null, narrower = sent
    assert "IS NULL" in null[0] and "IS NULL" not in bound[0]
    assert narrower[1][0] == "u1"
    assert list(narrower[1][-1]) == ["b"]
