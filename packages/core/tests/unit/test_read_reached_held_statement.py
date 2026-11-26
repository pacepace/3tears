"""a capped walk's statement is written once a call shape (:func:`read_reached` with ``depth``), each call binding its
own pins.

Why (2026-11-21): writing the visible-customer walk's statement cost 16.4 us of a 149 us call on the exclusive runner
and took the set over its bar of twice the statement it replaced; the text is the same every call of one shape.
"""

from __future__ import annotations

import dataclasses
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


async def _walk_joined(conn: InlineDuckDB, join: Joined) -> list[dict[str, Any]]:
    return await read_reached(
        conn,
        "members",
        start={"member_type": Const("user"), "member_id": "u1"},
        on="member_id",
        to="group_id",
        by={"member_type": Const("group")},
        columns=("group_id",),
        depth=2,
        joins=[join],
    )


async def test_two_joins_differing_in_one_field_are_two_statements(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a field of a join is left out of the shape a held statement is keyed by: two walks would share one
    text and the second read the first's join (an inner join answered as a left one, a literal pin as another)."""
    conn, sent = _recorded(db)
    on, columns = {"group_id": "group_id"}, {"grant_id": "grant_id"}
    left = await _walk_joined(conn, Joined("grants", on=on, columns=columns, left=True))
    inner = await _walk_joined(conn, Joined("grants", on=on, columns=columns, left=False))
    only_b = await _walk_joined(conn, Joined("grants", on=on, columns=columns, by={"scope": Const("b")}, left=True))
    texts = [sql for sql, _ in sent]
    assert len(set(texts)) == 3, texts
    assert sorted(row["grant_id"] for row in left) == ["x1", "x2"]
    assert sorted(row["grant_id"] for row in inner) == ["x1", "x2"]
    assert sorted((row["group_id"], row["grant_id"]) for row in only_b) == [("g1", None), ("g2", "x2")]


def test_every_field_of_a_join_is_in_its_shape() -> None:
    """Flips if a field added to Joined is not in its shape, whatever its kind."""
    base = Joined("grants", on={"group_id": "group_id"}, columns={"grant_id": "grant_id"})
    changed: dict[str, Any] = {
        "table": "other",
        "on": {"group_id": "other_id"},
        "columns": {"grant_id": "scope"},
        "left": True,
        "by": {"scope": "a"},
        "matched": "live",
        "casts": {"group_id": "text"},
        "counts": {"n": None},
    }
    fields = {each.name for each in dataclasses.fields(Joined)} - {"shape"}
    assert fields == set(changed), "a field of Joined has no case here"
    for name, value in changed.items():
        assert dataclasses.replace(base, **{name: value}).shape != base.shape, name


def test_a_joins_mappings_are_its_own_and_read_only() -> None:
    """Flips if a caller's dict changed after the join is built reaches the join (its shape no longer its SQL), or a
    join's mapping can be changed in place."""
    on = {"group_id": "group_id"}
    join = Joined("grants", on=on, columns={"grant_id": "grant_id"})
    shape = join.shape
    on["group_id"] = "other_id"
    assert dict(join.on) == {"group_id": "group_id"}
    assert join.shape == shape
    with pytest.raises(TypeError):
        join.on["group_id"] = "other_id"  # type: ignore[index]
