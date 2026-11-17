"""a pin the code fixes (:class:`Const`) is written into a keyed statement as its literal, every other value bound.

Why literals (Pace's ruling 2026-11-16): bound, the visible-customer walk's ``member_type`` discriminators made
PostgreSQL's generic plan look costlier, so it was never cached and every call was re-planned. Only a plain
identifier-like word from the code is a ``Const``; the enforcement test holds every ``Const(...)`` to a constant.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import Const, InlineDuckDB, by_conditions, read_reached


def test_a_const_pin_is_its_literal_and_binds_nothing() -> None:
    """Flips if a ``Const`` is bound as ``$n`` (the generic plan then never caches) or the values after it are
    numbered as though it took a placeholder."""
    conditions, params = by_conditions({"member_type": Const("user"), "member_id": 7}, alias="t")
    assert conditions == ["t.\"member_type\" = 'user'", 't."member_id" = $1']
    assert params == [7]


def test_a_plain_text_pin_is_still_bound() -> None:
    """Flips if text that is not a ``Const`` (a caller's, a request's) is written into the statement."""
    conditions, params = by_conditions({"member_type": "user"}, alias="t")
    assert conditions == ['t."member_type" = $1']
    assert params == ["user"]


@pytest.mark.parametrize("text", ["", "two words", "x'; DROP TABLE t; --", "1st", "a-b", "é", "a" * 64, "user\n"])
def test_a_const_refuses_anything_but_a_plain_word(text: str) -> None:
    """Flips if a ``Const`` takes text a statement could read as more than one plain literal."""
    with pytest.raises(ValueError, match="plain identifier-like word"):
        Const(text)


def test_a_const_refuses_a_non_text_value() -> None:
    """Flips if a ``Const`` takes a value that is not text (it would be rendered by its ``str``)."""
    with pytest.raises(ValueError, match="plain identifier-like word"):
        Const(5)  # type: ignore[arg-type]


def test_a_const_is_the_text_it_holds() -> None:
    """Flips if a ``Const`` stops comparing, hashing and printing as its text (readers compare rows to pins)."""
    pinned = Const("group")
    assert pinned == "group"
    assert hash(pinned) == hash("group")
    assert {pinned: 1}["group"] == 1
    assert pinned.sql == "'group'"


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE members (member_type VARCHAR, member_id VARCHAR, group_id VARCHAR)")
    conn.executemany(
        "INSERT INTO members VALUES (?, ?, ?)",
        [
            ("user", "u1", "g1"),
            ("group", "g1", "g2"),
            ("group", "g2", "g3"),
            ("user", "g1", "gx"),
            ("user", "u2", "g9"),
        ],
    )
    yield conn
    conn.close()


async def test_a_capped_walk_writes_its_const_pins_as_literals(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a walk binds a ``Const`` in ``start``, ``by`` or ``follow`` as ``$n``, or answers other rows than
    the same walk with the text bound."""
    sent: list[tuple[str, tuple[Any, ...]]] = []
    inline = InlineDuckDB(db)
    fetch = inline.fetch

    async def recorded(sql: str, *args: Any) -> Any:
        sent.append((sql, args))
        return await fetch(sql, *args)

    inline.fetch = recorded  # type: ignore[method-assign]
    walks = []
    for user, group in ((Const("user"), Const("group")), ("user", "group")):
        rows = await read_reached(
            inline,
            "members",
            start={"member_type": user, "member_id": "u1"},
            on="member_id",
            to="group_id",
            by={"member_type": group},
            follow={"member_type": group},
            columns=("group_id",),
            depth=4,
        )
        walks.append(sorted(row["group_id"] for row in rows))
    assert walks[0] == walks[1] == ["g1"]
    literal, bound = sent
    assert "t.\"member_type\" = 'user'" in literal[0] and "t.\"member_type\" = 'group'" in literal[0], literal[0]
    assert "r.\"member_type\" = 'group'" in literal[0], literal[0]
    assert literal[1] == ("u1", 4)
    assert bound[1] == ("user", "u1", "group", "group", 4)
