"""``clause_guards`` (:mod:`threetears.interfaces.keyed.filters`): a hand-written statement's clause builder (``build(
<subject>, ..., args)`` answering `` AND (...)`` over the statement's parameter list) as a keyed helper's ``where=``.

Correctness on an in-process DuckDB: the helper reading with the builder's guards answers the rows the statement read
with the builder's answer written in, the builder binding after the statement's own parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import Fragment, ThreadedDuckDB, clause_guards, read_by


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``t``: keyed ``(a, b)``, three groups of four rows, ``v`` NULL on odd ``b``."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (a VARCHAR, b BIGINT, v INTEGER, PRIMARY KEY (a, b))")
    conn.execute("INSERT INTO t SELECT 'k' || (i // 4), i, CASE WHEN i % 2 = 0 THEN i END FROM range(12) r(i)")
    yield conn
    conn.close()


def _at_least(subject: int | None, column: str, args: list[Any]) -> str:
    """a builder in the shape memory's read filter takes: nothing for no subject, else ``column >= <subject>``."""
    if subject is None:
        return ""
    args.append(subject)
    return f" AND ({column} >= ${len(args)})"


def test_a_builder_answering_nothing_is_no_guard() -> None:
    """Flips if a builder's ``""`` becomes a guard (an always-true fragment the helper still sends)."""
    assert clause_guards(_at_least, None, "t.v") == ()


def test_a_builders_clause_is_one_guard_numbered_from_one() -> None:
    """Flips if the guard keeps the builder's `` AND`` or numbers its placeholder past values it never bound."""
    assert clause_guards(_at_least, 4, "t.v") == (Fragment("(t.v >= $1)", (4,)),)


def test_a_builder_answering_other_than_a_conjunct_is_refused() -> None:
    """Flips if text that is not `` AND (...)`` (an ``ORDER BY``, an ``OR``) is taken as a guard."""

    def ordering(_: Any, args: list[Any]) -> str:
        return " ORDER BY b"

    builder: Callable[..., str] = ordering
    with pytest.raises(ValueError, match="AND"):
        clause_guards(builder, None)


@pytest.mark.parametrize("subject", [None, 4, 9])
async def test_the_helper_with_the_guards_reads_what_the_statement_read(
    db: duckdb.DuckDBPyConnection, subject: int | None
) -> None:
    """Flips if read_by with the builder's guards answers other rows than the statement with the builder's answer
    written after its own parameter."""
    args: list[Any] = ["k1"]
    clause = _at_least(subject, "v", args)
    expected = sorted(tuple(row) for row in db.execute(f"SELECT a, b FROM t WHERE a = $1{clause}", args).fetchall())
    rows = await read_by(
        ThreadedDuckDB(db), "t", by={"a": "k1"}, columns=("a", "b"), where=clause_guards(_at_least, subject, "t.v")
    )
    assert sorted((row["a"], row["b"]) for row in rows) == expected
