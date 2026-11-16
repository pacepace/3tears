"""point reads (:mod:`threetears.core.keyed.point`): ``read_by``, ``read_row`` and ``read_value`` answer what ``fetch``,
``fetchrow`` and ``fetchval`` of the hand-written ``WHERE a = $1 AND b = $2`` answered, and send that statement.

Correctness on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB); the
PostgreSQL / YugabyteDB proof of each rewrite is the repo's generated equivalence test
(:mod:`threetears.core.testing.rewrites`). A recording reader pins the statements.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import (
    AnyOf,
    Fragment,
    Joined,
    Lock,
    Nearest,
    PatternCount,
    Selected,
    Sort,
    TextRank,
    ThreadedDuckDB,
    read_by,
    read_nearest,
    read_row,
    read_value,
)
from threetears.interfaces.keyed.keys import DeclaredIndex, register_partial_predicates, register_table_indexes
from threetears.interfaces.keyed.point import read_by_statement


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``t``: keyed ``(a, b)``, three groups of four rows, ``v`` NULL on odd ``b``."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (a VARCHAR, b BIGINT, v INTEGER, flag BOOLEAN, PRIMARY KEY (a, b))")
    conn.execute(
        "INSERT INTO t SELECT 'k' || (i // 4), i, CASE WHEN i % 2 = 0 THEN i END, i % 3 = 0 FROM range(12) r(i)"
    )
    yield conn
    conn.close()


class _Recording:
    """a reader that answers from DuckDB and keeps every statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((query, params))
        return await self._inner.fetch(query, *params)


def _plain(db: duckdb.DuckDBPyConnection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in db.execute(sql, list(params)).fetchall()]


async def test_read_by_answers_every_row_the_pins_name(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_by keeps a row the statement would not, or drops one it would."""
    rows = await read_by(ThreadedDuckDB(db), "t", by={"a": "k1", "flag": True}, columns=("a", "b"))
    assert sorted((row["a"], row["b"]) for row in rows) == _plain(
        db, "SELECT a, b FROM t WHERE a = $1 AND flag = TRUE ORDER BY b", "k1"
    )


async def test_read_by_answers_its_rows_in_the_order_it_names(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``order=`` stops being the statement's ORDER BY (the rows the pins name, in the key's next column,
    either way), or an unordered read starts sending one."""
    reader = _Recording(db)
    for descending in (False, True):
        rows = await read_by(reader, "t", by={"a": "k2"}, columns=("b",), order=[Sort("b", descending=descending)])
        assert [row["b"] for row in rows] == sorted(range(8, 12), reverse=descending)
    await read_by(reader, "t", by={"a": "k2"}, columns=("b",))
    assert ["ORDER BY" in sql for sql, _ in reader.statements] == [True, True, False]


async def test_read_by_reads_every_column_when_none_is_named(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``columns=None`` stops reading the whole row, in the table's order."""
    rows = await read_by(ThreadedDuckDB(db), "t", by={"a": "k0", "b": 2})
    assert rows == [{"a": "k0", "b": 2, "v": 2, "flag": False}]


async def test_none_pins_is_null_and_anyof_a_set(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a None pin stops meaning IS NULL, or an AnyOf a set."""
    reader = _Recording(db)
    rows = await read_by(reader, "t", by={"a": AnyOf(["k0", "k2"], cast="text"), "v": None}, columns=("b",))
    assert sorted(row["b"] for row in rows) == [1, 3, 9, 11]
    assert reader.statements[0][0] == ('SELECT t."b" FROM "t" AS t WHERE t."a" = ANY($1::text[]) AND t."v" IS NULL')


async def test_each_pin_is_bound_as_the_hand_written_statement_bound_it(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a pin is sent other than the hand-written statement sent it: a value as ``= $n`` with the value as the
    parameter (the column types it, as before), a boolean as its literal (the hand-written ``flag = FALSE``: a bound
    ``flag = $2`` cannot prove a partial index's ``WHERE flag = false`` under a generic plan, which PostgreSQL
    switches a prepared statement to after five runs). The rows are the same either way."""
    reader = _Recording(db)
    rows = await read_by(reader, "t", by={"a": "k1", "flag": False, "b": 5}, columns=("b",), records=True)
    assert [row["b"] for row in rows] == [5]
    await read_by(reader, "t", by={"flag": True, "a": "k0"}, columns=("b",), records=True)
    assert reader.statements == [
        ('SELECT t."b" FROM "t" AS t WHERE t."a" = $1 AND t."flag" = FALSE AND t."b" = $2', ("k1", 5)),
        ('SELECT t."b" FROM "t" AS t WHERE t."flag" = TRUE AND t."a" = $1', ("k0",)),
    ]


async def test_a_text_a_declared_partial_predicate_names_is_written_as_its_literal(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if a pin of a text a declared partial index's predicate names (``WHERE a = 'k-stated'``) is sent as
    ``= $n`` (a generic plan cannot prove the predicate, so the planner passes over the index), or if any other pin is
    written as a literal: another text of that column, the same text of another table, or a pinned column no
    predicate names (coordinator ruling 2026-11-14, as Pace's boolean ruling)."""
    register_partial_predicates("t", ["a = 'k-stated' AND flag = FALSE", "v IS NOT NULL"])
    reader = _Recording(db)
    assert await read_by(reader, "t", by={"a": "k-stated", "b": 5}, columns=("b",), records=True) == []
    await read_by(reader, "t", by={"a": "k1", "b": 5}, columns=("b",), records=True)
    assert reader.statements == [
        ('SELECT t."b" FROM "t" AS t WHERE t."a" = \'k-stated\' AND t."b" = $1', (5,)),
        ('SELECT t."b" FROM "t" AS t WHERE t."a" = $1 AND t."b" = $2', ("k1", 5)),
    ]
    assert read_by_statement("u", by={"a": "k-stated"}, columns=("b",)) == (
        'SELECT t."b" FROM "u" AS t WHERE t."a" = $1',
        ["k-stated"],
    )


async def test_read_row_asks_for_one_row_and_answers_none_for_none(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_row asks for more than one row, or answers anything but None where the key names none."""
    reader = _Recording(db)
    assert await read_row(reader, "t", by={"a": "k1", "b": 5}) == {"a": "k1", "b": 5, "v": None, "flag": False}
    assert await read_row(reader, "t", by={"a": "k9", "b": 5}) is None
    assert reader.statements[0][0].endswith(" LIMIT 1")


async def test_read_value_answers_the_column_of_the_first_row_or_none(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_value answers another column, or anything but None for no row and for a NULL."""
    conn = ThreadedDuckDB(db)
    assert await read_value(conn, "t", by={"a": "k2", "b": 8}, column="v") == 8
    assert await read_value(conn, "t", by={"a": "k2", "b": 9}, column="v") is None
    assert await read_value(conn, "t", by={"a": "k7", "b": 9}, column="v") is None


async def test_read_value_keeps_its_guards_beside_the_pins(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_value drops a ``where=`` guard (the codemod sends a predicate other than equality as one): the
    row the pins name is answered only where the guard holds too."""
    conn = _Recording(db)
    assert await read_value(conn, "t", by={"a": "k2"}, column="b", where=[Fragment("b > $1", (9,))]) in {10, 11}
    assert await read_value(conn, "t", by={"a": "k2"}, column="b", where=[Fragment("b > $1", (11,))]) is None
    assert "b > $2" in conn.statements[-1][0]


async def test_a_read_pinning_nothing_is_refused(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a point read with no pin is sent: it reads the table whole."""
    with pytest.raises(ValueError, match="pins at least one column"):
        await read_by(ThreadedDuckDB(db), "t", by={})
    with pytest.raises(ValueError, match="names the columns"):
        await read_row(ThreadedDuckDB(db), "t", by={"a": "k1"}, columns=())


class _Sent:
    """keeps every statement sent; no row is there."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        return []


@pytest.mark.parametrize(
    ("lock", "clause"),
    [
        (Lock.UPDATE_NOWAIT, " LIMIT 1 FOR UPDATE OF t NOWAIT"),
        (Lock.SHARE, " LIMIT 1 FOR SHARE OF t"),
        (Lock.UPDATE_SKIP_LOCKED, " LIMIT 1 FOR UPDATE OF t SKIP LOCKED"),
    ],
)
async def test_a_locked_point_read_locks_its_row_as_asked(lock: Lock, clause: str) -> None:
    """Flips if a point read's lock is dropped, or its wait (NOWAIT, SKIP LOCKED) is lost or misplaced."""
    conn = _Sent()
    await read_row(conn, "epochs", by={"scope": "*"}, columns=("writing",), lock=lock)
    [(sql, params)] = conn.calls
    assert sql == f'SELECT t."writing" FROM "epochs" AS t WHERE t."scope" = $1{clause}'
    assert params == ("*",)


async def test_an_unlocked_point_read_takes_no_lock() -> None:
    """Flips if a read asking no lock takes one."""
    conn = _Sent()
    await read_by(conn, "epochs", by={"scope": "*"}, columns=("writing",))
    [(sql, _params)] = conn.calls
    assert "FOR " not in sql


async def test_a_nearest_read_orders_by_the_distance_alone_under_its_limit() -> None:
    """Flips if ``read_nearest`` does not send its pins and guards, the similarity (``1 -`` the distance) after the
    columns, the order by the vector column's distance from the vector bound once past them (no tie-break column, so an
    approximate-nearest-neighbour index answers it), and its LIMIT last, all inside a subquery the outer read answers
    the columns and the similarity from, re-sorted by the distance (an iterative HNSW scan in relaxed order hands rows
    back slightly out of order), naming no relation but the table (no CTE: the hub broker's gates read every name); or
    takes a distance pgvector has no operator for, or a negative limit."""
    conn = _Sent()
    await read_nearest(
        conn,
        "memories",
        by={"agent_id": "a", "user_id": "u"},
        nearest=Nearest("embedding", "[1,0]", similarity="similarity"),
        limit=4,
        columns=("memory_id",),
        where=[Fragment("t.salience >= $1", (0.5,))],
    )
    [(sql, params)] = conn.calls
    distance = 't."embedding" OPERATOR(public.<=>) $4::text::public.vector'
    assert sql == (
        f'SELECT n."memory_id", n."similarity" FROM (SELECT t."memory_id", 1 - ({distance}) AS "similarity", '
        f'{distance} AS "_nearest_distance" FROM "memories" AS t WHERE t."agent_id" = $1 AND t."user_id" = $2 AND '
        f'(t.salience >= $3) ORDER BY {distance} LIMIT $5) AS n ORDER BY n."_nearest_distance"'
    )
    assert params == ("a", "u", 0.5, "[1,0]", 4)
    conn.calls.clear()
    # every column (none named): the row travels whole through the wrapper and is spread again; a Selected column is
    # answered under its own name; no similarity named, none answered
    await read_nearest(conn, "memories", by={"agent_id": "a"}, nearest=Nearest("embedding", "[1]"), limit=2)
    await read_nearest(
        conn,
        "memories",
        by={"agent_id": "a"},
        nearest=Nearest("embedding", "[1]"),
        limit=2,
        columns=(Selected("embedding::text", "embedding"),),
    )
    [(whole, _), (named, _)] = conn.calls
    distance = 't."embedding" OPERATOR(public.<=>) $2::text::public.vector'
    assert whole == (
        f'SELECT (n."_nearest_row").* FROM (SELECT t AS "_nearest_row", {distance} AS "_nearest_distance" FROM '
        f'"memories" AS t WHERE t."agent_id" = $1 ORDER BY {distance} LIMIT $3) AS n ORDER BY n."_nearest_distance"'
    )
    assert named.startswith('SELECT n."embedding" FROM (SELECT embedding::text AS "embedding", ')
    assert named.endswith('LIMIT $3) AS n ORDER BY n."_nearest_distance"')


async def test_a_text_ranked_read_orders_by_the_rank_best_first_under_its_limit() -> None:
    """Flips if ``read_nearest`` given a ``TextRank`` does not send its pins and guards, the rank under its name, the
    order by that rank descending (best first, as ``ORDER BY fts_rank DESC`` read it) with the query text bound once
    past them and its LIMIT last, or evaluates the columns before that LIMIT rather than over the kept rows alone,
    re-sorted by the rank (WHY at ``read_nearest_statement``)."""
    conn = _Sent()
    # a relation no declaration registers (an unkeyed one): agent-memory registers "memories" with its primary key
    # when imported (test_schema_backed's backward-compat test does), and a keyed table's ranked read is another
    # statement (the next test's), so this one failed whenever it ran after that import in one process (YB1)
    await read_nearest(
        conn,
        "ranked_unkeyed_probe",
        by={"agent_id": "a", "user_id": "u"},
        nearest=TextRank("search_vector", "red apples"),
        limit=3,
        columns=("memory_id",),
        where=[Fragment("(t.search_vector @@ websearch_to_tsquery('english', $1)) IS TRUE", ("red apples",))],
    )
    [(sql, params)] = conn.calls
    rank = "ts_rank_cd(t.\"search_vector\", websearch_to_tsquery('english', $4))"
    assert sql == (
        f'SELECT t."memory_id", t."fts_rank" FROM (SELECT t.*, {rank} AS "fts_rank" FROM "ranked_unkeyed_probe" AS t WHERE '
        't."agent_id" = $1 AND t."user_id" = $2 AND ((t.search_vector @@ websearch_to_tsquery(\'english\', $3)) IS '
        f'TRUE) ORDER BY {rank} DESC LIMIT $5) AS t ORDER BY t."fts_rank" DESC'
    )
    assert params == ("a", "u", "red apples", "red apples", 3)


async def test_a_text_ranked_read_of_a_keyed_table_reads_its_columns_again_for_the_kept_rows_alone() -> None:
    """Flips if a ``TextRank`` read of a table with a declared primary key reads more than that key and the rank under
    its order and LIMIT, or does not read each kept row again by every key column (driven from the kept rows) for its
    columns, ``Selected`` expressions over ``t`` included, re-sorted by the rank with its ties (WHY at
    ``read_nearest_statement``): the select list then ran on every ranked row."""
    register_table_indexes(
        "ranked_by_key_probe", [DeclaredIndex("ranked_by_key_probe_pkey", ("agent_id", "doc_id"), primary=True)]
    )
    conn = _Sent()
    await read_nearest(
        conn,
        "ranked_by_key_probe",
        by={"agent_id": "a"},
        nearest=TextRank("search_vector", "red", then=("doc_id",)),
        limit=3,
        columns=("doc_id", Selected("t.embedding::text", "embedding")),
    )
    [(sql, params)] = conn.calls
    rank = "ts_rank_cd(t.\"search_vector\", websearch_to_tsquery('english', $2))"
    assert sql == (
        f'SELECT t."doc_id", t.embedding::text AS "embedding", r."fts_rank" FROM (SELECT t."agent_id", t."doc_id", '
        f'{rank} AS "fts_rank" FROM "ranked_by_key_probe" AS t WHERE t."agent_id" = $1 ORDER BY {rank} DESC, '
        't."doc_id" LIMIT $3) AS r CROSS JOIN LATERAL (SELECT * FROM "ranked_by_key_probe" AS t WHERE t."agent_id" = '
        'r."agent_id" AND t."doc_id" = r."doc_id") AS t ORDER BY r."fts_rank" DESC, t."doc_id"'
    )
    assert params == ("a", "red", 3)


@pytest.mark.parametrize("config", ["english", "pg_catalog.simple", "Spanish_2"])
def test_a_text_rank_takes_a_configuration_named_as_an_identifier(config: str) -> None:
    """Flips if a text search configuration named as a plain or schema-qualified identifier is refused: the rank
    writes it as a literal, and every such name is a safe one."""
    assert config in TextRank("search_vector", "x", config=config).distance(1)


@pytest.mark.parametrize(
    "config",
    ["english', $1)) OR TRUE OR (('", "english; DROP TABLE t", "", "a.b.c", "1english", "english'", 'en"glish'],
)
def test_a_text_rank_refuses_a_configuration_that_is_not_an_identifier(config: str) -> None:
    """Flips if a configuration that is not an identifier reaches the statement: the rank writes it as a literal
    (``websearch_to_tsquery('<config>', $n)``), so a caller's language with a quote in it would be SQL."""
    with pytest.raises(ValueError, match="text search configuration"):
        TextRank("search_vector", "x", config=config)


async def test_a_text_ranked_read_breaks_rank_ties_by_its_then_terms() -> None:
    """Flips if a ``TextRank``'s ``then`` terms are not ordered after the rank, each in its own direction with its NULLs
    where it says (a hand-written ``ORDER BY fts_rank DESC, last_used_at DESC NULLS LAST, date_created DESC, id``):
    rows tied on the rank must come back in that order, never the engine's."""
    conn = _Sent()
    await read_nearest(
        conn,
        "agent_skills",
        by={"agent_id": "a"},
        nearest=TextRank(
            "search_vector",
            "red",
            then=(
                Sort("last_used_at", descending=True, nullable=True),
                Sort("date_created", descending=True),
                Sort("skill_id", descending=True),
            ),
        ),
        limit=3,
        columns=("skill_id",),
    )
    [(sql, params)] = conn.calls
    rank = "ts_rank_cd(t.\"search_vector\", websearch_to_tsquery('english', $2))"
    ties = 't."last_used_at" DESC NULLS FIRST, t."date_created" DESC, t."skill_id" DESC'
    assert sql.endswith(f'ORDER BY {rank} DESC, {ties} LIMIT $3) AS t ORDER BY t."fts_rank" DESC, {ties}'), sql
    assert params == ("a", "red", 3)


async def test_text_ranked_ties_come_back_in_the_then_order(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if rows tied on the rank are not answered in the ``then`` order on a real engine: every row of ``t`` ties
    (a constant rank over a missing text column is not DuckDB's, so the tie is the order's own: ``v`` descending, NULLs
    last, then the key ``b``)."""
    db.execute("CREATE MACRO ts_rank_cd(a, b) AS 1")
    db.execute("CREATE MACRO websearch_to_tsquery(c, q) AS q")
    db.execute("ALTER TABLE t ADD COLUMN sv VARCHAR")
    rows = await read_nearest(
        ThreadedDuckDB(db),
        "t",
        by={"a": "k1"},
        nearest=TextRank("sv", "x", then=(Sort("v", descending=True, nulls="last"), Sort("b", cast=None))),
        limit=4,
        columns=("b", "v"),
    )
    assert [(row["b"], row["v"]) for row in rows] == [(6, 6), (4, 4), (5, None), (7, None)]


async def test_a_pattern_ranked_read_counts_its_patterns_in_one_statement_with_its_left_joins_inside() -> None:
    """Flips if ``read_nearest`` given a ``PatternCount`` does not send ONE statement: the pins first, a row passing
    when any field matches any pattern, the patterns bound once as one array, each ``LEFT`` join inside the ranked
    read (a joined field ranks the row before the limit cuts; a later join reads an earlier one's name), the count
    under its name, best first then the ``then`` columns, its LIMIT last."""
    conn = _Answering([{"id": 1, "match_score": 2}])
    rows = await read_nearest(
        conn,
        "columns",
        by={"source_id": "s"},
        nearest=PatternCount(("name", "doc"), ["%a%", "%b%"], then=("name",)),
        limit=5,
        joins=(
            Joined(
                "tables", on={"source_id": "source_id", "table": "table"}, columns={"bound": "template_id"}, left=True
            ),
            Joined("template_docs", on={"template_id": "bound", "name": "name"}, columns={"doc": "text"}, left=True),
        ),
    )
    assert rows == [{"id": 1, "match_score": 2}]
    [(sql, params)] = conn.calls
    fields = ['t."name"', 'j2."text"']
    assert sql == (
        'SELECT t.*, j1."template_id" AS "bound", j2."text" AS "doc", '
        f"(CASE WHEN {fields[0]} ILIKE ($2::text[])[1] OR {fields[1]} ILIKE ($2::text[])[1] THEN 1 ELSE 0 END + "
        f"CASE WHEN {fields[0]} ILIKE ($2::text[])[2] OR {fields[1]} ILIKE ($2::text[])[2] THEN 1 ELSE 0 END) "
        'AS "match_score" '
        'FROM "columns" AS t LEFT JOIN "tables" AS j1 ON j1."source_id" = t."source_id" AND j1."table" = t."table" '
        'LEFT JOIN "template_docs" AS j2 ON j2."template_id" = j1."template_id" AND j2."name" = t."name" '
        f'WHERE t."source_id" = $1 AND ({fields[0]} ILIKE ANY($2::text[]) OR {fields[1]} ILIKE ANY($2::text[])) '
        'ORDER BY "match_score" DESC, t."name" LIMIT $3'
    )
    assert params == ("s", ["%a%", "%b%"], 5)


async def test_both_ranks_order_their_ties_by_one_then_spelling() -> None:
    """Flips if a ``PatternCount``'s and a ``TextRank``'s ``then`` are spelled two ways: each takes a column named as
    text (ascending, as written) and a ``Sort`` (its direction and NULLs), ordered after the rank the same way."""
    then = ("name", Sort("seen", descending=True, nulls="last"))
    conn = _Answering([], [])
    await read_nearest(
        conn, "columns", by={"source_id": "s"}, nearest=PatternCount(("name",), ["%a%"], then=then), limit=2
    )
    await read_nearest(conn, "columns", by={"source_id": "s"}, nearest=TextRank("sv", "a", then=then), limit=2)
    ties = ', t."name", t."seen" DESC NULLS LAST LIMIT $'
    assert all(ties in sql for sql, _ in conn.calls), conn.calls


async def test_a_pattern_ranked_read_refuses_what_it_cannot_rank() -> None:
    """Flips if a pattern-ranked read pinning nothing, or joining other than a plain LEFT join, is sent."""
    rank = PatternCount(("name",), ["%a%"])
    with pytest.raises(ValueError, match="pins at least one column"):
        await read_nearest(_Answering([]), "columns", by={}, nearest=rank, limit=5)
    with pytest.raises(ValueError, match="plain LEFT joins only"):
        await read_nearest(
            _Answering([]),
            "columns",
            by={"source_id": "s"},
            nearest=rank,
            limit=5,
            joins=(Joined("tables", on={"source_id": "source_id"}, columns={"bound": "template_id"}),),
        )
    with pytest.raises(ValueError, match="at least one field"):
        PatternCount((), ["%a%"])


class _Answering:
    """answers each statement sent with the next of its scripted row lists, and keeps the statements."""

    def __init__(self, *answers: list[dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        return self.answers.pop(0)


async def test_a_ranked_read_with_a_left_join_is_one_statement_joining_each_ranked_row_by_key() -> None:
    """Flips if a ranked read's plain ``LEFT`` join is not sent as one statement: the ranked read as a
    derived table, each row ``LEFT JOIN LATERAL``-ed to its parents by the join's columns
    and pins, best first by the similarity it selects, under the same ``LIMIT`` (so a row matching two parents is answered once for
    each, and the copies count against the limit, as ``ORDER BY ... LIMIT`` over the hand-written join counted them);
    or if the column read only to join on is left in the rows. That the database answers this as the
    hand-written join did is the memory package's same-results tests' (PostgreSQL and YugabyteDB)."""
    conn = _Answering(
        [
            {"chunk_id": "c1", "agent_id": "a", "memory_id": "m1", "similarity": 0.9, "media_id": "f1"},
            {"chunk_id": "c1", "agent_id": "a", "memory_id": "m1", "similarity": 0.9, "media_id": "f2"},
            {"chunk_id": "c2", "agent_id": "a", "memory_id": "m2", "similarity": 0.8, "media_id": None},
        ],
    )
    rows = await read_nearest(
        conn,
        "memory_chunks",
        by={"agent_id": "a", "user_id": "u"},
        nearest=Nearest("embedding", "[1,0]", similarity="similarity"),
        limit=3,
        columns=("chunk_id", "agent_id"),
        joins=[
            Joined(
                "media",
                on={"agent_id": "agent_id", "memory_id": "memory_id"},
                columns={"media_id": "media_id"},
                left=True,
                by={"kind": "file"},
            )
        ],
    )
    assert rows == [
        {"chunk_id": "c1", "agent_id": "a", "similarity": 0.9, "media_id": "f1"},
        {"chunk_id": "c1", "agent_id": "a", "similarity": 0.9, "media_id": "f2"},
        {"chunk_id": "c2", "agent_id": "a", "similarity": 0.8, "media_id": None},
    ]
    [(sql, params)] = conn.calls
    distance = 't."embedding" OPERATOR(public.<=>) $3::text::public.vector'
    ranked = (
        'SELECT n."chunk_id", n."agent_id", n."memory_id", n."similarity" FROM (SELECT t."chunk_id", t."agent_id",'
        f' t."memory_id", 1 - ({distance}) AS "similarity", {distance} AS "_nearest_distance" FROM "memory_chunks"'
        f' AS t WHERE t."agent_id" = $1 AND t."user_id" = $2 ORDER BY {distance} LIMIT $4) AS n'
        ' ORDER BY n."_nearest_distance"'
    )
    assert sql == (
        f'SELECT r.*, j1."media_id" AS "media_id" FROM ({ranked}) AS r LEFT JOIN LATERAL (SELECT "media_id" FROM'
        ' "media" AS j WHERE j."agent_id" = r."agent_id" AND j."memory_id" = r."memory_id" AND j."kind" = $5) AS j1'
        ' ON TRUE ORDER BY r."similarity" DESC LIMIT $4'
    )
    assert params == ("a", "u", "[1,0]", 3, "file")


async def test_a_ranked_read_with_a_counted_join_reads_its_parents_by_one_keyed_batch() -> None:
    """Flips if a ranked read's counted join is not read as the ranked rows and then one keyed batch of the parents
    (each row kept once beside its count), the shape a single joined statement does not take."""
    conn = _Answering(
        [
            {"chunk_id": "c1", "agent_id": "a", "memory_id": "m1", "similarity": 0.9},
            {"chunk_id": "c2", "agent_id": "a", "memory_id": "m2", "similarity": 0.8},
        ],
        [
            {"agent_id": "a", "memory_id": "m1"},
            {"agent_id": "a", "memory_id": "m1"},
        ],
    )
    rows = await read_nearest(
        conn,
        "memory_chunks",
        by={"agent_id": "a", "user_id": "u"},
        nearest=Nearest("embedding", "[1,0]", similarity="similarity"),
        limit=3,
        columns=("chunk_id", "agent_id"),
        joins=[
            Joined(
                "media",
                on={"agent_id": "agent_id", "memory_id": "memory_id"},
                columns={},
                counts={"files": None},
                casts={"agent_id": "text", "memory_id": "text"},
            )
        ],
    )
    assert rows == [
        {"chunk_id": "c1", "agent_id": "a", "similarity": 0.9, "files": 2},
        {"chunk_id": "c2", "agent_id": "a", "similarity": 0.8, "files": 0},
    ]
    (ranked_sql, _), (joined_sql, _) = conn.calls
    assert 't."memory_id"' in ranked_sql and "LIMIT" in ranked_sql
    assert '"media"' in joined_sql


async def test_a_ranked_read_joins_left_only() -> None:
    """Flips if a ranked read takes an inner join: its rows would be dropped after the limit, where the hand-written
    join dropped them before it."""
    with pytest.raises(ValueError, match="LEFT only"):
        await read_nearest(
            _Sent(),
            "media_content",
            by={"agent_id": "a"},
            nearest=Nearest("embedding", "[1]"),
            limit=2,
            joins=[Joined("media", on={"agent_id": "agent_id"}, columns={"m": "media_id"})],
        )
