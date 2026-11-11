"""relational steps over change relations: each shape answers what the hand-written statement it replaces answers.

Every case runs through the equivalence harness (:func:`assert_same_results`) on an inline DuckDB, seeded with empty,
one, many, NULL, tied and two-tenant rows: the old statement is the shape as a load wrote it by hand; the helper is
the step. ``changed`` is the change relation (its keys hold a NULL office, matched null-safe where the shape says).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from threetears.core.keyed import (
    Fragment,
    InlineDuckDB,
    ThreadedDuckDB,
    Match,
    Quoted,
    Selected,
    Sort,
    anti,
    any_step,
    change_relation,
    count_step,
    delete_matching,
    distinct,
    except_rows,
    first_difference,
    grouped,
    joined,
    largest,
    latest_per_key,
    materialize,
    read_step,
    read_step_arrow,
    row_count,
    run_inline,
    scan,
    semi,
    union,
    update_from,
    windowed,
    xor_hash,
)
from threetears.core.testing.same_results import Seeded, assert_same_results

_EPOCH = datetime(2026, 11, 3)


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    db = duckdb.connect()
    db.execute("CREATE TABLE held (state VARCHAR, office VARCHAR, run_id BIGINT, instant TIMESTAMP, v BIGINT)")
    db.execute("CREATE TABLE changed (state VARCHAR, office VARCHAR, run_id BIGINT)")
    db.execute(
        "INSERT INTO changed VALUES ('S0', NULL, 1), ('S0', 'o0', 2), ('S0', 'o1', 3), ('S1', 'o1', 3), "
        "('S0', 'o2', 999), ('S0', 'o0', 2)"
    )
    yield db
    db.close()


def _row(seed: Seeded) -> dict[str, Any]:
    return {
        "state": f"S{seed.tenant}",
        "office": None if seed.null else f"o{seed.tie % 3}",
        "run_id": seed.index,
        "instant": None if seed.null else _EPOCH + timedelta(hours=seed.tie),
        "v": seed.index * 10,
    }


_CHANGED = change_relation("changed")
_KEY = [Match("state"), Match("office", null_safe=True), Match("run_id")]
_TENANT = Fragment('t."state" = $1', ("S0",))


def _same(conn: duckdb.DuckDBPyConnection, old_sql: str, helper: Any, **kwargs: Any) -> None:
    run_inline(
        assert_same_results(
            InlineDuckDB(conn), table="held", row=_row, old_sql=old_sql, params=["S0"], helper=helper, **kwargs
        )
    )


def test_a_semi_step_answers_the_semi_join_it_replaces(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if semi drops a row a NULL office matches null-safe, keeps one twice for a key named twice, or loses
    the tenant filter."""
    _same(
        conn,
        "SELECT t.* FROM held AS t SEMI JOIN changed AS k ON t.state = k.state AND t.office IS NOT DISTINCT FROM "
        "k.office AND t.run_id = k.run_id WHERE t.state = $1",
        lambda c: read_step(c, semi("held", _CHANGED, on=_KEY, where=[_TENANT])),
    )


def test_an_anti_step_with_its_own_condition_answers_the_not_exists_it_replaces(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if anti keeps a row a key matches, drops one a NULL key fails to match, or ignores ``also=``."""
    _same(
        conn,
        "SELECT t.* FROM held AS t WHERE NOT EXISTS (SELECT 1 FROM changed AS k WHERE t.state = k.state AND "
        "t.office IS NOT DISTINCT FROM k.office AND t.run_id = k.run_id AND t.v IS NOT NULL) AND t.state = $1",
        lambda c: read_step(c, anti("held", _CHANGED, on=_KEY, also=[Fragment('t."v" IS NOT NULL')], where=[_TENANT])),
    )


def test_a_count_of_an_anti_step_is_the_count_it_replaces(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if count_step counts other rows than the anti join's ``count(*)``."""

    async def counted(c: Any) -> list[dict[str, Any]]:
        return [{"n": await count_step(c, anti("held", _CHANGED, on=_KEY[:1] + _KEY[2:], where=[_TENANT]))}]

    _same(
        conn,
        "SELECT count(*) AS n FROM held AS t ANTI JOIN changed AS k ON t.state = k.state AND t.run_id = k.run_id "
        "WHERE t.state = $1",
        counted,
    )


def test_a_distinct_step_of_a_semi_step_is_one_statement_of_both(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if steps do not compose (a step read by a step), its parameters are misnumbered, or a cast column
    loses its name."""
    office = Selected('CAST(t."office" AS VARCHAR)', "office")
    _same(
        conn,
        "SELECT DISTINCT r.state, CAST(r.office AS VARCHAR) AS office FROM held AS r SEMI JOIN changed AS k ON "
        "r.state = k.state AND r.run_id = k.run_id WHERE r.state = $1",
        lambda c: read_step(
            c,
            distinct(semi("held", _CHANGED, on=[Match("state"), Match("run_id")], where=[_TENANT]), ["state", office]),
        ),
    )


def test_a_grouped_step_answers_the_group_by_it_replaces(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a group's mark, its no-value default, its count or its order-free digest differs from the
    hand-written aggregate."""
    _same(
        conn,
        "SELECT t.state, t.run_id, max(t.v) AS v, coalesce(max(CAST(t.instant AS TIMESTAMPTZ)), "
        "'-infinity'::TIMESTAMPTZ) AS instant, count(*) AS n, bit_xor(hash(t.office, t.v)) AS content "
        "FROM held AS t WHERE t.state = $1 GROUP BY t.state, t.run_id",
        lambda c: read_step(
            c,
            grouped(
                "held",
                by=["state", "run_id"],
                aggregates=[
                    largest("v"),
                    largest("instant", cast="TIMESTAMPTZ", default="'-infinity'::TIMESTAMPTZ"),
                    row_count(),
                    xor_hash(["office", "v"]),
                ],
                where=[_TENANT],
            ),
        ),
    )


def test_the_newest_row_of_each_key_is_the_one_qualify_keeps(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if latest_per_key keeps another row of a key than the QUALIFY row_number() = 1 it replaces."""
    _same(
        conn,
        "SELECT * FROM held AS t WHERE t.state = $1 QUALIFY row_number() OVER (PARTITION BY t.office ORDER BY "
        "t.instant DESC NULLS LAST, t.run_id DESC) = 1",
        lambda c: read_step(
            c,
            latest_per_key(
                "held",
                key=["office"],
                order=[Sort("instant", descending=True, nullable=True), Sort("run_id", descending=True)],
                where=[_TENANT],
            ),
        ),
        ordered=False,  # the window's ORDER BY orders no rows
    )


def test_every_row_tied_newest_is_kept_with_ties(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if ties=True keeps one row a key (DISTINCT ON) where the QUALIFY max() OVER it replaces keeps each tie."""
    _same(
        conn,
        "SELECT DISTINCT t.state, t.office, t.instant FROM held AS t WHERE t.state = $1 AND t.instant IS NOT NULL "
        "QUALIFY t.instant = max(t.instant) OVER (PARTITION BY t.state)",
        lambda c: read_step(
            c,
            distinct(
                latest_per_key(
                    "held",
                    key=["state"],
                    order=[Sort("instant", descending=True)],
                    columns=["state", "office", "instant"],
                    where=[_TENANT, Fragment('t."instant" IS NOT NULL')],
                    ties=True,
                ),
                ["state", "office", "instant"],
            ),
        ),
    )


def test_a_windowed_step_answers_the_qualify_it_replaces(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if windowed keeps other rows than a caller's window condition, or leaks the window's column."""
    _same(
        conn,
        "SELECT t.* FROM held AS t WHERE t.state = $1 QUALIFY ntile(2) OVER (PARTITION BY t.office ORDER BY "
        "t.run_id) = 1",
        lambda c: read_step(
            c,
            windowed(
                "held",
                windows=[Selected('ntile(2) OVER (PARTITION BY t."office" ORDER BY t."run_id")', "tier")],
                keep=[Fragment("w.tier = $1", (1,))],
                where=[_TENANT],
            ),
        ),
        ordered=False,  # the window's ORDER BY orders no rows
    )


def test_except_and_union_answer_the_set_operations_they_replace(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if except_rows or union(keep_all=) answers other rows (or other duplicates) than EXCEPT / UNION ALL."""
    keys = scan("held", ["state", "office", "run_id"], where=[_TENANT])
    _same(
        conn,
        "SELECT state, office, run_id FROM held WHERE state = $1 EXCEPT SELECT * FROM changed "
        "UNION ALL SELECT * FROM changed",
        lambda c: read_step(c, union(except_rows(keys, _CHANGED), _CHANGED, keep_all=True)),
    )


def test_a_step_reads_into_arrow_typed_as_asked(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_step_arrow answers other rows than read_step, or other columns or types than ``types``."""
    conn.execute(
        "INSERT INTO held VALUES ('S0', 'o0', 2, NULL, 20), ('S0', NULL, 1, NULL, 10), ('S1', 'o1', 3, NULL, 3)"
    )
    step = semi("held", _CHANGED, on=_KEY, columns=["run_id", "office"])
    types = {"run_id": pa.int32(), "office": pa.string()}
    table = run_inline(read_step_arrow(InlineDuckDB(conn), step, types=types, order=[Sort("run_id")]))
    assert table.schema == pa.schema(list(types.items()))
    assert table.to_pylist() == [
        {"run_id": 1, "office": None},
        {"run_id": 2, "office": "o0"},
        {"run_id": 3, "office": "o1"},
    ]


def test_a_step_read_into_arrow_on_duckdb_with_no_types_is_duckdbs_own_answer(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_step_arrow on DuckDB (inline or on a thread) without ``types`` answers other rows, columns or
    Arrow types than DuckDB's own answer to the step's statement, in the order asked; or if a reader that is no
    DuckDB is read without them (its rows carry no types to make the table of)."""
    conn.execute(
        "INSERT INTO held VALUES ('S0', 'o0', 2, NULL, 20), ('S0', NULL, 1, NULL, 10), ('S1', 'o1', 3, NULL, 3)"
    )
    step = semi("held", _CHANGED, on=_KEY, columns=["run_id", "office", "instant"])
    own = conn.execute(
        f"SELECT o.* FROM ({step.sql}) AS o ORDER BY o.run_id ASC", list(step.params)
    ).fetch_arrow_table()
    inline = run_inline(read_step_arrow(InlineDuckDB(conn), step, order=[Sort("run_id")]))
    threaded = asyncio.run(read_step_arrow(ThreadedDuckDB(conn), step, order=[Sort("run_id")]))
    for table in (inline, threaded):
        assert table.schema == own.schema
        assert table.to_pylist() == own.to_pylist()
    assert [row["run_id"] for row in inline.to_pylist()] == [1, 2, 3]
    whole = run_inline(read_step_arrow(InlineDuckDB(conn), "held"))
    assert whole.schema == conn.execute("SELECT * FROM held").fetch_arrow_table().schema

    class _Rows:
        async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
            return []

    with pytest.raises(TypeError, match="types"):
        asyncio.run(read_step_arrow(_Rows(), step))


def test_a_delete_by_a_change_relation_s_keys_deletes_what_the_exists_delete_does(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if delete_matching deletes a row no key names, or keeps one a NULL office names null-safe."""

    async def deleted(c: Any) -> list[dict[str, Any]]:
        # DuckDB answers a DELETE with its count, as the old statement's one row
        return [
            {"Count": await delete_matching(c, "held", _CHANGED, on=_KEY, also=[Fragment('t."state" = $1', ("S0",))])}
        ]

    _same(
        conn,
        "DELETE FROM held AS t WHERE EXISTS (SELECT 1 FROM changed AS k WHERE t.state = k.state AND t.office IS NOT "
        "DISTINCT FROM k.office AND t.run_id = k.run_id AND t.state = $1)",
        deleted,
        state_sql="SELECT * FROM held ORDER BY run_id, state",
    )


def test_an_update_from_a_windowed_step_by_rowid_writes_what_the_hand_written_update_does(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if update_from cannot take a step with its parameters as its source, or join on DuckDB's rowid."""

    async def updated(c: Any) -> list[dict[str, Any]]:
        source = scan(
            windowed(
                "held",
                windows=[Selected("t.rowid", "rowid"), Selected('row_number() OVER (ORDER BY t."run_id")', "rank")],
                keep=[Fragment("TRUE")],
                where=[Fragment('t."state" = $1', ("S0",))],
                columns=["rowid", Selected('w."rank"', "v")],
            ),
        )
        # DuckDB answers an UPDATE with its count, as the old statement's one row
        return [{"Count": (await update_from(c, "held", key=["rowid"], source=source, columns=["v"])).count}]

    _same(
        conn,
        "UPDATE held SET v = c.v FROM (SELECT rowid AS _row, row_number() OVER (ORDER BY run_id) AS v FROM held "
        "WHERE state = $1) AS c WHERE held.rowid = c._row",
        updated,
        state_sql="SELECT * FROM held ORDER BY run_id, state",
    )


def test_the_first_differing_row_is_the_one_the_hand_written_join_finds(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if first_difference misses a value that differs only by a NULL, or answers a later row than the first."""
    conn.execute("CREATE TABLE raw AS SELECT range AS __row, 'x' || range AS a FROM range(50)")
    conn.execute("CREATE TABLE seen AS SELECT __row, CASE WHEN __row IN (17, 40) THEN NULL ELSE a END AS a FROM raw")
    found = run_inline(
        first_difference(
            InlineDuckDB(conn),
            change_relation("raw"),
            change_relation("seen"),
            on=[Match("__row")],
            compare=[("a", "a")],
            order=["__row"],
        )
    )
    expected = conn.execute(
        "SELECT raw.__row FROM raw JOIN seen ON seen.__row = raw.__row WHERE raw.a IS DISTINCT FROM seen.a "
        "ORDER BY raw.__row LIMIT 1"
    ).fetchall()
    assert found == {"__row": expected[0][0]} == {"__row": 17}
    conn.execute("UPDATE seen SET a = 'x' || __row")
    assert (
        run_inline(
            first_difference(
                InlineDuckDB(conn), "raw", "seen", on=[Match("__row")], compare=[("a", "a")], order=["__row"]
            )
        )
        is None
    )


def test_a_joined_step_answers_the_join_and_the_left_join_it_replaces(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if joined answers other rows than the hand-written JOIN (a row once for each match, a NULL office matched
    null-safe), or with ``left`` drops a row no key matches or fills its key's columns."""
    on = [Match("state"), Match("office", null_safe=True)]
    columns: list[str | Selected] = ["state", "office", "run_id", Selected('k."run_id"', "k_run")]
    _same(
        conn,
        "SELECT t.state, t.office, t.run_id, k.run_id AS k_run FROM held AS t JOIN changed AS k "
        "ON t.state = k.state AND t.office IS NOT DISTINCT FROM k.office WHERE t.state = $1",
        lambda c: read_step(c, joined("held", _CHANGED, on=on, columns=columns, where=[_TENANT])),
    )
    _same(
        conn,
        "SELECT t.state, t.office, t.run_id, k.run_id AS k_run FROM held AS t LEFT JOIN changed AS k "
        "ON t.state = k.state AND t.office IS NOT DISTINCT FROM k.office AND k.run_id < 100 WHERE t.state = $1",
        lambda c: read_step(
            c,
            joined(
                "held",
                _CHANGED,
                on=on,
                columns=columns,
                where=[_TENANT],
                also=[Fragment("k.run_id < $1", (100,))],
                left=True,
            ),
        ),
    )


def test_a_step_read_up_to_a_limit_answers_the_first_rows_in_its_order(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_step or read_step_arrow with ``limit`` answers other rows than the hand-written ``ORDER BY ...
    LIMIT``, or takes a limit that is no count of rows."""
    step = scan("held", ["state", "run_id"], where=[_TENANT])
    _same(
        conn,
        "SELECT t.state, t.run_id FROM held AS t WHERE t.state = $1 ORDER BY t.run_id LIMIT 3",
        lambda c: read_step(c, step, order=[Sort("run_id")], limit=3),
    )
    conn.execute(
        "INSERT INTO held VALUES ('S0', 'o0', 2, NULL, 20), ('S0', NULL, 1, NULL, 10), ('S0', 'o1', 3, NULL, 3)"
    )
    table = run_inline(read_step_arrow(InlineDuckDB(conn), step, order=[Sort("run_id", descending=True)], limit=2))
    assert table.to_pylist() == [{"state": "S0", "run_id": 3}, {"state": "S0", "run_id": 2}]
    assert len(run_inline(read_step(InlineDuckDB(conn), step, limit=1))) == 1
    for refused in (-1, True, 1.5):
        with pytest.raises(ValueError, match="limit"):
            run_inline(read_step(InlineDuckDB(conn), step, limit=refused))  # type: ignore[arg-type]


def test_whether_a_step_answers_any_row_is_the_exists_it_replaces(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if any_step answers otherwise than the hand-written ``SELECT EXISTS`` over the same rows."""

    async def held(c: Any) -> list[dict[str, Any]]:
        return [{"held": await any_step(c, semi("held", _CHANGED, on=_KEY, where=[_TENANT]))}]

    _same(
        conn,
        "SELECT EXISTS (SELECT 1 FROM held AS t WHERE EXISTS (SELECT 1 FROM changed AS k WHERE t.state = k.state AND "
        "t.office IS NOT DISTINCT FROM k.office AND t.run_id = k.run_id) AND t.state = $1) AS held",
        held,
    )
    assert run_inline(any_step(InlineDuckDB(conn), "changed")) is True
    assert run_inline(any_step(InlineDuckDB(conn), scan("changed", where=[Fragment("FALSE")]))) is False


def test_a_table_of_a_steps_rows_holds_what_the_hand_written_create_table_as_does(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if materialize makes a table of other rows than ``CREATE TABLE ... AS`` the step's statement, answers
    another count, keeps a table it replaces, makes no temporary table when asked, or replaces one when told not to."""

    async def made(c: Any) -> list[dict[str, Any]]:
        step = semi("held", _CHANGED, on=_KEY, columns=["state", "run_id", "v"], where=[_TENANT])
        return [{"Count": await materialize(c, "made", step)}]

    conn.execute("CREATE TABLE made AS SELECT 1 AS stale")
    _same(
        conn,
        "CREATE OR REPLACE TABLE made AS SELECT t.state, t.run_id, t.v FROM held AS t WHERE EXISTS (SELECT 1 FROM "
        "changed AS k WHERE t.state = k.state AND t.office IS NOT DISTINCT FROM k.office AND t.run_id = k.run_id) "
        "AND t.state = $1",
        made,
        state_sql="SELECT * FROM made",
    )
    db = InlineDuckDB(conn)
    assert run_inline(materialize(db, "_named_keys", distinct(_CHANGED, ["state"]), temp=True)) == 2
    temporary = conn.execute("SELECT temporary FROM duckdb_tables() WHERE table_name = '_named_keys'").fetchall()
    assert temporary == [(True,)]
    assert asyncio.run(materialize(ThreadedDuckDB(conn), "again", "changed")) == 6
    with pytest.raises(duckdb.CatalogException):
        run_inline(materialize(db, "again", "changed", replace=False))


def test_the_first_differing_row_among_those_a_condition_names_is_the_hand_written_joins(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if first_difference's ``where`` is dropped (a differing row it rules out answered) or is bound to other
    parameters than its own."""
    conn.execute("CREATE TABLE raw AS SELECT range AS __row, 'x' || range AS a FROM range(50)")
    conn.execute("CREATE TABLE seen AS SELECT __row, CASE WHEN __row IN (17, 40) THEN NULL ELSE a END AS a FROM raw")
    found = run_inline(
        first_difference(
            InlineDuckDB(conn),
            change_relation("raw"),
            scan(change_relation("seen"), where=[Fragment("t.__row >= $1", (0,))]),
            on=[Match("__row")],
            compare=[("a", "a")],
            order=["__row"],
            where=[Fragment("t.__row > $1", (20,))],
        )
    )
    expected = conn.execute(
        "SELECT raw.__row FROM raw JOIN seen ON seen.__row = raw.__row WHERE raw.__row > 20 "
        "AND (raw.a IS DISTINCT FROM seen.a) ORDER BY raw.__row LIMIT 1"
    ).fetchall()
    assert found == {"__row": expected[0][0]} == {"__row": 40}


def test_a_relation_already_quoted_is_read_as_written_never_quoted_again(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a step over a :class:`Quoted` name quotes it again (``\"\"report\"\"`` reads no relation), reads other
    rows than the hand-written statement over it, or takes text that is no relation's name."""
    conn.execute("CREATE SCHEMA report")
    conn.execute("CREATE VIEW report.held AS SELECT * FROM main.held")
    step = scan(Quoted('"report"."held"'), ["state", "run_id"], where=[_TENANT])
    assert 'FROM "report"."held" AS t' in step.sql
    _same(
        conn,
        'SELECT t.state, t.run_id FROM "report"."held" AS t WHERE t.state = $1',
        lambda c: read_step(c, step),
    )
    _same(
        conn,
        "SELECT t.* FROM report.held AS t WHERE EXISTS (SELECT 1 FROM changed AS k WHERE t.state = k.state AND "
        "t.office IS NOT DISTINCT FROM k.office AND t.run_id = k.run_id) AND t.state = $1",
        lambda c: read_step(c, semi(Quoted("report.held"), _CHANGED, on=_KEY, where=[_TENANT])),
    )
    for refused in ("", "held; DROP TABLE held", '"report".held AS x', "(SELECT 1)"):
        with pytest.raises(ValueError, match="quoted relation"):
            Quoted(refused)


def test_steps_refuse_what_cannot_describe_them() -> None:
    """Flips if a step that matches nothing, groups nothing, or unions one relation is rendered."""
    with pytest.raises(ValueError, match="matches on"):
        semi("held", "changed", on=[])
    with pytest.raises(ValueError, match="aggregates"):
        grouped("held", by=["state"], aggregates=[])
    with pytest.raises(ValueError, match="two relations"):
        union("held")
    with pytest.raises(ValueError, match="named"):
        change_relation(" ")
    with pytest.raises(ValueError, match="one sort column"):
        latest_per_key("held", key=["state"], order=[Sort("a"), Sort("b")], ties=True)
