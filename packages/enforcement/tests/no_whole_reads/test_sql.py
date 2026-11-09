"""The text half of the no-whole-reads rule: which statements read or write a table whole."""

from __future__ import annotations

import pytest

from threetears.enforcement.no_whole_reads.sql import is_sql, judge_sql


def _kinds(sql: str) -> list[tuple[str, str]]:
    return [(finding.kind, finding.table) for finding in judge_sql(sql)]


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # election night: the writer's per-load check (loader.checks_sql), every source row grouped
        ("SELECT {} AS ask, {} AS state, count(*) AS n, {} FROM {} GROUP BY {}{}", [("whole-aggregate", "{}")]),
        # election night: _check_counts' count (storage.count)
        ("SELECT COUNT(*) FROM {}", [("whole-aggregate", "{}")]),
        ("SELECT DISTINCT state FROM results", [("whole-aggregate", "results")]),
        ("SELECT max(epoch) FROM t", [("whole-aggregate", "t")]),
        ("SELECT a FROM t WHERE pending IS NOT NULL", [("unbound-where", "t")]),
        ("SELECT a FROM t WHERE k IN (SELECT k FROM s)", [("unbound-where", "t"), ("whole-select", "s")]),
        ("INSERT INTO t SELECT * FROM s", [("whole-select", "s")]),
        ("CREATE TABLE x AS SELECT * FROM t", [("whole-select", "t")]),
        ("UPDATE stamps SET v = $1", [("whole-update", "stamps")]),
        ("UPDATE t SET a = 1 WHERE b IS NULL", [("unbound-where", "t")]),
        ("DELETE FROM t", [("whole-delete", "t")]),
        ("TRUNCATE TABLE t", [("truncate", "t")]),
        ("ANALYZE t", [("whole-maintenance", "t")]),
        ("COPY t TO STDOUT", [("whole-copy-out", "t")]),
        ("SELECT count(*) FROM t WHERE state = $1", [("keyed-count", "t")]),
    ],
)
def test_whole_reads_and_writes_are_found(sql: str, expected: list[tuple[str, str]]) -> None:
    """Flips if a statement that reads or writes every row stops being named, or is named as another kind."""
    assert is_sql(sql)
    assert _kinds(sql) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT writing FROM {} WHERE scope = $1 FOR UPDATE NOWAIT",
        "SELECT id FROM gateway_providers WHERE status = 'active'",
        "SELECT a FROM t ORDER BY k LIMIT 1",
        "SELECT t.* FROM unnest($1::text[]) AS k(id) JOIN t ON t.id = k.id",
        "WITH x AS (SELECT a FROM t WHERE k = $1) SELECT * FROM x",
        "SELECT extract(year FROM ts) FROM t WHERE id = $1",
        "UPDATE stamps SET v = $1 WHERE k = $2",
        "DELETE FROM t WHERE k = ANY($1)",
        "INSERT INTO t (a) VALUES ($1) ON CONFLICT (a) DO UPDATE SET a = excluded.a",
        "CREATE VIEW v AS SELECT * FROM t",
        "SELECT -1 AS ask, state, count(*) AS n FROM {} WHERE 1 = 0 GROUP BY state",
        "SELECT * FROM (SELECT a, k FROM t) AS d WHERE d.k = $1",
        "SELECT m.a FROM memories m WHERE m.agent_id = $1 AND NOT EXISTS (SELECT 1 FROM memories g WHERE g.id = m.id)",
        "SELECT * FROM memories {} AND embedding IS NOT NULL",
        "DELETE FROM role_assignments{} RETURNING assignment_id",
        "SELECT * FROM pg_class",
        "SELECT a, max(b) FROM t WHERE k = $1 GROUP BY a",
    ],
)
def test_keyed_and_bounded_statements_pass(sql: str) -> None:
    """Flips if a read narrowed by a bound key, a LIMIT, a correlated key, a CTE, a contradiction, an outer WHERE
    pushed into a derived table, or a predicate the walk cannot see is called whole."""
    assert _kinds(sql) == []


def test_a_bare_select_from_a_table_is_a_head_judged_where_it_runs() -> None:
    """Flips if a pager's ``SELECT ... FROM <table>`` head is judged as a whole read where it is written."""
    assert _kinds("SELECT {} FROM {}") == [("head", "{}")]


@pytest.mark.parametrize(
    "text", ["Select a model from the list", "delete from another scope", "truncate failed", "analyze images"]
)
def test_prose_is_not_sql(text: str) -> None:
    """Flips if English that starts with a SQL verb is judged as a statement."""
    assert not is_sql(text)
