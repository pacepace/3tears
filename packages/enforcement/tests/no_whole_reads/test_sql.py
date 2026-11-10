"""The text half of the no-whole-reads rule: which statements read or write a table whole, judged against each
table's declared key and indexes."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads.schema import SchemaCatalog, collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, is_sql, judge_sql

#: the function the purposed index below names
_DIGEST = "src/hub/schema_digest.py::SchemaDigest.refresh_active"

#: the tables these statements read, each with the key its declaration gives it
_DDL = f"""
CREATE TABLE t (k TEXT PRIMARY KEY, a TEXT, flag TEXT, created TIMESTAMPTZ, ts TIMESTAMPTZ);
CREATE TABLE s (k TEXT PRIMARY KEY, flag TEXT);
CREATE TABLE results (state TEXT, geo TEXT, candidate TEXT, winner TEXT, a TEXT, PRIMARY KEY (state, geo, candidate));
CREATE TABLE gateway_providers (id UUID PRIMARY KEY, status TEXT);
CREATE TABLE datasources (id UUID PRIMARY KEY, customer_id UUID, status TEXT, kind TEXT);
-- purpose: {_DIGEST}
CREATE INDEX datasources_by_kind ON datasources (kind);
CREATE TABLE memories (id UUID PRIMARY KEY, agent_id UUID, a TEXT);
CREATE TABLE dataset_sources (dataset_id UUID, source_kind TEXT, source_id UUID,
    PRIMARY KEY (dataset_id, source_kind, source_id));
CREATE TABLE roots (id UUID PRIMARY KEY);
CREATE TABLE stamps (k TEXT PRIMARY KEY, v TEXT, kind TEXT);
CREATE TABLE role_assignments (assignment_id UUID PRIMARY KEY);
"""


@pytest.fixture
def schema(tmp_path: Path) -> SchemaCatalog:
    """the catalog the gate reads from a migration declaring these tables."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "v001.sql").write_text(_DDL, encoding="utf-8")
    return collect_schema(tmp_path, (tmp_path / "src",))


def _kinds(sql: str, schema: SchemaCatalog, *names: str) -> list[tuple[str, str]]:
    return [(finding.kind, finding.table) for finding in judge_sql(sql, Reader(schema, frozenset(names)))]


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
        # a key read by a whole subquery: the subquery is the whole read
        ("SELECT a FROM t WHERE k IN (SELECT k FROM s)", [("whole-select", "s")]),
        ("SELECT a FROM t WHERE k IN (SELECT k FROM s WHERE flag = $1)", [("non-key-where", "s")]),
        ("INSERT INTO t SELECT * FROM s", [("whole-select", "s")]),
        ("CREATE TABLE x AS SELECT * FROM t", [("whole-select", "t")]),
        ("UPDATE stamps SET v = $1", [("whole-update", "stamps")]),
        ("UPDATE t SET a = 1 WHERE b IS NULL", [("unbound-where", "t")]),
        ("DELETE FROM t", [("whole-delete", "t")]),
        ("TRUNCATE TABLE t", [("truncate", "t")]),
        ("ANALYZE t", [("whole-maintenance", "t")]),
        ("COPY t TO STDOUT", [("whole-copy-out", "t")]),
        ("SELECT count(*) FROM results WHERE state = $1", [("keyed-count", "results")]),
        # the hole: a bound filter on a status or a kind is a whole read with a filter
        ("SELECT id FROM gateway_providers WHERE status = 'active'", [("non-key-where", "gateway_providers")]),
        (
            "SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1",
            [("non-key-where", "datasources")],
        ),
        ("SELECT a FROM results WHERE geo = $1", [("non-key-where", "results")]),
        ("SELECT count(*) FROM t WHERE flag = $1", [("whole-aggregate", "t")]),
        ("UPDATE stamps SET v = $1 WHERE kind = $2", [("non-key-where", "stamps")]),
        ("DELETE FROM t WHERE created < $1", [("non-key-where", "t")]),
        # a LIMIT is no key: ordered by anything but the key, every row is sorted to find the first
        ("SELECT a FROM t ORDER BY created DESC LIMIT 10", [("whole-select", "t")]),
        ("SELECT a FROM t WHERE flag = $1 LIMIT 1", [("non-key-where", "t")]),
        # a keyset page with a filter beside its range scans until the filter fills the page
        ("SELECT a FROM t WHERE k > $1 AND flag = $2 ORDER BY k LIMIT 100", [("non-key-where", "t")]),
        # a table declared nowhere has no known key
        ("SELECT a FROM nowhere WHERE k = $1", [("unknown-schema", "nowhere")]),
        ("SELECT a FROM {} WHERE scope = $1", [("unknown-schema", "{}")]),
    ],
)
def test_whole_reads_and_writes_are_found(sql: str, expected: list[tuple[str, str]], schema: SchemaCatalog) -> None:
    """Flips if a statement that reads or writes every row stops being named, or is named as another kind."""
    assert is_sql(sql)
    assert _kinds(sql, schema) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT writing FROM t WHERE k = $1 FOR UPDATE NOWAIT",
        "SELECT a FROM t ORDER BY k LIMIT 1",
        "SELECT 1 FROM t LIMIT 1",
        "SELECT a FROM results WHERE state = $1 AND winner IS NOT NULL",
        "SELECT a FROM results WHERE state = $1 AND geo = ANY($2::text[])",
        "SELECT a FROM t WHERE k = $1 OR k = $2",
        "SELECT a FROM {} WHERE {pk} = ANY($1)",
        "SELECT a FROM {} WHERE {pk} > $1 ORDER BY {pk} LIMIT $2",
        "SELECT t.* FROM unnest($1::text[]) AS keys(id) JOIN t ON t.k = keys.id",
        "WITH x AS (SELECT a FROM t WHERE k = $1) SELECT * FROM x",
        "SELECT extract(year FROM ts) FROM t WHERE k = $1",
        "UPDATE stamps SET v = $1 WHERE k = $2",
        "DELETE FROM t WHERE k = ANY($1)",
        "INSERT INTO t (a) VALUES ($1) ON CONFLICT (a) DO UPDATE SET a = excluded.a",
        "CREATE VIEW v AS SELECT * FROM t",
        "SELECT -1 AS ask, state, count(*) AS n FROM {} WHERE 1 = 0 GROUP BY state",
        "SELECT * FROM (SELECT a, k FROM t) AS d WHERE d.k = $1",
        "SELECT m.a FROM memories m WHERE m.id = $1 AND NOT EXISTS (SELECT 1 FROM memories g WHERE g.id = m.id)",
        "SELECT * FROM memories {} AND embedding IS NOT NULL",
        "DELETE FROM role_assignments{} RETURNING assignment_id",
        "SELECT * FROM pg_class",
        "SELECT a, max(b) FROM t WHERE k = $1 GROUP BY a",
    ],
)
def test_keyed_and_bounded_statements_pass(sql: str, schema: SchemaCatalog) -> None:
    """Flips if a read narrowed by a prefix of its key, a keyset page, a one-row probe, a correlated key, a CTE, a
    contradiction, an outer WHERE pushed into a derived table, the key generic code names, or a predicate the walk
    cannot see is called whole."""
    assert _kinds(sql, schema) == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE k > $1 ORDER BY k LIMIT $2",
        "SELECT a FROM results WHERE state = $1 AND geo > $2 ORDER BY geo LIMIT 100",
        "SELECT a FROM results WHERE state = $1 AND (geo, candidate) > ($2, $3) ORDER BY geo, candidate LIMIT 100",
    ],
)
def test_a_keyset_page_passes(sql: str, schema: SchemaCatalog) -> None:
    """Flips if an ordered key range under a LIMIT, after a bound key prefix, is called whole."""
    assert _kinds(sql, schema) == []


def test_a_key_prefix_passes_and_a_later_key_column_alone_does_not(schema: SchemaCatalog) -> None:
    """Flips if a read bound on the key's leading column is called whole, or one bound only on a later key column
    (no leading prefix) passes."""
    assert _kinds("SELECT a FROM results WHERE state = $1", schema) == []
    assert _kinds("SELECT a FROM results WHERE state = $1 AND geo = $2", schema) == []
    assert _kinds("SELECT a FROM results WHERE candidate = $1", schema) == [("non-key-where", "results")]


def test_a_purposed_index_keys_only_the_read_it_names(schema: SchemaCatalog) -> None:
    """Flips if an index keys a read in a function its purpose does not name, or fails to key the one it does, or
    if the use is not recorded (so an index whose purpose no read uses can be named)."""
    sql = "SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1"
    assert _kinds(sql, schema, "src/hub/other.py::elsewhere") == [("non-key-where", "datasources")]
    assert schema.used == set()
    assert _kinds(sql, schema, _DIGEST) == []
    assert schema.used == {("datasources", "datasources_by_kind")}


def test_without_a_schema_every_predicate_on_a_table_is_unknown() -> None:
    """Flips if a statement judged against no declarations passes on the strength of any bound value."""
    assert _kinds("SELECT a FROM t WHERE k = $1", SchemaCatalog()) == [("unknown-schema", "t")]


def test_a_bare_select_from_a_table_is_a_head_judged_where_it_runs(schema: SchemaCatalog) -> None:
    """Flips if a pager's ``SELECT ... FROM <table>`` head is judged as a whole read where it is written."""
    assert _kinds("SELECT {} FROM {}", schema) == [("head", "{}")]


@pytest.mark.parametrize(
    "text", ["Select a model from the list", "delete from another scope", "truncate failed", "analyze images"]
)
def test_prose_is_not_sql(text: str) -> None:
    """Flips if English that starts with a SQL verb is judged as a statement."""
    assert not is_sql(text)


_CLOSURE = """
WITH RECURSIVE reachable(dataset_id, path) AS (
    SELECT $1::uuid AS dataset_id, ARRAY[$1::uuid] AS path
  UNION ALL
    SELECT s.source_id, r.path || s.source_id
      FROM dataset_sources s
      JOIN reachable r ON s.dataset_id = r.dataset_id
     WHERE s.source_kind = 'dataset'
       AND NOT (s.source_id = ANY(r.path))
)
SELECT DISTINCT s.source_kind AS source_kind, s.source_id AS source_id
  FROM dataset_sources s
  JOIN reachable r ON s.dataset_id = r.dataset_id
"""


def test_a_join_to_a_recursive_cte_seeded_by_a_key_is_keyed(schema: SchemaCatalog) -> None:
    """Flips if a closure walked from one bound key, joined back to the table on column equality, is called whole."""
    assert _kinds(_CLOSURE, schema) == []


@pytest.mark.parametrize(
    "seed",
    [
        # no key at all: the CTE starts from every row
        "SELECT id AS dataset_id, ARRAY[id] AS path FROM roots",
        # a key in the select list of a whole read binds nothing
        "SELECT $1::uuid AS dataset_id, ARRAY[id] AS path FROM roots",
    ],
)
def test_a_cte_seeded_without_a_bound_key_leaves_the_join_whole(seed: str, schema: SchemaCatalog) -> None:
    """Flips if a join to a CTE counts as keyed when the CTE's seed is unbound, or is itself a whole read."""
    sql = _CLOSURE.replace("SELECT $1::uuid AS dataset_id, ARRAY[$1::uuid] AS path", seed)
    assert ("whole-aggregate", "dataset_sources") in _kinds(sql, schema)


def test_a_join_to_a_bound_cte_without_column_equality_is_still_whole(schema: SchemaCatalog) -> None:
    """Flips if naming a bound CTE in a clause, without joining the table to it by a column, keys the read."""
    sql = _CLOSURE.replace(
        "  FROM dataset_sources s\n  JOIN reachable r ON s.dataset_id = r.dataset_id\n",
        "  FROM dataset_sources s CROSS JOIN reachable r\n",
    )
    assert ("whole-aggregate", "dataset_sources") in _kinds(sql, schema)
