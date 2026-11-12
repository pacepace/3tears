"""The text half of the no-whole-reads rule: which statements read or write a table whole, judged against each
table's declared key and indexes."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads.schema import SchemaCatalog, collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, is_sql, judge_sql

#: the function the purposed index below names
_DIGEST = "src/hub/schema_digest.py::SchemaDigest.refresh_active"
#: the functions the namespaces' purposed indexes name
_TAGGED = "src/hub/namespaces.py::tagged"
_PLATFORM = "src/hub/namespaces.py::platform_rows"
_LOWER = "src/hub/namespaces.py::by_lower_kind"
_POD = "src/hub/namespaces.py::by_pod"
_MCP = "src/hub/removal.py::tool_rows_by_mcp_name"
_SPACED = "src/hub/namespaces.py::by_spaced"
_BY_NAME = "src/hub/providers.py::ProviderCollection.get_by_name"
_PAGED = "src/hub/grants.py::page_by_state"
_NEWEST = "src/hub/results.py::newest_in_state"
_UNDER = "src/hub/paths.py::under"

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
CREATE TABLE api_keys (customer_id UUID, id UUID, key_hash TEXT UNIQUE, PRIMARY KEY (customer_id, id));
CREATE TABLE grants (customer_id UUID, id UUID, name TEXT, state TEXT, PRIMARY KEY (customer_id, id),
    CONSTRAINT grants_name_uq UNIQUE (customer_id, name));
CREATE UNIQUE INDEX grants_live ON grants (state) WHERE state = 'live';
-- purpose: {_NEWEST}
CREATE INDEX results_by_state_votes ON results (state, winner);
-- purpose: {_PAGED}
CREATE INDEX grants_by_state ON grants (customer_id, state);
ALTER TABLE gateway_providers ADD CONSTRAINT gateway_providers_name_uq UNIQUE (provider_name);
CREATE TABLE namespaces (row_scope TEXT, namespace_id UUID, kind TEXT, tags JSONB, customer_id UUID,
    PRIMARY KEY (row_scope, namespace_id));
-- purpose: {_TAGGED}
CREATE INDEX namespaces_by_tags ON namespaces USING gin (tags);
-- purpose: {_PLATFORM}
CREATE INDEX namespaces_by_customer ON namespaces (customer_id);
CREATE TABLE singles (customer_id UUID PRIMARY KEY, n INT);
CREATE UNIQUE INDEX namespaces_segment ON namespaces ((replace(name, '.', '-')));
-- purpose: {_LOWER}
CREATE INDEX namespaces_lower_kind ON namespaces ((lower(kind)));
-- purpose: {_POD}
CREATE INDEX namespaces_by_pod ON namespaces ((tags->>'pod'));
-- purpose: {_MCP}
CREATE INDEX namespaces_tool_mcp_name ON namespaces ((tags ->> 'mcp_name')) WHERE kind = 'tool';
-- purpose: {_SPACED}
CREATE INDEX namespaces_by_spaced ON namespaces ((tags #>> '{{a,b}}'), customer_id);
CREATE TABLE providers (id UUID PRIMARY KEY, name TEXT);
-- purpose: {_BY_NAME}
CREATE UNIQUE INDEX providers_by_name ON providers (name);
CREATE TABLE paths (id UUID PRIMARY KEY, name TEXT, owner UUID);
-- purpose: {_UNDER}
CREATE INDEX paths_name_c ON paths ((name COLLATE "C"));
-- purpose: {_UNDER}
CREATE INDEX paths_owner_name ON paths (owner, name);
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
        # a view reads nothing when created, but is declared with the keys a read of it uses
        ("CREATE VIEW v AS SELECT * FROM t", [("undeclared-view", "v")]),
        # a predicate assembled at run time that the walk cannot see is never a pass
        ("SELECT * FROM memories {} AND embedding IS NOT NULL", [("unknown-predicate", "memories")]),
        ("DELETE FROM role_assignments{} RETURNING assignment_id", [("unknown-predicate", "role_assignments")]),
        ("SELECT count(*) FROM t WHERE {}", [("unknown-predicate", "t")]),
        ("UPDATE stamps SET v = $1 WHERE {}", [("unknown-predicate", "stamps")]),
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
        # a bounded probe reads at most its bound count of rows, of any relation
        "SELECT a FROM nowhere LIMIT 1",
        "SELECT * FROM t LIMIT $1",
        # interpolated relations too (hub table_rows.rows_up_to)
        "SELECT 1 FROM {schema}.{table} LIMIT $1",
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
        "SELECT -1 AS ask, state, count(*) AS n FROM {} WHERE 1 = 0 GROUP BY state",
        "SELECT * FROM (SELECT a, k FROM t) AS d WHERE d.k = $1",
        "SELECT m.a FROM memories m WHERE m.id = $1 AND NOT EXISTS (SELECT 1 FROM memories g WHERE g.id = m.id)",
        # what the walk can see keys it, whatever the interpolation adds; generic code names its key predicate
        # a shape probe reads no row
        "SELECT * FROM t LIMIT 0",
        "SELECT a, flag FROM t WHERE flag = $1 LIMIT 0",
        "SELECT a FROM t WHERE k = $1 AND {}",
        "SELECT a FROM {} WHERE {pk}",
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
    assert schema.used == {("datasources", "datasources_by_kind", _DIGEST)}


def test_an_or_whose_every_branch_keys_the_read_is_keyed_and_credits_each_index(schema: SchemaCatalog) -> None:
    """Flips if an OR of branches each keyed by its own index or the key is refused, if its purposed index is not
    recorded as used, or if an OR with one branch that keys nothing (or keys through an index another function's
    purpose names) passes."""
    keyed = "SELECT id, kind FROM datasources WHERE id = ANY($1::uuid[]) OR kind = $2"
    assert _kinds(keyed, schema, _DIGEST) == []
    assert ("datasources", "datasources_by_kind", _DIGEST) in schema.used
    mixed = "SELECT id, kind FROM datasources WHERE id = ANY($1::uuid[]) OR status = $2"
    assert _kinds(mixed, schema, _DIGEST) == [("non-key-where", "datasources")]
    assert _kinds(keyed, schema, "src/hub/other.py::elsewhere") == [("non-key-where", "datasources")]


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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM api_keys WHERE key_hash = $1",
        "SELECT * FROM grants WHERE customer_id = $1 AND name = $2",
        "SELECT id FROM gateway_providers WHERE provider_name = $1",
        "SELECT * FROM api_keys WHERE customer_id = $1 AND id = $2",
    ],
)
def test_a_declared_unique_is_a_key(sql: str, schema: SchemaCatalog) -> None:
    """Flips if a read by a column declared UNIQUE (inline, a table constraint, ALTER TABLE ... ADD UNIQUE) is
    called whole."""
    assert _kinds(sql, schema) == []


@pytest.mark.parametrize(
    ("sql", "table"),
    [
        # the primary key leads with the tenant: by id alone, YugabyteDB reads every tablet
        ("SELECT * FROM api_keys WHERE id = $1", "api_keys"),
        # a unique key leading with the tenant, read without it
        ("SELECT * FROM grants WHERE name = $1", "grants"),
        # a partial unique index holds only for the rows its WHERE names: no key
        ("SELECT * FROM grants WHERE state = $1", "grants"),
    ],
)
def test_a_read_missing_the_leading_tenant_or_by_a_partial_unique_is_whole(
    sql: str, table: str, schema: SchemaCatalog
) -> None:
    """Flips if a read by id on a tenant-leading key, by a unique key without its tenant, or by a partial unique
    index passes."""
    assert _kinds(sql, schema) == [("non-key-where", table)]


@pytest.mark.parametrize(
    "sql",
    [
        # the tenant alone is every row of the tenant
        "SELECT namespace_id FROM namespaces WHERE row_scope = $1",
        # every value of a low-cardinality column is every row
        "SELECT namespace_id FROM namespaces WHERE row_scope IN ('platform', 'customer')",
        "SELECT * FROM namespaces WHERE kind = ANY(ARRAY['agent', 'tool'])",
    ],
)
def test_a_tenant_only_prefix_or_every_value_of_a_column_is_whole(sql: str, schema: SchemaCatalog) -> None:
    """Flips if a key prefix covering only the tenant, or an IN listing fixed categories, passes as keyed."""
    assert _kinds(sql, schema) == [("non-key-where", "namespaces")]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT namespace_id FROM namespaces WHERE row_scope = $1 AND namespace_id = $2",
        "SELECT namespace_id FROM namespaces WHERE row_scope = $1 AND namespace_id > $2 ORDER BY namespace_id LIMIT 100",
        "SELECT namespace_id FROM namespaces WHERE row_scope = $1 ORDER BY namespace_id LIMIT 100",
        # a key that is only the tenant column, bound whole, is one row
        "SELECT n FROM singles WHERE customer_id = $1",
        # unnested bound arrays covering the key, joined or as a subquery
        "SELECT n.* FROM namespaces n JOIN unnest($1::text[], $2::uuid[]) AS k(s, i) "
        "ON n.row_scope = k.s AND n.namespace_id = k.i",
        "SELECT * FROM namespaces WHERE (row_scope, namespace_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))",
    ],
)
def test_past_the_tenant_a_keyset_page_or_unnested_keys_are_keyed(sql: str, schema: SchemaCatalog) -> None:
    """Flips if a key prefix reaching past the tenant, a keyset page within a tenant, a whole key that is the
    tenant, or a join to unnested bound keys is called whole."""
    assert _kinds(sql, schema) == []


@pytest.mark.parametrize(
    ("sql", "purpose"),
    [
        ("SELECT * FROM namespaces WHERE tags @> $1", _TAGGED),
        ("SELECT * FROM namespaces WHERE customer_id IS NULL", _PLATFORM),
    ],
)
def test_containment_and_is_null_key_only_through_their_purposed_index(
    sql: str, purpose: str, schema: SchemaCatalog
) -> None:
    """Flips if ``@>`` or ``IS NULL`` keys a read outside the function its index's purpose names, or fails to key
    the one it names."""
    assert _kinds(sql, schema, purpose) == []
    elsewhere = _kinds(sql, schema, "src/hub/other.py::elsewhere")
    # IS NULL binds no value, so outside its index it is a WHERE with nothing bound
    assert elsewhere == [("unbound-where" if "IS NULL" in sql else "non-key-where", "namespaces")]


@pytest.mark.parametrize("sql", ["SELECT * FROM t WHERE k @> $1", "SELECT * FROM t WHERE k IS NULL"])
def test_containment_and_is_null_are_never_a_key(sql: str, schema: SchemaCatalog) -> None:
    """Flips if ``@>`` or ``IS NULL`` on a key column counts as binding the key (a)."""
    assert _kinds(sql, schema)[0][1] == "t"


def test_an_expression_index_keys_a_read_of_the_same_expression(schema: SchemaCatalog) -> None:
    """Flips if a read bound on the expression a unique expression index holds (a), or one a purposed expression
    index holds in the function it names (b), is called whole, or if another expression of the column passes."""
    assert _kinds("SELECT * FROM namespaces WHERE replace(name, '.', '-') = ANY($1)", schema) == []
    assert _kinds("SELECT * FROM namespaces WHERE lower(kind) = $1", schema, _LOWER) == []
    assert _kinds("SELECT * FROM namespaces WHERE lower(kind) = $1", schema) == [("non-key-where", "namespaces")]
    assert _kinds("SELECT * FROM namespaces WHERE upper(name) = $1", schema) == [("non-key-where", "namespaces")]


def test_a_json_field_index_keys_a_read_of_the_same_field(schema: SchemaCatalog) -> None:
    """Flips if a read bound on the json field a purposed expression index holds is called whole in the function it
    names, or passes elsewhere, or a json operator is taken for the comparison. (A field's name is a quoted literal,
    which the judge reads as ``'_'`` as it reads every literal: the purpose names the one read.)"""
    assert _kinds("SELECT * FROM namespaces WHERE tags->>'pod' = $1", schema, _POD) == []
    assert _kinds("SELECT * FROM namespaces WHERE tags ->> 'pod' = $1", schema, _POD) == []
    assert _kinds("SELECT * FROM namespaces WHERE tags->>'pod' = $1", schema) == [("non-key-where", "namespaces")]
    assert _kinds("SELECT * FROM namespaces WHERE tags->'pod' = $1", schema, _POD) == [("non-key-where", "namespaces")]


def test_a_unique_index_keying_a_read_is_used_only_by_the_read_its_purpose_names(schema: SchemaCatalog) -> None:
    """Flips if a read through a unique index is not recorded as its use when the index's purpose names the reader,
    or is recorded when it does not (the key still keys both reads)."""
    sql = "SELECT * FROM providers WHERE name = $1"
    assert _kinds(sql, schema, "src/hub/other.py::elsewhere") == []
    assert schema.used == set()
    assert _kinds(sql, schema, _BY_NAME) == []
    assert schema.used == {("providers", "providers_by_name", _BY_NAME)}


_SUPERSEDED = (
    "SELECT id, (SELECT g.a FROM memories g WHERE g.id = memories.superseded_by) AS s FROM memories WHERE id = $1"
)


def test_an_aliased_relation_answers_only_to_its_alias(schema: SchemaCatalog) -> None:
    """Flips if, in a subquery over the same table aliased, the outer table's name binds the inner alias (so the
    correlated key is lost), or if a qualifier naming nothing at a level keys it."""
    assert _kinds(_SUPERSEDED, schema) == []
    unkeyed = _SUPERSEDED.replace("g.id = memories.superseded_by", "g.a = memories.a")
    # the outer row's non-key column binds no value of the inner read's key
    assert _kinds(unkeyed, schema) == [("unbound-where", "memories")]


@pytest.mark.parametrize(
    ("sql", "keyed"),
    [
        ("SELECT r.* FROM results r JOIN unnest($1::text[]) AS k(s) ON r.state = k.s", True),
        (
            "SELECT r.* FROM results r JOIN unnest($1::text[], $2::text[]) AS k(s, g) ON r.state = k.s AND r.geo = k.g",
            True,
        ),
        ("SELECT r.* FROM results r JOIN unnest($1::text[]) AS k(g) ON r.geo = k.g", False),
    ],
)
def test_an_unnest_join_keys_a_leading_prefix_of_the_key(sql: str, keyed: bool, schema: SchemaCatalog) -> None:
    """Flips if a join to unnested bound keys covering a leading prefix of the key is called whole, or one binding
    only a later key column passes."""
    assert (_kinds(sql, schema) == []) is keyed, _kinds(sql, schema)


@pytest.mark.parametrize(
    ("order", "keyed"),
    [("state", True), ("name", False)],
)
def test_a_page_through_an_index_orders_by_its_next_column(order: str, keyed: bool, schema: SchemaCatalog) -> None:
    """Flips if a page ranging on an index's next column is called whole when it orders by that column, or passes
    when it orders by another (sorting the tenant's every row)."""
    sql = f"SELECT * FROM grants WHERE customer_id = $1 AND state > $2 ORDER BY {order} LIMIT 5"
    assert (_kinds(sql, schema, _PAGED) == []) is keyed, _kinds(sql, schema, _PAGED)


def test_a_dollar_interpolation_is_a_bind_placeholder(schema: SchemaCatalog) -> None:
    """Flips if ``${n}`` (a bind placeholder rendered at run time) is unseen text rather than a bound value, or if a
    bare interpolated predicate stops being unseen."""
    assert _kinds("SELECT a FROM t WHERE k = ${n}", schema) == []
    assert _kinds("SELECT a FROM t WHERE flag = ${n}", schema) == [("non-key-where", "t")]
    assert _kinds("SELECT a FROM t WHERE {x}", schema) == [("unknown-predicate", "t")]


def test_a_purposed_index_serving_a_key_read_is_recorded_as_used(schema: SchemaCatalog) -> None:
    """Flips if a read keyed by the key's lead (a) does not record the purposed index whose leading columns are that
    lead then the column it orders by, or records it for a read ordered by another column."""
    _kinds("SELECT a FROM results WHERE state = $1 ORDER BY votes LIMIT 5", schema, _NEWEST)
    assert ("results", "results_by_state_votes", _NEWEST) not in schema.used
    assert _kinds("SELECT a FROM results WHERE state = $1 ORDER BY winner LIMIT 5", schema, _NEWEST) == []
    assert ("results", "results_by_state_votes", _NEWEST) in schema.used


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # an L1 cache's trim to its cap: rowid is the key its b-tree is stored by
        ("DELETE FROM t WHERE rowid IN (SELECT rowid FROM t ORDER BY rowid LIMIT $1)", []),
        ("SELECT rowid FROM t ORDER BY rowid DESC LIMIT 1000 OFFSET 5000", []),
        # ordered by anything else, still every row sorted
        ("SELECT rowid FROM t ORDER BY created LIMIT 5", [("whole-select", "t")]),
        # a bounded probe of a table's first rows
        ("SELECT count(*) FROM (SELECT 1 FROM t LIMIT $1) AS held", []),
    ],
)
def test_a_sqlite_rowid_is_a_key_and_a_bound_probe_passes(
    sql: str, expected: list[tuple[str, str]], schema: SchemaCatalog
) -> None:
    """Flips if a rowid-ordered cap trim is called a whole read, or a non-key order under a LIMIT passes."""
    assert _kinds(sql, schema) == expected


@pytest.mark.parametrize("order", ["ASC", "DESC", "ASC NULLS LAST"])
def test_a_collated_index_item_with_an_order_is_still_that_collation_s_column(tmp_path: Path, order: str) -> None:
    """Flips if an ordered collated index item (``(name COLLATE "C") ASC``, how YugabyteDB is asked for a range
    index) is not read as the column in that collation, so the byte range beneath a node it serves is flagged."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "v001.sql").write_text(
        "CREATE TABLE paths (id UUID PRIMARY KEY, name TEXT);\n"
        f"-- purpose: {_UNDER}\n"
        f'CREATE INDEX paths_name_c ON paths ((name COLLATE "C") {order});\n',
        encoding="utf-8",
    )
    catalog = collect_schema(tmp_path, (tmp_path / "src",))
    either = 'SELECT id FROM paths WHERE id = $1 OR (name COLLATE "C" >= $2 AND name COLLATE "C" < $3)'
    assert _kinds(either, catalog, _UNDER) == []
    assert ("paths", "paths_name_c", _UNDER) in catalog.used


def test_a_range_bounded_both_sides_on_a_purposed_index_is_its_key(schema: SchemaCatalog) -> None:
    """Flips if a range bounded on both sides by bound values, on the next column of an index whose purpose names the
    read (after its bound prefix), is not keyed and credited to that index, or if a one-sided range, one bounded by a
    literal, or one in another collation than the index's is."""
    collated = 'SELECT id FROM paths WHERE name COLLATE "C" >= $1 AND name COLLATE "C" < $2'
    assert _kinds(collated, schema, _UNDER) == []
    assert ("paths", "paths_name_c", _UNDER) in schema.used
    assert _kinds(collated, schema) == [("non-key-where", "paths")], "only the read the purpose names"
    assert _kinds("SELECT id FROM paths WHERE owner = $1 AND name >= $2 AND name < $3", schema, _UNDER) == []
    assert _kinds("SELECT id FROM paths WHERE owner = $1 AND name BETWEEN $2 AND $3", schema, _UNDER) == []
    either = 'SELECT id FROM paths WHERE id = $1 OR (name COLLATE "C" >= $2 AND name COLLATE "C" < $3)'
    assert _kinds(either, schema, _UNDER) == []
    for unkeyed in (
        'SELECT id FROM paths WHERE name COLLATE "C" >= $1',
        """SELECT id FROM paths WHERE name COLLATE "C" >= '' AND name COLLATE "C" < $1""",
        "SELECT id FROM paths WHERE name >= $1 AND name < $2",
        'SELECT id FROM paths WHERE name COLLATE "POSIX" >= $1 AND name COLLATE "POSIX" < $2',
    ):
        before = set(schema.used)
        assert _kinds(unkeyed, schema, _UNDER) == [("non-key-where", "paths")], unkeyed
        assert not {(t, i) for t, i, _ in schema.used - before} & {("paths", "paths_name_c")}, unkeyed


def test_a_json_field_index_written_with_spaces_is_the_field_not_its_column(schema: SchemaCatalog) -> None:
    """Flips if an expression index written with spaces (``(tags ->> 'mcp_name')``, hub v152) is catalogued under its
    bare column, so the read of the field it holds (by ``= ANY``, the hub's removal read) is not keyed by it in the
    function it names; or if it then keys a read of the bare column, or a read elsewhere."""
    field = "SELECT * FROM namespaces WHERE kind = 'tool' AND tags ->> 'mcp_name' = ANY($1::text[])"
    assert _kinds(field, schema, _MCP) == []
    assert _kinds("SELECT * FROM namespaces WHERE tags->>'mcp_name' = $1 AND kind = 'tool'", schema, _MCP) == []
    assert _kinds(field, schema) == [("non-key-where", "namespaces")]
    assert _kinds("SELECT * FROM namespaces WHERE tags = $1", schema, _MCP) == [("non-key-where", "namespaces")]
    # a path expression leading a two-column index: the expression, then its next column
    spaced = "SELECT * FROM namespaces WHERE tags #>> '{a,b}' = $1 AND customer_id = $2"
    assert _kinds(spaced, schema, _SPACED) == []
    assert _kinds("SELECT * FROM namespaces WHERE tags = $1 AND customer_id = $2", schema, _SPACED) == [
        ("non-key-where", "namespaces")
    ]


#: hub #38: a datasource's readiness counts, each a correlated count of a table by the outer datasource row
_READINESS = "src/hub/scope_injection.py::_knowledge_readiness_template"
_COUNTED = (
    "SELECT d.name, (SELECT COUNT(*) FROM {inner} AS pe WHERE pe.status = 'active' AND {match}) AS active "
    "FROM datasources AS d{outer}"
)
_EITHER = "(pe.datasource_id = d.id OR pe.datasource_id = d.origin_datasource_id)"


@pytest.mark.parametrize(
    ("inner", "match", "outer", "expected"),
    [
        # the outer datasources read by key: each count reads by that row's datasource, through its purposed index
        ("playbook_entries", _EITHER, " WHERE d.id = ANY($3)", []),
        ("playbook_entries", "pe.datasource_id = d.id", " WHERE d.id = ANY($3)", []),
        # no index on the inner column: still every active row counted for each outer row
        ("concepts", _EITHER, " WHERE d.id = ANY($3)", [("whole-aggregate", "concepts")]),
        ("concepts", "pe.datasource_id = d.id", " WHERE d.id = ANY($3)", [("whole-aggregate", "concepts")]),
        # the outer row read whole, or by a non-key filter: its values are every datasource, so is the count
        ("playbook_entries", _EITHER, "", [("whole-select", "datasources"), ("whole-aggregate", "playbook_entries")]),
        (
            "playbook_entries",
            "pe.datasource_id = d.id",
            "",
            [("whole-select", "datasources"), ("whole-aggregate", "playbook_entries")],
        ),
        (
            "playbook_entries",
            _EITHER,
            " WHERE d.name = $1",
            [("non-key-where", "datasources"), ("whole-aggregate", "playbook_entries")],
        ),
        # a branch equal to the inner row's own column binds nothing
        (
            "playbook_entries",
            "(pe.datasource_id = d.id OR pe.datasource_id = pe.id)",
            " WHERE d.id = ANY($3)",
            [("whole-aggregate", "playbook_entries")],
        ),
    ],
)
def test_a_correlated_equality_binds_only_through_a_keyed_outer_row(
    tmp_path: Path, inner: str, match: str, outer: str, expected: list[tuple[str, str]]
) -> None:
    """Flips if a correlated count bound by an outer row read by key (each ``OR`` branch one outer column) is not
    keyed through the inner table's purposed index, or if a correlated equality to an outer row read whole, or one on
    an inner column no index serves, stops being a finding."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "v001.sql").write_text(
        "CREATE TABLE datasources (id UUID PRIMARY KEY, name TEXT, origin_datasource_id UUID);\n"
        "CREATE TABLE playbook_entries (id UUID PRIMARY KEY, datasource_id UUID, status TEXT);\n"
        f"-- purpose: {_READINESS}\n"
        "CREATE INDEX playbook_entries_active_by_datasource ON playbook_entries (datasource_id)"
        " WHERE status = 'active';\n"
        "CREATE TABLE concepts (id UUID PRIMARY KEY, datasource_id UUID, status TEXT);\n",
        encoding="utf-8",
    )
    catalog = collect_schema(tmp_path, (tmp_path / "src",))
    sql = _COUNTED.format(inner=inner, match=match, outer=outer)
    found = [kind for kind in _kinds(sql, catalog, _READINESS) if kind[0] != "keyed-count"]
    assert found == expected
    if not expected:
        assert ("playbook_entries", "playbook_entries_active_by_datasource", _READINESS) in catalog.used
