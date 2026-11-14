"""the use-the-helper rewrite (:mod:`~threetears.enforcement.no_whole_reads.rewrite`): a plain point read is rewritten
onto its helper, recorded in the manifest, and passes both rules after; anything not certain is left and listed.

Each end-to-end test synthesises a small repo, plans and applies the rewrite with the gate's own walk, and walks the
result again.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

from threetears.core.testing.rewrites import assert_source_says
from threetears.enforcement.no_whole_reads import (
    NoWholeReadsConfig,
    no_whole_reads_findings,
    use_the_helper_findings,
)
from threetears.enforcement.no_whole_reads.rewrite import (
    Pin,
    PointSql,
    WriteSql,
    apply_rewrites,
    parse_count,
    parse_first,
    parse_join,
    parse_list,
    parse_point,
    parse_write,
    plan_count_rewrites,
    plan_first_rewrites,
    plan_join_rewrites,
    plan_list_rewrites,
    plan_point_rewrites,
    plan_write_rewrites,
    render_test_module,
    write_manifest,
)

_TABLES = """
    from threetears.core.collections.schema_backed import TableSchema

    AGENTS_SCHEMA = TableSchema(name="agents", primary_key="id", columns=[])
    GEOS_SCHEMA = TableSchema(name="geos", primary_key=("state", "geo_id"), columns=[])
"""

_READS = """
    import asyncpg

    #: a geo by its key
    _GEO_SQL = "SELECT geo_id, name FROM geos WHERE state = $1 AND geo_id = $2"


    async def one_agent(conn, agent_id):
        return await conn.fetchrow("SELECT * FROM agents WHERE id = $1 AND is_deleted = FALSE", agent_id)


    async def agent_name(pool, agent_id):
        return await pool.fetchval("SELECT name, slug FROM agents WHERE id = $1", agent_id)


    async def a_geo(conn, state, geo):
        return await conn.fetch(_GEO_SQL, state, geo)


    async def another_geo(conn, state, geo):
        return await conn.fetch(_GEO_SQL, state, geo)


    async def by_status(conn, status):
        return await conn.fetch("SELECT id FROM agents WHERE status = $1", status)


    async def by_ids(conn, ids):
        return await conn.fetch("SELECT id FROM agents WHERE id = ANY($1::uuid[])", ids)


    async def interpolated(conn, table, agent_id):
        return await conn.fetchrow(f"SELECT * FROM {table} WHERE id = $1", agent_id)


    async def live_one(conn, agent_id):
        return await conn.fetchrow("SELECT id FROM agents WHERE id = $1 AND live AND NOT gone", agent_id)


    async def scoped_agent(conn, agent_id, realm):
        return await conn.fetchrow(
            "SELECT * FROM agents WHERE scope IN ('platform', 'customer') AND id = $1 AND realm IS NOT DISTINCT FROM $2",
            agent_id,
            realm,
        )
"""


def _unlined(findings: list[str]) -> list[str]:
    """findings without their line numbers, which a rewrite above them moves."""
    return sorted(re.sub(r"^([^:]+):\d+ ", r"\1 ", finding) for finding in findings)


def _config(tmp_path: Path) -> NoWholeReadsConfig:
    src = tmp_path / "src"
    for name, source in {"pkg/tables.py": _TABLES, "pkg/reads.py": _READS}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM agents WHERE id = $1", PointSql("agents", None, ())),
        ("SELECT a.id, name FROM agents a WHERE a.id = $1 AND a.flag = TRUE", PointSql("agents", ("id", "name"), ())),
        ("SELECT id FROM platform.agents WHERE id = $1 AND realm IS NULL AND kind = 'x''y'", None),
    ],
)
def test_a_plain_point_read_is_parsed(sql: str, expected: PointSql | None) -> None:
    """Flips if a plain equality read stops parsing, or its table or columns come out wrong."""
    parsed = parse_point(sql)
    assert isinstance(parsed, PointSql), parsed
    if expected is not None:
        assert (parsed.table, parsed.columns) == (expected.table, expected.columns)
    assert parsed.pins[0].kind == "param" and parsed.pins[0].param == 1


def test_any_and_boolean_column_pins_are_read() -> None:
    """Flips if ``= ANY($n::type[])`` is not read as a set pin with its cast (none when the array is untyped), or a
    bare boolean column is not read as pinned TRUE and a NOT one as pinned FALSE."""
    parsed = parse_point("SELECT id FROM t WHERE a = ANY($1::UUID[]) AND b = any($2) AND live AND NOT t.gone")
    assert isinstance(parsed, PointSql)
    assert [(pin.column, pin.kind, pin.param, pin.literal, pin.cast) for pin in parsed.pins] == [
        ("a", "any", 1, None, "uuid"),
        ("b", "any", 2, None, None),
        ("live", "literal", None, True, None),
        ("gone", "literal", None, False, None),
    ]
    written = parse_write("DELETE FROM t WHERE k = $1 AND id = ANY($2::bigint[]) AND protected")
    assert isinstance(written, WriteSql)
    assert written.pins == (
        Pin("k", "param", param=1),
        Pin("id", "any", param=2, cast="bigint"),
        Pin("protected", "literal", literal=True),
    )


def test_literal_and_null_pins_keep_their_values() -> None:
    """Flips if a literal pin loses its value or type, or IS NULL is read as anything else."""
    parsed = parse_point(
        "SELECT id FROM t WHERE a = $2 AND b = FALSE AND c = 'it''s' AND d = -3 AND e IS NULL AND f = $1"
    )
    assert isinstance(parsed, PointSql)
    assert [(pin.column, pin.kind, pin.param, pin.literal) for pin in parsed.pins] == [
        ("a", "param", 2, None),
        ("b", "literal", None, False),
        ("c", "literal", None, "it's"),
        ("d", "literal", None, -3),
        ("e", "null", None, None),
        ("f", "param", 1, None),
    ]


def test_null_safe_and_literal_list_pins_are_read() -> None:
    """Flips if ``IS NOT DISTINCT FROM $n`` is not read as a null-safe pin of its parameter (sent ``= $n``, or
    ``IS NULL`` for None), or ``IN (<literals>)`` not as the literals' set, in a read and in a write."""
    parsed = parse_point(
        "SELECT * FROM t WHERE row_scope IN ('platform', 'customer') AND a = $1 AND b IS NOT DISTINCT FROM $2"
        " AND n IN (1, -2)"
    )
    assert isinstance(parsed, PointSql)
    assert [(pin.column, pin.kind, pin.param, pin.literal) for pin in parsed.pins] == [
        ("row_scope", "anylit", None, ("platform", "customer")),
        ("a", "param", 1, None),
        ("b", "nullsafe", 2, None),
        ("n", "anylit", None, (1, -2)),
    ]
    written = parse_write("DELETE FROM t WHERE k = $1 AND scope IS NOT DISTINCT FROM $2 AND kind IN ('x''y')")
    assert isinstance(written, WriteSql)
    assert written.pins == (
        Pin("k", "param", param=1),
        Pin("scope", "nullsafe", param=2),
        Pin("kind", "anylit", literal=("x'y",)),
    )
    # literals only: no bound parameter is still a constant read, judged by hand
    assert isinstance(parse_point("SELECT * FROM t WHERE kind IN ('a', 'b')"), str)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM t WHERE id = ANY($1::uuid[]) OR a = $2",
        "SELECT id FROM t WHERE a IN ($1, 'b')",
        "SELECT id FROM t WHERE a IN ('b', lower($1))",
        "SELECT id FROM t WHERE a IS DISTINCT FROM $1",
        "SELECT id FROM t WHERE a IS NOT DISTINCT FROM b",
        "SELECT id FROM t WHERE id = ANY(ARRAY[$1])",
        "SELECT id FROM t WHERE id = ANY($1::uuid[] || $2)",
        "SELECT id FROM t WHERE id <> ALL($1::uuid[])",
        "SELECT id FROM t WHERE a = $1 AND NOT (b)",
        "SELECT id FROM t WHERE a = $1 AND true",
        "SELECT id FROM t WHERE a = $1 AND NOT NULL",
        "SELECT id FROM t WHERE id IN ($1, $2)",
        "SELECT id FROM t WHERE a = $1 OR b = $1",
        "SELECT id FROM t WHERE a > $1",
        "SELECT id FROM t WHERE a = $1::uuid",
        "SELECT id FROM t WHERE lower(a) = $1",
        "SELECT id FROM t WHERE a = $1 ORDER BY id",
        "SELECT id FROM t WHERE a = $1 LIMIT 1",
        "SELECT count(*) FROM t WHERE a = $1",
        "SELECT DISTINCT a FROM t WHERE a = $1",
        "SELECT * FROM {} WHERE a = $1",
        "SELECT id FROM t WHERE a = 'x'",
        "SELECT id FROM t WHERE a = $2",
        "SELECT t.id FROM t JOIN u ON u.id = t.id WHERE t.a = $1",
        "SELECT id FROM t WHERE a IS NOT NULL AND b = $1",
    ],
)
def test_anything_but_plain_equality_is_refused(sql: str) -> None:
    """Flips if a statement the helper cannot send as written is taken for a plain point read."""
    assert isinstance(parse_point(sql), str)


def test_point_reads_are_rewritten_recorded_and_pass_both_rules(tmp_path: Path) -> None:
    """Flips if a plain point read is not rewritten onto its helper exactly, the manifest does not record it, or the
    rewritten module still holds a use-the-helper finding for it or a new whole-read finding."""
    config = _config(tmp_path)
    before = no_whole_reads_findings(config)
    rewrites, refusals = plan_point_rewrites(config)
    assert sorted((rewrite.function, rewrite.helper) for rewrite in rewrites) == [
        ("a_geo", "read_by"),
        ("agent_name", "read_value"),
        ("another_geo", "read_by"),
        ("by_ids", "read_by"),
        ("live_one", "read_row"),
        ("one_agent", "read_row"),
        ("scoped_agent", "read_row"),
    ]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() == {"by_status", "interpolated"}, refused
    assert "unkeyed" in refused["by_status"]

    assert apply_rewrites(tmp_path, rewrites, ruff=None) == ["src/pkg/reads.py"]
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    write_manifest(manifest, rewrites)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(entries) == 7

    source = (tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "from threetears.core.keyed import AnyOf, read_by, read_row, read_value" in source
    assert "read_by(conn, 'agents', by={'id': AnyOf(ids, cast='uuid')}, columns=('id',), records=True)" in source
    assert (
        "read_row(conn, 'agents', by={'id': agent_id, 'live': True, 'gone': False}, columns=('id',), records=True)"
        in source
    )
    assert "_GEO_SQL" not in source and "a geo by its key" not in source
    assert (
        "read_row(conn, 'agents', by={'scope': AnyOf(('platform', 'customer'), cast=None), 'id': agent_id,"
        " 'realm': realm}, records=True)" in " ".join(source.split()).replace("( ", "(").replace(" )", ")")
    )
    assert "read_row(conn, 'agents', by={'id': agent_id, 'is_deleted': False}, records=True)" in source
    assert "read_value(pool, 'agents', by={'id': agent_id}, column='name')" in source
    assert (
        "read_by(conn, 'geos', by={'state': state, 'geo_id': geo}, columns=('geo_id', 'name'), records=True)" in source
    )
    assert tree is not None
    for entry in entries:
        assert_source_says(tmp_path, entry)

    left = [finding for finding in use_the_helper_findings(config) if "use-the-helper point" in finding]
    assert {finding.split(" in ")[1].split(":")[0] for finding in left} == {"by_status", "interpolated"}
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


_LISTINGS = """
    async def geos_of(conn, state):
        return await conn.fetch(
            "SELECT geo_id, name FROM geos WHERE state = $1 ORDER BY name DESC NULLS LAST, geo_id", state
        )


    async def geos_unordered(conn, state):
        return await conn.fetch("SELECT * FROM geos WHERE state = $1 ORDER BY geo_id ASC", state)


    async def first_geo(conn, state):
        return await conn.fetchrow("SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id", state)


    async def geo_page(conn, state, n):
        return await conn.fetch("SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id LIMIT $2", state, n)


    async def by_hidden(conn, state):
        return await conn.fetch("SELECT name FROM geos WHERE state = $1 ORDER BY geo_id", state)


    async def by_lower(conn, state):
        return await conn.fetch("SELECT geo_id, name FROM geos WHERE state = $1 ORDER BY lower(name)", state)
"""


def test_a_listing_s_order_is_parsed_term_for_term() -> None:
    """Flips if a listing's ``ORDER BY`` is not read as its plain-column terms (direction, NULLs as written), or a
    point read starts taking one."""
    parsed = parse_list("SELECT g.geo_id, name FROM geos g WHERE g.state = $1 ORDER BY g.name DESC NULLS LAST, geo_id")
    assert isinstance(parsed, PointSql), parsed
    assert parsed.order == (
        {"column": "name", "descending": True, "nulls": "last"},
        {"column": "geo_id", "descending": False, "nulls": None},
    )
    assert [(pin.column, pin.param) for pin in parsed.pins] == [("state", 1)]
    unordered = parse_list("SELECT * FROM geos WHERE state = $1")
    assert isinstance(unordered, PointSql) and unordered.order == ()
    assert isinstance(parse_point("SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id"), str)
    for refused in (
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id LIMIT 5",
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id OFFSET 5",
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY lower(geo_id)",
        "SELECT name FROM geos WHERE state = $1 ORDER BY geo_id",
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY u.geo_id",
    ):
        assert isinstance(parse_list(refused), str), refused


def test_listings_are_rewritten_onto_read_by_in_their_order(tmp_path: Path) -> None:
    """Flips if a plain listing is not rewritten onto ``read_by(order=[Sort(...)])`` term for term and recorded, or a
    page, a first-of read, an expression order or an unselected order column is not left and listed."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_LISTINGS), encoding="utf-8")
    before = no_whole_reads_findings(config)
    rewrites, refusals = plan_list_rewrites(config)
    assert sorted(rewrite.function for rewrite in rewrites) == ["geos_of", "geos_unordered"]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() == {"first_geo", "geo_page", "by_hidden", "by_lower"}, refused
    assert "first-of" in refused["first_geo"]

    apply_rewrites(tmp_path, rewrites, ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    assert {entry["shape"] for entry in entries} == {"list"}
    source = " ".join((tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8").split())
    assert "from threetears.core.keyed import Sort, read_by" in source
    assert (
        "read_by(conn, 'geos', by={'state': state}, columns=('geo_id', 'name'), records=True, "
        "order=[Sort('name', descending=True, nulls='last'), Sort('geo_id')])" in source
    )
    assert "read_by(conn, 'geos', by={'state': state}, records=True, order=[Sort('geo_id')])" in source
    for entry in entries:
        assert_source_says(tmp_path, entry)
    left = [
        finding for finding in use_the_helper_findings(config) if " in geos_of:" in finding or "unordered" in finding
    ]
    assert not left, left
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


def test_a_listing_s_source_check_holds_its_order(tmp_path: Path) -> None:
    """Flips if the source check passes a listing whose ``order=`` no longer says the recorded sort."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_LISTINGS), encoding="utf-8")
    rewrites, _ = plan_list_rewrites(config)
    apply_rewrites(tmp_path, rewrites, ruff=None)
    entry = next(rewrite.record() for rewrite in rewrites if rewrite.function == "geos_unordered")
    path = tmp_path / "src/pkg/reads.py"
    path.write_text(path.read_text(encoding="utf-8").replace("order=[Sort('geo_id')]", "order=[]"), encoding="utf-8")
    with pytest.raises(AssertionError, match="no read_by call"):
        assert_source_says(tmp_path, entry)


_CLASS_CONSTANTS = """
    class Agents:
        #: an agent by its key
        _ONE_SQL = "SELECT * FROM agents WHERE id = $1"
        _NAME_SQL = "SELECT name FROM agents WHERE id = $1"

        async def one(self, conn, agent_id):
            return await conn.fetchrow(self._ONE_SQL, agent_id)

        @classmethod
        async def again(cls, conn, agent_id):
            return await conn.fetchrow(cls._ONE_SQL, agent_id)

        async def name(self, conn, agent_id):
            return await conn.fetchval(self._NAME_SQL, agent_id)


    async def name_elsewhere(other, conn, agent_id):
        return await conn.fetchval(other._NAME_SQL, agent_id)


    async def by_branch(conn, agent_id, live):
        if live:
            sql = "SELECT id FROM agents WHERE id = $1 AND live"
        else:
            sql = "SELECT id FROM agents WHERE id = $1"
        return await conn.fetchrow(sql, agent_id)
"""


def test_a_class_constant_sent_through_self_or_cls_is_rewritten(tmp_path: Path) -> None:
    """Flips if a class-level statement constant sent as ``self.X`` / ``cls.X`` is taken for unused (and left), if
    one also read through another object is rewritten away, or if a local bound by branch is rewritten (two
    statements onto one call) or refused as named elsewhere in the repo."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_CLASS_CONSTANTS), encoding="utf-8")
    rewrites, refusals = plan_point_rewrites(config)
    assert sorted(rewrite.function for rewrite in rewrites) == ["Agents.again", "Agents.one"]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert "read through other" in refused["Agents"], refused
    assert "bound more than once" in refused["by_branch"], refused
    apply_rewrites(tmp_path, rewrites, ruff=None)
    source = (tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8")
    assert "_ONE_SQL" not in source and "an agent by its key" not in source
    assert "await read_row(conn, 'agents', by={'id': agent_id}, records=True)" in source
    assert "_NAME_SQL = " in source
    for rewrite in rewrites:
        assert_source_says(tmp_path, rewrite.record())


def test_a_constant_named_elsewhere_is_left(tmp_path: Path) -> None:
    """Flips if a statement constant another module names in code (an import) is rewritten away."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/purposes.py").write_text("from pkg.reads import _GEO_SQL\n", encoding="utf-8")
    rewrites, refusals = plan_point_rewrites(config)
    assert not [rewrite for rewrite in rewrites if rewrite.table == "geos"]
    assert any("_GEO_SQL is named elsewhere" in refusal.reason for refusal in refusals)


def test_the_generated_test_module_parses_and_names_its_fixtures() -> None:
    """Flips if the generated module stops being Python, or stops using the repo's fixtures and manifest."""
    text = render_test_module(
        Path("tests/unit/test_rewrites.py"),
        "tests/helper_rewrites.json",
        pg_fixture="pg_conn",
        yb_fixture="yb_conn",
        overrides="tests.support.seeds:OVERRIDES",
    )
    ast.parse(text)
    assert "parents[2]" in text and "tests/helper_rewrites.json" in text
    assert "pg_conn: Any" in text and "yb_conn: Any" in text
    assert "from tests.support.seeds import OVERRIDES" in text and "overrides=OVERRIDES" in text


def test_the_generated_test_module_marks_itself_integration() -> None:
    """Flips if the generated module stops carrying its own ``integration`` mark, which is what puts it in an
    integration suite (the pool's, CI's) and keeps it out of the unit run, whatever conftest sits beside it."""
    text = render_test_module(
        Path("tests/integration/test_rewrites.py"), "tests/m.json", pg_fixture="conn", yb_fixture="yb"
    )
    module = ast.parse(text)
    marks = [
        ast.unparse(node.value)
        for node in module.body
        if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["pytestmark"]
    ]
    assert marks == ["pytest.mark.integration"]


_WRITES = """
    #: forget one geo
    _FORGET_SQL = "DELETE FROM geos WHERE state = $1 AND geo_id = $2"


    async def rename(conn, agent_id, name):
        await conn.execute("UPDATE agents SET name = $2, date_updated = now(), flag = TRUE WHERE id = $1", agent_id, name)


    async def forget(conn, state, geo):
        await conn.execute(_FORGET_SQL, state, geo)


    async def renamed_count(conn, agent_id, name):
        status = await conn.execute("UPDATE agents SET name = $1 WHERE id = $2", name, agent_id)
        return parse_rowcount(status)


    async def renamed_status(conn, agent_id, name):
        status = await conn.execute("UPDATE agents SET name = $1 WHERE id = $2", name, agent_id)
        return status


    async def by_status(conn, status):
        await conn.execute("DELETE FROM agents WHERE status = $1", status)


    async def bumped(conn, agent_id):
        await conn.execute("UPDATE agents SET hits = hits + 1, date_updated = now() WHERE id = $1", agent_id)


    async def raised(conn, version, at, agent_id):
        await conn.execute(
            "UPDATE agents SET version = GREATEST(version, $1), date_updated = $2, note = COALESCE($1::text, note)"
            " WHERE id = $3",
            version,
            at,
            agent_id,
        )


    async def from_elsewhere(conn, agent_id):
        await conn.execute("UPDATE agents SET hits = (SELECT max(hits) FROM agents) WHERE id = $1", agent_id)


    async def returned(conn, agent_id):
        return await conn.fetch("DELETE FROM agents WHERE id = $1 RETURNING id", agent_id)


    async def claimed(conn, agent_id, name):
        claimed = await conn.fetchval("UPDATE agents SET name = $1 WHERE id = $2 RETURNING id, name", name, agent_id)
        return claimed is not None


    async def moved(conn, agent_id, name):
        row = await conn.fetchrow("UPDATE agents SET name = $1 WHERE id = $2 RETURNING id, name", name, agent_id)
        return dict(row) if row is not None else None


    async def rename_many(conn, ids, name):
        await conn.execute("UPDATE agents SET name = $1 WHERE id = ANY($2::uuid[]) AND NOT gone", name, ids)


    async def returned_all(conn, agent_id):
        return await conn.fetch("DELETE FROM agents WHERE id = $1 RETURNING *", agent_id)


    async def returned_by_execute(conn, agent_id):
        await conn.execute("DELETE FROM agents WHERE id = $1 RETURNING id", agent_id)
"""


def _write_config(tmp_path: Path) -> NoWholeReadsConfig:
    src = tmp_path / "src"
    for name, source in {"pkg/tables.py": _TABLES, "pkg/writes.py": _WRITES}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )


def test_a_keyed_write_by_equality_is_parsed() -> None:
    """Flips if an UPDATE's SET (a parameter, a literal, NULL, now()) or a DELETE's pins are read other than written."""
    assert parse_write(
        "UPDATE a.t SET x = $1, y = NULL, z = now(), w = 'k' WHERE id = $2 AND gone IS NULL"
    ) == WriteSql(
        "update",
        "a.t",
        (Pin("x", "param", param=1), Pin("y", "null"), Pin("z", "now"), Pin("w", "literal", literal="k")),
        (Pin("id", "param", param=2), Pin("gone", "null")),
    )
    assert parse_write("DELETE FROM t WHERE k = $1 AND live = TRUE") == WriteSql(
        "delete", "t", (), (Pin("k", "param", param=1), Pin("live", "literal", literal=True))
    )


def test_an_increment_and_a_set_expression_are_parsed() -> None:
    """Flips if ``col = col + n`` is not read as an increment (``- n`` negative), another column's arithmetic or a
    function as anything but a fragment with its parameters renumbered from $1 in first-use order, or a comma inside
    the expression's parentheses splits the SET."""
    parsed = parse_write(
        "UPDATE t SET n = n + 1, m = m - 2, v = GREATEST(v, $3), w = COALESCE($1::jsonb, w, $3), x = y + 1"
        " WHERE id = $2"
    )
    assert isinstance(parsed, WriteSql), parsed
    assert parsed.sets == (
        Pin("n", "increment", literal=1),
        Pin("m", "increment", literal=-2),
        Pin("v", "fragment", literal="GREATEST(v, $1)", params=(3,)),
        Pin("w", "fragment", literal="COALESCE($1::jsonb, w, $2)", params=(1, 3)),
        Pin("x", "fragment", literal="y + 1", params=()),
    )


def test_a_returning_of_plain_columns_is_parsed() -> None:
    """Flips if a write's trailing RETURNING of plain columns is not read as those columns, in order, or is read into
    the WHERE's last pin."""
    assert parse_write("UPDATE t SET v = $1 WHERE id = $2 RETURNING id, V") == WriteSql(
        "update", "t", (Pin("v", "param", param=1),), (Pin("id", "param", param=2),), ("id", "v")
    )
    assert parse_write("DELETE FROM t WHERE id = $1 AND s = 'a returning b' RETURNING id;") == WriteSql(
        "delete", "t", (), (Pin("id", "param", param=1), Pin("s", "literal", literal="a returning b")), ("id",)
    )


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE t SET n = m + 1 WHERE id = $1 AND m = (SELECT 1)",
        "UPDATE t SET v = (SELECT $1::jsonb) WHERE id = $2",
        "UPDATE t SET v = $1 WHERE id IN ($2)",
        "UPDATE t SET v = $1 WHERE id = ANY($2) OR k = $3",
        "UPDATE t SET v = $1 FROM u WHERE t.id = u.id AND u.k = $2",
        "DELETE FROM t WHERE id IN (SELECT id FROM t WHERE at < $1 LIMIT $2)",
        "DELETE FROM t WHERE id = $1 RETURNING *",
        "DELETE FROM t WHERE id = $1 RETURNING id + 1",
        "DELETE FROM t WHERE id = $1 RETURNING id, id",
        "UPDATE t SET at = now() WHERE id = $1 RETURNING at",
        "DELETE FROM t USING u WHERE t.id = u.id AND u.k = $1",
        "UPDATE t SET v = (SELECT max(v) FROM u) WHERE id = $1",
        "DELETE FROM {} WHERE id = $1",
        "DELETE FROM t WHERE live = TRUE",
    ],
)
def test_anything_but_a_write_by_equality_is_refused(sql: str) -> None:
    """Flips if a write the helpers cannot send as written is taken for a keyed write by equality."""
    assert isinstance(parse_write(sql), str), sql


def test_keyed_writes_are_rewritten_recorded_and_pass_both_rules(tmp_path: Path) -> None:
    """Flips if a keyed write by equality, its status discarded (or its RETURNING of plain columns read by fetch,
    fetchrow or fetchval), is not rewritten onto update_keyed / delete_keyed exactly, read as the method it replaces
    read it, the manifest does not record it, or the rewritten module still holds a use-the-helper finding for it or
    a new whole-read finding; or a write whose status is read but not as a count (returned as it is), a RETURNING *
    or a RETURNING sent by execute, or an unkeyed write is rewritten. A write whose status is read as a count
    (``parse_rowcount(status)``) is rewritten onto the helper's ``.count`` (coordinator ruling 2026-11-12: the
    caller's status read becomes the helper's count; there is no ``KeyedWrite.status``)."""
    config = _write_config(tmp_path)
    before = no_whole_reads_findings(config)
    rewrites, refusals = plan_write_rewrites(config)
    assert sorted((rewrite.function, rewrite.helper) for rewrite in rewrites) == [
        ("bumped", "update_keyed"),
        ("claimed", "update_keyed"),
        ("forget", "delete_keyed"),
        ("moved", "update_keyed"),
        ("raised", "update_keyed"),
        ("rename", "update_keyed"),
        ("rename_many", "update_keyed"),
        ("renamed_count", "update_keyed"),
        ("returned", "delete_keyed"),
    ]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() >= {
        "renamed_status",
        "by_status",
        "from_elsewhere",
        "returned_all",
        "returned_by_execute",
    }, refused
    assert "status is read as 'return status', which no count reads" in refused["renamed_status"]
    assert "unkeyed" in refused["by_status"]
    assert "plain columns" in refused["returned_all"]
    assert "execute" in refused["returned_by_execute"]
    assert {rewrite.function: rewrite.returning for rewrite in rewrites if rewrite.returning} == {
        "claimed": ["id"],
        "moved": ["id", "name"],
        "returned": ["id"],
    }

    assert apply_rewrites(tmp_path, rewrites, ruff=None) == ["src/pkg/writes.py"]
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    assert {entry["shape"] for entry in entries} == {"write"}

    source = (tmp_path / "src/pkg/writes.py").read_text(encoding="utf-8")
    assert "from threetears.core.keyed import AnyOf, Fragment, Increment, delete_keyed, update_keyed" in source
    assert (
        "await update_keyed(conn, 'agents', values={'hits': Increment(1), 'date_updated': Fragment(\"now()\")},"
        " by={'id': agent_id})" in source
    )
    assert (
        "await update_keyed(conn, 'agents', values={'version': Fragment('GREATEST(version, $1)', (version,)),"
        " 'date_updated': at, 'note': Fragment('COALESCE($1::text, note)', (version,))}, by={'id': agent_id})" in source
    )
    assert (
        "await update_keyed(conn, 'agents', values={'name': name}, by={'id': AnyOf(ids, cast='uuid'), 'gone': False})"
        in source
    )
    assert "_FORGET_SQL" not in source and "forget one geo" not in source
    assert (
        "await update_keyed(conn, 'agents', values={'name': name, 'date_updated': Fragment(\"now()\"), 'flag': True},"
        " by={'id': agent_id})" in source
    )
    assert "await delete_keyed(conn, 'geos', by={'state': state, 'geo_id': geo})" in source
    assert "return (await delete_keyed(conn, 'agents', by={'id': agent_id}, returning=('id',))).rows" in source
    assert (
        "claimed = (await update_keyed(conn, 'agents', values={'name': name}, by={'id': agent_id},"
        " returning=('id',))).value" in source
    )
    assert (
        "row = (await update_keyed(conn, 'agents', values={'name': name}, by={'id': agent_id},"
        " returning=('id', 'name'))).row" in source
    )
    assert (
        "status = (await update_keyed(conn, 'agents', values={'name': name}, by={'id': agent_id})).count\n"
        "    return status\n" in source
    )
    for entry in entries:
        assert_source_says(tmp_path, entry)

    left = [finding for finding in use_the_helper_findings(config) if "use-the-helper write" in finding]
    assert not [
        finding
        for finding in left
        if any(
            f" in {function}:" in finding
            for function in (
                "rename",
                "forget",
                "returned",
                "claimed",
                "moved",
                "rename_many",
                "bumped",
                "raised",
                "renamed_count",
            )
        )
    ], left
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


_STATUS_READS = """
    from typing import Final

    from threetears.core.backends.protocol import parse_rowcount
    from threetears.interfaces.keyed.counts import rows_affected

    #: forget one agent, or one geo
    _FORGET_SQL: Final[dict[str, str]] = {
        "agents": "DELETE FROM agents WHERE id = $1",
        "geos": "DELETE FROM geos WHERE state = $1",
    }
    _HIDE_SQL = "UPDATE agents SET gone = TRUE WHERE id = $1"
    _DROP_SQL = "DELETE FROM agents WHERE id = $1"


    async def compared(conn, agent_id):
        if await conn.execute("DELETE FROM agents WHERE id = $1", agent_id) == "DELETE 0":
            return False
        return True


    async def kept(conn, agent_id, name):
        status = await conn.execute("UPDATE agents SET name = $1 WHERE id = $2", name, agent_id)
        log = "UPDATE 0" != status
        return status != "UPDATE 0", log


    async def ended(conn, agent_id):
        status = await conn.execute("UPDATE agents SET hits = 0 WHERE id = $1", agent_id)
        if status.endswith(" 0"):
            raise LookupError(agent_id)


    async def ended_not(conn, agent_id):
        return not (await conn.execute("UPDATE agents SET hits = 1 WHERE id = $1", agent_id)).endswith(" 1")


    async def counted(conn, agent_id):
        n = int((await conn.execute("DELETE FROM agents WHERE gone AND id = $1", agent_id)).split()[-1])
        return n


    async def worded(conn, agent_id):
        status = await conn.execute("UPDATE agents SET hits = 2 WHERE id = $1", agent_id)
        return status.split()[-1] == "0", int(status.rsplit(" ", 1)[-1])


    async def parsed(conn, agent_id):
        return parse_rowcount(await conn.execute("UPDATE agents SET hits = 3 WHERE id = $1", agent_id))


    async def affected(conn, agent_id):
        status: str = await conn.execute("UPDATE agents SET hits = 4 WHERE id = $1", agent_id)
        return rows_affected(status) == 1


    async def forget(conn, table, key):
        return parse_rowcount(await conn.execute(_FORGET_SQL[table], key))


    async def drop_or_hide(conn, agent_id, hard):
        sql = _DROP_SQL if hard else _HIDE_SQL
        return parse_rowcount(await conn.execute(sql, agent_id))


    async def wrong_verb(conn, agent_id):
        status = await conn.execute("UPDATE agents SET hits = 5 WHERE id = $1", agent_id)
        return status == "DELETE 0"


    async def logged(conn, agent_id):
        status = await conn.execute("UPDATE agents SET hits = 6 WHERE id = $1", agent_id)
        print(status)
        return parse_rowcount(status)


    async def suffix(conn, agent_id):
        status = await conn.execute("UPDATE agents SET hits = 7 WHERE id = $1", agent_id)
        return status.endswith("0")


    async def rebound(conn, agent_id):
        status = await conn.execute("UPDATE agents SET hits = 8 WHERE id = $1", agent_id)
        status = parse_rowcount(status)
        return status
"""


def test_a_write_whose_status_is_read_reads_the_helpers_count(tmp_path: Path) -> None:
    """Flips if a keyed write whose status is read in a common spelling (compared with its own verb's tag, ``!=``
    either side, ``.endswith(" n")``, ``int(status.split()[-1])`` / ``rsplit(" ", 1)``, the last word compared with a
    number, ``parse_rowcount`` / ``rows_affected``; read where it is awaited or through a local bound once, through a
    dict of statements or a conditional choice of two) is not rewritten onto the helper with each read of the status
    reading the helper's ``.count`` exactly (coordinator ruling 2026-11-12; there is no ``KeyedWrite.status``), or if a
    spelling no count reads (another verb's tag, a bare ``endswith("0")``, the status printed, the local bound twice)
    is rewritten at all."""
    src = tmp_path / "src"
    for name, source in {"pkg/tables.py": _TABLES, "pkg/status.py": _STATUS_READS}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    config = NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )
    before = no_whole_reads_findings(config)
    rewrites, refusals = plan_write_rewrites(config)
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert {rewrite.function for rewrite in rewrites} == {
        "compared",
        "kept",
        "ended",
        "ended_not",
        "counted",
        "worded",
        "parsed",
        "affected",
        "<module>",
        "drop_or_hide",
    }, refused
    assert refused.keys() == {"wrong_verb", "logged", "suffix", "rebound"}, refused
    assert "compared with 'DELETE 0', not its own verb's tag" in refused["wrong_verb"]
    assert "read as 'print(status)', which no count reads" in refused["logged"]
    assert "read as 'status.endswith(\"0\")'" in refused["suffix"]
    assert "bound more than once" in refused["rebound"]

    assert apply_rewrites(tmp_path, rewrites, ruff=None) == ["src/pkg/status.py"]
    source = (src / "pkg/status.py").read_text(encoding="utf-8")
    ast.parse(source)
    for text in (
        "if (await delete_keyed(conn, 'agents', by={'id': agent_id})).count == 0:",
        "status = (await update_keyed(conn, 'agents', values={'name': name}, by={'id': agent_id})).count",
        "log = 0 != status",
        "return status != 0, log",
        "if status == 0:",
        "return not ((await update_keyed(conn, 'agents', values={'hits': 1}, by={'id': agent_id})).count == 1)",
        "n = (await delete_keyed(conn, 'agents', by={'gone': True, 'id': agent_id})).count",
        "return status == 0, status",
        "return (await update_keyed(conn, 'agents', values={'hits': 3}, by={'id': agent_id})).count",
        "status: int = (await update_keyed(conn, 'agents', values={'hits': 4}, by={'id': agent_id})).count",
        "return status == 1",
        "return (await _FORGET_SQL[table](conn, key)).count",
        "return (await (delete_keyed(conn, 'agents', by={'id': agent_id}) if hard else update_keyed(conn, 'agents',"
        " values={'gone': True}, by={'id': agent_id}))).count",
    ):
        assert text in source, (text, source)
    assert "_DROP_SQL" not in source and "_HIDE_SQL" not in source and "sql = " not in source
    assert "parse_rowcount(status)\n" in source and 'status == "DELETE 0"' in source
    # a status parser the rewrite left unused is no longer imported; one still read stays
    assert "rows_affected" not in source, source
    assert "from threetears.core.backends.protocol import parse_rowcount\n" in source
    left = [finding for finding in use_the_helper_findings(config) if "use-the-helper write" in finding]
    assert {finding.split(" in ")[1].split(":")[0] for finding in left} == set(refused), left
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


_DICTS = """
    from typing import Final

    #: each table's rows of one agent
    _ROWS_SQL: Final[dict[str, str]] = {
        "agents": "SELECT id FROM agents WHERE id = $1",
        "geos": "SELECT geo_id FROM geos WHERE state = $1",
    }

    _MIXED_SQL = {
        "agents": "SELECT id FROM agents WHERE id = $1",
        "geos": "SELECT geo_id FROM geos WHERE geo_id > $1",
    }

    _FORGET_SQL: Final[dict[str, str]] = {
        "agents": "DELETE FROM agents WHERE id = $1",
        "geos": "DELETE FROM geos WHERE state = $1",
    }

    _TWICE_SQL = {
        "agents": "SELECT id FROM agents WHERE id = $1",
    }


    async def rows_of(conn, table, key):
        return await conn.fetch(_ROWS_SQL[table], key)


    async def every_table(pool, key):
        return [await pool.fetch(_ROWS_SQL[table], key) for table in sorted(_ROWS_SQL)]


    async def mixed(conn, table, key):
        return await conn.fetch(_MIXED_SQL[table], key)


    async def forget(conn, table, key):
        await conn.execute(_FORGET_SQL[table], key)


    async def twice(conn, key):
        return await conn.fetch(_TWICE_SQL["agents"], key), await conn.fetchrow(_TWICE_SQL["agents"], key)
"""


def _dict_config(tmp_path: Path) -> NoWholeReadsConfig:
    src = tmp_path / "src"
    for name, source in {"pkg/tables.py": _TABLES, "pkg/dicts.py": _DICTS}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_keys.py").write_text(
        "from pkg import dicts\n\n\ndef test_keys():\n    assert set(dicts._ROWS_SQL) == {'agents', 'geos'}\n",
        encoding="utf-8",
    )
    return NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )


def test_a_dict_of_statements_becomes_a_dict_of_helper_calls(tmp_path: Path) -> None:
    """Flips if a module dict of point reads (or keyed writes), every use of which sends one value indexed at the
    call, is not rewritten into a dict of ``lambda conn, <params>:`` helper calls with each call sending through it
    and the annotation retyped; if a test reading only its keys blocks it; or if a dict holding an unkeyed value, or
    sent by two methods, is rewritten at all."""
    config = _dict_config(tmp_path)
    before = no_whole_reads_findings(config)
    reads, read_refusals = plan_point_rewrites(config)
    writes, write_refusals = plan_write_rewrites(config)
    assert sorted((rewrite.table, rewrite.helper, rewrite.function) for rewrite in reads) == [
        ("agents", "read_by", "<module>"),
        ("geos", "read_by", "<module>"),
    ]
    assert sorted((rewrite.table, rewrite.helper) for rewrite in writes) == [
        ("agents", "delete_keyed"),
        ("geos", "delete_keyed"),
    ]
    reasons = {refusal.reason for refusal in read_refusals}
    assert any("_MIXED_SQL" in reason and "unkeyed" in reason for reason in reasons), reasons
    assert any("_TWICE_SQL" in reason and "one helper per value" in reason for reason in reasons), reasons
    assert not write_refusals, write_refusals

    apply_rewrites(tmp_path, [*reads, *writes], ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [*reads, *writes])
    source = (tmp_path / "src/pkg/dicts.py").read_text(encoding="utf-8")
    assert "_ROWS_SQL: Final[dict[str, Callable[..., Awaitable[Any]]]] = {" in source
    assert "lambda conn, p1: read_by(conn, 'agents', by={'id': p1}, columns=('id',), records=True)" in source
    assert "lambda conn, state: read_by(conn, 'geos', by={'state': state}, columns=('geo_id',), records=True)" in source
    assert "lambda conn, state: delete_keyed(conn, 'geos', by={'state': state})" in source
    assert "return await _ROWS_SQL[table](conn, key)" in source
    assert "await pool.fetch" not in source and "await _ROWS_SQL[table](pool, key) for table" in source
    assert "await _FORGET_SQL[table](conn, key)" in source
    assert "from collections.abc import Awaitable, Callable" in source and "from typing import Any, Final" in source
    ast.parse(source)
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        assert_source_says(tmp_path, entry)

    left = use_the_helper_findings(config)
    assert not [finding for finding in left if "_ROWS_SQL" in finding or "_FORGET_SQL" in finding], left
    assert not [f for f in left if ("agents" in f or "geos" in f) and ("rows_of" in f or "forget" in f)], left
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


def test_a_dict_of_statements_another_module_sends_is_left(tmp_path: Path) -> None:
    """Flips if a dict of statements is rewritten while another module reads its values, not only its keys."""
    config = _dict_config(tmp_path)
    (tmp_path / "tests/test_values.py").write_text(
        "from pkg import dicts\n\n\ndef test_sql():\n    assert 'agents' in dicts._ROWS_SQL['agents']\n",
        encoding="utf-8",
    )
    reads, refusals = plan_point_rewrites(config)
    assert not [rewrite for rewrite in reads if rewrite.old_sql.endswith("WHERE state = $1")]
    assert any("_ROWS_SQL is named elsewhere" in refusal.reason for refusal in refusals), refusals


def test_a_constant_named_in_an_index_purpose_is_rewritten_and_the_purpose_repointed(tmp_path: Path) -> None:
    """Flips if a statement constant whose only other mention is an index purpose entry (in a migration or the
    declaration ledger) is left, if the purpose is not repointed to each function now making its call, or if a
    comment naming it blocks the rewrite rather than being listed."""
    config = _config(tmp_path)
    migration = tmp_path / "src/pkg/v001.py"
    migration.write_text(
        'SQL = """\n-- purpose: src/pkg/other.py::f; src/pkg/reads.py::_GEO_SQL\nCREATE INDEX i ON geos (state)\n"""\n',
        encoding="utf-8",
    )
    ledger = tmp_path / "tests/_index_declarations.txt"
    ledger.parent.mkdir()
    ledger.write_text("i :: on geos :: purpose=src/pkg/reads.py::_GEO_SQL :: plan_test=t.py::t\n", encoding="utf-8")
    (tmp_path / "tests/test_geo.py").write_text("# the read behind _GEO_SQL\n", encoding="utf-8")
    rewrites, refusals = plan_point_rewrites(config)
    geos = [rewrite for rewrite in rewrites if rewrite.table == "geos"]
    assert sorted(rewrite.function for rewrite in geos) == ["a_geo", "another_geo"], refusals
    noted = [note for rewrite in geos for note in rewrite.notes]
    assert noted == ["tests/test_geo.py:1: still names _GEO_SQL (prose; reword by hand)"], noted
    apply_rewrites(tmp_path, rewrites, ruff=None)
    repointed = "src/pkg/reads.py::a_geo; src/pkg/reads.py::another_geo"
    assert f"-- purpose: src/pkg/other.py::f; {repointed}\n" in migration.read_text(encoding="utf-8")
    assert f"purpose={repointed} ::" in ledger.read_text(encoding="utf-8")


_CHOICES = """
    _STAGES_SQL = "UPDATE agents SET stages = $2 WHERE id = $1 AND gone = false"
    _UNITS_SQL = "UPDATE agents SET units = $2 WHERE id = $1 AND gone = false"
    _ONE_SQL = "SELECT id, name FROM agents WHERE id = $1"
    _OTHER_SQL = "SELECT geo_id, name FROM geos WHERE state = $1"
    _COUNTED_A_SQL = "UPDATE agents SET name = $2 WHERE id = $1"
    _COUNTED_B_SQL = "UPDATE agents SET slug = $2 WHERE id = $1"


    async def store(conn, agent_id, counts, per_unit):
        statement = _UNITS_SQL if per_unit else _STAGES_SQL
        await conn.execute(statement, agent_id, counts)


    async def either(conn, key, of_geos):
        sql = _OTHER_SQL if of_geos else _ONE_SQL
        return await conn.fetch(sql, key)


    async def counted(conn, agent_id, value, slug):
        sql = _COUNTED_B_SQL if slug else _COUNTED_A_SQL
        status = await conn.execute(sql, agent_id, value)
        return status
"""


def test_two_constants_chosen_by_a_conditional_become_a_conditional_of_helper_calls(tmp_path: Path) -> None:
    """Flips if two statement constants chosen by ``x = A if c else B`` and sent by the next statement are not
    rewritten into ``(<A's helper call> if c else <B's>)`` with the local and both constants removed; or if a
    choice whose write's status is read is rewritten."""
    src = tmp_path / "src"
    for name, source in {"pkg/tables.py": _TABLES, "pkg/choices.py": _CHOICES}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    config = NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )
    before = no_whole_reads_findings(config)
    reads, _ = plan_point_rewrites(config)
    writes, write_refusals = plan_write_rewrites(config)
    assert sorted((rewrite.function, rewrite.table) for rewrite in reads) == [("either", "agents"), ("either", "geos")]
    assert sorted((rewrite.function, rewrite.table) for rewrite in writes) == [("store", "agents"), ("store", "agents")]
    assert {refusal.function for refusal in write_refusals} >= {"<module>"}
    assert any("status is read" in refusal.reason for refusal in write_refusals), write_refusals

    apply_rewrites(tmp_path, [*reads, *writes], ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [*reads, *writes])
    source = (tmp_path / "src/pkg/choices.py").read_text(encoding="utf-8")
    assert (
        "await (update_keyed(conn, 'agents', values={'units': counts}, by={'id': agent_id, 'gone': False}) if per_unit"
        " else update_keyed(conn, 'agents', values={'stages': counts}, by={'id': agent_id, 'gone': False}))" in source
    )
    assert (
        "return await (read_by(conn, 'geos', by={'state': key}, columns=('geo_id', 'name'), records=True) if of_geos"
        " else read_by(conn, 'agents', by={'id': key}, columns=('id', 'name'), records=True))" in source
    )
    for gone in ("statement =", "sql = _OTHER", "_STAGES_SQL", "_UNITS_SQL", "_ONE_SQL", "_OTHER_SQL"):
        assert gone not in source, gone
    assert "_COUNTED_A_SQL" in source and "status = await conn.execute(sql, agent_id, value)" in source
    ast.parse(source)
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        assert_source_says(tmp_path, entry)
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


_FIRSTS = """
    async def top_names(conn, state, n):
        return await conn.fetch(
            "SELECT geo_id, name FROM geos WHERE state = $1 ORDER BY name DESC LIMIT $2", state, n
        )


    async def first_name(conn, state):
        return await conn.fetchrow("SELECT geo_id, name FROM geos WHERE state = $1 ORDER BY name LIMIT 1", state)


    async def first_id(conn, state):
        return await conn.fetchval("SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id LIMIT 1", state)


    async def hidden_key(conn, state):
        return await conn.fetchrow("SELECT name FROM geos WHERE state = $1 ORDER BY name LIMIT 1", state)


    async def unordered(conn, state):
        return await conn.fetchrow("SELECT geo_id FROM geos WHERE state = $1 LIMIT 1", state)


    async def listed(conn, state):
        return await conn.fetch("SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id", state)
"""


def test_a_first_of_read_is_parsed_with_its_limit() -> None:
    """Flips if a trailing ``LIMIT 1`` / ``LIMIT $n`` after a plain-column ``ORDER BY`` is not read as the shape's
    limit, or a LIMIT with no order, an OFFSET or a LIMIT parameter not last is taken."""
    top = parse_first("SELECT geo_id, name FROM geos WHERE state = $1 ORDER BY name DESC LIMIT $2")
    assert isinstance(top, PointSql) and top.limit == {"param": 2}, top
    assert top.order == ({"column": "name", "descending": True, "nulls": None},)
    one = parse_first("SELECT * FROM geos WHERE state = $1 ORDER BY name LIMIT 1")
    assert isinstance(one, PointSql) and one.limit == {"literal": 1} and one.columns is None, one
    for refused in (
        "SELECT geo_id FROM geos WHERE state = $1 LIMIT 1",
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id LIMIT 1 OFFSET 2",
        "SELECT geo_id FROM geos WHERE state = $2 ORDER BY geo_id LIMIT $1",
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id",
        "SELECT geo_id FROM geos WHERE state = $1 ORDER BY geo_id LIMIT 0",
    ):
        assert isinstance(parse_first(refused), str), refused


def test_first_of_and_top_n_reads_are_rewritten_onto_keyset_page_s_first_page(tmp_path: Path) -> None:
    """Flips if a first-of or top-n read is not rewritten onto ``keyset_page``'s first page with records, its order
    completed by the table's key (the pins and sort holding a key complete nothing), read as the old method answered;
    or a read with a key column unselected, no order, or no LIMIT is taken."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_FIRSTS), encoding="utf-8")
    (tmp_path / "src/pkg/indexes.py").write_text(textwrap.dedent(_GEO_NAME_INDEX), encoding="utf-8")
    rewrites, refusals = plan_first_rewrites(config)
    assert sorted(rewrite.function for rewrite in rewrites) == ["first_id", "first_name", "top_names"]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    # ``unordered`` (a LIMIT with no ORDER BY) is a point finding, refused by the point shape: never planned here
    assert refused.keys() == {"hidden_key", "listed"}, refused
    assert "not selected" in refused["hidden_key"]

    apply_rewrites(tmp_path, rewrites, ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    entries = {entry["function"]: entry for entry in json.loads(manifest.read_text(encoding="utf-8"))}
    assert {entry["shape"] for entry in entries.values()} == {"first"}
    assert entries["top_names"]["key_from"] == 1 and entries["top_names"]["limit"] == {"param": 2}
    assert [term["column"] for term in entries["top_names"]["order"]] == ["name", "geo_id"]
    assert [term["column"] for term in entries["first_id"]["order"]] == ["geo_id"]
    source = " ".join((tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8").split())
    assert "from threetears.core.keyed import Sort, keyset_page" in source
    assert (
        "(await keyset_page(conn, 'geos', columns=('geo_id', 'name'), by={'state': state}, "
        "order=[Sort('name', descending=True), Sort('geo_id')], limit=n, records=True)).rows" in source
    )
    assert "next(iter((await keyset_page(conn, 'geos', columns=('geo_id', 'name')," in source
    assert "next((row['geo_id'] for row in (await keyset_page(" in source
    for entry in entries.values():
        assert_source_says(tmp_path, entry)


#: the index the first-of reads by state walk in name order, completed by the key's geo_id
_GEO_NAME_INDEX = """
    BY_NAME = "CREATE INDEX IF NOT EXISTS geos_by_name ON geos (state, name, geo_id)"
"""

_SLOTS = """
    from threetears.core.collections.schema_backed import TableSchema

    SLOTS_SCHEMA = TableSchema(name="slots", primary_key=("zone", "slot_id"), columns=[])
    SLOT_ID = "CREATE UNIQUE INDEX IF NOT EXISTS slots_slot_id ON slots (slot_id)"
    BY_KIND = (
        "-- purpose: src/pkg/slots.py::first_due\\n"
        "CREATE INDEX IF NOT EXISTS slots_by_kind ON slots (kind, due_at, slot_id)"
    )


    async def first_due(conn, kind):
        return await conn.fetchrow(
            "SELECT zone, slot_id, due_at FROM slots WHERE kind = $1 ORDER BY due_at LIMIT 1", kind
        )
"""


def test_a_first_of_read_whose_completed_order_no_index_holds_is_refused(tmp_path: Path) -> None:
    """Flips if a first-of read is rewritten with its order completed by a key no declared index holds after its
    pins and sort (the database would then read and sort every row the pins leave: YugabyteDB has no incremental
    sort), or if the completion is not the declared key (primary or ``UNIQUE``) the pins and sort hold most of."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_FIRSTS), encoding="utf-8")
    rewrites, refusals = plan_first_rewrites(config)
    assert [rewrite.function for rewrite in rewrites] == ["first_id"]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    for function in ("top_names", "first_name"):
        assert "no declared index of geos holds its order completed by the key (name, geo_id" in refused[function]

    (tmp_path / "src/pkg/slots.py").write_text(textwrap.dedent(_SLOTS), encoding="utf-8")
    rewrites, refusals = plan_first_rewrites(config)
    slots = [rewrite for rewrite in rewrites if rewrite.path.endswith("slots.py")]
    # the UNIQUE slot_id completes due_at (slots_by_kind holds it after the pinned kind), not the primary key
    assert [rewrite.function for rewrite in slots] == ["first_due"], refusals
    apply_rewrites(tmp_path, slots, ruff=None)
    source = " ".join((tmp_path / "src/pkg/slots.py").read_text(encoding="utf-8").split())
    assert "order=[Sort('due_at'), Sort('slot_id')]" in source, source


_COUNTS = """
    #: whether a state holds any geo
    _HELD_SQL = "SELECT EXISTS (SELECT 1 FROM geos WHERE state = $1) AS held"


    async def geo_count(conn, state):
        return await conn.fetchval("SELECT count(*) FROM geos WHERE state = $1", state)


    async def any_geo(conn, state):
        return await conn.fetchval(_HELD_SQL, state)


    async def live_agents(conn, ids):
        return await conn.fetchval(
            "SELECT COUNT(*) AS n FROM agents WHERE id = ANY($1::uuid[]) AND NOT is_deleted", ids
        )


    async def named_geo(conn, state, name):
        return await conn.fetchval("SELECT EXISTS (SELECT 1 FROM geos WHERE name = $2 AND state = $1)", state, name)


    async def summed(conn, state):
        return await conn.fetchval("SELECT sum(pop) FROM geos WHERE state = $1", state)


    async def counted_row(conn, state):
        row = await conn.fetchrow("SELECT count(*) AS n FROM geos WHERE state = $1", state)
        return row["n"]
"""


def test_a_count_and_an_existence_test_are_parsed() -> None:
    """Flips if ``SELECT count(*)`` / ``SELECT EXISTS (SELECT 1 ...)`` of a point read's rows is not read as a count
    with its pins, or another select list, an ``EXISTS`` of other than ``1``, or a range is taken."""
    counted = parse_count("SELECT COUNT(*) AS cnt FROM agents a WHERE a.id = $1 AND a.status = 'active'")
    assert isinstance(counted, PointSql) and counted.aggregate == "count" and counted.columns is None, counted
    assert [pin.kind for pin in counted.pins] == ["param", "literal"]
    held = parse_count("SELECT EXISTS(SELECT 1 FROM geos WHERE state = $1 AND geo_id = $2)")
    assert isinstance(held, PointSql) and held.aggregate == "exists" and held.table == "geos", held
    for refused in (
        "SELECT sum(pop) FROM geos WHERE state = $1",
        "SELECT count(*), max(pop) FROM geos WHERE state = $1",
        "SELECT count(*) FROM geos WHERE state = $1 GROUP BY name",
        "SELECT EXISTS (SELECT name FROM geos WHERE state = $1)",
        "SELECT count(*) FROM geos WHERE state = $1 AND pop > $2",
        "SELECT count(*) FROM {table} WHERE state = $1",
    ):
        assert isinstance(parse_count(refused), str), refused


def test_counts_and_existence_tests_are_rewritten_onto_aggregate_keyed_and_holds_any(tmp_path: Path) -> None:
    """Flips if a count sent by ``fetchval`` is not rewritten onto ``aggregate_keyed``'s one row's count, an
    ``EXISTS`` onto ``holds_any`` walking the declared key its pins lead, each recorded and found by the source check;
    or a sum, or a count read off a row by its name, is taken."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_COUNTS), encoding="utf-8")
    rewrites, refusals = plan_count_rewrites(config)
    assert sorted(rewrite.function for rewrite in rewrites) == ["any_geo", "geo_count", "live_agents", "named_geo"]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() == {"summed", "counted_row"}, refused
    assert "fetchrow" in refused["counted_row"]

    apply_rewrites(tmp_path, rewrites, ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    entries = {entry["function"]: entry for entry in json.loads(manifest.read_text(encoding="utf-8"))}
    assert {entry["shape"] for entry in entries.values()} == {"count"}
    assert entries["any_geo"]["helper"] == "holds_any" and entries["any_geo"]["walk"] == ["state", "geo_id"]
    assert entries["geo_count"]["helper"] == "aggregate_keyed" and "walk" not in entries["geo_count"]
    # a pin past the key's lead it walks (name) narrows the walk: the key's leading state is the one it holds
    assert entries["named_geo"]["walk"] == ["state", "geo_id"]
    source = " ".join((tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8").split())
    assert "_HELD_SQL" not in source
    assert "from threetears.core.keyed import AnyOf, Count, aggregate_keyed, holds_any" in source
    assert "(await aggregate_keyed(conn, 'geos', aggregates={'n': Count()}, by={'state': state}))[0]['n']" in source
    assert "return (await holds_any(conn, 'geos', order=['state', 'geo_id'], by={'state': state}))" in source
    assert "by={'id': AnyOf(ids, cast='uuid'), 'is_deleted': False}" in source
    for entry in entries.values():
        assert_source_says(tmp_path, entry)
    left = [line for line in use_the_helper_findings(config) if "reads.py" in line]
    assert [line.split(":")[1].split(" in ")[1] for line in left] == ["summed", "counted_row"], left


_GUARDED = """
    async def big_geos(conn, state, floor):
        return await conn.fetchval("SELECT count(*) FROM geos WHERE state = $1 AND pop > $2", state, floor)


    async def named_or_all(conn, state, name):
        return await conn.fetch(
            "SELECT geo_id, name FROM geos WHERE state = $1 AND ($2::text IS NULL OR geos.name = $2) ORDER BY geo_id",
            state,
            name,
        )


    async def live_name(conn, agent_id):
        return await conn.fetchval(
            "SELECT name FROM agents WHERE id = $1 AND NOT (COALESCE(tags, '[]'::jsonb) ? 'gone')", agent_id
        )


    async def rename_big(conn, name, state, floor):
        await conn.execute("UPDATE geos SET name = $1 WHERE state = $2 AND pop >= $3", name, state, floor)


    async def either(conn, state, name):
        return await conn.fetch("SELECT geo_id FROM geos WHERE state = $1 AND name = $2 OR pop > 3", state, name)


    async def flagged(conn, state, on):
        return await conn.fetch("SELECT geo_id FROM geos WHERE state = $1 AND length($2) > 3", state, on)
"""


def test_a_predicate_other_than_equality_is_kept_as_a_guard_when_asked() -> None:
    """Flips if, with ``guards=True``, a range, an optional pin or a ``NOT (...)`` beside the equalities is not kept
    as a ``where=`` guard (qualifiers dropped, parameters renumbered from ``$1``, each paired with the column it is
    compared with), or a top-level ``OR``, a ``BETWEEN``, a subquery or another relation's column is taken."""
    counted = parse_count("SELECT count(*) FROM geos g WHERE g.state = $1 AND g.pop > $2", guards=True)
    assert isinstance(counted, PointSql), counted
    assert [(guard.literal, guard.params, guard.compared) for guard in counted.guards] == [("pop > $1", (2,), ("pop",))]
    optional = parse_list(
        "SELECT geo_id FROM geos WHERE state = $1 AND ($2::text IS NULL OR geos.name = $2) ORDER BY geo_id",
        guards=True,
    )
    assert isinstance(optional, PointSql), optional
    assert [(guard.literal, guard.params, guard.compared) for guard in optional.guards] == [
        ("($1::text IS NULL OR name = $1)", (2,), ("name",))
    ]
    flagged = parse_count("SELECT count(*) FROM geos WHERE state = $1 AND ($2 OR name <> 'x')", guards=True)
    assert isinstance(flagged, PointSql), flagged
    # a truth value standing alone: the proof binds it False so the predicate beside it decides
    assert [(guard.literal, guard.compared) for guard in flagged.guards] == [("($1 OR name <> 'x')", (":bool",))]
    written = parse_write("UPDATE geos SET name = $1 WHERE state = $2 AND pop >= $3", guards=True)
    assert isinstance(written, WriteSql) and [guard.literal for guard in written.guards] == ["pop >= $1"], written
    for refused in (
        "SELECT id FROM t WHERE a = $1 AND b = $2 OR c = $3",
        "SELECT id FROM t WHERE a = $1 AND b BETWEEN $2 AND $3",
        "SELECT id FROM t WHERE a = $1 AND EXISTS (SELECT 1 FROM u WHERE u.id = t.id)",
        "SELECT id FROM t WHERE a = $1 AND u.b > $2",
        "SELECT id FROM t WHERE a = $1 AND b = ?",
        "SELECT id FROM t WHERE b > $1",
    ):
        assert isinstance(parse_point(refused, guards=True), str), refused


def test_a_read_or_write_with_a_guard_is_rewritten_with_where_fragments(tmp_path: Path) -> None:
    """Flips if a keyed read or write whose ``WHERE`` holds a predicate other than equality is not rewritten onto its
    helper with that predicate as a ``where=`` ``Fragment`` beside the ``by=`` pins (recorded, found by the source
    check, and no use-the-helper finding left for it), or a top-level ``OR`` or a guard parameter no seed can bind is
    taken."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_GUARDED), encoding="utf-8")
    before = no_whole_reads_findings(config)
    rewrites = []
    refused: dict[str, str] = {}
    for planner in (plan_count_rewrites, plan_list_rewrites, plan_point_rewrites, plan_write_rewrites):
        planned, refusals = planner(config)
        rewrites.extend(planned)
        refused.update({refusal.function: refusal.reason for refusal in refusals})
    assert sorted(rewrite.function for rewrite in rewrites) == ["big_geos", "live_name", "named_or_all", "rename_big"]
    assert "OR outside" in refused["either"], refused
    assert "compared with no column" in refused["flagged"], refused

    apply_rewrites(tmp_path, rewrites, ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    entries = {entry["function"]: entry for entry in json.loads(manifest.read_text(encoding="utf-8"))}
    assert entries["big_geos"]["where"] == [
        {
            "column": "",
            "kind": "guard",
            "param": None,
            "literal": "pop > $1",
            "cast": None,
            "params": [2],
            "compared": ["pop"],
        }
    ]
    source = " ".join((tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8").split())
    assert "Fragment" in source.split("import")[1]
    assert (
        "(await aggregate_keyed(conn, 'geos', aggregates={'n': Count()}, by={'state': state}, "
        "where=[Fragment('pop > $1', (floor,))]))[0]['n']" in source
    )
    assert (
        "read_by(conn, 'geos', by={'state': state}, where=[Fragment('($1::text IS NULL OR name = $1)', (name,))], "
        "columns=('geo_id', 'name'), records=True, order=[Sort('geo_id')])" in source
    )
    assert (
        "read_value(conn, 'agents', by={'id': agent_id}, where=[Fragment(\"NOT (COALESCE(tags, '[]'::jsonb) ? "
        "'gone')\", ())], column='name')" in source
    )
    assert (
        "update_keyed(conn, 'geos', values={'name': name}, by={'state': state}, where=[Fragment('pop >= $1', "
        "(floor,))])" in source
    )
    for entry in entries.values():
        assert_source_says(tmp_path, entry)
    left = [line for line in use_the_helper_findings(config) if "reads.py" in line]
    assert sorted(line.split(":")[1].split(" in ")[1] for line in left) == ["either", "flagged"], left
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


_CONSTANT_COLUMNS = """
    _COLUMNS = "geo_id, " "name"
    _LISTED = ("geo_id", "name")
    _JOINED = ", ".join(_LISTED)
    _COMPUTED = ", ".join(column for column in _LISTED)


    async def by_constant(conn, state, geo):
        return await conn.fetchrow(f"SELECT {_COLUMNS} FROM geos WHERE state = $1 AND geo_id = $2", state, geo)


    async def by_joined(conn, state):
        return await conn.fetch(f"SELECT {_JOINED} FROM geos WHERE state = $1 ORDER BY geo_id", state)


    async def computed(conn, state):
        return await conn.fetch(f"SELECT {_COMPUTED} FROM geos WHERE state = $1", state)


    async def by_local(conn, state, columns):
        return await conn.fetch(f"SELECT {columns} FROM geos WHERE state = $1", state)
"""


def test_an_f_string_of_module_constants_is_rewritten_as_the_statement_it_sends(tmp_path: Path) -> None:
    """Flips if an f-string statement whose every interpolation is a module constant of text (a literal, or a
    ``", ".join`` of a literal tuple) is not rewritten as the statement it sends (recorded as that text, found by the
    source check), or one interpolating a value computed at run time (a generator join, a local) is taken."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/reads.py").write_text(textwrap.dedent(_CONSTANT_COLUMNS), encoding="utf-8")
    rewrites = []
    refused: dict[str, str] = {}
    for planner in (plan_point_rewrites, plan_list_rewrites):
        planned, refusals = planner(config)
        rewrites.extend(planned)
        refused.update({refusal.function: refusal.reason for refusal in refusals})
    assert sorted(rewrite.function for rewrite in rewrites) == ["by_constant", "by_joined"], refused
    assert "_COMPUTED" in refused["computed"] and "columns" in refused["by_local"], refused
    olds = {rewrite.function: rewrite.old_sql for rewrite in rewrites}
    assert olds["by_constant"] == "SELECT geo_id, name FROM geos WHERE state = $1 AND geo_id = $2"
    assert olds["by_joined"] == "SELECT geo_id, name FROM geos WHERE state = $1 ORDER BY geo_id"

    apply_rewrites(tmp_path, rewrites, ruff=None)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    source = " ".join((tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8").split())
    assert (
        "read_row(conn, 'geos', by={'state': state, 'geo_id': geo}, columns=('geo_id', 'name'), records=True)" in source
    )
    assert (
        "read_by(conn, 'geos', by={'state': state}, columns=('geo_id', 'name'), records=True, order=[Sort('geo_id')])"
        in source
    )
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        assert_source_says(tmp_path, entry)
    left = [line for line in use_the_helper_findings(config) if "reads.py" in line]
    assert sorted(line.split(":")[1].split(" in ")[1] for line in left) == ["by_local", "computed"], left


_NULL_SAFE_KINDS = """
    from enum import Enum


    class Status(str, Enum):
        PROPOSED = "proposed"
        ACTIVE = "active"
"""

_NULL_SAFE_TABLES = """
    from threetears.core.collections.schema_backed import Index, TableSchema

    VERSIONS = TableSchema(
        name="versions",
        primary_key=("agent_id", "version_id"),
        columns=[],
        indexes=(Index("versions_pending", "agent_id", "user_id", where="status = 'proposed'"),),
    )
"""

_NULL_SAFE_READS = """
    from pkg.kinds import Status


    async def pending(conn, agent_id, user_id):
        return await conn.fetch(
            "SELECT version_id FROM versions WHERE agent_id = $1 AND user_id IS NOT DISTINCT FROM $2 "
            f"AND status = '{Status.PROPOSED.value}'",
            agent_id,
            user_id,
        )


    async def by_owner(conn, owner):
        return await conn.fetch("SELECT version_id FROM versions WHERE owner IS NOT DISTINCT FROM $1", owner)
"""


def test_null_safe_pins_and_text_enum_values_are_rewritten_onto_the_helper(tmp_path: Path) -> None:
    """Flips if a read the rule finds unkeyed only because a pin is null-safe (``IS NOT DISTINCT FROM``, which no
    index binds) is refused when the helper's spelling (``= $n``, ``IS NULL`` for None) reads a declared index, if an
    f-string interpolating a text enum member's value is not read as that literal, or if a null-safe read no index
    serves even so is taken."""
    config = _config(tmp_path)
    for name, source in {
        "kinds.py": _NULL_SAFE_KINDS,
        "tables.py": _NULL_SAFE_TABLES,
        "reads.py": _NULL_SAFE_READS,
    }.items():
        (tmp_path / "src/pkg" / name).write_text(textwrap.dedent(source), encoding="utf-8")
    rewrites, refusals = plan_point_rewrites(config)
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert [rewrite.function for rewrite in rewrites] == ["pending"], refused
    assert "unkeyed" in refused["by_owner"], refused
    (rewrite,) = rewrites
    assert rewrite.old_sql == (
        "SELECT version_id FROM versions WHERE agent_id = $1 AND user_id IS NOT DISTINCT FROM $2 "
        "AND status = 'proposed'"
    )
    apply_rewrites(tmp_path, rewrites, ruff=None)
    source = " ".join((tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8").split())
    assert "read_by(conn, 'versions', by={'agent_id': agent_id, 'user_id': user_id, 'status': 'proposed'}" in source, (
        source
    )


def test_a_select_list_expression_is_read_as_selected_over_the_helpers_row() -> None:
    """Flips if an aliased expression in the select list (a cast, a sub-select naming the row) is not read as the
    helper's ``Selected(expression, alias)`` with the row's own table or alias respelled ``t.`` (outside quoted text and
    other relations' qualifiers), or a sub-select there ends the select list early."""
    parsed = parse_point(
        "SELECT id, embedding::text AS embedding, (SELECT g.made FROM memories g WHERE g.id = memories.parent "
        "AND g.note = 'memories.x') AS parent_made FROM memories WHERE agent_id = $1"
    )
    assert isinstance(parsed, PointSql), parsed
    assert parsed.columns == (
        "id",
        {"expr": "embedding::text", "alias": "embedding"},
        {
            "expr": "(SELECT g.made FROM memories g WHERE g.id = t.parent AND g.note = 'memories.x')",
            "alias": "parent_made",
        },
    )
    aliased = parse_point("SELECT m.id, lower(m.name) AS name FROM memories m WHERE m.agent_id = $1")
    assert isinstance(aliased, PointSql), aliased
    assert aliased.columns == ("id", {"expr": "lower(t.name)", "alias": "name"})
    renamed = parse_point("SELECT a AS b FROM t WHERE a = $1")
    assert isinstance(renamed, PointSql) and renamed.columns == ({"expr": "a", "alias": "b"},)


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("SELECT lower(name) FROM t WHERE a = $1", "no alias"),
        ("SELECT coalesce(name, $2) AS name FROM t WHERE a = $1", "binding a parameter"),
        ("SELECT (SELECT max(x) FROM t WHERE x > 0) AS top FROM t WHERE a = $1", "reading t again unaliased"),
        ("SELECT u.name FROM t WHERE a = $1", "another relation"),
        ("SELECT COALESCE(MAX(version), 0) AS top FROM t WHERE a = $1", "an aggregate"),
    ],
)
def test_a_select_list_expression_the_helper_cannot_send_is_refused(sql: str, reason: str) -> None:
    """Flips if a select-list expression is taken that has no alias to answer under, binds a parameter the helper would
    not number, reads the row's own table again with no alias of its own (its ``<table>.`` would be the inner read's),
    names another relation, or aggregates (one row of the whole match: ``aggregate_keyed``'s, not a row each)."""
    refused = parse_point(sql)
    assert isinstance(refused, str) and reason in refused, refused


def test_a_selected_read_is_rewritten_with_selected_and_its_source_says_so(tmp_path: Path) -> None:
    """Flips if a read with an aliased expression in its select list is not rewritten onto its helper with
    ``Selected`` (imported beside it), or its manifest entry is not one the source check reads back."""
    src = tmp_path / "src"
    sources = {
        "pkg/tables.py": _TABLES,
        "pkg/reads.py": """
            async def geo_named(conn, state, geo):
                return await conn.fetch(
                    "SELECT geo_id, lower(name) AS name FROM geos WHERE state = $1 AND geo_id = $2", state, geo
                )
        """,
    }
    for name, source in sources.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    config = NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )
    rewrites, refusals = plan_point_rewrites(config)
    assert [rewrite.function for rewrite in rewrites] == ["geo_named"], refusals
    assert apply_rewrites(tmp_path, rewrites, ruff=None) == ["src/pkg/reads.py"]
    source = (tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8")
    assert "from threetears.core.keyed import Selected, read_by" in source, source
    assert "columns=('geo_id', Selected('lower(name)', 'name'))" in source, source
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    (entry,) = json.loads(manifest.read_text(encoding="utf-8"))
    assert entry["columns"] == ["geo_id", {"expr": "lower(name)", "alias": "name"}]
    assert_source_says(tmp_path, entry)


_CLAUSE_BUILDERS = """
    def tag_clause(read_filter, tags, args):
        if read_filter is None:
            return ""
        args.append(read_filter)
        return f" AND ({tags} ? ${len(args)})"


    def other_clause(read_filter, tags, args):
        return ""
"""

_CLAUSE_READS = """
    from pkg.filters import other_clause, tag_clause


    async def by_key(conn, read_filter, state, geo):
        args = [state, geo]
        allowed = tag_clause(read_filter, "geos.tags", args)
        return await conn.fetchrow(
            f"SELECT geo_id, name FROM geos WHERE state = $1 AND geo_id = $2{allowed}",
            *args,
        )


    async def aliased(conn, read_filter, state):
        args = [state]
        allowed = tag_clause(read_filter, "g.tags", args)
        return await conn.fetch(f"SELECT g.geo_id FROM geos g WHERE g.state = $1{allowed} ORDER BY g.geo_id", *args)


    async def list_read_again(conn, read_filter, state):
        args = [state]
        allowed = tag_clause(read_filter, "geos.tags", args)
        rows = await conn.fetch(f"SELECT geo_id FROM geos WHERE state = $1{allowed} ORDER BY geo_id", *args)
        return rows, len(args)


    async def other_row(conn, read_filter, state):
        args = [state]
        allowed = tag_clause(read_filter, "parents.tags", args)
        return await conn.fetch(f"SELECT geo_id FROM geos WHERE state = $1{allowed} ORDER BY geo_id", *args)


    async def undeclared(conn, read_filter, state):
        args = [state]
        allowed = other_clause(read_filter, "geos.tags", args)
        return await conn.fetch(f"SELECT geo_id FROM geos WHERE state = $1{allowed} ORDER BY geo_id", *args)
"""


def test_a_declared_clause_builders_answer_is_rewritten_as_its_guards(tmp_path: Path) -> None:
    """Flips if a statement interpolating a declared clause builder's answer (bound once over a list literal spread
    into the call) is not rewritten onto its helper with ``where=[*clause_guards(<builder>, <subject>, <row 't'>)]``,
    its list and clause statements removed and its manifest entry naming the builder, its subjects, its arguments as
    written and respelled, and where the clause stood; or if the list read elsewhere, a builder over another relation's
    row, or an undeclared builder is taken."""
    config = _config(tmp_path)
    for name, source in {"filters.py": _CLAUSE_BUILDERS, "reads.py": _CLAUSE_READS}.items():
        (tmp_path / "src/pkg" / name).write_text(textwrap.dedent(source), encoding="utf-8")
    config = NoWholeReadsConfig(
        repo_root=tmp_path,
        src_roots=config.src_roots,
        ledger_dir=tmp_path,
        dependency_schema=False,
        table_lists=(),
        clause_builders={"pkg.filters:tag_clause": "pkg.filters:subjects"},
    )
    rewrites = []
    refused: dict[str, str] = {}
    for planner in (plan_point_rewrites, plan_list_rewrites):
        planned, refusals = planner(config)
        rewrites.extend(planned)
        refused.update({refusal.function: refusal.reason for refusal in refusals})
    assert sorted(rewrite.function for rewrite in rewrites) == ["aliased", "by_key"], refused
    assert "list_read_again" in refused and "args" in refused["list_read_again"], refused
    assert "parents.tags" in refused["other_row"], refused
    assert "{allowed}" in refused["undeclared"], refused
    planned_by = {rewrite.function: rewrite for rewrite in rewrites}
    assert planned_by["by_key"].old_sql == "SELECT geo_id, name FROM geos WHERE state = $1 AND geo_id = $2"
    assert planned_by["by_key"].clauses == [
        {
            "build": "pkg.filters:tag_clause",
            "subjects": "pkg.filters:subjects",
            "leading": ["geos.tags"],
            "respelled": ["t.tags"],
            "at": len("SELECT geo_id, name FROM geos WHERE state = $1 AND geo_id = $2"),
            "call": "clause_guards(tag_clause, read_filter, 't.tags')",
        }
    ]
    assert planned_by["aliased"].clauses[0]["respelled"] == ["t.tags"]

    apply_rewrites(tmp_path, rewrites, ruff=None)
    text = (tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8")
    source = " ".join(text.split())
    assert "from threetears.core.keyed import Sort, clause_guards, read_by, read_row" in source, text
    assert (
        "read_row(conn, 'geos', by={'state': state, 'geo_id': geo}, "
        "where=[*clause_guards(tag_clause, read_filter, 't.tags')], columns=('geo_id', 'name'), records=True)"
    ) in source, text
    assert source.count("args = [state, geo]") == 0 and source.count("allowed = tag_clause") == 2, text
    ast.parse(text)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        assert entry["clauses"][0]["build"] == "pkg.filters:tag_clause"
        assert_source_says(tmp_path, entry)


def test_a_clause_builders_parent_seeder_is_recorded_beside_it(tmp_path: Path) -> None:
    """Flips if a clause builder declared with a parent seeder (``clause_parents``: its clause decides on another
    table's rows) is recorded without it, so the generated proof would seed no parent and both sides would read
    nothing under every subject; or if a builder declared without one gains a ``parents`` key."""
    config = _config(tmp_path)
    for name, source in {"filters.py": _CLAUSE_BUILDERS, "reads.py": _CLAUSE_READS}.items():
        (tmp_path / "src/pkg" / name).write_text(textwrap.dedent(source), encoding="utf-8")

    def planned(parents: dict[str, str]) -> dict[str, Any]:
        declared = NoWholeReadsConfig(
            repo_root=tmp_path,
            src_roots=config.src_roots,
            ledger_dir=tmp_path,
            dependency_schema=False,
            table_lists=(),
            clause_builders={"pkg.filters:tag_clause": "pkg.filters:subjects"},
            clause_parents=parents,
        )
        rewrites, _ = plan_point_rewrites(declared)
        (clause,) = next(rewrite for rewrite in rewrites if rewrite.function == "by_key").clauses
        return clause

    assert planned({"pkg.filters:tag_clause": "pkg.filters:parents"})["parents"] == "pkg.filters:parents"
    assert "parents" not in planned({})


_JOIN_TABLES = """
    from threetears.core.collections.schema_backed import TableSchema

    KEYS_SCHEMA = TableSchema(name="api_keys", primary_key="id", columns=[])
    CUSTOMERS_SCHEMA = TableSchema(name="customers", primary_key="id", columns=[])
    LINKS_SCHEMA = TableSchema(name="links", primary_key=("ref", "link", "relation_id"), columns=[])
    RELATIONS_SCHEMA = TableSchema(name="relations", primary_key="id", columns=[])
"""

_JOINS = """
    async def with_slug(pool, key_id):
        row = await pool.fetchrow(
            "SELECT ak.*, c.slug AS customer_slug FROM api_keys ak JOIN customers c ON c.id = ak.customer_id "
            "WHERE ak.id = $1",
            key_id,
        )
        return None if row is None else dict(row)


    async def relations_of(conn, ref):
        rows = await conn.fetch(
            "SELECT r.id, r.name FROM relations r JOIN links l ON l.relation_id = r.id "
            "WHERE l.ref = $1 AND l.link = 'member' ORDER BY r.id",
            ref,
        )
        return [dict(row) for row in rows]


    async def with_template(conn, key_id):
        row = await conn.fetchrow(
            "SELECT ak.id, c.name AS template FROM api_keys ak LEFT JOIN customers c ON c.id = ak.customer_id "
            "WHERE ak.id = $1",
            key_id,
        )
        return row["template"] if row else None


    async def by_place(conn, key_id):
        row = await conn.fetchrow(
            "SELECT ak.id, c.slug FROM api_keys ak JOIN customers c ON c.id = ak.customer_id WHERE ak.id = $1",
            key_id,
        )
        return row[1]


    async def by_slug(conn, slug):
        return await conn.fetch(
            "SELECT ak.id FROM api_keys ak JOIN customers c ON c.id = ak.customer_id WHERE c.slug = $1", slug
        )


    async def ordered_by_theirs(conn, key_id):
        rows = await conn.fetch(
            "SELECT ak.id, c.name FROM api_keys ak LEFT JOIN customers c ON c.id = ak.customer_id "
            "WHERE ak.id = $1 ORDER BY c.name",
            key_id,
        )
        return list(rows)
"""


def _join_config(tmp_path: Path) -> NoWholeReadsConfig:
    src = tmp_path / "src"
    for name, source in {"pkg/tables.py": _JOIN_TABLES, "pkg/joins.py": _JOINS}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, table_lists=()
    )


def test_a_point_read_with_joins_is_rewritten_onto_read_by_joined(tmp_path: Path) -> None:
    """Flips if a keyed read with joins is not rewritten as the read of the table its parameters pin and each other
    table joined to it (``read_by_joined``: a ``fetchrow`` its first row, a ``LEFT JOIN`` ``left=True``, a table the
    parameters pin on the other side of a ``JOIN`` read first and the ``FROM`` table joined back, an ``ORDER BY`` of
    a joined column equal to one of the read's ordered by that one), or if one whose rows are read by position, one
    ordered by a joined column, or one keyed by nothing is taken; or if the rewrite leaves a finding."""
    config = _join_config(tmp_path)
    before_nwr = no_whole_reads_findings(config)
    rewrites, refusals = plan_join_rewrites(config)
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert sorted(rewrite.function for rewrite in rewrites) == ["relations_of", "with_slug", "with_template"], refused
    assert "by position" in refused["by_place"], refused
    assert "ORDER BY of a joined" in refused["ordered_by_theirs"], refused
    calls = {rewrite.function: rewrite.new_call for rewrite in rewrites}
    assert calls["with_slug"] == (
        "next(iter(await read_by_joined(pool, 'api_keys', by={'id': key_id}, columns=None, "
        "joins=[Joined('customers', on={'id': 'customer_id'}, columns={'customer_slug': 'slug'})])), None)"
    )
    assert calls["relations_of"] == (
        "read_by_joined(conn, 'links', by={'ref': ref, 'link': 'member'}, columns=(), "
        "joins=[Joined('relations', on={'id': 'relation_id'}, columns={'id': 'id', 'name': 'name'})], "
        "order=[Sort('relation_id')])"
    )
    assert "left=True" in calls["with_template"]
    apply_rewrites(tmp_path, rewrites, ruff=None)
    source = (tmp_path / "src/pkg/joins.py").read_text(encoding="utf-8")
    assert "from threetears.core.keyed import Joined, Sort, read_by_joined" in source, source
    ast.parse(source)
    remaining = {hit.split(" in ")[1].split(":")[0] for hit in use_the_helper_findings(config) if "joins.py" in hit}
    assert remaining == {"by_place", "ordered_by_theirs", "by_slug"}, remaining
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before_nwr)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    entries = {entry["function"]: entry for entry in json.loads(manifest.read_text(encoding="utf-8"))}
    assert entries["relations_of"]["joins"][0]["on"] == {"id": "relation_id"}
    for entry in entries.values():
        assert_source_says(tmp_path, entry)


@pytest.mark.parametrize(
    ("sql", "refused"),
    [
        ("SELECT a.id, b.x FROM a JOIN b ON b.id = a.b_id JOIN c ON c.id = b.c_id WHERE a.id = $1", "another joined"),
        ("SELECT a.id, b.x FROM a LEFT JOIN b ON b.id = a.b_id WHERE a.id = $1 AND b.k = 'z'", "LEFT JOIN's table"),
        ("SELECT a.id, b.x FROM a JOIN b ON b.id = a.b_id WHERE a.id = $1 AND b.k = $2", "by a parameter"),
        ("SELECT a.id, upper(b.x) AS x FROM a JOIN b ON b.id = a.b_id WHERE a.id = $1", "an expression"),
        ("SELECT a.id, b.id FROM a JOIN b ON b.id = a.b_id WHERE a.id = $1", "under one name"),
        ("SELECT a.id, b.x FROM a JOIN b ON b.id = a.b_id WHERE a.id = $1 LIMIT 5", "LIMIT"),
        ("SELECT a.id, b.x FROM a RIGHT JOIN b ON b.id = a.b_id WHERE a.id = $1", "RIGHT"),
        ("SELECT a.id, b.x FROM a JOIN b ON b.id = a.b_id WHERE id = $1", "naming no table"),
        ("SELECT a.id, b.x FROM a LEFT JOIN b ON b.id = a.b_id WHERE b.k = $1", "no parameter pins the FROM"),
    ],
)
def test_a_joined_read_the_helper_cannot_send_is_refused(sql: str, refused: str) -> None:
    """Flips if a joined read ``read_by_joined`` would answer otherwise is parsed: a chain of joins, a LEFT JOIN's
    table pinned in the WHERE, a joined table pinned by a parameter, an expression, two items under one name, a
    LIMIT, a RIGHT JOIN, an unqualified column, a LEFT JOIN read from its right side."""
    parsed = parse_join(sql, guards=True)
    assert isinstance(parsed, str) and refused in parsed, parsed


def test_a_correlated_exists_is_kept_as_a_guard_over_the_helper_s_row() -> None:
    """Flips if a ``[NOT] EXISTS (SELECT ... FROM <table> <alias> WHERE ...)`` conjunct beside the pins is not kept as
    a guard (the outer table's qualifier respelled ``t.`` in a read, the table's own name in a write, which names it
    bare; its inner alias kept, parameters renumbered from ``$1``),
    or if one whose inner table is unaliased, which names an unqualified column, or another relation, is taken; or
    if a ``DELETE`` under an alias with a ``RETURNING`` of that alias's columns is not parsed as the keyed write."""
    deleted = parse_write(
        "DELETE FROM datasource_columns fresh WHERE fresh.datasource_id = ANY($1::uuid[]) AND fresh.schema_name = $2 "
        "AND fresh.table_name = $4 AND EXISTS ( SELECT 1 FROM datasource_columns kept WHERE kept.datasource_id = "
        "fresh.datasource_id AND kept.schema_name = $2 AND kept.table_name = $3 AND kept.column_name = "
        "fresh.column_name ) RETURNING fresh.id",
        guards=True,
    )
    assert isinstance(deleted, WriteSql), deleted
    assert deleted.table == "datasource_columns" and deleted.returning == ("id",)
    assert [pin.column for pin in deleted.pins] == ["datasource_id", "schema_name", "table_name"]
    assert [(guard.literal, guard.params, guard.compared) for guard in deleted.guards] == [
        (
            "EXISTS ( SELECT 1 FROM datasource_columns kept WHERE kept.datasource_id = datasource_columns.datasource_id "
            "AND kept.schema_name = $1 AND kept.table_name = $2 AND kept.column_name = datasource_columns.column_name )",
            (2, 3),
            ("schema_name", "table_name"),
        )
    ]
    absent = parse_point(
        "SELECT id FROM geos g WHERE g.state = $1 AND NOT EXISTS (SELECT 1 FROM grants k WHERE k.geo = g.id)",
        guards=True,
    )
    assert isinstance(absent, PointSql), absent
    assert [guard.literal for guard in absent.guards] == ["NOT EXISTS (SELECT 1 FROM grants k WHERE k.geo = t.id)"]
    for refused in (
        "SELECT id FROM geos g WHERE g.state = $1 AND EXISTS (SELECT 1 FROM grants WHERE grants.geo = g.id)",
        "SELECT id FROM geos g WHERE g.state = $1 AND EXISTS (SELECT 1 FROM grants k WHERE geo = g.id)",
        "SELECT id FROM geos g WHERE g.state = $1 AND EXISTS (SELECT 1 FROM grants k WHERE k.geo = x.id)",
        "SELECT id FROM geos g WHERE g.state = $1 AND EXISTS (SELECT 1 FROM grants g WHERE g.geo = 1)",
    ):
        assert isinstance(parse_point(refused, guards=True), str), refused
