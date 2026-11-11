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

import pytest

from threetears.core.testing.rewrites import assert_source_says
from threetears.enforcement.no_whole_reads import (
    NoWholeReadsConfig,
    no_whole_reads_findings,
    use_the_helper_findings,
)
from threetears.enforcement.no_whole_reads.rewrite import (
    PointSql,
    apply_rewrites,
    parse_point,
    plan_point_rewrites,
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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM t WHERE id = ANY($1::uuid[])",
        "SELECT id FROM t WHERE id IN ($1, $2)",
        "SELECT id FROM t WHERE a = $1 OR b = $1",
        "SELECT id FROM t WHERE a > $1",
        "SELECT id FROM t WHERE a = $1::uuid",
        "SELECT id FROM t WHERE lower(a) = $1",
        "SELECT id FROM t WHERE a = $1 ORDER BY id",
        "SELECT id FROM t WHERE a = $1 LIMIT 1",
        "SELECT count(*) FROM t WHERE a = $1",
        "SELECT a AS b FROM t WHERE a = $1",
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
        ("one_agent", "read_row"),
    ]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() == {"by_status", "by_ids", "interpolated"}, refused
    assert "unkeyed" in refused["by_status"]

    assert apply_rewrites(tmp_path, rewrites, ruff=None) == ["src/pkg/reads.py"]
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    write_manifest(manifest, rewrites)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(entries) == 4

    source = (tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "from threetears.core.keyed import read_by, read_row, read_value" in source
    assert "_GEO_SQL" not in source and "a geo by its key" not in source
    assert "read_row(conn, 'agents', by={'id': agent_id, 'is_deleted': False}, records=True)" in source
    assert "read_value(pool, 'agents', by={'id': agent_id}, column='name')" in source
    assert (
        "read_by(conn, 'geos', by={'state': state, 'geo_id': geo}, columns=('geo_id', 'name'), records=True)" in source
    )
    assert tree is not None
    for entry in entries:
        assert_source_says(tmp_path, entry)

    left = [finding for finding in use_the_helper_findings(config) if "use-the-helper point" in finding]
    assert {finding.split(" in ")[1].split(":")[0] for finding in left} == {"by_status", "by_ids", "interpolated"}
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


def test_a_constant_named_elsewhere_is_left(tmp_path: Path) -> None:
    """Flips if a statement constant another module names (an index purpose, an import) is rewritten away."""
    config = _config(tmp_path)
    (tmp_path / "src/pkg/purposes.py").write_text("# purpose: src/pkg/reads.py::_GEO_SQL\n", encoding="utf-8")
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
