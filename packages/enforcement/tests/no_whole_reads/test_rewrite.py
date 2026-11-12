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
    Pin,
    PointSql,
    WriteSql,
    apply_rewrites,
    parse_point,
    parse_write,
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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM t WHERE id = ANY($1::uuid[]) OR a = $2",
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
        ("by_ids", "read_by"),
        ("live_one", "read_row"),
        ("one_agent", "read_row"),
    ]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() == {"by_status", "interpolated"}, refused
    assert "unkeyed" in refused["by_status"]

    assert apply_rewrites(tmp_path, rewrites, ruff=None) == ["src/pkg/reads.py"]
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rewrites)
    write_manifest(manifest, rewrites)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(entries) == 6

    source = (tmp_path / "src/pkg/reads.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "from threetears.core.keyed import AnyOf, read_by, read_row, read_value" in source
    assert "read_by(conn, 'agents', by={'id': AnyOf(ids, cast='uuid')}, columns=('id',), records=True)" in source
    assert (
        "read_row(conn, 'agents', by={'id': agent_id, 'live': True, 'gone': False}, columns=('id',), records=True)"
        in source
    )
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
    assert {finding.split(" in ")[1].split(":")[0] for finding in left} == {"by_status", "interpolated"}
    assert _unlined(no_whole_reads_findings(config)) == _unlined(before)


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


_WRITES = """
    #: forget one geo
    _FORGET_SQL = "DELETE FROM geos WHERE state = $1 AND geo_id = $2"


    async def rename(conn, agent_id, name):
        await conn.execute("UPDATE agents SET name = $2, date_updated = now(), flag = TRUE WHERE id = $1", agent_id, name)


    async def forget(conn, state, geo):
        await conn.execute(_FORGET_SQL, state, geo)


    async def renamed_count(conn, agent_id, name):
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
    a new whole-read finding; or a write whose status is read, an increment, a RETURNING * or a RETURNING sent by
    execute, or an unkeyed write is rewritten."""
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
        ("returned", "delete_keyed"),
    ]
    refused = {refusal.function: refusal.reason for refusal in refusals}
    assert refused.keys() >= {
        "renamed_count",
        "by_status",
        "from_elsewhere",
        "returned_all",
        "returned_by_execute",
    }, refused
    assert "status is read" in refused["renamed_count"]
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
    for entry in entries:
        assert_source_says(tmp_path, entry)

    left = [finding for finding in use_the_helper_findings(config) if "use-the-helper write" in finding]
    assert not [
        finding
        for finding in left
        if any(
            f" in {function}:" in finding
            for function in ("rename", "forget", "returned", "claimed", "moved", "rename_many", "bumped", "raised")
        )
    ], left
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
