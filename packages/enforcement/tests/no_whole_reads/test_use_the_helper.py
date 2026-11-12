"""use-the-helper: a statement written as text against a data table outside 3tears' helpers is a finding.

Each test synthesises a small repo and runs the rule over it with the gate's own walk, both ways: the statement a
module hand-writes is named, with its shape; the same read through a keyed helper, in a sanctioned helper module,
against a local copy or a change set, or ledgered with its count and reason, is not; a ledger line that no longer
matches the function is.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, use_the_helper_counts, use_the_helper_findings
from threetears.enforcement.no_whole_reads.sql import data_relations, statement_shape

#: the ledger's name (``use_the_helper._LEDGER``), as a repo writes it
USE_THE_HELPER_LEDGER = "_use_the_helper_exemptions.txt"

_TABLES = """
    from threetears.core.collections.schema_backed import TableSchema

    GEOS = "geos"
    GEOS_SCHEMA = TableSchema(name=GEOS, primary_key=("state", "geo_id"), columns=[])
    RUNS_SCHEMA = TableSchema(name="runs", primary_key="run_id", columns=[])
"""

_HAND_WRITTEN = """
    from threetears.core.keyed import read_keyed

    from pkg.tables import GEOS


    async def one_geo(conn, state, geo):
        return await conn.fetch("SELECT * FROM geos WHERE state = $1 AND geo_id = $2", state, geo)


    async def a_page(conn, state):
        return await conn.fetch("SELECT * FROM geos WHERE state = $1 ORDER BY geo_id LIMIT 100", state)


    async def how_many(conn, state):
        return await conn.fetchval("SELECT count(*) FROM geos WHERE state = $1", state)


    async def with_runs(conn, run):
        return await conn.fetch(
            "SELECT g.geo_id FROM geos g JOIN runs r ON r.run_id = g.run_id WHERE r.run_id = $1", run
        )


    async def drop_state(conn, state):
        await conn.execute("DELETE FROM geos WHERE state = $1", state)


    async def any_table(conn, table, state):
        return await conn.fetch(f"SELECT * FROM {table} WHERE state = $1", state)


    async def through_the_helper(conn, keys):
        return await read_keyed(conn, GEOS, keys=keys, columns=["geo_id"])


    async def a_copy(snapshot):
        return snapshot.ask(f"SELECT * FROM {snapshot.relation(GEOS)} WHERE state = 'TX'")


    async def a_stage(db, state):
        return db.execute("SELECT * FROM __stage_geos WHERE state = ?", [state])


    async def from_a_stage(db):
        db.execute("INSERT INTO geos SELECT * FROM __stage_geos")


    async def a_plain_insert(conn, state, geo):
        await conn.execute("INSERT INTO geos (state, geo_id) VALUES ($1, $2)", state, geo)


    async def the_catalog(conn):
        return await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = $1", "x")
"""

_HELPER_MODULE = """
    async def read_rows(conn, state):
        return await conn.fetch("SELECT * FROM geos WHERE state = $1", state)
"""


def _config(tmp_path: Path, modules: dict[str, str], *, ledger: str = "", **options: object) -> NoWholeReadsConfig:
    """write the modules under ``tmp_path/src`` and the use-the-helper ledger beside them, and configure the gate.

    :param tmp_path: the repo root
    :ptype tmp_path: Path
    :param modules: module path onto its source
    :ptype modules: dict[str, str]
    :param ledger: the ledger's body
    :ptype ledger: str
    :param options: other config fields
    :ptype options: object
    :return: the config
    :rtype: NoWholeReadsConfig
    """
    src = tmp_path / "src"
    for name, source in modules.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / USE_THE_HELPER_LEDGER).write_text(textwrap.dedent(ledger), encoding="utf-8")
    options.setdefault("dependency_schema", False)
    options.setdefault("table_lists", ())
    return NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, **options)  # type: ignore[arg-type]


def _named(findings: list[str]) -> dict[str, str]:
    """each finding's function onto its ``<shape> <table>``."""
    named: dict[str, str] = {}
    for finding in findings:
        if ": use-the-helper " in finding:
            where, rest = finding.split(": use-the-helper ", 1)
            named[where.split(" in ", 1)[1]] = rest.split(" -- ", 1)[0]
    return named


def test_a_statement_written_as_text_against_a_data_table_is_named_with_its_shape(tmp_path: Path) -> None:
    """Flips if a hand-written point read, page, count, join, write (a change set's rows written into a table too),
    read of a table named at run time or read of only a change set (its home is core.keyed's steps, Pace 2026-11-10)
    is not a finding with its shape, or if a keyed helper's read, a local copy's, a plain insert or a catalog read is."""
    config = _config(
        tmp_path, {"pkg/tables.py": _TABLES, "pkg/reads.py": _HAND_WRITTEN}, transient_relations=r"^__stage_\w+$"
    )
    named = _named(use_the_helper_findings(config))
    assert named == {
        "one_geo": "point geos",
        "a_page": "list geos",
        "how_many": "count geos",
        "with_runs": "join geos",
        "drop_state": "write geos",
        "from_a_stage": "write geos",
        "any_table": "point {}",
        "a_stage": "point change:__stage_geos",
    }, named
    counts = use_the_helper_counts(config)
    assert counts == {
        "point": 3,
        "list": 1,
        "count": 1,
        "join": 1,
        "write": 2,
        "named-at-run-time": 1,
        "total": 8,
    }, counts


def test_a_helper_module_writes_its_sql_and_a_repo_names_its_own(tmp_path: Path) -> None:
    """Flips if a statement in one of 3tears' helper modules (``threetears/core/keyed/``) or a repo's reviewed
    helper module (``helper_modules``) is a finding, or if the same statement elsewhere is not."""
    modules = {
        "pkg/tables.py": _TABLES,
        "threetears/core/keyed/rows.py": _HELPER_MODULE,
        "pkg/store/rows.py": _HELPER_MODULE,
        "pkg/other.py": _HELPER_MODULE,
    }
    bare = use_the_helper_findings(_config(tmp_path, modules))
    assert {line.split(":", 1)[0] for line in bare} == {"src/pkg/store/rows.py", "src/pkg/other.py"}, bare
    reviewed = use_the_helper_findings(_config(tmp_path, modules, helper_modules=("src/pkg/store/",)))
    assert {line.split(":", 1)[0] for line in reviewed} == {"src/pkg/other.py"}, reviewed


def test_the_ledger_pins_each_functions_statements_and_its_reason(tmp_path: Path) -> None:
    """Flips if a ledgered function's statements are still findings, if a line whose count no longer matches (or
    whose function writes none) passes, or if a short reason or a malformed line is accepted."""
    modules = {"pkg/tables.py": _TABLES, "pkg/reads.py": _HAND_WRITTEN}
    reason = "no helper reads a join of geos and runs yet: keyed.read_joined over runs"
    ledger = f"""
        src/pkg/reads.py :: with_runs :: 1 :: {reason}
        src/pkg/reads.py :: drop_state :: 2 :: {reason}
        src/pkg/reads.py :: through_the_helper :: 1 :: {reason}
        src/pkg/reads.py :: how_many :: 1 :: too short
        src/pkg/reads.py :: a_page :: 1
    """
    findings = use_the_helper_findings(_config(tmp_path, modules, ledger=ledger))
    named = _named(findings)
    assert "with_runs" not in named, named
    assert "drop_state" not in named, named
    assert "how_many" in named and "a_page" in named, named
    defects = [line for line in findings if line.startswith(USE_THE_HELPER_LEDGER)]
    assert any("drop_state pins 2 statement(s), the walk finds 1" in line for line in defects), defects
    assert any("through_the_helper pins 1 statement(s), the walk finds 0" in line for line in defects), defects
    assert any("the reason must say why" in line for line in defects), defects
    assert any("expected 4 fields" in line for line in defects), defects


@pytest.mark.parametrize(
    ("sql", "shape"),
    [
        ("SELECT a FROM t WHERE k = $1", "point"),
        ("SELECT a FROM t WHERE k = $1 LIMIT 1", "point"),
        ("SELECT a FROM t WHERE k = $1 ORDER BY a", "list"),
        ("SELECT a FROM t WHERE k > $1 LIMIT 50", "list"),
        ("SELECT a FROM t", "list"),
        ("SELECT count(*) FROM t WHERE k = $1", "count"),
        ("SELECT EXISTS (SELECT 1 FROM t WHERE k = $1)", "count"),
        ("SELECT k, max(v) FROM t WHERE k = ANY($1) GROUP BY k", "count"),
        ("SELECT a FROM t, u WHERE t.k = u.k AND t.k = $1", "join"),
        ("SELECT a FROM t LEFT JOIN u ON u.k = t.k WHERE t.k = $1", "join"),
        ("WITH x AS (SELECT k FROM t WHERE k = $1) SELECT a FROM x WHERE a = $2", "point"),
        ("UPDATE t SET a = $2 WHERE k = $1", "write"),
        ("DELETE FROM t WHERE k = $1", "write"),
        ("INSERT INTO t (a) SELECT a FROM u WHERE k = $1", "write"),
        ("WITH gone AS (SELECT k FROM u) DELETE FROM t USING gone WHERE t.k = gone.k", "write"),
        ("INSERT INTO t (a) VALUES ($1)", None),
        ("CREATE VIEW v AS SELECT a FROM t", None),
    ],
)
def test_a_statements_shape(sql: str, shape: str | None) -> None:
    """Flips if a statement is put in the wrong backlog bucket."""
    assert statement_shape(sql) == shape


def test_a_statements_data_relations_leave_out_ctes_catalogs_and_functions() -> None:
    """Flips if a CTE, a catalog, a function of bound values or a subquery is read as a data table."""
    sql = (
        "WITH x AS (SELECT k FROM t WHERE k = $1) SELECT a FROM x JOIN u ON u.k = x.k "
        "JOIN unnest($2::text[]) AS w(k) ON w.k = u.k JOIN pg_class c ON c.oid = 1 JOIN (SELECT 1) s ON true"
    )
    assert data_relations(sql) == ["t", "u"]


_LOCAL = """
    def copy_rows(db, name, state):
        return db.execute(f"SELECT * FROM main.{name} WHERE state = ?", [state])


    def _view_sql(view):
        return f"SELECT * FROM {view} WHERE state = 'TX'"


    def loose(db, table, state):
        return db.execute(f"SELECT * FROM {table} WHERE state = ?", [state])
"""


def test_a_local_copys_relation_and_the_l1_storage_layer_are_not_counted(tmp_path: Path) -> None:
    """Flips if a ``main.{}`` the repo's copy declarations key, a run-time name in a declared view producer, or a
    statement in 3tears' L1 storage layer (``threetears/core/cache/``) is counted, or if a run-time name the rule
    cannot resolve stops being."""
    modules = {"pkg/tables.py": _TABLES, "pkg/local.py": _LOCAL, "threetears/core/cache/duck.py": _HELPER_MODULE}
    config = _config(tmp_path, modules, local_relations={"*": ("state",)}, view_producers={"_view_sql": ("state",)})
    assert _named(use_the_helper_findings(config)) == {"loose": "point {}"}
    undeclared = _config(tmp_path, modules)
    assert set(_named(use_the_helper_findings(undeclared))) == {"copy_rows", "_view_sql", "loose"}


_CHANGE_SQL = """
    from threetears.core.keyed import Match, change_relation, read_step, semi

    _CHANGED = change_relation("changed_runs")


    async def by_hand(db):
        db.register("held_rows", rows)
        return db.execute("SELECT h.* FROM held_rows AS h SEMI JOIN changed_runs AS k ON h.run_id = k.run_id")


    async def through_the_step(db):
        db.register("held_rows", rows)
        return await read_step(db, semi("held_rows", _CHANGED, on=[Match("run_id")]))
"""


def test_hand_written_sql_over_a_change_relation_is_named_and_its_step_is_not(tmp_path: Path) -> None:
    """Flips if SQL written by hand over a registered or declared change relation is not a use-the-helper finding
    naming the relation, or if the same computation through core.keyed's steps is."""
    config = _config(tmp_path, {"pkg/settle.py": _CHANGE_SQL})
    assert _named(use_the_helper_findings(config)) == {"by_hand": "join change:held_rows"}


def test_a_hand_written_count_or_sum_is_pointed_at_the_keyed_aggregate(tmp_path: Path) -> None:
    """Flips if a hand-written count (a sum over a window, a count of a tenant set) is not told its home,
    aggregate_keyed, or a point read is told it."""
    config = _config(tmp_path, {"pkg/tables.py": _TABLES, "pkg/reads.py": _HAND_WRITTEN})
    findings = [finding for finding in use_the_helper_findings(config) if ": use-the-helper " in finding]
    counted = [finding for finding in findings if "use-the-helper count " in finding]
    assert counted and all("aggregate_keyed" in finding.rsplit(" -- ", 1)[1] for finding in counted)
    assert all("aggregate_keyed" not in finding for finding in findings if finding not in counted)
