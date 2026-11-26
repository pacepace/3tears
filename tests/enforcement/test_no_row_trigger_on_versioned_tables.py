"""enforcement: no source creates a per-row trigger on a versioned (page-written) table.

Append-only L3 chunk 7 (``design-append-only-l3.md`` decision 4): on YugabyteDB a row trigger switches off a table's
write batching, and every row then pays its own read and flush (1.58 ms a row against 0.43 without, 2026-11-25). A
versioned table is its own change record and needs none. The runtime half refuses one found in the catalog
(``threetears.core.ddl.epoch_catalog``); this rule (``threetears.enforcement.row_triggers``) finds the source that would
create one. A consumer runs the same rule over its own tree.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.row_triggers import row_triggers_in, row_triggers_on_versioned, versioned_tables_in

_REPO = Path(__file__).resolve().parents[2]


def _source() -> list[Path]:
    """every Python and SQL file of the packages' source (tests and fixtures left out)."""
    return sorted(
        path
        for pattern in ("packages/**/src/**/*.py", "packages/**/src/**/*.sql")
        for path in _REPO.glob(pattern)
        if "__pycache__" not in path.parts
    )


def test_no_source_creates_a_per_row_trigger_on_a_versioned_table() -> None:
    """Flips if any package's source creates a FOR EACH ROW trigger on a table it declares versioned."""
    found = row_triggers_on_versioned(_source())
    assert not found, "per-row triggers on versioned tables (drop them; see epoch_catalog.PerRowTriggerError):\n" + (
        "\n".join(str(each) for each in found)
    )


def test_the_rule_finds_a_per_row_trigger_on_a_declared_versioned_table(tmp_path: Path) -> None:
    """Flips if the rule misses a per-row trigger, written plain, as an f-string or in a .sql file, on a table declared
    versioned; or flags a statement-level trigger or one on a table not versioned."""
    schema = tmp_path / "tables.py"
    schema.write_text(
        "from threetears.core.collections.schema_backed import TableSchema\n"
        "from threetears.core.ddl.row_epoch import ROW_STAMPED, row_versions\n"
        "RESULTS = TableSchema(name='results', primary_key=('id',), columns=[],"
        " row_epoch=row_versions('state', epochs='e'))\n"
        "RUNS = TableSchema(name='runs', primary_key=('id',), columns=[], row_epoch=ROW_STAMPED)\n",
        encoding="utf-8",
    )
    migration = tmp_path / "v001.py"
    migration.write_text(
        "SCHEMA = 'rpt'\n"
        "A = 'CREATE TRIGGER audit AFTER INSERT ON rpt.\"results\" FOR EACH ROW EXECUTE FUNCTION f()'\n"
        "B = f'CREATE OR REPLACE TRIGGER stamp BEFORE UPDATE ON {SCHEMA}.results\\n FOR EACH ROW EXECUTE FUNCTION g()'\n"
        "C = 'CREATE TRIGGER whole AFTER TRUNCATE ON results FOR EACH STATEMENT EXECUTE FUNCTION h()'\n"
        "D = 'CREATE TRIGGER inplace AFTER UPDATE ON runs FOR EACH ROW EXECUTE FUNCTION k()'\n",
        encoding="utf-8",
    )
    sql = tmp_path / "v002.sql"
    sql.write_text("SELECT 1;\nCREATE TRIGGER t2 AFTER DELETE ON results\nFOR EACH ROW EXECUTE FUNCTION f();\n")
    files = [schema, migration, sql]
    assert versioned_tables_in(files) == frozenset({"results"})
    assert {(each.trigger, each.table) for each in row_triggers_in(files)} == {
        ("audit", "results"),
        ("stamp", "results"),
        ("inplace", "runs"),
        ("t2", "results"),
    }
    found = row_triggers_on_versioned(files)
    assert [(each.path.name, each.trigger) for each in found] == [
        ("v001.py", "audit"),
        ("v001.py", "stamp"),
        ("v002.sql", "t2"),
    ]
    assert next(each for each in found if each.path == sql).line == 2
