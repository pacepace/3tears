"""The schema the no-whole-reads rule judges is the migrations' in their own order: by each file's number, not its name
as text, so a later migration's drop or rename is applied after what it drops or renames."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_whole_reads.schema import collect_schema


def test_migrations_are_replayed_in_number_order_not_name_order(tmp_path: Path) -> None:
    """Flips if ``v1001_*`` is replayed before ``v118_*`` (``"v1001" < "v118"`` as text): its drop of an index ``v118``
    creates would be applied first, and the dropped index judged as declared."""
    migrations = tmp_path / "src" / "migrations"
    migrations.mkdir(parents=True)
    (migrations / "v118_results.sql").write_text(
        "CREATE TABLE results (k TEXT PRIMARY KEY, state TEXT, geo TEXT);\n"
        "CREATE INDEX results_by_state ON results (state);\n",
        encoding="utf-8",
    )
    (migrations / "v185_results_geo.sql").write_text(
        "CREATE INDEX results_by_geo ON results (geo);\n",
        encoding="utf-8",
    )
    (migrations / "v1001_results_contract.sql").write_text(
        "DROP INDEX IF EXISTS results_by_state;\nDROP INDEX IF EXISTS results_by_geo;\n",
        encoding="utf-8",
    )

    catalog = collect_schema(tmp_path, (tmp_path / "src",))

    assert {site.name for site in catalog.sites} == set(), "v1001 drops what v118 and v185 made"
