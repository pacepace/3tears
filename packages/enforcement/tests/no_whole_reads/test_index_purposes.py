"""The index-purpose tool: a discovering walk credits each read to every index that would serve it, and the tool writes
the purpose, generated plan test and reason the purpose rule asks of each index, or lists it as a drop candidate."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings
from threetears.enforcement.no_whole_reads.index_purposes import main

_MIGRATION = """
    UP = (
        "CREATE TABLE IF NOT EXISTS items (id UUID PRIMARY KEY, owner_id UUID, kind TEXT, made_at TIMESTAMPTZ,"
        " note TEXT, CONSTRAINT items_owner_fk FOREIGN KEY (owner_id) REFERENCES owners (id))"
    )
    BY_KIND = "CREATE INDEX IF NOT EXISTS items_by_kind ON items (kind, made_at)"
    BY_OWNER = "CREATE INDEX IF NOT EXISTS items_by_owner ON items (owner_id)"
    BY_NOTE = "CREATE INDEX IF NOT EXISTS items_by_note ON items (note)"
"""

_READER = """
    class Items:
        async def latest_of_kind(self, conn, kind):
            return await conn.fetch("SELECT id FROM items WHERE kind = $1 ORDER BY made_at DESC LIMIT 10", kind)

        async def one(self, conn, item_id):
            return await conn.fetchrow("SELECT id, kind FROM items WHERE id = $1", item_id)
"""

_SUPPORT = """
    from pathlib import Path

    from threetears.enforcement.no_whole_reads import NoWholeReadsConfig

    ROOT = Path(__file__).resolve().parent
    CONFIG = NoWholeReadsConfig(
        repo_root=ROOT,
        src_roots=(ROOT / "src",),
        ledger_dir=ROOT,
        dependency_schema=False,
        frozen_migrations=("src/versions/v*.py",),
    )
"""


def _repo(tmp_path: Path) -> NoWholeReadsConfig:
    """a repo of one frozen migration, its reader, and the gate's config module."""
    for name, source in {
        "src/versions/v001.py": _MIGRATION,
        "src/items.py": _READER,
        "support.py": _SUPPORT,
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "tests" / "integration").mkdir(parents=True)
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    return NoWholeReadsConfig(
        repo_root=tmp_path,
        src_roots=(tmp_path / "src",),
        ledger_dir=tmp_path,
        dependency_schema=False,
        frozen_migrations=("src/versions/v*.py",),
    )


def _index(findings: list[str]) -> list[str]:
    return [finding for finding in findings if ": index-" in finding]


def test_the_tool_declares_each_used_index_and_lists_the_unused(tmp_path: Path, capsys: object) -> None:
    """Flips if the tool does not credit a read to the index whose columns it binds then orders by, does not give a
    foreign key's index its ``fk:`` purpose, gives an index no read or key uses a purpose, or writes declarations the
    gate does not accept (its generated plan tests name each index and call plan_uses_index)."""
    config = _repo(tmp_path)
    before = _index(no_whole_reads_findings(config))
    assert sum("index-no-purpose" in finding for finding in before) == 3, before
    drops = tmp_path / "drops.txt"

    assert main([str(tmp_path), "support.py", "CONFIG", "--write", "--drop-list", str(drops)]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    assert "items_by_kind :: on items :: purpose=src/items.py::Items.latest_of_kind" in ledger
    assert "items_by_owner :: on items :: purpose=fk:items_owner_fk" in ledger
    assert "items_by_note" not in ledger
    assert drops.read_text(encoding="utf-8").startswith("items.items_by_note")
    entries = {e["index"]: e for e in json.loads((tmp_path / "tests/integration/index_plan_manifest.json").read_text())}
    assert entries["items_by_kind"]["sql"] == 'SELECT * FROM "items" WHERE "kind" = $1 ORDER BY "made_at" LIMIT 10'
    assert entries["items_by_owner"]["sql"] == 'SELECT * FROM "items" WHERE "owner_id" = $1'
    after = _index(no_whole_reads_findings(config))
    assert [finding for finding in after if "items_by_note" not in finding] == [], after
