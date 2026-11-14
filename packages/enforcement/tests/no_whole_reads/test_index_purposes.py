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

#: a unique index no read uses, partial on a literal: its purpose is the rule it enforces
_UNIQUE = """
    UQ_REF = "CREATE UNIQUE INDEX IF NOT EXISTS uq_items_ref ON items (owner_id, note) WHERE kind = 'live'"
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


def test_a_repo_names_its_own_plan_test_module_and_the_declarations_name_it(tmp_path: Path) -> None:
    """Flips if ``--module`` does not move the generated plan tests and their manifest together, or the declarations
    still name the default module (a repo whose engine fixtures live under its unit tree, the hub's, needs this)."""
    config = _repo(tmp_path)
    module = "tests/unit/test_index_plans_generated.py"
    (tmp_path / "tests" / "unit").mkdir(parents=True)

    assert main([str(tmp_path), "support.py", "CONFIG", "--write", "--module", module]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    assert f"plan_test={module}::test_items_by_kind" in ledger
    assert "tests/integration/" not in ledger
    assert (tmp_path / module).is_file()
    assert not (tmp_path / "tests/integration/test_index_plans_generated.py").exists()
    entries = json.loads((tmp_path / "tests/unit/index_plan_manifest.json").read_text())
    assert {e["index"] for e in entries} == {"items_by_kind", "items_by_owner"}
    after = _index(no_whole_reads_findings(config))
    assert [finding for finding in after if "items_by_note" not in finding] == [], after


def test_a_unique_index_no_read_uses_is_declared_by_the_rule_it_enforces(tmp_path: Path) -> None:
    """Flips if a unique index no read uses is listed to drop (it holds a data rule) instead of declared
    ``purpose=constraint`` with a generated test that inserts a duplicate and asserts the database refuses it (calling
    ``raises``, partial on its own ``WHERE``'s literal), or if the gate does not accept that declaration."""
    config = _repo(tmp_path)
    (tmp_path / "src/versions/v002.py").write_text(textwrap.dedent(_UNIQUE), encoding="utf-8")
    drops = tmp_path / "drops.txt"

    assert main([str(tmp_path), "support.py", "CONFIG", "--write", "--drop-list", str(drops)]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    assert (
        "uq_items_ref :: on items :: purpose=constraint :: "
        "plan_test=tests/integration/test_index_plans_generated.py::test_uq_items_ref"
    ) in ledger
    assert "uq_items_ref" not in drops.read_text(encoding="utf-8")
    entries = {e["index"]: e for e in json.loads((tmp_path / "tests/integration/index_plan_manifest.json").read_text())}
    assert entries["uq_items_ref"]["refuses"] is True
    assert {pin["column"]: pin["kind"] for pin in entries["uq_items_ref"]["pins"]} == {
        "owner_id": "param",
        "note": "param",
        "kind": "literal",
    }
    module = (tmp_path / "tests/integration/test_index_plans_generated.py").read_text(encoding="utf-8")
    assert "pytest.raises(" in module and 'insert_duplicate(engine_conn, _ENTRIES["uq_items_ref"])' in module
    after = _index(no_whole_reads_findings(config))
    assert [finding for finding in after if "uq_items_ref" in finding] == [], after


def test_only_writes_the_named_indexes(tmp_path: Path) -> None:
    """Flips if ``--index`` does not keep the tool to the indexes it names (the rest of its verdicts printed, not
    written): an owner applies the proven ones and leaves an engine-red one for its drop proof."""
    _repo(tmp_path)

    assert main([str(tmp_path), "support.py", "CONFIG", "--write", "--index", "items_by_owner"]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    assert "items_by_owner :: on items" in ledger
    assert "items_by_kind" not in ledger
    entries = json.loads((tmp_path / "tests/integration/index_plan_manifest.json").read_text())
    assert [e["index"] for e in entries] == ["items_by_owner"]


#: a partial index of the open items by kind, and a reader of the open ones: only that reader states its predicate
_PARTIAL = """
    OPEN_BY_KIND = "CREATE INDEX IF NOT EXISTS items_open_by_kind ON items (kind, made_at) WHERE note IS NULL"
"""

_OPEN_READER = """
    class OpenItems:
        async def open_of_kind(self, conn, kind):
            return await conn.fetch(
                "SELECT id FROM items WHERE kind = $1 AND note IS NULL ORDER BY made_at DESC LIMIT 10", kind
            )
"""


def test_a_partial_index_is_credited_only_to_reads_stating_its_predicate(tmp_path: Path) -> None:
    """Flips if the discovering walk credits a partial index to a read whose ``WHERE`` does not state the index's
    predicate (the planner cannot read a partial index for rows it may not hold: idx_intentions_open_ranked credited to
    find_by_user, 3tears #86), or stops crediting it to a read that does."""
    _repo(tmp_path)
    (tmp_path / "src/versions/v002.py").write_text(textwrap.dedent(_PARTIAL), encoding="utf-8")
    drops = tmp_path / "drops.txt"

    assert main([str(tmp_path), "support.py", "CONFIG", "--write", "--drop-list", str(drops)]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    assert "items_open_by_kind" not in ledger, ledger
    assert "items.items_open_by_kind" in drops.read_text(encoding="utf-8")

    (tmp_path / "src/open_items.py").write_text(textwrap.dedent(_OPEN_READER), encoding="utf-8")
    assert main([str(tmp_path), "support.py", "CONFIG", "--write", "--drop-list", str(drops)]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    line = next(entry for entry in ledger.splitlines() if entry.startswith("items_open_by_kind "))
    assert "src/open_items.py::OpenItems.open_of_kind" in line, line
    assert "Items.latest_of_kind" not in line, line


#: a partial index of the live items by owner, and a keyed helper's reads of them: by a literal pin, and by a bound one
_LIVE = """
    LIVE_BY_OWNER = "CREATE INDEX IF NOT EXISTS items_live_by_owner ON items (owner_id, made_at) WHERE kind = 'live'"
"""

_HELPER_READER = """
    from threetears.core.keyed import Sort, read_by

    class LiveItems:
        async def live_of_owner(self, conn, owner_id):
            return await read_by(
                conn, "items", by={"owner_id": owner_id, "kind": "live"}, order=[Sort("made_at", descending=True)]
            )

        async def of_owner_and_kind(self, conn, owner_id, kind):
            return await read_by(
                conn, "items", by={"owner_id": owner_id, "kind": kind}, order=[Sort("made_at", descending=True)]
            )
"""


def test_a_keyed_helper_s_literal_pin_states_a_partial_index_s_predicate(tmp_path: Path) -> None:
    """Flips if a keyed helper's literal pin (``by={"kind": "live"}``) does not state a partial index's predicate
    (``WHERE kind = 'live'``: find_pending's ``status: "proposed"`` over idx_identity_pending, 3tears #86), or if a
    pin bound at run time (``by={"kind": kind}``) is taken to state it."""
    _repo(tmp_path)
    (tmp_path / "src/versions/v002.py").write_text(textwrap.dedent(_LIVE), encoding="utf-8")
    (tmp_path / "src/live_items.py").write_text(textwrap.dedent(_HELPER_READER), encoding="utf-8")

    assert main([str(tmp_path), "support.py", "CONFIG", "--write"]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    line = next(entry for entry in ledger.splitlines() if entry.startswith("items_live_by_owner "))
    assert "src/live_items.py::LiveItems.live_of_owner" in line, line
    assert "of_owner_and_kind" not in line, line
    # a partial index serves no foreign key: the lookup of a deleted owner's items asks every item, live or not
    assert "fk:" not in line, line


_NOTED = """
    NOTED = "CREATE INDEX IF NOT EXISTS items_noted ON items (note, made_at) WHERE note IS NOT NULL"
"""

_NOTE_READER = """
    class Notes:
        async def by_note(self, conn, note):
            return await conn.fetch("SELECT id FROM items WHERE note = $1 ORDER BY made_at LIMIT 10", note)
"""


def test_a_value_comparison_states_a_not_null_predicate(tmp_path: Path) -> None:
    """Flips if a read comparing a column to a value (``note = $1``) is not credited to a partial index whose predicate
    is only that the column is not NULL (``WHERE note IS NOT NULL``): no NULL equals a value."""
    _repo(tmp_path)
    (tmp_path / "src/versions/v002.py").write_text(textwrap.dedent(_NOTED), encoding="utf-8")
    (tmp_path / "src/notes.py").write_text(textwrap.dedent(_NOTE_READER), encoding="utf-8")

    assert main([str(tmp_path), "support.py", "CONFIG", "--write"]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    line = next(entry for entry in ledger.splitlines() if entry.startswith("items_noted "))
    assert "src/notes.py::Notes.by_note" in line, line


_EITHER = """
    EITHER = "CREATE INDEX IF NOT EXISTS items_open_either ON items (kind) WHERE note IS NULL OR made_at IS NULL"
    LIVE = "CREATE INDEX IF NOT EXISTS items_live_by_owner ON items (owner_id, made_at) WHERE kind = 'live'"
"""

_EITHER_READER = """
    from enum import StrEnum

    from threetears.core.keyed import Sort, read_by

    class Kind(StrEnum):
        LIVE = "live"

    class Either:
        async def unnoted(self, conn, kind):
            return await conn.fetch("SELECT id FROM items WHERE kind = $1 AND note IS NULL", kind)

        async def live(self, conn, owner_id):
            return await read_by(
                conn, "items", by={"owner_id": owner_id, "kind": Kind.LIVE.value}, order=[Sort("made_at")]
            )
"""


def test_a_read_states_an_or_predicate_by_one_arm_and_a_pin_by_an_enum_value(tmp_path: Path) -> None:
    """Flips if a read stating one arm of a partial index's ``OR`` predicate (``note IS NULL`` of ``note IS NULL OR
    made_at IS NULL``) is not credited to it, or a keyed helper's pin to a text enum member's value
    (``ScopeType.TYPE_CUSTOMER.value``) does not state the literal predicate it names (3tears #87)."""
    _repo(tmp_path)
    (tmp_path / "src/versions/v002.py").write_text(textwrap.dedent(_EITHER), encoding="utf-8")
    (tmp_path / "src/either.py").write_text(textwrap.dedent(_EITHER_READER), encoding="utf-8")

    assert main([str(tmp_path), "support.py", "CONFIG", "--write"]) == 0

    ledger = (tmp_path / "_index_declarations.txt").read_text(encoding="utf-8")
    either = next(entry for entry in ledger.splitlines() if entry.startswith("items_open_either "))
    assert "src/either.py::Either.unnoted" in either, either
    live = next(entry for entry in ledger.splitlines() if entry.startswith("items_live_by_owner "))
    assert "src/either.py::Either.live" in live, live
