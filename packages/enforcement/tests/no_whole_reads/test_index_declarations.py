"""The index declarations ledger: an index whose DDL sits in a frozen migration declares its purpose, plan test and
reason in ``_index_declarations.txt``, judged by the same checks as comments beside its DDL, plus the ledger's own."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings

_MIGRATION = """
    UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT, customer_id UUID)"
    BY_KIND = "CREATE INDEX IF NOT EXISTS datasources_by_kind ON datasources (kind)"
"""

_INLINE = """
    UP = "CREATE TABLE IF NOT EXISTS datasources (id UUID PRIMARY KEY, kind TEXT, status TEXT, customer_id UUID)"
    BY_KIND = (
        "-- purpose: src/registry.py::Digest.refresh_active\\n"
        "-- plan_test: tests/test_plans.py::test_refresh_active_uses_datasources_by_kind\\n"
        "CREATE INDEX IF NOT EXISTS datasources_by_kind ON datasources (kind)"
    )
"""

_REGISTRY = """
    class Digest:
        async def one(self, conn, datasource_id):
            return await conn.fetchrow("SELECT id, kind FROM datasources WHERE id = $1", datasource_id)

        async def refresh_active(self, conn, kind):
            return await conn.fetch("SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1", kind)
"""

_PLANS = """
    from threetears.core.testing.plans import explain_statement, plan_uses_index

    async def test_refresh_active_uses_datasources_by_kind(conn):
        sql = "SELECT id, customer_id FROM datasources WHERE status = 'active' AND kind = $1"
        assert plan_uses_index(await explain_statement(conn, sql, "pg"), "datasources_by_kind")
"""

_LINE = (
    "datasources_by_kind :: on datasources :: purpose=src/registry.py::Digest.refresh_active :: "
    "plan_test=tests/test_plans.py::test_refresh_active_uses_datasources_by_kind"
)

_FROZEN = ("src/versions/v001.py",)


def _findings(
    tmp_path: Path, ledger: str | None, *, migration: str = _MIGRATION, frozen: tuple[str, ...] = _FROZEN
) -> list[str]:
    """the gate's findings over a migration, a reader and its plan test, with ``ledger`` as the declarations."""
    src = tmp_path / "src"
    for name, source in {"versions/v001.py": migration, "registry.py": _REGISTRY}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_plans.py").write_text(textwrap.dedent(_PLANS), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    if ledger is not None:
        (tmp_path / "_index_declarations.txt").write_text(f"# frozen v001\n{ledger}\n", encoding="utf-8")
    config = NoWholeReadsConfig(
        repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False, frozen_migrations=frozen
    )
    return no_whole_reads_findings(config)


def _index(findings: list[str]) -> list[str]:
    return [finding for finding in findings if ": index-" in finding]


def test_without_a_declaration_the_frozen_index_has_no_purpose_and_its_read_is_whole(tmp_path: Path) -> None:
    """Flips if an index in a frozen migration with no declaration anywhere passes, or its read is keyed by it."""
    findings = _findings(tmp_path, None)
    assert any("index datasources_by_kind on datasources: index-no-purpose" in f for f in findings), findings
    assert any("in Digest.refresh_active: non-key-where datasources" in f for f in findings), findings


def test_a_correct_frozen_entry_declares_the_index_and_keys_its_read(tmp_path: Path) -> None:
    """Flips if a ledger line for an index in a frozen migration, naming the read that uses it and its plan test, does
    not clear index-no-purpose, or leaves that read a whole read."""
    findings = _findings(tmp_path, _LINE)
    assert _index(findings) == [], findings
    assert not [f for f in findings if "Digest.refresh_active" in f], findings


def test_an_entry_for_an_index_in_no_frozen_migration_is_refused(tmp_path: Path) -> None:
    """Flips if the ledger may declare an index whose DDL can still be edited (no frozen migration holds it)."""
    findings = _findings(tmp_path, _LINE, frozen=())
    assert any("index-ledger-not-frozen" in f for f in findings), findings
    assert any("index-no-purpose" in f for f in findings), "a refused line declared the index anyway"


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        (_LINE.replace("datasources_by_kind ::", "datasources_by_gone ::"), "index-ledger-stale"),
        (_LINE.replace("on datasources", "on other_table"), "index-ledger-stale"),
        (_LINE.replace("Digest.refresh_active", "Digest.one"), "index-purpose-unused"),
        (_LINE.replace("Digest.refresh_active", "Digest.gone"), "index-purpose-unknown"),
        (_LINE.replace("test_refresh_active_uses_datasources_by_kind", "test_not_there"), "index-plan-test"),
        (_LINE.replace(" :: purpose=", " purpose="), "index-ledger-malformed"),
    ],
)
def test_a_wrong_entry_is_reported(tmp_path: Path, line: str, kind: str) -> None:
    """Flips if a stale entry (an index no migration creates, or on another table), a purpose whose function holds no
    keyed read by the index or names no function, a plan test that is not there, or a malformed line passes."""
    findings = _findings(tmp_path, line)
    assert any(kind in f for f in findings), findings


def test_an_index_declared_inline_and_in_the_ledger_is_declared_twice(tmp_path: Path) -> None:
    """Flips if an index carrying its comments beside its DDL may be declared in the ledger too."""
    findings = _findings(tmp_path, _LINE, migration=_INLINE)
    assert any("index-declared-twice" in f for f in findings), findings


def test_a_crowded_or_wide_index_s_reason_comes_from_its_line(tmp_path: Path) -> None:
    """Flips if a ledger line's reason is not the index's reason (a short one passes a wide index)."""
    wide = _MIGRATION.replace("(kind)", "(kind) INCLUDE (customer_id)")
    assert any("index-wide-no-reason" in f for f in _findings(tmp_path, _LINE, migration=wide))
    reasoned = f"{_LINE} :: reason=the digest answers each active datasource's customer from the index alone"
    assert _index(_findings(tmp_path, reasoned, migration=wide)) == []


def test_an_index_declared_on_two_lines_is_declared_twice(tmp_path: Path) -> None:
    """Flips if a ledger may declare one index on two lines."""
    findings = _findings(tmp_path, f"{_LINE}\n{_LINE}")
    assert any("index-declared-twice -- datasources_by_kind is declared on line 2 too" in f for f in findings), findings
