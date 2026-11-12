"""A read whole by nature, declared where it is written (``whole_read(<read>, kind=..., ...)``, Pace 2026-11-12):
the one mechanism. Each test flips if the gate lets through what the declaration does not cover, or refuses what it
does."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings, use_the_helper_findings

_RULING = "Pace 2026-11-12: a backup's own count and checksum read whole"

_STORE = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.declared import whole_read

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])
    _COUNT_ITEMS = "SELECT count(*) FROM items"
    _SHARED = "SELECT * FROM items"

    async def count_items(conn):
        return await whole_read(
            conn.fetchval(_COUNT_ITEMS),
            kind="backup_verification",
            reason="the manifest's counts are taken inside the dump's own snapshot",
            ruling="RULING",
            proof="tests/test_backup.py::test_counts_only_in_a_snapshot",
            speed="tests/test_backup.py::test_count_speed",
        )

    async def fill(conn):
        return await whole_read(
            conn.fetch("SELECT id, body FROM items"),
            kind="cold_start",
            reason="an empty copy is filled once at its cold start, nothing in L2",
            ruling="RULING",
            proof="tests/test_backup.py::test_fill_only_when_empty",
            speed="tests/test_backup.py::test_fill_speed",
        )
"""

_PROOFS = """
    from threetears.core.testing.calibrate import budget_problem

    def test_counts_only_in_a_snapshot():
        assert count_items

    def test_count_speed():
        assert count_items
        budget_problem()

    def test_fill_only_when_empty():
        assert fill

    def test_fill_speed():
        assert fill
        budget_problem()
"""


def _config(tmp_path: Path, modules: dict[str, str], proofs: str = _PROOFS) -> NoWholeReadsConfig:
    """the gate over synthesised modules and their proof tests (3tears' own declarations not read)."""
    src = tmp_path / "src"
    for name, source in modules.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).replace("RULING", _RULING), encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_backup.py").write_text(textwrap.dedent(proofs), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    return NoWholeReadsConfig(repo_root=tmp_path, src_roots=(src,), ledger_dir=tmp_path, dependency_schema=False)


def _findings(tmp_path: Path, modules: dict[str, str], proofs: str = _PROOFS) -> list[str]:
    return no_whole_reads_findings(_config(tmp_path, modules, proofs))


def test_a_sound_declaration_lets_its_read_through_and_is_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Flips if a declared count (its SQL in a module constant only it names) or a declared cold-start fill is still
    reported, if the declaration call itself reads as a whole call, or if an accepted declaration is not printed."""
    assert _findings(tmp_path, {"store.py": _STORE}) == []
    printed = capsys.readouterr().err
    assert f"declared-whole backup_verification src/store.py:10 count_items -- {_RULING}" in printed, printed
    assert "declared-whole cold_start src/store.py:20 fill" in printed, printed


def test_a_kind_lets_through_only_its_own_whole_reads(tmp_path: Path) -> None:
    """Flips if a kind lets through a whole read of another kind (an operator reset wrapping a whole select), or a
    drain reads a declared L3 table, or a declaration covering nothing it may let through is not named."""
    module = (
        _STORE
        + """
    async def reset(conn):
        return await whole_read(
            conn.fetch(_SHARED),
            kind="operator_reset",
            reason="an operator empties the copy it names, by hand only",
            ruling="RULING",
            proof="tests/test_backup.py::test_reset_entry",
        )

    async def drain(conn):
        return await whole_read(
            conn.fetch("SELECT id FROM items"),
            kind="drain",
            reason="the process-local buffer's rows are taken whole on flush",
            ruling="RULING",
            proof="tests/test_backup.py::test_drain_entry",
        )
"""
    )
    proofs = _PROOFS + textwrap.indent(
        "\ndef test_reset_entry():\n    assert reset\n\ndef test_drain_entry():\n    assert drain\n", "    "
    )
    findings = _findings(tmp_path, {"store.py": module}, proofs)
    assert any("in reset: whole-select items" in finding for finding in findings), findings
    assert any("in drain: whole-select items" in finding for finding in findings), findings
    assert any("reset whole_read(kind='operator_reset'): covers no whole read" in f for f in findings), findings
    assert any("drain whole_read(kind='drain'): covers no whole read" in f for f in findings), findings


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("RULING", "approved 2026-11-12"), "the ruling must cite Pace's ruling"),
        (("the manifest's counts are taken inside the dump's own snapshot", "snapshot counts"), "the reason must say"),
        (("test_counts_only_in_a_snapshot", "test_missing"), "the proof tests/test_backup.py defines no test_missing"),
        (('kind="backup_verification"', 'kind="nightly"'), "kind must be one of"),
        (('speed="tests/test_backup.py::test_fill_speed"', 'speed=""'), "the speed test names no test file"),
    ],
)
def test_a_declaration_is_checked_not_trusted(tmp_path: Path, change: tuple[str, str], message: str) -> None:
    """Flips if a declaration is accepted with a ruling not citing Pace and a date, a short reason, a proof naming
    no test, an unknown kind or a missing speed test -- and its read then still passes."""
    module = (
        _STORE.replace('ruling="RULING"', f'ruling="{change[1]}"') if change[0] == "RULING" else _STORE.replace(*change)
    )
    findings = _findings(tmp_path, {"store.py": module})
    assert any(message in finding for finding in findings), findings
    still = "in fill: whole-select items" if "fill_speed" in change[0] else "in <module>: whole-aggregate items"
    assert any(still in finding for finding in findings), findings


def test_a_proof_must_name_the_enclosing_function(tmp_path: Path) -> None:
    """Flips if a proof test that never names the declaring function is accepted."""
    proofs = _PROOFS.replace("assert count_items\n", "assert True\n", 1)
    findings = _findings(tmp_path, {"store.py": _STORE}, proofs)
    assert any("test_counts_only_in_a_snapshot never names count_items" in finding for finding in findings), findings


def test_a_constant_named_elsewhere_is_not_the_declaration_s_alone(tmp_path: Path) -> None:
    """Flips if a declaration covers a module constant's statement that an undeclared read names too."""
    module = (
        _STORE
        + """
    async def again(conn):
        return await conn.fetchval(_COUNT_ITEMS)
"""
    )
    findings = _findings(tmp_path, {"store.py": module})
    assert any("in <module>: whole-aggregate items" in finding for finding in findings), findings


def test_a_call_into_a_declared_read_is_declared_there_with_the_same_kind(tmp_path: Path) -> None:
    """Flips if an undeclared call into a function holding a declared read passes, if the call declared with the
    same kind is refused, or if one declared with another kind passes."""
    caller = """
    from threetears.core.declared import whole_read
    from store import fill

    async def plain(conn):
        return await fill(conn)

    async def same(conn):
        return await whole_read(
            fill(conn),
            kind="cold_start",
            reason="the copy's start fills it once when it holds nothing at all",
            ruling="RULING",
            proof="tests/test_backup.py::test_same_entry",
            speed="tests/test_backup.py::test_same_speed",
        )

    async def other(conn):
        return await whole_read(
            fill(conn),
            kind="backup_verification",
            reason="the restore check compares the manifest's counts with its own",
            ruling="RULING",
            proof="tests/test_backup.py::test_other_entry",
            speed="tests/test_backup.py::test_other_speed",
        )
"""
    proofs = _PROOFS + textwrap.indent(
        "".join(
            f"\ndef test_{name}_entry():\n    assert {name}\n\ndef test_{name}_speed():\n    budget_problem({name})\n"
            for name in ("same", "other")
        ),
        "    ",
    )
    findings = _findings(tmp_path, {"store.py": _STORE, "restore.py": caller}, proofs)
    assert any("in plain: calls-whole-read fill" in finding for finding in findings), findings
    assert any("in other: calls-whole-read fill" in finding for finding in findings), findings
    assert not any("in same:" in finding or " same whole_read" in finding for finding in findings), findings


def test_a_declared_statement_is_not_a_use_the_helper_finding(tmp_path: Path) -> None:
    """Flips if the use-the-helper rule counts a declared whole read (no keyed helper answers it), or stops counting
    the same statement undeclared."""
    declared = use_the_helper_findings(_config(tmp_path, {"store.py": _STORE}))
    assert not any("SELECT count(*) FROM items" in finding for finding in declared), declared
    plain = _STORE.replace("whole_read(\n            conn.fetchval(", "keep(\n            conn.fetchval(")
    assert plain != _STORE
    undeclared = use_the_helper_findings(_config(tmp_path, {"store.py": plain}))
    assert any("SELECT count(*) FROM items" in finding for finding in undeclared), undeclared


def test_a_call_into_a_declared_constant_read_is_declared_there_too(tmp_path: Path) -> None:
    """Flips if a call into a function whose declared read is a module constant (count_items) passes undeclared, or
    is refused when declared with the same kind."""
    caller = """
    from threetears.core.declared import whole_read
    from store import count_items

    async def plain(conn):
        return await count_items(conn)

    async def same(conn):
        return await whole_read(
            count_items(conn),
            kind="backup_verification",
            reason="the restore check compares the manifest's counts with its own",
            ruling="RULING",
            proof="tests/test_backup.py::test_same_entry",
            speed="tests/test_backup.py::test_same_speed",
        )
"""
    proofs = _PROOFS + textwrap.indent(
        "\ndef test_same_entry():\n    assert same\n\ndef test_same_speed():\n    budget_problem(same)\n", "    "
    )
    findings = _findings(tmp_path, {"store.py": _STORE, "restore.py": caller}, proofs)
    assert any("in plain: calls-whole-read count_items" in finding for finding in findings), findings
    assert not any("in same:" in finding or " same whole_read" in finding for finding in findings), findings


_LOCAL_SQL = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.declared import whole_read

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])

    async def count_items(conn, table):
        sql = f"SELECT count(*) FROM {table}"
        return await whole_read(
            conn.fetchval(sql),
            kind="backup_verification",
            reason="the manifest's counts are taken inside the dump's own snapshot",
            ruling="RULING",
            proof="tests/test_backup.py::test_counts_only_in_a_snapshot",
            speed="tests/test_backup.py::test_count_speed",
        )
"""


@pytest.mark.parametrize("reused", [False, True])
def test_a_local_statement_named_only_by_the_declaration_is_covered(tmp_path: Path, reused: bool) -> None:
    """Flips if a statement bound to a local just before the declaration (named only in the wrapped read) is still
    reported, or if one the function also runs undeclared is covered."""
    module = _LOCAL_SQL
    if reused:
        module = module.replace(
            "        return await whole_read(", "        await conn.fetchval(sql)\n        return await whole_read("
        )
    findings = _findings(tmp_path, {"store.py": module})
    reported = any("in count_items: unknown-schema" in f or "in count_items: whole-aggregate" in f for f in findings)
    assert reported is reused, findings


_BARE = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.declared import whole_read

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])
    OWNERS = TableSchema(name="owners", primary_key="id", columns=[])

    async def count_items(conn):
        return await whole_read(
            conn.fetchval("STATEMENT"),
            kind="backup_verification",
            reason="the manifest's counts are taken inside the dump's own snapshot",
            ruling="RULING",
            proof="tests/test_backup.py::test_counts_only_in_a_snapshot",
        )

    async def inventory(conn):
        return await whole_read(
            count_items(conn),
            kind="backup_verification",
            reason="the inventory of every table, counted in the dump's own snapshot",
            ruling="RULING",
            proof="tests/test_backup.py::test_inventory_entry",
        )
"""


@pytest.mark.parametrize(
    ("statement", "accepted"),
    [
        ("SELECT count(*) FROM items", True),
        ("SELECT count(*) AS n, coalesce(sum(hashtext(t::text)::bigint), 0) AS s FROM items t", True),
        ("SELECT count(*) FROM items i JOIN owners o ON o.id = i.id", False),
        ("SELECT count(*) FROM items WHERE body <> ''", False),
        ("SELECT count(*) FROM items GROUP BY body ORDER BY 1", False),
    ],
)
def test_a_backup_count_omits_its_speed_test_only_when_it_is_the_bare_count(
    tmp_path: Path, statement: str, accepted: bool
) -> None:
    """Flips if a backup_verification declaration with no speed test is accepted over a joined, filtered or ordered
    statement, or refused over a bare count(*) or row checksum of one table (and its declared caller with it).
    Coordinator ruling 2026-11-12: the bare count IS the bare operation; anything richer is timed."""
    proofs = _PROOFS + textwrap.indent("\ndef test_inventory_entry():\n    assert inventory\n", "    ")
    findings = _findings(tmp_path, {"store.py": _BARE.replace("STATEMENT", statement)}, proofs)
    refused = [finding for finding in findings if "names no speed test" in finding]
    if accepted:
        assert findings == [], findings
    else:
        assert any("count_items whole_read" in finding for finding in refused), findings
        assert any("in count_items:" in finding for finding in findings), findings
        assert any("in inventory: calls-whole-read count_items" in finding for finding in findings), findings
