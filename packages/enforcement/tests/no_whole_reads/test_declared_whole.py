"""A read whole by nature, declared where it is written (``whole_read(<read>, kind=..., ...)``, Pace 2026-11-12):
the one mechanism. Each test flips if the gate lets through what the declaration does not cover, or refuses what it
does."""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import pytest

from threetears.core.declared import whole_read
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


def test_a_call_into_a_declared_drain_is_declared_there_with_the_same_kind(tmp_path: Path) -> None:
    """Flips if a drain cannot be declared at a call into the function holding it (flush_pending's call of
    WriteBuffer.drain), if an undeclared call into it passes, or if a call declared with another kind passes."""
    reason = "the process's own buffer (never L3): every pending write is taken whole to flush"
    held = f"""
    from threetears.core.declared import whole_read

    async def held(conn):
        return await whole_read(
            conn.fetch("SELECT * FROM write_buffer"),
            kind="drain",
            reason="{reason}",
            ruling="RULING",
            proof="tests/test_backup.py::test_held_entry",
        )
"""
    caller = """
    from threetears.core.declared import whole_read
    from held import held

    async def plain(conn):
        return await held(conn)
""" + "".join(
        f"""
    async def {name}(conn):
        return await whole_read(
            held(conn),
            kind="{kind}",
            reason="{reason}",
            ruling="RULING",
            proof="tests/test_backup.py::test_{name}_entry",
        )
"""
        for name, kind in (("same", "drain"), ("other", "cold_start"))
    )
    proofs = _PROOFS + textwrap.indent(
        "".join(f"\ndef test_{name}_entry():\n    assert {name}\n" for name in ("held", "same", "other")), "    "
    )
    findings = _findings(tmp_path, {"held.py": held, "caller.py": caller}, proofs)
    assert any("in plain: calls-whole-read held" in finding for finding in findings), findings
    assert any("in other: calls-whole-read held" in finding for finding in findings), findings
    assert not any("in same:" in finding or " same whole_read" in finding for finding in findings), findings
    assert not any(" held whole_read" in finding or "in held:" in finding for finding in findings), findings


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


_ROLL = """
    from threetears.core.declared import whole_read
    from threetears.core.collections.schema_backed import TableSchema

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])

    async def legacy(conn):
        return await whole_read(
            conn.fetch("SELECT id, body FROM items"),
            kind="rolling_upgrade",
            reason="a writer older than the roll left a whole record: the table is read again whole",
            ruling="RULING",
            proof="tests/test_backup.py::test_legacy_only_for_old_writers",
            speed="tests/test_backup.py::test_legacy_speed",
            UNTIL
        )
"""

_ROLL_PROOFS = """
    from threetears.core.testing.calibrate import budget_problem

    def test_legacy_only_for_old_writers():
        assert legacy

    def test_legacy_speed():
        assert legacy
        budget_problem()
"""


@pytest.mark.parametrize(
    ("release", "until", "message"),
    [
        ("0.95.0", 'until="0.96.0",', None),
        ("0.96.0", 'until="0.96.0",', "the roll is over: the package is at 0.96.0"),
        ("0.97.1", 'until="0.96.0",', "the roll is over: the package is at 0.97.1"),
        ("0.95.0", 'until="soon",', 'names until="<major>.<minor>.<patch>"'),
        ("0.95.0", "", 'names until="<major>.<minor>.<patch>"'),
    ],
)
def test_a_rolling_upgrade_read_is_refused_once_its_release_is_reached(
    tmp_path: Path, release: str, until: str, message: str | None
) -> None:
    """Flips if a rolling_upgrade read is let through at or past the release it names (the package's own
    pyproject.toml version), or with no release or one that is not a version, or if one before its release is refused
    (coordinator ruling 2026-11-12)."""
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "pkg"\nversion = "{release}"\n', encoding="utf-8")
    findings = _findings(tmp_path, {"store.py": _ROLL.replace("UNTIL", until)}, _ROLL_PROOFS)
    if message is None:
        assert findings == [], findings
    else:
        assert any(message in finding and "legacy" in finding for finding in findings), findings


def test_until_is_a_rolling_upgrade_s_alone(tmp_path: Path) -> None:
    """Flips if another kind may name until= (a cold start that quietly expires), at the gate or at run time."""
    module = _STORE.replace(
        'speed="tests/test_backup.py::test_fill_speed",',
        'speed="tests/test_backup.py::test_fill_speed",\n            until="0.96.0",',
    )
    assert any(
        "until= is a rolling_upgrade's alone" in finding for finding in _findings(tmp_path, {"store.py": module})
    )
    with pytest.raises(ValueError, match="until"):
        whole_read(1, kind="cold_start", reason="r", ruling="Pace 2026-11-12", proof="a::b", until="0.96.0")
    with pytest.raises(ValueError, match="until"):
        whole_read(1, kind="rolling_upgrade", reason="r", ruling="Pace 2026-11-12", proof="a::b")
    assert (
        whole_read(1, kind="rolling_upgrade", reason="r", ruling="Pace 2026-11-12", proof="a::b", until="0.96.0") == 1
    )


_WAREHOUSE = """
    from dataclasses import dataclass

    from threetears.core import keyed
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.declared import whole_read

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])


    @dataclass
    class SourceTable:
        relation: str
        key: tuple[str, ...] = ()


    async def moved(reader, source: SourceTable, state, held):
        return await whole_read(
            keyed.read_changed(
                reader, source.relation, key=list(source.key), columns=["a"], held=held, hash_columns=["a"],
                by={"state": state},
            ),
            kind="unbounded_by_warehouse",
            reason="the warehouse relation holds no sort key: the state bound here binds no key",
            ruling="RULING",
            proof="tests/test_backup.py::test_moved_entry",
        )


    async def ours(reader, state, held):
        return await whole_read(
            keyed.read_changed(
                reader, "items", key=["id"], columns=["a"], held=held, hash_columns=["a"], by={"state": state}
            ),
            kind="unbounded_by_warehouse",
            reason="the warehouse relation holds no sort key: the state bound here binds no key",
            ruling="RULING",
            proof="tests/test_backup.py::test_ours_entry",
        )
"""


def test_a_warehouse_read_of_a_produced_relation_declared_whole_is_let_through(tmp_path: Path) -> None:
    """Flips if a read of a run-time relation a declared producer yields read whole (``SourceTable.relation: ()``, a
    warehouse relation: no 3tears table) is refused under ``unbounded_by_warehouse`` as if it were a declared L3
    table, or if the same declaration lets a read of a declared L3 table through."""
    proofs = _PROOFS + textwrap.indent(
        "\ndef test_moved_entry():\n    assert moved\n\ndef test_ours_entry():\n    assert ours\n", "    "
    )
    config = dataclasses.replace(
        _config(tmp_path, {"loader.py": _WAREHOUSE}, proofs), relation_producers={"SourceTable.relation": ()}
    )
    findings = no_whole_reads_findings(config)
    assert not [finding for finding in findings if "moved" in finding], findings
    assert any("ours whole_read(kind='unbounded_by_warehouse'): covers no whole read" in f for f in findings), findings


_SCHEMA_CHANGE = """
    from threetears.core.declared import whole_read

    async def held(conn, schema, table, column):
        return await whole_read(
            conn.fetchval(f'SELECT count(*) FROM "{schema}"."{table}" WHERE "{column}" IS NOT NULL'),
            kind="SCHEMA_KIND",
            reason="a resize clears every embedding the column holds, and says how many first",
            ruling="RULING",
            proof="tests/test_backup.py::test_held_entry",
            speed="tests/test_backup.py::test_held_speed",
        )
"""


@pytest.mark.parametrize(("kind", "passes"), [("schema_change", True), ("cold_start", False)])
def test_a_schema_change_admits_a_table_named_at_run_time(tmp_path: Path, kind: str, passes: bool) -> None:
    """Flips if a schema change's count over a table it names at run time (the schema it runs in, as a migration's)
    is still reported once declared, or if another kind lets that unknown table through."""
    proofs = _PROOFS + textwrap.indent(
        "\ndef test_held_entry():\n    assert held\n\ndef test_held_speed():\n    budget_problem(held)\n", "    "
    )
    findings = _findings(tmp_path, {"held.py": _SCHEMA_CHANGE.replace("SCHEMA_KIND", kind)}, proofs)
    assert (findings == []) is passes, findings
    if not passes:
        assert any("in held: unknown-schema" in finding for finding in findings), findings


def test_a_whole_call_into_a_declared_function_is_declared_there_with_the_same_kind(tmp_path: Path) -> None:
    """Flips if a call named whole (a rebuild) into a function holding a declared schema change cannot be declared
    there with the same kind, if an undeclared one passes, if one declared with another kind passes, or if a declared
    call into a rebuild that never declared its own read passes."""
    reason = "an index left unusable is built again; its unique keys are checked over the table first"
    inner = f"""
    from threetears.core.declared import whole_read

    async def duplicates(conn, schema, table):
        return await whole_read(
            conn.fetch(f'SELECT k, count(*) AS n FROM "{{schema}}"."{{table}}" GROUP BY k HAVING count(*) > 1'),
            kind="schema_change",
            reason="{reason}",
            ruling="RULING",
            proof="tests/test_backup.py::test_duplicates_entry",
            speed="tests/test_backup.py::test_duplicates_speed",
        )

    async def rebuild_declared(conn, schema, table):
        return await whole_read(
            duplicates(conn, schema, table),
            kind="schema_change",
            reason="{reason}",
            ruling="RULING",
            proof="tests/test_backup.py::test_rebuild_declared_entry",
            speed="tests/test_backup.py::test_rebuild_declared_speed",
        )

    async def rebuild_bare(conn, schema, table):
        return await duplicates(conn, schema, table)
"""
    caller = """
    from threetears.core.declared import whole_read
    from inner import rebuild_bare, rebuild_declared

    async def plain(conn):
        return await rebuild_declared(conn, "s", "t")
""" + "".join(
        f"""
    async def {name}(conn):
        return await whole_read(
            {callee}(conn, "s", "t"),
            kind="{kind}",
            reason="{reason}",
            ruling="RULING",
            proof="tests/test_backup.py::test_{name}_entry",
            speed="tests/test_backup.py::test_{name}_speed",
        )
"""
        for name, callee, kind in (
            ("same", "rebuild_declared", "schema_change"),
            ("other", "rebuild_declared", "cold_start"),
            ("unchained", "rebuild_bare", "schema_change"),
        )
    )
    names = ("duplicates", "rebuild_declared", "same", "other", "unchained")
    proofs = _PROOFS + textwrap.indent(
        "".join(
            f"\ndef test_{name}_entry():\n    assert {name}\n\ndef test_{name}_speed():\n    budget_problem({name})\n"
            for name in names
        ),
        "    ",
    )
    findings = _findings(tmp_path, {"inner.py": inner, "caller.py": caller}, proofs)
    assert any("in plain: whole-call rebuild_declared" in finding for finding in findings), findings
    assert any("in other: whole-call rebuild_declared" in finding for finding in findings), findings
    assert any("in unchained: whole-call rebuild_bare" in finding for finding in findings), findings
    assert any("in rebuild_bare: calls-whole-read duplicates" in finding for finding in findings), findings
    assert not any("in same:" in finding or " same whole_read" in finding for finding in findings), findings
    assert not any("in rebuild_declared:" in finding or "in duplicates:" in finding for finding in findings), findings


_MAINTAINED = """
    from threetears.core.collections.schema_backed import TableSchema
    from threetears.core.declared import whole_read

    ITEMS = TableSchema(name="items", primary_key="id", columns=[])

    async def refresh_statistics(conn):
        return await whole_read(
            conn.execute('STATEMENT'),
            kind="schema_change",
            reason="the planner's statistics are gathered again after the table was rebuilt",
            ruling="RULING",
            proof="tests/test_backup.py::test_counts_only_in_a_snapshot",
        )
"""


@pytest.mark.parametrize(
    ("statement", "accepted"),
    [
        ("ANALYZE items", True),
        ('ANALYZE "items"', True),
        ("SELECT count(*) FROM items", False),
        ("SELECT * FROM items", False),
    ],
)
def test_a_schema_change_omits_its_speed_test_only_when_it_is_the_maintenance_operation(
    tmp_path: Path, statement: str, accepted: bool
) -> None:
    """Flips if a schema_change declaration with no speed test is accepted over a read of the table, or refused over a
    bare ANALYZE of it. Coordinator ruling 2026-11-13: the maintenance operation (ANALYZE, REINDEX) IS the bare
    operation, as a backup's bare count is; anything richer is timed."""
    proofs = _PROOFS.replace("assert count_items", "assert refresh_statistics", 1)
    findings = _findings(tmp_path, {"store.py": _MAINTAINED.replace("STATEMENT", statement)}, proofs)
    if accepted:
        assert findings == [], findings
    else:
        assert any("refresh_statistics whole_read(kind='schema_change'): names no speed test" in f for f in findings), (
            findings
        )
