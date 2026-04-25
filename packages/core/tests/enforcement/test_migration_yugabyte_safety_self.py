"""
meta-tests for the migration yugabyte-safety walker.

exercises the walker's contracts against synthetic source files in
``tmp_path``:

- positive cases: clean source surfaces no violation
- negative cases: one fixture per rule (M-1 through M-5) deliberately
  triggers the rule
- exemption case: a documented exemption suppresses the violation

mirrors :mod:`tests.enforcement.test_partition_column_enforcement_self`
in shape -- review-task-01 finding D-8: every AST walker should ship
with at least one positive + negative + exemption fixture so a buggy
walker that silently passes everything is caught at CI time.

the walker logic itself lives in
:mod:`threetears.core.data.migrations.enforcement`; the per-repo
``test_migration_yugabyte_safety.py`` invocation modules import the
walker and run it against their repo's migration tree.
"""

from __future__ import annotations

from pathlib import Path

from threetears.core.data.migrations.enforcement import (
    RULE_M1_DDL_DML_MIXING,
    RULE_M2_REPLAY_GUARD,
    RULE_M3_IDEMPOTENCY,
    RULE_M4_PK_UNIQUE,
    RULE_M5_TRUNCATE,
    find_migration_violations,
    load_exemptions,
)

__all__: list[str] = []


def _write(path: Path, text: str) -> Path:
    """
    write ``text`` to ``path``, creating parents; return ``path``.

    :param path: target file
    :ptype path: Path
    :param text: file contents
    :ptype text: str
    :return: same path, for chaining
    :rtype: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# positive: clean source, no violations
# ---------------------------------------------------------------------------


class TestMigrationWalkerPositive:
    """clean migration source surfaces no walker violation."""

    def test_split_ddl_dml_passes(self, tmp_path: Path) -> None:
        """separate execute calls for DDL and DML produce no M-1 hit."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "ADD = 'ALTER TABLE t ADD COLUMN IF NOT EXISTS c VARCHAR(8)'\n"
                "BACKFILL = \"UPDATE t SET c = 'x' WHERE c IS NULL\"\n"
            ),
        )
        violations = find_migration_violations(target)
        assert violations == []

    def test_check_constraint_with_existence_probe_passes(
        self, tmp_path: Path,
    ) -> None:
        """DO block with NOT EXISTS guard satisfies M-3."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "SQL = '''\n"
                "DO $$\n"
                "BEGIN\n"
                "    IF NOT EXISTS (SELECT 1 FROM pg_constraint "
                "WHERE conname = 't_x_ck') THEN\n"
                "        ALTER TABLE t ADD CONSTRAINT t_x_ck "
                "CHECK (x > 0);\n"
                "    END IF;\n"
                "END $$;\n"
                "'''\n"
            ),
        )
        violations = find_migration_violations(target)
        assert violations == []

    def test_helper_call_skips_walker(self, tmp_path: Path) -> None:
        """helper call body has no SQL string literals; walker emits nothing."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "from threetears.core.data.migrations.helpers import "
                "add_column_with_backfill\n"
                "async def apply(store):\n"
                "    await add_column_with_backfill(\n"
                "        store, table='t', column='c',\n"
                "        column_type='VARCHAR(8)', default=\"'a'\",\n"
                "        backfill_value_sql=\"'b'\",\n"
                "        backfill_predicate='customer_id IS NULL',\n"
                "    )\n"
            ),
        )
        violations = find_migration_violations(target)
        assert violations == []


# ---------------------------------------------------------------------------
# negative: one deliberate violation per rule
# ---------------------------------------------------------------------------


class TestMigrationWalkerNegative:
    """one deliberate-violation fixture per rule (M-1 through M-5)."""

    def test_m1_ddl_dml_mixing_flagged(self, tmp_path: Path) -> None:
        """DO block mixing ALTER TABLE and UPDATE produces M-1 hit."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "SQL = '''\n"
                "DO $$\n"
                "BEGIN\n"
                "    ALTER TABLE t ADD COLUMN c VARCHAR(8) DEFAULT 'a';\n"
                "    UPDATE t SET c = 'b' WHERE c IS NULL;\n"
                "END $$;\n"
                "'''\n"
            ),
        )
        violations = find_migration_violations(target)
        rule_hits = [v for v in violations if v.rule == RULE_M1_DDL_DML_MIXING]
        assert rule_hits != [], (
            f"expected M-1 violation; got: {[v.format() for v in violations]}"
        )

    def test_m2_missing_replay_guard_flagged(self, tmp_path: Path) -> None:
        """UPDATE backfill without replay guard produces M-2 hit."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "SQL = \"UPDATE t SET c = 'x' WHERE customer_id IS NULL\"\n"
            ),
        )
        violations = find_migration_violations(target)
        rule_hits = [v for v in violations if v.rule == RULE_M2_REPLAY_GUARD]
        assert rule_hits != [], (
            f"expected M-2 violation; got: {[v.format() for v in violations]}"
        )

    def test_m3_missing_if_not_exists_flagged(self, tmp_path: Path) -> None:
        """bare CREATE TABLE produces M-3 hit."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "SQL = 'CREATE TABLE t (id INTEGER PRIMARY KEY)'\n"
            ),
        )
        violations = find_migration_violations(target)
        rule_hits = [v for v in violations if v.rule == RULE_M3_IDEMPOTENCY]
        assert rule_hits != [], (
            f"expected M-3 violation; got: {[v.format() for v in violations]}"
        )

    def test_m4_composite_pk_without_unique_flagged(
        self, tmp_path: Path,
    ) -> None:
        """ADD CONSTRAINT PRIMARY KEY (a, b) without UNIQUE produces M-4 hit."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "SQL = 'ALTER TABLE t ADD CONSTRAINT t_pkey "
                "PRIMARY KEY (customer_id, id)'\n"
            ),
        )
        violations = find_migration_violations(target)
        rule_hits = [v for v in violations if v.rule == RULE_M4_PK_UNIQUE]
        assert rule_hits != [], (
            f"expected M-4 violation; got: {[v.format() for v in violations]}"
        )

    def test_m5_truncate_flagged(self, tmp_path: Path) -> None:
        """TRUNCATE TABLE produces M-5 hit."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            (
                "SQL = 'TRUNCATE TABLE media CASCADE'\n"
            ),
        )
        violations = find_migration_violations(target)
        rule_hits = [v for v in violations if v.rule == RULE_M5_TRUNCATE]
        assert rule_hits != [], (
            f"expected M-5 violation; got: {[v.format() for v in violations]}"
        )


# ---------------------------------------------------------------------------
# exemption-respected case
# ---------------------------------------------------------------------------


class TestMigrationWalkerExemption:
    """documented exemptions suppress the walker hit."""

    def test_exemption_suppresses_truncate_hit(self, tmp_path: Path) -> None:
        """``<file>:M-5 # rationale: ...`` skips the M-5 hit on that file."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v009.py",
            "SQL = 'TRUNCATE TABLE media CASCADE'\n",
        )
        exempt_file = _write(
            tmp_path / "exemptions.txt",
            (
                f"{target}:M-5 # rationale: pre-GA agent-memory v009 "
                f"clears media to land NOT NULL without per-row backfill\n"
            ),
        )
        exemptions, missing = load_exemptions(exempt_file)
        assert missing == []
        violations = find_migration_violations(target, exemptions=exemptions)
        rule_hits = [v for v in violations if v.rule == RULE_M5_TRUNCATE]
        assert rule_hits == []

    def test_exemption_meta_rejects_blank_rationale(
        self, tmp_path: Path,
    ) -> None:
        """exemption line lacking a rationale is surfaced via ``missing``."""
        exempt_file = _write(
            tmp_path / "exemptions.txt",
            (
                "/some/path.py:M-5 # rationale:\n"
                "/another/path.py:M-1 # internal access needed\n"
            ),
        )
        _, missing = load_exemptions(exempt_file)
        assert len(missing) == 2

    def test_exemption_with_specific_rationale_accepted(
        self, tmp_path: Path,
    ) -> None:
        """non-blank rationale is accepted; pair is included in the set."""
        exempt_file = _write(
            tmp_path / "exemptions.txt",
            (
                "/foo/v009.py:M-5 # rationale: pre-GA TRUNCATE before "
                "NOT NULL switch on partitioned media table\n"
            ),
        )
        exemptions, missing = load_exemptions(exempt_file)
        assert missing == []
        assert ("/foo/v009.py", "M-5") in exemptions
