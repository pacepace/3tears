"""tests for the migration-yugabyte-safety walker adapter.

each test builds a synthetic ``migrations/`` directory containing a
single migration ``.py`` file with one or more SQL string literals
that match (or deliberately don't match) the rules implemented in
:mod:`threetears.core.data.migrations.enforcement`. the adapter's
job is to project the underlying ``MigrationViolation`` records into
the common :class:`Violation
<threetears.enforcement.common.violations.Violation>` shape; we
verify both that the underlying walker fires correctly and that the
projection preserves the data.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.migration_yugabyte_safety.walkers import (
    find_migration_violations,
)


__all__: list[str] = []


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo_with_migrations(repo_root: Path) -> Path:
    """build a repo skeleton with an empty ``migrations/`` directory.

    :return: path to the migrations directory (caller writes files into it)
    :rtype: Path
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    migrations = repo_root / "migrations"
    migrations.mkdir()
    return migrations


# ----------------------------------------------------------------
# rule-fire happy path
# ----------------------------------------------------------------


class TestRuleM3Idempotency:
    def test_bare_create_table_flagged(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE foo (id uuid PRIMARY KEY)"""\n',
        )
        violations = find_migration_violations(
            (migrations,), repo, frozenset(),
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "migration_yugabyte_safety.M-3"
        assert v.symbol == "M-3"
        assert v.line >= 1
        assert v.file == migrations / "v001.py"
        assert "CREATE TABLE" in v.reason

    def test_bare_create_index_flagged(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v002.py",
            'SQL = """CREATE INDEX foo_id_idx ON foo (id)"""\n',
        )
        violations = find_migration_violations(
            (migrations,), repo, frozenset(),
        )
        assert len(violations) == 1
        assert violations[0].category == "migration_yugabyte_safety.M-3"
        assert "CREATE INDEX" in violations[0].reason

    def test_bare_add_column_flagged(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v003.py",
            'SQL = """ALTER TABLE foo ADD COLUMN bar text"""\n',
        )
        violations = find_migration_violations(
            (migrations,), repo, frozenset(),
        )
        assert len(violations) == 1
        assert violations[0].category == "migration_yugabyte_safety.M-3"
        assert "ADD COLUMN" in violations[0].reason

    def test_create_table_if_not_exists_clean(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE IF NOT EXISTS foo (id uuid)"""\n',
        )
        assert find_migration_violations((migrations,), repo, frozenset()) == []


class TestRuleM5Truncate:
    def test_truncate_table_flagged(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v010.py",
            'SQL = """TRUNCATE TABLE foo"""\n',
        )
        violations = find_migration_violations(
            (migrations,), repo, frozenset(),
        )
        # TRUNCATE TABLE matches both M-5 (truncate forbidden) AND, via the
        # DML pattern set, may participate in other heuristics — in this
        # synthetic case M-5 fires alone because there's no DO block.
        rules = {v.symbol for v in violations}
        assert "M-5" in rules


# ----------------------------------------------------------------
# missing dirs / no-input behaviour
# ----------------------------------------------------------------


class TestNoInput:
    def test_no_migration_dirs_returns_empty(self, tmp_path: Path) -> None:
        # passing an empty tuple yields an empty result.
        assert find_migration_violations((), tmp_path, frozenset()) == []

    def test_nonexistent_dir_silently_skipped(self, tmp_path: Path) -> None:
        # underlying walker silently skips a directory that does not
        # exist on disk; the adapter must inherit that behaviour.
        repo = tmp_path / "repo"
        repo.mkdir()
        missing = repo / "no-such-dir"
        # missing path; walker yields []; adapter yields [].
        assert find_migration_violations(
            (missing,), repo, frozenset(),
        ) == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        # no .py files in migrations/.
        assert find_migration_violations(
            (migrations,), repo, frozenset(),
        ) == []

    def test_init_py_skipped(self, tmp_path: Path) -> None:
        # underlying walker skips __init__.py.
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "__init__.py",
            'SQL = """CREATE TABLE foo (id uuid)"""\n',
        )
        assert find_migration_violations(
            (migrations,), repo, frozenset(),
        ) == []


# ----------------------------------------------------------------
# multi-file / multi-violation aggregation
# ----------------------------------------------------------------


class TestMultiFile:
    def test_violations_across_multiple_files(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE a (id uuid)"""\n',
        )
        _write(
            migrations / "v002.py",
            'SQL = """CREATE INDEX b_idx ON b (id)"""\n',
        )
        violations = find_migration_violations(
            (migrations,), repo, frozenset(),
        )
        assert len(violations) == 2
        # both fire the same rule (M-3) but from different files
        files = {str(v.file) for v in violations}
        assert any(f.endswith("v001.py") for f in files)
        assert any(f.endswith("v002.py") for f in files)
        rules = {v.symbol for v in violations}
        assert rules == {"M-3"}

    def test_multiple_dirs_aggregated(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("")
        a = repo / "a"
        b = repo / "b"
        a.mkdir()
        b.mkdir()
        _write(a / "v001.py", 'SQL = """CREATE TABLE a (id uuid)"""\n')
        _write(b / "v001.py", 'SQL = """CREATE TABLE b (id uuid)"""\n')
        violations = find_migration_violations(
            (a, b), repo, frozenset(),
        )
        assert len(violations) == 2


# ----------------------------------------------------------------
# exemption application — exemptions are applied INSIDE the core walker
# ----------------------------------------------------------------


class TestExemptionApplication:
    def test_exemption_silences_violation(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE foo (id uuid)"""\n',
        )
        # exemption key uses repo-relative POSIX path.
        exemptions = frozenset({("migrations/v001.py", "M-3")})
        assert find_migration_violations(
            (migrations,), repo, exemptions,
        ) == []

    def test_exemption_only_silences_matching_rule(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """TRUNCATE TABLE foo"""\n',
        )
        # exempt M-3, but the only fire is M-5 — exemption must not match.
        exemptions = frozenset({("migrations/v001.py", "M-3")})
        violations = find_migration_violations(
            (migrations,), repo, exemptions,
        )
        rules = {v.symbol for v in violations}
        assert "M-5" in rules

    def test_exemption_only_silences_listed_file(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "exempt.py",
            'SQL = """CREATE TABLE foo (id uuid)"""\n',
        )
        _write(
            migrations / "violator.py",
            'SQL = """CREATE TABLE bar (id uuid)"""\n',
        )
        exemptions = frozenset({("migrations/exempt.py", "M-3")})
        violations = find_migration_violations(
            (migrations,), repo, exemptions,
        )
        assert len(violations) == 1
        assert violations[0].file == migrations / "violator.py"
