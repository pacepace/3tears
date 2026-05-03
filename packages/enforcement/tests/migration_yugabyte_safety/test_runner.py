"""tests for ``run_migration_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import (
    MODE_REPORT,
    MODE_STRICT,
    ExemptionError,
)
from threetears.enforcement.migration_yugabyte_safety import (
    MigrationYugabyteConfig,
    run_migration_enforcement,
)


__all__: list[str] = []


_VALID_RATIONALE = "legacy v007 migration ships TRUNCATE for the bootstrap reset path; rewrite tracked under sub-task 7"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo_with_migrations(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    migrations = repo_root / "migrations"
    migrations.mkdir()
    return migrations


def _build_violating_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one M-3 violation."""
    repo = tmp_path / "repo"
    migrations = _make_repo_with_migrations(repo)
    _write(
        migrations / "v001.py",
        'SQL = """CREATE TABLE foo (id uuid)"""\n',
    )
    return repo, migrations


# ----------------------------------------------------------------
# input validation
# ----------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
        )
        with pytest.raises(ValueError, match="walker must be one of"):
            run_migration_enforcement(config, walker="bogus")

    def test_default_walker_is_all(self, tmp_path: Path) -> None:
        # report mode (the canonical default) is silent on no input.
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
        )
        run_migration_enforcement(config)


# ----------------------------------------------------------------
# strict / report mode behaviour
# ----------------------------------------------------------------


class TestModes:
    def test_report_mode_returns_silently_with_violations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, migrations = _build_violating_repo(tmp_path)
        monkeypatch.setenv("MIGR_TEST_MODE", "report")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            mode_env_var="MIGR_TEST_MODE",
        )
        # must not raise even though a violation exists.
        run_migration_enforcement(config, walker="all")

    def test_strict_fails_on_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, migrations = _build_violating_repo(tmp_path)
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            mode_env_var="MIGR_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_migration_enforcement(config, walker="all")

    def test_strict_clean_passes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE IF NOT EXISTS foo (id uuid)"""\n',
        )
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            mode_env_var="MIGR_TEST_MODE",
        )
        run_migration_enforcement(config, walker="all")

    def test_default_mode_is_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # config.default_mode = MODE_REPORT (canonical default); env var
        # unset; violation present; runner returns silently.
        repo, migrations = _build_violating_repo(tmp_path)
        monkeypatch.delenv("MIGR_TEST_MODE", raising=False)
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            mode_env_var="MIGR_TEST_MODE",
        )
        assert config.default_mode == MODE_REPORT
        run_migration_enforcement(config, walker="all")

    def test_default_mode_strict_overrides(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # explicit default_mode=strict (no env var) flips the runner to
        # hard-fail on the same violation.
        repo, migrations = _build_violating_repo(tmp_path)
        monkeypatch.delenv("MIGR_TEST_MODE", raising=False)
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            mode_env_var="MIGR_TEST_MODE",
            default_mode=MODE_STRICT,
        )
        with pytest.raises(pytest.fail.Exception):
            run_migration_enforcement(config, walker="all")


# ----------------------------------------------------------------
# missing-dir handling
# ----------------------------------------------------------------


class TestMissingDirs:
    def test_nonexistent_dir_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("")
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        # path does not exist; runner must not raise.
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(repo / "no-such-dir",),
            mode_env_var="MIGR_TEST_MODE",
        )
        run_migration_enforcement(config, walker="all")

    def test_one_existing_one_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, migrations = _build_violating_repo(tmp_path)
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        # one dir exists (with a violation), the other doesn't.
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations, repo / "future-migrations"),
            mode_env_var="MIGR_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_migration_enforcement(config, walker="all")


# ----------------------------------------------------------------
# exemption file handling — uses core's load_exemptions
# ----------------------------------------------------------------


class TestExemptionsFile:
    def test_exemption_silences_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, migrations = _build_violating_repo(tmp_path)
        ex_path = repo / "_migration_exemptions.txt"
        ex_path.write_text(
            f"migrations/v001.py:M-3 # rationale: {_VALID_RATIONALE}\n",
        )
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=ex_path,
            mode_env_var="MIGR_TEST_MODE",
        )
        run_migration_enforcement(config, walker="all")

    def test_exemption_only_silences_listed_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
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
        ex_path = repo / "_migration_exemptions.txt"
        ex_path.write_text(
            f"migrations/exempt.py:M-3 # rationale: {_VALID_RATIONALE}\n",
        )
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=ex_path,
            mode_env_var="MIGR_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_migration_enforcement(config, walker="all")
        msg = str(excinfo.value)
        assert "violator.py" in msg
        assert "exempt.py" not in msg

    def test_exemption_path_none_skips_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE IF NOT EXISTS foo (id uuid)"""\n',
        )
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=None,
            mode_env_var="MIGR_TEST_MODE",
        )
        run_migration_enforcement(config, walker="all")

    def test_missing_exemption_file_raises(self, tmp_path: Path) -> None:
        repo, migrations = _build_violating_repo(tmp_path)
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=repo / "missing.txt",
        )
        with pytest.raises(FileNotFoundError):
            run_migration_enforcement(config, walker="all")

    def test_blank_rationale_rejected(self, tmp_path: Path) -> None:
        # core's load_exemptions returns the line in `missing` if the
        # rationale is blank; the runner converts that to ExemptionError.
        repo, migrations = _build_violating_repo(tmp_path)
        ex_path = repo / "_migration_exemptions.txt"
        ex_path.write_text(
            "migrations/v001.py:M-3 # rationale: \n",
        )
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=ex_path,
        )
        with pytest.raises(ExemptionError, match="rationale"):
            run_migration_enforcement(config, walker="all")

    def test_missing_rationale_rejected(self, tmp_path: Path) -> None:
        # entry has no `# rationale:` suffix at all.
        repo, migrations = _build_violating_repo(tmp_path)
        ex_path = repo / "_migration_exemptions.txt"
        ex_path.write_text(
            "migrations/v001.py:M-3\n",
        )
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=ex_path,
        )
        with pytest.raises(ExemptionError, match="rationale"):
            run_migration_enforcement(config, walker="all")

    def test_empty_exemption_file_ok(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # bot trio repos ship empty (header-only) exemption files;
        # this must continue to load without error.
        repo = tmp_path / "repo"
        migrations = _make_repo_with_migrations(repo)
        _write(
            migrations / "v001.py",
            'SQL = """CREATE TABLE IF NOT EXISTS foo (id uuid)"""\n',
        )
        ex_path = repo / "_migration_exemptions.txt"
        ex_path.write_text(
            "# header comment about migration exemptions\n#\n# format: <file_path>:<rule_name> # rationale: <reason>\n",
        )
        monkeypatch.setenv("MIGR_TEST_MODE", "strict")
        config = MigrationYugabyteConfig(
            repo_root=repo,
            migration_dirs=(migrations,),
            exemptions_path=ex_path,
            mode_env_var="MIGR_TEST_MODE",
        )
        run_migration_enforcement(config, walker="all")
