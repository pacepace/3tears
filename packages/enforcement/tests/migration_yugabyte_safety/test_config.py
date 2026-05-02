"""tests for ``MigrationYugabyteConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.common import MODE_REPORT, MODE_STRICT
from threetears.enforcement.migration_yugabyte_safety import (
    MigrationYugabyteConfig,
)


__all__: list[str] = []


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(tmp_path / "migrations",),
        )
        assert config.repo_root == tmp_path
        assert config.migration_dirs == (tmp_path / "migrations",)
        assert config.exemptions_path is None
        assert config.mode_env_var == "MIGRATION_ENFORCEMENT_MODE"
        assert config.default_mode == MODE_REPORT

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(tmp_path / "migrations",),
        )
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")

    def test_default_mode_is_report(self, tmp_path: Path) -> None:
        # canonical's retro-rewrite-window default — preserved verbatim.
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
        )
        assert config.default_mode == MODE_REPORT


class TestOverrides:
    def test_explicit_migration_dirs(self, tmp_path: Path) -> None:
        dirs = (tmp_path / "a", tmp_path / "b")
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=dirs,
        )
        assert config.migration_dirs == dirs

    def test_empty_migration_dirs(self, tmp_path: Path) -> None:
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
        )
        assert config.migration_dirs == ()

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "_migration_exemptions.txt"
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
            exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
            mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_explicit_default_mode_strict(self, tmp_path: Path) -> None:
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=(),
            default_mode=MODE_STRICT,
        )
        assert config.default_mode == MODE_STRICT

    def test_full_roundtrip(self, tmp_path: Path) -> None:
        dirs = (tmp_path / "src" / "migrations",)
        ex = tmp_path / "ex.txt"
        config = MigrationYugabyteConfig(
            repo_root=tmp_path,
            migration_dirs=dirs,
            exemptions_path=ex,
            mode_env_var="MIGR_TEST_MODE",
            default_mode=MODE_STRICT,
        )
        assert config.repo_root == tmp_path
        assert config.migration_dirs == dirs
        assert config.exemptions_path == ex
        assert config.mode_env_var == "MIGR_TEST_MODE"
        assert config.default_mode == MODE_STRICT
