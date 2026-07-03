"""tests for ``LoggerCoverageConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.logger_coverage import LoggerCoverageConfig


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = LoggerCoverageConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "LOGGER_COVERAGE_ENFORCEMENT_MODE"
        assert config.exempt_files == {}
        assert config.logger_factory_names == frozenset({"get_logger"})
        assert config.expected_var_names == frozenset({"log", "_logger"})
        assert config.skip_basenames == frozenset({"__init__.py"})

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = LoggerCoverageConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")

    def test_default_factory_returns_independent_dicts(
        self,
        tmp_path: Path,
    ) -> None:
        a = LoggerCoverageConfig(repo_root=tmp_path)
        b = LoggerCoverageConfig(repo_root=tmp_path)
        assert a.exempt_files == b.exempt_files
        assert a.exempt_files is not b.exempt_files


class TestOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = LoggerCoverageConfig(repo_root=tmp_path, src_roots=roots)
        assert config.src_roots == roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "ex.txt"
        config = LoggerCoverageConfig(
            repo_root=tmp_path,
            exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = LoggerCoverageConfig(
            repo_root=tmp_path,
            mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_custom_exempt_files(self, tmp_path: Path) -> None:
        exempt = {
            "src/3tears/hub/agents/entities.py": ("pure pydantic model module, no runtime behaviour"),
        }
        config = LoggerCoverageConfig(
            repo_root=tmp_path,
            exempt_files=exempt,
        )
        assert config.exempt_files == exempt

    def test_custom_logger_factory_names(self, tmp_path: Path) -> None:
        config = LoggerCoverageConfig(
            repo_root=tmp_path,
            logger_factory_names=frozenset({"get_logger", "make_logger"}),
        )
        assert "make_logger" in config.logger_factory_names

    def test_custom_expected_var_names(self, tmp_path: Path) -> None:
        config = LoggerCoverageConfig(
            repo_root=tmp_path,
            expected_var_names=frozenset({"log"}),
        )
        # only ``log`` accepted; ``_logger`` no longer counts.
        assert config.expected_var_names == frozenset({"log"})

    def test_custom_skip_basenames(self, tmp_path: Path) -> None:
        config = LoggerCoverageConfig(
            repo_root=tmp_path,
            skip_basenames=frozenset({"__init__.py", "_compat.py"}),
        )
        assert "_compat.py" in config.skip_basenames
