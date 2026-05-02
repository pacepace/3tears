"""tests for ``NoSilentSwallowConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.no_silent_swallow import NoSilentSwallowConfig


_DEFAULT_LOGGER_NAMES = frozenset(
    {"log", "logger", "_log", "_logger", "LOGGER"},
)
_DEFAULT_LOGGER_METHODS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception"},
)


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = NoSilentSwallowConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "NO_SILENT_SWALLOW_MODE"
        assert config.logger_names == _DEFAULT_LOGGER_NAMES
        assert config.logger_methods == _DEFAULT_LOGGER_METHODS
        assert config.nosilent_marker == "# NOSILENT:"

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = NoSilentSwallowConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")


class TestOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = NoSilentSwallowConfig(repo_root=tmp_path, src_roots=roots)
        assert config.src_roots == roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "ex.txt"
        config = NoSilentSwallowConfig(
            repo_root=tmp_path, exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = NoSilentSwallowConfig(
            repo_root=tmp_path, mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_custom_logger_names(self, tmp_path: Path) -> None:
        # a repo that uses, say, ``slog`` / ``audit`` as logger
        # receivers can override the recogniser without forking.
        custom = frozenset({"slog", "audit"})
        config = NoSilentSwallowConfig(
            repo_root=tmp_path, logger_names=custom,
        )
        assert config.logger_names == custom

    def test_custom_logger_methods(self, tmp_path: Path) -> None:
        custom = frozenset({"trace", "audit"})
        config = NoSilentSwallowConfig(
            repo_root=tmp_path, logger_methods=custom,
        )
        assert config.logger_methods == custom

    def test_custom_nosilent_marker(self, tmp_path: Path) -> None:
        config = NoSilentSwallowConfig(
            repo_root=tmp_path, nosilent_marker="# NOSWALLOW:",
        )
        assert config.nosilent_marker == "# NOSWALLOW:"

    def test_default_factory_returns_equal_sets(self, tmp_path: Path) -> None:
        a = NoSilentSwallowConfig(repo_root=tmp_path)
        b = NoSilentSwallowConfig(repo_root=tmp_path)
        assert a.logger_names == b.logger_names
        assert a.logger_methods == b.logger_methods
