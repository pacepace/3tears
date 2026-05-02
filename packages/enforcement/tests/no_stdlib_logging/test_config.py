"""tests for ``NoStdlibLoggingConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.no_stdlib_logging import NoStdlibLoggingConfig


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = NoStdlibLoggingConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "STDLIB_LOGGING_ENFORCEMENT_MODE"
        assert config.exempt_files == {}
        assert config.line_marker == "# stdlib-logging: ok"

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = NoStdlibLoggingConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")

    def test_default_factory_returns_independent_dicts(
        self, tmp_path: Path,
    ) -> None:
        a = NoStdlibLoggingConfig(repo_root=tmp_path)
        b = NoStdlibLoggingConfig(repo_root=tmp_path)
        assert a.exempt_files == b.exempt_files
        assert a.exempt_files is not b.exempt_files


class TestOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = NoStdlibLoggingConfig(repo_root=tmp_path, src_roots=roots)
        assert config.src_roots == roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "ex.txt"
        config = NoStdlibLoggingConfig(
            repo_root=tmp_path, exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = NoStdlibLoggingConfig(
            repo_root=tmp_path, mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_custom_exempt_files(self, tmp_path: Path) -> None:
        exempt = {
            "src/aibots/hub/common/config.py": (
                "platform bootstrap; stdlib logging used pre-observe init"
            ),
        }
        config = NoStdlibLoggingConfig(
            repo_root=tmp_path, exempt_files=exempt,
        )
        assert config.exempt_files == exempt

    def test_custom_line_marker(self, tmp_path: Path) -> None:
        config = NoStdlibLoggingConfig(
            repo_root=tmp_path, line_marker="# stdlogging-ok",
        )
        assert config.line_marker == "# stdlogging-ok"
