"""tests for ``CodebaseConventionsConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.codebase_conventions import (
    CodebaseConventionsConfig,
)


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = CodebaseConventionsConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "CODEBASE_CONVENTIONS_ENFORCEMENT_MODE"
        assert config.print_exempt_files == {}
        assert config.getlogger_exempt_files == {}
        assert config.getlogger_marker == "# stdlib-getlogger: ok"
        assert config.future_annotations_exempt_files == {}
        assert config.return_type_exempt_files == {}
        assert config.skip_basenames == frozenset({"__init__.py"})

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = CodebaseConventionsConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")

    def test_default_factories_return_independent_dicts(
        self, tmp_path: Path,
    ) -> None:
        a = CodebaseConventionsConfig(repo_root=tmp_path)
        b = CodebaseConventionsConfig(repo_root=tmp_path)
        assert a.print_exempt_files == b.print_exempt_files
        assert a.print_exempt_files is not b.print_exempt_files
        assert a.getlogger_exempt_files is not b.getlogger_exempt_files
        assert (
            a.future_annotations_exempt_files
            is not b.future_annotations_exempt_files
        )
        assert a.return_type_exempt_files is not b.return_type_exempt_files

    def test_default_skip_basenames_is_frozen(self, tmp_path: Path) -> None:
        config = CodebaseConventionsConfig(repo_root=tmp_path)
        assert isinstance(config.skip_basenames, frozenset)


class TestOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, src_roots=roots,
        )
        assert config.src_roots == roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "ex.txt"
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_custom_print_exempt_files(self, tmp_path: Path) -> None:
        exempt = {
            "src/aibots/cli.py": (
                "command-line entry point intentionally writes to stdout"
            ),
        }
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, print_exempt_files=exempt,
        )
        assert config.print_exempt_files == exempt

    def test_custom_getlogger_exempt_files(self, tmp_path: Path) -> None:
        exempt = {
            "src/aibots/config.py": (
                "platform bootstrap; stdlib logging used pre-observe init"
            ),
        }
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, getlogger_exempt_files=exempt,
        )
        assert config.getlogger_exempt_files == exempt

    def test_custom_getlogger_marker(self, tmp_path: Path) -> None:
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, getlogger_marker="# stdlogger-ok",
        )
        assert config.getlogger_marker == "# stdlogger-ok"

    def test_custom_future_annotations_exempt_files(
        self, tmp_path: Path,
    ) -> None:
        exempt = {
            "src/aibots/types_only.py": (
                "stub module; only TYPE_CHECKING-guarded imports"
            ),
        }
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, future_annotations_exempt_files=exempt,
        )
        assert config.future_annotations_exempt_files == exempt

    def test_custom_return_type_exempt_files(self, tmp_path: Path) -> None:
        exempt = {
            "src/aibots/legacy.py": (
                "legacy module pending migration to typed signatures"
            ),
        }
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, return_type_exempt_files=exempt,
        )
        assert config.return_type_exempt_files == exempt

    def test_custom_skip_basenames(self, tmp_path: Path) -> None:
        skip = frozenset({"__init__.py", "conftest.py"})
        config = CodebaseConventionsConfig(
            repo_root=tmp_path, skip_basenames=skip,
        )
        assert config.skip_basenames == skip
