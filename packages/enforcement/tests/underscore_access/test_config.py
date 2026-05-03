"""tests for ``UnderscoreAccessConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from threetears.enforcement.underscore_access import UnderscoreAccessConfig


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = UnderscoreAccessConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.scan_roots is None
        assert config.inheritance_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "UNDERSCORE_AUDIT_MODE"
        assert "conftest.py" in config.skip_basenames
        assert "__main__.py" in config.skip_basenames
        assert "_version.py" in config.skip_basenames
        assert config.enable_shape_b_ruff is True

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = UnderscoreAccessConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")


class TestOverrides:
    def test_explicit_scan_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = UnderscoreAccessConfig(repo_root=tmp_path, scan_roots=roots)
        assert config.scan_roots == roots

    def test_explicit_inheritance_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "lib" / "src")
        config = UnderscoreAccessConfig(
            repo_root=tmp_path,
            inheritance_roots=roots,
        )
        assert config.inheritance_roots == roots

    def test_scan_and_inheritance_roots_independent(self, tmp_path: Path) -> None:
        # the architecture fix: callers can pass distinct tuples for
        # each so violation scanning stays scoped to the consumer's
        # own code while shape-A / shape-D still see path-deps.
        scan = (tmp_path / "src",)
        inheritance = (tmp_path / "src", tmp_path / "lib" / "src")
        config = UnderscoreAccessConfig(
            repo_root=tmp_path,
            scan_roots=scan,
            inheritance_roots=inheritance,
        )
        assert config.scan_roots == scan
        assert config.inheritance_roots == inheritance
        assert config.scan_roots != config.inheritance_roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "ex.txt"
        config = UnderscoreAccessConfig(repo_root=tmp_path, exemptions_path=ex)
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = UnderscoreAccessConfig(repo_root=tmp_path, mode_env_var="MY_MODE")
        assert config.mode_env_var == "MY_MODE"

    def test_explicit_skip_basenames(self, tmp_path: Path) -> None:
        skip = frozenset({"foo.py"})
        config = UnderscoreAccessConfig(
            repo_root=tmp_path,
            skip_basenames=skip,
        )
        assert config.skip_basenames == skip

    def test_disable_shape_b_ruff(self, tmp_path: Path) -> None:
        config = UnderscoreAccessConfig(
            repo_root=tmp_path,
            enable_shape_b_ruff=False,
        )
        assert config.enable_shape_b_ruff is False


class TestNoBackwardsCompatShim:
    """``src_roots`` was renamed/replaced; constructing with it must fail.

    deliberately a hard break — the architecture fix splits scanning
    from the inheritance graph, and the right next move for any caller
    that passed ``src_roots`` is to choose explicitly.
    """

    def test_legacy_src_roots_kwarg_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            UnderscoreAccessConfig(  # type: ignore[call-arg]
                repo_root=tmp_path,
                src_roots=(tmp_path / "src",),
            )
