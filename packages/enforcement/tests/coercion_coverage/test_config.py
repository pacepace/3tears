"""tests for ``CoerceCoverageConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.coercion_coverage import CoerceCoverageConfig


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = CoerceCoverageConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "COERCE_ENFORCEMENT_MODE"
        assert config.base_class_suffixes == frozenset({"Tool"})

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = CoerceCoverageConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")


class TestOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = CoerceCoverageConfig(repo_root=tmp_path, src_roots=roots)
        assert config.src_roots == roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "ex.txt"
        config = CoerceCoverageConfig(repo_root=tmp_path, exemptions_path=ex)
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = CoerceCoverageConfig(
            repo_root=tmp_path,
            mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_explicit_base_class_suffixes(self, tmp_path: Path) -> None:
        suffixes = frozenset({"Tool", "Action", "Hook"})
        config = CoerceCoverageConfig(
            repo_root=tmp_path,
            base_class_suffixes=suffixes,
        )
        assert config.base_class_suffixes == suffixes

    def test_default_factory_is_independent(self, tmp_path: Path) -> None:
        # two configs share equal defaults but the factory returns a
        # frozenset (immutable) so identity does not actually matter;
        # this confirms the equality.
        a = CoerceCoverageConfig(repo_root=tmp_path)
        b = CoerceCoverageConfig(repo_root=tmp_path)
        assert a.base_class_suffixes == b.base_class_suffixes
