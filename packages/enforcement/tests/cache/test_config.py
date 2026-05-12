"""tests for the cache-enforcement config dataclass."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.cache import CacheEnforcementConfig


class TestDefaults:
    def test_default_scan_roots_is_none(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.scan_roots is None

    def test_default_inheritance_roots_is_none(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.inheritance_roots is None

    def test_default_exemptions_path_is_none(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.exemptions_path is None

    def test_default_mode_env_var(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.mode_env_var == "CACHE_ENFORCEMENT_MODE"

    def test_default_allowed_sites_empty(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.allowed_sqlite_construction_sites == frozenset()

    def test_default_collection_allowlist_empty(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.collection_table_allowlist == {}

    def test_default_migration_allowlist_empty(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.migration_table_allowlist == frozenset()

    def test_default_cache_method_names(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        # five canonical cache verbs; intentionally does not include
        # ``remove`` so consumers can opt in.
        assert config.cache_method_names == frozenset({"get", "put", "set", "delete", "upsert"})

    def test_default_base_collection_names(self, tmp_path: Path) -> None:
        # default is just ``BaseCollection`` — intermediate bases like
        # ``SchemaBackedCollection`` are walked transitively, NOT
        # hardcoded as targets.
        config = CacheEnforcementConfig(repo_root=tmp_path)
        assert config.base_collection_names == frozenset({"BaseCollection"})


class TestOverrides:
    def test_explicit_scan_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "src",)
        config = CacheEnforcementConfig(
            repo_root=tmp_path,
            scan_roots=roots,
        )
        assert config.scan_roots == roots

    def test_explicit_inheritance_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "src", tmp_path / "lib" / "src")
        config = CacheEnforcementConfig(
            repo_root=tmp_path,
            inheritance_roots=roots,
        )
        assert config.inheritance_roots == roots

    def test_scan_and_inheritance_roots_independent(self, tmp_path: Path) -> None:
        # the architecture fix: callers can pass distinct tuples for
        # each so violation scanning stays scoped while the inheritance
        # graph still spans path-deps.
        scan = (tmp_path / "src",)
        inheritance = (tmp_path / "src", tmp_path / "lib" / "src")
        config = CacheEnforcementConfig(
            repo_root=tmp_path,
            scan_roots=scan,
            inheritance_roots=inheritance,
        )
        assert config.scan_roots == scan
        assert config.inheritance_roots == inheritance
        assert config.scan_roots != config.inheritance_roots

    def test_explicit_allowed_sites(self, tmp_path: Path) -> None:
        sites = frozenset({"src/factory.py"})
        config = CacheEnforcementConfig(
            repo_root=tmp_path,
            allowed_sqlite_construction_sites=sites,
        )
        assert config.allowed_sqlite_construction_sites == sites

    def test_explicit_collection_allowlist(self, tmp_path: Path) -> None:
        allowlist = {"memories": "MemoriesCollection"}
        config = CacheEnforcementConfig(
            repo_root=tmp_path,
            collection_table_allowlist=allowlist,
        )
        assert config.collection_table_allowlist == allowlist

    def test_explicit_base_collection_names(self, tmp_path: Path) -> None:
        # consumers MAY override to add additional terminal names, but
        # the default + transitive walk should be sufficient for every
        # production consumer.
        names = frozenset({"BaseCollection", "AnotherBase"})
        config = CacheEnforcementConfig(
            repo_root=tmp_path,
            base_collection_names=names,
        )
        assert config.base_collection_names == names

    def test_frozen_dataclass(self, tmp_path: Path) -> None:
        config = CacheEnforcementConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("dataclass should be frozen")


class TestNoBackwardsCompatShim:
    """``src_roots`` was renamed/replaced; constructing with it must fail.

    deliberately a hard break — the architecture fix splits scanning
    from the inheritance graph, and the right next move for any caller
    that passed ``src_roots`` is to choose explicitly.
    """

    def test_legacy_src_roots_kwarg_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            CacheEnforcementConfig(  # type: ignore[call-arg]
                repo_root=tmp_path,
                src_roots=(tmp_path / "src",),
            )
