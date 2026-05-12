"""tests for the scan_roots / inheritance_roots split (architecture fix).

the original ``CacheEnforcementConfig.src_roots`` overloaded "where to
look for violations" with "where to build the inheritance graph". when
a consumer enabled Option B path-dep walking for inheritance, scanning
also dragged in path-dep code, producing spurious violations in
upstream packages. these tests prove the split fixes that:

- ``test_scan_roots_scopes_violations_to_local_only``: synthetic
  two-package fixture where the consumer's ``scan_roots`` is just the
  consumer's src and ``inheritance_roots`` is the union. the walker
  scans only consumer code (so sibling code never produces violations)
  while the transitive subclass detection still resolves the chain
  through the sibling.
- ``test_inheritance_roots_default_walks_path_deps``: a consumer with
  a path-dep declaration; ``inheritance_roots=None`` defaults to
  :func:`discover_src_roots` so the inheritance graph contains classes
  from the path-dep target.
- ``test_explicit_scan_and_inheritance_roots_independent``: pass
  different tuples for each and verify scope independence end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.cache import (
    CacheEnforcementConfig,
    find_missing_collections,
    find_sqlite_constructions,
    run_cache_enforcement,
)
from threetears.enforcement.common import discover_src_roots


_DEFAULT_BASE_NAMES = frozenset({"BaseCollection"})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_consumer_with_path_dep(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """build consumer + sibling library each with their own pyproject.

    returns ``(consumer_root, consumer_src, lib_src)``.
    """
    lib_root = tmp_path / "lib"
    lib_src = lib_root / "src"
    lib_root.mkdir(parents=True)
    (lib_root / "pyproject.toml").write_text('[project]\nname = "synthetic-lib"\nversion = "0"\n')
    _write(lib_src / "lib_pkg" / "__init__.py", "")
    _write(
        lib_src / "lib_pkg" / "bases.py",
        ("class BaseCollection: pass\nclass SchemaBackedCollection(BaseCollection): pass\n"),
    )
    # bare construction in the sibling library — would be flagged if
    # the consumer's scan_roots accidentally pulled in lib_src.
    _write(
        lib_src / "lib_pkg" / "wrong.py",
        ("from x import SQLiteBackend\ndef make() -> None:\n    backend = SQLiteBackend('/tmp/foo')\n"),
    )

    consumer_root = tmp_path / "consumer"
    consumer_src = consumer_root / "src"
    consumer_root.mkdir(parents=True)
    # consumer's pyproject declares the sibling as a poetry path-dep
    # so discover_src_roots picks lib_src up.
    (consumer_root / "pyproject.toml").write_text(
        "[tool.poetry]\n"
        'name = "synthetic-consumer"\n'
        'version = "0"\n'
        'description = ""\n'
        'authors = ["t"]\n'
        "\n"
        "[tool.poetry.dependencies]\n"
        'python = "^3.11"\n'
        f'synthetic-lib = {{path = "{lib_root}", develop = true}}\n'
    )
    _write(consumer_src / "consumer_pkg" / "__init__.py", "")
    _write(
        consumer_src / "consumer_pkg" / "memories.py",
        ("from lib_pkg.bases import SchemaBackedCollection\nclass MemoriesCollection(SchemaBackedCollection): pass\n"),
    )
    _write(
        consumer_src / "consumer_pkg" / "migrations" / "v001_init.py",
        'SQL = "CREATE TABLE memories (id INT)"\n',
    )
    return consumer_root, consumer_src, lib_src


class TestArchitectureFix:
    def test_scan_roots_scopes_violations_to_local_only(
        self,
        tmp_path: Path,
    ) -> None:
        """sibling-package code never appears in violations even when
        the inheritance graph spans both packages."""
        consumer_root, consumer_src, lib_src = _make_consumer_with_path_dep(
            tmp_path,
        )
        # scan ONLY the consumer; build inheritance graph over BOTH.
        sqlite_violations = find_sqlite_constructions(
            (consumer_src,),
            consumer_root,
            frozenset(),
        )
        # would be 1 if scan accidentally walked the sibling.
        assert sqlite_violations == []

        missing = find_missing_collections(
            (consumer_src,),
            consumer_root,
            (consumer_src, lib_src),
            {"memories": "MemoriesCollection"},
            frozenset(),
            _DEFAULT_BASE_NAMES,
        )
        # inheritance graph spans both src roots so the transitive
        # subclass walk reaches BaseCollection through lib_src.
        assert missing == []

    def test_inheritance_roots_default_walks_path_deps(
        self,
        tmp_path: Path,
    ) -> None:
        """``inheritance_roots=None`` defaults to discover_src_roots,
        which is path-dep aware."""
        consumer_root, _consumer_src, lib_src = _make_consumer_with_path_dep(
            tmp_path,
        )
        # discover_src_roots is what the runner falls back to when
        # CacheEnforcementConfig.inheritance_roots is None.
        discovered = discover_src_roots(consumer_root)
        resolved = {p.resolve() for p in discovered}
        # the path-dep target's src is in the discovered set.
        assert lib_src.resolve() in resolved

    def test_explicit_scan_and_inheritance_roots_independent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """end-to-end via the runner: scan_roots and inheritance_roots
        govern independent scopes; bug behaviour does NOT bleed."""
        consumer_root, consumer_src, lib_src = _make_consumer_with_path_dep(
            tmp_path,
        )
        monkeypatch.setenv("CACHE_TEST_MODE", "strict")
        config = CacheEnforcementConfig(
            repo_root=consumer_root,
            scan_roots=(consumer_src,),
            inheritance_roots=(consumer_src, lib_src),
            mode_env_var="CACHE_TEST_MODE",
            collection_table_allowlist={"memories": "MemoriesCollection"},
        )
        # strict mode: must pass. if scan_roots leaked into the lib
        # src tree, the bare ``SQLiteBackend(...)`` in lib_pkg/wrong.py
        # would raise. if the inheritance graph didn't include lib_src,
        # the migration table would be flagged as a missing Collection.
        run_cache_enforcement(config, walker="all")
