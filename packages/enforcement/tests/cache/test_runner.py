"""tests for ``run_cache_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.cache import (
    CacheEnforcementConfig,
    run_cache_enforcement,
)


_VALID_RATIONALE = "factory-only construction; legacy hub L1 cache pending migration"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_sqlite_violation(tmp_path: Path) -> tuple[Path, Path]:
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "foo.py",
        "from x import SQLiteBackend\ndef make() -> None:\n    backend = SQLiteBackend('/tmp/foo')\n",
    )
    return repo, src


def _build_collections_repo(tmp_path: Path) -> tuple[Path, Path]:
    """clean repo: every migrated table maps to a transitive Collection."""
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "bases.py",
        ("class BaseCollection: pass\nclass SchemaBackedCollection(BaseCollection): pass\n"),
    )
    _write(
        src / "memories.py",
        ("from .bases import SchemaBackedCollection\nclass MemoriesCollection(SchemaBackedCollection): pass\n"),
    )
    _write(
        src / "migrations" / "v001_init.py",
        'SQL = "CREATE TABLE memories (id INT)"\n',
    )
    return repo, src


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(),
            inheritance_roots=(),
        )
        with pytest.raises(ValueError, match="walker must be one of"):
            run_cache_enforcement(config, walker="bogus")

    def test_valid_walker_names(self, tmp_path: Path) -> None:
        # smoke-test that each named walker is accepted.
        repo, src = _build_collections_repo(tmp_path)
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            collection_table_allowlist={"memories": "MemoriesCollection"},
        )
        for walker in (
            "sqlite_construction",
            "wrapper_class",
            "pool_access",
            "missing_collection",
            "all",
        ):
            run_cache_enforcement(config, walker=walker)


# ----------------------------------------------------------------------
# strict / report mode behaviour
# ----------------------------------------------------------------------


class TestModes:
    def test_strict_fails_on_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_sqlite_violation(tmp_path)
        monkeypatch.setenv("CACHE_TEST_MODE", "strict")
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="CACHE_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_cache_enforcement(config, walker="sqlite_construction")

    def test_report_mode_returns_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_sqlite_violation(tmp_path)
        monkeypatch.setenv("CACHE_TEST_MODE", "report")
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="CACHE_TEST_MODE",
        )
        # no raise even though a violation exists.
        run_cache_enforcement(config, walker="sqlite_construction")

    def test_clean_repo_strict_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_collections_repo(tmp_path)
        monkeypatch.setenv("CACHE_TEST_MODE", "strict")
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="CACHE_TEST_MODE",
            collection_table_allowlist={"memories": "MemoriesCollection"},
        )
        run_cache_enforcement(config, walker="all")


# ----------------------------------------------------------------------
# exemption application via the runner
# ----------------------------------------------------------------------


class TestExemptions:
    def test_exemption_filters_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_sqlite_violation(tmp_path)
        exemptions = repo / "_cache_exemptions.txt"
        exemptions.write_text("# rationale: " + _VALID_RATIONALE + "\nsrc/foo.py:3:SQLiteBackend\n")
        monkeypatch.setenv("CACHE_TEST_MODE", "strict")
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            exemptions_path=exemptions,
            mode_env_var="CACHE_TEST_MODE",
        )
        # exemption filters the only violation -> strict still passes.
        run_cache_enforcement(config, walker="sqlite_construction")


# ----------------------------------------------------------------------
# bug-fix verification — transitive Collection chain through runner
# ----------------------------------------------------------------------


class TestTransitiveBugFixViaRunner:
    def test_transitive_chain_no_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # the central bug-fix verification: clean repo with
        # MemoriesCollection -> SchemaBackedCollection -> BaseCollection.
        # canonical produced a spurious "missing_collection" violation
        # because its check only walked direct AST bases. the new
        # walker passes via the transitive walker.
        repo, src = _build_collections_repo(tmp_path)
        monkeypatch.setenv("CACHE_TEST_MODE", "strict")
        config = CacheEnforcementConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="CACHE_TEST_MODE",
            collection_table_allowlist={"memories": "MemoriesCollection"},
        )
        # raises iff the transitive walk fails -> bug fix verified.
        run_cache_enforcement(config, walker="missing_collection")
