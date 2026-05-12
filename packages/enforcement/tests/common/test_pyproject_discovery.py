"""tests for ``pyproject_discovery`` module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from threetears.enforcement.common.pyproject_discovery import (
    PyprojectError,
    discover_src_roots,
)


def _make_repo(root: Path, pyproject_text: str, has_src: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(pyproject_text)
    if has_src:
        (root / "src").mkdir(exist_ok=True)
    return root


def test_python_version_supported() -> None:
    """tomllib is required; sanity-check the runtime."""
    assert sys.version_info >= (3, 11)


class TestPoetryPathDeps:
    def test_basic_path_dep(self, tmp_path: Path) -> None:
        target = _make_repo(
            tmp_path / "core",
            '[tool.poetry]\nname = "core"\n[tool.poetry.dependencies]\n',
        )
        consumer = _make_repo(
            tmp_path / "consumer",
            '[tool.poetry]\nname = "consumer"\n'
            "[tool.poetry.dependencies]\n"
            f'core = {{path = "../core", develop = true}}\n',
        )
        roots = discover_src_roots(consumer)
        assert (target / "src").resolve() in roots
        assert (consumer / "src").resolve() in roots

    def test_dev_group_path_dep(self, tmp_path: Path) -> None:
        target = _make_repo(
            tmp_path / "tools",
            '[tool.poetry]\nname = "tools"\n',
        )
        consumer = _make_repo(
            tmp_path / "consumer",
            f'[tool.poetry]\nname = "consumer"\n[tool.poetry.group.dev.dependencies]\ntools = {{path = "../tools"}}\n',
        )
        roots = discover_src_roots(consumer)
        assert (target / "src").resolve() in roots

    def test_version_string_deps_skipped(self, tmp_path: Path) -> None:
        consumer = _make_repo(
            tmp_path / "consumer",
            '[tool.poetry]\nname = "consumer"\n[tool.poetry.dependencies]\nfastapi = ">=0.115"\n',
        )
        roots = discover_src_roots(consumer)
        assert roots == ((consumer / "src").resolve(),)


class TestUvWorkspace:
    def test_workspace_globs_expand(self, tmp_path: Path) -> None:
        root = tmp_path / "monorepo"
        root.mkdir()
        (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\n')
        # workspace root itself has no top-level src
        for name in ("core", "observe"):
            pkg = root / "packages" / name
            pkg.mkdir(parents=True)
            (pkg / "src").mkdir()
            (pkg / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
        # decoy: a packages/* dir without a pyproject — must raise
        bad = root / "packages" / "broken"
        bad.mkdir(parents=True)
        # don't write pyproject.toml so visiting it raises
        with pytest.raises(PyprojectError, match="no pyproject.toml"):
            discover_src_roots(root)

    def test_workspace_clean(self, tmp_path: Path) -> None:
        root = tmp_path / "monorepo"
        root.mkdir()
        (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\n')
        for name in ("core", "observe"):
            pkg = root / "packages" / name
            pkg.mkdir(parents=True)
            (pkg / "src").mkdir()
            (pkg / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
        roots = discover_src_roots(root)
        assert (root / "packages" / "core" / "src").resolve() in roots
        assert (root / "packages" / "observe" / "src").resolve() in roots

    def test_workspace_exclude_drops_matching_dirs(self, tmp_path: Path) -> None:
        """parent grouping dirs flagged via exclude must be skipped.

        models the 3tears workspace shape: ``packages/*`` glob picks up
        every direct child including the bare ``packages/agent``
        grouping directory which holds no pyproject of its own; the
        ``exclude = ["packages/agent"]`` clause tells uv (and this
        walker) to ignore it. without exclude support the walker would
        try to read ``packages/agent/pyproject.toml`` and raise
        :class:`PyprojectError`.
        """
        root = tmp_path / "monorepo"
        root.mkdir()
        (root / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["packages/*", "packages/agent/*"]\nexclude = ["packages/agent"]\n'
        )
        for name in ("core", "observe"):
            pkg = root / "packages" / name
            pkg.mkdir(parents=True)
            (pkg / "src").mkdir()
            (pkg / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
        # bare grouping directory under packages/agent: no pyproject of its own.
        agent_dir = root / "packages" / "agent"
        agent_dir.mkdir(parents=True)
        for name in ("acl", "tools"):
            pkg = agent_dir / name
            pkg.mkdir(parents=True)
            (pkg / "src").mkdir()
            (pkg / "pyproject.toml").write_text(f'[project]\nname = "agent-{name}"\n')
        roots = discover_src_roots(root)
        assert (root / "packages" / "core" / "src").resolve() in roots
        assert (root / "packages" / "agent" / "acl" / "src").resolve() in roots
        assert (root / "packages" / "agent" / "tools" / "src").resolve() in roots

    def test_workspace_exclude_must_be_list(self, tmp_path: Path) -> None:
        """malformed exclude raises rather than silently ignoring it."""
        root = tmp_path / "monorepo"
        root.mkdir()
        (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\nexclude = "not-a-list"\n')
        with pytest.raises(PyprojectError, match="exclude must be a list"):
            discover_src_roots(root)


class TestUvSources:
    def test_workspace_true(self, tmp_path: Path) -> None:
        # workspace at tmp_path with two members
        root = tmp_path / "monorepo"
        root.mkdir()
        (root / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["packages/*"]\n[tool.uv.sources]\ncore = {workspace = true}\n'
        )
        core_pkg = root / "packages" / "core"
        core_pkg.mkdir(parents=True)
        (core_pkg / "src").mkdir()
        (core_pkg / "pyproject.toml").write_text('[project]\nname = "core"\n')
        roots = discover_src_roots(root)
        assert (core_pkg / "src").resolve() in roots

    def test_workspace_true_with_unresolved_name_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "monorepo"
        root.mkdir()
        (root / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["packages/*"]\n[tool.uv.sources]\nghost = {workspace = true}\n'
        )
        with pytest.raises(PyprojectError, match="workspace=true"):
            discover_src_roots(root)

    def test_explicit_path(self, tmp_path: Path) -> None:
        target = _make_repo(
            tmp_path / "lib",
            '[project]\nname = "lib"\n',
        )
        consumer_dir = tmp_path / "consumer"
        consumer_dir.mkdir()
        (consumer_dir / "src").mkdir()
        (consumer_dir / "pyproject.toml").write_text(
            f'[project]\nname = "consumer"\n[tool.uv.sources]\nlib = {{path = "../lib", editable = true}}\n'
        )
        roots = discover_src_roots(consumer_dir)
        assert (target / "src").resolve() in roots


class TestTransitiveResolution:
    def test_transitive_chain(self, tmp_path: Path) -> None:
        # consumer -> middle -> bottom
        bottom = _make_repo(
            tmp_path / "bottom",
            '[tool.poetry]\nname = "bottom"\n',
        )
        middle_text = '[tool.poetry]\nname = "middle"\n[tool.poetry.dependencies]\nbottom = {path = "../bottom"}\n'
        middle = _make_repo(tmp_path / "middle", middle_text)
        consumer_text = '[tool.poetry]\nname = "consumer"\n[tool.poetry.dependencies]\nmiddle = {path = "../middle"}\n'
        consumer = _make_repo(tmp_path / "consumer", consumer_text)
        roots = discover_src_roots(consumer)
        assert (bottom / "src").resolve() in roots
        assert (middle / "src").resolve() in roots
        assert (consumer / "src").resolve() in roots


class TestCycleHandling:
    def test_cycle_terminates(self, tmp_path: Path) -> None:
        # a -> b -> a
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "src").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "src").mkdir()
        (tmp_path / "a" / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "a"\n[tool.poetry.dependencies]\nb = {path = "../b"}\n'
        )
        (tmp_path / "b" / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "b"\n[tool.poetry.dependencies]\na = {path = "../a"}\n'
        )
        roots = discover_src_roots(tmp_path / "a")
        assert (tmp_path / "a" / "src").resolve() in roots
        assert (tmp_path / "b" / "src").resolve() in roots


class TestErrors:
    def test_malformed_toml_raises(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "pyproject.toml").write_text("[broken\n")
        with pytest.raises(PyprojectError, match="invalid TOML"):
            discover_src_roots(repo)

    def test_missing_pyproject_raises(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(PyprojectError, match="no pyproject.toml"):
            discover_src_roots(repo)

    def test_uv_workspace_members_must_be_list(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = "not-a-list"\n')
        with pytest.raises(PyprojectError, match="must be a list"):
            discover_src_roots(repo)

    def test_uv_sources_path_must_be_string(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "pyproject.toml").write_text("[tool.uv.sources]\nlib = {path = 42}\n")
        with pytest.raises(PyprojectError, match="must be a string"):
            discover_src_roots(repo)


class TestRealPyprojectShapesRoundTrip:
    """smoke-test the four real consumer pyproject shapes parse without error."""

    def test_3tears_root(self) -> None:
        # 3tears root uses [tool.uv.workspace] members = ["packages/*"]
        root = Path(__file__).resolve().parents[3].parent
        # path: packages/enforcement/tests/_common/test_pyproject_discovery.py
        # parents[3] = packages/enforcement, parent = packages, parent = 3tears
        # but tests are inside the package, so simpler:
        # packages/enforcement/tests/_common -> parent x4 = 3tears repo root
        threetears = Path(__file__).resolve().parents[4]
        if not (threetears / "pyproject.toml").exists():
            pytest.skip(f"3tears repo root not at expected location: {threetears}")
        roots = discover_src_roots(threetears)
        assert any(r.name == "src" for r in roots)
