"""tests for ``repo_layout`` module."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common.repo_layout import (
    find_local_src_roots,
    find_repo_root,
)


def _touch(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestFindRepoRoot:
    def test_walks_upward_to_pyproject(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        nested = repo / "src" / "pkg" / "subpkg"
        nested.mkdir(parents=True)
        _touch(repo / "pyproject.toml", "[tool.x]\n")
        result = find_repo_root(nested)
        assert result == repo.resolve()

    def test_at_repo_root_returns_self(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pyproject.toml", "[tool.x]\n")
        result = find_repo_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_starting_from_file_walks_to_parent(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pyproject.toml", "[tool.x]\n")
        file_path = _touch(tmp_path / "src" / "x.py", "")
        result = find_repo_root(file_path)
        assert result == tmp_path.resolve()

    def test_raises_when_no_pyproject(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="no pyproject.toml"):
            find_repo_root(deep)


class TestFindLocalSrcRoots:
    def test_top_level_src(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        _touch(tmp_path / "pyproject.toml", "")
        result = find_local_src_roots(tmp_path)
        assert result == ((tmp_path / "src").resolve(),)

    def test_packages_monorepo(self, tmp_path: Path) -> None:
        (tmp_path / "packages" / "core" / "src").mkdir(parents=True)
        (tmp_path / "packages" / "observe" / "src").mkdir(parents=True)
        # decoy without src/
        (tmp_path / "packages" / "skipme").mkdir()
        result = find_local_src_roots(tmp_path)
        assert result == (
            (tmp_path / "packages" / "core" / "src").resolve(),
            (tmp_path / "packages" / "observe" / "src").resolve(),
        )

    def test_mixed_layout(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "packages" / "core" / "src").mkdir(parents=True)
        result = find_local_src_roots(tmp_path)
        assert (tmp_path / "src").resolve() in result
        assert (tmp_path / "packages" / "core" / "src").resolve() in result
        assert len(result) == 2

    def test_empty_tuple_when_nothing_found(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        result = find_local_src_roots(tmp_path)
        assert result == ()

    def test_packages_dir_without_src_subdir_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "packages" / "core" / "lib").mkdir(parents=True)
        result = find_local_src_roots(tmp_path)
        assert result == ()
