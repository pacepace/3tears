"""tests for ``repo_layout`` module."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from threetears.enforcement.common.repo_layout import (
    find_local_src_roots,
    find_repo_root,
)

#: this repo's own root — ``packages/enforcement/tests/common/`` is four levels down.
_THIS_REPO_ROOT = Path(__file__).resolve().parents[4]

#: the smallest number of src roots this repo can plausibly have. the number exists to make a
#: regression that returns nothing (or nearly nothing) FAIL rather than read as a clean pass:
#: the helper once walked ``packages/*/src`` only, which silently dropped ten nested packages,
#: and every gate built on it reported green over source it never opened.
_MINIMUM_PLAUSIBLE_SRC_ROOTS = 25


def _touch(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _declared_workspace_members(repo_root: Path) -> tuple[Path, ...]:
    """every uv workspace member directory declared by ``repo_root``'s pyproject.

    reads ``[tool.uv.workspace].members`` directly rather than re-deriving the layout, so this
    is an INDEPENDENT statement of what a package is in this repo: the helper under test may
    not simply agree with itself.

    :param repo_root: absolute repo root path
    :ptype repo_root: Path
    :return: sorted tuple of resolved member directories that carry a ``pyproject.toml``
    :rtype: tuple[Path, ...]
    """
    with (repo_root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    globs = data["tool"]["uv"]["workspace"]["members"]
    members: set[Path] = set()
    for glob in globs:
        for match in repo_root.glob(glob):
            if match.is_dir() and (match / "pyproject.toml").is_file():
                members.add(match.resolve())
    return tuple(sorted(members))


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

    def test_nested_packages_are_found(self, tmp_path: Path) -> None:
        (tmp_path / "packages" / "core" / "src").mkdir(parents=True)
        (tmp_path / "packages" / "agent" / "memory" / "src").mkdir(parents=True)
        (tmp_path / "packages" / "agent" / "wake" / "src").mkdir(parents=True)
        result = find_local_src_roots(tmp_path)
        assert result == (
            (tmp_path / "packages" / "agent" / "memory" / "src").resolve(),
            (tmp_path / "packages" / "agent" / "wake" / "src").resolve(),
            (tmp_path / "packages" / "core" / "src").resolve(),
        )

    def test_arbitrarily_deep_nesting_is_found(self, tmp_path: Path) -> None:
        deep = tmp_path / "packages" / "group" / "subgroup" / "leaf" / "src"
        deep.mkdir(parents=True)
        result = find_local_src_roots(tmp_path)
        assert result == (deep.resolve(),)

    def test_does_not_descend_into_a_discovered_src_tree(self, tmp_path: Path) -> None:
        outer = tmp_path / "packages" / "core" / "src"
        (outer / "threetears" / "core" / "src").mkdir(parents=True)
        result = find_local_src_roots(tmp_path)
        assert result == (outer.resolve(),)

    def test_tooling_cache_dirs_are_not_mistaken_for_packages(self, tmp_path: Path) -> None:
        (tmp_path / "packages" / "core" / "src").mkdir(parents=True)
        (tmp_path / "packages" / "core" / ".mypy_cache" / "3.14" / "src").mkdir(parents=True)
        (tmp_path / "packages" / "core" / ".venv" / "lib" / "src").mkdir(parents=True)
        (tmp_path / "packages" / "core" / "build" / "src").mkdir(parents=True)
        result = find_local_src_roots(tmp_path)
        assert result == ((tmp_path / "packages" / "core" / "src").resolve(),)


class TestFindLocalSrcRootsOverThisRepo:
    """non-vacuity guard — the lesson that produced the widening.

    a walker that scans NOTHING reports exactly what a walker that finds nothing reports. these
    tests assert against the live repo so a discovery regression fails loudly instead of turning
    every downstream gate green over unread source.
    """

    def test_result_is_non_empty(self) -> None:
        assert find_local_src_roots(_THIS_REPO_ROOT)

    def test_result_is_of_the_expected_order_of_magnitude(self) -> None:
        result = find_local_src_roots(_THIS_REPO_ROOT)
        assert len(result) >= _MINIMUM_PLAUSIBLE_SRC_ROOTS

    def test_covers_every_declared_workspace_member_that_has_a_src_tree(self) -> None:
        found = set(find_local_src_roots(_THIS_REPO_ROOT))
        expected = {
            member / "src" for member in _declared_workspace_members(_THIS_REPO_ROOT) if (member / "src").is_dir()
        }
        assert expected
        assert expected <= found

    def test_covers_the_nested_agent_family(self) -> None:
        found = set(find_local_src_roots(_THIS_REPO_ROOT))
        agent_dir = _THIS_REPO_ROOT / "packages" / "agent"
        nested = {child / "src" for child in agent_dir.iterdir() if (child / "src").is_dir()}
        assert len(nested) >= 10
        assert nested <= found
