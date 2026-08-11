"""ruff-config discovery: what counts as this repo's config, and what never does.

The module's own history sets the two failure directions these tests hold
apart. Narrowing discovery once deleted reviewed ledger entries (the root-only
config blindness the module docstring records), so a *nested* config with no
``.git`` above it must keep being discovered. And widening once buried the
gate under another working tree's files: a worktree checked out under
``.claude/worktrees/`` was read as this repo's own ruff configs, and every
file in it became an unlisted-access finding. A nested checkout is another
tree's code -- its configs and its files are excluded together, and nothing
of this repo's own tree can legitimately sit below a nested ``.git``.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import all_exempted_files, ruff_configs

_RUFF_SLF001_PYPROJECT = '[tool.ruff.lint.per-file-ignores]\n"tests/**" = ["SLF001"]\n'


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_repo(tmp_path: Path) -> Path:
    """a repo root with its own config, a nested non-checkout config, and two nested checkouts.

    layout::

        repo/
          .git/                     the repo's own marker -- never excludes the root itself
          pyproject.toml            SLF001 ignore for tests/**
          tests/test_own.py
          packages/deep/ruff.toml   nested config, NO .git above it -- must stay discovered
          packages/deep/tests/test_deep.py
          .claude/worktrees/wt/     nested worktree: .git is a FILE
          vendor/clone/             nested clone: .git is a DIRECTORY
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    _write(repo / "pyproject.toml", _RUFF_SLF001_PYPROJECT)
    _write(repo / "tests" / "test_own.py", "x = 1\n")

    _write(repo / "packages" / "deep" / "ruff.toml", '[lint.per-file-ignores]\n"tests/**" = ["SLF001"]\n')
    _write(repo / "packages" / "deep" / "tests" / "test_deep.py", "x = 1\n")

    worktree = repo / ".claude" / "worktrees" / "wt"
    _write(worktree / ".git", "gitdir: elsewhere\n")
    _write(worktree / "pyproject.toml", _RUFF_SLF001_PYPROJECT)
    _write(worktree / "tests" / "test_foreign.py", "x = 1\n")

    clone = repo / "vendor" / "clone"
    (clone / ".git").mkdir(parents=True)
    _write(clone / "pyproject.toml", _RUFF_SLF001_PYPROJECT)
    _write(clone / "tests" / "test_cloned.py", "x = 1\n")

    return repo


class TestNestedCheckoutsAreNotThisRepo:
    def test_configs_below_a_nested_git_are_not_discovered(self, tmp_path: Path) -> None:
        """a worktree's .git file and a clone's .git directory both mark foreign trees."""
        repo = _make_repo(tmp_path)

        found = {config.relative_to(repo).as_posix() for config in ruff_configs(repo)}

        assert found == {"pyproject.toml", "packages/deep/ruff.toml"}, (
            "nested-checkout configs must be excluded WITHOUT losing nested configs of this "
            f"repo's own tree -- the root-only blindness incident -- got {sorted(found)}"
        )

    def test_exempted_files_below_a_nested_git_are_not_this_repos_findings(self, tmp_path: Path) -> None:
        """the visible symptom of the defect: every worktree file became an unlisted access."""
        repo = _make_repo(tmp_path)

        exempted = {path.relative_to(repo).as_posix() for path in all_exempted_files(repo)}

        assert exempted == {"tests/test_own.py", "packages/deep/tests/test_deep.py"}

    def test_the_repos_own_git_directory_excludes_nothing(self, tmp_path: Path) -> None:
        """only a .git strictly BELOW the root marks a foreign tree; the root's own never can."""
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        _write(repo / "pyproject.toml", _RUFF_SLF001_PYPROJECT)
        _write(repo / "tests" / "test_own.py", "x = 1\n")

        assert [config.name for config in ruff_configs(repo)] == ["pyproject.toml"]
        assert [path.name for path in all_exempted_files(repo)] == ["test_own.py"]
