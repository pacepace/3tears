"""tests for ``git_layout`` module."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.common.git_layout import (
    find_worktree_root,
    repo_identity,
)


def _make_main_checkout(root: Path) -> Path:
    """build a directory shaped like an ordinary git checkout.

    :param root: directory to turn into a checkout
    :ptype root: Path
    :return: the same directory
    :rtype: Path
    """
    (root / ".git").mkdir(parents=True)
    return root


def _make_worktree(root: Path, main_checkout: Path, name: str) -> Path:
    """build a directory shaped like a git worktree of ``main_checkout``.

    mirrors git's real on-disk layout: the worktree root holds a ``.git``
    FILE pointing at ``<main>/.git/worktrees/<name>``, and that directory
    holds a ``commondir`` file whose contents locate the shared git dir.

    :param root: directory to turn into a worktree
    :ptype root: Path
    :param main_checkout: the worktree's originating checkout
    :ptype main_checkout: Path
    :param name: worktree name (the per-worktree gitdir basename)
    :ptype name: str
    :return: the same directory
    :rtype: Path
    """
    worktree_gitdir = main_checkout / ".git" / "worktrees" / name
    worktree_gitdir.mkdir(parents=True)
    (worktree_gitdir / "commondir").write_text("../..\n")
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text(f"gitdir: {worktree_gitdir}\n")
    return root


class TestFindWorktreeRoot:
    """``find_worktree_root`` walks up to the checkout that owns a path."""

    def test_checkout_root_itself(self, tmp_path: Path) -> None:
        """a checkout root resolves to itself."""
        checkout = _make_main_checkout(tmp_path / "repo")
        assert find_worktree_root(checkout) == checkout.resolve()

    def test_nested_directory_walks_up(self, tmp_path: Path) -> None:
        """a directory inside a checkout resolves to the checkout root."""
        checkout = _make_main_checkout(tmp_path / "repo")
        nested = checkout / "packages" / "core"
        nested.mkdir(parents=True)
        assert find_worktree_root(nested) == checkout.resolve()

    def test_worktree_root_is_found_via_git_file(self, tmp_path: Path) -> None:
        """a ``.git`` FILE marks a worktree root just as a dir marks a checkout."""
        main = _make_main_checkout(tmp_path / "repo")
        worktree = _make_worktree(tmp_path / "repo-feature", main, "repo-feature")
        assert find_worktree_root(worktree) == worktree.resolve()

    def test_outside_any_repo_returns_none(self, tmp_path: Path) -> None:
        """a path with no ``.git`` above it resolves to None."""
        loose = tmp_path / "not_a_repo"
        loose.mkdir()
        assert find_worktree_root(loose) is None


class TestRepoIdentity:
    """``repo_identity`` pairs the shared git dir with the owning checkout."""

    def test_main_checkout_identity(self, tmp_path: Path) -> None:
        """an ordinary checkout's common git dir is its own ``.git``."""
        checkout = _make_main_checkout(tmp_path / "repo")
        identity = repo_identity(checkout)
        assert identity == ((checkout / ".git").resolve(), checkout.resolve())

    def test_worktree_shares_common_git_dir_with_main(self, tmp_path: Path) -> None:
        """this is the whole point: same repo, different checkout root.

        a worktree and its originating checkout must agree on the common
        git dir and disagree on the worktree root -- that pair is what
        lets a consumer tell "another checkout of the repo I am already
        looking at" apart from "a genuinely different repo".
        """
        main = _make_main_checkout(tmp_path / "repo")
        worktree = _make_worktree(tmp_path / "repo-feature", main, "repo-feature")
        main_identity = repo_identity(main)
        worktree_identity = repo_identity(worktree)
        assert main_identity is not None
        assert worktree_identity is not None
        assert main_identity[0] == worktree_identity[0]
        assert main_identity[1] != worktree_identity[1]

    def test_nested_path_reports_owning_checkout(self, tmp_path: Path) -> None:
        """a workspace member inside a checkout shares that checkout's identity.

        3tears' own ``packages/*`` members live in ONE git repo, so they
        must NOT look like separate checkouts -- otherwise dedupe by
        identity would drop every member but the first.
        """
        checkout = _make_main_checkout(tmp_path / "repo")
        member = checkout / "packages" / "core"
        member.mkdir(parents=True)
        assert repo_identity(member) == repo_identity(checkout)

    def test_two_independent_repos_have_distinct_identities(self, tmp_path: Path) -> None:
        """separate clones are separate repos, never deduped against each other."""
        first = _make_main_checkout(tmp_path / "alpha")
        second = _make_main_checkout(tmp_path / "beta")
        assert repo_identity(first) != repo_identity(second)

    def test_outside_any_repo_returns_none(self, tmp_path: Path) -> None:
        """a path with no ``.git`` above it has no identity."""
        loose = tmp_path / "not_a_repo"
        loose.mkdir()
        assert repo_identity(loose) is None

    def test_worktree_without_commondir_falls_back_to_gitdir(self, tmp_path: Path) -> None:
        """a ``.git`` file with no ``commondir`` still yields an identity.

        older git versions and hand-rolled submodule-style ``.git`` files
        point at a gitdir with no ``commondir``; treat the gitdir itself
        as the identity rather than refusing to classify the checkout.
        """
        gitdir = tmp_path / "elsewhere" / "gitdir"
        gitdir.mkdir(parents=True)
        root = tmp_path / "repo-linked"
        root.mkdir()
        (root / ".git").write_text(f"gitdir: {gitdir}\n")
        assert repo_identity(root) == (gitdir.resolve(), root.resolve())

    def test_malformed_git_file_degrades_to_none(self, tmp_path: Path) -> None:
        """an unparseable ``.git`` file yields None rather than raising.

        identity is an OPTIONAL refinement: returning None costs only the
        dedupe, while raising would take down an enforcement run over a
        layout quirk that has nothing to do with the rule being checked.
        """
        root = tmp_path / "weird"
        root.mkdir()
        (root / ".git").write_text("this is not a gitdir pointer\n")
        assert repo_identity(root) is None
