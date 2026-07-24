"""git checkout identity — tell one repo's two checkouts apart.

enforcement walkers resolve modules to files across a set of src roots
discovered by following path-deps (:mod:`threetears.enforcement.common
.pyproject_discovery`). that walk assumes one checkout per repo. git
worktrees break the assumption: developing repo A in a worktree, where A
path-deps to sibling repo B and B path-deps back to A's ORIGINAL
checkout, yields a root set holding TWO copies of A's package tree.
module resolution then has two candidate files for every one of A's
modules, and walkers report phantom violations in files nobody touched.

this module supplies the signal needed to collapse that: a checkout's
``(common_git_dir, worktree_root)`` pair. two checkouts of the same repo
agree on the first and differ on the second; two genuinely separate
repos differ on the first. crucially, a directory NESTED inside a
checkout (a uv workspace member such as ``packages/core``) reports its
owning checkout, so members of a single repo all share one identity and
are never mistaken for rival checkouts.

read straight off disk rather than by shelling out to ``git``:
enforcement runs in-process under pytest with a sub-15s budget, and the
two layouts are a stable, documented part of git's repository format:

- an ordinary checkout has a ``.git`` DIRECTORY, which is the common
  git dir.
- a worktree has a ``.git`` FILE reading ``gitdir: <path>``, where
  ``<path>`` holds a ``commondir`` file locating the shared git dir
  (relative to ``<path>``, conventionally ``../..``).

every failure mode degrades to ``None``. identity is an OPTIONAL
refinement -- losing it costs only the dedupe, whereas raising would
take down an enforcement run over a repository-layout quirk unrelated
to the rule being checked.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "find_worktree_root",
    "repo_identity",
]

#: prefix of the single line a git worktree's ``.git`` file contains.
_GITDIR_PREFIX = "gitdir:"

#: filename inside a worktree's private gitdir holding the (usually
#: relative) path to the repository's shared git dir.
_COMMONDIR_FILENAME = "commondir"


def find_worktree_root(start: Path) -> Path | None:
    """walk up from ``start`` to the checkout root that owns it.

    the checkout root is the first ancestor (``start`` itself included)
    holding a ``.git`` entry, whether that is an ordinary checkout's
    directory or a worktree's pointer file.

    :param start: directory to begin the upward walk from
    :ptype start: Path
    :return: resolved checkout root, or ``None`` when ``start`` is not
        inside a git checkout
    :rtype: Path | None
    """
    current = start.resolve()
    result: Path | None = None
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            result = candidate
            break
    return result


def repo_identity(start: Path) -> tuple[Path, Path] | None:
    """resolve ``start`` to its ``(common_git_dir, worktree_root)`` pair.

    ``common_git_dir`` identifies the REPOSITORY and is shared by every
    checkout of it; ``worktree_root`` identifies the CHECKOUT. compare
    the first to ask "same repo?" and the second to ask "same working
    copy?".

    :param start: directory to identify (need not be a checkout root)
    :ptype start: Path
    :return: resolved ``(common_git_dir, worktree_root)``, or ``None``
        when ``start`` is outside any git checkout or the ``.git``
        pointer cannot be parsed
    :rtype: tuple[Path, Path] | None
    """
    worktree_root = find_worktree_root(start)
    result: tuple[Path, Path] | None = None
    if worktree_root is not None:
        common_git_dir = _resolve_common_git_dir(worktree_root / ".git")
        if common_git_dir is not None:
            result = (common_git_dir, worktree_root)
    return result


def _resolve_common_git_dir(git_entry: Path) -> Path | None:
    """resolve a ``.git`` entry to the repository's shared git dir.

    :param git_entry: the ``.git`` directory or pointer file
    :ptype git_entry: Path
    :return: resolved common git dir, or ``None`` when the pointer file
        is unreadable or does not carry a ``gitdir:`` line
    :rtype: Path | None
    """
    result: Path | None = None
    if git_entry.is_dir():
        result = git_entry.resolve()
    else:
        private_git_dir = _read_gitdir_pointer(git_entry)
        if private_git_dir is not None:
            result = _read_commondir(private_git_dir)
    return result


def _read_gitdir_pointer(git_file: Path) -> Path | None:
    """read the ``gitdir: <path>`` target out of a worktree's ``.git`` file.

    :param git_file: the ``.git`` pointer file
    :ptype git_file: Path
    :return: resolved private gitdir, or ``None`` when the file is
        unreadable or carries no ``gitdir:`` line
    :rtype: Path | None
    """
    try:
        content = git_file.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return None
    result: Path | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(_GITDIR_PREFIX):
            raw_target = stripped[len(_GITDIR_PREFIX) :].strip()
            if raw_target:
                # a relative gitdir is anchored to the directory holding
                # the pointer file, matching git's own interpretation.
                result = (git_file.parent / raw_target).resolve()
            break
    return result


def _read_commondir(private_git_dir: Path) -> Path:
    """resolve a worktree's private gitdir to the repository's shared one.

    falls back to ``private_git_dir`` itself when no ``commondir`` file
    is present -- the shape older git versions and submodule-style
    ``.git`` pointers produce. that fallback still yields a usable
    identity (it is stable and unique per checkout); it merely cannot
    recognise sibling worktrees, which is exactly the pre-existing
    behaviour and strictly better than refusing to classify at all.

    :param private_git_dir: per-worktree gitdir named by the pointer file
    :ptype private_git_dir: Path
    :return: resolved common git dir
    :rtype: Path
    """
    commondir_file = private_git_dir / _COMMONDIR_FILENAME
    result = private_git_dir.resolve()
    try:
        raw_common = commondir_file.read_text(encoding="utf-8").strip()
    except OSError, UnicodeDecodeError:
        raw_common = ""
    if raw_common:
        result = (private_git_dir / raw_common).resolve()
    return result
