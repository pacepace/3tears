"""The subdirectories a source-tree gate treats as the tree's own, for canaries that derive their input.

Shared so that every gate deriving a population from the directories under a package agrees on
what counts as one — today the engine's classification canary
(``test_no_host_names_in_shared_contract.py``, every engine tree is classified).

**A directory holding nothing but ``__pycache__`` is not the tree's.** A move or a deletion that
arrives by ``git pull`` removes the tracked files and leaves the untracked bytecode caches, so
every checkout that had imported the old tree keeps a directory the repository no longer has. A
gate that counts it goes red on the checkouts that pulled the move and green on a fresh clone,
which reads as a flake rather than as the gate reading a local artifact.

Only the interpreter's own cache is set aside. A directory holding any other file still counts,
whether or not that file is Python: a data-only tree (the eval seed corpus) is part of the
repository, and the gates that derive from these directories refuse such a tree until it is
classified.
"""

from __future__ import annotations

from pathlib import Path

#: The directory the interpreter writes bytecode into beside every imported module.
_BYTECODE_CACHE = "__pycache__"


def _holds_more_than_bytecode(directory: Path) -> bool:
    """Whether any file under ``directory`` lies outside a bytecode cache.

    Args:
        directory: The directory to inspect.

    Returns:
        ``True`` when at least one file is not inside a ``__pycache__`` directory.
    """
    return any(
        path.is_file() and _BYTECODE_CACHE not in path.relative_to(directory).parts for path in directory.rglob("*")
    )


def tree_subdirectories(root: Path) -> list[Path]:
    """The immediate subdirectories of ``root`` that belong to the source tree, sorted by name.

    Private (``_``-prefixed) and hidden (``.``-prefixed) directories are not trees, and neither is
    a directory whose only content is bytecode cache.

    Args:
        root: The package directory to list.

    Returns:
        The subdirectories that hold anything besides ``__pycache__``.
    """
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(("_", ".")) and _holds_more_than_bytecode(path)
    )
