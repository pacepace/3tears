"""repo-anchoring helpers — find the root, find the local src trees.

walkers always anchor on a repo's root (the directory containing the
top-level ``pyproject.toml``) and discover the source trees under it
without speculating about which layout style the repo uses.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "find_local_src_roots",
    "find_repo_root",
]


def find_repo_root(start: Path) -> Path:
    """walk upward from ``start`` to the nearest directory with ``pyproject.toml``.

    every consumer repo anchors on a single ``pyproject.toml``; the
    walker's per-test entry point passes ``Path(__file__)`` and expects
    this helper to find the corresponding repo root regardless of
    package layout.

    :param start: path to start search from
    :ptype start: Path
    :return: directory containing the nearest ``pyproject.toml``
    :rtype: Path
    :raises RuntimeError: no ``pyproject.toml`` ancestor exists
    """
    current = start.resolve()
    candidates = [current, *current.parents] if current.is_dir() else list(current.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"no pyproject.toml ancestor found above {start}")


def find_local_src_roots(repo_root: Path) -> tuple[Path, ...]:
    """discover this repo's own ``src/`` trees, sorted for stable order.

    recognises two layouts:

    - **single-package**: a top-level ``src/`` directory under the repo
      root (the standard single-package layout).
    - **monorepo / uv workspace**: ``packages/*/src/`` directories (the
      3tears layout).

    a repo with both shapes returns both. a repo with neither returns
    an empty tuple.

    this helper does NOT follow path-deps; that responsibility belongs
    to :mod:`threetears.enforcement.common.pyproject_discovery`. keep
    the two concerns separate so domain walkers can pick: scan only
    this repo, or scan this repo + every transitively-reachable
    sibling.

    :param repo_root: absolute repo root path
    :ptype repo_root: Path
    :return: sorted tuple of absolute src-root paths
    :rtype: tuple[Path, ...]
    """
    roots: set[Path] = set()
    repo_root = repo_root.resolve()

    top_level = repo_root / "src"
    if top_level.is_dir():
        roots.add(top_level)

    packages_dir = repo_root / "packages"
    if packages_dir.is_dir():
        for package in sorted(packages_dir.iterdir(), key=lambda p: p.name):
            if not package.is_dir():
                continue
            candidate = package / "src"
            if candidate.is_dir():
                roots.add(candidate)

    return tuple(sorted(roots))
