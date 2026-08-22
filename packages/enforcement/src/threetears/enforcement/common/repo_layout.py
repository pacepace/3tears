"""repo-anchoring helpers — find the root, find the local src trees.

walkers always anchor on a repo's root (the directory containing the
top-level ``pyproject.toml``) and discover the source trees under it
without speculating about which layout style the repo uses.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.common.ast_helpers import SKIP_DIRS

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
    - **monorepo / uv workspace**: every ``src/`` directory found at ANY
      depth under ``packages/`` (the 3tears layout).

    a repo with both shapes returns both. a repo with neither returns
    an empty tuple.

    depth is deliberately unbounded rather than fixed at one level.
    3tears groups ten packages under ``packages/agent/`` for visual
    tidiness, and a ``packages/*/src`` walk returned NOTHING for them --
    which is indistinguishable, in every gate built on this helper, from
    finding nothing wrong. the layout is a naming choice; the discovery
    must not encode a guess about it, or the next grouping directory
    reopens the same silent hole.

    two exclusions keep the widened walk honest: directories named in
    :data:`threetears.enforcement.common.ast_helpers.SKIP_DIRS` and any
    dot-prefixed directory are never entered (a tooling cache such as
    ``packages/registry/.mypy_cache/3.14/src`` is a directory named
    ``src`` that holds no source), and a discovered ``src/`` tree is
    never itself descended into (``threetears/core/src`` inside a
    package's own sources is a module path, not a second package).

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
        _collect_src_roots(packages_dir, roots)

    return tuple(sorted(roots))


def _collect_src_roots(directory: Path, roots: set[Path]) -> None:
    """recurse through ``directory`` accumulating every package ``src/`` tree.

    a directory that owns a ``src/`` child contributes that child and is still descended into
    for siblings of it, but the ``src/`` tree itself is left alone -- package sources may
    legitimately contain a module path ending in ``src``.

    :param directory: directory to inspect
    :ptype directory: Path
    :param roots: accumulator of discovered src-root paths
    :ptype roots: set[Path]
    :return: nothing
    :rtype: None
    """
    candidate = directory / "src"
    if candidate.is_dir():
        roots.add(candidate.resolve())

    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIRS or entry.name.startswith("."):
            continue
        if entry == candidate:
            continue
        _collect_src_roots(entry, roots)
