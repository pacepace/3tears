"""Which engine package a module belongs to: its path, and nothing else.

``threetears.evals`` is four subpackages -- ``contracts``, ``run``, ``analysis`` and ``gen`` -- and the
gates over it agree on one rule: **a module is in a package because it lives in that package's
directory.** ``test_package_matrix.py`` holds each package to its row of the allowed-dependency
matrix and ``test_contracts_installable.py`` probes the contracts package standalone. Both need the
same answer to "where does this module live", so the answer lives here once rather than in one of
those files for the other to import.

Like :mod:`packages.evals.tests.import_resolution`, this is machinery with a single right answer: the
tree decides placement, no gate has a position on it, and a second copy is one more chance to hold
a different one. Standard library only.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "EVAL_PACKAGE",
    "PACKAGE_DIRS",
    "TREE_MARKERS",
    "discover",
    "eval_modules",
    "eval_relative",
    "join",
    "module_name",
    "modules_placed_in",
    "placement",
    "unplaced_modules",
]

#: The package every placed module hangs off.
EVAL_PACKAGE = "threetears.evals"

#: Package directory (dotted, relative to ``threetears.evals``) -> the package it places a module in.
PACKAGE_DIRS: dict[str, str] = {
    "contracts": "contracts",
    "run": "run",
    "analysis": "analysis",
    "gen": "gen",
}

#: The tree marker a path cannot place: the root is the namespace over the four packages and is held
#: to contracts' row, so it may import nothing but contracts.
TREE_MARKERS: dict[str, str] = {
    "": "contracts",
}


def module_name(path: Path, root: Path) -> str:
    """Dotted module name of ``path`` relative to the directory holding the top-level package.

    Args:
        path: A ``.py`` file under ``root``.
        root: The directory the dotted name is rooted at.

    Returns:
        The dotted name; a package's ``__init__`` is named by the package.
    """
    parts = path.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover(root: Path) -> dict[str, Path]:
    """Every module under ``root / 'threetears' / 'evals'``, keyed by dotted name.

    Args:
        root: The directory holding the ``threetears`` namespace -- the package's ``src``.

    Returns:
        Dotted name -> file.
    """
    return {module_name(path, root): path for path in sorted((root / "threetears" / "evals").rglob("*.py"))}


def eval_relative(module: str) -> str | None:
    """``module`` relative to ``threetears.evals`` ('' for the root), or None when outside it.

    Args:
        module: A dotted module name.

    Returns:
        The relative name, or None.
    """
    if module == EVAL_PACKAGE:
        return ""
    if module.startswith(EVAL_PACKAGE + "."):
        return module[len(EVAL_PACKAGE) + 1 :]
    return None


def join(relative: str) -> str:
    """The absolute dotted name of a name relative to ``threetears.evals``.

    Args:
        relative: A name as :func:`eval_relative` returns it.

    Returns:
        The dotted module name.
    """
    return f"{EVAL_PACKAGE}.{relative}" if relative else EVAL_PACKAGE


def placement(module: str) -> str | None:
    """The package ``module`` belongs to, from its path.

    Args:
        module: A dotted module name.

    Returns:
        ``"contracts"``, ``"run"``, ``"analysis"`` or ``"gen"``; ``"outside"`` for a module outside
        ``threetears.evals``; None for a module under it that no package directory holds.
    """
    relative = eval_relative(module)
    if relative is None:
        return "outside"
    if relative in TREE_MARKERS:
        return TREE_MARKERS[relative]
    for prefix in sorted(PACKAGE_DIRS, key=len, reverse=True):
        if relative == prefix or relative.startswith(prefix + "."):
            return PACKAGE_DIRS[prefix]
    return None


def eval_modules(root: Path) -> dict[str, Path]:
    """Every module under ``threetears/evals/``, keyed by its name relative to ``threetears.evals``.

    Args:
        root: The directory holding the ``threetears`` namespace.

    Returns:
        Relative name -> file.
    """
    found = {}
    for module, path in discover(root).items():
        relative = eval_relative(module)
        if relative is not None:
            found[relative] = path
    return found


def modules_placed_in(root: Path, package: str) -> dict[str, Path]:
    """Every module under ``threetears/evals/`` whose path places it in ``package``.

    Args:
        root: The directory holding the ``threetears`` namespace.
        package: A package name as :func:`placement` returns it.

    Returns:
        Relative name -> file, keyed like :func:`eval_modules`.
    """
    return {relative: path for relative, path in eval_modules(root).items() if placement(join(relative)) == package}


def unplaced_modules(root: Path) -> set[str]:
    """Modules under ``threetears/evals/`` whose path places them in no package.

    Args:
        root: The directory holding the ``threetears`` namespace.

    Returns:
        Their names relative to ``threetears.evals``.
    """
    return {relative for relative in eval_modules(root) if placement(join(relative)) is None}
