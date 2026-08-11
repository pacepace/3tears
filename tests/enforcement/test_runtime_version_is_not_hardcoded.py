"""
enforcement: a package's runtime ``__version__`` is read, never written.

Every ``3tears*`` package that exposes ``__version__`` derives it from
installed package metadata::

    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("3tears-epoch")
    except _PackageNotFoundError:  # pragma: no cover - dev fallback
        __version__ = "unknown"

and the comment above that block in a dozen packages says exactly why: "a
future release that bumps pyproject without updating ``__init__.py`` can't
drift the runtime ``__version__``."

``3tears-media-contracts`` was the one that had it written out as a literal.
It read ``__version__ = "0.10.6"`` while the package shipped at 0.24.0 --
fourteen minor releases of drift, published to PyPI every time, reporting a
version the package had not been since well before the family convergence
work started. Nothing failed, which is the point: a hardcoded version is
wrong silently and stays wrong until somebody reads it and believes it.

The lockstep release rule (``CLAUDE.md``) makes this worse than it looks. The
bump is mechanical across ~30 packages and touches ``pyproject.toml``; a
literal in a *different* file is exactly what a mechanical bump walks past.
The version-bounds guard next door catches the same class of drift in
dependency declarations, and it exists because the same thing happened there.

Static parsing only (no imports, no network), consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*", "packages/agent/*")


def _package_inits() -> list[Path]:
    """Every leaf ``__init__.py`` a workspace package publishes.

    :return: the ``__init__.py`` paths under each package's ``src/``
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for package in sorted(_REPO_ROOT.glob(glob)):
            source = package / "src"
            if not source.is_dir():
                continue
            found.extend(sorted(source.rglob("__init__.py")))
    return found


def _hardcoded_version_lines(tree: ast.Module) -> list[int]:
    """Line numbers where ``__version__`` is assigned a literal string.

    A module-level ``__version__ = "1.2.3"`` is the defect; every other
    assignment -- from a call, a name, a conditional -- is reading something
    rather than restating it.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: 1-based line numbers of literal assignments, if any
    :rtype: list[int]
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # "unknown" is the sanctioned fallback in the except branch: it
            # states that the version is not knowable, which is the opposite
            # of asserting one.
            if node.value.value != "unknown":
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("init_path", _package_inits(), ids=lambda path: str(path.relative_to(_REPO_ROOT)))
def test_runtime_version_is_read_from_metadata(init_path: Path) -> None:
    """A version literal in source is a version that will be wrong later.

    :param init_path: the ``__init__.py`` under inspection
    :ptype init_path: Path
    :return: nothing
    :rtype: None
    """
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    offending = _hardcoded_version_lines(tree)
    relative = init_path.relative_to(_REPO_ROOT)
    assert not offending, (
        f"{relative} assigns __version__ a literal string at line(s) {offending}. "
        "Derive it from installed metadata instead -- importlib.metadata.version('<pypi-name>') "
        "inside a try/except PackageNotFoundError, with 'unknown' as the fallback. A literal here "
        "survives the lockstep version bump untouched and reports a version the package is not."
    )
