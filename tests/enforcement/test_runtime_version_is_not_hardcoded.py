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

    Annotated assignments count. ``__version__: Final[str] = "1.2.3"`` drifts
    exactly as silently as the bare form, and it is the likelier spelling for
    a reintroduction here: this codebase annotates module constants with
    ``Final`` as a matter of course, so the shape a future author reaches for
    is the annotated one. A guard that read only ``ast.Assign`` would pass it.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: 1-based line numbers of literal assignments, if any
    :rtype: list[int]
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            # ``__version__: str`` with no value declares a type and asserts
            # no version, so there is nothing to drift.
            if node.value is None:
                continue
            value = node.value
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            # "unknown" is the sanctioned fallback in the except branch: it
            # states that the version is not knowable, which is the opposite
            # of asserting one.
            if value.value != "unknown":
                lines.append(node.lineno)
    return lines


#: shapes the walker must classify, kept beside it because the gap this pins
#: was live: the first version of the walker read only ``ast.Assign``, so an
#: annotated literal passed the guard whose whole purpose is to refuse one.
_WALKER_CASES = [
    pytest.param('__version__ = "0.10.6"', True, id="bare-literal"),
    pytest.param('__version__: str = "0.10.6"', True, id="annotated-literal"),
    pytest.param('__version__: Final[str] = "0.10.6"', True, id="final-annotated-literal"),
    pytest.param('__version__ = "unknown"', False, id="sanctioned-unknown-fallback"),
    pytest.param('__version__ = _version("3tears-x")', False, id="read-from-metadata"),
    pytest.param("__version__: str", False, id="annotation-without-value"),
]


@pytest.mark.parametrize(("source", "is_hardcoded"), _WALKER_CASES)
def test_the_walker_classifies_a_version_assignment(source: str, is_hardcoded: bool) -> None:
    """The detector itself, so a narrowed walker fails here rather than silently.

    Without this, shrinking the walker breaks nothing: every package already
    complies, so the parametrized scan over real files passes just as happily
    against a detector that finds nothing at all.

    :param source: the assignment under inspection
    :ptype source: str
    :param is_hardcoded: whether the walker must flag it
    :ptype is_hardcoded: bool
    :return: nothing
    :rtype: None
    """
    assert bool(_hardcoded_version_lines(ast.parse(source))) is is_hardcoded


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
