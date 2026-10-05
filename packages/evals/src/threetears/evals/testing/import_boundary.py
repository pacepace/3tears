"""A check an adopter runs over its own source tree: every ``threetears.evals`` import comes from a public root.

The package's API is its public roots (:data:`threetears.evals.PUBLIC_ROOTS`) and the names each root's
``__all__`` declares. Everything below a root is implementation: it moves, splits and is renamed in any
release, and a host that reached it breaks with an ``ImportError`` the package's own version gate never
warned about. The package holds its own tree to this (``tests/test_package_matrix.py``); this is the same
rule for a host's tree, so a host's suite fails on the import rather than on the upgrade.

Use it from any test runner::

    from pathlib import Path

    from threetears.evals.testing import nonpublic_evals_imports


    def test_every_evals_import_is_public() -> None:
        assert nonpublic_evals_imports(Path("src"), Path("tests")) == ()

**What it reads.** Every ``import`` and ``from ... import`` statement in every ``.py`` file under the
paths given — at module level, under ``TYPE_CHECKING`` and inside functions alike, since each is a name
the host's code binds to. It parses; it does not execute the host's code. It imports each public root a
statement names, to read that root's ``__all__``, so a root that needs an extra (``vega``,
``transports.fastmcp``) must be importable where the check runs — as it must be where the host's code runs.

**What it cannot see.** ``importlib.import_module("threetears.evals...")`` with a computed name, and an
attribute reached through a module object (``import threetears.evals.contracts as c; c._helper``).
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from threetears.evals import PUBLIC_ROOTS

#: The package itself. Importing it is allowed (it declares :data:`~threetears.evals.PUBLIC_ROOTS`), and
#: so is ``from threetears.evals import <root>`` for a public root directly below it.
_PACKAGE = "threetears.evals"


@dataclass(frozen=True)
class NonPublicImport:
    """One import of a ``threetears.evals`` name that does not come from a public root.

    Attributes:
        path: The file the statement is in.
        line: Its line number.
        module: The module it imports from, or the module it imports.
        name: The name it binds from ``module``; None for an ``import`` statement.
        reason: Why it is not public, and where the name is public when it is anywhere.
    """

    path: Path
    line: int
    module: str
    name: str | None
    reason: str

    def __str__(self) -> str:
        """The finding as one line: where, what, and why.

        Returns:
            ``path:line: from module import name — reason``.
        """
        statement = f"import {self.module}" if self.name is None else f"from {self.module} import {self.name}"
        return f"{self.path}:{self.line}: {statement} — {self.reason}"


def nonpublic_evals_imports(*sources: Path | str) -> tuple[NonPublicImport, ...]:
    """Every ``threetears.evals`` import under ``sources`` that reaches below a public root.

    A statement is public when it imports from a public root a name in that root's ``__all__``, imports
    a public root itself (``import threetears.evals.run``; ``from threetears.evals import run``), or
    imports the package's own root. Anything else is reported: a module below a root, a name a root does
    not declare, or a star import from a root.

    Args:
        sources: Files and directories to read; a directory is read recursively for ``.py`` files.

    Returns:
        One finding per offending name, in path and line order; empty when every import is public.

    Raises:
        FileNotFoundError: A source does not exist — a check over a path that is not there would pass
            by reading nothing.
        SyntaxError: A file does not parse.
    """
    exported: dict[str, frozenset[str]] = {}
    findings: list[NonPublicImport] = []
    for path in _python_files(sources):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                findings.extend(
                    NonPublicImport(path=path, line=node.lineno, module=alias.name, name=None, reason=reason)
                    for alias in node.names
                    if (reason := _module_import_defect(alias.name)) is not None
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                findings.extend(
                    NonPublicImport(path=path, line=node.lineno, module=node.module, name=alias.name, reason=reason)
                    for alias in node.names
                    if (reason := _from_import_defect(node.module, alias.name, exported)) is not None
                )
    return tuple(sorted(findings, key=lambda finding: (str(finding.path), finding.line, finding.name or "")))


def _python_files(sources: tuple[Path | str, ...]) -> Iterator[Path]:
    """Every ``.py`` file the sources name, each once, sorted.

    Args:
        sources: Files and directories.

    Yields:
        The files.

    Raises:
        FileNotFoundError: A source does not exist.
    """
    seen: set[Path] = set()
    for source in map(Path, sources):
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist, so nothing under it could be checked")
        for path in sorted(source.rglob("*.py")) if source.is_dir() else [source]:
            if path not in seen:
                seen.add(path)
                yield path


def _in_package(module: str) -> bool:
    """Whether ``module`` is the package or a module below it.

    Args:
        module: An absolute module name.

    Returns:
        Whether it is ``threetears.evals`` or starts with ``threetears.evals.``.
    """
    return module == _PACKAGE or module.startswith(f"{_PACKAGE}.")


def _module_import_defect(module: str) -> str | None:
    """Why ``import <module>`` reaches below a public root, or None when it does not.

    Args:
        module: The module imported.

    Returns:
        One sentence, or None.
    """
    if not _in_package(module) or module == _PACKAGE or module in PUBLIC_ROOTS:
        return None
    return f"{module} is below a public root; import from {_root_above(module)} instead"


def _from_import_defect(module: str, name: str, exported: dict[str, frozenset[str]]) -> str | None:
    """Why ``from <module> import <name>`` reaches below a public root, or None when it does not.

    Args:
        module: The module imported from.
        name: The name bound.
        exported: Each public root's ``__all__``, filled as roots are first named.

    Returns:
        One sentence, or None.
    """
    if not _in_package(module):
        return None
    if f"{module}.{name}" in PUBLIC_ROOTS:
        return None
    if module == _PACKAGE:
        return None if name in _exports(_PACKAGE, exported) else _undeclared(module, name, exported)
    if module not in PUBLIC_ROOTS:
        homes = _public_homes(name, exported)
        where = f"import it from {' or '.join(homes)}" if homes else "no public root exports it"
        return f"{module} is below a public root; {where}"
    if name == "*" or name not in _exports(module, exported):
        return _undeclared(module, name, exported)
    return None


def _undeclared(module: str, name: str, exported: dict[str, frozenset[str]]) -> str:
    """The reason for a name a root does not declare.

    Args:
        module: The root.
        name: The name.
        exported: The roots' ``__all__``.

    Returns:
        One sentence, naming where the name is public when it is.
    """
    if name == "*":
        return f"a star import binds whatever {module} happens to hold; import the names its __all__ declares"
    homes = [home for home in _public_homes(name, exported) if home != module]
    where = f"; it is public from {' or '.join(homes)}" if homes else "; no public root exports it"
    return f"{name} is not in {module}.__all__{where}"


def _exports(root: str, exported: dict[str, frozenset[str]]) -> frozenset[str]:
    """A root's ``__all__``, imported once.

    Args:
        root: A public root, or the package itself.
        exported: The cache.

    Returns:
        The names it declares.
    """
    if root not in exported:
        exported[root] = frozenset(getattr(importlib.import_module(root), "__all__", ()))
    return exported[root]


def _public_homes(name: str, exported: dict[str, frozenset[str]]) -> list[str]:
    """Every public root whose ``__all__`` declares ``name``.

    A root whose extra is not installed cannot be read and is passed over: it is a suggestion, and the
    finding stands either way.

    Args:
        name: The name.
        exported: The cache.

    Returns:
        The roots, in :data:`~threetears.evals.PUBLIC_ROOTS` order.
    """
    homes = []
    for root in PUBLIC_ROOTS:
        try:
            names = _exports(root, exported)
        # NOSILENT: a root whose extra is not installed drops out of the suggestion only; the finding stands.
        except ImportError:
            continue
        if name in names:
            homes.append(root)
    return homes


def _root_above(module: str) -> str:
    """The nearest public root above ``module``, or the package when none is.

    Args:
        module: A module below a root.

    Returns:
        The root's name.
    """
    parts = module.split(".")
    for end in range(len(parts) - 1, 1, -1):
        candidate = ".".join(parts[:end])
        if candidate in PUBLIC_ROOTS:
            return candidate
    return _PACKAGE


__all__ = ["NonPublicImport", "nonpublic_evals_imports"]
