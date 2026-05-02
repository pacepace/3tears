"""walkers for the nats-wrapper-usage enforcement domain.

two walkers in this module — :func:`find_direct_nats_imports` for
production source and :func:`find_test_nats_imports` for the tests
tree. they share the same import-detection logic
(:func:`_scan_module`) but emit different violation categories so
reports and exemption matching can keep the two surfaces distinct.

matched shapes:

- ``import nats`` -> symbol ``"*"``.
- ``import nats.aio`` (or any dotted submodule) -> symbol equals the
  full dotted name.
- ``from nats import connect`` -> one violation per imported name,
  symbol equals the imported name.
- ``from nats.aio import Client`` -> symbol ``"Client"`` (one
  violation per imported name).
- ``from nats import *`` -> symbol ``"*"``.

intentionally NOT flagged:

- ``import natural`` / ``import natalie`` -> the ``forbidden_module``
  test compares the import's package root, not a substring overlap.
- ``from threetears.nats import NatsClient`` -> the wrapper itself.
- ``from . import nats`` -> relative imports cannot reach the
  top-level package.
- aliases (``import nats as natz``) — alias does not change what was
  imported; the violation is on the package, not the local name.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    iter_python_files,
    parse_python_file,
)

__all__ = [
    "find_direct_nats_imports",
    "find_test_nats_imports",
    "is_forbidden_module",
]


_PRODUCTION_CATEGORY = "nats_wrapper_usage.production_import"
_TEST_CATEGORY = "nats_wrapper_usage.test_import"


def is_forbidden_module(name: str | None, forbidden_module: str) -> bool:
    """true iff ``name`` resolves to ``forbidden_module`` or a submodule.

    matches the literal ``forbidden_module`` itself and any dotted
    submodule (``nats.aio``, ``nats.errors``). does not match third-
    party packages whose name only contains the substring (``natural``,
    ``natalie``); the comparison is on the package root component, not
    on substring overlap.

    :param name: module name as written in the import statement
    :ptype name: str | None
    :param forbidden_module: top-level package being guarded
    :ptype forbidden_module: str
    :return: whether ``name`` resolves to the forbidden module
    :rtype: bool
    """
    if name is None:
        return False
    if name == forbidden_module:
        return True
    return name.startswith(forbidden_module + ".")


def find_direct_nats_imports(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    forbidden_module: str = "nats",
) -> list[Violation]:
    """walk production src-trees for direct ``nats`` imports.

    one :class:`Violation` per offending import, with category
    ``nats_wrapper_usage.production_import``. file-level exemptions
    (the ``_nats_exemptions.txt`` whole-file allowlist) are applied
    by the runner via
    :func:`~threetears.enforcement.common.exemptions.apply_exemptions`,
    not inside the walker — this matches the canonical's separation
    of concerns and keeps the walker focused on detection.

    symbol resolution:

    - ``import nats`` -> ``"*"``
    - ``import nats.aio`` -> ``"nats.aio"``
    - ``import nats as natz`` -> ``"*"`` (alias does not change what
      was imported)
    - ``from nats import connect`` -> ``"connect"`` (one violation per
      imported name)
    - ``from nats.aio import Client`` -> ``"Client"``
    - ``from nats import *`` -> ``"*"``

    relative imports (``from . import x``) and non-nats packages are
    ignored.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        walkers; relative-path rendering happens at report time)
    :ptype repo_root: Path
    :param forbidden_module: top-level package being guarded; defaults
        to ``"nats"``
    :ptype forbidden_module: str
    :return: violations in source order
    :rtype: list[Violation]
    """
    return _scan_roots(
        src_roots,
        repo_root,
        forbidden_module,
        _PRODUCTION_CATEGORY,
        _PRODUCTION_HINT,
    )


def find_test_nats_imports(
    tests_root: Path | None,
    repo_root: Path,
    forbidden_module: str = "nats",
) -> list[Violation]:
    """walk the tests tree for direct ``nats`` imports.

    same matched shapes as :func:`find_direct_nats_imports`, but
    emits category ``nats_wrapper_usage.test_import`` so exemption
    matching and reports can keep production and test surfaces
    distinct. returns ``[]`` when ``tests_root`` is ``None`` or when
    the directory does not exist on disk — the canonical preserves
    this skip-on-absent behaviour.

    :param tests_root: tests tree to scan, or ``None`` to skip
    :ptype tests_root: Path | None
    :param repo_root: repo root (retained for parity with sibling
        walkers)
    :ptype repo_root: Path
    :param forbidden_module: top-level package being guarded; defaults
        to ``"nats"``
    :ptype forbidden_module: str
    :return: violations in source order, or ``[]`` when ``tests_root``
        is missing or absent
    :rtype: list[Violation]
    """
    if tests_root is None or not tests_root.exists():
        return []
    return _scan_roots(
        (tests_root,),
        repo_root,
        forbidden_module,
        _TEST_CATEGORY,
        _TEST_HINT,
    )


_PRODUCTION_HINT = (
    "use `from threetears.nats import NatsClient, Subjects` "
    "instead, or add a whole-file exemption with a specific "
    "rationale to `_nats_exemptions.txt`"
)

_TEST_HINT = (
    "tests must also route through `threetears.nats.NatsClient` "
    "unless the file is explicitly exempted in `_nats_exemptions.txt` "
    "with a specific rationale"
)


def _scan_roots(
    roots: tuple[Path, ...],
    repo_root: Path,
    forbidden_module: str,
    category: str,
    hint: str,
) -> list[Violation]:
    """walk ``roots`` and aggregate per-module violations.

    :param roots: directory roots to walk
    :ptype roots: tuple[Path, ...]
    :param repo_root: repo root (parity with sibling walkers)
    :ptype repo_root: Path
    :param forbidden_module: top-level package being guarded
    :ptype forbidden_module: str
    :param category: violation category emitted for every match
    :ptype category: str
    :param hint: trailing remediation hint embedded in each violation
        ``reason``
    :ptype hint: str
    :return: violations in source order (root-major, then file-major)
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in roots:
        for module_path in iter_python_files(root):
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            _scan_module(
                tree,
                module_path,
                forbidden_module,
                category,
                hint,
                violations,
            )
    return violations


def _scan_module(
    tree: ast.Module,
    module_path: Path,
    forbidden_module: str,
    category: str,
    hint: str,
    violations: list[Violation],
) -> None:
    """walk a single parsed module, appending violations in place.

    :param tree: parsed ast module
    :ptype tree: ast.Module
    :param module_path: file containing the imports
    :ptype module_path: Path
    :param forbidden_module: top-level package being guarded
    :ptype forbidden_module: str
    :param category: violation category to emit
    :ptype category: str
    :param hint: remediation hint embedded in each violation reason
    :ptype hint: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _scan_import(
                node, module_path, forbidden_module, category, hint, violations,
            )
        elif isinstance(node, ast.ImportFrom):
            _scan_import_from(
                node, module_path, forbidden_module, category, hint, violations,
            )


def _scan_import(
    node: ast.Import,
    module_path: Path,
    forbidden_module: str,
    category: str,
    hint: str,
    violations: list[Violation],
) -> None:
    """flag every forbidden alias in an ``import a, b, ...`` statement.

    each alias is checked independently because a single ``import``
    line may name multiple modules (``import os, nats, sys``). the
    forbidden-module check uses the import target's name, not the
    local ``as``-alias.

    :param node: ast.Import node
    :ptype node: ast.Import
    :param module_path: file containing the import
    :ptype module_path: Path
    :param forbidden_module: top-level package being guarded
    :ptype forbidden_module: str
    :param category: violation category to emit
    :ptype category: str
    :param hint: remediation hint embedded in each violation reason
    :ptype hint: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    for alias in node.names:
        if not is_forbidden_module(alias.name, forbidden_module):
            continue
        if alias.name == forbidden_module:
            symbol = "*"
            description = f"import {forbidden_module}"
        else:
            symbol = alias.name
            description = f"import {alias.name}"
        violations.append(
            Violation(
                category=category,
                file=module_path,
                line=node.lineno,
                symbol=symbol,
                reason=f"{description}; {hint}",
            )
        )


def _scan_import_from(
    node: ast.ImportFrom,
    module_path: Path,
    forbidden_module: str,
    category: str,
    hint: str,
    violations: list[Violation],
) -> None:
    """flag a ``from <forbidden>[.x] import ...`` statement.

    one violation per imported name (``from nats import a, b`` yields
    two violations). ``from nats import *`` yields one violation with
    symbol ``"*"``. relative imports (``from . import x``) and
    non-forbidden modules are ignored.

    :param node: ast.ImportFrom node
    :ptype node: ast.ImportFrom
    :param module_path: file containing the import
    :ptype module_path: Path
    :param forbidden_module: top-level package being guarded
    :ptype forbidden_module: str
    :param category: violation category to emit
    :ptype category: str
    :param hint: remediation hint embedded in each violation reason
    :ptype hint: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    if node.level and node.level > 0:
        return
    if not is_forbidden_module(node.module, forbidden_module):
        return
    module_name = node.module or forbidden_module
    for alias in node.names:
        symbol = alias.name
        description = f"from {module_name} import {alias.name}"
        violations.append(
            Violation(
                category=category,
                file=module_path,
                line=node.lineno,
                symbol=symbol,
                reason=f"{description}; {hint}",
            )
        )
