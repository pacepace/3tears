"""walker for the coercion-coverage enforcement domain.

the single walker, :func:`find_run_overrides`, scans every top-level
class in the configured src trees and flags any class with a Tool-ish
base whose body declares a ``run`` method (sync or async). the
``TearsTool`` base provides a ``run`` that calls
``normalize_kwargs(...)`` and then ``execute(...)``; overriding
``run`` on a subclass bypasses the coercion step.

implementation notes:

- only top-level :class:`ast.ClassDef` nodes are considered. nested
  classes (``class Outer: class Inner(Tool): ...``) are intentionally
  skipped to match the canonical scanner's behaviour.
- a base is Tool-ish when its last textual segment (``Foo`` for a bare
  ``ast.Name``, or ``pkg.Foo`` -> ``Foo`` for an ``ast.Attribute``)
  ends with one of the configured suffixes. ``Tool`` itself qualifies
  because ``"Tool".endswith("Tool")`` is true.
- a method must be **directly** defined on the class body — overrides
  inherited from a mixin are not flagged here. that's the whole point:
  we're catching deliberate overrides, not inherited definitions.
- ``__init__.py`` files are scanned same as any other module; the
  canonical excluded them but the new contract is that AST-based
  walking treats every ``.py`` file equally and lets the pyproject
  discovery layer decide what's in scope.
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
    "base_looks_toolish",
    "find_run_overrides",
]


_CATEGORY = "coercion_coverage.run_override"


def base_looks_toolish(
    node: ast.expr, base_class_suffixes: frozenset[str],
) -> bool:
    """return True when a base-class expression names a Tool-ish class.

    matches bare ``FooTool`` or attribute-style ``pkg.FooTool``. used
    to narrow the scan to class trees rooted in something Tool-ish so
    unrelated classes that legitimately define ``run`` do not trip the
    enforcement.

    matching uses last-segment textual comparison:

    - :class:`ast.Name` ``Foo`` -> name ``Foo``
    - :class:`ast.Attribute` ``pkg.Foo`` -> name ``Foo``
    - any other base shape (``Generic[T]``, ``make_base()``) -> not Tool-ish

    a base is Tool-ish iff its name ends with any of
    ``base_class_suffixes``. with the default suffix set ``{"Tool"}``
    this matches ``Tool``, ``DataSourceTool``, ``SchemaTool``, etc.

    :param node: AST base expression from a class's ``bases`` list
    :ptype node: ast.expr
    :param base_class_suffixes: suffix set marking a base as Tool-ish
    :ptype base_class_suffixes: frozenset[str]
    :return: whether the expression names something Tool-ish
    :rtype: bool
    """
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        return False
    return any(name.endswith(suffix) for suffix in base_class_suffixes)


def find_run_overrides(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    base_class_suffixes: frozenset[str],
) -> list[Violation]:
    """walk every top-level Tool-ish class and flag ``run`` method overrides.

    one :class:`Violation` is produced per offending method (a class
    that declares both a sync and async ``run`` would produce two
    violations, but that combination is illegal at runtime and
    unlikely in practice). the violation's ``line`` field points at
    the offending method definition, not the enclosing class.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        domains that use it for path rendering)
    :ptype repo_root: Path
    :param base_class_suffixes: suffix set marking a base as Tool-ish
    :ptype base_class_suffixes: frozenset[str]
    :return: violations in source order, one per offending ``run``
        method on a Tool-ish class
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if not any(
                    base_looks_toolish(b, base_class_suffixes)
                    for b in node.bases
                ):
                    continue
                for member in node.body:
                    if not isinstance(
                        member, (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        continue
                    if member.name != "run":
                        continue
                    reason = (
                        f"class '{node.name}' overrides run(); override "
                        f"execute() instead so TearsTool.run input "
                        f"coercion still fires"
                    )
                    violations.append(
                        Violation(
                            category=_CATEGORY,
                            file=module_path,
                            line=member.lineno,
                            symbol=node.name,
                            reason=reason,
                        )
                    )
    return violations
