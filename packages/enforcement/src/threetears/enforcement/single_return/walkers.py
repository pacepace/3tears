"""walker for the single-return enforcement domain.

the single walker, :func:`find_multiple_business_returns`, reports
every function whose business logic returns more than once.

the accounting has two rules, and both exist because a naive version
of this walker got them wrong:

**leading guards do not count.** a plain ``if x: return y`` before
any other statement is a guard clause, and a function may have any
number of them. they are the shape that makes a single business-logic
return possible in the first place, so counting them would penalise
exactly the style the rule is trying to produce. the first non-guard
statement ends the guard prologue permanently -- a late ``if x:
return y`` in the middle of the body is business logic and counts.

**nested scopes are charged to themselves.** a ``def``, ``async
def``, or ``lambda`` inside a function body owns its own returns. the
module-level walk visits every function definition independently, so
descending into one would charge a nested helper's returns to its
parent and report the same returns twice. an ``ast.walk``-based
implementation does exactly that; :func:`_own_return_lines` stops at
the scope boundary instead.

a leading docstring is stripped before the guard prologue is read, so
a documented function's first real statement is still eligible to be
a guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    iter_python_files,
    parse_python_file,
    relative_posix_path,
)

__all__ = ["find_multiple_business_returns"]


_CATEGORY = "single_return.multiple"

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _is_guard_clause(node: ast.If) -> bool:
    """true for a plain ``if x: return`` -- no else/elif, body is one return.

    :param node: the ``if`` statement to classify
    :ptype node: ast.If
    :return: whether this is a guard clause
    :rtype: bool
    """
    if node.orelse:
        return False
    if len(node.body) != 1:
        return False
    return isinstance(node.body[0], ast.Return)


def _own_return_lines(node: ast.AST) -> list[int]:
    """line numbers of the returns belonging to ``node``'s own scope.

    stops at any nested ``def`` / ``async def`` / ``lambda``: those
    returns belong to that definition, which the module-level walk
    visits separately. descending past the boundary is the bug this
    function exists to avoid.

    :param node: the AST node to account for
    :ptype node: ast.AST
    :return: return-statement line numbers, in source order
    :rtype: list[int]
    """
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        return []
    lines: list[int] = [node.lineno] if isinstance(node, ast.Return) else []
    for child in ast.iter_child_nodes(node):
        lines.extend(_own_return_lines(child))
    return lines


def _returns_after_guards(body: list[ast.stmt]) -> list[int]:
    """line numbers of this function's own non-guard returns.

    :param body: the function body, docstring already stripped
    :ptype body: list[ast.stmt]
    :return: business-logic return lines, in source order
    :rtype: list[int]
    """
    in_guards = True
    business_returns: list[int] = []
    for stmt in body:
        if in_guards and isinstance(stmt, ast.If) and _is_guard_clause(stmt):
            continue
        in_guards = False
        business_returns.extend(_own_return_lines(stmt))
    return business_returns


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """drop a leading docstring so the guard prologue starts at real code.

    :param body: the raw function body
    :ptype body: list[ast.stmt]
    :return: the body without its leading docstring expression
    :rtype: list[ast.stmt]
    """
    if not body:
        return body
    first = body[0]
    is_docstring = (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    )
    return body[1:] if is_docstring else body


def _business_returns(node: _FunctionNode, excluded_function_names: frozenset[str]) -> list[int]:
    """the offending return lines for ``node``, or ``[]`` when it complies.

    :param node: the function definition to check
    :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
    :param excluded_function_names: function names skipped entirely
    :ptype excluded_function_names: frozenset[str]
    :return: return lines when there is more than one, else ``[]``
    :rtype: list[int]
    """
    if not node.body or node.name in excluded_function_names:
        return []
    body = _strip_docstring(node.body)
    returns = _returns_after_guards(body) if body else []
    return returns if len(returns) > 1 else []


def find_multiple_business_returns(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    excluded_function_names: frozenset[str],
    exempt_files: dict[str, str],
) -> list[Violation]:
    """walk every module for the single-business-return contract.

    one :class:`Violation` per offending function. ``line`` is the
    function's own ``def`` line rather than any one return, because
    the function is the unit that has to be restructured; the
    offending return lines are named in ``reason`` so a failure reads
    as a work list.

    file-level filters, applied in order:

    1. **empty-file skip** -- zero-size files hold no functions.
    2. **exempt-file skip** -- files whose
       :func:`~threetears.enforcement.common.ast_helpers.relative_posix_path`
       relative to ``repo_root`` is a key in ``exempt_files``.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching and violation reporting
    :ptype repo_root: Path
    :param excluded_function_names: function names skipped entirely
    :ptype excluded_function_names: frozenset[str]
    :param exempt_files: relative-posix-path -> rationale mapping
    :ptype exempt_files: dict[str, str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            try:
                if module_path.stat().st_size == 0:
                    continue
            except OSError:
                continue
            if relative_posix_path(module_path, repo_root) in exempt_files:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                lines = _business_returns(node, excluded_function_names)
                if lines:
                    violations.append(_violation(module_path, node, lines))
    return violations


def _violation(module_path: Path, node: _FunctionNode, lines: list[int]) -> Violation:
    """build the canonical :class:`Violation` for a multi-return function.

    :param module_path: absolute path to the offending module
    :ptype module_path: Path
    :param node: the offending function definition
    :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
    :param lines: the business-logic return lines
    :ptype lines: list[int]
    :return: the violation record
    :rtype: Violation
    """
    return Violation(
        category=_CATEGORY,
        file=module_path,
        line=node.lineno,
        symbol=node.name,
        reason=(
            f"{node.name}() has {len(lines)} business-logic returns at lines {lines}. "
            f"one return in business logic; leading `if x: return` guards are allowed. "
            f"collect the value into a result variable and return once at the end."
        ),
    )
