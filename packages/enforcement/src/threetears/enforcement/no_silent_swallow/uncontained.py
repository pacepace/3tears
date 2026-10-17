"""walker for broad and cancellation handlers that neither re-raise nor contain.

two rules, both decidable from the syntax alone, and neither excused by a comment or a log
call -- a log line does not tell the caller anything, and a comment is read by nobody at
runtime:

- **broad** (``no_silent_swallow.broad``): an ``except`` naming ``Exception`` or
  ``BaseException`` (alone or in a tuple) re-raises somewhere in its body, or calls
  ``contain(...)``. ``BaseException`` must re-raise; containing it is not allowed, because it
  is what a cancellation, ``KeyboardInterrupt`` and ``SystemExit`` are. a
  ``suppress(Exception)`` is flagged too: the fall-through form is ``with contained(...)``.
- **cancel** (``no_silent_swallow.cancel``): an ``except`` naming ``CancelledError``
  re-raises, and ``suppress(CancelledError)`` is flagged. absorbing a cancellation leaves the
  task running after it was told to stop, and breaks every ``asyncio.timeout`` and
  ``TaskGroup`` above it. stopping another task is ``await cancel_and_wait(task)``.

a bare ``except:`` is left to the silent-body walker, which bans it outright. ``try`` and
``try``/``except*`` are both scanned. a ``raise`` or ``contain`` inside a nested function or
class in the handler does not count: it does not run when the handler does.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    is_suppress_call,
    iter_python_files,
    parse_python_file,
)

__all__ = ["find_uncontained"]

BROAD_CATEGORY = "no_silent_swallow.broad"
CANCEL_CATEGORY = "no_silent_swallow.cancel"

_BROAD = frozenset({"Exception", "BaseException"})
_CANCELLED = "CancelledError"
_OTHER_FRAMES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _type_names(node: ast.expr | None) -> list[str]:
    """the trailing names of the exception types an ``except`` clause or ``suppress`` names.

    :param node: the clause's type expression
    :ptype node: ast.expr | None
    :return: one name per listed type; ``Name`` and ``Attribute`` give their last component
    :rtype: list[str]
    """
    if node is None:
        return []
    elements = node.elts if isinstance(node, ast.Tuple) else [node]
    names = []
    for element in elements:
        if isinstance(element, ast.Name):
            names.append(element.id)
        elif isinstance(element, ast.Attribute):
            names.append(element.attr)
    return names


def _same_frame(body: list[ast.stmt]) -> Iterator[ast.AST]:
    """every node in ``body`` that runs in the handler's own frame.

    :param body: a handler's statements
    :ptype body: list[ast.stmt]
    :return: the nodes, not descending into nested functions, lambdas or classes
    :rtype: Iterator[ast.AST]
    """
    pending: list[ast.AST] = [node for node in body if not isinstance(node, _OTHER_FRAMES)]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, _OTHER_FRAMES))


def _raises(body: list[ast.stmt]) -> bool:
    """true when the handler re-raises (or raises anything) on some path.

    :param body: a handler's statements
    :ptype body: list[ast.stmt]
    :return: whether a ``raise`` runs in the handler's frame
    :rtype: bool
    """
    return any(isinstance(node, ast.Raise) for node in _same_frame(body))


def _contains(body: list[ast.stmt], contain_names: frozenset[str]) -> bool:
    """true when the handler calls ``contain(...)``.

    :param body: a handler's statements
    :ptype body: list[ast.stmt]
    :param contain_names: the names the containment helper is called by
    :ptype contain_names: frozenset[str]
    :return: whether a call to one of ``contain_names`` runs in the handler's frame
    :rtype: bool
    """
    for node in _same_frame(body):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in contain_names:
            return True
        if isinstance(func, ast.Attribute) and func.attr in contain_names:
            return True
    return False


def _handler_violation(
    handler: ast.ExceptHandler,
    module_path: Path,
    contain_names: frozenset[str],
) -> Violation | None:
    """the violation ``handler`` commits, if any.

    :param handler: one ``except`` clause
    :ptype handler: ast.ExceptHandler
    :param module_path: the file it sits in
    :ptype module_path: Path
    :param contain_names: the names the containment helper is called by
    :ptype contain_names: frozenset[str]
    :return: the violation, or ``None``
    :rtype: Violation | None
    """
    names = _type_names(handler.type)
    if not names or _raises(handler.body):
        return None
    symbol = names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"
    if _CANCELLED in names:
        return Violation(
            category=CANCEL_CATEGORY,
            file=module_path,
            line=handler.lineno,
            symbol=symbol,
            reason=(
                "a caught CancelledError is re-raised; to stop another task, "
                "`await cancel_and_wait(task)` instead of awaiting it under this handler"
            ),
        )
    if "BaseException" in names:
        return Violation(
            category=BROAD_CATEGORY,
            file=module_path,
            line=handler.lineno,
            symbol=symbol,
            reason="`except BaseException` must re-raise: it catches cancellation, KeyboardInterrupt and SystemExit",
        )
    if "Exception" in names and not _contains(handler.body, contain_names):
        return Violation(
            category=BROAD_CATEGORY,
            file=module_path,
            line=handler.lineno,
            symbol=symbol,
            reason=(
                "a broad `except Exception` re-raises or calls `contain(exc, log=..., what=...)`; "
                "a log line or a NOSILENT comment does not excuse it. if the failure is an expected "
                "condition, name its type instead"
            ),
        )
    return None


def _suppress_violation(call: ast.Call, module_path: Path) -> Violation | None:
    """the violation a ``suppress(...)`` call commits, if any.

    :param call: the ``suppress(...)`` call
    :ptype call: ast.Call
    :param module_path: the file it sits in
    :ptype module_path: Path
    :return: the violation, or ``None``
    :rtype: Violation | None
    """
    names = [name for arg in call.args for name in _type_names(arg)]
    symbol = names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"
    if _CANCELLED in names:
        return Violation(
            category=CANCEL_CATEGORY,
            file=module_path,
            line=call.lineno,
            symbol=symbol,
            reason="suppress(CancelledError) absorbs the caller's own cancellation too; use `await cancel_and_wait(task)`",
        )
    if _BROAD & set(names):
        return Violation(
            category=BROAD_CATEGORY,
            file=module_path,
            line=call.lineno,
            symbol=symbol,
            reason="suppress(Exception) records nothing; use `with contained(what, log=log):`",
        )
    return None


def find_uncontained(
    src_roots: tuple[Path, ...],
    contain_names: frozenset[str],
) -> list[Violation]:
    """walk every module for broad or cancellation handlers that neither re-raise nor contain.

    :param src_roots: every src root to scan
    :ptype src_roots: tuple[Path, ...]
    :param contain_names: the names the containment helper is called by
    :ptype contain_names: frozenset[str]
    :return: violations, in file order and then source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            found: list[Violation] = []
            for node in ast.walk(tree):
                violation: Violation | None = None
                if isinstance(node, ast.ExceptHandler):
                    violation = _handler_violation(node, module_path, contain_names)
                elif is_suppress_call(node):
                    assert isinstance(node, ast.Call)
                    violation = _suppress_violation(node, module_path)
                if violation is not None:
                    found.append(violation)
            violations.extend(sorted(found, key=lambda v: v.line))
    return violations
