"""walker for the no-silent-swallow enforcement domain.

the single walker, :func:`find_silent_swallows`, scans every module
under the configured src trees for two violation shapes:

- ``ast.ExceptHandler`` whose body is "silent" (``pass``, ``...``,
  ``return None``, ``continue``, ``break``) and which neither logs nor
  re-raises and lacks a ``# NOSILENT: <reason>`` marker in the source
  window between the line above the ``except`` and the first body
  statement (inclusive). bare ``except:`` clauses are flagged
  unconditionally because they swallow ``SystemExit`` and
  ``KeyboardInterrupt``.
- ``contextlib.suppress(...)`` (or bare ``suppress(...)``) call
  expressions without a ``# NOSILENT: <reason>`` marker within 2
  lines above.

implementation notes:

- logger-call recognition is local to this walker (rather than reused
  from
  :func:`threetears.enforcement.common.ast_helpers.is_logger_call`)
  because the canonical contract is more restrictive: an indirect
  receiver counts as a logger only when the attribute name is itself
  ``log`` or ``logger`` (so ``self.log.error(...)`` is logged but
  ``self.foo.error(...)`` is not). reusing the broader common helper
  here would silently weaken the contract.
- the marker contract is "the substring (default ``# NOSILENT:``)
  followed by a non-empty rationale". ``# NOSILENT:`` with nothing
  after the colon does not count; the rationale (after stripping
  whitespace) must be at least one character. this preserves the
  canonical's intent that the comment carries the operator's reason
  rather than acting as a silencer.
- ``ast.walk`` is used to find every ``ast.Try`` and every
  ``contextlib.suppress(...)`` call regardless of nesting — silent
  swallowing in nested functions, conditional branches, etc. is still
  flagged.
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
    is_suppress_call,
    iter_python_files,
    parse_python_file,
)

__all__ = [
    "body_contains_log",
    "body_reraises",
    "body_silent_category",
    "find_silent_swallows",
    "has_nosilent_marker",
    "suppress_has_nosilent",
]


_EXCEPT_CATEGORY = "no_silent_swallow.except"
_SUPPRESS_CATEGORY = "no_silent_swallow.suppress"


def _is_log_call(
    node: ast.AST,
    logger_names: frozenset[str],
    logger_methods: frozenset[str],
) -> bool:
    """true iff ``node`` is a recognised ``logger.<method>(...)`` call.

    matches the canonical contract (more restrictive than the common
    :func:`~threetears.enforcement.common.ast_helpers.is_logger_call`):

    - bare-name receiver: ``Name(id in logger_names).method(...)``
      where ``method`` is in ``logger_methods``.
    - attribute receiver: ``Attribute(attr in {"log", "logger"}).method(...)``
      where ``method`` is in ``logger_methods``. this handles
      indirect references like ``self.log.error(...)`` /
      ``ctx.logger.warning(...)`` while still rejecting unrelated
      attributes such as ``self.cache.error(...)``.

    accepts the node either as an :class:`ast.Expr` (statement form)
    or directly as an :class:`ast.Call`.

    :param node: candidate ast node
    :ptype node: ast.AST
    :param logger_names: known logger receiver names
    :ptype logger_names: frozenset[str]
    :param logger_methods: known logger method names
    :ptype logger_methods: frozenset[str]
    :return: whether ``node`` invokes a logger method
    :rtype: bool
    """
    if isinstance(node, ast.Expr):
        node = node.value
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in logger_methods:
        return False
    receiver = func.value
    if isinstance(receiver, ast.Name) and receiver.id in logger_names:
        return True
    if isinstance(receiver, ast.Attribute) and receiver.attr in {"log", "logger"}:
        return True
    return False


def body_contains_log(
    body: list[ast.stmt],
    logger_names: frozenset[str],
    logger_methods: frozenset[str],
) -> bool:
    """true iff any statement (incl. nested) in ``body`` calls a logger.

    walks every node in every body statement so a logger call inside
    a nested ``if`` / ``try`` / function still counts as "logged".

    :param body: statement list from an ``ExceptHandler.body``
    :ptype body: list[ast.stmt]
    :param logger_names: known logger receiver names
    :ptype logger_names: frozenset[str]
    :param logger_methods: known logger method names
    :ptype logger_methods: frozenset[str]
    :return: whether a logger call appears anywhere in the body tree
    :rtype: bool
    """
    for stmt in body:
        if _is_log_call(stmt, logger_names, logger_methods):
            return True
        for child in ast.walk(stmt):
            if _is_log_call(child, logger_names, logger_methods):
                return True
    return False


def body_silent_category(body: list[ast.stmt]) -> str | None:
    """classify handler body as a silent swallow variant, else ``None``.

    a body is silent iff it is exactly one statement of one of the
    recognised shapes:

    - :class:`ast.Pass` -> ``"pass"``
    - :class:`ast.Expr` wrapping ``Ellipsis`` -> ``"ellipsis"``
    - :class:`ast.Return` with no value or a bare ``None`` constant
      -> ``"return-none"``
    - :class:`ast.Continue` -> ``"continue"``
    - :class:`ast.Break` -> ``"break"``

    multi-statement bodies and other single-statement shapes (e.g.
    ``except: x = 1``) are not classified as silent: they are
    legitimate recovery code that the walker should not flag.

    :param body: statement list from an ``ExceptHandler.body``
    :ptype body: list[ast.stmt]
    :return: silent-shape label, or ``None`` for non-silent bodies
    :rtype: str | None
    """
    if len(body) != 1:
        return None
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return "pass"
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    ):
        return "ellipsis"
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return "return-none"
        if (
            isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None
        ):
            return "return-none"
        return None
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Break):
        return "break"
    return None


def body_reraises(body: list[ast.stmt]) -> bool:
    """true when handler body contains a top-level ``raise`` statement.

    only top-level statements are checked; a ``raise`` buried inside a
    nested ``if`` / ``with`` / function does not count, because such a
    body cannot be classified as silent in the first place (it is
    multi-statement or non-trivial).

    :param body: statement list from an ``ExceptHandler.body``
    :ptype body: list[ast.stmt]
    :return: whether body contains a raise at top level
    :rtype: bool
    """
    return any(isinstance(stmt, ast.Raise) for stmt in body)


def _marker_with_reason(
    line_text: str,
    nosilent_marker: str,
) -> bool:
    """true iff ``line_text`` contains ``nosilent_marker`` with a rationale.

    the marker substring must appear, and the text immediately
    following the marker must contain at least one non-whitespace
    character. ``# NOSILENT: legitimate cleanup`` passes;
    ``# NOSILENT:`` (nothing after the colon, or whitespace only)
    fails.

    :param line_text: a single line of source (without trailing newline)
    :ptype line_text: str
    :param nosilent_marker: the marker substring (default ``# NOSILENT:``)
    :ptype nosilent_marker: str
    :return: whether the line carries a marker with non-empty reason
    :rtype: bool
    """
    idx = line_text.find(nosilent_marker)
    if idx < 0:
        return False
    rationale = line_text[idx + len(nosilent_marker):].strip()
    return bool(rationale)


def has_nosilent_marker(
    source_lines: list[str],
    except_line: int,
    body_line: int | None,
    nosilent_marker: str,
) -> bool:
    """check for a ``# NOSILENT:`` comment near the except or body.

    accepts the marker on the line immediately above the ``except``
    statement, on the except line itself, or anywhere between the
    except line and the first body statement (inclusive of both
    endpoints). the inclusive window covers multi-line ``#`` comment
    blocks placed between the ``except`` and its body.

    a marker without a non-empty rationale (``# NOSILENT:`` with
    nothing after the colon) does not count.

    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param except_line: 1-indexed line of the ``except`` statement
    :ptype except_line: int
    :param body_line: 1-indexed line of the first body statement, or
        ``None`` when the body has no anchor
    :ptype body_line: int | None
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :return: whether a marker with non-empty rationale is present in
        the accepted window
    :rtype: bool
    """
    start_1 = max(1, except_line - 1)
    end_1 = body_line if body_line is not None else except_line
    for line_1 in range(start_1, end_1 + 1):
        idx = line_1 - 1
        if 0 <= idx < len(source_lines):
            if _marker_with_reason(source_lines[idx], nosilent_marker):
                return True
    return False


def suppress_has_nosilent(
    source_lines: list[str],
    call_line: int,
    nosilent_marker: str,
) -> bool:
    """check for a marker within 3 lines above (or on) a suppress call.

    the canonical scans ``call_line - 1``, ``call_line - 2``, and
    ``call_line - 3`` (1-indexed); that window catches both
    ``# NOSILENT: ...`` placed on the line above ``with suppress(...)``
    and the marker placed on the same line as the suppress call (when
    the call's lineno is reported one line below the comment).

    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param call_line: 1-indexed line of the ``suppress(...)`` call
    :ptype call_line: int
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :return: whether a marker with non-empty rationale is present
        within 3 lines above the call
    :rtype: bool
    """
    for idx in (call_line - 1, call_line - 2, call_line - 3):
        if 0 <= idx < len(source_lines):
            if _marker_with_reason(source_lines[idx], nosilent_marker):
                return True
    return False


def _handler_symbol(handler: ast.ExceptHandler) -> str:
    """render a handler's caught-type as a human-readable symbol.

    bare ``except:`` -> ``"BaseException"`` (what python actually
    catches). single-name type -> the name. attribute type
    (``foo.Error``) -> the trailing attribute. tuple of types ->
    ``"(A, B)"``. anything else -> ``"<expr>"``.

    :param handler: except handler
    :ptype handler: ast.ExceptHandler
    :return: short symbol for the violation record
    :rtype: str
    """
    if handler.type is None:
        return "BaseException"
    return _expr_symbol(handler.type)


def _expr_symbol(node: ast.expr) -> str:
    """render an exception-type expression as a short symbol string.

    :param node: ast expression in the ``except`` clause
    :ptype node: ast.expr
    :return: short readable symbol
    :rtype: str
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Tuple):
        parts = [_expr_symbol(e) for e in node.elts]
        return "(" + ", ".join(parts) + ")"
    return "<expr>"


def _suppress_symbol(call: ast.Call) -> str:
    """render a ``suppress(...)`` call's argument list as a symbol.

    :param call: ``suppress(...)`` call expression
    :ptype call: ast.Call
    :return: short readable symbol
    :rtype: str
    """
    if not call.args:
        return "<expr>"
    parts = [_expr_symbol(a) for a in call.args]
    return "(" + ", ".join(parts) + ")" if len(parts) > 1 else parts[0]


def find_silent_swallows(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    logger_names: frozenset[str],
    logger_methods: frozenset[str],
    nosilent_marker: str,
) -> list[Violation]:
    """walk every module for silent ``except`` and unmarked ``suppress(...)``.

    one :class:`Violation` is produced per offending site:

    - ``no_silent_swallow.except`` for a silent except handler or a
      bare ``except:`` clause. ``symbol`` is the caught type name (or
      ``"BaseException"`` for the bare case).
    - ``no_silent_swallow.suppress`` for an unmarked
      ``contextlib.suppress(...)``. ``symbol`` is the first argument's
      name, or ``"(A, B)"`` for multi-argument forms, or ``"<expr>"``
      for non-name arguments.

    violations are emitted in source order (the order
    :func:`ast.walk` visits nodes), keyed by the offending node's
    line.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        domains that use it for path rendering)
    :ptype repo_root: Path
    :param logger_names: known logger receiver names
    :ptype logger_names: frozenset[str]
    :param logger_methods: known logger method names
    :ptype logger_methods: frozenset[str]
    :param nosilent_marker: the marker substring (default ``# NOSILENT:``)
    :ptype nosilent_marker: str
    :return: violations in source order
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            try:
                source_text = module_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            source_lines = source_text.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    _scan_try(
                        node,
                        module_path,
                        source_lines,
                        logger_names,
                        logger_methods,
                        nosilent_marker,
                        violations,
                    )
                if is_suppress_call(node):
                    assert isinstance(node, ast.Call)
                    if not suppress_has_nosilent(
                        source_lines, node.lineno, nosilent_marker,
                    ):
                        violations.append(
                            Violation(
                                category=_SUPPRESS_CATEGORY,
                                file=module_path,
                                line=node.lineno,
                                symbol=_suppress_symbol(node),
                                reason=(
                                    "contextlib.suppress without "
                                    f"{nosilent_marker} marker; add "
                                    f"`{nosilent_marker} <reason>` within 3 "
                                    "lines above the call"
                                ),
                            )
                        )
    return violations


def _scan_try(
    node: ast.Try,
    module_path: Path,
    source_lines: list[str],
    logger_names: frozenset[str],
    logger_methods: frozenset[str],
    nosilent_marker: str,
    violations: list[Violation],
) -> None:
    """scan a single ``ast.Try`` node for handler-level violations.

    extracted from :func:`find_silent_swallows` to keep the per-node
    nesting shallow. mutates ``violations`` in place to preserve
    source order.

    :param node: the try node to inspect
    :ptype node: ast.Try
    :param module_path: file containing the try
    :ptype module_path: Path
    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param logger_names: known logger receiver names
    :ptype logger_names: frozenset[str]
    :param logger_methods: known logger method names
    :ptype logger_methods: frozenset[str]
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    for handler in node.handlers:
        if handler.type is None:
            violations.append(
                Violation(
                    category=_EXCEPT_CATEGORY,
                    file=module_path,
                    line=handler.lineno,
                    symbol="BaseException",
                    reason=(
                        "bare `except:` is banned; narrow to "
                        "`except Exception:` or narrower"
                    ),
                )
            )
            continue
        category = body_silent_category(handler.body)
        if category is None:
            continue
        if body_contains_log(handler.body, logger_names, logger_methods):
            continue
        if body_reraises(handler.body):
            continue
        body_line = handler.body[0].lineno if handler.body else None
        if has_nosilent_marker(
            source_lines, handler.lineno, body_line, nosilent_marker,
        ):
            continue
        violations.append(
            Violation(
                category=_EXCEPT_CATEGORY,
                file=module_path,
                line=handler.lineno,
                symbol=_handler_symbol(handler),
                reason=(
                    f"silent swallow (body={category}); add a log call, "
                    f"re-raise, or a `{nosilent_marker} <reason>` comment "
                    "justifying silence"
                ),
            )
        )
