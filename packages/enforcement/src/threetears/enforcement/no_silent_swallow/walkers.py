"""walker for the no-silent-swallow enforcement domain.

the single walker, :func:`find_silent_swallows`, scans every module
under the configured src trees for two violation shapes:

- ``ast.ExceptHandler`` whose body is "silent" and which neither logs
  nor re-raises and lacks a valid ``# NOSILENT(<kind>): <reason>``
  marker in the source window between the line above the ``except``
  and the first body statement (inclusive). a body is silent when
  every statement in it is inert: ``pass``, ``...``, ``continue``,
  ``break``, a ``return`` of ``None`` / a constant / an empty
  collection, or an assignment of one of those (the "default" shape,
  ``except ValueError: ok = False``). such a body turns the error into
  an answer the caller cannot tell from a real one. bare ``except:``
  clauses are flagged unconditionally because they swallow
  ``SystemExit`` and ``KeyboardInterrupt``.
- ``ast.ExceptHandler`` that LOGS and then still leaves with an inert
  answer -- a ``return`` of ``None`` / a constant / an empty
  collection, a ``continue`` or ``break``, or a last statement that
  assigns a default -- without a valid kinded marker. a log line
  tells the operator; it does not tell the caller, who still gets a
  value it cannot tell from success. a handler that logs and then
  raises, returns a typed failure, or does other work stays accepted.
- ``contextlib.suppress(...)`` (or bare ``suppress(...)``) call
  expressions without a valid ``# NOSILENT(<kind>): <reason>`` marker
  within 2 lines above.

**the marker names a kind, and the walker checks the kind against the
code.** a bare ``# NOSILENT: <reason>`` no longer excuses a silent
handler: a reason is prose, and prose was accepted for handlers that
returned ``False`` for an unreadable case and dropped it from an arm.
the kinds, and what the walker verifies for each:

- ``expected`` -- the exception IS the answer the code asked for (a
  timeout that is the interval elapsing, a "not found" that is the
  documented miss, the refusal a conformance check exists to see).
  every caught type must be named and narrower than ``Exception``:
  "expected" cannot be said of everything.
- ``attempt`` -- one attempt of several (a retry, a candidate parse,
  a polling loop); exhaustion is raised or reported elsewhere. the
  handler must sit inside a loop in its function, or a ``raise`` must
  follow the ``try`` in that function.
- ``cleanup`` -- best-effort cleanup while another error is already
  propagating. the handler or ``suppress`` must sit inside an
  ``except`` body that re-raises, so it cannot mask the primary error.
- ``optional-import`` -- an optional-dependency probe. every caught
  type must be ``ImportError`` / ``ModuleNotFoundError`` and the
  guarded body must import something.
- ``logged`` -- the handler logs the error and the answer it then
  gives is still TRUE for the caller (a health probe's ``False``, a
  background loop with no caller, a side effect whose failure does not
  change what was asked). the handler must log at WARNING or above;
  a DEBUG / INFO line is not an operator-visible record.

an unknown kind, or a kind whose check fails, is itself a violation,
wherever the marker sits -- on a silent handler or on one that logs.
and a bare ``# NOSILENT: <reason>`` comment is a violation anywhere:
every marker states its kind, so none is left over from the days when
prose was enough.
exemptions live on the code, as the marker, never in a central file:
a marker moves with the handler it describes.

implementation notes:

- logger-call recognition is local to this walker (rather than reused
  from
  :func:`threetears.enforcement.common.ast_helpers.is_logger_call`)
  because the canonical contract is more restrictive: an indirect
  receiver counts as a logger only when the attribute name is itself
  ``log`` or ``logger`` (so ``self.log.error(...)`` is logged but
  ``self.foo.error(...)`` is not). reusing the broader common helper
  here would silently weaken the contract.
- the marker contract is "the marker followed by a non-empty
  rationale". ``# NOSILENT(expected):`` with nothing after the colon
  does not count.
- ``ast.walk`` is used to find every ``ast.Try`` / ``ast.TryStar`` and
  every ``contextlib.suppress(...)`` call regardless of nesting --
  silent swallowing in nested functions, conditional branches, etc. is
  still flagged.
- ``__init__.py`` files are scanned same as any other module; the
  canonical excluded them but the new contract is that AST-based
  walking treats every ``.py`` file equally and lets the pyproject
  discovery layer decide what's in scope.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    is_suppress_call,
    iter_python_files,
    note_unscanned,
    parse_python_file,
)

__all__ = [
    "MARKER_KINDS",
    "NosilentMarker",
    "body_contains_log",
    "body_reraises",
    "body_silent_category",
    "body_silent_exit",
    "find_nosilent_marker",
    "find_silent_swallows",
    "has_nosilent_marker",
    "marker_kind_problem",
    "suppress_has_nosilent",
]


_EXCEPT_CATEGORY = "no_silent_swallow.except"
_SUPPRESS_CATEGORY = "no_silent_swallow.suppress"
_MARKER_CATEGORY = "no_silent_swallow.marker"

MARKER_KINDS: frozenset[str] = frozenset({"expected", "attempt", "cleanup", "optional-import", "logged"})
"""the kinds a ``# NOSILENT(<kind>): <reason>`` marker may name; see the module docstring."""

_BROAD_TYPES: frozenset[str] = frozenset({"Exception", "BaseException"})
_IMPORT_ERRORS: frozenset[str] = frozenset({"ImportError", "ModuleNotFoundError"})
_EMPTY_CONSTRUCTORS: frozenset[str] = frozenset({"list", "dict", "set", "tuple", "frozenset"})
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_OPERATOR_LEVELS: frozenset[str] = frozenset({"warning", "error", "critical", "exception"})
_DEFAULT_LOGGER_NAMES: frozenset[str] = frozenset({"log", "logger", "_log", "_logger", "LOGGER"})


@dataclass(frozen=True)
class NosilentMarker:
    """one ``# NOSILENT`` marker found in a handler's or suppress's window.

    :ivar kind: the kind named in parentheses, or ``None`` for the bare
        ``# NOSILENT: <reason>`` form (which no longer excuses silence)
    :ivar reason: the rationale after the colon, stripped
    :ivar line: 1-indexed line carrying the marker
    """

    kind: str | None
    reason: str
    line: int


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


def _is_silent_value(node: ast.expr | None) -> bool:
    """true iff ``node`` is a value that carries no trace of an error.

    ``None`` (or no value), any constant (``False``, ``0``, ``""``,
    ...), an empty list / tuple / dict / set literal, or a call of an
    empty builtin constructor (``list()``, ``dict()``, ...).

    :param node: the returned or assigned expression, ``None`` for a bare ``return``
    :ptype node: ast.expr | None
    :return: whether the value is indistinguishable from a real answer
    :rtype: bool
    """
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id in _EMPTY_CONSTRUCTORS
            and not node.args
            and not node.keywords
        )
    return False


def _statement_category(stmt: ast.stmt) -> str | None:
    """label one inert statement, or ``None`` when it does real work.

    :param stmt: one statement of a handler body
    :ptype stmt: ast.stmt
    :return: the inert shape's label, or ``None``
    :rtype: str | None
    """
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return "ellipsis"
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Break):
        return "break"
    if isinstance(stmt, ast.Return):
        if stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
            return "return-none"
        if isinstance(stmt.value, ast.Constant):
            return "return-constant"
        return "return-empty" if _is_silent_value(stmt.value) else None
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return "assign-default" if stmt.value is not None and _is_silent_value(stmt.value) else None
    return None


def body_silent_category(body: list[ast.stmt]) -> str | None:
    """classify handler body as a silent swallow variant, else ``None``.

    a body is silent iff every statement in it is inert -- nothing in
    it raises, logs, calls, or carries the caught exception anywhere:

    - :class:`ast.Pass` -> ``"pass"``
    - :class:`ast.Expr` wrapping ``Ellipsis`` -> ``"ellipsis"``
    - :class:`ast.Return` with no value or a bare ``None`` constant
      -> ``"return-none"``
    - :class:`ast.Return` of any other constant (``False``, ``0``,
      ``""``) -> ``"return-constant"``
    - :class:`ast.Return` of an empty collection (``[]``, ``{}``,
      ``dict()``) -> ``"return-empty"``
    - an assignment of any of those values (``ok = False``,
      ``result = None``) -> ``"assign-default"``
    - :class:`ast.Continue` -> ``"continue"``
    - :class:`ast.Break` -> ``"break"``

    a multi-statement inert body (``ok = False`` then ``continue``) is
    labelled by its last statement. a body with any other statement --
    a call, a raise, an ``if`` -- is not classified as silent.

    :param body: statement list from an ``ExceptHandler.body``
    :ptype body: list[ast.stmt]
    :return: silent-shape label, or ``None`` for non-silent bodies
    :rtype: str | None
    """
    if not body:
        return None
    labels = [_statement_category(stmt) for stmt in body]
    if any(label is None for label in labels):
        return None
    return labels[-1]


def body_silent_exit(body: list[ast.stmt]) -> str | None:
    """the first inert way out of a handler body that does real work, else ``None``.

    :func:`body_silent_category` answers for a body that is inert
    throughout. this answers for one that is not -- typically one that
    logs -- and asks how control LEAVES it: a ``return`` of ``None`` /
    a constant / an empty collection, a ``continue`` / ``break`` aimed
    at an enclosing loop, or a last statement that assigns a default.
    any of those hands the caller an answer it cannot tell from
    success, whatever was logged on the way. ``raise`` statements do not
    excuse a silent exit on another branch; nested functions and
    classes are not part of the handler's flow. a ``continue`` /
    ``break`` inside a loop the handler itself runs belongs to that loop.

    :param body: statement list from an ``ExceptHandler.body``
    :ptype body: list[ast.stmt]
    :return: the silent exit's label (``return-none``, ``return-constant``,
        ``return-empty``, ``continue``, ``break``, ``assign-default``), or ``None``
    :rtype: str | None
    """
    stack: list[tuple[ast.AST, bool]] = [(stmt, False) for stmt in reversed(body)]
    while stack:
        node, in_inner_loop = stack.pop()
        if isinstance(node, _SCOPES):
            continue
        if isinstance(node, ast.Return):
            label = _statement_category(node)
            if label is not None:
                return label
        elif isinstance(node, (ast.Continue, ast.Break)) and not in_inner_loop:
            return "continue" if isinstance(node, ast.Continue) else "break"
        inner = in_inner_loop or isinstance(node, _LOOPS)
        stack.extend((child, inner) for child in reversed(list(ast.iter_child_nodes(node))))
    if body and _statement_category(body[-1]) == "assign-default":
        return "assign-default"
    return None


def _logs_at_operator_level(body: list[ast.stmt], logger_names: frozenset[str]) -> bool:
    """true iff ``body`` holds a logger call at WARNING or above.

    :param body: statement list from an ``ExceptHandler.body``
    :ptype body: list[ast.stmt]
    :param logger_names: known logger receiver names
    :ptype logger_names: frozenset[str]
    :return: whether an operator-visible log line is written
    :rtype: bool
    """
    return body_contains_log(body, logger_names, _OPERATOR_LEVELS)


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


def _parse_marker(line_text: str, nosilent_marker: str, line_1: int) -> NosilentMarker | None:
    """the marker on ``line_text``, if it carries one with a rationale.

    recognises ``<marker-stem>(<kind>): <reason>`` and the bare
    ``<marker> <reason>`` form, where ``<marker-stem>`` is the
    configured marker without its trailing colon (``# NOSILENT``).

    :param line_text: a single line of source (without trailing newline)
    :ptype line_text: str
    :param nosilent_marker: the marker substring (default ``# NOSILENT:``)
    :ptype nosilent_marker: str
    :param line_1: the line's 1-indexed number
    :ptype line_1: int
    :return: the marker, or ``None`` when absent or reason-less
    :rtype: NosilentMarker | None
    """
    stem = nosilent_marker.rstrip(":")
    kinded = re.search(re.escape(stem) + r"\(([^)]*)\):(.*)$", line_text)
    if kinded is not None:
        reason = kinded.group(2).strip()
        return NosilentMarker(kind=kinded.group(1).strip(), reason=reason, line=line_1) if reason else None
    idx = line_text.find(nosilent_marker)
    if idx < 0:
        return None
    reason = line_text[idx + len(nosilent_marker) :].strip()
    return NosilentMarker(kind=None, reason=reason, line=line_1) if reason else None


def _marker_with_reason(
    line_text: str,
    nosilent_marker: str,
) -> bool:
    """true iff ``line_text`` contains a marker (bare or kinded) with a rationale.

    :param line_text: a single line of source (without trailing newline)
    :ptype line_text: str
    :param nosilent_marker: the marker substring (default ``# NOSILENT:``)
    :ptype nosilent_marker: str
    :return: whether the line carries a marker with non-empty reason
    :rtype: bool
    """
    return _parse_marker(line_text, nosilent_marker, 0) is not None


def find_nosilent_marker(
    source_lines: list[str],
    first_line: int,
    last_line: int,
    nosilent_marker: str,
) -> NosilentMarker | None:
    """the first marker with a rationale on lines ``first_line..last_line`` (1-indexed, inclusive).

    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param first_line: first 1-indexed line of the window
    :ptype first_line: int
    :param last_line: last 1-indexed line of the window
    :ptype last_line: int
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :return: the marker, or ``None`` when the window carries none
    :rtype: NosilentMarker | None
    """
    for line_1 in range(max(1, first_line), last_line + 1):
        idx = line_1 - 1
        if 0 <= idx < len(source_lines):
            marker = _parse_marker(source_lines[idx], nosilent_marker, line_1)
            if marker is not None:
                return marker
    return None


def has_nosilent_marker(
    source_lines: list[str],
    except_line: int,
    body_line: int | None,
    nosilent_marker: str,
) -> bool:
    """check for a ``# NOSILENT`` comment near the except or body.

    accepts the marker on the line immediately above the ``except``
    statement, on the except line itself, or anywhere between the
    except line and the first body statement (inclusive of both
    endpoints). the inclusive window covers multi-line ``#`` comment
    blocks placed between the ``except`` and its body.

    this answers only whether a marker is PRESENT; whether it excuses
    the handler is :func:`marker_kind_problem`'s question.

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
    end_1 = body_line if body_line is not None else except_line
    return find_nosilent_marker(source_lines, except_line - 1, end_1, nosilent_marker) is not None


def suppress_has_nosilent(
    source_lines: list[str],
    call_line: int,
    nosilent_marker: str,
) -> bool:
    """check for a marker within 2 lines above (or on) a suppress call.

    the window covers ``call_line - 2`` through ``call_line`` (1-indexed);
    that catches both ``# NOSILENT(<kind>): ...`` placed on the line
    above ``with suppress(...)`` and the marker placed on the same line
    as the suppress call.

    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param call_line: 1-indexed line of the ``suppress(...)`` call
    :ptype call_line: int
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :return: whether a marker with non-empty rationale is present
    :rtype: bool
    """
    return _suppress_marker(source_lines, call_line, nosilent_marker) is not None


def _suppress_marker(source_lines: list[str], call_line: int, nosilent_marker: str) -> NosilentMarker | None:
    """the marker governing a ``suppress(...)`` call on ``call_line``, nearest line first.

    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param call_line: 1-indexed line of the call
    :ptype call_line: int
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :return: the marker, or ``None``
    :rtype: NosilentMarker | None
    """
    for line_1 in (call_line, call_line - 1, call_line - 2):
        marker = find_nosilent_marker(source_lines, line_1, line_1, nosilent_marker)
        if marker is not None:
            return marker
    return None


def _type_names(node: ast.expr | None) -> list[str | None]:
    """the caught (or suppressed) type names; ``None`` for one the walker cannot name.

    :param node: ``ExceptHandler.type``, or one ``suppress`` argument
    :ptype node: ast.expr | None
    :return: one entry per type
    :rtype: list[str | None]
    """
    if node is None:
        return [None]
    if isinstance(node, ast.Tuple):
        return [name for elt in node.elts for name in _type_names(elt)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return [None]


def _ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    """``node``'s ancestors, nearest first, stopping at (and including) its scope.

    :param node: the starting node
    :ptype node: ast.AST
    :param parents: child -> parent map for the module
    :ptype parents: dict[ast.AST, ast.AST]
    :return: the chain up to the enclosing function / class / module
    :rtype: list[ast.AST]
    """
    chain: list[ast.AST] = []
    current = parents.get(node)
    while current is not None:
        chain.append(current)
        if isinstance(current, _SCOPES):
            break
        current = parents.get(current)
    return chain


def _raise_follows(anchor: ast.AST, scope: ast.AST) -> bool:
    """true iff ``scope`` holds a ``raise`` after ``anchor`` ends, outside nested scopes.

    :param anchor: the ``try`` (or ``with``) the marker sits in
    :ptype anchor: ast.AST
    :param scope: the enclosing function or module
    :ptype scope: ast.AST
    :return: whether the function raises after the anchor
    :rtype: bool
    """
    end = getattr(anchor, "end_lineno", None) or 0
    stack: list[ast.AST] = list(ast.iter_child_nodes(scope))
    while stack:
        current = stack.pop()
        if isinstance(current, _SCOPES):
            continue
        if isinstance(current, ast.Raise) and current.lineno > end:
            return True
        stack.extend(ast.iter_child_nodes(current))
    return False


def marker_kind_problem(
    marker: NosilentMarker,
    *,
    site: ast.AST,
    anchor: ast.AST,
    caught: list[str | None],
    guarded: list[ast.stmt],
    parents: dict[ast.AST, ast.AST],
    logger_names: frozenset[str] = _DEFAULT_LOGGER_NAMES,
) -> str | None:
    """why ``marker`` does not excuse the silent site, or ``None`` when it does.

    :param marker: the marker found in the site's window
    :ptype marker: NosilentMarker
    :param site: the ``ExceptHandler`` or ``suppress(...)`` call
    :ptype site: ast.AST
    :param anchor: the ``try`` / ``with`` statement the site belongs to
    :ptype anchor: ast.AST
    :param caught: the caught or suppressed type names
    :ptype caught: list[str | None]
    :param guarded: the statements the site guards (the ``try`` / ``with`` body)
    :ptype guarded: list[ast.stmt]
    :param parents: child -> parent map for the module
    :ptype parents: dict[ast.AST, ast.AST]
    :param logger_names: known logger receiver names, for the ``logged`` check
    :ptype logger_names: frozenset[str]
    :return: the problem, or ``None``
    :rtype: str | None
    """
    kinds = ", ".join(sorted(MARKER_KINDS))
    if marker.kind is None:
        return (
            "a bare `# NOSILENT: <reason>` no longer excuses an error turned into a quiet answer; "
            "let the error reach the caller (raise, a typed failure, a log), or name the kind "
            f"the walker can check: `# NOSILENT(<kind>): <reason>`, kind one of {kinds}"
        )
    if marker.kind not in MARKER_KINDS:
        return f"unknown NOSILENT kind {marker.kind!r}; kinds are {kinds}"
    chain = _ancestors(site, parents)
    if marker.kind == "logged":
        if isinstance(site, ast.ExceptHandler) and _logs_at_operator_level(site.body, logger_names):
            return None
        return "NOSILENT(logged) needs the handler itself to log at WARNING or above"
    if marker.kind == "expected":
        if any(name is None or name in _BROAD_TYPES for name in caught):
            return "NOSILENT(expected) needs every caught type named and narrower than Exception"
        return None
    if marker.kind == "optional-import":
        if any(name not in _IMPORT_ERRORS for name in caught):
            return "NOSILENT(optional-import) may catch only ImportError / ModuleNotFoundError"
        if not any(isinstance(node, (ast.Import, ast.ImportFrom)) for stmt in guarded for node in ast.walk(stmt)):
            return "NOSILENT(optional-import) guards no import"
        return None
    if marker.kind == "attempt":
        in_loop = any(isinstance(node, _LOOPS) for node in chain)
        scope = chain[-1] if chain else anchor
        if in_loop or _raise_follows(anchor, scope):
            return None
        return "NOSILENT(attempt) needs the handler inside a loop, or a raise after the try in its function"
    for node in chain:
        if isinstance(node, ast.ExceptHandler) and body_reraises(node.body):
            return None
    return "NOSILENT(cleanup) needs the site inside an except body that re-raises the primary error"


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
    """walk every module for silent ``except`` and unexcused ``suppress(...)``.

    one :class:`Violation` is produced per offending site:

    - ``no_silent_swallow.except`` for a silent except handler or a
      bare ``except:`` clause. ``symbol`` is the caught type name (or
      ``"BaseException"`` for the bare case).
    - ``no_silent_swallow.suppress`` for a ``contextlib.suppress(...)``
      without a valid kinded marker. ``symbol`` is the first argument's
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
                note_unscanned(module_path, "not valid utf-8")
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            source_lines = source_text.splitlines()
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Try, ast.TryStar)):
                    _scan_try(
                        node,
                        module_path,
                        source_lines,
                        logger_names,
                        logger_methods,
                        nosilent_marker,
                        parents,
                        violations,
                    )
                if is_suppress_call(node):
                    assert isinstance(node, ast.Call)
                    _scan_suppress(node, module_path, source_lines, nosilent_marker, parents, violations)
            _scan_marker_comments(source_text, module_path, nosilent_marker, violations)
    return violations


def _scan_suppress(
    call: ast.Call,
    module_path: Path,
    source_lines: list[str],
    nosilent_marker: str,
    parents: dict[ast.AST, ast.AST],
    violations: list[Violation],
) -> None:
    """flag one ``suppress(...)`` call unless a valid kinded marker excuses it.

    :param call: the suppress call
    :ptype call: ast.Call
    :param module_path: file containing the call
    :ptype module_path: Path
    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :param parents: child -> parent map for the module
    :ptype parents: dict[ast.AST, ast.AST]
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    marker = _suppress_marker(source_lines, call.lineno, nosilent_marker)
    if marker is None:
        problem: str | None = (
            f"contextlib.suppress without a `{nosilent_marker.rstrip(':')}(<kind>): <reason>` marker "
            "within 2 lines above the call"
        )
    else:
        item = parents.get(call)
        anchor = parents.get(item) if isinstance(item, ast.withitem) else None
        guarded = list(anchor.body) if isinstance(anchor, (ast.With, ast.AsyncWith)) else []
        caught = [name for arg in call.args for name in _type_names(arg)] or [None]
        problem = marker_kind_problem(
            marker,
            site=call,
            anchor=anchor if anchor is not None else call,
            caught=caught,
            guarded=guarded,
            parents=parents,
        )
    if problem is None:
        return
    violations.append(
        Violation(
            category=_SUPPRESS_CATEGORY,
            file=module_path,
            line=call.lineno,
            symbol=_suppress_symbol(call),
            reason=problem,
        )
    )


def _scan_try(
    node: ast.Try | ast.TryStar,
    module_path: Path,
    source_lines: list[str],
    logger_names: frozenset[str],
    logger_methods: frozenset[str],
    nosilent_marker: str,
    parents: dict[ast.AST, ast.AST],
    violations: list[Violation],
) -> None:
    """scan a single ``ast.Try`` node for handler-level violations.

    extracted from :func:`find_silent_swallows` to keep the per-node
    nesting shallow. mutates ``violations`` in place to preserve
    source order.

    :param node: the try node to inspect
    :ptype node: ast.Try | ast.TryStar
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
    :param parents: child -> parent map for the module
    :ptype parents: dict[ast.AST, ast.AST]
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
                    reason=("bare `except:` is banned; narrow to `except Exception:` or narrower"),
                )
            )
            continue
        category = body_silent_category(handler.body)
        if category is None:
            # a body that does work and leaves inertly is judged only when the work is a log line:
            # a log tells the operator, never the caller, who still gets the inert answer.
            exit_label = body_silent_exit(handler.body)
            if exit_label is None or not body_contains_log(handler.body, logger_names, logger_methods):
                _check_stated_kind(
                    node, handler, module_path, source_lines, nosilent_marker, parents, logger_names, violations
                )
                continue
            category = f"logged-then-{exit_label}"
        elif body_contains_log(handler.body, logger_names, logger_methods) or body_reraises(handler.body):
            _check_stated_kind(
                node, handler, module_path, source_lines, nosilent_marker, parents, logger_names, violations
            )
            continue
        body_line = handler.body[0].lineno if handler.body else handler.lineno
        marker = find_nosilent_marker(source_lines, handler.lineno - 1, body_line, nosilent_marker)
        if marker is None:
            problem: str | None = (
                f"silent swallow (body={category}); let the error reach the caller (re-raise, a typed "
                f"failure{', a log call' if not category.startswith('logged-') else ''}), or add a `{nosilent_marker.rstrip(':')}(<kind>): <reason>` "
                "marker the walker can check"
            )
        else:
            problem = marker_kind_problem(
                marker,
                site=handler,
                anchor=node,
                caught=_type_names(handler.type),
                guarded=list(node.body),
                parents=parents,
                logger_names=logger_names,
            )
            if problem is not None:
                problem = f"silent swallow (body={category}): {problem}"
        if problem is None:
            continue
        violations.append(
            Violation(
                category=_EXCEPT_CATEGORY,
                file=module_path,
                line=handler.lineno,
                symbol=_handler_symbol(handler),
                reason=problem,
            )
        )


def _scan_marker_comments(
    source_text: str,
    module_path: Path,
    nosilent_marker: str,
    violations: list[Violation],
) -> None:
    """flag every bare or unknown-kind marker COMMENT in the module.

    only real ``#`` comments count (a docstring that quotes the marker
    is documentation, not a marker). a module that does not tokenize was
    already reported as unscanned by the parse step.

    :param source_text: the module's source
    :ptype source_text: str
    :param module_path: the module's path
    :ptype module_path: Path
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    stem = nosilent_marker.rstrip(":")
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(source_text).readline)
            if token.type == tokenize.COMMENT and stem in token.string
        ]
    except tokenize.TokenError, SyntaxError:
        # NOSILENT(expected): the parse step already named this module as unscanned; there are no comments to read
        return
    kinds = ", ".join(sorted(MARKER_KINDS))
    for token in tokens:
        marker = _parse_marker(token.string, nosilent_marker, token.start[0])
        if marker is None:
            continue
        if marker.kind is None:
            reason = f"a bare `{nosilent_marker} <reason>` marker; name its kind, one of {kinds}"
        elif marker.kind not in MARKER_KINDS:
            reason = f"unknown NOSILENT kind {marker.kind!r}; kinds are {kinds}"
        else:
            continue
        violations.append(
            Violation(category=_MARKER_CATEGORY, file=module_path, line=marker.line, symbol="NOSILENT", reason=reason)
        )


def _check_stated_kind(
    node: ast.Try | ast.TryStar,
    handler: ast.ExceptHandler,
    module_path: Path,
    source_lines: list[str],
    nosilent_marker: str,
    parents: dict[ast.AST, ast.AST],
    logger_names: frozenset[str],
    violations: list[Violation],
) -> None:
    """hold a kinded marker on a handler that needs none to the kind it states.

    such a handler passes without a marker, but a marker that claims
    ``logged`` over a DEBUG line, or ``expected`` over ``Exception``,
    is a false statement a reader will trust; it is flagged like any
    other. unknown kinds and bare markers are reported by the comment
    scan, so only a known kind is checked here.

    :param node: the try the handler belongs to
    :ptype node: ast.Try | ast.TryStar
    :param handler: the handler
    :ptype handler: ast.ExceptHandler
    :param module_path: file containing the try
    :ptype module_path: Path
    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param nosilent_marker: the marker substring
    :ptype nosilent_marker: str
    :param parents: child -> parent map for the module
    :ptype parents: dict[ast.AST, ast.AST]
    :param logger_names: known logger receiver names
    :ptype logger_names: frozenset[str]
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    if body_reraises(handler.body):
        return  # it cannot swallow; a marker in its window speaks for a suppress or handler inside it
    body_line = handler.body[0].lineno if handler.body else handler.lineno
    marker = find_nosilent_marker(source_lines, handler.lineno - 1, body_line, nosilent_marker)
    if marker is None or marker.kind not in MARKER_KINDS:
        return
    problem = marker_kind_problem(
        marker,
        site=handler,
        anchor=node,
        caught=_type_names(handler.type),
        guarded=list(node.body),
        parents=parents,
        logger_names=logger_names,
    )
    if problem is not None:
        violations.append(
            Violation(
                category=_EXCEPT_CATEGORY,
                file=module_path,
                line=handler.lineno,
                symbol=_handler_symbol(handler),
                reason=f"the marker's kind does not hold: {problem}",
            )
        )
