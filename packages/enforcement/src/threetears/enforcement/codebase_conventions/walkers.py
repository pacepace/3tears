"""walkers for the codebase-conventions enforcement domain.

four AST-based walkers cover the four conventions:

- :func:`find_print_calls` — flag every ``print(...)`` call where the
  callee is the bare builtin ``Name(id="print")``. attribute calls
  (``obj.print(...)``) are not flagged because they invoke a method,
  not the builtin.
- :func:`find_stdlib_getlogger_calls` — flag ``logging.getLogger(...)``
  attribute calls (receiver is the bare ``Name(id="logging")``) and
  ``getLogger(...)`` bare-name calls (which require a prior ``from
  logging import getLogger`` to mean the stdlib symbol; the walker
  conservatively flags every bare ``getLogger`` call so unsanctioned
  imports get caught). per-line marker discipline (default
  ``# stdlib-getlogger: ok``) and the file-level allowlist both
  exempt offending sites.
- :func:`find_missing_future_annotations` — flag every module that
  does not have ``from __future__ import annotations`` at top level.
- :func:`find_missing_return_types` — flag every non-dunder, non-test
  function definition (sync or async) that lacks a ``returns``
  annotation. dunder names (``__init__``, ``__repr__``, ...) and
  pytest test functions (``def test_*`` or any function in a
  ``test_*.py`` file) are skipped by construction.
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

__all__ = [
    "find_missing_future_annotations",
    "find_missing_return_types",
    "find_print_calls",
    "find_stdlib_getlogger_calls",
]


_PRINT_CATEGORY = "codebase_conventions.print"
_GETLOGGER_CATEGORY = "codebase_conventions.stdlib_getlogger"
_FUTURE_ANNOTATIONS_CATEGORY = "codebase_conventions.future_annotations"
_RETURN_TYPE_CATEGORY = "codebase_conventions.return_type"


# ----------------------------------------------------------------------
# walker 1 — find_print_calls
# ----------------------------------------------------------------------


def find_print_calls(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
) -> list[Violation]:
    """flag every bare ``print(...)`` call in production source.

    one :class:`Violation` per offending call site. category is
    ``codebase_conventions.print`` for every record. the symbol is
    always the literal ``"print"``.

    flagged shapes:

    - ``print(x)`` — callee is ``Name(id="print")``.
    - ``print()`` — same; argument count does not matter.

    NOT flagged:

    - ``obj.print(x)`` — callee is ``Attribute``; this is a method
      call, not the builtin.
    - ``logger.print(...)`` — same reason.
    - ``"print"`` (string literal) — not a call expression.

    file-level exemption: a file whose
    :func:`~threetears.enforcement.common.ast_helpers.relative_posix_path`
    relative to ``repo_root`` is a key in ``exempt_files`` is skipped
    before any AST work is done.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping;
        listed files are skipped entirely
    :ptype exempt_files: dict[str, str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            rel = relative_posix_path(module_path, repo_root)
            if rel in exempt_files:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Name):
                    continue
                if func.id != "print":
                    continue
                violations.append(
                    Violation(
                        category=_PRINT_CATEGORY,
                        file=module_path,
                        line=node.lineno,
                        symbol="print",
                        reason=("bare print() found; use the project logger (threetears.observe.get_logger) instead."),
                    )
                )
    return violations


# ----------------------------------------------------------------------
# walker 2 — find_stdlib_getlogger_calls
# ----------------------------------------------------------------------


def find_stdlib_getlogger_calls(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    marker: str,
) -> list[Violation]:
    """flag every stdlib ``getLogger(...)`` call in production source.

    one :class:`Violation` per offending call site. category is
    ``codebase_conventions.stdlib_getlogger`` for every record. the
    symbol is always the literal ``"getLogger"``.

    flagged shapes:

    - ``logging.getLogger(__name__)`` — ``Attribute(Name("logging"),
      "getLogger")``.
    - ``getLogger(__name__)`` — bare ``Name("getLogger")`` call,
      which only resolves to the stdlib symbol after ``from logging
      import getLogger``. the walker conservatively flags every bare
      ``getLogger(...)`` so unsanctioned imports cannot slip through.

    NOT flagged:

    - ``obj.getLogger(...)`` — receiver is not ``Name("logging")``;
      this is a method on some other object.
    - ``foo.bar.getLogger(...)`` — receiver is ``Attribute``, not
      ``Name("logging")``.

    two escape hatches:

    - ``exempt_files``: a file-relative-posix-path -> rationale dict.
      a listed file is skipped entirely.
    - per-line marker (default ``# stdlib-getlogger: ok``): the
      marker substring must appear on the offending call's source
      line. it scopes to that line — adjacent calls without the
      marker are still flagged.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping;
        listed files are skipped entirely
    :ptype exempt_files: dict[str, str]
    :param marker: per-line marker substring (default
        ``# stdlib-getlogger: ok``)
    :ptype marker: str
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            rel = relative_posix_path(module_path, repo_root)
            if rel in exempt_files:
                continue
            try:
                source_text = module_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            source_lines = source_text.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not _is_stdlib_getlogger_call(node):
                    continue
                if _line_has_marker(source_lines, node.lineno, marker):
                    continue
                violations.append(
                    Violation(
                        category=_GETLOGGER_CATEGORY,
                        file=module_path,
                        line=node.lineno,
                        symbol="getLogger",
                        reason=(
                            "stdlib logging.getLogger() found; use "
                            "threetears.observe.get_logger instead. "
                            f"if configuring a third-party logger by "
                            f"name, add '{marker}' on the call line "
                            f"with a justification, or add the module "
                            f"to the config's `getlogger_exempt_files` "
                            f"mapping."
                        ),
                    )
                )
    return violations


def _is_stdlib_getlogger_call(node: ast.Call) -> bool:
    """true iff ``node`` is a ``logging.getLogger(...)`` or bare
    ``getLogger(...)`` call.

    matches:

    - ``Attribute(Name("logging"), "getLogger")`` — the canonical
      stdlib call.
    - ``Name("getLogger")`` — the bare-name form, which resolves to
      the stdlib symbol only after ``from logging import getLogger``.
      the walker conservatively flags every bare ``getLogger`` call
      so unsanctioned imports cannot slip through.

    explicitly does NOT match attribute calls whose receiver is not
    ``Name("logging")`` — those are method calls on some other object.

    :param node: ast.Call node to test
    :ptype node: ast.Call
    :return: whether the node is a stdlib getLogger call
    :rtype: bool
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr != "getLogger":
            return False
        receiver = func.value
        return isinstance(receiver, ast.Name) and receiver.id == "logging"
    if isinstance(func, ast.Name):
        return func.id == "getLogger"
    return False


def _line_has_marker(
    source_lines: list[str],
    lineno: int,
    marker: str,
) -> bool:
    """true iff ``source_lines[lineno-1]`` carries the per-line marker.

    ``lineno`` is 1-indexed (matches python AST convention). the
    marker is matched as a case-sensitive substring after stripping
    trailing whitespace from the line.

    :param source_lines: file contents split by lines (no newlines)
    :ptype source_lines: list[str]
    :param lineno: 1-indexed line number of the offending site
    :ptype lineno: int
    :param marker: the marker substring (case-sensitive)
    :ptype marker: str
    :return: whether the marker is present on the offending line
    :rtype: bool
    """
    idx = lineno - 1
    if idx < 0 or idx >= len(source_lines):
        return False
    return marker in source_lines[idx].rstrip()


# ----------------------------------------------------------------------
# walker 3 — find_missing_future_annotations
# ----------------------------------------------------------------------


def find_missing_future_annotations(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    skip_basenames: frozenset[str],
) -> list[Violation]:
    """flag every module that lacks ``from __future__ import annotations``.

    one :class:`Violation` per offending module. category is
    ``codebase_conventions.future_annotations`` for every record. the
    symbol is always the literal ``"__future__"``.

    the check is performed against top-level ``ast.ImportFrom`` nodes
    (children of ``ast.Module.body``); ``__future__`` imports nested
    inside conditionals or functions are not the python-language form
    and are not detected.

    file-level exemptions: a file is skipped iff its basename is in
    ``skip_basenames`` (default ``{"__init__.py"}``) OR its
    relative-posix path is in ``exempt_files``.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping;
        listed files are skipped entirely
    :ptype exempt_files: dict[str, str]
    :param skip_basenames: basenames to skip (default
        ``{"__init__.py"}``)
    :ptype skip_basenames: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            if module_path.name in skip_basenames:
                continue
            rel = relative_posix_path(module_path, repo_root)
            if rel in exempt_files:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            if _has_future_annotations(tree):
                continue
            violations.append(
                Violation(
                    category=_FUTURE_ANNOTATIONS_CATEGORY,
                    file=module_path,
                    line=1,
                    symbol="__future__",
                    reason=(
                        "module is missing 'from __future__ import "
                        "annotations'; the project depends on PEP 563-style "
                        "evaluation for all type hints."
                    ),
                )
            )
    return violations


def _has_future_annotations(tree: ast.Module) -> bool:
    """true iff the module's top-level body has ``from __future__ import annotations``.

    only top-level imports count. nested ``__future__`` imports are
    not the python-language form and are ignored.

    :param tree: parsed module AST
    :ptype tree: ast.Module
    :return: whether the import is present at module top level
    :rtype: bool
    """
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "__future__":
            continue
        for alias in node.names:
            if alias.name == "annotations":
                return True
    return False


# ----------------------------------------------------------------------
# walker 4 — find_missing_return_types
# ----------------------------------------------------------------------


def find_missing_return_types(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    skip_basenames: frozenset[str],
) -> list[Violation]:
    """flag every non-dunder, non-test function lacking a return type.

    one :class:`Violation` per offending function definition (sync
    or async). category is ``codebase_conventions.return_type`` for
    every record. the symbol is the function's name.

    skipped by construction:

    - dunder names (``__init__``, ``__repr__``, ``__eq__``, ...): a
      name that starts AND ends with double underscore.
    - pytest test functions: function name starts with ``test_``.
    - all functions in test files: filename matches ``test_*.py``.
    - all functions in files whose basename is in ``skip_basenames``
      (default ``{"__init__.py"}``).
    - all functions in files listed in ``exempt_files``.

    walks the AST recursively (``ast.walk``) so methods on nested
    classes and inner functions are checked. each function with
    ``returns is None`` produces one violation.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping;
        listed files are skipped entirely
    :ptype exempt_files: dict[str, str]
    :param skip_basenames: basenames to skip (default
        ``{"__init__.py"}``)
    :ptype skip_basenames: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            if module_path.name in skip_basenames:
                continue
            if module_path.name.startswith("test_"):
                continue
            rel = relative_posix_path(module_path, repo_root)
            if rel in exempt_files:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue
                if _is_dunder(node.name):
                    continue
                if node.name.startswith("test_"):
                    continue
                if node.returns is not None:
                    continue
                violations.append(
                    Violation(
                        category=_RETURN_TYPE_CATEGORY,
                        file=module_path,
                        line=node.lineno,
                        symbol=node.name,
                        reason=(
                            f"function {node.name}() is missing a return "
                            f"type annotation; add '-> <type>' before the "
                            f"colon."
                        ),
                    )
                )
    return violations


def _is_dunder(name: str) -> bool:
    """true iff ``name`` starts AND ends with ``__``.

    matches python's "dunder" convention: ``__init__``, ``__repr__``,
    ``__eq__``, etc. name-mangled names (``__foo``) are NOT dunders
    because they don't end with ``__``.

    :param name: identifier to classify
    :ptype name: str
    :return: whether ``name`` is a dunder
    :rtype: bool
    """
    return name.startswith("__") and name.endswith("__")
