"""walkers for the print-log-lines enforcement domain.

:func:`find_print_log_lines` reports, in every in-scope module (one that imports a module under a
configured root, ``threetears`` by default), each call that writes a log line around
``threetears.observe``:

- ``print(..., file=sys.stderr)`` -- also ``sys.__stderr__``, ``import sys as s`` then
  ``file=s.stderr``, and ``from sys import stderr`` then ``file=stderr``. category
  ``print_log_lines.stderr_print``, symbol ``print``.
- a stdlib ``logging`` call: ``getLogger``, ``basicConfig``, or a root-logger call (``debug``,
  ``info``, ``warning``, ``warn``, ``error``, ``exception``, ``critical``, ``fatal``, ``log``),
  through the module (``logging.getLogger``, ``import logging as lg`` then ``lg.basicConfig``) or a
  name imported from it (``from logging import getLogger``). category
  ``print_log_lines.stdlib_logging``, symbol ``logging.<name>``.

intentionally NOT flagged: ``print`` to stdout (a command's output) or to any other file; a method
that merely shares a name (``log.warning(...)`` on a ``get_logger`` logger, ``obj.getLogger()``);
any module that does not import an in-scope root.

:func:`find_scoped_modules` lists the in-scope modules, for the runner's non-vacuity floor.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    iter_python_files,
    note_unscanned,
    parse_python_file,
    relative_posix_path,
)

__all__ = [
    "find_print_log_lines",
    "find_scoped_modules",
    "imports_any_root",
]

_STDERR_CATEGORY = "print_log_lines.stderr_print"
_LOGGING_CATEGORY = "print_log_lines.stdlib_logging"

#: ``logging`` module functions that configure or write to stdlib logging directly.
_LOGGING_FUNCTIONS: frozenset[str] = frozenset(
    {
        "getLogger",
        "basicConfig",
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "fatal",
        "log",
    }
)

_STDERR_ATTRIBUTES: frozenset[str] = frozenset({"stderr", "__stderr__"})


def _under_root(name: str, roots: tuple[str, ...]) -> bool:
    """whether dotted module ``name`` is one of ``roots`` or inside one.

    :param name: module name as written in the import
    :ptype name: str
    :param roots: dotted roots
    :ptype roots: tuple[str, ...]
    :return: ``True`` for ``threetears`` or ``threetears.x`` under root ``threetears``
    :rtype: bool
    """
    return any(name == root or name.startswith(f"{root}.") for root in roots)


def imports_any_root(tree: ast.Module, roots: tuple[str, ...]) -> bool:
    """whether the module imports anything under one of ``roots``, anywhere in its body.

    an import inside a function counts: a module that imports 3tears lazily still runs on it.
    relative imports never count (they name the module's own package). an empty ``roots`` puts
    every module in scope.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param roots: dotted roots that put a module in scope
    :ptype roots: tuple[str, ...]
    :return: whether the module is in scope
    :rtype: bool
    """
    if not roots:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(_under_root(alias.name, roots) for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and node.module and _under_root(node.module, roots):
            return True
    return False


class _Bindings:
    """the local names a module binds to ``sys``, ``sys.stderr``, ``logging`` and its functions.

    :ivar sys_modules: names bound to the ``sys`` module
    :ivar stderr_names: names bound to ``sys.stderr`` / ``sys.__stderr__``
    :ivar logging_modules: names bound to the ``logging`` module
    :ivar logging_functions: local name -> the ``logging`` function it is bound to
    """

    def __init__(self, tree: ast.Module) -> None:
        """collect the bindings from every import in ``tree``.

        :param tree: the parsed module
        :ptype tree: ast.Module
        :return: None
        :rtype: None
        """
        self.sys_modules: set[str] = set()
        self.stderr_names: set[str] = set()
        self.logging_modules: set[str] = set()
        self.logging_functions: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sys":
                        self.sys_modules.add(alias.asname or "sys")
                    elif alias.name == "logging":
                        self.logging_modules.add(alias.asname or "logging")
                    elif alias.name.startswith("logging.") and alias.asname is None:
                        # ``import logging.handlers`` binds ``logging`` itself
                        self.logging_modules.add("logging")
            elif isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    local = alias.asname or alias.name
                    if node.module == "sys" and alias.name in _STDERR_ATTRIBUTES:
                        self.stderr_names.add(local)
                    elif node.module == "logging" and alias.name in _LOGGING_FUNCTIONS:
                        self.logging_functions[local] = alias.name

    def is_stderr(self, expr: ast.expr) -> bool:
        """whether ``expr`` names the process's stderr.

        :param expr: the ``file=`` argument
        :ptype expr: ast.expr
        :return: ``True`` for ``sys.stderr`` in any spelling this module bound
        :rtype: bool
        """
        if isinstance(expr, ast.Attribute):
            return (
                expr.attr in _STDERR_ATTRIBUTES
                and isinstance(expr.value, ast.Name)
                and expr.value.id in self.sys_modules
            )
        return isinstance(expr, ast.Name) and expr.id in self.stderr_names

    def logging_function(self, call: ast.Call) -> str | None:
        """the stdlib ``logging`` function ``call`` invokes, if it is one.

        :param call: the call
        :ptype call: ast.Call
        :return: the function's name in ``logging``, or ``None``
        :rtype: str | None
        """
        func = call.func
        if isinstance(func, ast.Attribute):
            if (
                func.attr in _LOGGING_FUNCTIONS
                and isinstance(func.value, ast.Name)
                and func.value.id in self.logging_modules
            ):
                return func.attr
            return None
        if isinstance(func, ast.Name):
            return self.logging_functions.get(func.id)
        return None


def _marked(source_lines: list[str], node: ast.expr, marker: str) -> bool:
    """whether ``marker`` appears on any line the call spans.

    :param source_lines: the module's lines
    :ptype source_lines: list[str]
    :param node: the call
    :ptype node: ast.expr
    :param marker: the excusing comment
    :ptype marker: str
    :return: whether the call is excused
    :rtype: bool
    """
    last = node.end_lineno or node.lineno
    return any(marker in line for line in source_lines[node.lineno - 1 : last])


def _is_stderr_print(call: ast.Call, bindings: _Bindings) -> bool:
    """whether ``call`` is the builtin ``print`` writing to stderr.

    :param call: the call
    :ptype call: ast.Call
    :param bindings: the module's bindings
    :ptype bindings: _Bindings
    :return: ``True`` for ``print(..., file=<stderr>)``
    :rtype: bool
    """
    if not (isinstance(call.func, ast.Name) and call.func.id == "print"):
        return False
    return any(keyword.arg == "file" and bindings.is_stderr(keyword.value) for keyword in call.keywords)


def _scoped_trees(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    scope_import_roots: tuple[str, ...],
) -> list[tuple[Path, ast.Module]]:
    """every in-scope, non-exempt module under ``src_roots``, parsed.

    :param src_roots: trees to scan
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: root that ``exempt_files`` keys are relative to
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale; listed modules are skipped
    :ptype exempt_files: dict[str, str]
    :param scope_import_roots: dotted roots that put a module in scope
    :ptype scope_import_roots: tuple[str, ...]
    :return: (path, tree) per in-scope module, in walk order
    :rtype: list[tuple[Path, ast.Module]]
    """
    scoped: list[tuple[Path, ast.Module]] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            if relative_posix_path(module_path, repo_root) in exempt_files:
                continue
            tree = parse_python_file(module_path)
            if tree is not None and imports_any_root(tree, scope_import_roots):
                scoped.append((module_path, tree))
    return scoped


def find_scoped_modules(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    scope_import_roots: tuple[str, ...],
) -> list[Path]:
    """the modules :func:`find_print_log_lines` would scan.

    :param src_roots: trees to scan
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: root that ``exempt_files`` keys are relative to
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale; listed modules are skipped
    :ptype exempt_files: dict[str, str]
    :param scope_import_roots: dotted roots that put a module in scope
    :ptype scope_import_roots: tuple[str, ...]
    :return: the in-scope module paths
    :rtype: list[Path]
    """
    return [path for path, _ in _scoped_trees(src_roots, repo_root, exempt_files, scope_import_roots)]


def find_print_log_lines(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    line_marker: str,
    scope_import_roots: tuple[str, ...],
) -> list[Violation]:
    """every stderr ``print`` and direct stdlib-logging call in an in-scope module.

    :param src_roots: trees to scan
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: root that ``exempt_files`` keys are relative to
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale; listed modules are skipped
    :ptype exempt_files: dict[str, str]
    :param line_marker: comment excusing one call, on any line it spans
    :ptype line_marker: str
    :param scope_import_roots: dotted roots that put a module in scope; empty scans every module
    :ptype scope_import_roots: tuple[str, ...]
    :return: violations, in walk order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for module_path, tree in _scoped_trees(src_roots, repo_root, exempt_files, scope_import_roots):
        try:
            source_lines = module_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            note_unscanned(module_path, "not valid utf-8")
            continue
        bindings = _Bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_stderr_print(node, bindings):
                category, symbol = _STDERR_CATEGORY, "print"
                reason = (
                    "print(..., file=sys.stderr) is a log line with no level, no fields and no place in the "
                    "log pipeline; log it with threetears.observe.get_logger"
                )
            elif (function := bindings.logging_function(node)) is not None:
                category, symbol = _LOGGING_CATEGORY, f"logging.{function}"
                reason = (
                    f"stdlib logging.{function}() bypasses threetears.observe's formatter and context; use "
                    "threetears.observe.get_logger (configure_logging sets levels and handlers)"
                )
            else:
                continue
            if _marked(source_lines, node, line_marker):
                continue
            violations.append(
                Violation(
                    category=category,
                    file=module_path,
                    line=node.lineno,
                    symbol=symbol,
                    reason=(
                        f"{reason}. a command whose stderr output is its answer to the person who ran it "
                        f"goes in the config's `exempt_files`; one justified call takes '{line_marker}'."
                    ),
                )
            )
    return violations
