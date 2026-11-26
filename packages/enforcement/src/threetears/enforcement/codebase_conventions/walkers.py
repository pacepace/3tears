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

all four read one fact per file, :class:`_ModuleConventions`, which the
scan cache keeps by the file's bytes (``file_facts``): a file seen before
is not parsed again, by any of the walkers.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    iter_python_files,
    relative_posix_path,
)
from threetears.enforcement.common.scan_cache import CACHE_DIRECTORY, file_facts

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


@dataclass(frozen=True)
class _ModuleConventions:
    """what the four walkers read of one module, kept in the scan cache by its bytes.

    :ivar prints: the line of every bare ``print(...)`` call, in :func:`ast.walk` order
    :ivar getloggers: the line of every stdlib ``getLogger(...)`` call, in :func:`ast.walk` order,
        with that source line right-stripped (``None`` past the file's last line), for the marker
    :ivar future_annotations: whether the module imports ``annotations`` from ``__future__`` at top level
    :ivar untyped: ``(line, name)`` of every non-dunder, non-``test_`` function without a return
        annotation, in :func:`ast.walk` order
    """

    prints: tuple[int, ...]
    getloggers: tuple[tuple[int, str | None], ...]
    future_annotations: bool
    untyped: tuple[tuple[int, str], ...]


def _module_conventions(path: Path, tree: ast.Module) -> _ModuleConventions:
    """read one module's fact; the file is read again only when it calls ``getLogger``, for its lines.

    :param path: the module's file
    :ptype path: Path
    :param tree: its parsed source
    :ptype tree: ast.Module
    :return: the module's fact
    :rtype: _ModuleConventions
    """
    prints: list[int] = []
    getlogger_lines: list[int] = []
    untyped: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                prints.append(node.lineno)
            if _is_stdlib_getlogger_call(node):
                getlogger_lines.append(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_dunder(node.name) and not node.name.startswith("test_") and node.returns is None:
                untyped.append((node.lineno, node.name))
    getloggers: tuple[tuple[int, str | None], ...] = ()
    if getlogger_lines:
        source_lines = path.read_text(encoding="utf-8").splitlines()
        getloggers = tuple(
            (line, source_lines[line - 1].rstrip() if 0 <= line - 1 < len(source_lines) else None)
            for line in getlogger_lines
        )
    return _ModuleConventions(
        prints=tuple(prints),
        getloggers=getloggers,
        future_annotations=_has_future_annotations(tree),
        untyped=tuple(untyped),
    )


def _conventions(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    scanned: Callable[[Path], bool],
) -> list[tuple[Path, _ModuleConventions]]:
    """each scanned module under the roots, in walk order, with its fact.

    a file that does not parse goes through ``note_unscanned`` and is left out. the files under
    ``repo_root`` are keyed relative to it, any other root's relative to that root, and every fact
    lives in ``repo_root``'s scan cache.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root, where the scan cache lives
    :ptype repo_root: Path
    :param scanned: whether a walker reads a file at all, before it is parsed
    :ptype scanned: Callable[[Path], bool]
    :return: ``(path, fact)`` pairs, a file under two roots once per root
    :rtype: list[tuple[Path, _ModuleConventions]]
    """
    walked: list[Path] = []
    by_anchor: dict[Path, list[Path]] = {}
    for root in src_roots:
        paths = [path for path in iter_python_files(root) if scanned(path)]
        walked.extend(paths)
        by_anchor.setdefault(repo_root if root.is_relative_to(repo_root) else root, []).extend(paths)
    facts: dict[Path, _ModuleConventions] = {}
    for anchor, paths in by_anchor.items():
        facts.update(
            file_facts(
                _module_conventions,
                paths,
                root=anchor,
                directory=repo_root / CACHE_DIRECTORY,
                unparsable="note",
            )
        )
    return [(path, facts[path]) for path in walked if path in facts]


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
    return [
        Violation(
            category=_PRINT_CATEGORY,
            file=module_path,
            line=line,
            symbol="print",
            reason=("bare print() found; use the project logger (threetears.observe.get_logger) instead."),
        )
        for module_path, fact in _conventions(
            src_roots,
            repo_root,
            lambda path: relative_posix_path(path, repo_root) not in exempt_files,
        )
        for line in fact.prints
    ]


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
    for module_path, fact in _conventions(
        src_roots,
        repo_root,
        lambda path: relative_posix_path(path, repo_root) not in exempt_files,
    ):
        for lineno, source_line in fact.getloggers:
            if source_line is not None and marker in source_line:
                continue
            violations.append(
                Violation(
                    category=_GETLOGGER_CATEGORY,
                    file=module_path,
                    line=lineno,
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
    return [
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
        for module_path, fact in _conventions(
            src_roots,
            repo_root,
            lambda path: path.name not in skip_basenames and relative_posix_path(path, repo_root) not in exempt_files,
        )
        if not fact.future_annotations
    ]


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
    return [
        Violation(
            category=_RETURN_TYPE_CATEGORY,
            file=module_path,
            line=line,
            symbol=name,
            reason=(f"function {name}() is missing a return type annotation; add '-> <type>' before the colon."),
        )
        for module_path, fact in _conventions(
            src_roots,
            repo_root,
            lambda path: (
                path.name not in skip_basenames
                and not path.name.startswith("test_")
                and relative_posix_path(path, repo_root) not in exempt_files
            ),
        )
        for line, name in fact.untyped
    ]


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
