"""walker for the logger-coverage enforcement domain.

the single walker, :func:`find_modules_without_logger`, scans every
non-empty, non-skipped module under the configured src trees for a
module-level logger assignment. recognised shapes:

- ``log = get_logger(__name__)`` — bare-name callee, target ``log``
- ``_logger = get_logger(__name__)`` — bare-name callee, legacy alias
- ``log = observe.get_logger(__name__)`` — attribute callee, target ``log``

the comparison is configurable: ``logger_factory_names`` is the set
of acceptable function names (matched on attribute name for
attribute callees), ``expected_var_names`` is the set of acceptable
left-hand identifiers.

NOT recognised (i.e. they are violations):

- ``something = get_logger(...)`` — non-canonical variable name
- ``log = make_logger(...)`` — unknown factory function
- ``log = "string"`` — value is not a call

skip rules:

- empty files (``stat().st_size == 0``) are skipped — placeholder /
  generated-empty modules cannot be expected to declare a logger.
- files whose basename is in ``skip_basenames`` (default
  ``{"__init__.py"}``) are skipped entirely.
- files listed as keys in ``exempt_files`` are skipped before any
  AST work.
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
    "STDLIB_LOG_CALL_KWARGS",
    "find_modules_without_logger",
    "find_structlog_shaped_log_calls",
]


_CATEGORY = "logger_coverage.missing"


def _has_module_level_logger(
    tree: ast.Module,
    logger_factory_names: frozenset[str],
    expected_var_names: frozenset[str],
) -> bool:
    """true iff the module top-level assigns a logger to a known name.

    matches both ``log = get_logger(__name__)`` (Name callee) and
    ``log = observe.get_logger(__name__)`` (Attribute callee). the
    target name must be in ``expected_var_names``; the function name
    (or, for attribute callees, the attribute name) must be in
    ``logger_factory_names``.

    only single-target assignments are considered. tuple / chained
    assignments such as ``a = b = get_logger(...)`` are intentionally
    rejected — the canonical form is the cost of entry, and ambiguous
    shapes are not worth recognising.

    :param tree: parsed AST module
    :ptype tree: ast.Module
    :param logger_factory_names: acceptable factory function names
    :ptype logger_factory_names: frozenset[str]
    :param expected_var_names: acceptable target identifier names
    :ptype expected_var_names: frozenset[str]
    :return: whether a qualifying assignment exists at module scope
    :rtype: bool
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in expected_var_names:
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        fn = value.func
        if isinstance(fn, ast.Name) and fn.id in logger_factory_names:
            return True
        if isinstance(fn, ast.Attribute) and fn.attr in logger_factory_names:
            return True
    return False


def _missing_logger_violation(
    module_path: Path,
    rel: str,
    expected_var_names: frozenset[str],
) -> Violation:
    """build the canonical :class:`Violation` for a silent module.

    the violation is keyed on line 1: the file as a whole is the
    offender, not any specific line within it. the symbol is the
    relative posix path so reports group naturally per module.

    :param module_path: absolute path to the offending module
    :ptype module_path: Path
    :param rel: relative posix path for the offender (used as symbol)
    :ptype rel: str
    :param expected_var_names: accepted target names, surfaced in the
        reason for actionable error messaging
    :ptype expected_var_names: frozenset[str]
    :return: the violation record
    :rtype: Violation
    """
    accepted = " or ".join(f"`{name}`" for name in sorted(expected_var_names))
    return Violation(
        category=_CATEGORY,
        file=module_path,
        line=1,
        symbol=rel,
        reason=(
            f"missing module-level {accepted} = get_logger(__name__). "
            f"add the logger, or add the module to the config's "
            f"`exempt_files` mapping with a one-line rationale if it "
            f"legitimately produces no observable behaviour (re-export "
            f"shim, pure pydantic models, constants table)."
        ),
    )


def find_modules_without_logger(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    logger_factory_names: frozenset[str],
    expected_var_names: frozenset[str],
    skip_basenames: frozenset[str],
) -> list[Violation]:
    """walk every module for the module-level logger contract.

    one :class:`Violation` per non-empty, non-skipped, non-exempt
    module that lacks a qualifying ``<name> = get_logger(...)``
    assignment at module scope. category is ``logger_coverage.missing``
    for every record; ``line`` is always 1 (the file as a whole is
    the offender); ``symbol`` is the file's relative posix path.

    file-level filters, applied in order:

    1. **basename skip** — files whose basename is in
       ``skip_basenames`` are dropped (default
       ``{"__init__.py"}``).
    2. **empty-file skip** — files with zero size on disk are dropped
       (placeholder / generated-empty modules cannot reasonably be
       expected to declare a logger).
    3. **exempt-file skip** — files whose
       :func:`~threetears.enforcement.common.ast_helpers.relative_posix_path`
       relative to ``repo_root`` is a key in ``exempt_files`` are
       dropped.

    surviving files are AST-parsed; any that lacks a module-level
    logger assignment matching the configured factory names + target
    names is flagged.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching and violation reporting
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping;
        listed files are skipped entirely
    :ptype exempt_files: dict[str, str]
    :param logger_factory_names: acceptable factory function names
        (matches both Name and Attribute callees)
    :ptype logger_factory_names: frozenset[str]
    :param expected_var_names: acceptable assignment target names
    :ptype expected_var_names: frozenset[str]
    :param skip_basenames: file basenames the walker skips outright
    :ptype skip_basenames: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            if module_path.name in skip_basenames:
                continue
            try:
                if module_path.stat().st_size == 0:
                    continue
            except OSError as exc:
                note_unscanned(module_path, f"could not stat: {type(exc).__name__}")
                continue
            rel = relative_posix_path(module_path, repo_root)
            if rel in exempt_files:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            if _has_module_level_logger(
                tree,
                logger_factory_names,
                expected_var_names,
            ):
                continue
            violations.append(
                _missing_logger_violation(
                    module_path,
                    rel,
                    expected_var_names,
                )
            )
    return violations


_CALL_KWARGS_CATEGORY = "logger_coverage.call_kwargs"

#: the only keyword arguments ``logging.Logger``'s level methods accept. everything else
#: reaches ``Logger._log`` as an unexpected keyword and raises ``TypeError``.
STDLIB_LOG_CALL_KWARGS: frozenset[str] = frozenset({"exc_info", "extra", "stack_info", "stacklevel"})

#: the ``Logger`` methods that forward ``**kwargs`` to ``Logger._log``.
_LOG_LEVEL_METHODS: frozenset[str] = frozenset(
    {"debug", "info", "warning", "warn", "error", "critical", "exception", "log"}
)


def _is_logger_receiver(node: ast.expr, expected_var_names: frozenset[str]) -> bool:
    """true iff ``node`` is one of the recognised module-logger names.

    Deliberately narrow: only the canonical receivers this domain already requires
    (``log``, ``_logger``, and ``self.log`` / ``self._logger`` for the classes that hold
    one). Matching any identifier containing "log" would sweep in domain loggers -- an
    ``audit_logger.log(event_id=..., ...)`` is a different object with a keyword api of
    its own, and flagging it would be a false positive that gets this walker switched off.

    :param node: the call's receiver expression
    :ptype node: ast.expr
    :param expected_var_names: the module-logger names this domain recognises
    :ptype expected_var_names: frozenset[str]
    :return: whether the receiver is a recognised stdlib-backed logger
    :rtype: bool
    """
    if isinstance(node, ast.Name):
        return node.id in expected_var_names
    result = False
    if isinstance(node, ast.Attribute) and node.attr in expected_var_names:
        result = isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}
    return result


def find_structlog_shaped_log_calls(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    expected_var_names: frozenset[str],
    skip_basenames: frozenset[str],
) -> list[Violation]:
    """flag ``log.info("msg", key=value)`` -- structlog shape on a stdlib logger.

    ``get_logger`` returns a ``logging.Logger`` subclass, not a structlog logger, so an
    arbitrary keyword reaches ``Logger._log`` and raises ``TypeError``. The canonical form
    is ``extra={"extra_data": {...}}``.

    **This is why an AST walker earns its keep here rather than a test.** ``Logger.info``
    guards the ``_log`` call with ``isEnabledFor``, so a call site below the configured
    level never executes and never raises. A suite that leaves logging at WARNING is green
    on code that dies the moment a process calls ``configure_logging(level="INFO")`` --
    which is exactly what production entry points do, and only production entry points do.
    The defect is invisible to every test that does not deliberately raise the level on the
    module under test.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping; listed files are skipped
    :ptype exempt_files: dict[str, str]
    :param expected_var_names: recognised module-logger receiver names
    :ptype expected_var_names: frozenset[str]
    :param skip_basenames: file basenames the walker skips outright
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
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not isinstance(fn, ast.Attribute) or fn.attr not in _LOG_LEVEL_METHODS:
                    continue
                if not _is_logger_receiver(fn.value, expected_var_names):
                    continue
                offenders = sorted(
                    kw.arg for kw in node.keywords if kw.arg is not None and kw.arg not in STDLIB_LOG_CALL_KWARGS
                )
                if not offenders:
                    continue
                violations.append(
                    Violation(
                        category=_CALL_KWARGS_CATEGORY,
                        file=module_path,
                        line=node.lineno,
                        symbol=f"{rel}:{fn.attr}",
                        reason=(
                            f"log.{fn.attr}(...) passes {', '.join(offenders)} -- `get_logger` returns a "
                            f"`logging.Logger`, not a structlog logger, so these reach `Logger._log` as "
                            f"unexpected keywords and raise TypeError. this call is level-gated, so it is "
                            f"silent until a process raises the level to {fn.attr.upper()}. "
                            f'use extra={{"extra_data": {{...}}}} instead.'
                        ),
                    )
                )
    return violations
