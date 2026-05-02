"""walker for the no-stdlib-logging enforcement domain.

the single walker, :func:`find_stdlib_logging_imports`, scans every
module under the configured src trees for any direct stdlib
``logging`` import. matched shapes:

- ``import logging`` — flagged with symbol ``"*"``.
- ``import logging.handlers`` (or any other submodule) — flagged with
  the full dotted name as the symbol.
- ``from logging import getLogger`` — one violation per imported name,
  symbol equals the imported name.
- ``from logging.handlers import RotatingFileHandler`` — flagged the
  same way (``logging.handlers`` is part of the stdlib logging
  package).

intentionally NOT flagged:

- ``import some_logging_lib`` — third-party packages whose name
  merely contains the substring ``logging`` are not stdlib logging.
- ``from my_logging import x`` — the ``logging`` test is on the
  package root, not on substring overlap.

two escape hatches:

- ``exempt_files``: a file-relative-posix-path -> rationale dict in
  the config. a listed file is skipped entirely.
- per-line marker (default ``# stdlib-logging: ok``): the marker
  substring must appear on the offending import's source line. it
  scopes to that line — adjacent imports without the marker are
  still flagged.
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
    "find_stdlib_logging_imports",
    "is_stdlib_logging_module",
]


_CATEGORY = "no_stdlib_logging.import"


def is_stdlib_logging_module(name: str | None) -> bool:
    """true iff ``name`` refers to the stdlib ``logging`` package.

    matches ``"logging"`` itself and any dotted submodule
    (``"logging.handlers"``, ``"logging.config"``). does not match
    third-party packages whose name only contains the substring
    ``logging`` (``"my_logging"``, ``"loggingx"``).

    :param name: module name as written in the import statement
    :ptype name: str | None
    :return: whether ``name`` resolves to a stdlib logging module
    :rtype: bool
    """
    if name is None:
        return False
    if name == "logging":
        return True
    return name.startswith("logging.")


def _import_violation(
    module_path: Path,
    lineno: int,
    symbol: str,
    description: str,
    line_marker: str,
) -> Violation:
    """build the canonical :class:`Violation` for an offending import."""
    return Violation(
        category=_CATEGORY,
        file=module_path,
        line=lineno,
        symbol=symbol,
        reason=(
            f"{description}; use threetears.observe.get_logger "
            f"instead. if configuring a third-party logger by "
            f"name, add '{line_marker}' on the import line with a "
            f"justification, or add the module to the config's "
            f"`exempt_files` mapping."
        ),
    )


def _line_has_marker(
    source_lines: list[str], lineno: int, line_marker: str,
) -> bool:
    """true iff ``source_lines[lineno-1]`` carries the per-line marker.

    ``lineno`` is 1-indexed (matches python AST convention). the
    marker is matched as a case-sensitive substring after stripping
    trailing whitespace from the line.

    :param source_lines: file contents split by lines (no newlines)
    :ptype source_lines: list[str]
    :param lineno: 1-indexed line number of the offending import
    :ptype lineno: int
    :param line_marker: the marker substring (case-sensitive)
    :ptype line_marker: str
    :return: whether the marker is present on the offending line
    :rtype: bool
    """
    idx = lineno - 1
    if idx < 0 or idx >= len(source_lines):
        return False
    return line_marker in source_lines[idx].rstrip()


def find_stdlib_logging_imports(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    exempt_files: dict[str, str],
    line_marker: str,
) -> list[Violation]:
    """walk every module for direct stdlib ``logging`` imports.

    one :class:`Violation` per offending import line that is neither
    in ``exempt_files`` nor carries the per-line marker. category is
    ``no_stdlib_logging.import`` for every record.

    symbol resolution:

    - ``import logging`` -> ``"*"``
    - ``import logging.handlers`` -> ``"logging.handlers"``
    - ``import logging as stdlog`` -> ``"*"`` (alias does not change
      what was imported)
    - ``from logging import getLogger`` -> ``"getLogger"`` (one
      violation per imported name)
    - ``from logging.handlers import RotatingFileHandler`` ->
      ``"RotatingFileHandler"``
    - ``from logging import *`` -> symbol ``"*"`` (the literal
      ``Import * from logging`` form)

    file-level exemption: a file whose
    :func:`~threetears.enforcement.common.ast_helpers.relative_posix_path`
    relative to ``repo_root`` is a key in ``exempt_files`` is skipped
    before any AST work is done. line-level exemption: the marker
    substring on the import's source line skips that single line.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root used to render relative paths for
        exemption matching
    :ptype repo_root: Path
    :param exempt_files: relative-posix-path -> rationale mapping;
        listed files are skipped entirely
    :ptype exempt_files: dict[str, str]
    :param line_marker: per-line marker substring (default
        ``# stdlib-logging: ok``)
    :ptype line_marker: str
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
                if isinstance(node, ast.Import):
                    _scan_import(
                        node,
                        module_path,
                        source_lines,
                        line_marker,
                        violations,
                    )
                elif isinstance(node, ast.ImportFrom):
                    _scan_import_from(
                        node,
                        module_path,
                        source_lines,
                        line_marker,
                        violations,
                    )
    return violations


def _scan_import(
    node: ast.Import,
    module_path: Path,
    source_lines: list[str],
    line_marker: str,
    violations: list[Violation],
) -> None:
    """flag every stdlib-logging alias in an ``import a, b, ...`` node.

    each alias is checked independently because a single ``import``
    statement may import multiple modules on the same line. when the
    line carries the per-line marker the entire statement is
    exempted (the marker is a line-scoped pragma).

    :param node: ast.Import node
    :ptype node: ast.Import
    :param module_path: file containing the import
    :ptype module_path: Path
    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param line_marker: per-line marker substring
    :ptype line_marker: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    for alias in node.names:
        if not is_stdlib_logging_module(alias.name):
            continue
        if _line_has_marker(source_lines, node.lineno, line_marker):
            continue
        if alias.name == "logging":
            symbol = "*"
            description = "import logging"
        else:
            symbol = alias.name
            description = f"import {alias.name}"
        violations.append(
            _import_violation(
                module_path, node.lineno, symbol, description, line_marker,
            )
        )


def _scan_import_from(
    node: ast.ImportFrom,
    module_path: Path,
    source_lines: list[str],
    line_marker: str,
    violations: list[Violation],
) -> None:
    """flag a ``from logging[.x] import ...`` node.

    one violation per imported name (``from logging import a, b``
    yields two violations). ``from logging import *`` yields one
    violation with symbol ``"*"``. relative imports (``from . import
    x``) and non-logging modules are ignored.

    :param node: ast.ImportFrom node
    :ptype node: ast.ImportFrom
    :param module_path: file containing the import
    :ptype module_path: Path
    :param source_lines: file contents split by lines
    :ptype source_lines: list[str]
    :param line_marker: per-line marker substring
    :ptype line_marker: str
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    if node.level and node.level > 0:
        return
    if not is_stdlib_logging_module(node.module):
        return
    if _line_has_marker(source_lines, node.lineno, line_marker):
        return
    module_name = node.module or "logging"
    for alias in node.names:
        symbol = alias.name
        description = f"from {module_name} import {alias.name}"
        violations.append(
            _import_violation(
                module_path, node.lineno, symbol, description, line_marker,
            )
        )
