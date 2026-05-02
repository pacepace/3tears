"""ast-walking helpers shared across every enforcement domain.

these helpers exist because every domain walker re-implemented the same
file-iteration / parse / private-name / logger-call recognisers slightly
differently. consolidating them here keeps the walkers focused on the
shape they're actually checking.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "iter_python_files",
    "parse_python_file",
    "relative_posix_path",
    "is_private_name",
    "is_logger_call",
    "is_suppress_call",
]

# directories never walked for source-scanning passes. covers virtual
# environments, build artefacts, IDE caches, vcs metadata, and the
# legacy sphinx ``_build`` output dir. cookiecutter template trees are
# detected separately because their names contain ``{{`` placeholders.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".git",
    "dist",
    "build",
    ".eggs",
    "_build",
})


def _has_cookiecutter_marker(path: Path) -> bool:
    """true iff any path component contains a cookiecutter placeholder.

    cookiecutter directories embed ``{{cookiecutter.x}}`` in their path
    components; the files inside are jinja2 templates rather than valid
    python source.

    :param path: path to classify
    :ptype path: Path
    :return: whether any path component contains ``{{``
    :rtype: bool
    """
    return any("{{" in part for part in path.parts)


def iter_python_files(root: Path) -> Iterable[Path]:
    """yield every ``.py`` file under ``root``, deterministically ordered.

    skips :data:`_SKIP_DIRS` (venv / build / cache / vcs) and any path
    that lives inside a cookiecutter template tree (component contains
    ``{{``). entries within each directory are sorted so report output
    is stable across runs.

    :param root: directory to walk
    :ptype root: Path
    :return: iterator of python files
    :rtype: Iterable[Path]
    """
    if not root.is_dir():
        return
    yield from _walk_sorted(root)


def _walk_sorted(directory: Path) -> Iterable[Path]:
    """recursively yield ``.py`` files in deterministic order.

    :param directory: directory to walk
    :ptype directory: Path
    :return: iterator of python files
    :rtype: Iterable[Path]
    """
    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except (PermissionError, FileNotFoundError):
        return
    for entry in entries:
        if entry.is_dir():
            if entry.name in _SKIP_DIRS:
                continue
            if "{{" in entry.name:
                continue
            yield from _walk_sorted(entry)
        elif entry.is_file() and entry.suffix == ".py":
            if _has_cookiecutter_marker(entry):
                continue
            yield entry


def parse_python_file(path: Path) -> ast.Module | None:
    """parse a python file as utf-8, returning None on parse failure.

    swallows :class:`SyntaxError` and :class:`UnicodeDecodeError` since
    walkers must be robust to template / non-python files that slipped
    past the iterator filters. every other I/O error propagates.

    :param path: file to parse
    :ptype path: Path
    :return: parsed module AST, or None on syntax / unicode error
    :rtype: ast.Module | None
    """
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def relative_posix_path(path: Path, root: Path) -> str:
    """return ``path`` relative to ``root`` with forward-slash separators.

    falls back to the absolute posix string when ``path`` is not under
    ``root`` (so cross-repo paths still render cleanly in reports). the
    forward-slash output is stable across linux / mac / windows hosts,
    which matters because exemption files store paths in the canonical
    posix form.

    :param path: target path
    :ptype path: Path
    :param root: reference root for relativisation
    :ptype root: Path
    :return: forward-slash relative path, or absolute posix path on
        cross-root
    :rtype: str
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.as_posix()
    return rel.as_posix()


def is_private_name(name: str) -> bool:
    """true iff ``name`` is a single-leading-underscore identifier.

    excludes:

    - the throwaway ``_``
    - dunders (``__foo__`` — both starts and ends with double underscore)
    - trailing-underscore identifiers (``class_`` / ``id_`` — the
      escape-keyword convention; also catches single-char variants like
      ``_x_`` since these end with ``_``)

    matches PEP 8 / SLF001 semantics: ``_helper``, ``_foo``,
    ``__name_mangled`` (starts with ``__`` but does not end with
    ``__``) — those are private. note that name-mangled ``__foo`` is
    private under this classifier because the dunder check requires
    both the start and end pattern.

    :param name: identifier to classify
    :ptype name: str
    :return: whether ``name`` is a private (underscore-prefixed)
        identifier
    :rtype: bool
    """
    if name == "_":
        return False
    if not name.startswith("_"):
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    if name.endswith("_"):
        return False
    return True


def is_logger_call(
    node: ast.AST,
    logger_names: frozenset[str],
    method_names: frozenset[str],
) -> bool:
    """true iff ``node`` is a ``logger.<method>(...)`` call expression.

    accepts either of two shapes:

    - ``Expr(Call(Attribute(Name(logger_name), method_name)))`` — a bare
      logger reference such as ``log.debug(...)`` whose receiver is one
      of the known logger names (``log`` / ``logger`` / ``_logger`` /
      etc., as configured per domain).
    - ``Expr(Call(Attribute(_, method_name)))`` — any attribute call
      whose method name is in ``method_names``. this catches indirect
      receivers (``self.log.error(...)`` / ``ctx.logger.warning(...)``)
      where the receiver is not a bare name.

    :param node: ast node to test (typically an ``Expr`` or ``Call``)
    :ptype node: ast.AST
    :param logger_names: known receiver names (``log``, ``logger``, ...)
    :ptype logger_names: frozenset[str]
    :param method_names: known logger method names (``debug``, ``info``,
        ``warning``, ``error``, ``exception``, ``critical``)
    :ptype method_names: frozenset[str]
    :return: whether the node is a logger-method call
    :rtype: bool
    """
    if isinstance(node, ast.Expr):
        node = node.value
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in method_names:
        return False
    receiver = func.value
    if isinstance(receiver, ast.Name) and receiver.id in logger_names:
        return True
    if isinstance(receiver, ast.Attribute):
        return True
    return False


def is_suppress_call(node: ast.AST) -> bool:
    """true iff ``node`` is a ``suppress(...)`` / ``contextlib.suppress(...)`` call.

    used by the no-silent-swallow walker to recognise the canonical
    "silently allow this exception" idiom. recognises both the bare-name
    callee (``suppress(KeyError)``) and the attribute callee
    (``contextlib.suppress(KeyError)``); does not require the import to
    be detected.

    :param node: ast node to test
    :ptype node: ast.AST
    :return: whether the node is a suppress call
    :rtype: bool
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "suppress":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "suppress":
        return True
    return False
