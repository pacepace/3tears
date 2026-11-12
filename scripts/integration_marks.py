"""Which test files mark a test ``integration``: the one reading both of its users share.

``scripts/test-integration.sh`` runs it (``python3 -I scripts/integration_marks.py PATH...``) to learn whether a
local path still holds an integration test once the suite's paths are left out, and (``--process-only PATH...``)
which of the suite's modules run only where the services are child processes, and
``tests/enforcement/test_integration_suite_selection.py`` loads it to hold the pool's and the laptop's lists to every
integration test in the tree. A test counts as marked when ``pytest.mark.<name>`` appears in a decorator or a
``pytestmark`` assignment, at module or class level: the word in a docstring or a string is no mark.

Standard library only, and no syntax newer than Python 3.11: the script runs whatever ``python3`` is on PATH, which
may be older than the tree's, and refuses one older than 3.11 (it reads ``.testament.toml`` with ``tomllib``). A file
that interpreter cannot parse or decode is reported as marked, not dropped: keeping a path pytest then runs costs a
collection, and dropping one would let a test run nowhere.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path


def marks(tree: ast.Module) -> set[str]:
    """The ``pytest.mark.<name>`` names a module's decorators and ``pytestmark`` assignments apply.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: the mark names
    :rtype: set[str]
    """
    applied: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            applied.extend(node.decorator_list)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            applied.append(node.value)
    names: set[str] = set()
    for expr in applied:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Attribute) and sub.value.attr == "mark":
                names.add(sub.attr)
    return names


def _applied(exprs: list[ast.expr]) -> set[str]:
    """The ``pytest.mark.<name>`` names a list of decorators or ``pytestmark`` values applies.

    :param exprs: the expressions
    :ptype exprs: list[ast.expr]
    :return: the mark names
    :rtype: set[str]
    """
    names: set[str] = set()
    for expr in exprs:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Attribute) and sub.value.attr == "mark":
                names.add(sub.attr)
    return names


def _pytestmark(body: list[ast.stmt]) -> list[ast.expr]:
    """The values a module's or class's body assigns to ``pytestmark``.

    :param body: the body
    :ptype body: list[ast.stmt]
    :return: the values
    :rtype: list[ast.expr]
    """
    return [
        node.value
        for node in body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    ]


def marks_by_test(tree: ast.Module) -> dict[str, set[str]]:
    """Each test's marks, its module's and class's included: ``{"test_x": {...}, "TestY::test_z": {...}}``.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: the marks per test
    :rtype: dict[str, set[str]]
    """
    module = _applied(_pytestmark(tree.body))
    found: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found[node.name] = module | _applied(node.decorator_list)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            held = module | _applied(node.decorator_list) | _applied(_pytestmark(node.body))
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test"):
                    found[f"{node.name}::{member.name}"] = held | _applied(member.decorator_list)
    return found


def file_marks(path: Path) -> set[str] | None:
    """The marks ``path`` applies, or None when this interpreter cannot parse it.

    :param path: a test file
    :ptype path: Path
    :return: the mark names, or None
    :rtype: set[str] | None
    """
    try:
        return marks(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, ValueError):
        # NOSILENT: newer syntax, or bytes that aren't UTF-8 (UnicodeDecodeError is a ValueError) or hold a null
        # byte: an unreadable file is answered as one (None), and the caller keeps it rather than drops it
        return None


def marked_files(roots: list[str], mark: str, *, keep_unreadable: bool) -> Iterator[Path]:
    """Every ``test_*.py`` under ``roots`` (files or directories) that marks a test ``mark``.

    :param roots: the paths
    :ptype roots: list[str]
    :param mark: the mark's name
    :ptype mark: str
    :param keep_unreadable: whether a file this interpreter cannot read counts as marked
    :ptype keep_unreadable: bool
    :return: the files, in order
    :rtype: Iterator[Path]
    """
    for root in map(Path, roots):
        found = [root] if root.is_file() else sorted(root.glob("**/test_*.py")) if root.is_dir() else []
        for path in found:
            if "__pycache__" in path.parts:
                continue
            applied = file_marks(path)
            marked = keep_unreadable if applied is None else mark in applied
            if marked:
                yield path


def integration_files(roots: list[str]) -> Iterator[Path]:
    """Every ``test_*.py`` under ``roots`` that marks a test ``integration``, or can't be read (kept, not dropped).

    :param roots: the paths
    :ptype roots: list[str]
    :return: the files, in order
    :rtype: Iterator[Path]
    """
    return marked_files(roots, "integration", keep_unreadable=True)


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        sys.exit(f"integration_marks.py needs Python 3.11 or later, not {sys.version.split()[0]}")
    # `--process-only PATH...`: the files marking a test `process_only`, which run only where the services are
    # child processes (an unreadable one is not listed: leaving a test in the run is the safe side here)
    if sys.argv[1:2] == ["--process-only"]:
        listed = marked_files(sys.argv[2:], "process_only", keep_unreadable=False)
    else:
        listed = integration_files(sys.argv[1:])
    for found_path in listed:
        print(found_path)
