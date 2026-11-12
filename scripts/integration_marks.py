"""Which test files mark a test ``integration``: the one reading both of its users share.

``scripts/test-integration.sh`` runs it (``python3 -I scripts/integration_marks.py PATH...``) to learn whether a
local path still holds an integration test once the suite's paths are left out, and
``tests/enforcement/test_integration_suite_selection.py`` loads it to hold the pool's and the laptop's lists to every
integration test in the tree. A test counts as marked when ``pytest.mark.<name>`` appears in a decorator or a
``pytestmark`` assignment, at module or class level: the word in a docstring or a string is no mark.

Standard library only, and no syntax newer than the oldest ``python3`` that runs it: the script may find a system
interpreter older than the tree's. A file that interpreter cannot parse is reported as marked, not dropped: keeping
a path pytest then runs costs a collection, and dropping one would let a test run nowhere.
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


def file_marks(path: Path) -> set[str] | None:
    """The marks ``path`` applies, or None when this interpreter cannot parse it.

    :param path: a test file
    :ptype path: Path
    :return: the mark names, or None
    :rtype: set[str] | None
    """
    try:
        return marks(ast.parse(path.read_text(encoding="utf-8")))
    except SyntaxError:
        # NOSILENT: an unreadable file is answered as one (None), and the caller keeps it rather than drops it
        return None


def integration_files(roots: list[str]) -> Iterator[Path]:
    """Every ``test_*.py`` under ``roots`` (files or directories) that marks a test ``integration``, or can't be read.

    :param roots: the paths
    :ptype roots: list[str]
    :return: the files, in order
    :rtype: Iterator[Path]
    """
    for root in map(Path, roots):
        found = [root] if root.is_file() else sorted(root.glob("**/test_*.py")) if root.is_dir() else []
        for path in found:
            if "__pycache__" in path.parts:
                continue
            applied = file_marks(path)
            if applied is None or "integration" in applied:
                yield path


if __name__ == "__main__":
    for found_path in integration_files(sys.argv[1:]):
        print(found_path)
