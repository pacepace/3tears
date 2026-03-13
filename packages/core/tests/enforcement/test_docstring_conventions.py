"""Enforcement tests -- docstring conventions for threetears.core.

Pure static analysis via AST + regex. Enforces:
1. Ban :type X: directive (must use :ptype X:)
2. Public functions must have docstrings
3. Every :param X: must have a matching :ptype X:
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "core"


def _collect_src_files() -> list[Path]:
    """Collect all Python source files under src/, excluding __init__.py."""
    return sorted(
        p for p in _SRC_ROOT.rglob("*.py")
        if p.name != "__init__.py"
    )


def _parse_file(path: Path) -> ast.Module:
    """Parse a Python file into an AST module."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


_SRC_FILES = _collect_src_files()
_SRC_IDS = [str(p.relative_to(_SRC_ROOT)) for p in _SRC_FILES]

_PARAM_RE = re.compile(r":param\s+(\w+)\s*:")
_PTYPE_RE = re.compile(r":ptype\s+(\w+)\s*:")
_TYPE_RE = re.compile(r":type\s+(\w+)\s*:")


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function is decorated with @overload."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _is_public_function(node: ast.AST) -> bool:
    """Return True if node is a public function (not private, not dunder, not test)."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    name = node.name
    if name.startswith("_"):
        return False
    if name.startswith("test_"):
        return False
    return True


class TestNoTypeDirective:
    """Docstrings must use :ptype X: not :type X: for parameter types."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_type_directive(self, src_file: Path) -> None:
        """No :type param: directives in docstrings."""
        tree = _parse_file(src_file)
        violations: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Module)):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                continue
            for match in _TYPE_RE.finditer(docstring):
                lineno = getattr(node, "lineno", 0)
                violations.append(
                    f"  line ~{lineno}: :type {match.group(1)}: found "
                    f"(use :ptype {match.group(1)}: instead)"
                )

        if violations:
            rel = src_file.relative_to(_SRC_ROOT)
            detail = "\n".join(violations)
            pytest.fail(f"{rel} uses banned :type: directive:\n{detail}")


class TestDocstringsRequired:
    """All public functions/methods must have a docstring."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_public_functions_have_docstrings(self, src_file: Path) -> None:
        """Every public function/method must have a docstring."""
        tree = _parse_file(src_file)
        violations = self._check_node_children(tree, src_file)

        if violations:
            rel = src_file.relative_to(_SRC_ROOT)
            detail = "\n".join(violations)
            pytest.fail(
                f"{rel} has public functions without docstrings:\n{detail}"
            )

    def _check_node_children(
        self,
        parent: ast.AST,
        src_file: Path,
    ) -> list[str]:
        """Check direct children of module/class for missing docstrings.

        :param parent: The parent AST node (module or class).
        :ptype parent: ast.AST
        :param src_file: Path to the source file being checked.
        :ptype src_file: Path
        :rtype: list[str]
        """
        violations: list[str] = []
        for node in ast.iter_child_nodes(parent):
            if isinstance(node, ast.ClassDef):
                violations.extend(self._check_node_children(node, src_file))
                continue

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if not _is_public_function(node):
                continue
            if _is_overload(node):
                continue

            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                violations.append(
                    f"  line {node.lineno}: {node.name}() missing docstring"
                )

        return violations


class TestParamPtypePairing:
    """:param X: must always be paired with :ptype X: in the same docstring."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_param_has_matching_ptype(self, src_file: Path) -> None:
        """Every :param X: must have a matching :ptype X:."""
        tree = _parse_file(src_file)
        violations: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Module)):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if not docstring:
                continue

            params = set(_PARAM_RE.findall(docstring))
            ptypes = set(_PTYPE_RE.findall(docstring))
            missing = params - ptypes

            if missing:
                lineno = getattr(node, "lineno", 0)
                func_name = getattr(node, "name", "<module>")
                for param in sorted(missing):
                    violations.append(
                        f"  line ~{lineno} {func_name}(): "
                        f":param {param}: without matching :ptype {param}:"
                    )

        if violations:
            rel = src_file.relative_to(_SRC_ROOT)
            detail = "\n".join(violations)
            pytest.fail(
                f"{rel} has :param: without matching :ptype::\n{detail}"
            )
