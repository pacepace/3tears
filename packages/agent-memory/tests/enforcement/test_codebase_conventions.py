"""Enforcement tests -- codebase conventions for threetears.agent.memory.

Pure static analysis via AST. Enforces:
1. No bare print() in source code
2. No logging.getLogger() -- must use threetears.core.logging.get_logger
3. from __future__ import annotations in every module
4. Type annotations on public function return types
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "memory"


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


class TestLoggingConsistency:
    """Ensure all source files use the project logger, never print() or stdlib getLogger."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_print_statements(self, src_file: Path) -> None:
        """No bare print() calls in production source code."""
        tree = _parse_file(src_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    pytest.fail(
                        f"{src_file.relative_to(_SRC_ROOT)}:{node.lineno} -- "
                        f"bare print() found. Use get_logger(__name__) instead."
                    )

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_stdlib_get_logger(self, src_file: Path) -> None:
        """No direct logging.getLogger() -- use threetears.core.logging.get_logger."""
        tree = _parse_file(src_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "getLogger"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                ):
                    pytest.fail(
                        f"{src_file.relative_to(_SRC_ROOT)}:{node.lineno} -- "
                        f"logging.getLogger() found. Use get_logger(__name__) from "
                        f"threetears.core.logging instead."
                    )

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_uses_project_logger(self, src_file: Path) -> None:
        """Files that log must import get_logger from threetears.core.logging."""
        tree = _parse_file(src_file)

        uses_logger = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "_logger":
                uses_logger = True
                break

        if not uses_logger:
            return

        imports_get_logger = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "logging" in node.module:
                    for alias in node.names:
                        if alias.name == "get_logger":
                            imports_get_logger = True
                            break

        assert imports_get_logger, (
            f"{src_file.relative_to(_SRC_ROOT)} uses _logger but does not "
            f"import get_logger from the project logging module."
        )


class TestFutureAnnotations:
    """Every source file must have `from __future__ import annotations`."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_future_annotations_present(self, src_file: Path) -> None:
        """All source files must have from __future__ import annotations."""
        tree = _parse_file(src_file)
        has_future = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    for alias in node.names:
                        if alias.name == "annotations":
                            has_future = True
                            break
        assert has_future, (
            f"{src_file.relative_to(_SRC_ROOT)} is missing "
            f"`from __future__ import annotations`"
        )


class TestTypeAnnotations:
    """Public functions must have return type annotations."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_functions_have_return_annotations(self, src_file: Path) -> None:
        """All public functions must have return type annotations."""
        tree = _parse_file(src_file)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                if node.name.startswith("test_"):
                    continue
                if node.name.startswith("_"):
                    continue
                if node.returns is None:
                    violations.append(
                        f"  line {node.lineno}: {node.name}() missing -> return type"
                    )
        if violations:
            rel = src_file.relative_to(_SRC_ROOT)
            detail = "\n".join(violations)
            pytest.fail(f"{rel} has functions without return annotations:\n{detail}")
