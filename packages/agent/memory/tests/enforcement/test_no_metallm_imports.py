"""Enforcement test: no residual parent product import paths in threetears.agent.memory.

Guards against accidental imports from the parent product application codebase
(src.*, api.src.*, api.*) that should have been replaced during extraction.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "memory"

_BANNED_PREFIXES = ["src.", "api.src.", "api."]


def _collect_src_files() -> list[Path]:
    """Collect all Python source files under src/."""
    return sorted(p for p in _SRC_ROOT.rglob("*.py"))


def _is_banned(module_name: str) -> bool:
    """Return True if the import starts with a banned prefix."""
    return any(module_name.startswith(prefix) or module_name == prefix.rstrip(".") for prefix in _BANNED_PREFIXES)


class TestNoParentProductImports:
    """No parent product import paths in agent-memory."""

    def test_no_parent_product_imports(self) -> None:
        """Scan all imports in agent-memory source for parent product paths."""
        violations: list[str] = []
        for path in _collect_src_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _is_banned(alias.name):
                            violations.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and _is_banned(node.module):
                        violations.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: from {node.module} import ...")

        assert not violations, (
            f"Parent product imports found in agent-memory ({len(violations)} location(s)):\n" + "\n".join(violations)
        )
