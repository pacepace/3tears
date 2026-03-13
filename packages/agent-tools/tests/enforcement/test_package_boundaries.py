"""Enforcement test: agent-tools must not import from threetears.agent.memory.

Package dependency DAG:
  core <- agent.memory
  core <- agent.tools
  agent.memory X agent.tools (no cross-imports)
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "tools"

_BANNED_IMPORTS = ["threetears.agent.memory"]


def _collect_src_files() -> list[Path]:
    """Collect all Python source files under src/."""
    return sorted(p for p in _SRC_ROOT.rglob("*.py"))


def _is_banned(module_name: str) -> str | None:
    """Return the banned module prefix if the import matches, else None."""
    for banned in _BANNED_IMPORTS:
        if module_name == banned or module_name.startswith(banned + "."):
            return banned
    return None


class TestPackageBoundaries:
    """Agent-tools must not import from agent-memory."""

    def test_no_agent_memory_imports(self) -> None:
        """Scan all imports in agent-tools source for banned cross-package imports."""
        violations: list[str] = []
        for path in _collect_src_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        banned = _is_banned(alias.name)
                        if banned:
                            violations.append(
                                f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: import {alias.name} (banned: {banned})"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        banned = _is_banned(node.module)
                        if banned:
                            violations.append(
                                f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: "
                                f"from {node.module} import ... (banned: {banned})"
                            )

        assert not violations, (
            f"Cross-package imports found in agent-tools ({len(violations)} location(s)):\n"
            + "\n".join(violations)
            + "\n\nagent-tools must not import from threetears.agent.memory"
        )
