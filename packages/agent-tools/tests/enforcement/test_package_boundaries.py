"""Enforcement test: agent-tools package boundary constraints.

Package dependency DAG:
  core <- agent.memory
  core <- agent.tools
  agent.memory <- agent.tools (ledger used by ToolContextManager)

Agent-tools may import from agent.memory (MemoryLedger), but
agent.memory must not import from agent.tools (checked in
agent-memory's own enforcement tests).
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "tools"

# Only the ledger module is allowed from agent.memory
_ALLOWED_MEMORY_IMPORTS = frozenset(
    {
        "threetears.agent.memory.ledger",
    }
)


def _collect_src_files() -> list[Path]:
    """Collect all Python source files under src/.

    :return: sorted list of source file paths
    :rtype: list[Path]
    """
    return sorted(p for p in _SRC_ROOT.rglob("*.py"))


def _is_banned(module_name: str) -> str | None:
    """Return the banned module if the import is an unauthorized agent.memory import.

    :param module_name: fully qualified module name
    :ptype module_name: str
    :return: banned prefix or None
    :rtype: str | None
    """
    if module_name in _ALLOWED_MEMORY_IMPORTS:
        return None
    if module_name == "threetears.agent.memory" or module_name.startswith("threetears.agent.memory."):
        return module_name
    return None


class TestPackageBoundaries:
    """Agent-tools boundary: only allowed agent.memory imports."""

    def test_only_allowed_memory_imports(self) -> None:
        """Scan all imports in agent-tools source for unauthorized agent.memory imports."""
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
                            violations.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        banned = _is_banned(node.module)
                        if banned:
                            violations.append(
                                f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: from {node.module} import ..."
                            )

        assert not violations, (
            f"Unauthorized agent.memory imports in agent-tools ({len(violations)}):\n"
            + "\n".join(violations)
            + f"\n\nAllowed: {sorted(_ALLOWED_MEMORY_IMPORTS)}"
        )
