"""the registry reads the tool server's wire protocol from core, and costs nothing of agent-tools.

the registry parses what a tool pod sends -- registration, call, reply,
heartbeat -- and those envelopes are :mod:`threetears.core.tool_wire`'s. it does
not depend on ``3tears-agent-tools``, whose closure carries langchain,
langgraph and pgvector: an import of agent-tools from registry ``src`` would
fail in an install that resolved only what registry declares, so neither a
runtime import nor a ``TYPE_CHECKING`` one may name it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: the package registry no longer depends on.
_REFUSED_PACKAGE = "threetears.agent.tools"

#: imports the registry with agent-tools and its closure refused, as an install of registry alone
#: refuses them, then checks none was loaded. refusing rather than only looking matters for pgvector:
#: core loads it when the environment has it (the workspace does, for other packages), and works
#: without it, so what this proves is that the registry never needs it.
_PROBE = """
import importlib.abc, sys
_REFUSED_ROOTS = ("langchain", "langchain_core", "langgraph", "pgvector")
def _refused(name):
    return (
        name == "threetears.agent.tools"
        or name.startswith("threetears.agent.tools.")
        or name.split(".")[0] in _REFUSED_ROOTS
    )
class _NotInstalled(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _refused(name):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NotInstalled())
import threetears.registry
loaded = sorted(name for name in sys.modules if _refused(name))
assert not loaded, loaded
print("imported")
"""


def _agent_tools_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """every import in a module that names agent-tools, wherever it sits.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: (line, module) for each offending import
    :rtype: list[tuple[int, str]]
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        for name in names:
            if name == _REFUSED_PACKAGE or name.startswith(f"{_REFUSED_PACKAGE}."):
                found.append((node.lineno, name))
                break
    return found


def test_registry_source_never_imports_agent_tools() -> None:
    violations = [
        f"{path.relative_to(_SRC_ROOT)}:{line} {name}"
        for path in sorted(_SRC_ROOT.rglob("*.py"))
        for line, name in _agent_tools_imports(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    ]
    assert not violations, "registry src imports agent-tools, which it does not depend on:\n" + "\n".join(violations)


def test_the_scan_sees_every_import_form() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "import threetears.agent.tools.server\n"
        "from threetears.agent.tools import ToolServer\n"
        "from threetears.agent import tools\n"
        "if TYPE_CHECKING:\n"
        "    from threetears.agent.tools.server import CallRequest\n"
        "def later():\n"
        "    from threetears.agent.tools.context_envelope import CallContext\n"
        "from threetears.core.tool_wire import CallRequest\n"
        "from threetears.agent.toolsmith import nothing\n"
    )
    assert [line for line, _ in _agent_tools_imports(ast.parse(source))] == [2, 3, 4, 6, 8]


def test_importing_the_registry_loads_no_agent_tools_closure() -> None:
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
