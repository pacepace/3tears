"""enforcement test: no module under ``threetears.mcp`` writes to stdout / stderr.

stdio MCP servers communicate over stdin/stdout. every byte written
to stdout that isn't a properly-framed MCP message corrupts the
protocol. every byte written to stderr is sent to the client which
may interpret it as protocol noise. logging in stdio-mode servers
MUST go to a file or to NATS -- never stdout / stderr.

this test scans the framework's source for the common offenders:

- ``print(...)`` calls
- ``sys.stdout.write(...)`` / ``sys.stderr.write(...)``
- ``print(..., file=sys.stdout)`` / ``..., file=sys.stderr)``

per-product MCP servers (e.g. product-a-mcp-server.py, product-b-mcp-server.py,
etc.) have their own enforcement test in their respective repos.
this guard is for the shared framework only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2] / "src" / "threetears" / "mcp"


def _iter_py_files(root: Path) -> list[Path]:
    """return every .py file under ``root`` (excluding __pycache__)."""
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _find_offenders(tree: ast.AST) -> list[tuple[int, str]]:
    """walk the AST; report every line that writes to stdio."""
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        # print(...) -- with or without file= kwarg; even file=sys.stdout
        # is forbidden because the function name itself is the smell
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                offenders.append((node.lineno, "print(...)"))
            elif isinstance(func, ast.Attribute) and func.attr == "write":
                # sys.stdout.write / sys.stderr.write
                value = func.value
                if isinstance(value, ast.Attribute) and value.attr in (
                    "stdout",
                    "stderr",
                ):
                    inner = value.value
                    if isinstance(inner, ast.Name) and inner.id == "sys":
                        offenders.append((node.lineno, f"sys.{value.attr}.write(...)"))
    return offenders


class TestNoStdioWrites:
    """every framework module is stdio-clean."""

    def test_no_stdio_writes_in_framework(self) -> None:
        """assert no module under ``threetears.mcp`` writes to stdout / stderr."""
        violations: list[str] = []
        for path in _iter_py_files(_FRAMEWORK_ROOT):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                pytest.fail(f"{path}: SyntaxError {exc}")
            for line, msg in _find_offenders(tree):
                rel = path.relative_to(_FRAMEWORK_ROOT.parents[2])
                violations.append(f"{rel}:{line} -- {msg}")
        assert not violations, (
            "stdio MCP servers cannot tolerate stdout / stderr writes; "
            "log to a file or NATS instead. Offenders:\n  " + "\n  ".join(violations)
        )
