"""the TearsTool contract on its own: a tool runs through it with no platform module loaded."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from threetears.tool_contract import (
    TOOL_RESULT_ERROR_CODES,
    MCPToolDefinition,
    TearsTool,
    ToolResult,
    normalize_kwargs,
)


class _Echo(TearsTool):
    def mcp_name(self) -> str:
        return "example.echo"

    def mcp_version(self) -> str:
        return "1.0.0"

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="echo the arguments",
            input_schema={"type": "object", "properties": {"items": {"type": "array"}, "text": {"type": "string"}}},
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content=repr(sorted(kwargs.items())))


@pytest.mark.asyncio
async def test_run_coerces_loose_values_then_calls_execute() -> None:
    result = await _Echo().run(items='["a"]', text="t")
    assert result.content == repr([("items", ["a"]), ("text", "t")])


def test_a_subclass_may_not_override_run() -> None:
    with pytest.raises(TypeError, match="overrides TearsTool.run"):

        class _Bad(_Echo):
            async def run(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
                return ToolResult(success=True, content="")


def test_a_result_names_only_a_declared_code_and_only_on_a_failure() -> None:
    assert ToolResult(success=False, content="", error_code="CONFLICT").error_code in TOOL_RESULT_ERROR_CODES
    with pytest.raises(ValueError, match="declared spelling is 'CONFLICT'"):
        ToolResult(success=False, content="", error_code="conflict")
    with pytest.raises(ValueError, match="names a refusal"):
        ToolResult(success=True, content="", error_code="CONFLICT")


def test_the_coercion_reads_the_declared_type() -> None:
    assert normalize_kwargs({"o": ""}, {"properties": {"o": {"type": "object"}}}) == {"o": {}}


#: refuses the platform's modules, as an install of the contract alone does not have them
_BLOCK_PLATFORM = """
import importlib.abc, sys
_REFUSED = ("threetears.core", "threetears.nats", "threetears.agent", "nats", "asyncpg", "sqlalchemy",
            "langchain_core", "langgraph", "mcp")
class _NoPlatform(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == refused or name.startswith(refused + ".") for refused in _REFUSED):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoPlatform())
import asyncio
from threetears.tool_contract import MCPToolDefinition, TearsTool, ToolResult

class Echo(TearsTool):
    def mcp_name(self): return "example.echo"
    def mcp_version(self): return "1.0.0"
    def mcp_schema(self):
        return MCPToolDefinition(name="example.echo", version="1.0.0", description="d",
                                 input_schema={"type": "object", "properties": {"text": {"type": "string"}}})
    async def execute(self, **kwargs): return ToolResult(success=True, content=kwargs["text"])

assert asyncio.run(Echo().run(text="hi")).content == "hi"
print("ran")
"""


def test_a_tool_runs_with_no_platform_module_importable() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _BLOCK_PLATFORM], capture_output=True, text=True, check=False, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ran"
