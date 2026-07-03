"""tests for TearsTool ABC, ToolResult, and MCPToolDefinition."""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)


class DummyTool(TearsTool):
    """concrete test subclass of TearsTool."""

    async def execute(self, **kwargs):  # type: ignore[override]
        """execute dummy tool.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: success result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content="done")

    def mcp_schema(self) -> MCPToolDefinition:
        """return dummy MCP schema.

        :return: test tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name="test.dummy",
            version="1.0.0",
            description="test",
            input_schema={},
        )

    def mcp_name(self) -> str:
        """return dummy tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "test.dummy"

    def mcp_version(self) -> str:
        """return dummy tool version.

        :return: version string
        :rtype: str
        """
        return "1.0.0"


# -- ToolResult tests --


class TestToolResult:
    """tests for ToolResult dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """ToolResult can be created with all fields specified."""
        result = ToolResult(
            success=True,
            content="output text",
            metadata={"key": "value"},
            error="some error",
        )
        assert result.success is True
        assert result.content == "output text"
        assert result.metadata == {"key": "value"}
        assert result.error == "some error"

    def test_defaults(self) -> None:
        """ToolResult defaults metadata to None and error to None."""
        result = ToolResult(success=False, content="fail")
        assert result.metadata is None
        assert result.error is None


# -- MCPToolDefinition tests --


class TestMCPToolDefinition:
    """tests for MCPToolDefinition dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """MCPToolDefinition can be created with all fields specified."""
        defn = MCPToolDefinition(
            name="pkg.tool",
            version="2.1.0",
            description="does something",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
        )
        assert defn.name == "pkg.tool"
        assert defn.version == "2.1.0"
        assert defn.description == "does something"
        assert defn.input_schema == {"type": "object", "properties": {}}
        assert defn.output_schema == {"type": "string"}

    def test_output_schema_defaults_to_none(self) -> None:
        """MCPToolDefinition output_schema defaults to None."""
        defn = MCPToolDefinition(
            name="pkg.tool",
            version="1.0.0",
            description="test",
            input_schema={},
        )
        assert defn.output_schema is None


# -- TearsTool ABC tests --


class TestTearsTool:
    """tests for TearsTool abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """TearsTool raises TypeError when instantiated directly."""
        with pytest.raises(TypeError):
            TearsTool()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        """concrete subclass of TearsTool can be instantiated."""
        tool = DummyTool()
        assert isinstance(tool, TearsTool)

    def test_run_and_execute_are_async(self) -> None:
        """both the public run template and the execute hook must be async."""
        tool = DummyTool()
        assert inspect.iscoroutinefunction(tool.run)
        assert inspect.iscoroutinefunction(tool.execute)

    def test_run_returns_tool_result(self) -> None:
        """public run template returns ToolResult instance."""
        tool = DummyTool()
        result = asyncio.run(tool.run())
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.content == "done"

    def test_mcp_schema_returns_mcp_tool_definition(self) -> None:
        """mcp_schema returns MCPToolDefinition instance."""
        tool = DummyTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "test.dummy"
        assert schema.version == "1.0.0"

    def test_mcp_name_returns_string(self) -> None:
        """mcp_name returns string."""
        tool = DummyTool()
        name = tool.mcp_name()
        assert isinstance(name, str)
        assert name == "test.dummy"

    def test_mcp_version_returns_string(self) -> None:
        """mcp_version returns string."""
        tool = DummyTool()
        version = tool.mcp_version()
        assert isinstance(version, str)
        assert version == "1.0.0"


# -- AST enforcement: no platform imports --


class TestNoPlatformImports:
    """enforcement test: base_tool.py must not import platform dependencies."""

    def test_no_platform_imports_in_base_tool(self) -> None:
        """base_tool.py must not import NATS, LangChain, or platform modules."""
        source_path = Path(__file__).resolve().parents[3] / "src" / "threetears" / "agent" / "tools" / "base_tool.py"
        source = source_path.read_text()
        tree = ast.parse(source)

        banned_prefixes = (
            "nats",
            "langchain",
            "3tears",
            "fastapi",
            "uvicorn",
            "sqlalchemy",
        )

        violations: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(banned_prefixes):
                        violations.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(banned_prefixes):
                    violations.append(node.module)

        assert violations == [], f"banned imports found in base_tool.py: {violations}"
