"""tests for TearsTool subclasses on core built-in tools."""

from __future__ import annotations

import asyncio

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.builtin.calculator import CalculatorTool
from threetears.agent.tools.builtin.current_date import CurrentDateTool
from threetears.agent.tools.builtin.dictionary import DictionaryTool
from threetears.agent.tools.builtin.web_fetch import WebFetchTool
from threetears.agent.tools.builtin.web_search import WebSearchTool


# -- CalculatorTool --


class TestCalculatorTool:
    """tests for CalculatorTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """CalculatorTool is instance of TearsTool."""
        tool = CalculatorTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """CalculatorTool mcp_name returns correct string."""
        tool = CalculatorTool()
        assert tool.mcp_name() == "threetears.calculator"

    def test_mcp_version(self) -> None:
        """CalculatorTool mcp_version returns 1.0."""
        tool = CalculatorTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """CalculatorTool mcp_schema returns MCPToolDefinition with correct name and non-empty input_schema."""
        tool = CalculatorTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.calculator"
        assert schema.input_schema
        assert "expression" in schema.input_schema.get("properties", {})

    def test_execute_success(self) -> None:
        """CalculatorTool execute returns successful ToolResult for valid expression."""
        tool = CalculatorTool()
        result = asyncio.run(tool.execute(expression="2 + 3"))
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.content == "5"
        assert result.error is None

    def test_execute_math_function(self) -> None:
        """CalculatorTool execute handles math functions like sqrt."""
        tool = CalculatorTool()
        result = asyncio.run(tool.execute(expression="sqrt(16)"))
        assert result.success is True
        assert result.content == "4"

    def test_execute_error(self) -> None:
        """CalculatorTool execute returns failure ToolResult for invalid expression."""
        tool = CalculatorTool()
        result = asyncio.run(tool.execute(expression="invalid+++"))
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None


# -- CurrentDateTool --


class TestCurrentDateTool:
    """tests for CurrentDateTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """CurrentDateTool is instance of TearsTool."""
        tool = CurrentDateTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """CurrentDateTool mcp_name returns correct string."""
        tool = CurrentDateTool()
        assert tool.mcp_name() == "threetears.current_date"

    def test_mcp_version(self) -> None:
        """CurrentDateTool mcp_version returns 1.0."""
        tool = CurrentDateTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """CurrentDateTool mcp_schema returns MCPToolDefinition with correct name and non-empty input_schema."""
        tool = CurrentDateTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.current_date"
        assert schema.input_schema is not None

    def test_execute_success(self) -> None:
        """CurrentDateTool execute returns successful ToolResult with UTC date."""
        tool = CurrentDateTool()
        result = asyncio.run(tool.execute())
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "UTC" in result.content
        assert result.error is None

    def test_execute_with_timezone(self) -> None:
        """CurrentDateTool execute includes local timezone when configured."""
        tool = CurrentDateTool(timezone="US/Eastern")
        result = asyncio.run(tool.execute())
        assert result.success is True
        assert "US/Eastern" in result.content


# -- DictionaryTool --


class TestDictionaryTool:
    """tests for DictionaryTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """DictionaryTool is instance of TearsTool."""
        tool = DictionaryTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """DictionaryTool mcp_name returns correct string."""
        tool = DictionaryTool()
        assert tool.mcp_name() == "threetears.dictionary"

    def test_mcp_version(self) -> None:
        """DictionaryTool mcp_version returns 1.0."""
        tool = DictionaryTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """DictionaryTool mcp_schema returns MCPToolDefinition with correct name and non-empty input_schema."""
        tool = DictionaryTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.dictionary"
        assert schema.input_schema
        assert "word" in schema.input_schema.get("properties", {})


# -- WebSearchTool --


class TestWebSearchTool:
    """tests for WebSearchTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """WebSearchTool is instance of TearsTool."""
        tool = WebSearchTool(base_url="http://localhost:8888")
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """WebSearchTool mcp_name returns correct string."""
        tool = WebSearchTool(base_url="http://localhost:8888")
        assert tool.mcp_name() == "threetears.web_search"

    def test_mcp_version(self) -> None:
        """WebSearchTool mcp_version returns 1.0."""
        tool = WebSearchTool(base_url="http://localhost:8888")
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """WebSearchTool mcp_schema returns MCPToolDefinition with correct name and non-empty input_schema."""
        tool = WebSearchTool(base_url="http://localhost:8888")
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.web_search"
        assert schema.input_schema
        assert "query" in schema.input_schema.get("properties", {})

    def test_trailing_slash_stripped(self) -> None:
        """WebSearchTool strips trailing slash from base_url."""
        tool = WebSearchTool(base_url="http://localhost:8888/")
        assert tool._base_url == "http://localhost:8888"


# -- WebFetchTool --


class TestWebFetchTool:
    """tests for WebFetchTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """WebFetchTool is instance of TearsTool."""
        tool = WebFetchTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """WebFetchTool mcp_name returns correct string."""
        tool = WebFetchTool()
        assert tool.mcp_name() == "threetears.web_fetch"

    def test_mcp_version(self) -> None:
        """WebFetchTool mcp_version returns 1.0."""
        tool = WebFetchTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """WebFetchTool mcp_schema returns MCPToolDefinition with correct name and non-empty input_schema."""
        tool = WebFetchTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.web_fetch"
        assert schema.input_schema
        assert "url" in schema.input_schema.get("properties", {})
