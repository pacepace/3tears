"""tests for TearsTool subclasses on utility built-in tools."""

from __future__ import annotations

import asyncio
import base64

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.builtin.timezone_converter import TimezoneConverterTool
from threetears.agent.tools.builtin.unit_converter import UnitConverterTool
from threetears.agent.tools.builtin.analyze_media import AnalyzeMediaTool
from threetears.agent.tools.document import ParseDocumentTool


# -- Fake MediaStorage for AnalyzeMediaTool --


class _FakeMediaStorage:
    """minimal MediaStorage stub for construction tests."""

    async def get_media(self, media_id):  # noqa: ANN001, ANN201
        return None

    async def download_media(self, media_id):  # noqa: ANN001, ANN201
        return None

    async def get_content(self, media_id, content_type, *, model_name=None):  # noqa: ANN001, ANN201
        return None

    async def store_content(self, media_id, user_id, content_type, content, metadata=None):  # noqa: ANN001, ANN201
        return "fake-id"


# -- TimezoneConverterTool --


class TestTimezoneConverterTool:
    """tests for TimezoneConverterTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """TimezoneConverterTool is instance of TearsTool."""
        tool = TimezoneConverterTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """TimezoneConverterTool mcp_name returns correct string."""
        tool = TimezoneConverterTool()
        assert tool.mcp_name() == "threetears.timezone_converter"

    def test_mcp_version(self) -> None:
        """TimezoneConverterTool mcp_version returns 1.0."""
        tool = TimezoneConverterTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """TimezoneConverterTool mcp_schema returns MCPToolDefinition with correct name and input_schema."""
        tool = TimezoneConverterTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.timezone_converter"
        assert schema.input_schema
        assert "time_str" in schema.input_schema.get("properties", {})
        assert "from_timezone" in schema.input_schema.get("properties", {})
        assert "to_timezone" in schema.input_schema.get("properties", {})

    def test_execute_success(self) -> None:
        """TimezoneConverterTool execute returns successful ToolResult for valid conversion."""
        tool = TimezoneConverterTool()
        result = asyncio.run(tool.execute(
            time_str="2024-01-15 14:30",
            from_timezone="America/New_York",
            to_timezone="Europe/London",
        ))
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None

    def test_execute_error_invalid_timezone(self) -> None:
        """TimezoneConverterTool execute returns failure for invalid timezone."""
        tool = TimezoneConverterTool()
        result = asyncio.run(tool.execute(
            time_str="14:30",
            from_timezone="Invalid/Zone",
            to_timezone="Europe/London",
        ))
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None


# -- UnitConverterTool --


class TestUnitConverterTool:
    """tests for UnitConverterTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """UnitConverterTool is instance of TearsTool."""
        tool = UnitConverterTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """UnitConverterTool mcp_name returns correct string."""
        tool = UnitConverterTool()
        assert tool.mcp_name() == "threetears.unit_converter"

    def test_mcp_version(self) -> None:
        """UnitConverterTool mcp_version returns 1.0."""
        tool = UnitConverterTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """UnitConverterTool mcp_schema returns MCPToolDefinition with correct name and input_schema."""
        tool = UnitConverterTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.unit_converter"
        assert schema.input_schema
        assert "value" in schema.input_schema.get("properties", {})
        assert "from_unit" in schema.input_schema.get("properties", {})
        assert "to_unit" in schema.input_schema.get("properties", {})

    def test_execute_success(self) -> None:
        """UnitConverterTool execute returns successful ToolResult for valid conversion."""
        tool = UnitConverterTool()
        result = asyncio.run(tool.execute(value=1.0, from_unit="miles", to_unit="kilometers"))
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "kilometers" in result.content
        assert result.error is None

    def test_execute_error_invalid_unit(self) -> None:
        """UnitConverterTool execute returns failure for invalid unit."""
        tool = UnitConverterTool()
        result = asyncio.run(tool.execute(value=1.0, from_unit="miles", to_unit="kilograms"))
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None


# -- AnalyzeMediaTool --


class TestAnalyzeMediaTool:
    """tests for AnalyzeMediaTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """AnalyzeMediaTool is instance of TearsTool."""
        tool = AnalyzeMediaTool(storage=_FakeMediaStorage())
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """AnalyzeMediaTool mcp_name returns correct string."""
        tool = AnalyzeMediaTool(storage=_FakeMediaStorage())
        assert tool.mcp_name() == "threetears.analyze_media"

    def test_mcp_version(self) -> None:
        """AnalyzeMediaTool mcp_version returns 1.0."""
        tool = AnalyzeMediaTool(storage=_FakeMediaStorage())
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """AnalyzeMediaTool mcp_schema returns MCPToolDefinition with correct name and input_schema."""
        tool = AnalyzeMediaTool(storage=_FakeMediaStorage())
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.analyze_media"
        assert schema.input_schema
        assert "media_ids" in schema.input_schema.get("properties", {})
        assert "question" in schema.input_schema.get("properties", {})
        assert "analyzer" in schema.input_schema.get("properties", {})

    def test_execute_unknown_analyzer(self) -> None:
        """AnalyzeMediaTool execute returns error for unknown analyzer."""
        tool = AnalyzeMediaTool(storage=_FakeMediaStorage())
        result = asyncio.run(tool.execute(
            media_ids=["00000000-0000-0000-0000-000000000001"],
            question="describe this",
            analyzer="nonexistent",
        ))
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None


# -- ParseDocumentTool --


class TestParseDocumentTool:
    """tests for ParseDocumentTool TearsTool subclass."""

    def test_isinstance_tears_tool(self) -> None:
        """ParseDocumentTool is instance of TearsTool."""
        tool = ParseDocumentTool()
        assert isinstance(tool, TearsTool)

    def test_mcp_name(self) -> None:
        """ParseDocumentTool mcp_name returns correct string."""
        tool = ParseDocumentTool()
        assert tool.mcp_name() == "threetears.parse_document"

    def test_mcp_version(self) -> None:
        """ParseDocumentTool mcp_version returns 1.0."""
        tool = ParseDocumentTool()
        assert tool.mcp_version() == "1.0"

    def test_mcp_schema(self) -> None:
        """ParseDocumentTool mcp_schema returns MCPToolDefinition with correct name and input_schema."""
        tool = ParseDocumentTool()
        schema = tool.mcp_schema()
        assert isinstance(schema, MCPToolDefinition)
        assert schema.name == "threetears.parse_document"
        assert schema.input_schema
        assert "content_base64" in schema.input_schema.get("properties", {})
        assert "filename" in schema.input_schema.get("properties", {})

    def test_execute_success_txt(self) -> None:
        """ParseDocumentTool execute returns successful ToolResult for valid text file."""
        tool = ParseDocumentTool()
        content = base64.b64encode(b"Hello world document content").decode()
        result = asyncio.run(tool.execute(content_base64=content, filename="test.txt"))
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "Hello world" in result.content
        assert result.error is None

    def test_execute_error_bad_base64(self) -> None:
        """ParseDocumentTool execute returns failure for invalid base64."""
        tool = ParseDocumentTool()
        result = asyncio.run(tool.execute(content_base64="!!!not-base64!!!", filename="test.txt"))
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
