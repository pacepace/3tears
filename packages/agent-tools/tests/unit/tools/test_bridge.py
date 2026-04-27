"""tests for tears_tool_to_langchain bridge function."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.bridge import tears_tool_to_langchain


class DummyTool(TearsTool):
    """minimal TearsTool for bridge testing."""

    def __init__(self, succeed: bool = True) -> None:
        """initialize dummy tool.

        :param succeed: whether execute should return success
        :ptype succeed: bool
        """
        self._succeed = succeed

    async def execute(self, **kwargs: Any) -> ToolResult:
        """return configured result.

        :param kwargs: tool input parameters
        :ptype kwargs: Any
        :return: test result
        :rtype: ToolResult
        """
        if self._succeed:
            result = ToolResult(success=True, content=f"ok: {kwargs}")
            return result
        result = ToolResult(success=False, content="failed", error="something went wrong")
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return test schema.

        :return: test tool definition
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="dummy tool for testing",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "test message",
                    },
                    "count": {
                        "type": "integer",
                        "description": "test count",
                    },
                    "optional_flag": {
                        "type": "boolean",
                        "description": "optional flag",
                    },
                },
                "required": ["message"],
            },
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.dummy"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


class TestTearsToolToLangchain:
    """tests for tears_tool_to_langchain bridge function."""

    def test_returns_base_tool(self) -> None:
        """bridge returns BaseTool instance."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        assert isinstance(wrapped, BaseTool)

    def test_short_name_from_mcp_name(self) -> None:
        """wrapped tool name is short name from mcp_name (after last dot)."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        assert wrapped.name == "dummy"

    def test_description_from_schema(self) -> None:
        """wrapped tool description matches MCP schema description."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        assert wrapped.description == "dummy tool for testing"

    def test_arun_success_returns_content(self) -> None:
        """_arun delegates to execute and returns content on success."""
        tool = DummyTool(succeed=True)
        wrapped = tears_tool_to_langchain(tool)
        # _arun is LangChain BaseTool's documented override hook; see
        # tests/enforcement/_underscore_exemptions.txt for rationale.
        result = asyncio.run(wrapped._arun(message="hello"))  # noqa: SLF001
        assert "ok:" in result
        assert "hello" in result

    def test_arun_error_returns_error_message(self) -> None:
        """_arun returns 'Error: ...' message on failure."""
        tool = DummyTool(succeed=False)
        wrapped = tears_tool_to_langchain(tool)
        # _arun is LangChain BaseTool's documented override hook; see
        # tests/enforcement/_underscore_exemptions.txt for rationale.
        result = asyncio.run(wrapped._arun(message="hello"))  # noqa: SLF001
        assert result.startswith("Error:")
        assert "something went wrong" in result

    def test_args_schema_built_from_input_schema(self) -> None:
        """wrapped tool has args_schema with fields from input_schema."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        schema_cls = wrapped.args_schema
        fields = schema_cls.model_fields
        assert "message" in fields
        assert "count" in fields
        assert "optional_flag" in fields

    def test_required_field_has_no_default(self) -> None:
        """required fields in args_schema have no default value."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        schema_cls = wrapped.args_schema
        message_field = schema_cls.model_fields["message"]
        assert message_field.is_required() is True

    def test_optional_field_has_default_none(self) -> None:
        """optional fields in args_schema default to None."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        schema_cls = wrapped.args_schema
        optional_field = schema_cls.model_fields["optional_flag"]
        assert optional_field.default is None

    def test_run_raises_not_implemented(self) -> None:
        """_run raises NotImplementedError."""
        tool = DummyTool()
        wrapped = tears_tool_to_langchain(tool)
        raised = False
        try:
            # _run is LangChain BaseTool's documented sync-run override hook;
            # see tests/enforcement/_underscore_exemptions.txt for rationale.
            wrapped._run(message="hello")  # noqa: SLF001
        except NotImplementedError:
            raised = True
        assert raised is True

    def test_works_with_any_tears_tool(self) -> None:
        """bridge works generically with any TearsTool subclass."""
        from threetears.agent.tools.builtin.calculator import CalculatorTool

        calc = CalculatorTool()
        wrapped = tears_tool_to_langchain(calc)
        assert isinstance(wrapped, BaseTool)
        assert wrapped.name == "calculator"
        # _arun is LangChain BaseTool's documented override hook; see
        # tests/enforcement/_underscore_exemptions.txt for rationale.
        result = asyncio.run(wrapped._arun(expression="1 + 1"))  # noqa: SLF001
        assert result == "2"

    def test_error_result_without_error_field(self) -> None:
        """_arun falls back to content when error field is None on failure."""

        class FailNoErrorTool(TearsTool):
            """tool that returns failure with no error field."""

            async def execute(self, **kwargs: Any) -> ToolResult:
                """return failure without error.

                :param kwargs: tool input
                :ptype kwargs: Any
                :return: failure result
                :rtype: ToolResult
                """
                result = ToolResult(success=False, content="bad things happened")
                return result

            def mcp_schema(self) -> MCPToolDefinition:
                """return schema.

                :return: tool definition
                :rtype: MCPToolDefinition
                """
                result = MCPToolDefinition(
                    name="threetears.fail_no_error",
                    version="1.0",
                    description="test tool",
                    input_schema={"type": "object", "properties": {}},
                )
                return result

            def mcp_name(self) -> str:
                """return name.

                :return: name
                :rtype: str
                """
                return "threetears.fail_no_error"

            def mcp_version(self) -> str:
                """return version.

                :return: version
                :rtype: str
                """
                return "1.0"

        tool = FailNoErrorTool()
        wrapped = tears_tool_to_langchain(tool)
        # _arun is LangChain BaseTool's documented override hook; see
        # tests/enforcement/_underscore_exemptions.txt for rationale.
        result = asyncio.run(wrapped._arun())  # noqa: SLF001
        assert result == "Error: bad things happened"
