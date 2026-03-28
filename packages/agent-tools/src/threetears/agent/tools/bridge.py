"""bridge between TearsTool and LangChain BaseTool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from threetears.agent.tools.base_tool import TearsTool


def tears_tool_to_langchain(tool: TearsTool) -> BaseTool:
    """wrap TearsTool instance as LangChain BaseTool.

    creates dynamic BaseTool subclass with name, description,
    and args_schema from TearsTool's MCP definition.
    _arun delegates to tool.execute().

    :param tool: TearsTool instance to wrap
    :ptype tool: TearsTool
    :return: LangChain-compatible tool
    :rtype: BaseTool
    """
    schema = tool.mcp_schema()
    short_name = schema.name.split(".")[-1]

    # Build Pydantic args_schema from JSON Schema
    args_model = _build_args_model(short_name, schema.input_schema)

    class WrappedTool(BaseTool):
        name: str = short_name
        description: str = schema.description
        args_schema: type = args_model

        async def _arun(self, **kwargs: Any) -> str:
            """execute wrapped TearsTool asynchronously.

            :param kwargs: tool input parameters
            :ptype kwargs: Any
            :return: result content or error message
            :rtype: str
            """
            result = await tool.execute(**kwargs)
            if result.success:
                return result.content
            return f"Error: {result.error or result.content}"

        def _run(self, **kwargs: Any) -> str:
            """synchronous execution not supported.

            :param kwargs: tool input parameters
            :ptype kwargs: Any
            :raises NotImplementedError: always, use async _arun instead
            """
            raise NotImplementedError("Use async _arun")

    wrapped = WrappedTool()
    return wrapped


def _build_args_model(name: str, input_schema: dict[str, Any]) -> type[BaseModel]:
    """build Pydantic model from JSON Schema for tool args.

    :param name: model name prefix
    :ptype name: str
    :param input_schema: JSON Schema definition for tool input
    :ptype input_schema: dict[str, Any]
    :return: dynamically created Pydantic model class
    :rtype: type[BaseModel]
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_def in properties.items():
        prop_type = _json_type_to_python(prop_def.get("type", "string"))
        description = prop_def.get("description", "")
        if prop_name in required:
            fields[prop_name] = (prop_type, Field(description=description))
        else:
            fields[prop_name] = (prop_type | None, Field(default=None, description=description))

    result = create_model(f"{name}_args", **fields)
    return result


def _json_type_to_python(json_type: str) -> type:
    """map JSON Schema type to Python type.

    :param json_type: JSON Schema type string
    :ptype json_type: str
    :return: corresponding Python type
    :rtype: type
    """
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    result = mapping.get(json_type, str)
    return result
