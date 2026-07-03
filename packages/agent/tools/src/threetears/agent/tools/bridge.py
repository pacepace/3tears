"""bridge between TearsTool and LangChain BaseTool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from threetears.agent.tools.base_tool import TearsTool

__all__ = [
    "tears_tool_to_langchain",
]


def tears_tool_to_langchain(tool: TearsTool) -> BaseTool:
    """wrap TearsTool instance as LangChain BaseTool.

    creates dynamic BaseTool subclass with name, description,
    and args_schema from TearsTool's MCP definition.
    _arun delegates to tool.run().

    :param tool: TearsTool instance to wrap
    :ptype tool: TearsTool
    :return: LangChain-compatible tool
    :rtype: BaseTool
    """
    schema = tool.mcp_schema()
    # Use the full canonical name (e.g. ``threetears.calculator``)
    # rather than the short last-segment form. Earlier this code
    # extracted ``short_name = schema.name.split(".")[-1]`` so a
    # canonical ``threetears.calculator`` surfaced as ``calculator``;
    # that short form was incompatible with consumers that match by
    # the canonical name (a consumer's ``_execute_service_tool``,
    # ``compute_expected_tools`` glob matchers,
    # registry catalog full_name -> tool entry lookups). Provider-side
    # wire constraints (e.g. Bedrock's tool-name validator that
    # rejects dots) are handled at the chat-model boundary by the
    # OpenRouter factory's name-translating wrapper, not by
    # mangling the canonical name here.
    canonical_name = schema.name

    # Build Pydantic args_schema from JSON Schema. Pydantic's
    # ``create_model`` rejects dots in the model name, so the
    # args-model name is sanitised separately -- this is internal-only
    # and never appears on the LangChain ``BaseTool.name`` surface.
    args_model_name = canonical_name.replace(".", "_")
    args_model = _build_args_model(args_model_name, schema.input_schema)

    class WrappedTool(BaseTool):
        name: str = canonical_name
        description: str = schema.description
        args_schema: type = args_model

        async def _arun(self, **kwargs: Any) -> str:
            """execute wrapped TearsTool asynchronously.

            :param kwargs: tool input parameters
            :ptype kwargs: Any
            :return: result content or error message
            :rtype: str
            """
            result = await tool.run(**kwargs)
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

    result: type[BaseModel] = create_model(f"{name}_args", **fields)
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
