"""Unit converter tool using pint."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error

__all__ = [
    "UnitConverterInput",
    "UnitConverterTool",
    "create_unit_converter_tool",
]

_ureg: Any = None
try:
    import pint

    _ureg = pint.UnitRegistry()
    _HAS_PINT = True
except ImportError:
    _HAS_PINT = False


class UnitConverterInput(BaseModel):
    """Input for the unit converter tool."""

    value: float = Field(description="Numeric value to convert")
    from_unit: str = Field(description="Source unit (e.g. 'miles', 'kg', 'celsius')")
    to_unit: str = Field(description="Target unit (e.g. 'kilometers', 'pounds', 'fahrenheit')")


def _convert(value: float, from_unit: str, to_unit: str) -> str:
    if not _HAS_PINT:
        return tool_error("unit_converter", "convert", "pint package is not installed")
    try:
        quantity = _ureg.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
        return f"{value} {from_unit} = {converted.magnitude:.6g} {to_unit}"
    except pint.errors.UndefinedUnitError as exc:
        return tool_error("unit_converter", "convert", f"unknown unit: {exc}")
    except pint.errors.DimensionalityError as exc:
        return tool_error("unit_converter", "convert", f"incompatible units: {exc}")
    except Exception as exc:
        return tool_error("unit_converter", "convert", str(exc))


def create_unit_converter_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a unit converter tool."""
    return StructuredTool.from_function(
        func=_convert,
        name="unit_converter",
        description=description,
        args_schema=UnitConverterInput,
    )


class UnitConverterTool(TearsTool):
    """TearsTool wrapper for unit conversion via pint.

    converts numeric values between physical units using pint
    UnitRegistry. supports all standard unit types including
    length, mass, temperature, and more.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "source unit (e.g. 'miles', 'kg', 'celsius')",
            },
            "to_unit": {
                "type": "string",
                "description": "target unit (e.g. 'kilometers', 'pounds', 'fahrenheit')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """convert value between units.

        :param kwargs: must include 'value', 'from_unit', 'to_unit' keys
        :ptype kwargs: Any
        :return: result containing converted value or error
        :rtype: ToolResult
        """
        value = kwargs.get("value", 0.0)
        from_unit = kwargs.get("from_unit", "")
        to_unit = kwargs.get("to_unit", "")
        content = _convert(value, from_unit, to_unit)
        success = not content.startswith("[TOOL ERROR]")
        result = ToolResult(
            success=success,
            content=content,
            error=content if not success else None,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for unit converter.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="convert numeric values between physical units",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.unit_converter"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
