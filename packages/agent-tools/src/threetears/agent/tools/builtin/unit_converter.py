"""Unit converter tool using pint."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.utils import tool_error

try:
    import pint

    _ureg = pint.UnitRegistry()
    _HAS_PINT = True
except ImportError:
    _ureg = None  # type: ignore[assignment]
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
