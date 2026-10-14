"""Unit converter tool using pint.

People write units as they say them: "Miles", "Celsius", "F", "5 feet 10
inches". A capital letter made pint refuse "Miles", and a bare "F" to "C" was
read as farad to coulomb. Here a unit is read as written and then in lower case;
F, C and K mean degrees when the other side is a temperature or another of
them, while "F" to "uF" stays farads; a quantity of several parts is added up.
Currency is said to be out of reach, since rates change daily. The result says
it is rounded, and which of the US or UK measures a word like "gallon" means.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error, tool_no_change

__all__ = [
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

_TOOL = "unit_converter"

#: Significant figures a result is shown to, said in the result.
_SIGNIFICANT = 6

#: Bare letters that are a temperature scale when the other side is a temperature.
_BARE_TEMPERATURE = {"F": "degF", "C": "degC", "K": "kelvin"}

#: What a bare letter was read as, said in the result either way.
_BARE_READ = {
    "degF": "degrees Fahrenheit",
    "degC": "degrees Celsius",
    "kelvin": "kelvin",
    "F": "farads, the electrical unit; give a temperature on the other side for degrees",
    "C": "coulombs, the electrical unit; give a temperature on the other side for degrees",
    "K": "kelvin",
}

#: Written forms of the scales pint does not take as they are.
_TEMPERATURE_WORDS = {
    "fahrenheit": "degF",
    "degrees fahrenheit": "degF",
    "degree fahrenheit": "degF",
    "deg f": "degF",
    "degrees f": "degF",
    "°f": "degF",
    "ºf": "degF",
    "celsius": "degC",
    "centigrade": "degC",
    "degrees celsius": "degC",
    "degree celsius": "degC",
    "deg c": "degC",
    "degrees c": "degC",
    "°c": "degC",
    "ºc": "degC",
    "kelvin": "kelvin",
    "kelvins": "kelvin",
}

#: Words with a US and a UK measure: pint reads them as the US one.
_US_MEASURES = {
    "gallon": "imperial_gallon",
    "gallons": "imperial_gallon",
    "pint": "imperial_pint",
    "pints": "imperial_pint",
    "quart": "imperial_quart",
    "quarts": "imperial_quart",
    "cup": "imperial_cup",
    "cups": "imperial_cup",
    "fluid_ounce": "imperial_fluid_ounce",
    "floz": "imperial_fluid_ounce",
    "ton": "long_ton",
    "tons": "long_ton",
}

#: Codes and signs of money: there are no exchange rates here.
_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "NZD", "CHF", "SEK", "NOK", "DKK", "INR", "MXN",
        "BRL", "KRW", "SGD", "HKD", "ZAR", "RUB", "TRY", "PLN", "CZK", "HUF", "ILS", "AED", "SAR", "THB",
        "BTC", "ETH", "$", "€", "£", "¥", "DOLLAR", "DOLLARS", "EURO", "EUROS", "POUND STERLING", "YEN",
    }
)  # fmt: skip

_PART = re.compile(r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*([^\d\s][^\d]*?)\s*(?=-?\d|$)")


def _unit(text: str, other: str) -> str:
    """The unit name pint takes for what was written, with F, C and K read by the other side.

    :param text: the unit as written
    :ptype text: str
    :param other: the unit on the other side of the conversion, as written
    :ptype other: str
    :return: a name pint defines
    :rtype: str
    :raises pint.errors.UndefinedUnitError: when neither as written nor lower-cased is a unit
    """
    raw = text.strip()
    word = raw.lower()
    if word in _TEMPERATURE_WORDS:
        return _TEMPERATURE_WORDS[word]
    if raw in _BARE_TEMPERATURE:
        other_unit = _unit_part(other)
        if other_unit in _BARE_TEMPERATURE or _is_temperature(other_unit):
            return _BARE_TEMPERATURE[raw]
        return raw
    for candidate in (raw, raw.lower(), raw.replace(" ", "_"), raw.lower().replace(" ", "_")):
        try:
            _ureg.Unit(candidate)
        except pint.errors.UndefinedUnitError, AttributeError, ValueError, TypeError:
            # NOSILENT: this loop IS the probe -- each spelling is tried until one is a
            # unit; none being one is raised below, naming what was written.
            continue
        return candidate
    raise pint.errors.UndefinedUnitError(raw)


def _unit_part(text: str) -> str:
    """The unit of a side written as a quantity ("350 F" is "F"), or the side itself."""
    return re.sub(r"^[-+\d./\s]+", "", text).strip()


def _is_temperature(text: str) -> bool:
    """Whether a unit as written is a temperature scale, read without the F/C/K rule."""
    word = text.strip().lower()
    if word in _TEMPERATURE_WORDS:
        return True
    try:
        return bool(_ureg.Unit(word).dimensionality == _ureg.kelvin.dimensionality)
    except pint.errors.UndefinedUnitError, AttributeError, ValueError, TypeError:
        return False


def _is_money(text: str) -> bool:
    """Whether a side names money: a code, a word, or a sign, alone or after an amount."""
    raw = text.strip().upper()
    unit = _unit_part(raw)
    return raw in _CURRENCIES or unit in _CURRENCIES or any(sign in raw for sign in "$€£¥")


def _quantity(written: str, to_unit: str) -> tuple[Any, str]:
    """A quantity of one or more parts, "5 feet 10 inches", added up; and how it was read."""
    parts = _PART.findall(written.strip())
    if not parts:
        raise ValueError(f"could not read {written!r} as an amount with units, e.g. '5 feet 10 inches'")
    total = None
    shown: list[str] = []
    for number, unit in parts:
        numerator, _, denominator = number.partition("/")
        magnitude = float(numerator) / float(denominator) if denominator else float(numerator)
        name = _unit(unit, to_unit)
        part = _ureg.Quantity(magnitude, name)
        total = part if total is None else total + part
        shown.append(f"{number} {unit.strip()}")
    return total, " + ".join(shown)


def _convert(value: float | None, from_unit: str, to_unit: str, quantity: str | None = None) -> str:
    if not _HAS_PINT:
        return tool_error(_TOOL, "convert", "pint package is not installed")
    for side in (from_unit, to_unit, quantity or ""):
        if side and _is_money(side):
            return tool_no_change(
                _TOOL,
                f"{side.strip()} is money. There are no exchange rates here: they change every day. "
                "Search the web for today's rate. Nothing was converted.",
            )
    letters: list[str] = []
    try:
        written = (quantity or "").strip()
        if not written and any(c.isdigit() for c in from_unit):
            written = from_unit
        if written:
            source, read = _quantity(written, to_unit)
        else:
            if value is None:
                raise ValueError("give a value, or a quantity such as '5 feet 10 inches'")
            name = _unit(from_unit, to_unit)
            source = _ureg.Quantity(value, name)
            read = f"{_format(value)} {from_unit.strip()}"
            if from_unit.strip() in _BARE_TEMPERATURE:
                letters.append(f"{from_unit.strip()} read as {_BARE_READ[name]}")
        target = _unit(to_unit, from_unit or written)
        if to_unit.strip() in _BARE_TEMPERATURE:
            letters.append(f"{to_unit.strip()} read as {_BARE_READ[target]}")
        converted = source.to(target)
    except pint.errors.UndefinedUnitError as exc:
        return tool_error(
            _TOOL,
            "convert",
            f"unknown unit {exc}. Write units as words or symbols, e.g. miles, km, kg, lb, degF, degC, ml, cup, mph.",
        )
    except pint.errors.DimensionalityError as exc:
        return tool_error(_TOOL, "convert", f"incompatible units, they measure different things: {exc}")
    except (ValueError, ZeroDivisionError) as exc:
        return tool_error(_TOOL, "convert", str(exc))
    except Exception as exc:
        return tool_error(_TOOL, "convert", str(exc))

    notes = [*letters, f"rounded to {_SIGNIFICANT} significant figures"]
    for side in (from_unit, to_unit, written):
        for word in re.findall(r"[a-z_]+", side.lower()):
            if word in _US_MEASURES:
                notes.append(f"{word} is the US measure; write {_US_MEASURES[word]} for the UK one")
    shown_to = to_unit.strip()
    return f"{read} = {converted.magnitude:.{_SIGNIFICANT}g} {shown_to} ({'; '.join(dict.fromkeys(notes))})"


def _format(value: float) -> str:
    return f"{value:g}"


def create_unit_converter_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a unit converter tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`UnitConverterTool.execute` as their single
    execution body. ``config`` is unused.
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        UnitConverterTool(),
        description=description,
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
            "value": {"type": "number", "description": "The number to convert. Leave out when quantity is given."},
            "from_unit": {
                "type": "string",
                "description": "e.g. miles, kg, celsius, F. F, C and K are degrees when the other unit is a temperature.",
            },
            "to_unit": {
                "type": "string",
                "description": "e.g. kilometers, pounds, fahrenheit, cm.",
            },
            "quantity": {
                "type": "string",
                "description": "An amount of several parts to convert as one, e.g. '5 feet 10 inches' or '2 lb 3 oz'.",
            },
        },
        "required": ["to_unit"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """convert value between units.

        :param kwargs: 'to_unit', and 'value' with 'from_unit', or 'quantity'
        :ptype kwargs: Any
        :return: result containing converted value or error
        :rtype: ToolResult
        """
        raw_value = kwargs.get("value")
        value = float(raw_value) if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) else None
        from_unit = str(kwargs.get("from_unit") or "")
        to_unit = str(kwargs.get("to_unit") or "")
        quantity = kwargs.get("quantity")
        content = _convert(value, from_unit, to_unit, quantity)
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
