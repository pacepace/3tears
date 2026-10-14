"""Calculator tool using simpleeval for safe math evaluation.

A result shows the expression it answers: "84.50*0.15 = 12.675", never the bare
number, because a later round reads the result without the call's arguments and
a lone "12.674999999999999" answers nothing it can see. Trig is in radians and
``log`` is the natural log, as in Python, and a result that used them says so;
``sind`` and friends take degrees. What people write is read: "15% of 84.50",
"$22.50*3.5", "1,000*2". An error says what to write instead.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error

__all__ = [
    "CalculatorTool",
    "create_calculator_tool",
]

try:
    from simpleeval import (
        FeatureNotAvailable,
        FunctionNotDefined,
        NameNotDefined,
        NumberTooHigh,
        OperatorNotDefined,
        simple_eval,
    )

    _HAS_SIMPLEEVAL = True
except ImportError:
    _HAS_SIMPLEEVAL = False

_TOOL = "calculator"

#: Significant digits a float result is shown to: enough for any money or
#: measurement, few enough that 0.1+0.2 reads 0.3 and not 0.30000000000000004.
_SIGNIFICANT = 12


def _variadic(fn: Any) -> Any:
    """A function of a list, taking its items as arguments: ``sum(1, 2, 3)``."""

    def _call(*values: float) -> Any:
        return fn(values)

    return _call


_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sind": lambda x: math.sin(math.radians(x)),
    "cosd": lambda x: math.cos(math.radians(x)),
    "tand": lambda x: math.tan(math.radians(x)),
    "asind": lambda x: math.degrees(math.asin(x)),
    "acosd": lambda x: math.degrees(math.acos(x)),
    "atand": lambda x: math.degrees(math.atan(x)),
    "radians": math.radians,
    "degrees": math.degrees,
    "hypot": math.hypot,
    "log": math.log,
    "ln": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "sum": _variadic(math.fsum),
    "mean": _variadic(statistics.fmean),
    "median": _variadic(statistics.median),
}

_SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_RADIAN_FUNCTIONS = ("sin", "cos", "tan", "asin", "acos", "atan", "atan2")
_RADIANS_USED = re.compile(r"\b(?:" + "|".join(_RADIAN_FUNCTIONS) + r")\s*\(")
_NATURAL_LOG_USED = re.compile(r"\blog\s*\(")

_WRITE_INSTEAD = (
    "Write one expression of numbers, + - * / ** % ( ), and these functions: "
    + ", ".join(sorted(_SAFE_FUNCTIONS))
    + "; and pi, e, tau. For example 84.50*0.15, 15% of 84.50, sqrt(2116), sind(30)."
)


# Postfix factorial in math notation (``n!``, ``(2+3)!``) is not valid
# Python expression syntax and so simpleeval cannot parse it. Translate
# the common math-notation forms into ``factorial(...)`` calls before
# handing the expression to simpleeval. Two cases:
#  - ``<digits>!``  -> ``factorial(<digits>)``
#  - ``(<expr>)!``  -> ``factorial((<expr>))``
# Both transforms are local rewrites of well-formed sub-expressions; they
# do NOT attempt to chain (``5!!`` -> ``factorial(factorial(5))``) which
# math notation reserves for double factorial anyway.
_FACTORIAL_LITERAL_RE = re.compile(r"(\d+)!")
_FACTORIAL_PAREN_RE = re.compile(r"(\([^()]*\))!")

#: "15% of 84.50" -> a fifteenth-hundredth times what follows.
_PERCENT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s+", re.IGNORECASE)
#: "84.50*15%" -> 15/100, but only where a % closes the number: "10 % 3" and
#: "10%3" stay the remainder operator.
_PERCENT_CLOSE = re.compile(r"(\d+(?:\.\d+)?)%(?=\s*(?:$|[)+\-*/,]))")
#: A number written with thousands commas, "1,000,000" -- but not one opening a
#: function's arguments or following another argument: in "max(1,000)" the comma
#: separates two arguments, as Python reads it.
_THOUSANDS = re.compile(r"(?<![\d(,.])(?<!,\s)(?<!\(\s)(\d{1,3}(?:,\d{3})+)(?![\d,])")
_CURRENCY = re.compile(r"[$€£¥]")


def _normalize_expression(expression: str) -> str:
    """rewrite what people write into simpleeval-friendly form.

    ``^`` is power, postfix ``!`` is factorial, ``×`` and ``÷`` are ``*`` and
    ``/``, currency signs and thousands commas go, and a percent becomes a
    fraction: "15% of 84.50" and "84.50*15%". A ``%`` between numbers is still
    the remainder.

    :param expression: raw expression as written by the caller
    :ptype expression: str
    :return: expression with notation conveniences rewritten
    :rtype: str
    """
    rewritten = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    rewritten = _CURRENCY.sub("", rewritten)
    rewritten = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), rewritten)
    rewritten = _PERCENT_OF.sub(r"(\1/100)*", rewritten)
    rewritten = _PERCENT_CLOSE.sub(r"(\1/100)", rewritten)
    rewritten = _FACTORIAL_PAREN_RE.sub(r"factorial(\1)", rewritten)
    rewritten = _FACTORIAL_LITERAL_RE.sub(r"factorial(\1)", rewritten)
    return rewritten


def _format(value: Any) -> str:
    """A number as the result shows it: no float noise, no trailing ``.0``."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        shown = float(f"{value:.{_SIGNIFICANT}g}")
        if shown == int(shown) and abs(shown) < 1e15:
            return str(int(shown))
        return f"{shown:.{_SIGNIFICANT}g}"
    return str(value)


def _units_note(expression: str) -> str:
    """What the result's units are, when trig or ``log`` could be read two ways."""
    notes: list[str] = []
    if _RADIANS_USED.search(expression):
        notes.append(
            "sin, cos, tan and their inverses work in radians; sind, cosd, tand, asind, acosd, atand use degrees"
        )
    if _NATURAL_LOG_USED.search(expression):
        notes.append("log is the natural log (base e); log10 is base 10 and log2 base 2")
    return f" ({'; '.join(notes)})" if notes else ""


def _evaluate(expression: str) -> str:
    if not _HAS_SIMPLEEVAL:
        return tool_error(_TOOL, "evaluate", "simpleeval package is not installed")
    shown = expression.strip()
    if not shown:
        return tool_error(_TOOL, "evaluate", f"no expression given. {_WRITE_INSTEAD}")
    try:
        result = simple_eval(
            _normalize_expression(shown),
            functions=_SAFE_FUNCTIONS,
            names=_SAFE_NAMES,
        )
    except ZeroDivisionError:
        return tool_error(_TOOL, "evaluate", f"{shown}: division by zero.")
    except NameNotDefined as exc:
        return tool_error(
            _TOOL, "evaluate", f"{shown}: {exc.name!r} is not a number or a name it knows. {_WRITE_INSTEAD}"
        )
    except FunctionNotDefined as exc:
        return tool_error(_TOOL, "evaluate", f"{shown}: there is no function {exc.func_name!r}. {_WRITE_INSTEAD}")
    except (FeatureNotAvailable, OperatorNotDefined, SyntaxError) as exc:
        return tool_error(_TOOL, "evaluate", f"{shown}: could not read it ({exc}). {_WRITE_INSTEAD}")
    except NumberTooHigh:
        return tool_error(_TOOL, "evaluate", f"{shown}: the result is too large to work out.")
    except (ValueError, OverflowError, TypeError) as exc:
        return tool_error(_TOOL, "evaluate", f"{shown}: {exc}. For example sqrt takes a number of 0 or more.")
    if not isinstance(result, (int, float)):
        return tool_error(_TOOL, "evaluate", f"{shown}: that is not a calculation with numbers. {_WRITE_INSTEAD}")
    return f"{shown} = {_format(result)}{_units_note(_normalize_expression(shown))}"


def create_calculator_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a calculator tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share one execution body (:meth:`CalculatorTool.execute`).
    ``config`` is unused for calculator (no per-agent config); kept
    in the signature for :func:`register_builtins` factory-shape
    parity.
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        CalculatorTool(),
        description=description,
    )


class CalculatorTool(TearsTool):
    """TearsTool wrapper for safe math evaluation via simpleeval.

    evaluates mathematical expressions using a restricted set of
    functions and constants. supports basic arithmetic, trigonometry
    (radians, and degrees through the ``d`` forms), logarithms, sums
    and means, and common math constants.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "The calculation, e.g. '84.50*0.15', '15% of 84.50', '$22.50*3.5', 'sqrt(2116)', 'sind(30)'. "
                    "sin, cos and tan take radians, sind, cosd and tand take degrees; log is the natural log, "
                    "log10 base 10."
                ),
            },
        },
        "required": ["expression"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """evaluate mathematical expression.

        :param kwargs: must include 'expression' key with math expression string
        :ptype kwargs: Any
        :return: result containing evaluated value or error
        :rtype: ToolResult
        """
        expression = str(kwargs.get("expression") or "")
        content = _evaluate(expression)
        success = not content.startswith("[TOOL ERROR]")
        result = ToolResult(
            success=success,
            content=content,
            error=content if not success else None,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for calculator.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="evaluate mathematical expressions safely; the result shows the expression it answers",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.calculator"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
