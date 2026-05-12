"""Calculator tool using simpleeval for safe math evaluation."""

from __future__ import annotations

import math
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error

__all__ = [
    "CalculatorInput",
    "CalculatorTool",
    "create_calculator_tool",
]

try:
    from simpleeval import simple_eval

    _HAS_SIMPLEEVAL = True
except ImportError:
    _HAS_SIMPLEEVAL = False

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}

_SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


# Postfix factorial in math notation (``n!``, ``(2+3)!``) is not valid
# Python expression syntax and so simpleeval cannot parse it. Translate
# the common math-notation forms into ``factorial(...)`` calls before
# handing the expression to simpleeval. Two cases:
#  - ``<digits>!``  -> ``factorial(<digits>)``
#  - ``(<expr>)!``  -> ``factorial((<expr>))``
# Both transforms are local rewrites of well-formed sub-expressions; they
# do NOT attempt to chain (``5!!`` -> ``factorial(factorial(5))``) which
# math notation reserves for double factorial anyway. Repeated use of the
# regex would be needed for chaining; the LLM has not asked for it.
_FACTORIAL_LITERAL_RE = re.compile(r"(\d+)!")
_FACTORIAL_PAREN_RE = re.compile(r"(\([^()]*\))!")


def _normalize_expression(expression: str) -> str:
    """rewrite math-notation conveniences into simpleeval-friendly form.

    two transforms:

    1. ``^`` -> ``**`` so callers can write ``2^10`` instead of
       ``2**10`` (LLMs frequently emit the math-notation form). xor
       is not exposed by the calculator's safe-function set, so the
       rewrite has no behavior conflict.
    2. postfix factorial: ``5!`` -> ``factorial(5)`` and
       ``(2+3)!`` -> ``factorial((2+3))``. two regex passes catch the
       most common literal-and-paren cases. ``n!`` operating on a bare
       identifier is intentionally not rewritten -- the calculator's
       name table only exposes math constants and a chained-factorial
       request would more often be a typo than intent.

    :param expression: raw expression as written by the caller
    :ptype expression: str
    :return: expression with notation conveniences rewritten
    :rtype: str
    """
    rewritten = expression.replace("^", "**")
    rewritten = _FACTORIAL_PAREN_RE.sub(r"factorial(\1)", rewritten)
    rewritten = _FACTORIAL_LITERAL_RE.sub(r"factorial(\1)", rewritten)
    return rewritten


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


def _evaluate(expression: str) -> str:
    if not _HAS_SIMPLEEVAL:
        return tool_error("calculator", "evaluate", "simpleeval package is not installed")
    try:
        normalized = _normalize_expression(expression)
        result = simple_eval(
            normalized,
            functions=_SAFE_FUNCTIONS,
            names=_SAFE_NAMES,
        )
        # Format: avoid trailing .0 for integer results
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            return str(int(result))
        return str(result)
    except Exception as exc:
        return tool_error("calculator", "evaluate", str(exc))


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
        args_schema=CalculatorInput,
    )


class CalculatorTool(TearsTool):
    """TearsTool wrapper for safe math evaluation via simpleeval.

    evaluates mathematical expressions using a restricted set of
    functions and constants. supports basic arithmetic, trigonometry,
    logarithms, and common math constants.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "mathematical expression to evaluate",
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
        expression = kwargs.get("expression", "")
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
            description="evaluate mathematical expressions safely",
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
