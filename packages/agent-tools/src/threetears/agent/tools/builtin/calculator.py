"""Calculator tool using simpleeval for safe math evaluation."""

from __future__ import annotations

import math
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.utils import tool_error

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


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


def _evaluate(expression: str) -> str:
    if not _HAS_SIMPLEEVAL:
        return tool_error("calculator", "evaluate", "simpleeval package is not installed")
    try:
        result = simple_eval(
            expression,
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
    """Factory: create a calculator tool."""
    return StructuredTool.from_function(
        func=_evaluate,
        name="calculator",
        description=description,
        args_schema=CalculatorInput,
    )
