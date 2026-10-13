"""Workflow tool factories that operate on a ToolContextManager."""

from __future__ import annotations

from datetime import tzinfo
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.observe import get_logger

from threetears.core.utils.local_time import resolve_zone

from threetears.agent.tools.context import ToolContextManager, VariableLimitError, recalled_text
from threetears.agent.tools.text_window import WindowedInput
from threetears.agent.tools.utils import TOOL_ERROR_PREFIX, tool_no_change

log = get_logger(__name__)

__all__ = [
    "DeclareWorkflowInput",
    "GetVariableInput",
    "RecallContextInput",
    "SetVariableInput",
    "load_workflow_tools",
]


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class SetVariableInput(BaseModel):
    """Input for the set_variable tool."""

    key: str = Field(description="The name to save it under.")
    value: str = Field(description="The value.")
    value_type: str = Field(default="string", description="What kind of value it is: string, number, json.")


class GetVariableInput(BaseModel):
    """Input for the get_variable tool."""

    key: str = Field(description="The name it was saved under.")


class RecallContextInput(WindowedInput):
    """Input for the recall_context tool."""

    context_id: str = Field(description="The id in a [ctx:<id>] mark, with or without the ctx: part.")


class DeclareWorkflowInput(BaseModel):
    """Input for the declare_workflow tool."""

    plan: str = Field(description="What the work is for.")
    steps: list[str] = Field(description="The steps, in order.")


# ---------------------------------------------------------------------------
# Tool implementations (async)
# ---------------------------------------------------------------------------


async def _saved_names(tool_context: ToolContextManager) -> str:
    """The names a variable is saved under in this conversation, as a sentence."""
    names = sorted(str(item["key"]) for item in await tool_context.get_all_variables())
    if not names:
        return "No variables are saved in this conversation."
    return f"Saved names: {', '.join(names)}."


async def _set_variable(tool_context: ToolContextManager, key: str, value: str, value_type: str = "string") -> str:
    try:
        context_id = await tool_context.set_variable(key, value, value_type)
    except VariableLimitError as exc:
        # At the cap: nothing was saved and nothing broke. Its own status, so a caller that
        # refuses a failed tool for the rest of a turn does not refuse this one.
        return tool_no_change("variable_set", f"{exc} {await _saved_names(tool_context)}")
    except Exception as exc:
        log.error("variable_set failed", extra={"extra_data": {"key": key, "error": str(exc)}})
        return f"{TOOL_ERROR_PREFIX} variable_set: {exc}"
    return f"Variable '{key}' saved (context_id: {context_id})"


async def _get_variable(tool_context: ToolContextManager, key: str) -> str:
    try:
        var = await tool_context.get_variable(key)
        if var is None:
            return tool_no_change("variable_get", f"no variable named '{key}'. {await _saved_names(tool_context)}")
    except Exception as exc:
        log.error("variable_get failed", extra={"extra_data": {"key": key, "error": str(exc)}})
        return f"{TOOL_ERROR_PREFIX} variable_get: {exc}"
    return f"{key} ({var['value_type']}): {var['value']}"


async def _recall_context(
    tool_context: ToolContextManager, context_id: str, zone: tzinfo | None = None, offset: int = 0
) -> str:
    item = await tool_context.get_context_item(context_id)
    if item is None:
        return f"Saved result '{context_id}' not found in this conversation."
    return recalled_text(item, offset=offset, tz=zone, label=f"{item.get('key', 'unknown')}: ")


def _declare_workflow(tool_context: ToolContextManager, plan: str, steps: list[str]) -> str:
    state = tool_context.declare_workflow(plan, steps)
    step_list = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
    return f"Workflow declared:\n  Plan: {state['plan']}\n  Steps:\n{step_list}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_workflow_tools(tool_context: ToolContextManager, *, tz: tzinfo | str | None = None) -> list[Any]:
    """Create workflow management tools bound to a ToolContextManager.

    :param tool_context: the conversation's context manager
    :ptype tool_context: ToolContextManager
    :param tz: the zone of the person the results are read to: a ``tzinfo`` or
        an IANA name. ``context_recall`` says when a stored result was
        refreshed in it; ``None`` says it in UTC.
    :ptype tz: tzinfo | str | None
    :return: a list of LangChain ``BaseTool`` instances
    :rtype: list[Any]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)
    return [
        StructuredTool.from_function(
            func=lambda key, value, value_type="string": None,
            coroutine=lambda key, value, value_type="string": _set_variable(tool_context, key, value, value_type),
            name="variable_set",
            description=(
                "Save a value by name for the rest of this conversation, such as a number or an "
                "id you will need again. To keep something for later conversations, use memory_add."
            ),
            args_schema=SetVariableInput,
        ),
        StructuredTool.from_function(
            func=lambda key: None,
            coroutine=lambda key: _get_variable(tool_context, key),
            name="variable_get",
            description="Read a value saved with variable_set.",
            args_schema=GetVariableInput,
        ),
        StructuredTool.from_function(
            func=lambda context_id, offset=0: None,
            coroutine=lambda context_id, offset=0: _recall_context(tool_context, context_id, zone, offset),
            name="context_recall",
            description=(
                "Read a saved tool result by the id in its [ctx:<id>] mark. A long one comes in parts: "
                "each part names the offset of the next."
            ),
            args_schema=RecallContextInput,
        ),
        StructuredTool.from_function(
            func=lambda plan, steps: _declare_workflow(tool_context, plan, steps),
            name="workflow_declare",
            description="Write down a plan and its steps before a task that takes several actions in order.",
            args_schema=DeclareWorkflowInput,
        ),
    ]
