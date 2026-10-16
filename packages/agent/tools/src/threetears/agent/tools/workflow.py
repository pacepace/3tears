"""Workflow tool factories that operate on a ToolContextManager."""

from __future__ import annotations

import json
from datetime import tzinfo
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.observe.text_block import under
from threetears.observe import get_logger

from threetears.core.utils.local_time import resolve_zone

from threetears.agent.tools.context import (
    ToolContextManager,
    VariableLimitError,
    missing_result_text,
    recalled_text,
)
from threetears.agent.tools.text_window import WindowedInput
from threetears.agent.tools.utils import TOOL_ERROR_PREFIX, tool_no_change

log = get_logger(__name__)

__all__ = [
    "DeclareWorkflowInput",
    "DeleteVariableInput",
    "GetVariableInput",
    "ListVariablesInput",
    "RecallContextInput",
    "SetVariableInput",
    "VALUE_TYPES",
    "load_workflow_tools",
]


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


#: The kinds of value a variable may be said to hold. A number must read as one and a
#: json value must parse, so the kind a listing shows is true of the value.
VALUE_TYPES: tuple[str, ...] = ("string", "number", "json")

#: How much of a value a listing, a delete or a replacement shows.
_PREVIEW_CHARS = 200


class SetVariableInput(BaseModel):
    """Input for the set_variable tool."""

    key: str = Field(description="The name to save it under.")
    value: str = Field(description="The value.")
    value_type: str = Field(default="string", description="What kind of value it is: string, number, json.")


class GetVariableInput(BaseModel):
    """Input for the get_variable tool."""

    key: str = Field(description="The name it was saved under.")


class ListVariablesInput(BaseModel):
    """Input for the variable_list tool: it takes nothing."""


class DeleteVariableInput(BaseModel):
    """Input for the variable_delete tool."""

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


def _preview(value: str) -> str:
    """A value as a listing or a result shows it: whole when short, else its start and how much is left."""
    if len(value) <= _PREVIEW_CHARS:
        return repr(value)
    return f"{value[:_PREVIEW_CHARS]!r} ... ({len(value) - _PREVIEW_CHARS:,} more chars)"


def _type_problem(value: str, value_type: str) -> str | None:
    """Why ``value`` is not the kind ``value_type`` says, or ``None`` when it is."""
    if value_type not in VALUE_TYPES:
        return f"value_type '{value_type}' is not one of: {', '.join(VALUE_TYPES)}"
    if value_type == "number":
        try:
            float(value)
        except ValueError:
            return f"value_type is number and {_preview(value)} is not a number"
    if value_type == "json":
        try:
            json.loads(value)
        except ValueError as exc:
            return f"value_type is json and the value does not parse as JSON ({exc})"
    return None


async def _set_variable(tool_context: ToolContextManager, key: str, value: str, value_type: str = "string") -> str:
    problem = _type_problem(value, value_type)
    if problem is not None:
        return f"{TOOL_ERROR_PREFIX} variable_set: {problem}. Nothing was saved."
    try:
        before = await tool_context.get_variable(key)
        await tool_context.set_variable(key, value, value_type)
        after = await tool_context.get_variable(key)
    except VariableLimitError as exc:
        # At the cap: nothing was saved and nothing broke. Its own status, so a caller that
        # refuses a failed tool for the rest of a turn does not refuse this one.
        return tool_no_change("variable_set", f"{exc} {await _saved_names(tool_context)}")
    except Exception as exc:
        log.error("variable_set failed", extra={"extra_data": {"key": key, "error": str(exc)}})
        return f"{TOOL_ERROR_PREFIX} variable_set: {exc}"
    kept = str((after or {}).get("value", value))
    said = [f"Saved '{key}' ({value_type}, {len(kept):,} chars)."]
    if len(kept) < len(value):
        said.append(
            f"It was cut: only the first {len(kept):,} of {len(value):,} chars were kept, "
            f"and the last {len(value) - len(kept):,} are not saved."
        )
    if before is None:
        said.append("Nothing was saved under that name before.")
    elif before["value"] == kept and before["value_type"] == value_type:
        said.append("It already held this value.")
    else:
        said.append(
            f"It replaced the earlier value, which is gone ({before['value_type']}, "
            f"{len(before['value']):,} chars): {_preview(before['value'])}"
        )
    return " ".join(said)


async def _get_variable(tool_context: ToolContextManager, key: str) -> str:
    try:
        var = await tool_context.get_variable(key)
        if var is None:
            return tool_no_change("variable_get", f"no variable named '{key}'. {await _saved_names(tool_context)}")
    except Exception as exc:
        log.error("variable_get failed", extra={"extra_data": {"key": key, "error": str(exc)}})
        return f"{TOOL_ERROR_PREFIX} variable_get: {exc}"
    return f"{key} ({var['value_type']}): {var['value']}"


async def _list_variables(tool_context: ToolContextManager) -> str:
    try:
        saved = sorted(await tool_context.get_all_variables(), key=lambda item: str(item["key"]))
    except Exception as exc:
        log.error("variable_list failed", extra={"extra_data": {"error": str(exc)}})
        return f"{TOOL_ERROR_PREFIX} variable_list: {exc}"
    if not saved:
        return f"No variables are saved in this conversation. It may hold {tool_context.var_limit}."
    lines = [f"{len(saved)} of {tool_context.var_limit} variables this conversation may hold:"]
    for item in saved:
        value = str(item.get("content") or "")
        kind = (item.get("metadata") or {}).get("value_type", "string")
        lines.append(f"- {item['key']} ({kind}, {len(value):,} chars): {_preview(value)}")
    return "\n".join(lines)


async def _delete_variable(tool_context: ToolContextManager, key: str) -> str:
    try:
        held = await tool_context.get_variable(key)
        if held is None:
            return tool_no_change("variable_delete", f"no variable named '{key}'. {await _saved_names(tool_context)}")
        await tool_context.delete_variable(key)
        left = len(await tool_context.get_all_variables())
    except Exception as exc:
        log.error("variable_delete failed", extra={"extra_data": {"key": key, "error": str(exc)}})
        return f"{TOOL_ERROR_PREFIX} variable_delete: {exc}"
    value = str(held["value"])
    return (
        f"Deleted '{key}' ({held['value_type']}, {len(value):,} chars). It cannot be brought back. "
        f"It held: {_preview(value)}. {left} of {tool_context.var_limit} variables are saved now."
    )


async def _recall_context(
    tool_context: ToolContextManager, context_id: str, zone: tzinfo | None = None, offset: int = 0
) -> str:
    item = await tool_context.get_context_item(context_id)
    if item is None:
        return missing_result_text(context_id, tool_context.items)
    return recalled_text(item, offset=offset, tz=zone, label=f"{item.get('key', 'unknown')}: ")


def _declare_workflow(tool_context: ToolContextManager, plan: str, steps: list[str]) -> str:
    before = tool_context.workflow_state
    was = _plan_before(before) if before else "No plan was there before."
    state = tool_context.declare_workflow(plan, steps)
    step_list = [line for i, s in enumerate(steps) for line in under(s, "     ", first=f"  {i + 1}. ")]
    plan = under(str(state["plan"]), "    ", first="  Plan: ") or ["  Plan:"]
    return "\n".join([f"Workflow declared. {was}", *plan, "  Steps:", *step_list])


def _plan_before(state: dict[str, Any]) -> str:
    """The plan a new declaration replaces: its name, its steps and how far it got."""
    steps = state.get("steps") or []
    done = min(int(state.get("current_step") or 0), len(steps))
    count = f"{len(steps)} step{'' if len(steps) == 1 else 's'}, {done} done"
    return f"It replaced the plan {str(state.get('plan') or '')!r} ({count}, {state.get('status') or 'unknown'})."


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
                "id you will need again. Saving under a name already used replaces its value; the result "
                "shows the value replaced. To keep something for later conversations, use memory_add."
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
            func=lambda: None,
            coroutine=lambda: _list_variables(tool_context),
            name="variable_list",
            description=(
                "List the values saved with variable_set in this conversation: each name, its kind, "
                "its size and its start."
            ),
            args_schema=ListVariablesInput,
        ),
        StructuredTool.from_function(
            func=lambda key: None,
            coroutine=lambda key: _delete_variable(tool_context, key),
            name="variable_delete",
            description="Delete a value saved with variable_set. It cannot be undone.",
            args_schema=DeleteVariableInput,
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
