"""Workflow tool factories that operate on a ToolContextManager."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.context import ToolContextManager

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


class RecallContextInput(BaseModel):
    """Input for the recall_context tool."""

    context_id: str = Field(description="The id in a [ctx:<id>] mark, with or without the ctx: part.")


class DeclareWorkflowInput(BaseModel):
    """Input for the declare_workflow tool."""

    plan: str = Field(description="What the work is for.")
    steps: list[str] = Field(description="The steps, in order.")


# ---------------------------------------------------------------------------
# Tool implementations (async)
# ---------------------------------------------------------------------------


async def _set_variable(tool_context: ToolContextManager, key: str, value: str, value_type: str = "string") -> str:
    try:
        context_id = await tool_context.set_variable(key, value, value_type)
        return f"Variable '{key}' saved (context_id: {context_id})"
    except ValueError as exc:
        return str(exc)


async def _get_variable(tool_context: ToolContextManager, key: str) -> str:
    var = await tool_context.get_variable(key)
    if var is None:
        return f"Variable '{key}' not found"
    return f"{key} ({var['value_type']}): {var['value']}"


async def _recall_context(tool_context: ToolContextManager, context_id: str) -> str:
    item = await tool_context.get_context_item(context_id)
    if item is None:
        return f"Saved result '{context_id}' not found in this conversation."
    return f"{item.get('key', 'unknown')}: {item.get('content', '')}"


def _declare_workflow(tool_context: ToolContextManager, plan: str, steps: list[str]) -> str:
    state = tool_context.declare_workflow(plan, steps)
    step_list = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
    return f"Workflow declared:\n  Plan: {state['plan']}\n  Steps:\n{step_list}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_workflow_tools(tool_context: ToolContextManager) -> list[Any]:
    """Create workflow management tools bound to a ToolContextManager.

    Returns a list of LangChain ``BaseTool`` instances.
    """
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
            func=lambda context_id: None,
            coroutine=lambda context_id: _recall_context(tool_context, context_id),
            name="context_recall",
            description="Read the whole of a saved tool result by the id in its [ctx:<id>] mark.",
            args_schema=RecallContextInput,
        ),
        StructuredTool.from_function(
            func=lambda plan, steps: _declare_workflow(tool_context, plan, steps),
            name="workflow_declare",
            description="Write down a plan and its steps before a task that takes several actions in order.",
            args_schema=DeclareWorkflowInput,
        ),
    ]
