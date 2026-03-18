"""Workflow tool factories that operate on a ToolContextManager."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.context import ToolContextManager


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class SetVariableInput(BaseModel):
    """Input for the set_variable tool."""

    key: str = Field(description="Variable name")
    value: str = Field(description="Variable value")
    value_type: str = Field(default="string", description="Type hint (string, number, json, etc.)")


class GetVariableInput(BaseModel):
    """Input for the get_variable tool."""

    key: str = Field(description="Variable name to retrieve")


class RecallContextInput(BaseModel):
    """Input for the recall_context tool."""

    context_id: str = Field(description="The context ID to recall")


class DeclareWorkflowInput(BaseModel):
    """Input for the declare_workflow tool."""

    plan: str = Field(description="Description of the workflow plan")
    steps: list[str] = Field(description="Ordered list of workflow steps")


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
        return f"Context item '{context_id}' not found"
    return f"{item.get('key', 'unknown')}: {item.get('value', '')}"


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
            name="set_variable",
            description=(
                "Store a key-value variable in conversation context. "
                "Use this to remember important values across turns."
            ),
            args_schema=SetVariableInput,
        ),
        StructuredTool.from_function(
            func=lambda key: None,
            coroutine=lambda key: _get_variable(tool_context, key),
            name="get_variable",
            description="Retrieve a previously stored variable by key.",
            args_schema=GetVariableInput,
        ),
        StructuredTool.from_function(
            func=lambda context_id: None,
            coroutine=lambda context_id: _recall_context(tool_context, context_id),
            name="recall_context",
            description="Recall a specific context item by its context_id.",
            args_schema=RecallContextInput,
        ),
        StructuredTool.from_function(
            func=lambda plan, steps: _declare_workflow(tool_context, plan, steps),
            name="declare_workflow",
            description=(
                "Declare a structured workflow with a plan and ordered steps. "
                "Use this when a task requires multiple sequential actions."
            ),
            args_schema=DeclareWorkflowInput,
        ),
    ]
