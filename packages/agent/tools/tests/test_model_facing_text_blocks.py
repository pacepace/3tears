"""Exact output of every renderer that puts free text in front of a model.

Each item's text sits under its heading: every line indented, no blank line inside an item. An item whose text
has nothing in it still shows its heading.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.relevance import tool_hit_line
from threetears.agent.tools.reports.mermaid_generator import MermaidGenerator
from threetears.agent.tools.reports.severity import SeverityConfig
from threetears.agent.tools.router import ToolRouter
from threetears.agent.tools.todo import load_todo_tools
from threetears.agent.tools.workflow import load_workflow_tools

from packages.agent.tools.tests.test_todo import InMemoryTodoStorage
from packages.agent.tools.tests.testing_utils import FakePool, make_context_metadata, make_nats_mock

# A text of several lines with a blank line and an indented line in it.
MULTI = "first line\n\n  indented line\nlast line"
BLANKS = "  \n\t\n"


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_tb_{uuid.uuid4().hex[:8]}")
    b.initialize(make_context_metadata())
    yield b
    from threetears.core.testing import drain_and_shutdown_bridge

    drain_and_shutdown_bridge()
    b.reset()


@pytest.fixture()
def ctx(l1_backend: SQLiteBackend) -> ToolContextManager:
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend, kv_key_scope="test-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    coll = ContextItemCollection(reg, config, nats_client=make_nats_mock())
    coll.l3_pool = FakePool()
    return ToolContextManager(coll, "00000000-0000-0000-0000-000000000001", "user1")


def test_an_active_workflow_prompt_indents_every_line_of_a_step(ctx: ToolContextManager) -> None:
    ctx.declare_workflow("Plan", [MULTI, "second"])
    assert ctx.build_workflow_prompt() == (
        "## Active Workflow: Plan\n1. [ ] first line\n         indented line\n       last line\n2. [ ] second"
    )


def test_an_active_workflow_prompt_shows_the_heading_of_a_step_with_no_text(ctx: ToolContextManager) -> None:
    ctx.declare_workflow("Plan", [BLANKS, "second"])
    assert ctx.build_workflow_prompt() == "## Active Workflow: Plan\n1. [ ]\n2. [ ] second"


@pytest.mark.asyncio
async def test_conversation_variables_sit_under_their_heading_with_no_blank_line(ctx: ToolContextManager) -> None:
    await ctx.set_variable("notes", MULTI)
    await ctx.set_variable("other", "one")
    result = ctx.build_conversation_context()
    assert result is not None
    assert result.splitlines()[0] == "[Conversation Variables]"
    assert result == (
        "[Conversation Variables]\n- notes (string): first line\n    indented line\n  last line\n- other (string): one"
    )


@pytest.mark.asyncio
async def test_a_variable_with_no_text_still_shows_its_heading(ctx: ToolContextManager) -> None:
    await ctx.set_variable("blank", BLANKS)
    assert ctx.build_conversation_context() == "[Conversation Variables]\n- blank (string):"


@pytest.mark.asyncio
async def test_a_tool_result_preview_is_one_line(ctx: ToolContextManager) -> None:
    context_id = await ctx.save_tool_result("fetch", "body", short_desc=MULTI)
    result = ctx.build_conversation_context() or ""
    previews = [line for line in result.splitlines() if line.startswith("- [")]
    assert previews == [f"- [{context_id}] fetch:{context_id}: first line indented line last line"]


@pytest.mark.asyncio
async def test_a_tool_result_with_no_description_still_shows_its_heading(ctx: ToolContextManager) -> None:
    context_id = await ctx.save_tool_result("fetch", "body", short_desc=BLANKS)
    result = ctx.build_conversation_context() or ""
    previews = [line for line in result.splitlines() if line.startswith("- [")]
    assert previews == [f"- [{context_id}] fetch:{context_id}:"]


@pytest.mark.asyncio
async def test_a_ledger_description_is_one_line_and_an_empty_one_keeps_its_heading(ctx: ToolContextManager) -> None:
    first = "00000000-0000-0000-0000-0000000000a1"
    second = "00000000-0000-0000-0000-0000000000a2"
    await ctx.add_ledger_ref(first, "memory", MULTI)
    await ctx.add_ledger_ref(second, "chunk", BLANKS)
    items = [line for line in ctx.build_ledger_prompt().splitlines() if line.startswith("- [")]
    assert items == [f"- [memory:{first}] first line indented line last line", f"- [chunk:{second}]"]


# ---------------------------------------------------------------------------
# workflow_declare
# ---------------------------------------------------------------------------


def test_a_declared_plan_and_its_steps_sit_under_their_headings(ctx: ToolContextManager) -> None:
    tools = {t.name: t for t in load_workflow_tools(ctx)}
    result = tools["workflow_declare"].invoke({"plan": MULTI, "steps": [MULTI, "second"]})
    assert result == (
        "Workflow declared. No plan was there before.\n"
        "  Plan: first line\n"
        "      indented line\n"
        "    last line\n"
        "  Steps:\n"
        "  1. first line\n"
        "       indented line\n"
        "     last line\n"
        "  2. second"
    )


def test_a_declared_plan_and_step_with_no_text_keep_their_headings(ctx: ToolContextManager) -> None:
    tools = {t.name: t for t in load_workflow_tools(ctx)}
    result = tools["workflow_declare"].invoke({"plan": BLANKS, "steps": [BLANKS, "second"]})
    assert result == "Workflow declared. No plan was there before.\n  Plan:\n  Steps:\n  1.\n  2. second"


def test_a_replaced_plan_is_named_on_one_line(ctx: ToolContextManager) -> None:
    tools = {t.name: t for t in load_workflow_tools(ctx)}
    tools["workflow_declare"].invoke({"plan": MULTI, "steps": ["a"]})
    result = tools["workflow_declare"].invoke({"plan": "new", "steps": ["b"]})
    first_line = result.splitlines()[0]
    assert "\n" not in first_line
    assert (
        first_line
        == "Workflow declared. It replaced the plan 'first line\\n\\n  indented line\\nlast line' (1 step, 0 done, active)."
    )


# ---------------------------------------------------------------------------
# todos
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_todo_title_is_one_line_in_the_listing_and_an_empty_one_keeps_its_mark() -> None:
    storage = InMemoryTodoStorage()
    conv, user = uuid.uuid4(), uuid.uuid4()
    tools = {t.name: t for t in load_todo_tools(storage=storage, conversation_id=conv, user_id=user)}
    await storage.add(conv, user, MULTI, "Todo List", None)
    await storage.add(conv, user, BLANKS, "Todo List", None)
    result = await tools["list_todos"].ainvoke({})
    items = [line for line in result.splitlines() if line.startswith("- ")]
    assert len(items) == 2
    assert items[0].startswith("- [ ] first line indented line last line [todo:")
    assert items[1].startswith("- [ ]") and "[todo:" in items[1]


# ---------------------------------------------------------------------------
# tool lines: relevance and router
# ---------------------------------------------------------------------------


def _tool(name: str, description: str) -> StructuredTool:
    return StructuredTool.from_function(func=lambda: "ok", name=name, description=description)


def test_a_found_tool_line_indents_every_line_of_its_description() -> None:
    assert tool_hit_line(_tool("lookup", MULTI)) == ("- lookup(): first line\n    indented line\n  last line")


def test_a_found_tool_with_no_description_still_shows_its_heading() -> None:
    assert tool_hit_line(_tool("lookup", BLANKS)) == "- lookup():"


class CapturingChatModelFactory:
    """Chat model factory whose model keeps the messages it was asked."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def create_chat_model(self, purpose: str = "routing") -> Any:
        outer = self

        class _Model:
            async def ainvoke(self, messages: list[Any]) -> Any:
                outer.messages = messages

                class _Reply:
                    content = '{"tool_name": null, "reasoning": "none"}'

                return _Reply()

        return _Model()


async def _routing_prompt(tool_llms: list[dict[str, Any]], mcp_tools: list[dict[str, Any]]) -> str:
    factory = CapturingChatModelFactory()
    await ToolRouter(chat_model_factory=factory).route("look something up", tool_llms, mcp_tools)
    return str(factory.messages[0].content)


@pytest.mark.asyncio
async def test_the_routing_prompt_indents_every_line_of_a_tool_and_an_mcp_description() -> None:
    prompt = await _routing_prompt(
        [{"name": "alpha", "description": MULTI, "tool_llm_id": "t1"}],
        [{"name": "beta", "description": MULTI, "mcp_server_config_id": "m1"}],
    )
    assert (
        "- alpha: first line\n    indented line\n  last line\n- mcp:beta: first line\n    indented line\n  last line"
    ) in prompt


@pytest.mark.asyncio
async def test_the_routing_prompt_keeps_the_heading_of_a_tool_with_no_description() -> None:
    prompt = await _routing_prompt(
        [
            {"name": "alpha", "description": BLANKS, "tool_llm_id": "t1"},
            {"name": "gamma", "description": "g", "tool_llm_id": "t3"},
        ],
        [{"name": "beta", "description": None, "mcp_server_config_id": "m1"}],
    )
    assert "- alpha:\n- gamma: g\n- mcp:beta:" in prompt


# ---------------------------------------------------------------------------
# mermaid
# ---------------------------------------------------------------------------


def test_a_pie_slice_label_is_one_line() -> None:
    config = SeverityConfig(order=["high"], labels={"high": "Very\n\n  High"})
    chart = MermaidGenerator.severity_pie_chart([{"severity": "high"}], config)
    assert chart == 'pie title Finding Severity Distribution\n    "Very High" : 1'


def test_an_attack_flow_node_label_is_one_line() -> None:
    config = SeverityConfig(order=["high"], labels={"high": "High"})
    flow = MermaidGenerator.attack_flow([{"title": "SQL\n\n  injection", "severity": "high"}], config)
    assert flow == 'graph TD\n    f0["SQL injection\\n[High]"]'


def test_a_chain_flow_node_label_is_one_line() -> None:
    flow = MermaidGenerator.chain_attack_flow(
        "chain", [{"title": "SQL\n\n  injection", "severity": "high"}], "critical"
    )
    assert flow.splitlines()[1] == '    step0["SQL injection\\n[high]"]'
