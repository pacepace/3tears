"""A subscription model hands tool calls back, and the caller's graph runs them.

Under a subscription the CLI used to run bound tools itself, so a caller's graph never saw a call:
no approval, no ledger, no shaping, no loading tools mid-turn (see ``_claude_cli``'s module
docstring). These pin the standard contract against a REAL compiled LangGraph graph with a real
``ToolNode`` -- the model asks, the graph's tool node runs the tool, an approval ``interrupt()`` in
the tool pauses the graph and a ``Command(resume=...)`` continues it, and the next model call reads
the result as history.

Only the Claude Agent SDK subprocess boundary is faked (:class:`_FakeSDKClient`, driven by a
per-test script); what it returns is what the bundled CLI was measured to send at
``--max-turns 1``: the tool-use blocks, then ``error_max_turns``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from typing import Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, Interrupt, interrupt
from typing_extensions import Annotated

# MUST come before any `claude_agent_sdk`/`langchain_claude_code` import (module-level or not) --
# CI's default install doesn't include the optional `claude-cli` extra, so those modules are
# absent there; a hard `from claude_agent_sdk import ...` above this line would fail collection
# outright instead of skipping gracefully.
pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL  # noqa: E402
from threetears.models.providers._claude_cli import create_subscription_chat  # noqa: E402


@pytest.fixture(autouse=True)
def _one_off_cli_per_call() -> Any:
    """These pins drive the one-off-client path on purpose, with a fake client that only works as a
    context manager. Pooling is turned off explicitly so that path is chosen, rather than reached by a
    pooled start failing and falling back. The pooled path is pinned in ``test_claude_cli_pooled_chat.py``."""
    from threetears.models import claude_cli_pool

    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    yield
    claude_cli_pool.configure_claude_cli_pool(enabled=True)


def _tripwire_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
    raise AssertionError(
        "the REAL ClaudeSDKClient.__init__ ran -- a mock-patch binding was missed or the patch "
        "context closed before an ainvoke()/astream() call it needed to cover. This must NEVER "
        "fire in CI: it exists so a regression here fails loudly and for free, instead of quietly "
        "placing a real, billed Claude Agent SDK call the way an earlier version of this test did."
    )


@contextmanager
def _no_real_sdk_calls():
    """Patch EVERY binding of ``ClaudeSDKClient`` a call through this backend can reach, so a bug
    in this fix can never accidentally place a real, billed Claude Agent SDK call -- MUST stay
    active for the full duration of any code that might call ``.ainvoke()``/``.astream()``, not
    just model construction (see the tripwire's message and the note below for why).

    Three layers, not one:

    1. ``claude_agent_sdk.ClaudeSDKClient`` -- this module's own ``_astream`` override does a
       FRESH ``from claude_agent_sdk import ClaudeSDKClient`` inside ``_subscription_model_cls()``
       every time a model is constructed; patching this attribute is read at THAT moment and
       captured into ``_astream``'s closure, so (for this one path only) the patch only needs to
       be active during model construction.
    2. ``langchain_claude_code.claude_chat_model.ClaudeSDKClient`` -- the BASE class's
       ``_agenerate`` reaches ``_aquery``; were ``_aquery`` ever the base class's again, it references ``langchain_claude_code``'s OWN module-level name,
       re-resolved via a plain global lookup EVERY CALL -- NOT closure-captured. The patch must
       still be active when ``.ainvoke()`` actually runs, not just at construction time. Missing
       this distinction is exactly how an earlier version of this test placed five real, billed
       API calls before this comment was written: the patch context was closed right after model
       construction, which is safe for (1) but not for (2).
    3. A tripwire on the REAL class's own ``__init__`` (:func:`_tripwire_init`) -- defense in
       depth. If some FOURTH binding surfaces in a future ``langchain-claude-code``/
       ``claude-agent-sdk`` release that (1) and (2) don't cover, this raises immediately instead
       of silently reaching the network.
    """
    with (
        patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
        patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
        patch("claude_agent_sdk.client.ClaudeSDKClient.__init__", _tripwire_init),
    ):
        yield


# Deliberately partial: only the two methods this backend's `_astream`/`_aquery` actually call
# (`query`, `receive_response`, plus the async-context-manager protocol) are faked.
# `ClaudeSDKClient`'s other public methods (connect/disconnect/interrupt/set_model/... -- 13 of
# them) are interactive-session controls no code path under test here ever calls.
# parity-exempt: intentionally implements only query/receive_response -- see comment above
class _FakeSDKClient:
    """Stands in for ``claude_agent_sdk.ClaudeSDKClient``. Ignores the real subprocess entirely --
    driven by a test-supplied ``script(prompt) -> list[message]`` callable that decides what to
    "say" for each call, letting a test simulate the CLI deciding to call (or retry) a tool by
    returning the tool-use blocks it would.
    """

    script: Callable[[str], "list[Any] | AsyncIterator[Any]"] | None = None
    prompts: list[str] = []
    options: list[Any] = []

    def __init__(self, options: Any) -> None:
        self._options = options
        _FakeSDKClient.prompts.append("")
        self._prompt = ""

    async def __aenter__(self) -> "_FakeSDKClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self._prompt = prompt
        _FakeSDKClient.prompts[-1] = prompt
        _FakeSDKClient.options.append(self._options)

    async def receive_response(self) -> AsyncIterator[Any]:
        assert _FakeSDKClient.script is not None, "test must set _FakeSDKClient.script"
        messages = await _FakeSDKClient.script(self._prompt)
        for msg in messages:
            yield msg


def _assistant_text(text: str, *, session_id: str = "fake-session") -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model=DEFAULT_CHAT_MODEL, session_id=session_id)


def _result(
    *, session_id: str = "fake-session", subtype: str = "success", usage: dict[str, Any] | None = None
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=subtype != "success",
        num_turns=2 if subtype == "error_max_turns" else 1,
        session_id=session_id,
        usage=usage,
    )


def _asks_for(*calls: tuple[str, str, dict[str, Any]], text: str = "") -> list[Any]:
    """What the CLI sends when the model asks for tools: the tool uses, then the turn limit."""
    blocks: list[Any] = [TextBlock(text=text)] if text else []
    blocks += [
        ToolUseBlock(id=call_id, name=f"mcp__langchain-tools__{wire}", input=args) for call_id, wire, args in calls
    ]
    return [AssistantMessage(content=blocks, model=DEFAULT_CHAT_MODEL), _result(subtype="error_max_turns")]


class _StagesAWriteTool(BaseTool):
    """A confirm-mode write tool: calls ``interrupt(...)`` for real, then acts on the decision."""

    name: str = "threetears.stage_a_write"
    description: str = "stages a write behind a HITL interrupt"
    ran: int = 0

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    async def _arun(self, **kwargs: Any) -> str:
        decision = interrupt({"tool": self.name, "args": kwargs})
        self.ran += 1
        return f"write landed ({kwargs.get('path')})" if decision == "accept" else "write discarded"


class _GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _compiled_graph(model: Any, tool: BaseTool) -> Any:
    """The ordinary agent loop: the model asks, the tool node runs, back to the model."""
    bound = model.bind_tools([tool])

    async def agent(state: _GraphState) -> dict[str, list[BaseMessage]]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    builder = StateGraph(_GraphState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([tool]))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


@pytest.fixture(autouse=True)
def _fresh_script() -> Any:
    _FakeSDKClient.script = None
    _FakeSDKClient.prompts = []
    _FakeSDKClient.options = []
    yield


def _replies(*scripted: list[Any]) -> Callable[[str], Any]:
    queue = list(scripted)

    async def script(prompt: str) -> list[Any]:
        return queue.pop(0)

    return script


async def test_the_graph_runs_the_tool_pauses_for_approval_and_the_model_reads_the_result() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        tool = _StagesAWriteTool()
        graph = _compiled_graph(model, tool)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        _FakeSDKClient.script = _replies(
            _asks_for(("tu-1", "threetears_stage_a_write", {"path": "notes.md"}), text="Writing it."),
            [AssistantMessage(content=[TextBlock(text="Done.")], model=DEFAULT_CHAT_MODEL), _result()],
        )

        paused = await graph.ainvoke({"messages": [HumanMessage(content="write notes.md")]}, config)

        [pending] = paused["__interrupt__"]
        assert isinstance(pending, Interrupt)
        assert pending.value == {"tool": "threetears.stage_a_write", "args": {"path": "notes.md"}}
        assert tool.ran == 0, "the tool ran before anyone approved it"
        assert len(_FakeSDKClient.prompts) == 1, "the model was asked again before the tool had a result"

        final = await graph.ainvoke(Command(resume="accept"), config)

        assert "__interrupt__" not in final
        assert tool.ran == 1
        assert final["messages"][-1].content == "Done."
        assert "Tool (threetears.stage_a_write): write landed (notes.md)" in _FakeSDKClient.prompts[-1]
        assert "[Tool calls: threetears.stage_a_write({'path': 'notes.md'})]" in _FakeSDKClient.prompts[-1]


async def test_a_rejected_approval_reaches_the_tools_own_reject_branch() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        tool = _StagesAWriteTool()
        graph = _compiled_graph(model, tool)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        _FakeSDKClient.script = _replies(
            _asks_for(("tu-1", "threetears_stage_a_write", {"path": "notes.md"})),
            [AssistantMessage(content=[TextBlock(text="Left it alone.")], model=DEFAULT_CHAT_MODEL), _result()],
        )
        await graph.ainvoke({"messages": [HumanMessage(content="write notes.md")]}, config)
        await graph.ainvoke(Command(resume="reject"), config)
        assert "Tool (threetears.stage_a_write): write discarded" in _FakeSDKClient.prompts[-1]


async def test_parallel_tool_calls_come_back_together_under_the_names_the_caller_bound() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_StagesAWriteTool()])
        _FakeSDKClient.script = _replies(
            _asks_for(
                ("tu-1", "threetears_stage_a_write", {"path": "a"}),
                ("tu-2", "threetears_stage_a_write", {"path": "b"}),
            )
        )
        message = await bound.ainvoke([HumanMessage(content="write a and b")])

    assert isinstance(message, AIMessage)
    assert [(c["id"], c["name"], c["args"]) for c in message.tool_calls] == [
        ("tu-1", "threetears.stage_a_write", {"path": "a"}),
        ("tu-2", "threetears.stage_a_write", {"path": "b"}),
    ]
    assert message.response_metadata["finish_reason"] == "tool_calls"
    assert message.response_metadata["is_error"] is False, "stopping to hand calls back is not a failure"
    assert "internal_tool_calls" not in message.response_metadata


async def test_streaming_hands_the_calls_back_on_the_last_chunk() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_StagesAWriteTool()])
        _FakeSDKClient.script = _replies(_asks_for(("tu-1", "threetears_stage_a_write", {"path": "a"}), text="On it."))
        merged: Any = None
        async for chunk in bound.astream([HumanMessage(content="write a")]):
            merged = chunk if merged is None else merged + chunk

    assert merged.content == "On it."
    assert [(c["name"], c["args"]) for c in merged.tool_calls] == [("threetears.stage_a_write", {"path": "a"})]


async def test_the_turn_limit_without_a_tool_call_is_still_a_failure() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        _FakeSDKClient.script = _replies(
            [
                AssistantMessage(content=[TextBlock(text="hm")], model=DEFAULT_CHAT_MODEL),
                _result(subtype="error_max_turns"),
            ]
        )
        message = await model.ainvoke([HumanMessage(content="hi")])

    assert message.response_metadata["is_error"] is True
    assert message.response_metadata["finish_reason"] == "error"


async def test_every_call_is_one_model_turn_whatever_the_caller_asked_for() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest", max_turns=12)
        _FakeSDKClient.script = _replies(
            [AssistantMessage(content=[TextBlock(text="hi")], model=DEFAULT_CHAT_MODEL), _result()]
        )
        await model.ainvoke([HumanMessage(content="hi")])

    assert _FakeSDKClient.options[0].max_turns == 1


async def test_usage_is_reported_the_way_every_chat_model_reports_it() -> None:
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200,
        "output_tokens": 30,
    }
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        _FakeSDKClient.script = _replies(
            [AssistantMessage(content=[TextBlock(text="hi")], model=DEFAULT_CHAT_MODEL), _result(usage=usage)],
            [AssistantMessage(content=[TextBlock(text="hi")], model=DEFAULT_CHAT_MODEL), _result(usage=usage)],
        )
        invoked = await model.ainvoke([HumanMessage(content="hi")])
        streamed: Any = None
        async for chunk in model.astream([HumanMessage(content="hi")]):
            streamed = chunk if streamed is None else streamed + chunk

    for message in (invoked, streamed):
        assert message.usage_metadata == {
            "input_tokens": 1210,
            "output_tokens": 30,
            "total_tokens": 1240,
            "input_token_details": {"cache_read": 1000, "cache_creation": 200},
        }


async def test_a_tool_result_without_a_name_is_named_by_the_call_it_answers() -> None:
    from langchain_core.messages import ToolMessage

    model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
    query, _system = model._convert_messages(  # noqa: SLF001 -- the method under test
        [
            HumanMessage(content="write a"),
            AIMessage(
                content="", tool_calls=[{"id": "tu-1", "name": "threetears.stage_a_write", "args": {"path": "a"}}]
            ),
            ToolMessage(content="write landed", tool_call_id="tu-1"),
        ]
    )
    assert "Tool (threetears.stage_a_write): write landed" in query
