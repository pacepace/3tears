"""End-to-end pause/resume proof for the subscription-CLI backend's HITL interrupt fix.

``test_claude_cli_tool_events.py::TestGraphInterruptCapture`` proves the handler-level half in
isolation (a bound tool's ``GraphInterrupt`` is captured, not swallowed into a fake tool error).
This file proves the OTHER half, against a REAL compiled LangGraph graph (not just our own code
in isolation): the graph genuinely pauses on the first turn (``__interrupt__`` present, matching
what a real caller like scriob's ``run_turn`` checks), and a genuine ``Command(resume=...)``
correctly reaches the SAME tool's own ``interrupt()`` call on replay, letting it complete for
real -- the exact round-trip a confirm-mode write tool relies on.

Only the Claude Agent SDK subprocess boundary is faked (:class:`_FakeSDKClient`, driven by a
per-test script) -- everything else (the compiled graph, the checkpointer, ``interrupt()``,
``Command(resume=...)``, this module's ``_wrap_langchain_tool``/``_astream``/``_agenerate``) is
the real thing.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from typing import Any, TypedDict
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Interrupt, interrupt
from typing_extensions import Annotated

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers._claude_cli import create_subscription_chat


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
       ``_agenerate``/``_aquery`` (which this fix's ``_agenerate`` override wraps via
       ``super()._agenerate(...)``) references ``langchain_claude_code``'s OWN module-level name,
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
    directly invoking that tool's wrapped MCP handler as part of building its scripted response.
    """

    script: Callable[[str], "list[Any] | AsyncIterator[Any]"] | None = None

    def __init__(self, options: Any) -> None:
        self._options = options
        self._prompt = ""

    async def __aenter__(self) -> "_FakeSDKClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self._prompt = prompt

    async def receive_response(self) -> AsyncIterator[Any]:
        assert _FakeSDKClient.script is not None, "test must set _FakeSDKClient.script"
        messages = await _FakeSDKClient.script(self._prompt)
        for msg in messages:
            yield msg


def _assistant_text(text: str, *, session_id: str = "fake-session") -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model=DEFAULT_CHAT_MODEL, session_id=session_id)


def _result(*, session_id: str = "fake-session") -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=session_id,
    )


class _StagesAWriteTool(BaseTool):
    """A REAL confirm-mode write tool (the same shape as scriob's ``edit_object``): calls
    ``interrupt(...)`` for real, and once resumed, acts on the human's decision for real."""

    name: str = "stage_a_write"
    description: str = "stages a write behind a HITL interrupt"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    async def _arun(self, **kwargs: Any) -> str:
        decision = interrupt({"tool": "stage_a_write", "args": kwargs, "summary": "write the thing"})
        return f"write landed (decision={decision!r})" if decision == "accept" else "write discarded (rejected)"


class _GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _compiled_graph(model: Any) -> tuple[Any, _StagesAWriteTool]:
    """A minimal one-node graph: "agent" calls the model with the tool bound; that's it. Mirrors
    the structural fact this fix relies on -- for this backend, decide-and-call live inside ONE
    model call, so there is no separate "tools" node for LangGraph to replay in isolation.

    Returns ``(compiled_graph, tool)`` -- the test needs the SAME tool instance to build the exact
    handler a real turn would dispatch through (:meth:`_wrap_langchain_tool`), simulating what the
    CLI subprocess's own internal tool-calling loop would invoke.
    """
    tool = _StagesAWriteTool()
    bound = model.bind_tools([tool])

    async def agent(state: _GraphState) -> dict[str, list[BaseMessage]]:
        result = await bound.ainvoke(state["messages"])
        return {"messages": [result]}

    builder = StateGraph(_GraphState)
    builder.add_node("agent", agent)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")
    return builder.compile(checkpointer=InMemorySaver()), tool


@pytest.mark.asyncio
async def test_the_graph_genuinely_pauses_then_resumes_and_the_tool_completes_for_real() -> None:
    # The patch MUST stay active for the entire body, through both `graph.ainvoke()` calls --
    # not just model construction (see `_no_real_sdk_calls`'s docstring for exactly why).
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        graph, tool = _compiled_graph(model)

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        schema = {"properties": {}, "required": []}

        # Turn 1 (send): the fake CLI "decides" to call the tool directly via
        # `_wrap_langchain_tool`'s own handler (bypassing the SDK's own dispatch machinery, which
        # isn't under test here) -- `interrupt()` inside the REAL tool raises for real,
        # `wrapped()` captures it, and the model's own `_agenerate` re-raises it from a point that
        # DOES sit inside this graph's own call stack.
        async def turn_one(prompt: str) -> list[Any]:
            handler = model._wrap_langchain_tool(tool, schema).handler  # noqa: SLF001 -- the real dispatch path
            await handler({})
            return [_assistant_text("Calling the write tool now."), _result()]

        _FakeSDKClient.script = turn_one
        result = await graph.ainvoke({"messages": [HumanMessage(content="please write the thing")]}, config)

        assert "__interrupt__" in result
        interrupts = result["__interrupt__"]
        assert len(interrupts) == 1
        assert isinstance(interrupts[0], Interrupt)
        assert interrupts[0].value == {"tool": "stage_a_write", "args": {}, "summary": "write the thing"}

        state = await graph.aget_state(config)
        assert state.next  # still paused -- the "agent" node's task is pending

        # Turn 2 (resume): the fake CLI "decides" to retry the SAME tool call (simulating what a
        # real model does when told explicitly a decision was made -- see
        # _messages_with_resume_hint). This time `interrupt()` resolves the cached resume value
        # instead of raising -- the tool completes for real -- no capture, no re-raise, the graph
        # runs to completion.
        async def turn_two(prompt: str) -> list[Any]:
            assert "retry the exact same tool call" in prompt  # the resume-hint message reached the model's own prompt
            handler = model._wrap_langchain_tool(tool, schema).handler  # noqa: SLF001
            tool_result = await handler({})
            assert tool_result["content"][0]["text"] == "write landed (decision='accept')"
            return [_assistant_text("Done -- the write landed."), _result()]

        _FakeSDKClient.script = turn_two
        final = await graph.ainvoke(Command(resume="accept"), config)

        assert "__interrupt__" not in final
        final_state = await graph.aget_state(config)
        assert not final_state.next  # fully completed, nothing pending


@pytest.mark.asyncio
async def test_a_rejected_resume_lets_the_tools_own_reject_branch_run() -> None:
    """The other half of the decision: a "reject" resume must reach the tool's OWN reject
    branch (never silently treated as an accept), matching how scriob's ``confirm_or_proceed``
    would gate an actual write on the decision."""
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        graph, tool = _compiled_graph(model)

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        schema = {"properties": {}, "required": []}

        async def turn_one(prompt: str) -> list[Any]:
            handler = model._wrap_langchain_tool(tool, schema).handler  # noqa: SLF001
            await handler({})
            return [_assistant_text("Calling the write tool now."), _result()]

        _FakeSDKClient.script = turn_one
        result = await graph.ainvoke({"messages": [HumanMessage(content="please write the thing")]}, config)
        assert "__interrupt__" in result

        async def turn_two(prompt: str) -> list[Any]:
            assert "retry the exact same tool call" in prompt  # the resume-hint message reached the model's own prompt
            handler = model._wrap_langchain_tool(tool, schema).handler  # noqa: SLF001
            tool_result = await handler({})
            assert tool_result["content"][0]["text"] == "write discarded (rejected)"
            return [_assistant_text("Understood -- not making that change."), _result()]

        _FakeSDKClient.script = turn_two
        final = await graph.ainvoke(Command(resume="reject"), config)

        assert "__interrupt__" not in final
        final_state = await graph.aget_state(config)
        assert not final_state.next
