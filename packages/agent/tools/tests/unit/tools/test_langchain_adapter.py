"""The one LangChain adapter for a TearsTool: what the model is shown, what the tool receives, what comes back.

A TearsTool run inside a LangGraph graph must behave as it does when the ToolServer dispatches it
over NATS: the model is shown the tool's own ``mcp_schema()``, the tool's ``run`` sees the raw
arguments the model sent (so its coercion runs), and a failed result reads as a failure.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid7

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
import pytest
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.builtin.calculator import CalculatorTool
from threetears.agent.tools.builtin.context_recall import ContextRecallTool
from threetears.agent.tools.builtin.current_date import CurrentDateTool, create_current_date_tool
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.langchain_adapter import to_langchain_tool


class Shot(BaseModel):
    """One camera shot."""

    prompt: str = Field(description="what the shot shows")


class StoryboardInput(BaseModel):
    shots: list[Shot] = Field(description="the shots, in order")
    tags: list[str] | None = Field(default=None, description="labels")


class _Storyboard(TearsTool):
    """A tool that records what ``execute`` received and answers with a configured result."""

    def __init__(self, result: ToolResult | None = None) -> None:
        """record what execute receives and answer with result.

        :param result: what ``execute`` answers; ``None`` answers success with the kwargs
        :ptype result: ToolResult | None
        """
        self.received: dict[str, Any] | None = None
        self._result = result

    async def execute(self, **kwargs: Any) -> ToolResult:
        """remember the input and answer with the configured result.

        :param kwargs: the tool's input, after coercion
        :ptype kwargs: Any
        :return: the configured result
        :rtype: ToolResult
        """
        self.received = kwargs
        return self._result or ToolResult(success=True, content=f"planned {len(kwargs['shots'])}", metadata={"k": 1})

    def mcp_schema(self) -> MCPToolDefinition:
        """the definition, its input schema rendered by pydantic.

        :return: the tool's definition, its input schema rendered by pydantic
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="plans shots",
            input_schema=StoryboardInput.model_json_schema(),
        )

    def mcp_name(self) -> str:
        """the canonical name.

        :return: the canonical dotted name
        :rtype: str
        """
        return "studio.storyboard"

    def mcp_version(self) -> str:
        """the version.

        :return: the version
        :rtype: str
        """
        return "1.0"


def _call(args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": "call-1", "name": "studio.storyboard", "args": args}


def _run(tool: BaseTool, args: dict[str, Any]) -> ToolMessage:
    message = asyncio.run(tool.ainvoke(_call(args)))
    assert isinstance(message, ToolMessage)
    return message


class TestWhatTheModelIsShown:
    def test_the_name_is_the_canonical_mcp_name(self) -> None:
        assert to_langchain_tool(_Storyboard()).name == "studio.storyboard"

    def test_the_description_is_the_schemas_unless_overridden(self) -> None:
        assert to_langchain_tool(_Storyboard()).description == "plans shots"
        assert to_langchain_tool(_Storyboard(), description="configured").description == "configured"

    def test_the_schema_is_the_tools_own_input_schema(self) -> None:
        """The NATS path advertises ``mcp_schema().input_schema``; so does this one -- one schema,
        nested definitions and all, rather than a second hand-written model that can drift."""
        tool = to_langchain_tool(_Storyboard())
        assert tool.args_schema == StoryboardInput.model_json_schema()

    def test_a_tool_that_brought_no_model_still_shows_its_arguments(self) -> None:
        """``current_date`` passed no ``args_schema``, so LangChain inferred one from the wrapper's
        ``**kwargs``: the model was shown a single ``kwargs`` object and never ``timezone``."""
        schema = create_current_date_tool({}, "what time is it").args_schema
        assert isinstance(schema, dict)
        assert list(schema["properties"]) == ["timezone"]


class TestWhatTheToolReceives:
    def test_loose_input_reaches_the_tools_coercion(self) -> None:
        """LangChain validated input against a pydantic ``args_schema`` before ``run`` -- and its
        coercion -- ever saw it, so a list sent as a JSON string was refused outright."""
        tool = _Storyboard()
        message = _run(to_langchain_tool(tool), {"shots": '[{"prompt": "dawn"}]', "tags": "[]"})
        assert message.status == "success"
        assert tool.received == {"shots": [{"prompt": "dawn"}], "tags": []}

    def test_nested_values_arrive_as_the_dicts_the_nats_path_delivers(self) -> None:
        tool = _Storyboard()
        _run(to_langchain_tool(tool), {"shots": [{"prompt": "dawn"}]})
        assert tool.received == {"shots": [{"prompt": "dawn"}]}

    def test_an_argument_the_model_left_out_stays_out(self) -> None:
        """A pydantic ``args_schema`` filled every omitted field with its default; the NATS path
        sends only what the model sent, and ``execute`` tells the two apart."""
        tool = _Storyboard()
        _run(to_langchain_tool(tool), {"shots": []})
        assert tool.received == {"shots": []}

    def test_a_tool_that_brought_no_model_receives_its_arguments(self) -> None:
        message = asyncio.run(
            create_current_date_tool({}, "what time is it").ainvoke(
                {"type": "tool_call", "id": "c", "name": "threetears.current_date", "args": {"timezone": "Asia/Tokyo"}}
            )
        )
        assert "Asia/Tokyo" in message.content


class TestWhatComesBack:
    def test_a_success_is_its_content_with_its_metadata_as_the_artifact(self) -> None:
        message = _run(to_langchain_tool(_Storyboard()), {"shots": [{"prompt": "a"}, {"prompt": "b"}]})
        assert message.status == "success"
        assert message.content == "planned 2"
        assert message.artifact == {"k": 1}

    def test_a_failure_reads_as_a_failure(self) -> None:
        failed = ToolResult(success=False, content="refused", error="refused")
        message = _run(to_langchain_tool(_Storyboard(failed)), {"shots": []})
        assert message.status == "error"
        assert message.content == "refused"

    def test_a_failure_keeps_its_metadata_as_the_artifact(self) -> None:
        """A refusal is exactly the case a consumer must not have to parse prose for: a failed
        search carries its typed failure record in its metadata, and the caller reads it off the
        artifact. Marking the call failed must not cost the record."""
        failed = ToolResult(success=False, content="refused", error="refused", metadata={"failure": "transport"})
        message = _run(to_langchain_tool(_Storyboard(failed)), {"shots": []})
        assert message.status == "error"
        assert message.artifact == {"failure": "transport"}
        assert message.tool_call_id == "call-1"
        assert message.name == "studio.storyboard"

    def test_a_sync_tool_call_failure_reads_as_a_failure_too(self) -> None:
        failed = ToolResult(success=False, content="", error="down", metadata={"failure": "down"})
        message = to_langchain_tool(_Storyboard(failed)).invoke(_call({"shots": []}))
        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        assert message.content == "down"
        assert message.artifact == {"failure": "down"}

    def test_a_failure_with_no_content_keeps_its_error(self) -> None:
        failed = ToolResult(success=False, content="", error="the storyboard service is down")
        message = _run(to_langchain_tool(_Storyboard(failed)), {"shots": []})
        assert message.status == "error"
        assert message.content == "the storyboard service is down"

    def test_a_failure_whose_content_differs_from_its_error_shows_both(self) -> None:
        failed = ToolResult(success=False, content="2 of 3 shots planned", error="shot 3 has no prompt")
        message = _run(to_langchain_tool(_Storyboard(failed)), {"shots": []})
        assert message.status == "error"
        assert message.content == "shot 3 has no prompt\n\n2 of 3 shots planned"

    def test_a_failure_with_no_error_field_is_its_content(self) -> None:
        failed = ToolResult(success=False, content="bad things happened")
        message = _run(to_langchain_tool(_Storyboard(failed)), {"shots": []})
        assert message.status == "error"
        assert message.content == "bad things happened"

    def test_a_failure_that_says_nothing_is_named_as_one(self) -> None:
        message = _run(to_langchain_tool(_Storyboard(ToolResult(success=False, content=""))), {"shots": []})
        assert message.status == "error"
        assert message.content == "studio.storyboard failed and gave no reason."

    def test_a_plain_invoke_answers_with_the_text(self) -> None:
        calculator = to_langchain_tool(CalculatorTool())
        assert calculator.invoke({"expression": "1 + 1"}) == "2"
        assert "[TOOL ERROR]" in calculator.invoke({"expression": "invalid!!!"})

    async def test_a_sync_invoke_from_inside_a_running_loop_completes(self) -> None:
        assert to_langchain_tool(CalculatorTool()).invoke({"expression": "6 * 7"}) == "42"


class TestRunAndArunAnswerAsInvokeDoes:
    """``run`` / ``arun`` are LangChain's classic ``AgentExecutor`` route: it calls them with the
    call's id as ``tool_call_id`` and, usually, no config. A failure answered there must keep its
    artifact as it does through ``invoke``, and a missing config is a call with no call context."""

    async def test_a_failure_through_arun_keeps_its_artifact(self) -> None:
        failed = ToolResult(success=False, content="", error="down", metadata={"failure": "down"})
        message = await to_langchain_tool(_Storyboard(failed)).arun({"shots": []}, tool_call_id="call-9")
        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        assert message.tool_call_id == "call-9"
        assert message.artifact == {"failure": "down"}

    def test_a_failure_through_run_keeps_its_artifact(self) -> None:
        failed = ToolResult(success=False, content="", error="down", metadata={"failure": "down"})
        message = to_langchain_tool(_Storyboard(failed)).run({"shots": []}, tool_call_id="call-9")
        assert isinstance(message, ToolMessage)
        assert message.artifact == {"failure": "down"}

    async def test_bare_arguments_with_no_config_answer_with_the_text(self) -> None:
        assert await to_langchain_tool(CalculatorTool()).arun({"expression": "6 * 7"}) == "42"


class _GatedStoryboard(_Storyboard):
    """a storyboard whose calls a person must approve first."""

    requires_confirmation = True


class TestTheConfirmationFlagTravels:
    """a gate reads ``requires_confirmation`` off the tool it is handed -- the aibots SDK's
    confirmation middleware by ``getattr`` -- so the wrapped tool must carry it."""

    def test_a_tool_that_requires_confirmation_still_does_once_wrapped(self) -> None:
        """wrapping must not drop the gate.

        :return: none
        :rtype: None
        """
        assert getattr(to_langchain_tool(_GatedStoryboard()), "requires_confirmation", False) is True

    def test_a_tool_that_does_not_stays_ungated(self) -> None:
        """the default carries too.

        :return: none
        :rtype: None
        """
        assert getattr(to_langchain_tool(_Storyboard()), "requires_confirmation", None) is False


# parity-exempt: the one ToolContextManager method context_recall calls, answering a fixed item by id
class _StoredResults:
    """a conversation's saved tool results, by context id."""

    def __init__(self, conversation_id: UUID) -> None:
        """hold one saved result for ``conversation_id``.

        :param conversation_id: the conversation these results belong to
        :ptype conversation_id: UUID
        :return: nothing
        :rtype: None
        """
        self.conversation_id = conversation_id

    async def get_context_item(self, context_id: str) -> dict[str, Any] | None:
        """the saved result named ``context_id``.

        :param context_id: the id from a ``[ctx:<id>]`` mark
        :ptype context_id: str
        :return: the item, or ``None`` when there is none
        :rtype: dict[str, Any] | None
        """
        return {"content": f"saved in {self.conversation_id}"} if context_id == "abc" else None


class TestAToolInAGraphRunsInTheCallsScope:
    """in a graph, the tool runs inside the same ToolCallScope the ToolServer installs, built from
    the graph config's ``call_context``; it used to run in none, so every tool that reads the call's
    identity answered as if it were outside any conversation."""

    def _factory(self, seen: list[tuple[UUID, UUID]]) -> Any:
        """a context factory recording what it was asked for.

        :param seen: where each ``(conversation_id, user_id)`` asked for is recorded
        :ptype seen: list[tuple[UUID, UUID]]
        :return: the factory
        :rtype: Any
        """

        async def _factory(conversation_id: UUID, user_id: UUID) -> Any:
            seen.append((conversation_id, user_id))
            return _StoredResults(conversation_id)

        return _factory

    async def test_context_recall_reads_the_conversation_the_call_names(self) -> None:
        """the conversation comes from the config's call context, through the host's factory.

        :return: none
        :rtype: None
        """
        conversation, user = uuid7(), uuid7()
        seen: list[tuple[UUID, UUID]] = []
        tool = to_langchain_tool(ContextRecallTool(), context_factory=self._factory(seen))
        config = {"configurable": {"call_context": CallContext(conversation_id=conversation, user_id=user)}}

        message = await tool.ainvoke(
            {"type": "tool_call", "id": "r1", "name": tool.name, "args": {"context_id": "abc"}}, config
        )

        assert message.status == "success"
        assert message.content == f"saved in {conversation}"
        assert seen == [(conversation, user)]

    async def test_current_date_reads_the_callers_timezone(self) -> None:
        """the per-user timezone rides the call context, as over NATS.

        :return: none
        :rtype: None
        """
        tool = to_langchain_tool(CurrentDateTool())
        config = {"configurable": {"call_context": CallContext(user_timezone="Asia/Tokyo")}}
        message = await tool.ainvoke({"type": "tool_call", "id": "d1", "name": tool.name, "args": {}}, config)
        assert "Asia/Tokyo" in message.content

    def test_the_sync_path_runs_in_the_scope_too(self) -> None:
        """a sync invoke builds and enters the same scope.

        :return: none
        :rtype: None
        """
        tool = to_langchain_tool(CurrentDateTool())
        config = {"configurable": {"call_context": CallContext(user_timezone="Asia/Tokyo")}}
        message = tool.invoke({"type": "tool_call", "id": "d2", "name": tool.name, "args": {}}, config)
        assert "Asia/Tokyo" in message.content

    async def test_no_call_context_names_what_is_missing(self) -> None:
        """a tool that needs the call's conversation fails naming the config key, not "unavailable".

        :return: none
        :rtype: None
        """
        tool = to_langchain_tool(ContextRecallTool(), context_factory=self._factory([]))
        message = await tool.ainvoke(
            {"type": "tool_call", "id": "r2", "name": tool.name, "args": {"context_id": "abc"}}, {"configurable": {}}
        )
        assert message.status == "error"
        assert "config['configurable']['call_context']" in message.content

    async def test_a_tool_that_needs_no_scope_works_without_one(self) -> None:
        """the calculator reads nothing from the call; no call context is needed.

        :return: none
        :rtype: None
        """
        message = await to_langchain_tool(CalculatorTool()).ainvoke(
            {"type": "tool_call", "id": "c9", "name": "threetears.calculator", "args": {"expression": "2 + 2"}}
        )
        assert message.status == "success"
        assert message.content == "4"

    async def test_a_call_context_of_the_wrong_type_is_refused_by_name(self) -> None:
        """a dict where a CallContext belongs is a host bug, said so rather than run as no scope.

        :return: none
        :rtype: None
        """
        tool = to_langchain_tool(CurrentDateTool())
        with pytest.raises(TypeError, match="call_context"):
            await tool.ainvoke(
                {"type": "tool_call", "id": "d3", "name": tool.name, "args": {}},
                {"configurable": {"call_context": {"user_timezone": "Asia/Tokyo"}}},
            )
