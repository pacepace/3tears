"""``ToolLoopBudgetMiddleware``: a model that loops on tool calls is stopped, and the turn ends with text.

Most tests run a real ``create_agent`` graph over a scripted chat model, so what is pinned is what a
consumer gets: the turn's last message, the tools each model call was offered, and the messages it
saw. The scripted model answers each call from a function of (call index, tools offered, messages).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from pydantic import Field

from threetears.langgraph import (
    DEFAULT_MAX_TOOL_CALLS_PER_TURN,
    DEFAULT_TOOL_BUDGET_FALLBACK_TEXT,
    ToolLoopBudgetMiddleware,
    recursion_limit_for,
)

_LOGGER = "threetears.langgraph.middleware_tool_budget"

#: (call index, tool names offered, messages seen) -> the model's response
Script = Callable[[int, list[str], list[BaseMessage]], AIMessage]


@tool
def read_note(topic: str) -> str:
    """Read the note about a topic."""
    return f"the note about {topic}"


class _ScriptedModel(BaseChatModel):
    """Answers each call from ``script`` and records what every call was offered and shown."""

    script: Any
    offered: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        # A shallow copy shares ``calls``, so the bound copy records into the same list.
        return self.model_copy(update={"offered": names})

    def _respond(self, messages: list[BaseMessage]) -> ChatResult:
        index = len(self.calls)
        self.calls.append({"offered": list(self.offered), "messages": list(messages)})
        return ChatResult(generations=[ChatGeneration(message=self.script(index, list(self.offered), messages))])

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return self._respond(messages)

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return self._respond(messages)


def _call(index: int, name: str = "read_note", n: int = 1) -> AIMessage:
    """A response that makes ``n`` tool calls and writes no text."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"topic": f"t{index}-{k}"}, "id": f"c{index}-{k}"} for k in range(n)],
    )


def _looping(answer: str = "here is what I found") -> Script:
    """Calls a tool whenever it is offered one; answers only when offered none."""

    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        return _call(index) if offered else AIMessage(content=answer)

    return script


def _scripted(script: Script) -> _ScriptedModel:
    return _ScriptedModel(script=script)


async def _turn(
    model: _ScriptedModel,
    budget: ToolLoopBudgetMiddleware | None,
    *,
    recursion_limit: int | None = None,
    tools: Sequence[Any] = (read_note,),
    extra: Sequence[AgentMiddleware] = (),
) -> list[BaseMessage]:
    middleware: list[AgentMiddleware] = ([budget] if budget else []) + list(extra)
    agent = create_agent(model=model, tools=list(tools), middleware=middleware)
    limit = (
        recursion_limit
        if recursion_limit is not None
        else recursion_limit_for(
            budget.max_tool_calls if budget else DEFAULT_MAX_TOOL_CALLS_PER_TURN, middleware=middleware
        )
    )
    state = await agent.ainvoke({"messages": [HumanMessage(content="Say hello.")]}, {"recursion_limit": limit})
    return list(state["messages"])


def _tool_results(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _last_seen(call: dict[str, Any]) -> BaseMessage:
    return cast("list[BaseMessage]", call["messages"])[-1]


# --- the loop is stopped and the turn ends with text -------------------------------------------------


async def test_a_looping_model_is_stopped_by_the_budget_and_the_turn_ends_with_its_answer() -> None:
    model = _scripted(_looping())
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=5))

    assert len(_tool_results(messages)) == 5
    last = messages[-1]
    assert isinstance(last, AIMessage) and last.content == "here is what I found" and not last.tool_calls
    # five rounds with tools, then one forced answer pass offered none
    assert [bool(c["offered"]) for c in model.calls] == [True] * 5 + [False]


async def test_without_the_budget_the_same_loop_ends_on_the_recursion_error() -> None:
    with pytest.raises(GraphRecursionError):
        await _turn(_scripted(_looping()), None, recursion_limit=25)


async def test_at_the_default_cap_the_default_recursion_limit_is_not_enough_and_the_helper_is() -> None:
    """The scriob failure: LangGraph's default 25 ends a 16-call budget's loop before the budget does."""
    with pytest.raises(GraphRecursionError):
        await _turn(_scripted(_looping()), ToolLoopBudgetMiddleware(), recursion_limit=25)

    messages = await _turn(_scripted(_looping()), ToolLoopBudgetMiddleware())
    assert len(_tool_results(messages)) == DEFAULT_MAX_TOOL_CALLS_PER_TURN
    assert messages[-1].content == "here is what I found"


async def test_the_forced_pass_offers_no_tools_and_says_to_answer_from_what_it_has() -> None:
    model = _scripted(_looping())
    await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=2))

    forced = model.calls[-1]
    assert forced["offered"] == []
    instruction = _last_seen(forced)
    assert isinstance(instruction, HumanMessage)
    assert "no more tool calls" in str(instruction.content)
    assert "information you already have" in str(instruction.content)


async def test_a_tool_call_returned_by_the_forced_pass_is_dropped_so_the_turn_ends() -> None:
    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        if offered:
            return _call(index)
        return AIMessage(content="final words", tool_calls=[{"name": "read_note", "args": {}, "id": "rogue"}])

    messages = await _turn(_scripted(script), ToolLoopBudgetMiddleware(max_tool_calls=2))
    last = messages[-1]
    assert isinstance(last, AIMessage) and last.content == "final words" and last.tool_calls == []
    assert len(_tool_results(messages)) == 2


async def test_a_round_that_would_pass_the_cap_is_discarded_and_the_answer_forced() -> None:
    """Three parallel calls a round under a cap of ten: 3, 6, 9, then a round of 3 would make 12."""

    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        return _call(index, n=3) if offered else AIMessage(content="answered")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=10))
    assert len(_tool_results(messages)) == 9
    assert messages[-1].content == "answered"
    # the fourth call's three tool calls were discarded; the fifth call is the forced pass
    assert [bool(c["offered"]) for c in model.calls] == [True, True, True, True, False]


async def test_a_model_that_answers_on_its_own_is_untouched() -> None:
    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        return _call(index) if index < 2 else AIMessage(content="done")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=5))
    assert messages[-1].content == "done"
    assert all(c["offered"] == ["read_note"] for c in model.calls)
    # no nudge below warn: after the first call, each call's last message is a tool result
    assert all(isinstance(_last_seen(c), ToolMessage) for c in model.calls[1:])


# --- tiers -------------------------------------------------------------------------------------------


async def test_each_call_tier_nudges_at_its_count() -> None:
    """Cap 10: warn at 70% (7 calls), urgent at 85% (9 calls), force at the cap (10)."""
    model = _scripted(_looping())
    await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=10))

    def nudge(index: int) -> str | None:
        last = _last_seen(model.calls[index])
        return str(last.content) if isinstance(last, HumanMessage) and index > 0 else None

    assert [nudge(i) for i in range(7)] == [None] * 7
    for i in (7, 8):
        assert (n := nudge(i)) is not None and n.startswith("Start wrapping up") and f"{i} of the 10" in n
    urgent = nudge(9)
    assert urgent is not None and urgent.startswith("Stop calling tools") and "9 of the 10" in urgent
    assert model.calls[9]["offered"] == ["read_note"]
    assert model.calls[10]["offered"] == []
    assert len(model.calls) == 11


async def test_the_window_tiers_nudge_and_force_by_the_prompts_share_of_the_window() -> None:
    """Ten tokens a message in a 200-token window: after round k the prompt is 5 + 10k percent."""
    model = _scripted(_looping())
    budget = ToolLoopBudgetMiddleware(
        max_tool_calls=100, context_window_tokens=200, count_tokens=lambda msgs: 10 * len(msgs)
    )
    messages = await _turn(model, budget)

    kinds = []
    for i, call in enumerate(model.calls):
        last = _last_seen(call)
        if not call["offered"]:
            kinds.append("force")
        elif i > 0 and isinstance(last, HumanMessage):
            kinds.append("urgent" if str(last.content).startswith("Stop calling tools") else "warn")
        else:
            kinds.append(None)
    # rounds 0..6 under 70%, round 7 at 75% (warn), 8 at 85% (urgent), 9 at 95% (force)
    assert kinds == [None] * 7 + ["warn", "urgent", "force"]
    assert messages[-1].content == "here is what I found"


async def test_the_window_comes_from_the_models_profile_when_not_configured() -> None:
    model = _scripted(_looping())
    model.profile = {"max_input_tokens": 200}
    budget = ToolLoopBudgetMiddleware(max_tool_calls=100, count_tokens=lambda msgs: 10 * len(msgs))
    await _turn(model, budget)
    assert len(model.calls) == 10
    assert model.calls[-1]["offered"] == []


async def test_thresholds_and_texts_are_the_consumers_to_tune() -> None:
    model = _scripted(_looping())
    budget = ToolLoopBudgetMiddleware(
        max_tool_calls=4,
        warn_pct=25,
        urgent_pct=50,
        force_pct=90,
        warn_text="WARN {tool_calls}/{max_tool_calls}",
        urgent_text="URGENT {tool_calls}/{max_tool_calls}",
        force_text="FORCE",
        nudge_role="system",
    )
    await _turn(model, budget)
    seen = [_last_seen(c) for c in model.calls]
    assert [type(m).__name__ for m in seen[1:]] == ["SystemMessage"] * 4
    assert [m.content for m in seen[1:]] == ["WARN 1/4", "URGENT 2/4", "URGENT 3/4", "FORCE"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tool_calls": 0},
        {"warn_pct": 0},
        {"warn_pct": 90, "urgent_pct": 80},
        {"urgent_pct": 95, "force_pct": 90},
        {"force_pct": 101},
        {"runaway_rounds": 0},
        {"context_window_tokens": 0},
        {"fallback_text": "  "},
    ],
)
def test_nonsense_thresholds_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ToolLoopBudgetMiddleware(**kwargs)


# --- the empty-answer guard --------------------------------------------------------------------------


async def test_an_empty_then_reasoning_only_forced_answer_is_retried_then_falls_back() -> None:
    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        if offered:
            return _call(index)
        if index == 2:
            return AIMessage(content="")
        return AIMessage(content=[{"type": "reasoning", "reasoning": "maybe I should look again"}])

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=2))

    assert [bool(c["offered"]) for c in model.calls] == [True, True, False, False]
    retry = _last_seen(model.calls[3])
    assert "Your last reply was empty" in str(retry.content)
    last = messages[-1]
    assert isinstance(last, AIMessage) and last.content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    assert not last.tool_calls


async def test_a_reasoning_only_answer_is_retried_with_its_reasoning_and_the_retry_is_the_answer() -> None:
    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        if offered:
            return _call(index)
        if index == 1:
            return AIMessage(content="", additional_kwargs={"reasoning_content": "the note says hello"})
        return AIMessage(content="Hello!")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=1))

    retry = str(_last_seen(model.calls[2]).content)
    assert "the note says hello" in retry and "wrote no reply" in retry
    assert model.calls[2]["offered"] == []
    assert messages[-1].content == "Hello!"
    # the empty answer is not in the conversation; only the retry's answer is
    assert [m.content for m in messages if isinstance(m, AIMessage) and not m.tool_calls] == ["Hello!"]


async def test_an_empty_natural_answer_is_guarded_too() -> None:
    """No budget pressure at all: the model just answers with nothing."""

    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="") if index == 0 else AIMessage(content="Hello there.")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware())
    assert messages[-1].content == "Hello there."
    assert [c["offered"] for c in model.calls] == [["read_note"], []]


async def test_the_fallback_text_is_configurable() -> None:
    model = _scripted(lambda index, offered, messages: AIMessage(content=""))
    messages = await _turn(model, ToolLoopBudgetMiddleware(fallback_text="Sorry -- nothing to say."))
    assert messages[-1].content == "Sorry -- nothing to say."


async def test_a_failing_retry_still_ends_the_turn_with_the_fallback() -> None:
    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        if index == 0:
            return AIMessage(content="")
        raise RuntimeError("provider went away")

    messages = await _turn(_scripted(script), ToolLoopBudgetMiddleware())
    assert messages[-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT


# --- the runaway guard -------------------------------------------------------------------------------


async def test_rounds_with_no_text_whose_calls_all_fail_force_the_answer_early() -> None:
    """Calls to a tool that does not exist come back as error results; three such rounds is a runaway."""

    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        return _call(index, name="no_such_tool") if offered else AIMessage(content="I could not do that.")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=16, runaway_rounds=3))

    assert [bool(c["offered"]) for c in model.calls] == [True, True, True, False]
    assert all(m.status == "error" for m in _tool_results(messages))
    assert "Your recent tool calls failed" in str(_last_seen(model.calls[3]).content)
    assert messages[-1].content == "I could not do that."


async def test_a_round_that_writes_text_or_succeeds_resets_the_runaway_count() -> None:
    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        if index in (0, 1, 3, 4):
            return _call(index, name="no_such_tool")
        if index == 2:
            return _call(index)  # a successful round between the failing ones
        return AIMessage(content="done")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=16, runaway_rounds=3))
    assert messages[-1].content == "done"
    assert all(c["offered"] for c in model.calls)


# --- the recursion limit -----------------------------------------------------------------------------


class _BeforeModel(AgentMiddleware):
    """A hook middleware: ``create_agent`` gives it a node on every round."""

    def before_model(self, state: Any, runtime: Any) -> None:
        return None


class _AfterAgent(AgentMiddleware):
    """A once-per-run hook middleware."""

    def after_agent(self, state: Any, runtime: Any) -> None:
        return None


def test_the_helper_counts_the_hook_nodes_the_agent_adds() -> None:
    plain = recursion_limit_for(4, margin=0)
    assert plain == 1 + 5 + 4  # the input step, five model calls, four tool rounds
    assert recursion_limit_for(4, middleware=[_BeforeModel()], margin=0) == plain + 5
    assert recursion_limit_for(4, middleware=[_AfterAgent()], margin=0) == plain + 1
    assert recursion_limit_for(4, middleware=[ToolLoopBudgetMiddleware()], margin=0) == plain
    assert ToolLoopBudgetMiddleware(max_tool_calls=4).recursion_limit(margin=0) == plain
    with pytest.raises(ValueError):
        recursion_limit_for(0)


@pytest.mark.parametrize("cap", [1, 3, 6])
@pytest.mark.parametrize("extra", [(), (_BeforeModel(),), (_BeforeModel(), _AfterAgent())])
async def test_the_helpers_limit_exceeds_the_worst_turn_the_budget_allows(
    cap: int, extra: tuple[AgentMiddleware, ...]
) -> None:
    """Worst case: every round uses the cap one call at a time, then the forced answer and its retry.

    It runs at the helper's limit with no margin at all; the budget, not the recursion error, ends it.
    One step less is not enough, so the margin-free limit is the worst turn exactly.
    """

    def script(index: int, offered: list[str], messages: list[BaseMessage]) -> AIMessage:
        if offered:
            return _call(index)
        return AIMessage(content="") if index == cap else AIMessage(content="ok")

    budget = ToolLoopBudgetMiddleware(max_tool_calls=cap)
    limit = recursion_limit_for(cap, middleware=[budget, *extra], margin=0)
    messages = await _turn(_scripted(script), budget, recursion_limit=limit, extra=extra)
    assert messages[-1].content == "ok"
    assert len(_tool_results(messages)) == cap

    with pytest.raises(GraphRecursionError):
        await _turn(_scripted(script), budget, recursion_limit=limit - 1, extra=extra)


# --- observability -----------------------------------------------------------------------------------


async def test_tier_transitions_and_the_forced_answer_are_logged_with_the_count_and_no_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await _turn(_scripted(_looping("a secret answer")), ToolLoopBudgetMiddleware(max_tool_calls=10))

    records = [r for r in caplog.records if r.name == _LOGGER]
    tiers = [r.__dict__["extra_data"] for r in records if r.getMessage() == "tool-loop budget tier reached"]
    # once per transition, not once per call in a tier
    assert [(t["tier"], t["tool_calls"]) for t in tiers] == [("warn", 7), ("urgent", 9), ("force", 10)]
    [forced] = [r.__dict__["extra_data"] for r in records if r.getMessage() == "tool-loop budget forced an answer"]
    assert forced == {"cause": "tool_calls", "tool_calls": 10, "max_tool_calls": 10}
    for record in records:
        rendered = record.getMessage() + repr(record.__dict__.get("extra_data"))
        assert "Say hello" not in rendered and "note about" not in rendered and "secret" not in rendered


# --- the seam, without a graph -----------------------------------------------------------------------


def test_the_sync_seam_runs_the_same_plan() -> None:
    history: list[BaseMessage] = [HumanMessage(content="hi")]
    for i in range(3):
        call = _call(i)
        history += [call, ToolMessage(content="x", tool_call_id=call.tool_calls[0]["id"])]
    seen: list[ModelRequest] = []

    def handler(request: ModelRequest) -> Any:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="" if len(seen) == 1 else "fine")])

    request = ModelRequest(
        model=cast("BaseChatModel", _scripted(_looping())),
        messages=history,
        tools=[read_note],
        system_message=SystemMessage(content="sys"),
        state={"messages": history},
    )
    result = ToolLoopBudgetMiddleware(max_tool_calls=3).wrap_model_call(request, handler)
    assert [r.tools for r in seen] == [[], []]
    assert result.result[-1].content == "fine"
