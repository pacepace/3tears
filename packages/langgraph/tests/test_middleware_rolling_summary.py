"""``RollingSummaryMiddleware``: fold older messages into a rolling summary without deleting history.

The existing ``SummarizationMiddleware`` rewrites the checkpointed messages (``RemoveMessage``) and
triggers on a message count. Scriob and metallm each hand-rolled the other shape -- keep the full
history in the checkpointer, trim only what the model sees, trigger on a token budget, and persist a
rolling summary with a cursor -- and their cursors disagreed (a count vs a message id). These tests
pin the one shared implementation.

Token counting is injected as a character count so every budget below is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from threetears.langgraph import NOSTREAM_TAG, RollingSummaryMiddleware, SummaryState, SummaryStore
from threetears.langgraph.middleware_rolling_summary import USAGE_PURPOSE_METADATA_KEY


class _Summarizer(BaseChatModel):
    """Answers every summarization call with a numbered summary and records what it was sent."""

    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "summarizer"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:  # pragma: no cover - async only
        raise NotImplementedError

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # noqa: A002
        self.calls.append({"messages": list(input), "config": dict(config or {})})
        return AIMessage(content=f"SUMMARY {len(self.calls)}")

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:  # pragma: no cover
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])


class _FakeSummaryStore(SummaryStore):
    """An in-memory store; ``lose_next_save`` makes the next save lose its compare-and-swap."""

    def __init__(self, state: SummaryState | None = None) -> None:
        self.state = state
        self.saves: list[SummaryState] = []
        self.winner: SummaryState | None = None

    async def load(self) -> SummaryState | None:
        return self.state

    async def save(self, state: SummaryState, *, expected: SummaryState | None) -> bool:
        if self.winner is not None:
            self.state, self.winner = self.winner, None
            return False
        if self.state != expected:
            return False
        self.state = state
        self.saves.append(state)
        return True


def _chars(messages: Sequence[BaseMessage]) -> int:
    return sum(len(str(m.content)) for m in messages)


def _h(i: int, text: str = "hello there") -> HumanMessage:
    return HumanMessage(content=text, id=f"m{i}")


def _a(i: int, text: str = "general kenobi") -> AIMessage:
    return AIMessage(content=text, id=f"m{i}")


def _history(n: int) -> list[BaseMessage]:
    """``n`` alternating human/assistant messages ending on a human one (a turn's start)."""
    out: list[BaseMessage] = []
    for i in range(n):
        out.append(_h(i) if (n - 1 - i) % 2 == 0 else _a(i))
    return out


async def _run(
    middleware: RollingSummaryMiddleware, messages: list[BaseMessage]
) -> tuple[list[BaseMessage], ModelRequest]:
    seen: dict[str, Any] = {}

    async def handler(request: ModelRequest) -> Any:
        seen["request"] = request
        return SimpleNamespace(result=[AIMessage(content="ok")])

    request = ModelRequest(model=cast("BaseChatModel", _Summarizer()), messages=messages)
    await middleware.awrap_model_call(request, handler)
    return list(seen["request"].messages), request


def _middleware(store: SummaryStore, model: _Summarizer, *, budget: int, **kwargs: Any) -> RollingSummaryMiddleware:
    return RollingSummaryMiddleware(model, store=store, token_budget=budget, count_tokens=_chars, **kwargs)


async def test_under_budget_the_request_is_untouched_and_nothing_is_summarized() -> None:
    store, model = _FakeSummaryStore(), _Summarizer()
    history = _history(3)
    sent, _ = await _run(_middleware(store, model, budget=1000), history)
    assert sent == history
    assert model.calls == [] and store.saves == []


async def test_over_budget_the_model_sees_a_summary_and_the_history_is_kept() -> None:
    store, model = _FakeSummaryStore(), _Summarizer()
    history = _history(7)
    original = list(history)
    sent, request = await _run(_middleware(store, model, budget=40), history)

    assert isinstance(sent[0], HumanMessage) and "SUMMARY 1" in str(sent[0].content)
    kept = sent[1:]
    assert kept == history[-len(kept) :] and _chars(kept) <= 40
    assert list(request.messages) == original, "the checkpointed history is never rewritten"
    [saved] = store.saves
    assert saved.text == "SUMMARY 1"
    assert saved.through_id == history[-len(kept) - 1].id, "the cursor is the last folded message's id"
    assert saved.through_count == len(history) - len(kept)


async def test_the_summary_call_is_hidden_from_the_stream_and_billed_as_summarization() -> None:
    store, model = _FakeSummaryStore(), _Summarizer()
    await _run(_middleware(store, model, budget=40), _history(7))
    [call] = model.calls
    assert NOSTREAM_TAG in call["config"]["tags"]
    assert call["config"]["metadata"][USAGE_PURPOSE_METADATA_KEY] == "summarization"


async def test_a_second_fold_includes_the_prior_summary_and_only_newer_messages() -> None:
    # Distinct contents, so the transcript shows exactly which messages were folded.
    history: list[BaseMessage] = [
        (_h(i, f"human-{i:02d}") if i % 2 == 0 else _a(i, f"assistant-{i:02d}")) for i in range(9)
    ]
    store = _FakeSummaryStore(SummaryState(text="EARLIER", through_id="m2", through_count=3))
    model = _Summarizer()
    # Each message is 8 or 12 characters: a 30-character budget keeps m6..m8 and folds m3..m5.
    sent, _ = await _run(_middleware(store, model, budget=30), history)
    [call] = model.calls
    transcript = " ".join(str(m.content) for m in call["messages"])
    assert "EARLIER" in transcript, "the prior summary is folded into the new one"
    for i in range(3):
        assert f"-{i:02d}" not in transcript, f"m{i} was already folded and must not be folded again"
    for i in range(3, 6):
        assert f"-{i:02d}" in transcript, f"m{i} is newer than the cursor and over budget"
    assert store.state == SummaryState(text="SUMMARY 1", through_id="m5", through_count=6)
    assert sent[1:] == history[6:]


async def test_a_mid_turn_step_trims_by_the_cursor_but_does_not_summarize_again() -> None:
    history = [*_history(5), AIMessage(content="", id="m5", tool_calls=[{"name": "t", "args": {}, "id": "c1"}])]
    history.append(ToolMessage(content="result", tool_call_id="c1", id="m6"))
    store = _FakeSummaryStore(SummaryState(text="EARLIER", through_id="m2", through_count=3))
    model = _Summarizer()
    sent, _ = await _run(_middleware(store, model, budget=10), history)
    assert model.calls == [], "only a turn's first model call may summarize"
    assert "EARLIER" in str(sent[0].content)
    assert sent[1:] == history[3:]


async def test_a_cursor_missing_from_history_falls_back_to_the_whole_window() -> None:
    history = _history(3)
    store = _FakeSummaryStore(SummaryState(text="EARLIER", through_id="gone", through_count=40))
    model = _Summarizer()
    sent, _ = await _run(_middleware(store, model, budget=1000), history)
    assert "EARLIER" in str(sent[0].content)
    assert sent[1:] == history, "the summary covers messages no longer present; everything here is new"


async def test_id_less_messages_fall_back_to_the_count_cursor() -> None:
    history: list[BaseMessage] = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]
    store = _FakeSummaryStore(SummaryState(text="EARLIER", through_id=None, through_count=2))
    sent, _ = await _run(_middleware(store, _Summarizer(), budget=1000), history)
    assert sent[1:] == history[2:]


async def test_a_lost_compare_and_swap_uses_the_winners_summary_without_summarizing_twice() -> None:
    store, model = _FakeSummaryStore(), _Summarizer()
    history = _history(7)
    store.winner = SummaryState(text="WINNER", through_id="m1", through_count=2)
    sent, _ = await _run(_middleware(store, model, budget=40), history)
    assert len(model.calls) == 1
    assert "WINNER" in str(sent[0].content)
    assert sent[1:] == history[2:]


async def test_the_kept_tail_never_starts_with_an_orphaned_tool_result() -> None:
    long = "x" * 30
    history: list[BaseMessage] = [
        _h(0, long),
        AIMessage(content="call", id="m1", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage(content="tool output", tool_call_id="c1", id="m2"),
        _a(3, "done"),
        _h(4, "next"),
    ]
    store, model = _FakeSummaryStore(), _Summarizer()
    # By size alone the budget keeps [tool result, "done", "next"] -- a tail opening on a tool result
    # whose requesting AIMessage was folded away, which providers reject.
    sent, _ = await _run(_middleware(store, model, budget=len("tool output") + len("done") + len("next")), history)
    assert not isinstance(sent[1], ToolMessage)
    assert sent[1:] == history[3:], "the orphaned tool result is folded with the call that requested it"


async def test_a_leading_system_message_stays_at_the_head() -> None:
    store, model = _FakeSummaryStore(), _Summarizer()
    history: list[BaseMessage] = [SystemMessage(content="be brief", id="sys"), *_history(7)]
    sent, _ = await _run(_middleware(store, model, budget=40), history)
    assert isinstance(sent[0], SystemMessage) and sent[0].content == "be brief"
    assert "SUMMARY 1" in str(sent[1].content)


async def test_on_summarized_hears_each_fold() -> None:
    heard: list[tuple[int, str]] = []

    async def on_summarized(folded: int, text: str) -> None:
        heard.append((folded, text))

    store, model = _FakeSummaryStore(), _Summarizer()
    await _run(_middleware(store, model, budget=40, on_summarized=on_summarized), _history(7))
    assert heard and heard[0][1] == "SUMMARY 1" and heard[0][0] == store.saves[0].through_count


def test_the_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="token_budget"):
        RollingSummaryMiddleware(_Summarizer(), store=_FakeSummaryStore(), token_budget=0)


def test_summary_state_is_immutable() -> None:
    state = SummaryState(text="t", through_id="m1", through_count=2)
    assert replace(state, text="u").text == "u"
    with pytest.raises(AttributeError):
        state.text = "v"  # type: ignore[misc]


class _Failing(_Summarizer):
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # noqa: A002
        self.calls.append({"messages": list(input), "config": dict(config or {})})
        raise ConnectionError("provider down")


async def test_a_failed_summary_keeps_the_prior_one_and_saves_nothing() -> None:
    """The fallback summary keeps only assistant sentences; storing it would erase the prior summary."""
    history = _history(9)
    prior = SummaryState(text="PRIOR: the user is Alice, budget $5000", through_id="m2", through_count=3)
    store = _FakeSummaryStore(prior)
    sent, _ = await _run(_middleware(store, _Failing(), budget=30), history)
    assert store.state == prior and store.saves == []
    assert "PRIOR: the user is Alice" in str(sent[0].content)
    assert sent[1:] == history[3:], "trimmed by the old cursor; over budget for one turn, nothing lost"


async def test_an_empty_summary_is_never_stored() -> None:
    class _Blank(_Summarizer):
        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:  # noqa: A002
            return AIMessage(content="   ")

    store = _FakeSummaryStore()
    await _run(_middleware(store, _Blank(), budget=40), _history(7))
    assert store.saves == []


async def test_a_count_cursor_at_or_past_the_window_keeps_the_newest_message() -> None:
    history: list[BaseMessage] = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c NEW")]
    store = _FakeSummaryStore(SummaryState(text="EARLIER", through_id=None, through_count=10))
    sent, _ = await _run(_middleware(store, _Summarizer(), budget=1000), history)
    assert sent[1:] == history, "a cursor past the window means the history was trimmed; all of it is new"


async def test_a_store_can_be_resolved_per_call() -> None:
    """An app that compiles its agent once serves many conversations through one middleware."""
    stores = {"t1": _FakeSummaryStore(), "t2": _FakeSummaryStore()}

    def store_for(request: ModelRequest) -> SummaryStore | None:
        return stores[request.state["thread"]]  # type: ignore[index]

    middleware = RollingSummaryMiddleware(_Summarizer(), store_for=store_for, token_budget=40, count_tokens=_chars)
    for thread in ("t1", "t2"):
        seen: dict[str, Any] = {}

        async def handler(request: ModelRequest) -> Any:
            seen["request"] = request
            return SimpleNamespace(result=[AIMessage(content="ok")])

        request = ModelRequest(
            model=cast("BaseChatModel", _Summarizer()), messages=_history(7), state={"thread": thread}
        )
        await middleware.awrap_model_call(request, handler)
    assert len(stores["t1"].saves) == 1 and len(stores["t2"].saves) == 1


def test_exactly_one_of_store_and_store_for_is_given() -> None:
    with pytest.raises(ValueError, match="store"):
        RollingSummaryMiddleware(_Summarizer(), token_budget=10)
    with pytest.raises(ValueError, match="store"):
        RollingSummaryMiddleware(_Summarizer(), store=_FakeSummaryStore(), store_for=lambda _r: None, token_budget=10)


async def test_a_failing_on_summarized_does_not_fail_the_turn() -> None:
    async def on_summarized(folded: int, text: str) -> None:
        raise RuntimeError("listener broke")

    store = _FakeSummaryStore()
    sent, _ = await _run(_middleware(store, _Summarizer(), budget=40, on_summarized=on_summarized), _history(7))
    assert store.saves and "SUMMARY 1" in str(sent[0].content)
