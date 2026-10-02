"""Unit tests for :mod:`threetears.langgraph.summarize`.

Pure-logic, no infra: a stub chat model drives the contracts the summarizer must hold --

* a model that returns a summary -> that summary is returned;
* a model that raises, times out, or answers with nothing -> :class:`SummarizationFailedError`,
  chained to the cause and logged once. Never a string: a caller that stores the summary as the
  conversation's narrative cannot tell a stand-in sentence from a real summary, and a consumer
  stored "The earlier part of this conversation could not be summarized." as exactly that;
* a cancellation is not a failure and propagates untouched;
* an over-long summary is truncated to the 2000-character cap.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from threetears.langgraph import SummarizationFailedError as ExportedSummarizationFailedError
from threetears.langgraph.summarize import (
    SummarizationFailedError,
    summarize_older_messages,
)

#: the documented summary cap, asserted as a number so a change to it is a decision made here.
_SUMMARY_CAP = 2000


class _StubModel(BaseChatModel):
    """A minimal chat model that returns a fixed reply (or raises ``raises``) on ``ainvoke``.

    ``BaseChatModel`` is abstract; we only exercise ``ainvoke`` here, so the sync
    ``_generate`` / ``_llm_type`` hooks are stubbed to satisfy the ABC without
    being called.
    """

    reply: str = ""
    raises: BaseException | None = None

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        if self.raises is not None:
            raise self.raises
        return AIMessage(content=self.reply)


_OLDER: list[BaseMessage] = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris. It sits on the Seine."),
]


async def test_returns_model_summary() -> None:
    """When the model returns a summary, that text is returned (stripped)."""
    model = _StubModel(reply="  The user asked about France; Paris was confirmed.  ")
    summary = await summarize_older_messages(_OLDER, model)
    assert summary == "The user asked about France; Paris was confirmed."


async def test_a_model_error_raises_a_typed_failure_chained_to_its_cause() -> None:
    """A failed model call is never answered with text: the caller gets the typed failure."""
    cause = RuntimeError("provider unavailable")
    model = _StubModel(raises=cause)
    with pytest.raises(SummarizationFailedError) as caught:
        await summarize_older_messages(_OLDER, model)
    assert caught.value.__cause__ is cause


async def test_a_model_timeout_raises_a_typed_failure_chained_to_its_cause() -> None:
    """A model call that ran out of time is a failure, never a summary."""
    cause = TimeoutError("summary call timed out")
    model = _StubModel(raises=cause)
    with pytest.raises(SummarizationFailedError) as caught:
        await summarize_older_messages(_OLDER, model)
    assert caught.value.__cause__ is cause


async def test_a_deadline_around_the_call_surfaces_as_a_typed_failure() -> None:
    """A model that hangs past an ``asyncio.timeout`` inside the model wrapper is still the typed failure."""

    class _Hangs(_StubModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
            async with asyncio.timeout(0.01):
                await asyncio.sleep(10)
            return AIMessage(content="never")  # pragma: no cover - the deadline fires first

    with pytest.raises(SummarizationFailedError) as caught:
        await summarize_older_messages(_OLDER, _Hangs())
    assert isinstance(caught.value.__cause__, TimeoutError)


@pytest.mark.parametrize("reply", ["", "   \n  "])
async def test_an_empty_answer_is_a_failure_not_a_summary(reply: str) -> None:
    """An empty summary stands in for the messages it replaces as surely as a placeholder does."""
    model = _StubModel(reply=reply)
    with pytest.raises(SummarizationFailedError) as caught:
        await summarize_older_messages(_OLDER, model)
    assert caught.value.__cause__ is None


async def test_no_placeholder_text_is_ever_returned() -> None:
    """The failure is not a return value of any kind -- not the old sentence, not a heuristic."""
    model = _StubModel(raises=ConnectionError("down"))
    returned: object = None
    with pytest.raises(SummarizationFailedError):
        returned = await summarize_older_messages([HumanMessage(content="hello")], model)
    assert returned is None


async def test_cancellation_propagates_untouched() -> None:
    """A cancelled turn is not a failed summary: ``CancelledError`` is never wrapped."""
    model = _StubModel(raises=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await summarize_older_messages(_OLDER, model)


async def test_the_failure_is_logged_once_with_its_cause(caplog: pytest.LogCaptureFixture) -> None:
    """The point of failure logs it, once, with the model's own exception attached."""
    model = _StubModel(raises=RuntimeError("provider unavailable"))
    with (
        caplog.at_level(logging.WARNING, logger="threetears.langgraph.summarize"),
        pytest.raises(SummarizationFailedError),
    ):
        await summarize_older_messages(_OLDER, model)
    records = [r for r in caplog.records if r.name == "threetears.langgraph.summarize"]
    assert len(records) == 1
    assert records[0].exc_info is not None and isinstance(records[0].exc_info[1], RuntimeError)


def test_the_failure_is_exported_from_the_package() -> None:
    """Callers import the error from where they import the function."""
    assert ExportedSummarizationFailedError is SummarizationFailedError
    assert issubclass(SummarizationFailedError, RuntimeError)


async def test_long_summary_is_truncated() -> None:
    """A summary longer than the cap is truncated to exactly the cap with an ellipsis."""
    model = _StubModel(reply="x" * (_SUMMARY_CAP + 500))
    summary = await summarize_older_messages(_OLDER, model)
    assert len(summary) == _SUMMARY_CAP
    assert summary.endswith("...")


async def test_custom_prompt_is_accepted() -> None:
    """A custom prompt overrides the default without changing the returned summary."""
    model = _StubModel(reply="custom summary")
    summary = await summarize_older_messages(_OLDER, model, custom_prompt="Summarize tersely.")
    assert summary == "custom summary"


class _MultipartModel(BaseChatModel):
    """A chat model that answers with multipart content and records the prompt it was sent."""

    reply_parts: list[Any] = []
    received: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "stub-multipart"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.received.append(list(messages))
        return AIMessage(content=self.reply_parts)


_MULTIPART = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "x"}}, "world"]


async def test_message_text_handles_multipart_content() -> None:
    """Multipart content is coalesced on both paths -- the model's reply and the transcript it
    is sent: text parts joined, non-text parts ignored."""
    model = _MultipartModel(reply_parts=_MULTIPART, received=[])

    summary = await summarize_older_messages([AIMessage(content=_MULTIPART)], model)

    assert summary == "helloworld"
    (prompt,) = model.received
    assert prompt[-1].content == "Assistant: helloworld"
