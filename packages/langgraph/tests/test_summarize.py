"""Unit tests for :mod:`threetears.langgraph.summarize`.

Pure-logic, no infra: a stub chat model drives the three contracts the lifted
summarizer must hold —

* a model that returns a summary → that summary is returned;
* a model that raises → the heuristic :func:`_fallback_summary` is used (the
  broad ``except`` is the provider-error boundary, logged, never swallowed);
* an over-long summary is truncated to :data:`_MAX_SUMMARY_LENGTH`.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from threetears.langgraph.summarize import (
    _MAX_SUMMARY_LENGTH,
    _fallback_summary,
    summarize_older_messages,
)


class _StubModel(BaseChatModel):
    """A minimal chat model that returns a fixed reply (or raises) on ``ainvoke``.

    ``BaseChatModel`` is abstract; we only exercise ``ainvoke`` here, so the sync
    ``_generate`` / ``_llm_type`` hooks are stubbed to satisfy the ABC without
    being called.
    """

    reply: str = ""
    raises: bool = False

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        if self.raises:
            raise RuntimeError("provider unavailable")
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


async def test_falls_back_when_model_raises() -> None:
    """A model error falls back to the heuristic summary (last sentence per AI msg)."""
    model = _StubModel(raises=True)
    summary = await summarize_older_messages(_OLDER, model)
    assert summary == _fallback_summary(_OLDER)
    # The fallback pulls the last sentence of the assistant message.
    assert summary == "It sits on the Seine."


async def test_fallback_when_no_assistant_content() -> None:
    """With no usable assistant text, the fallback returns its sentinel string."""
    only_human: list[BaseMessage] = [HumanMessage(content="hello")]
    assert _fallback_summary(only_human) == ("Earlier conversation context was summarized but details are unavailable.")


async def test_long_summary_is_truncated() -> None:
    """A summary longer than the cap is truncated to exactly the cap with an ellipsis."""
    model = _StubModel(reply="x" * (_MAX_SUMMARY_LENGTH + 500))
    summary = await summarize_older_messages(_OLDER, model)
    assert len(summary) == _MAX_SUMMARY_LENGTH
    assert summary.endswith("...")


async def test_custom_prompt_is_accepted() -> None:
    """A custom prompt overrides the default without changing the returned summary."""
    model = _StubModel(reply="custom summary")
    summary = await summarize_older_messages(_OLDER, model, custom_prompt="Summarize tersely.")
    assert summary == "custom summary"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


async def test_message_text_handles_multipart_content() -> None:
    """The fallback/transcript path coalesces LangChain multipart content (the mypy-strict fix the
    `_message_text` helper exists for): text parts joined, non-text parts ignored."""
    from threetears.langgraph.summarize import _message_text

    msg = AIMessage(
        content=[{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "x"}}, "world"]
    )
    assert _message_text(msg) == "helloworld"
