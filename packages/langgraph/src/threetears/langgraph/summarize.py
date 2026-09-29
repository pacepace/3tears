"""Conversation summarization for context-window management.

When a conversation's active message window grows past the model's context
budget, the older messages are distilled into a concise narrative summary
before the next model call. The summary stands in for the original messages
in the active context — the originals are NOT deleted (the caller keeps the
full history in its checkpointer / store), they are merely excluded from the
window fed to the model.

This is domain-agnostic: it takes a list of LangChain messages and a chat
model, and returns summary text. Lifted from a product's
``graph/nodes/summarize.py`` into 3tears so multiple products consume
one implementation (shared-infra directive). The caller decides *when* to
summarize (the token-threshold trigger) and *what* to do with the result
(persist a rolling summary, advance a cursor); this module owns only the
distillation.

A failed distillation raises :class:`SummarizationFailedError`. It is never
answered with text: a caller that stores the summary as the conversation's
narrative cannot tell a stand-in sentence from a real summary, and one stored
"The earlier part of this conversation could not be summarized." as exactly
that.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig

from threetears.observe import get_logger

__all__ = ["DEFAULT_SUMMARIZATION_PROMPT", "SummarizationFailedError", "summarize_older_messages"]

log = get_logger(__name__)

#: No person is named for the voice: a caller summarizing an agent's own turns
#: asks for the first person, and a default of third person would fight it.
DEFAULT_SUMMARIZATION_PROMPT = (
    "Summarize the conversation below. The summary replaces these messages, so keep "
    "everything needed to carry on: what was discussed and decided, names, numbers and "
    "facts, what each person wants, and anything still open.\n"
    "\n"
    "Write plain sentences in the past tense, not a list. Leave out greetings and small talk."
)

#: Hard cap on the returned summary length (characters). A summary that grows past
#: this is truncated with an ellipsis — a runaway summary defeats the purpose of
#: summarizing, and the cap keeps the assembled context bounded regardless of the
#: model's verbosity.
_MAX_SUMMARY_LENGTH = 2000


def _message_text(message: BaseMessage) -> str:
    """Return a message's text content as a plain string.

    LangChain message ``content`` is ``str | list[...]`` (multi-part). Coalesce:
    a ``str`` passes through; a list is joined over its text parts
    (``{"type": "text", "text": ...}`` or bare strings), ignoring non-text parts.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _format_transcript(messages: Sequence[BaseMessage]) -> str:
    """Format messages as a readable transcript for summarization."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, SystemMessage):
            role = "System"
        else:
            role = "Unknown"
        lines.append(f"{role}: {_message_text(msg)}")
    return "\n\n".join(lines)


class SummarizationFailedError(RuntimeError):
    """the summary model call failed, ran out of time, or answered with nothing.

    raised in place of a summary, so a failure never reaches a caller as the
    conversation's narrative. the model's own exception is chained as
    ``__cause__``; an empty answer has none. a caller keeps what it had -- the
    prior summary, or the un-summarized window -- stores nothing, and lets the
    next turn try again.

    :param reason: what went wrong, in words
    :ptype reason: str
    :param message_count: how many messages the call was summarizing
    :ptype message_count: int
    """

    def __init__(self, reason: str, *, message_count: int) -> None:
        self.reason = reason
        self.message_count = message_count
        super().__init__(f"summarizing {message_count} message(s) failed: {reason}")


async def summarize_older_messages(
    older_messages: Sequence[BaseMessage],
    chat_model: BaseChatModel,
    custom_prompt: str | None = None,
    config: RunnableConfig | None = None,
) -> str:
    """summarize a list of older messages into a concise narrative.

    invokes ``chat_model`` with a summarization prompt over the rendered
    transcript and returns the model's summary text, capped at
    :data:`_MAX_SUMMARY_LENGTH` characters. a failed call is logged once, here,
    with its cause, and raised as :class:`SummarizationFailedError` -- never
    answered with stand-in text. ``asyncio.CancelledError`` is not a failed
    summary and propagates untouched.

    :param older_messages: the messages to summarize (the window being rolled out
        of the active context)
    :ptype older_messages: Sequence[BaseMessage]
    :param chat_model: the chat model used to generate the summary
    :ptype chat_model: BaseChatModel
    :param custom_prompt: an optional override for :data:`DEFAULT_SUMMARIZATION_PROMPT`
    :ptype custom_prompt: str | None
    :param config: an optional ``RunnableConfig`` forwarded to the model call.
        callers streaming a response tag this with the framework's no-stream
        marker (``{"tags": [NOSTREAM_TAG]}``) so the internal summary call's
        tokens never leak into the user-facing token stream
    :ptype config: RunnableConfig | None
    :return: the summary text, capped at :data:`_MAX_SUMMARY_LENGTH` characters
    :rtype: str
    :raises SummarizationFailedError: the model call raised or timed out (chained
        as ``__cause__``), or it answered with no text
    """
    prompt = custom_prompt or DEFAULT_SUMMARIZATION_PROMPT
    transcript = _format_transcript(older_messages)
    extra = {"extra_data": {"message_count": len(older_messages)}}

    try:
        result = await chat_model.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=transcript),
            ],
            config=config,
        )
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- any provider failure is re-raised typed, chained, never answered as text
        log.warning("summary model call failed; no summary is returned", extra=extra, exc_info=True)
        raise SummarizationFailedError(
            f"the model call raised {type(exc).__name__}", message_count=len(older_messages)
        ) from exc

    summary = _message_text(result).strip()
    if not summary:
        log.warning("summary model call answered with no text; no summary is returned", extra=extra)
        raise SummarizationFailedError("the model answered with no text", message_count=len(older_messages))

    if len(summary) > _MAX_SUMMARY_LENGTH:
        summary = summary[: _MAX_SUMMARY_LENGTH - 3] + "..."

    return summary
