"""Conversation summarization for context-window management.

When a conversation's active message window grows past the model's context
budget, the older messages are distilled into a concise narrative summary
before the next model call. The summary stands in for the original messages
in the active context — the originals are NOT deleted (the caller keeps the
full history in its checkpointer / store), they are merely excluded from the
window fed to the model.

This is domain-agnostic: it takes a list of LangChain messages and a chat
model, and returns summary text. Lifted from MetaLLM's
``graph/nodes/summarize.py`` into 3tears so both MetaLLM and Scriob consume
one implementation (shared-infra directive). The caller decides *when* to
summarize (the token-threshold trigger) and *what* to do with the result
(persist a rolling summary, advance a cursor); this module owns only the
distillation.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from threetears.observe import get_logger

__all__ = ["DEFAULT_SUMMARIZATION_PROMPT", "summarize_older_messages"]

_logger = get_logger(__name__)

DEFAULT_SUMMARIZATION_PROMPT = (
    "Summarize the conversation history below into a concise narrative that preserves:\n"
    "- Key topics discussed and conclusions reached\n"
    "- Important facts, names, numbers, and decisions\n"
    "- The user's preferences and requests that are still relevant\n"
    "- Any unresolved questions or ongoing tasks\n"
    "\n"
    "Write in third person past tense. Be concise but complete — the summary "
    "replaces the original messages and the assistant will not have access to them. "
    "Do not include greetings or filler."
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


def _format_transcript(messages: list[BaseMessage]) -> str:
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


def _fallback_summary(messages: list[BaseMessage]) -> str:
    """Heuristic fallback when the LLM summarization call fails.

    Extracts the last sentence of each assistant message — enough to keep some
    continuity in the active context even when the model is unavailable, so a
    provider outage degrades the summary rather than dropping the turn.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            text = _message_text(msg).strip()
            if not text:
                continue
            # Take the last sentence.
            sentences = text.replace("\n", " ").split(". ")
            last = sentences[-1].strip().rstrip(".")
            if last:
                parts.append(last + ".")
    if not parts:
        return "Earlier conversation context was summarized but details are unavailable."
    return " ".join(parts[:20])  # Cap at 20 sentences.


async def summarize_older_messages(
    older_messages: list[BaseMessage],
    chat_model: BaseChatModel,
    custom_prompt: str | None = None,
) -> str:
    """Summarize a list of older messages into a concise narrative.

    Invokes ``chat_model`` with a summarization prompt over the rendered
    transcript and returns the model's summary text, capped at
    :data:`_MAX_SUMMARY_LENGTH` characters. If the model call fails for any
    reason, falls back to a heuristic summary (logged, never silently swallowed)
    so the turn proceeds rather than failing on a provider hiccup.

    :param older_messages: the messages to summarize (the window being rolled out
        of the active context).
    :param chat_model: the LangChain chat model used to generate the summary.
    :param custom_prompt: an optional override for :data:`DEFAULT_SUMMARIZATION_PROMPT`.
    :return: the summary text (capped at :data:`_MAX_SUMMARY_LENGTH` characters).
    """
    prompt = custom_prompt or DEFAULT_SUMMARIZATION_PROMPT
    transcript = _format_transcript(older_messages)

    try:
        result = await chat_model.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=transcript),
            ]
        )
        summary = _message_text(result).strip()
    except Exception:  # prawduct:allow prawduct/broad-except -- provider/LLM errors fall back to the heuristic summary
        _logger.warning(
            "Summarization LLM call failed, using fallback",
            exc_info=True,
        )
        summary = _fallback_summary(older_messages)

    if len(summary) > _MAX_SUMMARY_LENGTH:
        summary = summary[: _MAX_SUMMARY_LENGTH - 3] + "..."

    return summary
