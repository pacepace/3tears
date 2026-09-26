"""Fold older messages into a rolling summary without deleting the conversation's history.

:class:`~threetears.langgraph.SummarizationMiddleware` rewrites the checkpointed ``messages``
channel (``RemoveMessage``) once a message COUNT is crossed. Two consumers each hand-rolled the
other shape, and their persisted cursors disagreed, so this is that shape, once:

- **Non-destructive.** It overrides the model REQUEST for one call and returns no state update: the
  checkpointer keeps every message; only what the model sees is trimmed.
- **Token budget.** When the messages since the last fold exceed ``token_budget``, the newest suffix
  that fits is kept and the rest is folded into the rolling summary (the prior summary included).
- **At most one fold per turn.** It folds only on a turn's first model call -- when the window ends
  with the user's ``HumanMessage``; a tool loop's later calls only trim.
- **Never an orphaned tool result.** The kept tail never opens on a ``ToolMessage`` whose requesting
  ``AIMessage`` was folded away (providers reject that).
- **A durable cursor.** The summary and a cursor -- the id of the last folded message, with a count
  as the fallback for id-less messages -- live behind a :class:`SummaryStore`. Its ``save`` is a
  compare-and-swap, so two pods folding one conversation at once cannot interleave; the loser uses
  the winner's summary. ``threetears.conversations.ConversationSummaryStore`` is the store over a
  conversations row.
- **Billed and silent.** The summary call carries ``NOSTREAM_TAG`` (so it never streams into the
  user's reply) and ``metadata[USAGE_PURPOSE_METADATA_KEY] = "summarization"`` (so usage tracking
  can attribute it) -- run metadata, so this package needs no dependency on the models package.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from threetears.langgraph.streaming import NOSTREAM_TAG
from threetears.langgraph.summarize import summarize_older_messages
from threetears.observe import get_logger

__all__ = [
    "DEFAULT_ROLLING_SUMMARY_PREFIX",
    "RollingSummaryMiddleware",
    "SummaryState",
    "SummaryStore",
    "USAGE_PURPOSE_METADATA_KEY",
]

log = get_logger(__name__)

#: Run-metadata key carrying an LLM call's usage purpose. ``threetears.models``' usage tracking reads
#: it, so a call made here is attributed without this package importing that one.
USAGE_PURPOSE_METADATA_KEY = "threetears.usage.purpose"

#: Heads the summary message the model sees in place of the folded messages.
DEFAULT_ROLLING_SUMMARY_PREFIX = "Summary of the conversation so far:\n\n"

#: Heads the prior summary when it is folded into the next one.
_PRIOR_SUMMARY_PREFIX = "Prior rolling summary:\n"


@dataclass(frozen=True)
class SummaryState:
    """A conversation's rolling summary and how far into its history the summary reaches.

    :ivar text: the rolling summary
    :ivar through_id: id of the last message folded into it; ``None`` when that message had no id
    :ivar through_count: how many history messages it covers -- the cursor when ids are absent
    """

    text: str
    through_id: str | None
    through_count: int


class SummaryStore(Protocol):
    """Where one conversation's :class:`SummaryState` lives."""

    async def load(self) -> SummaryState | None:
        """The current state, or ``None`` before the first fold.

        :return: the stored state
        :rtype: SummaryState | None
        """
        ...

    async def save(self, state: SummaryState, *, expected: SummaryState | None) -> bool:
        """Store ``state`` only if the stored state still equals ``expected``.

        :param state: the new state
        :ptype state: SummaryState
        :param expected: the state this fold started from
        :ptype expected: SummaryState | None
        :return: ``True`` when stored; ``False`` when another writer got there first
        :rtype: bool
        """
        ...


def _cursor(window: Sequence[BaseMessage], state: SummaryState | None) -> int:
    """Index of the first message the summary does not yet cover.

    :param window: the history the model would see
    :ptype window: Sequence[BaseMessage]
    :param state: the stored summary state
    :ptype state: SummaryState | None
    :return: the start of the unsummarized window
    :rtype: int
    """
    start = 0
    if state is not None and state.through_id is not None:
        # The id is authoritative. Not found means the history was trimmed underneath the summary, so
        # everything still present is newer than it.
        ids = [m.id for m in window]
        start = ids.index(state.through_id) + 1 if state.through_id in ids else 0
    elif state is not None:
        start = min(state.through_count, len(window))
    return start


def _partition(
    messages: list[BaseMessage], *, budget: int, count_tokens: Callable[[Sequence[BaseMessage]], int]
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split ``messages`` into (to fold, to keep): the newest suffix within ``budget`` is kept.

    At least the last message is always kept. The kept tail never opens on a ``ToolMessage``: the
    cutoff moves past any, folding each tool result with the call that requested it.

    :param messages: the unsummarized window
    :ptype messages: list[BaseMessage]
    :param budget: the token budget for the kept tail
    :ptype budget: int
    :param count_tokens: token counter over a message list
    :ptype count_tokens: Callable[[Sequence[BaseMessage]], int]
    :return: the messages to fold and the messages to keep
    :rtype: tuple[list[BaseMessage], list[BaseMessage]]
    """
    if count_tokens(messages) <= budget:
        return [], list(messages)
    cutoff = len(messages) - 1
    while cutoff > 0 and count_tokens(messages[cutoff - 1 :]) <= budget:
        cutoff -= 1
    while cutoff < len(messages) - 1 and isinstance(messages[cutoff], ToolMessage):
        cutoff += 1
    return list(messages[:cutoff]), list(messages[cutoff:])


class RollingSummaryMiddleware(AgentMiddleware):
    """Keep the model's context within a token budget by folding older messages into a rolling summary.

    Build one per turn, bound to that conversation's store. See the module docstring for the
    guarantees.

    :param model: the model that writes summaries
    :ptype model: BaseChatModel
    :param store: this conversation's summary store
    :ptype store: SummaryStore
    :param token_budget: tokens the kept (unsummarized) messages may use; must be positive
    :ptype token_budget: int
    :param count_tokens: token counter over a message list (default: an approximation)
    :ptype count_tokens: Callable[[Sequence[BaseMessage]], int]
    :param prompt: summarization prompt (default: ``summarize``'s)
    :ptype prompt: str | None
    :param summary_prefix: heads the summary message the model sees
    :ptype summary_prefix: str
    :param on_summarized: awaited after each stored fold with (messages folded, summary text)
    :ptype on_summarized: Callable[[int, str], Awaitable[None]] | None
    """

    name = "RollingSummaryMiddleware"

    def __init__(
        self,
        model: BaseChatModel,
        *,
        store: SummaryStore,
        token_budget: int,
        count_tokens: Callable[[Sequence[BaseMessage]], int] = count_tokens_approximately,
        prompt: str | None = None,
        summary_prefix: str = DEFAULT_ROLLING_SUMMARY_PREFIX,
        on_summarized: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        self._model = model
        self._store = store
        self._budget = token_budget
        self._count_tokens = count_tokens
        self._prompt = prompt
        self._prefix = summary_prefix
        self._on_summarized = on_summarized

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Trim this call's messages to ``[summary, recent]`` when needed, then run it.

        :param request: the model request
        :ptype request: ModelRequest
        :param handler: the next handler
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model response
        :rtype: ModelResponse
        """
        messages: list[BaseMessage] = list(request.messages)
        system = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
        window = messages[1:] if system is not None else messages
        state = await self._store.load()
        start = _cursor(window, state)
        turn_start = bool(window) and isinstance(window[-1], HumanMessage)
        to_fold, recent = (
            _partition(window[start:], budget=self._budget, count_tokens=self._count_tokens)
            if turn_start
            else ([], list(window[start:]))
        )
        if to_fold:
            text = await self._summarize(state, to_fold)
            folded = SummaryState(text=text, through_id=to_fold[-1].id, through_count=start + len(to_fold))
            if await self._store.save(folded, expected=state):
                state = folded
                log.info(
                    "conversation summarized",
                    extra={"extra_data": {"messages_folded": len(to_fold), "token_budget": self._budget}},
                )
                if self._on_summarized is not None:
                    await self._on_summarized(len(to_fold), text)
            else:
                # Another writer folded this conversation first. One fold per turn: use theirs.
                state = await self._store.load()
                recent = list(window[_cursor(window, state) :])
        result: ModelResponse
        if state is None:
            result = await handler(request)
        else:
            summary = HumanMessage(content=f"{self._prefix}{state.text}")
            assembled = [*([system] if system is not None else []), summary, *recent]
            result = await handler(request.override(messages=cast("list[Any]", assembled)))
        return result

    async def _summarize(self, state: SummaryState | None, to_fold: list[BaseMessage]) -> str:
        """Write the next rolling summary: the prior one plus the messages being folded.

        :param state: the stored state (its text is the prior summary)
        :ptype state: SummaryState | None
        :param to_fold: the messages being folded
        :ptype to_fold: list[BaseMessage]
        :return: the new summary
        :rtype: str
        """
        older: list[BaseMessage] = list(to_fold)
        if state is not None:
            older.insert(0, SystemMessage(content=f"{_PRIOR_SUMMARY_PREFIX}{state.text}"))
        return await summarize_older_messages(
            older,
            self._model,
            custom_prompt=self._prompt,
            config={"tags": [NOSTREAM_TAG], "metadata": {USAGE_PURPOSE_METADATA_KEY: "summarization"}},
        )
