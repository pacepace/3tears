"""A conversation's rolling summary and cursor, stored on its own ``conversations`` row.

:class:`~threetears.langgraph.RollingSummaryMiddleware` keeps its summary behind a
:class:`~threetears.langgraph.SummaryStore`. This is that store over a conversations row: the
summary goes in the ``summary`` column the table has always had (through
:meth:`Conversation.summarize_into`), and the cursor -- how far into the history the summary reaches
-- in ``metadata[SUMMARY_CURSOR_KEY]``. No migration.

A save is a compare-and-swap twice over: the stored state must still be the one the fold started
from, and the row is written under the collection's own ``date_updated`` fence, so a racing fold
loses cleanly (``False``) instead of interleaving two summaries. A racing write that is NOT a fold
(the row's message count, a rename) also moves the fence; the save re-checks and retries once while
the summary state is unchanged. The fence needs a write-through collection: a ``conversations``
collection configured write-behind skips the L3 fence, leaving only the state comparison.

:func:`dispatch_conversation_summarized` is the matching ``on_summarized`` callback: it fires the
existing :class:`ConversationSummarizedEvent` on the run's custom-event transport.
"""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from langchain_core.runnables.config import ensure_config

from threetears.conversations.entity import Conversation
from threetears.conversations.events import ConversationSummarizedEvent
from threetears.core.exceptions import ConcurrentModificationError
from threetears.langgraph import SummaryState
from threetears.langgraph.events import dispatch_event
from threetears.observe import get_logger

__all__ = [
    "SUMMARY_CURSOR_KEY",
    "ConversationSummaryStore",
    "dispatch_conversation_summarized",
]

log = get_logger(__name__)

#: ``metadata`` key holding the rolling summary's cursor: ``{"through_id": str | None, "through_count": int}``.
SUMMARY_CURSOR_KEY = "summary_through"


class _ConversationRows(Protocol):
    """The two ``ConversationsCollection`` calls the store makes."""

    async def get(self, entity_id: Any) -> Conversation | None: ...  # noqa: ANN401

    async def save_entity(self, entity: Conversation) -> None: ...

    def evict_from_cache_sync(self, entity_id: Any) -> bool: ...  # noqa: ANN401


def _state_of(conversation: Conversation) -> SummaryState | None:
    """Read a row's summary state; a malformed cursor reads as none.

    :param conversation: the row
    :ptype conversation: Conversation
    :return: the state, or ``None`` when the row has never been summarized
    :rtype: SummaryState | None
    """
    if not conversation.summary:
        return None
    raw = (conversation.metadata or {}).get(SUMMARY_CURSOR_KEY)
    cursor = raw if isinstance(raw, dict) else {}
    through_id = cursor.get("through_id")
    through_count = cursor.get("through_count")
    return SummaryState(
        text=conversation.summary,
        through_id=through_id if isinstance(through_id, str) else None,
        through_count=through_count if isinstance(through_count, int) and not isinstance(through_count, bool) else 0,
    )


class ConversationSummaryStore:
    """The :class:`~threetears.langgraph.SummaryStore` for one conversations row.

    :param collection: the conversations collection
    :ptype collection: ConversationsCollection
    :param agent_id: the row's ``agent_id`` (the first half of its key)
    :ptype agent_id: UUID
    :param conversation_id: the row's ``conversation_id``
    :ptype conversation_id: UUID
    """

    def __init__(self, collection: _ConversationRows, *, agent_id: UUID, conversation_id: UUID) -> None:
        self._collection = collection
        self._key = (agent_id, conversation_id)

    async def load(self) -> SummaryState | None:
        """The row's summary state, or ``None`` (never summarized, or no such row).

        :return: the stored state
        :rtype: SummaryState | None
        """
        conversation = await self._collection.get(self._key)
        return None if conversation is None else _state_of(conversation)

    async def save(self, state: SummaryState, *, expected: SummaryState | None) -> bool:
        """Store ``state`` if the row still holds ``expected``.

        :param state: the new state
        :ptype state: SummaryState
        :param expected: the state the fold started from
        :ptype expected: SummaryState | None
        :return: ``True`` when stored; ``False`` when the row is gone or another writer got there first
        :rtype: bool
        """
        stored = False
        # Twice at most: any write to the row (a message-count flush, a rename) moves its date_updated
        # fence, and losing the summary to one of those would waste the fold. A retry is safe only
        # while the summary state is still the one this fold started from.
        for attempt in range(2):
            conversation = await self._collection.get(self._key)
            # On the retry, this fold's own unsaved edit may be what a cached entity shows; that is
            # not another writer's fold.
            unchanged = (expected,) if attempt == 0 else (expected, state)
            if conversation is None or _state_of(conversation) not in unchanged:
                break
            conversation.summarize_into(state.text)
            conversation.metadata = {
                **(conversation.metadata or {}),
                SUMMARY_CURSOR_KEY: {"through_id": state.through_id, "through_count": state.through_count},
            }
            try:
                await self._collection.save_entity(conversation)
            except ConcurrentModificationError:
                # setting the fields wrote through to this pod's L1 row, and the save did not land: drop
                # that dirty row so the re-check (and any later load) reads the real one, not this fold's.
                self._collection.evict_from_cache_sync(self._key)
                log.info("rolling summary save hit a concurrent row update; re-checking once")
            else:
                stored = True
                break
        return stored


async def dispatch_conversation_summarized(messages_summarized: int, summary_text: str) -> None:
    """Fire :class:`ConversationSummarizedEvent` for a fold; pass as ``on_summarized``.

    Best-effort: the summary is already durable, and outside a graph run there is no transport to
    dispatch on, so that case is skipped rather than failing the turn.

    :param messages_summarized: how many messages the fold covered
    :ptype messages_summarized: int
    :param summary_text: the new rolling summary
    :ptype summary_text: str
    :return: None
    :rtype: None
    """
    try:
        await dispatch_event(
            ConversationSummarizedEvent(messages_summarized=messages_summarized, summary_text=summary_text),
            config=ensure_config(),
        )
    except RuntimeError:
        log.debug("no run context for the summarized event; the summary is still stored")
