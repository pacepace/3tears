"""``ConversationSummaryStore``: a conversation's rolling summary and cursor, on its own row.

``RollingSummaryMiddleware`` keeps its summary behind a ``SummaryStore``; this is the store over a
conversations row -- the ``summary`` column the table already had, plus a cursor in ``metadata`` --
so a consumer on ``ConversationsCollection`` needs no store of its own and no migration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.conversations import Conversation, ConversationSummaryStore
from threetears.conversations.summary_store import SUMMARY_CURSOR_KEY
from threetears.core.exceptions import ConcurrentModificationError
from threetears.langgraph import SummaryState

_AGENT = uuid4()


def _conversation(conversation_id: UUID, **fields: Any) -> Conversation:
    now = datetime.now(UTC)
    data: dict[str, Any] = {
        "agent_id": _AGENT,
        "conversation_id": conversation_id,
        "customer_id": uuid4(),
        "user_id": uuid4(),
        "channel_type": "web",
        "conversation_ref": "ref",
        "status": "active",
        "summary": None,
        "date_created": now,
        "date_updated": now,
        "metadata": {"source": "test"},
        "message_count": 0,
        **fields,
    }
    return Conversation(data)


# parity-exempt: a get/save_entity stand-in for the two calls the summary store makes on ConversationsCollection
class _FakeConversations:
    def __init__(self, *rows: Conversation) -> None:
        self.rows = {(row.agent_id, row.conversation_id): row for row in rows}
        self.saved: list[Conversation] = []
        self.conflict_next = False

    async def get(self, key: tuple[UUID, UUID]) -> Conversation | None:
        return self.rows.get(key)

    async def save_entity(self, entity: Conversation) -> None:
        if self.conflict_next:
            self.conflict_next = False
            raise ConcurrentModificationError("conversations", entity.conversation_id, entity.date_updated)
        self.saved.append(entity)


def _store(rows: _FakeConversations, conversation_id: UUID) -> ConversationSummaryStore:
    return ConversationSummaryStore(rows, agent_id=_AGENT, conversation_id=conversation_id)  # type: ignore[arg-type]


async def test_a_conversation_never_summarized_loads_as_none() -> None:
    cid = uuid4()
    assert await _store(_FakeConversations(_conversation(cid)), cid).load() is None


async def test_a_missing_conversation_loads_as_none_and_refuses_a_save() -> None:
    cid = uuid4()
    store = _store(_FakeConversations(), cid)
    assert await store.load() is None
    assert await store.save(SummaryState("s", "m1", 1), expected=None) is False


async def test_a_saved_state_round_trips_and_keeps_other_metadata() -> None:
    cid = uuid4()
    rows = _FakeConversations(_conversation(cid))
    store = _store(rows, cid)
    state = SummaryState(text="the story so far", through_id="m9", through_count=10)
    assert await store.save(state, expected=None) is True
    assert await store.load() == state
    row = rows.rows[(_AGENT, cid)]
    assert row.summary == "the story so far"
    assert row.metadata is not None and row.metadata["source"] == "test", "other metadata survives"
    assert row.metadata[SUMMARY_CURSOR_KEY] == {"through_id": "m9", "through_count": 10}
    assert rows.saved == [row]


async def test_a_save_from_a_stale_expectation_is_refused() -> None:
    cid = uuid4()
    rows = _FakeConversations(_conversation(cid))
    store = _store(rows, cid)
    assert await store.save(SummaryState("first", "m1", 2), expected=None)
    assert await store.save(SummaryState("second", "m3", 4), expected=None) is False, "another writer folded first"
    assert (await store.load()) == SummaryState("first", "m1", 2)


async def test_an_unrelated_row_update_does_not_cost_the_summary() -> None:
    """Any write to the row (a message-count flush) moves its date_updated fence. When the summary
    state is still the one expected, the save is retried once rather than thrown away."""
    cid = uuid4()
    rows = _FakeConversations(_conversation(cid))
    rows.conflict_next = True
    assert await _store(rows, cid).save(SummaryState("s", "m1", 1), expected=None) is True
    assert await _store(rows, cid).load() == SummaryState("s", "m1", 1)


async def test_a_competing_fold_during_the_save_loses_the_compare_and_swap() -> None:
    cid = uuid4()
    rows = _FakeConversations(_conversation(cid))
    competitor = SummaryState("theirs", "m2", 3)

    async def conflict_then_theirs(entity: Conversation) -> None:
        # the other writer's fold landed between our read and our write
        row = rows.rows[(_AGENT, cid)]
        row.summarize_into(competitor.text)
        row.metadata = {SUMMARY_CURSOR_KEY: {"through_id": "m2", "through_count": 3}}
        raise ConcurrentModificationError("conversations", entity.conversation_id, entity.date_updated)

    rows.save_entity = conflict_then_theirs  # type: ignore[method-assign]
    assert await _store(rows, cid).save(SummaryState("mine", "m1", 1), expected=None) is False


async def test_a_summary_written_without_a_cursor_reads_as_covering_nothing_it_can_locate() -> None:
    """A row summarized by some other path (``summarize_into`` alone) still surfaces its summary."""
    cid = uuid4()
    rows = _FakeConversations(_conversation(cid, summary="written elsewhere"))
    assert await _store(rows, cid).load() == SummaryState("written elsewhere", None, 0)


def test_the_store_satisfies_the_middleware_protocol() -> None:
    from threetears.langgraph import SummaryStore

    store: SummaryStore = _store(_FakeConversations(), uuid4())
    assert callable(store.load) and callable(store.save)


@pytest.mark.parametrize("cursor", [{"through_count": "x"}, "not a dict", {"through_id": 7}])
async def test_a_malformed_cursor_is_read_as_no_cursor(cursor: object) -> None:
    cid = uuid4()
    rows = _FakeConversations(_conversation(cid, summary="s", metadata={SUMMARY_CURSOR_KEY: cursor}))
    state = await _store(rows, cid).load()
    assert state is not None and state.through_count == 0


async def test_the_summarized_event_is_skipped_outside_a_graph_run() -> None:
    """The summary is already stored; no run context means nothing to dispatch on, not a failure."""
    from threetears.conversations import dispatch_conversation_summarized

    await dispatch_conversation_summarized(3, "summary")
