"""unit tests for :class:`ConversationWriteBuffer`.

data-layer-task-01 sub-task 3. validates cross-conversation pod-wide
batching: the buffer coalesces deltas across many conversations and
flushes opportunistically (timer / threshold / shutdown). per Pace's
DQ-D2 nuance: per-conversation buffers don't help -- a multi-
conversation pod amortizes write cost via batched-across-conversations
writes.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from uuid import uuid7

from threetears.conversations.buffer import ConversationWriteBuffer


def _build_collection_mock(
    *, get_returns: dict[tuple, Any] | None = None,
) -> MagicMock:
    """build a fake :class:`ConversationsCollection` for buffer tests.

    :param get_returns: optional pk -> entity map; defaults to a
        single mock entity for any pk
    :ptype get_returns: dict[tuple, Any] | None
    :return: configured mock collection
    :rtype: MagicMock
    """
    collection = MagicMock()
    if get_returns is None:
        async def _default_get(pk: Any) -> Any:
            """return a mock entity exposing record_message + metadata.

            :param pk: composite primary key tuple
            :ptype pk: Any
            :return: mock entity
            :rtype: Any
            """
            entity = MagicMock()
            entity.record_message = MagicMock()
            return entity
        collection.get = AsyncMock(side_effect=_default_get)
    else:
        async def _mapped_get(pk: Any) -> Any:
            """return the pre-mapped entity for ``pk``.

            :param pk: composite primary key tuple
            :ptype pk: Any
            :return: mapped entity or ``None``
            :rtype: Any
            """
            return get_returns.get(pk)
        collection.get = AsyncMock(side_effect=_mapped_get)
    collection.save_entity = AsyncMock()
    return collection


class TestEnqueueAndFlush:
    """enqueue accumulates deltas; flush drains them through the collection."""

    async def test_enqueue_then_flush_persists(self) -> None:
        """one enqueue + one flush triggers one save_entity call.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        agent_id = uuid7()
        conv_id = uuid7()
        await buffer.enqueue(
            agent_id=agent_id,
            conversation_id=conv_id,
            at=datetime.now(UTC),
            role="user",
        )
        await buffer.flush()
        assert collection.save_entity.await_count == 1

    async def test_coalesces_same_conversation(self) -> None:
        """three enqueues for one conversation produce one save_entity.

        the entity's record_message is invoked three times to apply
        the accumulated counter.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        agent_id = uuid7()
        conv_id = uuid7()
        for _ in range(3):
            await buffer.enqueue(
                agent_id=agent_id,
                conversation_id=conv_id,
                at=datetime.now(UTC),
                role="user",
            )
        await buffer.flush()
        assert collection.save_entity.await_count == 1
        # the entity's record_message is the increment hook; mock
        # captures the apply path
        entity = collection.save_entity.await_args.args[0]
        assert entity.record_message.call_count == 3

    async def test_batches_across_conversations(self) -> None:
        """enqueues for two conversations produce two save_entity calls.

        cross-conversation batching: one flush drains both deltas,
        one round-trip per conversation. the regression-lock for
        Pace's DQ-D2 nuance.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        agent_id = uuid7()
        conv_a = uuid7()
        conv_b = uuid7()
        await buffer.enqueue(
            agent_id=agent_id, conversation_id=conv_a,
            at=datetime.now(UTC), role="user",
        )
        await buffer.enqueue(
            agent_id=agent_id, conversation_id=conv_b,
            at=datetime.now(UTC), role="user",
        )
        await buffer.flush()
        assert collection.save_entity.await_count == 2

    async def test_flush_clears_buffer(self) -> None:
        """second flush after first is a no-op.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        await buffer.enqueue(
            agent_id=uuid7(), conversation_id=uuid7(),
            at=datetime.now(UTC), role="user",
        )
        await buffer.flush()
        await buffer.flush()
        assert collection.save_entity.await_count == 1


class TestThresholdFlush:
    """exceeding ``flush_threshold_messages`` triggers a flush."""

    async def test_threshold_triggers_flush(self) -> None:
        """100-message threshold fires an immediate flush.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
            flush_threshold_messages=3,
        )
        agent_id = uuid7()
        conv_id = uuid7()
        for _ in range(3):
            await buffer.enqueue(
                agent_id=agent_id,
                conversation_id=conv_id,
                at=datetime.now(UTC),
                role="user",
            )
        # Yield so the threshold-fired task runs
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert collection.save_entity.await_count >= 1


class TestShutdownDrain:
    """:meth:`stop` flushes the buffer one last time."""

    async def test_stop_drains_buffer(self) -> None:
        """pending deltas land via the final drain in stop.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        await buffer.enqueue(
            agent_id=uuid7(), conversation_id=uuid7(),
            at=datetime.now(UTC), role="user",
        )
        await buffer.stop()
        assert collection.save_entity.await_count == 1

    async def test_enqueue_after_stop_is_dropped(self) -> None:
        """after stop, enqueues drop silently to prevent leaks.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock()
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        await buffer.stop()
        await buffer.enqueue(
            agent_id=uuid7(), conversation_id=uuid7(),
            at=datetime.now(UTC), role="user",
        )
        # No flush triggered, no saves
        assert collection.save_entity.await_count == 0


class TestMissingEntityIsSkipped:
    """flush against a missing entity drops the delta with a debug log."""

    async def test_missing_entity_skipped(self) -> None:
        """no save_entity when get returns None.

        :return: nothing
        :rtype: None
        """
        collection = _build_collection_mock(get_returns={})
        buffer = ConversationWriteBuffer(
            collection=collection,
            flush_interval_seconds=0,
        )
        await buffer.enqueue(
            agent_id=uuid7(), conversation_id=uuid7(),
            at=datetime.now(UTC), role="user",
        )
        await buffer.flush()
        assert collection.save_entity.await_count == 0
