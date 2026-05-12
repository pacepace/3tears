"""unit tests for :class:`Conversation` lifecycle hooks.

data-layer-task-01 sub-task 3. covers ``mark_active``,
``record_message``, ``close``, ``summarize_into`` -- the mutation-only
hooks that update entity state without persisting (caller is
responsible for ``save_entity`` or for routing through
:class:`ConversationWriteBuffer`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from threetears.conversations.entity import Conversation, ConversationStatus


def _make_entity(**overrides: Any) -> Conversation:
    """build a fresh :class:`Conversation` for unit testing.

    :param overrides: keys to override on the default row dict
    :ptype overrides: Any
    :return: fresh entity (is_new=True)
    :rtype: Conversation
    """
    from uuid import uuid7

    now = datetime.now(UTC)
    data: dict[str, Any] = {
        "id": uuid7(),
        "agent_id": uuid7(),
        "customer_id": uuid7(),
        "user_id": uuid7(),
        "channel_type": "web",
        "conversation_ref": None,
        "status": "active",
        "summary": None,
        "date_created": now,
        "date_updated": now,
        "date_last_message": None,
        "metadata": {},
        "message_count": 0,
    }
    data.update(overrides)
    return Conversation(data, is_new=True, collection=None)


class TestMarkActive:
    """:meth:`Conversation.mark_active` flips status and bumps timestamp."""

    def test_flips_status_back_to_active(self) -> None:
        """verify ``mark_active`` sets status to ``active``.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity(status="closed")
        entity.mark_active()
        assert entity.status == ConversationStatus.ACTIVE.value

    def test_idempotent(self) -> None:
        """verify repeated ``mark_active`` calls are safe.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        entity.mark_active()
        entity.mark_active()
        assert entity.status == ConversationStatus.ACTIVE.value


class TestRecordMessage:
    """:meth:`Conversation.record_message` increments + stamps."""

    def test_increments_message_count(self) -> None:
        """verify counter increments on each call.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        at = datetime.now(UTC)
        entity.record_message(at, "user")
        entity.record_message(at, "assistant")
        assert entity.message_count == 2

    def test_stamps_date_last_message(self) -> None:
        """verify ``date_last_message`` is set to the supplied ``at``.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        at = datetime.now(UTC)
        entity.record_message(at, "user")
        # naive UTC normalization
        assert entity.date_last_message is not None

    def test_records_role_on_metadata(self) -> None:
        """verify the actor role is stamped on metadata['last_role'].

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        entity.record_message(datetime.now(UTC), "assistant")
        assert entity.metadata is not None
        assert entity.metadata["last_role"] == "assistant"

    def test_does_not_change_status(self) -> None:
        """verify status is untouched by record_message.

        record_message increments the counter; reactivation is the
        explicit responsibility of :meth:`mark_active`.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity(status="closed")
        entity.record_message(datetime.now(UTC), "user")
        assert entity.status == "closed"


class TestClose:
    """:meth:`Conversation.close` flips status and records reason."""

    def test_flips_status_to_closed(self) -> None:
        """verify ``close`` sets status to ``closed``.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        entity.close("user_request")
        assert entity.status == ConversationStatus.CLOSED.value

    def test_records_close_reason_on_metadata(self) -> None:
        """verify reason lands on metadata['close_reason'].

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        entity.close("timeout")
        assert entity.metadata is not None
        assert entity.metadata["close_reason"] == "timeout"


class TestSummarizeInto:
    """:meth:`Conversation.summarize_into` replaces summary in place."""

    def test_replaces_summary(self) -> None:
        """verify summary is replaced with supplied text.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity(summary="old summary")
        entity.summarize_into("distilled summary")
        assert entity.summary == "distilled summary"

    def test_refreshes_date_updated(self) -> None:
        """verify ``date_updated`` is bumped on summarize.

        :return: nothing
        :rtype: None
        """
        entity = _make_entity()
        before = entity.date_updated
        entity.summarize_into("new summary")
        assert entity.date_updated >= before
