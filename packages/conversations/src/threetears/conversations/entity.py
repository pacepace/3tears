"""
Conversation entity -- thin cache proxy for the per-agent
``conversations`` table.

this module owns the canonical :class:`Conversation` shape. it is the
one value type every other package that keys off ``conversation_id``
hydrates rows into: memory consumers call
:meth:`ConversationsCollection.get_by_id` (or fetch from their own
foreign-key joins) and then project fields off :class:`Conversation`.

lifecycle hooks (:meth:`mark_active`, :meth:`record_message`,
:meth:`close`, :meth:`summarize_into`) are mutation-only operations
that update the entity's in-memory state; the caller is responsible
for ``await entity.save()`` (immediate write) or for routing the
delta through :class:`ConversationWriteBuffer` (cross-conversation
batched write). data-layer-task-01 sub-task 3.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from enum import StrEnum

from threetears.core.entities.base import BaseEntity

__all__ = [
    "Conversation",
    "ConversationStatus",
]


class ConversationStatus(StrEnum):
    """canonical conversation lifecycle status values.

    one-step lifecycle: ``ACTIVE`` (default after lazy-create) ->
    ``CLOSED`` (terminal). archival is a separate orthogonal concern;
    the platform does not promote archived conversations through this
    enum.

    :cvar ACTIVE: conversation is open and accepting messages
    :cvar CLOSED: conversation has been closed; no further messages
    """

    ACTIVE = "active"
    CLOSED = "closed"


def _as_uuid(value: object) -> UUID:
    """
    coerce raw cache / data-layer value to :class:`UUID`.

    caches serialize UUIDs as strings; callers of
    :class:`BaseEntity` properties expect concrete UUIDs. the coercion
    is identity for :class:`UUID` inputs so there is no cost on the
    hot path.

    :param value: raw value from the cache or database row
    :ptype value: object
    :return: strongly typed UUID
    :rtype: UUID
    """
    if isinstance(value, UUID):
        result: UUID = value
    else:
        result = UUID(str(value))
    return result


class Conversation(BaseEntity):
    """
    cache proxy entity for one row in the ``conversations`` table.

    the conversations table tracks user-facing conversations an agent
    is engaged in. every row carries agent + customer + user scoping,
    the external channel reference, a short status enum, an optional
    summary, and three timestamps. consumer packages read these
    properties to associate their own per-conversation rows with the
    conversation's identity and scope.

    composite primary key on ``(agent_id, id)`` so rows are
    partitioned per agent; ``_id`` holds the
    ``(agent_id, id)`` tuple after construction.

    :ivar primary_key_field: name of the first pk column on the table
    :ptype primary_key_field: str
    """

    primary_key_field: str = "agent_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with tuple ``_id`` for composite-pk lookup.

        :class:`BaseEntity.__init__` captures the first pk field by
        name; composite-pk entities overwrite ``_id`` with the
        declared-order tuple so :meth:`BaseCollection.normalize_pk`
        and :meth:`BaseCollection.l2_key` address the row uniformly
        across tiers.

        :param data: row dict; must carry both ``agent_id`` and ``id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_id", (data["agent_id"], data["id"]))

    @property
    def agent_id(self) -> UUID:
        """
        return the owning agent identifier.

        :return: agent UUID for the conversation
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """
        set the owning agent identifier.

        :param value: agent UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID:
        """
        return the owning customer identifier.

        :return: customer UUID for the conversation
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("customer_id"))

    @customer_id.setter
    def customer_id(self, value: UUID) -> None:
        """
        set the owning customer identifier.

        :param value: customer UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """
        return the invoking user identifier.

        :return: user UUID for the conversation
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """
        set the invoking user identifier.

        :param value: user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def channel_type(self) -> str:
        """
        return the channel-adapter type (slack, discord, ...).

        :return: channel adapter identifier string
        :rtype: str
        """
        value: str = self._get_raw("channel_type")
        return value

    @channel_type.setter
    def channel_type(self, value: str) -> None:
        """
        set the channel-adapter type.

        :param value: channel adapter identifier string
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "channel_type", value)

    @property
    def conversation_ref(self) -> str | None:
        """
        return the channel-specific reference string for the conversation.

        opaque to the platform; interpreted by the originating channel
        adapter when correlating downstream events.

        :return: channel-specific conversation reference or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("conversation_ref")
        return value

    @conversation_ref.setter
    def conversation_ref(self, value: str | None) -> None:
        """
        set the channel-specific reference string.

        :param value: channel-specific reference or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "conversation_ref", value)

    @property
    def name(self) -> str | None:
        """
        return the conversation display name.

        nullable: conversations are typically un-named at creation time
        and get a title (often LLM-generated) after the first user turn
        lands.  callers presenting conversations in a list UI fall back
        to ``summary`` or a synthesized label when this is ``None``.

        :return: display name or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str | None) -> None:
        """
        set the conversation display name.

        :param value: display name or ``None`` to clear
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "name", value)

    @property
    def status(self) -> str:
        """
        return the conversation status enum value.

        :return: short status token (e.g. ``active``, ``closed``)
        :rtype: str
        """
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """
        set the conversation status enum value.

        :param value: short status token
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "status", value)

    @property
    def summary(self) -> str | None:
        """
        return the rolling conversation summary.

        populated by memory-extraction passes; ``None`` while the
        conversation has no distilled summary yet.

        :return: summary text or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("summary")
        return value

    @summary.setter
    def summary(self, value: str | None) -> None:
        """
        set the rolling conversation summary.

        :param value: summary text or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "summary", value)

    @property
    def date_created(self) -> datetime:
        """
        return the creation timestamp (UTC).

        :return: UTC datetime when the row was first created
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """
        set the creation timestamp (UTC).

        :param value: UTC-aware datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """
        return the last-updated timestamp (UTC).

        :return: UTC datetime of the last mutation
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """
        set the last-updated timestamp (UTC).

        :param value: UTC-aware datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_updated", value)

    @property
    def date_last_message(self) -> datetime | None:
        """
        return the timestamp of the last message in the conversation.

        ``None`` until the first message lands; callers use this for
        inactivity sweeps.

        :return: UTC datetime of last message or ``None``
        :rtype: datetime | None
        """
        value: datetime | None = self._get_raw("date_last_message")
        return value

    @date_last_message.setter
    def date_last_message(self, value: datetime | None) -> None:
        """
        set the timestamp of the last message in the conversation.

        :param value: UTC-aware datetime or ``None``
        :ptype value: datetime | None
        """
        BaseEntity.__setattr__(self, "date_last_message", value)

    @property
    def metadata(self) -> dict[str, Any] | None:
        """
        return the free-form metadata JSONB blob.

        channel adapters and memory extractors use this to carry
        platform-specific fields that are not worth promoting to
        columns. always a dict or ``None``; never a list.

        :return: metadata dict or ``None``
        :rtype: dict[str, Any] | None
        """
        value: dict[str, Any] | None = self._get_raw("metadata")
        return value

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        """
        set the free-form metadata JSONB blob.

        :param value: metadata dict or ``None``
        :ptype value: dict[str, Any] | None
        """
        BaseEntity.__setattr__(self, "metadata", value)

    @property
    def message_count(self) -> int:
        """
        return the running count of messages recorded on this conversation.

        added in v002 of the conversations migration package
        (data-layer-task-01 sub-task 3) so admin queries can render the
        conversation list without a per-row COUNT(*) on a foreign-key
        message table. defaults to ``0`` for rows predating the
        backfill.

        :return: message counter
        :rtype: int
        """
        value: int | None = self._get_raw("message_count")
        return value if value is not None else 0

    @message_count.setter
    def message_count(self, value: int) -> None:
        """
        set the running message counter.

        :param value: new counter value
        :ptype value: int
        """
        BaseEntity.__setattr__(self, "message_count", value)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def mark_active(self) -> None:
        """
        flip ``status`` back to ``active`` and refresh ``date_updated``.

        idempotent. used by channel adapters that re-open a previously
        closed conversation when a user resumes the session. the
        caller is responsible for persisting the change via
        :meth:`save_entity` or by enqueueing the delta on
        :class:`ConversationWriteBuffer`.

        :return: nothing
        :rtype: None
        """
        self.status = ConversationStatus.ACTIVE.value
        self.date_updated = datetime.now(UTC)

    def record_message(self, at: datetime, role: str) -> None:
        """
        increment ``message_count`` and stamp ``date_last_message``.

        intentionally does NOT mark the conversation active or change
        status; reactivation is an explicit operation via
        :meth:`mark_active`. ``role`` is recorded on the metadata blob
        so downstream consumers (analytics, audit) can read which
        actor triggered the increment without reaching into the
        message table. ``at`` is normalized to aware-UTC; naive input
        is coerced as a boundary defense for legacy callers.

        :param at: timestamp the message was observed at
        :ptype at: datetime
        :param role: short actor token, e.g. ``user`` / ``assistant``
        :ptype role: str
        :return: nothing
        :rtype: None
        """
        normalized = at.astimezone(UTC) if at.tzinfo else at.replace(tzinfo=UTC)
        self.message_count = self.message_count + 1
        self.date_last_message = normalized
        self.date_updated = normalized
        meta = dict(self.metadata) if self.metadata is not None else {}
        meta["last_role"] = role
        self.metadata = meta

    def close(self, reason: str) -> None:
        """
        flip ``status`` to ``closed`` and record the reason on metadata.

        terminal lifecycle step; subsequent attempts to ``record_message``
        succeed at the entity level (counter still increments) but the
        admin surfaces filter ``status='closed'`` rows out of active-
        conversation queries by default. ``reason`` is short free-form
        text recorded on ``metadata['close_reason']``.

        :param reason: short token describing why the conversation
            closed (``user_request`` / ``timeout`` / ``error`` / ...)
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self.status = ConversationStatus.CLOSED.value
        self.date_updated = datetime.now(UTC)
        meta = dict(self.metadata) if self.metadata is not None else {}
        meta["close_reason"] = reason
        self.metadata = meta

    def summarize_into(self, text: str) -> None:
        """
        replace the rolling summary with ``text`` and refresh ``date_updated``.

        invoked by memory-extraction passes once the rolling summary
        is ready; idempotent at the value level (re-applying the same
        text is a no-op aside from the timestamp bump). callers must
        persist the change via :meth:`save_entity` or via
        :class:`ConversationWriteBuffer`.

        :param text: distilled summary text to replace the prior value
        :ptype text: str
        :return: nothing
        :rtype: None
        """
        self.summary = text
        self.date_updated = datetime.now(UTC)
