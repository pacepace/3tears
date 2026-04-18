"""
Conversation entity -- thin cache proxy for the per-agent
``conversations`` table.

this module owns the canonical :class:`Conversation` shape. it is the
one value type every other package that keys off ``conversation_id``
hydrates rows into: memory consumers call
:meth:`ConversationsCollection.get_by_id` (or fetch from their own
foreign-key joins) and then project fields off :class:`Conversation`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity


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

    :ivar _primary_key_field: primary key column name on the table
    :ptype _primary_key_field: str
    """

    _primary_key_field: str = "id"

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
