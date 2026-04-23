"""Memory entity -- thin cache proxy for user memory records.

Also defines entities for orphan tables adopted under namespace-task-01
phase 8.5b: :class:`MediaEntity` (parent record from migration v006),
:class:`MediaContentEntity` (content rows from v006), and
:class:`MemoryChunkEntity` (document-style chunks from v007).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "MediaContentEntity",
    "MediaEntity",
    "MemoryChunkEntity",
    "MemoryEntity",
]


def _as_uuid(value: object) -> UUID:
    """Coerce a value to UUID, handling strings from cache/data layers."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


class MemoryEntity(BaseEntity):
    """Cache proxy entity for the ``memories`` table."""

    primary_key_field: str = "memory_id"

    @property
    def memory_id(self) -> UUID:
        """Get the memory ID (alias for primary key)."""
        return _as_uuid(self.id)

    @property
    def agent_id(self) -> UUID:
        """Get agent ID for memory scoping."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set agent ID."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID:
        """Get customer ID for memory scoping."""
        return _as_uuid(self._get_raw("customer_id"))

    @customer_id.setter
    def customer_id(self, value: UUID) -> None:
        """Set customer ID."""
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """Get the user ID that owns this memory."""
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """Set the user ID."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def conversation_id(self) -> UUID:
        """Get the conversation ID where this memory was extracted."""
        return _as_uuid(self._get_raw("conversation_id"))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """Set the conversation ID."""
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def message_id_source(self) -> UUID:
        """Get the source message ID for this memory."""
        return _as_uuid(self._get_raw("message_id_source"))

    @message_id_source.setter
    def message_id_source(self, value: UUID) -> None:
        """Set the source message ID."""
        BaseEntity.__setattr__(self, "message_id_source", value)

    @property
    def type_memory(self) -> str:
        """Get the memory type classification."""
        value: str = self._get_raw("type_memory")
        return value

    @type_memory.setter
    def type_memory(self, value: str) -> None:
        """Set the memory type."""
        BaseEntity.__setattr__(self, "type_memory", value)

    @property
    def content(self) -> str:
        """Get the memory content text."""
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """Set the memory content."""
        BaseEntity.__setattr__(self, "content", value)

    @property
    def embedding(self) -> list[float]:
        """Get the embedding vector."""
        value: list[float] = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float]) -> None:
        """Set the embedding vector."""
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def media_id(self) -> UUID | None:
        """Get the associated media ID, if any."""
        value = self._get_raw("media_id")
        if value is None:
            return None
        return _as_uuid(value)

    @media_id.setter
    def media_id(self, value: UUID | None) -> None:
        """Set the associated media ID."""
        BaseEntity.__setattr__(self, "media_id", value)

    @property
    def is_deleted(self) -> bool:
        """Get the soft-delete flag."""
        value: bool = self._get_raw("is_deleted")
        return value

    @is_deleted.setter
    def is_deleted(self, value: bool) -> None:
        """Set the soft-delete flag."""
        BaseEntity.__setattr__(self, "is_deleted", value)

    @property
    def date_deleted(self) -> datetime | None:
        """Get the deletion timestamp."""
        value: datetime | None = self._get_raw("date_deleted")
        return value

    @date_deleted.setter
    def date_deleted(self, value: datetime | None) -> None:
        """Set the deletion timestamp."""
        BaseEntity.__setattr__(self, "date_deleted", value)

    @property
    def date_updated(self) -> datetime | None:
        """Get the last-updated timestamp."""
        value: datetime | None = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime | None) -> None:
        """Set the last-updated timestamp."""
        BaseEntity.__setattr__(self, "date_updated", value)


class MediaEntity(BaseEntity):
    """cache proxy entity for the ``media`` parent table (v006).

    columns match ``CREATE TABLE media`` in
    :mod:`threetears.agent.memory.migrations.v006_memory_media_content`:
    ``media_id`` PK, nullable ``agent_id`` / ``customer_id``, required
    ``user_id``, ``media_category`` discriminator, ``metadata_json``
    blob, plus ``date_created`` / ``date_updated``.
    """

    primary_key_field: str = "media_id"

    @property
    def media_id(self) -> UUID:
        """get the media ID (primary key).

        :return: media UUID
        :rtype: UUID
        """
        return _as_uuid(self.id)

    @property
    def agent_id(self) -> UUID | None:
        """get optional agent ID for scoping.

        :return: agent UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("agent_id")
        if value is None:
            return None
        return _as_uuid(value)

    @agent_id.setter
    def agent_id(self, value: UUID | None) -> None:
        """set the agent ID.

        :param value: new agent UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """get optional customer ID for scoping.

        :return: customer UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        if value is None:
            return None
        return _as_uuid(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set the customer ID.

        :param value: new customer UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """get owning user ID.

        :return: user UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """set owning user ID.

        :param value: new user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def media_category(self) -> str:
        """get the media category discriminator.

        :return: category string
        :rtype: str
        """
        value: str = self._get_raw("media_category")
        return value

    @media_category.setter
    def media_category(self, value: str) -> None:
        """set the media category discriminator.

        :param value: new category
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "media_category", value)

    @property
    def metadata_json(self) -> Any:
        """get the metadata JSONB blob.

        :return: metadata payload (dict / list / scalar / ``None``)
        :rtype: Any
        """
        return self._get_raw("metadata_json")

    @metadata_json.setter
    def metadata_json(self, value: Any) -> None:
        """set the metadata JSONB blob.

        :param value: new metadata payload
        :ptype value: Any
        """
        BaseEntity.__setattr__(self, "metadata_json", value)

    @property
    def date_created(self) -> datetime:
        """get the creation timestamp.

        :return: creation datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the creation timestamp.

        :param value: new creation datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """get the last-updated timestamp.

        :return: update datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """set the last-updated timestamp.

        :param value: new update datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_updated", value)


class MediaContentEntity(BaseEntity):
    """cache proxy entity for the ``media_content`` table (v006).

    content rows attached to a :class:`MediaEntity` parent through
    ``media_id``. carry extracted text / transcripts / OCR with their
    own embedding + FTS search_vector maintained by trigger.
    """

    primary_key_field: str = "content_id"

    @property
    def content_id(self) -> UUID:
        """get the content ID (primary key).

        :return: content UUID
        :rtype: UUID
        """
        return _as_uuid(self.id)

    @property
    def media_id(self) -> UUID:
        """get the parent media ID.

        :return: media UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("media_id"))

    @media_id.setter
    def media_id(self, value: UUID) -> None:
        """set the parent media ID.

        :param value: new media UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "media_id", value)

    @property
    def agent_id(self) -> UUID | None:
        """get optional agent ID for scoping.

        :return: agent UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("agent_id")
        if value is None:
            return None
        return _as_uuid(value)

    @agent_id.setter
    def agent_id(self, value: UUID | None) -> None:
        """set the agent ID.

        :param value: new agent UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """get optional customer ID for scoping.

        :return: customer UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        if value is None:
            return None
        return _as_uuid(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set the customer ID.

        :param value: new customer UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """get owning user ID.

        :return: user UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """set owning user ID.

        :param value: new user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def content_type(self) -> str:
        """get the content-type discriminator.

        :return: content type string
        :rtype: str
        """
        value: str = self._get_raw("content_type")
        return value

    @content_type.setter
    def content_type(self, value: str) -> None:
        """set the content-type discriminator.

        :param value: new content type
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "content_type", value)

    @property
    def content(self) -> str:
        """get the content text.

        :return: content text
        :rtype: str
        """
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """set the content text.

        :param value: new content text
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "content", value)

    @property
    def summary(self) -> str | None:
        """get optional summary text.

        :return: summary text or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("summary")
        return value

    @summary.setter
    def summary(self, value: str | None) -> None:
        """set optional summary text.

        :param value: new summary or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "summary", value)

    @property
    def embedding(self) -> list[float] | None:
        """get the embedding vector.

        :return: embedding vector or ``None``
        :rtype: list[float] | None
        """
        value: list[float] | None = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        """set the embedding vector.

        :param value: new embedding or ``None``
        :ptype value: list[float] | None
        """
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def date_created(self) -> datetime:
        """get the creation timestamp.

        :return: creation datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the creation timestamp.

        :param value: new creation datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)


class MemoryChunkEntity(BaseEntity):
    """cache proxy entity for the ``memory_chunks`` table (v007).

    document-style chunks with location metadata (``heading_context``,
    ``page_number``) joined back to a parent :class:`MediaEntity`
    through optional ``media_id``. same embedding / FTS shape as
    :class:`MediaContentEntity`.
    """

    primary_key_field: str = "chunk_id"

    @property
    def chunk_id(self) -> UUID:
        """get the chunk ID (primary key).

        :return: chunk UUID
        :rtype: UUID
        """
        return _as_uuid(self.id)

    @property
    def media_id(self) -> UUID | None:
        """get optional parent media ID.

        :return: media UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("media_id")
        if value is None:
            return None
        return _as_uuid(value)

    @media_id.setter
    def media_id(self, value: UUID | None) -> None:
        """set optional parent media ID.

        :param value: new media UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "media_id", value)

    @property
    def agent_id(self) -> UUID | None:
        """get optional agent ID for scoping.

        :return: agent UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("agent_id")
        if value is None:
            return None
        return _as_uuid(value)

    @agent_id.setter
    def agent_id(self, value: UUID | None) -> None:
        """set the agent ID.

        :param value: new agent UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """get optional customer ID for scoping.

        :return: customer UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        if value is None:
            return None
        return _as_uuid(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set the customer ID.

        :param value: new customer UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """get owning user ID.

        :return: user UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """set owning user ID.

        :param value: new user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def content(self) -> str:
        """get the chunk content.

        :return: chunk text
        :rtype: str
        """
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """set the chunk content.

        :param value: new chunk text
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "content", value)

    @property
    def summary(self) -> str | None:
        """get optional summary text.

        :return: summary text or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("summary")
        return value

    @summary.setter
    def summary(self, value: str | None) -> None:
        """set optional summary text.

        :param value: new summary or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "summary", value)

    @property
    def heading_context(self) -> str | None:
        """get optional heading / section context.

        :return: heading context or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("heading_context")
        return value

    @heading_context.setter
    def heading_context(self, value: str | None) -> None:
        """set optional heading / section context.

        :param value: new heading context or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "heading_context", value)

    @property
    def page_number(self) -> int | None:
        """get optional page number.

        :return: page number or ``None``
        :rtype: int | None
        """
        value: int | None = self._get_raw("page_number")
        return value

    @page_number.setter
    def page_number(self, value: int | None) -> None:
        """set optional page number.

        :param value: new page number or ``None``
        :ptype value: int | None
        """
        BaseEntity.__setattr__(self, "page_number", value)

    @property
    def embedding(self) -> list[float] | None:
        """get the embedding vector.

        :return: embedding vector or ``None``
        :rtype: list[float] | None
        """
        value: list[float] | None = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        """set the embedding vector.

        :param value: new embedding or ``None``
        :ptype value: list[float] | None
        """
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def date_created(self) -> datetime:
        """get the creation timestamp.

        :return: creation datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the creation timestamp.

        :param value: new creation datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)
