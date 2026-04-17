"""Memory entity -- thin cache proxy for user memory records."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from threetears.core.entities.base import BaseEntity


def _as_uuid(value: object) -> UUID:
    """Coerce a value to UUID, handling strings from cache/data layers."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


class MemoryEntity(BaseEntity):
    """Cache proxy entity for the ``memories`` table."""

    _primary_key_field: str = "memory_id"

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
