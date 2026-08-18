"""Context item entity — thin cache proxy for conversation context records."""

from __future__ import annotations


from threetears.core.entities.base import BaseEntity

__all__ = [
    "ContextItemEntity",
]


class ContextItemEntity(BaseEntity):
    """Entity representing a conversation context item.

    Covers all context types: variables, tool results, and media slots.
    All field access is via BaseEntity's ``__getattr__`` proxy. composite
    primary key on ``(conversation_id, context_id)`` so the entity
    addresses the partition slot it belongs to; ``BaseEntity`` derives
    ``_id`` as that tuple from the collection's declared
    ``primary_key_columns``.
    """

    primary_key_field: str = "context_id"
