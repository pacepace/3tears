"""Context item entity — thin cache proxy for conversation context records."""

from __future__ import annotations

from threetears.core.entities.base import BaseEntity


class ContextItemEntity(BaseEntity):
    """Entity representing a conversation context item.

    Covers all context types: variables, tool results, and media slots.
    All field access is via BaseEntity's ``__getattr__`` proxy.
    """

    _primary_key_field: str = "context_id"
