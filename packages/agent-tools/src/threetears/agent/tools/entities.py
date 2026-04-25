"""Context item entity — thin cache proxy for conversation context records."""

from __future__ import annotations

from typing import Any

from threetears.core.entities.base import BaseEntity

__all__ = [
    "ContextItemEntity",
]


class ContextItemEntity(BaseEntity):
    """Entity representing a conversation context item.

    Covers all context types: variables, tool results, and media slots.
    All field access is via BaseEntity's ``__getattr__`` proxy. composite
    primary key on ``(conversation_id, context_id)`` so the entity
    addresses the partition slot it belongs to; ``_id`` is the tuple
    ``(conversation_id, context_id)`` after construction.
    """

    primary_key_field: str = "conversation_id"

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

        :param data: row dict; must carry ``conversation_id`` and
            ``context_id`` keys
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(
            self, "_id", (data["conversation_id"], data["context_id"]),
        )
