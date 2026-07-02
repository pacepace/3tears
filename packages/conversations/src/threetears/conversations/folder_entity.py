"""
Folder entity -- thin cache proxy for the per-agent ``folders`` table.

a folder is an app-agnostic, mutable, per-owner named container that
groups conversations. it mirrors the multi-tenant scoping of
:class:`~threetears.conversations.entity.Conversation`: every row
carries agent + customer + user scoping and a human-facing ``name``,
plus a free-form ``metadata`` JSONB blob where app-specific
presentation bits (color, sort_order, icon) live so the canonical
shape stays app-agnostic.

a product-side folder primitive so multiple apps
that group conversations under named containers reuse one canonical
value type instead of re-inventing a per-product table.

composite primary key on ``(agent_id, folder_id)`` so rows are
partitioned per agent; ``_id`` holds the ``(agent_id, folder_id)``
tuple after construction (same discipline as :class:`Conversation`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "Folder",
]


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


class Folder(BaseEntity):
    """
    cache proxy entity for one row in the ``folders`` table.

    the folders table tracks per-owner named containers an agent's
    users organise their conversations into. every row carries agent +
    customer + user scoping, a display ``name`` (unique per
    ``(agent_id, user_id)``), an optional free-form ``metadata`` blob,
    and two timestamps. consumer apps read these properties to render
    folder lists and to associate conversations (via the mutable
    ``conversations.folder_id`` column) with the folder's identity and
    scope.

    composite primary key on ``(agent_id, folder_id)`` so rows are
    partitioned per agent; ``_id`` holds the ``(agent_id, folder_id)``
    tuple after construction.

    app-specific presentation (color, sort order, icon) is intentionally
    NOT promoted to columns -- it lives in ``metadata`` so the canonical
    shape stays app-agnostic and a new consumer never has to migrate the
    table to carry its own bits.

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

        :param data: row dict; must carry both ``agent_id`` and
            ``folder_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_id", (data["agent_id"], data["folder_id"]))

    @property
    def agent_id(self) -> UUID:
        """
        return the owning agent identifier.

        :return: agent UUID for the folder
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
    def folder_id(self) -> UUID:
        """
        return the folder identifier.

        :return: folder UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("folder_id"))

    @folder_id.setter
    def folder_id(self, value: UUID) -> None:
        """
        set the folder identifier.

        :param value: folder UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "folder_id", value)

    @property
    def customer_id(self) -> UUID:
        """
        return the owning customer identifier.

        :return: customer UUID for the folder
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
        return the owning user identifier.

        folders are scoped per user: the ``(agent_id, user_id, name)``
        uniqueness constraint pins a folder name to one owner so two
        users under the same agent can both have a folder named
        ``"Work"``.

        :return: user UUID for the folder
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """
        set the owning user identifier.

        :param value: user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def name(self) -> str:
        """
        return the folder display name.

        unique within ``(agent_id, user_id)`` (enforced by the v008
        migration); presented directly in folder-list UIs.

        :return: display name
        :rtype: str
        """
        value: str = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str) -> None:
        """
        set the folder display name.

        :param value: display name
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "name", value)

    @property
    def metadata(self) -> dict[str, Any] | None:
        """
        return the free-form metadata JSONB blob.

        consumer apps carry presentation bits here (``color``,
        ``sort_order``, ``icon``, ...) that are intentionally not
        promoted to columns so the canonical folder shape stays
        app-agnostic. always a dict or ``None``; never a list.

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
