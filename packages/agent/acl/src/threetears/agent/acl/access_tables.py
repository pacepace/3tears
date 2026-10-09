"""the four access tables an :class:`AclCache` is derived from, and how one changed row reaches it.

Every entry of an :class:`AclCache` is computed from rows of ``groups``, ``group_members``,
``roles`` and ``role_assignments``. Those tables carry write generations (epoch-task-06), so each
write to one of them is broadcast row by row, and a registry that follows the table drops it when a
broadcast was missed. :func:`bind_acl_cache_to_access_tables` registers the cache with the
registry as a cache derived from each table (``CollectionRegistry.register_derived_cache``):

- a row evicts exactly the entries it reaches (:meth:`AclCache.evict_group_member_row` and its
  siblings), from the columns its collection declared to ride on the broadcast;
- a row whose broadcast does not say what it names (an older writer) empties only the layer
  derived from that table, because its reach is unknown;
- a dropped table empties what was derived from it, and nothing more.

The broadcasts heard here are the registry's own invalidation listener's, so the registry must
have one running; following the generations is
:class:`threetears.agent.acl.generation_follow.AccessTableFollower`'s job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from uuid import UUID

from threetears.observe import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry

    from threetears.agent.acl.cache import AclCache

__all__ = ["ACCESS_TABLES", "bind_acl_cache_to_access_tables"]

log = get_logger(__name__)

#: the tables every :class:`AclCache` entry is derived from
ACCESS_TABLES: Final = ("groups", "group_members", "roles", "role_assignments")


def _uuid(value: str | None) -> UUID | None:
    """a broadcast's string value as a UUID, or ``None`` when absent or not one.

    :param value: the value as the broadcast carried it
    :ptype value: str | None
    :return: the UUID, or ``None``
    :rtype: UUID | None
    """
    result: UUID | None = None
    if value is not None:
        try:
            result = UUID(value)
        except ValueError:
            result = None
    return result


def _column(message: CacheInvalidationMessage, name: str) -> str | None:
    """one declared column's value as the broadcast carried it, or ``None``.

    :param message: the row broadcast
    :ptype message: CacheInvalidationMessage
    :param name: the column
    :ptype name: str
    :return: the value, or ``None`` when the broadcast carries no columns or not this one
    :rtype: str | None
    """
    return None if message.columns is None else message.columns.get(name)


def _key_part(message: CacheInvalidationMessage, index: int) -> str | None:
    """one primary-key value of the row, or ``None`` when the key has fewer parts.

    :param message: the row broadcast
    :ptype message: CacheInvalidationMessage
    :param index: the position in the declared primary key
    :ptype index: int
    :return: the value
    :rtype: str | None
    """
    return message.ids[index] if len(message.ids) > index else None


def bind_acl_cache_to_access_tables(registry: CollectionRegistry, cache: AclCache) -> Callable[[], None]:
    """make ``cache`` a cache derived from the four access tables on ``registry``.

    :param registry: the registry whose invalidation listener hears the tables' row broadcasts
    :ptype registry: CollectionRegistry
    :param cache: the cache to evict from
    :ptype cache: AclCache
    :return: a call that removes the four registrations
    :rtype: Callable[[], None]
    """

    def on_group_member(message: CacheInvalidationMessage) -> None:
        member_id = _uuid(_column(message, "member_id"))
        cache.evict_group_member_row(_column(message, "member_type"), member_id)

    def on_role_assignment(message: CacheInvalidationMessage) -> None:
        cache.evict_role_assignment_row(_uuid(_column(message, "group_id")))

    def on_role(message: CacheInvalidationMessage) -> None:
        # primary key ``role_id``
        role_id = _uuid(_key_part(message, 0))
        if role_id is None:
            cache.drop_assignment_layers()
        else:
            cache.evict_role_row(role_id)

    def on_group(message: CacheInvalidationMessage) -> None:
        # primary key ``(row_scope, group_id)``
        group_id = _uuid(_key_part(message, 1))
        if group_id is None:
            cache.invalidate_all()
        else:
            cache.evict_group_row(group_id)

    removers = (
        registry.register_derived_cache(
            "group_members", on_row=on_group_member, on_table_dropped=cache.drop_membership_layer
        ),
        registry.register_derived_cache(
            "role_assignments", on_row=on_role_assignment, on_table_dropped=cache.drop_assignment_layers
        ),
        registry.register_derived_cache("roles", on_row=on_role, on_table_dropped=cache.drop_assignment_layers),
        registry.register_derived_cache("groups", on_row=on_group, on_table_dropped=cache.invalidate_all),
    )
    log.debug("acl cache bound to the access tables' row broadcasts", extra={"extra_data": {"tables": ACCESS_TABLES}})

    def remove() -> None:
        for remover in removers:
            remover()

    return remove
