"""the tables an :class:`AclCache` is derived from, and how one changed row reaches it.

Every entry of an :class:`AclCache` is computed from rows of ``groups``, ``group_members``,
``roles``, ``role_assignments`` and ``namespaces`` (a per-namespace entry reads the namespace row's
customer and type). Those tables carry write generations (epoch-task-06), so each write to one of
them is broadcast row by row, and a registry that follows the table drops it when a broadcast was
missed. :func:`bind_acl_cache_to_access_tables` registers the cache with the registry as a cache
derived from each table (``CollectionRegistry.register_derived_cache``):

- a row evicts exactly the entries it reaches (:meth:`AclCache.evict_group_member_row` and its
  siblings), from the columns its collection declared to ride on the broadcast, or its key;
- a row whose broadcast does not say what it names (an older writer, a value that does not parse)
  empties only the layer derived from that table, because its reach is unknown, and says so in the
  log and in :attr:`DegradedEvictions`;
- a dropped table empties what was derived from it, and nothing more.

Table names and key positions are read off the collection classes, so a reordered or widened key
cannot quietly turn every heard row into an unknown reach.

The broadcasts heard here are the registry's own invalidation listener's, so the registry must have
one running. :func:`threetears.agent.acl.generation_follow.follow_access_tables` binds the cache and
follows the tables in one call; that is what a consumer calls.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.types import MemberType

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry

    from threetears.agent.acl.cache import AclCache

__all__ = ["ACCESS_TABLES", "DegradedEvictions", "bind_acl_cache_to_access_tables"]

log = get_logger(__name__)

_GROUPS: Final = GroupCollection.schema.name
_GROUP_MEMBERS: Final = GroupMemberCollection.schema.name
_ROLES: Final = RoleCollection.schema.name
_ROLE_ASSIGNMENTS: Final = RoleAssignmentCollection.schema.name
_NAMESPACES: Final = NamespaceCollection.schema.name

#: the tables every :class:`AclCache` entry is derived from
ACCESS_TABLES: Final = (_GROUPS, _GROUP_MEMBERS, _ROLES, _ROLE_ASSIGNMENTS, _NAMESPACES)

#: how often an unknown-reach eviction on one table is logged again while it keeps happening
_DEGRADED_LOG_INTERVAL_SECONDS: Final = 60.0


def _key_index(collection: type[Any], column: str) -> int:
    """where ``column`` sits in ``collection``'s declared primary key.

    :param collection: the collection class
    :ptype collection: type[Any]
    :param column: a primary-key column
    :ptype column: str
    :return: its position in the key a row broadcast carries
    :rtype: int
    """
    declared = collection.primary_key_column
    key: tuple[str, ...] = declared if isinstance(declared, tuple) else (declared,)
    return key.index(column)


_GROUP_ID_AT: Final = _key_index(GroupCollection, "group_id")
_ROLE_ID_AT: Final = _key_index(RoleCollection, "role_id")
_NAMESPACE_ID_AT: Final = _key_index(NamespaceCollection, "namespace_id")


class DegradedEvictions:
    """how many heard rows of each table had an unknown reach and emptied a layer, and why.

    A rising count means row-by-row eviction has stopped working for that table: an older writer
    still deployed, or a broadcast that lost its declared columns. Each table is logged at
    WARNING when it first happens, and again at most once a minute while it goes on.
    """

    def __init__(self) -> None:
        """start with nothing counted.

        :return: nothing
        :rtype: None
        """
        self._counts: Counter[str] = Counter()
        self._last_logged: dict[str, float] = {}

    @property
    def counts(self) -> dict[str, int]:
        """unknown-reach evictions so far, by table.

        :return: table to count
        :rtype: dict[str, int]
        """
        return dict(self._counts)

    def record(self, table: str, reason: str) -> None:
        """count one unknown-reach eviction, and log it when the table is due.

        :param table: the table whose row could not be placed
        :ptype table: str
        :param reason: what the broadcast lacked
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        self._counts[table] += 1
        now = time.monotonic()
        last = self._last_logged.get(table)
        if last is None or now - last >= _DEGRADED_LOG_INTERVAL_SECONDS:
            self._last_logged[table] = now
            log.warning(
                "an access-table row did not say what it reaches; the layer derived from the table was emptied",
                extra={"extra_data": {"table": table, "reason": reason, "count": self._counts[table]}},
            )


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


def _member_kind(value: str | None) -> str | None:
    """a broadcast's ``member_type`` as the membership key's actor kind, or ``None`` when it is not one.

    :param value: the value as the broadcast carried it
    :ptype value: str | None
    :return: the kind, spelled as :class:`MemberType` spells it
    :rtype: str | None
    """
    result: str | None = None
    if value is not None:
        try:
            result = MemberType(value).value
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


def bind_acl_cache_to_access_tables(
    registry: CollectionRegistry,
    cache: AclCache,
    *,
    degraded: DegradedEvictions | None = None,
) -> Callable[[], None]:
    """make ``cache`` a cache derived from the access tables on ``registry``.

    A consumer calls :func:`threetears.agent.acl.generation_follow.follow_access_tables`, which
    binds through this and follows the tables; binding alone never drops a table whose broadcast
    was missed.

    :param registry: the registry whose invalidation listener hears the tables' row broadcasts
    :ptype registry: CollectionRegistry
    :param cache: the cache to evict from
    :ptype cache: AclCache
    :param degraded: where unknown-reach evictions are counted; a new record when ``None``
    :ptype degraded: DegradedEvictions | None
    :return: a call that removes the registrations
    :rtype: Callable[[], None]
    """
    record = degraded if degraded is not None else DegradedEvictions()

    def on_group_member(message: CacheInvalidationMessage) -> None:
        kind = _member_kind(_column(message, "member_type"))
        member_id = _uuid(_column(message, "member_id"))
        if kind is None or member_id is None:
            record.record(_GROUP_MEMBERS, "no member_type and member_id it parses")
            cache.drop_membership_layer()
        else:
            cache.evict_group_member_row(kind, member_id)

    def on_role_assignment(message: CacheInvalidationMessage) -> None:
        group_id = _uuid(_column(message, "group_id"))
        if group_id is None:
            record.record(_ROLE_ASSIGNMENTS, "no group_id it parses")
            cache.drop_assignment_layers()
        else:
            cache.evict_role_assignment_row(group_id)

    def on_role(message: CacheInvalidationMessage) -> None:
        role_id = _uuid(_key_part(message, _ROLE_ID_AT))
        if role_id is None:
            record.record(_ROLES, "no role_id in its key")
            cache.drop_assignment_layers()
        else:
            cache.evict_role_row(role_id)

    def on_group(message: CacheInvalidationMessage) -> None:
        group_id = _uuid(_key_part(message, _GROUP_ID_AT))
        if group_id is None:
            record.record(_GROUPS, "no group_id in its key")
            cache.invalidate_all()
        else:
            cache.evict_group_row(group_id)

    def on_namespace(message: CacheInvalidationMessage) -> None:
        namespace_id = _uuid(_key_part(message, _NAMESPACE_ID_AT))
        if namespace_id is None:
            record.record(_NAMESPACES, "no namespace_id in its key")
            cache.drop_assignment_layers()
        else:
            cache.invalidate_namespace(namespace_id)

    removers = (
        registry.register_derived_cache(
            _GROUP_MEMBERS, on_row=on_group_member, on_table_dropped=cache.drop_membership_layer
        ),
        registry.register_derived_cache(
            _ROLE_ASSIGNMENTS, on_row=on_role_assignment, on_table_dropped=cache.drop_assignment_layers
        ),
        registry.register_derived_cache(_ROLES, on_row=on_role, on_table_dropped=cache.drop_assignment_layers),
        registry.register_derived_cache(_GROUPS, on_row=on_group, on_table_dropped=cache.invalidate_all),
        registry.register_derived_cache(
            _NAMESPACES, on_row=on_namespace, on_table_dropped=cache.drop_assignment_layers
        ),
    )
    log.debug("acl cache bound to the access tables' row broadcasts", extra={"extra_data": {"tables": ACCESS_TABLES}})

    def remove() -> None:
        for remover in removers:
            remover()

    return remove
