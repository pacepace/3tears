"""entity classes for the canonical rbac collections.

these :class:`BaseEntity` subclasses front the five rbac tables every
3tears app shares: ``groups``, ``group_members``, ``roles``,
``role_assignments``, and ``namespaces``. fields match the canonical
column shape (no app-specific table prefix; ``platform.`` qualification
is set on the L3 pool's ``search_path``, not in the schema name here).

each entity carries no business logic; the rbac evaluator
(:func:`threetears.agent.acl.evaluate_decision`) speaks in frozen
dataclasses (:class:`Group`, :class:`Role`, :class:`RoleAssignment`,
:class:`GroupMembership`) and the loaders (in
:mod:`threetears.agent.acl.loaders`) translate rows into those
dataclasses at the loader boundary. the entity subclasses exist purely
so callers have a canonical place to read / write / cache rows via the
standard three-tier collection api.

four of the five tables use composite primary keys post-row_scope
partitioning (``(row_scope, id)`` for ``groups`` /
``role_assignments`` / ``namespaces``; ``(group_id, id)`` for
``group_members``). every composite-pk entity overrides ``__init__`` to
expose ``_id`` as the matching tuple so :meth:`BaseCollection.normalize_pk`
+ :meth:`BaseCollection.l2_key` address the row uniformly across L1 /
L2 / L3. ``entity.id`` keeps returning the scalar row id via a property
override so callers (response models, audit trails, log lines) read the
same value across pre/post partitioning.
"""

from __future__ import annotations

from typing import Any

from threetears.core.entities.base import BaseEntity

__all__ = [
    "GroupEntity",
    "GroupMemberEntity",
    "NamespaceEntity",
    "RoleAssignmentEntity",
    "RoleEntity",
]


class GroupEntity(BaseEntity):
    """row in ``groups``.

    composite primary key ``(row_scope, id)``. the row_scope column
    discriminates platform-scope groups (``customer_id IS NULL``) from
    customer-scope groups; a single CHECK constraint at the database
    layer pins the invariant ``row_scope='platform' <-> customer_id IS
    NULL``.

    fields: ``row_scope`` / ``id`` / ``customer_id`` / ``name`` /
    ``description`` / ``date_created`` / ``date_updated``.
    """

    primary_key_field: str = "row_scope"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite-pk ``_id`` tuple.

        :param data: row dict carrying both ``row_scope`` and ``id``;
            ``row_scope`` is auto-derived from ``customer_id`` when
            absent so callers / fixtures keep their pre-partition shape
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        row_scope = data.get("row_scope")
        if row_scope is None:
            row_scope = "platform" if data.get("customer_id") is None else "customer"
            data = {**data, "row_scope": row_scope}
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_row_id", data["id"])
        object.__setattr__(self, "_id", (row_scope, data["id"]))

    @property
    def id(self) -> Any:
        """return scalar group UUID (pre-partition contract preserved).

        :return: group UUID
        :rtype: Any
        """
        return self._row_id


class GroupMemberEntity(BaseEntity):
    """row in ``group_members``.

    composite primary key ``(group_id, id)``. group_id partitions the
    table so per-group listing reads stay co-located.

    fields: ``group_id`` / ``id`` / ``member_type`` (``user`` or
    ``agent``) / ``member_id`` / ``customer_id`` / ``date_added``.
    """

    primary_key_field: str = "group_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite-pk ``_id`` tuple.

        :param data: row dict carrying both ``group_id`` and ``id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_row_id", data["id"])
        object.__setattr__(self, "_id", (data["group_id"], data["id"]))

    @property
    def id(self) -> Any:
        """return scalar membership UUID (pre-partition contract preserved).

        :return: membership UUID
        :rtype: Any
        """
        return self._row_id


class RoleEntity(BaseEntity):
    """row in ``roles``.

    not partitioned: small row count, by-name lookup shape, and the
    table acts as an FK target via ``role_assignments.role_id``.

    fields: ``id`` / ``name`` / ``description`` / ``permissions`` /
    ``is_builtin`` / ``date_created`` / ``date_updated``.
    """

    primary_key_field: str = "id"


class RoleAssignmentEntity(BaseEntity):
    """row in ``role_assignments``.

    composite primary key ``(row_scope, id)``. the row_scope column
    discriminates platform-scope assignments (``scope_type='all'`` or
    ``scope_type='type_customer'`` with NULL ``scope_customer_id``)
    from customer-scope assignments.

    fields: ``row_scope`` / ``id`` / ``role_id`` / ``group_id`` /
    ``scope_type`` / ``scope_namespace_id`` / ``scope_namespace_type``
    / ``scope_customer_id`` / ``granted_by`` / ``date_granted`` /
    ``managed_by``.
    """

    primary_key_field: str = "row_scope"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite-pk ``_id`` tuple.

        :param data: row dict carrying both ``row_scope`` and ``id``;
            ``row_scope`` is auto-derived from the scope shape when
            absent
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        row_scope = data.get("row_scope")
        if row_scope is None:
            scope_type = data.get("scope_type")
            scope_customer_id = data.get("scope_customer_id")
            if scope_type == "all":
                row_scope = "platform"
            elif scope_type == "type_customer" and scope_customer_id is None:
                row_scope = "platform"
            else:
                row_scope = "customer"
            data = {**data, "row_scope": row_scope}
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_row_id", data["id"])
        object.__setattr__(self, "_id", (row_scope, data["id"]))

    @property
    def id(self) -> Any:
        """return scalar assignment UUID (pre-partition contract preserved).

        :return: assignment UUID
        :rtype: Any
        """
        return self._row_id


class NamespaceEntity(BaseEntity):
    """row in ``namespaces``.

    composite primary key ``(row_scope, id)``. namespace rows are the
    target side of every authorization check; ``namespace_type`` carries
    the resource-type discriminator (``workspace`` / ``agent`` /
    ``shared`` / ``system`` / ...) the evaluator routes on.

    fields: ``row_scope`` / ``id`` / ``name`` / ``namespace_type`` /
    ``owner_agent_id`` / ``customer_id`` / ``schema_name`` /
    ``metadata`` / ``date_created`` / ``date_updated``.
    """

    primary_key_field: str = "row_scope"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite-pk ``_id`` tuple.

        :param data: row dict carrying both ``row_scope`` and ``id``;
            ``row_scope`` is auto-derived from ``customer_id`` when
            absent
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        row_scope = data.get("row_scope")
        if row_scope is None:
            row_scope = "platform" if data.get("customer_id") is None else "customer"
            data = {**data, "row_scope": row_scope}
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_row_id", data["id"])
        object.__setattr__(self, "_id", (row_scope, data["id"]))

    @property
    def id(self) -> Any:
        """return scalar namespace UUID (pre-partition contract preserved).

        :return: namespace UUID
        :rtype: Any
        """
        return self._row_id
