"""three-tier collections for the canonical rbac tables.

every 3tears app shares the five rbac tables: ``groups``,
``group_members``, ``roles``, ``role_assignments``, and
``namespaces``. the schemas are universal (column shapes, constraints,
partition discriminators) so the Collections live here and
deploying apps subclass them only when admin-specific query shapes
need to ride alongside the canonical CRUD + evaluator-loader paths.

scope of the canonical Collections (kept generic; nothing app-specific
leaks in):

- table-level CRUD via :class:`SchemaBackedCollection` and the
  declarative :class:`TableSchema`
- evaluator-loader query methods that the canonical
  :class:`threetears.agent.acl.MembershipLoader` and
  :class:`threetears.agent.acl.GrantLoader` Protocols call into
  (``load_for_user`` / ``load_for_agent`` / ``load_for_groups`` /
  ``get_many``)
- bulk fetch by id list (every app needs this for the introspection /
  audit / grant-resolver paths)
- a small set of universally-useful list / find queries

scope explicitly out of bounds for the canonical classes (lives on
deploying-app subclasses):

- admin-endpoint dynamic ``list_by_filter`` shapes
- per-cardinality counts driving deploy-specific audit envelopes
- discovery JOINs that span an app-specific multi-table query

table names use the canonical RBAC vocabulary (``groups`` etc.)
without any deploy-specific schema prefix; the prefix (``platform.``
in the aibots hub deployment) is applied at the L3 pool's
``search_path``, not in the schema name on the Collection.
"""

from __future__ import annotations

import json as _json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIME_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.observe import get_logger

from threetears.agent.acl.entities import (
    GroupEntity,
    GroupMemberEntity,
    NamespaceEntity,
    RoleAssignmentEntity,
    RoleEntity,
)
from threetears.agent.acl.types import (
    GroupMembership,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
)

log = get_logger(__name__)

__all__ = [
    "GroupCollection",
    "GroupMemberCollection",
    "NamespaceCollection",
    "RoleAssignmentCollection",
    "RoleCollection",
]


def _coerce_uuid(raw: Any) -> UUID | None:
    """coerce a database cell value to a :class:`UUID` (or ``None``).

    asyncpg returns ``UUID`` columns as native :class:`UUID` instances
    on a direct asyncpg pool; the agent-side
    :class:`NatsProxyL3Backend` pool round-trips rows through JSON
    which collapses UUIDs to their string representation. Collections
    whose method return types carry UUID fields (e.g.
    :class:`GroupMembership.member_id`) must normalize both shapes so
    callers get a stable Python type regardless of which pool answered
    the query. ``None`` passes through; any other type is passed to
    :class:`UUID`'s constructor via ``str()`` conversion.

    :param raw: value pulled directly from a row dictionary
    :ptype raw: Any
    :return: UUID instance, or ``None`` when the input is ``None``
    :rtype: UUID | None
    """
    result: UUID | None = None
    if raw is not None:
        if isinstance(raw, UUID):
            result = raw
        else:
            result = UUID(str(raw))
    return result


def _coerce_role_permissions(raw: Any) -> dict[str, frozenset[str]]:
    """coerce a JSONB ``permissions`` payload into ``{resource: frozenset(action)}``.

    asyncpg returns JSONB columns either as a parsed ``dict`` (when
    the connection has the JSONB codec registered) or as the raw
    ``str``; both shapes land here and normalize to the dataclass-
    friendly mapping that :class:`Role` expects.

    :param raw: raw JSONB column value as returned by asyncpg
    :ptype raw: Any
    :return: normalized permissions mapping
    :rtype: dict[str, frozenset[str]]
    """
    parsed: dict[str, Any] = {}
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw:
        loaded = _json.loads(raw)
        if isinstance(loaded, dict):
            parsed = loaded
    result: dict[str, frozenset[str]] = {}
    for resource_type, actions in parsed.items():
        if isinstance(actions, list):
            result[resource_type] = frozenset(str(a) for a in actions)
    return result


# ---------------------------------------------------------------------------
# GroupCollection
# ---------------------------------------------------------------------------


class GroupCollection(SchemaBackedCollection[GroupEntity]):
    """three-tier collection for ``groups`` rows.

    groups use hard-delete. cascading FKs on ``group_members`` and
    ``role_assignments`` (``ON DELETE CASCADE``) clean up member +
    assignment rows in the same transaction, so the collection only
    needs to delete the group row itself. CRUD comes from the
    declarative :class:`TableSchema`; the evaluator-loader / introspection
    helpers (``list_by_customer`` / ``list_all`` / ``get_many``) stay on
    the canonical class because every rbac-consuming app needs them.
    """

    primary_key_column: tuple[str, ...] = ("row_scope", "id")
    _partition_exempt_methods = frozenset(
        {
            "list_by_customer",
            "list_all",
            "get_many",
            "delete_from_postgres",
            "save_entity",
            "create",
            "find_by_id",
        }
    )
    schema = TableSchema(
        name="groups",
        primary_key=("row_scope", "id"),
        columns=[
            Column("row_scope", STRING_TYPE, partition=True),
            Column("id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column("name", STRING_TYPE),
            Column("description", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"groups"``
        :rtype: str
        """
        return "groups"

    @property
    def entity_class(self) -> type[GroupEntity]:
        """return entity class for this collection.

        :return: :class:`GroupEntity`
        :rtype: type[GroupEntity]
        """
        return GroupEntity

    def create(self, data: dict[str, Any]) -> GroupEntity:
        """construct new group entity, auto-deriving ``row_scope``.

        every group row carries ``customer_id`` (nullable; platform-
        scoped groups have ``customer_id IS NULL``); the partition
        column ``row_scope`` is the defensive discriminator
        (``platform`` / ``customer``) and the database CHECK constraint
        pins ``row_scope='platform' <-> customer_id IS NULL`` at the
        row level. this override sets row_scope from customer_id so
        callers keep their pre-partition shape.

        :param data: row payload; may omit ``row_scope`` (override
            sets it) or include it (override leaves explicit values
            untouched)
        :ptype data: dict[str, Any]
        :return: newly constructed (not-yet-persisted) group entity
        :rtype: GroupEntity
        """
        if "row_scope" not in data:
            customer_id = data.get("customer_id")
            data = {
                **data,
                "row_scope": "platform" if customer_id is None else "customer",
            }
        return super().create(data)

    async def find_by_id(
        self,
        group_id: UUID,
    ) -> GroupEntity | None:
        """resolve group by ``id`` alone via the ``UNIQUE (id)`` constraint.

        every endpoint that takes ``{group_id}`` in the URL knows the
        row's id but not the partition column ``row_scope``. uniqueness
        is preserved by the table-level ``UNIQUE (id)`` constraint so
        an id-only fetch is unambiguous.

        :param group_id: group UUID
        :ptype group_id: UUID
        :return: group entity or ``None`` when no row exists
        :rtype: GroupEntity | None
        """
        result: GroupEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM groups WHERE id = $1",
                group_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def list_by_customer(
        self,
        customer_id: UUID,
    ) -> list[GroupEntity]:
        """list every ``groups`` row owned by ``customer_id``.

        rows are promoted into L1/L2 caches for subsequent ``get(id)``
        lookups. returns an empty list (never ``None``) when the
        customer has no groups so callers can iterate unconditionally.

        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: list of group entities ordered by ``date_created``
            ascending
        :rtype: list[GroupEntity]
        """
        result: list[GroupEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM groups
                 WHERE row_scope = 'customer'
                   AND customer_id = $1
                 ORDER BY date_created ASC
                """,
                customer_id,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def get_many(
        self,
        group_ids: Sequence[UUID],
    ) -> list[GroupEntity]:
        """fetch every group row whose id is in ``group_ids``.

        used by the introspection / grant-loader paths that need a
        bulk group lookup keyed by the assignment rows' ``group_id``.
        row order is database-determined (no ``ORDER BY`` clause)
        since the evaluator consumes the result as an unordered map;
        callers that need a specific order should sort the returned
        list themselves.

        empty input short-circuits without a SQL round-trip and
        returns an empty list. promotes resolved rows into L1/L2
        caches.

        :param group_ids: sequence of group UUIDs to resolve
        :ptype group_ids: Sequence[UUID]
        :return: list of group entities (subset of ``group_ids`` that
            exist in L3); order is database-determined
        :rtype: list[GroupEntity]
        """
        result: list[GroupEntity] = []
        if self.l3_pool is not None and len(group_ids) > 0:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM groups WHERE id = ANY($1::uuid[])",
                list(group_ids),
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                # the NATS proxy pool round-trips UUID columns through
                # JSON which collapses them to strings; the schema's
                # _coerce_row handles UUID columns it knows about, but
                # belt-and-suspenders for the two pk-adjacent columns.
                if "id" in data:
                    data["id"] = _coerce_uuid(data["id"])
                if "customer_id" in data:
                    data["customer_id"] = _coerce_uuid(data["customer_id"])
                self.write_to_cache_sync(data)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def list_all(
        self,
        customer_id: UUID | None = None,
    ) -> list[GroupEntity]:
        """list every ``groups`` row, optionally filtered by customer.

        platform admins may list every group (``customer_id=None``) or
        scope to a specific customer; customer admins pass their own
        ``customer_id``. rows are ordered by ``date_created`` ascending
        and promoted into L1/L2 caches.

        :param customer_id: optional customer scope; ``None`` returns
            every row
        :ptype customer_id: UUID | None
        :return: list of group entities ordered by ``date_created``
            ascending
        :rtype: list[GroupEntity]
        """
        result: list[GroupEntity] = []
        if self.l3_pool is not None:
            if customer_id is None:
                rows = await self.l3_pool.fetch(
                    """
                    SELECT * FROM groups
                     WHERE row_scope IN ('platform', 'customer')
                     ORDER BY date_created ASC
                    """,
                )
            else:
                rows = await self.l3_pool.fetch(
                    """
                    SELECT * FROM groups
                     WHERE row_scope = 'customer'
                       AND customer_id = $1
                     ORDER BY date_created ASC
                    """,
                    customer_id,
                )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result


# ---------------------------------------------------------------------------
# GroupMemberCollection
# ---------------------------------------------------------------------------


class GroupMemberCollection(SchemaBackedCollection[GroupMemberEntity]):
    """three-tier collection for ``group_members`` rows.

    CRUD comes from the declarative :class:`TableSchema`;
    evaluator-loader queries (``load_for_user`` / ``load_for_agent`` /
    ``list_by_group`` / ``find_by_group_and_id``) stay on the canonical
    class.
    """

    primary_key_column: tuple[str, ...] = ("group_id", "id")
    _partition_exempt_methods = frozenset(
        {
            "load_for_user",
            "load_for_agent",
            "delete_from_postgres",
            "save_entity",
        }
    )
    schema = TableSchema(
        name="group_members",
        primary_key=("group_id", "id"),
        columns=[
            Column("id", UUID_TYPE),
            Column("group_id", UUID_TYPE, partition=True),
            Column("member_type", STRING_TYPE, immutable=True),
            Column("member_id", UUID_TYPE, immutable=True),
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column("date_added", DATETIMETZ_TYPE, immutable=True),
        ],
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"group_members"``
        :rtype: str
        """
        return "group_members"

    @property
    def entity_class(self) -> type[GroupMemberEntity]:
        """return entity class for this collection.

        :return: :class:`GroupMemberEntity`
        :rtype: type[GroupMemberEntity]
        """
        return GroupMemberEntity

    async def load_for_user(
        self,
        user_id: UUID,
    ) -> list[GroupMembership]:
        """resolve ``user_id`` to its :class:`GroupMembership` rows.

        returns the protocol-shape :class:`GroupMembership` dataclass
        instances (NOT :class:`GroupMemberEntity`) because the unified
        evaluator's :class:`MembershipLoader` Protocol speaks in the
        ACL types. rows are NOT promoted into L1/L2 because the ACL
        flow does not re-read by primary key; the evaluator consumes
        the membership list, not per-row entities.

        empty result is an empty list (never ``None``).

        :param user_id: user UUID to resolve
        :ptype user_id: UUID
        :return: list of memberships naming ``user_id`` as a user
            member
        :rtype: list[GroupMembership]
        """
        result: list[GroupMembership] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM group_members
                 WHERE member_type = 'user'
                   AND member_id = $1
                """,
                user_id,
            )
            result = [
                GroupMembership(
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    member_type=MemberType(row["member_type"]),
                    member_id=_coerce_uuid(row["member_id"]),  # type: ignore[arg-type]
                    customer_id=_coerce_uuid(row["customer_id"]),
                )
                for row in rows
            ]
        return result

    async def load_for_agent(
        self,
        agent_id: UUID,
    ) -> list[GroupMembership]:
        """resolve ``agent_id`` to its :class:`GroupMembership` rows.

        symmetric counterpart of :meth:`load_for_user` for the agent
        side of an intersection evaluation.

        :param agent_id: agent UUID to resolve
        :ptype agent_id: UUID
        :return: list of memberships naming ``agent_id`` as an agent
            member
        :rtype: list[GroupMembership]
        """
        result: list[GroupMembership] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM group_members
                 WHERE member_type = 'agent'
                   AND member_id = $1
                """,
                agent_id,
            )
            result = [
                GroupMembership(
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    member_type=MemberType(row["member_type"]),
                    member_id=_coerce_uuid(row["member_id"]),  # type: ignore[arg-type]
                    customer_id=_coerce_uuid(row["customer_id"]),
                )
                for row in rows
            ]
        return result

    async def list_by_group(
        self,
        group_id: UUID,
    ) -> list[GroupMemberEntity]:
        """list every membership row for ``group_id`` ordered by ``date_added``.

        rows are promoted into L1/L2 caches so subsequent ``get(id)``
        calls hit L1.

        :param group_id: owning group UUID
        :ptype group_id: UUID
        :return: list of membership entities ordered by ``date_added``
            ascending
        :rtype: list[GroupMemberEntity]
        """
        result: list[GroupMemberEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM group_members
                 WHERE group_id = $1
                 ORDER BY date_added ASC
                """,
                group_id,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def find_by_group_and_id(
        self,
        group_id: UUID,
        member_row_id: UUID,
    ) -> GroupMemberEntity | None:
        """fetch a membership by PK and assert it belongs to ``group_id``.

        used where the URL carries ``(group_id, member_row_id)`` and
        the caller wants to fail closed (404) when the membership row
        exists but lives in a different group. the method is a
        :meth:`BaseCollection.get` followed by an in-Python
        ``group_id`` predicate check so the three-tier cache path
        (L1 -> L2 -> L3) answers the fetch.

        :param group_id: expected owning group UUID
        :ptype group_id: UUID
        :param member_row_id: ``group_members.id`` UUID
        :ptype member_row_id: UUID
        :return: membership entity, or ``None`` when the row is absent
            OR exists but belongs to a different group
        :rtype: GroupMemberEntity | None
        """
        entity = await self.get((group_id, member_row_id))
        result: GroupMemberEntity | None = None
        if entity is not None:
            data = entity.to_dict()
            if data.get("group_id") == group_id:
                result = entity
        return result


# ---------------------------------------------------------------------------
# RoleCollection
# ---------------------------------------------------------------------------


class RoleCollection(SchemaBackedCollection[RoleEntity]):
    """three-tier collection for ``roles`` rows.

    roles use hard-delete; admin endpoints typically guard against
    deleting builtins or any role that is referenced by an assignment
    (the ``role_assignments.role_id`` FK is ``ON DELETE RESTRICT``).
    CRUD comes from the declarative :class:`TableSchema`;
    evaluator-loader queries (``list_all`` / ``list_builtin`` /
    ``get_many``) stay on the canonical class.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="roles",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("description", STRING_TYPE),
            Column("permissions", JSONB_TYPE),
            Column("is_builtin", BOOL_TYPE, immutable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"roles"``
        :rtype: str
        """
        return "roles"

    @property
    def entity_class(self) -> type[RoleEntity]:
        """return entity class for this collection.

        :return: :class:`RoleEntity`
        :rtype: type[RoleEntity]
        """
        return RoleEntity

    async def list_all(self) -> list[RoleEntity]:
        """list every role row ordered by ``date_created`` ascending.

        rows are promoted into L1/L2 caches so subsequent ``get(id)``
        calls hit L1.

        :return: list of role entities ordered by ``date_created``
            ascending
        :rtype: list[RoleEntity]
        """
        result: list[RoleEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM roles ORDER BY date_created ASC",
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def list_builtin(self) -> list[RoleEntity]:
        """list every platform-shipped builtin role.

        rows are promoted into L1/L2 caches so subsequent ``get(id)``
        calls hit L1.

        :return: list of builtin role entities ordered by ``name``
            ascending
        :rtype: list[RoleEntity]
        """
        result: list[RoleEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM roles
                 WHERE is_builtin = TRUE
                 ORDER BY name ASC
                """,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def get_many(
        self,
        role_ids: Sequence[UUID],
    ) -> list[Role]:
        """resolve ``role_ids`` to :class:`Role` rows.

        returns the protocol-shape :class:`Role` dataclass instances
        (NOT :class:`RoleEntity`) because the unified evaluator's
        :class:`GrantLoader` Protocol speaks in the ACL types.

        empty input short-circuits without a SQL round-trip and
        returns an empty list. order is database-determined; the
        evaluator consumes the result as an unordered collection.

        :param role_ids: sequence of role UUIDs to resolve
        :ptype role_ids: Sequence[UUID]
        :return: list of :class:`Role` instances (subset of
            ``role_ids`` that exist in L3)
        :rtype: list[Role]
        """
        result: list[Role] = []
        if self.l3_pool is not None and len(role_ids) > 0:
            rows = await self.l3_pool.fetch(
                """
                SELECT id, name, permissions, is_builtin
                  FROM roles
                 WHERE id = ANY($1::uuid[])
                """,
                list(role_ids),
            )
            result = [
                Role(
                    id=_coerce_uuid(row["id"]),  # type: ignore[arg-type]
                    name=row["name"],
                    permissions=_coerce_role_permissions(row["permissions"]),
                    is_built_in=bool(row["is_builtin"]),
                )
                for row in rows
            ]
        return result


# ---------------------------------------------------------------------------
# RoleAssignmentCollection
# ---------------------------------------------------------------------------


class RoleAssignmentCollection(SchemaBackedCollection[RoleAssignmentEntity]):
    """three-tier collection for ``role_assignments`` rows.

    CRUD comes from the declarative :class:`TableSchema`;
    evaluator-loader queries (``load_for_groups`` /
    ``ensure_group_role_assignment`` / ``delete_by_group_and_scope``)
    stay on the canonical class. apps that need admin-specific filter /
    listing shapes (``list_by_filter`` / ``list_by_namespace`` /
    ``count_by_*``) subclass and add their own methods.
    """

    primary_key_column: tuple[str, ...] = ("row_scope", "id")
    _partition_exempt_methods = frozenset(
        {
            "load_for_groups",
            "ensure_group_role_assignment",
            "delete_by_group_and_scope",
            "delete_from_postgres",
            "save_entity",
            "create",
            "find_by_id",
        }
    )
    schema = TableSchema(
        name="role_assignments",
        primary_key=("row_scope", "id"),
        columns=[
            Column("row_scope", STRING_TYPE, partition=True),
            Column("id", UUID_TYPE),
            Column("role_id", UUID_TYPE, immutable=True),
            Column("group_id", UUID_TYPE, immutable=True),
            Column("scope_type", STRING_TYPE, immutable=True),
            Column("scope_namespace_id", UUID_TYPE, nullable=True, immutable=True),
            Column("scope_namespace_type", STRING_TYPE, nullable=True, immutable=True),
            Column("scope_customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column("granted_by", UUID_TYPE, nullable=True, immutable=True),
            Column("date_granted", DATETIMETZ_TYPE, immutable=True),
            # ``managed_by`` declares assignment provenance. default
            # ``'manual'`` covers admin-authored rows; agent-side
            # automation passes ``'auto:agent-yaml'``.
            Column("managed_by", STRING_TYPE),
        ],
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"role_assignments"``
        :rtype: str
        """
        return "role_assignments"

    @property
    def entity_class(self) -> type[RoleAssignmentEntity]:
        """return entity class for this collection.

        :return: :class:`RoleAssignmentEntity`
        :rtype: type[RoleAssignmentEntity]
        """
        return RoleAssignmentEntity

    def create(
        self,
        data: dict[str, Any],
    ) -> RoleAssignmentEntity:
        """construct new assignment entity, auto-deriving ``row_scope``.

        role_assignments has no ``customer_id`` column on the row
        itself; the row's effective customer flows from the scope
        triple (``scope_type`` / ``scope_namespace_id`` /
        ``scope_customer_id``). ``row_scope`` stores the discriminator
        explicitly (``platform`` for super_admin ``scope_type='all'``
        rows or ``scope_type='type_customer'`` rows whose
        ``scope_customer_id IS NULL``; ``customer`` for every other
        shape) so the partition primitive can guard the row uniformly.

        :param data: row payload; may include ``row_scope`` directly
            (override leaves it alone) or rely on derivation from the
            scope triple
        :ptype data: dict[str, Any]
        :return: newly constructed assignment entity
        :rtype: RoleAssignmentEntity
        """
        if "row_scope" not in data:
            scope_type = data.get("scope_type")
            scope_customer_id = data.get("scope_customer_id")
            if scope_type == "all":
                row_scope = "platform"
            elif scope_type == "type_customer" and scope_customer_id is None:
                row_scope = "platform"
            else:
                row_scope = "customer"
            data = {**data, "row_scope": row_scope}
        return super().create(data)

    async def find_by_id(
        self,
        assignment_id: UUID,
    ) -> RoleAssignmentEntity | None:
        """resolve assignment by ``id`` alone via the ``UNIQUE (id)`` constraint.

        admin endpoints take ``{assignment_id}`` in the URL but not the
        partition column ``row_scope``. uniqueness across the whole
        table is preserved by the table-level ``UNIQUE (id)``
        constraint.

        :param assignment_id: assignment UUID
        :ptype assignment_id: UUID
        :return: assignment entity or ``None`` when no row exists
        :rtype: RoleAssignmentEntity | None
        """
        result: RoleAssignmentEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM role_assignments WHERE id = $1",
                assignment_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def load_for_groups(
        self,
        group_ids: Sequence[UUID],
    ) -> list[RoleAssignment]:
        """resolve ``group_ids`` to every assignment they hold.

        returns the protocol-shape :class:`RoleAssignment` dataclass
        instances (NOT :class:`RoleAssignmentEntity`) because the
        unified evaluator's :class:`GrantLoader` Protocol speaks in
        the ACL types.

        this method does NOT accept a ``namespace`` filter; it returns
        every assignment every group in the input set holds. callers
        run :meth:`RoleAssignment.covers` themselves to scope the
        result to a specific namespace (the evaluator does this on the
        hot path; the canonical :class:`CollectionGrantLoader` filters
        on its way out so the Protocol contract is preserved).

        empty input short-circuits without a SQL round-trip.

        :param group_ids: sequence of group UUIDs to resolve
        :ptype group_ids: Sequence[UUID]
        :return: list of assignments held by any group in
            ``group_ids``
        :rtype: list[RoleAssignment]
        """
        result: list[RoleAssignment] = []
        if self.l3_pool is not None and len(group_ids) > 0:
            # row_scope spans both 'platform' (scope_type='all' / NULL
            # scope_customer_id) and 'customer' grants; the unified
            # evaluator's hot path needs every assignment regardless
            # of scope.
            rows = await self.l3_pool.fetch(
                """
                SELECT id, role_id, group_id, scope_type,
                       scope_namespace_id, scope_namespace_type,
                       scope_customer_id
                  FROM role_assignments
                 WHERE row_scope IN ('platform', 'customer')
                   AND group_id = ANY($1::uuid[])
                """,
                list(group_ids),
            )
            result = [
                RoleAssignment(
                    id=_coerce_uuid(row["id"]),  # type: ignore[arg-type]
                    role_id=_coerce_uuid(row["role_id"]),  # type: ignore[arg-type]
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    scope_type=ScopeType(row["scope_type"]),
                    scope_namespace_id=_coerce_uuid(row["scope_namespace_id"]),
                    scope_namespace_type=row["scope_namespace_type"],
                    scope_customer_id=_coerce_uuid(row["scope_customer_id"]),
                )
                for row in rows
            ]
        return result

    async def ensure_group_role_assignment(
        self,
        *,
        group_id: UUID,
        role_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
        managed_by: str = "manual",
    ) -> UUID:
        """idempotent insert of ``(group, role, scope)`` assignment row.

        returns the assignment's UUID — either an existing row's id
        when the tuple already exists, or a freshly minted ``uuid7``
        for a newly inserted row.

        the underlying ``role_assignments`` table does NOT carry a
        unique constraint over the lookup tuple (only ``id`` is PK)
        so this method does a SELECT-then-INSERT under a single
        round-trip pattern rather than ``ON CONFLICT (...) DO
        UPDATE``. a concurrent inserter can race; the worst case is
        two physical rows for the same logical grant, which the
        evaluator treats as a no-op duplicate. callers serialize
        admin-path writes themselves so the race is theoretical.

        the ``scope_type`` argument maps to :class:`ScopeType`:

        - ``"namespace"`` — ``scope_id`` is the namespace UUID
        - ``"all"`` — ``scope_id`` must be ``None``
        - ``"type_customer"`` — not supported via this method (raises
          ``ValueError``)

        the ``managed_by`` argument stamps provenance onto freshly-
        inserted rows. when the row already exists the existing
        ``managed_by`` stays untouched so a manual row accidentally
        re-discovered by a translator is never silently re-classed.

        :param group_id: group UUID to bind
        :ptype group_id: UUID
        :param role_id: role UUID to grant
        :ptype role_id: UUID
        :param scope_type: scope discriminator (``"namespace"`` or
            ``"all"``)
        :ptype scope_type: str
        :param scope_id: namespace UUID for ``"namespace"`` scope;
            must be ``None`` for ``"all"`` scope
        :ptype scope_id: UUID | None
        :param managed_by: provenance marker (``"manual"`` |
            ``"auto:agent-yaml"``); applied only on insert
        :ptype managed_by: str
        :return: assignment UUID (existing or newly inserted)
        :rtype: UUID
        :raises ValueError: if ``scope_type`` is unsupported or
            ``scope_id`` shape mismatches the scope
        :raises RuntimeError: if no L3 pool is bound
        """
        if scope_type not in ("namespace", "all"):
            raise ValueError(
                f"unsupported scope_type for idempotent ensure: {scope_type}; use save_entity for type_customer scope",
            )
        if scope_type == "namespace" and scope_id is None:
            raise ValueError(
                "scope_type='namespace' requires a non-None scope_id",
            )
        if scope_type == "all" and scope_id is not None:
            raise ValueError(
                "scope_type='all' requires scope_id=None",
            )
        if self.l3_pool is None:
            raise RuntimeError(
                "RoleAssignmentCollection.ensure_group_role_assignment requires an L3 pool",
            )

        row_scope = "platform" if scope_type == "all" else "customer"
        existing_row = await self.l3_pool.fetchrow(
            """
            SELECT id FROM role_assignments
             WHERE row_scope = $1
               AND group_id = $2
               AND role_id = $3
               AND scope_type = $4
               AND scope_namespace_id IS NOT DISTINCT FROM $5
             ORDER BY id ASC
             LIMIT 1
            """,
            row_scope,
            group_id,
            role_id,
            scope_type,
            scope_id,
        )
        result: UUID
        if existing_row is not None:
            result = existing_row["id"]
        else:
            new_id = uuid7()
            now = datetime.now(UTC).replace(tzinfo=None)
            await self.l3_pool.execute(
                """
                INSERT INTO role_assignments (
                    row_scope, id, role_id, group_id, scope_type,
                    scope_namespace_id, scope_namespace_type,
                    scope_customer_id, granted_by, date_granted,
                    managed_by
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, NULL, NULL, NULL, $7, $8
                )
                """,
                row_scope,
                new_id,
                role_id,
                group_id,
                scope_type,
                scope_id,
                now,
                managed_by,
            )
            result = new_id
        return result

    async def delete_by_group_and_scope(
        self,
        *,
        group_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
        managed_by: str | None = None,
    ) -> int:
        """delete every assignment matching ``(group, scope)`` predicate.

        symmetric counterpart to :meth:`ensure_group_role_assignment`.
        used when revoking a previously-granted scope; returns the
        number of rows the DB confirms it deleted so callers can
        detect a no-op (zero) vs an actual revocation.

        the ``managed_by`` filter restricts the delete to
        provenance-matched rows; ``None`` (the default) means "no
        filter on managed_by" so every row matching the
        ``(group, scope)`` tuple is removed.

        :param group_id: group UUID whose assignments should be
            removed
        :ptype group_id: UUID
        :param scope_type: scope discriminator (``"namespace"`` or
            ``"all"``)
        :ptype scope_type: str
        :param scope_id: namespace UUID for ``"namespace"`` scope;
            must be ``None`` for ``"all"`` scope
        :ptype scope_id: UUID | None
        :param managed_by: optional provenance filter (``"manual"`` |
            ``"auto:agent-yaml"``); ``None`` means no filter
        :ptype managed_by: str | None
        :return: number of rows deleted (zero when nothing matched)
        :rtype: int
        :raises ValueError: if ``scope_type`` / ``scope_id`` shape is
            invalid
        """
        if scope_type not in ("namespace", "all"):
            raise ValueError(
                f"unsupported scope_type for delete_by_group_and_scope: "
                f"{scope_type}; use the per-id delete for "
                "type_customer scope",
            )
        if scope_type == "namespace" and scope_id is None:
            raise ValueError(
                "scope_type='namespace' requires a non-None scope_id",
            )
        if scope_type == "all" and scope_id is not None:
            raise ValueError(
                "scope_type='all' requires scope_id=None",
            )
        row_scope = "platform" if scope_type == "all" else "customer"
        result: int = 0
        if self.l3_pool is not None:
            if managed_by is None:
                status = await self.l3_pool.execute(
                    """
                    DELETE FROM role_assignments
                     WHERE row_scope = $1
                       AND group_id = $2
                       AND scope_type = $3
                       AND scope_namespace_id IS NOT DISTINCT FROM $4
                    """,
                    row_scope,
                    group_id,
                    scope_type,
                    scope_id,
                )
            else:
                status = await self.l3_pool.execute(
                    """
                    DELETE FROM role_assignments
                     WHERE row_scope = $1
                       AND group_id = $2
                       AND scope_type = $3
                       AND scope_namespace_id IS NOT DISTINCT FROM $4
                       AND managed_by = $5
                    """,
                    row_scope,
                    group_id,
                    scope_type,
                    scope_id,
                    managed_by,
                )
            # asyncpg returns "DELETE <count>" status string
            parts = status.split()
            if len(parts) >= 2 and parts[0].upper() == "DELETE":
                result = int(parts[1])
        return result


# ---------------------------------------------------------------------------
# NamespaceCollection
# ---------------------------------------------------------------------------


class NamespaceCollection(SchemaBackedCollection[NamespaceEntity]):
    """three-tier collection for ``namespaces`` rows.

    CRUD comes from the declarative :class:`TableSchema`; canonical
    lookup helpers (``find_by_id`` / ``get_by_name`` /
    ``find_by_type_and_customer`` / ``get_by_owner_and_customer`` /
    ``list_ids_by_customer_and_type`` / ``list_all_ids``) stay here
    because every rbac-consuming app needs to resolve namespace rows
    by these shapes during evaluator hydration / audit-snapshot
    composition. discovery JOINs that span app-specific tables live on
    deploying-app subclasses.
    """

    primary_key_column: tuple[str, ...] = ("row_scope", "id")
    _partition_exempt_methods = frozenset(
        {
            "delete_from_postgres",
            "save_entity",
            "create",
            "find_by_type_and_customer",
            "list_ids_by_customer_and_type",
            "list_all_ids",
            "get_by_name",
            "get_by_agent_id",
            "get_by_owner_and_customer",
            "find_by_id",
        }
    )
    schema = TableSchema(
        name="namespaces",
        primary_key=("row_scope", "id"),
        columns=[
            Column("row_scope", STRING_TYPE, partition=True),
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("namespace_type", STRING_TYPE, immutable=True),
            Column("owner_agent_id", UUID_TYPE, nullable=True, immutable=True),
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column("schema_name", STRING_TYPE, nullable=True, immutable=True),
            Column("metadata", JSONB_TYPE, nullable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_updated", DATETIME_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"namespaces"``
        :rtype: str
        """
        return "namespaces"

    @property
    def entity_class(self) -> type[NamespaceEntity]:
        """return entity class for this collection.

        :return: :class:`NamespaceEntity`
        :rtype: type[NamespaceEntity]
        """
        return NamespaceEntity

    def create(self, data: dict[str, Any]) -> NamespaceEntity:
        """construct new namespace entity, auto-deriving ``row_scope``.

        every namespace row carries ``customer_id`` (nullable for
        platform-scoped rows); ``row_scope`` is the defensive
        discriminator (``platform`` / ``customer``) the partition
        primitive enforces. this override pins ``row_scope`` to the
        value implied by ``customer_id`` so callers continue to pass
        the customer-bearing shape.

        :param data: row payload; may omit ``row_scope`` (override
            sets it) or include it (override leaves explicit values
            untouched)
        :ptype data: dict[str, Any]
        :return: newly constructed (not-yet-persisted) namespace entity
        :rtype: NamespaceEntity
        """
        if "row_scope" not in data:
            customer_id = data.get("customer_id")
            data = {
                **data,
                "row_scope": "platform" if customer_id is None else "customer",
            }
        return super().create(data)

    async def find_by_id(
        self,
        namespace_id: UUID,
    ) -> NamespaceEntity | None:
        """resolve namespace by ``id`` alone via the ``UNIQUE (id)`` constraint.

        callers know the namespace's ``id`` (often computed
        deterministically from the owning agent_id / customer_id) but
        not the partition column ``row_scope``. uniqueness is
        preserved by the ``UNIQUE (id)`` constraint so an id-only
        fetch is unambiguous.

        :param namespace_id: namespace UUID
        :ptype namespace_id: UUID
        :return: namespace entity or ``None`` when no row exists
        :rtype: NamespaceEntity | None
        """
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM namespaces WHERE id = $1",
                namespace_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_name(self, name: str) -> NamespaceEntity | None:
        """look up namespace by unique name.

        searches L3 for namespace matching name. promotes found
        namespace into L1/L2 caches.

        :param name: unique namespace name
        :ptype name: str
        :return: namespace entity or ``None`` if not found
        :rtype: NamespaceEntity | None
        """
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM namespaces WHERE name = $1",
                name,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_agent_id(
        self,
        agent_id: UUID,
        namespace_type: str = "agent",
    ) -> NamespaceEntity | None:
        """look up agent-private namespace by owning agent.

        searches L3 for namespace where ``owner_agent_id`` matches and
        ``namespace_type`` equals the supplied value (default
        ``"agent"`` for the per-agent private namespace shape every
        rbac-consuming app shares). promotes found namespace into
        L1/L2 caches.

        :param agent_id: agent UUID to look up namespace for
        :ptype agent_id: UUID
        :param namespace_type: namespace type discriminator; defaults
            to ``"agent"``
        :ptype namespace_type: str
        :return: namespace entity or ``None`` if not found
        :rtype: NamespaceEntity | None
        """
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            # private agent namespaces always live in the customer
            # partition: every agent belongs to one customer and the
            # namespace inherits that customer's row_scope.
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM namespaces "
                "WHERE row_scope = 'customer' "
                "  AND owner_agent_id = $1 AND namespace_type = $2",
                agent_id,
                namespace_type,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> NamespaceEntity | None:
        """look up namespace by ``(namespace_type, owner_agent_id, customer_id)``.

        natural lookup key for per-agent / per-customer typed
        namespaces (one row per ``(agent, customer)`` pair).

        the underlying ``namespaces`` table does NOT carry a unique
        constraint over the triple (only ``id`` is PK and ``name`` /
        ``schema_name`` are unique). callers create one row per triple
        by convention; this method orders by ``id`` ASC and returns
        the first row to guarantee deterministic resolution if
        duplicates ever land.

        promotes the resolved row into L1/L2 caches.

        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID, or ``None`` for
            agent-agnostic namespaces (e.g. shared rows)
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID, or ``None`` for
            platform-scoped rows
        :ptype customer_id: UUID | None
        :return: namespace entity or ``None`` if no row matches the
            triple
        :rtype: NamespaceEntity | None
        """
        row_scope = "platform" if customer_id is None else "customer"
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                """
                SELECT * FROM namespaces
                 WHERE row_scope = $1
                   AND namespace_type = $2
                   AND owner_agent_id IS NOT DISTINCT FROM $3
                   AND customer_id IS NOT DISTINCT FROM $4
                 ORDER BY id ASC
                 LIMIT 1
                """,
                row_scope,
                namespace_type,
                owner_agent_id,
                customer_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def find_by_type_and_customer(
        self,
        *,
        namespace_type: str,
        customer_id: UUID,
    ) -> list[NamespaceEntity]:
        """return every namespace entity for ``(namespace_type, customer_id)``.

        used where the caller needs the full entity surface (not just
        ids) so it can evaluate per-row authorization against each
        candidate namespace via the unified evaluator and then extract
        the authorized ids from the surviving namespaces.

        rows are NOT promoted into L1/L2 here. typical call shape is a
        scan over a per-customer slice (typically 0-10 rows per
        customer) so the per-row promotion overhead outweighs the L1
        hit ratio. callers that want a single namespace by id chase
        the warmer ``get(id)`` path which promotes naturally.

        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: list of namespace entities matching both filters
        :rtype: list[NamespaceEntity]
        """
        result: list[NamespaceEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM namespaces WHERE row_scope = 'customer'   AND namespace_type = $1 AND customer_id = $2",
                namespace_type,
                customer_id,
            )
            result = [
                self.entity_class(
                    self._coerce_row(dict(row)),
                    is_new=False,
                    collection=self,
                )
                for row in rows
            ]
        return result

    async def list_ids_by_customer_and_type(
        self,
        customer_id: UUID,
        namespace_type: str,
    ) -> list[UUID]:
        """return every namespace id for ``(customer_id, namespace_type)``.

        used by audit-snapshot paths that need to enumerate the
        namespace set a ``type_customer`` scoped assignment covers.
        returning only ids (not full entities) keeps the call cheap.

        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :return: list of namespace UUIDs matching both filters
        :rtype: list[UUID]
        """
        result: list[UUID] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT id FROM namespaces WHERE row_scope = 'customer'   AND customer_id = $1 AND namespace_type = $2",
                customer_id,
                namespace_type,
            )
            result = [row["id"] for row in rows if row["id"] is not None]
        return result

    async def list_all_ids(self) -> list[UUID]:
        """return every namespace id in the table.

        used by audit-snapshot paths for ``scope='all'`` assignments.
        returning ids only mirrors :meth:`list_ids_by_customer_and_type`
        so the caller composes a single id set regardless of scope.

        :return: list of every namespace UUID
        :rtype: list[UUID]
        """
        result: list[UUID] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT id FROM namespaces WHERE row_scope IN ('platform', 'customer')",
            )
            result = [row["id"] for row in rows if row["id"] is not None]
        return result
