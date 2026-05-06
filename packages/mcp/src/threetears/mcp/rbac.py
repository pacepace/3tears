"""``mcp_tool_grants`` collection -- per-tool RBAC grants for MCP servers.

shape one row per ``(principal, tool, permission)`` triple. principals
are users / groups / roles; permissions are platform-format strings
(``"metallm.conversations.read"``, ``"hub.audit.read"``, etc.). default-
deny: a tool's :attr:`McpTool.required_permission` runs only when an
active grant matches the calling identity.

mutation discipline:

- :meth:`add_grant` writes the row, then the **caller** is responsible
  for bumping :func:`Subjects.mcp_rbac_epoch` (this collection does not
  hold an :class:`EpochClient` reference -- the bump call site is the
  REST endpoint that mutated the row, so the bump lives there for
  symmetry with task-02 Chunks B and C).
- :meth:`remove_grant` same shape.
- :class:`~threetears.mcp.auth.LocalGrantAuthorizer` subscribes to the
  rbac epoch via :class:`~threetears.epoch.EpochListener`; on bump the
  authorizer reloads its in-memory cache from this collection.

cross-pod coherence is therefore: write -> bump -> broadcast -> sibling
authorizers reload. missed broadcasts recover via the periodic catch-up
tick the authorizer wires in :meth:`LocalGrantAuthorizer.start`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column as SAColumn
from sqlalchemy import DateTime, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.entities.base import BaseEntity
from threetears.observe import get_logger

__all__ = [
    "McpToolGrantCollection",
    "McpToolGrantEntity",
    "mcp_tool_grants_table",
]

log = get_logger(__name__)


def mcp_tool_grants_table(metadata: MetaData) -> Table:
    """register the ``mcp_tool_grants`` table on the given SA metadata.

    call this before ``SQLiteBackend.initialize(metadata)`` so the L1
    cache gets the correct schema. safe to call multiple times --
    returns the existing table if already registered.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``mcp_tool_grants`` :class:`Table`
    :rtype: Table
    """
    if "mcp_tool_grants" in metadata.tables:
        return metadata.tables["mcp_tool_grants"]
    return Table(
        "mcp_tool_grants",
        metadata,
        SAColumn("grant_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("principal_type", Text(), nullable=False),
        SAColumn("principal_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn("tool_name", Text(), nullable=False),
        SAColumn("permission", Text(), nullable=False),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
    )


class McpToolGrantEntity(BaseEntity):
    """Entity representing one MCP tool grant row.

    every grant is one ``(principal_type, principal_id, tool_name,
    permission)`` quadruple. ``grant_id`` is the surrogate PK and the
    addressable identity for revocation.

    field access flows through :class:`BaseEntity.__getattr__`.
    """

    primary_key_field: str = "grant_id"


class McpToolGrantCollection(SchemaBackedCollection[McpToolGrantEntity]):
    """three-tier collection for ``mcp_tool_grants``.

    CRUD comes from :class:`SchemaBackedCollection`; the domain
    methods are :meth:`add_grant`, :meth:`remove_grant`, and
    :meth:`load_all_grants` (the loader the authorizer cache calls).

    instances do NOT publish epoch bumps themselves -- callers (REST
    admin endpoints) own the bump-after-commit pattern so the row
    write and the broadcast happen in the same scope. see task-02
    Chunks B + C for the same discipline.
    """

    primary_key_column: str | tuple[str, ...] = "grant_id"
    schema = TableSchema(
        name="mcp_tool_grants",
        primary_key=("grant_id",),
        columns=[
            Column("grant_id", UUID_TYPE),
            Column("principal_type", STRING_TYPE, immutable=True),
            Column("principal_id", UUID_TYPE, immutable=True),
            Column("tool_name", STRING_TYPE, immutable=True),
            Column("permission", STRING_TYPE, immutable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
    )

    @property
    def table_name(self) -> str:
        """return the collection's table name.

        :return: ``"mcp_tool_grants"``
        :rtype: str
        """
        return "mcp_tool_grants"

    @property
    def entity_class(self) -> type[McpToolGrantEntity]:
        """return the entity class for this collection.

        :return: :class:`McpToolGrantEntity`
        :rtype: type[McpToolGrantEntity]
        """
        return McpToolGrantEntity

    async def add_grant(
        self,
        *,
        principal_type: str,
        principal_id: UUID,
        tool_name: str,
        permission: str,
    ) -> McpToolGrantEntity:
        """insert a new grant row and return the created entity.

        the caller is responsible for bumping the rbac epoch after
        the row commits (see module docstring for rationale).

        :param principal_type: ``"user"`` / ``"group"`` / ``"role"``
        :ptype principal_type: str
        :param principal_id: principal UUID
        :ptype principal_id: UUID
        :param tool_name: target tool name (matches :attr:`McpTool.name`)
        :ptype tool_name: str
        :param permission: permission string the grant authorizes
        :ptype permission: str
        :return: created grant entity
        :rtype: McpToolGrantEntity
        """
        grant_id = uuid4()
        entity = self.entity_class(
            {
                "grant_id": grant_id,
                "principal_type": principal_type,
                "principal_id": principal_id,
                "tool_name": tool_name,
                "permission": permission,
                "date_created": datetime.now(UTC),
            },
            is_new=True,
            collection=self,
        )
        await self.save_entity(entity)
        return entity

    async def remove_grant(self, grant_id: UUID) -> bool:
        """delete a grant by id.

        the caller is responsible for bumping the rbac epoch after
        the row commits. uses the public :meth:`BaseCollection.get`
        + :meth:`BaseCollection.delete` extension seams (the
        ``find_by_id`` / ``delete_entity`` shorthand the original
        prototype reached for does not exist on the canonical
        BaseCollection surface; this method shapes around the real
        contract).

        :param grant_id: target grant UUID
        :ptype grant_id: UUID
        :return: True when the row existed and was removed; False when
            no row matched
        :rtype: bool
        """
        existing = await self.get(grant_id)
        if existing is None:
            return False
        return await self.delete(grant_id)

    async def load_all_grants(self) -> list[dict[str, Any]]:
        """return every grant row as a dict suitable for cache rebuild.

        the framework's :class:`~threetears.mcp.auth.LocalGrantAuthorizer`
        calls this on cold start and on every ``mcp.rbac`` epoch bump.
        rows go straight from L3 (no caching here -- the authorizer
        owns the in-memory cache).

        uses the public :attr:`l3_pool` extension seam (matches every
        other peer collection) and fails closed when the registry
        has not been configured.

        :return: list of dict rows with keys
            ``(grant_id, principal_type, principal_id, tool_name,
            permission, date_created)``
        :rtype: list[dict[str, Any]]
        :raises RuntimeError: when the L3 pool is not configured
            on this collection's registry
        """
        if self.l3_pool is None:
            raise RuntimeError(
                "McpToolGrantCollection L3 pool is not configured; "
                "wire CollectionRegistry.configure(l3_pool=...) before "
                "calling load_all_grants",
            )
        rows = await self.l3_pool.fetch(
            "SELECT grant_id, principal_type, principal_id, "
            "tool_name, permission, date_created "
            "FROM mcp_tool_grants",
        )
        return [dict(row) for row in rows]
