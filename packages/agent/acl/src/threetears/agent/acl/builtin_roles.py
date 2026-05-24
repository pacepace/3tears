"""bootstrap helpers for 3tears-shipped builtin roles.

agent-tools-eligibility shard 01 (TE-12) ships one builtin role
that grants the ``tool.call`` action on the three pre-check tools
the ``3tears-agent-wake`` foundation depends on
(``http_get``, ``loki_query``, ``postgres_query``). The role lives
in the platform-managed ``roles`` table and is named
:data:`PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME` so the deploying app
can bind it to its canonical "every user belongs here" group
(``platform-users`` in the metallm hub) via the normal
``role_assignments`` machinery.

The role definition is intentionally minimal: it grants
``tool.call`` on a ``tool``-type namespace. The actual
per-tool assignment rows
(``scope_type='namespace'``, ``scope_namespace_id=<tool-ns-uuid>``)
are created by the deploying app at tool-registration time -- the
3tears side does not write into ``role_assignments`` because the
target namespace ids are only known once the tool's
``platform.namespaces`` row exists, and that row is materialized
hub-side (see
:class:`aibots.hub.tools.namespace_emitter.ToolNamespaceEmitter`).

This module exposes:

- :data:`PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME` -- canonical role
  name; deploying apps reference this constant when looking the
  role up (do NOT hard-code the string).
- :data:`PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS` -- the
  permission map persisted into ``roles.permissions`` as JSONB
  (``{"tool": ["call"]}``); kept here so the deploying app's seed
  path and 3tears' own tests reference the same source of truth.
- :data:`PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES` -- the three
  pre-check tool ``mcp_name`` values the role's grants must cover
  once each tool registers.
- :func:`ensure_platform_builtin_tool_user_role` -- idempotent
  helper that inserts the role row if it isn't already present and
  returns its UUID. callers run this once per platform bootstrap.

Per-customer overrides (revoke for one customer's users) are out
of scope here -- they happen at the assignment layer via the
normal admin grant API. The role itself is platform-wide.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from threetears.observe import get_logger

__all__ = [
    "PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_DESCRIPTION",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS",
    "ensure_platform_builtin_tool_user_role",
]

log = get_logger(__name__)


PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME: str = "PlatformBuiltinToolUser"
"""canonical name of the role that grants ``tool.call`` on the
3tears-shipped pre-check tools (``http_get``, ``loki_query``,
``postgres_query``). The deploying app binds this role to its
"every user belongs here" group so every user can include the
pre-check tools in a skill's ``tool_additions`` list without an
admin grant for each tool."""


PLATFORM_BUILTIN_TOOL_USER_ROLE_DESCRIPTION: str = (
    "Default access to the 3tears-shipped pre-check tools "
    "(http_get, loki_query, postgres_query). Bound to the "
    "platform-users group by deploying-app bootstrap; revocable "
    "per-customer at the assignment layer."
)
"""human-readable description persisted into ``roles.description``;
written once at insert time and never updated by the helper so a
deploying app may rewrite it without the next bootstrap reverting
the edit."""


PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS: dict[str, list[str]] = {
    "tool": ["call"],
}
"""permission map persisted into ``roles.permissions``. Single
``tool.call`` action because pre-check tools only need invocation
(not introspection / mutation of the namespace itself). Kept as a
plain ``dict[str, list[str]]`` so the JSONB encode/decode path is
symmetrical with every other role row in the table."""


PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES: tuple[str, ...] = (
    "http_get",
    "loki_query",
    "postgres_query",
)
"""``mcp_name`` values of the three pre-check tools the role's
assignments must cover once each tool registers. The 3tears side
does not seed assignment rows -- the deploying app does, once it
knows the per-version namespace ids -- but the list is published
here so the deploying app's seed code and the shard's
documentation reference the same source of truth."""


async def ensure_platform_builtin_tool_user_role(
    role_collection: Any,
) -> UUID:
    """idempotently ensure the ``PlatformBuiltinToolUser`` row exists.

    looks up the role by canonical name; returns the existing row's
    id when present, otherwise inserts a new row with the canonical
    name + description + permissions + ``is_builtin=TRUE`` and
    returns the freshly-minted UUID. concurrent bootstrap callers
    can race; the worst case is two physical rows for the same
    logical role (the table's only unique constraint is on
    ``role_id``), which the deploying app guards against by
    serializing platform bootstrap.

    :param role_collection: instance of :class:`RoleCollection`
        (typed ``Any`` to keep this module free of an import on the
        collections module, which would close a small import cycle
        through the entity stack)
    :ptype role_collection: Any
    :return: UUID of the existing-or-newly-inserted role row
    :rtype: UUID
    :raises RuntimeError: if the collection has no L3 pool bound
        (every production bootstrap path has one; this is a fail-
        fast guard for tests that wire the collection without a
        pool)
    """
    pool = getattr(role_collection, "l3_pool", None)
    if pool is None:
        raise RuntimeError(
            "ensure_platform_builtin_tool_user_role: role_collection "
            "has no l3_pool bound; cannot resolve or create the role",
        )
    existing = await pool.fetchrow(
        "SELECT role_id FROM roles WHERE name = $1",
        PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME,
    )
    if existing is not None:
        result: UUID = existing["role_id"] if isinstance(existing["role_id"], UUID) else UUID(str(existing["role_id"]))
        log.debug(
            "platform builtin tool-user role already present",
            extra={
                "extra_data": {
                    "role_id": str(result),
                    "role_name": PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME,
                }
            },
        )
        return result
    new_id = uuid7()
    now = datetime.now(UTC)
    # JSONB encoding is the pool's responsibility (asyncpg or the
    # NATS-proxy L3 backend register a JSONB codec); passing the
    # dict directly matches the pattern used by every other 3tears
    # caller that writes a roles-shaped permissions payload.
    import json as _json  # noqa: PLC0415

    await pool.execute(
        "INSERT INTO roles ("
        "role_id, name, description, permissions, is_builtin, "
        "date_created, date_updated"
        ") VALUES ($1, $2, $3, $4::jsonb, TRUE, $5, $5)",
        new_id,
        PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME,
        PLATFORM_BUILTIN_TOOL_USER_ROLE_DESCRIPTION,
        _json.dumps(PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS),
        now,
    )
    log.info(
        "platform builtin tool-user role inserted",
        extra={
            "extra_data": {
                "role_id": str(new_id),
                "role_name": PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME,
                "covered_tools": list(PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES),
            }
        },
    )
    return new_id
