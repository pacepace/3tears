"""3tears-mcp package migrations.

single entry point :func:`register` wires the package's versioned
migration callable into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. the mcp package owns one
platform-scope table -- ``mcp_tool_grants`` -- shared by every
consumer that hosts an MCP server with per-tool RBAC.

PLATFORM scope: the table lives in the consumer application's
platform schema (whatever schema the caller binds via search_path
before calling :meth:`MigrationRunner.apply_for_platform_schema`).

version history:

- v001 -- create the ``mcp_tool_grants`` table with surrogate UUID
  PK plus principal-and-tool lookup indexes.
- v002 (CONTRACT, held) -- drop both lookup indexes (no read uses them):
  attached only with ``held_contract=True`` (the test schemas now; every
  deployment from the release after this one).
"""

from __future__ import annotations

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)
from threetears.mcp.migrations.v001_create_mcp_tool_grants import (
    create_mcp_tool_grants_table,
)
from threetears.mcp.migrations.v002_contract_lookup_indexes_dropped import (
    drop_lookup_indexes,
)
from threetears.mcp.rbac import McpToolGrantCollection

PACKAGE_NAME = "mcp"


def register(runner: MigrationRunner, *, held_contract: bool = False) -> PackageMigrations:
    """register mcp migrations with the given runner.

    produces a platform-scoped :class:`PackageMigrations` with no
    ``depends_on`` edges, attaches every migration in version order,
    and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param held_contract: attach v002, the held contract step (drops the two unread lookup indexes)
    :ptype held_contract: bool
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.PLATFORM,
    )
    pkg.version(1)(create_mcp_tool_grants_table)
    if held_contract:
        pkg.version(2)(drop_lookup_indexes)
    # the grants are followed by write generation: the run that creates the table, and every run
    # after, gives it its write-mark and change-note triggers
    pkg.mark_write_generation(McpToolGrantCollection.schema)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_mcp_tool_grants_table",
    "register",
]
