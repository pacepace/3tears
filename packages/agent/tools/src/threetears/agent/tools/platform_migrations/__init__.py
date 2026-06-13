"""agent-tools platform-scope migrations.

separate :class:`PackageMigrations` from the agent-scope set in
:mod:`threetears.agent.tools.migrations` because one package
declares one scope. these migrations run once against the
deploying app's platform schema (whatever schema search_path
binds before the runner applies them) and own the
``platform.namespaces`` columns the tool registration path stamps
with each tool's visibility flags.

version history:

- v001 (agent-tools-eligibility shard 01) adds two ``BOOLEAN NOT
  NULL DEFAULT`` columns to the ``namespaces`` table:
  ``tool_eligible`` (defaults TRUE -- existing tool-type rows keep
  their pre-shard "appears in default tool surface" behaviour) and
  ``skill_eligible`` (defaults FALSE -- no row leaks into the
  skills catalog without explicit opt-in). Idempotent via ``ADD
  COLUMN IF NOT EXISTS``.
"""

from __future__ import annotations

from threetears.agent.tools.platform_migrations.v001_add_tool_eligibility_columns import (
    add_tool_eligibility_columns,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_tools_platform"


def register(
    runner: MigrationRunner,
    *,
    depends_on: tuple[str, ...] = (),
) -> PackageMigrations:
    """register the agent-tools platform-scope migration package.

    produces a platform-scoped :class:`PackageMigrations`, attaches every
    migration in version order, and calls ``runner.register``.

    ``depends_on`` lets the DEPLOYING app declare which platform package
    creates the ``namespaces`` table this package ALTERs, so the runner
    orders that package first. it is deployment-specific (different apps
    name their namespaces-owning package differently), so it is supplied
    by the caller rather than hard-coded here. the default ``()`` keeps
    single-runner consumers that create ``namespaces`` in a separate pass
    (or against an already-migrated schema) working unchanged. without
    this edge the runner's alphabetical tie-break would order
    ``agent_tools_platform`` BEFORE a later-sorting namespaces package on
    a fresh schema, and the ``ALTER TABLE`` would run before the table
    exists.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param depends_on: platform package names that must run first (e.g.
        the deploying app's package that creates ``namespaces``)
    :ptype depends_on: tuple[str, ...]
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.PLATFORM,
        depends_on=depends_on,
    )
    pkg.version(1)(add_tool_eligibility_columns)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_tool_eligibility_columns",
    "register",
]
