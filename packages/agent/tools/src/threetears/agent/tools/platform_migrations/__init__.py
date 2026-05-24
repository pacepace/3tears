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


def register(runner: MigrationRunner) -> PackageMigrations:
    """register the agent-tools platform-scope migration package.

    produces a platform-scoped :class:`PackageMigrations` with no
    ``depends_on`` edges (the ``namespaces`` table is platform-managed
    by the deploying app and assumed to exist by the time this
    migration runs), attaches every migration in version order, and
    calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.PLATFORM,
    )
    pkg.version(1)(add_tool_eligibility_columns)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_tool_eligibility_columns",
    "register",
]
