"""agent-skills package migrations.

single entry point :func:`register` wires the package's versioned
migration callables into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`.

This package owns two tables in every agent schema:

- ``agent_skills`` -- one row per skill (partition column ``agent_id``).
- ``agent_skill_invocations`` -- one row per skill load (partition
  column ``agent_id``; composite FK ``(agent_id, skill_id) REFERENCES
  agent_skills`` with ``ON DELETE CASCADE``).

The package declares ``depends_on=("conversations",)`` because the
invocation rows carry ``conversation_id``. No FK constraint exists on
``conversation_id`` -- the constraint would force a cross-package
teardown order; declaring the dependency on apply is enough.

Version history:

- v001 creates ``agent_skills`` + indexes + FTS trigger.
- v002 creates ``agent_skill_invocations`` + indexes.
"""

from __future__ import annotations

from threetears.agent.skills.migrations.v001_create_agent_skills import (
    create_agent_skills,
)
from threetears.agent.skills.migrations.v002_create_agent_skill_invocations import (
    create_agent_skill_invocations,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_skills"


def register(runner: MigrationRunner) -> PackageMigrations:
    """Register agent-skills migrations with ``runner``.

    Produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)`` so the conversations table
    exists before this package's tables are created. Attaches every
    migration in version order and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("conversations",),
    )
    pkg.version(1)(create_agent_skills)
    pkg.version(2)(create_agent_skill_invocations)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_agent_skill_invocations",
    "create_agent_skills",
    "register",
]
