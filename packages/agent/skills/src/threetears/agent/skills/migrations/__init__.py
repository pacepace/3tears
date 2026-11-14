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
- v003 drops the two v001 GIN indexes, ``idx_skills_search_vector`` and
  ``idx_skills_tags``. ``list_for_user`` / ``count_for_user`` filter the
  typed query and the tag overlap through
  ``threetears.core.data.gin_filter``, which keeps the planner off the
  index because YugabyteDB's ybgin refuses a multi-entry scan, so neither
  index had a reader and both only cost writes and storage. The
  ``search_vector`` column and its trigger stay. ``DROP INDEX IF EXISTS``,
  so replay is a no-op.
- v004 adds ``tool`` (TEXT) and ``arguments`` (JSONB) to ``agent_skills``
  so a skill can be one tool call run with no model, plus the checks that
  keep a row one kind: never both a body and a tool, arguments only with a
  tool and only as a JSON object, and a payload check that accepts a
  tool-only row.
- v005 adds ``outcome_notes`` (TEXT) to ``agent_skill_invocations``: the
  agent's line on how a use went, kept apart from ``notes``, which holds the
  reason it gave when it invoked the skill.
- v006 (expand) builds ``idx_skill_invocations_conv_key`` (``agent_id, conversation_id, invoked_at DESC,
  invocation_id``): a conversation's page ordered by time completed by the key; v002's
  ``idx_skill_invocations_conv`` stays until the contract step.
- v007 (CONTRACT, held) drops ``idx_skill_invocations_conv``: attached only with
  ``held_contract=True`` (the test schemas now; every deployment from the release after v006's).
"""

from __future__ import annotations

from threetears.agent.skills.migrations.v006_conversation_key_index import (
    add_conversation_key_index,
)
from threetears.agent.skills.migrations.v001_create_agent_skills import (
    create_agent_skills,
)
from threetears.agent.skills.migrations.v002_create_agent_skill_invocations import (
    create_agent_skill_invocations,
)
from threetears.agent.skills.migrations.v003_drop_gin_indexes import (
    drop_gin_indexes,
)
from threetears.agent.skills.migrations.v004_add_tool_call_columns import (
    add_tool_call_columns,
)
from threetears.agent.skills.migrations.v005_add_outcome_notes import (
    add_outcome_notes,
)
from threetears.agent.skills.migrations.v007_contract_replaced_index_dropped import (
    drop_replaced_conversation_index,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_skills"


def register(runner: MigrationRunner, *, held_contract: bool = False) -> PackageMigrations:
    """Register agent-skills migrations with ``runner``.

    Produces an agent-scoped :class:`PackageMigrations` declaring
    ``depends_on=("conversations",)`` so the conversations table
    exists before this package's tables are created. Attaches every
    migration in version order and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param held_contract: attach v007, the held contract step (drops the index v006 replaced)
    :ptype held_contract: bool
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
    pkg.version(3)(drop_gin_indexes)
    pkg.version(4)(add_tool_call_columns)
    pkg.version(5)(add_outcome_notes)
    pkg.version(6)(add_conversation_key_index)
    if held_contract:
        pkg.version(7)(drop_replaced_conversation_index)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_outcome_notes",
    "add_tool_call_columns",
    "create_agent_skill_invocations",
    "create_agent_skills",
    "drop_gin_indexes",
    "register",
]
