"""agent-skills v004: a skill can be one tool call.

Adds two nullable columns to ``agent_skills``:

- ``tool TEXT`` -- a tool's canonical name. A skill with a ``tool`` and no
  ``body`` runs by calling that tool, with no model.
- ``arguments JSONB`` -- the JSON object passed to that tool, fixed when the
  skill is written.

And the checks that keep every row one kind or the other:

- ``agent_skills_body_or_tool_check``: ``NOT (body IS NOT NULL AND tool IS NOT
  NULL)``. A row is never both, so nothing checks the kind at run time.
- ``agent_skills_arguments_need_tool_check``: ``arguments IS NULL OR tool IS
  NOT NULL``.
- ``agent_skills_arguments_object_check``: ``arguments IS NULL OR
  jsonb_typeof(arguments) = 'object'``. A tool takes named arguments.
- ``agent_skills_payload_check`` (from v001) is replaced so a tool-only row
  satisfies it: body, or a tool, or a non-empty ``tool_additions``, or a
  non-empty ``tool_restrictions``.

Every catalog probe is scoped to ``current_schema()``: each agent schema holds
its own ``agent_skills``, and a probe that matched a sibling schema's
constraint would skip this schema's.

Idempotent: ``ADD COLUMN IF NOT EXISTS``; each new check is added only when
absent; the payload check is swapped only when its stored definition differs
from :data:`PAYLOAD_CHECK_ENGINE_DEF`, so a replay changes nothing. No DML, so
no replay guard is needed and no DO block mixes DDL with DML. Existing rows
get ``NULL`` in both columns and already satisfy every check.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "ARGUMENTS_NEED_TOOL_CHECK",
    "ARGUMENTS_OBJECT_CHECK",
    "BODY_OR_TOOL_CHECK",
    "PAYLOAD_CHECK",
    "PAYLOAD_CHECK_ENGINE_DEF",
    "add_tool_call_columns",
]

log = get_logger(__name__)


#: ``(constraint name, expression)`` for the check that refuses a row with both a body and a tool.
BODY_OR_TOOL_CHECK: tuple[str, str] = (
    "agent_skills_body_or_tool_check",
    "NOT (body IS NOT NULL AND tool IS NOT NULL)",
)

#: ``(constraint name, expression)`` for the check that refuses arguments with no tool.
ARGUMENTS_NEED_TOOL_CHECK: tuple[str, str] = (
    "agent_skills_arguments_need_tool_check",
    "arguments IS NULL OR tool IS NOT NULL",
)

#: ``(constraint name, expression)`` for the check that refuses arguments that are not a JSON object.
ARGUMENTS_OBJECT_CHECK: tuple[str, str] = (
    "agent_skills_arguments_object_check",
    "arguments IS NULL OR jsonb_typeof(arguments) = 'object'",
)

#: ``(constraint name, expression)`` for the at-least-one-payload check, now accepting a tool.
PAYLOAD_CHECK: tuple[str, str] = (
    "agent_skills_payload_check",
    "body IS NOT NULL "
    "OR tool IS NOT NULL "
    "OR array_length(tool_additions, 1) IS NOT NULL "
    "OR array_length(tool_restrictions, 1) IS NOT NULL",
)

#: :data:`PAYLOAD_CHECK` as Postgres stores it (``pg_get_constraintdef``). The swap compares against
#: this; a mismatch only costs a redundant drop and re-add, never a wrong constraint.
PAYLOAD_CHECK_ENGINE_DEF = (
    "CHECK (((body IS NOT NULL) OR (tool IS NOT NULL) "
    "OR (array_length(tool_additions, 1) IS NOT NULL) "
    "OR (array_length(tool_restrictions, 1) IS NOT NULL)))"
)


_ADD_TOOL_COLUMN_SQL = "ALTER TABLE agent_skills ADD COLUMN IF NOT EXISTS tool TEXT"

_ADD_ARGUMENTS_COLUMN_SQL = "ALTER TABLE agent_skills ADD COLUMN IF NOT EXISTS arguments JSONB"


def _add_check_if_absent_sql(name: str, expression: str) -> str:
    """Build a DO block adding one CHECK to ``agent_skills`` unless this schema already has it.

    :param name: constraint name
    :ptype name: str
    :param expression: boolean SQL expression, without ``CHECK``
    :ptype expression: str
    :return: the DO block
    :rtype: str
    """
    return f"""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint pc
          JOIN pg_class cls ON cls.oid = pc.conrelid
          JOIN pg_namespace ns ON ns.oid = cls.relnamespace
         WHERE ns.nspname = current_schema()
           AND cls.relname = 'agent_skills'
           AND pc.conname = '{name}'
    ) THEN
        ALTER TABLE agent_skills ADD CONSTRAINT {name} CHECK ({expression});
    END IF;
END
$$
"""


def _replace_payload_check_sql() -> str:
    """Build the DO block swapping ``agent_skills_payload_check`` for the tool-aware form.

    :return: the DO block
    :rtype: str
    """
    name, expression = PAYLOAD_CHECK
    target = PAYLOAD_CHECK_ENGINE_DEF.replace("'", "''")
    return f"""
DO $$
DECLARE
    current_def TEXT;
BEGIN
    SELECT pg_get_constraintdef(pc.oid)
      INTO current_def
      FROM pg_constraint pc
      JOIN pg_class cls ON cls.oid = pc.conrelid
      JOIN pg_namespace ns ON ns.oid = cls.relnamespace
     WHERE ns.nspname = current_schema()
       AND cls.relname = 'agent_skills'
       AND pc.conname = '{name}';

    IF current_def IS NOT NULL AND current_def = '{target}' THEN
        RETURN;
    END IF;

    ALTER TABLE agent_skills DROP CONSTRAINT IF EXISTS {name};
    ALTER TABLE agent_skills ADD CONSTRAINT {name} CHECK ({expression});
END
$$
"""


async def add_tool_call_columns(store: DataStore) -> None:
    """Add ``tool`` / ``arguments`` to ``agent_skills`` and the checks that keep a row one kind.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding agent_skills.tool + agent_skills.arguments and their checks (v004)")
    await store.execute(_ADD_TOOL_COLUMN_SQL)
    await store.execute(_ADD_ARGUMENTS_COLUMN_SQL)
    for name, expression in (BODY_OR_TOOL_CHECK, ARGUMENTS_NEED_TOOL_CHECK, ARGUMENTS_OBJECT_CHECK):
        await store.execute(_add_check_if_absent_sql(name, expression))
    await store.execute(_replace_payload_check_sql())
