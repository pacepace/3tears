"""agent-skills v001: create ``agent_skills`` table + FTS trigger.

agent-skills shard 01. partition column is ``agent_id``; composite
primary key is ``(agent_id, skill_id)``. ``UNIQUE (skill_id)`` lets
cross-package FKs reference the bare column without partition
knowledge -- specifically the wake-side
``agent_wake_schedules.skill_id`` declared in a later shard.

CHECK constraints:

- ``prompt_mode IN ('additive', 'replace')`` -- enum-by-app.
- at-least-one-payload across ``body`` / ``tool_additions`` /
  ``tool_restrictions`` -- enforces the "skill must do *something*"
  invariant from PLACEMENT §1.1.

FTS: trigger-maintained ``search_vector`` (weighted A/B/C across
``name`` / ``trigger_keywords`` / ``body``) drives the optional
query-filter ranking in ``skill_list``. NOT used for auto-load (that
path was dropped per the planning set's redesign).

every statement is idempotent so re-running this migration on a
schema that already has the table is a no-op:

- ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` /
  ``CREATE UNIQUE INDEX IF NOT EXISTS`` for table + indexes.
- ``CREATE OR REPLACE FUNCTION`` for the trigger function.
- ``DROP TRIGGER IF EXISTS ... ; CREATE TRIGGER ...`` for the trigger
  (Postgres has no ``CREATE TRIGGER IF NOT EXISTS``).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_agent_skills",
]

log = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_skills (
    agent_id          UUID         NOT NULL,
    skill_id          UUID         NOT NULL,
    user_id           UUID         NOT NULL,
    name              TEXT         NOT NULL,
    summary           TEXT         NOT NULL,
    body              TEXT         NULL,
    prompt_mode       TEXT         NOT NULL DEFAULT 'additive',
    tool_additions    TEXT[]       NOT NULL DEFAULT '{}',
    tool_restrictions TEXT[]       NOT NULL DEFAULT '{}',
    trigger_keywords  TEXT         NOT NULL DEFAULT '',
    tags              TEXT[]       NOT NULL DEFAULT '{}',
    source            TEXT         NOT NULL DEFAULT 'manual',
    enabled           BOOLEAN      NOT NULL DEFAULT true,
    use_count         INTEGER      NOT NULL DEFAULT 0,
    last_used_at      TIMESTAMPTZ,
    success_count     INTEGER      NOT NULL DEFAULT 0,
    failure_count     INTEGER      NOT NULL DEFAULT 0,
    last_failure_at   TIMESTAMPTZ,
    date_created      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    date_updated      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    search_vector     TSVECTOR,
    PRIMARY KEY (agent_id, skill_id),
    UNIQUE (skill_id),
    CONSTRAINT agent_skills_prompt_mode_check
        CHECK (prompt_mode IN ('additive', 'replace')),
    CONSTRAINT agent_skills_payload_check CHECK (
        body IS NOT NULL
        OR array_length(tool_additions, 1) IS NOT NULL
        OR array_length(tool_restrictions, 1) IS NOT NULL
    )
)
"""

_CREATE_UQ_NAME_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_skills_agent_user_name ON agent_skills (agent_id, user_id, name)"
)

_CREATE_AGENT_USER_ENABLED_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_skills_agent_user_enabled ON agent_skills (agent_id, user_id, enabled)"
)

_CREATE_SEARCH_VECTOR_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_skills_search_vector ON agent_skills USING GIN (search_vector)"
)

_CREATE_TAGS_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_skills_tags ON agent_skills USING GIN (tags)"


_CREATE_FTS_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION agent_skills_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
  NEW.search_vector :=
      setweight(to_tsvector('english', coalesce(NEW.name, '')), 'A') ||
      setweight(to_tsvector('english', coalesce(NEW.trigger_keywords, '')), 'B') ||
      setweight(to_tsvector('english', coalesce(NEW.body, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""

_DROP_FTS_TRIGGER_SQL = "DROP TRIGGER IF EXISTS trg_agent_skills_search_vector ON agent_skills"

_CREATE_FTS_TRIGGER_SQL = """
CREATE TRIGGER trg_agent_skills_search_vector
  BEFORE INSERT OR UPDATE OF name, trigger_keywords, body ON agent_skills
  FOR EACH ROW EXECUTE FUNCTION agent_skills_search_vector_update()
"""


async def create_agent_skills(store: DataStore) -> None:
    """Create ``agent_skills`` table, indexes, and FTS trigger.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating agent_skills table + FTS trigger (v001)")
    await store.execute(_CREATE_TABLE_SQL)
    await store.execute(_CREATE_UQ_NAME_INDEX_SQL)
    await store.execute(_CREATE_AGENT_USER_ENABLED_INDEX_SQL)
    await store.execute(_CREATE_SEARCH_VECTOR_INDEX_SQL)
    await store.execute(_CREATE_TAGS_INDEX_SQL)
    await store.execute(_CREATE_FTS_FUNCTION_SQL)
    await store.execute(_DROP_FTS_TRIGGER_SQL)
    await store.execute(_CREATE_FTS_TRIGGER_SQL)
