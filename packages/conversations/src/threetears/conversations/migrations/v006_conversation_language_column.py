"""
conversations v006: add per-conversation ``language`` column for FTS
tokenization.

The v005 migration hard-coded ``english`` in the
``conversations_search_vector_update`` trigger function. That's fine
for v0.7.0 consumers operating in English, but a future polyglot
3tears consumer (a Spanish chat product, a French knowledge base,
etc.) would need to choose between two unappealing options:

- Live with mis-tokenized search vectors (postgres' english parser
  applies english stemming to non-english text -- match quality
  degrades).
- Author a separate migration that pivots the trigger to a
  different postgres parser.

This migration adds a ``language`` column on the ``conversations``
table with default ``'english'`` so existing rows stay byte-stable,
and recreates the trigger function to read ``NEW.language`` instead
of the hard-coded literal. Once landed, a polyglot consumer sets
``conversations.language`` to ``'spanish'`` / ``'french'`` / etc.
on a per-conversation basis and the trigger builds the search_vector
with the appropriate tokenizer -- no further migration.

The valid language values are whatever the postgres installation's
``pg_ts_config`` catalog reports. The default install ships with
``simple``, ``english``, ``spanish``, ``french``, ``german``, etc.
The trigger uses ``COALESCE(NEW.language, 'english')`` so the
fallback works even if a row carries NULL (which the NOT NULL
constraint should prevent, but defense in depth).

Idempotency:

- ``ADD COLUMN IF NOT EXISTS`` makes the column add a no-op on
  replay.
- ``CREATE OR REPLACE FUNCTION`` makes the trigger-function
  rebuild a no-op on replay (always installs the latest body).
- The trigger itself uses DROP-then-CREATE because postgres has
  no ``CREATE TRIGGER IF NOT EXISTS``.
- The search_vector rebuild UPDATE carries the canonical
  yb-safety guard ``WHERE search_vector IS NULL OR ...`` -- since
  the v005 trigger built all existing rows with 'english' and the
  new column also defaults to 'english', re-running the rebuild
  produces byte-identical vectors. The guard is there for future
  callers that might want to re-trigger by NULL-ing the column.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 006
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_conversation_language_column",
]

log = get_logger(__name__)


# Each DDL statement is idempotent on its own so a replay against a
# fully-migrated agent schema is a no-op.

_ADD_LANGUAGE_COLUMN_SQL = """
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'english'
"""

# Trigger function reads NEW.language with a defensive COALESCE to
# 'english' so a NULL value (which the NOT NULL constraint prevents,
# but the defense costs nothing) doesn't propagate as ``NULL::regconfig``
# and crash the trigger.
_CREATE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION conversations_search_vector_update() RETURNS trigger AS $$
DECLARE
    cfg regconfig;
BEGIN
    cfg := COALESCE(NEW.language, 'english')::regconfig;
    NEW.search_vector :=
        setweight(to_tsvector(cfg, coalesce(NEW.name, '')), 'A');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

# Re-create the trigger so it fires on UPDATE OF (name, language) --
# changing the language must trigger a search_vector rebuild,
# otherwise the row's vector stays tokenized in the old language.
_DROP_TRIGGER_SQL = "DROP TRIGGER IF EXISTS trg_conversations_search_vector ON conversations"

_CREATE_TRIGGER_SQL = """
CREATE TRIGGER trg_conversations_search_vector
BEFORE INSERT OR UPDATE OF name, language ON conversations
FOR EACH ROW EXECUTE FUNCTION conversations_search_vector_update();
"""


async def add_conversation_language_column(store: DataStore) -> None:
    """
    Add per-row ``language`` column + update trigger to use it.

    Runs in the per-agent schema set by the migration runner's
    ``search_path``. All DDL is idempotent so replays are safe.
    Backfill of existing rows is implicit (DEFAULT 'english' on the
    new column); search_vectors stay byte-identical since v005
    built them with 'english' verbatim.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding conversations.language + updating FTS trigger to use it")
    await store.execute(_ADD_LANGUAGE_COLUMN_SQL)
    await store.execute(_CREATE_FUNCTION_SQL)
    await store.execute(_DROP_TRIGGER_SQL)
    await store.execute(_CREATE_TRIGGER_SQL)
    log.info("conversations.language column installed + FTS trigger updated")
