"""agent-tools v004: partial-unique index for tool_result dedup.

Enables dedup of repeated identical tool calls by giving ``tool_result``
rows the same upsert-on-key treatment ``variable`` rows already have
(``ix_context_items_var_key``). New tool results are keyed
``tool_name + ':' + hash(input)`` and upserted on
``(conversation_id, key) WHERE context_type = 'tool_result'``, so a
repeat call with the same input refreshes the existing row instead of
appending a duplicate.

Existing tool_result rows were keyed by bare ``tool_name``, so a
conversation that called one tool N times holds N rows sharing a key --
a plain unique index cannot be created over them. This migration is
NON-DESTRUCTIVE: it first suffixes every existing tool_result key with
its own ``context_id`` so all legacy keys become unique (they will not
dedup, but age out via LRU eviction), THEN creates the unique index.
No rows are deleted.

Idempotent: the suffix UPDATE skips rows whose key already ends with
``:<context_id>`` (so a re-run is a no-op), and the index uses
``CREATE UNIQUE INDEX IF NOT EXISTS``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "tool_result_dedup_index",
]

log = get_logger(__name__)


# Suffix legacy tool_result keys with their context_id so they are unique
# before the unique index is built. ``key NOT LIKE '%:<context_id>'``
# excludes already-suffixed rows, making the UPDATE safe to re-run. The
# ``left(..., 255)`` keeps the result inside ``key``'s VARCHAR(255) bound.
_DEDUP_LEGACY_KEYS_SQL = """
UPDATE context_items
SET key = left(key || ':' || context_id::text, 255)
WHERE context_type = 'tool_result'
  AND key NOT LIKE ('%:' || context_id::text)
"""


_CREATE_TOOL_RESULT_KEY_IDX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_context_items_tool_result_key "
    "ON context_items (conversation_id, key) "
    "WHERE context_type = 'tool_result'"
)


async def tool_result_dedup_index(store: DataStore) -> None:
    """Suffix legacy tool_result keys, then add the dedup unique index.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("suffixing legacy tool_result keys for uniqueness (v004)")
    await store.execute(_DEDUP_LEGACY_KEYS_SQL)
    log.info("creating ix_context_items_tool_result_key unique index (v004)")
    await store.execute(_CREATE_TOOL_RESULT_KEY_IDX_SQL)
