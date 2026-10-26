"""
agent-identity v003: block variants.

Adds ``variant`` to ``identity_versions``: a consumer-named copy of a block
with its own version chain and its own active version. ``''`` is the block
itself, so every existing row keeps its meaning and every caller that never
names a variant reads and writes exactly what it did before.

The one-active rule and the history index move from ``(agent, customer,
user, block_key)`` to ``(..., block_key, variant)``. Each new index is
created under a new name before the old one is dropped, so a replay finds
the new index present and the old one gone, and no moment of the
migration leaves a block without its one-active guard. The old unique index
must go: it would refuse a variant's active version beside the block's.

The runner serialises pods with an advisory lock and runs each statement on
its own (see v002), so the column add commits before the index that names it.
Idempotent throughout (``IF NOT EXISTS`` / ``IF EXISTS``).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_block_variants",
]

log = get_logger(__name__)

_ADD_VARIANT_SQL = "ALTER TABLE identity_versions ADD COLUMN IF NOT EXISTS variant TEXT NOT NULL DEFAULT ''"

_CREATE_ACTIVE_UNIQUE_IDX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_identity_active_per_variant "
    "ON identity_versions (agent_id, customer_id, user_id, block_key, variant) "
    "WHERE status = 'active'"
)

_CREATE_HISTORY_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_identity_variant_history "
    "ON identity_versions (agent_id, user_id, block_key, variant, date_created DESC)"
)

_DROP_OLD_ACTIVE_UNIQUE_IDX_SQL = "DROP INDEX IF EXISTS uq_identity_active_per_block"

_DROP_OLD_HISTORY_IDX_SQL = "DROP INDEX IF EXISTS idx_identity_block_history"


async def add_block_variants(store: DataStore) -> None:
    """add the variant column and move the block indexes onto it.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding identity block variants (v003)")
    await store.execute(_ADD_VARIANT_SQL)
    await store.execute(_CREATE_ACTIVE_UNIQUE_IDX_SQL)
    await store.execute(_CREATE_HISTORY_IDX_SQL)
    await store.execute(_DROP_OLD_ACTIVE_UNIQUE_IDX_SQL)
    await store.execute(_DROP_OLD_HISTORY_IDX_SQL)
