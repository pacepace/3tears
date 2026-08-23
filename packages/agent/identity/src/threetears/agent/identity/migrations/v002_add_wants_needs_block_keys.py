"""
agent-identity v002: add ``wants`` / ``needs`` to ``identity_block_key``.

Extends the block-key enum with the agent's standing appetite
(``wants``) and its regulatory conditions (``needs``) so both become
proposable, consentable identity blocks like the original five. Both are
tier 2 (auto-apply, human vetoes after the fact) -- see
``IDENTITY_BLOCK_TIERS`` for the reasoning behind that placement.

v001 shipped in 3tears v0.26.1, so its ``CREATE TYPE`` is a historical
record: a schema that already recorded v001 applied will never re-run it,
and editing that DDL in place would land the new members on fresh
installs only. Hence a forward migration.

``ALTER TYPE ... ADD VALUE`` is the one DDL statement that cannot run
inside a transaction block on PostgreSQL before 12, and even on 12+ the
added label is unusable until the adding transaction commits. Both
constraints are satisfied here for the same reason: the migration runner
deliberately does NOT wrap a version in a transaction (DDL auto-commits
under YugabyteDB, so an advisory lock serialises pods instead of a
transaction -- see ``migrations/runner.py``). Each statement therefore
self-commits, and the labels are usable by the time any later migration
or query references them.

``IF NOT EXISTS`` makes the replay a no-op, so re-running is safe. It
also keeps this correct if v001's enum list is ever regenerated for fresh
installs: the adds simply find the labels already present.

Appending rather than inserting: new labels go on the end of the enum's
sort order, matching declaration order in :class:`IdentityBlockKey`. No
``BEFORE``/``AFTER`` placement is used, because nothing orders identity
blocks by their enum position -- ordering is by ``date_created`` in the
history index and by the layout's own block sequence when rendering.
"""

from __future__ import annotations

from threetears.agent.identity.types import IdentityBlockKey
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_wants_needs_block_keys",
]

log = get_logger(__name__)

# Derived from the enum rather than spelled out, so the DDL cannot drift
# from the Python members the schema Column and the tools validate against.
# Scoped to the members v002 introduces -- a later version adding an
# eighth block gets its own migration, and widening this tuple would
# silently make THIS migration responsible for it while every schema that
# already recorded v002 skips it.
_V002_BLOCK_KEYS: tuple[str, ...] = (
    IdentityBlockKey.WANTS.value,
    IdentityBlockKey.NEEDS.value,
)

# current_schema()-scoped like v001's create: an agent schema's enum is
# its own, and a sibling schema's copy must not mask this one.
_ADD_BLOCK_KEY_SQL = "ALTER TYPE identity_block_key ADD VALUE IF NOT EXISTS '{value}'"


async def add_wants_needs_block_keys(store: DataStore) -> None:
    """add the wants / needs labels to the identity_block_key enum.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding wants / needs identity block keys (v002)")
    for value in _V002_BLOCK_KEYS:
        # Interpolated, not bound: ALTER TYPE takes a literal label, and a
        # bound parameter is a value expression PostgreSQL rejects here.
        # Safe because the values come from the enum, never from input.
        await store.execute(_ADD_BLOCK_KEY_SQL.format(value=value))
