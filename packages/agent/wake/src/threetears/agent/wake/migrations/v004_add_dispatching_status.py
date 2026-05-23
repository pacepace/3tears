"""agent-wake v004: add ``'dispatching'`` placeholder to ``wake_fires.status``.

shard-02's :meth:`WakeFireCollection.create_dispatching` inserts the
initial in-flight row before the dispatch callback runs. In v002 that
row landed with ``status='fired'`` -- the same terminal value the
context_from resolver keys off via ``status IN {'fired',
'fired_silent'}``.

Today this is safe because the tick body is sequential inside the
cross-pod lock + the per-conv lock spans claim through finalize, so
the placeholder row is overwritten before any observer keys off it.
But the safety is structural, not invariant: if a future refactor
parallelises per-schedule dispatch inside the tick body
(PLACEMENT.md §1.3 calls this out as an anti-pattern for exactly this
reason), a downstream wake with ``context_from = upstream`` could fetch
the placeholder mid-tick and materialise an empty context block from
``output_text=NULL`` -- silent context drop, no error.

The clean fix is a distinct placeholder status so a half-completed row
is queryable as in-flight rather than terminal. This migration extends
the ``wake_fires_status_check`` CHECK constraint to accept
``'dispatching'``; :meth:`create_dispatching` is updated in the same
commit to write that value; :meth:`finalize_success` /
:meth:`finalize_failed` overwrite to the real terminal status as
before. ``context_from`` already filters on ``status IN {'fired',
'fired_silent'}`` so the placeholder is invisible to that path by
construction.

Idempotent via ``DROP CONSTRAINT IF EXISTS`` + ``ADD CONSTRAINT``.
PostgreSQL has no ``ALTER CONSTRAINT`` for CHECK predicates, so the
drop-then-add pair is the canonical pattern.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_dispatching_status",
]

log = get_logger(__name__)


# Drop the existing constraint (idempotent via IF EXISTS) and re-add
# it with the 'dispatching' value included. PostgreSQL has no
# ALTER CONSTRAINT for CHECK predicates -- drop + add is the canonical
# pattern. The constraint name matches v002's declaration.
_DROP_OLD_CHECK_SQL = "ALTER TABLE wake_fires DROP CONSTRAINT IF EXISTS wake_fires_status_check"


_ADD_NEW_CHECK_SQL = """
ALTER TABLE wake_fires
    ADD CONSTRAINT wake_fires_status_check
        CHECK (status IN (
            'dispatching',
            'fired',
            'fired_silent',
            'yielded',
            'skipped_busy',
            'skipped_rate_limit',
            'skipped_cap',
            'skipped_no_handler',
            'failed'
        ))
"""


async def add_dispatching_status(store: DataStore) -> None:
    """Extend ``wake_fires.status`` CHECK to accept ``'dispatching'``.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("extending wake_fires.status CHECK to include 'dispatching' (v004)")
    await store.execute(_DROP_OLD_CHECK_SQL)
    await store.execute(_ADD_NEW_CHECK_SQL)
