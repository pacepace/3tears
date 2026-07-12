"""
agent-memory v024: stored salience substrate + scope-model relaxation.

Presence/aliveness program (3tears v0.15.0). Two additive changes to
``memories``:

1. **Salience substrate** — five columns backing scheduled decay and
   reinforcement:

   - ``salience`` NUMERIC(5,4) NOT NULL DEFAULT 0.5 — the stored,
     decayed ranking weight (query-time recency stays; this is the
     durable, reinforced signal).
   - ``last_decayed_at`` TIMESTAMPTZ NULL — decay anchor. Anchoring age
     on the last decay run (not last access) makes total decay over a
     period cadence-independent, so nightly and hourly passes agree.
   - ``last_accessed`` TIMESTAMPTZ NULL — reinforcement telemetry,
     stamped on ambient retrieval.
   - ``evergreen`` BOOLEAN NOT NULL DEFAULT FALSE — pin for core
     identity facts: excluded from BOTH decay and the access-bump.
   - ``superseded_by`` UUID NULL — soft ref (no FK) to a consolidation
     gist. Ambient retrieval excludes non-null; direct recall still
     finds it; the source's own salience is preserved for clean un-merge.

2. **Scope-model relaxation** — ``customer_id`` and ``user_id`` become
   nullable so the memory primitive supports all three scope grains
   (agent / customer / user). metallm enforces NOT NULL at its own
   consumer layer; the relaxation is a constraint RELAXATION (no
   existing row violates it).

The NOT-NULL columns carry a server-side DEFAULT, so on modern Postgres
the add is metadata-only (no table rewrite) and every existing INSERT
that omits them lets the default apply — the other five consumers never
read them.

Idempotent: ``ADD COLUMN IF NOT EXISTS`` for the new columns;
``ALTER COLUMN ... DROP NOT NULL`` is a no-op when the column is already
nullable, so replay is safe.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_memory_salience_and_relax_scope",
]

log = get_logger(__name__)


_ADD_SALIENCE_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS salience NUMERIC(5,4) NOT NULL DEFAULT 0.5"

_ADD_LAST_DECAYED_AT_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMPTZ NULL"

_ADD_LAST_ACCESSED_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ NULL"

_ADD_EVERGREEN_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS evergreen BOOLEAN NOT NULL DEFAULT FALSE"

_ADD_SUPERSEDED_BY_SQL = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS superseded_by UUID NULL"

_RELAX_CUSTOMER_ID_SQL = "ALTER TABLE memories ALTER COLUMN customer_id DROP NOT NULL"

_RELAX_USER_ID_SQL = "ALTER TABLE memories ALTER COLUMN user_id DROP NOT NULL"


async def add_memory_salience_and_relax_scope(store: DataStore) -> None:
    """add the salience substrate columns and relax the scope columns.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding memory salience substrate + relaxing scope NOT NULL (v024)")
    await store.execute(_ADD_SALIENCE_SQL)
    await store.execute(_ADD_LAST_DECAYED_AT_SQL)
    await store.execute(_ADD_LAST_ACCESSED_SQL)
    await store.execute(_ADD_EVERGREEN_SQL)
    await store.execute(_ADD_SUPERSEDED_BY_SQL)
    await store.execute(_RELAX_CUSTOMER_ID_SQL)
    await store.execute(_RELAX_USER_ID_SQL)
