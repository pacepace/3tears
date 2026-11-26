"""mcp v003: ``mcp_tool_grants`` carries its epoch, and a restore reads it a bucket at a time.

Append-only L3 (``design-append-only-l3.md`` decision 3, Pace 2026-11-25): the tables are their own change record, so
the change-note ring can be retired. The grants are switched on (every committed grant write advances the table's
generation, which ``LocalGrantAuthorizer`` follows), so the ring was their only restore path; the table is declared
``ROW_STAMPED`` (one row a key, carrying its last write's epoch, a revoke marking the grant gone in place).

The six steps of :func:`~threetears.core.data.migrations.stamp_tables.stamp_existing_table`: the epoch added nullable,
backfilled a page of keys a keyed ``UPDATE``, made ``NOT NULL``; the gone flag; the stored bucket, its
``(epoch_bucket ASC, epoch)`` index (split a tablet a bucket on YugabyteDB) and its ``COMMENT ON INDEX`` declaration.

Not held behind ``held_contract`` as v002 is: it only adds, and every pod of this release writes the epoch (the
outage-window cutover, Pace 2026-11-25: one rollout, everything scaled to zero). Idempotent (each DDL ``IF NOT
EXISTS``, the backfill only rows whose epoch is NULL), never truncating. Forward-only.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.migrations.stamp_tables import StampingTable, stamp_existing_tables
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["grants_carry_their_epoch"]

log = get_logger(__name__)

#: the table v003 stamps, its key as declared at v003 (FROZEN: a later change of its key is its own migration)
_STAMPED_TABLES: Final[tuple[StampingTable, ...]] = (
    StampingTable("mcp_tool_grants", (("grant_id", "uuid"),), tenancy=("undeclared", None)),
)


async def grants_carry_their_epoch(store: DataStore) -> None:
    """stamp ``mcp_tool_grants``: its epoch, gone flag, restore bucket, index and declaration (v003).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    stamped = await stamp_existing_tables(store, _STAMPED_TABLES)
    log.info("mcp v003: the tool grants are ROW_STAMPED", extra={"extra_data": {"rows_stamped": stamped}})
