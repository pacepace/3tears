"""the datasource tables this package declares ``ROW_STAMPED``, as a migration that stamps them names them.

Append-only L3 (``design-append-only-l3.md`` decision 3, Pace 2026-11-25): the tables are their own change record, so
the change-note ring can be retired. ``datasources``, ``datasource_tables`` and ``table_templates`` are switched on
(each write advances its write generation), so the ring was their only restore path; each is declared ``ROW_STAMPED``
(:mod:`threetears.datasources.collections`).

This package has no migrations: the tables' DDL is the hub's (hub v001, v006, v007, v016, ...). The hub's migration
that stamps them calls :func:`datasource_tables_carry_their_epoch` (the six steps of
:func:`~threetears.core.data.migrations.stamp_tables.stamp_existing_table` for each), so the frozen keys and the steps
live beside the declarations they must match.
"""

from __future__ import annotations

from typing import Any, Final

from threetears.core.data.migrations.stamp_tables import StampingTable, stamp_existing_tables
from threetears.observe import get_logger

__all__ = ["STAMPED_TABLES", "datasource_tables_carry_their_epoch"]

log = get_logger(__name__)

#: the tables a hub migration stamps, each key as declared at 0.95 (FROZEN: a later change of a key is its own
#: migration)
STAMPED_TABLES: Final[tuple[StampingTable, ...]] = (
    StampingTable("datasources", (("id", "uuid"),)),
    StampingTable("datasource_tables", (("id", "uuid"),)),
    StampingTable("table_templates", (("id", "uuid"),)),
)


async def datasource_tables_carry_their_epoch(store: Any) -> int:
    """stamp the three datasource tables: each one's epoch, gone flag, restore bucket, index and declaration.

    :param store: the hub migration's ``DataStore``, bound to the platform schema via search_path
    :ptype store: DataStore
    :return: the rows the backfills stamped
    :rtype: int
    """
    stamped = await stamp_existing_tables(store, STAMPED_TABLES)
    log.info("the datasource tables are ROW_STAMPED", extra={"extra_data": {"rows_stamped": stamped}})
    return stamped
