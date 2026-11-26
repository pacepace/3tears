"""the migration run's refusals at its end, under its DDL lock: a per-row trigger on a versioned table, a
switched-on table that carries no epoch, and a moveable stamped table whose schema holds no moves table.

Every table carries its own epoch since the change-note ring retired (append-only L3, ``design-append-only-l3.md``
decisions 3 and 4): a restore of a lost change log reads it back from its epochs, so no migration gives a table a
trigger any more. What the run checks instead is that none brings the per-row path back
(:func:`refuse_row_triggers_here`) and that every table it switches on can be read back
(:func:`switched_tables_without_epoch`), and that a table a write may move between tenants has the moves table its
writes append to (:func:`refuse_moveable_without_moves_here`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.core.ddl.epoch_catalog import (
    refuse_moveable_without_moves,
    refuse_row_triggers,
    refuse_tables_without_epoch,
)

__all__ = [
    "refuse_moveable_without_moves_here",
    "refuse_row_triggers_here",
    "switched_tables_without_epoch",
]


class _Fetched:
    """a run's ``DataStore`` read as the catalog reads of :mod:`~threetears.core.ddl.epoch_catalog` take it: its
    ``query`` is their ``fetch``."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """rows of ``query``, by the store's ``query``."""
        rows: list[Any] = await self._store.query(query, *args)
        return rows


async def refuse_row_triggers_here(store: Any) -> None:
    """refuse a per-row trigger on any versioned table of the run's schema (:mod:`~threetears.core.ddl.epoch_catalog`):
    called at the end of every run, under its DDL lock, so no migration leaves the slow path behind.

    :param store: the run's locked ``DataStore``, bound to the target schema
    :ptype store: Any
    :return: nothing
    :rtype: None
    :raises PerRowTriggerError: when one is found, naming each
    """
    rows = await store.query("SELECT current_schema() AS schema_name")
    if not rows:
        # a shim that is no database (a preview's): it holds no table to judge
        return
    await refuse_row_triggers(_Fetched(store), str(rows[0]["schema_name"]))


async def refuse_moveable_without_moves_here(store: Any) -> None:
    """refuse a stamped table of the run's schema that declares a moveable column while the schema holds no moves table
    (:func:`~threetears.core.ddl.epoch_catalog.refuse_moveable_without_moves`): called at the end of every run, under
    its DDL lock, so a moving write never meets a missing table at write time.

    :param store: the run's locked ``DataStore``, bound to the target schema
    :ptype store: Any
    :return: nothing
    :rtype: None
    :raises MovesTableMissingError: when one is found, naming each table, its schema and how to create the moves table
    """
    rows = await store.query("SELECT current_schema() AS schema_name")
    if not rows:
        # a shim that is no database (a preview's): it holds no table to judge
        return
    await refuse_moveable_without_moves(_Fetched(store), str(rows[0]["schema_name"]))


async def switched_tables_without_epoch(store: Any, tables: Sequence[str]) -> frozenset[str]:
    """refuse the switched-on ``tables`` of the run's schema that carry no epoch
    (:func:`~threetears.core.ddl.epoch_catalog.refuse_tables_without_epoch`).

    :param store: the run's locked ``DataStore``, bound to the target schema
    :ptype store: Any
    :param tables: the switched-on tables the run's packages declare
    :ptype tables: Sequence[str]
    :return: none: a table without an epoch is refused
    :rtype: frozenset[str]
    :raises TableWithoutEpochError: when one carries none
    """
    rows = await store.query("SELECT current_schema() AS schema_name")
    if not rows:
        # a shim that is no database (a preview's): it holds no table to judge
        return frozenset()
    return await refuse_tables_without_epoch(_Fetched(store), str(rows[0]["schema_name"]), tables)
