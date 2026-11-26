"""the migration run's refusals at its end, under its DDL lock: a per-row trigger on a versioned table, and a
switched-on table that carries no epoch.

Every table carries its own epoch since the change-note ring retired (append-only L3, ``design-append-only-l3.md``
decisions 3 and 4): a restore of a lost change log reads it back from its epochs, so no migration gives a table a
trigger any more. What the run checks instead is that none brings the per-row path back
(:func:`refuse_row_triggers_here`) and that every table it switches on can be read back
(:func:`switched_tables_without_epoch`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.core.ddl.epoch_catalog import refuse_row_triggers, refuse_tables_without_epoch
from threetears.observe import get_logger

__all__ = [
    "refuse_row_triggers_here",
    "switched_tables_without_epoch",
]

log = get_logger(__name__)


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


async def switched_tables_without_epoch(store: Any, tables: Sequence[str]) -> frozenset[str]:
    """the switched-on ``tables`` of the run's schema that carry no epoch, refused by
    :func:`~threetears.core.ddl.epoch_catalog.refuse_tables_without_epoch` once its refusal is on; said at WARNING
    until then.

    :param store: the run's locked ``DataStore``, bound to the target schema
    :ptype store: Any
    :param tables: the switched-on tables the run's packages declare
    :ptype tables: Sequence[str]
    :return: those without an epoch
    :rtype: frozenset[str]
    :raises TableWithoutEpochError: when one carries none and the refusal is on
    """
    rows = await store.query("SELECT current_schema() AS schema_name")
    if not rows:
        # a shim that is no database (a preview's): it holds no table to judge
        return frozenset()
    missing = await refuse_tables_without_epoch(_Fetched(store), str(rows[0]["schema_name"]), tables)
    if missing:
        log.warning(
            "switched-on tables carry no epoch: their writes reach no reader of what changed until they do",
            extra={"extra_data": {"tables": sorted(missing)}},
        )
    return missing
