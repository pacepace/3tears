"""the order a compare-and-swap row was written in, carried into L3 so L3 cannot go backwards.

:meth:`~threetears.core.collections.base.BaseCollection.l2_cas_mutate` orders writes by the L2
(NATS KV) revision the compare-and-swap won. Each winner persists its row to L3 on its own --
synchronously or through the write buffer -- so two winners of consecutive revisions can reach L3
in either order. Unfenced, the earlier row could land last and stay there; L2 hides that until it
loses the key (a memory-backed broker loses every key on restart), and the next mutation then
seeds from L3's stale row and the later change is gone for good.

The fence is the pair ``(epoch, revision)``, stored in two columns of the row itself:

- ``revision`` is the L2 revision the write won -- the bucket's stream sequence, monotonic across
  every key in the bucket for the life of the stream;
- ``epoch`` is the creation time of the stream that revision belongs to. A wiped bucket is
  recreated with its sequence back at 1, so a revision alone would order every write after a
  broker restart below every write before it and refuse them all. The creation time moves forward
  on every recreation, which is what makes the pair monotonic across incarnations.

An L3 write carrying an order lands only over a row whose stored order is strictly older. A row
with a newer-or-equal stored order is left alone, and that is not an error: the newer value is
already there. ``NULL`` in either column is older than every order, so a row written by some
other path never blocks a compare-and-swap.

**The column contract** a three-tier collection that uses ``l2_cas_mutate`` must meet:

- ``l2_epoch`` -- ``TIMESTAMPTZ``, nullable, mutable;
- ``l2_revision`` -- ``BIGINT``, nullable, mutable (a busy bucket passes 2**31 revisions).

:func:`l2_order_migration_statements` renders the migration that adds both to an existing table
and backfills existing rows to :data:`L2_ORDER_FLOOR`, below every order a live bucket can
produce, so the fence works on the first write after the migration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

__all__ = [
    "L2_EPOCH_COLUMN",
    "L2_ORDER_COLUMNS",
    "L2_ORDER_FLOOR",
    "L2_REVISION_COLUMN",
    "L2Order",
    "l2_order_migration_statements",
    "l2_order_of",
    "with_l2_order",
    "without_l2_order",
]

#: the column holding the creation time of the L2 stream a row's revision belongs to.
L2_EPOCH_COLUMN: Final = "l2_epoch"

#: the column holding the L2 revision a row's compare-and-swap won.
L2_REVISION_COLUMN: Final = "l2_revision"

#: both order columns, in the order they are declared and migrated.
L2_ORDER_COLUMNS: Final[tuple[str, str]] = (L2_EPOCH_COLUMN, L2_REVISION_COLUMN)

#: an unqualified or schema-qualified SQL identifier, as the migration interpolates it.
_TABLE_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


@dataclass(frozen=True, order=True, slots=True)
class L2Order:
    """where one compare-and-swap write sits in its key's history.

    Compared field by field: a later incarnation of the bucket orders after every revision of an
    earlier one, and within one incarnation the higher revision is the later write.

    :ivar epoch: creation time of the L2 stream the write landed in, timezone-aware UTC
    :ivar revision: the L2 revision the write produced
    """

    epoch: datetime
    revision: int

    def __post_init__(self) -> None:
        """refuse an order that could not compare correctly against a stored one.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``epoch`` is timezone-naive or ``revision`` is negative
        """
        if self.epoch.tzinfo is None:
            raise ValueError(
                "L2Order.epoch must be timezone-aware; a naive time cannot be ordered against a stored one"
            )
        if self.revision < 0:
            raise ValueError(f"L2Order.revision must not be negative, got {self.revision}")


#: the order every row written before the fence existed is backfilled to: below every order a
#: live bucket can produce, since no stream was created before the Unix epoch.
L2_ORDER_FLOOR: Final = L2Order(datetime(1970, 1, 1, tzinfo=UTC), 0)


def l2_order_of(row: dict[str, Any]) -> L2Order | None:
    """read the order a row carries, or ``None`` when it carries none.

    Accepts the epoch as the aware ``datetime`` every tier hands back, or as the ISO string a
    row takes on through a JSON boundary (the write buffer's durable copy).

    :param row: a row dict
    :ptype row: dict[str, Any]
    :return: the row's order, or ``None`` when either column is absent or ``NULL``
    :rtype: L2Order | None
    :raises ValueError: when an epoch string does not parse or is timezone-naive
    """
    epoch = row.get(L2_EPOCH_COLUMN)
    revision = row.get(L2_REVISION_COLUMN)
    if epoch is None or revision is None:
        return None
    if isinstance(epoch, str):
        epoch = datetime.fromisoformat(epoch)
    return L2Order(epoch=epoch, revision=int(revision))


def with_l2_order(row: dict[str, Any], order: L2Order) -> dict[str, Any]:
    """return a copy of ``row`` carrying ``order`` in its order columns.

    :param row: the row to stamp
    :ptype row: dict[str, Any]
    :param order: the order its compare-and-swap won
    :ptype order: L2Order
    :return: the stamped copy
    :rtype: dict[str, Any]
    """
    return {**row, L2_EPOCH_COLUMN: order.epoch, L2_REVISION_COLUMN: order.revision}


def without_l2_order(row: dict[str, Any]) -> dict[str, Any]:
    """return a copy of ``row`` carrying no order: both columns ``NULL``.

    What every write that did not win a compare-and-swap stores, so an order copied from a row it
    read can never be mistaken for one it won.

    :param row: the row to clear
    :ptype row: dict[str, Any]
    :return: the cleared copy
    :rtype: dict[str, Any]
    """
    return {**row, L2_EPOCH_COLUMN: None, L2_REVISION_COLUMN: None}


def l2_order_migration_statements(table_name: str) -> tuple[str, ...]:
    """the statements that give an existing table the order columns and backfill its rows.

    Every statement is idempotent, so a migration replayed on recovery is safe, and the DDL and
    the backfill are separate statements: YugabyteDB does not run DDL and DML in one transaction.
    The backfill writes :data:`L2_ORDER_FLOOR` into rows that carry no order, so the first
    compare-and-swap after the migration supersedes them and every later one is fenced.

    :param table_name: the table to migrate, unqualified or ``schema.table``
    :ptype table_name: str
    :return: the ``ALTER TABLE`` statements, then the backfill ``UPDATE``
    :rtype: tuple[str, ...]
    :raises ValueError: when ``table_name`` is not a plain SQL identifier
    """
    if not _TABLE_NAME.match(table_name):
        raise ValueError(f"{table_name!r} is not a plain SQL table name; refusing to interpolate it")
    floor = L2_ORDER_FLOOR.epoch.isoformat()
    return (
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {L2_EPOCH_COLUMN} TIMESTAMPTZ",
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {L2_REVISION_COLUMN} BIGINT",
        (
            f"UPDATE {table_name} SET {L2_EPOCH_COLUMN} = '{floor}'::timestamptz, "
            f"{L2_REVISION_COLUMN} = {L2_ORDER_FLOOR.revision} "
            f"WHERE {L2_EPOCH_COLUMN} IS NULL OR {L2_REVISION_COLUMN} IS NULL"
        ),
    )
