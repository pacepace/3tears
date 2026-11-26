"""a keyed delete of a stamped table's rows (``ROW_STAMPED``) is a gone mark, never a ``DELETE``.

Append-only L3 (``design-append-only-l3.md`` decision 2): "a real delete leaves nothing for a restore to find". A
stamped table keeps one row a key, so a delete of its rows marks each gone in place: one keyed ``UPDATE`` setting the
``gone`` flag and rising the epoch past the delete's stamp (``GREATEST(epoch + 1, <stamp>)``,
:func:`~threetears.core.ddl.row_epoch.rising_stamp`), a row already gone left as it is. A restore reading the table's
``(epoch_bucket, epoch)`` index since X hears the delete, and every shared keyed read skips the row
(``threetears.core.keyed.live_read``). 3tears refuses a real keyed ``DELETE`` of a stamped L3 table
(``keyed.refuse_in_place``).

:func:`mark_gone_keyed` takes what ``delete_keyed`` takes (``by``, a ``keys`` batch, a key ``prefix``, ``where``
guards, ``returning``), so a delete site changes its call and nothing else. A prefix names the rows whose leading text
key column starts with it: a code-point range on that column (``prefix_conditions``, read through a ``COLLATE "C"``
index), never a scan. The stamp is drawn from the caller's :class:`CallerTransaction` when ``conn`` joins one
(counted in its commit's ``s_min``), else from the process clock as that one statement's own (``stamp_for``); the
statement says it stamps on the wire (``update_keyed`` reads the stamp off the epoch's value), as every stamped
write does, so the broker that commits it records it.

The one gone-mark ``UPDATE`` every 3tears site that names its rows by key sends (a declared table's
``mark_gone_in_store``, a stamped child's foreign-key action, a fence's clear). ``SqlL3Backend.mark_gone`` keeps its
own, a batch of Arrow key columns stamped with the epoch its caller drew.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, GONE_COLUMN, rising_stamp
from threetears.core.keyed import DEFAULT_KEY_BATCH, Fragment, KeyBatch, update_keyed
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.keys import stamped_table
from threetears.interfaces.keyed.writes import KeyedWrite

__all__ = ["mark_gone_keyed"]

#: the guard every gone mark carries: a row already gone is left as it is, its epoch not risen again
_LIVE: Final[Fragment] = Fragment(f"NOT {quote_identifier(GONE_COLUMN)}")


async def mark_gone_keyed(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any] | None = None,
    keys: KeyBatch | None = None,
    prefix: Mapping[str, str] | None = None,
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
    batch: int = DEFAULT_KEY_BATCH,
) -> KeyedWrite:
    """delete the rows of the stamped ``table`` that ``by``, ``keys`` and ``prefix`` (and ``where``) name by marking
    each gone in place: one keyed ``UPDATE`` a batch, ``gone`` set and the epoch risen past the delete's stamp, a row
    already gone left as it is. Never a ``DELETE`` (the module's docstring).

    :param conn: the connection (inside the caller's transaction, perhaps a :class:`CallerTransaction`'s) or the pool
    :ptype conn: Any
    :param table: the stamped table, a TRUSTED identifier, as its schema names it
    :ptype table: str
    :param by: columns pinned (a leading part of its key, or of an index declared for it), as ``update_keyed`` takes
    :ptype by: Mapping[str, Any] | None
    :param keys: a batch of keys the rows are named by, ``batch`` keys a statement
    :ptype keys: KeyBatch | None
    :param prefix: text key columns -> the prefix the rows marked start with, a code-point range (``update_keyed``)
    :ptype prefix: Mapping[str, str] | None
    :param where: guards beside the key, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :param returning: the columns the marked rows return; none when empty
    :ptype returning: Sequence[str]
    :param batch: the most keys a statement names
    :ptype batch: int
    :return: the count marked, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when ``table`` is not a stamped table (its schema registers it so), or neither ``by``, ``keys``
        nor ``prefix`` names a column
    """
    if stamped_table(table, l3=True) is None:
        raise ValueError(f"{table} is not a stamped table (ROW_STAMPED): a delete of its rows is no gone mark")
    stamp = CallerTransaction.stamp_for(conn, table)
    return await update_keyed(
        conn,
        table,
        # a RisingStamp, not a bare Fragment: update_keyed reads the stamp off it and sends it as the write's s_min
        values={GONE_COLUMN: True, EPOCH_COLUMN: rising_stamp(stamp, quote_identifier(EPOCH_COLUMN))},
        by=by,
        keys=keys,
        prefix=prefix,
        where=(_LIVE, *where),
        returning=returning,
        batch=batch,
    )
