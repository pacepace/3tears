"""quiet-time cleanup of a stamped table's gone rows below the restore window: by key, in batches, a bucket at a time.

Append-only L3 (``design-append-only-l3.md``, decisions 2 and 5). A stamped table (``ROW_STAMPED``: one row a key,
updated in place) never deletes a key's row: its delete marks the row gone in place, the ``gone`` flag set and the epoch
risen past the delete's stamp (``SqlL3Backend.mark_gone``), so a restore of a lost change log, reading each bucket's
``epoch >= X`` (:mod:`~threetears.core.collections.table_restore`), hears it. Every shared keyed read skips a gone row
(``keyed.live_read``). Once its epoch is below the cleanup floor F (``row_epoch.cleanup_floor``, below every restore
floor the change log can still ask for) no restore reads it, so this removes it, the one ``DELETE`` a stamped table
takes (``keyed.refuse_in_place`` refuses every other).

**Keyed, in small batches, a bucket at a time.** A batch is two statements, Arrow end to end:

1. the bucket's next gone rows stamped in ``[since, F)``, oldest first, at most ``batch`` of them: one range of the
   table's ``(epoch_bucket, epoch)`` index (the restore's own), the bucket bound, read by the index's own order;
2. those rows deleted by their whole key, each only while it is still gone and still below F (``gone AND epoch <
   F``): a key written again since (live, its epoch risen) is kept.

The next batch starts at the highest epoch the last one read (``>=``: a delete statement stamps its rows alike, so
several share an epoch; the ones removed are gone, and the rest are live or risen past F). A bucket is done at a short
batch. ``since`` is the caller's progress: the epoch below which an earlier pass cleared every bucket (as
``QuietCleanup`` keeps a scope's ``cleaned_to``); a row marked gone later carries a newer epoch, so it is never behind
it.

**A row another row still references is left this round.** A stamped parent's delete marks its children gone before
itself, in one transaction (``BaseCollection.delete``), so every child's epoch is at most its parent's and the floor
reaches the children first. Cleanup still visits a bucket at a time, so it can reach a gone parent while a gone child
(or a live one, or one risen past F) still names it; under a ``NO ACTION`` / ``RESTRICT`` foreign key the database
refuses that parent's ``DELETE``. Cleanup then deletes the page's rows a key at a time, leaves each the database refuses
for a foreign key, counts it (``threetears_quiet_cleanup_referenced_total``) and never raises; the pass is left
unfinished, so its progress is not recorded past the row, and a later pass, its children gone, removes it. WHY a key at
a time only then: one statement is all or nothing, so one refused parent would hold back every row of its page.

**Only while nothing loads.** ``stirred`` is asked before every batch (``QuietCleanup``'s ``LoadGate.stirred``): the
pass stops there and the caller resumes it from ``since`` on its next quiet look. Nothing runs on a timer, nothing here
reads a table whole, scans it, or truncates it, and F is at least ten minutes older than any transaction a writer may
still commit, so cleanup never contends for a row a writer writes.

**Run by** ``QuietCleanup`` (:mod:`~threetears.core.collections.quiet_cleanup`), beside its rule for the versioned
tables: for each stamped table it is given, from its progress in ``threetears_cleanup_progress`` under
``(table_name, "*")``, its rows counted in ``threetears_quiet_cleanup_removed_total{kind="stamped_gone"}``. Each delete
is sent inside ``cleanup_deleting(F)`` (``broker_generation``), as every cleanup delete is: the one context a delete of
an append-only table is taken in.

Needs the ``arrow`` extra (pyarrow at module top).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import asyncpg
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.broker_generation import cleanup_deleting
from threetears.core.backends.protocol import parse_rowcount
from threetears.core.backends.schema_sql import key_array_type
from threetears.core.collections.schema_backed import TableSchema, rail_arrow_types
from threetears.core.ddl.row_epoch import BUCKET_COLUMN, EPOCH_COLUMN, GONE_COLUMN, RESTORE_BUCKETS
from threetears.core.keyed import Fragment, KeyBatch, Sort
from threetears.core.keyed._filters import binds_native_arrays
from threetears.core.keyed.binding import binder_for
from threetears.interfaces.keyed import page_statement
from threetears.interfaces.keyed.writes import delete_keyed_statements
from threetears.observe import get_logger
from threetears.observe.metrics import counter

__all__ = ["StampedSweep", "clear_stamped_gone"]

log = get_logger(__name__)

_REFERENCED = counter(
    "threetears_quiet_cleanup_referenced_total",
    description="gone rows quiet-time cleanup left for a round because a row of another table still references them "
    "(the database refused the DELETE for a foreign key); a later pass removes each once its children are gone",
    label_names=("table",),
)

#: the gone rows a batch removes by default: QuietCleanup's measured batch (500, quiet_cleanup._DEFAULT_BATCH: past it
#: the saving is under 10 % while a load waits longer for the batch in flight)
DEFAULT_BATCH = 500


@dataclass(frozen=True, slots=True)
class StampedSweep:
    """what one :func:`clear_stamped_gone` did.

    :ivar removed: the gone rows it removed
    :ivar finished: every bucket is cleared to the floor (the caller may record ``floor`` as its progress)
    :ivar referenced: the gone rows it left because another row still references them (never finished then)
    """

    removed: int
    finished: bool
    referenced: int = 0


def _casts(schema: TableSchema) -> tuple[str, ...]:
    """each key column's SQL type, its text cast back to it.

    :param schema: the stamped table
    :ptype schema: TableSchema
    :return: the types, in key order
    :rtype: tuple[str, ...]
    :raises ValueError: for a key column with no type a key batch binds
    """
    casts: list[str] = []
    for column in schema.pk_columns:
        kind = key_array_type(schema, column)
        if kind is None:
            raise ValueError(f"{schema.name}: key column {column!r} has no type cleanup can bind a key by")
        casts.append(kind.removesuffix("[]"))
    return tuple(casts)


async def _gone_page(pool: Any, schema: TableSchema, bucket: int, since: int, floor: int, batch: int) -> pa.Table:
    """the bucket's next ``batch`` gone rows stamped in ``[since, floor)``, oldest first: their key and epoch, one
    range of the ``(epoch_bucket, epoch)`` index, as Arrow.

    :param pool: the L3 pool
    :ptype pool: Any
    :param schema: the stamped table
    :ptype schema: TableSchema
    :param bucket: the bucket, ``0`` to ``RESTORE_BUCKETS - 1``
    :ptype bucket: int
    :param since: the lowest epoch read
    :ptype since: int
    :param floor: F
    :ptype floor: int
    :param batch: the most rows read
    :ptype batch: int
    :return: the key columns and ``epoch``
    :rtype: pyarrow.Table
    """
    columns = (*schema.pk_columns, EPOCH_COLUMN)
    sql, params = page_statement(
        schema.name,
        columns,
        {BUCKET_COLUMN: bucket},
        # the index's own order after its bound bucket: one range, no sort
        (Sort(EPOCH_COLUMN, cast="bigint"),),
        batch,
        None,
        False,
        (),
        (
            Fragment(f't."{GONE_COLUMN}"'),
            Fragment(f't."{EPOCH_COLUMN}" >= $1', (since,)),
            Fragment(f't."{EPOCH_COLUMN}" < $1', (floor,)),
        ),
        None,
    )
    types = rail_arrow_types([schema.column(name) for name in columns])
    return await l3_fetch_arrow(pool, sql, *params, types=types)


def _refused_for_a_reference(error: BaseException) -> bool:
    """whether the database refused a ``DELETE`` because another row still references a row it names: PostgreSQL's
    and YugabyteDB's ``foreign_key_violation`` (SQLSTATE 23503; the rail's proxy raises the same asyncpg class), or
    another engine's error saying so (DuckDB's ``Violates foreign key constraint``).

    :param error: what the delete raised
    :ptype error: BaseException
    :return: whether it is a foreign key's refusal
    :rtype: bool
    """
    return (
        isinstance(error, asyncpg.exceptions.ForeignKeyViolationError) or "foreign key constraint" in str(error).lower()
    )


async def _delete_keys(pool: Any, schema: TableSchema, page: pa.Table, floor: int) -> int:
    """delete ``page``'s rows by their whole key in one statement a page, each only while it is still gone and below
    ``floor``, inside ``cleanup_deleting(floor)``.

    :param pool: the L3 pool
    :ptype pool: Any
    :param schema: the stamped table
    :ptype schema: TableSchema
    :param page: the rows, their key columns
    :ptype page: pyarrow.Table
    :param floor: F
    :ptype floor: int
    :return: the rows deleted
    :rtype: int
    """
    key = schema.pk_columns
    columns = [page.column(name).combine_chunks() for name in key]
    casts = _casts(schema)
    statements = delete_keyed_statements(
        schema.name,
        # the table's own key, written where the delete is so the key it binds is the one deleted by
        keys=KeyBatch(key, keys=columns, casts=casts),
        where=(Fragment(f'"{GONE_COLUMN}"'), Fragment(f'"{EPOCH_COLUMN}" < $1', (floor,))),
        batch=max(1, page.num_rows),
        natively=binds_native_arrays(pool),
        binder=binder_for(pool, KeyBatch(key, keys=columns, casts=casts)),
    )
    deleted = 0
    for sql, params in statements:
        # marked cleanup's with its floor, as every cleanup delete is (and bounded by it in its own WHERE)
        with cleanup_deleting(floor):
            deleted += parse_rowcount(await pool.execute(sql, *params))
    return deleted


async def _delete_gone(pool: Any, schema: TableSchema, page: pa.Table, floor: int) -> tuple[int, int]:
    """delete a page's rows by their whole key, each only while it is still gone and below ``floor``: a key written
    again since is live (or risen past F) and stays. Cleanup's own statements, inside ``cleanup_deleting(floor)``: the
    one ``DELETE`` a stamped table takes (``keyed.refuse_in_place`` refuses every other). When the database refuses
    the page's statement for a foreign key, its rows go a key at a time and each it refuses so is left (the module's
    docstring).

    :param pool: the L3 pool
    :ptype pool: Any
    :param schema: the stamped table
    :ptype schema: TableSchema
    :param page: the rows, their key columns
    :ptype page: pyarrow.Table
    :param floor: F
    :ptype floor: int
    :return: the rows deleted, and the rows left because another row still references them
    :rtype: tuple[int, int]
    """
    try:
        return await _delete_keys(pool, schema, page, floor), 0
    except Exception as error:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- only a foreign key's refusal is taken (engines raise it as different classes); any other is raised
        if not _refused_for_a_reference(error):
            raise
    deleted = 0
    referenced = 0
    for at in range(page.num_rows):
        try:
            deleted += await _delete_keys(pool, schema, page.slice(at, 1), floor)
        except Exception as error:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- only a foreign key's refusal is taken (engines raise it as different classes); any other is raised
            if not _refused_for_a_reference(error):
                raise
            referenced += 1
    if referenced:
        _REFERENCED.labels(table=schema.name).inc(referenced)
        log.info(
            "stamped cleanup left gone rows other rows still reference; a later pass removes them",
            extra={"extra_data": {"table": schema.name, "referenced": referenced, "floor": floor}},
        )
    return deleted, referenced


async def clear_stamped_gone(
    pool: Any,
    schema: TableSchema,
    *,
    since: int,
    floor: int,
    batch: int = DEFAULT_BATCH,
    stirred: Callable[[], bool] = lambda: False,
) -> StampedSweep:
    """remove ``schema``'s gone rows stamped in ``[since, floor)``, a bucket at a time, ``batch`` rows a batch,
    stopping before a batch when ``stirred`` says a load started (the module's docstring).

    :param pool: the L3 pool the table lives in
    :ptype pool: Any
    :param schema: the stamped table (``ROW_STAMPED``)
    :ptype schema: TableSchema
    :param since: the caller's progress: every bucket was cleared below it
    :ptype since: int
    :param floor: F, the cleanup floor (``row_epoch.cleanup_floor``): no row at or above it is touched
    :ptype floor: int
    :param batch: the most gone rows a batch removes
    :ptype batch: int
    :param stirred: whether a load started; asked before every batch
    :ptype stirred: Callable[[], bool]
    :return: what it did
    :rtype: StampedSweep
    :raises ValueError: for a table that is not stamped, or a batch under one
    """
    if schema.row_epoch is None or schema.versioned:
        raise ValueError(f"{schema.name}: stamped cleanup clears a ROW_STAMPED table's gone rows")
    if batch < 1:
        raise ValueError(f"a cleanup batch removes at least one row; got {batch}")
    removed = 0
    referenced = 0
    for bucket in range(RESTORE_BUCKETS):
        low = since
        while True:
            if stirred():
                log.debug(
                    "stamped cleanup stopped for a load",
                    extra={"extra_data": {"table": schema.name, "bucket": bucket, "removed": removed}},
                )
                return StampedSweep(removed=removed, finished=False, referenced=referenced)
            page = await _gone_page(pool, schema, bucket, low, floor, batch)
            deleted, left = await _delete_gone(pool, schema, page, floor) if page.num_rows else (0, 0)
            removed += deleted
            referenced += left
            if page.num_rows < batch:
                break
            highest = int(pc.max(page.column(EPOCH_COLUMN)).as_py())
            if left:
                # the rows left are still gone below F and would come back at ``>=``: this pass goes on past the
                # page's highest epoch (any row there it does not reach waits, with them, for a later pass: a pass that
                # left a row is never recorded finished)
                low = highest + 1
                continue
            if not deleted:
                # NOSILENT: a full page of gone rows below F of which none went: the same page would come back, so the
                # pass stops unfinished rather than spin; the caller's next look tries again
                log.warning(
                    "stamped cleanup removed none of a full page of gone rows; stopped",
                    extra={"extra_data": {"table": schema.name, "bucket": bucket, "since": low, "floor": floor}},
                )
                return StampedSweep(removed=removed, finished=False, referenced=referenced)
            # the next batch from the highest epoch read: those removed are gone, the rest live or risen past F
            low = highest
    log.debug(
        "stamped cleanup cleared to the floor",
        extra={"extra_data": {"table": schema.name, "removed": removed, "referenced": referenced}},
    )
    return StampedSweep(removed=removed, finished=not referenced, referenced=referenced)
