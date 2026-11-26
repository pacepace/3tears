"""a stamped table's keyed delete is a gone mark (``mark_gone_keyed``): by key, by a batch, by a key prefix."""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.collections.gone_marks import mark_gone_keyed
from threetears.core.collections.schema_backed import STRING_TYPE, Column, Index, TableSchema
from threetears.core.ddl.row_epoch import ROW_STAMPED, next_epoch
from threetears.core.testing.stamped_tables import RecordingPool, stamps_in
from threetears.interfaces.keyed.rows import KeyBatch


def _stamped(name: str) -> TableSchema:
    """a stamped table keyed ``(owner, item)``, its ``item`` ranged in code-point order (declaring it registers it)."""
    return TableSchema(
        name=name,
        primary_key=("owner", "item"),
        columns=[Column("owner", STRING_TYPE), Column("item", STRING_TYPE)],
        row_epoch=ROW_STAMPED,
        indexes=(Index(f"{name}_by_item_c", "item", asc="item", collate={"item": "C"}),),
    )


def _marked(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [{"item": "x"}] if sql.startswith("UPDATE") else []


async def test_a_prefix_mark_is_a_keyed_range_under_its_pins() -> None:
    """Flips if a prefix gone mark is not one UPDATE of the C-ordered range under the key's pinned lead, or is sent
    as a DELETE, or leaves the epoch where it was."""
    _stamped("gm_items")
    pool = RecordingPool(_marked)
    before = next_epoch()
    marked = await mark_gone_keyed(pool, "gm_items", by={"owner": "o1"}, prefix={"item": "job/"}, returning=("item",))
    assert marked.count == 1
    [(sql, params)] = pool.sent
    assert sql == (
        'UPDATE "gm_items" SET "gone" = $1, "epoch" = GREATEST("epoch" + 1, $2::bigint) WHERE "owner" = $3 AND '
        '"item" COLLATE "C" >= $4 AND "item" COLLATE "C" < $5 AND (NOT "gone") RETURNING "item"'
    ), sql
    assert params[0] is True and params[2:] == ("o1", "job/", "job0"), params
    assert len(stamps_in(params, before)) == 1


async def test_a_batch_of_keys_is_marked_gone() -> None:
    """Flips if a batch of keys is deleted rather than marked gone."""
    _stamped("gm_batch")
    pool = RecordingPool()
    await mark_gone_keyed(
        pool, "gm_batch", keys=KeyBatch(("owner", "item"), keys=[["o1", "o2"], ["a", "b"]], casts=("text", "text"))
    )
    assert pool.sent and all(sql.startswith('UPDATE "gm_batch" SET "gone" = $1') for sql, _ in pool.sent), pool.sent


async def test_a_table_not_stamped_is_refused() -> None:
    """Flips if a gone mark is sent to a table with no gone flag (no declaration registers it stamped)."""
    with pytest.raises(ValueError, match="not a stamped table"):
        await mark_gone_keyed(RecordingPool(), "gm_never_declared", by={"owner": "o1"})
