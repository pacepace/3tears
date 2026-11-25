"""A write of many rows is broadcast as one message naming them all: heard exactly as one message a row, and handed to
a copy as one Arrow batch."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.invalidation_rows import (
    InvalidatedRows,
    invalidation_batches,
    key_batch,
    row_ids,
)
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_ROWS = 1_200

#: the most bytes of packed keys one message carries (invalidation_rows' own bound, restated so a test says what it
#: holds it to)
_BYTES_A_MESSAGE = 512 * 1024

#: a page of a report's rows: the write the copy must hear as one message and one Arrow batch
_PAGE = 5_000


class _Heard:
    """what a receiving registry did with what it heard: each row applied, each generation mark counted."""

    def __init__(self, registry: CollectionRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        self.applied: list[tuple[str, tuple[str, ...], Any, Any]] = []
        self.counted: list[tuple[str, str, int]] = []

        def apply(message: CacheInvalidationMessage) -> None:
            self.applied.append((message.table, tuple(message.ids), message.l2_current_scope, message.columns))

        # every row applied reaches the caches derived from its table, in order
        registry.register_derived_cache(
            "results", on_row=apply, on_table_dropped=lambda: None, on_partition=lambda _prefix: None
        )
        marks = registry.generation_marks
        real_hear = type(marks).hear

        def hear(owner: Any, table: str, generation: str, bump_rows: int, *, rows: int = 1) -> None:
            if owner is marks:
                self.counted.extend([(table, generation, bump_rows)] * rows)
            real_hear(owner, table, generation, bump_rows, rows=rows)

        monkeypatch.setattr(type(marks), "hear", hear)


def _keys(count: int) -> list[tuple[str, str]]:
    return [(f"r{n % 7}", f"g{n}") for n in range(count)]


def _columns(count: int) -> list[dict[str, str | None] | None]:
    # a row the write saw none of names no columns; a NULL column is named as None: the two stay apart
    return [None if n % 5 == 0 else {"state": f"S{n % 3}" if n % 4 else None} for n in range(count)]


async def _broadcast(batched: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[_Heard, int]:
    bus = FakeNatsClient()
    sender, receiver = CollectionRegistry(), CollectionRegistry()
    heard = _Heard(receiver, monkeypatch)
    await receiver.start_invalidation_listener(bus)
    keys, columns = _keys(_ROWS), _columns(_ROWS)
    if batched:
        await sender.publish_invalidations(
            bus, "results", invalidation_batches(keys, columns), generation="inc:9", bump_rows=_ROWS
        )
    else:
        for key, row in zip(keys, columns, strict=True):
            await sender.publish_invalidation(bus, "results", key, generation="inc:9", bump_rows=_ROWS, columns=row)
    await receiver.stop_invalidation_listener()
    return heard, len(bus.published)


async def test_a_batch_is_heard_row_for_row_as_the_singles_are(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a batched broadcast evicts other rows, in another order, with other columns, or counts its advance
    otherwise than one message a row would."""
    singles, single_publishes = await _broadcast(False, monkeypatch)
    batched, batch_publishes = await _broadcast(True, monkeypatch)

    assert single_publishes == _ROWS
    assert batch_publishes == 1
    assert batched.applied == singles.applied
    assert len(batched.applied) == _ROWS
    assert batched.counted == singles.counted == [("results", "inc:9", _ROWS)] * _ROWS


async def test_a_batch_from_this_registry_is_not_applied_here() -> None:
    """Flips if a registry applies the rows of its own batched broadcast on hearing it, as it never does its own
    single ones: a derived cache is told of each of its own rows once, by the write (``_publish_rows``), and not
    again by the echo (``publish_invalidations`` is the bus half alone)."""
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    applied: list[Any] = []
    registry.register_derived_cache(
        "results", on_row=applied.append, on_table_dropped=lambda: None, on_partition=lambda _prefix: None
    )
    await registry.start_invalidation_listener(bus)
    await registry.publish_invalidations(bus, "results", invalidation_batches([("a", "b"), ("a", "c"), ("a", "d")]))
    await registry.stop_invalidation_listener()
    assert len(bus.published) == 1
    assert applied == []


async def test_a_batch_reaches_a_cache_taking_key_batches_as_one_arrow_batch_a_message() -> None:
    """Flips if a batched broadcast reaches a cache that takes key batches (a copy, ``on_keys``) as a row message a
    key (a Python object a key in the copy, decision 35) instead of one Arrow batch a message at its advance, or if
    a cache taking only rows stops hearing each of them."""
    pa = pytest.importorskip("pyarrow")
    bus = FakeNatsClient()
    sender, receiver = CollectionRegistry(), CollectionRegistry()
    batches: list[tuple[Any, str | None]] = []
    copy_rows: list[Any] = []
    rows: list[Any] = []
    receiver.register_derived_cache(
        "results",
        on_row=copy_rows.append,
        on_table_dropped=lambda: None,
        on_partition=lambda _prefix: None,
        on_keys=lambda keys, epoch: batches.append((keys, epoch)),
    )
    receiver.register_derived_cache(
        "results", on_row=rows.append, on_table_dropped=lambda: None, on_partition=lambda _prefix: None
    )
    await receiver.start_invalidation_listener(bus)
    await sender.publish_invalidations(
        bus, "results", invalidation_batches(_keys(_ROWS)), generation="inc:9", bump_rows=_ROWS
    )
    await receiver.stop_invalidation_listener()

    assert copy_rows == [], "no row message a key to a cache that takes batches"
    assert len(batches) == 1 and all(isinstance(keys, pa.Table) and epoch == "inc:9" for keys, epoch in batches)
    heard = batches[0][0]
    assert list(zip(heard.column("k0").to_pylist(), heard.column("k1").to_pylist(), strict=True)) == _keys(_ROWS)
    assert len(rows) == _ROWS, "a cache taking rows hears each"


@pytest.mark.parametrize("batched", [False, True])
async def test_a_broadcast_of_an_advance_already_applied_from_its_record_is_not_told_to_a_copy_again(
    batched: bool,
) -> None:
    """Flips if a row broadcast that lands after the catch-up applied its epoch's record (the record told the copy
    its keys then) is told to a cache that takes key batches again: the copy read the key from L3 twice for one
    change (the live refresh test, 2 runs in 6, 2026-11-24). A cache taking only rows still hears it, and a
    broadcast of an advance not yet applied still reaches the copy."""
    bus = FakeNatsClient()
    sender, receiver = CollectionRegistry(), CollectionRegistry()
    receiver.follow_generation("results")
    receiver.generation_marks.adopt_generation("results", "inc:8")
    copy: list[Any] = []
    rows: list[Any] = []
    receiver.register_derived_cache(
        "results",
        on_row=copy.append,
        on_table_dropped=lambda: None,
        on_partition=lambda _prefix: None,
        on_keys=lambda keys, epoch: copy.append((keys.num_rows, epoch)),
    )
    receiver.register_derived_cache(
        "results", on_row=rows.append, on_table_dropped=lambda: None, on_partition=lambda _prefix: None
    )
    await receiver.start_invalidation_listener(bus)

    async def publish(keys: list[tuple[str, str]], generation: str) -> None:
        if batched:
            await sender.publish_invalidations(
                bus, "results", invalidation_batches(keys), generation=generation, bump_rows=len(keys)
            )
            return
        for key in keys:
            await sender.publish_invalidation(bus, "results", key, generation=generation, bump_rows=len(keys))

    # the catch-up applied epoch inc:9 from its record before its broadcast landed
    receiver.generation_marks.applied("results", "inc:9")
    await publish([("r", f"g{n}") for n in range(2)], "inc:9")
    assert copy == [], "a broadcast of an advance applied from its record was told to the copy again"
    assert len(rows) == 2
    await publish([("r", f"h{n}") for n in range(2)], "inc:10")
    await receiver.stop_invalidation_listener()
    assert copy, "a broadcast of an advance not yet applied did not reach the copy"


class _Vote(BaseEntity):
    """a precinct's row: its own id is the last key column."""

    primary_key_field = "geo_id"


def _collection(name: str, nats_client: Any) -> Any:
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal")
    schema = TableSchema(
        name=name,
        primary_key=("state", "geo_id"),
        columns=[Column("state", STRING_TYPE), Column("geo_id", STRING_TYPE)],
        l1_eviction=TEST_L1_TABLE,
    )
    return collection_for_schema(schema, entity_class=_Vote)(registry, DefaultCoreConfig(), nats_client)


async def test_a_page_of_5000_keys_reaches_a_copy_as_one_message_and_one_arrow_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a write of a page of keys is broadcast as more than one message (one a row was a pydantic message a
    row decoded on every peer's loop, ~80 ms of it per 200,000 rows, 2026-11-25), or reaches a peer's copy as anything
    but one Arrow batch of exactly the page's keys, or if a peer whose only cache of the table is the copy builds a
    Python object a row to apply it. Over the JSON wire (the fake bus encodes and decodes every message)."""
    pa = pytest.importorskip("pyarrow")
    bus = FakeNatsClient()
    writer = _collection("votes", bus)
    receiver = CollectionRegistry()
    receiver.follow_generation("votes")
    batches: list[tuple[Any, str | None]] = []
    rows_to_copy: list[Any] = []
    receiver.register_derived_cache(
        "votes",
        on_row=rows_to_copy.append,
        on_table_dropped=lambda: None,
        on_partition=lambda _prefix: None,
        on_keys=lambda keys, epoch: batches.append((keys, epoch)),
    )
    expanded: list[int] = []
    real_each_row = CacheInvalidationMessage.each_row

    def each_row(message: CacheInvalidationMessage) -> list[CacheInvalidationMessage]:
        expanded.append(message.row_count)
        return real_each_row(message)

    monkeypatch.setattr(CacheInvalidationMessage, "each_row", each_row)
    await receiver.start_invalidation_listener(bus)
    page = [(f"S{n % 50:02d}", f"{n:08d}") for n in range(_PAGE)]
    await writer.invalidate_cache_many(page)
    await receiver.stop_invalidation_listener()

    assert len(bus.published) == 1, f"the page went as {len(bus.published)} messages"
    assert bus.published[0].row_count == _PAGE
    assert rows_to_copy == [], "the copy was handed a row message a key"
    assert len(batches) == 1, f"the copy was told {len(batches)} batches"
    keys, _epoch = batches[0]
    assert keys.equals(
        pa.table({"k0": pa.array([s for s, _ in page], pa.string()), "k1": pa.array([g for _, g in page], pa.string())})
    ), "the copy was told other keys than the page's"
    assert expanded == [], "a peer whose only cache of the table is the copy built a row message a row"


def test_the_wire_spells_every_key_as_a_row_message_does() -> None:
    """Flips if a key on the batched wire reads back, as Arrow or as a row's ``ids``, otherwise than the one-row
    message spells it (``key_text``): text with any character, an empty text, integers, an instant, a decimal's scale."""
    pa = pytest.importorskip("pyarrow")
    keys: list[tuple[Any, ...]] = [
        ("Zürich ☃", 12, datetime(2026, 11, 3, 21, 45, tzinfo=UTC), Decimal("1.50")),
        ("", -3, datetime(2026, 11, 4, 1, 0, 0, 5, tzinfo=UTC), Decimal("1.5")),
        ('a "quoted"\n\x00line', 0, datetime(2026, 11, 5, tzinfo=UTC), Decimal("0")),
    ]
    (rows,) = invalidation_batches(keys)
    wire = InvalidatedRows.model_validate_json(rows.model_dump_json())
    spelled = [[key_text(value) for value in key] for key in keys]
    assert row_ids(wire) == spelled
    batch = key_batch(wire)
    assert batch.column_names == ["k0", "k1", "k2", "k3"]
    assert [list(row.values()) for row in batch.to_pylist()] == spelled
    assert all(batch.column(at).type == pa.string() for at in range(4))


def test_a_write_past_the_message_budget_goes_as_few_messages_each_within_it() -> None:
    """Flips if a write too big for one message is cut into more messages than its size needs, into one past the
    budget, or loses or reorders a key on the way."""
    keys = [("x" * 40, f"{n:030d}") for n in range(50_000)]
    batches = invalidation_batches(keys)
    assert 2 <= len(batches) <= 8
    for rows in batches:
        assert sum(len(packed.data) + len(packed.offsets) for packed in rows.keys) <= _BYTES_A_MESSAGE
    assert [tuple(ids) for rows in batches for ids in row_ids(rows)] == keys


def test_rows_whose_columns_name_another_count_are_refused() -> None:
    """Flips if a batched message whose key offsets or invalidation columns name another number of rows than it
    says is read rather than refused as malformed."""
    (rows,) = invalidation_batches(["a", "b"], [{"c": "1"}, None])
    with pytest.raises(ValueError, match="offsets"):
        InvalidatedRows(count=3, keys=rows.keys, columns=None)
    with pytest.raises(ValueError, match="invalidation columns"):
        InvalidatedRows(count=2, keys=rows.keys, columns=[None])
