"""A write of many rows may be broadcast as one message naming them all: heard exactly as one message a row."""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.testing.kv import FakeNatsClient

_ROWS = 1_200


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

        def hear(owner: Any, table: str, generation: str, rows: int) -> None:
            if owner is marks:
                self.counted.append((table, generation, rows))
            real_hear(owner, table, generation, rows)

        monkeypatch.setattr(type(marks), "hear", hear)


async def _broadcast(batched: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[_Heard, int]:
    bus = FakeNatsClient()
    sender, receiver = CollectionRegistry(), CollectionRegistry()
    heard = _Heard(receiver, monkeypatch)
    await receiver.start_invalidation_listener(bus)
    sender.batch_invalidations = batched
    published: list[Any] = []
    real_publish = bus.publish

    async def counting(*args: Any, **kwargs: Any) -> Any:
        published.append(kwargs)
        return await real_publish(*args, **kwargs)

    monkeypatch.setattr(bus, "publish", counting)
    messages = [
        sender.row_message(
            "results", (f"r{n % 7}", f"g{n}"), generation="inc:9", bump_rows=_ROWS, columns={"state": f"S{n % 3}"}
        )
        for n in range(_ROWS)
    ]
    await sender.publish_invalidations(bus, "results", messages)
    await receiver.stop_invalidation_listener()
    return heard, len(published)


async def test_a_batch_is_heard_row_for_row_as_the_singles_are(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a batched broadcast evicts other rows, in another order, or counts its advance otherwise than
    one message a row would."""
    singles, single_publishes = await _broadcast(False, monkeypatch)
    batched, batch_publishes = await _broadcast(True, monkeypatch)

    assert single_publishes == _ROWS
    assert batch_publishes == 3
    assert batched.applied == singles.applied
    assert len(batched.applied) == _ROWS
    assert batched.counted == singles.counted == [("results", "inc:9", _ROWS)] * _ROWS


async def test_a_batch_from_this_registry_is_not_applied_here() -> None:
    """Flips if a registry applies the rows of its own batched broadcast on hearing it, as it never does its own
    single ones: a derived cache is told of each of its own rows once, by the write, and not again by the echo."""
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.batch_invalidations = True
    applied: list[Any] = []
    registry.register_derived_cache(
        "results", on_row=applied.append, on_table_dropped=lambda: None, on_partition=lambda _prefix: None
    )
    await registry.start_invalidation_listener(bus)
    messages = [registry.row_message("results", ("a", "b")) for _ in range(3)]
    await registry.publish_invalidations(bus, "results", messages)
    await registry.stop_invalidation_listener()
    assert len(applied) == 3


async def test_a_batch_reaches_a_cache_taking_key_batches_as_one_arrow_batch_a_message() -> None:
    """Flips if a batched broadcast reaches a cache that takes key batches (a copy, ``on_keys``) as a row message a
    key (a Python object a key in the copy, decision 35) instead of one Arrow batch a message at its advance, or if
    a cache taking only rows stops hearing each of them."""
    pa = pytest.importorskip("pyarrow")
    bus = FakeNatsClient()
    sender, receiver = CollectionRegistry(), CollectionRegistry()
    sender.batch_invalidations = True
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
    messages = [
        sender.row_message("results", (f"r{n % 7}", f"g{n}"), generation="inc:9", bump_rows=_ROWS) for n in range(_ROWS)
    ]
    await sender.publish_invalidations(bus, "results", messages)
    await receiver.stop_invalidation_listener()

    assert copy_rows == [], "no row message a key to a cache that takes batches"
    assert len(batches) == 3 and all(isinstance(keys, pa.Table) and epoch == "inc:9" for keys, epoch in batches)
    heard = pa.concat_tables([keys for keys, _ in batches])
    assert list(zip(heard.column("k0").to_pylist(), heard.column("k1").to_pylist(), strict=True)) == [
        (f"r{n % 7}", f"g{n}") for n in range(_ROWS)
    ]
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
    sender.batch_invalidations = batched
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
    # the catch-up applied epoch inc:9 from its record before its broadcast landed
    receiver.generation_marks.applied("results", "inc:9")
    told = [sender.row_message("results", ("r", f"g{n}"), generation="inc:9", bump_rows=2) for n in range(2)]
    await sender.publish_invalidations(bus, "results", told)
    assert copy == [], "a broadcast of an advance applied from its record was told to the copy again"
    assert len(rows) == 2
    fresh = [sender.row_message("results", ("r", f"h{n}"), generation="inc:10", bump_rows=2) for n in range(2)]
    await sender.publish_invalidations(bus, "results", fresh)
    await receiver.stop_invalidation_listener()
    assert copy, "a broadcast of an advance not yet applied did not reach the copy"
