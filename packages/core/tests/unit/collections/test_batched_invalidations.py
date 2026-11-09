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

        async def apply(message: CacheInvalidationMessage) -> None:
            self.applied.append((message.table, tuple(message.ids), message.l2_current_scope, message.columns))

        monkeypatch.setattr(registry, "_apply_row_message", apply)
        marks = registry._generation_marks  # noqa: SLF001 - what the listener counts toward
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
    await receiver.start_invalidation_listener(bus)  # type: ignore[arg-type]
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
    await sender.publish_invalidations(bus, "results", messages)  # type: ignore[arg-type]
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
    """Flips if a registry evicts the rows of its own batched broadcast, as it never does its own single ones."""
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.batch_invalidations = True
    applied: list[Any] = []

    async def apply(message: CacheInvalidationMessage) -> None:
        applied.append(message)

    registry._apply_row_message = apply  # type: ignore[method-assign]  # noqa: SLF001
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    await registry.publish_invalidations(
        bus, "results", [registry.row_message("results", ("a", "b")) for _ in range(3)]
    )  # type: ignore[arg-type]
    await registry.stop_invalidation_listener()
    assert applied == []
