"""A bridged async store offers ``ResultReadings`` exactly when it offers ``AsyncResultReadings``, and passes the kit.

* **It conforms.** Every case of ``RESULT_READINGS_CONFORMANCE_CASES`` runs against an async store offering the async
  form, through ``bridge_async_store`` on a loop of its own thread. The kit's backend hook is installed on the
  in-memory store behind it, so "one backend call, made at the call" holds of the bridge: it starts the store's async
  iterator when ``iter_narrowed`` is called, not when the first row is asked for.
* **It streams.** A long narrowed read is taken from the store's async iterator a batch at a time, in order, in the
  caller's context; one abandoned part way closes the store's iterator.
* **Offered only when the store offers it**, and composed with the clock: all four (clock, readings) combinations
  bridge to exactly those capabilities, each working. A store offering the readings in the wrong shape is refused.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator, Callable, ItemsView, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.schema import AsyncDocumentStore, GroupMeasure, ResultReadings, StoreClock
from threetears.evals.storage import InMemoryDocumentStore, bridge_async_store
from threetears.evals.testing import RESULT_READINGS_CONFORMANCE_CASES, ResultReadingsConformanceCase

_STORE_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

#: What an adapter would scope by, set by the caller and read inside the store's iterator.
_TENANT: contextvars.ContextVar[str] = contextvars.ContextVar("tenant", default="unset")


class _HookedDocuments(dict[tuple[str, str], dict[str, Any]]):
    """A store's documents, calling the installed hook on each pass a read makes over them."""

    hook: Callable[[], None] | None = None

    def items(self) -> ItemsView[tuple[str, str], dict[str, Any]]:
        if self.hook is not None:
            self.hook()
        return super().items()


class _FakeAsyncStore(AsyncDocumentStore):
    """The in-memory reference store behind the async port, and nothing more."""

    def __init__(self) -> None:
        self.inner = InMemoryDocumentStore()

    async def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        return self.inner.get(doc_id, scope_id)

    async def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        return self.inner.get_with_etag(doc_id, scope_id)

    async def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return self.inner.get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)

    async def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        return self.inner.by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )

    async def iter_by_doc_type(self, doc_type: str, scope_id: str) -> AsyncIterator[str]:
        for doc_id in list(self.inner.iter_by_doc_type(doc_type, scope_id)):
            yield doc_id

    async def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        self.inner.upsert(document, if_match=if_match, create_only=create_only)

    async def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return self.inner.merge_fields(doc_id, scope_id, fields)

    async def delete(self, doc_id: str, scope_id: str) -> bool:
        return self.inner.delete(doc_id, scope_id)


# parity-with: threetears.evals.schema.AsyncStoreClock
class _FakeAsyncClock:
    """An async store's clock, mixed into a store below."""

    async def now(self) -> datetime:
        return _STORE_TIME


# parity-with: threetears.evals.schema.AsyncResultReadings
class _FakeAsyncReadings:
    """The in-memory store's readings, awaited, mixed into a store below; it records how its iterator was driven.

    :ivar pulled: Rows its latest narrowed read has yielded so far.
    :ivar finished: Whether that read's iterator has finished, run out or closed.
    :ivar tenants: The tenant each row was yielded under.
    """

    inner: InMemoryDocumentStore
    pulled: int = 0
    finished: bool = False
    tenants: list[str]

    async def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self.pulled, self.finished, self.tenants = 0, False, []
        try:
            for row in self.inner.iter_narrowed(
                doc_type, scope_id, exclude=exclude, exclude_in_items=exclude_in_items, field_in=field_in, **field_eq
            ):
                self.pulled += 1
                self.tenants.append(_TENANT.get())
                yield row
        finally:
            self.finished = True

    async def grouped(
        self,
        doc_type: str,
        scope_id: str,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        return self.inner.grouped(doc_type, scope_id, by=by, measures=measures, field_in=field_in, **field_eq)


class _FakeClockedAsyncStore(_FakeAsyncClock, _FakeAsyncStore):
    """An async store keeping a clock, without the readings."""


class _FakeReadingAsyncStore(_FakeAsyncReadings, _FakeAsyncStore):
    """An async store offering the readings, without a clock."""


class _FakeClockedReadingAsyncStore(_FakeAsyncClock, _FakeAsyncReadings, _FakeAsyncStore):
    """An async store offering the clock and the readings."""


class _FakeSyncGroupedStore(_FakeAsyncStore):
    """Readings in the wrong shape: ``grouped`` is not awaited."""

    async def iter_narrowed(self, doc_type: str, scope_id: str, **predicates: Any) -> AsyncIterator[dict[str, Any]]:
        for row in self.inner.iter_narrowed(doc_type, scope_id, **predicates):
            yield row

    def grouped(self, doc_type: str, scope_id: str, **arguments: Any) -> list[dict[str, Any]]:
        return self.inner.grouped(doc_type, scope_id, **arguments)


class _FakeAwaitedNarrowedStore(_FakeAsyncStore):
    """Readings in the wrong shape: ``iter_narrowed`` is awaited for a list, not iterated."""

    async def iter_narrowed(self, doc_type: str, scope_id: str, **predicates: Any) -> list[dict[str, Any]]:
        return list(self.inner.iter_narrowed(doc_type, scope_id, **predicates))

    async def grouped(self, doc_type: str, scope_id: str, **arguments: Any) -> list[dict[str, Any]]:
        return self.inner.grouped(doc_type, scope_id, **arguments)


class _FakeSyncNarrowedStore(_FakeAsyncStore):
    """Readings in the wrong shape, seen only at the call: ``iter_narrowed`` gives a synchronous iterator."""

    def iter_narrowed(self, doc_type: str, scope_id: str, **predicates: Any) -> Iterator[dict[str, Any]]:
        return self.inner.iter_narrowed(doc_type, scope_id, **predicates)

    async def grouped(self, doc_type: str, scope_id: str, **arguments: Any) -> list[dict[str, Any]]:
        return self.inner.grouped(doc_type, scope_id, **arguments)


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A running loop on a thread of its own, the async store's home."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


def _installer(store: _FakeAsyncStore) -> Callable[[Callable[[], None]], None]:
    """Hook the in-memory store behind ``store``, and return the installer the kit calls."""
    documents = _HookedDocuments(store.inner.documents)
    store.inner.documents = documents

    def install(hook: Callable[[], None]) -> None:
        documents.hook = hook

    return install


def _doc(index: int) -> dict[str, Any]:
    return {"id": f"d{index:04d}", "scope_id": "s", "doc_type": "t", "n": index, "scores": [{"reasoning": "x"}]}


# --- it conforms ----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_a_bridged_async_store_offering_readings_conforms(
    case: ResultReadingsConformanceCase, loop: asyncio.AbstractEventLoop
) -> None:
    store = _FakeReadingAsyncStore()
    case.run(bridge_async_store(store, loop), _installer(store))


# --- it streams -------------------------------------------------------------------------------------


def test_a_long_narrowed_read_is_streamed_in_order(loop: asyncio.AbstractEventLoop) -> None:
    store = _FakeReadingAsyncStore()
    for index in reversed(range(1000)):
        store.inner.upsert(_doc(index))
    bridge = bridge_async_store(store, loop)
    assert isinstance(bridge, ResultReadings)

    rows = bridge.iter_narrowed("t", "s", exclude_in_items={"scores": ["reasoning"]})
    assert 0 < store.pulled < 1000, "the call started the store's iterator, and did not drain it"
    narrowed = list(rows)
    assert [row["n"] for row in narrowed] == list(reversed(range(1000)))
    assert all(row["scores"] == [{}] for row in narrowed)
    assert store.pulled == 1000 and store.finished


def test_every_batch_is_taken_in_the_callers_context(loop: asyncio.AbstractEventLoop) -> None:
    store = _FakeReadingAsyncStore()
    for index in range(1000):
        store.inner.upsert(_doc(index))
    bridge = bridge_async_store(store, loop)
    assert isinstance(bridge, ResultReadings)
    token = _TENANT.set("tenant-7")
    try:
        assert len(list(bridge.iter_narrowed("t", "s"))) == 1000
    finally:
        _TENANT.reset(token)
    assert set(store.tenants) == {"tenant-7"}


def test_a_read_abandoned_part_way_closes_the_stores_iterator(loop: asyncio.AbstractEventLoop) -> None:
    store = _FakeReadingAsyncStore()
    for index in range(1000):
        store.inner.upsert(_doc(index))
    bridge = bridge_async_store(store, loop)
    assert isinstance(bridge, ResultReadings)

    rows = bridge.iter_narrowed("t", "s")
    assert next(rows)["n"] == 0
    assert not store.finished
    rows.close()
    assert store.finished and store.pulled < 1000
    assert len(list(bridge.iter_narrowed("t", "s"))) == 1000, "the store still reads after the abandoned read"


# --- offered only when the store offers it, and composed with the clock ----------------------------------


_STORES: dict[tuple[bool, bool], type[_FakeAsyncStore]] = {
    (False, False): _FakeAsyncStore,
    (True, False): _FakeClockedAsyncStore,
    (False, True): _FakeReadingAsyncStore,
    (True, True): _FakeClockedReadingAsyncStore,
}


@pytest.mark.parametrize("clock", [False, True], ids=["no-clock", "clock"])
@pytest.mark.parametrize("readings", [False, True], ids=["no-readings", "readings"])
def test_each_clock_and_readings_combination_bridges_to_exactly_those(
    clock: bool, readings: bool, loop: asyncio.AbstractEventLoop
) -> None:
    store = _STORES[clock, readings]()
    store.inner.upsert({"id": "r1", "scope_id": "s", "doc_type": "t", "cost": 4, "heavy": "x"})
    bridge = bridge_async_store(store, loop)
    assert isinstance(bridge, StoreClock) is clock
    assert isinstance(bridge, ResultReadings) is readings
    if not readings:
        assert not hasattr(bridge, "iter_narrowed") and not hasattr(bridge, "grouped"), "no method that raises"
    if isinstance(bridge, StoreClock):
        assert bridge.now() == _STORE_TIME
    if isinstance(bridge, ResultReadings):
        assert list(bridge.iter_narrowed("t", "s", exclude=["heavy"])) == [
            {"id": "r1", "scope_id": "s", "doc_type": "t", "cost": 4}
        ]
        assert bridge.grouped("t", "s", by=[], measures={"cost": ("sum_int", "cost")}) == [{"cost": 4}]


@pytest.mark.parametrize(
    "store_type", [_FakeSyncGroupedStore, _FakeAwaitedNarrowedStore], ids=["sync-grouped", "awaited-narrowed"]
)
def test_readings_in_the_wrong_shape_are_refused_when_bridged(
    store_type: type[_FakeAsyncStore], loop: asyncio.AbstractEventLoop
) -> None:
    with pytest.raises(TypeError, match="wrong shape"):
        bridge_async_store(store_type(), loop)


def test_a_narrowed_read_that_gives_no_async_iterator_is_refused_at_the_call(loop: asyncio.AbstractEventLoop) -> None:
    store = _FakeSyncNarrowedStore()
    store.inner.upsert(_doc(0))
    bridge = bridge_async_store(store, loop)
    assert isinstance(bridge, ResultReadings)
    with pytest.raises(TypeError, match="not an async iterator"):
        bridge.iter_narrowed("t", "s")
