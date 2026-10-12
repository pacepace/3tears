"""An async store, bridged, is a store the engine runs on; and no store is ever called on the event loop.

* **It conforms.** Every case of ``STORE_CONFORMANCE_CASES`` runs against an async store through
  ``bridge_async_store``, on a loop of its own thread, the way the kit's docstring tells an adopter to.
* **The caller's context reaches the store**, as a context variable an adapter scopes by must.
* **A call on a loop thread is refused**, by the bridge and by ``EvalStorage`` over any store, rather than
  stalling the loop or waiting on itself; the same call made through ``run_blocking`` lands.
* **The clock crosses exactly when the store keeps one**, and a synchronous ``now`` on an async store is refused.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.kernel import EvalStorage
from threetears.evals.run import run_blocking
from threetears.evals.schema import (
    AsyncDocumentStore,
    StoreCalledOnEventLoop,
    StoreClock,
    StoreConflict,
)
from threetears.evals.storage import InMemoryDocumentStore, bridge_async_store
from threetears.evals.testing import STORE_CONFORMANCE_CASES, StoreConformanceCase

#: What an adapter would scope by — a tenant, say — set by the caller and read inside the store.
_TENANT: contextvars.ContextVar[str] = contextvars.ContextVar("tenant", default="unset")

_STORE_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class _FakeAsyncStore(AsyncDocumentStore):
    """The in-memory reference store behind the async port, recording the tenant each call ran under."""

    def __init__(self) -> None:
        self.inner = InMemoryDocumentStore()
        self.tenants: list[str] = []

    def _seen(self) -> None:
        self.tenants.append(_TENANT.get())

    async def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        self._seen()
        return self.inner.get(doc_id, scope_id)

    async def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        self._seen()
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
        self._seen()
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
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        self._seen()
        return self.inner.by_doc_type(
            doc_type, scope_id, order_by=order_by, descending=descending, limit=limit, exclude=exclude, **field_eq
        )

    async def iter_by_doc_type(self, doc_type: str, scope_id: str) -> AsyncIterator[str]:
        self._seen()
        for doc_id in list(self.inner.iter_by_doc_type(doc_type, scope_id)):
            yield doc_id

    async def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        self._seen()
        self.inner.upsert(document, if_match=if_match, create_only=create_only)

    async def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        self._seen()
        return self.inner.merge_fields(doc_id, scope_id, fields)

    async def delete(self, doc_id: str, scope_id: str) -> bool:
        self._seen()
        return self.inner.delete(doc_id, scope_id)


class _FakeClockedAsyncStore(_FakeAsyncStore):
    """An async store that keeps a clock of its own."""

    async def now(self) -> datetime:
        return _STORE_TIME


class _FakeSyncClockAsyncStore(_FakeAsyncStore):
    """An async store whose ``now`` is not awaited: the shape the bridge refuses."""

    def now(self) -> datetime:
        return _STORE_TIME


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


def _doc(doc_id: str, **fields: object) -> dict[str, object]:
    return {"id": doc_id, "scope_id": "scope-a", "doc_type": "eval_run", **fields}


@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_a_bridged_async_store_conforms(case: StoreConformanceCase, loop: asyncio.AbstractEventLoop) -> None:
    case.run(bridge_async_store(_FakeAsyncStore(), loop))


def test_the_callers_context_reaches_the_store(loop: asyncio.AbstractEventLoop) -> None:
    store = _FakeAsyncStore()
    bridged = bridge_async_store(store, loop)
    token = _TENANT.set("tenant-7")
    try:
        bridged.upsert(_doc("r1"))
        assert bridged.get("r1", "scope-a") == _doc("r1")
        assert list(bridged.iter_by_doc_type("eval_run", "scope-a")) == ["r1"]
    finally:
        _TENANT.reset(token)
    assert store.tenants == ["tenant-7", "tenant-7", "tenant-7"]


def test_what_the_store_raises_reaches_the_caller_unchanged(loop: asyncio.AbstractEventLoop) -> None:
    bridged = bridge_async_store(_FakeAsyncStore(), loop)
    bridged.upsert(_doc("r1", n=1))
    _, etag = bridged.get_with_etag("r1", "scope-a")
    bridged.upsert(_doc("r1", n=2))
    with pytest.raises(StoreConflict):
        bridged.upsert(_doc("r1", n=3), if_match=etag)


async def test_a_bridged_call_on_its_own_loop_is_refused_rather_than_waiting_on_itself() -> None:
    bridged = bridge_async_store(_FakeAsyncStore(), asyncio.get_running_loop())
    with pytest.raises(StoreCalledOnEventLoop):
        bridged.get("r1", "scope-a")


async def test_the_same_call_through_run_blocking_lands() -> None:
    store = _FakeAsyncStore()
    bridged = bridge_async_store(store, asyncio.get_running_loop())
    await run_blocking(None, bridged.upsert, _doc("r1"))
    assert await run_blocking(None, bridged.get, "r1", "scope-a") == _doc("r1")


def test_a_loop_that_is_not_running_is_named_rather_than_waited_on() -> None:
    stopped = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="is not running"):
            bridge_async_store(_FakeAsyncStore(), stopped).get("r1", "scope-a")
    finally:
        stopped.close()


def test_the_clock_crosses_when_the_store_keeps_one(loop: asyncio.AbstractEventLoop) -> None:
    bridged = bridge_async_store(_FakeClockedAsyncStore(), loop)
    assert isinstance(bridged, StoreClock)
    assert bridged.now() == _STORE_TIME
    assert EvalStorage(bridged).store_time() == _STORE_TIME


def test_no_clock_crosses_when_the_store_keeps_none(loop: asyncio.AbstractEventLoop) -> None:
    assert not isinstance(bridge_async_store(_FakeAsyncStore(), loop), StoreClock)


def test_a_synchronous_clock_on_an_async_store_is_refused(loop: asyncio.AbstractEventLoop) -> None:
    with pytest.raises(TypeError, match="not a coroutine function"):
        bridge_async_store(_FakeSyncClockAsyncStore(), loop)


async def test_eval_storage_refuses_a_call_on_the_event_loop_over_any_store() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    with pytest.raises(StoreCalledOnEventLoop):
        storage.store_time()
    with pytest.raises(StoreCalledOnEventLoop):
        storage.query_eval_runs("scope-a")


def _a_sync_helper(storage: EvalStorage) -> None:
    """A synchronous helper that hides the store call from the async code above it."""
    storage.store_time()


async def test_the_refusal_names_the_call_site_outside_the_engine() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    with pytest.raises(StoreCalledOnEventLoop) as refused:
        storage.query_eval_runs("scope-a")
    assert "test_the_refusal_names_the_call_site_outside_the_engine" in str(refused.value)
    assert __file__ in str(refused.value)


async def test_the_refusal_names_a_sync_helper_rather_than_the_code_awaiting_it() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    with pytest.raises(StoreCalledOnEventLoop, match=r"\._a_sync_helper \("):
        _a_sync_helper(storage)


async def test_eval_storage_over_a_bridged_store_runs_off_the_loop() -> None:
    store = _FakeAsyncStore()
    storage = EvalStorage(bridge_async_store(store, asyncio.get_running_loop()))
    token = _TENANT.set("tenant-9")
    try:
        assert await run_blocking(None, storage.query_eval_runs, "scope-a") == []
    finally:
        _TENANT.reset(token)
    assert store.tenants == ["tenant-9"]
