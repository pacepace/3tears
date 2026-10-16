"""An :class:`~threetears.evals.schema.AsyncDocumentStore` presented as the engine's synchronous port.

A host whose database driver is async (asyncpg, an async cloud SDK) writes its adapter against that
driver, as an :class:`~threetears.evals.schema.AsyncDocumentStore`, and hands the engine
``bridge_async_store(adapter, loop)``. The engine calls the bridge as it calls any
:class:`~threetears.evals.schema.DocumentStore` — from its blocking-I/O executor, off the loop — and
each call is run as a task on ``loop``, the loop the adapter's connections belong to, while the calling
worker thread waits for its answer.

**Three things a hand-written bridge gets wrong, kept here once:**

- **The caller's context goes with the call.** ``asyncio.run_coroutine_threadsafe`` runs the coroutine
  in the loop thread's context, not the caller's, so a context variable the adapter reads — a tenant id
  its row-level security scopes by, the request id its traces correlate on — would read its default.
  The engine copies its context onto the executor thread
  (:func:`~threetears.evals.kernel.offload.run_blocking`); the bridge carries it on into the task.
- **A call on a loop thread is refused**, before anything is scheduled
  (:func:`~threetears.evals.schema.refuse_on_event_loop`). Waiting on ``loop`` from ``loop``'s own thread
  never returns, and waiting from another loop's thread stalls that loop for the round trip.
- **A loop that is not running is named**, rather than waited on: a call scheduled on a stopped loop never
  starts, and its caller would wait forever.

**The clock comes across too.** A store offering :class:`~threetears.evals.schema.AsyncStoreClock` is
bridged as one offering :class:`~threetears.evals.schema.StoreClock`, so job leases are stamped in the
database's time; one that does not is bridged without it, so the engine reads its own clock rather than
being told the store keeps one.

``iter_by_doc_type`` is drained on the loop and handed back as a list's iterator: its one caller deletes
what it yields, and a generator held open across threads would hold its cursor while it does.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
from collections.abc import Callable, Coroutine, Iterator, Sequence
from datetime import datetime
from functools import partial
from typing import Any

from threetears.evals.schema.store_port import AsyncDocumentStore, AsyncStoreClock, refuse_on_event_loop

__all__ = ["AsyncStoreBridge", "bridge_async_store"]


class AsyncStoreBridge:
    """A :class:`~threetears.evals.schema.DocumentStore` whose every call runs on an async store's loop.

    Built by :func:`bridge_async_store`, which also decides whether it offers the store's clock.

    :ivar store: The async store each call is made on.
    :ivar loop: The loop each call runs on: the one the store's connections belong to.
    """

    def __init__(self, store: AsyncDocumentStore, loop: asyncio.AbstractEventLoop) -> None:
        """Bridge ``store``, whose calls run on ``loop``.

        :param store: The async store.
        :ptype store: AsyncDocumentStore
        :param loop: The loop its calls run on.
        :ptype loop: asyncio.AbstractEventLoop
        """
        self.store = store
        self.loop = loop

    def _call[T](self, make: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Run the coroutine ``make`` builds as a task on :attr:`loop`, in the caller's context, and wait for it.

        :param make: Builds the store call's coroutine; called on the loop, inside the task.
        :ptype make: Callable[[], Coroutine[Any, Any, T]]
        :return: What the coroutine returned.
        :rtype: T
        :raises StoreCalledOnEventLoop: The calling thread is running an event loop.
        :raises RuntimeError: :attr:`loop` is not running, so the call would never start.
        :raises concurrent.futures.CancelledError: The task was cancelled on its loop before it answered.
        :raises Exception: Whatever the store raised, unchanged — a :class:`~threetears.evals.schema.StoreConflict`
            included.
        """
        refuse_on_event_loop()
        if not self.loop.is_running():
            raise RuntimeError(
                f"the async store's loop {self.loop!r} is not running, so a store call would never start"
            )
        context = contextvars.copy_context()
        answer: concurrent.futures.Future[T] = concurrent.futures.Future()

        def settle(task: asyncio.Task[T]) -> None:
            if task.cancelled():
                answer.set_exception(concurrent.futures.CancelledError("the store call was cancelled on its loop"))
                return
            error = task.exception()
            if error is not None:
                answer.set_exception(error)
                return
            answer.set_result(task.result())

        def start() -> None:
            if not answer.set_running_or_notify_cancel():
                return
            self.loop.create_task(make(), context=context).add_done_callback(settle)

        self.loop.call_soon_threadsafe(start)
        return answer.result()

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """See :meth:`~threetears.evals.schema.DocumentStore.get`."""
        return self._call(partial(self.store.get, doc_id, scope_id))

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """See :meth:`~threetears.evals.schema.DocumentStore.get_with_etag`."""
        return self._call(partial(self.store.get_with_etag, doc_id, scope_id))

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """See :meth:`~threetears.evals.schema.DocumentStore.get_many`."""
        return self._call(partial(self.store.get_many, doc_type, doc_ids, scope_id, exclude=exclude, keep=keep))

    def by_doc_type(
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
        """See :meth:`~threetears.evals.schema.DocumentStore.by_doc_type`."""
        return self._call(
            partial(
                self.store.by_doc_type,
                doc_type,
                scope_id,
                order_by=order_by,
                descending=descending,
                limit=limit,
                exclude=exclude,
                **field_eq,
            )
        )

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """See :meth:`~threetears.evals.schema.DocumentStore.iter_by_doc_type`; drained on the loop, then iterated."""

        async def drained() -> list[str]:
            return [doc_id async for doc_id in self.store.iter_by_doc_type(doc_type, scope_id)]

        return iter(self._call(drained))

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """See :meth:`~threetears.evals.schema.DocumentStore.upsert`."""
        self._call(partial(self.store.upsert, document, if_match=if_match, create_only=create_only))

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """See :meth:`~threetears.evals.schema.DocumentStore.merge_fields`."""
        return self._call(partial(self.store.merge_fields, doc_id, scope_id, fields))

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """See :meth:`~threetears.evals.schema.DocumentStore.delete`."""
        return self._call(partial(self.store.delete, doc_id, scope_id))


class ClockedAsyncStoreBridge(AsyncStoreBridge):
    """An :class:`AsyncStoreBridge` over a store that keeps a clock, offering it as :class:`StoreClock`."""

    def now(self) -> datetime:
        """See :meth:`~threetears.evals.schema.StoreClock.now`: the async store's ``now``, on its loop."""
        clock = self.store
        if not isinstance(clock, AsyncStoreClock):
            raise TypeError(f"{type(clock).__name__} offers no clock, so its bridge offers none")
        return self._call(clock.now)


def bridge_async_store(store: AsyncDocumentStore, loop: asyncio.AbstractEventLoop) -> AsyncStoreBridge:
    """Present ``store`` as the engine's synchronous :class:`~threetears.evals.schema.DocumentStore`.

    Hand the result to :class:`~threetears.evals.kernel.EvalStorage` as any store. Build it with the loop the
    store's connections belong to — in an async app, the running loop at startup
    (``asyncio.get_running_loop()``) — and keep the engine's blocking executor a thread pool, so its calls
    arrive off that loop.

    :param store: The async adapter.
    :ptype store: AsyncDocumentStore
    :param loop: The loop its calls run on.
    :ptype loop: asyncio.AbstractEventLoop
    :return: The bridge: one that offers :class:`~threetears.evals.schema.StoreClock` exactly when ``store`` offers
        :class:`~threetears.evals.schema.AsyncStoreClock`.
    :rtype: AsyncStoreBridge
    :raises TypeError: ``store`` has a ``now`` that is not a coroutine function: an async store's clock is awaited,
        and a synchronous one would be the calling process's time passed off as the database's.
    """
    if isinstance(store, AsyncStoreClock):
        if not inspect.iscoroutinefunction(store.now):
            raise TypeError(
                f"{type(store).__name__}.now is not a coroutine function; an async store's clock is awaited "
                "(AsyncStoreClock), and a store with no clock of its own offers no now at all"
            )
        return ClockedAsyncStoreBridge(store, loop)
    return AsyncStoreBridge(store, loop)
