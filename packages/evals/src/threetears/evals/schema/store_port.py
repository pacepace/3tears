"""The storage seam the eval core depends on — a narrow document store.

The engine stores through one port and names no backend: this module is that port. It
deliberately imports nothing of the engine — not the host, not the eval models, not even
:mod:`threetears.evals.kernel.storage` — so an adapter implementing it depends on nothing else.

**One store, every eval document.** A :class:`DocumentStore` addresses one
collection of JSON documents keyed by ``(scope_id, doc_type, id)``:

* ``scope_id`` is the **scope** — the partition a document lives in, carried on every
  document as its own ``scope_id`` field. The core carries it and never interprets it;
  the adapter maps it onto whatever the backing store partitions and isolates by, and
  how that isolation is enforced is the adapter's business.
* ``doc_type`` is the core's own discriminator, since several document kinds
  co-tenant one collection. Routing a kind to a table of its own is the adapter's
  business too.
* ``id`` is the document's own identity, unique within its scope.

**There is no scope-free read.** Every method names the scope it reads, writes or
deletes in, or — for :meth:`DocumentStore.upsert` — takes it from the document. A
caller that must reach several scopes (a boot-time reclaim, an operator wipe) is told
which ones by its host and asks each in turn; "every scope this connection can see"
is a property of one backend's isolation, and the engine does not lean on it.

**Reads return stripped documents.** Every method that yields a document yields
one the core can hand straight to a model constructor: the adapter has already
removed whatever it injected on write — an etag, a timestamp, a column its isolation
mechanism stamps. The strip is load-bearing rather than defensive: every stored eval
model reads strictly and refuses a key it does not declare, so an adapter that returned
raw rows would make every read raise.

That contract is why :meth:`DocumentStore.get_with_etag` exists as a separate
method. The etag is stripped like everything else, so the one caller that needs
it — a read-modify-write under optimistic concurrency — has to ask for it.

**Errors.** A miss is a normal outcome, never an exception: ``get`` returns
``None``, the list reads return empty, ``delete`` returns ``False``. A genuine
backend failure raises, and the core decides what to do about it (
:meth:`threetears.evals.kernel.storage.EvalStorage._save` raises it on as a typed eval
error; :func:`threetears.evals.run.run_document.update_eval_run` retries). An implementation
that collapses a failure into a miss would let an outage read as "nothing stored".

A lost conditional write is the one failure the port names: an implementation raises
:class:`StoreConflict` for it, because a lost race has a remedy the core owns (re-read
and re-apply) and every other failure does not.

**Two preconditions, and only two.** A write may be conditional on the document still carrying an
etag (``if_match``: it was read, and nothing wrote since) or on there being no document at all
(``create_only``: nothing is stored under its ``(scope_id, id)``, of any type). The second is how two
processes racing to create one deterministic document learn which of them won: without it, two
unconditional creates both land and the later silently replaces the earlier. ``if_match`` never means
"absent": a conditional write to a document that is not stored is refused.

**An optional clock.** A store that keeps a clock of its own may also offer :class:`StoreClock` — its
``now()``. Job leases (:mod:`~threetears.evals.schema.job_lease`) are stamped and compared in that time when it
is offered, so processes whose clocks disagree still agree on whether a lease has lapsed; a store that does not
offer it is read on each process's own clock, and the processes sharing it must keep their clocks in step. It
is a separate protocol rather than a method of :class:`DocumentStore`, because a store that cannot tell its own
time honestly must not be made to pretend: the engine asks ``isinstance(store, StoreClock)``.

**Never on the event loop.** The port is synchronous and the engine is async: every call the engine
makes on a store runs on its blocking-I/O executor, off the loop. The engine's repository over the
store (:class:`~threetears.evals.kernel.EvalStorage`) refuses a call made on a thread running an event
loop (:class:`StoreCalledOnEventLoop`), so a path that forgot to offload fails at once, by name, rather
than stalling the loop under a synchronous store or deadlocking an async one behind
:func:`~threetears.evals.storage.bridge_async_store`. An adapter can call :func:`refuse_on_event_loop`
itself to hold its own callers to the same rule.

**Proving an adapter.** Every rule above is a case in the store conformance kit,
``threetears.evals.testing.STORE_CONFORMANCE_CASES``: an adapter's own test suite hands each case
a fresh, empty store, and a case that fails names the rule it broke. The engine's in-memory
adapter, ``threetears.evals.storage.InMemoryDocumentStore``, passes every case and is the shape to
compare a real adapter against.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


class StoreConflict(Exception):
    """A conditional write lost its race: the stored document no longer carries ``if_match``.

    Distinct from every other write failure because the caller can do something about
    it — read the winner's document and apply its change again — and because the stored
    document is intact, which an outage does not promise.
    """


class StoreCalledOnEventLoop(RuntimeError):
    """A store call was made on a thread that is running an event loop.

    The document store is synchronous, and the engine calls it from a blocking-I/O executor
    (:func:`~threetears.evals.kernel.offload.run_blocking`), never on the loop: a synchronous store
    called there stalls every coroutine the loop runs until the call returns, and a store bridged from
    an async one (:func:`~threetears.evals.storage.bridge_async_store`) waits on the very loop it is
    blocking, which never returns. So the call is refused before it reaches the store.

    A caller in async code offloads the call, ``await run_blocking(host.blocking_executor, fn, ...)``, as the
    action catalogue does for every synchronous operation in :mod:`threetears.evals.ops`; an async operation there
    (a launch, a job's poll) offloads its own.
    """


def refuse_on_event_loop() -> None:
    """Raise :class:`StoreCalledOnEventLoop` when the calling thread is running an event loop.

    Raises:
        StoreCalledOnEventLoop: It is.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # NOSILENT: the answer, not a failure: no loop runs on this thread (an executor's worker, or synchronous
        # code), so the store may be called here.
        return
    raise StoreCalledOnEventLoop(
        f"a document store was called on the thread running event loop {loop!r}; a store call blocks, so it runs "
        "off the loop: await run_blocking(host.blocking_executor, fn, ...) runs it on the host's executor"
    )


def omit_paths(document: dict[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """Return ``document`` without each dotted path in ``paths`` — the meaning of ``exclude``.

    A path names one leaf: ``"a.b.c"`` removes key ``c`` from ``document["a"]["b"]`` and keeps
    every sibling. A path that is absent, or runs through something that is not an object, is a
    no-op rather than an error, because the stores this port abstracts behave that way (a SQL
    JSON path-delete does) and a bulk read must not fail on a document whose payload never had the value.

    Defined here, beside the port, so every implementation that cannot push the projection into
    its backend applies the one definition rather than its own.

    Args:
        document: The document. Not mutated.
        paths: Dotted paths to drop.

    Returns:
        A copy, sharing every subtree the paths do not pass through.
    """
    result = dict(document)
    for path in paths:
        *parents, leaf = path.split(".")
        node = result
        for key in parents:
            child = node.get(key)
            if not isinstance(child, dict):
                break
            node[key] = dict(child)
            node = node[key]
        else:
            node.pop(leaf, None)
    return result


def keep_fields(document: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Return ``document`` reduced to the top-level ``fields`` it has — the meaning of ``keep``.

    A field the document does not carry stays absent rather than arriving as ``None``, so a
    reader's own default for a missing field still applies. Defined beside the port for the
    reason :func:`omit_paths` is.

    Args:
        document: The document. Not mutated.
        fields: The top-level fields to keep.

    Returns:
        A new dict holding only those fields.
    """
    return {key: document[key] for key in fields if key in document}


class DocumentStore(Protocol):
    """A narrow document store over one ``(scope_id, doc_type, id)``-keyed collection.

    See the module docstring for the scope model, the strip-on-read contract, and
    the error contract — all three are properties of *every* implementation, not
    of any one method.
    """

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """Return one document by id within a scope, or ``None`` if absent.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            The stored document with storage-injected fields removed, or ``None``.
        """
        ...

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return one document plus the token a conditional write must present.

        The read half of optimistic concurrency: pass the returned token back as
        :meth:`upsert`'s ``if_match`` and the write lands only if nothing else
        wrote in between.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Every store implements conditional writes: a found document always comes
        back with a token, and every write of it — whole or merged — mints a new
        one. A store that could not would turn the engine's read-modify-writes into
        blind overwrites, which lose the other writer's change without a trace;
        several writers share a run document as it finishes, so that is the
        ordinary case rather than an edge.

        Returns:
            ``(document, etag)``, or ``(None, None)`` when there is no such
            document.
        """
        ...

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Return the documents of one type whose ids are in ``doc_ids``, within a scope.

        Silently skips ids that are absent — the batch form of a miss being a
        normal outcome. ``doc_type`` is part of the key, so an id that exists in
        the scope under a different type is not returned. An id named twice is
        returned once.

        A projected document carries no record of the projection: one with a path
        left out is indistinguishable from one stored without it. So the caller that
        asks for ``exclude`` or ``keep`` owns saying so wherever the document goes on
        as a model that could pass for the whole one — the eval run reads mark each
        run they hydrate, and a host implementing ``load_eval_runs`` itself carries
        the same obligation (see :class:`~threetears.evals.run.curation.CurationStore`).

        Args:
            doc_type: The discriminator every returned document carries.
            doc_ids: The ids to fetch. An empty sequence returns ``[]`` and asks the
                backend nothing.
            scope_id: The scope to read.
            exclude: Dotted paths the store leaves out of every returned document, with
                the meaning :func:`omit_paths` gives them — the batch form of
                :meth:`by_doc_type`'s ``exclude``.
            keep: The narrow alternative to ``exclude``: when given, each returned
                document carries only these top-level fields — those it has, with the
                meaning :func:`keep_fields` gives them — for a reader of a few scalars
                off many large documents. Never both.

        Returns:
            The matching documents, in no guaranteed order.

        Raises:
            ValueError: Both ``exclude`` and ``keep`` were given.
        """
        ...

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
        """Return documents of one type within a scope, matching ANDed equality predicates.

        Args:
            doc_type: The discriminator to select on.
            scope_id: The scope to read.
            order_by: Document field to sort by; ``None`` leaves the order to the
                store. Sorting is by the field's stored type, so a number orders
                numerically and an ISO-8601 timestamp lexicographically. A document
                without the field, or with it null, sorts below every document that
                has it — first ascending, last descending.
            descending: Sort direction when ``order_by`` is given.
            limit: Maximum documents to return; ``None`` is unbounded.
            exclude: Dotted paths the store leaves out of every returned document, with the
                meaning :func:`omit_paths` gives them. For a bulk read whose consumer needs
                the documents but not a heavy value inside them; the store drops it before
                the document is shipped, so the value is never materialised at all.
            **field_eq: Field-equality predicates over the document's other fields, compared on
                the stored JSON type: a number never equals its text, and ``True`` never equals
                ``1`` (nor ``False`` ``0``), as no JSON column's boolean equals a number. A
                ``None`` value matches documents where the field is absent or null. Values are
                scalars (``str``, ``int``, ``float``, ``bool``, ``None``): a store whose backend
                compares an object or list by its serialised text cannot match one, so it may
                refuse such a value with ``TypeError``, and the engine never passes one.

        Returns:
            The matching documents.
        """
        ...

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Yield the id of every document of one type within a scope.

        The identity-only sweep, for an operator wipe of a scope its host names. It
        yields identities rather than documents because the only thing that can be done
        with the answer is delete it, and hauling whole documents to decide that is what
        makes a wipe of a large collection expensive.

        Args:
            doc_type: The discriminator to sweep.
            scope_id: The scope to sweep.

        Yields:
            Document ids suitable for :meth:`delete` in the same scope.
        """
        ...

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Write a document, creating or replacing it.

        The document supplies its own ``scope_id`` and ``id``; the store derives the
        partition from the document rather than from a caller-supplied guess, so
        a write cannot land in a partition the document does not claim.

        Args:
            document: The document to store.
            if_match: A token from :meth:`get_with_etag`. When given, the write
                lands only if the stored document still carries that token; a
                document that is not stored carries none, and refuses it.
            create_only: When True, the write lands only if no document is stored
                under the document's ``(scope_id, id)`` — of any ``doc_type``. The
                check and the write are one step, so of several racing creates
                exactly one lands. Never together with ``if_match``.

        Raises:
            StoreConflict: ``if_match`` was given and the stored document no longer
                carries it, or ``create_only`` was set and a document is stored there.
            ValueError: Both ``if_match`` and ``create_only`` were given.
            Exception: Any other backend failure.
        """
        ...

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Set top-level fields on one stored document, leaving every other field as stored.

        The partial form of :meth:`upsert`, for a write that changes a flag on a large
        document: the caller neither reads the document nor sends it back. The write is
        unconditional, so it cannot lose a race and never raises :class:`StoreConflict`.
        It mints a fresh :meth:`get_with_etag` token exactly as a whole-document write
        does, so a conditional writer holding the old token is refused and re-reads
        rather than silently putting the old value back.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.
            fields: Top-level fields and their new values. Never the fields that
                locate the document — ``id``, ``scope_id`` and ``doc_type`` — nor any
                storage-injected field: a store refuses those.

        Returns:
            ``True`` when the document was written; ``False`` when there is no such
            document — the one way this write reports not landing.

        Raises:
            ValueError: ``fields`` is empty or names a field that locates the document.
            Exception: Any other backend failure.
        """
        ...

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete one document by id within a scope.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in. Required — a scope cannot be derived from
                an id, and guessing it wrong deletes nothing while reporting
                success.

        Returns:
            ``True`` when a document was deleted, ``False`` when there was none.
        """
        ...


@runtime_checkable
class StoreClock(Protocol):
    """A store's own clock: the optional capability a :class:`DocumentStore` may offer beside its methods.

    Offered only by a store whose time is the store's — a database server's clock, which every process
    writing to it reads alike. A store whose ``now`` would be the calling process's own clock (an
    in-process store, or a file opened by each process on its own machine) does not offer it: the engine
    reads that clock itself, and a store pretending otherwise would promise an agreement it cannot keep.
    """

    def now(self) -> datetime:
        """The store's current time.

        Returns:
            A timezone-aware datetime.
        """
        ...


class AsyncDocumentStore(Protocol):
    """:class:`DocumentStore`, every method a coroutine: the port for a host whose database driver is async.

    The same port, rule for rule — the scope model, the strip on read, the error contract, both write
    preconditions — with each method awaited, and :meth:`iter_by_doc_type` an async iterator. The engine
    never calls it directly: :func:`~threetears.evals.storage.bridge_async_store` presents it as a
    :class:`DocumentStore`, whose calls run on the host's event loop, so an adapter is written against
    the driver the host already has rather than a synchronous one kept for the engine alone. The store
    conformance kit proves it through that bridge (:mod:`threetears.evals.testing.store_conformance`).

    See :class:`DocumentStore` for what each method promises; only the calling convention differs.
    """

    async def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """See :meth:`DocumentStore.get`."""
        ...

    async def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """See :meth:`DocumentStore.get_with_etag`."""
        ...

    async def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """See :meth:`DocumentStore.get_many`."""
        ...

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
        """See :meth:`DocumentStore.by_doc_type`."""
        ...

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> AsyncIterator[str]:
        """See :meth:`DocumentStore.iter_by_doc_type`; an async iterator here."""
        ...

    async def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """See :meth:`DocumentStore.upsert`."""
        ...

    async def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """See :meth:`DocumentStore.merge_fields`."""
        ...

    async def delete(self, doc_id: str, scope_id: str) -> bool:
        """See :meth:`DocumentStore.delete`."""
        ...


@runtime_checkable
class AsyncStoreClock(Protocol):
    """:class:`StoreClock` for an :class:`AsyncDocumentStore`: the store's own time, awaited.

    Offered on the same terms as :class:`StoreClock`, and only then: a store that offers it is bridged as a
    :class:`DocumentStore` that offers :class:`StoreClock`, so job leases are stamped in its time.
    """

    async def now(self) -> datetime:
        """The store's current time.

        Returns:
            A timezone-aware datetime.
        """
        ...


__all__ = [
    "AsyncDocumentStore",
    "AsyncStoreClock",
    "DocumentStore",
    "StoreCalledOnEventLoop",
    "StoreClock",
    "StoreConflict",
    "keep_fields",
    "omit_paths",
    "refuse_on_event_loop",
]
