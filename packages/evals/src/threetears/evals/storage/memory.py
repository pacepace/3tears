"""The in-memory reference :class:`~threetears.evals.schema.store_port.DocumentStore`.

Every rule the port states, kept in one process's memory: documents keyed by ``(scope_id, id)``
with ``doc_type`` part of every typed read, a miss that is an outcome rather than an exception,
``exclude`` and ``keep`` with the one meaning :func:`~threetears.evals.schema.store_port.omit_paths`
and :func:`~threetears.evals.schema.store_port.keep_fields` give them, and optimistic concurrency:
every write mints a fresh etag, a conditional write that presents a stale one raises
:class:`~threetears.evals.schema.store_port.StoreConflict`, and :meth:`InMemoryDocumentStore.merge_fields`
moves the etag on as a whole-document write does. It passes every case of the store conformance
kit (``threetears.evals.testing``).

It offers :class:`~threetears.evals.schema.store_port.ResultReadings` too, computed in Python over a snapshot taken
in one pass at the call, and passes every case of the readings conformance kit.

It is for tests, examples and a host's first afternoon — nothing persists past the process — and it
is the shape to compare a real adapter against. It injects nothing into a stored document, so it has
nothing to strip on the way out: the etag lives beside the document, not in it. Documents are copied
in and out, so a caller mutating what it wrote or read never reaches the stored copy. A copy is only as deep as
that needs: each object and list is copied, and a string, number, boolean or null is shared, since nothing can
change one in place (:func:`~threetears.evals.schema.store_port.copy_json`).

**One lock over every method.** The engine calls its store from a blocking-I/O executor, so two
threads can write one document at once; a conditional write's compare and its write happen under
the same lock, or two writers holding one etag would both land and the first change would be lost
without a conflict — exactly what the etag exists to prevent.
"""

from __future__ import annotations

import copy
import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from threetears.evals.schema.store_port import (
    GroupMeasure,
    StoreConflict,
    check_after,
    copy_json,
    field_in_values,
    fold_groups,
    keep_fields,
    narrow_items,
    omit_paths,
    order_and_page,
)

__all__ = ["MAX_DOCUMENTS", "InMemoryDocumentStore"]

#: The most documents one in-memory store holds. It is for tests, examples and quick runs, never a host's evidence:
#: the largest test writes 60,001 documents (``test_reads_past_fifty_thousand``), and a quick run's few thousand fit
#: far below it. A write past it is refused, naming a real adapter: past it, a store belongs in one.
MAX_DOCUMENTS: Final = 250_000

#: How many writes the newest layer takes before it is sealed into the layers below. Each write rebuilds the newest
#: layer whole, so it is kept small; sealed layers merge two of a size into one (log-structured), so a document is
#: copied about log2(n / this) times over its life and 60,000 writes take well under a second, where rebuilding one
#: map of every document per write took 68s.
_FRESH_LIMIT: Final = 32

#: The fields that locate a document, which :meth:`InMemoryDocumentStore.merge_fields` refuses to set.
_LOCATING_FIELDS = frozenset({"id", "scope_id", "doc_type"})


class _Entry(NamedTuple):
    """One write of a document: what it stored, or its deletion; and whether it takes a new place in the store's
    insertion order (a first write, a write after a deletion, or a deletion) rather than keeping the one it had."""

    document: dict[str, Any] | None
    etag: str | None
    placed: bool


type _Key = tuple[str, str]
type _Layer = Mapping[_Key, _Entry]


def _merged(older: _Layer, newer: _Layer, *, bottom: bool) -> _Layer:
    """``newer`` laid over ``older``, as one layer, in the store's insertion order: an entry that keeps its place keeps
    ``older``'s, one ``placed`` anew follows every entry of ``older``. At the bottom, deletions are dropped: nothing
    below is left for them to hide.

    The dicts here are this function's own, built and frozen before it returns.
    """
    moved = frozenset(key for key, entry in newer.items() if entry.placed)
    merged = {key: entry for key, entry in older.items() if key not in moved}
    for key, entry in newer.items():
        below = merged.get(key)
        merged[key] = entry if below is None else entry._replace(placed=below.placed)
    if bottom:
        merged = {key: entry for key, entry in merged.items() if entry.document is not None}
    return MappingProxyType(merged)


class _DocumentsView(Mapping[_Key, dict[str, Any]]):
    """The stored documents, read-only, in insertion order: what :attr:`InMemoryDocumentStore.documents` hands out.
    A deep copy is a plain dict, so a test can hold what the store held and compare it later."""

    __slots__ = ("_documents",)

    def __init__(self, documents: Mapping[_Key, dict[str, Any]]) -> None:
        self._documents = documents

    def __getitem__(self, key: _Key) -> dict[str, Any]:
        return self._documents[key]

    def __iter__(self) -> Iterator[_Key]:
        return iter(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self._documents)!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[_Key, dict[str, Any]]:
        return copy.deepcopy(dict(self._documents), memo)


def _stored_equal(stored: Any, wanted: Any) -> bool:
    """Equality on the stored JSON type, as a SQL/JSON store compares: ``true`` is not ``1``.

    Python's ``==`` reads ``True == 1`` and ``False == 0``, which no JSON column does, so a predicate
    compared that way matches documents a real adapter would not return.
    """
    if isinstance(stored, bool) or isinstance(wanted, bool):
        return type(stored) is type(wanted) and stored == wanted
    return bool(stored == wanted)


class InMemoryDocumentStore:
    """A :class:`~threetears.evals.schema.store_port.DocumentStore` over frozen layers of writes.

    Each document is kept beside its etag, keyed by ``(scope_id, id)``, in layers that are never changed in place: a
    write rebuilds the small newest layer whole (:data:`_FRESH_LIMIT`), which is sealed into the layers below when it
    fills, two of a size merging into one. It holds at most :data:`MAX_DOCUMENTS`.
    """

    def __init__(self) -> None:
        """Start empty."""
        # The sealed layers, oldest first, and the newest one; each frozen, replaced whole under the lock.
        self._layers: tuple[_Layer, ...] = ()
        self._fresh: _Layer = MappingProxyType({})
        self._count = 0
        # The documents as one read-only view, built on the first read after a write.
        self._view: _DocumentsView | None = None
        self._etag_counter = itertools.count(1)
        self._lock = threading.RLock()

    @property
    def documents(self) -> Mapping[tuple[str, str], dict[str, Any]]:
        """The stored documents, keyed by ``(scope_id, id)``, in insertion order, read-only: so a test can inspect
        what was written; a reader goes through the port's methods, a writer through :meth:`upsert`,
        :meth:`merge_fields` and :meth:`delete`."""
        with self._lock:
            if self._view is None:
                whole: _Layer = MappingProxyType({})
                for layer in (*self._layers, self._fresh):
                    whole = _merged(whole, layer, bottom=True)
                # Compacted: the next read until a write reads the one layer.
                self._layers, self._fresh = (whole,), MappingProxyType({})
                self._view = _DocumentsView(
                    MappingProxyType(
                        {key: entry.document for key, entry in whole.items() if entry.document is not None}
                    )
                )
            return self._view

    def _entry(self, key: _Key) -> _Entry | None:
        """The live entry stored under ``key``, newest layer first; None when absent or deleted."""
        for layer in (self._fresh, *reversed(self._layers)):
            if (entry := layer.get(key)) is not None:
                return entry if entry.document is not None else None
        return None

    def _put(self, key: _Key, document: dict[str, Any] | None) -> None:
        """Write ``document`` under ``key`` (``None`` deletes it), with a fresh etag, under the lock: the newest layer
        rebuilt whole, sealed into the layers below when it fills.

        :raises RuntimeError: The write would hold more than :data:`MAX_DOCUMENTS`.
        """
        live = self._entry(key) is not None
        if document is not None and not live and self._count >= MAX_DOCUMENTS:
            raise RuntimeError(
                f"an in-memory store holds at most {MAX_DOCUMENTS} documents, and this write would hold more: "
                "it is for tests and quick runs; use a real store adapter"
            )
        if document is None:
            entry = _Entry(None, None, placed=True)
        else:
            entry = _Entry(document, f"etag-{next(self._etag_counter)}", placed=not live)
        self._fresh = _merged(self._fresh, MappingProxyType({key: entry}), bottom=False)
        self._count += (1 if document is not None else -1) if live != (document is not None) else 0
        self._view = None
        if len(self._fresh) > _FRESH_LIMIT:
            layers = (*self._layers, self._fresh)
            while len(layers) > 1 and len(layers[-2]) <= 2 * len(layers[-1]):
                layers = (*layers[:-2], _merged(layers[-2], layers[-1], bottom=len(layers) == 2))
            self._layers, self._fresh = layers, MappingProxyType({})

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """Return one document by id within a scope, or ``None`` if absent.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: A copy of the stored document, or ``None``.
        :rtype: dict[str, Any] | None
        """
        with self._lock:
            entry = self._entry((scope_id, doc_id))
            return copy_json(entry.document) if entry is not None and entry.document is not None else None

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return one document plus the token a conditional write must present.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``(document, etag)``, or ``(None, None)`` when there is no such document.
        :rtype: tuple[dict[str, Any] | None, str | None]
        """
        key = (scope_id, doc_id)
        with self._lock:
            entry = self._entry(key)
            if entry is None or entry.document is None:
                return None, None
            return copy_json(entry.document), entry.etag

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

        :param doc_type: The discriminator every returned document carries.
        :ptype doc_type: str
        :param doc_ids: The ids to fetch; an absent one is skipped, and one named twice is returned once.
        :ptype doc_ids: Sequence[str]
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param exclude: Dotted paths left out of every returned document.
        :ptype exclude: Sequence[str]
        :param keep: When given, the only top-level fields each returned document carries.
        :ptype keep: Sequence[str]
        :return: The matching documents, projected as asked.
        :rtype: list[dict[str, Any]]
        :raises ValueError: Both ``exclude`` and ``keep`` were given.
        """
        if exclude and keep:
            raise ValueError("get_many takes exclude or keep, never both")
        with self._lock:
            found = [
                document
                for doc_id in dict.fromkeys(doc_ids)
                if (entry := self._entry((scope_id, doc_id))) is not None
                and (document := entry.document) is not None
                and document.get("doc_type") == doc_type
            ]
            return [copy_json(keep_fields(d, keep) if keep else omit_paths(d, exclude)) for d in found]

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        after: tuple[Any, str] | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Return documents of one type within a scope, matching ANDed equality and membership predicates.

        A predicate compares on the stored JSON type: ``True`` matches only ``true``, never ``1``, and a
        number never matches its text. A ``None`` predicate matches a field that is absent or null. A membership
        predicate (``field_in``) matches a field equal, by the same comparison, to any of its values, and one with no
        values matches nothing. Under ``order_by``, documents without the field (or with it null) sort as lower than
        every document that has it, and ties sort by ``id`` in the same direction; ``after`` starts past a place in
        that order (:func:`~threetears.evals.schema.store_port.order_and_page`).

        :param doc_type: The discriminator to select on.
        :ptype doc_type: str
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param order_by: The field to sort by, or ``None`` for insertion order.
        :ptype order_by: str | None
        :param descending: Sort direction when ``order_by`` is given.
        :ptype descending: bool
        :param limit: Maximum documents to return; ``None`` is unbounded.
        :ptype limit: int | None
        :param after: The last document of the page before, as ``(its order_by value, its id)``; ``None`` reads from
            the start. Only with ``order_by``.
        :ptype after: tuple[Any, str] | None
        :param exclude: Dotted paths left out of every returned document.
        :ptype exclude: Sequence[str]
        :param field_in: Field-membership predicates: field → the values it may equal.
        :ptype field_in: Mapping[str, Sequence[Any]] | None
        :param **field_eq: Field-equality predicates.
        :ptype field_eq: Any
        :return: The matching documents.
        :rtype: list[dict[str, Any]]
        :raises ValueError: A field is named in both ``field_in`` and ``field_eq``, or ``after`` is given without
            ``order_by``.
        :raises TypeError: A ``field_in`` entry is a string rather than a sequence of values, or holds a value that is
            not a scalar; or ``after`` is not a pair of a scalar and a string id.
        """
        check_after(after, order_by)
        with self._lock:
            rows = self._matching(doc_type, scope_id, field_in, field_eq)
            if order_by is not None:
                rows = order_and_page(rows, order_by=order_by, descending=descending, after=after, limit=limit)
            elif limit is not None:
                rows = rows[:limit]
            return [copy_json(omit_paths(d, exclude)) for d in rows]

    def _matching(
        self, doc_type: str, scope_id: str, field_in: Mapping[str, Sequence[Any]] | None, field_eq: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """The documents a typed read selects, in insertion order: one pass over the stored documents.

        Each is a shallow copy taken under the lock, so a caller changing a row never reaches the stored document; a
        write stores a new document rather than changing one, and nested values are never changed in place, so sharing
        them is safe.

        :raises ValueError: A field is named in both ``field_in`` and ``field_eq``.
        :raises TypeError: A ``field_in`` entry is a string, or holds a value that is not a scalar.
        """
        members = field_in_values(field_in, field_eq)
        if any(not values for values in members.values()):
            return []
        with self._lock:
            return [
                dict(document)
                for (scope, _), document in self.documents.items()
                if scope == scope_id
                and document.get("doc_type") == doc_type
                and all(_stored_equal(document.get(field), value) for field, value in field_eq.items())
                and all(
                    any(_stored_equal(document.get(field), value) for value in values)
                    for field, values in members.items()
                )
            ]

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = MappingProxyType({}),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield the rows :meth:`by_doc_type` would return for the same predicates, each narrowed.

        The rows are a snapshot taken when this is called, in one pass over the stored documents; each is narrowed
        and copied as it is yielded. See :meth:`~threetears.evals.schema.ResultReadings.iter_narrowed`.

        :param doc_type: The discriminator to select on.
        :ptype doc_type: str
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param exclude: Dotted paths left out of every row.
        :ptype exclude: Sequence[str]
        :param exclude_in_items: List-field path → dotted paths left out of each object item of that list.
        :ptype exclude_in_items: Mapping[str, Sequence[str]]
        :param field_in: Field-membership predicates: field → the values it may equal.
        :ptype field_in: Mapping[str, Sequence[Any]] | None
        :param **field_eq: Field-equality predicates.
        :ptype field_eq: Any
        :return: The narrowed rows, in insertion order.
        :rtype: Iterator[dict[str, Any]]
        :raises ValueError: A field is named in both ``field_in`` and ``field_eq``.
        :raises TypeError: A ``field_in`` entry is a string rather than a sequence of values, or holds a value that is
            not a scalar.
        """
        rows = self._matching(doc_type, scope_id, field_in, field_eq)
        return (copy_json(narrow_items(omit_paths(row, exclude), exclude_in_items)) for row in rows)

    def grouped(
        self,
        doc_type: str,
        scope_id: str,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
        exclude_in_items: Mapping[str, Sequence[str]] = MappingProxyType({}),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Group the rows :meth:`by_doc_type` would return and compute each measure per group.

        Folded in Python over a snapshot taken in one pass, by
        :func:`~threetears.evals.schema.store_port.fold_groups`. See
        :meth:`~threetears.evals.schema.ResultReadings.grouped` for the measures and their refusals.

        :param doc_type: The discriminator to select on.
        :ptype doc_type: str
        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param by: Dotted paths to group by.
        :ptype by: Sequence[str]
        :param measures: Measure name → the measure.
        :ptype measures: Mapping[str, GroupMeasure]
        :param exclude_in_items: List-field path → dotted paths left out of each object item of that list, in every
            row before it is folded.
        :ptype exclude_in_items: Mapping[str, Sequence[str]]
        :param field_in: Field-membership predicates: field → the values it may equal.
        :ptype field_in: Mapping[str, Sequence[Any]] | None
        :param **field_eq: Field-equality predicates.
        :ptype field_eq: Any
        :return: One dict per group, in the order each group was first seen.
        :rtype: list[dict[str, Any]]
        :raises ValueError: The grouping is refused, or a field is named in both ``field_in`` and ``field_eq``.
        :raises TypeError: A measure met a value its kind refuses, a ``by`` value is an object or a list, or a
            predicate is refused.
        """

        def matching() -> Iterator[dict[str, Any]]:
            # Lazy, so fold_groups refuses a grouping before any document is read.
            for row in self._matching(doc_type, scope_id, field_in, field_eq):
                yield narrow_items(row, exclude_in_items) if exclude_in_items else row

        return fold_groups(matching(), by=by, measures=measures)

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Yield the id of every document of one type within a scope.

        :param doc_type: The discriminator to sweep.
        :ptype doc_type: str
        :param scope_id: The scope to sweep.
        :ptype scope_id: str
        :yield: Document ids, from a snapshot taken when iteration starts, so deleting each as it is
            yielded is safe.
        :rtype: Iterator[str]
        """
        with self._lock:
            ids = [
                doc_id
                for (scope, doc_id), document in self.documents.items()
                if scope == scope_id and document.get("doc_type") == doc_type
            ]
        yield from ids

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Write a document under the scope and id it carries, creating or replacing it.

        :param document: The document to store; it supplies its own ``scope_id`` and ``id``.
        :ptype document: dict[str, Any]
        :param if_match: A token from :meth:`get_with_etag`; when given, the write lands only if the
            stored document still carries it.
        :ptype if_match: str | None
        :param create_only: When True, the write lands only if nothing is stored under its scope and id.
        :ptype create_only: bool
        :raises StoreConflict: ``if_match`` was given and the stored document no longer carries it,
            or is gone; or ``create_only`` was set and a document is stored there.
        :raises ValueError: Both ``if_match`` and ``create_only`` were given.
        :raises KeyError: The document carries no ``scope_id`` or no ``id``. The store reads the scope
            as the key it is and never judges its value: every eval model refuses an empty one.
        """
        if if_match is not None and create_only:
            raise ValueError("a write is conditional on an etag or on absence, never both")
        scope_id, doc_id = document["scope_id"], document["id"]
        key = (scope_id, doc_id)
        stored = copy_json(document)
        with self._lock:
            held = self._entry(key)
            if if_match is not None and (held.etag if held is not None else None) != if_match:
                raise StoreConflict(f"document {doc_id!r} in scope {scope_id!r} no longer carries etag {if_match!r}")
            if create_only and held is not None:
                raise StoreConflict(f"document {doc_id!r} in scope {scope_id!r} already exists")
            self._put(key, stored)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Set top-level fields on one stored document, leaving every other field as stored.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :param fields: Top-level fields and their new values.
        :ptype fields: dict[str, Any]
        :return: ``True`` when the document was written; ``False`` when there is no such document.
        :rtype: bool
        :raises ValueError: ``fields`` is empty or names a field that locates the document.
        """
        if not fields:
            raise ValueError("merge_fields needs at least one field")
        if locating := sorted(_LOCATING_FIELDS & fields.keys()):
            raise ValueError(f"merge_fields cannot set the fields that locate a document: {locating}")
        key = (scope_id, doc_id)
        merged = copy_json(fields)
        with self._lock:
            held = self._entry(key)
            if held is None or held.document is None:
                return False
            self._put(key, {**held.document, **merged})
            return True

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete one document by id within a scope.

        :param doc_id: The document's own id.
        :ptype doc_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``True`` when a document was deleted, ``False`` when there was none.
        :rtype: bool
        """
        key = (scope_id, doc_id)
        with self._lock:
            if self._entry(key) is None:
                return False
            self._put(key, None)
            return True
