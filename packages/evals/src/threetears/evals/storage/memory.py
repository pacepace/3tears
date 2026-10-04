"""The in-memory reference :class:`~threetears.evals.contracts.store_port.DocumentStore`.

Every rule the port states, kept in one process's memory: documents keyed by ``(scope_id, id)``
with ``doc_type`` part of every typed read, a miss that is an outcome rather than an exception,
``exclude`` and ``keep`` with the one meaning :func:`~threetears.evals.contracts.store_port.omit_paths`
and :func:`~threetears.evals.contracts.store_port.keep_fields` give them, and optimistic concurrency:
every write mints a fresh etag, a conditional write that presents a stale one raises
:class:`~threetears.evals.contracts.store_port.StoreConflict`, and :meth:`InMemoryDocumentStore.merge_fields`
moves the etag on as a whole-document write does. It passes every case of the store conformance
kit (``threetears.evals.testing``).

It is for tests, examples and a host's first afternoon — nothing persists past the process — and it
is the shape to compare a real adapter against. It injects nothing into a stored document, so it has
nothing to strip on the way out: the etag lives beside the document, not in it. Documents are copied
in and out, so a caller mutating what it wrote or read never reaches the stored copy.

**One lock over every method.** The engine calls its store from a blocking-I/O executor, so two
threads can write one document at once; a conditional write's compare and its write happen under
the same lock, or two writers holding one etag would both land and the first change would be lost
without a conflict — exactly what the etag exists to prevent.
"""

from __future__ import annotations

import copy
import itertools
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from threetears.evals.contracts.store_port import StoreConflict, keep_fields, omit_paths

__all__ = ["InMemoryDocumentStore"]

#: The fields that locate a document, which :meth:`InMemoryDocumentStore.merge_fields` refuses to set.
_LOCATING_FIELDS = frozenset({"id", "scope_id", "doc_type"})


class InMemoryDocumentStore:
    """A :class:`~threetears.evals.contracts.store_port.DocumentStore` over one dict.

    Attributes:
        documents: The stored documents, keyed by ``(scope_id, id)``. The store's own state, exposed
            so a test can inspect what was written; a reader goes through the port's methods.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.documents: dict[tuple[str, str], dict[str, Any]] = {}
        self._etags: dict[tuple[str, str], str] = {}
        self._etag_counter = itertools.count(1)
        self._lock = threading.RLock()

    def _mint_etag(self, key: tuple[str, str]) -> None:
        """Give the document at ``key`` a token no earlier write of it carried."""
        self._etags[key] = f"etag-{next(self._etag_counter)}"

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """Return one document by id within a scope, or ``None`` if absent.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            A copy of the stored document, or ``None``.
        """
        with self._lock:
            document = self.documents.get((scope_id, doc_id))
            return copy.deepcopy(document) if document is not None else None

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return one document plus the token a conditional write must present.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            ``(document, etag)``, or ``(None, None)`` when there is no such document.
        """
        key = (scope_id, doc_id)
        with self._lock:
            document = self.documents.get(key)
            if document is None:
                return None, None
            return copy.deepcopy(document), self._etags[key]

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

        Args:
            doc_type: The discriminator every returned document carries.
            doc_ids: The ids to fetch; an absent one is skipped, and one named twice is returned once.
            scope_id: The scope to read.
            exclude: Dotted paths left out of every returned document.
            keep: When given, the only top-level fields each returned document carries.

        Returns:
            The matching documents, projected as asked.

        Raises:
            ValueError: Both ``exclude`` and ``keep`` were given.
        """
        if exclude and keep:
            raise ValueError("get_many takes exclude or keep, never both")
        with self._lock:
            found = [
                document
                for doc_id in dict.fromkeys(doc_ids)
                if (document := self.documents.get((scope_id, doc_id))) is not None
                and document.get("doc_type") == doc_type
            ]
            return [copy.deepcopy(keep_fields(d, keep) if keep else omit_paths(d, exclude)) for d in found]

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

        A ``None`` predicate matches a field that is absent or null. Under ``order_by``, documents
        without the field (or with it null) sort as lower than every document that has it.

        Args:
            doc_type: The discriminator to select on.
            scope_id: The scope to read.
            order_by: The field to sort by, or ``None`` for insertion order.
            descending: Sort direction when ``order_by`` is given.
            limit: Maximum documents to return; ``None`` is unbounded.
            exclude: Dotted paths left out of every returned document.
            **field_eq: Field-equality predicates.

        Returns:
            The matching documents.
        """
        with self._lock:
            rows = [
                document
                for (scope, _), document in self.documents.items()
                if scope == scope_id
                and document.get("doc_type") == doc_type
                and all(document.get(field) == value for field, value in field_eq.items())
            ]
            if order_by is not None:
                rows.sort(
                    key=lambda d: (d.get(order_by) is not None, d.get(order_by)),
                    reverse=descending,
                )
            if limit is not None:
                rows = rows[:limit]
            return [copy.deepcopy(omit_paths(d, exclude)) for d in rows]

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Yield the id of every document of one type within a scope.

        Args:
            doc_type: The discriminator to sweep.
            scope_id: The scope to sweep.

        Yields:
            Document ids, from a snapshot taken when iteration starts, so deleting each as it is
            yielded is safe.
        """
        with self._lock:
            ids = [
                doc_id
                for (scope, doc_id), document in self.documents.items()
                if scope == scope_id and document.get("doc_type") == doc_type
            ]
        yield from ids

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        """Write a document under the scope and id it carries, creating or replacing it.

        Args:
            document: The document to store; it supplies its own ``scope_id`` and ``id``.
            if_match: A token from :meth:`get_with_etag`; when given, the write lands only if the
                stored document still carries it.

        Raises:
            StoreConflict: ``if_match`` was given and the stored document no longer carries it,
                or is gone.
            KeyError: The document carries no ``scope_id`` or no ``id``. The store reads the scope
                as the key it is and never judges its value: every eval model refuses an empty one.
        """
        scope_id, doc_id = document["scope_id"], document["id"]
        key = (scope_id, doc_id)
        stored = copy.deepcopy(document)
        with self._lock:
            if if_match is not None and self._etags.get(key) != if_match:
                raise StoreConflict(f"document {doc_id!r} in scope {scope_id!r} no longer carries etag {if_match!r}")
            self.documents[key] = stored
            self._mint_etag(key)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Set top-level fields on one stored document, leaving every other field as stored.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.
            fields: Top-level fields and their new values.

        Returns:
            ``True`` when the document was written; ``False`` when there is no such document.

        Raises:
            ValueError: ``fields`` is empty or names a field that locates the document.
        """
        if not fields:
            raise ValueError("merge_fields needs at least one field")
        if locating := sorted(_LOCATING_FIELDS & fields.keys()):
            raise ValueError(f"merge_fields cannot set the fields that locate a document: {locating}")
        key = (scope_id, doc_id)
        merged = copy.deepcopy(fields)
        with self._lock:
            document = self.documents.get(key)
            if document is None:
                return False
            document.update(merged)
            self._mint_etag(key)
            return True

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete one document by id within a scope.

        Args:
            doc_id: The document's own id.
            scope_id: The scope it lives in.

        Returns:
            ``True`` when a document was deleted, ``False`` when there was none.
        """
        key = (scope_id, doc_id)
        with self._lock:
            self._etags.pop(key, None)
            return self.documents.pop(key, None) is not None
