"""Stores for reading one store's documents on either read path: as the store is, and with its narrowed and grouped
reads hidden.

``EvalStorage`` reads readings and groups on the store's path when its store offers
:class:`~threetears.evals.schema.store_port.ResultReadings`, and on the engine's otherwise. :class:`EngineOnlyStore`
forwards every method of the port to the store it wraps and nothing else, so the same documents are read on the
engine's path. It never forwards by ``__getattr__``: ``isinstance`` against a runtime-checkable protocol checks only
that the methods exist, so a forwarding fallback would offer the capability it is meant to hide.

:class:`CountingStore` is the reference store naming each backend call it is asked for, the way a test counts the store
calls a read makes.

:func:`hook_reads` hooks each pass a reference store's read makes over its documents, for a kit's backend hook.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from threetears.evals.schema.store_port import DocumentStore, GroupMeasure
from threetears.evals.storage import InMemoryDocumentStore

__all__ = ["CountingStore", "EngineOnlyStore"]


class EngineOnlyStore(DocumentStore):
    """``inner``'s documents behind the port's methods alone: no narrowed or grouped reads, whatever ``inner`` offers."""

    def __init__(self, inner: DocumentStore) -> None:
        self.inner = inner

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        return self.inner.get(doc_id, scope_id)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        return self.inner.get_with_etag(doc_id, scope_id)

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return self.inner.get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)

    def by_doc_type(
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

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        return self.inner.iter_by_doc_type(doc_type, scope_id)

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        self.inner.upsert(document, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return self.inner.merge_fields(doc_id, scope_id, fields)

    def delete(self, doc_id: str, scope_id: str) -> bool:
        return self.inner.delete(doc_id, scope_id)


class CountingStore(InMemoryDocumentStore):
    """The reference store, naming each read it is asked for, in order (``calls``)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, field_in: Mapping[str, Sequence[Any]] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append("by_doc_type")
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append("get_many")
        return super().get_many(doc_type, doc_ids, scope_id, **kwargs)

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        self.calls.append("iter_narrowed")
        return super().iter_narrowed(
            doc_type, scope_id, exclude=exclude, exclude_in_items=exclude_in_items, field_in=field_in, **field_eq
        )

    def grouped(
        self,
        doc_type: str,
        scope_id: str,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append("grouped")
        return super().grouped(
            doc_type,
            scope_id,
            by=by,
            measures=measures,
            exclude_in_items=exclude_in_items,
            field_in=field_in,
            **field_eq,
        )


class HookedDocuments(dict[tuple[str, str], dict[str, Any]]):
    """A store's documents, calling the installed hook on each pass a read makes over them (``items()``)."""

    hook: Callable[[], None] | None = None

    # Any: dict's own items view type lives in a private module, and this hands back exactly what dict's does.
    def items(self) -> Any:
        if self.hook is not None:
            self.hook()
        return super().items()


def hook_reads(store: InMemoryDocumentStore) -> HookedDocuments:
    """Hook each pass ``store``'s reads make over its documents; set the hook on what this returns.

    The store reads its documents through its own read-only :attr:`~InMemoryDocumentStore.documents`, so ``store``
    becomes an instance of a subclass of its own class whose ``documents`` hands each pass a hooked copy: every
    other method, its own or a fault's, stays as it was.
    """
    hooks = HookedDocuments()

    def documents(self: InMemoryDocumentStore) -> Mapping[tuple[str, str], dict[str, Any]]:
        seen = HookedDocuments(InMemoryDocumentStore.documents.__get__(self))
        seen.hook = hooks.hook
        return seen

    store.__class__ = type(f"Hooked{type(store).__name__}", (type(store),), {"documents": property(documents)})
    return hooks
