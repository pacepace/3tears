"""Storage adapters the engine ships: implementations of the one port a host stores through.

The port itself — :class:`~threetears.evals.schema.DocumentStore`, its
:class:`~threetears.evals.schema.StoreConflict` and the projection helpers — is a contract and
lives in :mod:`threetears.evals.schema`, which depends on nothing but pydantic. This package holds
adapters BEHIND that port, so an adapter's dependencies never reach the schema.

It holds two adapters and a router. :class:`InMemoryDocumentStore` is the reference adapter for tests,
examples and a quick start, and the shape to compare a real adapter against; nothing it holds outlives the
process. :class:`SqliteDocumentStore` keeps every document in one SQLite file through the standard library, for
keeping runs without running a database (``run_eval(..., store=SqliteDocumentStore("evals.sqlite"))``).
:class:`RoutedDocumentStore` is one store over several, each document type kept in the store a
:class:`DocTypeRoute` over its facts names. All three pass every case of the store conformance kit. A host with its own database writes
its own adapter, passes it to :class:`~threetears.evals.kernel.EvalStorage`, and proves it with
the store conformance kit in :mod:`threetears.evals.testing`.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.storage.memory import InMemoryDocumentStore
from threetears.evals.storage.routed import (
    ROUTING_SCOPE,
    DocTypeRoute,
    MisplacedDocument,
    RouteChanged,
    RoutedDocumentStore,
)
from threetears.evals.storage.sqlite import SQLITE_STORE_LAYOUT, SqliteDocumentStore

__all__ = [
    "ROUTING_SCOPE",
    "SQLITE_STORE_LAYOUT",
    "DocTypeRoute",
    "InMemoryDocumentStore",
    "MisplacedDocument",
    "RouteChanged",
    "RoutedDocumentStore",
    "SqliteDocumentStore",
]
