"""Storage adapters the engine ships: implementations of the one port a host stores through.

The port itself — :class:`~threetears.evals.contracts.DocumentStore`, its
:class:`~threetears.evals.contracts.StoreConflict` and the projection helpers — is a contract and
lives in :mod:`threetears.evals.contracts`, which depends on nothing but pydantic. This package holds
adapters BEHIND that port, so an adapter's dependencies never reach the contracts.

Today it holds one: :class:`InMemoryDocumentStore`, the reference adapter for tests, examples and a
quick start, and the shape to compare a real adapter against. A host with its own database writes
its own adapter, passes it to :class:`~threetears.evals.contracts.EvalStorage`, and proves it with
the store conformance kit in :mod:`threetears.evals.testing`.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.storage.memory import InMemoryDocumentStore

__all__ = ["InMemoryDocumentStore"]
