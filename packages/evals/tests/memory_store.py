"""The package's own tests' document store: the shipped in-memory reference store, with a write that can be made to fail.

The example hosts import :class:`threetears.evals.contracts.InMemoryDocumentStore` itself, as an adopter
would; this subclass adds only the failure switch the package's tests need.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.contracts import EvalStorage
from threetears.evals.contracts import InMemoryDocumentStore as _ReferenceStore

__all__ = ["InMemoryDocumentStore", "memory_storage"]


class InMemoryDocumentStore(_ReferenceStore):
    """The reference store; while ``fail_writes`` is set, every ``upsert`` raises."""

    def __init__(self) -> None:
        """Start empty, with writes succeeding."""
        super().__init__()
        self.fail_writes = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        """Store the document, unless writes are set to fail."""
        if self.fail_writes:
            raise RuntimeError("simulated write failure")
        super().upsert(document, if_match=if_match)


def memory_storage() -> tuple[EvalStorage, InMemoryDocumentStore]:
    """An ``EvalStorage`` over a fresh in-memory store, and the store itself."""
    store = InMemoryDocumentStore()
    return EvalStorage(store), store
