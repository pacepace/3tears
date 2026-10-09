"""a rescope whose advance fails still moves, evicts and announces its row, and says what it decided.

The UPDATE commits before the namespaces table's write generation is advanced. When the advance
fails, the row has moved all the same, so the caller still owes it its own invalidations and its
audit: :meth:`NamespaceCollection.rescope` raises :class:`NamespaceRescopeNotAdvanced` carrying the
outcome, for the caller to finish with and raise last.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.acl import NamespaceCollection, NamespaceRescopeNotAdvanced
from threetears.core.collections import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeNatsClient


class _Pool:
    """a platform-scoped row, and an UPDATE that moves it."""

    def __init__(self, namespace_id: Any) -> None:
        self.namespace_id = namespace_id

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        if sql.startswith("SELECT row_scope, customer_id FROM namespaces"):
            return {"row_scope": "platform", "customer_id": None}
        assert sql.startswith("UPDATE namespaces"), sql
        return {"namespace_id": self.namespace_id}


class _FailingSource:
    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        raise GenerationUnavailableError(f"epoch bucket unreachable for {table_name}")


async def test_a_failed_advance_carries_the_outcome_after_the_row_was_announced() -> None:
    namespace_id, customer_id = uuid4(), uuid4()
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=_Pool(namespace_id), l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    registry.set_generation_source(_FailingSource())
    namespaces = NamespaceCollection(
        registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client=bus
    )

    with pytest.raises(NamespaceRescopeNotAdvanced) as raised:
        await namespaces.rescope(namespace_id, customer_id=customer_id)

    assert isinstance(raised.value, GenerationUnavailableError)
    outcome = raised.value.outcome
    assert (outcome.moved, outcome.previous_row_scope, outcome.row_scope, outcome.customer_id) == (
        True,
        "platform",
        "customer",
        customer_id,
    )
    announced = [m.ids for m in bus.published if isinstance(m, CacheInvalidationMessage)]
    assert announced == [["platform", f"{namespace_id}"], ["customer", f"{namespace_id}"]]
