"""core's durable ledger is a ``threetears.models.key_pool.store.KeyPoolStore``, structurally.

Core's source never imports ``threetears.models`` (models depends on neither core nor nats, and core
does not depend on models), so nothing in either package's ``src`` ties :class:`BudgetLedger` to the
Protocol it is built for. This test is that tie, from core's tests, where the workspace installs both:

- at run time, ``isinstance`` against the ``runtime_checkable`` Protocol, which checks that every
  method is present;
- at type-check time, the annotated assignment below, which mypy checks signature by signature --
  the snapshot's ``TypedDict`` included, since core declares its own of the same shape.
"""

from __future__ import annotations

import uuid

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import BudgetLedger
from threetears.core.testing.kv import FakeNatsClient
from threetears.models.key_pool.store import KeyPoolStore


def _ledger() -> BudgetLedger:
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=SQLiteBackend(db_name=f"ledger_conformance_{uuid.uuid4().hex[:8]}"),
        l2_client=FakeNatsClient(),
        # the ledger refuses a registry with no L3; this test calls no method, so any pool stands in
        l3_pool=object(),  # type: ignore[arg-type]
        kv_key_scope="ledger-conformance",
    )
    return BudgetLedger(registry, purpose="pool")


def test_the_ledger_is_a_key_pool_store() -> None:
    store: KeyPoolStore = _ledger()
    assert isinstance(store, KeyPoolStore)
