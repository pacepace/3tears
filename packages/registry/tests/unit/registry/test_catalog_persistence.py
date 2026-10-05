"""the registry's catalog bucket comes back after a NATS restart that lost it, with the catalog in it.

Found live in a cold-start validation: a NATS restart that lost its JetStream storage -- what a
restarted NATS pod on Kubernetes can do, whatever the declared storage -- took the ``tool_catalog``
bucket with it, and every registration answered ``CATALOG_UNAVAILABLE`` until the registry itself was
restarted by hand. Found again on cobalt-dev after a NATS rolling restart: the bucket was held through
a raw nats-py handle that stayed bound to a retired connection.

The bucket is now owned by :class:`threetears.nats.PersistedCopyBucket`, built by
:func:`threetears.registry.catalog_persistence.catalog_bucket`. These drive that owner over the
shipped in-memory client, :class:`threetears.core.testing.kv.FakeNatsClient`, whose
``restart_broker`` loses every bucket, puts back the declared ones empty and runs their refills, as
the real client does. What the owner itself guarantees for any persisted copy -- the start retry,
load once, write back on every refill -- is pinned in ``packages/nats/tests/unit/test_persisted_copy.py``;
what is pinned here is the catalog's side of it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats import KvError, PersistedCopyBucket
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.catalog_persistence import catalog_bucket

from .copy_entries import uniform_entry

_BUCKET = "tool_catalog"

#: how long a test waits for the background declaration to reach the state it expects; the owner's
#: retry schedule starts at half a second and doubles
_SETTLE_SECONDS = 5.0


class _UnansweredClient(FakeNatsClient):
    """the shipped client, whose first ``unanswered`` declarations time out as a broker still starting does."""

    def __init__(self, *, unanswered: int) -> None:
        super().__init__()
        self.unanswered = unanswered
        self.declarations: list[dict[str, Any]] = []

    async def ensure_kv_bucket(self, **kwargs: Any) -> FakeKvBucket:  # type: ignore[override]
        self.declarations.append(kwargs)
        if self.unanswered > 0:
            self.unanswered -= 1
            raise KvError("nats: timeout")
        return await super().ensure_kv_bucket(**kwargs)


def _entry(name: str, *, status: str = "available") -> CatalogEntry:
    return uniform_entry(
        tool_name=name,
        tool_version="1.0.0",
        full_name=f"{name}@1.0.0",
        description=f"test tool {name}",
        input_schema={"type": "object", "properties": {}},
        endpoints=[ToolEndpoint(pod_id="pod-001", status=status)],
    )


async def _until(condition: Callable[[], bool]) -> None:
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the catalog bucket did not reach the expected state in time")
        await asyncio.sleep(0.01)


async def _started(
    client: FakeNatsClient, catalog: ToolCatalog | None = None
) -> tuple[ToolCatalog, PersistedCopyBucket]:
    held = catalog if catalog is not None else ToolCatalog()
    owner = catalog_bucket(catalog=held, client=client, bucket=_BUCKET)
    await owner.start()
    return held, owner


async def _declared(client: FakeNatsClient) -> tuple[ToolCatalog, PersistedCopyBucket]:
    """a started catalog whose bucket's first declaration has landed."""
    catalog, owner = await _started(client)
    await _until(lambda: owner.bucket is not None and client.bucket_exists(_BUCKET))
    # the first declaration loads then writes back; let both finish before the test acts
    for _ in range(20):
        await asyncio.sleep(0)
    return catalog, owner


async def _stored_names(bucket: FakeKvBucket) -> set[str]:
    names: set[str] = set()
    for key in await bucket.list_keys():
        value = await bucket.get(key=key)
        if value is not None:
            names.add(json.loads(value)["full_name"])
    return names


@pytest.mark.asyncio
async def test_the_bucket_is_declared_under_its_exact_name_with_its_stated_shape() -> None:
    client = FakeNatsClient()
    _, owner = await _declared(client)

    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)
    assert owner.name == _BUCKET
    assert bucket.storage == "file"
    assert bucket.direct is True
    assert _BUCKET in client.remembered_declarations, "the client puts the bucket back after a restart"
    await owner.stop()


@pytest.mark.asyncio
async def test_start_creates_an_absent_bucket_and_loads_an_existing_one() -> None:
    client = FakeNatsClient()
    catalog, first = await _declared(client)
    await catalog.register(_entry("threetears.calculator"))
    await first.stop()

    again, second = await _declared(client)

    assert again.get("threetears.calculator@1.0.0") is not None, "what an earlier registry persisted is loaded"
    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)
    assert await _stored_names(bucket) == {"threetears.calculator@1.0.0"}, "a live bucket keeps its entries"
    await second.stop()


@pytest.mark.asyncio
async def test_a_bucket_lost_with_the_brokers_storage_comes_back_holding_the_catalog() -> None:
    client = FakeNatsClient()
    catalog, owner = await _declared(client)
    await catalog.register(_entry("threetears.calculator"))
    await catalog.register(_entry("threetears.clock"))

    await client.restart_broker()

    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)
    assert await _stored_names(bucket) == {"threetears.calculator@1.0.0", "threetears.clock@1.0.0"}
    assert not client.refill_owed(_BUCKET)
    await catalog.register(_entry("threetears.weather"))
    assert len(await bucket.list_keys()) == 3, "registrations write to the bucket that came back"
    await owner.stop()


@pytest.mark.asyncio
async def test_a_tool_deregistered_while_its_delete_failed_is_not_brought_back_by_a_restart() -> None:
    """memory is what the recreated bucket is refilled from, and memory no longer holds the tool.

    A tool deregistered while its delete could not reach the bucket is gone from memory but still in
    the bucket. After a restart that lost the bucket, neither the catalog nor the bucket the client
    puts back may hold it again.
    """
    client = FakeNatsClient()
    catalog, owner = await _declared(client)
    await catalog.register(_entry("threetears.calculator"))
    await catalog.register(_entry("threetears.clock"))
    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)
    bucket.become_unreachable(KvError("nats: connection closed"))
    assert await catalog.deregister("threetears.clock@1.0.0") is False, "the delete did not reach the bucket"
    bucket.become_reachable()

    await client.restart_broker()

    assert catalog.get("threetears.clock@1.0.0") is None
    assert await _stored_names(bucket) == {"threetears.calculator@1.0.0"}
    await owner.stop()


@pytest.mark.asyncio
async def test_a_bucket_that_survived_a_reconnect_is_neither_refilled_nor_reloaded() -> None:
    client = FakeNatsClient()
    catalog, owner = await _declared(client)
    await catalog.register(_entry("threetears.calculator"))
    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)
    revision_before = (await bucket.get_entry(key="threetears_calculator_AT_1_0_0") or (b"", 0))[1]

    await client.reconnect()

    assert not client.refill_owed(_BUCKET)
    entry = await bucket.get_entry(key="threetears_calculator_AT_1_0_0")
    assert entry is not None and entry[1] == revision_before, "nothing rewrote an entry that survived"
    await owner.stop()


@pytest.mark.asyncio
async def test_a_failed_refill_stays_owed_and_is_retried(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeNatsClient()
    catalog, owner = await _declared(client)
    await catalog.register(_entry("threetears.calculator"))
    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)

    bucket.become_unreachable(KvError("nats: timeout"))
    with caplog.at_level(logging.ERROR):
        await client.restart_broker()
    assert client.refill_owed(_BUCKET), "a write-back that did not land leaves the refill owed"
    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert any("catalog entries were not written back" in message for message in errors), errors

    bucket.become_reachable()
    await client.reconnect()

    assert not client.refill_owed(_BUCKET)
    assert await _stored_names(bucket) == {"threetears.calculator@1.0.0"}
    await owner.stop()


@pytest.mark.asyncio
async def test_a_declaration_the_broker_does_not_answer_creates_nothing_and_is_retried() -> None:
    """a timeout is retried, never papered over with a bucket nobody declared."""
    client = _UnansweredClient(unanswered=1000)
    catalog, owner = await _started(client)
    await _until(lambda: len(client.declarations) >= 2)
    await owner.stop()

    assert not client.bucket_exists(_BUCKET)
    assert owner.bucket is None
    assert catalog.list_available(None) == []


@pytest.mark.asyncio
async def test_stop_ends_a_declaration_still_retrying() -> None:
    client = _UnansweredClient(unanswered=1000)
    _, owner = await _started(client)
    await _until(lambda: len(client.declarations) >= 1)

    await asyncio.wait_for(owner.stop(), timeout=2.0)
    attempts = len(client.declarations)
    await asyncio.sleep(1.0)

    assert len(client.declarations) == attempts, "the declaration kept retrying after stop"


@pytest.mark.asyncio
async def test_a_start_whose_catalog_bucket_is_unreachable_recovers_persistence_without_a_reconnect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """a registry must not serve with persistence silently off until a reconnect that may never come.

    the start returns at once -- the registry serves from an in-memory catalog its pods fill -- and
    says so at ERROR; the background declaration then lands, loads what an earlier registry
    persisted, and writes what was registered meanwhile.
    """
    client = _UnansweredClient(unanswered=0)
    earlier, first = await _declared(client)
    await earlier.register(_entry("threetears.calculator"))
    await first.stop()
    client.unanswered = 2

    with caplog.at_level(logging.ERROR, logger="threetears.nats.persisted_copy"):
        catalog, owner = await asyncio.wait_for(_started(client), timeout=0.5)
        await catalog.register(_entry("threetears.clock"))
        await _until(lambda: client.unanswered == 0 and catalog.get("threetears.calculator@1.0.0") is not None)
    assert any(r.levelno == logging.ERROR for r in caplog.records), "a start without persistence is loud"
    for _ in range(20):
        await asyncio.sleep(0)

    bucket = await client.kv_bucket(name=_BUCKET, create_if_missing=False)
    assert await _stored_names(bucket) == {"threetears.calculator@1.0.0", "threetears.clock@1.0.0"}
    await owner.stop()


@pytest.mark.asyncio
async def test_the_late_load_never_replaces_an_entry_registered_meanwhile() -> None:
    client = _UnansweredClient(unanswered=0)
    earlier, first = await _declared(client)
    await earlier.register(_entry("threetears.calculator"))
    await first.stop()
    client.unanswered = 1

    catalog, owner = await _started(client)
    await catalog.register(_entry("threetears.calculator"))
    await _until(lambda: client.unanswered == 0 and owner.bucket is not None)
    for _ in range(50):
        await asyncio.sleep(0.01)

    held = catalog.get("threetears.calculator@1.0.0")
    assert held is not None and held.endpoints[0].status == "available", "the live registration was kept"
    await owner.stop()
