"""the registry's catalog bucket comes back after a NATS restart that lost it, with the catalog in it.

Found live in a cold-start validation: a NATS restart that lost its JetStream storage -- what a
restarted NATS pod on Kubernetes can do, whatever the declared storage -- took the ``tool_catalog``
bucket with it. The registry created that bucket only at startup, through a raw nats-py handle, so
from then on every registration answered ``CATALOG_UNAVAILABLE`` (the catalog write failed with
"no response from stream"), and a restarted agent could not register its tools until the registry
itself was restarted by hand.

These drive :class:`CatalogPersistence` through the reconnect hook a real client fires, against a
scripted JetStream whose buckets a test can drop, as such a restart does.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from threetears.nats import KvBucketNotFoundError
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.catalog_persistence import CatalogPersistence

from .copy_entries import uniform_entry

_BUCKET = "tool_catalog"

#: how long a test waits for the background restore to reach the state it expects
_SETTLE_SECONDS = 5.0


# parity-exempt: one raw nats-py KeyValue's keys/get/put/delete over a dict; the real handle is nats-py's own class
class _FakeKeyValue:
    """a bucket's entries, as the server holds them."""

    def __init__(self) -> None:
        self.entries: dict[str, bytes] = {}

    async def keys(self) -> list[str]:
        if not self.entries:
            raise RuntimeError("nats: no keys found")
        return list(self.entries)

    async def get(self, key: str) -> Any:
        return type("_Entry", (), {"value": self.entries[key]})()

    async def put(self, key: str, value: bytes) -> int:
        self.entries[key] = value
        return len(self.entries)

    async def delete(self, key: str) -> bool:
        self.entries.pop(key, None)
        return True


# parity-exempt: scripted JetStream holding KV buckets a test can drop, for the two calls the catalog's owner makes
class _FakeJetStream:
    """the bind and the create the catalog's owner makes, over buckets a test can lose."""

    def __init__(self) -> None:
        self.buckets: dict[str, _FakeKeyValue] = {}
        self.created: list[str] = []
        self.bind_failures: list[Exception] = []

    def lose_storage(self) -> None:
        """drop every bucket, as a restart that lost the broker's storage does."""
        self.buckets.clear()

    async def key_value(self, bucket: str) -> _FakeKeyValue:
        if self.bind_failures:
            raise self.bind_failures.pop(0)
        if bucket not in self.buckets:
            raise KvBucketNotFoundError(f"bucket {bucket} not found", bucket=bucket)
        return self.buckets[bucket]

    async def create_key_value(self, bucket: str) -> _FakeKeyValue:
        self.created.append(bucket)
        kv = _FakeKeyValue()
        self.buckets[bucket] = kv
        return kv


# parity-with: threetears.registry.catalog_persistence.CatalogBucketClient
class _FakeClient:
    """the slice of the registry's NATS client the catalog's owner uses."""

    def __init__(self, js: _FakeJetStream) -> None:
        self._js = js
        self.hooks: list[Callable[[], Awaitable[None]]] = []

    def jetstream_context(self) -> Any:
        return self._js

    def add_reconnect_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        self.hooks.append(callback)

    async def reconnect(self) -> None:
        for hook in self.hooks:
            await hook()


def _entry(name: str) -> CatalogEntry:
    return uniform_entry(
        tool_name=name,
        tool_version="1.0.0",
        full_name=f"{name}@1.0.0",
        description=f"test tool {name}",
        input_schema={"type": "object", "properties": {}},
        endpoints=[ToolEndpoint(pod_id="pod-001", status="available")],
    )


async def _until(condition: Callable[[], bool]) -> None:
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the catalog bucket was not restored in time")
        await asyncio.sleep(0.01)


async def _started(js: _FakeJetStream) -> tuple[ToolCatalog, _FakeClient, CatalogPersistence]:
    catalog = ToolCatalog()
    client = _FakeClient(js)
    persistence = CatalogPersistence(catalog=catalog, nc=client, bucket=_BUCKET)
    await persistence.start()
    return catalog, client, persistence


def _stored_names(kv: _FakeKeyValue) -> set[str]:
    return {json.loads(value)["full_name"] for value in kv.entries.values()}


@pytest.mark.asyncio
async def test_start_creates_an_absent_bucket_and_loads_an_existing_one() -> None:
    js = _FakeJetStream()
    catalog, _, persistence = await _started(js)
    await catalog.register(_entry("threetears.calculator"))
    await persistence.stop()

    again, _, second = await _started(js)

    assert js.created == [_BUCKET], "a live bucket is bound, never created again"
    assert again.get("threetears.calculator@1.0.0") is not None
    await second.stop()


@pytest.mark.asyncio
async def test_a_bucket_lost_with_the_brokers_storage_comes_back_holding_the_catalog() -> None:
    js = _FakeJetStream()
    catalog, client, persistence = await _started(js)
    await catalog.register(_entry("threetears.calculator"))
    await catalog.register(_entry("threetears.clock"))

    js.lose_storage()
    await client.reconnect()
    await _until(lambda: _BUCKET in js.buckets and len(js.buckets[_BUCKET].entries) == 2)

    assert _stored_names(js.buckets[_BUCKET]) == {"threetears.calculator@1.0.0", "threetears.clock@1.0.0"}
    await catalog.register(_entry("threetears.weather"))
    assert len(js.buckets[_BUCKET].entries) == 3, "registrations write to the bucket that came back"
    await persistence.stop()


@pytest.mark.asyncio
async def test_a_bucket_that_survived_is_bound_and_not_created() -> None:
    js = _FakeJetStream()
    catalog, client, persistence = await _started(js)
    await catalog.register(_entry("threetears.calculator"))
    created_before = list(js.created)

    await client.reconnect()
    await _until(lambda: len(js.buckets[_BUCKET].entries) == 1)
    for _ in range(20):
        await asyncio.sleep(0)

    assert js.created == created_before
    await persistence.stop()


@pytest.mark.asyncio
async def test_a_failed_restore_is_logged_at_error_and_retried(caplog: pytest.LogCaptureFixture) -> None:
    js = _FakeJetStream()
    catalog, client, persistence = await _started(js)
    await catalog.register(_entry("threetears.calculator"))

    js.lose_storage()
    js.bind_failures.append(RuntimeError("nats: timeout"))
    with caplog.at_level(logging.ERROR, logger="threetears.registry.catalog_persistence"):
        await client.reconnect()
        await _until(lambda: _BUCKET in js.buckets and len(js.buckets[_BUCKET].entries) == 1)

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("nats: timeout" in message for message in errors), errors
    await persistence.stop()


@pytest.mark.asyncio
async def test_a_bind_failure_that_is_not_an_absent_bucket_never_creates_one() -> None:
    """only an answered "bucket not found" may create; a refusal or timeout is retried, not papered over."""
    js = _FakeJetStream()
    catalog, client, persistence = await _started(js)
    js.lose_storage()
    js.bind_failures.extend(RuntimeError("nats: timeout") for _ in range(1000))
    created_before = list(js.created)

    await client.reconnect()
    await _until(lambda: len(js.bind_failures) < 1000)
    await persistence.stop()

    assert js.created == created_before
    assert catalog.list_available(None) == []


@pytest.mark.asyncio
async def test_stop_ends_a_restore_still_retrying() -> None:
    js = _FakeJetStream()
    _, client, persistence = await _started(js)
    js.lose_storage()
    js.bind_failures.extend(RuntimeError("nats: timeout") for _ in range(1000))

    await client.reconnect()
    await _until(lambda: len(js.bind_failures) < 1000)
    await asyncio.wait_for(persistence.stop(), timeout=2.0)
    remaining = len(js.bind_failures)
    await asyncio.sleep(1.0)

    assert len(js.bind_failures) == remaining, "the restore kept running after stop"


@pytest.mark.asyncio
async def test_a_start_whose_catalog_bucket_is_unreachable_recovers_persistence_without_a_reconnect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """a registry must not serve with persistence silently off until a reconnect that may never come.

    the start returns at once -- the registry serves from an in-memory catalog its pods fill -- and
    says so at ERROR; the same background restore the reconnect path runs then declares the bucket,
    loads what an earlier registry persisted, and writes what was registered meanwhile.
    """
    js = _FakeJetStream()
    earlier, _, first = await _started(js)
    await earlier.register(_entry("threetears.calculator"))
    await first.stop()
    js.bind_failures.extend(RuntimeError("nats: timeout") for _ in range(2))

    catalog = ToolCatalog()
    persistence = CatalogPersistence(catalog=catalog, nc=_FakeClient(js), bucket=_BUCKET)
    with caplog.at_level(logging.ERROR, logger="threetears.registry.catalog_persistence"):
        await asyncio.wait_for(persistence.start(), timeout=0.5)
    assert any(r.levelno == logging.ERROR for r in caplog.records), "a start without persistence is loud"

    await catalog.register(_entry("threetears.clock"))
    await _until(lambda: len(js.buckets[_BUCKET].entries) == 2)

    assert catalog.get("threetears.calculator@1.0.0") is not None, "what an earlier registry persisted is loaded"
    assert _stored_names(js.buckets[_BUCKET]) == {"threetears.calculator@1.0.0", "threetears.clock@1.0.0"}
    await persistence.stop()


@pytest.mark.asyncio
async def test_the_late_load_never_replaces_an_entry_registered_meanwhile() -> None:
    js = _FakeJetStream()
    earlier, _, first = await _started(js)
    await earlier.register(_entry("threetears.calculator"))
    await first.stop()
    js.bind_failures.append(RuntimeError("nats: timeout"))

    catalog = ToolCatalog()
    persistence = CatalogPersistence(catalog=catalog, nc=_FakeClient(js), bucket=_BUCKET)
    await persistence.start()
    live = _entry("threetears.calculator")
    await catalog.register(live)
    await _until(lambda: js.bind_failures == [] and len(js.buckets[_BUCKET].entries) == 1)
    for _ in range(50):
        await asyncio.sleep(0.01)

    held = catalog.get("threetears.calculator@1.0.0")
    assert held is not None and held.endpoints[0].status == "available", "the live registration was kept"
    await persistence.stop()
