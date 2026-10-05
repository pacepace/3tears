"""a registry whose catalog writes keep failing reports itself not ready, and ready again once one lands.

Found on cobalt-dev after a NATS rolling restart: every registration answered ``CATALOG_UNAVAILABLE``
with ``nats: connection closed`` while the registry reported itself healthy, until its pod was deleted
by hand. ``RegistryServer`` wires ``HealthCheck(name="catalog_persisting", probe=lambda:
catalog.persisting, tier=HealthTier.READY)``; these drive that probe over a real
:class:`ToolCatalog` writing to the shipped in-memory bucket, through a real :class:`HealthServer`,
the way ``test_readiness.py`` drives the registry's JWKS gate.
"""

from __future__ import annotations

import logging

import pytest

from threetears.core.testing.kv import FakeKvBucket
from threetears.nats import KvError
from threetears.observe import HealthCheck, HealthServer, HealthTier
from threetears.registry.catalog import WRITE_FAILURE_THRESHOLD, CatalogEntry, ToolCatalog, ToolEndpoint

from .copy_entries import uniform_entry


def _entry(name: str) -> CatalogEntry:
    return uniform_entry(
        tool_name=name,
        tool_version="1.0.0",
        full_name=f"{name}@1.0.0",
        description=f"test tool {name}",
        input_schema={"type": "object", "properties": {}},
        endpoints=[ToolEndpoint(pod_id="pod-001", status="available")],
    )


def _health_server(catalog: ToolCatalog) -> HealthServer:
    """a health server carrying the EXACT probe ``RegistryServer`` wires for the catalog."""
    return HealthServer(
        port=0,
        service_name="registry",
        checks=[HealthCheck(name="catalog_persisting", probe=lambda: catalog.persisting, tier=HealthTier.READY)],
    )


async def _bound_catalog() -> tuple[ToolCatalog, FakeKvBucket]:
    bucket = FakeKvBucket(bucket_name="tool_catalog", storage="file", direct=True)
    catalog = ToolCatalog()
    await catalog.load_from_kv(bucket)
    return catalog, bucket


async def _failed_registration(catalog: ToolCatalog, name: str) -> None:
    with pytest.raises(KvError, match="connection closed"):
        await catalog.register(_entry(name))


@pytest.mark.asyncio
async def test_a_streak_of_failed_writes_makes_the_registry_not_ready_and_one_that_lands_ends_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog, bucket = await _bound_catalog()
    server = _health_server(catalog)
    assert (await server.get_status(HealthTier.READY)).healthy is True

    bucket.become_unreachable(KvError("nats: connection closed"))
    for attempt in range(WRITE_FAILURE_THRESHOLD - 1):
        await _failed_registration(catalog, f"tool.n{attempt}")
    assert catalog.persisting is True, "a failure short of the threshold is a blip, not an outage"
    assert (await server.get_status(HealthTier.READY)).healthy is True

    with caplog.at_level(logging.ERROR, logger="threetears.registry.catalog"):
        await _failed_registration(catalog, "tool.last")
    assert catalog.persisting is False
    ready = await server.get_status(HealthTier.READY)
    assert ready.healthy is False
    assert {c.name: c.healthy for c in ready.components}["catalog_persisting"] is False
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        f"failed {WRITE_FAILURE_THRESHOLD} times in a row" in message and "connection closed" in message
        for message in errors
    ), errors

    bucket.become_reachable()
    await catalog.register(_entry("tool.recovered"))
    assert catalog.persisting is True
    assert (await server.get_status(HealthTier.READY)).healthy is True


@pytest.mark.asyncio
async def test_the_catalog_check_is_never_a_liveness_failure() -> None:
    """a NATS outage fails catalog writes, and a restart does not fix an outage."""
    catalog, bucket = await _bound_catalog()
    bucket.become_unreachable(KvError("nats: connection closed"))
    for attempt in range(WRITE_FAILURE_THRESHOLD + 2):
        await _failed_registration(catalog, f"tool.n{attempt}")

    live = await _health_server(catalog).get_status(HealthTier.LIVE)
    assert live.healthy is True
    assert live.components == []


@pytest.mark.asyncio
async def test_every_kind_of_catalog_write_counts_toward_the_streak() -> None:
    """a registration, a copy withdrawn, a deregistration and a promotion all write the bucket."""
    catalog, bucket = await _bound_catalog()
    await catalog.register(_entry("tool.a"))
    await catalog.register(_entry("tool.b"))
    pending = _entry("tool.c")
    pending.endpoints[0].status = "pending"
    await catalog.register(pending)
    bucket.become_unreachable(KvError("nats: connection closed"))

    assert await catalog.deregister("tool.a@1.0.0") is False
    with pytest.raises(KvError):
        await catalog.mark_ready("pod-001")
    assert catalog.persisting is True
    # the last copy withdrawn deregisters the tool, whose delete fails and is reported, not raised
    assert await catalog.remove_copy("tool.b@1.0.0", "pod-001") is True

    assert catalog.persisting is False


@pytest.mark.asyncio
async def test_a_catalog_with_no_bucket_bound_is_persisting() -> None:
    """no bucket yet means no write has failed; the owner's own declaration logs that wait."""
    catalog = ToolCatalog()
    await catalog.register(_entry("tool.a"))
    assert catalog.persisting is True
