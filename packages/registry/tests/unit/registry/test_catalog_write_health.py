"""a registry whose catalog writes keep failing reports itself not ready, and ready again once one lands.

Found on cobalt-dev after a NATS rolling restart: every registration answered ``CATALOG_UNAVAILABLE``
with ``nats: connection closed`` while the registry reported itself healthy, until its pod was deleted
by hand. ``RegistryServer`` wires ``HealthCheck(name="catalog_persisting", probe=lambda:
catalog.persisting, tier=HealthTier.READY)`` and ``HealthCheck(name="catalog_connection_usable",
probe=lambda: catalog.connection_usable, tier=HealthTier.LIVE)``; these drive both probes over a
real :class:`ToolCatalog` writing to the shipped in-memory bucket, through a real
:class:`HealthServer`, the way ``test_readiness.py`` drives the registry's JWKS gate.

Any streak of failed writes takes the registry out of rotation. Only writes failing on a CLOSED
connection fail liveness: nats-py never reopens a closed connection, so a restart is the one thing
that clears it, while a NATS outage -- timeouts, no responders, a lost bucket -- ends on its own and
must never put the registry in a restart loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import nats.errors
import pytest

from threetears.core.testing.kv import FakeKvBucket
from threetears.nats import KvBucketNotFoundError, KvError, NatsClientError
from threetears.nats.errors import PublishTimeoutError
from threetears.observe import HealthCheck, HealthServer, HealthTier
from threetears.observe.health import HealthStatus
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
    """a health server carrying the EXACT probes ``RegistryServer`` wires for the catalog."""
    return HealthServer(
        port=0,
        service_name="registry",
        checks=[
            HealthCheck(name="catalog_persisting", probe=lambda: catalog.persisting, tier=HealthTier.READY),
            HealthCheck(
                name="catalog_connection_usable", probe=lambda: catalog.connection_usable, tier=HealthTier.LIVE
            ),
        ],
    )


def _chained(error: KvError, cause: BaseException) -> KvError:
    """``error`` raised ``from cause``, the way the wrapper chains every KV failure to what nats-py raised.

    :param error: the wrapper's error
    :ptype error: KvError
    :param cause: what nats-py raised
    :ptype cause: BaseException
    :return: ``error``, carrying ``cause`` as its explicit cause
    :rtype: KvError
    """
    try:
        raise error from cause
    except KvError as raised:
        return raised


def _closed_connection() -> KvError:
    """what a ``NatsKvBucket`` write raises on a closed connection: a KvError chained, through its
    failed self-heal re-bind, to nats-py's ``ConnectionClosedError``.

    :return: the error
    :rtype: KvError
    """
    rebind = _chained(KvError("open KV bucket failed: bucket=tool_catalog"), nats.errors.ConnectionClosedError())
    return _chained(KvError("KV put failed: bucket=tool_catalog key=k: nats: connection closed"), rebind)


def _live_components(status: HealthStatus) -> dict[str, bool]:
    return {c.name: c.healthy for c in status.components}


async def _bound_catalog() -> tuple[ToolCatalog, FakeKvBucket]:
    bucket = FakeKvBucket(bucket_name="tool_catalog", storage="file", direct=True)
    catalog = ToolCatalog()
    await catalog.load_from_kv(bucket)
    return catalog, bucket


async def _failed_registration(catalog: ToolCatalog, bucket: FakeKvBucket, name: str) -> None:
    """register ``name`` and require it to fail with exactly the error the unreachable bucket raised."""
    with pytest.raises(NatsClientError) as raised:
        await catalog.register(_entry(name))
    assert raised.value is bucket.unreachable_error


@pytest.mark.asyncio
async def test_a_streak_of_failed_writes_makes_the_registry_not_ready_and_one_that_lands_ends_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog, bucket = await _bound_catalog()
    server = _health_server(catalog)
    assert (await server.get_status(HealthTier.READY)).healthy is True

    bucket.become_unreachable(KvError("nats: connection closed"))
    for attempt in range(WRITE_FAILURE_THRESHOLD - 1):
        await _failed_registration(catalog, bucket, f"tool.n{attempt}")
    assert catalog.persisting is True, "a failure short of the threshold is a blip, not an outage"
    assert (await server.get_status(HealthTier.READY)).healthy is True

    with caplog.at_level(logging.ERROR, logger="threetears.registry.catalog"):
        await _failed_registration(catalog, bucket, "tool.last")
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


@pytest.mark.parametrize(
    "outage",
    [
        pytest.param(lambda: _chained(KvError("KV put failed"), nats.errors.TimeoutError()), id="nats-timeout"),
        pytest.param(lambda: PublishTimeoutError("KV put timed out"), id="wrapper-publish-timeout"),
        pytest.param(lambda: _chained(KvError("KV put failed"), nats.errors.NoRespondersError()), id="no-responders"),
        pytest.param(lambda: KvBucketNotFoundError("absent", bucket="tool_catalog"), id="bucket-not-found"),
        # the message says "connection closed", but nothing beneath it is a closed connection: the
        # liveness rule reads the type, never the text.
        pytest.param(lambda: KvError("nats: connection closed"), id="message-text-only"),
    ],
)
@pytest.mark.asyncio
async def test_a_failure_that_is_not_a_closed_connection_is_never_a_liveness_failure(
    outage: Callable[[], NatsClientError],
) -> None:
    """a NATS outage fails catalog writes, and a restart does not fix an outage."""
    catalog, bucket = await _bound_catalog()
    bucket.become_unreachable(outage())
    for attempt in range(WRITE_FAILURE_THRESHOLD + 2):
        await _failed_registration(catalog, bucket, f"tool.n{attempt}")
    assert catalog.persisting is False, "the outage does take the registry out of rotation"

    live = await _health_server(catalog).get_status(HealthTier.LIVE)
    assert catalog.connection_usable is True
    assert live.healthy is True
    assert _live_components(live) == {"catalog_connection_usable": True}


@pytest.mark.asyncio
async def test_writes_failing_on_a_closed_connection_fail_liveness_and_say_a_restart_clears_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog, bucket = await _bound_catalog()
    server = _health_server(catalog)
    assert (await server.get_status(HealthTier.LIVE)).healthy is True

    bucket.become_unreachable(_closed_connection())
    for attempt in range(WRITE_FAILURE_THRESHOLD - 1):
        await _failed_registration(catalog, bucket, f"tool.n{attempt}")
    assert catalog.connection_usable is True, "a failure short of the threshold does not restart the registry"
    assert (await server.get_status(HealthTier.LIVE)).healthy is True

    with caplog.at_level(logging.ERROR, logger="threetears.registry.catalog"):
        await _failed_registration(catalog, bucket, "tool.last")
    assert catalog.connection_usable is False
    live = await server.get_status(HealthTier.LIVE)
    assert live.healthy is False
    assert _live_components(live) == {"catalog_connection_usable": False}
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("closed NATS connection" in message and "only a restart clears it" in message for message in errors), (
        errors
    )


@pytest.mark.asyncio
async def test_outage_failures_between_closed_connection_ones_neither_count_nor_reset_them() -> None:
    catalog, bucket = await _bound_catalog()
    for attempt in range(WRITE_FAILURE_THRESHOLD - 1):
        bucket.become_unreachable(_closed_connection())
        await _failed_registration(catalog, bucket, f"tool.closed{attempt}")
        bucket.become_unreachable(_chained(KvError("KV put failed"), nats.errors.TimeoutError()))
        await _failed_registration(catalog, bucket, f"tool.timeout{attempt}")
        assert catalog.connection_usable is True

    bucket.become_unreachable(_closed_connection())
    await _failed_registration(catalog, bucket, "tool.closed-last")

    assert catalog.connection_usable is False


@pytest.mark.asyncio
async def test_a_write_that_lands_makes_the_registry_live_again_and_restarts_the_count() -> None:
    catalog, bucket = await _bound_catalog()
    server = _health_server(catalog)
    bucket.become_unreachable(_closed_connection())
    for attempt in range(WRITE_FAILURE_THRESHOLD):
        await _failed_registration(catalog, bucket, f"tool.n{attempt}")
    assert (await server.get_status(HealthTier.LIVE)).healthy is False

    bucket.become_reachable()
    await catalog.register(_entry("tool.recovered"))
    assert catalog.connection_usable is True
    assert (await server.get_status(HealthTier.LIVE)).healthy is True

    # the count started again from zero: one short of the threshold is still live
    bucket.become_unreachable(_closed_connection())
    for attempt in range(WRITE_FAILURE_THRESHOLD - 1):
        await _failed_registration(catalog, bucket, f"tool.again{attempt}")
    assert catalog.connection_usable is True


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
    assert catalog.connection_usable is True
