"""a registry whose catalog writes keep failing reports itself not ready, and ready again once one lands.

A catalog that cannot write its bucket answers every registration ``CATALOG_UNAVAILABLE``, so it must
leave rotation rather than report itself healthy. ``RegistryServer`` wires
``HealthCheck(name="catalog_persisting", probe=lambda: catalog.persisting, tier=HealthTier.READY)``;
these drive that probe over a real :class:`ToolCatalog` writing to the shipped in-memory bucket,
through a real :class:`HealthServer`, the way ``test_readiness.py`` drives the registry's JWKS gate.

A failed catalog write is never a liveness failure, whatever its kind. An outage ends on its own and a
restart through one is a restart loop; a closed connection is already the registry's ``nats`` liveness
check to report, because the catalog's bucket writes through the client's current connection.
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
    "failure",
    [
        pytest.param(lambda: _chained(KvError("KV put failed"), nats.errors.TimeoutError()), id="nats-timeout"),
        pytest.param(lambda: PublishTimeoutError("KV put timed out"), id="wrapper-publish-timeout"),
        pytest.param(lambda: _chained(KvError("KV put failed"), nats.errors.NoRespondersError()), id="no-responders"),
        pytest.param(lambda: KvBucketNotFoundError("absent", bucket="tool_catalog"), id="bucket-not-found"),
        pytest.param(lambda: KvError("nats: connection closed"), id="connection-closed-message"),
        pytest.param(_closed_connection, id="closed-connection"),
    ],
)
@pytest.mark.asyncio
async def test_the_catalog_check_is_never_a_liveness_failure(failure: Callable[[], NatsClientError]) -> None:
    """a NATS outage fails catalog writes and a restart does not fix it; a closed connection is the
    ``nats`` check's to report, since the catalog follows the client's connection."""
    catalog, bucket = await _bound_catalog()
    bucket.become_unreachable(failure())
    for attempt in range(WRITE_FAILURE_THRESHOLD + 2):
        await _failed_registration(catalog, bucket, f"tool.n{attempt}")
    assert catalog.persisting is False, "the failure does take the registry out of rotation"

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


@pytest.mark.asyncio
async def test_one_failed_write_back_over_many_entries_counts_once() -> None:
    """a write-back writes every entry; against a dead bucket it is ONE failed operation, so a single
    pass over a large catalog does not take the registry out of rotation on its own."""
    catalog, bucket = await _bound_catalog()
    for index in range(WRITE_FAILURE_THRESHOLD + 2):
        await catalog.register(_entry(f"tool.n{index}"))
    bucket.become_unreachable(KvError("nats: connection closed"))

    failed = await catalog.restore_to_kv(bucket)
    assert len(failed) == WRITE_FAILURE_THRESHOLD + 2
    assert catalog.persisting is True

    for _ in range(WRITE_FAILURE_THRESHOLD - 1):
        await catalog.restore_to_kv(bucket)
    assert catalog.persisting is False, "failed passes in a row do"
