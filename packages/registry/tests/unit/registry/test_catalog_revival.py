"""a registry replica answers calls only once the copies its catalog warm-loaded are routable again.

A starting replica loads the persisted catalog with every copy unavailable, and each comes back only with its
pod's next heartbeat. On cobalt dev on 2026-10-30 a scheduled ENR refresh reached a new replica during a platform
rollout and was refused ``TOOL_UNAVAILABLE: tool enr.reload@1.0 has no available endpoints`` while every other
replica could route it. :meth:`ToolCatalog.loaded_copies_settled` is what the replica holds its calls and its
readiness on.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint

from .copy_entries import uniform_entry

_FULL = "enr.reload@1.0"


def _entry(*pods: str) -> CatalogEntry:
    """a catalog entry for the tool, served by ``pods``, as a bucket persists it."""
    return uniform_entry(
        tool_name="enr.reload",
        tool_version="1.0",
        full_name=_FULL,
        description="reload",
        input_schema={"type": "object", "properties": {}},
        endpoints=[ToolEndpoint(pod_id=pod, status="available") for pod in pods],
    )


def _bucket(entry: CatalogEntry) -> AsyncMock:
    """a bucket holding ``entry``, as a starting replica finds it."""
    kv = AsyncMock()
    kv.list_keys = AsyncMock(return_value=[entry.full_name])
    raw = json.dumps(entry.to_dict()).encode("utf-8")

    async def get(*, key: str) -> bytes | None:
        return raw if key == entry.full_name else None

    kv.get = get
    return kv


def test_a_catalog_that_loaded_nothing_waits_for_nothing() -> None:
    """Flips if a replica whose bucket was empty or unreachable holds its calls."""
    assert ToolCatalog().loaded_copies_settled(window=60.0) is True


@pytest.mark.asyncio
async def test_a_loaded_copy_holds_the_replica_until_its_pod_announces_again() -> None:
    """Flips if a replica answers calls while a copy it loaded is still unavailable, or keeps waiting once the
    pod's heartbeat has made it routable."""
    catalog = ToolCatalog()
    await catalog.load_from_kv(_bucket(_entry("pod-a", "pod-b")))
    assert catalog.loaded_copies_settled(window=60.0) is False
    catalog.mark_pod_endpoints_available("pod-a")
    assert catalog.loaded_copies_settled(window=60.0) is False
    catalog.mark_pod_endpoints_available("pod-b")
    assert catalog.loaded_copies_settled(window=60.0) is True


@pytest.mark.asyncio
async def test_a_copy_whose_pod_never_returns_is_waited_for_no_longer_than_the_window() -> None:
    """Flips if a pod that died keeps the replica out of rotation past one heartbeat round."""
    catalog = ToolCatalog()
    await catalog.load_from_kv(_bucket(_entry("pod-gone")))
    assert catalog.loaded_copies_settled(window=60.0) is False
    assert catalog.loaded_copies_settled(window=0.0) is True


@pytest.mark.asyncio
async def test_a_copy_swept_from_the_catalog_settles() -> None:
    """Flips if a copy the catalog no longer holds is waited for."""
    catalog = ToolCatalog()
    await catalog.load_from_kv(_bucket(_entry("pod-a")))
    catalog.mark_pod_endpoints_unavailable("pod-a")
    assert catalog.loaded_copies_settled(window=60.0) is False
    await catalog.deregister_pod("pod-a")  # the pod's row swept
    assert catalog.loaded_copies_settled(window=60.0) is True


@pytest.mark.asyncio
async def test_asking_whether_the_copies_settled_changes_nothing() -> None:
    """Flips if the readiness probe edits what the catalog waits on: a copy whose window passed is still named as
    unrouted, so the replica can say which pods never came back."""
    catalog = ToolCatalog()
    await catalog.load_from_kv(_bucket(_entry("pod-a", "pod-gone")))
    catalog.mark_pod_endpoints_available("pod-a")
    assert catalog.loaded_copies_settled(window=0.0) is True
    assert catalog.loaded_copies_settled(window=60.0) is False
    assert catalog.loaded_copies_pending() == {(_FULL, "pod-gone")}
