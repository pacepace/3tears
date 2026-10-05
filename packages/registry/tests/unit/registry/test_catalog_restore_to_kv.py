"""``ToolCatalog.restore_to_kv`` writes back what the catalog holds WHEN it writes, not what it held when it began.

The write-back runs in the background whenever the bucket is created again, and awaits each write.
A tool deregistered while it runs has its key deleted by the deregistration; a write-back working
from a snapshot taken before that would put the key back, and the shared bucket would advertise a
tool no replica holds -- read back by the next warm-loading registry.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import timedelta

import pytest

from threetears.core.testing.kv import FakeKvBucket
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint

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


# parity-with: threetears.nats.kv.KvBucketLike
class _FakeBucketPausingOnFirstPut(FakeKvBucket):
    """the shipped in-memory bucket, whose first write lets a deregistration land, as one can while a restore awaits it."""

    def __init__(self) -> None:
        super().__init__(bucket_name="tool_catalog", storage="file", direct=True)
        self.on_first_put: Callable[[], Awaitable[None]] | None = None

    async def put(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int:
        hook, self.on_first_put = self.on_first_put, None
        if hook is not None:
            await hook()
        return await super().put(key=key, value=value, ttl=ttl)


@pytest.mark.asyncio
async def test_a_tool_deregistered_during_the_restore_is_not_written_back() -> None:
    catalog = ToolCatalog()
    await catalog.register(_entry("threetears.calculator"))
    await catalog.register(_entry("threetears.clock"))
    kv = _FakeBucketPausingOnFirstPut()

    async def _deregister_the_other() -> None:
        await catalog.deregister("threetears.clock@1.0.0")

    kv.on_first_put = _deregister_the_other
    failed = await catalog.restore_to_kv(kv)

    assert failed == []
    stored = set()
    for key in await kv.list_keys():
        value = await kv.get(key=key)
        assert value is not None
        stored.add(json.loads(value)["full_name"])
    assert stored == {"threetears.calculator@1.0.0"}
