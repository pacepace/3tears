"""``ToolCatalog.restore_to_kv`` writes back what the catalog holds WHEN it writes, not what it held when it began.

The restore runs in the background after every reconnect, and awaits each write. A tool
deregistered while it runs has its key deleted by the deregistration; a restore working from a
snapshot taken before that would put the key back, and the shared bucket would advertise a tool no
replica holds -- read back by the next warm-loading registry.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

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


# parity-exempt: a raw nats-py KeyValue's put/delete over a dict, with a hook that runs mid-restore
class _FakeKeyValue:
    """a bucket whose first write lets a deregistration land, as one can while a restore awaits it."""

    def __init__(self) -> None:
        self.entries: dict[str, bytes] = {}
        self.on_first_put: Any = None

    async def put(self, key: str, value: bytes) -> int:
        hook, self.on_first_put = self.on_first_put, None
        if hook is not None:
            await hook()
        self.entries[key] = value
        return len(self.entries)

    async def delete(self, key: str) -> bool:
        self.entries.pop(key, None)
        return True


@pytest.mark.asyncio
async def test_a_tool_deregistered_during_the_restore_is_not_written_back() -> None:
    catalog = ToolCatalog()
    await catalog.register(_entry("threetears.calculator"))
    await catalog.register(_entry("threetears.clock"))
    kv = _FakeKeyValue()

    async def _deregister_the_other() -> None:
        await catalog.deregister("threetears.clock@1.0.0")

    kv.on_first_put = _deregister_the_other
    failed = await catalog.restore_to_kv(kv)

    assert failed == []
    stored = {json.loads(value)["full_name"] for value in kv.entries.values()}
    assert stored == {"threetears.calculator@1.0.0"}
