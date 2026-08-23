"""Bucket-ownership policy is a wiring decision, not a library constant (coll-task-04a KVC-04).

``BaseCollection._ensure_kv`` used to call ``kv_bucket`` with no
``create_if_missing`` at all, taking the library default. A deployment that wants
its pods to be readers of a bucket somebody else declares needs to say so, and
the place to say it is the registry the process wires -- not a literal in
``base.py``, which would bake one deployment's policy into the library and, for
the collections bucket specifically, would neuter the restart self-heal that
exists because a NATS restart on ephemeral storage wiped every bucket in
production.

The default therefore STAYS ``True``, and this file pins both halves: that the
default is unchanged, and that the flag actually reaches the open.
"""

from __future__ import annotations

import inspect
import json
import uuid
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities import BaseEntity


# parity-with: threetears.nats.kv.KvBucketLike
class _RecordingBucket:
    """Minimal bucket that records nothing; the client below is what is under test."""

    @property
    def name(self) -> str:
        return "recording"

    @property
    def ttl(self) -> timedelta | None:
        return None

    async def get(self, *, key: str) -> bytes | None:
        del key
        return None

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        del key
        return None

    async def put(self, *, key: str, value: bytes) -> int:
        del key, value
        return 1

    async def create(self, *, key: str, value: bytes) -> int | None:
        del key, value
        return 1

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None:
        del key, value, revision
        return 1

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        del key, revision
        return True


# parity-with: threetears.nats.kv.KvCapable
class _RecordingClient:
    """Records the ``create_if_missing`` every bucket open asked for."""

    def __init__(self) -> None:
        self.opens: list[bool] = []

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> _RecordingBucket:
        del name, ttl, storage, history
        self.opens.append(create_if_missing)
        return _RecordingBucket()


class _StubEntity(BaseEntity):
    """Minimal entity; nothing here reaches a durable tier."""

    id: str = ""


class _StubCollection(BaseCollection[_StubEntity]):
    """Concrete collection with no durable tier; only ``_ensure_kv`` is exercised."""

    def __init__(self, registry: CollectionRegistry, nats_client: Any) -> None:
        super().__init__(
            registry,
            DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
            nats_client,
        )

    @property
    def table_name(self) -> str:
        return "stub_rows"

    @property
    def entity_class(self) -> type[_StubEntity]:
        return _StubEntity

    async def fetch_from_store(self, entity_id: object) -> dict[str, Any] | None:
        del entity_id
        return None

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None) -> int:
        del data, original_timestamp
        return 1

    async def delete_from_store(self, entity_id: object) -> None:
        del entity_id

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(data)
        return loaded


def _wired(*, l2_create_if_missing: bool | None) -> tuple[_StubCollection, _RecordingClient]:
    client = _RecordingClient()
    registry = CollectionRegistry()
    registry.configure(
        l2_client=client,
        kv_key_scope=f"probe-{uuid.uuid4().hex[:8]}",
        l2_create_if_missing=l2_create_if_missing,
    )
    return _StubCollection(registry, client), client


class TestTheDefaultIsUnchanged:
    """Flipping it globally would break ~12 buckets that rely on first-use creation.

    It would also neuter ``NatsKvBucket._reopen``, which re-runs the opener with
    the stored flag and exists because a single-node NATS restart on ephemeral
    JetStream storage wiped every bucket and silenced the wake scheduler in
    production.
    """

    def test_the_library_default_is_still_create(self) -> None:
        from threetears.nats import NatsClient

        assert inspect.signature(NatsClient.kv_bucket).parameters["create_if_missing"].default is True

    def test_a_fresh_registry_creates(self) -> None:
        assert CollectionRegistry().l2_create_if_missing is True

    @pytest.mark.asyncio
    async def test_an_unconfigured_registry_opens_with_create(self) -> None:
        collection, client = _wired(l2_create_if_missing=None)
        await collection._ensure_kv()  # noqa: SLF001 - the resolution under test
        assert client.opens == [True]


class TestTheFlagReachesTheOpen:
    """The point of the flag: a reader process can say so, in ONE place.

    ``configure`` is where every other tier is wired, and the scope alongside it,
    so the ownership policy belongs there too -- one line per process rather than
    one argument threaded through every collection.
    """

    @pytest.mark.asyncio
    async def test_a_reader_registry_binds_without_creating(self) -> None:
        collection, client = _wired(l2_create_if_missing=False)
        await collection._ensure_kv()  # noqa: SLF001 - the resolution under test
        assert client.opens == [False]

    def test_it_merges_like_every_other_configure_argument(self) -> None:
        """Two-pass wiring is the normal shape at several call sites."""
        registry = CollectionRegistry()
        registry.configure(l2_create_if_missing=False)
        registry.configure(kv_key_scope="later")
        assert registry.l2_create_if_missing is False

    def test_base_py_carries_no_literal(self) -> None:
        """The anti-pattern this requirement names, checked at the source.

        A ``create_if_missing=False`` literal in ``base.py`` would be one
        deployment's policy compiled into the library.
        """
        from pathlib import Path

        source = Path(inspect.getfile(BaseCollection)).read_text(encoding="utf-8")
        assert "create_if_missing=False" not in source
        assert "create_if_missing=True" not in source
        assert "create_if_missing=self._registry.l2_create_if_missing" in source
