"""integration tests for HeartbeatCollection with shared L2 NATS KV.

validates the L1+L2 coherence contract that was introduced in
namespace-task-01 phase 8.5l-3:

- a save populates L1; ``get`` returns from L1 without hitting L3
  (``fetch_from_postgres`` raises by design, so any test that
  exercises pull-through implicitly asserts L3 was not reached).
- ``fetch_from_postgres`` raises on direct invocation (defensive
  guard -- L1+L2 only).
- cross-registry coherence: pod A writes; pod B's L1 miss resolves
  via L2 pull-through into pod B's L1. when pod A deletes, pod B's
  L1 is invalidated via the cross-pod invalidation subject and the
  next read returns ``None``.

the subscriber-flow tests exercise the two main orchestration
paths on :class:`HeartbeatSubscriber`: a heartbeat persists the
right entity through the Collection; a health-check sweep marks
the right pod unresponsive and removes its catalog endpoints.

no testcontainer is required: HeartbeatCollection is L1+L2 only,
so an in-process NATS bus mock suffices for L2 + invalidation.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import HeartbeatMessage
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.l1_cache import create_registry_l1_backend


# ---------------------------------------------------------------------------
# in-process NATS stand-in supporting both KV and pub/sub paths
# ---------------------------------------------------------------------------


class _InMemoryKvBucket:
    """typed-wrapper KV bucket stand-in matching :class:`NatsKvBucket`."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def get(self, *, key: str) -> bytes | None:
        return self._store.get(key)

    async def put(self, *, key: str, value: bytes) -> int:
        self._store[key] = value
        return len(self._store)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:  # noqa: ARG002
        existed = key in self._store
        self._store.pop(key, None)
        return existed or revision is None


class InMemoryNatsBus:
    """typed-wrapper NATS stand-in: KV (for L2) + typed pub/sub.

    every :class:`HeartbeatCollection` instance wired against the same
    bus shares the bucket and the invalidation subject, mirroring two
    registry processes reading one NATS cluster. no persistence, no
    JetStream queue groups -- just enough surface to exercise the
    L1+L2 contract.
    """

    def __init__(self) -> None:
        self._bucket = _InMemoryKvBucket()
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}

    async def kv_bucket(
        self,
        *,
        name: str,  # noqa: ARG002 -- single shared bucket suffices for tests
        ttl: Any = None,  # noqa: ARG002
        storage: str = "file",  # noqa: ARG002
        create_if_missing: bool = True,  # noqa: ARG002
        history: int = 1,  # noqa: ARG002
    ) -> _InMemoryKvBucket:
        return self._bucket

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:  # noqa: ARG002
        """typed publish: dispatch to every typed subscriber on ``subject``."""
        subject_str = str(subject)
        for cb, message_type in self._subscribers.get(subject_str, []):
            payload = message.model_dump_json()
            decoded = message_type.model_validate_json(payload)
            await cb(decoded)

    async def subscribe_typed(
        self,
        *,
        subject: Any,
        cb: Any,
        message_type: Any,
        queue: Any = None,  # noqa: ARG002
        max_in_flight: Any = None,  # noqa: ARG002
        deadletter_on_error: bool = True,  # noqa: ARG002
    ) -> None:
        subject_str = str(subject)
        self._subscribers.setdefault(subject_str, []).append((cb, message_type))


def _make_pod(
    nats: InMemoryNatsBus,
) -> tuple[HeartbeatCollection, CollectionRegistry]:
    """construct one registry-process pod's Collection + registry pair.

    :param nats: shared L2 bus
    :ptype nats: InMemoryNatsBus
    :return: (collection, registry) pair
    :rtype: tuple[HeartbeatCollection, CollectionRegistry]
    """
    l1 = create_registry_l1_backend()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = HeartbeatCollection(registry, config, nats_client=nats)
    return collection, registry


# ---------------------------------------------------------------------------
# L1 + L3 guards
# ---------------------------------------------------------------------------


class TestHeartbeatCollectionL1Only:
    """L1 population and L3 raise-loudly guards."""

    @pytest.mark.asyncio
    async def test_save_populates_l1(self) -> None:
        """save_entity puts the row into L1 and subsequent get hits L1."""
        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        now = datetime.now(UTC)
        entity = collection.create(
            {
                "pod_id": "pod-one",
                "date_last_heartbeat": now,
                "tools": ["t.a@1.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(entity)

        hit = await collection.get("pod-one")
        assert hit is not None
        assert hit.pod_id == "pod-one"
        assert hit.tools == ["t.a@1.0"]

    @pytest.mark.asyncio
    async def test_fetch_from_postgres_raises(self) -> None:
        """fetch_from_postgres raises -- L3 is intentionally off."""
        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        with pytest.raises(RuntimeError):
            await collection.fetch_from_postgres("pod-x")

    @pytest.mark.asyncio
    async def test_save_to_postgres_raises(self) -> None:
        """save_to_postgres raises -- L3 is intentionally off."""
        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        with pytest.raises(RuntimeError):
            await collection.save_to_postgres({"pod_id": "pod-x"})

    @pytest.mark.asyncio
    async def test_delete_from_postgres_raises(self) -> None:
        """delete_from_postgres raises -- L3 is intentionally off."""
        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        with pytest.raises(RuntimeError):
            await collection.delete_from_postgres("pod-x")


# ---------------------------------------------------------------------------
# L2 coherence across two pods
# ---------------------------------------------------------------------------


class TestHeartbeatCollectionL2Coherence:
    """cross-registry coherence via the shared L2 KV + invalidation subject."""

    @pytest.mark.asyncio
    async def test_l2_pull_through_on_cold_l1(self) -> None:
        """pod B's cold L1 reads pod A's saved row via L2 pull-through."""
        nats = InMemoryNatsBus()
        pod_a_collection, _ = _make_pod(nats)
        pod_b_collection, _ = _make_pod(nats)

        now = datetime.now(UTC)
        entity = pod_a_collection.create(
            {
                "pod_id": "pod-shared",
                "date_last_heartbeat": now,
                "tools": ["t.shared@1.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await pod_a_collection.save_entity(entity)

        hit = await pod_b_collection.get("pod-shared")
        assert hit is not None
        assert hit.pod_id == "pod-shared"
        assert hit.tools == ["t.shared@1.0"]
        assert hit.status == "healthy"

    @pytest.mark.asyncio
    async def test_cross_pod_invalidation(self) -> None:
        """pod A save publishes an invalidation envelope that pod B consumes.

        pod B warms its L1 with an older copy; pod A publishes a
        save; the invalidation listener on pod B evicts its L1; pod
        B's next get returns the freshly-saved row through L2
        pull-through.
        """
        nats = InMemoryNatsBus()
        pod_a_collection, _ = _make_pod(nats)
        pod_b_collection, pod_b_registry = _make_pod(nats)

        # pod B subscribes to the invalidation subject
        await pod_b_registry.start_invalidation_listener(nats)

        # seed both pods with an initial row
        now = datetime.now(UTC)
        seed = pod_a_collection.create(
            {
                "pod_id": "pod-inv",
                "date_last_heartbeat": now,
                "tools": ["t.old@1.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await pod_a_collection.save_entity(seed)
        # warm pod B's L1
        warmed = await pod_b_collection.get("pod-inv")
        assert warmed is not None
        assert warmed.tools == ["t.old@1.0"]

        # pod A mutates (new tools list) and saves; this publishes an
        # invalidation envelope that pod B's listener consumes
        mutated = await pod_a_collection.get("pod-inv")
        assert mutated is not None
        mutated.tools = ["t.new@2.0"]
        mutated.tools_count = 1
        await pod_a_collection.save_entity(mutated)

        # allow any queued invalidation callbacks to run
        await asyncio.sleep(0)

        # pod B's next read resolves via L2 pull-through and returns
        # the mutated row
        refreshed = await pod_b_collection.get("pod-inv")
        assert refreshed is not None
        assert refreshed.tools == ["t.new@2.0"]

    @pytest.mark.asyncio
    async def test_delete_clears_l2_and_notifies_peers(self) -> None:
        """pod A delete removes L2 row; pod B's next get returns None."""
        nats = InMemoryNatsBus()
        pod_a_collection, _ = _make_pod(nats)
        pod_b_collection, pod_b_registry = _make_pod(nats)
        await pod_b_registry.start_invalidation_listener(nats)

        now = datetime.now(UTC)
        entity = pod_a_collection.create(
            {
                "pod_id": "pod-del",
                "date_last_heartbeat": now,
                "tools": [],
                "tools_count": 0,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await pod_a_collection.save_entity(entity)
        warmed = await pod_b_collection.get("pod-del")
        assert warmed is not None

        await pod_a_collection.delete("pod-del")
        await asyncio.sleep(0)

        assert await pod_b_collection.get("pod-del") is None


# ---------------------------------------------------------------------------
# subscriber orchestration
# ---------------------------------------------------------------------------


class TestHeartbeatSubscriberFlow:
    """subscriber orchestration against a live HeartbeatCollection."""

    @pytest.mark.asyncio
    async def test_heartbeat_saves_entity_through_collection(self) -> None:
        """an incoming heartbeat message persists via the Collection."""
        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        catalog = ToolCatalog()

        endpoint = ToolEndpoint(pod_id="pod-sub", status="unavailable")
        entry = CatalogEntry(
            tool_name="threetears.sub_tool",
            tool_version="1.0.0",
            full_name="threetears.sub_tool@1.0.0",
            description="test tool",
            input_schema={"type": "object", "properties": {}},
            endpoints=[endpoint],
        )
        await catalog.register(entry)

        subscriber = HeartbeatSubscriber(
            catalog,
            collection,
            namespace="test",
            check_interval=100.0,
            timeout=30.0,
        )
        mock_nc = AsyncMock()
        mock_sub = AsyncMock()
        mock_nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(mock_nc)

        heartbeat = HeartbeatMessage(
            pod_id="pod-sub",
            timestamp=datetime.now(UTC).isoformat(),
            tools_count=1,
        )
        msg = MagicMock()
        msg.data = heartbeat.model_dump_json().encode("utf-8")
        await subscriber.handle_heartbeat(msg)

        saved = await collection.get("pod-sub")
        assert saved is not None
        assert saved.pod_id == "pod-sub"
        assert saved.status == "healthy"
        assert saved.tools == ["threetears.sub_tool@1.0.0"]
        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_sweep_marks_unresponsive_and_evicts(self) -> None:
        """sweep marks stale pod unresponsive and evicts catalog endpoints."""
        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        catalog = ToolCatalog()
        endpoint = ToolEndpoint(pod_id="pod-stale", status="available")
        entry = CatalogEntry(
            tool_name="threetears.calc",
            tool_version="1.0.0",
            full_name="threetears.calc@1.0.0",
            description="test",
            input_schema={"type": "object", "properties": {}},
            endpoints=[endpoint],
        )
        await catalog.register(entry)

        subscriber = HeartbeatSubscriber(
            catalog,
            collection,
            namespace="test",
            check_interval=100.0,
            timeout=30.0,
        )

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        stale_entity = collection.create(
            {
                "pod_id": "pod-stale",
                "date_last_heartbeat": stale_time,
                "tools": ["threetears.calc@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(stale_entity)
        subscriber.track_pod("pod-stale")

        await subscriber.run_health_check()

        # pod removed from subscriber's known-set and Collection
        assert "pod-stale" not in subscriber.known_pod_ids
        assert await collection.get("pod-stale") is None
        # catalog entry gone (only endpoint was pod-stale)
        assert catalog.get("threetears.calc@1.0.0") is None

    @pytest.mark.asyncio
    async def test_invalidation_envelope_carries_pod_id(self) -> None:
        """the invalidation envelope on save uses the pod_id as ids[0]."""
        from threetears.core.collections.registry import (
            CacheInvalidationMessage,
        )
        from threetears.nats import Subjects

        nats = InMemoryNatsBus()
        collection, _ = _make_pod(nats)
        captured: list[CacheInvalidationMessage] = []

        async def _capture(message: CacheInvalidationMessage) -> None:
            captured.append(message)

        await nats.subscribe_typed(
            subject=Subjects.cache_invalidate(),
            cb=_capture,
            message_type=CacheInvalidationMessage,
        )

        entity = collection.create(
            {
                "pod_id": "pod-env",
                "date_last_heartbeat": datetime.now(UTC),
                "tools": [],
                "tools_count": 0,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(entity)

        assert len(captured) == 1
        assert captured[0].table == "pod_heartbeats"
        assert captured[0].ids == ["pod-env"]
