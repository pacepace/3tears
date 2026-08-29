"""The tool-pod runtime's first two-tier collection, and the cache it replaces.

``ObjectResolutionCollection`` is L1+L2 with no L3, in the shipped shape: the pool is
forced to ``None`` and the three store methods raise rather than silently no-op. The
raise is the point. A tool pod cannot reach L3 at all yet -- the broker reads the
principal off a hub-minted token and a tool pod holds none -- so a collection that
quietly accepted a durable write would report success for a row nobody stored, which
is the defect shape this whole line of work keeps finding.

The second half of the file is the payload: ``HubObjectResolver`` kept a hand-rolled
``dict`` with FIFO eviction, pod-local, so every replica paid its own hub round trip
for a mapping a sibling had already resolved. Handed the collection it uses that
instead, and the two replicas share one L2 key.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.media.contracts import ObjectHandle

from threetears.agent.tools.l1_cache import create_tool_pod_l1_backend
from threetears.agent.tools.object_resolution_collection import (
    OBJECT_RESOLUTIONS_TABLE,
    ObjectResolutionCollection,
)
from threetears.agent.tools.object_resolver import (
    HubObjectResolver,
    ObjectResolveRequestModel,
    ObjectResolveResponseModel,
)

_SCOPE = "01947100-0000-7000-8000-0000000000aa"
_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OTHER_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-2222aaaa2222")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_KEY = f"{_CUSTOMER}/conversation-x/reports/2026/06/30/{_OBJECT}/report.md"
_TOKEN = "hub.identity.token.value"


# named ``_Recording`` / ``_InMemory`` rather than ``_Fake`` for the reason
# ``test_object_resolver.py`` already records: the fake-parity walker targets
# ``Fake<Name>`` and would demand a full mirror of a protocol these shims serve three
# methods of.
class _InMemoryKvBucket:
    """the two-call KV surface ``BaseCollection`` reaches L2 through."""

    def __init__(self, store: dict[str, bytes]) -> None:
        self.store = store

    async def get(self, *, key: str) -> bytes | None:
        return self.store.get(key)

    async def put(self, *, key: str, value: bytes) -> int:
        self.store[key] = value
        return len(self.store)

    async def delete(self, *, key: str) -> bool:
        return self.store.pop(key, None) is not None


class _RecordingNats:
    """a NATS client stand-in that hands out one shared in-memory bucket."""

    def __init__(self, store: dict[str, bytes]) -> None:
        self.store = store
        self.published: list[Any] = []

    async def kv_bucket(self, *, name: str, create_if_missing: bool) -> _InMemoryKvBucket:
        del name, create_if_missing
        return _InMemoryKvBucket(self.store)

    async def publish(self, *, subject: Any, message: Any) -> None:
        self.published.append(message)


def _replica(store: dict[str, bytes], nats: _RecordingNats) -> ObjectResolutionCollection:
    """wire one pod replica's registry and return its resolution collection.

    Two calls with the SAME ``store`` are two replicas of one pod: they share the L2
    bucket and the key scope (``tool_pods.id`` is per deployment), and hold separate
    L1 databases, which is exactly the arrangement cross-pod coherence has to work in.

    The distinct ``db_name`` is what makes that true and is not incidental.
    :func:`create_tool_pod_l1_backend` names its in-memory database, and
    :class:`SQLiteBackend` keys its connection on that name -- deliberately, so every
    collection in ONE process shares one L1 tier. Left at the default, both "replicas"
    here would share one L1 as well, and an L2 round trip this file claims to prove
    would be served out of the writer's own L1 without touching the bucket at all.

    :param store: the shared L2 backing dict
    :ptype store: dict[str, bytes]
    :param nats: the client whose bucket handle reaches that store
    :ptype nats: _RecordingNats
    :return: a collection wired L1 + L2, no L3
    :rtype: ObjectResolutionCollection
    """
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=create_tool_pod_l1_backend(MetaData(), db_name=f"replica_{uuid4().hex}"),
        kv_key_scope=_SCOPE,
        l2_create_if_missing=False,
    )
    return ObjectResolutionCollection(registry, DefaultCoreConfig(), nats)  # type: ignore[arg-type]


def _handle(s3_key: str = _KEY, mime: str = "text/markdown", size: int = 42) -> ObjectHandle:
    """build the handle a successful resolve produces."""
    return ObjectHandle(
        object_id=_OBJECT,
        s3_key=s3_key,
        mime_type=mime,
        size_bytes=size,
        summary=None,
        category=None,
    )


class TestTheL3MethodsRefuse:
    """L3 is not reachable from a tool pod, so the store methods must say so."""

    async def test_fetch_from_store_raises(self) -> None:
        collection = _replica({}, _RecordingNats({}))
        with pytest.raises(RuntimeError, match="L1\\+L2 only"):
            await collection.fetch_from_store((str(_CUSTOMER), str(_OBJECT)))

    async def test_save_to_store_raises(self) -> None:
        collection = _replica({}, _RecordingNats({}))
        with pytest.raises(RuntimeError, match="L1\\+L2 only"):
            await collection.save_to_store({"customer_id": str(_CUSTOMER), "object_id": str(_OBJECT)})

    async def test_delete_from_store_raises(self) -> None:
        collection = _replica({}, _RecordingNats({}))
        with pytest.raises(RuntimeError, match="L1\\+L2 only"):
            await collection.delete_from_store((str(_CUSTOMER), str(_OBJECT)))

    async def test_the_pool_is_forced_off_even_when_the_registry_offers_one(self) -> None:
        """a registry default L3 pool must not be inherited: the raise is the contract."""
        registry = CollectionRegistry()
        registry.configure(
            l1_backend=create_tool_pod_l1_backend(MetaData()),
            l3_pool=object(),  # type: ignore[arg-type]
            kv_key_scope=_SCOPE,
            l2_create_if_missing=False,
        )
        collection = ObjectResolutionCollection(registry, DefaultCoreConfig(), _RecordingNats({}))  # type: ignore[arg-type]
        assert collection.l3_pool is None


class TestTheL1TableIsRegisteredByTheCacheModule:
    """the runtime's own table is mirrored whatever the host pod declared."""

    def test_a_pod_that_declares_nothing_still_gets_the_table(self) -> None:
        backend = create_tool_pod_l1_backend(MetaData())
        assert backend.has_table(OBJECT_RESOLUTIONS_TABLE) is True

    def test_the_pod_s_own_tables_survive_beside_it(self) -> None:
        metadata = MetaData()
        Table("widgets", metadata, Column("id", String(64), primary_key=True))
        backend = create_tool_pod_l1_backend(metadata)
        assert backend.has_table("widgets") is True
        assert backend.has_table(OBJECT_RESOLUTIONS_TABLE) is True

    def test_registering_twice_over_one_metadata_is_not_an_error(self) -> None:
        """the host may hold one MetaData for the life of the process."""
        metadata = MetaData()
        create_tool_pod_l1_backend(metadata)
        backend = create_tool_pod_l1_backend(metadata)
        assert backend.has_table(OBJECT_RESOLUTIONS_TABLE) is True


class TestRememberAndLookup:
    """the two-tier read path, and what each tier contributes."""

    async def test_a_remembered_handle_comes_back(self) -> None:
        collection = _replica({}, _RecordingNats({}))
        await collection.remember(_CUSTOMER, _handle())
        found = await collection.lookup(_CUSTOMER, _OBJECT)
        assert found == _handle()

    async def test_an_unknown_object_is_a_miss_not_a_raise(self) -> None:
        collection = _replica({}, _RecordingNats({}))
        assert await collection.lookup(_CUSTOMER, _OBJECT) is None

    async def test_another_customer_does_not_see_it(self) -> None:
        """the pk carries the VERIFIED customer, so no cross-tenant reuse."""
        collection = _replica({}, _RecordingNats({}))
        await collection.remember(_CUSTOMER, _handle())
        assert await collection.lookup(_OTHER_CUSTOMER, _OBJECT) is None

    async def test_a_second_replica_reads_it_out_of_l2(self) -> None:
        """replica B never wrote the row and holds its own empty L1."""
        store: dict[str, bytes] = {}
        nats = _RecordingNats(store)
        await _replica(store, nats).remember(_CUSTOMER, _handle())
        assert await _replica(store, nats).lookup(_CUSTOMER, _OBJECT) == _handle()

    async def test_the_l2_read_fills_l1_so_the_next_one_is_local(self) -> None:
        store: dict[str, bytes] = {}
        nats = _RecordingNats(store)
        await _replica(store, nats).remember(_CUSTOMER, _handle())
        reader = _replica(store, nats)
        assert await reader.lookup(_CUSTOMER, _OBJECT) == _handle()
        store.clear()
        assert await reader.lookup(_CUSTOMER, _OBJECT) == _handle()

    async def test_forget_clears_both_tiers(self) -> None:
        store: dict[str, bytes] = {}
        nats = _RecordingNats(store)
        writer = _replica(store, nats)
        await writer.remember(_CUSTOMER, _handle())
        await writer.forget(_CUSTOMER, _OBJECT)
        assert await writer.lookup(_CUSTOMER, _OBJECT) is None
        assert store == {}

    async def test_a_write_announces_itself_to_the_other_replicas(self) -> None:
        """the invalidation publish is what evicts a peer's L1 copy."""
        nats = _RecordingNats({})
        await _replica({}, nats).remember(_CUSTOMER, _handle())
        assert [m.table for m in nats.published] == [OBJECT_RESOLUTIONS_TABLE]
        assert nats.published[0].ids == [str(_CUSTOMER), str(_OBJECT)]

    async def test_a_forget_announces_itself_too(self) -> None:
        nats = _RecordingNats({})
        collection = _replica({}, nats)
        await collection.remember(_CUSTOMER, _handle())
        nats.published.clear()
        await collection.forget(_CUSTOMER, _OBJECT)
        assert [m.table for m in nats.published] == [OBJECT_RESOLUTIONS_TABLE]


class _RecordingResolveNats:
    """records resolve requests and returns one queued reply."""

    def __init__(self, response: ObjectResolveResponseModel) -> None:
        self._response = response
        self.requests: list[ObjectResolveRequestModel] = []

    async def request(self, *, subject: object, message: object, response_type: object, timeout: object) -> object:
        del subject, response_type, timeout
        assert isinstance(message, ObjectResolveRequestModel)
        self.requests.append(message)
        return self._response


def _ok() -> ObjectResolveResponseModel:
    """build a success resolve reply."""
    return ObjectResolveResponseModel(success=True, s3_key=_KEY, mime_type="text/markdown", size_bytes=42)


class TestTheResolverUsesTheCollection:
    """the payload: a resolution one replica paid for serves every replica."""

    async def test_a_repeat_resolve_asks_the_hub_once(self) -> None:
        nc = _RecordingResolveNats(_ok())
        resolver = HubObjectResolver(nc, request_timeout_seconds=5.0, resolution_cache=_replica({}, _RecordingNats({})))  # type: ignore[arg-type]
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
        assert len(nc.requests) == 1

    async def test_a_sibling_replica_does_not_pay_the_round_trip(self) -> None:
        """the whole reason the cache moved out of a process-local dict."""
        store: dict[str, bytes] = {}
        nats = _RecordingNats(store)
        first = _RecordingResolveNats(_ok())
        await HubObjectResolver(
            first,  # type: ignore[arg-type]
            request_timeout_seconds=5.0,
            resolution_cache=_replica(store, nats),
        ).resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)

        second = _RecordingResolveNats(_ok())
        handle = await HubObjectResolver(
            second,  # type: ignore[arg-type]
            request_timeout_seconds=5.0,
            resolution_cache=_replica(store, nats),
        ).resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
        assert second.requests == []
        assert handle.s3_key == _KEY

    async def test_another_customer_still_asks_the_hub(self) -> None:
        nc = _RecordingResolveNats(_ok())
        resolver = HubObjectResolver(nc, request_timeout_seconds=5.0, resolution_cache=_replica({}, _RecordingNats({})))  # type: ignore[arg-type]
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
        await resolver.resolve(_OBJECT, customer_id=_OTHER_CUSTOMER, identity_token=_TOKEN)
        assert len(nc.requests) == 2

    async def test_without_a_cache_the_in_process_dict_still_serves(self) -> None:
        """a pod that declared no collection tables keeps the historical behaviour."""
        nc = _RecordingResolveNats(_ok())
        resolver = HubObjectResolver(nc, request_timeout_seconds=5.0)  # type: ignore[arg-type]
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token=_TOKEN)
        assert len(nc.requests) == 1
