"""integration fixtures: real-NATS two-pod presence wiring.

builds two independent ``PresenceCollection`` / ``RoomState`` /
``PresenceSweeper`` triples over ONE NATS testcontainer — two "pods"
reading one NATS cluster — so the cross-pod proofs exercise the real L2
KV + invalidation envelope, not an in-memory stand-in.

the session-scoped ``nats_container`` fixture comes from the canonical
harness (``threetears.core.testing.fixtures``, wired via the
workspace-root ``conftest.py``); a checkout without docker skips
cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pytest

from threetears.channels.presence.collection import PresenceCollection
from threetears.channels.presence.fanout import RoomFanout
from threetears.channels.presence.l1_cache import create_presence_l1_backend
from threetears.channels.presence.room_state import RoomState
from threetears.channels.presence.sweeper import PresenceSweeper
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.nats import NatsClient, set_default_namespace

_NAMESPACE = "channelspresenceit"


@dataclass
class Pod:
    """one pod's presence wiring over the shared NATS container."""

    client: NatsClient
    registry: CollectionRegistry
    collection: PresenceCollection
    state: RoomState
    sweeper: PresenceSweeper
    fanout: RoomFanout


async def _make_pod(nats_url: str, pod_id: str) -> Pod:
    """connect a NatsClient and build the presence stack for one pod.

    starts the registry's invalidation listener so a peer pod's write
    evicts this pod's stale L1 copy — the cross-pod coherence path.
    """
    client = await NatsClient.connect(
        nats_url=nats_url,
        nats_subject_namespace=_NAMESPACE,
        client_name=f"presence-{pod_id}",
    )
    l1 = create_presence_l1_backend()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=client)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = PresenceCollection(registry, config, nats_client=client)
    await registry.start_invalidation_listener(client)
    state = RoomState(collection, pod_id=pod_id)
    sweeper = PresenceSweeper(collection, check_interval=100.0, timeout=30.0)
    fanout = RoomFanout(state, client)
    return Pod(
        client=client,
        registry=registry,
        collection=collection,
        state=state,
        sweeper=sweeper,
        fanout=fanout,
    )


@pytest.fixture
async def two_pods(nats_container: str) -> AsyncIterator[tuple[Pod, Pod]]:
    """two pods (A, B) over one real NATS container; cleaned up on teardown."""
    set_default_namespace(_NAMESPACE)
    pod_a = await _make_pod(nats_container, "pod-a")
    pod_b = await _make_pod(nats_container, "pod-b")
    try:
        yield pod_a, pod_b
    finally:
        await pod_a.client.shutdown()
        await pod_b.client.shutdown()


@pytest.fixture
def room_key() -> Callable[[str, str], str]:
    """build a per-test-unique ``{customer}:{story}:{branch}:{file}`` room id.

    the NATS container is session-scoped, so its KV bucket persists
    across tests; a unique story segment per test keeps each test's room
    rows disjoint so they cannot bleed into one another.

    :return: ``room_key(customer, file) -> room_id`` builder
    :rtype: Callable[[str, str], str]
    """
    suffix = uuid.uuid4().hex[:12]

    def _build(customer: str, file: str) -> str:
        return f"{customer}:story-{suffix}:main:{file}"

    return _build
