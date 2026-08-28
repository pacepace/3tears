"""Two replicas of one tool pod, one live bus, and a mapping that crosses between them.

This is the property a unit test over a permission table cannot show, and the reason the
chunk exists: a tool pod's L2 grant is only worth having if a write on one replica is
actually visible to the next one, and if a change on one replica actually drops the copy
the other is serving. Both halves run here against a real NATS server.

**What each half proves, and why the first is not decoration.** Replica B reading a
mapping it never wrote proves the shared ``{ns}-collections`` bucket is reachable and
scoped the same way from both connections -- without it, "B saw the new value" would be
satisfied just as well by a B that read nothing and re-resolved. Replica B then LOSING its
cached copy when A rewrites the mapping proves the invalidation broadcast lands: nothing
else evicts an L1 row here, so a B that never hears the broadcast serves the first value
until the process dies.

Replica A is wired through the production builder
(:func:`~threetears.agent.tools.bootstrap.build_tool_pod_collection_stack`), so the bucket
bind, the ``tool_pods.id`` key scope and the invalidation listener are the real ones rather
than a test's approximation of them. Both replicas carry the SAME pod id, because that is
what two replicas of one deployment are.

Requires docker; gated by ``pytest.mark.integration``, which the default gate excludes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import MetaData

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.media.contracts import ObjectHandle
from threetears.nats import NatsClient, set_default_namespace

from threetears.agent.tools.bootstrap import build_tool_pod_collection_stack
from threetears.agent.tools.object_resolution_collection import (
    OBJECT_RESOLUTIONS_TABLE,
    ObjectResolutionCollection,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

_NAMESPACE = "3tears"
#: one deployment, so one ``tool_pods.id`` and one key scope for both replicas.
_POD_ID = "01947100-0000-7000-8000-0000000000aa"
_CUSTOMER = uuid.UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OBJECT = uuid.UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_FIRST_KEY = "cust/conv/first.md"
_SECOND_KEY = "cust/conv/second.md"


def _handle(s3_key: str) -> ObjectHandle:
    """build the handle a hub resolve would have produced.

    :param s3_key: the stored key the mapping points at
    :ptype s3_key: str
    :return: the handle
    :rtype: ObjectHandle
    """
    return ObjectHandle(
        object_id=_OBJECT,
        s3_key=s3_key,
        mime_type="text/markdown",
        size_bytes=11,
        summary=None,
        category=None,
    )


# session loop scope, matching the tests below: an async fixture left at the default
# function scope is set up on a DIFFERENT event loop from the test that consumes it, and a
# NATS client whose reader task lives on one loop while the test awaits on another does not
# fail -- it hangs.
@pytest_asyncio.fixture(loop_scope="session")
async def bus(nats_container: str) -> AsyncIterator[tuple[NatsClient, NatsClient]]:
    """one connection per replica, with the shared bucket already declared.

    The declaring third client stands in for the hub: a tool pod BINDS the collections
    bucket and never declares it (its grant carries ``STREAM.INFO`` and no ``CREATE``), so
    without a declaration first, replica A's production wiring path would spend its whole
    retry budget and raise -- which is the correct production behaviour and useless here.

    :param nats_container: the container fixture's connection url
    :ptype nats_container: str
    :return: async iterator yielding both replicas' clients
    :rtype: AsyncIterator[tuple[NatsClient, NatsClient]]
    """
    set_default_namespace(_NAMESPACE)
    declarer = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace=_NAMESPACE,
        client_name="object-resolution-declarer",
    )
    await declarer.ensure_kv_bucket(name="collections", create_if_missing=True)
    first = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace=_NAMESPACE,
        client_name="object-resolution-replica-a",
    )
    second = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace=_NAMESPACE,
        client_name="object-resolution-replica-b",
    )
    try:
        yield (first, second)
    finally:
        await first.shutdown()
        await second.shutdown()
        await declarer.shutdown()


async def _replica(client: NatsClient, label: str) -> tuple[CollectionRegistry, ObjectResolutionCollection]:
    """wire one replica exactly as a running tool pod does.

    The ``l1_db_name`` override is what makes two replicas possible inside ONE test
    process: the L1 database is named so that every collection in a process shares one
    tier, so left at the default both replicas here would read and write the same L1 and
    the L2 hop this file exists to prove would never happen.

    :param client: this replica's own connection
    :ptype client: NatsClient
    :param label: distinguishes the replica's L1 database
    :ptype label: str
    :return: the replica's registry and its resolution collection
    :rtype: tuple[CollectionRegistry, ObjectResolutionCollection]
    """
    registry = await build_tool_pod_collection_stack(
        nats_client=client,
        pod_id=_POD_ID,
        l1_metadata=MetaData(),
        l1_db_name=f"tool_pod_l1_{label}_{uuid.uuid4().hex[:8]}",
    )
    collection = ObjectResolutionCollection(registry, DefaultCoreConfig(), client)
    return registry, collection


def _l1_row(registry: CollectionRegistry, table: str = OBJECT_RESOLUTIONS_TABLE) -> dict[str, Any] | None:
    """read the mapping row straight out of one replica's L1, bypassing every tier above.

    :param registry: the replica's registry
    :ptype registry: CollectionRegistry
    :param table: the table to read
    :ptype table: str
    :return: the row, or ``None`` when this replica holds no local copy
    :rtype: dict[str, Any] | None
    """
    l1 = registry.get_l1_backend(table)
    assert l1 is not None
    row: dict[str, Any] | None = l1.select_by_id(
        table,
        (str(_CUSTOMER), str(_OBJECT)),
        ("customer_id", "object_id"),
    )
    return row


async def _wait_until_evicted(registry: CollectionRegistry, timeout: float = 10.0) -> None:
    """poll until the mapping leaves this replica's L1, or fail saying what that means.

    :param registry: the replica's registry
    :ptype registry: CollectionRegistry
    :param timeout: seconds to wait before giving up
    :ptype timeout: float
    :return: nothing
    :rtype: None
    :raises AssertionError: if the row is still cached when the timeout lapses
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while _l1_row(registry) is not None:
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(
                f"the peer replica still holds its own L1 copy after {timeout}s -- "
                "the invalidation broadcast never landed, so it would serve the old mapping forever"
            )
        await asyncio.sleep(0.05)


class TestCrossPodResolutionCoherence:
    """the two halves: a mapping crosses, and a change drops the copy that crossed."""

    async def test_a_write_on_one_replica_reaches_and_then_invalidates_the_other(
        self,
        bus: tuple[NatsClient, NatsClient],
    ) -> None:
        """B reads A's mapping out of L2, then loses that copy the moment A rewrites it.

        :param bus: one connected client per replica
        :ptype bus: tuple[NatsClient, NatsClient]
        :return: nothing
        :rtype: None
        """
        client_a, client_b = bus
        registry_a, replica_a = await _replica(client_a, "a")
        registry_b, replica_b = await _replica(client_b, "b")
        # subscribe -> flush -> publish. nats-py does not flush on subscribe, so without the
        # barrier B's SUB can reach the server after the PUB it was meant to catch, and the
        # test fails on a race rather than on the behaviour it asserts.
        await client_a.flush()
        await client_b.flush()
        try:
            await replica_a.remember(_CUSTOMER, _handle(_FIRST_KEY))

            # half one: B never wrote this and holds no L1 copy, so the only way it can
            # answer is the shared L2 key.
            assert _l1_row(registry_b) is None
            first = await replica_b.lookup(_CUSTOMER, _OBJECT)
            assert first is not None
            assert first.s3_key == _FIRST_KEY
            assert _l1_row(registry_b) is not None, "the L2 read must fill B's L1, or half two proves nothing"

            # half two: A rewrites the mapping. Nothing but the broadcast can dislodge the
            # copy B just cached.
            await replica_a.remember(_CUSTOMER, _handle(_SECOND_KEY))
            await _wait_until_evicted(registry_b)

            second = await replica_b.lookup(_CUSTOMER, _OBJECT)
            assert second is not None
            assert second.s3_key == _SECOND_KEY

            # and the writer keeps its own copy: a registry ignores its own broadcast, so A
            # does not evict the row it just wrote.
            assert _l1_row(registry_a) is not None
        finally:
            await registry_a.stop_invalidation_listener()
            await registry_b.stop_invalidation_listener()

    async def test_a_forget_on_one_replica_removes_the_mapping_everywhere(
        self,
        bus: tuple[NatsClient, NatsClient],
    ) -> None:
        """the delete half: B must stop answering, not merely answer something else.

        :param bus: one connected client per replica
        :ptype bus: tuple[NatsClient, NatsClient]
        :return: nothing
        :rtype: None
        """
        client_a, client_b = bus
        registry_a, replica_a = await _replica(client_a, "a")
        registry_b, replica_b = await _replica(client_b, "b")
        await client_a.flush()
        await client_b.flush()
        try:
            await replica_a.remember(_CUSTOMER, _handle(_FIRST_KEY))
            assert await replica_b.lookup(_CUSTOMER, _OBJECT) is not None

            await replica_a.forget(_CUSTOMER, _OBJECT)
            await _wait_until_evicted(registry_b)

            assert await replica_b.lookup(_CUSTOMER, _OBJECT) is None
        finally:
            await registry_a.stop_invalidation_listener()
            await registry_b.stop_invalidation_listener()
