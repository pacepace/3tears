"""Tests for CollectionRegistry."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.registry import CollectionRegistry
from threetears.nats import NatsClient


def _make_mock_collection(table_name: str) -> MagicMock:
    coll = MagicMock()
    coll.table_name = table_name
    return coll


def _routes_to(resolved: Any, pool: Any) -> bool:
    """whether a resolved L3 backend sends its SQL to ``pool``.

    ``configure`` / ``bind_table`` / ``register`` normalize a raw L3 transport (a
    bare pool) to a :class:`SqlL3Backend` so the resolved backend exposes the
    structured ``DurableStore`` ops the collection CRUD lifecycle needs (L3B-03). A
    backend that already satisfies ``DurableStore`` passes through un-wrapped. These
    routing/isolation/override tests assert WHICH pool reaches WHICH table, so this
    asks the wrapper the way production code does: through its raw-SQL escape hatch,
    which forwards every attribute it does not define to the pool it wraps.

    :param resolved: the value returned by ``get_l3_pool``.
    :ptype resolved: Any
    :param pool: the pool the test configured for that table.
    :ptype pool: Any
    :return: ``True`` when ``resolved`` is ``pool`` or a wrapper forwarding to it.
    :rtype: bool
    """
    if resolved is pool:
        return True
    return isinstance(resolved, SqlL3Backend) and resolved.copy_records_to_table is pool.copy_records_to_table


class TestCollectionRegistry:
    """Tests for CollectionRegistry."""

    def test_register_and_lookup(self) -> None:
        registry = CollectionRegistry()
        coll = _make_mock_collection("users")
        registry.register(coll)

        assert registry.get_collection("users") is coll

    def test_get_collection_returns_none_for_unregistered(self) -> None:
        registry = CollectionRegistry()

        assert registry.get_collection("nonexistent") is None

    def test_configure_sets_defaults(self) -> None:
        registry = CollectionRegistry()
        l1 = MagicMock()
        l2 = MagicMock()
        l3 = MagicMock()

        registry.configure(l1_backend=l1, l2_client=l2, l3_pool=l3, kv_key_scope="hub")

        assert registry.get_l1_backend("any_table") is l1
        assert registry.get_l2_client("any_table") is l2
        assert _routes_to(registry.get_l3_pool("any_table"), l3)

    def test_get_l1_backend_returns_default(self) -> None:
        registry = CollectionRegistry()
        default_l1 = MagicMock()
        registry.configure(l1_backend=default_l1)

        assert registry.get_l1_backend("messages") is default_l1

    def test_per_collection_override(self) -> None:
        registry = CollectionRegistry()
        default_l1 = MagicMock()
        override_l1 = MagicMock()
        registry.configure(l1_backend=default_l1)

        coll = _make_mock_collection("special_table")
        registry.register(coll, l1_backend=override_l1)

        assert registry.get_l1_backend("special_table") is override_l1
        assert registry.get_l1_backend("other_table") is default_l1

    def test_per_collection_override_l2(self) -> None:
        registry = CollectionRegistry()
        default_l2 = MagicMock()
        override_l2 = MagicMock()
        registry.configure(l2_client=default_l2, kv_key_scope="hub")

        coll = _make_mock_collection("cached_table")
        registry.register(coll, l2_client=override_l2)

        assert registry.get_l2_client("cached_table") is override_l2
        assert registry.get_l2_client("other_table") is default_l2

    def test_per_collection_override_l3(self) -> None:
        registry = CollectionRegistry()
        default_l3 = MagicMock()
        override_l3 = MagicMock()
        registry.configure(l3_pool=default_l3)

        coll = _make_mock_collection("sharded_table")
        registry.register(coll, l3_pool=override_l3)

        assert _routes_to(registry.get_l3_pool("sharded_table"), override_l3)
        assert _routes_to(registry.get_l3_pool("other_table"), default_l3)

    def test_clear_removes_all(self) -> None:
        registry = CollectionRegistry()
        coll1 = _make_mock_collection("t1")
        coll2 = _make_mock_collection("t2")
        registry.register(coll1, l1_backend=MagicMock())
        registry.register(coll2)

        registry.clear()

        assert registry.get_collection("t1") is None
        assert registry.get_collection("t2") is None
        # Defaults are NOT cleared by clear()
        # Overrides are cleared: t1's per-table l1 override registered
        # above must no longer win over the (absent) default; since no
        # default is set, the public get_l1_backend returns None.
        assert registry.get_l1_backend("t1") is None

    def test_configure_partial_update(self) -> None:
        """Calling configure multiple times only updates provided fields."""
        registry = CollectionRegistry()
        l1 = MagicMock()
        l2 = MagicMock()

        registry.configure(l1_backend=l1)
        registry.configure(kv_key_scope="hub")
        registry.configure(l2_client=l2)

        assert registry.get_l1_backend("any") is l1
        assert registry.get_l2_client("any") is l2

    def test_defaults_are_none_initially(self) -> None:
        registry = CollectionRegistry()

        assert registry.get_l1_backend("any") is None
        assert registry.get_l2_client("any") is None
        assert registry.get_l3_pool("any") is None


class TestBindTable:
    """tests for :meth:`CollectionRegistry.bind_table` (Phase C2)."""

    def test_bind_table_l3_pool_overrides_default(self) -> None:
        """per-table l3 override wins over the registry default."""
        registry = CollectionRegistry()
        default_l3 = MagicMock()
        override_l3 = MagicMock()
        registry.configure(l3_pool=default_l3)

        registry.bind_table("groups", l3_pool=override_l3)

        assert _routes_to(registry.get_l3_pool("groups"), override_l3)
        assert _routes_to(registry.get_l3_pool("conversations"), default_l3)

    def test_bind_table_accepts_pool_without_instance(self) -> None:
        """bind_table pins a pool BEFORE any collection is constructed."""
        registry = CollectionRegistry()
        pool = MagicMock()

        registry.bind_table("roles", l3_pool=pool)

        # subsequent register() calls merge with the earlier binding
        # rather than overwriting it
        coll = _make_mock_collection("roles")
        registry.register(coll)
        assert _routes_to(registry.get_l3_pool("roles"), pool)

    def test_bind_table_layers_l1_and_l3_independently(self) -> None:
        """l1 and l3 bindings on the same table are independent."""
        registry = CollectionRegistry()
        l1_override = MagicMock()
        l3_override = MagicMock()

        registry.bind_table("namespaces", l1_backend=l1_override)
        registry.bind_table("namespaces", l3_pool=l3_override)

        assert registry.get_l1_backend("namespaces") is l1_override
        assert _routes_to(registry.get_l3_pool("namespaces"), l3_override)

    def test_bind_table_no_op_when_every_arg_none(self) -> None:
        """bind_table with no overrides leaves existing overrides untouched."""
        registry = CollectionRegistry()
        pool = MagicMock()
        registry.bind_table("roles", l3_pool=pool)

        registry.bind_table("roles")

        assert _routes_to(registry.get_l3_pool("roles"), pool)

    def test_bind_table_isolates_to_named_table(self) -> None:
        """per-table binding never leaks onto an unrelated table."""
        registry = CollectionRegistry()
        default_l3 = MagicMock()
        registry.configure(l3_pool=default_l3)
        rbac_pool = MagicMock()

        registry.bind_table("groups", l3_pool=rbac_pool)

        assert _routes_to(registry.get_l3_pool("groups"), rbac_pool)
        assert _routes_to(registry.get_l3_pool("workspace_files"), default_l3)
        assert _routes_to(registry.get_l3_pool("memories"), default_l3)


# parity-exempt: subset shim for the nats-py Subscription the wrapper holds; only unsubscribe and the messages iterator are driven, and nats-py's own class is library-internal with no importable protocol
class _FakeRawSubscription:
    """fake nats-py subscription with a never-yielding message stream.

    the wrapper's dispatch task iterates :attr:`messages`, so the stream
    parks on an empty queue for the subscription's lifetime -- exactly
    like a live subscription with no traffic. cancelling the dispatch
    task (what ``unsubscribe`` does) is what ends it.
    """

    def __init__(self, unsubscribe_error: Exception | None = None) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.unsubscribe_calls = 0
        self.unsubscribe_error = unsubscribe_error

    async def unsubscribe(self) -> None:
        self.unsubscribe_calls += 1
        if self.unsubscribe_error is not None:
            raise self.unsubscribe_error

    @property
    def messages(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            while True:
                msg = await self.queue.get()
                if msg is None:
                    return
                yield msg

        return _gen()


# parity-exempt: subset shim for nats.aio.client.Client implementing subscribe only; that is the whole surface the wrapper drives on this path and nats-py's full Client surface is enormous
class _FakeNatsPyClient:
    """minimal fake of ``nats.aio.client.Client`` recording every subscribe."""

    def __init__(self, unsubscribe_error: Exception | None = None) -> None:
        self.subscriptions: list[_FakeRawSubscription] = []
        self.subscribed_subjects: list[str] = []
        self.unsubscribe_error = unsubscribe_error

    async def subscribe(self, subject: str, queue: str = "") -> _FakeRawSubscription:
        del queue
        sub = _FakeRawSubscription(unsubscribe_error=self.unsubscribe_error)
        self.subscriptions.append(sub)
        self.subscribed_subjects.append(subject)
        return sub


def _make_nats_client(unsubscribe_error: Exception | None = None) -> tuple[NatsClient, _FakeNatsPyClient]:
    """build the real wrapper over a fake nats-py client.

    :param unsubscribe_error: exception every raw subscription raises from
        ``unsubscribe()``, modelling a connection already draining; ``None``
        for the healthy path
    :ptype unsubscribe_error: Exception | None
    :return: the wrapper and the fake transport underneath it
    :rtype: tuple[NatsClient, _FakeNatsPyClient]
    """
    raw = _FakeNatsPyClient(unsubscribe_error=unsubscribe_error)
    client = NatsClient(raw=raw, namespace="aibots", client_name="registry-lifecycle-test")  # type: ignore[arg-type]
    return client, raw


class TestInvalidationListenerLifecycle:
    """start / stop lifecycle for the cross-pod invalidation subscription.

    Consumers need teardown to be expressible: a process that subscribes on
    startup has to unsubscribe on shutdown, and before this existed the handle
    was discarded so nothing could.
    """

    async def test_start_subscribes_the_invalidation_subject(self) -> None:
        """The subject is the cross-platform constant, spelled out rather than derived.

        This shard is lifecycle only; the wire subject is untouched. Asserting
        the literal is what makes a change to it visible here.
        """
        registry = CollectionRegistry()
        client, raw = _make_nats_client()

        await registry.start_invalidation_listener(client)

        assert raw.subscribed_subjects == ["threetears.cache.invalidate"]

        await registry.stop_invalidation_listener()

    async def test_a_second_start_while_one_is_live_creates_no_second_consumer(self) -> None:
        """SUB-03. Asserted on the subscribe COUNT, not on a boolean.

        A guard that flips a flag but still subscribes leaves two consumers on
        one subject, so every invalidation is handled twice and teardown
        releases one of them. Only the count can tell those apart.
        """
        registry = CollectionRegistry()
        client, raw = _make_nats_client()

        await registry.start_invalidation_listener(client)
        await registry.start_invalidation_listener(client)

        assert len(raw.subscriptions) == 1

        await registry.stop_invalidation_listener()

    async def test_stop_unsubscribes_the_live_subscription(self) -> None:
        registry = CollectionRegistry()
        client, raw = _make_nats_client()
        await registry.start_invalidation_listener(client)

        await registry.stop_invalidation_listener()

        assert raw.subscriptions[0].unsubscribe_calls == 1

    async def test_a_second_stop_is_a_no_op(self) -> None:
        """SUB-04. Callers run stop from a ``finally``, so it cannot double-fire."""
        registry = CollectionRegistry()
        client, raw = _make_nats_client()
        await registry.start_invalidation_listener(client)
        await registry.stop_invalidation_listener()

        await registry.stop_invalidation_listener()

        assert raw.subscriptions[0].unsubscribe_calls == 1

    async def test_stop_without_a_start_is_a_no_op(self) -> None:
        """SUB-04, the never-started case: a ``finally`` runs whether start reached or not."""
        registry = CollectionRegistry()

        await registry.stop_invalidation_listener()

    async def test_start_after_stop_subscribes_again(self) -> None:
        """SUB-05. The registry is reusable, not one-shot."""
        registry = CollectionRegistry()
        client, raw = _make_nats_client()

        await registry.start_invalidation_listener(client)
        await registry.stop_invalidation_listener()
        await registry.start_invalidation_listener(client)

        assert len(raw.subscriptions) == 2
        assert raw.subscriptions[0].unsubscribe_calls == 1
        assert raw.subscriptions[1].unsubscribe_calls == 0

        await registry.stop_invalidation_listener()

    async def test_stop_on_a_draining_connection_does_not_raise(self) -> None:
        """The shutdown-ordering case: the connection is already going away.

        ``NatsClient.unsubscribe`` absorbs the transport failure itself (it
        wraps both the raw unsubscribe and the dispatch-task join), which is
        why the registry adds no catch of its own. This is the test that keeps
        that reasoning true: if the wrapper ever stopped absorbing it, a
        ``close()`` path would start raising and this fails.
        """
        registry = CollectionRegistry()
        client, raw = _make_nats_client(unsubscribe_error=RuntimeError("nats: connection closed"))
        await registry.start_invalidation_listener(client)

        await registry.stop_invalidation_listener()

        assert raw.subscriptions[0].unsubscribe_calls == 1

    async def test_a_failed_stop_still_leaves_the_registry_restartable(self) -> None:
        """A draining-connection stop must not strand the handle.

        If the failed teardown left the old subscription in place, the next
        start would be swallowed by the re-entry guard and the process would
        come back up deaf to invalidations.
        """
        registry = CollectionRegistry()
        client, raw = _make_nats_client(unsubscribe_error=RuntimeError("nats: connection closed"))
        await registry.start_invalidation_listener(client)
        await registry.stop_invalidation_listener()

        await registry.start_invalidation_listener(client)

        assert len(raw.subscriptions) == 2

        await registry.stop_invalidation_listener()
