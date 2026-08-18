"""Tests for the checkpoint saver's optional customer dimension.

``ThreeTierCheckpointSaver`` may be bound to one customer at construction. When
it is, every key it addresses -- the ``thread_id`` bound into L3 SQL, the L2
bucket key, the L1 thread key -- carries that customer, so a saver bound to one
customer cannot name another customer's row at all. When it is not bound, every
byte on the wire is what it was before: the whole point, because
``adelete_thread`` has a live production caller (scriob) that passes no customer.

The suite is deliberately split from ``test_checkpoint.py``: that file pins the
un-tenanted behaviour, and leaving it untouched is what proves the default path
did not move.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.protocols import CheckpointL2Cache, CheckpointL2PrefixCache

_LOGGER = "threetears.langgraph.checkpoint"

_CUSTOMER_A = UUID("11111111-1111-1111-1111-111111111111")
_CUSTOMER_B = UUID("22222222-2222-2222-2222-222222222222")

_CHECKPOINT: dict[str, Any] = {
    "id": "cp-1",
    "ts": "2026-01-01T00:00:00Z",
    "channel_values": {},
    "channel_versions": {},
    "versions_seen": {},
    "pending_sends": [],
}


def _make_executor() -> Any:
    """build a MagicMock standing in for an AsyncQueryExecutor.

    :return: mock executor with async fetch/fetchrow/execute
    :rtype: Any
    """
    executor = MagicMock()
    executor.fetch = AsyncMock(return_value=[])
    executor.fetchrow = AsyncMock(return_value=None)
    executor.execute = AsyncMock(return_value="INSERT 0 1")
    return executor


class _DictL2Cache(CheckpointL2Cache):
    """an exact-key L2 cache over a plain dict, with no prefix sweep.

    stands in for the shape every wired L2 adapter has today (the survey
    engine's ``NatsKvL2CacheAdapter`` is exactly this surface), so a test can
    assert what a bucket shared between two customers actually returns.
    """

    def __init__(self) -> None:
        """start empty.

        :return: nothing
        :rtype: None
        """
        self.store: dict[tuple[str, str], bytes] = {}

    async def get(self, bucket: str, key: str) -> bytes | None:
        """read a value, or None on miss.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: stored bytes or None
        :rtype: bytes | None
        """
        return self.store.get((bucket, key))

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """write a value.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: nothing
        :rtype: None
        """
        self.store[(bucket, key)] = value

    async def delete(self, bucket: str, key: str) -> None:
        """drop one key; a missing key is not an error.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        self.store.pop((bucket, key), None)


class _PrefixSweepingL2Cache(_DictL2Cache, CheckpointL2PrefixCache):
    """a dict L2 cache that also implements the optional prefix sweep."""

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        """drop every key in *bucket* starting with *prefix*.

        :param bucket: bucket name
        :ptype bucket: str
        :param prefix: key prefix to sweep
        :ptype prefix: str
        :return: nothing
        :rtype: None
        """
        for bucket_name, key in list(self.store):
            if bucket_name == bucket and key.startswith(prefix):
                del self.store[(bucket_name, key)]


class _FailingSweepL2Cache(_DictL2Cache, CheckpointL2PrefixCache):
    """a prefix-capable cache whose sweep always fails, as a network one can."""

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        """fail the way a NATS timeout does.

        :param bucket: bucket name
        :ptype bucket: str
        :param prefix: key prefix to sweep
        :ptype prefix: str
        :return: never returns
        :rtype: None
        :raises RuntimeError: always
        """
        raise RuntimeError("nats: timeout")


class TestStorageThreadId:
    """the customer lives inside the storage thread id, not in a new column."""

    def test_unbound_saver_leaves_the_thread_id_alone(self) -> None:
        """no customer means byte-identical keys to the pre-tenancy saver.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        assert saver.storage_thread_id("t-1") == "t-1"

    def test_bound_saver_prefixes_with_the_customer(self) -> None:
        """the composite is what lands in the ``thread_id`` column.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_A)

        assert saver.storage_thread_id("t-1") == f"{_CUSTOMER_A}/t-1"

    def test_two_customers_never_produce_the_same_storage_id(self) -> None:
        """the same logical thread under two customers is two distinct rows.

        this is the property that makes the primary key
        ``(thread_id, checkpoint_ns, checkpoint_id)`` unique THROUGH the
        customer without altering it: a collision on the caller-chosen thread id
        can no longer collide in storage.

        :return: nothing
        :rtype: None
        """
        saver_a = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_A)
        saver_b = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_B)

        assert saver_a.storage_thread_id("shared") != saver_b.storage_thread_id("shared")

    def test_an_overlong_composite_raises_rather_than_reaching_the_column(self) -> None:
        """``thread_id`` is VARCHAR(255); the prefix must not silently overflow it.

        the un-tenanted saver has always handed an over-long id straight to the
        column and let the database reject it. adding 37 characters to every id
        makes overflow OUR doing, so it fails here with the arithmetic named
        rather than as a truncation or an opaque driver error.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_A)

        with pytest.raises(ValueError, match="255"):
            saver.storage_thread_id("t" * 255)

    def test_an_overlong_id_still_passes_through_when_unbound(self) -> None:
        """without a customer nothing is added, so nothing new can overflow.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        assert saver.storage_thread_id("t" * 300) == "t" * 300


class TestL2KeyIsScoped:
    """the KV bucket carries the customer too -- tenanting only L3 is the anti-pattern."""

    def test_root_namespace_key_carries_the_customer(self) -> None:
        """:return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_A)

        assert saver.l2_key("t-1", "") == f"{_CUSTOMER_A}/t-1"

    def test_namespaced_key_carries_the_customer(self) -> None:
        """:return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_A)

        assert saver.l2_key("t-1", "inner") == f"{_CUSTOMER_A}/t-1.inner"

    def test_unbound_key_is_unchanged(self) -> None:
        """:return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        assert saver.l2_key("t-1", "inner") == "t-1.inner"

    async def test_a_shared_bucket_does_not_leak_between_customers(self) -> None:
        """one bucket, two customers, same logical thread: B must miss.

        this is the KV half of "a checkpoint written under one customer is
        unreadable under another" -- asserted against a real cache rather than a
        mock, because the whole question is what the key lookup returns.

        :return: nothing
        :rtype: None
        """
        shared = _DictL2Cache()
        saver_a = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=shared, customer_id=_CUSTOMER_A)
        saver_b = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=shared, customer_id=_CUSTOMER_B)

        await saver_a.l2_put("t-1", "", b"customer-a-state")

        assert await saver_a.l2_get("t-1", "") == b"customer-a-state"
        assert await saver_b.l2_get("t-1", "") is None

    async def test_l1_is_addressed_by_the_scoped_thread_id(self) -> None:
        """L1 is keyed on thread too, so it needs the same treatment.

        :return: nothing
        :rtype: None
        """
        l1 = AsyncMock()
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, customer_id=_CUSTOMER_A)

        await saver.l1_put("t-1", "", b"state")
        await saver.l1_get("t-1", "")
        await saver.l1_delete("t-1")

        assert l1.put.await_args.args[0] == f"{_CUSTOMER_A}/t-1"
        assert l1.get.await_args.args[0] == f"{_CUSTOMER_A}/t-1"
        assert l1.delete.await_args.args[0] == f"{_CUSTOMER_A}/t-1"


class TestL3StatementsAreScoped:
    """every statement binds the scoped id; every returned config keeps the logical one."""

    async def test_aput_binds_the_scoped_thread_id(self) -> None:
        """:return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        await saver.aput({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}, _CHECKPOINT, {}, {})

        _sql, *params = executor.execute.await_args.args
        assert params[0] == f"{_CUSTOMER_A}/t-1"

    async def test_aput_returns_the_logical_thread_id(self) -> None:
        """LangGraph gets back the id it passed in; the prefix is storage-only.

        a leaked prefix would be handed straight back into the next call and
        double-prefixed, so this is load-bearing rather than cosmetic.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), customer_id=_CUSTOMER_A)

        result = await saver.aput({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}, _CHECKPOINT, {}, {})

        assert result["configurable"]["thread_id"] == "t-1"

    async def test_aget_tuple_binds_the_scoped_thread_id(self) -> None:
        """:return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        _sql, *params = executor.fetchrow.await_args.args
        assert params[0] == f"{_CUSTOMER_A}/t-1"

    async def test_aget_tuple_returns_the_logical_thread_id(self) -> None:
        """a row read back is reported under the caller's own thread id.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)
        cp_type, cp_blob = saver.serde.dumps_typed(_CHECKPOINT)
        executor.fetchrow = AsyncMock(
            return_value={
                "checkpoint_id": "cp-1",
                "parent_checkpoint_id": None,
                "type": cp_type,
                "checkpoint": cp_blob,
                "metadata_": b"\x00",
            },
        )

        tup = await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert tup is not None
        assert tup.config["configurable"]["thread_id"] == "t-1"

    async def test_alist_binds_the_scoped_thread_id(self) -> None:
        """:return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        [item async for item in saver.alist({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})]

        _sql, *params = executor.fetch.await_args.args
        assert params[0] == f"{_CUSTOMER_A}/t-1"

    async def test_aput_writes_binds_the_scoped_thread_id(self) -> None:
        """:return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        await saver.aput_writes(
            {"configurable": {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "cp-1"}},
            [("channel-a", "value-a")],
            "task-1",
        )

        _sql, *params = executor.execute.await_args.args
        assert params[0] == f"{_CUSTOMER_A}/t-1"

    async def test_control_write_invalidation_uses_the_scoped_keys(self) -> None:
        """the invalidation that keeps an approval gate visible must find the key.

        it bypasses ``l1_delete``/``l2_delete`` on purpose, so it has its own
        scoping and its own test.

        :return: nothing
        :rtype: None
        """
        l1, l2 = AsyncMock(), AsyncMock()
        saver = ThreeTierCheckpointSaver(
            executor=_make_executor(),
            l1_cache=l1,
            l2_cache=l2,
            customer_id=_CUSTOMER_A,
        )

        await saver.aput_writes(
            {"configurable": {"thread_id": "t-1", "checkpoint_ns": "inner", "checkpoint_id": "cp-1"}},
            [("__interrupt__", "approve?")],
            "task-1",
        )

        l1.delete.assert_awaited_once_with(f"{_CUSTOMER_A}/t-1")
        assert l2.delete.await_args.args[1] == f"{_CUSTOMER_A}/t-1.inner"

    async def test_an_unbound_saver_issues_the_same_statements_as_before(self) -> None:
        """resumability for a single-tenant deployment is unchanged.

        the bound parameters are the whole contract with the existing rows: if
        they still read exactly as they did, a deployment that never sets a
        customer resumes from checkpoints written before this change.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor)

        await saver.aput({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}, _CHECKPOINT, {}, {})
        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert executor.execute.await_args.args[1] == "t-1"
        assert executor.fetchrow.await_args.args[1] == "t-1"


class TestPerThreadPurge:
    """``adelete_thread`` keeps its signature -- scriob calls it in production."""

    async def test_unbound_delete_is_byte_identical_to_before(self) -> None:
        """the live consumer passes no customer and must not change.

        scriob's delete-session route calls ``adelete_thread(str(session_id))``
        (``scriob/server/src/scriob_server/chat/routes.py``). a required customer
        argument, or a silently rewritten parameter, breaks it.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor)

        await saver.adelete_thread("t-42")

        assert [call.args[1] for call in executor.execute.await_args_list] == ["t-42", "t-42"]

    async def test_bound_delete_only_names_its_own_customer(self) -> None:
        """a bound saver cannot address another customer's row, even to delete it.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        await saver.adelete_thread("t-42")

        expected = f"{_CUSTOMER_A}/t-42"
        assert [call.args[1] for call in executor.execute.await_args_list] == [expected, expected]

    async def test_delete_sweeps_namespaced_l2_keys_when_the_cache_can(self) -> None:
        """the documented exact-key gap: a namespaced bundle used to survive.

        ``l2_delete`` is exact-key, so purging a thread cleared only its root
        key and left every ``thread.ns`` bundle cached. a prefix-capable cache
        closes it.

        :return: nothing
        :rtype: None
        """
        cache = _PrefixSweepingL2Cache()
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache)
        await saver.l2_put("t-1", "", b"root")
        await saver.l2_put("t-1", "inner", b"namespaced")
        await saver.l2_put("t-10", "", b"different-thread")

        await saver.adelete_thread("t-1")

        assert cache.store == {("checkpoints", "t-10"): b"different-thread"}

    async def test_delete_warns_when_the_cache_cannot_sweep(self, caplog: pytest.LogCaptureFixture) -> None:
        """an exact-key cache leaves namespaced bundles behind, and says so.

        silence here is what made the gap invisible; the warning names the
        thread so an operator purging for an erasure request can see what was
        not reached.

        :param caplog: pytest log capture fixture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        cache = _DictL2Cache()
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache)
        await saver.l2_put("t-1", "inner", b"namespaced")

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await saver.adelete_thread("t-1")

        assert any("checkpoint_ns" in record.getMessage() for record in caplog.records)

    async def test_delete_does_not_warn_when_no_l2_is_wired(self, caplog: pytest.LogCaptureFixture) -> None:
        """no cache, nothing stale, nothing to say.

        :param caplog: pytest log capture fixture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await saver.adelete_thread("t-1")

        assert caplog.records == []


class TestPerCustomerPurge:
    """erasure needs a handle that is not per-thread; tenant offboarding needs it too."""

    async def test_purge_requires_a_bound_customer(self) -> None:
        """an unbound saver would delete every row in the table.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with pytest.raises(ValueError, match="customer_id"):
            await saver.adelete_customer_threads()

    async def test_purge_deletes_writes_before_checkpoints(self) -> None:
        """same order as the per-thread purge, for the same reason.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        await saver.adelete_customer_threads()

        statements = [call.args[0] for call in executor.execute.await_args_list]
        assert "DELETE FROM checkpoint_writes" in statements[0]
        assert "DELETE FROM checkpoints" in statements[1]

    async def test_purge_matches_only_this_customers_prefix(self) -> None:
        """the pattern is the customer's own prefix and nothing wider.

        a UUID's text form contains no ``%`` and no ``_``, so the pattern needs
        no ESCAPE clause -- asserted here so a future change of separator or id
        type has to confront it.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, customer_id=_CUSTOMER_A)

        await saver.adelete_customer_threads()

        patterns = [call.args[1] for call in executor.execute.await_args_list]
        assert patterns == [f"{_CUSTOMER_A}/%", f"{_CUSTOMER_A}/%"]
        assert "%" not in str(_CUSTOMER_A)
        assert "_" not in str(_CUSTOMER_A)

    async def test_purge_sweeps_the_customers_l2_keys(self) -> None:
        """a purged blob still served from cache is not a purge.

        :return: nothing
        :rtype: None
        """
        cache = _PrefixSweepingL2Cache()
        saver_a = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, customer_id=_CUSTOMER_A)
        saver_b = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, customer_id=_CUSTOMER_B)
        await saver_a.l2_put("t-1", "", b"a-root")
        await saver_a.l2_put("t-1", "inner", b"a-namespaced")
        await saver_b.l2_put("t-1", "", b"b-root")

        await saver_a.adelete_customer_threads()

        assert cache.store == {("checkpoints", f"{_CUSTOMER_B}/t-1"): b"b-root"}

    async def test_purge_warns_when_a_cache_cannot_be_swept(self, caplog: pytest.LogCaptureFixture) -> None:
        """an unsweepable cache is an erasure gap and must not pass quietly.

        :param caplog: pytest log capture fixture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(
            executor=_make_executor(),
            l1_cache=AsyncMock(),
            l2_cache=_DictL2Cache(),
            customer_id=_CUSTOMER_A,
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await saver.adelete_customer_threads()

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "L1" in messages
        assert "L2" in messages


class TestPrefixCacheCapability:
    """the optional protocol is detected structurally, never assumed."""

    def test_an_exact_key_cache_is_not_prefix_capable(self) -> None:
        """:return: nothing
        :rtype: None
        """
        assert not isinstance(_DictL2Cache(), CheckpointL2PrefixCache)

    def test_a_sweeping_cache_is_prefix_capable(self) -> None:
        """:return: nothing
        :rtype: None
        """
        assert isinstance(_PrefixSweepingL2Cache(), CheckpointL2PrefixCache)

    def test_a_sweeping_cache_is_still_an_ordinary_l2(self) -> None:
        """the optional protocol widens nothing; existing adapters stay valid.

        :return: nothing
        :rtype: None
        """
        assert isinstance(_PrefixSweepingL2Cache(), CheckpointL2Cache)
        assert isinstance(_DictL2Cache(), CheckpointL2Cache)

    async def test_a_failing_sweep_degrades(self) -> None:
        """a cache fault must not abort a purge whose L3 half already ran.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=_FailingSweepL2Cache())

        assert await saver.l2_delete_prefix("t-1.") is False
