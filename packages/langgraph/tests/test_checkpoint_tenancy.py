"""Tests for the checkpoint saver's required customer scope.

``ThreeTierCheckpointSaver`` takes a :class:`CheckpointScope` at construction and
has no default for it, so every caller states one of exactly two things: this
saver belongs to one customer, or it deliberately belongs to none and here is
why. When it names a customer, every key the saver addresses -- the ``thread_id``
bound into L3 SQL, the L2 bucket key, the L1 thread key -- carries that customer,
so a saver scoped to one customer cannot name another customer's row at all.

The unscoped answer is not a degraded mode; it is the pre-tenancy behaviour
preserved byte for byte, which is what makes the required-scope change adoptable
without a data migration. That property is pinned by
:class:`TestUnscopedIsByteIdenticalToThePreTenancySaver` rather than assumed.

The suite is deliberately split from ``test_checkpoint.py``: that file pins the
un-tenanted behaviour, and it now constructs every saver with a single shared
unscoped scope, so it keeps proving that the unscoped path did not move.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.protocols import CheckpointL2Cache, CheckpointL2PrefixCache

_LOGGER = "threetears.langgraph.checkpoint"
_SCOPE_LOGGER = "threetears.langgraph.checkpoint_scope"

_CUSTOMER_A = UUID("11111111-1111-1111-1111-111111111111")
_CUSTOMER_B = UUID("22222222-2222-2222-2222-222222222222")

_SCOPE_A = CheckpointScope.for_customer(_CUSTOMER_A)
_SCOPE_B = CheckpointScope.for_customer(_CUSTOMER_B)

#: built once at import so the warning it emits lands outside every ``caplog``
#: block below. tests that assert on the warning build their own.
_UNSCOPED = CheckpointScope.unscoped(reason="tests pin the un-tenanted keyspace")

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


class TestCheckpointScope:
    """the scope decision is a value with exactly two legal answers."""

    def test_for_customer_carries_the_customer(self) -> None:
        """the ordinary answer names a customer and no reason.

        :return: nothing
        :rtype: None
        """
        scope = CheckpointScope.for_customer(_CUSTOMER_A)

        assert scope.customer_id == _CUSTOMER_A
        assert scope.reason is None

    def test_for_customer_refuses_anything_but_a_uuid(self) -> None:
        """a string customer would produce a plausible-looking, wrong prefix.

        the prefix is interpolated into a ``LIKE`` pattern by the per-customer
        purge, and the "no ESCAPE clause needed" property that purge relies on
        holds because a UUID's text form contains no ``%`` and no ``_``. a
        string identifier carries no such guarantee, so it is refused where it
        enters rather than where it widens a DELETE.

        :return: nothing
        :rtype: None
        """
        # bound through Any rather than suppressed inline: the call is a type
        # error, and the point is what happens at RUNTIME when a caller makes it
        # anyway (an untyped host, a value off a JSON body).
        customer_as_text: Any = "11111111-1111-1111-1111-111111111111"

        with pytest.raises(TypeError, match="UUID"):
            CheckpointScope.for_customer(customer_as_text)

    def test_unscoped_carries_a_reason_and_no_customer(self) -> None:
        """the opt-out records why, so it is answerable in review.

        :return: nothing
        :rtype: None
        """
        scope = CheckpointScope.unscoped(reason="single-tenant deployment")

        assert scope.customer_id is None
        assert scope.reason == "single-tenant deployment"

    def test_unscoped_warns_and_names_the_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """an unscoped deployment has to be visible in logs, not only in source.

        greppability in source is the other half and comes free from the
        constructor's name; this half is what an operator reading a running
        system sees.

        :param caplog: pytest log capture fixture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        with caplog.at_level(logging.WARNING, logger=_SCOPE_LOGGER):
            CheckpointScope.unscoped(reason="no customer exists in this process")

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "no customer exists in this process" in messages

    def test_unscoped_refuses_an_empty_reason(self) -> None:
        """``unscoped("")`` would be the falsy default this type exists to remove.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValueError, match="reason"):
            CheckpointScope.unscoped(reason="")

    def test_unscoped_refuses_a_whitespace_reason(self) -> None:
        """whitespace is an empty reason wearing a disguise.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValueError, match="reason"):
            CheckpointScope.unscoped(reason="   ")

    def test_there_is_no_public_constructor(self) -> None:
        """``CheckpointScope()`` must not be a way to reach the unsafe answer.

        a bare constructor would give the unscoped state back its default: build
        one with nothing and get "sees everything" with no reason recorded and no
        warning logged. the two named constructors are the only doors.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(TypeError, match="for_customer"):
            CheckpointScope()

    def test_a_scope_cannot_be_rewritten_after_construction(self) -> None:
        """a saver holds its scope for life; a mutable one would be a re-scoping.

        :return: nothing
        :rtype: None
        """
        scope: Any = CheckpointScope.for_customer(_CUSTOMER_A)

        with pytest.raises(AttributeError):
            scope.customer_id = _CUSTOMER_B
        with pytest.raises(AttributeError):
            scope.smuggled = _CUSTOMER_B
        with pytest.raises(AttributeError):
            del scope.customer_id

    def test_a_scope_carries_no_instance_dict(self) -> None:
        """slots, so there is no back door around the read-only attributes.

        :return: nothing
        :rtype: None
        """
        assert not hasattr(CheckpointScope.for_customer(_CUSTOMER_A), "__dict__")

    def test_scopes_compare_by_value(self) -> None:
        """two scopes naming one customer are one scope.

        :return: nothing
        :rtype: None
        """
        assert CheckpointScope.for_customer(_CUSTOMER_A) == CheckpointScope.for_customer(_CUSTOMER_A)
        assert CheckpointScope.for_customer(_CUSTOMER_A) != CheckpointScope.for_customer(_CUSTOMER_B)
        assert CheckpointScope.for_customer(_CUSTOMER_A) != _UNSCOPED
        assert len({CheckpointScope.for_customer(_CUSTOMER_A), CheckpointScope.for_customer(_CUSTOMER_A)}) == 1

    def test_the_repr_says_which_answer_was_given(self) -> None:
        """a saver in a traceback should say which of the two it holds.

        :return: nothing
        :rtype: None
        """
        assert "for_customer" in repr(_SCOPE_A)
        assert "unscoped" in repr(_UNSCOPED)


class TestScopeIsRequired:
    """the defect being fixed was the DEFAULT, so the absence of one is the test."""

    def test_omitting_the_scope_is_a_type_error(self) -> None:
        """saying nothing used to mean "see everything"; now it means nothing.

        this is the whole change. a caller that has not thought about tenancy
        cannot get a saver at all, which turns the feature from a convention
        into a gate.

        the class is bound through ``Any`` rather than suppressed inline: both
        calls below are type errors, and the assertion is about what a RUNTIME
        caller gets -- a type checker is the first gate, not the only one.

        :return: nothing
        :rtype: None
        """
        saver_class: Any = ThreeTierCheckpointSaver

        with pytest.raises(TypeError, match="scope"):
            saver_class(executor=_make_executor())

    def test_there_is_no_customer_id_parameter_left(self) -> None:
        """the old optional parameter is gone, not merely discouraged.

        leaving it accepted alongside ``scope`` would keep the unsafe default
        reachable for anyone who never read the changelog.

        :return: nothing
        :rtype: None
        """
        saver_class: Any = ThreeTierCheckpointSaver

        with pytest.raises(TypeError, match="customer_id"):
            saver_class(executor=_make_executor(), scope=_UNSCOPED, customer_id=_CUSTOMER_A)


class TestStorageThreadId:
    """the customer lives inside the storage thread id, not in a new column."""

    def test_unscoped_saver_leaves_the_thread_id_alone(self) -> None:
        """no customer means byte-identical keys to the pre-tenancy saver.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        assert saver.storage_thread_id("t-1") == "t-1"

    def test_scoped_saver_prefixes_with_the_customer(self) -> None:
        """the composite is what lands in the ``thread_id`` column.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_A)

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
        saver_a = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_A)
        saver_b = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_B)

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
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_A)

        with pytest.raises(ValueError, match="255"):
            saver.storage_thread_id("t" * 255)

    def test_an_overlong_id_still_passes_through_when_unscoped(self) -> None:
        """without a customer nothing is added, so nothing new can overflow.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        assert saver.storage_thread_id("t" * 300) == "t" * 300


class TestL2KeyIsScoped:
    """the KV bucket carries the customer too -- tenanting only L3 is the anti-pattern."""

    def test_root_namespace_key_carries_the_customer(self) -> None:
        """:return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_A)

        assert saver.l2_key("t-1", "") == f"{_CUSTOMER_A}/t-1"

    def test_namespaced_key_carries_the_customer(self) -> None:
        """:return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_A)

        assert saver.l2_key("t-1", "inner") == f"{_CUSTOMER_A}/t-1.inner"

    def test_unscoped_key_is_unchanged(self) -> None:
        """:return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

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
        saver_a = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=shared, scope=_SCOPE_A)
        saver_b = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=shared, scope=_SCOPE_B)

        await saver_a.l2_put("t-1", "", b"customer-a-state")

        assert await saver_a.l2_get("t-1", "") == b"customer-a-state"
        assert await saver_b.l2_get("t-1", "") is None

    async def test_l1_is_addressed_by_the_scoped_thread_id(self) -> None:
        """L1 is keyed on thread too, so it needs the same treatment.

        :return: nothing
        :rtype: None
        """
        l1 = AsyncMock()
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, scope=_SCOPE_A)

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
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

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
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_SCOPE_A)

        result = await saver.aput({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}, _CHECKPOINT, {}, {})

        assert result["configurable"]["thread_id"] == "t-1"

    async def test_aget_tuple_binds_the_scoped_thread_id(self) -> None:
        """:return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        _sql, *params = executor.fetchrow.await_args.args
        assert params[0] == f"{_CUSTOMER_A}/t-1"

    async def test_aget_tuple_returns_the_logical_thread_id(self) -> None:
        """a row read back is reported under the caller's own thread id.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)
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
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

        [item async for item in saver.alist({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})]

        _sql, *params = executor.fetch.await_args.args
        assert params[0] == f"{_CUSTOMER_A}/t-1"

    async def test_aput_writes_binds_the_scoped_thread_id(self) -> None:
        """:return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

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
            scope=_SCOPE_A,
        )

        await saver.aput_writes(
            {"configurable": {"thread_id": "t-1", "checkpoint_ns": "inner", "checkpoint_id": "cp-1"}},
            [("__interrupt__", "approve?")],
            "task-1",
        )

        l1.delete.assert_awaited_once_with(f"{_CUSTOMER_A}/t-1")
        assert l2.delete.await_args.args[1] == f"{_CUSTOMER_A}/t-1.inner"


class TestUnscopedIsByteIdenticalToThePreTenancySaver:
    """the migration story rests on this: opting out moves no row and no key.

    an existing deployment upgrades by passing
    ``scope=CheckpointScope.unscoped(reason=...)`` and nothing else. that is only
    an honest instruction if the resulting saver addresses exactly the keyspace
    its existing rows already live in, so the claim is asserted rather than
    described.
    """

    async def test_every_statement_binds_the_bare_thread_id(self) -> None:
        """read, write, list, and per-thread delete all address the un-prefixed row.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)
        config: Any = {"configurable": {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "cp-1"}}

        await saver.aput(config, _CHECKPOINT, {}, {})
        await saver.aput_writes(config, [("channel-a", "value-a")], "task-1")
        await saver.aget_tuple(config)
        [item async for item in saver.alist(config)]
        await saver.adelete_thread("t-1")

        bound = [call.args[1] for call in executor.execute.await_args_list]
        bound += [call.args[1] for call in executor.fetchrow.await_args_list]
        bound += [call.args[1] for call in executor.fetch.await_args_list]
        assert set(bound) == {"t-1"}

    async def test_every_cache_key_is_the_bare_thread_id(self) -> None:
        """an existing L2 bucket keeps serving the bundles it already holds.

        :return: nothing
        :rtype: None
        """
        cache = _DictL2Cache()
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, scope=_UNSCOPED)

        await saver.l2_put("t-1", "", b"root")
        await saver.l2_put("t-1", "inner", b"namespaced")

        assert set(cache.store) == {("checkpoints", "t-1"), ("checkpoints", "t-1.inner")}


class TestPerThreadPurge:
    """``adelete_thread`` keeps its signature -- scriob calls it in production."""

    async def test_unscoped_delete_is_byte_identical_to_before(self) -> None:
        """the live consumer passes no customer and must not change.

        scriob's delete-session route calls ``adelete_thread(str(session_id))``
        (``scriob/server/src/scriob_server/chat/routes.py``). a customer argument
        on this method, or a silently rewritten parameter, breaks it.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        await saver.adelete_thread("t-42")

        assert [call.args[1] for call in executor.execute.await_args_list] == ["t-42", "t-42"]

    async def test_scoped_delete_only_names_its_own_customer(self) -> None:
        """a scoped saver cannot address another customer's row, even to delete it.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

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
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, scope=_UNSCOPED)
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
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, scope=_UNSCOPED)
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
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await saver.adelete_thread("t-1")

        assert caplog.records == []


class TestPerCustomerPurge:
    """erasure needs a handle that is not per-thread; tenant offboarding needs it too."""

    async def test_purge_refuses_on_an_unscoped_saver(self) -> None:
        """an unscoped saver would delete every row in the table.

        the refusal names the scope decision, because that is what the caller
        has to change -- not an argument to this method, which deliberately
        takes none.

        :return: nothing
        :rtype: None
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with pytest.raises(ValueError, match="unscoped"):
            await saver.adelete_customer_threads()

    async def test_the_refusal_issues_no_statement(self) -> None:
        """refusing after a DELETE would be no refusal at all.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        with pytest.raises(ValueError):
            await saver.adelete_customer_threads()

        assert executor.execute.await_args_list == []

    async def test_purge_deletes_writes_before_checkpoints(self) -> None:
        """same order as the per-thread purge, for the same reason.

        :return: nothing
        :rtype: None
        """
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

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
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_SCOPE_A)

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
        saver_a = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, scope=_SCOPE_A)
        saver_b = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=cache, scope=_SCOPE_B)
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
            scope=_SCOPE_A,
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
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=_FailingSweepL2Cache(), scope=_UNSCOPED)

        assert await saver.l2_delete_prefix("t-1.") is False
