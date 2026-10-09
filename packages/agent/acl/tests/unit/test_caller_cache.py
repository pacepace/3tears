"""a per-caller answer is dropped by the access tables' rows and generations, and by nothing else.

The rules: a membership row naming a person or an agent drops those callers' answers only; any other
access-table row (a nested group's membership among them), a row that does not say what it names, and
a dropped table drop every answer; an answer asked before an eviction is never stored.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.acl import ACCESS_TABLES, CallerAccessCache, CallerKey, bind_caller_cache_to_access_tables
from threetears.agent.acl.access_tables import DegradedEvictions
from threetears.core.collections import CollectionRegistry
from threetears.core.collections.registry import CacheInvalidationMessage


def _member_row(member_type: str, member_id: Any) -> CacheInvalidationMessage:
    return CacheInvalidationMessage(
        table="group_members",
        ids=[f"{uuid4()}", f"{uuid4()}"],
        columns={"member_type": member_type, "member_id": f"{member_id}"},
    )


def _filled() -> tuple[CollectionRegistry, CallerAccessCache[Any], CallerKey, CallerKey, CallerKey]:
    registry = CollectionRegistry()
    cache: CallerAccessCache[Any] = CallerAccessCache()
    bind_caller_cache_to_access_tables(registry, cache)
    ingress = uuid4()
    reader, other = CallerKey(ingress, uuid4()), CallerKey(ingress, uuid4())
    agent_only = CallerKey(uuid4(), None)
    for key in (reader, other, agent_only):
        assert cache.put(key, frozenset({"va"}), fence=cache.read_fence())
    return registry, cache, reader, other, agent_only


def test_a_membership_row_naming_a_person_drops_that_persons_answer_only() -> None:
    registry, cache, reader, other, agent_only = _filled()
    registry.tell_derived_caches(_member_row("user", reader.user_id))
    assert cache.get(reader) is None
    assert cache.get(other) is not None
    assert cache.get(agent_only) is not None


def test_a_membership_row_naming_an_agent_drops_every_caller_through_it() -> None:
    registry, cache, reader, other, agent_only = _filled()
    registry.tell_derived_caches(_member_row("agent", reader.agent_id))
    assert cache.get(reader) is None
    assert cache.get(other) is None
    assert cache.get(agent_only) is not None


def test_a_nested_groups_membership_row_drops_every_answer() -> None:
    registry, cache, *_ = _filled()
    registry.tell_derived_caches(_member_row("group", uuid4()))
    assert cache.size == 0


@pytest.mark.parametrize("table", [t for t in ACCESS_TABLES if t != "group_members"])
def test_a_group_role_assignment_or_namespace_row_drops_every_answer(table: str) -> None:
    registry, cache, *_ = _filled()
    registry.tell_derived_caches(CacheInvalidationMessage(table=table, ids=[f"{uuid4()}"]))
    assert cache.size == 0


def test_a_membership_row_that_does_not_name_its_member_drops_all_and_is_counted() -> None:
    registry = CollectionRegistry()
    cache: CallerAccessCache[Any] = CallerAccessCache()
    degraded = DegradedEvictions()
    bind_caller_cache_to_access_tables(registry, cache, degraded=degraded)
    cache.put(CallerKey(uuid4(), uuid4()), 1, fence=cache.read_fence())
    registry.tell_derived_caches(CacheInvalidationMessage(table="group_members", ids=[f"{uuid4()}", f"{uuid4()}"]))
    assert cache.size == 0
    assert degraded.counts == {"group_members": 1}


@pytest.mark.parametrize("table", ACCESS_TABLES)
def test_a_dropped_table_drops_every_answer(table: str) -> None:
    registry, cache, *_ = _filled()
    registry.drop_table(table, reason="a broadcast was missed")
    assert cache.size == 0


def test_a_row_of_another_table_drops_nothing() -> None:
    registry, cache, *_ = _filled()
    registry.tell_derived_caches(CacheInvalidationMessage(table="conversations", ids=[f"{uuid4()}"]))
    assert cache.size == 3


def test_an_answer_asked_before_an_eviction_is_not_stored() -> None:
    registry, cache, reader, other, _ = _filled()
    fence = cache.read_fence()
    registry.tell_derived_caches(_member_row("user", other.user_id))
    assert not cache.put(reader, frozenset({"tx"}), fence=fence)
    assert cache.get(reader) == frozenset({"va"})
    assert cache.fence_skipped_stores == 1
    assert cache.put(reader, frozenset({"tx"}), fence=cache.read_fence())
    assert cache.get(reader) == frozenset({"tx"})


def test_unbinding_stops_the_drops() -> None:
    registry = CollectionRegistry()
    cache: CallerAccessCache[Any] = CallerAccessCache()
    remove = bind_caller_cache_to_access_tables(registry, cache)
    remove()
    cache.put(CallerKey(uuid4(), uuid4()), 1, fence=cache.read_fence())
    registry.tell_derived_caches(CacheInvalidationMessage(table="groups", ids=[f"{uuid4()}"]))
    assert cache.size == 1
    assert not any(registry.has_derived_caches(table) for table in ACCESS_TABLES)


class _Pushing:
    """a watcher that pushes one generation per table and then waits."""

    async def watch(self, table_name: str) -> Any:
        yield "inc:1"
        await asyncio.Event().wait()


async def test_following_binds_and_follows_every_access_table_and_stop_undoes_both() -> None:
    from threetears.agent.acl.generation_follow import follow_caller_access_cache
    from threetears.core.testing.kv import FakeNatsClient

    registry = CollectionRegistry()
    bus = FakeNatsClient()
    registry.configure(l2_client=bus, kv_key_scope="pod")
    cache: CallerAccessCache[Any] = CallerAccessCache()
    with pytest.raises(RuntimeError, match="invalidation listener"):
        follow_caller_access_cache(registry, cache, _Pushing())
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    following = follow_caller_access_cache(registry, cache, _Pushing())
    try:
        for _ in range(100):
            if following.healthy:
                break
            await asyncio.sleep(0.01)
        assert following.healthy
        assert all(registry.generation_marks.follows(table) for table in ACCESS_TABLES)
        assert all(registry.has_derived_caches(table) for table in ACCESS_TABLES)
        cache.put(CallerKey(uuid4(), uuid4()), 1, fence=cache.read_fence())
        registry.drop_table("roles", reason="a broadcast was missed")
        assert cache.size == 0
    finally:
        await following.stop()
    assert not any(registry.has_derived_caches(table) for table in ACCESS_TABLES)


class _Item:
    def __init__(self, name: str) -> None:
        self.name = name


class _Discovery:
    """records each ask; answers the names it holds, or raises what it is given."""

    def __init__(self, names: list[str], fail: Exception | None = None) -> None:
        self.names = names
        self.fail = fail
        self.asked: list[dict[str, Any]] = []

    async def discover(self, **kwargs: Any) -> list[_Item]:
        self.asked.append(kwargs)
        if self.fail is not None:
            raise self.fail
        return [_Item(name) for name in self.names]


def _asking(discovery: Any) -> tuple[Any, CallerAccessCache[frozenset[str]], CollectionRegistry]:
    from threetears.agent.acl import CallerNamespaces

    registry = CollectionRegistry()
    cache: CallerAccessCache[frozenset[str]] = CallerAccessCache()
    bind_caller_cache_to_access_tables(registry, cache)
    return CallerNamespaces(lambda: discovery, cache), cache, registry


async def test_a_callers_namespaces_are_asked_once_and_again_after_a_change_naming_them_or_a_group() -> None:
    discovery = _Discovery(["tools.enr.state.va.1-0"])
    names, _cache, registry = _asking(discovery)
    agent, user = uuid4(), uuid4()
    caller = {"agent_id": agent, "user_id": user, "identity_token": "t", "user_identity_token": "u"}
    assert await names.names_for(**caller) == frozenset({"tools.enr.state.va.1-0"})
    assert await names.names_for(**caller) == frozenset({"tools.enr.state.va.1-0"})
    assert len(discovery.asked) == 1
    assert discovery.asked[0]["identity_token"] == "t" and discovery.asked[0]["user_identity_token"] == "u"
    assert discovery.asked[0]["namespace_type"] == "tool"
    # another person's change leaves this caller's answer
    registry.tell_derived_caches(_member_row("user", uuid4()))
    await names.names_for(**caller)
    assert len(discovery.asked) == 1
    discovery.names = ["tools.enr.state.va.1-0", "tools.enr.state.tx.1-0"]
    registry.tell_derived_caches(_member_row("user", user))
    assert await names.names_for(**caller) == frozenset({"tools.enr.state.va.1-0", "tools.enr.state.tx.1-0"})
    assert len(discovery.asked) == 2
    discovery.names = ["tools.enr.state.tx.1-0"]
    registry.tell_derived_caches(_member_row("group", uuid4()))
    assert await names.names_for(**caller) == frozenset({"tools.enr.state.tx.1-0"})
    assert len(discovery.asked) == 3


async def test_two_callers_never_share_an_answer() -> None:
    discovery = _Discovery(["a"])
    names, _cache, _registry = _asking(discovery)
    agent = uuid4()
    first = await names.names_for(agent_id=agent, user_id=uuid4(), identity_token="t", user_identity_token="u")
    discovery.names = ["b"]
    second = await names.names_for(agent_id=agent, user_id=uuid4(), identity_token="t", user_identity_token="v")
    assert (first, second) == (frozenset({"a"}), frozenset({"b"}))


@pytest.mark.parametrize(
    ("discovery", "caller", "why"),
    [
        (_Discovery([], fail=RuntimeError("broker refused")), {}, "namespace.discover failed"),
        (None, {}, "not connected"),
        (_Discovery([]), {"identity_token": None, "user_identity_token": None}, "no hub credential"),
        (_Discovery([]), {"agent_id": None}, "no verified agent"),
    ],
)
async def test_an_answer_that_cannot_be_had_refuses_and_is_never_cached(
    discovery: Any, caller: dict[str, Any], why: str
) -> None:
    from threetears.agent.acl import CallerNamespacesUnavailable

    names, cache, _registry = _asking(discovery)
    asked = {"agent_id": uuid4(), "user_id": uuid4(), "identity_token": "t", "user_identity_token": "u", **caller}
    with pytest.raises(CallerNamespacesUnavailable, match=why):
        await names.names_for(**asked)
    assert cache.size == 0
