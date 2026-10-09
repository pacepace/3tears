"""``evict_after_rbac_write``: what an rbac writer evicts from its own cache, and nothing it publishes."""

from __future__ import annotations

import inspect
from uuid import uuid7

from threetears.agent.acl import evict_after_rbac_write
from threetears.agent.acl.cache import ActorMembershipKey, GroupNamespaceKey

from .fake_loaders import FakeStore, make_cache


def test_the_named_actor_and_group_are_evicted_and_nothing_else() -> None:
    cache = make_cache(FakeStore())
    user, other_user, group, other_group, ns = uuid7(), uuid7(), uuid7(), uuid7(), uuid7()
    cache.put_membership(ActorMembershipKey("user", user), ())
    cache.put_membership(ActorMembershipKey("user", other_user), ())
    cache.put_group_namespace(GroupNamespaceKey(group, ns), frozenset(), ())
    cache.put_group_namespace(GroupNamespaceKey(other_group, ns), frozenset(), ())

    evict_after_rbac_write(cache, member_actors=[("user", user)], group_ids=[group])

    assert cache.get_membership(ActorMembershipKey("user", user)) is None
    assert cache.get_group_namespace(GroupNamespaceKey(group, ns)) is None
    assert cache.get_membership(ActorMembershipKey("user", other_user)) is not None
    assert cache.get_group_namespace(GroupNamespaceKey(other_group, ns)) is not None


def test_a_write_that_changed_nothing_evicts_nothing() -> None:
    cache = make_cache(FakeStore())
    user = uuid7()
    cache.put_membership(ActorMembershipKey("user", user), ())

    evict_after_rbac_write(cache)

    assert cache.get_membership(ActorMembershipKey("user", user)) is not None


def test_it_takes_no_publisher() -> None:
    """the acl invalidation subjects are retired: other processes hear the row broadcasts and generations."""
    assert list(inspect.signature(evict_after_rbac_write).parameters) == ["cache", "member_actors", "group_ids"]
