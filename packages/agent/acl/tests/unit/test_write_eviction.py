"""``evict_after_rbac_write``: what an rbac writer evicts locally, and what it broadcasts."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid7

import pytest
from pydantic import BaseModel

from threetears.agent.acl import evict_after_rbac_write
from threetears.agent.acl.cache import ActorMembershipKey, GroupNamespaceKey
from threetears.agent.acl.invalidation import AssignmentInvalidatePayload, MembershipInvalidatePayload

from ._fake_loaders import FakeStore, make_cache


class _RecordingPublisher:
    """# parity-with: threetears.agent.acl.invalidation_bus.AclInvalidationPublisher"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, BaseModel]] = []

    async def publish(self, *, subject: Any, message: BaseModel) -> None:
        self.sent.append((str(getattr(subject, "path", subject)), message))


class _FailingPublisher:
    """# parity-with: threetears.agent.acl.invalidation_bus.AclInvalidationPublisher"""

    async def publish(self, *, subject: Any, message: BaseModel) -> None:
        raise ConnectionError("broker unreachable")


@pytest.fixture(autouse=True)
def _namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "t")


async def test_the_named_actor_and_group_are_evicted_and_nothing_else() -> None:
    cache = make_cache(FakeStore())
    user, other_user, group, other_group, ns = uuid7(), uuid7(), uuid7(), uuid7(), uuid7()
    cache.put_membership(ActorMembershipKey("user", user), ())
    cache.put_membership(ActorMembershipKey("user", other_user), ())
    cache.put_group_namespace(GroupNamespaceKey(group, ns), frozenset(), ())
    cache.put_group_namespace(GroupNamespaceKey(other_group, ns), frozenset(), ())

    await evict_after_rbac_write(cache, member_actors=[("user", user)], group_ids=[group])

    assert cache.get_membership(ActorMembershipKey("user", user)) is None
    assert cache.get_group_namespace(GroupNamespaceKey(group, ns)) is None
    assert cache.get_membership(ActorMembershipKey("user", other_user)) is not None
    assert cache.get_group_namespace(GroupNamespaceKey(other_group, ns)) is not None


async def test_each_evicted_entry_is_broadcast_on_the_subject_its_subscriber_evicts_on() -> None:
    cache = make_cache(FakeStore())
    pub = _RecordingPublisher()
    user, group = uuid7(), uuid7()

    await evict_after_rbac_write(cache, pub, member_actors=[("user", user)], group_ids=[group])

    assert [subject for subject, _ in pub.sent] == ["t.acl.membership.invalidate", "t.acl.assignment.invalidate"]
    membership, assignment = pub.sent[0][1], pub.sent[1][1]
    assert isinstance(membership, MembershipInvalidatePayload)
    assert (membership.actor_type, membership.actor_id) == ("user", user)
    assert isinstance(assignment, AssignmentInvalidatePayload)
    assert assignment.group_id == group


async def test_a_write_that_changed_nothing_publishes_nothing() -> None:
    """an idempotent ensure runs on every request; one that wrote no row must not flood the bus."""
    pub = _RecordingPublisher()

    await evict_after_rbac_write(make_cache(FakeStore()), pub)

    assert pub.sent == []


async def test_a_failed_broadcast_is_logged_and_the_local_eviction_still_holds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """the write committed before this ran; raising would fail work that succeeded, and a retry would publish nothing."""
    cache = make_cache(FakeStore())
    user = uuid7()
    cache.put_membership(ActorMembershipKey("user", user), ())

    with caplog.at_level(logging.WARNING):
        await evict_after_rbac_write(cache, _FailingPublisher(), member_actors=[("user", user)])

    assert cache.get_membership(ActorMembershipKey("user", user)) is None
    assert any("cross-pod invalidation broadcast failed" in record.getMessage() for record in caplog.records)
