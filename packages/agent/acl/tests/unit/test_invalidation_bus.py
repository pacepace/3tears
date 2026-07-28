"""The rbac invalidation bus: what each publisher sends, and what each subscriber evicts."""

from __future__ import annotations

import inspect
from typing import Any
from uuid import uuid7

import pytest
from pydantic import BaseModel

from threetears.agent.acl.cache import ActorMembershipKey
from threetears.agent.acl.invalidation import (
    AssignmentInvalidatePayload,
    MembershipInvalidatePayload,
    RoleInvalidatePayload,
)
from tests.unit._fake_loaders import FakeStore, make_cache

from threetears.agent.acl.invalidation_bus import (
    publish_assignment_invalidation,
    publish_membership_invalidation,
    publish_role_invalidation,
    subscribe_acl_invalidation,
)


class _RecordingPublisher:
    """# parity-with: threetears.agent.acl.invalidation_bus.AclInvalidationPublisher"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, BaseModel]] = []

    async def publish(self, *, subject: Any, message: BaseModel) -> None:
        self.sent.append((str(getattr(subject, "path", subject)), message))


class _RecordingSubscriber:
    """# parity-with: threetears.agent.acl.invalidation_bus.AclInvalidationSubscriber"""

    def __init__(self) -> None:
        self.bound: dict[str, Any] = {}
        self.kwargs: list[dict[str, Any]] = []

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any) -> Any:
        # The protocol's exact signature, not **kw. Swallowing arbitrary kwargs meant a misspelled
        # or extra one that the real NatsClient would TypeError on passed here in silence.
        self.kwargs.append({"subject": subject, "cb": cb, "message_type": message_type})
        self.bound[str(subject.path)] = cb
        return object()


@pytest.fixture(autouse=True)
def _namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "t")


async def test_membership_publish_names_the_actor() -> None:
    pub = _RecordingPublisher()
    actor = uuid7()
    await publish_membership_invalidation(pub, actor_type="user", actor_id=actor)

    subject, message = pub.sent[0]
    assert subject == "t.acl.membership.invalidate"
    assert isinstance(message, MembershipInvalidatePayload)
    assert message.actor_id == actor and message.actor_type == "user"


async def test_assignment_publish_names_the_group() -> None:
    pub = _RecordingPublisher()
    group = uuid7()
    await publish_assignment_invalidation(pub, group_id=group)

    subject, message = pub.sent[0]
    assert subject == "t.acl.assignment.invalidate"
    assert isinstance(message, AssignmentInvalidatePayload)
    assert message.group_id == group


async def test_role_publish_carries_the_empty_payload() -> None:
    pub = _RecordingPublisher()
    await publish_role_invalidation(pub)

    subject, message = pub.sent[0]
    assert subject == "t.acl.role.invalidate"
    assert isinstance(message, RoleInvalidatePayload)


@pytest.mark.parametrize(
    "publish",
    [
        lambda p: publish_membership_invalidation(p, actor_type="user", actor_id=uuid7()),
        lambda p: publish_assignment_invalidation(p, group_id=uuid7()),
        publish_role_invalidation,
    ],
)
async def test_a_none_publisher_is_an_explicit_no_op(publish: Any) -> None:
    """A deployment with no broker falls back to TTL expiry rather than failing the mutation."""
    await publish(None)  # must not raise


async def test_subscribe_binds_all_three_subjects_without_a_queue_group() -> None:
    """Plain subscriptions, never a queue group.

    A queue group delivers each broadcast to exactly ONE member, so every other pod would keep
    serving the decision it was just told to drop -- the precise failure this bus prevents.
    """
    sub = _RecordingSubscriber()
    await subscribe_acl_invalidation(sub, make_cache(FakeStore()))

    assert set(sub.bound) == {
        "t.acl.membership.invalidate",
        "t.acl.assignment.invalidate",
        "t.acl.role.invalidate",
    }
    # subscribe_typed DOES take a `queue`; the bus must never pass one. Asserted against the real
    # client's signature rather than against the fake, which cannot prove an omission by itself.
    from threetears.nats import NatsClient

    assert "queue" in inspect.signature(NatsClient.subscribe_typed).parameters, (
        "if the client loses its queue parameter this assertion is meaningless and must be rewritten"
    )
    for kw in sub.kwargs:
        assert "queue" not in kw, "a queue group would deliver each broadcast to ONE pod only"


async def test_a_membership_broadcast_drops_that_actor_and_leaves_others() -> None:
    cache, sub = make_cache(FakeStore()), _RecordingSubscriber()
    kept, dropped = uuid7(), uuid7()
    kept_key = ActorMembershipKey(actor_kind="user", actor_id=kept)
    dropped_key = ActorMembershipKey(actor_kind="user", actor_id=dropped)
    cache.put_membership(dropped_key, ())
    cache.put_membership(kept_key, ())
    await subscribe_acl_invalidation(sub, cache)

    await sub.bound["t.acl.membership.invalidate"](MembershipInvalidatePayload(actor_type="user", actor_id=dropped))

    assert cache.get_membership(dropped_key) is None
    assert cache.get_membership(kept_key) is not None, "only the named actor is evicted"


async def test_a_role_broadcast_clears_every_layer() -> None:
    """A role's permissions changed: every assignment referencing it may now decide differently."""
    cache, sub = make_cache(FakeStore()), _RecordingSubscriber()
    key = ActorMembershipKey(actor_kind="user", actor_id=uuid7())
    cache.put_membership(key, ())
    await subscribe_acl_invalidation(sub, cache)

    await sub.bound["t.acl.role.invalidate"](RoleInvalidatePayload())

    assert cache.get_membership(key) is None


async def test_a_failed_bind_unwinds_the_subscriptions_already_made() -> None:
    """A partial bind must not orphan live subscriptions.

    The three binds used to be elements of a list literal, so a failure on the second or third
    propagated and the caller received NO handles for the ones that had succeeded -- nothing could
    ever unsubscribe them, and they held the AclCache alive for the process lifetime.
    """

    class _FailsOnSecond:
        """# parity-with: threetears.agent.acl.invalidation_bus.AclInvalidationSubscriber"""

        def __init__(self) -> None:
            self.made: list[_Unsubscribable] = []

        async def subscribe_typed(self, **_kw: Any) -> Any:
            if len(self.made) == 1:
                raise RuntimeError("broker went away mid-bind")
            sub = _Unsubscribable()
            self.made.append(sub)
            return sub

    class _Unsubscribable:
        def __init__(self) -> None:
            self.unsubscribed = False

        async def unsubscribe(self) -> None:
            self.unsubscribed = True

    nats = _FailsOnSecond()
    with pytest.raises(RuntimeError):
        await subscribe_acl_invalidation(nats, make_cache(FakeStore()))

    assert nats.made and all(s.unsubscribed for s in nats.made), "the first bind must be unwound"
