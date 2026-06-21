"""integration proof: cross-pod presence + self-heal over REAL NATS.

two ``PresenceCollection`` / ``RoomState`` instances over one NATS
testcontainer (two "pods") prove the four acceptance criteria:

1. cross-pod membership — a join on pod A is visible in
   ``members(room)`` on pod B (L2 + invalidation); a leave on A
   removes it on B.
2. self-heal — a connection whose ``date_last_heartbeat`` goes stale is
   evicted by the sweep and pruned from its room-index, observed on both
   pods.
3. concurrency — join/leave churn during a ``local_sockets`` read does
   not race (snapshot iteration).
4. tenancy — two ``customer_id`` s produce disjoint room keys; isolation
   holds even with one customer configured.

the NATS container is session-scoped, so each test scopes its rooms +
connection ids under a per-test-unique suffix (the ``room_key`` fixture)
to stay disjoint. a checkout without docker skips cleanly via the
``nats_container`` fixture's docker gate.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest

from .conftest import Pod

pytestmark = pytest.mark.integration


class _FakeSocket:
    """stand-in live socket handle for the concurrency proof."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def cid() -> Callable[[str], str]:
    """per-test-unique connection-id builder (the KV bucket is shared)."""
    suffix = uuid.uuid4().hex[:12]

    def _build(name: str) -> str:
        return f"{name}-{suffix}"

    return _build


async def _await_until(
    predicate: Callable[[], Awaitable[object]],
    *,
    attempts: int = 50,
    delay: float = 0.05,
) -> object:
    """poll an async predicate until truthy (bounded), for cross-pod settle.

    cross-pod coherence is eventual: the invalidation envelope must
    round-trip the broker before pod B's stale L1 is evicted. poll
    rather than sleep-a-fixed-time so the test is both fast and robust.
    """
    last: object = None
    for _ in range(attempts):
        last = await predicate()
        if last:
            return last
        await asyncio.sleep(delay)
    return last


# ---------------------------------------------------------------------------
# 1. cross-pod membership
# ---------------------------------------------------------------------------


async def test_join_on_a_visible_on_b(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """join on pod A → members(room) on pod B returns the member."""
    pod_a, pod_b = two_pods
    room = room_key("cust", "scene.md")
    conn = cid("conn")

    await pod_a.state.join(room, conn, "user-1", "cust", pod_id="pod-a")

    members = await _await_until(lambda: pod_b.state.members(room))
    assert members, "pod B never saw the member from pod A"
    assert [m.connection_id for m in members] == [conn]  # type: ignore[union-attr]
    assert members[0].pod_id == "pod-a"  # type: ignore[index]
    assert members[0].user_id == "user-1"  # type: ignore[index]


async def test_leave_on_a_clears_on_b(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """leave on pod A → pod B no longer lists the member."""
    pod_a, pod_b = two_pods
    room = room_key("cust", "scene.md")
    conn = cid("conn")

    await pod_a.state.join(room, conn, "user-1", "cust", pod_id="pod-a")
    await _await_until(lambda: pod_b.state.members(room))  # warm B's view

    await pod_a.state.leave(room, conn)

    gone = await _await_until(lambda: _empty(pod_b, room))
    assert gone, "pod B still lists the member after pod A left"


async def _empty(pod: Pod, room: str) -> bool:
    return (await pod.state.members(room)) == []


async def test_two_members_two_pods(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """a member joined on each pod is visible to both pods."""
    pod_a, pod_b = two_pods
    room = room_key("cust", "scene.md")
    conn_a, conn_b = cid("conn-a"), cid("conn-b")

    await pod_a.state.join(room, conn_a, "user-a", "cust", pod_id="pod-a")
    await pod_b.state.join(room, conn_b, "user-b", "cust", pod_id="pod-b")

    seen_a = await _await_until(lambda: _has_both(pod_a, room, conn_a, conn_b))
    seen_b = await _await_until(lambda: _has_both(pod_b, room, conn_a, conn_b))
    assert seen_a, "pod A does not see both members"
    assert seen_b, "pod B does not see both members"


async def _has_both(pod: Pod, room: str, conn_a: str, conn_b: str) -> bool:
    ids = {m.connection_id for m in await pod.state.members(room)}
    return {conn_a, conn_b} <= ids


# ---------------------------------------------------------------------------
# 2. self-heal (heartbeat + sweep)
# ---------------------------------------------------------------------------


async def test_stale_connection_swept_and_pruned_cross_pod(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """a stale connection is evicted by A's sweep and pruned on both pods."""
    pod_a, pod_b = two_pods
    room = room_key("cust", "scene.md")
    stale, fresh = cid("conn-stale"), cid("conn-fresh")

    await pod_a.state.join(room, stale, "user-1", "cust", pod_id="pod-a")
    await pod_a.state.join(room, fresh, "user-2", "cust", pod_id="pod-a")
    pod_a.sweeper.track_many([stale, fresh])

    # force conn-stale's heartbeat outside the 30s window
    entity = await pod_a.collection.connections.get(stale)
    assert entity is not None
    entity.date_last_heartbeat = datetime.now(UTC) - timedelta(seconds=120)
    await pod_a.collection.connections.save_entity(entity)

    evicted = await pod_a.sweeper.run_sweep()
    assert evicted == [stale]

    # observed on pod A
    a_ids = {m.connection_id for m in await pod_a.state.members(room)}
    assert a_ids == {fresh}

    # observed on pod B (cross-pod, after invalidation settles)
    settled = await _await_until(lambda: _is_exactly(pod_b, room, {fresh}))
    assert settled, "pod B still lists the swept connection"


async def test_heartbeat_keeps_connection_alive(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """a connection that keeps heart-beating is NOT swept."""
    pod_a, _ = two_pods
    room = room_key("cust", "scene.md")
    conn = cid("conn")

    await pod_a.state.join(room, conn, "user-1", "cust", pod_id="pod-a")
    pod_a.sweeper.track(conn)
    assert await pod_a.state.heartbeat(conn) is True

    evicted = await pod_a.sweeper.run_sweep()
    assert evicted == []
    assert [m.connection_id for m in await pod_a.state.members(room)] == [conn]


# ---------------------------------------------------------------------------
# 3. concurrency — churn during a local_sockets read does not race
# ---------------------------------------------------------------------------


async def test_churn_during_local_sockets_read_no_race(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """join/leave churn concurrent with local_sockets reads must not raise."""
    pod_a, _ = two_pods
    room = room_key("cust", "scene.md")

    seeded = [cid(f"conn-{i}") for i in range(10)]
    for i, c in enumerate(seeded):
        await pod_a.state.register(c, _FakeSocket(c))
        await pod_a.state.join(room, c, f"user-{i}", "cust", pod_id="pod-a")

    stop = False

    async def churn() -> None:
        i = 0
        while not stop:
            c = cid(f"churn-{i}")
            await pod_a.state.register(c, _FakeSocket(c))
            await pod_a.state.join(room, c, "user-c", "cust", pod_id="pod-a")
            await pod_a.state.leave(room, c)
            await pod_a.state.unregister(c)
            i += 1
            await asyncio.sleep(0)

    async def read() -> None:
        for _ in range(60):
            sockets = await pod_a.state.local_sockets(room)
            for s in sockets:  # iterate the snapshot, not the live map
                _ = s.name
            await asyncio.sleep(0)

    churn_task = asyncio.create_task(churn())
    try:
        await asyncio.gather(read(), read())
    finally:
        stop = True
        await churn_task

    # the 10 seeded local sockets survive the churn
    assert await pod_a.state.local_connection_count() >= 10


# ---------------------------------------------------------------------------
# 4. tenancy — disjoint room keys across customers
# ---------------------------------------------------------------------------


async def test_tenancy_isolation_cross_pod(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """two customers' rooms stay disjoint across pods."""
    pod_a, pod_b = two_pods
    room_a = room_key("custA", "scene.md")
    room_b = room_key("custB", "scene.md")
    conn_a, conn_b = cid("conn-a"), cid("conn-b")

    await pod_a.state.join(room_a, conn_a, "user-1", "custA", pod_id="pod-a")
    await pod_a.state.join(room_b, conn_b, "user-2", "custB", pod_id="pod-a")

    # pod B sees each customer's room independently, with no bleed-through
    seen_a = await _await_until(lambda: _is_exactly(pod_b, room_a, {conn_a}))
    seen_b = await _await_until(lambda: _is_exactly(pod_b, room_b, {conn_b}))
    assert seen_a, "custA room wrong on pod B"
    assert seen_b, "custB room wrong on pod B"

    members_a = await pod_b.state.members(room_a)
    assert members_a[0].customer_id == "custA"


async def _is_exactly(pod: Pod, room: str, expected: set[str]) -> bool:
    ids = {m.connection_id for m in await pod.state.members(room)}
    return ids == expected


# ---------------------------------------------------------------------------
# 5. concurrent same-room CAS — the room-index optimistic-lock under real
#    cross-pod contention (the highest-risk path: a lost update here loses a
#    member silently). exercises both the create-race and the update-race.
# ---------------------------------------------------------------------------


async def test_concurrent_same_room_joins_keep_both_cross_pod(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """two pods joining the SAME room at once must not lose a member (CAS retry).

    Drives genuinely concurrent `add_member` against one room-index KV row from
    two pods over real NATS, so the optimistic-lock conflict + retry branch
    actually fires (the create-race when the row is absent, then the update-race
    once it exists). A last-writer-wins bug would drop one member; the CAS keeps
    both.
    """
    pod_a, pod_b = two_pods
    room = room_key("cust", "scene.md")
    a1, b1 = cid("a1"), cid("b1")

    # create-race: both pods create the room-index row at once. assert on the
    # room-index member set directly (the thing the CAS protects), not via
    # state.members (which would also require per-connection entries).
    await asyncio.gather(
        pod_a.collection.rooms.add_member(room, "cust", a1),
        pod_b.collection.rooms.add_member(room, "cust", b1),
    )
    assert await _await_until(lambda: _room_is(pod_a, room, {a1, b1})), "create-race lost a member (pod A)"
    assert await _await_until(lambda: _room_is(pod_b, room, {a1, b1})), "create-race lost a member (pod B)"

    # update-race: both pods add to the now-existing row at once.
    a2, b2 = cid("a2"), cid("b2")
    await asyncio.gather(
        pod_a.collection.rooms.add_member(room, "cust", a2),
        pod_b.collection.rooms.add_member(room, "cust", b2),
    )
    assert await _await_until(lambda: _room_is(pod_a, room, {a1, b1, a2, b2})), "update-race lost a member"


async def _room_is(pod: Pod, room: str, expected: set[str]) -> bool:
    """the room-index member set (the CAS-protected value), direct — no connection entries."""
    return set(await pod.collection.rooms.members(room)) == expected


# ---------------------------------------------------------------------------
# 6. l2_key robustness — room ids that the old ':'->'=' replace would collide,
#    and out-of-grammar characters that a bare interpolation would reject.
# ---------------------------------------------------------------------------


async def test_special_char_room_ids_do_not_collide_or_throw(
    two_pods: tuple[Pod, Pod],
    cid: Callable[[str], str],
) -> None:
    """room ids with `=`/spaces map to distinct, valid KV keys (SHA-256), no bleed.

    `"x=y:z"` and `"x:y=z"` both collapse to `"x=y=z"` under a naive `:`->`=`
    replace — a silent cross-room collision. A space is outside the JetStream KV
    grammar and would throw `KvError` under a bare interpolation. The SHA-256 key
    must keep them distinct and valid.
    """
    pod_a, pod_b = two_pods
    suffix = uuid.uuid4().hex[:12]
    r1 = f"x=y:z:{suffix}"  # collides with r2 under ':'->'='
    r2 = f"x:y=z:{suffix}"
    r3 = f"cust:story:main:my file {suffix}.md"  # space: out-of-grammar
    c1, c2, c3 = cid("c1"), cid("c2"), cid("c3")

    await pod_a.collection.rooms.add_member(r1, "cust", c1)
    await pod_a.collection.rooms.add_member(r2, "cust", c2)
    await pod_a.collection.rooms.add_member(r3, "cust", c3)  # must not raise KvError

    # no collision: r1 holds only c1, not c2 (a `:`->`=` collision would bleed c2 in).
    assert await _await_until(lambda: _room_is(pod_b, r1, {c1})), "r1 collided"
    assert await _await_until(lambda: _room_is(pod_b, r2, {c2})), "r2 collided"
    assert await _await_until(lambda: _room_is(pod_b, r3, {c3})), "space room id failed"
