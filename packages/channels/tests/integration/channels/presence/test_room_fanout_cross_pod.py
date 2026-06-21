"""integration proof: cross-pod room fanout over REAL NATS (channels-task-02).

two ``RoomFanout`` instances (each wrapping its pod's ``RoomState``) over
ONE NATS testcontainer — two "pods" — prove the one net-new mechanism of
the cross-pod design (``docs/channels-cross-pod-design.md`` D1 / Shard B):
a room message published on pod A is delivered to room members on **every**
pod, each pod fanning to its **own** live sockets.

The single scenario exercises all four delivery rules at once:

- pod A holds ``a1`` (the sender, excluded) and ``a2`` (a second local
  member) of room ``R``;
- pod B holds ``b1`` (a member of ``R``) and ``b2`` (a member of a
  *different* room — a non-member of ``R``);
- ``fanout_A.broadcast(R, payload, exclude=a1)`` must result in:
  - ``a2`` **receives** — **same-pod** delivery via the sender pod's own
    subscription (the NATS-echo path; a no-echo regression drops this);
  - ``b1`` **receives** — **cross-pod** delivery;
  - ``a1`` does **NOT** — excluded by connection-id, honoured after the hop;
  - ``b2`` does **NOT** — not a member of ``R``.

The NATS container is session-scoped, so rooms + connection ids are scoped
under a per-test-unique suffix to stay disjoint. A checkout without docker
skips cleanly via the ``nats_container`` fixture's docker gate.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable

import pytest

from .conftest import Pod

pytestmark = pytest.mark.integration


class _RecordingSocket:
    """live-socket stand-in that captures frames delivered via ``send_text``.

    matches the ``WebSocketProtocol.send_text`` shape the fanout delivers
    through; ``frames`` is the ordered list of payloads this socket received.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.frames: list[str] = []

    async def send_text(self, text: str) -> None:
        self.frames.append(text)


@pytest.fixture
def cid() -> Callable[[str], str]:
    """per-test-unique connection-id builder (the KV bucket is shared)."""
    suffix = uuid.uuid4().hex[:12]

    def _build(name: str) -> str:
        return f"{name}-{suffix}"

    return _build


async def _await_until(
    predicate: Callable[[], Awaitable[object] | object],
    *,
    attempts: int = 50,
    delay: float = 0.05,
) -> object:
    """poll a predicate until truthy (bounded), for cross-pod settle.

    accepts either a sync predicate (``lambda: sock.frames == [...]``) or
    an async one; an awaitable result is awaited, a plain value is used
    as-is. the delivery assertions below are unchanged — this only lets
    the harness poll the sync membership lambdas the test passes.
    """
    last: object = None
    for _ in range(attempts):
        result = predicate()
        last = await result if inspect.isawaitable(result) else result
        if last:
            return last
        await asyncio.sleep(delay)
    return last


async def test_broadcast_reaches_every_pod_honours_exclude_and_membership(
    two_pods: tuple[Pod, Pod],
    room_key: Callable[[str, str], str],
    cid: Callable[[str], str],
) -> None:
    """one broadcast: same-pod + cross-pod deliver; exclude + non-member stay silent."""
    pod_a, pod_b = two_pods
    room = room_key("cust", "scene.md")
    other_room = room_key("cust", "other.md")
    payload = f"frame-{uuid.uuid4().hex[:8]}"

    a1, a2 = cid("a1"), cid("a2")  # both on pod A, in room R (a1 is the sender)
    b1 = cid("b1")  # on pod B, in room R
    b2 = cid("b2")  # on pod B, in a DIFFERENT room (non-member of R)

    sock_a1, sock_a2 = _RecordingSocket(a1), _RecordingSocket(a2)
    sock_b1, sock_b2 = _RecordingSocket(b1), _RecordingSocket(b2)

    # register live handles, then join rooms through the fanout (join_room both
    # records membership AND subscribes the pod to the room subject on the first
    # local member, so the pod is listening before the broadcast).
    await pod_a.state.register(a1, sock_a1)
    await pod_a.state.register(a2, sock_a2)
    await pod_b.state.register(b1, sock_b1)
    await pod_b.state.register(b2, sock_b2)

    await pod_a.fanout.join_room(room, a1, "user-a1", "cust")
    await pod_a.fanout.join_room(room, a2, "user-a2", "cust")
    await pod_b.fanout.join_room(room, b1, "user-b1", "cust")
    await pod_b.fanout.join_room(other_room, b2, "user-b2", "cust")

    # one publish; every pod with a local member of R fans to its own sockets.
    await pod_a.fanout.broadcast(room, payload, exclude=a1)

    # positives (delivery is eventual across the broker): poll until delivered.
    assert await _await_until(lambda: sock_a2.frames == [payload]), (
        "a2 (same-pod, non-sender) never received the frame — a no-echo regression "
        "would drop the sender pod's own delivery"
    )
    assert await _await_until(lambda: sock_b1.frames == [payload]), "b1 (cross-pod member) never received the frame"

    # negatives (proving absence): once both positives have settled, give the
    # broker a brief grace and assert the excluded sender and the non-member of R
    # are still empty.
    await asyncio.sleep(0.2)
    assert sock_a1.frames == [], "a1 was excluded but still received the frame"
    assert sock_b2.frames == [], "b2 is not a member of the room but received the frame"

    # exactly-once: nobody got a duplicate (publish-only; no double local fan).
    assert sock_a2.frames == [payload]
    assert sock_b1.frames == [payload]
