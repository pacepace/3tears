"""unit tests for :class:`RoomFanout` (channels-task-02, fake-NATS level).

these assert the fanout's contract WITHOUT a NATS container — the
cross-pod truth (echo delivery + the broker hop) is the integration proof
(``tests/integration/.../test_room_fanout_cross_pod.py``). here we prove:

- **ref-counted subscribe/unsubscribe** — the first ``join_room`` on a pod
  subscribes to the room subject; the last ``leave_room`` unsubscribes;
  N joins / N−1 leaves keep the subscription live (transport bookkeeping).
- **publish-only** — ``broadcast`` performs exactly ONE publish and ZERO
  local ``send_text`` on the publish path (the no-double-fan rule: every
  pod, incl. the sender, fans on *receive*, not at publish).
- **exclude is honoured by connection-id** — ``_deliver`` skips exactly
  the excluded connection and delivers to the rest.
- **typed wire round-trips** — ``RoomFrame`` encode/decode is lossless.
- **no ordering/seq state** (design D4) — an enforcement check that
  neither ``RoomFrame`` nor ``RoomFanout`` carries a sequence/ordering
  field, so the durable-source-of-truth rule cannot silently regress.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.channels.presence.fanout import RoomFanout
from threetears.channels.presence.wire import RoomFrame
from threetears.nats import Subjects


class _FakeSocket:
    """live-socket stand-in capturing payloads delivered via ``send_text``."""

    def __init__(self) -> None:
        self.frames: list[str] = []

    async def send_text(self, text: str) -> None:
        self.frames.append(text)


class _BoomSocket:
    """live-socket stand-in whose ``send_text`` always raises."""

    async def send_text(self, text: str) -> None:
        raise RuntimeError("socket is dead")


class _FakeSubscription:
    """records whether it was torn down via ``unsubscribe``."""

    def __init__(self, subject: str) -> None:
        self.subject = subject
        self.unsubscribed = False

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class _FakeNats:
    """fake ``NatsClient``: records publishes + hands back fake subscriptions."""

    def __init__(self) -> None:
        self.publishes: list[tuple[str, RoomFrame]] = []
        self.subscriptions: list[_FakeSubscription] = []

    async def publish(self, *, subject: Any, message: Any) -> None:
        self.publishes.append((str(subject), message))

    async def subscribe_typed(self, *, subject: Any, message_type: Any, cb: Any) -> _FakeSubscription:  # noqa: ARG002
        sub = _FakeSubscription(str(subject))
        self.subscriptions.append(sub)
        return sub


class _GatedNats(_FakeNats):
    """``_FakeNats`` whose ``subscribe_typed`` parks until released.

    lets a test deterministically interleave a ``leave_room`` against an
    in-flight first ``join_room`` subscribe — the window that, if the ref
    count and the subscription handle are not mutated atomically, orphans
    a subscription that no leave can ever tear down.
    """

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_started = asyncio.Event()
        self._gate = asyncio.Event()

    async def subscribe_typed(self, *, subject: Any, message_type: Any, cb: Any) -> _FakeSubscription:
        self.subscribe_started.set()
        await self._gate.wait()
        return await super().subscribe_typed(subject=subject, message_type=message_type, cb=cb)

    def release(self) -> None:
        self._gate.set()


class _FakeRoomState:
    """fake ``RoomState``: records join/leave, serves canned local sockets."""

    def __init__(self, pod_id: str = "pod-a") -> None:
        self._pod_id = pod_id
        self.joins: list[tuple[str, str, str, str]] = []
        self.leaves: list[tuple[str, str]] = []
        # room_id -> list[(connection_id, socket)] this pod delivers to
        self.local: dict[str, list[tuple[str, Any]]] = {}

    @property
    def pod_id(self) -> str:
        return self._pod_id

    async def join(self, room_id: str, connection_id: str, user_id: str, customer_id: str) -> None:
        self.joins.append((room_id, connection_id, user_id, customer_id))

    async def leave(self, room_id: str, connection_id: str) -> None:
        self.leaves.append((room_id, connection_id))

    async def local_member_sockets(self, room_id: str) -> list[tuple[str, Any]]:
        return list(self.local.get(room_id, []))


@pytest.fixture
def nats() -> _FakeNats:
    return _FakeNats()


@pytest.fixture
def state() -> _FakeRoomState:
    return _FakeRoomState()


@pytest.fixture
def fanout(state: _FakeRoomState, nats: _FakeNats) -> RoomFanout:
    return RoomFanout(state, nats)  # type: ignore[arg-type]


ROOM = "cust:story-1:main:scene.md"


class TestRefCountedSubscription:
    """subscribe on the first local member, unsubscribe on the last."""

    async def test_first_join_subscribes(self, fanout: RoomFanout, state: _FakeRoomState, nats: _FakeNats) -> None:
        await fanout.join_room(ROOM, "c1", "user-1", "cust")
        assert state.joins == [(ROOM, "c1", "user-1", "cust")]
        assert len(nats.subscriptions) == 1
        assert nats.subscriptions[0].subject == str(Subjects.room(ROOM))

    async def test_second_join_does_not_resubscribe(self, fanout: RoomFanout, nats: _FakeNats) -> None:
        await fanout.join_room(ROOM, "c1", "user-1", "cust")
        await fanout.join_room(ROOM, "c2", "user-2", "cust")
        assert len(nats.subscriptions) == 1  # still just the first

    async def test_last_leave_unsubscribes(self, fanout: RoomFanout, nats: _FakeNats) -> None:
        await fanout.join_room(ROOM, "c1", "user-1", "cust")
        await fanout.leave_room(ROOM, "c1")
        assert nats.subscriptions[0].unsubscribed is True

    async def test_n_joins_n_minus_one_leaves_keeps_subscription(self, fanout: RoomFanout, nats: _FakeNats) -> None:
        for i in range(3):
            await fanout.join_room(ROOM, f"c{i}", f"user-{i}", "cust")
        await fanout.leave_room(ROOM, "c0")
        await fanout.leave_room(ROOM, "c1")
        assert len(nats.subscriptions) == 1
        assert nats.subscriptions[0].unsubscribed is False  # one member remains
        # the final leave (3→...→0) tears it down.
        await fanout.leave_room(ROOM, "c2")
        assert nats.subscriptions[0].unsubscribed is True

    async def test_rejoin_after_full_leave_resubscribes(self, fanout: RoomFanout, nats: _FakeNats) -> None:
        await fanout.join_room(ROOM, "c1", "user-1", "cust")
        await fanout.leave_room(ROOM, "c1")
        await fanout.join_room(ROOM, "c1", "user-1", "cust")
        assert len(nats.subscriptions) == 2  # a fresh subscription on the second 0→1


class TestPublishOnly:
    """broadcast publishes exactly once and never fans locally at publish time."""

    async def test_broadcast_publishes_once(self, fanout: RoomFanout, nats: _FakeNats) -> None:
        await fanout.broadcast(ROOM, "hello")
        assert len(nats.publishes) == 1
        subject, frame = nats.publishes[0]
        assert subject == str(Subjects.room(ROOM))
        assert isinstance(frame, RoomFrame)
        assert frame.room_id == ROOM
        assert frame.payload == "hello"
        assert frame.exclude is None
        assert frame.origin_pod == "pod-a"

    async def test_broadcast_carries_exclude(self, fanout: RoomFanout, nats: _FakeNats) -> None:
        await fanout.broadcast(ROOM, "hello", exclude="c1")
        _subject, frame = nats.publishes[0]
        assert frame.exclude == "c1"

    async def test_broadcast_does_not_locally_fan(self, fanout: RoomFanout, state: _FakeRoomState) -> None:
        """no ``send_text`` happens on the publish path — every pod fans on receive."""
        sock = _FakeSocket()
        state.local[ROOM] = [("c1", sock), ("c2", _FakeSocket())]
        await fanout.broadcast(ROOM, "hello")
        # publish-only: NO local delivery happened at publish time.
        assert sock.frames == []
        for _cid, s in state.local[ROOM]:
            assert s.frames == []


class TestDeliver:
    """_deliver fans a received frame to local sockets, honouring exclude."""

    async def test_deliver_sends_to_every_local_member(self, fanout: RoomFanout, state: _FakeRoomState) -> None:
        s1, s2 = _FakeSocket(), _FakeSocket()
        state.local[ROOM] = [("c1", s1), ("c2", s2)]
        # _deliver is the subscription callback; unit-tested directly here, exercised
        # via the real broker in the integration proof.
        await fanout._deliver(RoomFrame(room_id=ROOM, payload="p", origin_pod="pod-a"))  # noqa: SLF001 -- testing the subscription callback
        assert s1.frames == ["p"]
        assert s2.frames == ["p"]

    async def test_deliver_skips_excluded_connection(self, fanout: RoomFanout, state: _FakeRoomState) -> None:
        excluded, kept = _FakeSocket(), _FakeSocket()
        state.local[ROOM] = [("c1", excluded), ("c2", kept)]
        await fanout._deliver(RoomFrame(room_id=ROOM, payload="p", exclude="c1", origin_pod="pod-a"))  # noqa: SLF001 -- testing the subscription callback
        assert excluded.frames == []  # excluded by connection-id
        assert kept.frames == ["p"]

    async def test_deliver_no_local_members_is_noop(self, fanout: RoomFanout) -> None:
        # no entry for the room → nothing to deliver, no error.
        await fanout._deliver(RoomFrame(room_id="unknown", payload="p", origin_pod="pod-a"))  # noqa: SLF001 -- testing the subscription callback

    async def test_one_failing_socket_does_not_starve_the_rest(self, fanout: RoomFanout, state: _FakeRoomState) -> None:
        """a socket whose send raises must not abort delivery to the room's other members.

        transient fast-notify is best-effort PER socket: one dead/slow
        handle dropping its own frame is fine, but it must not starve every
        member after it in the snapshot (nor propagate out of the callback
        and deadletter the whole frame).
        """
        before, boom, after = _FakeSocket(), _BoomSocket(), _FakeSocket()
        state.local[ROOM] = [("c1", before), ("c2", boom), ("c3", after)]
        await fanout._deliver(RoomFrame(room_id=ROOM, payload="p", origin_pod="pod-a"))  # noqa: SLF001 -- testing the subscription callback
        assert before.frames == ["p"]
        assert after.frames == ["p"]  # delivered despite the dead socket between them


class TestSubscriptionLifecycleRaces:
    """ref-count + subscription handle must move atomically (no orphaned subs)."""

    async def test_leave_during_in_flight_first_join_does_not_orphan_subscription(self) -> None:
        """a leave racing an in-flight first-join subscribe must not leak the subscription.

        regression for the orphaned-subscription race: ``join_room`` bumping
        the ref-count and the subscription handle landing in the registry
        must be one atomic step. otherwise a ``leave_room`` that runs while
        the first ``join_room`` is parked mid-subscribe sees no handle to
        tear down, the subscribe then completes and stores a handle, and the
        pod stays subscribed to a room it holds nobody in — forever.
        """
        nats = _GatedNats()
        state = _FakeRoomState()
        fanout = RoomFanout(state, nats)  # type: ignore[arg-type]

        join_task = asyncio.create_task(fanout.join_room(ROOM, "c1", "user-1", "cust"))
        await nats.subscribe_started.wait()  # first join is parked inside subscribe

        # a concurrent leave of the same (only) member; let it make all the
        # progress it can while the subscribe is still in flight.
        leave_task = asyncio.create_task(fanout.leave_room(ROOM, "c1"))
        await asyncio.sleep(0)

        nats.release()  # let the subscribe complete
        await asyncio.gather(join_task, leave_task)

        # the room ended with zero members → no live subscription may remain.
        assert len(nats.subscriptions) == 1
        assert nats.subscriptions[0].unsubscribed is True, (
            "subscription orphaned: room has no members but stays subscribed"
        )
        assert ROOM not in fanout._subscriptions  # noqa: SLF001 -- white-box: no retained handle
        assert ROOM not in fanout._ref_counts  # noqa: SLF001 -- white-box: no retained ref


class TestRoomFrameWire:
    """the typed wire envelope round-trips losslessly."""

    async def test_encode_decode_round_trip(self) -> None:
        frame = RoomFrame(room_id=ROOM, payload="body", exclude="c1", origin_pod="pod-a")
        decoded = RoomFrame.model_validate_json(frame.model_dump_json())
        assert decoded == frame

    async def test_exclude_defaults_to_none(self) -> None:
        frame = RoomFrame(room_id=ROOM, payload="body", origin_pod="pod-a")
        assert frame.exclude is None

    async def test_frame_is_frozen(self) -> None:
        frame = RoomFrame(room_id=ROOM, payload="body", origin_pod="pod-a")
        with pytest.raises(Exception):  # pydantic ValidationError on frozen mutation
            frame.payload = "mutated"  # type: ignore[misc]


class TestNoOrderingState:
    """design D4: transient fanout carries NO seq/ordering anywhere."""

    _ORDERING_NAMES = {"seq", "sequence", "ordinal", "order", "index", "offset", "version", "epoch"}

    def test_room_frame_has_no_ordering_field(self) -> None:
        offenders = set(RoomFrame.model_fields) & self._ORDERING_NAMES
        assert offenders == set(), (
            f"RoomFrame declares ordering field(s) {offenders}; the fanout is transient "
            "fast-notify — ordering is the durable op-log's job (design D4)"
        )
        # the field set is exactly the transient transport shape.
        assert set(RoomFrame.model_fields) == {"room_id", "payload", "exclude", "origin_pod"}

    def test_room_fanout_keeps_no_ordering_attribute(self, fanout: RoomFanout) -> None:
        attrs = {name for name in vars(fanout)}
        offenders = {a for a in attrs if a.lstrip("_") in self._ORDERING_NAMES}
        assert offenders == set(), (
            f"RoomFanout holds ordering state {offenders}; the only fanout state is the "
            "per-room subscription ref-count (transport bookkeeping), never a seq (design D4)"
        )
