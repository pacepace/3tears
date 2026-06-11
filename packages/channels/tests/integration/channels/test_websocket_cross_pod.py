"""integration proofs (REAL NATS): the cross-pod WebSocket handler (channels-task-03).

This is the headline of the channels stack: two ``WebSocketHandler``s over
ONE NATS testcontainer (two pods), each with its own ``RoomState`` +
``RoomFanout`` and the injected seams (an ``op_handler`` returning
incrementing op-log seqs, an ``AclCache``-backed authorizer over the REAL
``agent-acl`` ``authorize_on_entity``). The proofs:

- **vertical slice:** A connects→auth→``join``→``editor.op`` on pod A; B
  (same room, pod B, kept joined + listening) **receives** the op frame
  carrying the op-log seq; A does **not** (exclude); a client in another
  room does not. Delivery arrives via the task-02 fanout subscription
  callback, so we assert on B's socket's ``sent``.
- **authorization:** a denied user's ``join`` writes **no** presence row
  (asserted via ``RoomState.members`` on **both** pods) and triggers
  **no** broadcast.
- **reconnect-resume:** a reconnecting client with ``last_seq=N`` receives
  the replayed tail (fake ``replay_source`` yielding N+1..M) **before** any
  live frame, in order.
- **disconnect cleanup:** dropping the socket leaves the room (presence
  gone on **both** pods) and unregisters the local handle.

A checkout without docker skips cleanly via the ``nats_container`` gate.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    AclCache,
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.channels.frames import Frame, OpResult
from threetears.channels.presence.collection import PresenceCollection
from threetears.channels.presence.fanout import RoomFanout
from threetears.channels.presence.l1_cache import create_presence_l1_backend
from threetears.channels.presence.room_state import RoomState
from threetears.channels.websocket import WebSocketHandler
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_NAMESPACE = "channelstask03it"
_NS_TYPE = "story"


# -- a real allowing/denying AclCache (REAL agent-acl evaluation) -----------


@dataclass
class _FakeStore:
    """in-memory MembershipLoader + GrantLoader (the acl test pattern, inlined)."""

    memberships: list[GroupMembership] = field(default_factory=list)
    groups: dict[UUID, Group] = field(default_factory=dict)
    roles: dict[UUID, Role] = field(default_factory=dict)
    assignments: list[RoleAssignment] = field(default_factory=list)

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        return tuple(m for m in self.memberships if m.member_type == MemberType.USER and m.member_id == user_id)

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        return tuple(m for m in self.memberships if m.member_type == MemberType.AGENT and m.member_id == agent_id)

    async def load_assignments_for_groups(
        self, group_ids: tuple[UUID, ...], namespace: Namespace
    ) -> tuple[RoleAssignment, ...]:
        ids = set(group_ids)
        return tuple(a for a in self.assignments if a.group_id in ids)

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        return {rid: self.roles[rid] for rid in role_ids if rid in self.roles}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, object]:
        return {gid: self.groups[gid] for gid in group_ids if gid in self.groups}


@dataclass
class _StubNs:
    """duck-typed namespace entity (the four fields authorize_on_entity reads)."""

    id: UUID
    customer_id: UUID
    namespace_type: str
    owner_agent_id: UUID


def _allowing_cache(user_id: UUID, customer_id: UUID) -> tuple[AclCache, _StubNs]:
    """build an AclCache granting ``user_id`` room.join + entry.write on the ns.

    Seeds a customer-scoped group with ``user_id`` as member, a Role
    granting ``{story: [room.join, entry.write]}``, and a TYPE_CUSTOMER
    assignment binding the group to the role for the namespace's type +
    customer. The cache evaluates the REAL ``agent-acl`` path.
    """
    ns = _StubNs(id=uuid4(), customer_id=customer_id, namespace_type=_NS_TYPE, owner_agent_id=uuid4())
    role = Role(
        id=uuid4(),
        name="Collaborator",
        permissions={_NS_TYPE: frozenset(["room.join", "entry.write"])},
        is_built_in=True,
    )
    group = Group(id=uuid4(), name="collab", customer_id=customer_id)
    membership = GroupMembership(
        group_id=group.id, member_type=MemberType.USER, member_id=user_id, customer_id=customer_id
    )
    assignment = RoleAssignment(
        id=uuid4(),
        role_id=role.id,
        group_id=group.id,
        scope_type=ScopeType.TYPE_CUSTOMER,
        scope_namespace_id=None,
        scope_namespace_type=_NS_TYPE,
        scope_customer_id=customer_id,
    )
    store = _FakeStore()
    store.roles[role.id] = role
    store.groups[group.id] = group
    store.memberships.append(membership)
    store.assignments.append(assignment)
    cache = AclCache(membership_loader=store, grant_loader=store)
    return cache, ns


def _empty_cache(customer_id: UUID) -> tuple[AclCache, _StubNs]:
    """build an AclCache with NO grants — every action denies."""
    ns = _StubNs(id=uuid4(), customer_id=customer_id, namespace_type=_NS_TYPE, owner_agent_id=uuid4())
    store = _FakeStore()
    return AclCache(membership_loader=store, grant_loader=store), ns


# -- a blocking mock socket: stays "live" so its pod keeps its subscription --


class _BlockingMockWebSocket:
    """WebSocketProtocol mock whose ``receive_text`` blocks after queued frames.

    Client B must stay in its message loop (so its pod keeps the room
    subscription alive) while A broadcasts. After delivering its queued
    frames, ``receive_text`` awaits an Event that the test sets to end the
    loop — modelling a live, idle, still-subscribed peer. ``sent`` is what
    the fanout delivered to B.
    """

    def __init__(self, messages: list[str] | None = None, query_params: dict[str, str] | None = None) -> None:
        self.messages: list[str] = list(messages or [])
        self.sent: list[str] = []
        self.accepted = False
        self.closed = False
        self.query_params: dict[str, str] = query_params or {}
        self._release = asyncio.Event()

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        # idle but alive: block until the test releases, then end the loop.
        await self._release.wait()
        raise Exception("disconnect")

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True

    def release(self) -> None:
        """unblock ``receive_text`` so the connection's loop ends + cleans up."""
        self._release.set()


async def _auth_for(user_id: UUID, customer_id: UUID) -> Callable[[str], Awaitable[dict[str, object] | None]]:
    """build an auth validator returning a fixed user_id + customer_id (UUID strings)."""

    async def _v(token: str) -> dict[str, object] | None:
        if token == "valid-token":
            return {"user_id": str(user_id), "customer_id": str(customer_id)}
        return None

    return _v


# -- two-pod handler wiring over one real NATS container --------------------


@dataclass
class _PodHandler:
    """one pod's RoomState + RoomFanout + WebSocketHandler over shared NATS."""

    client: NatsClient
    state: RoomState
    fanout: RoomFanout

    def handler(
        self,
        *,
        auth_validator: Callable[[str], Awaitable[dict[str, object] | None]],
        acl_cache: AclCache,
        ns: _StubNs,
        op_start: int = 100,
        replay_source: object | None = None,
    ) -> WebSocketHandler:
        op = _IncrementingOpHandler(op_start)

        async def _resolver(room_id: str) -> _StubNs:
            return ns

        return WebSocketHandler(
            router=_NullRouter(),
            auth_validator=auth_validator,
            room_state=self.state,
            room_fanout=self.fanout,
            acl_cache=acl_cache,
            ns_resolver=_resolver,  # type: ignore[arg-type]
            op_handler=op,  # type: ignore[arg-type]
            replay_source=replay_source,  # type: ignore[arg-type]
        )


class _NullRouter:
    async def route_inbound(self, message: object) -> None:
        return None


class _IncrementingOpHandler:
    """op_handler seam: returns op-log seqs N, N+1, … and records appends."""

    def __init__(self, start: int) -> None:
        self._next = start
        self.appended: list[Frame] = []

    async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
        self.appended.append(frame)
        seq = self._next
        self._next += 1
        return OpResult(seq=seq)


async def _make_pod(nats_url: str, pod_id: str) -> _PodHandler:
    client = await NatsClient.connect(
        nats_url=nats_url, nats_subject_namespace=_NAMESPACE, client_name=f"task03-{pod_id}"
    )
    l1 = create_presence_l1_backend()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=client)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = PresenceCollection(registry, config, nats_client=client)
    await registry.start_invalidation_listener(client)
    state = RoomState(collection, pod_id=pod_id)
    fanout = RoomFanout(state, client)
    return _PodHandler(client=client, state=state, fanout=fanout)


@pytest.fixture
async def two_handlers(nats_container: str) -> AsyncIterator[tuple[_PodHandler, _PodHandler]]:
    set_default_namespace(_NAMESPACE)
    pod_a = await _make_pod(nats_container, "pod-a")
    pod_b = await _make_pod(nats_container, "pod-b")
    try:
        yield pod_a, pod_b
    finally:
        await pod_a.client.shutdown()
        await pod_b.client.shutdown()


@pytest.fixture
def room_key() -> Callable[[UUID, str], str]:
    """build a per-test-unique ``{customer}:{story}:{branch}:{file}`` room id."""
    suffix = uuid.uuid4().hex[:12]

    def _build(customer: UUID, file: str) -> str:
        return f"{customer.hex}:story-{suffix}:main:{file}"

    return _build


async def _await_until(predicate: Callable[[], object], *, attempts: int = 60, delay: float = 0.05) -> object:
    """poll a sync OR async predicate until truthy (bounded), for cross-pod settle."""
    last: object = None
    for _ in range(attempts):
        result = predicate()
        last = await result if inspect.isawaitable(result) else result
        if last:
            return last
        await asyncio.sleep(delay)
    return last


# ============================================================
# the vertical slice
# ============================================================


async def test_vertical_slice_cross_pod_editor_op(
    two_handlers: tuple[_PodHandler, _PodHandler],
    room_key: Callable[[UUID, str], str],
) -> None:
    """A's editor.op reaches B (cross-pod) carrying the op-log seq; A excluded; other room silent."""
    pod_a, pod_b = two_handlers
    customer = uuid4()
    user_a, user_b, user_c = uuid4(), uuid4(), uuid4()
    room = room_key(customer, "scene.md")
    other_room = room_key(customer, "other.md")

    cache_a, ns = _allowing_cache(user_a, customer)
    cache_b, _ = _allowing_cache(user_b, customer)
    cache_c, _ = _allowing_cache(user_c, customer)
    # B + C share the SAME ns entity object so the room→ns policy is identical.
    cache_b_ns = ns
    cache_c_ns = ns

    # --- B joins the room on pod B and stays live + subscribed ---
    handler_b = pod_b.handler(auth_validator=await _auth_for(user_b, customer), acl_cache=cache_b, ns=cache_b_ns)
    ws_b = _BlockingMockWebSocket(
        messages=[json.dumps({"type": "join", "room": room})], query_params={"token": "valid-token"}
    )
    task_b = asyncio.create_task(handler_b.handle_connection(ws_b))

    # --- C joins a DIFFERENT room on pod B; must NOT receive A's op ---
    handler_c = pod_b.handler(auth_validator=await _auth_for(user_c, customer), acl_cache=cache_c, ns=cache_c_ns)
    ws_c = _BlockingMockWebSocket(
        messages=[json.dumps({"type": "join", "room": other_room})], query_params={"token": "valid-token"}
    )
    task_c = asyncio.create_task(handler_c.handle_connection(ws_c))

    # wait until B + C have actually joined (their pods now subscribed to their rooms).
    await _await_until(lambda: any("connected" in s for s in ws_b.sent))
    await _await_until(lambda: any("connected" in s for s in ws_c.sent))
    await asyncio.sleep(0.3)  # let the pod-B subscriptions settle on the broker

    # --- A connects on pod A, joins, sends editor.op ---
    handler_a = pod_a.handler(auth_validator=await _auth_for(user_a, customer), acl_cache=cache_a, ns=ns, op_start=500)
    ws_a = _BlockingMockWebSocket(
        messages=[
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": "editor.op", "room": room, "payload": "replace(0,1,'x')"}),
        ],
        query_params={"token": "valid-token"},
    )
    task_a = asyncio.create_task(handler_a.handle_connection(ws_a))

    # B receives the broadcast op frame carrying the op-log seq (500).
    def _b_got_op() -> bool:
        return any(json.loads(s).get("type") == "editor.op" for s in ws_b.sent)

    assert await _await_until(_b_got_op), "B (cross-pod, same room) never received A's editor.op"
    op_frames = [json.loads(s) for s in ws_b.sent if json.loads(s).get("type") == "editor.op"]
    assert op_frames[0]["seq"] == 500, "broadcast op frame did not carry the op-log handler's seq"
    assert op_frames[0]["payload"] == "replace(0,1,'x')"

    # grace, then the negatives: A excluded (no editor.op echo), C (other room) silent.
    await asyncio.sleep(0.3)
    a_op_echo = [s for s in ws_a.sent if json.loads(s).get("type") == "editor.op"]
    assert a_op_echo == [], "A received its own editor.op (exclude failed)"
    c_ops = [s for s in ws_c.sent if json.loads(s).get("type") == "editor.op"]
    assert c_ops == [], "C (different room) received the op frame"

    # tear down the three live loops.
    for ws, task in ((ws_a, task_a), (ws_b, task_b), (ws_c, task_c)):
        ws.release()
        await asyncio.wait_for(task, timeout=5)


# ============================================================
# authorization integration
# ============================================================


async def test_denied_join_writes_no_presence_and_no_broadcast(
    two_handlers: tuple[_PodHandler, _PodHandler],
    room_key: Callable[[UUID, str], str],
) -> None:
    """a denied join writes NO presence row (both pods) and triggers NO broadcast."""
    pod_a, pod_b = two_handlers
    customer = uuid4()
    user_denied = uuid4()
    room = room_key(customer, "scene.md")

    cache, ns = _empty_cache(customer)  # no grants → every action denies
    handler = pod_a.handler(auth_validator=await _auth_for(user_denied, customer), acl_cache=cache, ns=ns)
    ws = _BlockingMockWebSocket(
        messages=[json.dumps({"type": "join", "room": room})], query_params={"token": "valid-token"}
    )
    task = asyncio.create_task(handler.handle_connection(ws))

    # the join was denied → an error frame came back.
    assert await _await_until(lambda: any(json.loads(s).get("type") == "error" for s in ws.sent)), (
        "denied join did not produce an error frame"
    )
    await asyncio.sleep(0.2)

    # NO presence row on either pod's view of the room.
    members_a = await pod_a.state.members(room)
    members_b = await pod_b.state.members(room)
    assert members_a == [], "denied join wrote a presence row (pod A)"
    assert members_b == [], "denied join wrote a presence row (pod B)"

    ws.release()
    await asyncio.wait_for(task, timeout=5)


# ============================================================
# reconnect-resume
# ============================================================


async def test_reconnect_resume_replays_tail_before_live(
    two_handlers: tuple[_PodHandler, _PodHandler],
    room_key: Callable[[UUID, str], str],
) -> None:
    """a reconnect with last_seq=N receives the replayed tail (N+1..M) before any live frame, in order."""
    pod_a, _pod_b = two_handlers
    customer = uuid4()
    user = uuid4()
    room = room_key(customer, "scene.md")
    cache, ns = _allowing_cache(user, customer)

    async def _replay(room_id: str, from_seq: int) -> AsyncIterator[str]:
        for seq in range(from_seq + 1, from_seq + 4):  # N+1, N+2, N+3
            yield Frame(type="editor.op", room=room_id, payload=f"op-{seq}", seq=seq).model_dump_json()

    handler = pod_a.handler(
        auth_validator=await _auth_for(user, customer), acl_cache=cache, ns=ns, replay_source=_replay
    )
    # the connect-path resume cursor rides the query string (room + last_seq=10).
    ws = _BlockingMockWebSocket(
        messages=[],
        query_params={"token": "valid-token", "resume_room": room, "resume_seq": "10"},
    )
    task = asyncio.create_task(handler.handle_connection(ws))

    def _got_three_ops() -> bool:
        return len([s for s in ws.sent if json.loads(s).get("type") == "editor.op"]) >= 3

    assert await _await_until(_got_three_ops), "resume did not replay the tail"
    replayed = [json.loads(s) for s in ws.sent if json.loads(s).get("type") == "editor.op"]
    assert [f["seq"] for f in replayed] == [11, 12, 13], "replayed tail out of order / wrong range"

    # the replay ran BEFORE the live loop: the connected frame precedes the ops.
    types_in_order = [json.loads(s).get("type") for s in ws.sent]
    assert types_in_order.index("connected") < types_in_order.index("editor.op"), (
        "an editor.op preceded the connected frame — resume must finish before going live"
    )

    ws.release()
    await asyncio.wait_for(task, timeout=5)


# ============================================================
# disconnect cleanup
# ============================================================


async def test_disconnect_leaves_room_on_both_pods_and_unregisters(
    two_handlers: tuple[_PodHandler, _PodHandler],
    room_key: Callable[[UUID, str], str],
) -> None:
    """dropping the socket leaves the room (presence gone on both pods) + unregisters."""
    pod_a, pod_b = two_handlers
    customer = uuid4()
    user = uuid4()
    room = room_key(customer, "scene.md")
    cache, ns = _allowing_cache(user, customer)

    handler = pod_a.handler(auth_validator=await _auth_for(user, customer), acl_cache=cache, ns=ns)
    ws = _BlockingMockWebSocket(
        messages=[json.dumps({"type": "join", "room": room})], query_params={"token": "valid-token"}
    )
    task = asyncio.create_task(handler.handle_connection(ws))

    # wait until the join is visible cross-pod.
    await _await_until(lambda: any("connected" in s for s in ws.sent))
    assert await _await_until(lambda: _members_nonempty(pod_b, room)), "join never became visible on pod B"

    # local handle is registered while live.
    assert await pod_a.state.local_connection_count() == 1

    # drop the socket → disconnect path leaves the room + unregisters.
    ws.release()
    await asyncio.wait_for(task, timeout=5)
    await asyncio.sleep(0.2)

    members_a = await pod_a.state.members(room)
    members_b = await pod_b.state.members(room)
    assert members_a == [], "presence row remained on pod A after disconnect"
    assert members_b == [], "presence row remained on pod B after disconnect"
    assert await pod_a.state.local_connection_count() == 0, "live handle was not unregistered"


async def _members_nonempty(pod: _PodHandler, room: str) -> bool:
    members = await pod.state.members(room)
    return len(members) > 0
