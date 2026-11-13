"""The lame-duck handover scenarios, run against a two-node cluster however it was started.

A rolling NATS restart puts each server in lame-duck mode: it stops accepting connections, tells its
clients, and closes them over its lame-duck duration. Left to nats-py, each close is a reconnect --
a real disconnect that drops the replies, owed replies and messages in flight. A
:class:`~threetears.nats.NatsClient` moves first instead, the way a credential renewal does: a
successor opens on another server of the cluster and takes over, and the old connection is kept
for the work it carries until the server closes it.

Two modules run these same scenarios, so their assertions are one text:

- ``test_lame_duck_handover_live.py``: two routed containers on a Docker network, put in lame-duck mode
  with ``docker exec`` (the ``docker`` marker: the laptop and CI);
- ``test_lame_duck_handover_process_live.py``: two routed nats-server child processes
  (:func:`threetears.core.testing.services.running_nats`, the testament pool).

Each passes :func:`hands_its_clients_over` and :func:`pinned_run_stays_in_order` a ``cluster`` factory: an
async context manager yielding the two :class:`Node` s, node A first, once :func:`wait_for_route` has seen a
message cross the route between them (a node is ready before its route is, and a request from A to a responder on
B made before then has nowhere to go).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import timedelta

from threetears.nats import REAUTH_RETIRE_DRAIN_SECONDS, SYNC_REPLY_BUDGET_SECONDS, IncomingMessage, NatsClient, Subject
from threetears.nats.subjects import get_default_namespace, set_default_namespace

#: the image both modules run (or stand for, in process mode)
IMAGE = "nats:2.14.2-alpine"
NS = "ldmlive"
#: the shortest lame-duck duration nats-server accepts; the grace before it starts closing clients.
LAME_DUCK_DURATION = "30s"
LAME_DUCK_GRACE = "3s"
#: how long the workload runs after the server is told to shut down: the grace, the spread of its
#: closes, and its exit, with margin.
WORKLOAD_AFTER_LDM_SECONDS = 38.0
#: how long the workload runs before the server is told to shut down, and again after the handover ends: the
#: stretches on either side where the feed must be in order
STEADY_SECONDS = 1.0
TICK_SECONDS = 0.02


@dataclass
class Node:
    """one server of the cluster.

    :ivar name: its server name
    :ivar uri: its client URI from this host
    :ivar enter_lame_duck: puts its server in lame-duck mode, as a rolling restart does
    """

    name: str
    uri: str
    enter_lame_duck: Callable[[], None]


#: what each module hands the scenarios: two routed nodes, node A first
Cluster = Callable[[], AbstractAsyncContextManager[tuple[Node, Node]]]

#: how long two fresh nodes get to form their route
ROUTE_TIMEOUT_SECONDS = 10.0


def lame_duck_settings() -> str:
    """the lame-duck lines of every node's configuration.

    :return: the configuration lines
    :rtype: str
    """
    return f'lame_duck_duration: "{LAME_DUCK_DURATION}"\nlame_duck_grace_period: "{LAME_DUCK_GRACE}"\n'


async def connect(uri: str, name: str) -> NatsClient:
    """a client connected to one node only.

    :param uri: the node's URI
    :ptype uri: str
    :param name: the client's name
    :ptype name: str
    :return: the connected client
    :rtype: NatsClient
    """
    return await NatsClient.connect(nats_url=uri, nats_subject_namespace=NS, client_name=name, verify_jetstream=False)


def port_of(client: NatsClient) -> int | None:
    """the host port of the node a client's current connection is on.

    :param client: the client
    :ptype client: NatsClient
    :return: the port, or ``None`` while it has none
    :rtype: int | None
    """
    # rationale: which server a connection sits on is exactly what this test observes, and the
    # wrapper has no reason to expose it; nats-py's public connected_url carries it
    url = client.raw.connected_url
    return None if url is None else url.port


async def wait_for_route(first: Node, second: Node) -> None:
    """return once a message published on ``first`` reaches a subscriber on ``second``: the route is up.

    :param first: one node
    :ptype first: Node
    :param second: the other
    :ptype second: Node
    :return: None
    :rtype: None
    :raises AssertionError: no message crossed within :data:`ROUTE_TIMEOUT_SECONDS`
    """
    arrived = asyncio.Event()

    async def _arrive(_msg: IncomingMessage) -> None:
        arrived.set()

    async with contextlib.AsyncExitStack() as clients:
        listener = await connect(second.uri, "route-check-b")
        clients.push_async_callback(listener.shutdown, drain_timeout=timedelta(seconds=2))
        talker = await connect(first.uri, "route-check-a")
        clients.push_async_callback(talker.shutdown, drain_timeout=timedelta(seconds=2))
        await listener.subscribe(Subject.raw("route.check"), cb=_arrive)
        await listener.flush()
        deadline = time.monotonic() + ROUTE_TIMEOUT_SECONDS
        while not arrived.is_set():
            assert time.monotonic() < deadline, f"{first.name} and {second.name} never formed their route"
            await talker.publish_raw(subject=Subject.raw("route.check"), payload=b"?")
            await talker.flush()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(arrived.wait(), timeout=0.1)


@contextlib.asynccontextmanager
async def namespace() -> AsyncIterator[None]:
    """bind this test's subject namespace, and restore the previous one after.

    :return: nothing, yielded
    :rtype: AsyncIterator[None]
    """
    previous = get_default_namespace()
    try:
        yield
    finally:
        set_default_namespace(previous)


async def hands_its_clients_over(cluster: Cluster) -> None:
    """a server in lame-duck mode hands its clients over; it does not drop them.

    A pod on node A keeps requesting a responder and reading a sequenced feed, both served from node B,
    while node A is put in lame-duck mode and then exits. Nothing is lost -- no request fails, no feed
    message is missing -- and the pod is on node B before node A closes it.

    The feed is in order everywhere but the handover itself. Core NATS orders a subject's messages along one
    route, and a server move puts a second route beside the first for a moment: a message still on node A's
    route can arrive after one node B delivered directly, for as long as the pod still holds its old connection
    on node A (kept for the work it carries until node A closes it). So an inversion is allowed only inside the
    handover window -- the feed numbers published from the moment node A entered lame-duck mode to the moment
    the pod's old connection was closed, widened at each end by the most messages the pod was ever behind by
    before the move -- and order is required everywhere else. Nothing that carries data depends on cross-server
    order: a follower walks the numbered change log from its mark, so a late row broadcast only evicts, and an
    epoch is applied by its sequence, never by its arrival.

    :param cluster: the two-node cluster
    :ptype cluster: Cluster
    :return: None
    :rtype: None
    """
    async with cluster() as (first, second):
        async with namespace():
            # a pod that knows both nodes, as a pod given the cluster's URLs does. nats-py shuffles
            # its pool, so the node it lands on is the one shut down, and the other serves.
            pod = await NatsClient.connect(
                nats_url=first.uri,
                cluster_urls=[second.uri],
                nats_subject_namespace=NS,
                client_name="ldm-pod",
                verify_jetstream=False,
            )
            port_by = {int(node.uri.rsplit(":", 1)[1]): node for node in (first, second)}
            node_a = port_by[port_of(pod) or 0]
            node_b = second if node_a is first else first
            port_b = int(node_b.uri.rsplit(":", 1)[1])
            service = await connect(node_b.uri, "ldm-service")

            async def _echo(msg: IncomingMessage) -> None:
                assert msg.reply_subject is not None
                await service.publish_raw_reply(reply_subject=msg.reply_subject, payload=bytes(msg.data))

            await service.subscribe(Subject.raw("svc.echo"), cb=_echo)
            feed_seen: list[int] = []

            async def _on_feed(msg: IncomingMessage) -> None:
                feed_seen.append(int(bytes(msg.data)))

            await pod.subscribe(Subject.raw("feed"), cb=_on_feed)
            await service.flush()
            await pod.flush()

            stop = asyncio.Event()
            failures: list[str] = []
            answered = 0
            published = 0
            moved_at: float | None = None
            #: the feed numbers published when node A entered lame-duck mode, when the pod was on node B, and
            #: when the pod's old connection on node A was closed (the end of the handover)
            published_at_ldm: int | None = None
            published_at_move: int | None = None
            published_at_retired: int | None = None
            # rationale: the end of the handover is when the old connection closes, and only the raw connection
            # it was says so; the wrapper keeps no public handle on a connection it has replaced
            leaving = pod.raw
            #: the most feed messages the pod was behind by before the move: the in-flight depth of one route
            in_flight = 0

            async def _requests() -> None:
                nonlocal answered, moved_at, published_at_move, published_at_retired, in_flight
                seq = 0
                while not stop.is_set():
                    seq += 1
                    try:
                        reply = await pod.request_raw(
                            subject=Subject.raw("svc.echo"), payload=str(seq).encode(), timeout=timedelta(seconds=5)
                        )
                    except Exception as exc:  # noqa: BLE001 -- every failure is the finding
                        failures.append(f"request {seq}: {type(exc).__name__}: {exc}")
                    else:
                        if reply != str(seq).encode():
                            failures.append(f"request {seq}: wrong reply {reply!r}")
                        answered += 1
                    if moved_at is None and port_of(pod) == port_b:
                        moved_at = time.monotonic()
                        published_at_move = published
                    if published_at_ldm is None:
                        in_flight = max(in_flight, published - len(feed_seen))
                    elif published_at_retired is None and leaving.is_closed:
                        published_at_retired = published
                    await asyncio.sleep(TICK_SECONDS)

            async def _feed() -> None:
                nonlocal published
                while not stop.is_set():
                    published += 1
                    await service.publish_raw(subject=Subject.raw("feed"), payload=str(published).encode())
                    await asyncio.sleep(TICK_SECONDS)

            workers = [asyncio.create_task(_requests()), asyncio.create_task(_feed())]
            await asyncio.sleep(STEADY_SECONDS)
            ldm_at = time.monotonic()
            published_at_ldm = published
            node_a.enter_lame_duck()
            await asyncio.sleep(WORKLOAD_AFTER_LDM_SECONDS)
            # the handover ends when node A has closed the pod's old connection: bounded by the lame-duck grace and
            # duration, and the pod's own retirement of it (its longest request, then the drain)
            ends_by = time.monotonic() + SYNC_REPLY_BUDGET_SECONDS + REAUTH_RETIRE_DRAIN_SECONDS
            while not leaving.is_closed and time.monotonic() < ends_by:
                await asyncio.sleep(TICK_SECONDS)
            # and the feed runs on past it, so order after the handover is checked too
            await asyncio.sleep(STEADY_SECONDS)
            stop.set()
            await asyncio.gather(*workers)
            await service.flush()
            await asyncio.sleep(0.5)

            missing = sorted(set(range(1, published + 1)) - set(feed_seen))
            assert published_at_ldm is not None
            # the handover ends when the old connection closes; still open at the end, it spans the whole run
            ended = published if published_at_retired is None else published_at_retired
            window = range(published_at_ldm - max(in_flight, 1), ended + max(in_flight, 1) + 1)
            inversions = [(a, b) for a, b in zip(feed_seen, feed_seen[1:], strict=False) if a > b]
            outside = [number for number in feed_seen if number not in window]
            print(  # noqa: T201 - the counts are the evidence this test exists for
                f"\nlame-duck handover: moved to node B {((moved_at or ldm_at) - ldm_at) * 1000:.0f}ms after "
                f"node A entered lame-duck mode; {answered} requests answered, {len(failures)} failed; "
                f"{published} feed messages published, {len(missing)} missing, "
                f"{len(inversions)} inversion(s) {inversions}, handover window {window.start}..{window.stop - 1} "
                f"(in flight {in_flight}, old connection closed at feed {published_at_retired})"
            )
            assert published_at_retired is not None, "the pod's old connection on node A was never closed"
            assert published > published_at_retired, "the feed ran on past the handover"
            assert moved_at is not None, "the pod never moved to node B"
            assert moved_at - ldm_at < 3.0, "the pod did not move before node A began closing clients"
            assert failures == []
            assert missing == []
            assert outside == sorted(outside), "the feed is in order outside the handover window"
            assert all(a in window and b in window for a, b in inversions), (
                f"an inversion outside the handover window {window.start}..{window.stop - 1}: {inversions}"
            )
            await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await service.shutdown(drain_timeout=timedelta(seconds=2))


async def pinned_run_stays_in_order(cluster: Cluster) -> None:
    """a stream split over two nodes' routes can reorder; a pinned run stays on its first connection.

    :param cluster: the two-node cluster
    :ptype cluster: Cluster
    :return: None
    :rtype: None
    """
    async with cluster() as (node_a, node_b):
        async with namespace():
            reader = await connect(node_b.uri, "pin-reader")
            received: list[int] = []

            async def _on_token(msg: IncomingMessage) -> None:
                received.append(int(bytes(msg.data)))

            await reader.subscribe(Subject.raw("hub.stream.tokens"), cb=_on_token)
            await reader.flush()
            writer = await NatsClient.connect(
                nats_url=node_a.uri,
                cluster_urls=[node_b.uri],
                nats_subject_namespace=NS,
                client_name="pin-writer",
                verify_jetstream=False,
            )
            started_on = port_of(writer)
            pin = writer.publish_pin()
            total = 400
            for seq in range(1, total + 1):
                await writer.publish_raw(subject=Subject.raw("hub.stream.tokens"), payload=str(seq).encode(), pin=pin)
                if seq == total // 2:
                    # the handover lands mid-stream, and is repeated until the successor sits on the
                    # OTHER node -- the case where two publishers travel two routes. nats-py shuffles
                    # its pool, so each renewal lands on either node.
                    for _ in range(20):
                        await writer.renew_connection(retire_after=timedelta(seconds=30))
                        if port_of(writer) != started_on:
                            break
            assert port_of(writer) != started_on, "no renewal landed on the other node"
            unpinned = total + 1
            await writer.publish_raw(subject=Subject.raw("hub.stream.tokens"), payload=str(unpinned).encode())
            for _ in range(200):
                if len(received) == unpinned:
                    break
                await asyncio.sleep(0.01)

            # the run's second half left on its first connection -- the node it started on -- so the
            # whole run is one publisher's, in order. the unpinned publish left on the successor, a
            # second publisher on another route: nothing orders it against the run, and when it
            # overtakes the run's tail that is the reordering a pin exists to prevent.
            run = [seq for seq in received if seq != unpinned]
            overtook = received.index(unpinned) < len(received) - 1
            print(  # noqa: T201 - whether a second publisher overtook is evidence, not an assertion
                f"\npinned run of {total}: in order={run == list(range(1, total + 1))}; a publish on the "
                f"successor, on the other node, overtook the run's tail: {overtook}"
            )
            assert run == list(range(1, total + 1))
            assert sorted(received) == [*range(1, total + 1), unpinned]
            await writer.shutdown(drain_timeout=timedelta(seconds=2))
            await reader.shutdown(drain_timeout=timedelta(seconds=2))
