"""Integration test: a server in lame-duck mode hands its clients over; it does not drop them.

A rolling NATS restart puts each server in lame-duck mode: it stops accepting connections, tells its
clients, and closes them over its lame-duck duration. Left to nats-py, each close is a reconnect --
a real disconnect that drops the replies, owed replies and messages in flight. A
:class:`~threetears.nats.NatsClient` moves first instead, the way a credential renewal does: a
successor opens on another server of the cluster and takes over, and the old connection is kept
for the work it carries until the server closes it.

Proven against a real two-node cluster (nats 2.14.2, what the clusters run): a pod on node A keeps
requesting a responder and reading a sequenced feed, both served from node B, while node A is put
in lame-duck mode (``nats-server --signal ldm``) and then exits. Nothing is lost -- no request fails,
no feed message is missing -- and the pod is on node B before node A closes it.

A second test pins the same property for a PINNED run of publishes (a token stream): a handover in
the middle of the run, onto the other node, still delivers the run in order. It also reports whether a
publish made on the successor overtook the run's tail -- two connections on two nodes are two
publishers on two routes, and that reordering is what the pin prevents within a run.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import IncomingMessage, NatsClient, Subject
from threetears.nats.subjects import get_default_namespace, set_default_namespace

pytestmark = pytest.mark.integration

_IMAGE = "nats:2.14.2-alpine"
_NS = "ldmlive"
#: the shortest lame-duck duration nats-server accepts; the grace before it starts closing clients.
_LAME_DUCK_DURATION = "30s"
_LAME_DUCK_GRACE = "3s"
#: how long the workload runs after the server is told to shut down: the grace, the spread of its
#: closes, and its exit, with margin.
_WORKLOAD_AFTER_LDM_SECONDS = 38.0
_TICK_SECONDS = 0.02


@dataclass
class _Node:
    """one server of the cluster.

    :ivar name: its server name and network alias
    :ivar uri: its client URI from the host
    :ivar container: the running container
    """

    name: str
    uri: str
    container: object


def _config(name: str) -> str:
    """one node's configuration: a two-node cluster, advertising nothing the host cannot reach.

    :param name: the node's name and network alias
    :ptype name: str
    :return: the server configuration
    :rtype: str
    """
    return (
        "port: 4222\n"
        f"server_name: {name}\n"
        f'lame_duck_duration: "{_LAME_DUCK_DURATION}"\n'
        f'lame_duck_grace_period: "{_LAME_DUCK_GRACE}"\n'
        "cluster {\n"
        "  name: ldm\n"
        '  listen: "0.0.0.0:6222"\n'
        "  no_advertise: true\n"
        '  routes: ["nats-route://node-a:6222", "nats-route://node-b:6222"]\n'
        "}\n"
    )


@contextlib.contextmanager
def _cluster(tmp_path: Path) -> Iterator[tuple[_Node, _Node]]:
    """two routed nats-servers on one docker network.

    :param tmp_path: a directory for the configurations
    :ptype tmp_path: Path
    :return: the two nodes, yielded
    :rtype: Iterator[tuple[_Node, _Node]]
    """
    from testcontainers.core.network import Network  # noqa: PLC0415
    from testcontainers.nats import NatsContainer  # noqa: PLC0415

    with Network() as network:
        nodes: list[_Node] = []
        containers = []
        try:
            for name in ("node-a", "node-b"):
                conf_dir = tmp_path / name
                conf_dir.mkdir()
                (conf_dir / "nats.conf").write_text(_config(name))
                container = (
                    NatsContainer(image=_IMAGE, jetstream=False)
                    .with_network(network)
                    .with_network_aliases(name)
                    .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
                    .with_command(["-c", "/etc/nats/nats.conf"])
                )
                container.start()
                containers.append(container)
                nodes.append(_Node(name=name, uri=container.nats_uri(), container=container))
            yield nodes[0], nodes[1]
        finally:
            for container in containers:
                with contextlib.suppress(Exception):
                    container.stop()


def _enter_lame_duck(node: _Node) -> None:
    """put ``node``'s server in lame-duck mode, as a rolling restart does.

    :param node: the node
    :ptype node: _Node
    :return: nothing
    :rtype: None
    """
    container_id = node.container.get_wrapped_container().id  # type: ignore[attr-defined]
    subprocess.run(  # noqa: S603 - fixed argv
        ["docker", "exec", container_id, "nats-server", "--signal", "ldm=1"],  # noqa: S607
        check=True,
        capture_output=True,
    )


async def _connect(uri: str, name: str) -> NatsClient:
    """a client connected to one node only.

    :param uri: the node's URI
    :ptype uri: str
    :param name: the client's name
    :ptype name: str
    :return: the connected client
    :rtype: NatsClient
    """
    return await NatsClient.connect(nats_url=uri, nats_subject_namespace=_NS, client_name=name, verify_jetstream=False)


def _port_of(client: NatsClient) -> int | None:
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


@contextlib.asynccontextmanager
async def _namespace() -> AsyncIterator[None]:
    """bind this test's subject namespace, and restore the previous one after.

    :return: nothing, yielded
    :rtype: AsyncIterator[None]
    """
    previous = get_default_namespace()
    try:
        yield
    finally:
        set_default_namespace(previous)


async def test_a_lame_duck_server_hands_its_clients_over_without_losing_anything(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    with _cluster(tmp_path) as (first, second):
        async with _namespace():
            # a pod that knows both nodes, as a pod given the cluster's URLs does. nats-py shuffles
            # its pool, so the node it lands on is the one shut down, and the other serves.
            pod = await NatsClient.connect(
                nats_url=first.uri,
                cluster_urls=[second.uri],
                nats_subject_namespace=_NS,
                client_name="ldm-pod",
                verify_jetstream=False,
            )
            port_of = {int(node.uri.rsplit(":", 1)[1]): node for node in (first, second)}
            node_a = port_of[_port_of(pod) or 0]
            node_b = second if node_a is first else first
            port_b = int(node_b.uri.rsplit(":", 1)[1])
            service = await _connect(node_b.uri, "ldm-service")

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

            async def _requests() -> None:
                nonlocal answered, moved_at
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
                    if moved_at is None and _port_of(pod) == port_b:
                        moved_at = time.monotonic()
                    await asyncio.sleep(_TICK_SECONDS)

            async def _feed() -> None:
                nonlocal published
                while not stop.is_set():
                    published += 1
                    await service.publish_raw(subject=Subject.raw("feed"), payload=str(published).encode())
                    await asyncio.sleep(_TICK_SECONDS)

            workers = [asyncio.create_task(_requests()), asyncio.create_task(_feed())]
            await asyncio.sleep(1.0)
            ldm_at = time.monotonic()
            _enter_lame_duck(node_a)
            await asyncio.sleep(_WORKLOAD_AFTER_LDM_SECONDS)
            stop.set()
            await asyncio.gather(*workers)
            await service.flush()
            await asyncio.sleep(0.5)

            missing = sorted(set(range(1, published + 1)) - set(feed_seen))
            print(  # noqa: T201 - the counts are the evidence this test exists for
                f"\nlame-duck handover: moved to node B {((moved_at or ldm_at) - ldm_at) * 1000:.0f}ms after "
                f"node A entered lame-duck mode; {answered} requests answered, {len(failures)} failed; "
                f"{published} feed messages published, {len(missing)} missing, "
                f"in order: {feed_seen == sorted(feed_seen)}"
            )
            assert moved_at is not None, "the pod never moved to node B"
            assert moved_at - ldm_at < 3.0, "the pod did not move before node A began closing clients"
            assert failures == []
            assert missing == []
            assert feed_seen == sorted(feed_seen)
            await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await service.shutdown(drain_timeout=timedelta(seconds=2))


async def test_a_pinned_run_arrives_in_order_across_a_handover_onto_another_node(tmp_path: Path) -> None:
    """a stream split over two nodes' routes can reorder; a pinned run stays on its first connection."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    with _cluster(tmp_path) as (node_a, node_b):
        async with _namespace():
            reader = await _connect(node_b.uri, "pin-reader")
            received: list[int] = []

            async def _on_token(msg: IncomingMessage) -> None:
                received.append(int(bytes(msg.data)))

            await reader.subscribe(Subject.raw("hub.stream.tokens"), cb=_on_token)
            await reader.flush()
            writer = await NatsClient.connect(
                nats_url=node_a.uri,
                cluster_urls=[node_b.uri],
                nats_subject_namespace=_NS,
                client_name="pin-writer",
                verify_jetstream=False,
            )
            started_on = _port_of(writer)
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
                        if _port_of(writer) != started_on:
                            break
            assert _port_of(writer) != started_on, "no renewal landed on the other node"
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
