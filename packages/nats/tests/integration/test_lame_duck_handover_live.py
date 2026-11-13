"""Integration test: a server in lame-duck mode hands its clients over; it does not drop them -- in Docker.

The scenarios and their assertions are :mod:`lame_duck_scenarios`' (why the handover matters, and what
each proves). This module runs them on two routed nats-server containers (nats 2.14.2, what the clusters
run) on one Docker network, and puts node A in lame-duck mode with ``docker exec ... nats-server --signal
ldm``. It drives Docker itself, so it carries the ``docker`` marker and runs on the laptop and in CI; the
testament pool runs the same scenarios on two child processes (``test_lame_duck_handover_process_live.py``).
"""

from __future__ import annotations

import contextlib
import functools
import subprocess
from collections.abc import AsyncIterator

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.core.testing.services import require_services
from threetears.observe import contained, get_logger

from .lame_duck_scenarios import (
    IMAGE,
    Node,
    hands_its_clients_over,
    lame_duck_settings,
    pinned_run_stays_in_order,
    wait_for_route,
)

pytestmark = [pytest.mark.integration, pytest.mark.docker]

log = get_logger(__name__)


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
        f"{lame_duck_settings()}"
        "cluster {\n"
        "  name: ldm\n"
        '  listen: "0.0.0.0:6222"\n'
        "  no_advertise: true\n"
        '  routes: ["nats-route://node-a:6222", "nats-route://node-b:6222"]\n'
        "}\n"
    )


def _enter_lame_duck(container: object) -> None:
    """put a container's server in lame-duck mode, as a rolling restart does.

    :param container: the running container
    :ptype container: object
    :return: nothing
    :rtype: None
    """
    container_id = container.get_wrapped_container().id  # type: ignore[attr-defined]
    subprocess.run(  # noqa: S603 - fixed argv
        ["docker", "exec", container_id, "nats-server", "--signal", "ldm=1"],  # noqa: S607
        check=True,
        capture_output=True,
    )


@contextlib.asynccontextmanager
async def _cluster() -> AsyncIterator[tuple[Node, Node]]:
    """two routed nats-servers on one docker network, once the route between them is up.

    :return: the two nodes, yielded
    :rtype: AsyncIterator[tuple[Node, Node]]
    """
    if not check_docker_available():
        require_services("Docker not available")
    from testcontainers.core.network import Network  # noqa: PLC0415
    from threetears.core.testing.fixtures import nats_server, nats_server_uri  # noqa: PLC0415

    with Network() as network:
        nodes: list[Node] = []
        containers = []
        try:
            for name in ("node-a", "node-b"):
                container = nats_server(IMAGE, config=_config(name)).with_network(network).with_network_aliases(name)
                container.start()
                containers.append(container)
                nodes.append(
                    Node(
                        name=name,
                        uri=nats_server_uri(container),
                        enter_lame_duck=functools.partial(_enter_lame_duck, container),
                    )
                )
            await wait_for_route(nodes[0], nodes[1])
            yield nodes[0], nodes[1]
        finally:
            for container in containers:
                with contained("stopping a lame-duck test container", log=log):
                    container.stop()


async def test_a_lame_duck_server_hands_its_clients_over_without_losing_anything() -> None:
    await hands_its_clients_over(_cluster)


async def test_a_pinned_run_arrives_in_order_across_a_handover_onto_another_node() -> None:
    """a stream split over two nodes' routes can reorder; a pinned run stays on its first connection."""
    await pinned_run_stays_in_order(_cluster)
