"""Integration test: a server in lame-duck mode hands its clients over; it does not drop them -- as processes.

The scenarios and their assertions are :mod:`lame_duck_scenarios`' (why the handover matters, and what
each proves), the same ones ``test_lame_duck_handover_live.py`` runs on two Docker containers. Here the
two nodes are nats-server child processes (:func:`threetears.core.testing.services.running_nats`, nats
2.14.2): node A listens for routes on a free port, node B routes to it, and node A is put in lame-duck
mode with ``nats-server --signal ldm=<pid>``. It runs where the services are processes, the testament
pool's integration suite; elsewhere the ``process_only`` marker deselects it (``tests/require_services.py``),
and the Docker module runs the scenarios instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import pytest

from threetears.core.testing.services import running_nats

from .lame_duck_scenarios import (
    IMAGE,
    Node,
    hands_its_clients_over,
    lame_duck_settings,
    pinned_run_stays_in_order,
    wait_for_route,
)

pytestmark = [pytest.mark.integration, pytest.mark.process_only]


def _config(name: str, route: str | None) -> str:
    """one node's configuration: a cluster listener on a free loopback port, and a route to node A's.

    :param name: the node's name
    :ptype name: str
    :param route: node A's ``nats-route://`` URL, or None for node A itself
    :ptype route: str | None
    :return: the server configuration
    :rtype: str
    """
    routes = f'  routes: ["{route}"]\n' if route is not None else ""
    return (
        "port: 4222\n"
        f"server_name: {name}\n"
        f"{lame_duck_settings()}"
        "cluster {\n"
        "  name: ldm\n"
        '  listen: "127.0.0.1:-1"\n'
        "  no_advertise: true\n"
        f"{routes}"
        "}\n"
    )


@contextlib.asynccontextmanager
async def _cluster() -> AsyncIterator[tuple[Node, Node]]:
    """two routed nats-server child processes: node A, then node B routing to it, once the route is up.

    :return: the two nodes, yielded
    :rtype: AsyncIterator[tuple[Node, Node]]
    """
    with running_nats(_config("node-a", None), image=IMAGE) as node_a:
        assert node_a.cluster_url is not None, "node A reported no cluster listener"
        listener = urlsplit(node_a.cluster_url)
        route = f"nats-route://{listener.hostname}:{listener.port}"
        with running_nats(_config("node-b", route), image=IMAGE) as node_b:
            nodes = (
                Node(name="node-a", uri=node_a.uri, enter_lame_duck=node_a.enter_lame_duck),
                Node(name="node-b", uri=node_b.uri, enter_lame_duck=node_b.enter_lame_duck),
            )
            await wait_for_route(*nodes)
            yield nodes


async def test_a_lame_duck_server_hands_its_clients_over_without_losing_anything() -> None:
    await hands_its_clients_over(_cluster)


async def test_a_pinned_run_arrives_in_order_across_a_handover_onto_another_node() -> None:
    """a stream split over two nodes' routes can reorder; a pinned run stays on its first connection."""
    await pinned_run_stays_in_order(_cluster)
