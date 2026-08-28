"""A tool pod learns, from its registration reply, which namespace it owns.

**The gap this closes.** A tool pod's subject grants are minted at CONNECT, from the
tool-name NODES on its ``tool_pods`` row. The pod never sees that row: it presents a key
and receives a credential, and nothing in that exchange names the node the credential was
scoped to. Every value the pod holds locally is a tool LEAF
(``tools.pentest.sqlmap.1-0-0``), and a leaf cannot be reduced to its node -- ``pentest``
is one component and ``aibots.admin`` is two, so leaves of the same shape split in
different places.

So the answer has to arrive from the process that read the row. It rides the registration
reply, which already crosses that boundary in the right direction.

**Registration stays a PUBLISH by default.** ``publish_registration`` is called on every
heartbeat and on every dynamic register/deregister; turning all of those into round trips
would make a registry that is merely slow into a pod that stalls. ``learn_identity=True``
asks for the reply, and every failure of that ask degrades to a warning: the manifest was
published either way, because a request IS a publish.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import RegistrationManifest, RegistrationResponse, ToolServer
from threetears.nats.errors import RequestError

_POD = "01947100-0000-7000-8000-00000000ab01"


class _StubTool(TearsTool):
    """the smallest concrete tool a manifest can carry."""

    async def execute(self, **kwargs: Any) -> ToolResult:
        """no-op body.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: a trivial result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content="")

    def mcp_schema(self) -> MCPToolDefinition:
        """the stub's schema.

        :return: schema with an empty-object input
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name="pentest.sqlmap",
            version="1.0",
            description="stub",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """the stub's mcp name.

        :return: the name
        :rtype: str
        """
        return "pentest.sqlmap"

    def mcp_version(self) -> str:
        """the stub's version.

        :return: the version
        :rtype: str
        """
        return "1.0"


def _server() -> ToolServer:
    """a server with one tool registered and no live connection.

    :return: the server under test
    :rtype: ToolServer
    """
    server = ToolServer(
        agent_id=uuid7(),
        customer_id=uuid7(),
        nats_url="nats://test:4222",
        pod_id=_POD,
    )
    server.register(_StubTool())
    return server


def _replying_nc(*owned: str) -> AsyncMock:
    """a NATS wrapper whose request answers with a successful registration reply.

    :param owned: the namespaces the reply names
    :ptype owned: str
    :return: the configured mock
    :rtype: AsyncMock
    """
    nc = AsyncMock()
    nc.request = AsyncMock(
        return_value=RegistrationResponse(success=True, pod_id=_POD, owned_namespaces=list(owned)),
    )
    return nc


class TestRegistrationStaysAPublishUnlessAsked:
    """the default path is unchanged, and that is deliberate rather than incidental."""

    async def test_the_default_publishes_and_never_waits(self) -> None:
        """no reply is asked for, so a registry that is down costs nothing.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = AsyncMock()
        server._nc = nc  # noqa: SLF001
        await server.publish_registration()
        assert isinstance(nc.publish.await_args.kwargs["message"], RegistrationManifest)
        nc.request.assert_not_awaited()

    async def test_the_default_leaves_the_pod_with_no_self_identity(self) -> None:
        """``None`` means "not yet learned", which is distinct from "owns nothing".

        an empty tuple is a real answer -- a tokenless pod with no owning agent genuinely
        owns nothing -- so it cannot double as "never asked".

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = AsyncMock()  # noqa: SLF001
        await server.publish_registration()
        assert server.owned_namespaces is None


class TestLearningTheOwnedNamespace:
    """what ``learn_identity=True`` buys, and what it costs when it fails."""

    async def test_the_reply_becomes_the_pods_self_identity(self) -> None:
        """the canonical node names come back and are held on the server.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying_nc("tools.pentest")  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        assert server.owned_namespaces == ("tools.pentest",)

    async def test_the_manifest_is_still_the_one_a_publish_sends(self) -> None:
        """a request IS a publish, so nothing about the manifest changes.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = _replying_nc("tools.pentest")
        server._nc = nc  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        manifest = nc.request.await_args.kwargs["message"]
        assert isinstance(manifest, RegistrationManifest)
        assert manifest.pod_id == _POD
        assert [t.name for t in manifest.tools] == ["pentest.sqlmap"]

    async def test_a_pod_owning_several_nodes_learns_all_of_them(self) -> None:
        """one row may authorize more than one node.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying_nc("tools.pentest", "tools.threetears")  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        assert server.owned_namespaces == ("tools.pentest", "tools.threetears")

    async def test_owning_nothing_is_recorded_as_owning_nothing(self) -> None:
        """an empty reply is an ANSWER, and is not retried as though it were a failure.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying_nc()  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        assert server.owned_namespaces == ()


class TestALearnThatFailsDoesNotBreakRegistration:
    """the manifest was published either way; the identity is what goes missing."""

    async def test_a_transport_failure_leaves_the_pod_unidentified_and_running(self) -> None:
        """a registry that does not answer must not stop a pod registering.

        the manifest left this process before the reply was awaited, so a pod that cannot
        hear the answer is in exactly the state it was in before this existed.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=RequestError("no responders"))
        server._nc = nc  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        assert server.owned_namespaces is None

    async def test_a_refused_registration_leaves_the_pod_unidentified(self) -> None:
        """a reply that says the registration failed carries no identity to keep.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = AsyncMock()
        nc.request = AsyncMock(
            return_value=RegistrationResponse(success=False, pod_id=_POD, error="invalid bootstrap token"),
        )
        server._nc = nc  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        assert server.owned_namespaces is None

    async def test_a_later_learn_replaces_an_earlier_one(self) -> None:
        """the row can change, and the reply is the only thing that reports it.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying_nc("tools.pentest")  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        server._nc = _replying_nc("tools.pentest", "tools.threetears")  # noqa: SLF001
        await server.publish_registration(learn_identity=True)
        assert server.owned_namespaces == ("tools.pentest", "tools.threetears")


@pytest.mark.parametrize("learn", [False, True])
async def test_registration_requires_a_connection_either_way(learn: bool) -> None:
    """the guard is on the connection, not on which path is taken.

    :param learn: whether the identity is asked for
    :ptype learn: bool
    :return: none
    :rtype: None
    """
    server = _server()
    with pytest.raises(RuntimeError, match="publish_registration called before NATS connected"):
        await server.publish_registration(learn_identity=learn)
