"""A registering pod is told which namespace it OWNS, on the reply it already gets.

**What was missing, and why a pod could not work it out for itself.** A tool pod's
subject grants are minted at CONNECT, from the tool-name NODES on its ``tool_pods`` row.
The pod itself never sees that row: it presents a key, the broker answers with a
credential, and nothing in the exchange names the node the credential was scoped to. Every
value the pod does hold is a tool LEAF (``tools.pentest.sqlmap.1-0-0``), and a leaf cannot
be reduced to its node -- ``aibots.admin`` is two components and ``pentest`` is one, so two
leaves of the same shape split in different places.

So the pod was deriving its human-in-the-loop family from a leaf while its grant was minted
from a node. Different strings, a digest in between, and an ungranted SUBSCRIBE that is
created client-side and receives nothing forever.

The registration reply is where this is answered because it is the one exchange that
already crosses the boundary in the right direction, carrying a manifest the pod authored
to a process holding the row. The reply's ``owned_namespaces`` is the CANONICAL namespace
name of each node -- ``tools.pentest`` -- which is the same value hub migration v089
materialized as the provider node's row.

**It is derived here, never asserted by the pod.** It comes from the ``ToolPodAuth`` the
authenticator returned for the token the pod presented, so a pod cannot name a namespace it
does not own by putting one on its manifest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from threetears.agent.tools.server import RegistrationManifest, RegistrationResponse, ToolManifestEntry
from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.auth import ToolPodAuth
from threetears.registry.catalog import ToolCatalog
from threetears.registry.registration import RegistrationHandler

_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000a7")


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """Bind a deterministic subject namespace for the probe subjects."""
    set_default_namespace("test")


def _nc() -> AsyncMock:
    """a NATS wrapper that answers every reachability probe and records the reply.

    :return: the configured mock
    :rtype: AsyncMock
    """

    async def _probe(*, subject: Any, message: Any, response_type: Any, timeout: Any) -> Any:
        del message, timeout
        return response_type(pod_id=subject.path.rsplit(".", 1)[-1], ready=True)

    nc = AsyncMock()
    nc.request = AsyncMock(side_effect=_probe)
    return nc


def _manifest(
    pod_id: str = "pod-001",
    *,
    token: str | None = None,
    owner: UUID | None = None,
    tool: str = "pentest.sqlmap",
) -> bytes:
    """one manifest offering a single tool, by default under the ``pentest`` node.

    :param pod_id: the registering pod's id
    :ptype pod_id: str
    :param token: the pod's self-minted identity token, or ``None`` for a tokenless pod
    :ptype token: str | None
    :param owner: the owning agent for an agent-spun pod
    :ptype owner: UUID | None
    :param tool: the mcp name offered, so an agent-owned pod can offer one that
        sits under no provider node anybody owns
    :ptype tool: str
    :return: the serialized manifest
    :rtype: bytes
    """
    entry = ToolManifestEntry(
        name=tool,
        version="1.0.0",
        description="a tool",
        input_schema={"type": "object", "properties": {}},
    )
    manifest = RegistrationManifest(pod_id=pod_id, tools=[entry], bootstrap_token=token, owner_agent_id=owner)
    return manifest.model_dump_json().encode("utf-8")


async def _register(handler: RegistrationHandler, nc: AsyncMock, data: bytes) -> RegistrationResponse:
    """drive one registration and return the reply the handler published.

    :param handler: the handler under test, already started against ``nc``
    :ptype handler: RegistrationHandler
    :param nc: the mock NATS wrapper the handler replies through
    :ptype nc: AsyncMock
    :param data: the serialized manifest
    :ptype data: bytes
    :return: the published reply
    :rtype: RegistrationResponse
    """
    await handler.handle_registration(
        IncomingMessage(data=data, reply_subject="reply.to", subject="test.tools.register")
    )
    reply = nc.publish_reply.await_args.kwargs["message"]
    assert isinstance(reply, RegistrationResponse)
    return reply


def _authenticator(*nodes: str) -> AsyncMock:
    """an authenticator that verifies any token and reports ``nodes`` as OWNED.

    ``provider_nodes`` answers with the same nodes, rooted: a pod's nodes are rows
    in the very graph the directory reports, so a double that let the two disagree
    would be testing a state that cannot exist.

    :param nodes: the pod's ownership entries, as a host might hold them
    :ptype nodes: str
    :return: the configured mock
    :rtype: AsyncMock
    """
    from threetears.core.namespaces import build_tool_provider_node_name

    rooted: list[str] = []
    for node in nodes:
        try:
            rooted.append(build_tool_provider_node_name(node))
        except ValueError:
            continue
    auth = AsyncMock()
    auth.verify_pod = AsyncMock(
        return_value=ToolPodAuth(pod_entity_id="pod-001", name="a pod", owned_namespaces=list(nodes)),
    )
    auth.provider_nodes = AsyncMock(return_value=tuple(rooted))
    return auth


class TestAVerifiedPodLearnsTheNodeItOwns:
    """the platform tool pod: a row, a token, and a node the grant was minted from."""

    async def test_the_reply_names_the_canonical_provider_namespace(self) -> None:
        """``pentest`` on the row comes back as ``tools.pentest`` on the reply.

        the canonical form, because that is the namespace ROW the platform materialized and
        the value an ownership comparison is made against. the pod can hand it straight to
        the subject builders, which root either spelling onto one family.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("pentest"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(token="a-token"))
        assert reply.success is True
        assert reply.owned_namespaces == ["tools.pentest"]

    async def test_a_multi_component_node_comes_back_whole(self) -> None:
        """``aibots.admin`` is ONE node; its dots are boundaries, not a split point.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("aibots.admin"))
        await handler.start(nc)
        entry = ToolManifestEntry(
            name="aibots.admin.list_agents",
            version="1.0.0",
            description="a tool",
            input_schema={"type": "object", "properties": {}},
        )
        data = (
            RegistrationManifest(pod_id="pod-001", tools=[entry], bootstrap_token="a-token")
            .model_dump_json()
            .encode("utf-8")
        )
        reply = await _register(handler, nc, data)
        assert reply.owned_namespaces == ["tools.aibots.admin"]

    async def test_every_authorized_node_is_named(self) -> None:
        """a pod may be authorized at more than one node, and owns all of them.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("pentest", "threetears"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(token="a-token"))
        assert reply.owned_namespaces == ["tools.pentest", "tools.threetears"]

    async def test_a_node_that_cannot_be_rooted_is_dropped_rather_than_raising(self) -> None:
        """a malformed row value must not turn a registration into a failure.

        the ownership record is operator-written and hub-side validation is not this
        process's to rely on. A value that cannot compose a node name (empty, or the bare
        ``tools`` prefix, which names the whole tree) is left out of the reply -- the pod
        then holds no self-identity for it, which is what it already had. Raising instead
        would refuse a pod whose OTHER nodes are perfectly good.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("", "tools", "pentest"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(token="a-token"))
        assert reply.success is True
        assert reply.owned_namespaces == ["tools.pentest"]

    async def test_the_pod_cannot_name_its_own_namespace(self) -> None:
        """it is derived from the ROW, so a manifest cannot assert ownership.

        the manifest here carries an ``owner_agent_id`` the pod chose; the reply still
        names only what the authenticator returned.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("pentest"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(token="a-token", owner=_AGENT))
        assert reply.owned_namespaces == ["tools.pentest"]


class TestAnAgentOwnedPodLearnsItsAgentNamespace:
    """the in-process pod: no ``tool_pods`` row, no token, and an agent it belongs to."""

    async def test_a_tokenless_pod_is_told_its_owning_agents_namespace(self) -> None:
        """its identity was settled at the NATS layer, and its namespace IS its agent's.

        a tokenless manifest is the agent-owned in-process tool server, admitted here
        because the auth callout already authenticated it per-key as an AGENT. It is not a
        row in ``tool_pods``, so it owns no provider node -- what it owns is
        ``agents.<uuid>``, which is exactly the ``owner_namespace`` its own tool-namespace
        rows are stamped with.

        The tool it offers sits under no provider node anybody owns, which is the
        ordinary case for an agent's own tools -- and it is offered explicitly here
        because owning no provider node is now a REAL constraint on what such a pod
        may register, not merely a gap in what it is told.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("pentest"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(owner=_AGENT, tool="myagent.summarize"))
        assert reply.success is True
        assert reply.owned_namespaces == [f"agents.{_AGENT}"]

    async def test_a_tokenless_pod_may_not_take_a_name_inside_a_node_it_does_not_own(self) -> None:
        """the path that used to return before any filter ran, now refused.

        Paired with the admission above: the same pod, the same handler, one name
        under nobody's node and one under ``pentest``'s.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("pentest"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(owner=_AGENT, tool="pentest.sqlmap"))
        assert reply.success is False
        assert reply.owned_namespaces == []

    async def test_a_tokenless_pod_with_no_owner_is_told_nothing(self) -> None:
        """no row and no agent is no self-identity, said as an empty list rather than a guess.

        :return: none
        :rtype: None
        """
        nc = _nc()
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=_authenticator("pentest"))
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(tool="myagent.summarize"))
        assert reply.success is True
        assert reply.owned_namespaces == []


class TestARefusedRegistrationNamesNothing:
    """a reply that failed must not hand out an identity as a consolation."""

    async def test_a_rejected_token_yields_no_owned_namespace(self) -> None:
        """the refusal is the whole answer.

        :return: none
        :rtype: None
        """
        nc = _nc()
        auth = AsyncMock()
        auth.verify_pod = AsyncMock(return_value=None)
        handler = RegistrationHandler(catalog=ToolCatalog(), authenticator=auth)
        await handler.start(nc)
        reply = await _register(handler, nc, _manifest(token="a-bad-token"))
        assert reply.success is False
        assert reply.owned_namespaces == []
