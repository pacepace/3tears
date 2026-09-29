"""integration: per-copy definitions and verified registrants, over a real NATS bus.

Real :class:`ToolServer` pods register with a real :class:`RegistrationHandler` and are asked
about by a real :class:`DiscoveryHandler`, every hop crossing a NATS testcontainer. Only the
host's authenticator is a double -- a table of which token verifies as which pod or agent -- so
what is proved is the registry's handling of verified identity, not any host's cryptography.

What each test pins:

* two copies serving everyone, with different descriptions, are both available and discovery
  shows one of them, the same one, on every ask;
* a verified pod that owns no node and is not the platform is refused ``NOT_PLATFORM_SHARED``,
  hears the refusal as :class:`ToolRegistrationRefused`, and the incumbent is untouched;
* a tokenless manifest naming the shared pod's id is refused, and the shared pod stays available
  with its own definition;
* an agent's signed in-process copy is shown to that agent and to nobody else;
* an agent registering under another agent's pod id is refused ``POD_ID_MISMATCH``.

requires docker; marked integration. run with::

    ./scripts/test-integration.sh registry -rs
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest
from threetears.core.testing.replay_guard import FakeReplayGuard

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import (
    RegistrationManifest,
    RegistrationResponse,
    ToolManifestEntry,
    ToolRegistrationRefused,
    ToolServer,
)
from threetears.nats import NatsClient, Subjects, set_default_namespace
from threetears.registry.auth import ToolPodAuth
from threetears.registry.catalog import CopyStatus, ToolCatalog
from threetears.registry.discovery import DiscoverRequest, DiscoverResponse, DiscoverToolEntry, DiscoveryHandler
from threetears.registry.registration import RegistrationHandler

pytestmark = pytest.mark.integration

__all__: list[str] = []

_NS = "itest_copies"
_TOOL = "threetears.calculator"
_VERSION = "1.0"
_FULL = f"{_TOOL}@{_VERSION}"
_SHARED_POD = "builtin-tool-server"
_HUB_POD = "019f6000-0000-7000-8000-00000000b0b1"
_STRAY_POD = "stray-tool-pod"
_AGENT_A = UUID("01948a00-aaaa-7000-8000-00000000000a")
_AGENT_B = UUID("01948a00-aaaa-7000-8000-00000000000b")
_A_POD = Subjects.agent_inprocess_pod_id(_AGENT_A, "inst-1")
_B_POD = Subjects.agent_inprocess_pod_id(_AGENT_B, "inst-1")


class _Directory:
    """the host's authenticator, as a table of which token verifies as whom."""

    def __init__(self) -> None:
        """the platform's cast: the shared pod, a hub-run pod, a stray, and two agents."""
        self._pods = {
            "shared-token": ToolPodAuth(
                pod_entity_id=_SHARED_POD, name="builtin-tool-server", owned_namespaces=[], platform_shared=True
            ),
            "hub-token": ToolPodAuth(
                pod_entity_id=_HUB_POD, name="hub datasource pod", owned_namespaces=[], platform_shared=True
            ),
            "stray-token": ToolPodAuth(pod_entity_id=_STRAY_POD, name="stray", owned_namespaces=[]),
        }
        self._agents = {"agent-a-token": _AGENT_A, "agent-b-token": _AGENT_B}

    async def verify_pod(self, token: str) -> ToolPodAuth | None:
        """the pod the token verifies as.

        :param token: the manifest token
        :ptype token: str
        :return: the pod, or ``None``
        :rtype: ToolPodAuth | None
        """
        return self._pods.get(token)

    async def verify_agent(self, token: str) -> UUID | None:
        """the agent the token verifies as.

        :param token: the manifest token
        :ptype token: str
        :return: the agent, or ``None``
        :rtype: UUID | None
        """
        return self._agents.get(token)

    async def provider_nodes(self) -> tuple[str, ...]:
        """one provider node nobody here serves under.

        :return: the graph
        :rtype: tuple[str, ...]
        """
        return ("tools.pentest",)


class _Calculator(TearsTool):
    """one calculator whose description says which copy it is."""

    def __init__(self, description: str) -> None:
        """name the copy.

        :param description: this copy's description
        :ptype description: str
        :return: nothing
        :rtype: None
        """
        super().__init__()
        self._description = description

    async def execute(self, **kwargs: Any) -> ToolResult:
        """answer with the copy's description.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: the description
        :rtype: ToolResult
        """
        return ToolResult(success=True, content=self._description)

    def mcp_schema(self) -> MCPToolDefinition:
        """this copy's definition.

        :return: the schema
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=_TOOL,
            version=_VERSION,
            description=self._description,
            input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
        )

    def mcp_name(self) -> str:
        """the mcp name.

        :return: the name
        :rtype: str
        """
        return _TOOL

    def mcp_version(self) -> str:
        """the version.

        :return: the version
        :rtype: str
        """
        return _VERSION


class _Registry:
    """a real registration and discovery pair on its own connection."""

    def __init__(self, nc: NatsClient) -> None:
        """build both handlers over one catalog.

        :param nc: the registry's connection
        :ptype nc: NatsClient
        :return: nothing
        :rtype: None
        """
        self.nc = nc
        self.catalog = ToolCatalog()
        self.registration = RegistrationHandler(self.catalog, namespace=_NS, authenticator=_Directory())
        self.discovery = DiscoveryHandler(self.catalog, namespace=_NS)

    async def discover(self, requester: str) -> DiscoverResponse:
        """ask discovery about the calculator, as ``requester``, over the bus.

        :param requester: the agent id or pod id asking
        :ptype requester: str
        :return: the answer
        :rtype: DiscoverResponse
        """
        return await self.nc.request(
            subject=Subjects.tools_discover(),
            message=DiscoverRequest(
                agent_id=requester, tool_manifest=[DiscoverToolEntry(name=_TOOL, version=_VERSION)]
            ),
            response_type=DiscoverResponse,
            timeout=timedelta(seconds=5),
        )


@contextlib.asynccontextmanager
async def _registry(nats_url: str) -> AsyncIterator[_Registry]:
    """a started registry, stopped on exit.

    :param nats_url: the bus
    :ptype nats_url: str
    :return: the registry
    :rtype: AsyncIterator[_Registry]
    """
    async with await NatsClient.connect(nats_url=nats_url, nats_subject_namespace=_NS, client_name="registry") as nc:
        registry = _Registry(nc)
        await registry.registration.start(nc)
        await registry.discovery.start(nc)
        await asyncio.sleep(0.2)
        try:
            yield registry
        finally:
            await registry.discovery.stop()
            await registry.registration.stop()


@contextlib.asynccontextmanager
async def _pod(nats_url: str, pod_id: str, token: str | None, description: str) -> AsyncIterator[ToolServer]:
    """a real ToolServer serving one calculator copy, presenting ``token`` on its manifest.

    :param nats_url: the bus
    :ptype nats_url: str
    :param pod_id: the pod's id
    :ptype pod_id: str
    :param token: the credential its manifest carries, or ``None``
    :ptype token: str | None
    :param description: its copy's description
    :ptype description: str
    :return: the serving server
    :rtype: AsyncIterator[ToolServer]
    """
    async with await NatsClient.connect(nats_url=nats_url, nats_subject_namespace=_NS, client_name=pod_id) as nc:
        server = ToolServer(
            nats_client=nc,
            namespace=_NS,
            pod_id=pod_id,
            auth_token=(lambda: token) if token is not None else None,
            jwks_provider=lambda: {"keys": []},
            assertion_replay_guard=FakeReplayGuard(),
            heartbeat_interval=3600.0,
        )
        server.register(_Calculator(description))
        serving = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                if server.is_ready:
                    break
                await asyncio.sleep(0.05)
            yield server
        finally:
            await server.shutdown()
            serving.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serving


async def _register_raw(nc: NatsClient, pod_id: str, token: str | None, description: str) -> RegistrationResponse:
    """publish one manifest by hand and return the registry's reply.

    :param nc: a connection on the bus
    :ptype nc: NatsClient
    :param pod_id: the pod id the manifest claims
    :ptype pod_id: str
    :param token: the credential it carries, or ``None``
    :ptype token: str | None
    :param description: the calculator description it offers
    :ptype description: str
    :return: the reply
    :rtype: RegistrationResponse
    """
    manifest = RegistrationManifest(
        pod_id=pod_id,
        tools=[
            ToolManifestEntry(
                name=_TOOL,
                version=_VERSION,
                description=description,
                input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
            )
        ],
        bootstrap_token=token,
    )
    return await nc.request(
        subject=Subjects.tools_register(),
        message=manifest,
        response_type=RegistrationResponse,
        timeout=timedelta(seconds=10),
    )


def _descriptions(registry: _Registry, pod_id: str) -> list[str]:
    """every description one pod's copy holds.

    :param registry: the registry
    :ptype registry: _Registry
    :param pod_id: the pod
    :ptype pod_id: str
    :return: the descriptions
    :rtype: list[str]
    """
    entry = registry.catalog.get(_FULL)
    copy = entry.get_endpoint(pod_id) if entry is not None else None
    return [] if copy is None else [d.definition.description for d in copy.definitions.values()]


async def test_two_serve_everyone_copies_are_both_available_and_discovery_is_stable(nats_container: str) -> None:
    """the shared pod and a hub-run pod, different descriptions, one stable answer."""
    set_default_namespace(_NS)
    async with _registry(nats_container) as registry:
        async with _pod(nats_container, _SHARED_POD, "shared-token", "SHARED") as shared:
            assert await shared.wait_until_ready(timeout=10.0) is True
            await asyncio.sleep(0.05)  # a strictly later first announcement for the second copy
            async with _pod(nats_container, _HUB_POD, "hub-token", "HUB") as hub:
                assert await hub.wait_until_ready(timeout=10.0) is True

                entry = registry.catalog.get(_FULL)
                assert entry is not None
                assert entry.copy_status(_SHARED_POD) is CopyStatus.AVAILABLE
                assert entry.copy_status(_HUB_POD) is CopyStatus.AVAILABLE
                assert _descriptions(registry, _SHARED_POD) == ["SHARED"]
                assert _descriptions(registry, _HUB_POD) == ["HUB"]

                answers = [(await registry.discover("unknown")).tools[0] for _ in range(10)]
                assert {a.description for a in answers} == {"HUB"}
                assert {a.endpoint_count for a in answers} == {2}
                assert shared.refused_tools == ()
                assert hub.refused_tools == ()


async def test_a_stray_token_pod_is_refused_and_the_incumbent_is_untouched(nats_container: str) -> None:
    """NOT_PLATFORM_SHARED reaches the stray pod as a refusal; the shared copy is as it was."""
    set_default_namespace(_NS)
    async with _registry(nats_container) as registry:
        async with _pod(nats_container, _SHARED_POD, "shared-token", "SHARED") as shared:
            assert await shared.wait_until_ready(timeout=10.0) is True
            async with _pod(nats_container, _STRAY_POD, "stray-token", "STRAY") as stray:
                with pytest.raises(ToolRegistrationRefused) as excinfo:
                    await stray.wait_until_ready(timeout=10.0)
                assert [(r.name, r.code) for r in excinfo.value.refused] == [(_TOOL, "NOT_PLATFORM_SHARED")]

                assert _descriptions(registry, _STRAY_POD) == []
                assert _descriptions(registry, _SHARED_POD) == ["SHARED"]
                (answer,) = (await registry.discover("unknown")).tools
                assert answer.description == "SHARED"


async def test_a_tokenless_manifest_claiming_the_shared_pods_id_is_refused(nats_container: str) -> None:
    """UNVERIFIED_PUBLISHER; the shared pod stays available with its own definition."""
    set_default_namespace(_NS)
    async with _registry(nats_container) as registry:
        async with _pod(nats_container, _SHARED_POD, "shared-token", "SHARED") as shared:
            assert await shared.wait_until_ready(timeout=10.0) is True
            async with await NatsClient.connect(
                nats_url=nats_container, nats_subject_namespace=_NS, client_name="impostor"
            ) as impostor:
                reply = await _register_raw(impostor, _SHARED_POD, None, "IMPOSTOR")

            assert reply.success is False
            assert reply.error_code == "UNVERIFIED_PUBLISHER"
            entry = registry.catalog.get(_FULL)
            assert entry is not None
            assert entry.copy_status(_SHARED_POD) is CopyStatus.AVAILABLE
            assert _descriptions(registry, _SHARED_POD) == ["SHARED"]
            (answer,) = (await registry.discover("unknown")).tools
            assert answer.description == "SHARED"


async def test_an_agents_signed_copy_is_visible_only_to_that_agent(nats_container: str) -> None:
    """agent A reads its own copy; agent B and an unnamed caller read the shared one."""
    set_default_namespace(_NS)
    async with _registry(nats_container) as registry:
        async with _pod(nats_container, _SHARED_POD, "shared-token", "SHARED") as shared:
            assert await shared.wait_until_ready(timeout=10.0) is True
            async with _pod(nats_container, _A_POD, "agent-a-token", "AGENT-A ONLY") as mine:
                assert await mine.wait_until_ready(timeout=10.0) is True

                seen = {
                    who: (await registry.discover(who)).tools[0].description
                    for who in (str(_AGENT_A), str(_AGENT_B), "unknown")
                }
                assert seen == {str(_AGENT_A): "AGENT-A ONLY", str(_AGENT_B): "SHARED", "unknown": "SHARED"}
                entry = registry.catalog.get(_FULL)
                assert entry is not None
                copy = entry.get_endpoint(_A_POD)
                assert copy is not None and copy.verified_publisher is True


async def test_an_agent_claiming_another_agents_pod_id_is_refused(nats_container: str) -> None:
    """agent A's valid identity under agent B's pod id: POD_ID_MISMATCH, B's copy untouched."""
    set_default_namespace(_NS)
    async with _registry(nats_container) as registry:
        async with _pod(nats_container, _B_POD, "agent-b-token", "B REAL") as theirs:
            assert await theirs.wait_until_ready(timeout=10.0) is True
            async with await NatsClient.connect(
                nats_url=nats_container, nats_subject_namespace=_NS, client_name="agent-a"
            ) as agent_a:
                reply = await _register_raw(agent_a, _B_POD, "agent-a-token", "FROM A")

            assert reply.success is False
            assert reply.error_code == "POD_ID_MISMATCH"
            assert _descriptions(registry, _B_POD) == ["B REAL"]
            (answer,) = (await registry.discover(str(_AGENT_B))).tools
            assert answer.description == "B REAL"
