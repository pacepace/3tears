"""an agent's in-process tool is routed only to that agent's own process.

Several agents serve the same tool name on their own in-process ToolServers -- a drafts tool, a
conversation-recall tool -- and the catalog merges them into ONE entry keyed on ``name@version``.
Those tools answer from the serving agent's own state (its scope, its per-conversation store), so
a call from agent A that lands on agent B's process is answered wrongly: A's drafts refused or
filtered by B's scope, A's conversation "not found" in B's store. Least-busy routing with a random
tie-break made that the ordinary case rather than a rare one.

The owner of an endpoint is read from its pod-id (``{agent_id}.{instance}``, the only shape an
agent's grant lets it answer the reachability probe under); the caller is the VERIFIED principal.
A Tool Pod's single-token endpoint serves everyone, as before.

Random selection is the reason every routing assertion here loops: one call proves nothing about a
coin flip, sixty calls against three owners do.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.server import CallResponse, RegistrationManifest, ToolManifestEntry

from threetears.core.security import PLATFORM_CUSTOMER_SENTINEL
from threetears.nats import IncomingMessage, RequestError, Subjects, set_default_namespace
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.discovery import DiscoverRequest, DiscoverToolEntry, DiscoveryHandler
from threetears.registry.proxy import CallProxy, ProxyCallResponse
from threetears.registry.registration import RegistrationHandler

from ._dispatch_auth import make_authed_request, make_proxy

_NS = "test"
_TOOL = "aibots.knowledge_drafts"
_VERSION = "1.0"
_FULL_NAME = f"{_TOOL}@{_VERSION}"
_CALLS = 60

_AGENT_A = UUID("01948a00-aaaa-7000-8000-00000000000a")
_AGENT_B = UUID("01948a00-aaaa-7000-8000-00000000000b")
_AGENT_C = UUID("01948a00-aaaa-7000-8000-00000000000c")
_TOOL_POD = "01948a00-dddd-7000-8000-0000000000d1"
_TOOL_POD_PRINCIPAL = UUID("01948a00-dddd-7000-8000-0000000000e1")


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace(_NS)


def _inproc(agent_id: UUID, instance: str) -> str:
    """the pod-id an agent's in-process tool server registers under.

    :param agent_id: the owning agent
    :ptype agent_id: UUID
    :param instance: the replica's instance id
    :ptype instance: str
    :return: the ``{agent_id}.{instance}`` composite
    :rtype: str
    """
    return Subjects.agent_inprocess_pod_id(agent_id, instance)


def _endpoint(pod_id: str, *, status: str = "available", in_flight: int = 0) -> ToolEndpoint:
    """one endpoint of the tool under test.

    :param pod_id: the serving pod's id
    :ptype pod_id: str
    :param status: endpoint lifecycle status
    :ptype status: str
    :param in_flight: in-flight count the routing strategy reads
    :ptype in_flight: int
    :return: the endpoint
    :rtype: ToolEndpoint
    """
    return ToolEndpoint(pod_id=pod_id, status=status, in_flight=in_flight)


async def _catalog(*endpoints: ToolEndpoint, tool_name: str = _TOOL) -> ToolCatalog:
    """a catalog holding one tool served by ``endpoints``.

    :param endpoints: the tool's endpoints
    :ptype endpoints: ToolEndpoint
    :param tool_name: the tool's name
    :ptype tool_name: str
    :return: the catalog
    :rtype: ToolCatalog
    """
    catalog = ToolCatalog()
    await _add(catalog, tool_name, *endpoints)
    return catalog


async def _add(catalog: ToolCatalog, tool_name: str, *endpoints: ToolEndpoint) -> None:
    """register one more tool, served by ``endpoints``, into ``catalog``.

    :param catalog: the catalog to add to
    :ptype catalog: ToolCatalog
    :param tool_name: the tool's name
    :ptype tool_name: str
    :param endpoints: the tool's endpoints
    :ptype endpoints: ToolEndpoint
    """
    await catalog.register(
        CatalogEntry(
            tool_name=tool_name,
            tool_version=_VERSION,
            full_name=f"{tool_name}@{_VERSION}",
            description=f"{tool_name} tool",
            input_schema={"type": "object", "properties": {}},
            endpoints=list(endpoints),
        )
    )


def _ok_reply() -> bytes:
    """the bytes a pod answers a successful call with.

    :return: serialized response
    :rtype: bytes
    """
    return ProxyCallResponse(success=True, content="ok", context=CallContext()).model_dump_json().encode("utf-8")


async def _proxy(catalog: ToolCatalog, *, replies: list[object] | None = None) -> tuple[CallProxy, AsyncMock]:
    """a started proxy over ``catalog`` whose forwards are recorded by a mock transport.

    :param catalog: the catalog the proxy routes against
    :ptype catalog: ToolCatalog
    :param replies: side effects for successive forwards; every forward succeeds when omitted
    :ptype replies: list[object] | None
    :return: the proxy and its transport
    :rtype: tuple[CallProxy, AsyncMock]
    """
    proxy = make_proxy(catalog, namespace=_NS, timeout=5.0)
    nc = AsyncMock()
    nc.request_raw = AsyncMock(side_effect=replies) if replies is not None else AsyncMock(return_value=_ok_reply())
    await proxy.start(nc)
    return proxy, nc


async def _call(
    proxy: CallProxy, nc: AsyncMock, caller: UUID, *, customer_claim: str | None = None
) -> ProxyCallResponse:
    """drive one authenticated call from ``caller`` through the proxy and return its answer.

    :param proxy: the proxy under test
    :ptype proxy: CallProxy
    :param nc: the proxy's transport
    :ptype nc: AsyncMock
    :param caller: the verified principal making the call
    :ptype caller: UUID
    :param customer_claim: the token's customer claim verbatim; the platform sentinel makes the
        caller a tool pod
    :ptype customer_claim: str | None
    :return: the answer the proxy published to the caller
    :rtype: ProxyCallResponse
    """
    before = nc.publish_reply.await_count
    request = make_authed_request(
        agent_id=caller, tool_name=_TOOL, tool_version=_VERSION, customer_claim=customer_claim
    )
    await proxy.handle_call(
        IncomingMessage(data=request.model_dump_json().encode("utf-8"), reply_subject="_INBOX.c", subject="t")
    )
    for _ in range(100):
        if nc.publish_reply.await_count > before:
            break
        await asyncio.sleep(0)
    answer: ProxyCallResponse = nc.publish_reply.await_args.kwargs["message"]
    return answer


def _forwarded(nc: AsyncMock) -> list[str]:
    """the pod-id of every forward the proxy made, in order.

    :param nc: the proxy's transport
    :ptype nc: AsyncMock
    :return: the target pod-id of each forward
    :rtype: list[str]
    """
    prefix = f"{_NS}.tools.internal."
    return [call.kwargs["subject"].path.removeprefix(prefix) for call in nc.request_raw.await_args_list]


class TestEachAgentReachesOnlyItsOwnProcess:
    @pytest.mark.asyncio
    async def test_three_agents_serving_one_tool_each_only_ever_reach_their_own(self) -> None:
        own = {agent: _inproc(agent, "inst-1") for agent in (_AGENT_A, _AGENT_B, _AGENT_C)}
        catalog = await _catalog(*(_endpoint(pod_id) for pod_id in own.values()))

        for caller, own_pod in own.items():
            proxy, nc = await _proxy(catalog)
            for _ in range(_CALLS):
                assert (await _call(proxy, nc, caller)).success is True

            assert set(_forwarded(nc)) == {own_pod}
            assert len(_forwarded(nc)) == _CALLS

    @pytest.mark.asyncio
    async def test_replicas_of_one_agent_still_share_that_agents_calls(self) -> None:
        a1, a2 = _inproc(_AGENT_A, "inst-1"), _inproc(_AGENT_A, "inst-2")
        catalog = await _catalog(_endpoint(a1), _endpoint(a2), _endpoint(_inproc(_AGENT_B, "inst-1")))
        proxy, nc = await _proxy(catalog)

        for _ in range(_CALLS):
            await _call(proxy, nc, _AGENT_A)

        # both of A's replicas carry load, and B's process is never touched
        assert set(_forwarded(nc)) == {a1, a2}

    @pytest.mark.asyncio
    async def test_the_owners_least_busy_replica_is_preferred(self) -> None:
        """least-busy still decides among the owner's replicas; a peer's idle process does not compete."""
        a_busy, a_idle = _inproc(_AGENT_A, "inst-1"), _inproc(_AGENT_A, "inst-2")
        catalog = await _catalog(
            _endpoint(a_busy, in_flight=4),
            _endpoint(a_idle, in_flight=2),
            _endpoint(_inproc(_AGENT_B, "inst-1"), in_flight=0),
        )
        proxy, nc = await _proxy(catalog)

        for _ in range(_CALLS):
            await _call(proxy, nc, _AGENT_A)

        assert set(_forwarded(nc)) == {a_idle}


class TestToolPodToolsServeEveryone:
    @pytest.mark.asyncio
    async def test_a_tool_pod_endpoint_reaches_every_agent_and_every_tool_pod(self) -> None:
        catalog = await _catalog(_endpoint(_TOOL_POD))

        for caller, claim in (
            (_AGENT_A, None),
            (_AGENT_B, None),
            (_TOOL_POD_PRINCIPAL, PLATFORM_CUSTOMER_SENTINEL),
        ):
            proxy, nc = await _proxy(catalog)
            answer = await _call(proxy, nc, caller, customer_claim=claim)

            assert answer.success is True, answer.error
            assert _forwarded(nc) == [_TOOL_POD]

    @pytest.mark.asyncio
    async def test_beside_an_in_process_endpoint_a_peer_gets_the_tool_pod_and_the_owner_gets_both(self) -> None:
        a1 = _inproc(_AGENT_A, "inst-1")
        catalog = await _catalog(_endpoint(_TOOL_POD), _endpoint(a1))

        proxy_b, nc_b = await _proxy(catalog)
        for _ in range(_CALLS):
            await _call(proxy_b, nc_b, _AGENT_B)
        assert set(_forwarded(nc_b)) == {_TOOL_POD}

        proxy_a, nc_a = await _proxy(catalog)
        for _ in range(_CALLS):
            await _call(proxy_a, nc_a, _AGENT_A)
        assert set(_forwarded(nc_a)) == {_TOOL_POD, a1}


class TestFailoverStaysWithTheOwner:
    @pytest.mark.asyncio
    async def test_a_dead_replica_fails_over_to_the_owners_other_replica_not_a_peer(self) -> None:
        """the peer is the least-busy survivor, so only the owner rule keeps the retry at home."""
        a_dead, a_live = _inproc(_AGENT_A, "inst-1"), _inproc(_AGENT_A, "inst-2")
        catalog = await _catalog(
            _endpoint(a_dead, in_flight=0),
            _endpoint(a_live, in_flight=5),
            _endpoint(_inproc(_AGENT_B, "inst-1"), in_flight=0),
        )
        proxy, nc = await _proxy(catalog, replies=[RequestError("no responders available for request"), _ok_reply()])

        answer = await _call(proxy, nc, _AGENT_A)

        assert answer.success is True, answer.error
        assert _forwarded(nc) == [a_dead, a_live]

    @pytest.mark.asyncio
    async def test_when_the_owner_has_no_replica_left_the_call_fails_rather_than_crossing(self) -> None:
        a_dead = _inproc(_AGENT_A, "inst-1")
        catalog = await _catalog(_endpoint(a_dead), _endpoint(_inproc(_AGENT_B, "inst-1")))
        proxy, nc = await _proxy(catalog, replies=[RequestError("no responders available for request"), _ok_reply()])

        answer = await _call(proxy, nc, _AGENT_A)

        assert answer.success is False
        assert answer.error_code == "TOOL_UNAVAILABLE"
        assert _forwarded(nc) == [a_dead]


class TestThePodsOwnRefusalReachesTheCaller:
    @pytest.mark.asyncio
    async def test_the_pod_refusal_code_passes_through_and_is_not_failed_over(self) -> None:
        """what a caller sees if a call ever reaches another agent's process: the pod's own refusal.

        not a transport failure, so it is not retried against a sibling -- the pod answered, and
        its answer is that this routing was wrong.
        """
        catalog = await _catalog(_endpoint(_inproc(_AGENT_A, "inst-1")), _endpoint(_inproc(_AGENT_A, "inst-2")))
        refusal = CallResponse(success=False, content="", error="not yours", error_code="TOOL_CALLER_NOT_OWNER")
        proxy, nc = await _proxy(catalog, replies=[refusal.model_dump_json().encode("utf-8"), _ok_reply()])

        answer = await _call(proxy, nc, _AGENT_A)

        assert answer.success is False
        assert answer.error_code == "TOOL_CALLER_NOT_OWNER"
        assert len(_forwarded(nc)) == 1


class TestACallerWithNoEndpointOfItsOwn:
    @pytest.mark.asyncio
    async def test_an_agent_that_serves_no_endpoint_of_an_in_process_tool_is_refused_without_a_forward(self) -> None:
        catalog = await _catalog(_endpoint(_inproc(_AGENT_B, "inst-1")), _endpoint(_inproc(_AGENT_C, "inst-1")))
        proxy, nc = await _proxy(catalog)

        answer = await _call(proxy, nc, _AGENT_A)

        assert answer.success is False
        assert answer.error_code == "TOOL_UNAVAILABLE"
        assert answer.error is not None
        assert "in-process" in answer.error
        assert "calling agent" in answer.error
        assert _forwarded(nc) == []

    @pytest.mark.asyncio
    async def test_a_tool_pod_cannot_borrow_an_agents_in_process_tool(self) -> None:
        catalog = await _catalog(_endpoint(_inproc(_AGENT_B, "inst-1")))
        proxy, nc = await _proxy(catalog)

        answer = await _call(proxy, nc, _TOOL_POD_PRINCIPAL, customer_claim=PLATFORM_CUSTOMER_SENTINEL)

        assert answer.error_code == "TOOL_UNAVAILABLE"
        assert _forwarded(nc) == []

    @pytest.mark.asyncio
    async def test_the_callers_own_pending_endpoint_answers_not_ready_rather_than_borrowing_a_peer(self) -> None:
        catalog = await _catalog(
            _endpoint(_inproc(_AGENT_A, "inst-1"), status="pending"),
            _endpoint(_inproc(_AGENT_B, "inst-1")),
        )
        proxy, nc = await _proxy(catalog)

        answer = await _call(proxy, nc, _AGENT_A)

        assert answer.error_code == "TOOL_NOT_READY"
        assert _forwarded(nc) == []

    @pytest.mark.asyncio
    async def test_a_peers_pending_endpoint_does_not_tell_the_caller_to_wait(self) -> None:
        """a retry would never help: the endpoint that is coming up is not one this caller may use."""
        catalog = await _catalog(_endpoint(_inproc(_AGENT_B, "inst-1"), status="pending"))
        proxy, nc = await _proxy(catalog)

        answer = await _call(proxy, nc, _AGENT_A)

        assert answer.error_code == "TOOL_UNAVAILABLE"
        assert _forwarded(nc) == []

    @pytest.mark.asyncio
    async def test_an_endpoint_whose_id_names_no_agent_is_routable_by_no_one(self) -> None:
        """a malformed dotted id loaded from shared state is not read as a tool pod."""
        catalog = await _catalog(_endpoint("agent-A.inst-1"))

        for caller in (_AGENT_A, _AGENT_B):
            proxy, nc = await _proxy(catalog)
            answer = await _call(proxy, nc, caller)

            assert answer.error_code == "TOOL_UNAVAILABLE"
            assert _forwarded(nc) == []


class TestRegistrationRefusesAnIdNamingNoAgent:
    @staticmethod
    def _manifest(pod_id: str) -> bytes:
        return (
            RegistrationManifest(
                pod_id=pod_id,
                tools=[ToolManifestEntry(name=_TOOL, version=_VERSION, description="d", input_schema={})],
            )
            .model_dump_json()
            .encode("utf-8")
        )

    @pytest.mark.asyncio
    async def test_a_dotted_id_that_names_no_agent_is_refused_and_catalogs_nothing(self) -> None:
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace=_NS)
        nc = AsyncMock()
        await handler.start(nc)

        await handler.handle_registration(
            IncomingMessage(data=self._manifest("agent-A.inst-1"), reply_subject="_INBOX.r", subject="t")
        )

        reply = nc.publish_reply.await_args.kwargs["message"]
        assert reply.success is False
        assert "agent-A.inst-1" in reply.error
        assert catalog.get(_FULL_NAME) is None

    @pytest.mark.asyncio
    async def test_an_agents_composite_id_is_admitted(self) -> None:
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace=_NS)
        nc = AsyncMock()
        await handler.start(nc)
        pod_id = _inproc(_AGENT_A, "inst-1")

        await handler.handle_registration(
            IncomingMessage(data=self._manifest(pod_id), reply_subject="_INBOX.r", subject="t")
        )

        entry = catalog.get(_FULL_NAME)
        assert entry is not None
        assert [ep.pod_id for ep in entry.endpoints] == [pod_id]


class TestDiscoveryOffersOnlyWhatTheRequesterCanCall:
    _SHARED = "aibots.knowledge_drafts"  # served in-process by A and B
    _B_ONLY = "threetears.context_recall"  # served in-process by B alone
    _POD_TOOL = "threetears.calculator"  # served by a tool pod

    async def _catalog(self, *, a_status: str = "available") -> ToolCatalog:
        catalog = ToolCatalog()
        await _add(
            catalog,
            self._SHARED,
            _endpoint(_inproc(_AGENT_A, "inst-1"), status=a_status),
            _endpoint(_inproc(_AGENT_B, "inst-1")),
        )
        await _add(catalog, self._B_ONLY, _endpoint(_inproc(_AGENT_B, "inst-1")))
        await _add(catalog, self._POD_TOOL, _endpoint(_TOOL_POD))
        return catalog

    @staticmethod
    async def _discover(
        catalog: ToolCatalog, requester: str, names: list[str] | None = None
    ) -> dict[str, dict[str, object]]:
        handler = DiscoveryHandler(catalog, namespace=_NS)
        nc = AsyncMock()
        await handler.start(nc)
        request = DiscoverRequest(
            agent_id=requester,
            tool_manifest=[DiscoverToolEntry(name=name, version=_VERSION) for name in (names or [])],
        )
        await handler.handle_discover(
            IncomingMessage(data=request.model_dump_json().encode("utf-8"), reply_subject="_INBOX.d", subject="t")
        )
        payload = json.loads(nc.publish_reply.await_args.kwargs["message"].model_dump_json())
        return {tool["name"]: tool for tool in payload["tools"]}

    @pytest.mark.asyncio
    async def test_the_full_listing_leaves_out_a_peers_in_process_tool_and_counts_only_own_endpoints(self) -> None:
        tools = await self._discover(await self._catalog(), str(_AGENT_A))

        assert set(tools) == {self._SHARED, self._POD_TOOL}
        assert tools[self._SHARED]["endpoint_count"] == 1
        assert tools[self._POD_TOOL]["endpoint_count"] == 1

    @pytest.mark.asyncio
    async def test_a_pinned_peer_only_tool_resolves_unavailable(self) -> None:
        tools = await self._discover(await self._catalog(), str(_AGENT_A), [self._B_ONLY, self._SHARED])

        assert tools[self._B_ONLY]["status"] == "unavailable"
        assert tools[self._SHARED]["status"] == "available"
        assert tools[self._SHARED]["endpoint_count"] == 1

    @pytest.mark.asyncio
    async def test_the_owner_sees_its_own_in_process_tool(self) -> None:
        tools = await self._discover(await self._catalog(), str(_AGENT_B))

        assert set(tools) == {self._SHARED, self._B_ONLY, self._POD_TOOL}

    @pytest.mark.asyncio
    async def test_an_in_process_server_polling_under_its_pod_id_sees_its_own_tools(self) -> None:
        """``ToolServer.wait_until_ready`` names itself by pod-id, which for an agent is the composite."""
        tools = await self._discover(await self._catalog(), _inproc(_AGENT_B, "inst-9"), [self._B_ONLY])

        assert tools[self._B_ONLY]["status"] == "available"

    @pytest.mark.asyncio
    async def test_the_callers_own_pending_endpoint_is_not_covered_by_a_peers_available_one(self) -> None:
        tools = await self._discover(await self._catalog(a_status="pending"), str(_AGENT_A), [self._SHARED])

        assert tools[self._SHARED]["status"] == "unavailable"

    @pytest.mark.parametrize("requester", ["agent-001", "unknown", "agent-A.inst-1"])
    @pytest.mark.asyncio
    async def test_a_requester_that_names_no_agent_sees_only_tool_pod_tools(self, requester: str) -> None:
        tools = await self._discover(await self._catalog(), requester)

        assert set(tools) == {self._POD_TOOL}
