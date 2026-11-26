"""MCP resources over Streamable HTTP: list, read, templates, a permission per resource, subscriptions.

the SDK's own clients over real loopback connections: ``ClientSession`` on the handshake-era protocol (2025-11-25),
where a stateful session subscribes with ``resources/subscribe``, and ``mcp.Client`` on 2026-07-28, where a
``subscriptions/listen`` stream names the resources it wants updates of.
"""

from __future__ import annotations

import base64
import contextlib
import warnings
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import anyio
import httpx2
import mcp
import mcp.types as mcp_types
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import ResourceUpdated
from mcp.shared.exceptions import MCPDeprecationWarning, MCPError
from threetears.mcp import (
    ErrorCode,
    ErrorCodes,
    Identity,
    McpResource,
    McpResourceTemplate,
    McpServer,
    Refusal,
    ResourceContent,
    ResourceRegistry,
    ResourceRequest,
    ToolRegistry,
)
from threetears.mcp.resources import RESOURCE_REFUSED

from .conftest import LoopbackMcp

_DEADLINE_S = 5
_PROTOCOLS = pytest.mark.parametrize("modern", [False, True], ids=["2025-11-25", "2026-07-28"])


class _Identity:
    def __init__(self) -> None:
        self.resolvable = True

    async def identify(self) -> Identity:
        if not self.resolvable:
            raise RuntimeError("no caller")
        return Identity(principal_type="user", principal_id=uuid4())


class _Grants:
    """grants every permission but ``secret.read``; records what it was asked."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def allows(self, identity: Identity, permission: str) -> bool:
        self.asked.append(permission)
        return permission != "secret.read"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


async def _config() -> str:
    return '{"mode": "local"}'


async def _secret() -> str:
    return "hidden"


async def _logo() -> bytes:
    return b"\x89PNG"


async def _status(*, name: str) -> str:
    return f"{name}: idle"


async def _pair() -> list[ResourceContent]:
    return [
        ResourceContent(uri="app://pair/a", text="a"),
        ResourceContent(uri="app://pair/b", blob=b"b", mime_type="application/octet-stream"),
    ]


async def _broken() -> str:
    raise OSError("disk gone")


def _registry() -> ResourceRegistry:
    registry = ResourceRegistry()
    registry.register(
        McpResource(
            uri="app://config",
            name="config",
            required_permission="app.read",
            reader=_config,
            mime_type="application/json",
        )
    )
    registry.register(McpResource(uri="app://secret", name="secret", required_permission="secret.read", reader=_secret))
    registry.register(
        McpResource(uri="app://logo", name="logo", required_permission="app.read", reader=_logo, mime_type="image/png")
    )
    registry.register(McpResource(uri="app://pair", name="pair", required_permission="app.read", reader=_pair))
    registry.register(McpResource(uri="app://broken", name="broken", required_permission="app.read", reader=_broken))
    registry.register(
        McpResourceTemplate(
            uri_template="app://sessions/{name}/status",
            name="session status",
            required_permission="app.read",
            reader=_status,
            mime_type="text/plain",
        )
    )
    registry.register(
        McpResourceTemplate(
            uri_template="app://vault/{key}", name="vault", required_permission="secret.read", reader=_status
        )
    )
    return registry


def _server(identity: _Identity | None = None, *, resources: ResourceRegistry | None = None) -> McpServer:
    return McpServer(
        name="resources",
        identity_provider=identity or _Identity(),
        authorizer=_Grants(),
        registry=ToolRegistry(),
        resources=resources if resources is not None else _registry(),
    )


@contextlib.asynccontextmanager
async def _client(url: str, *, modern: bool, **options: Any) -> AsyncIterator[Any]:
    async with httpx2.AsyncClient() as http:
        if modern:
            async with mcp.Client(streamable_http_client(url, http_client=http), **options) as client:
                assert client.protocol_version == "2026-07-28"
                yield client
        else:
            async with streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write, **options) as session:
                    await session.initialize()
                    yield session


class TestListAndRead:
    @pytest.mark.asyncio
    @_PROTOCOLS
    async def test_the_listing_holds_only_what_the_caller_may_read(
        self, loopback_mcp: LoopbackMcp, modern: bool
    ) -> None:
        async with loopback_mcp(_server(), stateless=True, json_response=False) as url:
            async with _client(url, modern=modern) as client:
                listed = await client.list_resources()
                templates = await client.list_resource_templates()
        assert [resource.uri for resource in listed.resources] == [
            "app://config",
            "app://logo",
            "app://pair",
            "app://broken",
        ]
        assert listed.resources[0].mime_type == "application/json"
        assert [template.uri_template for template in templates.resource_templates] == ["app://sessions/{name}/status"]

    @pytest.mark.asyncio
    @_PROTOCOLS
    async def test_text_bytes_template_and_many_items_are_read(self, loopback_mcp: LoopbackMcp, modern: bool) -> None:
        async with loopback_mcp(_server(), stateless=True, json_response=False) as url:
            async with _client(url, modern=modern) as client:
                config = await client.read_resource("app://config")
                logo = await client.read_resource("app://logo")
                status = await client.read_resource("app://sessions/s1/status")
                pair = await client.read_resource("app://pair")
        [text] = config.contents
        assert isinstance(text, mcp_types.TextResourceContents)
        assert (text.uri, text.text, text.mime_type) == ("app://config", '{"mode": "local"}', "application/json")
        [blob] = logo.contents
        assert isinstance(blob, mcp_types.BlobResourceContents)
        assert (base64.b64decode(blob.blob), blob.mime_type) == (b"\x89PNG", "image/png")
        [line] = status.contents
        assert isinstance(line, mcp_types.TextResourceContents)
        assert (line.text, line.mime_type) == ("s1: idle", "text/plain")
        assert [item.uri for item in pair.contents] == ["app://pair/a", "app://pair/b"]

    @pytest.mark.asyncio
    @_PROTOCOLS
    async def test_refusals_are_json_rpc_errors_carrying_the_error_model(
        self, loopback_mcp: LoopbackMcp, modern: bool
    ) -> None:
        identity = _Identity()
        async with loopback_mcp(_server(identity), stateless=True, json_response=False) as url:
            async with _client(url, modern=modern) as client:
                with pytest.raises(MCPError) as unknown:
                    await client.read_resource("app://nothing")
                with pytest.raises(MCPError) as denied:
                    await client.read_resource("app://secret")
                with pytest.raises(MCPError) as denied_template:
                    await client.read_resource("app://vault/k")
                with pytest.raises(MCPError) as broken:
                    await client.read_resource("app://broken")
                identity.resolvable = False
                with pytest.raises(MCPError) as anonymous:
                    await client.read_resource("app://config")
                with pytest.raises(MCPError) as anonymous_list:
                    await client.list_resources()
        assert (unknown.value.code, unknown.value.data["code"]) == (INVALID_PARAMS, "UNKNOWN_RESOURCE")
        assert (denied.value.code, denied.value.data["code"]) == (RESOURCE_REFUSED, "PERMISSION_DENIED")
        assert denied_template.value.data["code"] == "PERMISSION_DENIED"
        assert (broken.value.code, broken.value.data["code"]) == (INTERNAL_ERROR, "HANDLER_ERROR")
        assert "OSError" in broken.value.data["message"]
        assert (anonymous.value.code, anonymous.value.data["code"]) == (RESOURCE_REFUSED, "IDENTITY_UNAVAILABLE")
        assert anonymous_list.value.data["code"] == "IDENTITY_UNAVAILABLE"
        assert denied.value.data["retryable"] is False

    @pytest.mark.asyncio
    async def test_a_listing_asks_each_permission_once(self, loopback_mcp: LoopbackMcp) -> None:
        grants = _Grants()
        server = McpServer(
            name="resources",
            identity_provider=_Identity(),
            authorizer=grants,
            registry=ToolRegistry(),
            resources=_registry(),
        )
        async with loopback_mcp(server, stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as client:
                await client.list_resources()
        assert sorted(grants.asked) == ["app.read", "secret.read"]

    @pytest.mark.asyncio
    async def test_a_live_provider_is_read_on_every_request(self, loopback_mcp: LoopbackMcp) -> None:
        live: list[McpResource] = []
        server = McpServer(
            name="resources",
            identity_provider=_Identity(),
            authorizer=_Grants(),
            registry=ToolRegistry(),
            resource_provider=lambda: list(live),
        )
        async with loopback_mcp(server, stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as client:
                assert (await client.list_resources()).resources == []
                live.append(McpResource(uri="app://new", name="new", required_permission="app.read", reader=_config))
                assert [resource.uri for resource in (await client.list_resources()).resources] == ["app://new"]

    @pytest.mark.asyncio
    async def test_a_failing_provider_is_an_internal_error(self, loopback_mcp: LoopbackMcp) -> None:
        def _fails() -> list[McpResource]:
            raise RuntimeError("registry down")

        server = McpServer(
            name="resources",
            identity_provider=_Identity(),
            authorizer=_Grants(),
            registry=ToolRegistry(),
            resource_provider=_fails,
        )
        async with loopback_mcp(server, stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as client:
                with pytest.raises(MCPError) as failed:
                    await client.list_resources()
        assert failed.value.code == INTERNAL_ERROR
        assert "registry down" not in failed.value.message


async def _subscribe(session: ClientSession, uri: str) -> None:
    """``resources/subscribe``, the handshake-era call; the SDK warns that 2026-07-28 removed it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MCPDeprecationWarning)
        await session.subscribe_resource(uri)


async def _unsubscribe(session: ClientSession, uri: str) -> None:
    """``resources/unsubscribe``, the handshake-era call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MCPDeprecationWarning)
        await session.unsubscribe_resource(uri)


class _Notes:
    def __init__(self) -> None:
        self.updated: list[str] = []
        self.list_changed = 0
        self.tools_changed = anyio.Event()
        self.arrived = anyio.Event()

    async def handler(self, message: Any) -> None:
        inner = getattr(message, "root", message)
        if isinstance(inner, mcp_types.ResourceUpdatedNotification):
            self.updated.append(str(inner.params.uri))
            self.arrived.set()
        elif isinstance(inner, mcp_types.ResourceListChangedNotification):
            self.list_changed += 1
            self.arrived.set()
        elif isinstance(inner, mcp_types.ToolListChangedNotification):
            self.tools_changed.set()


class TestSubscriptions:
    @pytest.mark.asyncio
    async def test_a_stateful_session_subscribes_and_is_told_of_updates_until_it_unsubscribes(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        server = _server()
        notes = _Notes()
        async with loopback_mcp(server, stateless=False, json_response=False) as url:
            async with _client(url, modern=False, message_handler=notes.handler) as session:
                capabilities = session.server_capabilities
                assert capabilities.resources is not None
                assert capabilities.resources.subscribe is True
                assert capabilities.resources.list_changed is True
                await _subscribe(session, "app://sessions/s1/status")
                await server.notify_resource_updated("app://config")
                await server.notify_resource_updated("app://sessions/s1/status")
                with anyio.fail_after(_DEADLINE_S):
                    await notes.arrived.wait()
                await _unsubscribe(session, "app://sessions/s1/status")
                await server.notify_resource_updated("app://sessions/s1/status")
                # a tools change sent after the update arrives after it: the update was not sent if it is absent
                await server.notify_tools_changed()
                with anyio.fail_after(_DEADLINE_S):
                    await notes.tools_changed.wait()
        assert notes.updated == ["app://sessions/s1/status"]

    @pytest.mark.asyncio
    async def test_a_stateful_session_is_told_the_resource_list_changed(self, loopback_mcp: LoopbackMcp) -> None:
        server = _server()
        notes = _Notes()
        async with loopback_mcp(server, stateless=False, json_response=False) as url:
            async with _client(url, modern=False, message_handler=notes.handler) as session:
                await session.list_resources()
                await server.notify_resources_changed()
                with anyio.fail_after(_DEADLINE_S):
                    await notes.arrived.wait()
        assert notes.list_changed == 1

    @pytest.mark.asyncio
    async def test_a_subscription_is_refused_without_the_permission(self, loopback_mcp: LoopbackMcp) -> None:
        async with loopback_mcp(_server(), stateless=False, json_response=False) as url:
            async with _client(url, modern=False) as session:
                with pytest.raises(MCPError) as denied:
                    await _subscribe(session, "app://secret")
        assert denied.value.data["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_a_stateless_transport_offers_no_subscriptions_and_refuses_one(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        async with loopback_mcp(_server(), stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as session:
                capabilities = session.server_capabilities
                assert capabilities.resources is not None
                assert not capabilities.resources.subscribe
                with pytest.raises(MCPError) as refused:
                    await _subscribe(session, "app://config")
        assert refused.value.code == INVALID_REQUEST

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stateless", [True, False], ids=["stateless", "stateful"])
    async def test_a_2026_listen_stream_naming_a_resource_gets_its_updates(
        self, loopback_mcp: LoopbackMcp, stateless: bool
    ) -> None:
        server = _server()
        events: list[Any] = []
        async with loopback_mcp(server, stateless=stateless, json_response=False) as url:
            async with _client(url, modern=True) as client:
                async with client.listen(resource_subscriptions=["app://config"]) as subscription:
                    await server.notify_resource_updated("app://logo")
                    await server.notify_resource_updated("app://config")
                    with anyio.fail_after(_DEADLINE_S):
                        async for event in subscription:
                            events.append(event)
                            break
        assert events == [ResourceUpdated(uri="app://config")]

    @pytest.mark.asyncio
    async def test_a_2026_listen_stream_naming_a_forbidden_resource_is_refused(self, loopback_mcp: LoopbackMcp) -> None:
        async with loopback_mcp(_server(), stateless=True, json_response=False) as url:
            async with _client(url, modern=True) as client:
                with pytest.raises(MCPError) as denied:
                    async with client.listen(resource_subscriptions=["app://secret"]):
                        pass
        assert denied.value.data["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_a_listen_stream_needs_an_identity(self, loopback_mcp: LoopbackMcp) -> None:
        identity = _Identity()
        identity.resolvable = False
        async with loopback_mcp(_server(identity), stateless=True, json_response=False) as url:
            async with _client(url, modern=True) as client:
                with pytest.raises(MCPError) as anonymous:
                    async with client.listen(tools_list_changed=True):
                        pass
        assert anonymous.value.data["code"] == "IDENTITY_UNAVAILABLE"


class TestAServerWithoutResources:
    @pytest.mark.asyncio
    async def test_it_declares_no_resources_capability_and_a_listen_naming_one_is_refused(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        server = McpServer(
            name="tools-only", identity_provider=_Identity(), authorizer=_Grants(), registry=ToolRegistry()
        )
        async with loopback_mcp(server, stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as session:
                assert session.server_capabilities.resources is None
            async with _client(url, modern=True) as client:
                with pytest.raises(MCPError) as unknown:
                    async with client.listen(resource_subscriptions=["app://config"]):
                        pass
        assert unknown.value.data["code"] == "UNKNOWN_RESOURCE"


class TestTheTypes:
    def test_a_content_item_is_text_or_bytes(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            ResourceContent(uri="app://x")
        with pytest.raises(ValueError, match="exactly one"):
            ResourceContent(uri="app://x", text="t", blob=b"b")

    def test_empty_fields_and_a_bad_template_are_refused(self) -> None:
        with pytest.raises(ValueError, match="uri"):
            McpResource(uri="", name="n", required_permission="p", reader=_config)
        with pytest.raises(ValueError, match="required_permission"):
            McpResource(uri="app://x", name="n", required_permission="", reader=_config)
        with pytest.raises(ValueError):
            McpResourceTemplate(uri_template="app://{a}{b}", name="n", required_permission="p", reader=_status)

    def test_a_registry_refuses_a_uri_twice_and_a_server_takes_one_source(self) -> None:
        registry = _registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(McpResource(uri="app://config", name="again", required_permission="p", reader=_config))
        with pytest.raises(ValueError, match="at most one"):
            McpServer(
                name="x",
                identity_provider=_Identity(),
                authorizer=_Grants(),
                registry=ToolRegistry(),
                resources=registry,
                resource_provider=registry.all,
            )


TAINTED = ErrorCode("tainted", "the session is tainted")


class _SessionGate:
    """grants every permission, and decides on each resource request by the session its URI names."""

    def __init__(self) -> None:
        self.requests: list[ResourceRequest] = []
        self.allows_calls = 0

    async def allows(self, identity: Identity, permission: str) -> bool:
        self.allows_calls += 1
        return True

    async def authorize_resource(self, identity: Identity, request: ResourceRequest) -> bool | Refusal:
        self.requests.append(request)
        if request.uri == "app://secret":
            return False
        if request.variables.get("name") == "bad":
            return Refusal(TAINTED, "session bad is tainted")
        if request.variables.get("name") == "odd":
            return Refusal("never_registered", "refused")
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _gated(gate: _SessionGate) -> McpServer:
    return McpServer(
        name="resources",
        identity_provider=_Identity(),
        authorizer=gate,
        registry=ToolRegistry(),
        resources=_registry(),
        error_codes=ErrorCodes([TAINTED]),
    )


class TestAnAuthorizerThatSeesTheRequest:
    """an authorizer whose class defines ``authorize_resource`` decides on each request's URI and variables."""

    @pytest.mark.asyncio
    async def test_a_read_is_decided_by_the_uris_variables_with_its_own_code(self, loopback_mcp: LoopbackMcp) -> None:
        gate = _SessionGate()
        async with loopback_mcp(_gated(gate), stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as client:
                served = await client.read_resource("app://sessions/s1/status")
                with pytest.raises(MCPError) as tainted:
                    await client.read_resource("app://sessions/bad/status")
                with pytest.raises(MCPError) as unregistered:
                    await client.read_resource("app://sessions/odd/status")
        [line] = served.contents
        assert isinstance(line, mcp_types.TextResourceContents)
        assert line.text == "s1: idle"
        assert (tainted.value.code, tainted.value.data["code"]) == (RESOURCE_REFUSED, "tainted")
        assert unregistered.value.data["code"] == "PERMISSION_DENIED"
        read = [request for request in gate.requests if request.action == "read"]
        assert [(request.uri, dict(request.variables)) for request in read][0] == (
            "app://sessions/s1/status",
            {"name": "s1"},
        )
        assert read[0].required_permission == "app.read"
        assert gate.allows_calls == 0

    @pytest.mark.asyncio
    async def test_a_listing_asks_for_each_fixed_resource_and_leaves_out_what_it_refuses(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        gate = _SessionGate()
        async with loopback_mcp(_gated(gate), stateless=True, json_response=False) as url:
            async with _client(url, modern=False) as client:
                listed = await client.list_resources()
        assert "app://secret" not in [resource.uri for resource in listed.resources]
        assert {request.uri for request in gate.requests if request.action == "list"} == {
            "app://config",
            "app://secret",
            "app://logo",
            "app://pair",
            "app://broken",
        }

    @pytest.mark.asyncio
    async def test_a_subscription_and_a_listen_stream_are_decided_the_same_way(self, loopback_mcp: LoopbackMcp) -> None:
        gate = _SessionGate()
        server = _gated(gate)
        async with loopback_mcp(server, stateless=False, json_response=False) as url:
            async with _client(url, modern=False) as session:
                await _subscribe(session, "app://sessions/s1/status")
                with pytest.raises(MCPError) as refused:
                    await _subscribe(session, "app://sessions/bad/status")
            async with _client(url, modern=True) as client:
                with pytest.raises(MCPError) as listen_refused:
                    async with client.listen(resource_subscriptions=["app://sessions/bad/status"]):
                        pass
        assert refused.value.data["code"] == listen_refused.value.data["code"] == "tainted"
        assert {request.action for request in gate.requests} >= {"subscribe", "listen"}
