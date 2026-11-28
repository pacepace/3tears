"""a tool's progress and log entries cross the mesh: caller -> the real registry proxy -> a real tool server, and back.

Three processes in one: a :class:`ToolCallClient` (the caller), the real :class:`CallProxy` with its proxy-assertion
signer, and the real :class:`ToolServer` serving a ``TearsTool`` whose ``execute`` declares the MCP call context. The
hops are an in-memory bus that delivers bytes by exact subject and nothing else, so every update travels the
subjects the code names: the pod's ``tools.progress.{pod}.{call}``, relayed by the registry to the caller's
``tools.updates.{caller}.{call}``. A last test puts a real MCP client in front of the caller, whose MCP face hands the
call its own context, and reads ``notifications/progress`` off the client.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4, uuid7

import httpx2
import mcp.types as mcp_types
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, SecretStr
from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer
from threetears.core.security import (
    PLATFORM_CUSTOMER_SENTINEL,
    IdentityClaims,
    ProxyAssertionSigner,
    access_token_hash,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    jwk_thumbprint,
    make_pop_proof,
    sign_identity_token,
)
from threetears.core.testing.replay_guard import FakeReplayGuard
from threetears.core.tool_wire import CallAccepted, CallRequest, CallUpdate, CallUpdates, UpdateLogLevel
from threetears.mcp import CallContext, Identity, McpServer, McpTool, ToolRegistry
from threetears.nats import RELAY_UPDATE_BURST, IncomingMessage, Subject, Subjects, set_default_namespace

from threetears.registry.auth import AllowAllAuthorizer, AllowAllLimitGuard
from threetears.registry.catalog import ToolCatalog, ToolEndpoint
from threetears.registry.client import ToolCallClient
from threetears.registry.proxy import CallProxy, ProxyCallAccepted, ProxyCallResponse

from .copy_entries import uniform_entry
from .dispatch_auth import DEFAULT_AGENT_ID, make_authed_request, make_proxy

_NS = "test"
_SERVING_POD = "serving-pod"
_TOOL = "probe.build"
_VERSION = "1.0"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


class _Bus:
    """core publish/subscribe by exact subject, delivered inline: what the hops share."""

    def __init__(self) -> None:
        self.subscriptions: dict[int, tuple[str, Callable[[IncomingMessage], Awaitable[None]]]] = {}
        self.published: list[tuple[str, bytes]] = []
        self.ever_subscribed: list[str] = []

    async def subscribe(self, subject: Subject | str, cb: Callable[[IncomingMessage], Awaitable[None]]) -> int:
        path = subject.path if isinstance(subject, Subject) else subject
        handle = len(self.ever_subscribed) + 1
        self.ever_subscribed.append(path)
        self.subscriptions[handle] = (path, cb)
        return handle

    async def unsubscribe(self, handle: int) -> None:
        self.subscriptions.pop(handle, None)

    async def publish_raw(self, subject: Subject, payload: bytes) -> None:
        self.published.append((subject.path, payload))
        for path, cb in list(self.subscriptions.values()):
            if path == subject.path:
                await cb(IncomingMessage(data=payload, reply_subject=None, subject=path))


# parity-exempt: subset stand-in for NatsClient: the pod's replies, its audit publish and its core publishes
class _PodNats:
    max_payload = 16 * 1024 * 1024

    def __init__(self, bus: _Bus) -> None:
        self._bus = bus
        self.replies: list[Any] = []

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        del reply_subject
        self.replies.append(message)

    async def jetstream_publish(
        self, *, subject: Any, payload: bytes, timeout: Any = None, headers: Any = None
    ) -> None:
        del subject, payload, timeout, headers

    async def publish_raw(self, *, subject: Subject, payload: bytes) -> None:
        await self._bus.publish_raw(subject, payload)


# parity-exempt: subset stand-in for NatsClient: the proxy's forward delivered in-process to a real ToolServer
class _ProxyNats:
    def __init__(self, server: ToolServer, pod_nats: _PodNats, bus: _Bus) -> None:
        self._server = server
        self._pod_nats = pod_nats
        self._bus = bus
        self.forwarded: list[CallRequest] = []
        self.replies: list[Any] = []
        #: when set, the forward's updates subject is replaced by this one on the way to the pod
        self.rewrite_updates_subject: str | None = None

    async def subscribe(self, *, subject: Subject | str, cb: Any, queue: str | None = None) -> Any:
        del queue
        if isinstance(subject, Subject) and subject.path == str(Subjects.tools_call()):
            return object()  # the proxy's own call subscription; calls are handed to handle_call directly
        return await self._bus.subscribe(subject, cb)

    async def unsubscribe(self, sub: Any) -> None:
        if isinstance(sub, int):
            await self._bus.unsubscribe(sub)

    async def publish_raw(self, *, subject: Subject, payload: bytes) -> None:
        await self._bus.publish_raw(subject, payload)

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        del timeout
        forwarded = CallRequest.model_validate_json(payload)
        if self.rewrite_updates_subject is not None and forwarded.updates is not None:
            body = json.loads(payload)
            body["updates"]["subject"] = self.rewrite_updates_subject
            payload = json.dumps(body).encode("utf-8")
        self.forwarded.append(forwarded)
        before = len(self._pod_nats.replies)
        await self._server.handle_call(
            IncomingMessage(data=payload, reply_subject="_INBOX.proxy", subject=subject.path)
        )
        reply: BaseModel = self._pod_nats.replies[before]
        return reply.model_dump_json().encode("utf-8")

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        del reply_subject
        self.replies.append(message)


# parity-exempt: subset stand-in for NatsClient: the caller's request to the real proxy, and its own subscriptions
class _CallerNats:
    def __init__(self, proxy: CallProxy, proxy_nats: _ProxyNats, bus: _Bus) -> None:
        self._proxy = proxy
        self._proxy_nats = proxy_nats
        self._bus = bus
        self.sent: list[Any] = []

    async def subscribe(self, subject: Subject | str | None = None, *, cb: Any = None) -> Any:
        assert subject is not None
        return await self._bus.subscribe(subject, cb)

    async def unsubscribe(self, sub: Any) -> None:
        await self._bus.unsubscribe(sub)

    async def request(self, *, subject: Subject, message: Any, response_type: Any, timeout: timedelta) -> Any:
        del timeout
        self.sent.append(message)
        before = len(self._proxy_nats.replies)
        await self._proxy.handle_call(
            IncomingMessage(
                data=message.model_dump_json().encode("utf-8"), reply_subject="_INBOX.c", subject=subject.path
            )
        )
        for _ in range(200):
            if len(self._proxy_nats.replies) > before:
                break
            await asyncio.sleep(0)
        reply = self._proxy_nats.replies[before]
        return response_type.model_validate_json(reply.model_dump_json())


class _Signer:
    """the SDK ``PopSigner`` shape over a holder key."""

    def __init__(self, holder_key: Ed25519PrivateKey) -> None:
        self._holder_key = holder_key

    def sign(self, *, access_token: str, tool_name: str, arguments: dict[str, Any], correlation_id: str | None) -> str:
        return make_pop_proof(
            holder_key=self._holder_key,
            access_token_hash=access_token_hash(access_token),
            body_hash=canonical_call_hash(tool_name, arguments, correlation_id),
            nonce=str(uuid7()),
            iat=int(time.time()),
        )


class _Building(TearsTool):
    """reports three steps and two log entries through the call context, when it is given one."""

    face_mcp = True

    def __init__(self) -> None:
        self.contexts: list[CallContext | None] = []

    async def execute(self, *, ctx: CallContext | None = None, **kwargs: Any) -> ToolResult:
        del kwargs
        self.contexts.append(ctx)
        if ctx is not None:
            await ctx.report_progress(1, 3, "cloning")
            await ctx.log("info", {"phase": "clone"})
            await ctx.log("debug", "noise the caller does not take")
            await ctx.report_progress(2, 3, "building")
            # follows 2 with nothing between, before the sender has run: the two coalesce to this one
            await ctx.report_progress(3, 3, "done")
        return ToolResult(success=True, content="built")

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name=_TOOL, version=_VERSION, description="builds", input_schema={"type": "object"})

    def mcp_name(self) -> str:
        return _TOOL

    def mcp_version(self) -> str:
        return _VERSION


class _Plain(TearsTool):
    """an execute that declares no context."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(success=True, content="plain")

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name=_TOOL, version=_VERSION, description="plain", input_schema={"type": "object"})

    def mcp_name(self) -> str:
        return _TOOL

    def mcp_version(self) -> str:
        return _VERSION


class _Sink:
    """a caller's sink, the shape a ``threetears.mcp.CallContext`` has: what its client takes, and what it got."""

    def __init__(self, *, progress_token: str | int | None, log_levels: frozenset[UpdateLogLevel]) -> None:
        self._progress_token = progress_token
        self._log_levels = log_levels
        self.received: list[tuple[Any, ...]] = []

    @property
    def progress_token(self) -> str | int | None:
        return self._progress_token

    @property
    def log_levels(self) -> frozenset[UpdateLogLevel]:
        return self._log_levels

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        self.received.append(("progress", progress, total, message))

    async def log(self, level: UpdateLogLevel, data: Any, *, logger: str | None = None) -> None:
        del logger
        self.received.append(("log", level, data))


_INFO_AND_UP: frozenset[UpdateLogLevel] = frozenset(
    ("info", "notice", "warning", "error", "critical", "alert", "emergency")
)


def _pod_token(priv: Any, *, pod_id: UUID, holder_key: Ed25519PrivateKey) -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub=str(pod_id),
            customer_id=PLATFORM_CUSTOMER_SENTINEL,
            sid="sid-1",
            pod_id="pod-1",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(holder_key.public_key()),
        ),
        signing_key=priv,
        kid="kid-1",
    )


class _Mesh:
    """caller, registry and pod wired over one bus."""

    def __init__(self, client: ToolCallClient, caller_id: UUID, bus: _Bus, hop: _ProxyNats) -> None:
        self.client = client
        self.caller_id = caller_id
        self.bus = bus
        self.hop = hop


async def _mesh(tool: TearsTool) -> _Mesh:
    hub_priv, hub_pub = generate_signing_keypair()
    seed = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
    proxy_signer = ProxyAssertionSigner.from_secret(SecretStr(seed))
    combined = {"keys": [*build_jwks({"kid-1": hub_pub})["keys"], *proxy_signer.public_jwks()["keys"]]}
    bus = _Bus()
    pod_nats = _PodNats(bus)
    server = ToolServer(
        nats_client=pod_nats,  # type: ignore[arg-type]
        pod_id=_SERVING_POD,
        namespace=_NS,
        jwks_provider=lambda: combined,
        assertion_replay_guard=FakeReplayGuard(),
    )
    server.register(tool)
    catalog = ToolCatalog()
    await catalog.register(
        uniform_entry(
            tool_name=_TOOL,
            tool_version=_VERSION,
            full_name=f"{_TOOL}@{_VERSION}",
            description="builds",
            input_schema={"type": "object", "properties": {}},
            endpoints=[ToolEndpoint(pod_id=_SERVING_POD, status="available")],
            timeout_seconds=5.0,
        )
    )
    proxy = CallProxy(
        catalog,
        AllowAllAuthorizer(),
        FakeReplayGuard(),
        limit_guard=AllowAllLimitGuard(),
        namespace=_NS,
        jwks_provider=lambda: {"keys": build_jwks({"kid-1": hub_pub})["keys"]},
        proxy_signer=proxy_signer,
    )
    proxy_nats = _ProxyNats(server, pod_nats, bus)
    await proxy.start(proxy_nats)  # type: ignore[arg-type]
    caller_id = uuid7()
    holder_key = Ed25519PrivateKey.generate()
    token = _pod_token(hub_priv, pod_id=caller_id, holder_key=holder_key)
    client = ToolCallClient(
        _CallerNats(proxy, proxy_nats, bus),
        principal_id=caller_id,
        identity_token=lambda: token,
        pop_signer=_Signer(holder_key),
    )
    return _Mesh(client, caller_id, bus, proxy_nats)


class TestUpdatesCrossTheMesh:
    @pytest.mark.asyncio
    async def test_progress_and_log_entries_reach_the_caller_in_order_before_the_answer(self) -> None:
        tool = _Building()
        mesh = await _mesh(tool)
        sink = _Sink(progress_token="tok-1", log_levels=_INFO_AND_UP)

        reply = await mesh.client.call(_TOOL, _VERSION, {}, updates=sink)

        assert reply.content == "built"
        # in order; the debug entry the caller does not take was never sent; 2 coalesced into 3 on the pod
        assert sink.received == [
            ("progress", 1.0, 3.0, "cloning"),
            ("log", "info", {"phase": "clone"}),
            ("progress", 3.0, 3.0, "done"),
        ]
        # the tool was given the caller's own progress token
        [ctx] = tool.contexts
        assert ctx is not None and ctx.progress_token == "tok-1"

    @pytest.mark.asyncio
    async def test_each_hop_uses_its_own_partys_subject(self) -> None:
        mesh = await _mesh(_Building())

        await mesh.client.call(_TOOL, _VERSION, {}, updates=_Sink(progress_token=7, log_levels=frozenset()))

        [forwarded] = mesh.hop.forwarded
        assert forwarded.updates is not None
        assert forwarded.updates.subject.startswith(f"{_NS}.tools.progress.{_SERVING_POD}.")
        assert forwarded.updates.progress_token == 7
        assert forwarded.updates.log_level is None
        caller_prefix = f"{_NS}.tools.updates.{mesh.caller_id}."
        published = [path for path, _payload in mesh.bus.published]
        assert published and {path.rsplit(".", 1)[0] + "." for path in published} == {
            forwarded.updates.subject.rsplit(".", 1)[0] + ".",
            caller_prefix,
        }
        # every subscription is gone once the call has answered
        assert mesh.bus.subscriptions == {}

    @pytest.mark.asyncio
    async def test_a_caller_that_takes_log_entries_only_gets_no_progress(self) -> None:
        tool = _Building()
        mesh = await _mesh(tool)
        sink = _Sink(progress_token=None, log_levels=_INFO_AND_UP)

        await mesh.client.call(_TOOL, _VERSION, {}, updates=sink)

        # the context had no progress token, so its progress was never sent, on either hop
        assert sink.received == [("log", "info", {"phase": "clone"})]
        [ctx] = tool.contexts
        assert ctx is not None and ctx.progress_token is None
        assert mesh.hop.forwarded[0].updates is not None
        assert mesh.hop.forwarded[0].updates.log_level == "info"
        assert {json.loads(payload)["kind"] for _path, payload in mesh.bus.published} == {"log"}

    @pytest.mark.asyncio
    async def test_entries_below_the_callers_level_are_not_sent_from_the_pod(self) -> None:
        tool = _Building()
        mesh = await _mesh(tool)

        await mesh.client.call(
            _TOOL, _VERSION, {}, updates=_Sink(progress_token=None, log_levels=frozenset(("warning", "error")))
        )

        [ctx] = tool.contexts
        assert ctx is not None
        assert mesh.hop.forwarded[0].updates is not None and mesh.hop.forwarded[0].updates.log_level == "warning"
        assert mesh.bus.published == []

    @pytest.mark.asyncio
    async def test_a_pod_handed_a_subject_that_is_not_its_own_publishes_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = _Building()
        mesh = await _mesh(tool)
        mesh.hop.rewrite_updates_subject = Subjects.tools_progress("another-pod", uuid7()).path

        reply = await mesh.client.call(_TOOL, _VERSION, {}, updates=_Sink(progress_token=1, log_levels=_INFO_AND_UP))

        assert reply.content == "built"
        assert tool.contexts == [None]
        assert mesh.bus.published == []
        assert any("not this pod's own" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_caller_that_asks_for_nothing_changes_nothing(self) -> None:
        """the A/B: no token and no levels is a call exactly as it was, with no context and no subscription."""
        tool = _Building()
        mesh = await _mesh(tool)

        reply = await mesh.client.call(_TOOL, _VERSION, {}, updates=_Sink(progress_token=None, log_levels=frozenset()))

        assert reply.content == "built"
        assert tool.contexts == [None]
        assert mesh.bus.ever_subscribed == []
        assert mesh.bus.published == []
        assert mesh.hop.forwarded[0].updates is None

    @pytest.mark.asyncio
    async def test_a_tool_that_declares_no_context_is_run_with_its_arguments_alone(self) -> None:
        tool = _Plain()
        mesh = await _mesh(tool)
        sink = _Sink(progress_token="tok-1", log_levels=_INFO_AND_UP)

        reply = await mesh.client.call(_TOOL, _VERSION, {"a": 1}, updates=sink)

        assert reply.content == "plain"
        assert tool.calls == [{"a": 1}]
        assert sink.received == []
        assert mesh.bus.subscriptions == {}


def _proxy_nats_double() -> Any:
    """the proxy's transport with a scripted pod: it publishes what it is given to the forward's updates subject."""

    class _ScriptedPod:
        """the proxy transport subset the relay uses, with a pod that publishes scripted updates."""

        def __init__(self) -> None:
            self.bus = _Bus()
            self.forwarded: list[CallRequest] = []
            self.replies: list[Any] = []
            self.updates_to_send: list[bytes] = []

        async def subscribe(self, *, subject: Subject | str, cb: Any, queue: str | None = None) -> Any:
            del queue
            return await self.bus.subscribe(subject, cb)

        async def unsubscribe(self, sub: Any) -> None:
            await self.bus.unsubscribe(sub)

        async def publish_raw(self, *, subject: Subject, payload: bytes) -> None:
            await self.bus.publish_raw(subject, payload)

        async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
            del subject, timeout
            request = CallRequest.model_validate_json(payload)
            self.forwarded.append(request)
            if request.updates is not None:
                for update in self.updates_to_send:
                    await self.bus.publish_raw(Subject.raw(request.updates.subject), update)
            if request.result_subject is not None:
                accepted = CallAccepted(accepted=True, pod_id="pod-1", result_subject=request.result_subject)
                return accepted.model_dump_json().encode("utf-8")
            return ProxyCallResponse(success=True, content="ok").model_dump_json().encode("utf-8")

        async def jetstream_result_waiter(self, *, subject: Subject, stream: str, wait_budget: timedelta) -> Any:
            del subject, stream, wait_budget

            class _Waiter:
                async def wait(self, *, timeout: timedelta) -> bytes:
                    del timeout
                    return ProxyCallResponse(success=True, content="long").model_dump_json().encode("utf-8")

                async def close(self) -> None:
                    return None

            return _Waiter()

        async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
            del reply_subject
            self.replies.append(message)

    return _ScriptedPod()


async def _catalog(timeout_seconds: float = 5.0) -> ToolCatalog:
    catalog = ToolCatalog()
    await catalog.register(
        uniform_entry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=timeout_seconds,
            endpoints=[ToolEndpoint(pod_id="pod-1", status="available")],
        )
    )
    return catalog


async def _dispatch(nats: Any, request: Any, *, timeout_seconds: float = 5.0) -> None:
    proxy = make_proxy(await _catalog(timeout_seconds), namespace=_NS)
    await proxy.start(nats)
    await proxy.handle_call(
        IncomingMessage(data=request.model_dump_json().encode("utf-8"), reply_subject="_INBOX.a", subject="x")
    )
    await proxy.stop()


def _update(seq: int) -> bytes:
    return CallUpdate(seq=seq, kind="progress", progress=float(seq)).model_dump_json().encode("utf-8")


class TestTheRegistryRelay:
    @pytest.mark.asyncio
    async def test_an_updates_subject_naming_another_caller_is_refused_and_nothing_runs(self) -> None:
        nats = _proxy_nats_double()
        peer = Subjects.tools_updates(uuid4(), uuid7()).path
        request = make_authed_request().model_copy(update={"updates": CallUpdates(subject=peer, progress_token=1)})

        await _dispatch(nats, request)

        assert nats.forwarded == []
        [refusal] = nats.replies
        assert isinstance(refusal, ProxyCallResponse)
        assert refusal.error_code == "MALFORMED_REQUEST"
        assert "does not belong to the verified caller" in (refusal.error or "")

    @pytest.mark.asyncio
    async def test_a_durable_caller_is_refused_where_it_listens(self) -> None:
        nats = _proxy_nats_double()
        peer = Subjects.tools_updates(uuid4(), uuid7()).path
        request = make_authed_request().model_copy(
            update={
                "updates": CallUpdates(subject=peer, progress_token=1),
                "result_subject": Subjects.tools_reply(DEFAULT_AGENT_ID, uuid7()).path,
            }
        )

        await _dispatch(nats, request)

        [refusal] = nats.replies
        assert isinstance(refusal, ProxyCallAccepted)
        assert refusal.accepted is False

    @pytest.mark.asyncio
    async def test_a_subject_one_token_too_deep_is_refused(self) -> None:
        nats = _proxy_nats_double()
        own_but_deeper = Subjects.tools_updates(DEFAULT_AGENT_ID, uuid7()).path + ".extra"
        request = make_authed_request().model_copy(
            update={"updates": CallUpdates(subject=own_but_deeper, progress_token=1)}
        )

        await _dispatch(nats, request)

        assert nats.forwarded == []

    @pytest.mark.asyncio
    async def test_the_relay_bounds_a_pod_that_floods_and_drops_what_it_cannot_read(self) -> None:
        nats = _proxy_nats_double()
        own = Subjects.tools_updates(DEFAULT_AGENT_ID, uuid7())
        received: list[bytes] = []

        async def caller(msg: IncomingMessage) -> None:
            received.append(msg.data)

        await nats.bus.subscribe(own, caller)
        nats.updates_to_send = [b"not an update", *(_update(seq) for seq in range(1, 201))]
        request = make_authed_request().model_copy(update={"updates": CallUpdates(subject=own.path, progress_token=1)})

        await _dispatch(nats, request)

        # the unreadable one is dropped; of 200 sent at once, the burst is relayed and the rest dropped
        assert RELAY_UPDATE_BURST <= len(received) <= RELAY_UPDATE_BURST + 2
        assert received[0] == _update(1)
        [answer] = nats.replies
        assert answer.content == "ok"

    @pytest.mark.asyncio
    async def test_a_long_call_on_the_durable_path_relays_its_updates_too(self) -> None:
        nats = _proxy_nats_double()
        own = Subjects.tools_updates(DEFAULT_AGENT_ID, uuid7())
        received: list[bytes] = []

        async def caller(msg: IncomingMessage) -> None:
            received.append(msg.data)

        await nats.bus.subscribe(own, caller)
        nats.updates_to_send = [_update(1), _update(2)]
        request = make_authed_request().model_copy(update={"updates": CallUpdates(subject=own.path, progress_token=1)})

        await _dispatch(nats, request, timeout_seconds=1200.0)

        [forwarded] = nats.forwarded
        assert forwarded.result_subject is not None
        assert forwarded.updates is not None and forwarded.updates.subject.startswith(f"{_NS}.tools.progress.pod-1.")
        assert received == [_update(1), _update(2)]
        assert nats.replies[-1].content == "long"

    @pytest.mark.asyncio
    async def test_each_forward_gets_a_fresh_pod_subject(self) -> None:
        nats = _proxy_nats_double()
        proxy = make_proxy(await _catalog(), namespace=_NS)
        await proxy.start(nats)
        for _ in range(2):
            own = Subjects.tools_updates(DEFAULT_AGENT_ID, uuid7()).path
            request = make_authed_request().model_copy(update={"updates": CallUpdates(subject=own, log_level="info")})
            await proxy.handle_call(
                IncomingMessage(data=request.model_dump_json().encode("utf-8"), reply_subject="_INBOX.a", subject="x")
            )
        await proxy.stop()

        subjects = [request.updates.subject for request in nats.forwarded if request.updates is not None]
        assert len(subjects) == 2 and len(set(subjects)) == 2
        assert all(subject.startswith(f"{_NS}.tools.progress.pod-1.") for subject in subjects)
        assert all(request.updates is not None and request.updates.log_level == "info" for request in nats.forwarded)


def _caller_face(client: ToolCallClient) -> McpServer:
    """an MCP server whose tool ``build`` forwards to the mesh, handing the call its own context."""

    async def build(*, ctx: CallContext) -> str:
        reply = await client.call(_TOOL, _VERSION, {}, updates=ctx)
        return reply.content

    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="build",
            description="builds on the mesh",
            input_schema={"type": "object", "properties": {}},
            required_permission="t.build.call",
            handler=build,
        )
    )
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4(), is_admin=True))
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(name="caller-face", identity_provider=provider, authorizer=authorizer, registry=registry)


class TestTheCallersMcpFace:
    @pytest.mark.asyncio
    async def test_an_mcp_client_gets_notifications_for_its_own_progress_token(self) -> None:
        mesh = await _mesh(_Building())
        server = _caller_face(mesh.client)
        progress: list[tuple[float, float | None, str | None]] = []
        logs: list[tuple[str, Any]] = []

        async def on_progress(value: float, total: float | None, message: str | None) -> None:
            progress.append((value, total, message))

        async def on_log(params: mcp_types.LoggingMessageNotificationParams) -> None:
            logs.append((params.level, params.data))

        await server.start()
        app = server.build_http_app(json_response=False)
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://mcp.test")
        try:
            async with app.router.lifespan_context(app), http:
                async with streamable_http_client("http://mcp.test/mcp", http_client=http) as (read, write):
                    async with ClientSession(read, write, logging_callback=on_log) as session:
                        await session.initialize()
                        result = await session.call_tool("build", {}, progress_callback=on_progress)
        finally:
            await server.stop()

        assert result.is_error is False
        block = result.content[0]
        assert isinstance(block, mcp_types.TextContent) and block.text == "built"
        assert progress == [(1.0, 3.0, "cloning"), (3.0, 3.0, "done")]
        # a handshake-era client with no logging/setLevel takes every level; the tool's debug entry is sent too
        assert ("info", {"phase": "clone"}) in logs
