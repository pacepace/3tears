"""the composed path: a tool pod's call through the real proxy, with a signer, into a real tool server.

Every gate on this path was unit-correct on its own and the composition still
refused every pod-originated call: the registry admitted the pod, then minted no
proxy assertion for a caller with no customer, and the serving pod's mirror gate
parsed the forwarded sentinel as a UUID. A test that fakes the pod with a raw
subscriber and builds the proxy with no signer cannot see either. This one runs
the real :class:`CallProxy` WITH a :class:`ProxyAssertionSigner` and forwards --
over an in-process transport that delivers bytes and nothing else -- into the
real :class:`ToolServer` WITH the JWKS that verifies both the hub's token and the
proxy's assertion, and asserts the tool actually ran under the pod's identity.

The A/B is the signer: the same path with no proxy signer is refused by the pod
for the missing assertion, which is the shipped failure named by its cause.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr
from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.server import CallRequest, ToolServer
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
from threetears.nats import IncomingMessage, Subject, set_default_namespace

from threetears.registry.auth import AllowAllAuthorizer, AllowAllLimitGuard
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.client import ToolCallClient, ToolCallError
from threetears.registry.proxy import CallProxy, ProxyCallResponse

_NS = "test"
_SERVING_POD = "serving-pod"
_TOOL = "probe.echo"
_VERSION = "1.0"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


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


class _StubReplayGuard:
    async def record_unique(self, nonce: str) -> bool:
        return True


class _ScopeRecordingTool(TearsTool):
    """echoes its arguments and records the call scope's context it ran under."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        scope = current_scope()
        self.contexts.append(scope.context if scope is not None else None)
        return ToolResult(success=True, content=json.dumps(kwargs))

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name=_TOOL, version=_VERSION, description="echo", input_schema={"type": "object"})

    def mcp_name(self) -> str:
        return _TOOL

    def mcp_version(self) -> str:
        return _VERSION


# parity-exempt: subset stand-in for NatsClient exposing only the publish_reply the pod's handler answers on
class _PodNats:
    def __init__(self) -> None:
        self.replies: list[Any] = []

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        self.replies.append(message)


# parity-exempt: subset stand-in for NatsClient: the proxy's forward hop delivered in-process to a real ToolServer
class _ProxyNats:
    """delivers the proxy's forwarded bytes to the real pod and hands its reply bytes back."""

    def __init__(self, server: ToolServer, pod_nats: _PodNats) -> None:
        self._server = server
        self._pod_nats = pod_nats
        self.forwarded: list[CallRequest] = []
        self.replies: list[Any] = []

    async def subscribe(self, **_kwargs: Any) -> Any:
        return object()

    async def unsubscribe(self, _sub: Any) -> None:
        return None

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        del timeout
        self.forwarded.append(CallRequest.model_validate_json(payload))
        before = len(self._pod_nats.replies)
        await self._server.handle_call(
            IncomingMessage(data=payload, reply_subject="_INBOX.proxy", subject=subject.path)
        )
        reply = self._pod_nats.replies[before]
        return reply.model_dump_json().encode("utf-8")

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        del reply_subject
        self.replies.append(message)


# parity-exempt: subset stand-in for NatsClient: the caller's typed request delivered in-process to the real proxy
class _CallerNats:
    def __init__(self, proxy: CallProxy, proxy_nats: _ProxyNats) -> None:
        self._proxy = proxy
        self._proxy_nats = proxy_nats

    async def request(self, *, subject: Subject, message: Any, response_type: Any, timeout: timedelta) -> Any:
        del timeout
        before = len(self._proxy_nats.replies)
        await self._proxy.handle_call(
            IncomingMessage(
                data=message.model_dump_json().encode("utf-8"), reply_subject="_INBOX.caller", subject=subject.path
            )
        )
        # ``handle_call`` spawns the dispatch as a task; let it run to its reply.
        for _ in range(100):
            if len(self._proxy_nats.replies) > before:
                break
            await asyncio.sleep(0)
        reply = self._proxy_nats.replies[before]
        return response_type.model_validate_json(reply.model_dump_json())


def _pod_token(priv: Any, *, pod_id: UUID, holder_key: Ed25519PrivateKey, customer_claim: str) -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub=str(pod_id),
            customer_id=customer_claim,
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


async def _catalog() -> ToolCatalog:
    catalog = ToolCatalog()
    await catalog.register(
        CatalogEntry(
            tool_name=_TOOL,
            tool_version=_VERSION,
            full_name=f"{_TOOL}@{_VERSION}",
            description="echo",
            input_schema={"type": "object", "properties": {}},
            endpoints=[ToolEndpoint(pod_id=_SERVING_POD, status="available")],
            timeout_seconds=5.0,
        )
    )
    return catalog


async def _composed(*, with_signer: bool) -> tuple[ToolCallClient, _ScopeRecordingTool, _ProxyNats, UUID]:
    """wire caller -> real proxy -> real pod in process; returns the client, the pod's tool, the hop, the pod id."""
    hub_priv, hub_pub = generate_signing_keypair()
    seed = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
    proxy_signer = ProxyAssertionSigner.from_secret(SecretStr(seed))
    # the pod verifies BOTH the hub's token and the proxy's assertion off one JWKS carrying both keys
    combined = {"keys": [*build_jwks({"kid-1": hub_pub})["keys"], *proxy_signer.public_jwks()["keys"]]}

    server = ToolServer(
        nats_url="nats://localhost:9999",
        pod_id=_SERVING_POD,
        jwks_provider=lambda: combined,
        assertion_replay_guard=_StubReplayGuard(),
    )
    tool = _ScopeRecordingTool()
    server.register(tool)
    pod_nats = _PodNats()
    setattr(server, "_nc", pod_nats)

    proxy = CallProxy(
        await _catalog(),
        AllowAllAuthorizer(),
        _StubReplayGuard(),
        limit_guard=AllowAllLimitGuard(),
        namespace=_NS,
        jwks_provider=lambda: {"keys": build_jwks({"kid-1": hub_pub})["keys"]},
        proxy_signer=proxy_signer if with_signer else None,
    )
    proxy_nats = _ProxyNats(server, pod_nats)
    await proxy.start(proxy_nats)  # type: ignore[arg-type]

    pod_id = uuid7()
    holder_key = Ed25519PrivateKey.generate()
    token = _pod_token(hub_priv, pod_id=pod_id, holder_key=holder_key, customer_claim=PLATFORM_CUSTOMER_SENTINEL)
    client = ToolCallClient(
        _CallerNats(proxy, proxy_nats),  # type: ignore[arg-type]
        principal_id=pod_id,
        identity_token=lambda: token,
        pop_signer=_Signer(holder_key),
    )
    return client, tool, proxy_nats, pod_id


class TestTheComposedPath:
    @pytest.mark.asyncio
    async def test_a_tool_pods_call_is_signed_by_the_proxy_and_runs_on_the_real_pod(self) -> None:
        client, tool, hop, pod_id = await _composed(with_signer=True)

        reply = await client.call(_TOOL, _VERSION, {"text": "hi"})

        assert isinstance(reply, ProxyCallResponse)
        assert reply.success is True
        assert json.loads(reply.content) == {"text": "hi"}
        # the forward hop carried a proxy assertion for a caller with no customer
        assert len(hop.forwarded) == 1
        assert hop.forwarded[0].proxy_assertion is not None
        forwarded_context = hop.forwarded[0].context
        assert forwarded_context is not None
        assert forwarded_context.agent_id == pod_id
        assert forwarded_context.customer_id is None
        # and the tool ran under the pod's identity, with no customer and no user
        assert len(tool.contexts) == 1
        assert tool.contexts[0].agent_id == pod_id
        assert tool.contexts[0].customer_id is None
        assert tool.contexts[0].user_id is None

    @pytest.mark.asyncio
    async def test_without_the_proxy_signer_the_pod_refuses_the_same_call(self) -> None:
        """the A/B: the shipped failure was an unsigned forward, and the pod is bound to refuse one."""
        client, tool, hop, _pod_id = await _composed(with_signer=False)

        with pytest.raises(ToolCallError) as excinfo:
            await client.call(_TOOL, _VERSION, {"text": "hi"})

        assert "proxy assertion verification failed" in str(excinfo.value)
        assert hop.forwarded[0].proxy_assertion is None
        assert tool.contexts == []  # the tool never ran
