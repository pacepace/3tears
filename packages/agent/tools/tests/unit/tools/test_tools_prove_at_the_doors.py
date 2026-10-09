"""a tool reaching a platform door proves the credential it presents, and never picks a token itself."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid7

import pytest
from threetears.core.security import (
    DoorCredential,
    PopKey,
    RegistrationKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    mint_delegated_token,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims
from threetears.core.testing.replay_guard import FakeReplayGuard
from threetears.media.contracts import ObjectHandle

from packages.agent.tools.tests.unit.tools.pod_auth import (
    ScopeRecordingTool,
    deliver_call,
    jwks_provider,
    recording_tool_server,
    signed_call_payload,
)
from threetears.agent.tools.call_scope import ToolCallScope, build_call_scope, enter_call_scope
from threetears.agent.tools.consume import resolve_object
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.engagement_resolver import EngagementScopeResponseModel, HubEngagementScopeResolver
from threetears.agent.tools.namespace_discovery_client import NamespaceDiscoveryClient, NamespaceDiscoveryResponse
from threetears.agent.tools.object_resolver import HubObjectResolver, ObjectResolveResponseModel

_REGISTRY, _REGISTRY_PUB = generate_signing_keypair()
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"registry-1": _REGISTRY_PUB, "hub-1": _HUB_PUB})
_POD, _POD_PUB = generate_signing_keypair()
_AGENT, _AGENT_PUB = generate_signing_keypair()
_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_KEY = f"{_CUSTOMER}/conversation-x/{_OBJECT}/report.md"


def _delegated(door: str) -> str:
    now = int(time.time())
    return mint_delegated_token(
        signing_key=_REGISTRY,
        kid="registry-1",
        agent_id="agent-1",
        pod_id="pod-1",
        customer_id=f"{_CUSTOMER}",
        user_id="user-1",
        doors=(door,),
        holder_jkt=jwk_thumbprint(_POD_PUB),
        correlation_id="corr-1",
        deadline=now + 60,
        now=now,
    )


def _agent_token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id=f"{_CUSTOMER}",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_AGENT_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


class _RecordingNats:
    """answers a typed ``request`` with one reply, keeping what was sent."""

    def __init__(self, reply: object) -> None:
        self._reply = reply
        self.sent: list[dict[str, Any]] = []

    async def request(self, *, subject: object, message: Any, response_type: object, timeout: object) -> object:
        self.sent.append(json.loads(message.model_dump_json()))
        return self._reply


class _RecordingRawNats:
    """answers a bytes ``request`` with one reply, keeping what was sent."""

    def __init__(self, reply: bytes) -> None:
        self._reply = reply
        self.sent: list[dict[str, Any]] = []

    async def request(self, subject: str, payload: bytes, *, timeout: float) -> object:
        self.sent.append(json.loads(payload))
        return SimpleNamespace(data=self._reply)


# parity-with: threetears.agent.tools.object_resolver.ObjectResolver
class _FakeResolver:
    """returns one handle, keeping the credential it was given."""

    def __init__(self) -> None:
        self.presented: list[tuple[str, DoorCredential | None]] = []

    async def resolve(
        self, object_id: UUID, *, customer_id: UUID, identity_token: str, credential: DoorCredential | None = None
    ) -> ObjectHandle:
        self.presented.append((identity_token, credential))
        return ObjectHandle(
            object_id=object_id, s3_key=_KEY, mime_type="text/markdown", size_bytes=1, summary=None, category=None
        )


class _ResolverWrittenBeforeProofs:
    """an implementer of the protocol from before the credential existed."""

    async def resolve(self, object_id: UUID, *, customer_id: UUID, identity_token: str) -> ObjectHandle:
        return ObjectHandle(
            object_id=object_id, s3_key=_KEY, mime_type="text/markdown", size_bytes=1, summary=None, category=None
        )


class TestTheScopesCredential:
    """the server's one key decides which token is proven, by its type."""

    async def test_a_tool_pods_registration_key_proves_the_delegation(self) -> None:
        token = _delegated("hub.object.resolve")
        context = CallContext(customer_id=_CUSTOMER, identity_token=_agent_token(), delegated_token=token)
        scope = await build_call_scope(context, principal_is_tool_pod=False, door_key=RegistrationKey(_POD))
        assert scope.door_credential() == DoorCredential(token=token, holder_key=_POD)

    async def test_a_pod_without_its_key_still_forwards_the_callers_token(self) -> None:
        agent = _agent_token()
        context = CallContext(customer_id=_CUSTOMER, identity_token=agent, delegated_token=_delegated("l3"))
        scope = await build_call_scope(context, principal_is_tool_pod=False)
        assert scope.door_credential() == DoorCredential(token=agent)

    async def test_an_in_process_servers_pop_key_proves_the_agents_own_token(self) -> None:
        agent = _agent_token()
        context = CallContext(customer_id=_CUSTOMER, identity_token=agent)
        scope = await build_call_scope(context, principal_is_tool_pod=False, door_key=PopKey(_AGENT))
        credential = scope.door_credential()
        assert credential == DoorCredential(token=agent, holder_key=_AGENT)
        assert credential is not None and credential.proves

    async def test_a_registration_key_never_proves_the_agents_own_token(self) -> None:
        agent = _agent_token()
        scope = await build_call_scope(
            CallContext(customer_id=_CUSTOMER, identity_token=agent),
            principal_is_tool_pod=False,
            door_key=RegistrationKey(_AGENT),
        )
        assert scope.door_credential() == DoorCredential(token=agent)

    def test_a_hand_built_scope_presents_its_contexts_token_alone(self) -> None:
        scope = ToolCallScope(context=CallContext(identity_token="the-agents-own"))
        assert scope.door_credential() == DoorCredential(token="the-agents-own")
        assert ToolCallScope().door_credential() is None


class TestTheConsumeHelper:
    """it passes the credential only when it proves, so an older resolver keeps working."""

    async def test_it_presents_a_proving_credential(self) -> None:
        token = _delegated("hub.object.resolve")
        resolver = _FakeResolver()
        context = CallContext(customer_id=_CUSTOMER, identity_token=_agent_token(), delegated_token=token)
        scope = await build_call_scope(
            context, principal_is_tool_pod=False, object_resolver=resolver, door_key=RegistrationKey(_POD)
        )
        async with enter_call_scope(scope):
            await resolve_object(_OBJECT)
        assert resolver.presented == [(token, DoorCredential(token=token, holder_key=_POD))]

    async def test_a_resolver_written_before_proofs_still_resolves_a_bearer_call(self) -> None:
        context = CallContext(customer_id=_CUSTOMER, identity_token=_agent_token())
        scope = await build_call_scope(
            context, principal_is_tool_pod=False, object_resolver=_ResolverWrittenBeforeProofs()
        )
        async with enter_call_scope(scope):
            handle = await resolve_object(_OBJECT)
        assert handle.s3_key == _KEY


class TestTheDoorClients:
    """each client sends a request the door verifies for its own operation."""

    async def test_the_object_resolver(self) -> None:
        token = _delegated("hub.object.resolve")
        nats = _RecordingNats(
            ObjectResolveResponseModel(success=True, s3_key=_KEY, mime_type="text/markdown", size_bytes=1)
        )
        resolver = HubObjectResolver(nats, request_timeout_seconds=5.0)  # type: ignore[arg-type]
        await resolver.resolve(
            _OBJECT,
            customer_id=_CUSTOMER,
            identity_token=token,
            credential=DoorCredential(token=token, holder_key=_POD),
        )
        caller = verify_door_request(
            nats.sent[0], jwks=_JWKS, hub_issuer="hub", door="hub.object.resolve", op="resolve"
        )
        assert (caller.agent_id, caller.pod_id) == ("agent-1", "pod-1")

    async def test_the_engagement_resolver(self) -> None:
        token = _delegated("hub.engagement")
        nats = _RecordingNats(EngagementScopeResponseModel(success=True, customer_id=_CUSTOMER, targets=[]))
        resolver = HubEngagementScopeResolver(nats, request_timeout_seconds=5.0)  # type: ignore[arg-type]
        await resolver.resolve(
            uuid7(),
            customer_id=_CUSTOMER,
            identity_token=token,
            credential=DoorCredential(token=token, holder_key=_POD),
        )
        caller = verify_door_request(nats.sent[0], jwks=_JWKS, hub_issuer="hub", door="hub.engagement", op="scope")
        assert caller.pod_id == "pod-1"

    async def test_namespace_discovery(self) -> None:
        token = _agent_token()
        nats = _RecordingRawNats(NamespaceDiscoveryResponse(items=[]).model_dump_json().encode())
        client = NamespaceDiscoveryClient(nats, "test", timeout_seconds=1.0)
        await client.discover(
            correlation_id=uuid7(), identity_token=token, credential=DoorCredential(token=token, holder_key=_AGENT)
        )
        caller = verify_door_request(
            nats.sent[0], jwks=_JWKS, hub_issuer="hub", door="namespace.discover", op="discover"
        )
        assert (caller.agent_id, caller.pod_id) == ("agent-1", None)

    async def test_a_credential_for_another_token_is_refused_by_the_resolver(self) -> None:
        nats = _RecordingNats(
            ObjectResolveResponseModel(success=True, s3_key=_KEY, mime_type="text/markdown", size_bytes=1)
        )
        resolver = HubObjectResolver(nats, request_timeout_seconds=5.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="another token"):
            await resolver.resolve(
                _OBJECT,
                customer_id=_CUSTOMER,
                identity_token="bearer",
                credential=DoorCredential(token=_agent_token(), holder_key=_AGENT),
            )

    async def test_a_resolver_given_no_credential_sends_the_token_alone(self) -> None:
        nats = _RecordingNats(
            ObjectResolveResponseModel(success=True, s3_key=_KEY, mime_type="text/markdown", size_bytes=1)
        )
        resolver = HubObjectResolver(nats, request_timeout_seconds=5.0)  # type: ignore[arg-type]
        await resolver.resolve(_OBJECT, customer_id=_CUSTOMER, identity_token="bearer")
        assert nats.sent[0]["identity_token"] == "bearer"
        assert nats.sent[0]["identity_proof"] is None


async def test_the_tool_server_puts_its_key_on_each_calls_scope() -> None:
    """the server's one key reaches every call's scope.

    The calls in this harness carry a cnf-less forwarded token (pod_auth), so neither key binds it:
    the credential is bearer whichever key the server holds. A key only proves the token it binds.
    """
    for key in (PopKey(_AGENT), RegistrationKey(_POD)):
        server, rec = recording_tool_server(
            pod_id="door-pod",
            jwks_provider=jwks_provider,
            assertion_replay_guard=FakeReplayGuard(),
            door_key=lambda key=key: key,
        )
        tool = ScopeRecordingTool()
        server.register(tool)
        await deliver_call(server, signed_call_payload(pod_id="door-pod", customer_id=_CUSTOMER), pod_id="door-pod")
        assert rec.last_reply[1].success is True, rec.last_reply[1].error
        (scope,) = tool.scopes
        credential = scope.door_credential()
        assert credential is not None
        assert credential.token == scope.context.identity_token
        assert not credential.proves
