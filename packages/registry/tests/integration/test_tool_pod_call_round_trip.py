"""integration test: a tool pod calls a tool through the real proxy, on a real bus, with no user.

the scaffold the dataset-access work rests on. a TOOL-POD principal -- a hub-style
token whose customer claim is the platform sentinel and whose ``cnf`` binds a
holder key -- calls a registered tool through :class:`ToolCallClient`, over a NATS
testcontainer, into the real :class:`CallProxy` fronting the real
:class:`RbacEvaluatorAuthorizer`, and receives the tool's reply. the four refusals
that make the admission about the right things are driven through the same door:
the same pod with no grant, the same pod with a proof for another body, an AGENT
principal with no user, and a pod asking for a tool its grant does not cover.

the rbac rows are the parity-declared in-memory loaders the authorizer's unit
tests use, over the REAL evaluator; the bus, the proxy, the door and the client
are all real.

requires docker; marked integration. run with::

    uv run pytest -m integration packages/registry/tests/integration/
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID, uuid7

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from threetears.agent.acl import (
    AclCache,
    Group,
    GroupMembership,
    MemberType,
    Namespace as AclNamespace,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.server import CallRequest
from threetears.core.namespaces import build_tool_namespace_name
from threetears.core.security import (
    PLATFORM_CUSTOMER_SENTINEL,
    IdentityClaims,
    access_token_hash,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    jwk_thumbprint,
    make_pop_proof,
    sign_identity_token,
)
from threetears.nats import IncomingMessage, NatsClient, Subjects, set_default_namespace

from threetears.registry.auth import AllowAllLimitGuard
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.client import ToolCallClient, ToolCallError
from threetears.registry.proxy import CallProxy, ProxyCallResponse
from threetears.registry.rbac_authorizer import RbacEvaluatorAuthorizer

pytestmark = pytest.mark.integration

_NS = "itest"
_ISS = "hub"
_KID = "kid-1"
_TOOL = "probe.echo"
_VERSION = "1.0"
_SERVING_POD = "serving-pod"


class _Signer:
    """the SDK ``PopSigner`` shape over a holder key; ``body_override`` mints a spliced proof."""

    def __init__(self, holder_key: Ed25519PrivateKey, *, body_override: dict[str, Any] | None = None) -> None:
        self._holder_key = holder_key
        self._body_override = body_override

    def sign(
        self,
        *,
        access_token: str,
        tool_name: str,
        arguments: dict[str, Any],
        correlation_id: str | None,
    ) -> str:
        body = self._body_override if self._body_override is not None else arguments
        return make_pop_proof(
            holder_key=self._holder_key,
            access_token_hash=access_token_hash(access_token),
            body_hash=canonical_call_hash(tool_name, body, correlation_id),
            nonce=str(uuid7()),
            iat=int(time.time()),
        )


class _StubReplayGuard:
    """accepts every first-seen nonce; the real guard's compare-and-set has its own tests."""

    async def record_unique(self, nonce: str) -> bool:
        return True


class _StubToolNamespace:
    """the fields the authorizer reads off a tool namespace row."""

    __slots__ = ("customer_id", "id", "namespace_type", "owner_agent_id", "owner_namespace")

    def __init__(self, *, id: UUID) -> None:
        self.id = id
        self.namespace_type = "tool"
        self.owner_agent_id = None
        self.customer_id = None
        self.owner_namespace = None


# parity-exempt: subset shim for the NamespaceCollection exposing only the get_by_name lookup the rbac authorizer evaluates against
class _FakeNamespaceCollection:
    def __init__(self, rows: dict[str, _StubToolNamespace]) -> None:
        self._rows = rows

    async def get_by_name(self, name: str) -> _StubToolNamespace | None:
        return self._rows.get(name)


# parity-with: threetears.agent.acl.loader.MembershipLoader
class _FakeMembershipLoader:
    def __init__(self, agents: dict[UUID, tuple[GroupMembership, ...]]) -> None:
        self._agents = agents

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        return ()

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        return self._agents.get(agent_id, ())

    async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
        return ()


# parity-with: threetears.agent.acl.loader.GrantLoader
class _FakeGrantLoader:
    def __init__(
        self,
        *,
        assignments: dict[UUID, tuple[RoleAssignment, ...]],
        roles: dict[UUID, Role],
        groups: dict[UUID, Group],
    ) -> None:
        self._assignments = assignments
        self._roles = roles
        self._groups = groups

    async def load_assignments_for_groups(
        self, group_ids: tuple[UUID, ...], namespace: AclNamespace
    ) -> tuple[RoleAssignment, ...]:
        out: list[RoleAssignment] = []
        for gid in group_ids:
            out.extend(self._assignments.get(gid, ()))
        return tuple(out)

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        return {rid: self._roles[rid] for rid in role_ids if rid in self._roles}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        return {gid: self._groups[gid] for gid in group_ids if gid in self._groups}


def _token(priv: Any, *, sub: UUID, customer_id: str, holder_key: Ed25519PrivateKey) -> str:
    now = int(time.time())
    claims = IdentityClaims(
        sub=str(sub),
        customer_id=customer_id,
        user_id=None,
        sid="sid-1",
        pod_id="pod-1",
        iss=_ISS,
        iat=now,
        exp=now + 600,
        cnf=jwk_thumbprint(holder_key.public_key()),
    )
    return sign_identity_token(claims, signing_key=priv, kid=_KID)


def _authorizer_granting(pod_id: UUID, namespace_id: UUID) -> RbacEvaluatorAuthorizer:
    """the real authorizer over ONE ``ToolCaller`` row for the pod's platform group on one tool namespace."""
    group_id, role_id = uuid7(), uuid7()
    role = Role(id=role_id, name="ToolCaller", permissions={"tool": frozenset({"tool.call"})}, is_built_in=True)
    membership = GroupMembership(group_id=group_id, member_id=pod_id, member_type=MemberType.AGENT, customer_id=None)
    assignment = RoleAssignment(
        id=uuid7(),
        group_id=group_id,
        role_id=role_id,
        scope_type=ScopeType.NAMESPACE,
        scope_namespace_id=namespace_id,
        scope_namespace_type=None,
        scope_customer_id=None,
    )
    cache = AclCache(
        membership_loader=_FakeMembershipLoader({pod_id: (membership,)}),
        grant_loader=_FakeGrantLoader(
            assignments={group_id: (assignment,)},
            roles={role_id: role},
            groups={group_id: Group(id=group_id, name="tool-pod-access:pod", customer_id=None)},
        ),
    )
    rows = {build_tool_namespace_name(_TOOL, _VERSION): _StubToolNamespace(id=namespace_id)}
    return RbacEvaluatorAuthorizer(acl_cache=cache, namespace_collection=_FakeNamespaceCollection(rows))


async def _serve_fake_tool(nc: NatsClient, received: list[CallRequest]) -> Any:
    """answer the registry's forwarded call on the serving pod's internal subject, recording it.

    what the pod RECEIVES is where the verified identity is observable: the registry
    re-stamps the forwarded context off the token, and on the synchronous path passes
    the pod's own reply back to the caller verbatim.
    """

    async def respond(msg: IncomingMessage) -> None:
        assert msg.reply_subject is not None
        received.append(CallRequest.model_validate_json(msg.data))
        await nc.publish_reply(
            reply_subject=msg.reply_subject,
            message=ProxyCallResponse(success=True, content="echo", context=CallContext(correlation_id=uuid7())),
        )

    return await nc.subscribe(subject=Subjects.tools_internal(_SERVING_POD), cb=respond)


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
            # under the sync reply budget, so the registry answers on the reply inbox. a tool
            # declaring more rides the durable result stream on the registry-to-pod hop, which
            # this bus does not provision; that path has its own tests.
            timeout_seconds=5.0,
        )
    )
    return catalog


async def test_a_tool_pod_calls_a_tool_through_the_real_proxy_with_no_user(nats_container: str) -> None:
    set_default_namespace(_NS)
    priv, pub = generate_signing_keypair()
    jwks = build_jwks({_KID: pub})
    pod_id, namespace_id = uuid7(), uuid7()
    holder_key = Ed25519PrivateKey.generate()
    pod_token = _token(priv, sub=pod_id, customer_id=PLATFORM_CUSTOMER_SENTINEL, holder_key=holder_key)

    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=_NS, client_name="registry"
        ) as registry_nc,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=_NS, client_name="pod") as pod_nc,
    ):
        received: list[CallRequest] = []
        tool_sub = await _serve_fake_tool(registry_nc, received)
        proxy = CallProxy(
            await _catalog(),
            _authorizer_granting(pod_id, namespace_id),
            _StubReplayGuard(),
            limit_guard=AllowAllLimitGuard(),
            namespace=_NS,
            jwks_provider=lambda: jwks,
        )
        await proxy.start(registry_nc)
        await asyncio.sleep(0.2)  # let both subscriptions register
        try:
            granted = ToolCallClient(
                pod_nc, principal_id=pod_id, identity_token=lambda: pod_token, pop_signer=_Signer(holder_key)
            )

            # the admission: a pod with a ToolCaller row, no user, a bound token and a good proof.
            reply = await granted.call(_TOOL, _VERSION, {"text": "hi"})
            assert reply.success is True
            assert reply.content == "echo"
            assert len(received) == 1
            forwarded = received[0].context
            assert forwarded is not None
            assert forwarded.agent_id == pod_id  # the VERIFIED principal reached the pod
            assert forwarded.customer_id is None  # the sentinel was read as no customer
            assert forwarded.user_id is None
            assert received[0].arguments == {"text": "hi"}

            # the same pod asking for a tool its grant does not cover is refused at the authorizer.
            with pytest.raises(ToolCallError) as excinfo:
                await granted.call("probe.other", _VERSION, {})
            assert excinfo.value.error_code == "TOOL_NOT_AUTHORIZED"

            # a proof minted for another body is refused at the pop gate, before any authorizer.
            spliced = ToolCallClient(
                pod_nc,
                principal_id=pod_id,
                identity_token=lambda: pod_token,
                pop_signer=_Signer(holder_key, body_override={"text": "not this body"}),
            )
            with pytest.raises(ToolCallError) as excinfo:
                await spliced.call(_TOOL, _VERSION, {"text": "hi"})
            assert excinfo.value.error_code == "TOOL_POP_UNVERIFIED"

            # a pod holding NO grant: the mark buys evaluation, never authority.
            ungranted_pod = uuid7()
            ungranted_token = _token(
                priv, sub=ungranted_pod, customer_id=PLATFORM_CUSTOMER_SENTINEL, holder_key=holder_key
            )
            ungranted = ToolCallClient(
                pod_nc,
                principal_id=ungranted_pod,
                identity_token=lambda: ungranted_token,
                pop_signer=_Signer(holder_key),
            )
            with pytest.raises(ToolCallError) as excinfo:
                await ungranted.call(_TOOL, _VERSION, {"text": "hi"})
            assert excinfo.value.error_code == "TOOL_NOT_AUTHORIZED"

            # an AGENT principal (a customer UUID on the token) with no user is refused exactly as
            # before, even holding the very same row: the two-sided rule is untouched for agents.
            agent_token = _token(priv, sub=pod_id, customer_id=str(uuid7()), holder_key=holder_key)
            agent = ToolCallClient(
                pod_nc, principal_id=pod_id, identity_token=lambda: agent_token, pop_signer=_Signer(holder_key)
            )
            with pytest.raises(ToolCallError) as excinfo:
                await agent.call(_TOOL, _VERSION, {"text": "hi"})
            assert excinfo.value.error_code == "TOOL_NOT_AUTHORIZED"
        finally:
            await proxy.stop()
            await registry_nc.unsubscribe(tool_sub)
