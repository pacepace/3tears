"""a delegated agent call's namespaces are asked with the call's own credential, and never widen.

A tool pod that proves delegation is sent the call's delegated token in place of the agent's token
and the person's assertion. :meth:`CallerNamespaces.names_for` asks discovery with that credential;
the hub stand-in here verifies it as the door does and answers the person's grants narrowed by the
agent's, so the test holds the pod to forwarding a credential that names both.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    CallerAccessCache,
    CallerNamespaces,
    CallerNamespacesUnavailable,
    bind_caller_cache_to_access_tables,
)
from threetears.core.collections import CollectionRegistry
from threetears.core.security import (
    DoorCredential,
    RegistrationKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    mint_delegated_token,
    verify_door_request,
)
from threetears.core.security.delegated_token import NAMESPACE_DISCOVER_DOOR
from threetears.core.security.door_request import NAMESPACE_DISCOVER_OP

_REGISTRY, _REGISTRY_PUB = generate_signing_keypair()
_JWKS = build_jwks({"registry-1": _REGISTRY_PUB})
_POD, _POD_PUB = generate_signing_keypair()

_EVERY = frozenset(f"tools.enr.state.{code}.1-0" for code in ("va", "pa", "tx", "tn"))
_VA = "tools.enr.state.va.1-0"
_PA = "tools.enr.state.pa.1-0"


class _Item:
    def __init__(self, name: str) -> None:
        self.name = name


class _Hub:
    """namespace.discover as the hub answers it: the caller read off the proven credential, the person's
    grants narrowed by the agent's. A request with no proven credential is refused."""

    def __init__(self, grants: dict[str, frozenset[str]]) -> None:
        self.grants = grants
        self.asked = 0

    async def discover(
        self,
        *,
        correlation_id: UUID,
        identity_token: str | None = None,
        user_identity_token: str | None = None,
        namespace_type: Any = None,
        credential: DoorCredential | None = None,
    ) -> list[_Item]:
        self.asked += 1
        if credential is None or not identity_token:
            raise PermissionError("namespace.discover: no proven credential")
        body = {
            "identity_token": identity_token,
            "correlation_id": f"{correlation_id}",
            "namespace_type": namespace_type,
        }
        caller = verify_door_request(
            credential.present(body, door=NAMESPACE_DISCOVER_DOOR, op=NAMESPACE_DISCOVER_OP),
            jwks=_JWKS,
            hub_issuer="hub",
            door=NAMESPACE_DISCOVER_DOOR,
            op=NAMESPACE_DISCOVER_OP,
        )
        visible = self.grants.get(caller.agent_id, frozenset())
        if caller.user_id is not None:
            visible = visible & self.grants.get(caller.user_id, frozenset())
        return [_Item(name) for name in sorted(visible)]


def _delegated(agent: UUID, user: UUID) -> DoorCredential:
    """the call's credential as the tool server builds it: the registry's delegation, proven with the pod's key."""
    now = int(time.time())
    token = mint_delegated_token(
        signing_key=_REGISTRY,
        kid="registry-1",
        agent_id=f"{agent}",
        pod_id="pod-1",
        customer_id=f"{uuid4()}",
        user_id=f"{user}",
        doors=(NAMESPACE_DISCOVER_DOOR,),
        holder_jkt=jwk_thumbprint(_POD_PUB),
        correlation_id=f"{uuid4()}",
        deadline=now + 60,
        now=now,
    )
    credential = DoorCredential.for_tool_call(delegated_token=token, identity_token=None, key=RegistrationKey(_POD))
    assert credential is not None and credential.proves
    return credential


def _names(hub: _Hub) -> tuple[CallerNamespaces, CallerAccessCache[frozenset[str]]]:
    cache: CallerAccessCache[frozenset[str]] = CallerAccessCache()
    cache.followed_by(lambda: True)
    bind_caller_cache_to_access_tables(CollectionRegistry(), cache)
    return CallerNamespaces(lambda: hub, cache), cache


async def test_a_delegated_agent_call_resolves_to_the_persons_states_narrowed_by_the_agents() -> None:
    agent, user = uuid4(), uuid4()
    # the agent is granted every state; the person Virginia only
    hub = _Hub({f"{agent}": _EVERY, f"{user}": frozenset({_VA})})
    names, _cache = _names(hub)
    listed = await names.names_for(
        agent_id=agent,
        user_id=user,
        identity_token=None,
        user_identity_token=None,
        credential=_delegated(agent, user),
    )
    assert listed == frozenset({_VA})


async def test_a_state_outside_the_narrowing_is_refused() -> None:
    agent, user = uuid4(), uuid4()
    # the person may read Virginia and Pennsylvania; the agent Virginia only
    hub = _Hub({f"{agent}": frozenset({_VA}), f"{user}": frozenset({_VA, _PA})})
    names, _cache = _names(hub)
    listed = await names.names_for(
        agent_id=agent,
        user_id=user,
        identity_token=None,
        user_identity_token=None,
        credential=_delegated(agent, user),
    )
    assert _PA not in listed
    assert listed == frozenset({_VA})


async def test_a_delegated_call_carrying_no_credential_is_refused_without_asking() -> None:
    agent, user = uuid4(), uuid4()
    hub = _Hub({f"{agent}": _EVERY, f"{user}": _EVERY})
    names, cache = _names(hub)
    with pytest.raises(CallerNamespacesUnavailable, match="no hub credential"):
        await names.names_for(
            agent_id=agent,
            user_id=user,
            identity_token=None,
            user_identity_token=None,
            credential=None,
        )
    assert hub.asked == 0
    assert cache.size == 0
