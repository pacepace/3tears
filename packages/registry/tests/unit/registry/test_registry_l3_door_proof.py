"""the registry's L3 reads prove their token at the hub's ``l3`` door, where the host's identity holds the key.

The ``l3`` door takes a bearer token only until 2027-01-05; after that every request must carry a proof of
possession signed with the key the token binds (its ``cnf``). The registry's rbac pool is one such sender.
These pin:

- an identity that holds its key (``ProvingRegistryIdentity``) hands it to the server's
  ``pop_key_provider``, and one that does not hands nothing, so its requests stay bearer as before;
- the rbac stack built with that provider sends every L3 request with a proof the door verifies, for the
  operation the subject names;
- built without one, it sends the token alone, which today's hub accepts until its window closes.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.security import (
    DOOR_PROOF_FIELD,
    PopKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims
from threetears.core.testing.kv import FakeNatsClient
from threetears.registry.auth import AllowAllAuthorizer, ProvingRegistryIdentity
from threetears.registry.l1_cache import create_registry_l1_backend
from threetears.registry.rbac_stack import build_registry_rbac_stack
from threetears.registry.server import RegistryServer

_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_REGISTRY_KEY, _REGISTRY_PUB = generate_signing_keypair()


def _token() -> str:
    """a hub-minted identity token whose ``cnf`` binds the registry's key."""
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="registry",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="registry-pod",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_REGISTRY_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


# parity-with: threetears.registry.auth.ProvingRegistryIdentity
class _FakeProvingIdentity:
    """a host identity holding the key its token binds."""

    def __init__(self, token: str) -> None:
        self._token = token

    def token(self) -> str | None:
        return self._token

    def pop_key(self) -> PopKey | None:
        return PopKey(_REGISTRY_KEY)

    async def close(self) -> None:
        return None


# parity-with: threetears.registry.auth.RegistryIdentity
class _FakeBearerIdentity:
    """today's host identity: a token, no key."""

    def token(self) -> str | None:
        return "host.minted.token"

    async def close(self) -> None:
        return None


def _nats_client() -> MagicMock:
    """a client whose L3 requests are recorded and answered with no rows."""
    nc = MagicMock()
    nc.raw = MagicMock()
    nc.subscribe_typed = AsyncMock(side_effect=lambda **_kw: MagicMock())
    nc.unsubscribe = AsyncMock()
    fake = FakeNatsClient()
    nc.kv_bucket = fake.kv_bucket
    nc.request_raw = AsyncMock(return_value=json.dumps({"success": True, "rows": []}).encode())
    return nc


async def _server_with(identity: Any) -> RegistryServer:
    server = RegistryServer(
        namespace="testns", authorizer=AllowAllAuthorizer(), identity_factory=AsyncMock(return_value=identity)
    )
    await server.apply_identity_factory(MagicMock())
    return server


@pytest.mark.asyncio
async def test_an_identity_holding_its_key_hands_it_to_the_l3_senders() -> None:
    identity = _FakeProvingIdentity(_token())
    assert isinstance(identity, ProvingRegistryIdentity)
    provider = (await _server_with(identity)).pop_key_provider()
    assert provider is not None
    held = provider()
    assert held is not None and held.key is _REGISTRY_KEY


@pytest.mark.asyncio
async def test_an_identity_without_a_key_hands_none_and_stays_bearer() -> None:
    assert (await _server_with(_FakeBearerIdentity())).pop_key_provider() is None
    assert RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer()).pop_key_provider() is None


@pytest.mark.asyncio
async def test_the_rbac_pool_proves_its_token_at_the_l3_door() -> None:
    token = _token()
    nc = _nats_client()
    stack = build_registry_rbac_stack(
        nats_client=nc,
        subject_namespace="3tears",
        l1_backend=create_registry_l1_backend(),
        identity_token=lambda: token,
        pop_key=lambda: PopKey(_REGISTRY_KEY),
    )
    pool = stack.registry.get_l3_pool("namespaces")
    await pool.fetch("SELECT 1")
    call = nc.request_raw.await_args_list[-1]
    sent = json.loads(call.kwargs["payload"])
    assert DOOR_PROOF_FIELD in sent
    caller = verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")
    assert caller.agent_id == "registry"


@pytest.mark.asyncio
async def test_the_rbac_pool_without_a_key_sends_the_token_alone() -> None:
    nc = _nats_client()
    stack = build_registry_rbac_stack(
        nats_client=nc,
        subject_namespace="3tears",
        l1_backend=create_registry_l1_backend(),
        identity_token=lambda: "host.minted.token",
    )
    await stack.registry.get_l3_pool("namespaces").fetch("SELECT 1")
    sent = json.loads(nc.request_raw.await_args_list[-1].kwargs["payload"])
    assert DOOR_PROOF_FIELD not in sent
    assert sent["identity_token"] == "host.minted.token"
