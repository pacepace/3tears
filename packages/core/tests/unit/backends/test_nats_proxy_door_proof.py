"""an L3 request from a principal holding its key proves its token over the request body."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.security import (
    DOOR_PROOF_FIELD,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims

_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_AGENT, _AGENT_PUB = generate_signing_keypair()


def _token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
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


def _backend(holder_key: object) -> tuple[NatsProxyL3Backend, AsyncMock]:
    nc = MagicMock()
    nc.request_raw = AsyncMock(return_value=json.dumps({"success": True, "rows": []}).encode())
    token = _token()
    backend = NatsProxyL3Backend(
        nats_client=nc,
        namespace_prefix="test",
        agent_id="agent-1",
        identity_token=lambda: token,
        holder_key=holder_key,  # type: ignore[arg-type]
    )
    return backend, nc.request_raw


async def test_a_request_from_a_key_holder_verifies_at_the_door() -> None:
    backend, request_raw = _backend(lambda: _AGENT)
    await backend.fetch("SELECT $1, $2", datetime(2026, 10, 8, tzinfo=UTC), {"nested": [1.5, "x"]})
    sent = json.loads(request_raw.call_args.kwargs["payload"])
    caller = verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3")
    assert caller.agent_id == "agent-1"


async def test_a_backend_without_a_key_sends_the_token_alone() -> None:
    backend, request_raw = _backend(None)
    await backend.fetch("SELECT 1")
    assert DOOR_PROOF_FIELD not in json.loads(request_raw.call_args.kwargs["payload"])


async def test_a_key_provider_answering_none_sends_the_token_alone() -> None:
    backend, request_raw = _backend(lambda: None)
    await backend.fetch("SELECT 1")
    assert DOOR_PROOF_FIELD not in json.loads(request_raw.call_args.kwargs["payload"])
