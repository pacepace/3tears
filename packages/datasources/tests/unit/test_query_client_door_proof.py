"""a datasource query from a principal holding its key proves its token, freshly on every send."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from pydantic import BaseModel

from threetears.core.security import (
    DelegatedTokenError,
    PopKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims
from threetears.datasources import query_client
from threetears.datasources.query_client import DATASOURCE_BUSY, DatasourceQueryClient, DatasourceQueryResponse

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


class _BusyOnce:
    """answers busy once, then success, keeping every request as it would be on the wire."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def request(self, *, subject: object, message: BaseModel, response_type: object, timeout: object) -> Any:
        self.sent.append(json.loads(message.model_dump_json()))
        if len(self.sent) == 1:
            return DatasourceQueryResponse(success=False, error_code=DATASOURCE_BUSY, error_message="busy")
        return DatasourceQueryResponse(success=True, rows=[], row_count=0, correlation_id=message.correlation_id)  # type: ignore[attr-defined]


async def test_each_send_carries_its_own_verifiable_proof(monkeypatch: Any) -> None:
    monkeypatch.setattr(query_client, "BUSY_BACKOFF_SECONDS", 0.0)
    token = _token()
    nats = _BusyOnce()
    client = DatasourceQueryClient(nats, identity_token=lambda: token, pop_key=lambda: PopKey(_AGENT))  # type: ignore[arg-type]
    await client.query("warehouse", "SELECT 1")
    proofs = [
        verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="datasource", op="warehouse.query").proof.jti
        for sent in nats.sent
    ]
    assert len(proofs) == 2
    assert proofs[0] != proofs[1]
    # bound to the datasource, read off the subject: a proof for one is refused at another
    with pytest.raises(DelegatedTokenError):
        verify_door_request(nats.sent[0], jwks=_JWKS, hub_issuer="hub", door="datasource", op="other.query")


async def test_a_client_without_a_key_sends_the_token_alone() -> None:
    nats = _BusyOnce()
    nats.sent.append({})  # the next answer is the success
    client = DatasourceQueryClient(nats, identity_token=lambda: "bearer")  # type: ignore[arg-type]
    await client.query("warehouse", "SELECT 1")
    assert nats.sent[1]["identity_token"] == "bearer"
    assert "identity_proof" not in nats.sent[1]
