"""an audit anonymization request from an agent holding its key proves its token over each batch."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

from threetears.core.security import (
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims

from threetears.agent.audit.erasure import request_audit_anonymization

_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_KEY, _KEY_PUB = generate_signing_keypair()
_AGENT = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")


def _token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub=f"{_AGENT}",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_KEY_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


class _Answering:
    """answers each batch with a success, keeping what was sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def request_raw(self, *, subject: object, payload: bytes, timeout: timedelta) -> bytes:
        body = json.loads(payload)
        self.sent.append(body)
        return json.dumps(
            {
                "success": True,
                "correlation_id": body["correlation_id"],
                "agent_id": f"{_AGENT}",
                "rows_matched": 1,
                "rows_changed": 1,
            }
        ).encode()


async def test_a_batch_is_proven_over_its_own_body() -> None:
    nats = _Answering()
    await request_audit_anonymization(
        nats,  # type: ignore[arg-type]
        identity_token=_token(),
        agent_id=_AGENT,
        actor_user_ids=[uuid7()],
        holder_key=_KEY,
    )
    caller = verify_door_request(nats.sent[0], jwks=_JWKS, hub_issuer="hub", door="hub.audit.anonymize")
    assert caller.agent_id == f"{_AGENT}"


async def test_a_caller_without_a_key_sends_the_token_alone() -> None:
    nats = _Answering()
    await request_audit_anonymization(nats, identity_token="bearer", agent_id=_AGENT, actor_user_ids=[uuid7()])  # type: ignore[arg-type]
    assert nats.sent[0]["identity_proof"] is None
