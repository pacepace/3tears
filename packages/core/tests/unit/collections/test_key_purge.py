"""a tool pod's collection keys purge proves its token at the hub's door, and the hub reads it tolerantly."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import ValidationError
from threetears.nats.collection_key_requests import CollectionKeysPurgeRequest as StrictPurgeRequest

from threetears.core.collections.key_purge import CollectionKeysPurgeRequest, purge_pod_collection_keys
from threetears.core.security import (
    PLATFORM_CUSTOMER_SENTINEL,
    PopKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims

_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_KEY, _KEY_PUB = generate_signing_keypair()
_POD = UUID("06a41d51-a6d5-7824-8000-29ab66754fc1")


def _token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub=f"{_POD}",
            customer_id=PLATFORM_CUSTOMER_SENTINEL,
            sid="sid-1",
            pod_id=f"{_POD}",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_KEY_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


class _Answering:
    """answers each purge with a success, keeping what was sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def request_raw(self, *, subject: object, payload: bytes, timeout: timedelta) -> bytes:
        body = json.loads(payload)
        self.sent.append(body)
        return json.dumps({"success": True, "correlation_id": body["correlation_id"], "purged": 1}).encode()


async def test_a_purge_is_proven_over_its_own_body() -> None:
    nats = _Answering()
    purged = await purge_pod_collection_keys(
        nats,  # type: ignore[arg-type]
        identity_token=_token(),
        keys=["enr_answers.v1_abc"],
        pop_key=PopKey(_KEY),
    )
    assert purged == 1
    caller = verify_door_request(
        nats.sent[0], jwks=_JWKS, hub_issuer="hub", door="hub.collection_keys.purge", op="purge"
    )
    assert caller.agent_id == f"{_POD}"


async def test_a_pod_without_a_key_sends_the_token_alone_and_an_older_hub_reads_it() -> None:
    nats = _Answering()
    await purge_pod_collection_keys(nats, identity_token="bearer", keys=["enr_answers.v1_abc"])  # type: ignore[arg-type]
    assert "identity_proof" not in nats.sent[0]
    # the request as a hub from before the door proofs reads it: every unknown field refused
    StrictPurgeRequest.model_validate(nats.sent[0])


def test_the_hub_ignores_a_field_a_newer_pod_adds() -> None:
    body = {"identity_token": "t", "correlation_id": f"{uuid7()}", "keys": ["a.b"], "a_later_field": 1}
    assert CollectionKeysPurgeRequest.model_validate(body).keys == ["a.b"]


@pytest.mark.parametrize("key", ["a.*", "a.>", "a..b", ""])
def test_a_key_that_is_not_literal_tokens_is_still_refused(key: str) -> None:
    with pytest.raises(ValidationError):
        CollectionKeysPurgeRequest.model_validate(
            {"identity_token": "t", "correlation_id": f"{uuid7()}", "keys": [key]}
        )
