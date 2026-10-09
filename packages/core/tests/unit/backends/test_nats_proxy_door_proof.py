"""an L3 request from a principal holding its key proves its token over the request, its subject's operation included."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.security import (
    DOOR_PROOF_FIELD,
    DelegatedTokenError,
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
_AGENT, _AGENT_PUB = generate_signing_keypair()
_TX_ID = "01999999-0000-7000-8000-000000000001"


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


def _backend(holder_key: object, *replies: dict[str, object]) -> tuple[NatsProxyL3Backend, AsyncMock]:
    nc = MagicMock()
    answers = [json.dumps(reply).encode() for reply in replies] or [json.dumps({"success": True, "rows": []}).encode()]
    nc.request_raw = AsyncMock(side_effect=answers if len(answers) > 1 else None, return_value=answers[0])
    token = _token()
    backend = NatsProxyL3Backend(
        nats_client=nc,
        namespace_prefix="test",
        agent_id="agent-1",
        identity_token=lambda: token,
        pop_key=holder_key,  # type: ignore[arg-type]
    )
    return backend, nc.request_raw


def _sent(request_raw: AsyncMock, index: int = -1) -> tuple[str, dict[str, object]]:
    call = request_raw.await_args_list[index]
    return call.kwargs["subject"].path, json.loads(call.kwargs["payload"])


async def test_a_request_with_nested_non_json_values_verifies_at_the_door() -> None:
    backend, request_raw = _backend(lambda: PopKey(_AGENT))
    nested = {
        "at": datetime(2026, 10, 8, 12, 30, tzinfo=UTC),
        "id": UUID("01999999-0000-7000-8000-0000000000aa"),
        "amount": Decimal("12.50"),
        "list": [datetime(2026, 1, 1, tzinfo=UTC), Decimal("1")],
    }
    await backend.fetch("SELECT $1, $2", datetime(2026, 10, 8, tzinfo=UTC), nested)
    subject, sent = _sent(request_raw)
    assert subject == "test.l3.query"
    caller = verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")
    assert caller.agent_id == "agent-1"


async def test_a_commit_and_a_rollback_are_proven_for_their_own_operation() -> None:
    backend, request_raw = _backend(
        lambda: PopKey(_AGENT),
        {"success": True, "tx_id": _TX_ID},
        {"success": True},
        {"success": True, "tx_id": _TX_ID},
        {"success": True},
    )
    async with backend.acquire() as conn, conn.transaction():
        pass
    with pytest.raises(RuntimeError):
        async with backend.acquire() as conn, conn.transaction():
            raise RuntimeError("roll it back")
    commit_subject, commit = _sent(request_raw, 1)
    rollback_subject, rollback = _sent(request_raw, 3)
    assert (commit_subject, rollback_subject) == ("test.l3.tx.commit", "test.l3.tx.rollback")
    verify_door_request(commit, jwks=_JWKS, hub_issuer="hub", door="l3", op="tx.commit")
    verify_door_request(rollback, jwks=_JWKS, hub_issuer="hub", door="l3", op="tx.rollback")
    with pytest.raises(DelegatedTokenError):
        verify_door_request(rollback, jwks=_JWKS, hub_issuer="hub", door="l3", op="tx.commit")


async def test_a_backend_without_a_key_sends_the_token_alone() -> None:
    backend, request_raw = _backend(None)
    await backend.fetch("SELECT 1")
    assert DOOR_PROOF_FIELD not in _sent(request_raw)[1]


async def test_a_key_the_token_does_not_bind_sends_the_token_alone() -> None:
    thief, _ = generate_signing_keypair()
    for provider in (lambda: None, lambda: PopKey(thief)):
        backend, request_raw = _backend(provider)
        await backend.fetch("SELECT 1")
        assert DOOR_PROOF_FIELD not in _sent(request_raw)[1]


def test_a_second_backend_borrows_the_key_provider() -> None:
    key = PopKey(_AGENT)
    backend, _ = _backend(lambda: key)
    assert backend.current_pop_key() is key
    assert _backend(None)[0].current_pop_key() is None


async def test_an_unproven_backend_says_so_once(caplog: pytest.LogCaptureFixture) -> None:
    """a backend sending its token alone logs it once, not on every request."""
    backend, _request_raw = _backend(None)
    with caplog.at_level("WARNING"):
        await backend.fetch("SELECT 1")
        await backend.fetch("SELECT 2")
    assert sum("without the token's proof" in record.getMessage() for record in caplog.records) == 1
