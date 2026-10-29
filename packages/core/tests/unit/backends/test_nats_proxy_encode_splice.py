"""a proven L3 request's wire bytes are exactly what encoding the whole presented body gave, the proof spliced on.

``encode_request`` used to encode a request, parse it back, present it and encode the presented body
again. It now appends the proof to the first encoding instead of encoding the body a second time. These
pin the bytes: for every body here, the new path answers what the old one did, byte for byte.
"""

from __future__ import annotations

import json
import random
import secrets
import string
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.schema_sql import json_default, transport_param
from threetears.core.security import (
    DOOR_PROOF_FIELD,
    DoorCredential,
    PopKey,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
    build_jwks,
)
from threetears.core.security.identity_token import IdentityClaims

_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_AGENT, _AGENT_PUB = generate_signing_keypair()
_NOW = int(time.time())
_NONCE = "fixed-nonce-for-byte-comparison"


def _token() -> str:
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=_NOW,
            exp=_NOW + 600,
            cnf=jwk_thumbprint(_AGENT_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


_TOKEN = _token()


def _backend() -> NatsProxyL3Backend:
    return NatsProxyL3Backend(
        nats_client=MagicMock(),
        namespace_prefix="test",
        agent_id="agent-1",
        identity_token=lambda: _TOKEN,
        pop_key=lambda: PopKey(_AGENT),
    )


def _before(payload: dict[str, Any], op: str) -> bytes:
    """what ``encode_request`` answered before the splice: the presented body, encoded whole."""
    encoded = json.dumps(payload, default=json_default).encode("utf-8")
    presented = DoorCredential.own(_TOKEN, PopKey(_AGENT)).present(json.loads(encoded), door="l3", op=op)
    return json.dumps(presented, default=json_default).encode("utf-8")


def _page(rows: int) -> list[list[Any]]:
    rng = random.Random(1000)
    letters = string.ascii_letters + 'éü中 "\\\n\t'
    return [
        [
            transport_param(value)
            for value in (
                rng.randint(-(10**12), 10**12),
                "".join(rng.choices(letters, k=24)),
                rng.random() * 1000,
                None,
                rng.choice([True, False]),
                Decimal(f"{rng.randint(0, 99999)}.{rng.randint(0, 99):02d}"),
                datetime(2026, 10, 28, rng.randint(0, 23), tzinfo=UTC),
                UUID(int=rng.getrandbits(128)),
                {"nested": [1, 2.5, "x", None, {"k": float("nan")}], 3: "int key"},
                float("inf") if index % 97 == 0 else -0.0,
                2**70,
            )
        ]
        for index in range(rows)
    ]


def _bodies() -> list[tuple[str, str, dict[str, Any]]]:
    """(subject, operation, payload): a query, a write with nested values, a transaction, a 1,000-row copy."""
    return [
        (
            "test.l3.query",
            "query",
            {
                "correlation_id": "c-1",
                "identity_token": _TOKEN,
                "namespace": "ns",
                "operation": "select",
                "query": "SELECT 1",
                "params": [],
                "timeout_ms": 30000,
            },
        ),
        (
            "test.l3.query",
            "query",
            {
                "correlation_id": "c-2",
                "identity_token": _TOKEN,
                "namespace": "ns",
                "operation": "insert",
                "query": "INSERT INTO t VALUES ($1, $2, $3)",
                "params": [transport_param(v) for v in _page(1)[0][:3]] + [{"at": datetime(2026, 1, 1, tzinfo=UTC)}],
                "timeout_ms": 30000,
                "conv_customer_id": "01999999-0000-7000-8000-0000000000aa",
            },
        ),
        (
            "test.l3.tx.begin",
            "tx.begin",
            {"correlation_id": "c-3", "identity_token": _TOKEN, "namespace": "ns", "timeout_ms": 1},
        ),
        (
            "test.l3.tx.copy",
            "tx.copy",
            {
                "correlation_id": "c-4",
                "identity_token": _TOKEN,
                "tx_id": "01999999-0000-7000-8000-000000000001",
                "table": "lsf",
                "columns": [f"c{i}" for i in range(11)],
                "rows": _page(1000),
            },
        ),
    ]


@pytest.fixture(autouse=True)
def _fixed_proof_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    # a proof is deterministic (Ed25519) but for its nonce and issue time; fix both so two encodings compare
    monkeypatch.setattr(secrets, "token_urlsafe", lambda _n=None: _NONCE)
    monkeypatch.setattr(time, "time", lambda: float(_NOW))


@pytest.mark.parametrize(("subject", "op", "payload"), _bodies(), ids=["select", "insert", "tx-begin", "copy-1000"])
def test_the_wire_bytes_are_unchanged(subject: str, op: str, payload: dict[str, Any]) -> None:
    sent = _backend().encode_request(subject, payload)
    assert sent == _before(payload, op)
    assert DOOR_PROOF_FIELD in json.loads(sent)


@pytest.mark.parametrize(("subject", "op", "payload"), _bodies(), ids=["select", "insert", "tx-begin", "copy-1000"])
def test_the_spliced_request_verifies_at_the_door(subject: str, op: str, payload: dict[str, Any]) -> None:
    del subject
    sent = json.loads(_backend().encode_request(f"test.l3.{op}", payload))
    assert verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op=op).agent_id == "agent-1"


def test_a_body_already_carrying_a_proof_is_encoded_whole_as_before() -> None:
    payload = {"correlation_id": "c-5", "identity_token": _TOKEN, DOOR_PROOF_FIELD: "stale", "query": "SELECT 1"}
    assert _backend().encode_request("test.l3.query", payload) == _before(payload, "query")


def test_a_body_without_a_token_is_sent_as_encoded() -> None:
    payload = {"correlation_id": "c-6", "query": "SELECT 1"}
    assert _backend().encode_request("test.l3.query", payload) == json.dumps(payload).encode("utf-8")
