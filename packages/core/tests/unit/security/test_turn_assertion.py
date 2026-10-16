"""a turn assertion verifies only from the hub, for this agent, within its hour."""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest

from threetears.core.security import (
    TURN_ASSERTION_LIFETIME,
    TurnAssertionError,
    build_jwks,
    generate_signing_keypair,
    mint_turn_assertion,
    new_turn_id,
    verify_turn_assertion,
)


def _mint(priv, **overrides):
    args = {
        "signing_key": priv,
        "kid": "hub-1",
        "issuer": "hub",
        "agent_id": "agent-1",
        "user_id": "user-1",
        "customer_id": "cust-1",
        "conversation_id": "conv-1",
        "turn_id": "turn-1",
    }
    args.update(overrides)
    return mint_turn_assertion(**args)


def test_round_trip_carries_whose_turn_it_is() -> None:
    priv, pub = generate_signing_keypair()
    claims = verify_turn_assertion(_mint(priv), jwks=build_jwks({"hub-1": pub}), issuer="hub", agent_id="agent-1")
    assert (claims.user_id, claims.customer_id, claims.conversation_id, claims.turn_id) == (
        "user-1",
        "cust-1",
        "conv-1",
        "turn-1",
    )


def test_another_agents_turn_is_refused() -> None:
    priv, pub = generate_signing_keypair()
    with pytest.raises(TurnAssertionError, match="another agent"):
        verify_turn_assertion(_mint(priv), jwks=build_jwks({"hub-1": pub}), issuer="hub", agent_id="agent-2")


def test_a_forged_turn_is_refused() -> None:
    forger, _ = generate_signing_keypair()
    _, pub = generate_signing_keypair()
    with pytest.raises(TurnAssertionError, match="does not verify"):
        verify_turn_assertion(_mint(forger), jwks=build_jwks({"hub-1": pub}), issuer="hub", agent_id="agent-1")


def test_an_identity_token_is_not_a_turn_assertion() -> None:
    priv, pub = generate_signing_keypair()
    now = int(time.time())
    token = pyjwt.encode(
        {"iss": "hub", "sub": "agent-1", "iat": now, "exp": now + 60},
        key=priv,
        algorithm="EdDSA",
        headers={"kid": "hub-1"},
    )
    with pytest.raises(TurnAssertionError):
        verify_turn_assertion(token, jwks=build_jwks({"hub-1": pub}), issuer="hub", agent_id="agent-1")


def test_an_expired_turn_is_refused() -> None:
    priv, pub = generate_signing_keypair()
    old = int(time.time()) - int(TURN_ASSERTION_LIFETIME.total_seconds()) - 60
    with pytest.raises(TurnAssertionError, match="ExpiredSignature"):
        verify_turn_assertion(_mint(priv, now=old), jwks=build_jwks({"hub-1": pub}), issuer="hub", agent_id="agent-1")


def test_an_empty_key_set_refuses() -> None:
    priv, _ = generate_signing_keypair()
    with pytest.raises(TurnAssertionError):
        verify_turn_assertion(_mint(priv), jwks={}, issuer="hub", agent_id="agent-1")


def test_turn_ids_are_one_random_subject_token() -> None:
    first, second = new_turn_id(), new_turn_id()
    assert first != second
    assert all(character not in first for character in ".*> ")
