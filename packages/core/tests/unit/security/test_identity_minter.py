"""Unit tests for IdentityMinter: the mint↔verify contract, key loading, and fail-closed edges."""

from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.core.security import (
    IdentityMinter,
    IdentityTokenError,
    verify_identity_token,
)

_ISSUER = "test-principal"


def _pem(key: Ed25519PrivateKey | rsa.RSAPrivateKey) -> str:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def test_mint_verifies_against_its_own_jwks() -> None:
    """a minted token verifies against the minter's JWKS — the core self-mint / auth-callout contract."""
    minter = IdentityMinter.generate(kid="agent-1", issuer=_ISSUER, ttl_seconds=300)

    claims = verify_identity_token(
        minter.mint("agent-1", customer_id="cust-9"),
        jwks=minter.jwks(),
        issuer=_ISSUER,
    )

    assert claims.sub == "agent-1"
    assert claims.customer_id == "cust-9"
    assert claims.pod_id == "agent-1"  # defaults to the subject
    assert claims.iss == _ISSUER


def test_pod_id_defaults_to_subject_but_can_differ() -> None:
    """pod_id rides the subject by default, but a distinct pod id is carried when passed."""
    minter = IdentityMinter.generate(kid="agent-1", issuer=_ISSUER)

    claims = verify_identity_token(
        minter.mint("agent-1", customer_id="c", pod_id="pod-x"),
        jwks=minter.jwks(),
        issuer=_ISSUER,
    )

    assert claims.pod_id == "pod-x"


def test_from_pem_round_trips() -> None:
    """a minter loaded from a PKCS#8 PEM mints tokens that verify against its published JWKS."""
    minter = IdentityMinter.from_pem(_pem(Ed25519PrivateKey.generate()), kid="k1", issuer=_ISSUER)

    claims = verify_identity_token(
        minter.mint("subj", customer_id="c"),
        jwks=minter.jwks(),
        issuer=_ISSUER,
    )

    assert claims.sub == "subj"


def test_from_pem_rejects_non_ed25519_key() -> None:
    """FAIL CLOSED: an RSA PEM is not a valid identity signing key."""
    with pytest.raises(IdentityTokenError):
        IdentityMinter.from_pem(
            _pem(rsa.generate_private_key(public_exponent=65537, key_size=2048)),
            kid="k1",
            issuer=_ISSUER,
        )


def test_from_pem_rejects_junk() -> None:
    """FAIL CLOSED: a non-PEM string raises rather than yielding a minter on a junk key."""
    with pytest.raises(IdentityTokenError):
        IdentityMinter.from_pem(
            "-----BEGIN PRIVATE KEY-----\nnot base64\n-----END PRIVATE KEY-----", kid="k1", issuer=_ISSUER
        )


def test_nonpositive_ttl_rejected() -> None:
    """a non-positive TTL is a misconfiguration, not a mint on a zero-lifetime token."""
    with pytest.raises(IdentityTokenError):
        IdentityMinter.generate(kid="k1", issuer=_ISSUER, ttl_seconds=0)


def test_issuer_is_pinned_on_verify() -> None:
    """the resolver pins the issuer; a token minted under one issuer fails verification under another."""
    minter = IdentityMinter.generate(kid="k1", issuer=_ISSUER)
    token = minter.mint("subj", customer_id="c")

    with pytest.raises(IdentityTokenError):
        verify_identity_token(token, jwks=minter.jwks(), issuer="a-different-issuer")


def test_fresh_sid_per_mint() -> None:
    """each mint carries a fresh session id — two tokens from one minter differ in sid."""
    minter = IdentityMinter.generate(kid="k1", issuer=_ISSUER)

    first = verify_identity_token(minter.mint("s", customer_id="c"), jwks=minter.jwks(), issuer=_ISSUER)
    second = verify_identity_token(minter.mint("s", customer_id="c"), jwks=minter.jwks(), issuer=_ISSUER)

    assert first.sid != second.sid


def test_expired_token_is_rejected() -> None:
    """a token whose exp is in the past fails verification (TTL is honored end-to-end)."""
    minter = IdentityMinter.generate(kid="k1", issuer=_ISSUER, ttl_seconds=1)
    stale = minter.mint("s", customer_id="c", now=int(time.time()) - 3600)  # exp ~1h ago

    with pytest.raises(IdentityTokenError):
        verify_identity_token(stale, jwks=minter.jwks(), issuer=_ISSUER)
