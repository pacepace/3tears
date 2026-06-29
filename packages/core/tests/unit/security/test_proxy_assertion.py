"""tests for the proxy->pod assertion binding (platform-auth Option B)."""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest

from threetears.core.security.identity_token import (
    IdentityTokenError,
    build_jwks,
    generate_signing_keypair,
)
from threetears.core.security.proxy_assertion import mint_proxy_assertion, verify_proxy_assertion


def _mint(priv, *, kid="proxy-1", pod_id="pod-1", body_hash="bh-1", nonce="n-1", iat=None, exp=None):
    now = int(time.time())
    return mint_proxy_assertion(
        signing_key=priv,
        kid=kid,
        pod_id=pod_id,
        agent_id="agent-1",
        customer_id="cust-1",
        body_hash=body_hash,
        nonce=nonce,
        iat=iat if iat is not None else now,
        exp=exp if exp is not None else now + 30,
        user_id="user-1",
    )


class TestProxyAssertion:
    """an assertion verifies ONLY under the proxy key in the JWKS, for THIS pod + THIS call."""

    def test_round_trips_and_carries_verified_identity(self) -> None:
        priv, pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": pub})
        claims = verify_proxy_assertion(_mint(priv), jwks=jwks, expected_pod_id="pod-1", body_hash="bh-1")
        assert claims.sub == "agent-1"
        assert claims.customer_id == "cust-1"
        assert claims.pod_id == "pod-1"
        assert claims.body_hash == "bh-1"
        assert claims.jti == "n-1"
        assert claims.user_id == "user-1"

    def test_wrong_pod_audience_rejected(self) -> None:
        priv, pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": pub})
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(_mint(priv), jwks=jwks, expected_pod_id="OTHER-POD", body_hash="bh-1")

    def test_wrong_body_hash_rejected(self) -> None:
        priv, pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": pub})
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(_mint(priv), jwks=jwks, expected_pod_id="pod-1", body_hash="DIFFERENT")

    def test_expired_rejected(self) -> None:
        priv, pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": pub})
        now = int(time.time())
        expired = _mint(priv, iat=now - 120, exp=now - 60)
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(expired, jwks=jwks, expected_pod_id="pod-1", body_hash="bh-1")

    def test_signature_under_a_key_not_in_the_jwks_rejected(self) -> None:
        priv, _pub = generate_signing_keypair()
        _other_priv, other_pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": other_pub})  # jwks has a DIFFERENT key for kid proxy-1
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(_mint(priv), jwks=jwks, expected_pod_id="pod-1", body_hash="bh-1")

    def test_unknown_kid_rejected(self) -> None:
        priv, pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": pub})
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(
                _mint(priv, kid="proxy-UNKNOWN"), jwks=jwks, expected_pod_id="pod-1", body_hash="bh-1"
            )

    def test_non_eddsa_alg_rejected(self) -> None:
        _priv, pub = generate_signing_keypair()
        jwks = build_jwks({"proxy-1": pub})
        now = int(time.time())
        forged = pyjwt.encode(
            {
                "iss": "registry",
                "aud": "pod-1",
                "sub": "a",
                "customer_id": "c",
                "bh": "bh-1",
                "jti": "j",
                "iat": now,
                "exp": now + 30,
            },
            key="secret",
            algorithm="HS256",
            headers={"kid": "proxy-1"},
        )
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(forged, jwks=jwks, expected_pod_id="pod-1", body_hash="bh-1")
