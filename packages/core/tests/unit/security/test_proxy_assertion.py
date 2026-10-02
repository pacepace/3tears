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
            key="an-hmac-key-long-enough-for-hs256-x",
            algorithm="HS256",
            headers={"kid": "proxy-1"},
        )
        with pytest.raises(IdentityTokenError):
            verify_proxy_assertion(forged, jwks=jwks, expected_pod_id="pod-1", body_hash="bh-1")


class TestTheRefusalCodeIsSpelledOnce:
    """the code a pod answers a failed assertion with is exported from core, where the pod and
    the hub's error faces both read it, and its message names no check."""

    def test_the_code_and_message_are_exported_from_the_security_package(self) -> None:
        from threetears.core import security

        assert security.TOOL_PROXY_ASSERTION_UNVERIFIED == "TOOL_PROXY_ASSERTION_UNVERIFIED"
        assert "TOOL_PROXY_ASSERTION_UNVERIFIED" in security.__all__
        assert "TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE" in security.__all__

    def test_the_message_does_not_say_which_check_refused(self) -> None:
        from threetears.core.security import TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE

        for discriminator in ("absent", "missing", "replay", "nonce", "body", "guard", "kid", "Error"):
            assert discriminator not in TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE


class TestTheLedgerOutageCodeIsSpelledOnce:
    """the code a verifier answers when its replay ledger cannot be reached is exported from core,
    where the registry (the caller's proof) and the tool pod (the proxy's assertion) both read it."""

    def test_the_code_and_message_are_exported_from_the_security_package(self) -> None:
        from threetears.core import security

        assert security.TOOL_POP_LEDGER_UNAVAILABLE == "TOOL_POP_LEDGER_UNAVAILABLE"
        assert "TOOL_POP_LEDGER_UNAVAILABLE" in security.__all__
        assert "TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE" in security.__all__

    def test_the_message_names_no_exception(self) -> None:
        """the exception type and its text belong in the verifier's log, never in the reply."""
        from threetears.core.security import TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE

        for discriminator in ("Error", "Kv", "bucket", "nonce"):
            assert discriminator not in TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE
