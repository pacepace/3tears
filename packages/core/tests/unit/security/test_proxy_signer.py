"""tests for ProxyAssertionSigner (the proxy's assertion-signing key, custody + mint + JWKS)."""

from __future__ import annotations

import base64
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from threetears.core.security.identity_token import IdentityTokenError
from threetears.core.security.proxy_assertion import verify_proxy_assertion
from threetears.core.security.proxy_signer import ProxyAssertionSigner


def _seed_secret(key: Ed25519PrivateKey) -> SecretStr:
    return SecretStr(base64.urlsafe_b64encode(key.private_bytes_raw()).decode("ascii"))


class TestProxyAssertionSigner:
    """the signer mints assertions its own published JWKS verifies; custody fails closed."""

    def test_mint_then_verify_against_own_jwks(self) -> None:
        signer = ProxyAssertionSigner.from_secret(_seed_secret(Ed25519PrivateKey.generate()))
        assertion = signer.mint(
            pod_id="pod-1",
            agent_id="agent-1",
            customer_id="cust-1",
            body_hash="bh-1",
            nonce="n-1",
            now=int(time.time()),
            user_id="user-1",
        )
        claims = verify_proxy_assertion(assertion, jwks=signer.public_jwks(), expected_pod_id="pod-1", body_hash="bh-1")
        assert claims.sub == "agent-1"
        assert claims.customer_id == "cust-1"
        assert claims.user_id == "user-1"
        assert claims.jti == "n-1"

    def test_public_jwks_carries_the_key_under_the_kid(self) -> None:
        signer = ProxyAssertionSigner.from_secret(_seed_secret(Ed25519PrivateKey.generate()))
        jwks = signer.public_jwks()
        assert len(jwks["keys"]) == 1
        assert jwks["keys"][0]["kid"] == signer.kid
        assert "d" not in jwks["keys"][0]  # public material only

    def test_from_secret_is_stable(self) -> None:
        key = Ed25519PrivateKey.generate()
        assert ProxyAssertionSigner.from_secret(_seed_secret(key)).kid == (
            ProxyAssertionSigner.from_secret(_seed_secret(key)).kid
        )

    def test_from_secret_rejects_junk_without_echoing_it(self) -> None:
        with pytest.raises(IdentityTokenError) as excinfo:
            ProxyAssertionSigner.from_secret(SecretStr("not-a-real-key-@@@"))
        assert "not-a-real-key" not in str(excinfo.value)

    def test_from_secret_rejects_wrong_length(self) -> None:
        short = SecretStr(base64.urlsafe_b64encode(b"too-short").decode("ascii"))
        with pytest.raises(IdentityTokenError):
            ProxyAssertionSigner.from_secret(short)

    def test_non_positive_ttl_rejected(self) -> None:
        with pytest.raises(IdentityTokenError):
            ProxyAssertionSigner(signing_key=Ed25519PrivateKey.generate(), ttl_seconds=0)
