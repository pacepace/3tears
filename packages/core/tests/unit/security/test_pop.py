"""tests for the proof-of-possession binding (agent->proxy hop, platform-auth Option B)."""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.core.security.identity_token import IdentityTokenError, jwk_thumbprint
from threetears.core.security.pop import access_token_hash, make_pop_proof, verify_pop_proof


def _proof(
    holder: Ed25519PrivateKey,
    *,
    ath: str = "ath-1",
    bh: str = "bh-1",
    nonce: str = "nonce-1",
    iat: int | None = None,
) -> str:
    return make_pop_proof(
        holder_key=holder,
        access_token_hash=ath,
        body_hash=bh,
        nonce=nonce,
        iat=iat if iat is not None else int(time.time()),
    )


class TestProofOfPossession:
    """a proof verifies ONLY under the bound holder key, for THIS token + THIS call, fresh."""

    def test_round_trips_and_returns_nonce(self) -> None:
        holder = Ed25519PrivateKey.generate()
        jkt = jwk_thumbprint(holder.public_key())
        proof = _proof(holder, ath="ath-x", bh="bh-y", nonce="n-1")
        assert verify_pop_proof(
            proof, expected_jkt=jkt, access_token_hash="ath-x", body_hash="bh-y"
        ) == "n-1"

    def test_wrong_holder_key_rejected(self) -> None:
        holder = Ed25519PrivateKey.generate()
        other_jkt = jwk_thumbprint(Ed25519PrivateKey.generate().public_key())
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(
                _proof(holder), expected_jkt=other_jkt, access_token_hash="ath-1", body_hash="bh-1"
            )

    def test_wrong_ath_rejected(self) -> None:
        holder = Ed25519PrivateKey.generate()
        jkt = jwk_thumbprint(holder.public_key())
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(
                _proof(holder, ath="real-ath"),
                expected_jkt=jkt,
                access_token_hash="DIFFERENT",
                body_hash="bh-1",
            )

    def test_wrong_body_hash_rejected(self) -> None:
        holder = Ed25519PrivateKey.generate()
        jkt = jwk_thumbprint(holder.public_key())
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(
                _proof(holder, bh="real-bh"),
                expected_jkt=jkt,
                access_token_hash="ath-1",
                body_hash="DIFFERENT",
            )

    def test_stale_iat_rejected(self) -> None:
        holder = Ed25519PrivateKey.generate()
        jkt = jwk_thumbprint(holder.public_key())
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(
                _proof(holder, iat=int(time.time()) - 3600),  # an hour old
                expected_jkt=jkt,
                access_token_hash="ath-1",
                body_hash="bh-1",
                leeway_seconds=60,
            )

    def test_non_eddsa_alg_rejected(self) -> None:
        # an HS256 proof must be rejected at the alg pin, before signature handling.
        forged = pyjwt.encode(
            {"ath": "a", "bh": "b", "jti": "j", "iat": int(time.time())},
            key="secret",
            algorithm="HS256",
            headers={"jwk": {}},
        )
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(forged, expected_jkt="x", access_token_hash="a", body_hash="b")

    def test_missing_inline_jwk_rejected(self) -> None:
        # an EdDSA proof with no inline jwk: nothing to bind to the cnf -> reject.
        holder = Ed25519PrivateKey.generate()
        forged = pyjwt.encode(
            {"ath": "a", "bh": "b", "jti": "j", "iat": int(time.time())},
            key=holder,
            algorithm="EdDSA",
        )
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(forged, expected_jkt="x", access_token_hash="a", body_hash="b")


class TestAccessTokenHash:
    """the shared ``ath`` both ends compute so a proof binds to its exact identity token."""

    def test_deterministic_and_url_safe(self) -> None:
        # same token -> same hash; url-safe base64 with padding stripped (no '=', '+', '/').
        digest = access_token_hash("header.payload.signature")
        assert digest == access_token_hash("header.payload.signature")
        assert not (set(digest) & set("=+/"))

    def test_distinct_tokens_differ(self) -> None:
        assert access_token_hash("token-a") != access_token_hash("token-b")

    def test_binds_a_proof_to_its_token(self) -> None:
        # a proof minted with one token's ath does not verify when presented with another token.
        holder = Ed25519PrivateKey.generate()
        jkt = jwk_thumbprint(holder.public_key())
        proof = _proof(holder, ath=access_token_hash("real-token"), bh="bh-1")
        assert verify_pop_proof(
            proof, expected_jkt=jkt, access_token_hash=access_token_hash("real-token"), body_hash="bh-1"
        )
        with pytest.raises(IdentityTokenError):
            verify_pop_proof(
                proof,
                expected_jkt=jkt,
                access_token_hash=access_token_hash("other-token"),
                body_hash="bh-1",
            )
