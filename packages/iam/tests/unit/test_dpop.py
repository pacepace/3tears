"""DPoP (RFC 9449) proof validation.

This module had no direct tests at all. Nothing under ``packages/iam/tests`` imported it,
and its only real exercise was a downstream service's integration tier -- so every rejection
path in a security module that the JWT alg-pinning gate explicitly points at was unverified
in the package that owns it.

What is asserted here is one case per way a proof can be refused, plus the two properties
the module's docstring commits to and which are invisible from the outside: the algorithm and
curve are pinned rather than read off the proof, and the single-use ``jti`` check runs LAST
so an otherwise-invalid proof cannot burn a nonce.
"""

from __future__ import annotations

import time
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ec import (
    SECP256K1,
    SECP256R1,
    SECP384R1,
    EllipticCurvePrivateKey,
    generate_private_key,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jwt.algorithms import ECAlgorithm

from threetears.core.coordination import ReplayGuard
from threetears.core.security.identity_token import jwk_thumbprint
from threetears.core.testing.kv import FakeNatsClient
from threetears.iam.dpop import DEFAULT_IAT_WINDOW, DpopError, validate_dpop_proof

_HTM = "POST"
_HTU = "https://issuer.example/v1/token"


@pytest.fixture
def guard() -> ReplayGuard:
    """A real ReplayGuard over the shipped in-memory KV double.

    No cast: the guard names ``KvCapable``, which the double satisfies by construction.
    """
    return ReplayGuard(FakeNatsClient(), bucket_name="dpop-nonces", ttl_seconds=300)


def _key() -> EllipticCurvePrivateKey:
    return generate_private_key(SECP256R1())


def _public_jwk(key: Any) -> dict[str, Any]:
    jwk: dict[str, Any] = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
    return jwk


def _proof(
    key: EllipticCurvePrivateKey,
    *,
    htm: str = _HTM,
    htu: str = _HTU,
    jti: str | None = "jti-1",
    iat: int | None = None,
    typ: str = "dpop+jwt",
    jwk: dict[str, Any] | None = None,
    omit: tuple[str, ...] = (),
    algorithm: str = "ES256",
    signing_key: Any = None,
) -> str:
    """Mint a proof. Every part is overridable so a test can build one that must be refused."""
    payload: dict[str, Any] = {"htm": htm, "htu": htu, "jti": jti, "iat": iat if iat is not None else int(time.time())}
    for claim in omit:
        payload.pop(claim, None)
    headers: dict[str, Any] = {"typ": typ, "jwk": jwk if jwk is not None else _public_jwk(key)}
    return pyjwt.encode(payload, key=signing_key or key, algorithm=algorithm, headers=headers)


# -- the happy path ------------------------------------------------------------------------


class TestValidProof:
    async def test_a_well_formed_proof_yields_the_holder_thumbprint(self, guard: ReplayGuard) -> None:
        key = _key()
        result = await validate_dpop_proof(_proof(key), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)
        # The same thumbprint every other holder-key binding in the platform computes, which
        # is what makes it comparable against a token's `cnf.jkt`.
        assert result.jkt == jwk_thumbprint(key.public_key())

    async def test_several_acceptable_htus_are_an_allow_list_not_a_relaxation(self, guard: ReplayGuard) -> None:
        key = _key()
        other = "https://other.example/v1/token"
        result = await validate_dpop_proof(
            _proof(key, htu=other), expected_htm=_HTM, expected_htu=[_HTU, other], replay_guard=guard
        )
        assert result.jkt

    async def test_an_htu_outside_the_allow_list_is_still_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        with pytest.raises(DpopError, match="htu"):
            await validate_dpop_proof(
                _proof(key, htu="https://evil.example/v1/token"),
                expected_htm=_HTM,
                expected_htu=[_HTU, "https://other.example/v1/token"],
                replay_guard=guard,
            )

    async def test_an_htu_is_matched_exactly_never_as_a_prefix(self, guard: ReplayGuard) -> None:
        # A prefix match would accept https://issuer.example/v1/token.evil.com.
        key = _key()
        with pytest.raises(DpopError, match="htu"):
            await validate_dpop_proof(
                _proof(key, htu=f"{_HTU}/extra"), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
            )


# -- the header pins ------------------------------------------------------------------------


class TestHeaderPinning:
    async def test_a_malformed_proof_is_refused(self, guard: ReplayGuard) -> None:
        with pytest.raises(DpopError, match="malformed"):
            await validate_dpop_proof("not-a-jws", expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_the_wrong_typ_is_refused(self, guard: ReplayGuard) -> None:
        # A plain `JWT` typ would let an ordinary access token be replayed as a proof.
        key = _key()
        with pytest.raises(DpopError, match="typ"):
            await validate_dpop_proof(_proof(key, typ="JWT"), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_a_non_es256_algorithm_is_refused_before_any_key_is_selected(self, guard: ReplayGuard) -> None:
        """THE pin. The proof does not get to choose the algorithm that verifies it."""
        ed = Ed25519PrivateKey.generate()
        proof = pyjwt.encode(
            {"htm": _HTM, "htu": _HTU, "jti": "j", "iat": int(time.time())},
            key=ed,
            algorithm="EdDSA",
            headers={"typ": "dpop+jwt", "jwk": {"kty": "OKP"}},
        )
        with pytest.raises(DpopError, match="only ES256"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_hs256_proof_is_refused(self, guard: ReplayGuard) -> None:
        # The asymmetric-to-symmetric confusion: a proof signed with a public key as an HMAC
        # secret verifies fine if the algorithm is read off the token.
        proof = pyjwt.encode(
            {"htm": _HTM, "htu": _HTU, "jti": "j", "iat": int(time.time())},
            key="a-shared-secret",
            algorithm="HS256",
            headers={"typ": "dpop+jwt", "jwk": {"kty": "oct"}},
        )
        with pytest.raises(DpopError, match="only ES256"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)


# -- the inline key -------------------------------------------------------------------------


class TestInlineHolderKey:
    async def test_a_missing_jwk_is_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        proof = pyjwt.encode(
            {"htm": _HTM, "htu": _HTU, "jti": "j", "iat": int(time.time())},
            key=key,
            algorithm="ES256",
            headers={"typ": "dpop+jwt"},
        )
        with pytest.raises(DpopError, match="missing an inline jwk"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_inline_private_key_is_refused(self, guard: ReplayGuard) -> None:
        """A legitimate proof carries public material only."""
        key = _key()
        jwk = _public_jwk(key)
        jwk["d"] = "cHJpdmF0ZQ"
        with pytest.raises(DpopError, match="private key material"):
            await validate_dpop_proof(_proof(key, jwk=jwk), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_unparseable_jwk_is_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        with pytest.raises(DpopError, match="invalid"):
            await validate_dpop_proof(
                _proof(key, jwk={"kty": "EC", "crv": "nonsense"}),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )

    @pytest.mark.parametrize("curve", [SECP384R1(), SECP256K1()])
    async def test_a_curve_other_than_p256_is_refused(self, guard: ReplayGuard, curve: Any) -> None:
        """PyJWK parses P-384 and secp256k1 happily; accepting one would let the proof choose
        its own security level.

        The proof is signed with a real P-256 key and ADVERTISES the other curve, because
        PyJWT refuses to sign ES256 with a non-P-256 key at all -- so the only way such a
        header reaches a verifier is a mismatch like this one. The curve check runs before
        signature verification, which is why it is the curve error that surfaces and not a
        signature failure.
        """
        signer = _key()
        advertised = generate_private_key(curve)
        with pytest.raises(DpopError, match="P-256"):
            await validate_dpop_proof(
                _proof(signer, jwk=_public_jwk(advertised)),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )

    async def test_a_proof_signed_by_a_different_key_than_it_advertises_is_refused(self, guard: ReplayGuard) -> None:
        # The whole point: the inline jwk must be the key that actually signed.
        advertised, actual = _key(), _key()
        with pytest.raises(DpopError, match="verification failed"):
            await validate_dpop_proof(
                _proof(actual, jwk=_public_jwk(advertised)),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )


# -- the bound claims -----------------------------------------------------------------------


class TestBoundClaims:
    @pytest.mark.parametrize("claim", ["jti", "htm", "htu", "iat"])
    async def test_every_required_claim_is_required(self, guard: ReplayGuard, claim: str) -> None:
        key = _key()
        with pytest.raises(DpopError):
            await validate_dpop_proof(
                _proof(key, omit=(claim,)), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
            )

    async def test_a_mismatched_htm_is_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        with pytest.raises(DpopError, match="htm"):
            await validate_dpop_proof(_proof(key, htm="GET"), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_empty_acceptable_htu_set_is_refused_rather_than_matching_anything(
        self, guard: ReplayGuard
    ) -> None:
        # Fail closed: an empty allow-list must deny, never wave everything through.
        key = _key()
        with pytest.raises(DpopError, match="no acceptable dpop htu"):
            await validate_dpop_proof(_proof(key), expected_htm=_HTM, expected_htu=[], replay_guard=guard)


# -- freshness ------------------------------------------------------------------------------


class TestFreshness:
    async def test_a_stale_iat_is_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        stale = int(time.time() - DEFAULT_IAT_WINDOW.total_seconds() - 30)
        with pytest.raises(DpopError, match="freshness"):
            await validate_dpop_proof(_proof(key, iat=stale), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_a_future_iat_is_refused_too(self, guard: ReplayGuard) -> None:
        """Two-sided deliberately: an unbounded future iat would let an attacker mint proofs
        today for use after a key rotation.

        The rejection is asserted, not its wording. PyJWT's own ``iat`` validation refuses an
        immature signature during decode, so it lands first and the module's own future-side
        check sits behind it as the backstop rather than being what fires. The proof is denied
        either way, which is the property; matching on the message would pin which of the two
        layers happened to get there first.
        """
        key = _key()
        future = int(time.time() + DEFAULT_IAT_WINDOW.total_seconds() + 30)
        with pytest.raises(DpopError):
            await validate_dpop_proof(_proof(key, iat=future), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_iat_inside_the_window_is_accepted(self, guard: ReplayGuard) -> None:
        key = _key()
        recent = int(time.time() - DEFAULT_IAT_WINDOW.total_seconds() + 5)
        assert await validate_dpop_proof(
            _proof(key, iat=recent), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
        )

    async def test_a_boolean_iat_is_not_an_integer(self, guard: ReplayGuard) -> None:
        # bool is an int subclass; a boolean timestamp is malformed, not zero-or-one.
        key = _key()
        proof = pyjwt.encode(
            {"htm": _HTM, "htu": _HTU, "jti": "j", "iat": True},
            key=key,
            algorithm="ES256",
            headers={"typ": "dpop+jwt", "jwk": _public_jwk(key)},
        )
        with pytest.raises(DpopError, match="iat"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)


# -- single use -----------------------------------------------------------------------------


class TestSingleUse:
    async def test_a_replayed_jti_is_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        proof = _proof(key, jti="replay-me")
        assert await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)
        with pytest.raises(DpopError, match="replay"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_empty_jti_is_refused(self, guard: ReplayGuard) -> None:
        key = _key()
        with pytest.raises(DpopError, match="jti"):
            await validate_dpop_proof(_proof(key, jti=""), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_a_failing_proof_does_not_burn_its_nonce(self, guard: ReplayGuard) -> None:
        """THE ordering property the module docstring commits to.

        The jti check is last and is the only consuming step. If an earlier failure consumed
        it, anyone who could induce one -- a wrong htu, a stale clock -- could burn a
        legitimate client's nonce and deny it the retry.
        """
        key = _key()
        jti = "not-yet-spent"
        with pytest.raises(DpopError, match="htm"):
            await validate_dpop_proof(
                _proof(key, jti=jti, htm="GET"), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
            )
        # The same jti still works, which it would not if the failure above had spent it.
        assert await validate_dpop_proof(_proof(key, jti=jti), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)
