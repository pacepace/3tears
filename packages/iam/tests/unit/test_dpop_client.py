"""Signing a DPoP proof — the client half of `threetears.iam.dpop`.

The two halves are tested against each other rather than against a transcript: every
case here signs with :func:`sign_dpop_proof` and validates with
:func:`validate_dpop_proof`, so a change that breaks the wire format fails here
instead of in a downstream service's integration tier. That pairing is the reason
this module lives beside its validator.
"""

from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, EllipticCurvePrivateKey
from threetears.core.security.identity_token import jwk_thumbprint
from threetears.iam.dpop import DpopError, validate_dpop_proof
from threetears.iam.dpop_client import new_holder_key, sign_dpop_proof

_HTU = "https://edge.example/v1/token"
_HTM = "POST"


class _AcceptingReplayGuard:
    """Records every `jti` it is asked about and accepts each one once."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def record_unique(self, jti: str) -> bool:
        first = jti not in self.seen
        self.seen.append(jti)
        return first


class TestTheProofItSignsIsOneItsValidatorAccepts:
    async def test_a_freshly_signed_proof_validates(self) -> None:
        key = new_holder_key()
        proof = sign_dpop_proof(key, htm=_HTM, htu=_HTU)

        validated = await validate_dpop_proof(
            proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=_AcceptingReplayGuard()
        )

        assert validated.jkt == jwk_thumbprint(key.public_key())

    async def test_the_key_is_p256_because_the_validator_pins_that_curve(self) -> None:
        """The module's one hard commitment: any other curve produces a proof this
        platform refuses, so the generator cannot drift from the verifier."""
        key = new_holder_key()
        assert isinstance(key, EllipticCurvePrivateKey)
        assert isinstance(key.curve, SECP256R1)

    async def test_each_proof_carries_a_distinct_jti(self) -> None:
        """Proofs are single-use across the whole service, so a signer that repeated a
        `jti` would make its own second request unreplayable."""
        key = new_holder_key()
        guard = _AcceptingReplayGuard()

        await validate_dpop_proof(
            sign_dpop_proof(key, htm=_HTM, htu=_HTU), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
        )
        await validate_dpop_proof(
            sign_dpop_proof(key, htm=_HTM, htu=_HTU), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
        )

        assert len(set(guard.seen)) == 2

    async def test_one_key_signing_twice_keeps_one_thumbprint(self) -> None:
        """A client that keeps its key keeps its session: the `cnf` binding is the
        thumbprint, so two proofs from one key must present the same one."""
        key = new_holder_key()
        guard = _AcceptingReplayGuard()

        first = await validate_dpop_proof(
            sign_dpop_proof(key, htm=_HTM, htu=_HTU), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
        )
        second = await validate_dpop_proof(
            sign_dpop_proof(key, htm=_HTM, htu=_HTU), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
        )

        assert first.jkt == second.jkt


class TestTheOverridesExistSoARejectableProofCanBeBuilt:
    """Every parameter is overridable specifically so a caller can construct a proof
    that SHOULD be refused. Without these a negative test has to reimplement the wire
    format, which is how a test's idea of it drifts from production's."""

    async def test_a_proof_bound_to_another_url_is_refused(self) -> None:
        key = new_holder_key()
        proof = sign_dpop_proof(key, htm=_HTM, htu="https://edge.example/v1/token/refresh")

        with pytest.raises(DpopError):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=_AcceptingReplayGuard())

    async def test_a_proof_bound_to_another_method_is_refused(self) -> None:
        key = new_holder_key()
        proof = sign_dpop_proof(key, htm="GET", htu=_HTU)

        with pytest.raises(DpopError):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=_AcceptingReplayGuard())

    async def test_a_stale_proof_is_refused(self) -> None:
        key = new_holder_key()
        proof = sign_dpop_proof(key, htm=_HTM, htu=_HTU, iat=int(time.time()) - 3600)

        with pytest.raises(DpopError):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=_AcceptingReplayGuard())

    async def test_a_replayed_jti_is_refused(self) -> None:
        key = new_holder_key()
        guard = _AcceptingReplayGuard()
        fixed = "11111111-1111-7111-8111-111111111111"

        await validate_dpop_proof(
            sign_dpop_proof(key, htm=_HTM, htu=_HTU, jti=fixed),
            expected_htm=_HTM,
            expected_htu=_HTU,
            replay_guard=guard,
        )

        with pytest.raises(DpopError):
            await validate_dpop_proof(
                sign_dpop_proof(key, htm=_HTM, htu=_HTU, jti=fixed),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )
