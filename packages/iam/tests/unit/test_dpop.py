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

import inspect
import time
from datetime import UTC, datetime, timedelta
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
from freezegun import freeze_time
from jwt.algorithms import ECAlgorithm

from threetears.core.coordination import ReplayGuard
from threetears.core.security.identity_token import jwk_thumbprint
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.security import (
    CLIENT_ISSUE_TIME_FUTURE_TOLERANCE,
    DEFAULT_PROOF_MAX_AGE,
    ISSUE_TIME_FUTURE_TOLERANCE,
)
from threetears.iam.dpop import DpopError, validate_dpop_proof

_HTM = "POST"
_HTU = "https://issuer.example/v1/token"


def _frozen_clock() -> Any:
    """Stop the server's clock at this instant, for a test whose subject is an exact edge of a window.

    Frozen at NOW rather than at a fixed date, because the ``guard`` fixture's bucket is aged
    from the real clock and a proof stamped before that bucket's creation is refused as a
    possible replay. Frozen on a whole second, because ``iat`` is one: a clock stopped mid-second
    would put a proof's truncated issue time up to a second behind the moment it was minted, and
    an edge measured in whole seconds would land on the wrong side. The event loop keeps its
    real clock.

    :return: the freezer, a context manager whose value can ``tick`` the frozen clock forward
    :rtype: Any
    """
    return freeze_time(datetime.now(UTC).replace(microsecond=0), real_asyncio=True)


@pytest.fixture
def kv() -> FakeNatsClient:
    """The in-memory KV double the guard stores nonces in."""
    return FakeNatsClient()


@pytest.fixture
async def guard(kv: FakeNatsClient) -> ReplayGuard:
    """A real ReplayGuard over the shipped in-memory KV double, on a bucket that already existed.

    The bucket is aged an hour: a guard refuses proofs issued before its bucket was created, so a
    bucket born in the same instant as the proof would refuse every case here for a reason none
    of them is about. No cast: the guard names ``KvCapable``, which the double satisfies by
    construction.
    """
    bucket = await kv.kv_bucket(name="dpop-nonces")
    bucket.wipe(date_created=datetime.now(UTC) - timedelta(hours=1))
    return ReplayGuard(
        kv,
        bucket_name="dpop-nonces",
        ttl_seconds=300,
        verifier_future_tolerance=CLIENT_ISSUE_TIME_FUTURE_TOLERANCE,
    )


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
            key="an-hmac-key-long-enough-for-hs256-x",
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
        stale = int(time.time() - DEFAULT_PROOF_MAX_AGE.total_seconds() - 30)
        with pytest.raises(DpopError, match="freshness"):
            await validate_dpop_proof(_proof(key, iat=stale), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_a_future_iat_is_refused_too(self, guard: ReplayGuard) -> None:
        """Two-sided deliberately: an unbounded future iat would let an attacker mint proofs
        today for use after a key rotation.

        The message is matched, and that is the point of matching it. This module's window is
        the only thing adjudicating ``iat``, so "freshness" is the only wording a refusal can
        carry. Asserting merely that something was refused passed just as well when PyJWT's
        zero-leeway check got here first and refused EVERY future proof, one second included.
        """
        key = _key()
        future = int(time.time() + DEFAULT_PROOF_MAX_AGE.total_seconds() + 30)
        with pytest.raises(DpopError, match="freshness"):
            await validate_dpop_proof(_proof(key, iat=future), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_an_old_iat_inside_the_window_is_still_accepted(self, guard: ReplayGuard) -> None:
        """Separating the two directions must not shorten how long a slow request has to arrive."""
        key = _key()
        with _frozen_clock():
            for behind in (55, 60):
                assert await validate_dpop_proof(
                    _proof(key, jti=f"behind-{behind}", iat=int(time.time()) - behind),
                    expected_htm=_HTM,
                    expected_htu=_HTU,
                    replay_guard=guard,
                )
            with pytest.raises(DpopError, match="freshness"):
                await validate_dpop_proof(
                    _proof(key, jti="behind-61", iat=int(time.time()) - 61),
                    expected_htm=_HTM,
                    expected_htu=_HTU,
                    replay_guard=guard,
                )

    async def test_an_iat_ahead_of_the_server_is_accepted_up_to_the_client_tolerance(self, guard: ReplayGuard) -> None:
        """The admitted twin of the refusal below, and the case a real login actually hits.

        ``iat`` is required to be an integer, so a client whose clock leads the server's by a
        fraction of a second still stamps ``server_now + 1``. Refusing that makes whether a
        login succeeds depend on sub-second timing between minting the proof and receiving it.

        And the lead is not always a fraction of a second: the signer is a user's browser or a
        developer's laptop, whose clock the platform does not keep. So the client-signed proof
        keeps a full minute on the future side, where a pod-signed one gets five seconds. All
        of it is usable, to its last second. The clock is frozen so the edge is exact.
        """
        key = _key()
        with _frozen_clock():
            for ahead in (1, 2, 6, 55, 60):
                assert await validate_dpop_proof(
                    _proof(key, jti=f"ahead-{ahead}", iat=int(time.time()) + ahead),
                    expected_htm=_HTM,
                    expected_htu=_HTU,
                    replay_guard=guard,
                )

    async def test_an_iat_sixty_one_seconds_ahead_of_the_server_is_refused(self, guard: ReplayGuard) -> None:
        """The future side is bounded, by its own number, whatever the past side allows."""
        key = _key()
        with _frozen_clock():
            for ahead in (61, 90):
                with pytest.raises(DpopError, match="freshness"):
                    await validate_dpop_proof(
                        _proof(key, jti=f"ahead-{ahead}", iat=int(time.time()) + ahead),
                        expected_htm=_HTM,
                        expected_htu=_HTU,
                        replay_guard=guard,
                    )

    async def test_a_wide_past_window_does_not_widen_the_future(self, guard: ReplayGuard) -> None:
        key = _key()
        with _frozen_clock(), pytest.raises(DpopError, match="freshness"):
            await validate_dpop_proof(
                _proof(key, iat=int(time.time()) + 61),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
                max_age=timedelta(hours=1),
            )

    async def test_a_caller_may_choose_the_pod_tolerance_and_then_six_seconds_ahead_is_refused(
        self, guard: ReplayGuard
    ) -> None:
        # the future bound is the caller's to set, separately from the past one: a surface whose
        # signer is a platform host passes the small number, and gets the small window.
        key = _key()
        with _frozen_clock():
            assert await validate_dpop_proof(
                _proof(key, jti="ahead-4", iat=int(time.time()) + 4),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
                future_tolerance=ISSUE_TIME_FUTURE_TOLERANCE,
            )
            with pytest.raises(DpopError, match="freshness"):
                await validate_dpop_proof(
                    _proof(key, jti="ahead-6", iat=int(time.time()) + 6),
                    expected_htm=_HTM,
                    expected_htu=_HTU,
                    replay_guard=guard,
                    future_tolerance=ISSUE_TIME_FUTURE_TOLERANCE,
                )

    def test_the_default_future_tolerance_is_the_client_constant_and_the_past_window_its_own(self) -> None:
        signature = inspect.signature(validate_dpop_proof)
        assert signature.parameters["future_tolerance"].default is CLIENT_ISSUE_TIME_FUTURE_TOLERANCE
        assert signature.parameters["max_age"].default is DEFAULT_PROOF_MAX_AGE
        assert timedelta(seconds=60) == CLIENT_ISSUE_TIME_FUTURE_TOLERANCE
        assert "iat_window" not in signature.parameters

    async def test_an_iat_of_an_unconvertible_type_is_refused_rather_than_raising(self, guard: ReplayGuard) -> None:
        """A token endpoint is reachable unauthenticated, so the payload's TYPES are attacker-chosen.

        Every refusal in this module is a ``DpopError`` the caller collapses into one generic
        denial. An ``iat`` that is neither a number nor a numeric string must land there too,
        not escape as a bare ``TypeError`` for a request handler to turn into a 500.
        """
        key = _key()
        proof = pyjwt.encode(
            {"htm": _HTM, "htu": _HTU, "jti": "j", "iat": {"not": "a timestamp"}},
            key=key,
            algorithm="ES256",
            headers={"typ": "dpop+jwt", "jwk": _public_jwk(key)},
        )
        with pytest.raises(DpopError, match="iat"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

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

    async def test_a_proof_replayed_after_the_nonce_bucket_is_wiped_is_refused(
        self, guard: ReplayGuard, kv: FakeNatsClient
    ) -> None:
        """A broker restart empties the nonce bucket; the proof's own iat is what still refuses it.

        This is the proof reaching the guard with its signed issue time. Were that time not
        passed through, the wiped bucket would treat the replay as a first sighting.
        """
        key = _key()
        proof = _proof(key, jti="spent-before-the-restart")
        assert await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)
        bucket = await kv.kv_bucket(name="dpop-nonces")
        bucket.wipe(date_created=datetime.now(UTC))
        with pytest.raises(DpopError, match="replay"):
            await validate_dpop_proof(proof, expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)

    async def test_after_a_broker_restart_a_client_sized_guard_refuses_for_sixty_five_seconds(
        self, kv: FakeNatsClient
    ) -> None:
        """What the client tolerance costs: the reach after a wipe is the minute plus the drift allowance.

        The guard is bound, as a service binds it at startup, so the restart's reconnect hook
        recreates the bucket at the moment the broker comes back. A proof minted 64 seconds
        later is still inside the reach and refused; one minted 65 seconds later is accepted.
        ``refusing_until`` names that same moment, so a surface can say "try again shortly"
        instead.
        """
        key = _key()
        guard = ReplayGuard(
            kv,
            bucket_name="dpop-nonces",
            ttl_seconds=300,
            verifier_future_tolerance=CLIENT_ISSUE_TIME_FUTURE_TOLERANCE,
        )
        with _frozen_clock() as clock:
            await guard.bind()
            await kv.restart_broker()
            restarted = datetime.now(UTC)
            clock.tick(timedelta(seconds=64))
            assert await guard.refusing_until() == restarted + timedelta(seconds=65)
            with pytest.raises(DpopError, match="replay"):
                await validate_dpop_proof(
                    _proof(key, jti="sixty-four-seconds-after"),
                    expected_htm=_HTM,
                    expected_htu=_HTU,
                    replay_guard=guard,
                )
            clock.tick(timedelta(seconds=1))
            assert await guard.refusing_until() is None
            assert await validate_dpop_proof(
                _proof(key, jti="sixty-five-seconds-after"), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard
            )

    async def test_a_future_tolerance_wider_than_the_guard_was_sized_for_is_refused_before_the_proof(
        self, guard: ReplayGuard
    ) -> None:
        """Widening the future tolerance alone would let a replay stamped at the new edge past the wipe check.

        So it is a wiring error, raised whatever the proof -- never a quiet reopening of the hole.
        """
        key = _key()
        with pytest.raises(ValueError, match="verifier_future_tolerance"):
            await validate_dpop_proof(
                _proof(key),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
                future_tolerance=CLIENT_ISSUE_TIME_FUTURE_TOLERANCE + timedelta(seconds=1),
            )

    async def test_a_guard_sized_for_the_pod_tolerance_is_refused_by_the_default(self, kv: FakeNatsClient) -> None:
        # the mistake this split makes possible: sizing a DPoP guard with the pod constant. It
        # would let a replay stamped up to a minute ahead past the wipe check, so it fails loudly.
        key = _key()
        undersized = ReplayGuard(
            kv, bucket_name="dpop-nonces", ttl_seconds=300, verifier_future_tolerance=ISSUE_TIME_FUTURE_TOLERANCE
        )
        with pytest.raises(ValueError, match="verifier_future_tolerance"):
            await validate_dpop_proof(_proof(key), expected_htm=_HTM, expected_htu=_HTU, replay_guard=undersized)

    async def test_widening_the_past_window_needs_no_bigger_guard(self, guard: ReplayGuard) -> None:
        # the guard's reach follows the FUTURE tolerance only; how old a proof may be is not its concern.
        key = _key()
        assert await validate_dpop_proof(
            _proof(key, iat=int(time.time()) - 120),
            expected_htm=_HTM,
            expected_htu=_HTU,
            replay_guard=guard,
            max_age=timedelta(minutes=5),
        )

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


# -- what a refusal tells the operator -------------------------------------------------------


class TestRefusalDiagnostics:
    """An htu mismatch must name BOTH origins it failed to reconcile.

    Callers collapse every `DpopError` into one generic client-facing message on purpose --
    an unauthenticated caller learns nothing from probing. That makes the raise site the only
    place the actual reason ever exists, so an htu mismatch that does not carry the two
    values an operator must reconcile is a reason destroyed rather than a reason withheld.

    Real incident: an admin SPA proxied `/v1` under its own origin, so browsers signed `htu`
    against the ADMIN origin while the deployment's accepted list held only the API's. Every
    refresh was denied, the edge logged the RPC "200 OK" (the rejection rides in the reply
    envelope), the issuer logged nothing at all, and every session died at exactly one
    access-token lifetime.
    """

    _PRESENTED = "https://admin.example/v1/token"

    async def test_an_htu_mismatch_reports_both_the_presented_and_accepted_uris(self, guard: ReplayGuard) -> None:
        """Without both values the log names a problem but not the fix."""
        key = _key()
        with pytest.raises(DpopError) as exc:
            await validate_dpop_proof(
                _proof(key, htu=self._PRESENTED), expected_htm=_HTM, expected_htu=[_HTU], replay_guard=guard
            )
        assert exc.value.detail["presented_htu"] == self._PRESENTED
        assert exc.value.detail["accepted_htu"] == [_HTU]

    async def test_the_message_itself_still_leaks_nothing_situational(self, guard: ReplayGuard) -> None:
        """The structural reason stays generic; the specifics ride in `detail`, which only a
        server-side log consumes."""
        key = _key()
        with pytest.raises(DpopError) as exc:
            await validate_dpop_proof(
                _proof(key, htu=self._PRESENTED), expected_htm=_HTM, expected_htu=[_HTU], replay_guard=guard
            )
        assert self._PRESENTED not in str(exc.value)

    async def test_an_unrelated_failure_carries_an_empty_detail(self, guard: ReplayGuard) -> None:
        """`detail` is opt-in per raise site, never a required field."""
        key = _key()
        with pytest.raises(DpopError) as exc:
            await validate_dpop_proof(_proof(key, htm="GET"), expected_htm=_HTM, expected_htu=_HTU, replay_guard=guard)
        assert exc.value.detail == {}

    async def test_an_overlong_htu_is_truncated_before_it_reaches_the_log(self, guard: ReplayGuard) -> None:
        """The endpoint is unauthenticated, so this value is attacker-chosen in LENGTH.
        Echoed unbounded it is a log-volume amplifier."""
        key = _key()
        with pytest.raises(DpopError) as exc:
            await validate_dpop_proof(
                _proof(key, htu=f"https://evil.example/{'a' * 5000}"),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )
        presented = exc.value.detail["presented_htu"]
        assert presented.endswith("...[truncated]")
        assert len(presented) < 250

    async def test_a_newline_bearing_htu_cannot_forge_a_second_log_line(self, guard: ReplayGuard) -> None:
        """Attacker-chosen in CONTENT too: raw newlines would let one rejected proof write
        what looks like an independent log record."""
        key = _key()
        with pytest.raises(DpopError) as exc:
            await validate_dpop_proof(
                _proof(key, htu="https://evil.example/\n2026-01-01 INFO refresh accepted"),
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )
        assert "\n" not in exc.value.detail["presented_htu"]

    async def test_a_non_string_htu_is_rendered_rather_than_exploding(self, guard: ReplayGuard) -> None:
        """Attacker-chosen in TYPE. The diagnostic path must not raise its own exception on
        the way to reporting a refusal."""
        key = _key()
        with pytest.raises(DpopError) as exc:
            await validate_dpop_proof(
                _proof(key, htu={"not": "a string"}),  # type: ignore[arg-type]
                expected_htm=_HTM,
                expected_htu=_HTU,
                replay_guard=guard,
            )
        assert isinstance(exc.value.detail["presented_htu"], str)
