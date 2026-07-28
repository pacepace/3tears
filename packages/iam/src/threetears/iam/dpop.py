"""DPoP (RFC 9449) proof-of-possession validation.

A DPoP proof binds a token to a private key its bearer must actually hold. A
stolen bearer token alone is then useless: replaying it requires the key too.
The binding is the RFC 7638 thumbprint of the holder's public key, carried in
the token's ``cnf.jkt`` claim and recomputed from every proof.

Scope is the AUTHORIZATION-SERVER TOKEN ENDPOINT. RFC 9449 SS4.2 says a proof
presented there carries ``htm``/``htu``/``iat``/``jti`` and nothing else -- no
``ath``, because there is no access token to hash yet, and no server-issued
``nonce``, because the ``jti`` single-use check already provides replay
protection. A proof accompanying a RESOURCE request is a different shape and
this module does not validate one.

Sibling to :mod:`threetears.core.security.pop`, which does the same job for the
platform's internal agent-to-proxy hop with an EdDSA proof bound to a call
hash. Same discipline -- pin the algorithm from a literal, extract the inline
key, verify, check freshness, enforce single use -- applied to RFC 9449's
actual wire format: ``typ: dpop+jwt``, ``alg: ES256``, an inline P-256 ``jwk``
header.

**The curve is pinned to P-256.** PyJWK will happily parse a P-384, P-521, or
secp256k1 key from the header, and accepting one would mean the proof chose its
own security level. It does not get to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import jwt
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, EllipticCurvePublicKey

from threetears.core.coordination import ReplayGuard
from threetears.core.security.identity_token import jwk_thumbprint

__all__ = ["DEFAULT_IAT_WINDOW", "DpopError", "DpopProof", "validate_dpop_proof"]

_ALG: Final[str] = "ES256"
_TYP: Final[str] = "dpop+jwt"
_REQUIRED: Final[list[str]] = ["jti", "htm", "htu", "iat"]

#: Clock-skew and freshness tolerance for a proof's ``iat``, matching
#: :mod:`threetears.core.security.pop`'s default so the two proof formats age alike.
DEFAULT_IAT_WINDOW: Final[timedelta] = timedelta(seconds=60)


class DpopError(Exception):
    """A DPoP proof could not be trusted: bad signature, wrong ``typ``/``alg``, an inline key
    of the wrong shape, a stale or future ``iat``, a missing claim, or a replayed ``jti``.

    Carries only the structural reason -- never the proof string or key material -- so it is
    safe to log at a verification boundary.

    :ivar detail: operator-actionable specifics for the failures where the structural reason
        alone does not name the fix. Callers are expected to collapse every ``DpopError``
        into one generic client-facing message -- an unauthenticated caller must learn
        nothing from probing -- which makes the raise site the only place the actual reason
        ever exists. Without this, that reason is destroyed rather than withheld. Values are
        unvalidated, caller-supplied claim data: diagnostics to log, never inputs to trust.
    """

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


#: Cap on the attacker-supplied ``htu`` echoed into :attr:`DpopError.detail`.
#:
#: A token endpoint is reachable unauthenticated, and this value comes straight off an
#: unverified proof payload -- so it is attacker-chosen in both LENGTH and TYPE. Logged
#: unbounded it is a log-volume amplifier; logged raw it can carry newlines and forge log
#: lines. Truncated and coerced to ``str``, it stays exactly as useful for the one job it
#: has: showing an operator which origin a real browser signed.
_MAX_LOGGED_HTU_CHARS: Final[int] = 200


def _loggable_htu(htu: object) -> str:
    """Render an unverified ``htu`` claim safely for a server-side log.

    :param htu: raw ``htu`` claim from an unverified proof payload; any type.
    :ptype htu: object
    :return: single-line, length-capped rendering.
    :rtype: str
    """
    flattened = " ".join(str(htu).split())
    if len(flattened) > _MAX_LOGGED_HTU_CHARS:
        return f"{flattened[:_MAX_LOGGED_HTU_CHARS]}...[truncated]"
    return flattened


@dataclass(frozen=True, slots=True)
class DpopProof:
    """A validated proof's holder-key thumbprint.

    :ivar jkt: the RFC 7638 thumbprint to bind into, or match against, a token's ``cnf``
        claim. The same value :func:`~threetears.core.security.identity_token.jwk_thumbprint`
        computes for any other holder-key binding in this platform.
    """

    jkt: str


async def validate_dpop_proof(
    proof: str,
    *,
    expected_htm: str,
    expected_htu: str | Sequence[str],
    replay_guard: ReplayGuard,
    iat_window: timedelta = DEFAULT_IAT_WINDOW,
) -> DpopProof:
    """Validate a DPoP proof presented at a token endpoint.

    Checks run fail-closed and in this order: ``typ`` and ``alg`` are pinned before any key
    material is touched; the inline ``jwk`` must be a P-256 EC PUBLIC key; the signature must
    verify under that key; ``htm``/``htu`` must match what the caller expects; ``iat`` must be
    fresh; and ``jti`` must be unused. The ``jti`` check is last and is consuming, so an
    otherwise-invalid proof does not burn a nonce.

    :param proof: the compact DPoP JWS.
    :ptype proof: str
    :param expected_htm: the HTTP method the proof must bind to.
    :ptype expected_htm: str
    :param expected_htu: the URI the proof must bind to, or several acceptable URIs where one
        deployment is legitimately reachable at more than one origin. Each entry is matched
        EXACTLY -- never as a prefix or a wildcard -- so a list is not a relaxation, just a
        longer allow-list.
    :ptype expected_htu: str | Sequence[str]
    :param replay_guard: the fail-closed single-use guard for the proof's ``jti``.
    :ptype replay_guard: ReplayGuard
    :param iat_window: freshness tolerance for ``iat``.
    :ptype iat_window: timedelta
    :return: the resolved holder-key thumbprint.
    :rtype: DpopProof
    :raises DpopError: on any failure. The caller MUST treat this as deny -- never as a
        fallback to an unverified key.
    """
    try:
        header = jwt.get_unverified_header(proof)
    except jwt.PyJWTError as exc:
        raise DpopError(f"malformed dpop proof header ({type(exc).__name__}).") from None
    # Reject before selecting a key: defence in depth with the literal pin in decode, so an
    # HS256, `none`, or EdDSA header never reaches signature verification.
    if header.get("typ") != _TYP:
        raise DpopError("dpop proof header 'typ' must be 'dpop+jwt'.")
    if header.get("alg") != _ALG:
        raise DpopError("unexpected dpop proof algorithm; only ES256 is accepted.")
    holder_key = _holder_key_from_header(header)
    try:
        payload = jwt.decode(
            proof,
            key=holder_key,
            algorithms=[_ALG],  # literal pin -- statically auditable; never widen
            options={"require": _REQUIRED},
        )
    except jwt.PyJWTError as exc:
        raise DpopError(f"dpop proof verification failed ({type(exc).__name__}).") from None

    if payload.get("htm") != expected_htm:
        raise DpopError("dpop proof htm does not match the expected request method.")
    accepted_htu = (expected_htu,) if isinstance(expected_htu, str) else tuple(expected_htu)
    if not accepted_htu:
        raise DpopError("no acceptable dpop htu was supplied for validation.")
    if payload.get("htu") not in accepted_htu:
        # The one rejection an operator cannot diagnose from the structural reason alone:
        # a browser signs `htu` against the origin it actually called, so a deployment
        # reachable through a proxying front-end fails here while every URI involved looks
        # individually correct. `detail` names the two values to reconcile.
        raise DpopError(
            "dpop proof htu does not match the expected request URI.",
            detail={"presented_htu": _loggable_htu(payload.get("htu")), "accepted_htu": list(accepted_htu)},
        )

    iat = payload.get("iat")
    if not isinstance(iat, int) or isinstance(iat, bool):
        raise DpopError("dpop proof iat must be an integer.")
    now = int(datetime.now(UTC).timestamp())
    # Two-sided: a proof from the future is as suspect as a stale one, and an unbounded
    # future iat would let an attacker mint proofs today for use after a key rotation.
    #
    # The future half is a BACKSTOP today, not the thing that fires: PyJWT validates `iat`
    # during decode above and rejects an immature signature first. It is kept because that
    # is PyJWT's default with zero leeway -- a caller that ever passes leeway, or a PyJWT
    # release that relaxes the check, silently removes the only guard otherwise. The stale
    # half is genuinely this module's: PyJWT has no maximum-age notion at all.
    if abs(now - iat) > iat_window.total_seconds():
        raise DpopError("dpop proof iat is outside the acceptable freshness window.")

    jti = payload.get("jti")
    if not isinstance(jti, str) or not jti:
        raise DpopError("dpop proof jti must be a non-empty string.")
    if not await replay_guard.record_unique(jti):
        raise DpopError("dpop proof jti has already been used (replay).")
    return DpopProof(jkt=jwk_thumbprint(holder_key))


def _holder_key_from_header(header: dict[str, Any]) -> EllipticCurvePublicKey:
    """Extract and validate the inline holder public key from a proof header."""
    jwk = header.get("jwk")
    if not isinstance(jwk, dict):
        raise DpopError("dpop proof header is missing an inline jwk.")
    if "d" in jwk:
        # A legitimate proof carries public material only. An inline private key means
        # either a badly broken client or an attempt to confuse the verifier.
        raise DpopError("dpop proof inline jwk must not carry private key material.")
    try:
        key = jwt.PyJWK.from_dict(jwk).key
    except (
        jwt.exceptions.PyJWKError,
        jwt.exceptions.InvalidKeyError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise DpopError(f"dpop proof inline jwk is invalid ({type(exc).__name__}).") from None
    if not isinstance(key, EllipticCurvePublicKey) or not isinstance(key.curve, SECP256R1):
        raise DpopError("dpop proof inline jwk must be a P-256 EC public key.")
    return key
