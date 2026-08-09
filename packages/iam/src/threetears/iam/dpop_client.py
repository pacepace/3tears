"""Signing a DPoP proof (RFC 9449) — the client half of :mod:`threetears.iam.dpop`.

A proof is something a CLIENT presents, so this belongs beside the validator rather
than inside any one service: browsers, CLIs, SDKs and the occasional service acting
as its own client all need to produce the identical wire format, and two
implementations of it is two chances to disagree about what the verifier accepts.

**Keep the key for as long as you want the session.** The `cnf` binding on an issued
token pair is this key's thumbprint, so the key that obtained a session is the only
key that can refresh it. A caller that generates a proof, uses the resulting access
token immediately and never refreshes may throw the key away — a headless bootstrap
does exactly that, deliberately binding an unrefreshable session. Any caller that
intends to stay signed in must persist it.

**Every parameter is overridable so a caller can build a proof that SHOULD be
refused.** Tests need that — a replayed ``jti``, a stale ``iat``, a mismatched
``htu`` — and building those by hand elsewhere is how a test's idea of the wire
format drifts from production's.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid7

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, EllipticCurvePrivateKey, generate_private_key
from jwt.algorithms import ECAlgorithm

__all__ = ["new_holder_key", "sign_dpop_proof"]


def new_holder_key() -> EllipticCurvePrivateKey:
    """A fresh P-256 holder key.

    P-256 and nothing else: :mod:`threetears.iam.dpop` pins the curve on the verifying
    side, so any other choice produces a proof this platform refuses.

    :return: the private key whose public half rides in a proof's ``jwk`` header
    :rtype: EllipticCurvePrivateKey
    """
    return generate_private_key(SECP256R1())


def sign_dpop_proof(
    key: EllipticCurvePrivateKey,
    *,
    htm: str,
    htu: str,
    jti: str | None = None,
    iat: int | None = None,
) -> str:
    """A compact RFC 9449 proof JWS signed by ``key``.

    :param key: the holder key. Its public half rides in the ``jwk`` header, which is what
        makes the proof self-describing and what the thumbprint is computed over.
    :ptype key: EllipticCurvePrivateKey
    :param htm: the HTTP method the proof binds to.
    :ptype htm: str
    :param htu: the URL the proof binds to. Matched EXACTLY by the validator, never as a
        prefix, so this must be the endpoint's real externally-reachable URL.
    :ptype htu: str
    :param jti: the proof's unique id; a fresh uuid7 when omitted. Single-use across a
        whole deployment, mint endpoints and the refresh endpoint alike.
    :ptype jti: str | None
    :param iat: issued-at, unix seconds; now when omitted.
    :ptype iat: int | None
    :return: the compact JWS.
    :rtype: str
    """
    public_jwk = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
    payload: dict[str, Any] = {
        "htm": htm,
        "htu": htu,
        "jti": jti or str(uuid7()),
        "iat": iat if iat is not None else int(time.time()),
    }
    return pyjwt.encode(payload, key=key, algorithm="ES256", headers={"typ": "dpop+jwt", "jwk": public_jwk})
