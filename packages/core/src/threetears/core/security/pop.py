"""DPoP-style proof-of-possession for the agent->proxy hop (platform-auth Option B binding).

The Hub binds an identity token to a holder key via the ``cnf``/``jkt`` claim. On each tool call
the agent signs a short proof with that holder PRIVATE key; the proxy checks the proof's inline
key matches the token's ``cnf``, verifies the signature, and confirms the proof binds to THIS
token (``ath``) + THIS call (``bh``) + is fresh (``iat`` window + a single-use ``jti`` nonce the
caller records in a replay cache). So a leaked token alone -- without the holder private key --
is unusable.

EdDSA-pinned, exactly like the identity token: verification never reads the algorithm from the
proof to choose how to check the signature.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from jwt.algorithms import OKPAlgorithm

from threetears.core.security.identity_token import IdentityTokenError, jwk_thumbprint

__all__ = ["access_token_hash", "make_pop_proof", "verify_pop_proof"]

_ALG = "EdDSA"
_TYP = "pop+jwt"
_REQUIRED = ["ath", "bh", "jti", "iat"]


def access_token_hash(access_token: str) -> str:
    """RFC 9449 ``ath``: base64url(SHA-256(access_token)) with padding stripped.

    Both the holder (minting a proof) and the verifier (the proxy) compute the ``ath`` through
    this one function, so a proof is cryptographically bound to the exact identity token it is
    presented with -- a proof minted for one token cannot be replayed alongside another.

    :param access_token: the compact identity-token JWS the proof accompanies
    :ptype access_token: str
    :return: the base64url SHA-256 digest of the token, padding stripped
    :rtype: str
    """
    digest = hashlib.sha256(access_token.encode("ascii")).digest()
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")


def make_pop_proof(
    *,
    holder_key: Ed25519PrivateKey,
    access_token_hash: str,
    body_hash: str,
    nonce: str,
    iat: int,
) -> str:
    """sign a proof-of-possession JWS with the holder's private key.

    The proof carries the holder's PUBLIC key inline (``jwk`` header) so the verifier can check it
    against the identity token's ``cnf`` thumbprint, then verify this signature.
    ``access_token_hash`` binds the proof to a specific identity token, ``body_hash`` to a specific
    call, ``nonce`` makes it single-use.

    :param holder_key: the agent's per-pod holder private key (bound by the token's ``cnf``)
    :ptype holder_key: Ed25519PrivateKey
    :param access_token_hash: hash of the identity token the proof is presented with (``ath``)
    :ptype access_token_hash: str
    :param body_hash: canonical_call_hash of the call (``bh``)
    :ptype body_hash: str
    :param nonce: a unique single-use value (``jti``)
    :ptype nonce: str
    :param iat: unix-seconds issue time
    :ptype iat: int
    :return: a compact EdDSA JWS proof
    :rtype: str
    """
    public_jwk = OKPAlgorithm.to_jwk(holder_key.public_key(), as_dict=True)
    payload: dict[str, Any] = {
        "ath": access_token_hash,
        "bh": body_hash,
        "jti": nonce,
        "iat": iat,
    }
    return jwt.encode(
        payload,
        key=holder_key,
        algorithm=_ALG,
        headers={"typ": _TYP, "jwk": public_jwk},
    )


def verify_pop_proof(
    proof: str,
    *,
    expected_jkt: str,
    access_token_hash: str,
    body_hash: str,
    leeway_seconds: int = 60,
) -> str:
    """verify a proof-of-possession against the token's holder-key thumbprint + the call binding.

    Fail-closed checks, in order: EdDSA pin; the inline ``jwk`` thumbprint == ``expected_jkt`` (the
    token's ``cnf``); the signature under that inline key; ``ath`` == ``access_token_hash``; ``bh``
    == ``body_hash``; ``iat`` within ``leeway_seconds`` of now. Returns the proof's ``jti`` nonce so
    the caller can enforce single-use against its replay cache. Any failure raises
    :class:`IdentityTokenError`.

    :param proof: the compact JWS proof from the caller
    :ptype proof: str
    :param expected_jkt: the holder-key thumbprint from the verified identity token's ``cnf``
    :ptype expected_jkt: str
    :param access_token_hash: the expected ``ath`` (hash of the identity token presented)
    :ptype access_token_hash: str
    :param body_hash: the expected ``bh`` (canonical_call_hash of the received call)
    :ptype body_hash: str
    :param leeway_seconds: clock-skew tolerance for the ``iat`` freshness window
    :ptype leeway_seconds: int
    :return: the proof nonce (``jti``) for single-use enforcement
    :rtype: str
    :raises IdentityTokenError: on any verification failure
    """
    try:
        header = jwt.get_unverified_header(proof)
    except jwt.PyJWTError as exc:
        raise IdentityTokenError(f"malformed pop header ({type(exc).__name__}).") from None
    if header.get("alg") != _ALG:
        raise IdentityTokenError("unexpected pop algorithm; only EdDSA is accepted.")
    holder_key = _holder_key_from_header(header)
    if jwk_thumbprint(holder_key) != expected_jkt:
        raise IdentityTokenError("pop holder key does not match the token cnf thumbprint.")
    try:
        payload = jwt.decode(
            proof,
            key=holder_key,
            algorithms=["EdDSA"],  # literal pin -- statically auditable; never widen
            options={"require": _REQUIRED},
        )
    except jwt.PyJWTError as exc:
        raise IdentityTokenError(f"pop verification failed ({type(exc).__name__}).") from None
    if payload.get("ath") != access_token_hash:
        raise IdentityTokenError("pop ath does not match the presented identity token.")
    if payload.get("bh") != body_hash:
        raise IdentityTokenError("pop bh does not match the call body.")
    iat = payload.get("iat")
    if not isinstance(iat, int):
        raise IdentityTokenError("pop iat must be an integer.")
    now = int(datetime.now(UTC).timestamp())
    if abs(now - iat) > leeway_seconds:
        raise IdentityTokenError("pop iat is outside the acceptable freshness window.")
    jti = payload.get("jti")
    if not isinstance(jti, str) or not jti:
        raise IdentityTokenError("pop jti must be a non-empty string.")
    return jti


def _holder_key_from_header(header: dict[str, Any]) -> Ed25519PublicKey:
    """extract + validate the inline holder public key (``jwk`` header) from a proof."""
    jwk = header.get("jwk")
    if not isinstance(jwk, dict):
        raise IdentityTokenError("pop header is missing an inline jwk.")
    try:
        key = jwt.PyJWK.from_dict(jwk).key
    except (
        jwt.exceptions.PyJWKError,
        jwt.exceptions.InvalidKeyError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise IdentityTokenError(f"pop inline jwk is invalid ({type(exc).__name__}).") from None
    if not isinstance(key, Ed25519PublicKey):
        raise IdentityTokenError("pop inline jwk is not an Ed25519 public key.")
    return key
