"""The proxy->pod assertion: the second-hop binding (platform-auth Option B).

The proxy->pod hop is the weakest boundary -- the pod cannot see the transport's subject lock, and
an identity-only message can be spliced/replayed onto the discoverable internal subject. So the
proxy signs an assertion binding the VERIFIED caller identity + THIS call body (``bh``) + a
single-use nonce (``jti``) + the target pod (``aud``) with the proxy's OWN Ed25519 key (published
in the Hub JWKS alongside the hub key, selected by ``kid``). The pod verifies it and runs the tool
under the assertion's identity -- this is the pod's PRIMARY gate; the pod-side identity-token
re-verify (C6) is the belt behind it.

EdDSA-pinned, exactly like the identity token: verification never reads the algorithm from the
assertion to choose how to check the signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from threetears.core.security.identity_token import IdentityTokenError

__all__ = ["ProxyAssertionClaims", "mint_proxy_assertion", "verify_proxy_assertion"]

_ALG = "EdDSA"
_ISSUER = "registry"
_REQUIRED = ("iss", "aud", "sub", "customer_id", "bh", "jti", "iat", "exp")


@dataclass(frozen=True, slots=True)
class ProxyAssertionClaims:
    """the verified proxy assertion: which agent/customer, for which call body, to which pod.

    ``pod_id`` is the ``aud`` (the target pod), ``body_hash`` the ``bh``, ``jti`` the single-use
    nonce the pod records in its replay cache. ``user_id`` is ``None`` for agent-initiated calls.
    """

    sub: str  # the verified agent_id
    customer_id: str
    pod_id: str  # aud
    body_hash: str  # bh
    jti: str  # nonce
    iat: int
    exp: int
    user_id: str | None = None


def mint_proxy_assertion(
    *,
    signing_key: Ed25519PrivateKey,
    kid: str,
    pod_id: str,
    agent_id: str,
    customer_id: str,
    body_hash: str,
    nonce: str,
    iat: int,
    exp: int,
    user_id: str | None = None,
) -> str:
    """sign a proxy->pod assertion with the proxy's signing key, stamping ``kid`` for JWKS lookup.

    :param signing_key: the proxy's Ed25519 signing key (its public key is in the Hub JWKS)
    :ptype signing_key: Ed25519PrivateKey
    :param kid: the JWKS key id for the proxy's key
    :ptype kid: str
    :param pod_id: the target pod (``aud``)
    :ptype pod_id: str
    :param agent_id: the VERIFIED agent identity (``sub``)
    :ptype agent_id: str
    :param customer_id: the VERIFIED owning customer
    :ptype customer_id: str
    :param body_hash: canonical_call_hash of the forwarded call (``bh``)
    :ptype body_hash: str
    :param nonce: a unique single-use value (``jti``)
    :ptype nonce: str
    :param iat: unix-seconds issue time
    :ptype iat: int
    :param exp: unix-seconds expiry (a short window -- one forwarded hop)
    :ptype exp: int
    :param user_id: the VERIFIED human principal, when one is in the loop
    :ptype user_id: str | None
    :return: a compact EdDSA JWS assertion
    :rtype: str
    """
    payload: dict[str, Any] = {
        "iss": _ISSUER,
        "aud": pod_id,
        "sub": agent_id,
        "customer_id": customer_id,
        "bh": body_hash,
        "jti": nonce,
        "iat": iat,
        "exp": exp,
    }
    if user_id is not None:
        payload["user_id"] = user_id
    return jwt.encode(payload, key=signing_key, algorithm=_ALG, headers={"kid": kid})


def verify_proxy_assertion(
    assertion: str,
    *,
    jwks: dict[str, Any],
    expected_pod_id: str,
    body_hash: str,
    leeway_seconds: int = 0,
) -> ProxyAssertionClaims:
    """verify a proxy assertion against the Hub JWKS + the target pod + the call body.

    Fail-closed checks: EdDSA pin; ``kid`` selects the proxy key from ``jwks``; the signature; the
    ``registry`` issuer; ``aud`` == ``expected_pod_id``; expiry; the required claims; and ``bh`` ==
    ``body_hash``. Returns the verified :class:`ProxyAssertionClaims` (the pod runs the tool under
    that identity + records ``jti`` for single-use). Any failure raises :class:`IdentityTokenError`.

    :param assertion: the compact JWS assertion from the proxy
    :ptype assertion: str
    :param jwks: the Hub JWKS (carries the proxy's public key)
    :ptype jwks: dict[str, Any]
    :param expected_pod_id: this pod's id (the ``aud`` the assertion must target)
    :ptype expected_pod_id: str
    :param body_hash: the expected ``bh`` (canonical_call_hash of the received call)
    :ptype body_hash: str
    :param leeway_seconds: clock-skew tolerance
    :ptype leeway_seconds: int
    :return: the verified assertion claims
    :rtype: ProxyAssertionClaims
    :raises IdentityTokenError: on any verification failure
    """
    try:
        header = jwt.get_unverified_header(assertion)
    except jwt.PyJWTError as exc:
        raise IdentityTokenError(f"malformed proxy assertion header ({type(exc).__name__}).") from None
    if header.get("alg") != _ALG:
        raise IdentityTokenError("unexpected proxy assertion algorithm; only EdDSA is accepted.")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise IdentityTokenError("proxy assertion is missing a string kid header.")
    public_key = _select_public_key(jwks, kid)
    try:
        payload = jwt.decode(
            assertion,
            key=public_key,
            algorithms=["EdDSA"],  # literal pin -- statically auditable; never widen
            issuer=_ISSUER,
            audience=expected_pod_id,
            leeway=leeway_seconds,
            options={"require": list(_REQUIRED)},
        )
    except jwt.PyJWTError as exc:
        raise IdentityTokenError(f"proxy assertion verification failed ({type(exc).__name__}).") from None
    if payload.get("bh") != body_hash:
        raise IdentityTokenError("proxy assertion bh does not match the call body.")
    user_id = payload.get("user_id")
    return ProxyAssertionClaims(
        sub=_require_nonempty_str(payload, "sub"),
        customer_id=_require_nonempty_str(payload, "customer_id"),
        pod_id=_require_nonempty_str(payload, "aud"),
        body_hash=_require_nonempty_str(payload, "bh"),
        jti=_require_nonempty_str(payload, "jti"),
        iat=int(payload["iat"]),
        exp=int(payload["exp"]),
        user_id=None if user_id is None else _require_nonempty_str(payload, "user_id"),
    )


def _select_public_key(jwks: dict[str, Any], kid: str) -> Ed25519PublicKey:
    """resolve the Ed25519 public key for ``kid`` from a JWKS, or reject."""
    try:
        key_set = jwt.PyJWKSet.from_dict(jwks)
    except (
        jwt.exceptions.PyJWKError,
        jwt.exceptions.PyJWKSetError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise IdentityTokenError(f"malformed JWKS ({type(exc).__name__}).") from None
    for jwk in key_set.keys:
        if jwk.key_id == kid:
            key = jwk.key
            if not isinstance(key, Ed25519PublicKey):
                raise IdentityTokenError("JWKS key for kid is not an Ed25519 public key.")
            return key
    raise IdentityTokenError("no JWKS key matches the proxy assertion kid.")


def _require_nonempty_str(payload: dict[str, Any], claim: str) -> str:
    """return ``payload[claim]`` iff it is a non-empty string, else reject (claim NAME only)."""
    value = payload.get(claim)
    if not isinstance(value, str) or not value:
        raise IdentityTokenError(f"proxy assertion claim {claim!r} must be a non-empty string.")
    return value
