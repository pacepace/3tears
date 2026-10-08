"""a call-scoped delegated token, and the check a door makes on whoever presents a token.

A tool pod acts as its caller at the platform's doors (``l3.*``, ``datasource.*``, objects, ...).
Today it does so with the caller's own bearer tokens, forwarded whole in ``CallContext``: any pod
in the call holds the agent's token for its ten minutes and the user assertion for its hour, and
can use them anywhere, for anything, after the call. Instead the registry mints, per forwarded call,
a delegated token: the agent as ``sub``, the pod as ``act``, the user and customer bound in, the
doors the call may reach as ``aud``, the pod's own key as ``cnf``, the call's deadline as ``exp``.
The pod proves possession of its key on every door request
(:mod:`threetears.core.security.pop`), so the token is useless to anyone else, after the call, and
at any other door. An agent presents its own token the same way, proving its own key.

Signed with the registry's proxy-assertion key, which the hub JWKS publishes; the header's ``typ``
and the claims keep it from being taken for a proxy assertion or an identity token, and the other
way round. EdDSA-pinned.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from threetears.core.security.freshness import ISSUE_TIME_FUTURE_TOLERANCE
from threetears.core.security.identity_token import IdentityTokenError, verify_identity_token
from threetears.core.security.pop import VerifiedPopProof, access_token_hash, verify_pop_proof

__all__ = [
    "DELEGATED_TOKEN_ISSUER",
    "DELEGATED_TOKEN_MAX_LIFETIME",
    "DELEGATED_TOKEN_TYPE",
    "DelegatedClaims",
    "DelegatedTokenError",
    "DoorCaller",
    "ProofAbsentError",
    "is_delegated_token",
    "mint_delegated_token",
    "verify_delegated_token",
    "verify_door_caller",
]

#: the ``typ`` header every delegated token carries, and every verifier requires.
DELEGATED_TOKEN_TYPE: Final[str] = "delegated+jwt"

#: the issuer the registry signs delegated tokens as.
DELEGATED_TOKEN_ISSUER: Final[str] = "registry"

#: the longest a delegated token may claim: a tool call's own deadline never exceeds this.
DELEGATED_TOKEN_MAX_LIFETIME: Final[timedelta] = timedelta(hours=1)

_ALG: Final[str] = "EdDSA"
_REQUIRED: Final[tuple[str, ...]] = ("iss", "aud", "sub", "act", "customer_id", "cnf", "corr", "iat", "exp")


class DelegatedTokenError(Exception):
    """a delegated token, or a caller's proof, did not verify; the reason names the check."""


class ProofAbsentError(DelegatedTokenError):
    """the caller presented a token and no proof of possession: the door's window decides."""


@dataclass(frozen=True, slots=True)
class DelegatedClaims:
    """a verified delegated token.

    :ivar agent_id: the agent the pod acts for
    :ivar pod_id: the tool pod acting
    :ivar customer_id: the customer
    :ivar user_id: the user the call runs as, if any
    :ivar holder_jkt: the pod key's thumbprint the pod must prove
    :ivar correlation_id: the call it was minted for
    """

    agent_id: str
    pod_id: str
    customer_id: str
    user_id: str | None
    holder_jkt: str
    correlation_id: str


@dataclass(frozen=True, slots=True)
class DoorCaller:
    """whoever a door request proved it acts as.

    :ivar agent_id: the agent
    :ivar customer_id: the customer
    :ivar user_id: the user, if the credential binds one
    :ivar pod_id: the tool pod acting for the agent, or ``None`` for the agent itself
    :ivar proof: the request's verified proof, whose ``jti`` the door spends in its ledger
    """

    agent_id: str
    customer_id: str
    user_id: str | None
    pod_id: str | None
    proof: VerifiedPopProof


def is_delegated_token(token: str) -> bool:
    """whether ``token``'s header says it is a delegated token (unverified; routing only).

    :param token: a compact JWS
    :ptype token: str
    :return: whether its ``typ`` is :data:`DELEGATED_TOKEN_TYPE`
    :rtype: bool
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        header = {}
    return header.get("typ") == DELEGATED_TOKEN_TYPE


def mint_delegated_token(
    *,
    signing_key: Ed25519PrivateKey,
    kid: str,
    agent_id: str,
    pod_id: str,
    customer_id: str,
    user_id: str | None,
    doors: Iterable[str],
    holder_jkt: str,
    correlation_id: str,
    deadline: int,
    now: int | None = None,
) -> str:
    """sign a delegated token for one forwarded call.

    :param signing_key: the registry's proxy-assertion key
    :ptype signing_key: Ed25519PrivateKey
    :param kid: its JWKS id
    :ptype kid: str
    :param agent_id: the verified calling agent
    :ptype agent_id: str
    :param pod_id: the tool pod the call is forwarded to
    :ptype pod_id: str
    :param customer_id: the verified customer
    :ptype customer_id: str
    :param user_id: the verified user, if the call carries one
    :ptype user_id: str | None
    :param doors: the doors the call may reach
    :ptype doors: Iterable[str]
    :param holder_jkt: the pod's own key thumbprint
    :ptype holder_jkt: str
    :param correlation_id: the call's correlation id
    :ptype correlation_id: str
    :param deadline: unix-seconds end of the call
    :ptype deadline: int
    :param now: unix-seconds issue time; the current time when ``None``
    :ptype now: int | None
    :return: the compact JWS
    :rtype: str
    :raises ValueError: when the deadline is not after now, or further than the maximum, or no door is named
    """
    issued = int(time.time()) if now is None else now
    audience = sorted(set(doors))
    if not audience:
        raise ValueError("a delegated token names at least one door")
    if not issued < deadline <= issued + int(DELEGATED_TOKEN_MAX_LIFETIME.total_seconds()):
        raise ValueError(f"a delegated token lives past now and at most {DELEGATED_TOKEN_MAX_LIFETIME}")
    payload: dict[str, Any] = {
        "iss": DELEGATED_TOKEN_ISSUER,
        "aud": audience,
        "sub": agent_id,
        "act": pod_id,
        "customer_id": customer_id,
        "user_id": user_id,
        "cnf": holder_jkt,
        "corr": correlation_id,
        "iat": issued,
        "exp": deadline,
    }
    return jwt.encode(payload, key=signing_key, algorithm=_ALG, headers={"kid": kid, "typ": DELEGATED_TOKEN_TYPE})


def verify_delegated_token(token: str, *, jwks: Mapping[str, Any], door: str) -> DelegatedClaims:
    """verify ``token`` as a delegation that may reach ``door``.

    :param token: the compact JWS
    :ptype token: str
    :param jwks: the hub JWKS, which publishes the registry's key
    :ptype jwks: Mapping[str, Any]
    :param door: this door
    :ptype door: str
    :return: the verified claims
    :rtype: DelegatedClaims
    :raises DelegatedTokenError: naming the check that failed
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise DelegatedTokenError(f"malformed delegated token header ({type(exc).__name__})") from exc
    kid = header.get("kid")
    if header.get("alg") != _ALG or header.get("typ") != DELEGATED_TOKEN_TYPE or not isinstance(kid, str) or not kid:
        raise DelegatedTokenError("not an EdDSA delegated token with a kid")
    try:
        payload = jwt.decode(
            token,
            key=_public_key(jwks, kid),
            algorithms=["EdDSA"],  # literal pin -- never widen
            issuer=DELEGATED_TOKEN_ISSUER,
            audience=door,
            leeway=int(ISSUE_TIME_FUTURE_TOLERANCE.total_seconds()),
            options={"require": list(_REQUIRED)},
        )
    except jwt.PyJWTError as exc:
        raise DelegatedTokenError(f"the delegated token does not verify ({type(exc).__name__})") from exc
    if int(payload["exp"]) - int(payload["iat"]) > DELEGATED_TOKEN_MAX_LIFETIME.total_seconds():
        raise DelegatedTokenError("the delegated token claims a longer life than the maximum")
    user = payload.get("user_id")
    return DelegatedClaims(
        agent_id=str(payload["sub"]),
        pod_id=str(payload["act"]),
        customer_id=str(payload["customer_id"]),
        user_id=str(user) if user is not None else None,
        holder_jkt=str(payload["cnf"]),
        correlation_id=str(payload["corr"]),
    )


def verify_door_caller(
    token: str,
    proof: str | None,
    *,
    jwks: Mapping[str, Any],
    hub_issuer: str,
    door: str,
    body_hash: str,
) -> DoorCaller:
    """verify a door request's credential and its proof of possession.

    A delegated token is proven with the tool pod's key; an agent's identity token with the agent's
    own (its ``cnf``). The proof binds the token (``ath``) and this request's body (``bh``). The
    door spends the proof's ``jti`` in its ledger.

    :param token: the presented credential
    :ptype token: str
    :param proof: the presented proof of possession, or ``None``
    :ptype proof: str | None
    :param jwks: the hub JWKS
    :ptype jwks: Mapping[str, Any]
    :param hub_issuer: the hub's identity-token issuer
    :ptype hub_issuer: str
    :param door: this door, which a delegated token must name
    :ptype door: str
    :param body_hash: this request's body hash, as the caller signed it
    :ptype body_hash: str
    :return: the proven caller
    :rtype: DoorCaller
    :raises ProofAbsentError: when no proof is presented (the door's window decides)
    :raises DelegatedTokenError: when the token or the proof does not verify
    """
    if not proof:
        raise ProofAbsentError("the request carries no proof of possession")
    caller: tuple[str, str, str | None, str | None]
    if is_delegated_token(token):
        claims = verify_delegated_token(token, jwks=jwks, door=door)
        holder, caller = claims.holder_jkt, (claims.agent_id, claims.customer_id, claims.user_id, claims.pod_id)
    else:
        try:
            identity = verify_identity_token(token, jwks=dict(jwks), issuer=hub_issuer)
        except IdentityTokenError as exc:
            raise DelegatedTokenError(f"the identity token does not verify ({exc})") from exc
        if not identity.cnf:
            raise DelegatedTokenError("the identity token binds no key to prove")
        holder, caller = identity.cnf, (identity.sub, identity.customer_id, identity.user_id, None)
    try:
        verified = verify_pop_proof(
            proof, expected_jkt=holder, access_token_hash=access_token_hash(token), body_hash=body_hash
        )
    except IdentityTokenError as exc:
        raise DelegatedTokenError(f"the proof of possession does not verify ({exc})") from exc
    return DoorCaller(agent_id=caller[0], customer_id=caller[1], user_id=caller[2], pod_id=caller[3], proof=verified)


def _public_key(jwks: Mapping[str, Any], kid: str) -> Ed25519PublicKey:
    """the Ed25519 public key ``kid`` names in ``jwks``.

    :param jwks: the JWKS
    :ptype jwks: Mapping[str, Any]
    :param kid: the key id
    :ptype kid: str
    :return: the key
    :rtype: Ed25519PublicKey
    :raises DelegatedTokenError: when no Ed25519 key carries that id
    """
    try:
        key_set = jwt.PyJWKSet.from_dict(dict(jwks))
    except (jwt.exceptions.PyJWKError, jwt.exceptions.PyJWKSetError, KeyError, TypeError, ValueError) as exc:
        raise DelegatedTokenError(f"malformed JWKS ({type(exc).__name__})") from exc
    found = next((jwk.key for jwk in key_set.keys if jwk.key_id == kid), None)
    if not isinstance(found, Ed25519PublicKey):
        raise DelegatedTokenError("no Ed25519 key in the JWKS carries the delegated token's kid")
    return found
