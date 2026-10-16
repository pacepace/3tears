"""the hub's signed statement of whose turn an agent is about to run.

An agent turn travels from the hub (or a channel adapter) through the agent router's durable stream
to an agent pod, and the pod runs it as the user, customer and conversation it names. Any client in
the account can get a turn delivered onto the stream or onto the pod's own subject, so the body
proves nothing about whose turn it is. The hub signs a turn assertion with its identity key: the
agent, the user, the customer, the conversation, and a CSPRNG ``turn_id`` the turn's stream and
completion subjects are keyed on. The router and the agent verify it against the hub JWKS they
already trust, and take those four ids from it, never the body.

It is not the per-turn user assertion that rides into ``CallContext`` and so reaches every tool pod
in the turn: this one stays with the router and the agent runtime, so a tool pod never holds it.

Its life is an hour, not the service envelope's minute: a durable turn waits on the stream while
the router restarts. It is not single-use at verification, because a durable turn is legitimately
redelivered; the pod's dedup gate keeps a turn exactly-once.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from threetears.core.security.freshness import ISSUE_TIME_FUTURE_TOLERANCE

__all__ = [
    "TURN_ASSERTION_AUDIENCE",
    "TURN_ASSERTION_LIFETIME",
    "TurnAssertionClaims",
    "TurnAssertionError",
    "mint_turn_assertion",
    "new_turn_id",
    "verify_turn_assertion",
]

#: the audience every turn assertion names, so an identity token is never taken for one.
TURN_ASSERTION_AUDIENCE: Final[str] = "agents.turn"

#: how long a turn assertion stands: long enough for a durable turn to wait out a router restart.
TURN_ASSERTION_LIFETIME: Final[timedelta] = timedelta(hours=1)

_ALG: Final[str] = "EdDSA"
_REQUIRED: Final[tuple[str, ...]] = (
    "iss",
    "aud",
    "sub",
    "user_id",
    "customer_id",
    "conversation_id",
    "turn_id",
    "iat",
    "exp",
)


class TurnAssertionError(Exception):
    """a turn assertion did not verify; the reason names the check."""


@dataclass(frozen=True, slots=True)
class TurnAssertionClaims:
    """a verified turn assertion.

    :ivar agent_id: the agent the turn is for
    :ivar user_id: the user it runs as
    :ivar customer_id: the customer it belongs to
    :ivar conversation_id: the conversation it continues
    :ivar turn_id: the turn's CSPRNG id, which its stream and completion subjects key on
    """

    agent_id: str
    user_id: str
    customer_id: str
    conversation_id: str
    turn_id: str


def new_turn_id() -> str:
    """a fresh CSPRNG turn id, one subject token long.

    :return: the id
    :rtype: str
    """
    return secrets.token_hex(16)


def mint_turn_assertion(
    *,
    signing_key: Ed25519PrivateKey,
    kid: str,
    issuer: str,
    agent_id: str,
    user_id: str,
    customer_id: str,
    conversation_id: str,
    turn_id: str,
    now: int | None = None,
) -> str:
    """sign a turn assertion.

    :param signing_key: the hub's identity key, published in its JWKS under ``kid``
    :ptype signing_key: Ed25519PrivateKey
    :param kid: the key's JWKS id
    :ptype kid: str
    :param issuer: the hub's issuer
    :ptype issuer: str
    :param agent_id: the agent the turn is for
    :ptype agent_id: str
    :param user_id: the user it runs as
    :ptype user_id: str
    :param customer_id: the customer it belongs to
    :ptype customer_id: str
    :param conversation_id: the conversation it continues
    :ptype conversation_id: str
    :param turn_id: the turn's id (:func:`new_turn_id`)
    :ptype turn_id: str
    :param now: unix-seconds issue time; the current time when ``None``
    :ptype now: int | None
    :return: the compact JWS
    :rtype: str
    """
    issued = int(time.time()) if now is None else now
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": TURN_ASSERTION_AUDIENCE,
        "sub": agent_id,
        "user_id": user_id,
        "customer_id": customer_id,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "iat": issued,
        "exp": issued + int(TURN_ASSERTION_LIFETIME.total_seconds()),
    }
    return jwt.encode(payload, key=signing_key, algorithm=_ALG, headers={"kid": kid})


def verify_turn_assertion(
    token: str,
    *,
    jwks: Mapping[str, Any],
    issuer: str,
    agent_id: str,
) -> TurnAssertionClaims:
    """verify ``token`` as a turn for ``agent_id`` from ``issuer``.

    :param token: the compact JWS
    :ptype token: str
    :param jwks: the hub's trusted JWKS
    :ptype jwks: Mapping[str, Any]
    :param issuer: the hub's issuer
    :ptype issuer: str
    :param agent_id: the agent the verifier serves or routes for
    :ptype agent_id: str
    :return: the verified claims
    :rtype: TurnAssertionClaims
    :raises TurnAssertionError: naming the check that failed
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise TurnAssertionError(f"malformed turn assertion header ({type(exc).__name__})") from exc
    kid = header.get("kid")
    if header.get("alg") != _ALG or not isinstance(kid, str) or not kid:
        raise TurnAssertionError("the turn assertion is not EdDSA, or names no kid")
    try:
        payload = jwt.decode(
            token,
            key=_public_key(jwks, kid),
            algorithms=["EdDSA"],  # literal pin -- never widen
            issuer=issuer,
            audience=TURN_ASSERTION_AUDIENCE,
            leeway=int(ISSUE_TIME_FUTURE_TOLERANCE.total_seconds()),
            options={"require": list(_REQUIRED)},
        )
    except jwt.PyJWTError as exc:
        raise TurnAssertionError(f"the turn assertion does not verify ({type(exc).__name__})") from exc
    if payload.get("sub") != agent_id:
        raise TurnAssertionError("the turn assertion is for another agent")
    if int(payload["exp"]) - int(payload["iat"]) > TURN_ASSERTION_LIFETIME.total_seconds():
        raise TurnAssertionError("the turn assertion claims a longer life than the maximum")
    return TurnAssertionClaims(
        agent_id=str(payload["sub"]),
        user_id=str(payload["user_id"]),
        customer_id=str(payload["customer_id"]),
        conversation_id=str(payload["conversation_id"]),
        turn_id=str(payload["turn_id"]),
    )


def _public_key(jwks: Mapping[str, Any], kid: str) -> Ed25519PublicKey:
    """the Ed25519 public key ``kid`` names in ``jwks``.

    :param jwks: the JWKS
    :ptype jwks: Mapping[str, Any]
    :param kid: the key id
    :ptype kid: str
    :return: the key
    :rtype: Ed25519PublicKey
    :raises TurnAssertionError: when no Ed25519 key carries that id
    """
    try:
        key_set = jwt.PyJWKSet.from_dict(dict(jwks))
    except (jwt.exceptions.PyJWKError, jwt.exceptions.PyJWKSetError, KeyError, TypeError, ValueError) as exc:
        raise TurnAssertionError(f"malformed JWKS ({type(exc).__name__})") from exc
    found = next((jwk.key for jwk in key_set.keys if jwk.key_id == kid), None)
    if not isinstance(found, Ed25519PublicKey):
        raise TurnAssertionError("no Ed25519 key in the JWKS carries the turn assertion's kid")
    return found
