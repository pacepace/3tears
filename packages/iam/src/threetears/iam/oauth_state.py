"""OAuth ``state``: a signed, short-TTL, single-use CSRF token for the redirect round-trip.

An OAuth authorization redirect hands the provider a ``state`` value and gets it back on the
callback. Its job is to prove the callback belongs to a flow *this* deployment started, so an
attacker cannot walk a victim's browser into a callback carrying the attacker's authorization
code. That needs two properties, and they are not the same property:

**Authenticity and freshness** come from signing. The state is a JWT (HS256 under the
deployment's secret) with its own issuer, audience and a short expiry, so any pod validates it
without a shared cookie or session store. :func:`mint_oauth_state` issues one and
:func:`verify_oauth_state` checks it.

**Single use** cannot come from signing, because a signed token is by construction replayable
until it expires. A captured state would otherwise be usable for the whole TTL. So the minted
token carries a random ``nonce``, and the deployment records it on the redirect
(:func:`record_state_nonce`) and consumes it on the callback (:func:`consume_state_nonce`).
The consume is one atomic take, never a read-then-delete: two callbacks racing on the same
captured state could both pass a presence check before either removed it, which defeats the
guarantee this exists to provide.

A caller that skips the nonce store is choosing the stateless-only posture -- authentic and
fresh, replayable within the TTL. That is a real choice for a short enough TTL, which is why
:func:`verify_oauth_state` is public rather than folded into the consume.

The algorithm is pinned twice: once on the DECLARED header before a key is selected, and once
as a literal in the decode. The first stops an ``alg: none`` or an asymmetric-confusion header
from ever reaching signature verification; the second is the statically auditable pin. This is
the same two-stage discipline :mod:`threetears.iam.tokens` applies to session tokens.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import jwt

from threetears.iam.stores.base import StateStore

__all__ = [
    "DEFAULT_STATE_TTL",
    "DEFAULT_STATE_TYPE",
    "NONCE_BYTES",
    "OAuthState",
    "OAuthStateError",
    "consume_state_nonce",
    "mint_oauth_state",
    "record_state_nonce",
    "verify_oauth_state",
]

#: The redirect -> callback round trip. Short: the window a captured state stays replayable
#: for a deployment that opts out of the nonce store.
DEFAULT_STATE_TTL: Final[timedelta] = timedelta(minutes=5)

#: The ``type`` claim, checked on verify so a token minted for some other purpose under the
#: same secret cannot be presented as a state.
DEFAULT_STATE_TYPE: Final[str] = "oauth_state"

#: Entropy in the single-use nonce.
NONCE_BYTES: Final[int] = 16

_ALGORITHM: Final[str] = "HS256"


class OAuthStateError(Exception):
    """A ``state`` failed verification: bad signature, expired, wrong issuer/audience/type, or
    a nonce that was never recorded or has already been consumed.

    One error for every cause, carrying only a structural reason. The callback boundary cannot
    act differently on "forged" than on "replayed", and telling them apart in a message would
    hand an attacker the distinction for free.
    """


@dataclass(frozen=True)
class OAuthState:
    """A verified state token's payload.

    :ivar nonce: the single-use key, for :func:`record_state_nonce` / :func:`consume_state_nonce`.
    :ivar issued_at: when it was minted.
    :ivar expires_at: when it stops verifying. The nonce record is given exactly this life, so
        it neither outlives the token it guards nor dies before a still-valid callback arrives.
    """

    nonce: str
    issued_at: datetime
    expires_at: datetime


def mint_oauth_state(
    *,
    secret: str,
    issuer: str,
    audience: str,
    ttl: timedelta = DEFAULT_STATE_TTL,
    state_type: str = DEFAULT_STATE_TYPE,
    now: datetime | None = None,
) -> str:
    """Mint a signed ``state`` carrying a fresh single-use nonce.

    :param secret: the HMAC signing secret.
    :ptype secret: str
    :param issuer: the ``iss`` claim, checked on verify.
    :ptype issuer: str
    :param audience: the ``aud`` claim, checked on verify. Give the state its own audience so a
        session token cannot be presented here, or the reverse.
    :ptype audience: str
    :param ttl: how long it verifies for.
    :ptype ttl: timedelta
    :param state_type: the ``type`` claim.
    :ptype state_type: str
    :param now: the issuing instant; defaults to the current time.
    :ptype now: datetime | None
    :return: the encoded state token.
    :rtype: str
    """
    issued = now or datetime.now(UTC)
    payload = {
        "type": state_type,
        "nonce": secrets.token_hex(NONCE_BYTES),
        "iss": issuer,
        "aud": audience,
        "iat": int(issued.timestamp()),
        "exp": int((issued + ttl).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def verify_oauth_state(
    state: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    state_type: str = DEFAULT_STATE_TYPE,
) -> OAuthState:
    """Validate a returned ``state`` -- signature, issuer, audience, type and expiry.

    Does NOT check single use; :func:`consume_state_nonce` does that. A caller using this alone
    is accepting that a captured state stays replayable until it expires.

    :param state: the token returned on the callback.
    :ptype state: str
    :param secret: the HMAC signing secret.
    :ptype secret: str
    :param issuer: the required ``iss``.
    :ptype issuer: str
    :param audience: the required ``aud``.
    :ptype audience: str
    :param state_type: the required ``type``.
    :ptype state_type: str
    :return: the verified payload.
    :rtype: OAuthState
    :raises OAuthStateError: on any verification failure.
    """
    try:
        header = jwt.get_unverified_header(state)
    except jwt.PyJWTError as exc:
        raise OAuthStateError("invalid or expired OAuth state") from exc
    # Reject on the DECLARED algorithm before a key is selected -- defence in depth with the
    # literal pin below, so an `alg: none` or asymmetric-confusion header never reaches
    # signature verification.
    if header.get("alg") != _ALGORITHM:
        raise OAuthStateError("invalid or expired OAuth state")
    try:
        payload: dict[str, Any] = jwt.decode(
            state,
            secret,
            algorithms=[_ALGORITHM],  # literal pin -- statically auditable; never widen
            issuer=issuer,
            audience=audience,
            options={"require": ["iss", "aud", "iat", "exp"]},
        )
    except jwt.InvalidTokenError as exc:  # bad signature, malformed, expired, wrong iss/aud
        raise OAuthStateError("invalid or expired OAuth state") from exc
    if payload.get("type") != state_type:
        raise OAuthStateError("wrong token type for OAuth state")
    nonce = payload.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise OAuthStateError("OAuth state carries no usable nonce")
    return OAuthState(
        nonce=nonce,
        issued_at=datetime.fromtimestamp(int(payload["iat"]), tz=UTC),
        expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=UTC),
    )


async def record_state_nonce(
    store: StateStore,
    state: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    state_type: str = DEFAULT_STATE_TYPE,
    now: datetime | None = None,
) -> None:
    """Persist a freshly-minted state's nonce so :func:`consume_state_nonce` can enforce single use.

    The nonce is given the STATE's own remaining life rather than a fresh window: one that
    outlived its token would keep a dead state redeemable, and one that died first would reject
    a still-valid callback.

    :param store: where the nonce lives until the callback.
    :ptype store: StateStore
    :param state: the token just minted.
    :ptype state: str
    :param secret: the HMAC signing secret.
    :ptype secret: str
    :param issuer: the required ``iss``.
    :ptype issuer: str
    :param audience: the required ``aud``.
    :ptype audience: str
    :param state_type: the required ``type``.
    :ptype state_type: str
    :param now: the recording instant; defaults to the current time.
    :ptype now: datetime | None
    :return: nothing
    :rtype: None
    :raises OAuthStateError: if the state does not verify.
    """
    verified = verify_oauth_state(state, secret=secret, issuer=issuer, audience=audience, state_type=state_type)
    moment = now or datetime.now(UTC)
    await store.put(verified.nonce, {}, ttl=max(verified.expires_at - moment, timedelta(0)))


async def consume_state_nonce(
    store: StateStore,
    state: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    state_type: str = DEFAULT_STATE_TYPE,
) -> None:
    """Validate ``state`` and consume its nonce exactly once.

    One atomic take, never a read-then-delete: two callbacks racing on the same captured state
    could both pass a presence check before either removed it, silently defeating the single-use
    guarantee. A miss -- never recorded, or already consumed -- is a replay.

    :param store: where the nonce was recorded.
    :ptype store: StateStore
    :param state: the token returned on the callback.
    :ptype state: str
    :param secret: the HMAC signing secret.
    :ptype secret: str
    :param issuer: the required ``iss``.
    :ptype issuer: str
    :param audience: the required ``aud``.
    :ptype audience: str
    :param state_type: the required ``type``.
    :ptype state_type: str
    :return: nothing
    :rtype: None
    :raises OAuthStateError: if the state does not verify, or its nonce is unknown or spent.
    """
    verified = verify_oauth_state(state, secret=secret, issuer=issuer, audience=audience, state_type=state_type)
    if await store.take(verified.nonce) is None:
        raise OAuthStateError("OAuth state already used or unknown (replay)")
