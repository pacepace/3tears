"""a signed service-to-service envelope: "this message is from service X, for door Y, with this body, now, once".

Any client in a NATS account can get bytes delivered onto a subject it can name, so a message proves
nothing about its sender by arriving. Some services hold no token a receiver can verify -- identity-core
toward the hub, the channel adapters, the gateway. Each signs an envelope with its own Ed25519 key,
published in its JWKS, and the receiving door verifies it before acting on the message.

The envelope is a compact EdDSA JWS (the same construction as the registry's proxy assertion) whose
claims are:

- ``iss``: the sending service principal;
- ``aud``: the door it is addressed to, so an envelope for one door is refused at another;
- ``op``: the operation within the door;
- ``bh``: :func:`body_hash` over exactly the message body it travels with, so it cannot be moved onto
  another body;
- ``iat`` / ``exp``: a life of at most :data:`SERVICE_ENVELOPE_MAX_LIFETIME`;
- ``jti``: a single-use id the receiver records in a ``ReplayGuard`` (:func:`verify_and_record`), so a
  captured envelope cannot be replayed.

EdDSA-pinned: verification never reads the algorithm from the envelope to choose how to check it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from threetears.nats.errors import KvError
from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.security.door_proof import ProofOutcome
from threetears.core.security.freshness import ISSUE_TIME_FUTURE_TOLERANCE

if TYPE_CHECKING:
    from threetears.core.coordination.replay_guard import ReplayGuard
    from threetears.core.security.door_proof import DoorProofSwitch

__all__ = [
    "SERVICE_ENVELOPE_FIELD",
    "SERVICE_ENVELOPE_LEEWAY",
    "SERVICE_ENVELOPE_MAX_LIFETIME",
    "SERVICE_ENVELOPE_REPLAY_TTL",
    "EnvelopeJwksSource",
    "ServiceEnvelopeClaims",
    "ServiceEnvelopeDoor",
    "ServiceEnvelopeError",
    "body_hash",
    "mint_service_envelope",
    "verify_and_record",
    "verify_service_envelope",
]

log = get_logger(__name__)

#: the longest life an envelope may claim. It covers one hop and clock skew; a longer life would let a
#: captured envelope stand for its body long after it was sent.
SERVICE_ENVELOPE_MAX_LIFETIME: Final[timedelta] = timedelta(seconds=60)

#: how far an envelope's issue time may lead the verifier's clock, and how long past its expiry it is
#: still accepted. Every signer is a platform service on a clock the platform keeps, so it is the
#: platform's pod tolerance, not a client device's.
SERVICE_ENVELOPE_LEEWAY: Final[timedelta] = ISSUE_TIME_FUTURE_TOLERANCE

#: how long a door's ``ReplayGuard`` must remember a single-use id: the longest an envelope is
#: accepted, its life plus the leeway past its expiry. A shorter memory would forget an id while its
#: envelope could still be replayed.
SERVICE_ENVELOPE_REPLAY_TTL: Final[timedelta] = SERVICE_ENVELOPE_MAX_LIFETIME + SERVICE_ENVELOPE_LEEWAY

#: the field of a message that carries its envelope; :func:`body_hash` leaves it out of the hash.
SERVICE_ENVELOPE_FIELD: Final[str] = "service_envelope"

#: the prometheus counter of envelopes a door refused (labels ``door``, ``cause``).
_REFUSED_COUNTER: Final[str] = "service_envelope_refused"

_ALG: Final[str] = "EdDSA"
_REQUIRED: Final[tuple[str, ...]] = ("iss", "aud", "op", "bh", "jti", "iat", "exp")


class ServiceEnvelopeError(Exception):
    """a service envelope did not verify, or was already used; the reason names the check."""


@dataclass(frozen=True, slots=True)
class ServiceEnvelopeClaims:
    """a verified envelope.

    :ivar issuer: the sending service principal
    :ivar audience: the door it was addressed to
    :ivar op: the operation
    :ivar body_hash: the hash of the body it signed
    :ivar jti: its single-use id
    :ivar iat: its issue time, unix seconds
    :ivar exp: its expiry, unix seconds
    """

    issuer: str
    audience: str
    op: str
    body_hash: str
    jti: str
    iat: int
    exp: int


def body_hash(body: Mapping[str, Any]) -> str:
    """the hash an envelope signs: sha-256 over the body's canonical JSON, the envelope field left out.

    Canonical: compact JSON with recursively sorted keys, the form :func:`~threetears.core.security.identity_token.canonical_call_hash` uses.
    The body must already be JSON values -- a model's ``model_dump(mode="json")`` -- so the sender and
    the receiver hash the same text; there is deliberately no fallback encoder, which would let the
    two sides stringify one value two ways.

    :param body: the message body, as JSON values
    :ptype body: Mapping[str, Any]
    :return: the base64url digest, unpadded
    :rtype: str
    :raises TypeError: when the body holds a value that is not JSON
    """
    content = {key: value for key, value in body.items() if key != SERVICE_ENVELOPE_FIELD}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # str(bytes, "ascii"), as canonical_call_hash: the alg-pin enforcement reads every ``.decode(``
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")


def mint_service_envelope(
    *,
    signing_key: Ed25519PrivateKey,
    kid: str,
    issuer: str,
    audience: str,
    op: str,
    body: Mapping[str, Any],
    now: int | None = None,
    ttl: timedelta = SERVICE_ENVELOPE_MAX_LIFETIME,
    jti: str | None = None,
) -> str:
    """sign an envelope over ``body`` for ``audience``'s ``op``.

    :param signing_key: the sending service's Ed25519 key, published in its JWKS under ``kid``
    :ptype signing_key: Ed25519PrivateKey
    :param kid: the key's JWKS id
    :ptype kid: str
    :param issuer: the sending service principal
    :ptype issuer: str
    :param audience: the receiving door
    :ptype audience: str
    :param op: the operation within the door
    :ptype op: str
    :param body: the message body the envelope travels with
    :ptype body: Mapping[str, Any]
    :param now: unix-seconds issue time; the current time when ``None``
    :ptype now: int | None
    :param ttl: the envelope's life, at most :data:`SERVICE_ENVELOPE_MAX_LIFETIME`
    :ptype ttl: timedelta
    :param jti: the single-use id; a fresh random one when ``None``
    :ptype jti: str | None
    :return: the compact JWS
    :rtype: str
    :raises ValueError: when ``ttl`` exceeds the maximum
    """
    if ttl > SERVICE_ENVELOPE_MAX_LIFETIME:
        raise ValueError(f"a service envelope lives at most {SERVICE_ENVELOPE_MAX_LIFETIME}, not {ttl}")
    issued = int(time.time()) if now is None else now
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "op": op,
        "bh": body_hash(body),
        "jti": jti if jti is not None else secrets.token_urlsafe(24),
        "iat": issued,
        "exp": issued + int(ttl.total_seconds()),
    }
    return jwt.encode(payload, key=signing_key, algorithm=_ALG, headers={"kid": kid})


def verify_service_envelope(
    envelope: str,
    *,
    jwks: Mapping[str, Any],
    issuer: str,
    audience: str,
    op: str,
    body: Mapping[str, Any],
    leeway_seconds: int = int(SERVICE_ENVELOPE_LEEWAY.total_seconds()),
) -> ServiceEnvelopeClaims:
    """verify ``envelope`` for ``body`` at ``audience``'s ``op``, from ``issuer``.

    Fail closed: the EdDSA pin, the ``kid`` selecting a key in ``jwks``, the signature, the issuer,
    the audience, the operation, the life (expired, or claiming more than the maximum), the required
    claims, and the body hash. It does not record the ``jti``: :func:`verify_and_record` does.

    :param envelope: the compact JWS
    :ptype envelope: str
    :param jwks: the sending service's JWKS, from the verifier's trusted source
    :ptype jwks: Mapping[str, Any]
    :param issuer: the service the door takes this message from
    :ptype issuer: str
    :param audience: this door
    :ptype audience: str
    :param op: the operation this message is for
    :ptype op: str
    :param body: the message body as received
    :ptype body: Mapping[str, Any]
    :param leeway_seconds: clock-skew tolerance on ``exp`` and ``iat``
    :ptype leeway_seconds: int
    :return: the verified claims
    :rtype: ServiceEnvelopeClaims
    :raises ServiceEnvelopeError: naming the check that failed
    """
    try:
        header = jwt.get_unverified_header(envelope)
    except jwt.PyJWTError as exc:
        raise ServiceEnvelopeError(f"malformed envelope header ({type(exc).__name__})") from exc
    kid = header.get("kid")
    if header.get("alg") != _ALG or not isinstance(kid, str) or not kid:
        raise ServiceEnvelopeError("the envelope is not EdDSA, or names no kid")
    try:
        payload = jwt.decode(
            envelope,
            key=_public_key(jwks, kid),
            algorithms=["EdDSA"],  # literal pin -- never widen
            issuer=issuer,
            audience=audience,
            leeway=leeway_seconds,
            options={"require": list(_REQUIRED)},
        )
    except jwt.PyJWTError as exc:
        raise ServiceEnvelopeError(f"the envelope does not verify ({type(exc).__name__})") from exc
    if payload.get("op") != op:
        raise ServiceEnvelopeError("the envelope is for another operation")
    if int(payload["exp"]) - int(payload["iat"]) > SERVICE_ENVELOPE_MAX_LIFETIME.total_seconds():
        raise ServiceEnvelopeError("the envelope claims a longer life than the maximum")
    if payload.get("bh") != body_hash(body):
        raise ServiceEnvelopeError("the envelope was signed over another body")
    jti = payload.get("jti")
    if not isinstance(jti, str) or not jti:
        raise ServiceEnvelopeError("the envelope carries no single-use id")
    return ServiceEnvelopeClaims(
        issuer=str(payload["iss"]),
        audience=audience,
        op=op,
        body_hash=str(payload["bh"]),
        jti=jti,
        iat=int(payload["iat"]),
        exp=int(payload["exp"]),
    )


async def verify_and_record(
    envelope: str,
    *,
    jwks: Mapping[str, Any],
    issuer: str,
    audience: str,
    op: str,
    body: Mapping[str, Any],
    replay_guard: ReplayGuard,
) -> ServiceEnvelopeClaims:
    """verify ``envelope`` and consume its single-use id; a second use is refused.

    :param envelope: the compact JWS
    :ptype envelope: str
    :param jwks: the sending service's JWKS
    :ptype jwks: Mapping[str, Any]
    :param issuer: the service the door takes this message from
    :ptype issuer: str
    :param audience: this door
    :ptype audience: str
    :param op: the operation
    :ptype op: str
    :param body: the message body as received
    :ptype body: Mapping[str, Any]
    :param replay_guard: the door's single-use ledger, remembering ids for at least
        :data:`SERVICE_ENVELOPE_REPLAY_TTL` and sized for :data:`SERVICE_ENVELOPE_LEEWAY`
    :ptype replay_guard: ReplayGuard
    :return: the verified claims
    :rtype: ServiceEnvelopeClaims
    :raises ServiceEnvelopeError: when it does not verify, or its id was already used
    :raises ValueError: when ``replay_guard`` was not sized for :data:`SERVICE_ENVELOPE_LEEWAY`
    :raises threetears.nats.KvError: when the ledger cannot be reached; the caller denies
    """
    replay_guard.require_covers(SERVICE_ENVELOPE_LEEWAY)
    claims = verify_service_envelope(envelope, jwks=jwks, issuer=issuer, audience=audience, op=op, body=body)
    # keyed by issuer as well, so one service's ids can never spend another's
    nonce = f"{claims.issuer}|{claims.jti}"
    if not await replay_guard.record_unique(nonce, issued_at=datetime.fromtimestamp(claims.iat, tz=UTC)):
        log.warning(
            "service envelope refused: its single-use id was already used",
            extra={"extra_data": {"issuer": issuer, "audience": audience, "op": op}},
        )
        raise ServiceEnvelopeError("the envelope was already used")
    return claims


def _public_key(jwks: Mapping[str, Any], kid: str) -> Ed25519PublicKey:
    """the Ed25519 public key ``kid`` names in ``jwks``.

    :param jwks: the JWKS
    :ptype jwks: Mapping[str, Any]
    :param kid: the key id
    :ptype kid: str
    :return: the key
    :rtype: Ed25519PublicKey
    :raises ServiceEnvelopeError: when no Ed25519 key carries that id
    """
    try:
        key_set = jwt.PyJWKSet.from_dict(dict(jwks))
    except (jwt.exceptions.PyJWKError, jwt.exceptions.PyJWKSetError, KeyError, TypeError, ValueError) as exc:
        raise ServiceEnvelopeError(f"malformed JWKS ({type(exc).__name__})") from exc
    found = next((jwk.key for jwk in key_set.keys if jwk.key_id == kid), None)
    if not isinstance(found, Ed25519PublicKey):
        raise ServiceEnvelopeError("no Ed25519 key in the JWKS carries the envelope's kid")
    return found


class EnvelopeJwksSource(Protocol):
    """where a door reads its sender's JWKS from: a trusted, cached fetch, or a key set it holds."""

    async def get_jwks(self, *, refresh: bool = False) -> Mapping[str, Any]:
        """the sender's trusted JWKS.

        :param refresh: ask for a fresh fetch, because an envelope named a kid the set lacks
        :ptype refresh: bool
        :return: the JWKS
        :rtype: Mapping[str, Any]
        :raises ServiceEnvelopeError: when no trusted key set is available
        """
        ...


class ServiceEnvelopeDoor:
    """one door's acceptance of messages from one sending service, inside its dated window.

    A message carrying an envelope is acted on only when it verifies against the sender's JWKS and
    its single-use id is spent. A message without one is the door switch's to judge: accepted while
    the window is open, so a sender that predates the envelope keeps working until its release, and
    refused from the window's date. Every refusal is logged and counted; the door never raises, so a
    receiver's single answer is "act" or "do not act".
    """

    def __init__(
        self,
        *,
        switch: DoorProofSwitch,
        issuer: str,
        audience: str,
        jwks: EnvelopeJwksSource,
        replay_guard: ReplayGuard,
    ) -> None:
        """
        build a door for messages from ``issuer``.

        :param switch: the door's window, from the process's door registry
        :ptype switch: DoorProofSwitch
        :param issuer: the one service this door takes messages from
        :ptype issuer: str
        :param audience: this door's name in the envelope, so an envelope for another door is refused
        :ptype audience: str
        :param jwks: the sender's trusted JWKS
        :ptype jwks: EnvelopeJwksSource
        :param replay_guard: the door's single-use ledger
        :ptype replay_guard: ReplayGuard
        :raises ValueError: when ``replay_guard`` was not sized for :data:`SERVICE_ENVELOPE_LEEWAY`,
            so a mis-sized ledger fails the boot rather than a replay
        """
        replay_guard.require_covers(SERVICE_ENVELOPE_LEEWAY)
        self._switch = switch
        self._issuer = issuer
        self._audience = audience
        self._jwks = jwks
        self._replay_guard = replay_guard

    @property
    def door(self) -> str:
        """
        answer the door's name.

        :return: the switch's door name
        :rtype: str
        """
        return self._switch.door

    async def admit(self, body: Mapping[str, Any], *, op: str) -> bool:
        """whether the receiver may act on ``body`` for ``op``.

        :param body: the message as received, parsed from its JSON; its envelope field included
        :ptype body: Mapping[str, Any]
        :param op: the operation the receiver is about to perform
        :ptype op: str
        :return: ``True`` to act, ``False`` to refuse
        :rtype: bool
        """
        envelope = body.get(SERVICE_ENVELOPE_FIELD)
        if isinstance(envelope, str) and envelope:
            admitted = await self._verified(envelope, body=body, op=op)
        else:
            admitted = (
                self._switch.judge(presented=False, claimed_sender=self._issuer) is ProofOutcome.ACCEPTED_WITHOUT_PROOF
            )
        return admitted

    async def _verified(self, envelope: str, *, body: Mapping[str, Any], op: str) -> bool:
        """verify and spend ``envelope``, logging and counting a refusal.

        :param envelope: the compact JWS
        :ptype envelope: str
        :param body: the message as received
        :ptype body: Mapping[str, Any]
        :param op: the operation
        :ptype op: str
        :return: whether it verified and was not used before
        :rtype: bool
        """
        cause = ""
        try:
            jwks = await self._jwks_naming(envelope)
            await verify_and_record(
                envelope,
                jwks=jwks,
                issuer=self._issuer,
                audience=self._audience,
                op=op,
                body=body,
                replay_guard=self._replay_guard,
            )
        except ServiceEnvelopeError as exc:
            cause, reason = "unverified", str(exc)
        except KvError as exc:
            # the ledger cannot say whether the id was used: refuse rather than admit a replay
            cause, reason = "ledger_unavailable", type(exc).__name__
        if cause:
            counter(
                _REFUSED_COUNTER,
                description="service envelopes a door refused",
                label_names=("door", "cause"),
            ).labels(door=self.door, cause=cause).inc()
            log.warning(
                "service envelope refused",
                extra={"extra_data": {"door": self.door, "op": op, "cause": cause, "reason": reason}},
            )
        return not cause

    async def _jwks_naming(self, envelope: str) -> Mapping[str, Any]:
        """the sender's JWKS, fetched afresh once when it lacks the envelope's kid (a rotated key).

        :param envelope: the compact JWS
        :ptype envelope: str
        :return: the JWKS
        :rtype: Mapping[str, Any]
        :raises ServiceEnvelopeError: when the header is malformed or no key set is available
        """
        try:
            kid = jwt.get_unverified_header(envelope).get("kid")
        except jwt.PyJWTError as exc:
            raise ServiceEnvelopeError(f"malformed envelope header ({type(exc).__name__})") from exc
        jwks = await self._jwks.get_jwks()
        if not any(isinstance(key, Mapping) and key.get("kid") == kid for key in jwks.get("keys", ())):
            jwks = await self._jwks.get_jwks(refresh=True)
        return jwks
