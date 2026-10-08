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

from threetears.core.security.door_proof import DoorWindow, ProofOutcome
from threetears.core.security.identity_token import build_jwks, jwk_thumbprint
from threetears.core.security.freshness import ISSUE_TIME_FUTURE_TOLERANCE
from threetears.core.security.secret_refs import SecretResolutionError, resolve_secret

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.core.security.door_proof import DoorProofSwitch

__all__ = [
    "SERVICE_ENVELOPE_FIELD",
    "SERVICE_ENVELOPE_LEEWAY",
    "SERVICE_ENVELOPE_MAX_LIFETIME",
    "SERVICE_ENVELOPE_REPLAY_TTL",
    "EnvelopeJwksSource",
    "EnvelopeLedger",
    "HubJwksEnvelopeSource",
    "InProcessEnvelopeLedger",
    "PinnedEnvelopeJwks",
    "ServiceEnvelopeClaims",
    "ServiceEnvelopeDoor",
    "ServiceEnvelopeError",
    "ServiceEnvelopeSigner",
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


@dataclass(frozen=True, slots=True)
class ServiceEnvelopeSigner:
    """one sending service's key, for every door it writes to.

    :ivar signing_key: the service's Ed25519 key, published in its JWKS under ``kid``
    :ivar kid: the key's JWKS id
    :ivar issuer: the service principal the doors expect
    """

    signing_key: Ed25519PrivateKey
    kid: str
    issuer: str

    @classmethod
    def from_seed_text(cls, seed: str, *, issuer: str, what: str) -> ServiceEnvelopeSigner:
        """a sender's signer from its configured seed: 32 raw Ed25519 bytes, base64 or base64url.

        The kid is the public key's RFC 7638 thumbprint, the one :class:`PinnedEnvelopeJwks` names
        it by, so the pinned public half alone is enough for a door to find the key.

        :param seed: the configured seed; never echoed
        :ptype seed: str
        :param issuer: the sending service principal
        :ptype issuer: str
        :param what: the setting's name, for the error
        :ptype what: str
        :return: the signer
        :rtype: ServiceEnvelopeSigner
        :raises ValueError: when the value is not a 32-byte seed in either encoding
        """
        text = seed.strip().replace("+", "-").replace("/", "_")
        try:
            key = Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))
        except ValueError:
            raise ValueError(f"{what} is not a base64 32-byte Ed25519 seed") from None
        return cls(signing_key=key, kid=jwk_thumbprint(key.public_key()), issuer=issuer)

    @classmethod
    def from_secret_ref(cls, ref: str, *, issuer: str, what: str, window: DoorWindow) -> ServiceEnvelopeSigner | None:
        """a sender's signer from its seed's secret reference, or ``None`` while it is not provisioned.

        Unprovisioned, the sender sends unsigned, which the receiving door accepts only until
        ``window``'s date; the warning says so, naming the setting. A provisioned seed that is
        malformed fails, like every other signing key.

        :param ref: the secret reference (``scheme://locator``) to the base64 seed
        :ptype ref: str
        :param issuer: the sending service principal
        :ptype issuer: str
        :param what: the setting that provisions the seed, for the warning and the error
        :ptype what: str
        :param window: the receiving door's window, whose date the warning names
        :ptype window: DoorWindow
        :return: the signer, or ``None``
        :rtype: ServiceEnvelopeSigner | None
        :raises ValueError: when the referenced secret is not a 32-byte seed
        """
        signer: ServiceEnvelopeSigner | None = None
        try:
            secret = resolve_secret(ref)
        except SecretResolutionError:
            log.warning(
                "%s is not provisioned: messages to the %s door go unsigned, and it refuses them from %s",
                what,
                window.door,
                window.required_from.isoformat(),
            )
        else:
            signer = cls.from_seed_text(secret.get_secret_value(), issuer=issuer, what=what)
        return signer

    def sign(self, *, audience: str, op: str, body: Mapping[str, Any]) -> str:
        """an envelope over ``body`` for ``audience``'s ``op``.

        :param audience: the receiving door
        :ptype audience: str
        :param op: the operation within the door
        :ptype op: str
        :param body: the message body as it goes on the wire, as JSON values
        :ptype body: Mapping[str, Any]
        :return: the compact JWS
        :rtype: str
        """
        return mint_service_envelope(
            signing_key=self.signing_key, kid=self.kid, issuer=self.issuer, audience=audience, op=op, body=body
        )

    def seal(self, *, audience: str, op: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """``body`` with its envelope added under :data:`SERVICE_ENVELOPE_FIELD`, ready to send.

        :param audience: the receiving door
        :ptype audience: str
        :param op: the operation within the door
        :ptype op: str
        :param body: the message body, as JSON values
        :ptype body: Mapping[str, Any]
        :return: the body with its envelope
        :rtype: dict[str, Any]
        """
        return {**body, SERVICE_ENVELOPE_FIELD: self.sign(audience=audience, op=op, body=body)}


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
    replay_guard: EnvelopeLedger,
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
    :ptype replay_guard: EnvelopeLedger
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


class EnvelopeLedger(Protocol):
    """where a door spends envelope ids: a shared ``ReplayGuard``, or one process's own ledger."""

    def require_covers(self, future_tolerance: timedelta) -> None:
        """refuse a verifier whose future tolerance the ledger was not sized for.

        :param future_tolerance: the verifier's tolerance
        :ptype future_tolerance: timedelta
        :raises ValueError: when it exceeds the ledger's sizing
        """
        ...

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        """spend ``nonce``; ``False`` when it was spent before (or must be refused).

        :param nonce: the id
        :ptype nonce: str
        :param issued_at: the envelope's issue time, timezone-aware
        :ptype issued_at: datetime
        :return: whether it was fresh
        :rtype: bool
        """
        ...


class InProcessEnvelopeLedger:
    """one process's ledger of spent envelope ids, for a door only this process consumes.

    A runner's own inbox has exactly one consumer per runner, so the ids it has spent are the whole
    ledger: a replay can only reach this process again. Ids are forgotten once their envelope could
    no longer be accepted (:data:`SERVICE_ENVELOPE_REPLAY_TTL` after issue), so the ledger stays
    bounded. Nothing survives a restart, and nothing needs to: a restarted runner holds a new
    identity generation, which a replayed refusal does not name.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        """
        start with nothing spent.

        :param clock: the current time, timezone-aware; the system clock when ``None``
        :ptype clock: Callable[[], datetime] | None
        """
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._spent: dict[str, datetime] = {}

    def require_covers(self, future_tolerance: timedelta) -> None:
        """accept any verifier tolerance up to the envelope's own leeway.

        :param future_tolerance: the verifier's tolerance
        :ptype future_tolerance: timedelta
        :raises ValueError: when it exceeds :data:`SERVICE_ENVELOPE_LEEWAY`, past which an id could be
            forgotten while its envelope is still accepted
        """
        if future_tolerance > SERVICE_ENVELOPE_LEEWAY:
            raise ValueError(
                f"InProcessEnvelopeLedger is sized for a verifier future tolerance of {SERVICE_ENVELOPE_LEEWAY}, "
                f"not {future_tolerance}"
            )

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        """spend ``nonce``, forgetting ids whose envelopes have expired.

        :param nonce: the id
        :ptype nonce: str
        :param issued_at: the envelope's issue time, timezone-aware
        :ptype issued_at: datetime
        :return: whether it was fresh
        :rtype: bool
        :raises ValueError: when ``issued_at`` is timezone-naive
        """
        if issued_at.tzinfo is None:
            raise ValueError("InProcessEnvelopeLedger.record_unique requires a timezone-aware issued_at")
        horizon = self._clock() - SERVICE_ENVELOPE_REPLAY_TTL
        self._spent = {key: when for key, when in self._spent.items() if when >= horizon}
        fresh = nonce not in self._spent
        if fresh:
            self._spent[nonce] = issued_at
        return fresh


class HubJwksEnvelopeSource:
    """the hub's key set for a door on a pod, read from the pod's trusted ``CachedHubJwksProvider``.

    The provider takes a JWKS reply only under the hub's root statement, so an envelope verifies
    only under a key the hub published.
    """

    def __init__(self, provider: Any) -> None:
        """
        wrap the pod's provider.

        :param provider: a ``CachedHubJwksProvider``: called for the cached set, ``refresh_now`` to fetch
        :ptype provider: Any
        """
        self._provider = provider

    async def get_jwks(self, *, refresh: bool = False) -> Mapping[str, Any]:
        """the hub's trusted JWKS; a refresh asks the provider for one (rate-limited there).

        :param refresh: ask for a fresh fetch
        :ptype refresh: bool
        :return: the JWKS, empty until the provider's first fetch (nothing verifies then)
        :rtype: Mapping[str, Any]
        """
        if refresh:
            await self._provider.refresh_now()
        jwks: Mapping[str, Any] = self._provider()
        return jwks


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


class PinnedEnvelopeJwks:
    """a sender's key set from its one public key, pinned in the receiver's configuration.

    For a sender that publishes no JWKS of its own (the gateway, a channel adapter): the key is
    named by its thumbprint, as :meth:`ServiceEnvelopeSigner.from_seed_text` names it. Without a
    pinned key no envelope from the sender verifies.
    """

    def __init__(self, public_key: Ed25519PublicKey | None) -> None:
        """
        pin the sender's key.

        :param public_key: the sender's public key, or ``None`` while it is not provisioned
        :ptype public_key: Ed25519PublicKey | None
        """
        self._jwks = None if public_key is None else build_jwks({jwk_thumbprint(public_key): public_key})

    async def get_jwks(self, *, refresh: bool = False) -> Mapping[str, Any]:
        """the pinned key set; a refresh changes nothing.

        :param refresh: ignored: nothing is fetched
        :ptype refresh: bool
        :return: the JWKS
        :rtype: Mapping[str, Any]
        :raises ServiceEnvelopeError: when no key is pinned
        """
        if self._jwks is None:
            raise ServiceEnvelopeError("no public key is pinned for this sender")
        return self._jwks


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
        replay_guard: EnvelopeLedger,
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
        :ptype replay_guard: EnvelopeLedger
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

    async def admit_message(self, data: bytes, *, op: str) -> bool:
        """whether the receiver may act on the message ``data`` for ``op``, hashed exactly as sent.

        The door to use when the receiver's own model of the message is a tolerant reader: a model
        that ignores a field it does not know would hash less than the sender signed, so a sender a
        release ahead, adding a field, would be refused. Here every field as sent counts.

        :param data: the message bytes, a JSON object
        :ptype data: bytes
        :param op: the operation the receiver is about to perform
        :ptype op: str
        :return: ``True`` to act, ``False`` to refuse; a body that is not a JSON object is refused
        :rtype: bool
        """
        try:
            body = json.loads(data)
        except ValueError, UnicodeDecodeError:
            body = None
        return isinstance(body, dict) and await self.admit(body, op=op)

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
