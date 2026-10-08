"""a fetched JWKS is never trusted blind: what a verifier checks before a key set reaches it.

A verifier fetches the key set it checks tokens against over NATS request/reply, and any client in
the account can get bytes delivered onto a subject it can name. A forged reply that adds the
forger's own key would let it mint any identity the verifier then accepts.

:class:`JwksTrust` is the one place a verifier decides what to trust. Each reply is judged once,
into a :class:`ReplyTrust`, and its keys are adopted with their provenance:

**A signed statement.** The publisher signs a statement over exactly its key set with a root key
the verifier pins from deployment configuration, never from this channel (:func:`sign_jwks_statement`,
:func:`verify_jwks_statement`). A verified reply is :attr:`ReplyTrust.VERIFIED`. A statement is
refused when it is older than the newest one already verified by more than
:data:`STATEMENT_ORDER_LEEWAY` (replicas' clocks differ), or claims a life longer than
:data:`MAX_STATEMENT_LIFETIME`, so a stored genuine reply cannot be replayed to bring back a key the
publisher has rotated out.

**The window, and no downgrade.** A reply without a statement the verifier can check is
:attr:`ReplyTrust.UNVERIFIED`, taken only while the door's dated window is open. Once the
publisher has signed one reply, an unsigned one is refused as a downgrade: the publisher signs from
then on. This only tightens. Its cost, recorded: a publisher rolling out the signing release should
finish the rollout promptly, because a verifier that has seen one signed reply refuses the replies
of publisher replicas still on the old release; and rolling the publisher BACK to a release that
does not sign needs every long-lived verifier restarted.

**Provenance decides what a missing key costs.** :class:`JwksKeyring` holds a key missing from a
reply for :data:`ROTATION_OVERLAP` only if a verified reply brought that key in, so the publisher's
own rotation cannot strand a token it signed a moment before. A key only an unverified reply
vouched for is dropped by the first reply that omits it, verified or not: a forged key never
outlives the next genuine reply.

**Thumbprint = kid, for an unverified reply.** Every key a platform JWKS serves is named by its
RFC 7638 thumbprint (the hub identity issuer, the proxy assertion signer, identity-core's signer).
An unverified reply whose entry is not is refused from :data:`THUMBPRINT_WINDOW`'s date, and
before it accepted and counted. A verified reply needs no such check: its statement covers the keys.

**No attacker-timed refresh.** A verifier's reactive refresh on an unknown ``kid`` goes through a
:class:`RefreshGate`: single flight, at most once per :data:`REACTIVE_REFRESH_WINDOW_SECONDS`.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, Final

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from threetears.observe import get_logger

from threetears.core.security.door_proof import DoorProofSwitch, DoorWindow, ProofOutcome
from threetears.core.security.identity_token import jwk_thumbprint

__all__ = [
    "HUB_JWKS_ROOT_PUBLIC_KEY_ENV",
    "HUB_JWKS_STATEMENT_WINDOW",
    "JWKS_STATEMENT_FIELD",
    "MAX_STATEMENT_LIFETIME",
    "STATEMENT_ORDER_LEEWAY",
    "REACTIVE_REFRESH_WINDOW_SECONDS",
    "ROTATION_OVERLAP",
    "THUMBPRINT_WINDOW",
    "JwksKeyring",
    "JwksStatementError",
    "JwksTrust",
    "KeyringMerge",
    "RefreshGate",
    "ReplyJudgement",
    "ReplyTrust",
    "StatementClaims",
    "hub_jwks_root_public_key_from_env",
    "jwks_keys",
    "keys_digest",
    "root_public_key_from_text",
    "sign_jwks_statement",
    "thumbprint_mismatches",
    "verify_jwks_statement",
]

log = get_logger(__name__)

#: how long a key missing from the publisher's set is still trusted, when a verified reply brought
#: it in. At least the longest lifetime of anything signed under a platform key (the hub's user
#: assertion, an hour), so a key the publisher has just rotated out still verifies the tokens it
#: signed before the rotation.
ROTATION_OVERLAP: Final[timedelta] = timedelta(hours=1)

#: the shortest gap between two reactive refreshes. A genuine re-key heals on the first token
#: signed under the new key, within seconds; a flood of tokens naming unknown kids costs one fetch
#: per window, never one per token.
REACTIVE_REFRESH_WINDOW_SECONDS: Final[float] = 5.0

#: the longest life a verifier accepts a statement claiming. A publisher signs a fresh statement
#: per reply, so this needs to outlive only a round trip and clock skew; a longer claimed life
#: would let a stored reply stand for the key set long after it changed.
MAX_STATEMENT_LIFETIME: Final[timedelta] = timedelta(minutes=15)

#: how far behind the newest verified statement a statement may be issued and still be taken.
#: Replicas behind one queue group each stamp ``iat`` from their own clock, so genuine replies
#: arrive out of order by up to the replicas' clock skew; refusing every one of those would read
#: as a forgery signal on each fetch that lands on the slower replica. A replay inside this leeway
#: can bring back only a key the publisher served within it, which a verified key's rotation
#: overlap holds anyway.
STATEMENT_ORDER_LEEWAY: Final[timedelta] = timedelta(seconds=30)

#: the pod environment entry carrying the hub's pinned JWKS root public key: the 32 raw Ed25519
#: bytes, base64 or base64url. It reaches a pod through deployment configuration, never over NATS.
HUB_JWKS_ROOT_PUBLIC_KEY_ENV: Final[str] = "THREETEARS_HUB_JWKS_ROOT_PUBLIC_KEY"

#: the field of a JWKS reply that carries the publisher's signed statement over its ``keys``.
#: A reader that predates the statement ignores the field.
JWKS_STATEMENT_FIELD: Final[str] = "statement"

#: the hub JWKS door's window for adopting the signed statement. Until ``required_from`` a reply
#: without one is accepted (counted, logged); from it, refused. Every verifier -- the registry,
#: every tool pod, the gateway -- must carry :data:`HUB_JWKS_ROOT_PUBLIC_KEY_ENV` before that date,
#: or it refuses every reply from it.
HUB_JWKS_STATEMENT_WINDOW: Final[DoorWindow] = DoorWindow(
    door="hub.jwks.statement",
    opened=date(2026, 10, 7),
    required_from=date(2027, 1, 5),
)

#: the window for refusing an unverified reply whose key is not named by its thumbprint. Until
#: ``required_from`` such a reply is accepted (counted, logged); from it, refused.
THUMBPRINT_WINDOW: Final[DoorWindow] = DoorWindow(
    door="hub.jwks.thumbprint",
    opened=date(2026, 10, 7),
    required_from=date(2027, 1, 5),
)

_ALG: Final[str] = "EdDSA"
_STATEMENT_TYPE: Final[str] = "jwks-statement+jwt"
_STATEMENT_REQUIRED: Final[tuple[str, ...]] = ("iss", "iat", "exp", "keys_sha256")


class JwksStatementError(Exception):
    """a JWKS statement is malformed, signed by a key other than the pinned root, expired, or
    signed over a different key set. Carries the reason only, never key material."""


@dataclass(frozen=True)
class StatementClaims:
    """what a verified statement says about itself.

    :ivar iat: when it was signed, unix seconds
    :ivar exp: when it stops standing, unix seconds
    """

    iat: int
    exp: int


def keys_digest(keys: list[dict[str, Any]]) -> str:
    """base64url SHA-256 of the canonical JSON of a key list, sorted by ``kid``.

    The publisher signs this and the verifier recomputes it from the keys it received; both go
    through this one function so their canonical forms cannot diverge.

    :param keys: the JWKS ``keys`` entries
    :ptype keys: list[dict[str, Any]]
    :return: the digest, unpadded base64url
    :rtype: str
    """
    ordered = sorted(keys, key=lambda entry: str(entry.get("kid", "")))
    canonical = json.dumps(ordered, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return str(base64.urlsafe_b64encode(hashlib.sha256(canonical.encode("utf-8")).digest()).rstrip(b"="), "ascii")


def thumbprint_mismatches(keys: list[Any]) -> list[str]:
    """every entry whose key material does not hash to its ``kid``, as a reason per entry.

    An entry that is not an Ed25519 public JWK, or carries no ``kid``, is a mismatch too.

    :param keys: the JWKS ``keys`` entries
    :ptype keys: list[Any]
    :return: one line per bad entry, empty when every entry is named by its thumbprint
    :rtype: list[str]
    """
    bad: list[str] = []
    for index, entry in enumerate(keys):
        if not isinstance(entry, dict) or not isinstance(entry.get("kid"), str) or not entry["kid"]:
            bad.append(f"entry {index}: no kid")
            continue
        try:
            key = jwt.PyJWK.from_dict(entry).key
        except (jwt.exceptions.PyJWKError, jwt.exceptions.InvalidKeyError, KeyError, TypeError, ValueError) as exc:
            bad.append(f"entry {index}: not a public JWK ({type(exc).__name__})")
            continue
        if not isinstance(key, Ed25519PublicKey):
            bad.append(f"entry {index}: not an Ed25519 public key")
            continue
        if jwk_thumbprint(key) != entry["kid"]:
            bad.append(f"entry {index}: kid is not the key's thumbprint")
    return bad


def sign_jwks_statement(
    keys: list[dict[str, Any]],
    *,
    signing_key: Ed25519PrivateKey,
    issuer: str,
    now: int,
    ttl: timedelta,
) -> str:
    """sign a statement that ``keys`` is the publisher's key set, under the pinned root key.

    :param keys: the JWKS ``keys`` entries the reply carries
    :ptype keys: list[dict[str, Any]]
    :param signing_key: the root private key whose public half verifiers pin
    :ptype signing_key: Ed25519PrivateKey
    :param issuer: the publisher's name, the statement's ``iss``
    :ptype issuer: str
    :param now: unix seconds
    :ptype now: int
    :param ttl: how long the statement stands; at most :data:`MAX_STATEMENT_LIFETIME`
    :ptype ttl: timedelta
    :return: a compact EdDSA JWS
    :rtype: str
    """
    payload = {
        "iss": issuer,
        "iat": now,
        "exp": now + int(ttl.total_seconds()),
        "keys_sha256": keys_digest(keys),
    }
    headers = {"kid": jwk_thumbprint(signing_key.public_key()), "typ": _STATEMENT_TYPE}
    return jwt.encode(payload, key=signing_key, algorithm=_ALG, headers=headers)


def verify_jwks_statement(
    statement: str,
    keys: list[dict[str, Any]],
    *,
    root_public_key: Ed25519PublicKey,
    issuer: str,
    leeway_seconds: int = 30,
) -> StatementClaims:
    """check ``statement`` is the pinned root's signature over exactly ``keys``, and say when it was made.

    :param statement: the compact JWS from the reply
    :ptype statement: str
    :param keys: the JWKS ``keys`` entries the reply carries
    :ptype keys: list[dict[str, Any]]
    :param root_public_key: the pinned root key
    :ptype root_public_key: Ed25519PublicKey
    :param issuer: the publisher the statement must name
    :ptype issuer: str
    :param leeway_seconds: clock-skew tolerance on ``exp`` and ``iat``
    :ptype leeway_seconds: int
    :return: the statement's issue and expiry times
    :rtype: StatementClaims
    :raises JwksStatementError: when any check fails, including a claimed life longer than
        :data:`MAX_STATEMENT_LIFETIME`
    """
    try:
        header = jwt.get_unverified_header(statement)
    except jwt.PyJWTError as exc:
        raise JwksStatementError(f"malformed statement header ({type(exc).__name__})") from None
    if header.get("alg") != _ALG:
        raise JwksStatementError("unexpected statement algorithm; only EdDSA is accepted")
    if header.get("typ") != _STATEMENT_TYPE:
        raise JwksStatementError("not a JWKS statement")
    if header.get("kid") != jwk_thumbprint(root_public_key):
        raise JwksStatementError("statement is not signed by the pinned root key")
    try:
        payload = jwt.decode(
            statement,
            key=root_public_key,
            algorithms=["EdDSA"],  # literal pin -- statically auditable; never widen
            issuer=issuer,
            leeway=leeway_seconds,
            options={"require": list(_STATEMENT_REQUIRED)},
        )
    except jwt.PyJWTError as exc:
        raise JwksStatementError(f"statement verification failed ({type(exc).__name__})") from None
    if payload.get("keys_sha256") != keys_digest(keys):
        raise JwksStatementError("statement is signed over a different key set")
    iat, exp = int(payload["iat"]), int(payload["exp"])
    if exp - iat > MAX_STATEMENT_LIFETIME.total_seconds():
        raise JwksStatementError("statement claims a longer life than a verifier accepts")
    return StatementClaims(iat=iat, exp=exp)


@dataclass(frozen=True)
class KeyringMerge:
    """what one adoption changed in a :class:`JwksKeyring`.

    :ivar added: kids this reply brought in
    :ivar retired: kids this reply retired
    :ivar held: kids missing from this reply that are still trusted
    """

    added: tuple[str, ...]
    retired: tuple[str, ...]
    held: tuple[str, ...]


@dataclass
class JwksKeyring:
    """the key set a verifier trusts, each key with where it came from.

    :ivar overlap: how long a key a verified reply brought in stays trusted once missing
    :ivar clock: monotonic seconds
    """

    overlap: timedelta = ROTATION_OVERLAP
    clock: Callable[[], float] = time.monotonic
    _keys: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _verified: set[str] = field(default_factory=set, init=False)
    _missing_since: dict[str, float] = field(default_factory=dict, init=False)

    def adopt(self, keys: list[dict[str, Any]], *, verified: bool) -> KeyringMerge:
        """fold one taken reply's keys in.

        A verified reply is authoritative for every kid it names: its material replaces whatever
        that kid held, and the kid is marked verified. An unverified reply sets the material of a kid
        only an unverified reply vouched for, and never touches a kid holding verified material, so
        it cannot bind a known kid to a forger's key. A key missing from a reply is held until it has
        been missing for the overlap only if a verified reply brought it in; any other missing key
        -- one only an unverified reply vouched for -- is dropped at once.

        :param keys: the reply's ``keys`` entries
        :ptype keys: list[dict[str, Any]]
        :param verified: whether the reply's statement verified
        :ptype verified: bool
        :return: what changed
        :rtype: KeyringMerge
        """
        now = self.clock()
        seen = {entry["kid"]: entry for entry in keys if isinstance(entry, dict) and isinstance(entry.get("kid"), str)}
        added = tuple(sorted(kid for kid in seen if kid not in self._keys))
        for kid, entry in seen.items():
            if verified or kid not in self._verified:
                # the one place a kid is bound to material: a verified reply always, an unverified
                # reply only for a kid no verified reply has vouched for
                self._keys[kid] = dict(entry)
            self._missing_since.pop(kid, None)
            if verified:
                self._verified.add(kid)
        retired: list[str] = []
        held: list[str] = []
        for kid in sorted(set(self._keys) - set(seen)):
            first_missed = self._missing_since.setdefault(kid, now)
            if kid not in self._verified or now - first_missed >= self.overlap.total_seconds():
                del self._keys[kid]
                del self._missing_since[kid]
                self._verified.discard(kid)
                retired.append(kid)
            else:
                held.append(kid)
        return KeyringMerge(added=added, retired=tuple(retired), held=tuple(held))

    def document(self) -> dict[str, Any]:
        """the trusted keys as a JWKS document, a fresh copy each call.

        :return: ``{"keys": [...]}``
        :rtype: dict[str, Any]
        """
        return {"keys": [dict(entry) for _, entry in sorted(self._keys.items())]}

    def __contains__(self, kid: object) -> bool:
        """whether ``kid`` is trusted.

        :param kid: a key id
        :ptype kid: object
        :return: True when trusted
        :rtype: bool
        """
        return kid in self._keys


class ReplyTrust(StrEnum):
    """what one reply is worth."""

    #: refused: nothing in it is adopted
    REFUSED = "refused"
    #: its statement verified under the pinned root
    VERIFIED = "verified"
    #: no statement the verifier could check, taken because the door's window is still open
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ReplyJudgement:
    """the verdict on one reply.

    :ivar trust: what it is worth
    :ivar reason: why it was refused, empty otherwise
    """

    trust: ReplyTrust
    reason: str = ""


class JwksTrust:
    """everything a verifier trusts about one publisher's key set, decided in one place.

    :param door: the publisher's door name, for logs
    :ptype door: str
    :param issuer: the ``iss`` the publisher's statement names
    :ptype issuer: str
    :param root_public_key: the publisher's pinned root key, or ``None`` when this verifier has none
    :ptype root_public_key: Ed25519PublicKey | None
    :param statement_switch: the switch of the door's statement window
    :ptype statement_switch: DoorProofSwitch
    :param thumbprint_switch: the switch of the thumbprint window
    :ptype thumbprint_switch: DoorProofSwitch
    :param keyring: the key set; a fresh one when not given
    :ptype keyring: JwksKeyring | None
    """

    def __init__(
        self,
        *,
        door: str,
        issuer: str,
        root_public_key: Ed25519PublicKey | None,
        statement_switch: DoorProofSwitch,
        thumbprint_switch: DoorProofSwitch,
        keyring: JwksKeyring | None = None,
    ) -> None:
        self._door = door
        self._issuer = issuer
        self._root = root_public_key
        self._statement_switch = statement_switch
        self._thumbprint_switch = thumbprint_switch
        self._keyring = keyring if keyring is not None else JwksKeyring()
        #: the newest verified statement's issue time; set once the publisher has signed a reply,
        #: after which an unsigned one is a downgrade
        self._newest_statement_iat: int | None = None
        if root_public_key is None:
            required_from = statement_switch.window.required_from.isoformat()
            log.warning(
                "%s verifier has no pinned root key: it cannot check the publisher's statement, and refuses "
                "every reply from %s",
                door,
                required_from,
                extra={"extra_data": {"door": door, "required_from": required_from}},
            )

    @property
    def keyring(self) -> JwksKeyring:
        """the trusted key set.

        :return: the keyring
        :rtype: JwksKeyring
        """
        return self._keyring

    def take(self, document: Mapping[str, Any]) -> tuple[ReplyJudgement, KeyringMerge | None]:
        """judge one reply and adopt its keys when it is taken.

        :param document: the decoded reply
        :ptype document: Mapping[str, Any]
        :return: the verdict, and what adopting the reply changed (``None`` when refused)
        :rtype: tuple[ReplyJudgement, KeyringMerge | None]
        """
        judgement = self._judge(document)
        merge: KeyringMerge | None = None
        if judgement.trust is ReplyTrust.REFUSED:
            log.warning(
                "%s reply refused: %s",
                self._door,
                judgement.reason,
                extra={
                    "extra_data": {
                        "door": self._door,
                        "reason": judgement.reason,
                        "root_configured": self._root is not None,
                    }
                },
            )
        else:
            merge = self._keyring.adopt(jwks_keys(document), verified=judgement.trust is ReplyTrust.VERIFIED)
        return judgement, merge

    def _judge(self, document: Mapping[str, Any]) -> ReplyJudgement:
        """what one reply is worth; adopts nothing.

        :param document: the decoded reply
        :ptype document: Mapping[str, Any]
        :return: the verdict
        :rtype: ReplyJudgement
        """
        keys = document.get("keys")
        statement = document.get(JWKS_STATEMENT_FIELD)
        judgement: ReplyJudgement
        if not isinstance(keys, list):
            judgement = ReplyJudgement(ReplyTrust.REFUSED, "keys is not a list")
        elif isinstance(statement, str) and self._root is not None:
            judgement = self._judge_statement(statement, keys)
        elif self._newest_statement_iat is not None:
            judgement = ReplyJudgement(ReplyTrust.REFUSED, "no statement, after the publisher has signed one")
        else:
            if isinstance(statement, str):
                log.warning(
                    "%s reply carries a statement this verifier cannot check: it has no pinned root key",
                    self._door,
                    extra={"extra_data": {"door": self._door}},
                )
            judgement = self._judge_unverified(keys)
        return judgement

    def _judge_statement(self, statement: str, keys: list[Any]) -> ReplyJudgement:
        """the verdict on a reply carrying a statement this verifier can check.

        :param statement: the statement
        :ptype statement: str
        :param keys: the reply's keys
        :ptype keys: list[Any]
        :return: the verdict
        :rtype: ReplyJudgement
        """
        assert self._root is not None  # the caller checked
        judgement = ReplyJudgement(ReplyTrust.VERIFIED)
        try:
            claims = verify_jwks_statement(statement, keys, root_public_key=self._root, issuer=self._issuer)
        except JwksStatementError as exc:
            judgement = ReplyJudgement(ReplyTrust.REFUSED, f"statement does not verify: {exc}")
        else:
            newest = self._newest_statement_iat
            if newest is not None and claims.iat < newest - STATEMENT_ORDER_LEEWAY.total_seconds():
                judgement = ReplyJudgement(ReplyTrust.REFUSED, "statement is older than one already verified")
            else:
                self._newest_statement_iat = claims.iat if newest is None else max(newest, claims.iat)
                self._statement_switch.judge(presented=True, claimed_sender=self._issuer)
        return judgement

    def _judge_unverified(self, keys: list[Any]) -> ReplyJudgement:
        """the verdict on a reply this verifier cannot authenticate, through both windows.

        :param keys: the reply's keys
        :ptype keys: list[Any]
        :return: the verdict
        :rtype: ReplyJudgement
        """
        judgement = ReplyJudgement(ReplyTrust.UNVERIFIED)
        statement_outcome = self._statement_switch.judge(presented=False, claimed_sender=self._issuer)
        mismatches = thumbprint_mismatches(keys)
        if statement_outcome is ProofOutcome.REFUSED_WITHOUT_PROOF:
            judgement = ReplyJudgement(
                ReplyTrust.REFUSED, "no statement this verifier can check, and the door requires one"
            )
        elif mismatches:
            outcome = self._thumbprint_switch.judge(presented=False, claimed_sender=self._issuer)
            if outcome is ProofOutcome.REFUSED_WITHOUT_PROOF:
                judgement = ReplyJudgement(
                    ReplyTrust.REFUSED, "an entry's kid is not its key's thumbprint: " + "; ".join(mismatches)
                )
            else:
                log.warning(
                    "%s reply has entries not named by their thumbprint; taken while the window is open",
                    self._door,
                    extra={"extra_data": {"door": self._door, "entries": mismatches}},
                )
        return judgement


class RefreshGate:
    """single-flight, rate-limited reactive refresh: one fetch per window, whatever the callers.

    :param refresh: the fetch to run
    :ptype refresh: Callable[[], Awaitable[None]]
    :param window_seconds: the shortest gap between two refreshes this gate runs
    :ptype window_seconds: float
    :param clock: monotonic seconds
    :ptype clock: Callable[[], float]
    """

    def __init__(
        self,
        refresh: Callable[[], Awaitable[None]],
        *,
        window_seconds: float = REACTIVE_REFRESH_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._refresh = refresh
        self._window = window_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last: float | None = None

    def _recent(self) -> bool:
        """whether a refresh this gate ran is still inside the window.

        :return: True inside the window
        :rtype: bool
        """
        return self._last is not None and (self._clock() - self._last) < self._window

    async def run(self) -> bool:
        """run the refresh unless one ran within the window; concurrent callers share one.

        :return: True when this call ran the refresh
        :rtype: bool
        """
        if self._recent():
            return False
        async with self._lock:
            if self._recent():
                return False
            try:
                await self._refresh()
            finally:
                self._last = self._clock()
            return True


def jwks_keys(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """the ``keys`` list of a JWKS reply, or an empty list for a reply without one.

    :param document: the decoded reply
    :ptype document: Mapping[str, Any]
    :return: the entries
    :rtype: list[dict[str, Any]]
    """
    keys = document.get("keys")
    return list(keys) if isinstance(keys, list) else []


def hub_jwks_root_public_key_from_env(environ: Mapping[str, str] | None = None) -> Ed25519PublicKey | None:
    """the hub's pinned JWKS root key from :data:`HUB_JWKS_ROOT_PUBLIC_KEY_ENV`, or ``None`` when unset.

    :param environ: the environment to read; the process environment when not given
    :ptype environ: Mapping[str, str] | None
    :return: the public key, or ``None`` when this pod was given none
    :rtype: Ed25519PublicKey | None
    :raises ValueError: when the entry is set but is not a base64 32-byte Ed25519 public key
    """
    raw = (os.environ if environ is None else environ).get(HUB_JWKS_ROOT_PUBLIC_KEY_ENV, "").strip()
    return root_public_key_from_text(raw, what=HUB_JWKS_ROOT_PUBLIC_KEY_ENV) if raw else None


def root_public_key_from_text(raw: str, *, what: str) -> Ed25519PublicKey:
    """a pinned JWKS root key from its configured text: the 32 raw Ed25519 bytes, base64 or base64url.

    :param raw: the configured value
    :ptype raw: str
    :param what: the setting's name, for the error
    :ptype what: str
    :return: the public key
    :rtype: Ed25519PublicKey
    :raises ValueError: when the value is not a 32-byte Ed25519 public key in either encoding
    """
    text = raw.strip().replace("+", "-").replace("/", "_")
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        key = Ed25519PublicKey.from_public_bytes(decoded)
    except ValueError as exc:
        raise ValueError(f"{what} is not a base64 32-byte Ed25519 public key") from exc
    return key
