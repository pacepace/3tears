"""a JWKS reply is never trusted blind: what a verifier checks before a fetched key set reaches it.

A verifier fetches the key set it checks tokens against over NATS request/reply, and any client in
the account can get bytes delivered onto a subject it can name. A forged reply that adds the
forger's own key would let it mint any identity the verifier then accepts. Four rules close that:

**Thumbprint = kid.** Every platform issuer names a key by its RFC 7638 thumbprint
(:func:`~threetears.core.security.identity_token.jwk_thumbprint`). An entry whose key material does
not hash to its ``kid`` is a forgery, and the whole reply is refused: :func:`thumbprint_mismatches`.

**Add-only merge, for a reply the verifier could authenticate.** A reply whose signed statement
verified may add keys and never replaces the cache: a key missing from it is retired only once it
has been missing for :data:`ROTATION_OVERLAP`, so the publisher's own rotation cannot strand a
token it signed a moment before (:meth:`JwksKeyring.merge`). A reply the verifier could NOT
authenticate, accepted only while its door's window is open, replaces the cache exactly
(:meth:`JwksKeyring.replace`): merging it add-only would let a forged key outlive the next genuine
reply by the whole overlap.

**A signed statement.** The publisher signs a statement over the exact key set with a root key the
verifier pins, delivered to it out of band (deployment configuration, never this channel):
:func:`sign_jwks_statement` and :func:`verify_jwks_statement`. A verifier adopts the statement
through the dated window :data:`HUB_JWKS_STATEMENT_WINDOW`; before its date an unsigned reply is
accepted, counted and logged, and from its date it is refused.

**No attacker-timed refresh.** The reactive refresh a verifier makes on an unknown ``kid`` is single
flight and runs at most once per :data:`REACTIVE_REFRESH_WINDOW_SECONDS`: :class:`RefreshGate`.
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
from typing import Any, Final

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from threetears.observe import get_logger

from threetears.core.security.door_proof import DoorWindow
from threetears.core.security.identity_token import jwk_thumbprint

__all__ = [
    "HUB_JWKS_ROOT_PUBLIC_KEY_ENV",
    "HUB_JWKS_STATEMENT_WINDOW",
    "JWKS_STATEMENT_FIELD",
    "REACTIVE_REFRESH_WINDOW_SECONDS",
    "ROTATION_OVERLAP",
    "JwksKeyring",
    "JwksStatementError",
    "KeyringMerge",
    "RefreshGate",
    "hub_jwks_root_public_key_from_env",
    "jwks_keys",
    "root_public_key_from_text",
    "keys_digest",
    "sign_jwks_statement",
    "thumbprint_mismatches",
    "verify_jwks_statement",
]

log = get_logger(__name__)

#: how long a key missing from the publisher's set is still trusted. At least the longest lifetime
#: of anything signed under a platform key (the hub's user assertion, an hour), so a key the
#: publisher has just rotated out still verifies the tokens it signed before the rotation.
ROTATION_OVERLAP: Final[timedelta] = timedelta(hours=1)

#: the shortest gap between two reactive refreshes. A genuine re-key heals on the first token
#: signed under the new key, within seconds; a flood of tokens naming unknown kids costs one fetch
#: per window, never one per token.
REACTIVE_REFRESH_WINDOW_SECONDS: Final[float] = 5.0

#: the pod environment entry carrying the hub's pinned JWKS root public key: the 32 raw Ed25519
#: bytes, base64url. It reaches a pod through deployment configuration, never over NATS.
HUB_JWKS_ROOT_PUBLIC_KEY_ENV: Final[str] = "THREETEARS_HUB_JWKS_ROOT_PUBLIC_KEY"

#: the field of a JWKS reply that carries the publisher's signed statement over its ``keys``.
#: A reader that predates the statement ignores the field.
JWKS_STATEMENT_FIELD: Final[str] = "statement"

#: the hub JWKS door's window for adopting the signed statement. Until ``required_from`` a reply
#: without a statement is accepted (counted, logged); from it, refused.
HUB_JWKS_STATEMENT_WINDOW: Final[DoorWindow] = DoorWindow(
    door="hub.jwks.statement",
    opened=date(2026, 10, 7),
    required_from=date(2027, 1, 5),
)

_ALG: Final[str] = "EdDSA"
_STATEMENT_TYPE: Final[str] = "jwks-statement+jwt"
_STATEMENT_REQUIRED: Final[tuple[str, ...]] = ("iss", "iat", "exp", "keys_sha256")


class JwksStatementError(Exception):
    """a JWKS statement is malformed, signed by a key other than the pinned root, expired, or
    signed over a different key set. Carries the reason only, never key material."""


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

    An entry that is not an Ed25519 or EC public JWK, or carries no ``kid``, is a mismatch too.

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
    :param ttl: how long the statement stands
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
) -> None:
    """check ``statement`` is the pinned root's signature over exactly ``keys``.

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
    :return: nothing
    :rtype: None
    :raises JwksStatementError: when any check fails
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


@dataclass(frozen=True)
class KeyringMerge:
    """what one refresh changed in a :class:`JwksKeyring`.

    :ivar added: kids this refresh brought in
    :ivar retired: kids this refresh retired, missing for at least the rotation overlap
    :ivar held: kids missing from this refresh that are still trusted
    """

    added: tuple[str, ...]
    retired: tuple[str, ...]
    held: tuple[str, ...]


@dataclass
class JwksKeyring:
    """the key set a verifier trusts, merged add-only from successive refreshes.

    :ivar overlap: how long a key missing from the publisher's set stays trusted
    :ivar clock: monotonic seconds
    """

    overlap: timedelta = ROTATION_OVERLAP
    clock: Callable[[], float] = time.monotonic
    _keys: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _missing_since: dict[str, float] = field(default_factory=dict, init=False)

    def merge(self, keys: list[dict[str, Any]]) -> KeyringMerge:
        """fold one refresh's keys in: add new ones, hold missing ones, retire long-missing ones.

        The caller has already refused a reply whose kids are not their thumbprints, so a kid
        names one key: a known kid's material cannot change.

        :param keys: the refresh's ``keys`` entries, every one named by its thumbprint
        :ptype keys: list[dict[str, Any]]
        :return: what changed
        :rtype: KeyringMerge
        """
        now = self.clock()
        seen = {entry["kid"]: entry for entry in keys}
        added = tuple(sorted(kid for kid in seen if kid not in self._keys))
        for kid, entry in seen.items():
            self._keys.setdefault(kid, dict(entry))
            self._missing_since.pop(kid, None)
        retired: list[str] = []
        held: list[str] = []
        for kid in sorted(set(self._keys) - set(seen)):
            first_missed = self._missing_since.setdefault(kid, now)
            if now - first_missed >= self.overlap.total_seconds():
                del self._keys[kid]
                del self._missing_since[kid]
                retired.append(kid)
            else:
                held.append(kid)
        return KeyringMerge(added=added, retired=tuple(retired), held=tuple(held))

    def replace(self, keys: list[dict[str, Any]]) -> KeyringMerge:
        """trust exactly ``keys``, for a reply nothing authenticated: nothing it omits is held.

        :param keys: the refresh's ``keys`` entries, every one named by its thumbprint
        :ptype keys: list[dict[str, Any]]
        :return: what changed; ``held`` is always empty
        :rtype: KeyringMerge
        """
        seen = {entry["kid"]: dict(entry) for entry in keys}
        added = tuple(sorted(kid for kid in seen if kid not in self._keys))
        retired = tuple(sorted(kid for kid in self._keys if kid not in seen))
        self._keys = seen
        self._missing_since = {}
        return KeyringMerge(added=added, retired=retired, held=())

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
    :raises ValueError: when the entry is set but is not a base64url 32-byte Ed25519 public key
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
