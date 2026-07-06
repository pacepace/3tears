"""Holds an Ed25519 signing key and self-mints short-lived identity JWTs.

The stateful counterpart to :mod:`identity_token` (which is pure sign / verify / JWKS, no custody).
A self-minting principal — a scriob pod, an aibots agent pod, an aibots tool pod — holds its own
Ed25519 key and mints a short-lived identity JWT it presents as its NATS connect credential. It
re-mints on every (re)connect via ``NatsClient.connect``'s token provider, so the connection never
re-presents a credential that has since expired. The auth-callout responder verifies each minted
token against the matching JWKS (this minter's :meth:`jwks`, keyed on the same ``kid``).

Custody lives here (the private key is held); the signing itself delegates to the pure
:func:`sign_identity_token`. Mirrors :class:`~threetears.core.security.proxy_signer.ProxyAssertionSigner`
(a pure mint function plus a stateful, key-holding signer). The private key never leaves the minter
and is never logged.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from threetears.core.security.identity_token import (
    IdentityClaims,
    IdentityTokenError,
    build_jwks,
    generate_signing_keypair,
    sign_identity_token,
)

__all__ = ["DEFAULT_IDENTITY_TTL_SECONDS", "IdentityMinter", "static_token_provider"]

#: fallback identity-JWT lifetime. The token is a CONNECT credential re-minted on every reconnect
#: (the token provider), so a short TTL just means more frequent transparent re-mints, never
#: downtime — it only has to outlive the re-auth interval. Callers pass their own value.
DEFAULT_IDENTITY_TTL_SECONDS = 3600


def static_token_provider(token: str) -> Callable[[], str]:
    """wrap a STANDING static connect token in the zero-arg provider ``NatsClient.connect`` requires.

    ``NatsClient.connect``'s ``auth_token`` is a PROVIDER — a zero-arg callable nats-py invokes on
    every (re)connect — never a bare string. A platform service that authenticates with a fixed,
    standing credential (the Hub, the model gateway; they bypass the auth-callout) has no per-reconnect
    minting to do, so it wraps its one token in this constant provider: every call returns the same
    string. Self-minting principals (agent / tool pods) use :class:`IdentityMinter` instead, whose
    provider mints a FRESH short-lived token per reconnect. The token is captured by value and never
    logged.

    :param token: the fixed NATS auth token the provider returns on every invocation.
    :ptype token: str
    :return: a zero-arg callable that returns ``token`` unchanged on every call.
    :rtype: Callable[[], str]
    """
    return lambda: token


class IdentityMinter:
    """mints a principal's EdDSA identity JWT from a custody-held Ed25519 key.

    :param signing_key: the principal's Ed25519 signing key (deploy custody; never generated on a
        live mint path — use :meth:`from_pem`, or :meth:`generate` for tests/dev only).
    :ptype signing_key: Ed25519PrivateKey
    :param kid: the key id stamped in each token header + the JWKS entry (the resolver looks the
        verifying key up by it; e.g. the pod/agent id).
    :ptype kid: str
    :param issuer: the ``iss`` claim every minted token carries; the resolver pins it.
    :ptype issuer: str
    :param ttl_seconds: the lifetime each minted token carries.
    :ptype ttl_seconds: int
    """

    def __init__(
        self,
        signing_key: Ed25519PrivateKey,
        *,
        kid: str,
        issuer: str,
        ttl_seconds: int = DEFAULT_IDENTITY_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise IdentityTokenError("ttl_seconds must be positive")
        self._signing_key = signing_key
        self._kid = kid
        self._issuer = issuer
        self._ttl_seconds = ttl_seconds

    @classmethod
    def from_pem(
        cls,
        pem: str | bytes,
        *,
        kid: str,
        issuer: str,
        ttl_seconds: int = DEFAULT_IDENTITY_TTL_SECONDS,
    ) -> IdentityMinter:
        """load the signing key from a PKCS#8 PEM secret (deploy custody). FAILS CLOSED on a non-Ed25519 key.

        :param pem: the PKCS#8 PEM of the Ed25519 private signing key (never logged).
        :ptype pem: str | bytes
        :param kid: the key id stamped in each token + the JWKS entry.
        :ptype kid: str
        :param issuer: the pinned ``iss`` claim.
        :ptype issuer: str
        :param ttl_seconds: minted-token lifetime.
        :ptype ttl_seconds: int
        :return: a ready minter.
        :rtype: IdentityMinter
        :raises IdentityTokenError: when the PEM is not an unencrypted Ed25519 private key.
        """
        raw = pem.encode("utf-8") if isinstance(pem, str) else pem
        try:
            key = load_pem_private_key(raw, password=None)
        except ValueError as exc:
            raise IdentityTokenError(f"invalid identity signing key ({type(exc).__name__})") from None
        if not isinstance(key, Ed25519PrivateKey):
            raise IdentityTokenError("identity signing key must be an Ed25519 private key")
        return cls(key, kid=kid, issuer=issuer, ttl_seconds=ttl_seconds)

    @classmethod
    def generate(
        cls,
        *,
        kid: str,
        issuer: str,
        ttl_seconds: int = DEFAULT_IDENTITY_TTL_SECONDS,
    ) -> IdentityMinter:
        """generate a fresh keypair — for tests / the dev stack, NOT a production custody path."""
        signing_key, _ = generate_signing_keypair()
        return cls(signing_key, kid=kid, issuer=issuer, ttl_seconds=ttl_seconds)

    def mint(self, subject: str, *, customer_id: str, pod_id: str | None = None, now: int | None = None) -> str:
        """mint a short-lived identity JWT for ``subject`` (the authenticated principal id).

        ``sub`` is the principal id (the resolver scopes on it); ``pod_id`` defaults to ``subject``
        when the principal is its own pod. A fresh ``sid`` per mint (the schema requires a session id).

        :param subject: the principal id the token authenticates (the token's ``sub`` + default ``pod_id``).
        :ptype subject: str
        :param customer_id: the customer the principal is bound to (``customer_id`` claim).
        :ptype customer_id: str
        :param pod_id: the pod id, when distinct from ``subject``; defaults to ``subject``.
        :ptype pod_id: str | None
        :param now: unix-seconds issue time (``exp`` = ``now`` + ttl); defaults to the wall clock.
        :ptype now: int | None
        :return: a compact EdDSA JWS identity token.
        :rtype: str
        """
        issued_at = now if now is not None else int(time.time())
        claims = IdentityClaims(
            iss=self._issuer,
            sub=subject,
            customer_id=customer_id,
            sid=uuid.uuid4().hex,  # a fresh per-mint session id (the schema requires one)
            pod_id=pod_id if pod_id is not None else subject,
            iat=issued_at,
            exp=issued_at + self._ttl_seconds,
        )
        token: str = sign_identity_token(claims, signing_key=self._signing_key, kid=self._kid)
        return token

    def jwks(self) -> dict[str, Any]:
        """the public JWKS the resolver verifies this minter's tokens against (public material only)."""
        jwks_doc: dict[str, Any] = build_jwks({self._kid: self._signing_key.public_key()})
        return jwks_doc
