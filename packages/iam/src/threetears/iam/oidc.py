"""OpenID Connect: discovery, ``id_token`` verification, and identity extraction.

The relying-party half of OIDC. What is here is the part that is identical for
Entra, Okta, Google, Keycloak and anything else that publishes a discovery
document -- fetch the metadata, verify the token, pull out the identity. What is
not here is the session, the user record, or the redirect plumbing, because
those differ per service and none of them are protocol.

**The discovery document's ``issuer`` must equal the URL it was fetched from.**
OIDC Discovery 1.0 SS4.3 requires it, and skipping the check is a real hole
rather than a formality: without it, a compromised or misconfigured discovery
endpoint can assert a DIFFERENT issuer, and every later check -- including the
``id_token``'s own ``iss`` -- is then made against the attacker's claimed value
instead of the one the deployment actually configured and trusts.

**Signing algorithms are an explicit allow-list, and it can never contain a
symmetric one.** In OIDC the client secret is the HMAC key, so a provider that
is allowed to sign with HS256 can be impersonated by anyone who holds that
secret -- which includes every service that has ever been configured as a
client. :func:`verify_id_token` rejects an allow-list containing anything
symmetric or ``none`` before it verifies anything.

**Failures name their reason.** Signature, issuer, audience and expiry are
distinguished, because an operator debugging a federation problem needs to know
which one failed and an auditor needs it recorded. That is safe here in a way it
is not for a session token: the counterparty is a configured identity provider,
not an anonymous caller probing for an oracle.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
from joserfc import errors as jose_errors
from joserfc import jwt as jose_jwt
from joserfc.jwk import KeySet

from threetears.observe import get_logger

__all__ = [
    "DEFAULT_SIGNING_ALGORITHMS",
    "OidcDiscoveryClient",
    "OidcError",
    "OidcIdentity",
    "OidcProviderMetadata",
    "OidcTokenError",
    "coerce_email_verified",
    "extract_identity",
    "verify_id_token",
]

log = get_logger(__name__)

#: Conservative default. RS256 is what essentially every provider signs with. Widen
#: deliberately -- to add ES256, say -- never by accident.
DEFAULT_SIGNING_ALGORITHMS: Final[tuple[str, ...]] = ("RS256",)

#: Algorithm families that must never verify an ``id_token``. ``none`` is unsigned; the
#: symmetric families use the client secret as the key (module docstring).
_FORBIDDEN_ALG_PREFIXES: Final[tuple[str, ...]] = ("HS", "none")

_WELL_KNOWN_PATH: Final[str] = "/.well-known/openid-configuration"


class OidcError(Exception):
    """An OIDC step failed and the login must be denied.

    Covers discovery failures, a metadata mismatch, and a refused exchange. Callers must
    treat it as deny -- never as a fallback to an unverified assertion.
    """


class OidcTokenError(OidcError):
    """An ``id_token`` failed verification.

    :ivar reason: which check failed -- ``"signature"``, ``"issuer"``, ``"audience"``,
        ``"expiry"``, or ``"configuration"``. Recorded so a federation problem is debuggable
        and auditable; see the module docstring on why naming it is safe here.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class OidcProviderMetadata:
    """The part of a discovery document a relying party actually needs.

    :ivar issuer: the verified issuer -- equal to the URL it was fetched from.
    :ivar authorization_endpoint: where to send the user to authenticate.
    :ivar token_endpoint: where to exchange the authorization code.
    :ivar jwks: the provider's signing keys.
    """

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks: KeySet


@dataclass(frozen=True, slots=True)
class OidcIdentity:
    """The identity an ``id_token`` asserts.

    :ivar subject: the provider's stable identifier for the user. The identity key -- an
        email can change hands, this cannot.
    :ivar email: the asserted address, if any.
    :ivar email_verified: whether the PROVIDER says it verified that address. Never assume
        ``True``: an unverified address lets an attacker claim a local account by asserting
        someone else's email at an identity provider that does not check.
    :ivar display_name: the asserted display name, if any.
    :ivar claims: the full verified claim set, for claim-mapping rules.
    """

    subject: str
    email: str | None = None
    email_verified: bool = False
    display_name: str | None = None
    claims: Mapping[str, Any] = field(default_factory=dict)


class OidcDiscoveryClient:
    """Fetches and caches provider metadata.

    Caching is per instance and unbounded in time, which means a provider's signing-key
    rotation is picked up on the next process restart rather than automatically. That is a
    real limitation, stated rather than hidden: a deployment that needs seamless rotation
    should hold a short-lived instance or clear the cache on a verification failure.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        """
        :param client: the HTTP client to fetch through. Supplied rather than constructed so
            timeouts and lifecycle stay with the caller, and so a test can pass an
            ``httpx.MockTransport`` and never touch the network.
        :ptype client: httpx.AsyncClient
        """
        self._client = client
        self._cache: dict[str, OidcProviderMetadata] = {}

    async def discover(self, issuer: str) -> OidcProviderMetadata:
        """Resolve ``issuer``'s metadata, from cache when already fetched.

        :param issuer: the configured, trusted issuer URL.
        :ptype issuer: str
        :return: the provider's metadata.
        :rtype: OidcProviderMetadata
        :raises OidcError: the fetch failed, the document is malformed, or its ``issuer``
            does not exactly match ``issuer`` (module docstring -- this check is load-bearing).
        """
        cached = self._cache.get(issuer)
        if cached is not None:
            return cached
        try:
            discovery_response = await self._client.get(f"{issuer.rstrip('/')}{_WELL_KNOWN_PATH}")
            discovery_response.raise_for_status()
            discovery = discovery_response.json()
        except httpx.HTTPError as exc:
            raise OidcError(f"oidc discovery fetch failed ({type(exc).__name__}).") from exc
        except ValueError as exc:
            raise OidcError("oidc discovery document is not valid JSON.") from exc
        if not isinstance(discovery, Mapping):
            raise OidcError("oidc discovery document is not a JSON object.")

        asserted = discovery.get("issuer")
        if asserted != issuer:
            raise OidcError(f"oidc discovery issuer mismatch: configured {issuer!r}, document asserts {asserted!r}.")
        try:
            jwks_response = await self._client.get(str(discovery["jwks_uri"]))
            jwks_response.raise_for_status()
            jwks = KeySet.import_key_set(jwks_response.json())
        except httpx.HTTPError as exc:
            raise OidcError(f"oidc jwks fetch failed ({type(exc).__name__}).") from exc
        except KeyError as exc:
            raise OidcError("oidc discovery document has no jwks_uri.") from exc
        except (ValueError, jose_errors.JoseError) as exc:
            raise OidcError(f"oidc jwks is malformed ({type(exc).__name__}).") from exc

        try:
            metadata = OidcProviderMetadata(
                issuer=str(discovery["issuer"]),
                authorization_endpoint=str(discovery["authorization_endpoint"]),
                token_endpoint=str(discovery["token_endpoint"]),
                jwks=jwks,
            )
        except KeyError as exc:
            raise OidcError(f"oidc discovery document is missing {exc.args[0]!r}.") from None
        self._cache[issuer] = metadata
        return metadata

    def forget(self, issuer: str) -> None:
        """Drop ``issuer`` from the cache, so the next :meth:`discover` refetches.

        The escape hatch for key rotation: a verifier that fails on an unknown ``kid`` can
        call this and retry once, rather than waiting for a restart.
        """
        self._cache.pop(issuer, None)


def verify_id_token(
    id_token: str,
    *,
    jwks: KeySet,
    issuer: str,
    audience: str,
    algorithms: Sequence[str] = DEFAULT_SIGNING_ALGORITHMS,
    clock_skew_seconds: int = 60,
) -> Mapping[str, Any]:
    """Verify an ``id_token``'s signature, issuer, audience, and expiry.

    :param id_token: the compact JWS from the token endpoint.
    :ptype id_token: str
    :param jwks: the provider's signing keys, from :class:`OidcDiscoveryClient`.
    :ptype jwks: KeySet
    :param issuer: the issuer the token must assert.
    :ptype issuer: str
    :param audience: this client's id -- the audience the token must be addressed to. Without
        this check a token minted for a DIFFERENT client of the same provider would verify
        here, which is a complete authentication bypass wherever that other client is less
        trusted.
    :ptype audience: str
    :param algorithms: the permitted signing algorithms. Rejected outright if it contains a
        symmetric algorithm or ``none``.
    :ptype algorithms: Sequence[str]
    :param clock_skew_seconds: leeway on expiry.
    :ptype clock_skew_seconds: int
    :return: the verified claims.
    :rtype: Mapping[str, Any]
    :raises OidcTokenError: on any failure, with ``reason`` naming which check it was.
    """
    permitted = tuple(algorithms)
    if not permitted:
        raise OidcTokenError("no id_token signing algorithms were permitted.", reason="configuration")
    for algorithm in permitted:
        if algorithm.startswith(_FORBIDDEN_ALG_PREFIXES):
            raise OidcTokenError(
                f"refusing to verify an id_token with {algorithm!r}: symmetric and unsigned "
                "algorithms are never acceptable here (see the module docstring).",
                reason="configuration",
            )

    try:
        token = jose_jwt.decode(id_token, jwks, algorithms=list(permitted))
    except jose_errors.BadSignatureError as exc:
        raise OidcTokenError("id_token signature verification failed.", reason="signature") from exc
    except jose_errors.JoseError as exc:
        raise OidcTokenError(f"id_token could not be decoded ({type(exc).__name__}).", reason="signature") from exc

    registry = jose_jwt.JWTClaimsRegistry(
        now=int(time.time()),
        leeway=clock_skew_seconds,
        iss={"essential": True, "value": issuer},
        aud={"essential": True, "value": audience},
        exp={"essential": True},
    )
    try:
        registry.validate(token.claims)
    except jose_errors.InvalidClaimError as exc:
        raise OidcTokenError(f"id_token claim {exc.claim!r} is invalid.", reason=_reason_for(exc.claim)) from exc
    except jose_errors.ExpiredTokenError as exc:
        raise OidcTokenError("id_token has expired.", reason="expiry") from exc
    except jose_errors.MissingClaimError as exc:
        raise OidcTokenError(f"id_token is missing claim {exc.claim!r}.", reason=_reason_for(exc.claim)) from exc
    except jose_errors.JoseError as exc:
        raise OidcTokenError(f"id_token claims are invalid ({type(exc).__name__}).", reason="signature") from exc
    return dict(token.claims)


def extract_identity(claims: Mapping[str, Any]) -> OidcIdentity:
    """Pull the identity out of an ALREADY-VERIFIED claim set.

    Does no verification of its own. Calling it on unverified claims is calling it on
    attacker-controlled input.

    :raises OidcTokenError: the claims carry no usable ``sub``. Every OIDC token must have
        one; a token without it cannot be tied to any account.
    """
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise OidcTokenError("id_token has no usable 'sub' claim.", reason="signature")
    email = claims.get("email")
    name = claims.get("name")
    return OidcIdentity(
        subject=subject,
        email=str(email) if email else None,
        email_verified=coerce_email_verified(claims.get("email_verified")),
        display_name=str(name) if name else None,
        claims=dict(claims),
    )


def coerce_email_verified(raw: object) -> bool:
    """Normalize an ``email_verified`` claim to a bool, defaulting to ``False``.

    The claim is specified as a boolean, but real providers ship the strings ``"true"`` and
    ``"false"`` instead -- Apple most notably. A naive truthiness test reads the STRING
    ``"false"`` as verified, which is the exact inversion that lets an unverified address
    through, so the string forms are matched explicitly.

    Anything unrecognized is ``False``. Defaulting the other way would mean a provider that
    omits the claim silently grants verified status to every address it asserts.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return False


def _reason_for(claim: str) -> str:
    """Map a failed claim name onto the reason vocabulary."""
    return {"iss": "issuer", "aud": "audience", "exp": "expiry"}.get(claim, "signature")
