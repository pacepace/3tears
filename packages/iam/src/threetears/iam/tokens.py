"""Session tokens: claims, signing, verification, and pair minting.

A session token asserts **identity and nothing else**. It carries who the
subject is, which tenant they belong to, which session they are in, and when
they last actually authenticated. It never carries a role, a permission, a
group, or a scope.

That is a security property, not a style preference. Authorization evaluated
live at request time means a revoked grant takes effect immediately; a grant
baked into a token takes effect whenever that token happens to expire, which
for a long-lived session is exactly when you least want it. The claim set is
therefore pinned -- on mint AND on verify -- and a token carrying an
unrecognized claim is rejected rather than trusted. A smuggled ``role`` claim
must be a verification failure, not a field nobody reads yet.

**Two signing schemes, one claim vocabulary.** A service that publishes a JWKS
and wants its tokens verifiable by other services signs EdDSA
(:class:`Ed25519Signer`); a self-contained service that both mints and verifies
signs HS256 (:class:`HmacSigner`). The choice changes the signature and nothing
else -- the claims, the pinning, and the verification rules are identical, so
the two do not drift into different security postures.

**Algorithms are pinned from literals.** Verification never reads ``alg`` from
the token to decide how to check it. The header algorithm is compared against
the expected one BEFORE any key is selected, and the decode call passes a
literal single-element list. This is the ``alg: none`` and
RS256-downgraded-to-HS256 family of attacks, and the defence is cheap enough
that it is applied twice.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Protocol
from uuid import uuid7

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

__all__ = [
    "BASE_CLAIMS",
    "DEFAULT_ACCESS_TTL",
    "DEFAULT_REFRESH_TTL",
    "KNOWN_CLAIMS",
    "OPTIONAL_CLAIMS",
    "Ed25519JwksVerifier",
    "Ed25519Signer",
    "HmacSigner",
    "HmacVerifier",
    "SessionClaims",
    "TokenError",
    "TokenPair",
    "TokenSigner",
    "TokenType",
    "TokenVerifier",
    "mint_session_token",
    "mint_token_pair",
    "new_session_id",
    "new_token_id",
    "sole_audience",
    "verify_session_token",
]

#: Short by design. Authorization is evaluated live, and a refresh renews access, so a brief
#: access token costs a round trip and buys prompt loss of access for a disabled subject.
DEFAULT_ACCESS_TTL: Final[timedelta] = timedelta(minutes=15)

#: Long, because it is single-use and rotated: see :mod:`threetears.iam.rotation`.
DEFAULT_REFRESH_TTL: Final[timedelta] = timedelta(days=30)

_EDDSA: Final[str] = "EdDSA"
_HS256: Final[str] = "HS256"

#: Claims every token carries. Enforced as an exact set on mint, and as the required floor
#: on verify.
BASE_CLAIMS: Final[frozenset[str]] = frozenset(
    {
        "sub",
        "sid",
        "jti",
        "iss",
        "aud",
        "iat",
        "exp",
        "type",
        "auth_time",
        "step_up_window",
        "session_started_at",
    }
)

#: Claims a token MAY carry, each corresponding to an optional field on
#: :class:`SessionClaims`. Anything outside ``BASE_CLAIMS | OPTIONAL_CLAIMS`` is a
#: verification failure -- that is what stops a role claim from riding along unnoticed.
OPTIONAL_CLAIMS: Final[frozenset[str]] = frozenset({"customer_id", "cnf", "act", "act_reason", "act_restriction"})

KNOWN_CLAIMS: Final[frozenset[str]] = BASE_CLAIMS | OPTIONAL_CLAIMS


class TokenType(StrEnum):
    """What a token may be exchanged for."""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenError(Exception):
    """A token could not be trusted, or could not be minted safely.

    One error type for every failure mode -- tampered, expired, wrong signature, wrong
    issuer, wrong audience, unknown key, unexpected claim. The caller does not need to
    distinguish them, and telling an attacker WHICH check failed hands them a tuning signal
    for free. Carries only structural reasons, never token strings or key material, so it is
    safe to log at a verification boundary.
    """


@dataclass(frozen=True, slots=True)
class SessionClaims:
    """The identity a session token asserts.

    :ivar sub: the authenticated subject.
    :ivar sid: the session. Survives refresh rotation, so revoking a session revokes every
        token descended from it.
    :ivar jti: this token's unique id -- the handle a denylist or a single-use check names.
    :ivar iss: the issuer.
    :ivar aud: the audiences this token is valid for. Required and never defaulted: only the
        caller knows which trust boundary it will accept a token from, and a defaulted
        audience silently accepts tokens that have left the platform.
    :ivar iat: issued-at, unix seconds.
    :ivar exp: expiry, unix seconds.
    :ivar type: access or refresh.
    :ivar auth_time: when the subject last actually authenticated. Distinct from ``iat``: a
        refresh mints a new token without re-authenticating anyone, so ``iat`` moves and
        ``auth_time`` does not. This is what makes step-up meaningful.
    :ivar step_up_window: seconds after ``auth_time`` during which sensitive actions stay
        permitted. Denormalized onto the token so the check costs no round trip.
    :ivar session_started_at: when the session began, for absolute-lifetime caps. Defaults to
        ``iat`` at mint.
    :ivar customer_id: the tenant, where the deployment is multi-tenant.
    :ivar cnf: an RFC 9449 holder-key thumbprint binding this token to a private key its
        bearer must prove possession of. Set means a stolen token alone is not enough.
    :ivar act: RFC 8693 -- the real actor when this session is an impersonation. The token's
        ``sub`` stays the impersonated subject, so downstream checks see the session as the
        user experiences it, while the audit trail keeps both identities.
    :ivar act_reason: why the impersonation was permitted.
    :ivar act_restriction: an optional restriction on an impersonated session, e.g. read-only.
        A restriction, never a grant -- it can only narrow what live authorization allows.
    """

    sub: str
    sid: str
    jti: str
    iss: str
    aud: tuple[str, ...]
    iat: int
    exp: int
    type: TokenType
    auth_time: int
    step_up_window: int
    session_started_at: int
    customer_id: str | None = None
    cnf: str | None = None
    act: str | None = None
    act_reason: str | None = None
    act_restriction: str | None = None

    @property
    def is_impersonation(self) -> bool:
        """Whether this session is one subject acting as another."""
        return self.act is not None


@dataclass(frozen=True, slots=True)
class TokenPair:
    """An access token and the refresh token that renews it.

    :ivar access_token: the short-lived token to present on requests.
    :ivar refresh_token: the long-lived, single-use token that mints the next pair.
    :ivar claims: the access token's claims, so a caller need not re-verify what it just minted.
    """

    access_token: str
    refresh_token: str
    claims: SessionClaims


class TokenSigner(Protocol):
    """Turns a claims payload into a signed compact JWS."""

    @property
    def algorithm(self) -> str:
        """The JWS algorithm this signer produces."""
        ...

    def sign(self, payload: Mapping[str, Any]) -> str:
        """Sign ``payload`` and return the compact serialization."""
        ...


class TokenVerifier(Protocol):
    """Verifies a compact JWS and returns its payload, or raises :class:`TokenError`."""

    def verify(self, token: str) -> Mapping[str, Any]:
        """Verify ``token``, returning the trusted payload."""
        ...


class Ed25519Signer:
    """Signs EdDSA over Ed25519, stamping a ``kid`` header.

    Use this when tokens must be verifiable by services that do not hold the signing key:
    they verify against the published JWKS instead. The ``kid`` is what lets a verifier pick
    the right key across a rotation, so it is mandatory rather than optional.
    """

    def __init__(self, private_key: Ed25519PrivateKey, *, kid: str) -> None:
        if not kid:
            raise TokenError("an Ed25519 signer requires a non-empty kid.")
        self._private_key = private_key
        self._kid = kid

    @property
    def algorithm(self) -> str:
        return _EDDSA

    @property
    def kid(self) -> str:
        """The key id stamped into every token's header."""
        return self._kid

    def sign(self, payload: Mapping[str, Any]) -> str:
        return jwt.encode(dict(payload), key=self._private_key, algorithm=_EDDSA, headers={"kid": self._kid})


class HmacSigner:
    """Signs HS256 with a shared secret.

    Use this only when the same service both mints and verifies. A symmetric key that any
    verifier holds is a key any verifier can FORGE with, so the moment a second party needs
    to verify these tokens, switch to :class:`Ed25519Signer`.
    """

    def __init__(self, secret: str) -> None:
        if not secret:
            raise TokenError("an HMAC signer requires a non-empty secret.")
        self._secret = secret

    @property
    def algorithm(self) -> str:
        return _HS256

    def sign(self, payload: Mapping[str, Any]) -> str:
        return jwt.encode(dict(payload), key=self._secret, algorithm=_HS256)


class Ed25519JwksVerifier:
    """Verifies EdDSA tokens against a JWKS, selecting the key by header ``kid``."""

    def __init__(
        self,
        *,
        jwks: Mapping[str, Any],
        issuer: str,
        audience: str | Sequence[str],
        leeway_seconds: int = 0,
    ) -> None:
        """
        :param jwks: the key set to verify against. Supplied per call site rather than
            fetched here, so key caching and refresh stay the caller's concern -- see
            :class:`~threetears.core.security.jwks_provider.CachedHubJwksProvider`.
        :ptype jwks: Mapping[str, Any]
        :param issuer: the only issuer this verifier accepts.
        :ptype issuer: str
        :param audience: the audience, or audiences, this verifier accepts. A token matching
            ANY of them passes, so a single value is the norm; pass several only where one
            verifier genuinely serves several trust boundaries.
        :ptype audience: str | Sequence[str]
        :param leeway_seconds: clock-skew tolerance on ``exp`` and ``iat``.
        :ptype leeway_seconds: int
        """
        self._jwks = jwks
        self._issuer = issuer
        self._audience = _normalize_audience(audience)
        self._leeway = leeway_seconds

    def verify(self, token: str) -> Mapping[str, Any]:
        header = _unverified_header(token)
        # Reject a non-EdDSA token BEFORE selecting a key: defence in depth with the literal
        # pin in decode, so an HS256 or `none` header never reaches signature verification.
        if header.get("alg") != _EDDSA:
            raise TokenError("unexpected token algorithm; only EdDSA is accepted.")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise TokenError("token is missing a string kid header.")
        public_key = _select_public_key(self._jwks, kid)
        try:
            return dict(
                jwt.decode(
                    token,
                    key=public_key,
                    algorithms=[_EDDSA],  # literal pin -- statically auditable; never widen
                    issuer=self._issuer,
                    audience=list(self._audience),
                    leeway=self._leeway,
                    options={"require": sorted(BASE_CLAIMS)},
                )
            )
        except jwt.PyJWTError as exc:
            raise TokenError(f"token verification failed ({type(exc).__name__}).") from None


class HmacVerifier:
    """Verifies HS256 tokens against a shared secret."""

    def __init__(
        self,
        secret: str,
        *,
        issuer: str,
        audience: str | Sequence[str],
        leeway_seconds: int = 0,
    ) -> None:
        if not secret:
            raise TokenError("an HMAC verifier requires a non-empty secret.")
        self._secret = secret
        self._issuer = issuer
        self._audience = _normalize_audience(audience)
        self._leeway = leeway_seconds

    def verify(self, token: str) -> Mapping[str, Any]:
        header = _unverified_header(token)
        # Same two-stage pin as the EdDSA verifier. Here it also blocks the classic
        # asymmetric-to-symmetric confusion, where a token signed with a public key as an
        # HMAC secret would verify if the algorithm were read from the token.
        if header.get("alg") != _HS256:
            raise TokenError("unexpected token algorithm; only HS256 is accepted.")
        try:
            return dict(
                jwt.decode(
                    token,
                    key=self._secret,
                    algorithms=[_HS256],  # literal pin -- statically auditable; never widen
                    issuer=self._issuer,
                    audience=list(self._audience),
                    leeway=self._leeway,
                    options={"require": sorted(BASE_CLAIMS)},
                )
            )
        except jwt.PyJWTError as exc:
            raise TokenError(f"token verification failed ({type(exc).__name__}).") from None


def mint_session_token(claims: SessionClaims, *, signer: TokenSigner) -> str:
    """Sign ``claims`` into a compact JWS.

    The assembled payload is checked against the known claim set before signing. If a future
    edit adds a field to :class:`SessionClaims` without teaching this function about it, the
    mint fails loudly here rather than quietly shipping an unexpected claim.

    :param claims: the claims to assert.
    :ptype claims: SessionClaims
    :param signer: the signing strategy.
    :ptype signer: TokenSigner
    :return: the compact JWS.
    :rtype: str
    :raises TokenError: the audience is empty, or the assembled payload carries an
        unexpected claim.
    """
    # Checked here rather than only on the dataclass: an empty audience would encode as
    # `"aud": []`, which no verifier can match, so it is a token that is broken on arrival.
    _normalize_audience(claims.aud)
    payload: dict[str, Any] = {
        "sub": claims.sub,
        "sid": claims.sid,
        "jti": claims.jti,
        "iss": claims.iss,
        # A single audience is encoded as a string and several as an array: both are valid
        # per RFC 7519, and the string form is what most verifiers expect for the common case.
        "aud": claims.aud[0] if len(claims.aud) == 1 else list(claims.aud),
        "iat": claims.iat,
        "exp": claims.exp,
        "type": claims.type.value,
        "auth_time": claims.auth_time,
        "step_up_window": claims.step_up_window,
        "session_started_at": claims.session_started_at,
    }
    if claims.customer_id is not None:
        payload["customer_id"] = claims.customer_id
    if claims.cnf is not None:
        payload["cnf"] = {"jkt": claims.cnf}
    if claims.act is not None:
        payload["act"] = {"sub": claims.act}
    if claims.act_reason is not None:
        payload["act_reason"] = claims.act_reason
    if claims.act_restriction is not None:
        payload["act_restriction"] = claims.act_restriction
    unexpected = set(payload) - KNOWN_CLAIMS
    if unexpected:
        raise TokenError(f"refusing to mint a token carrying unexpected claims: {sorted(unexpected)}.")
    return signer.sign(payload)


def verify_session_token(
    token: str,
    *,
    verifier: TokenVerifier,
    expected_type: TokenType | None = None,
) -> SessionClaims:
    """Verify ``token`` and return the trusted claims.

    :param token: the compact JWS, with no ``Bearer `` prefix.
    :ptype token: str
    :param verifier: the verification strategy.
    :ptype verifier: TokenVerifier
    :param expected_type: reject the token unless its ``type`` matches. Pass this. A refresh
        token accepted where an access token was expected is a privilege escalation: refresh
        tokens live far longer and are handled far more casually.
    :ptype expected_type: TokenType | None
    :return: the verified claims.
    :rtype: SessionClaims
    :raises TokenError: on any verification failure.
    """
    payload = verifier.verify(token)
    unexpected = set(payload) - KNOWN_CLAIMS
    if unexpected:
        # The invariant that makes "identity only" enforceable rather than aspirational: a
        # token carrying a claim this package does not know about is rejected, not ignored.
        raise TokenError(f"token carries unexpected claims: {sorted(unexpected)}.")
    claims = _payload_to_claims(payload)
    if expected_type is not None and claims.type is not expected_type:
        raise TokenError(f"expected a {expected_type.value} token.")
    return claims


def mint_token_pair(
    *,
    subject: str,
    session_id: str,
    issuer: str,
    audience: str | Sequence[str],
    signer: TokenSigner,
    auth_time: int | None = None,
    step_up_window: int = 0,
    session_started_at: int | None = None,
    customer_id: str | None = None,
    cnf: str | None = None,
    act: str | None = None,
    act_reason: str | None = None,
    act_restriction: str | None = None,
    access_ttl: timedelta = DEFAULT_ACCESS_TTL,
    refresh_ttl: timedelta = DEFAULT_REFRESH_TTL,
    now: datetime | None = None,
) -> TokenPair:
    """Mint a fresh access/refresh pair for a session.

    The two tokens share ``sub``, ``sid``, ``auth_time`` and ``session_started_at`` and differ
    in ``type``, ``exp``, and ``jti``. The distinct ``jti`` matters: rotation marks the
    refresh token used, and it can only do that if the refresh token has its own identifier.

    ``auth_time`` defaults to now, which is correct for a fresh login and WRONG for a
    refresh. A rotation must carry the original ``auth_time`` forward -- otherwise every
    refresh silently re-satisfies step-up, and a stolen refresh token becomes a way to
    perform sensitive actions without ever proving possession of a credential.
    :func:`threetears.iam.rotation.rotate_refresh_token` does this correctly.

    :return: the minted pair, with the access token's claims attached.
    :rtype: TokenPair
    """
    issued = now or datetime.now(UTC)
    issued_at = int(issued.timestamp())
    access_claims = SessionClaims(
        sub=subject,
        sid=session_id,
        jti=str(uuid7()),
        iss=issuer,
        aud=_normalize_audience(audience),
        iat=issued_at,
        exp=int((issued + access_ttl).timestamp()),
        type=TokenType.ACCESS,
        auth_time=auth_time if auth_time is not None else issued_at,
        step_up_window=step_up_window,
        session_started_at=session_started_at if session_started_at is not None else issued_at,
        customer_id=customer_id,
        cnf=cnf,
        act=act,
        act_reason=act_reason,
        act_restriction=act_restriction,
    )
    refresh_claims = replace(
        access_claims,
        jti=str(uuid7()),
        type=TokenType.REFRESH,
        exp=int((issued + refresh_ttl).timestamp()),
    )
    return TokenPair(
        access_token=mint_session_token(access_claims, signer=signer),
        refresh_token=mint_session_token(refresh_claims, signer=signer),
        claims=access_claims,
    )


def sole_audience(claims: SessionClaims) -> str:
    """The single audience ``claims`` was issued for.

    :class:`SessionClaims` carries an open tuple because a token in general may serve several
    audiences. A service whose audiences are a closed vocabulary -- internal versus external,
    say -- usually issues exactly one per token, and that narrowing is load-bearing: a token
    claiming both is usable on either side of the boundary the claim exists to enforce. This
    is the checked way to read it back, so each such service does not re-derive the check.

    :param claims: claims from a token that has already been verified.
    :ptype claims: SessionClaims
    :return: the one audience value.
    :rtype: str
    :raises TokenError: the token carries several audiences, or none.
    """
    if len(claims.aud) != 1:
        raise TokenError(f"expected a token with exactly one audience; found {len(claims.aud)}.")
    return claims.aud[0]


def new_session_id() -> str:
    """A fresh session identifier. uuid7 so sessions sort by creation time."""
    return str(uuid7())


def new_token_id() -> str:
    """A fresh ``jti``. Random rather than time-ordered: it is a handle, not a sort key."""
    return secrets.token_hex(16)


def _normalize_audience(audience: str | Sequence[str]) -> tuple[str, ...]:
    """Coerce an audience argument to a non-empty tuple."""
    values = (audience,) if isinstance(audience, str) else tuple(audience)
    if not values or not all(values):
        raise TokenError("at least one non-empty audience is required.")
    return values


def _unverified_header(token: str) -> Mapping[str, Any]:
    """Read the JOSE header WITHOUT verifying anything.

    Only ever used to reject a token on its declared algorithm before a key is selected.
    Nothing read here is trusted for any other purpose.
    """
    try:
        return dict(jwt.get_unverified_header(token))
    except jwt.PyJWTError as exc:
        raise TokenError(f"malformed token header ({type(exc).__name__}).") from None


def _select_public_key(jwks: Mapping[str, Any], kid: str) -> Ed25519PublicKey:
    """Resolve the Ed25519 public key for ``kid`` from a JWKS, or reject.

    An empty key set is reported as "no key for this kid" rather than a malformed document:
    it is the normal state of a verifier whose JWKS has not warmed yet, and it must fail
    closed with a message that says so.
    """
    keys = jwks.get("keys") if isinstance(jwks, Mapping) else None
    if isinstance(keys, list) and not keys:
        raise TokenError("JWKS holds no keys for the token kid.")
    try:
        key_set = jwt.PyJWKSet.from_dict(dict(jwks))
    except (
        jwt.exceptions.PyJWKError,
        jwt.exceptions.PyJWKSetError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise TokenError(f"malformed JWKS ({type(exc).__name__}).") from None
    for jwk in key_set.keys:
        if jwk.key_id == kid:
            key = jwk.key
            if not isinstance(key, Ed25519PublicKey):
                raise TokenError("JWKS key for kid is not an Ed25519 public key.")
            return key
    raise TokenError("no JWKS key matches the token kid.")


def _payload_to_claims(payload: Mapping[str, Any]) -> SessionClaims:
    """Build :class:`SessionClaims` from an already-verified payload.

    Every field is type-checked rather than coerced. A claim of the wrong type means the
    token was minted by something this package does not understand, and guessing at what it
    meant is how a verification boundary stops being one.
    """
    type_raw = payload.get("type")
    if not isinstance(type_raw, str):
        raise TokenError("token claim 'type' must be a string.")
    try:
        token_type = TokenType(type_raw)
    except ValueError:
        raise TokenError(f"unrecognized token type {type_raw!r}.") from None

    aud_raw = payload.get("aud")
    audience: tuple[str, ...]
    if isinstance(aud_raw, str):
        audience = (aud_raw,)
    elif isinstance(aud_raw, list) and aud_raw and all(isinstance(item, str) for item in aud_raw):
        audience = tuple(str(item) for item in aud_raw)
    else:
        raise TokenError("token claim 'aud' must be a string or a non-empty list of strings.")

    cnf_claim = payload.get("cnf")
    cnf = cnf_claim.get("jkt") if isinstance(cnf_claim, Mapping) else None
    act_claim = payload.get("act")
    act = act_claim.get("sub") if isinstance(act_claim, Mapping) else None

    return SessionClaims(
        sub=_require_nonempty_str(payload, "sub"),
        sid=_require_nonempty_str(payload, "sid"),
        jti=_require_nonempty_str(payload, "jti"),
        iss=_require_nonempty_str(payload, "iss"),
        aud=audience,
        iat=_require_int(payload, "iat"),
        exp=_require_int(payload, "exp"),
        type=token_type,
        auth_time=_require_int(payload, "auth_time"),
        step_up_window=_require_int(payload, "step_up_window"),
        session_started_at=_require_int(payload, "session_started_at"),
        customer_id=_optional_str(payload.get("customer_id")),
        cnf=_optional_str(cnf),
        act=_optional_str(act),
        act_reason=_optional_str(payload.get("act_reason")),
        act_restriction=_optional_str(payload.get("act_restriction")),
    )


def _require_nonempty_str(payload: Mapping[str, Any], claim: str) -> str:
    value = payload.get(claim)
    if not isinstance(value, str) or not value:
        raise TokenError(f"token claim {claim!r} must be a non-empty string.")
    return value


def _require_int(payload: Mapping[str, Any], claim: str) -> int:
    value = payload.get(claim)
    # bool is an int subclass; a boolean timestamp is malformed, not zero-or-one.
    if not isinstance(value, int) or isinstance(value, bool):
        raise TokenError(f"token claim {claim!r} must be an integer.")
    return value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
