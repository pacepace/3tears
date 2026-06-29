"""Hub-issued identity token: an EdDSA-signed compact JWS the registry verifies before RBAC.

Tool-call caller identity is self-asserted on the bus today, so RBAC authorizes a CLAIMED
identity it never authenticated. This module is the application half of the fix (Option B):
the trust anchor (the Hub) mints a short-lived token binding the caller's verified identity
(agent, customer, optional user, session, pod); the registry proxy and tool pods VERIFY it
against the Hub's published JWKS and feed RBAC the VERIFIED identity instead of the envelope.

The one security-critical invariant: **the algorithm is pinned to EdDSA**. Verification never
reads the algorithm from the token header to choose how to check the signature — that is the
classic JWS forgery (``alg=none``; or HS256 signed with the public key as the HMAC secret). A
companion enforcement test (``tests/enforcement/test_identity_token_alg_pinning.py``) locks the
pin structurally so it can't be edited away.

This module is pure: sign / verify / JWKS build, no transport and no key custody. The signing
key comes from operator custody (HSM/KMS) in production; :func:`generate_signing_keypair` is a
convenience for tests and bootstrap. Identity ids stay ``str`` here (the token is the wire
border) — consumers convert to their domain types (e.g. ``UUID``) at their own border.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from jwt.algorithms import OKPAlgorithm

__all__ = [
    "IdentityClaims",
    "IdentityKeyNotFoundError",
    "IdentityTokenError",
    "build_jwks",
    "canonical_call_hash",
    "generate_signing_keypair",
    "jwk_thumbprint",
    "sign_identity_token",
    "verify_identity_token",
]

# the ONE algorithm this module signs and verifies. EdDSA over Ed25519: small keys, fast
# verify, no curve/parameter choices to get wrong. Decode pins this as a list literal (so the
# enforcement test can read it statically); the header pre-check and signing reuse the name.
_ALG = "EdDSA"

# claims that MUST be present for a token to be trusted. user_id is intentionally absent — an
# agent acting on its own behalf (no human in the loop) carries no user_id.
_REQUIRED_CLAIMS = ("iss", "sub", "customer_id", "sid", "pod_id", "iat", "exp")


class IdentityTokenError(Exception):
    """raised when a token cannot be trusted: bad signature, wrong alg/kid/issuer, expiry, or

    a malformed token / JWKS. Deliberately carries only the STRUCTURAL reason — never the token
    string, a claim value, or key material — so it is safe to log at the verification boundary.
    """


class IdentityKeyNotFoundError(IdentityTokenError):
    """raised specifically when no key for the token's ``kid`` is present in the verifier's JWKS.

    A SUBCLASS of :class:`IdentityTokenError`, so every existing ``except IdentityTokenError`` keeps
    catching it (fail-closed is unchanged). It is the distinct, RECOVERABLE signal a verifier reacts
    to: a token signed under a key the verifier's CACHED JWKS does not yet hold (the Hub re-keyed, or
    the cache is empty/stale after a Hub pod move) is well-formed but un-checkable against the stale
    cache, so the consumer can trigger ONE reactive JWKS refresh and re-verify before rejecting. An
    expired / bad-signature / malformed token raises the BASE error instead -- a refresh cannot fix
    those, so they must NOT provoke a Hub fetch.
    """


@dataclass(frozen=True, slots=True)
class IdentityClaims:
    """the verified identity a token asserts. ids are ``str`` (wire border); ``iat``/``exp`` are

    unix seconds. ``user_id`` is ``None`` for agent-initiated calls with no human principal.
    ``cnf`` is the holder-key JWK thumbprint (DPoP ``jkt``) binding the token to a
    proof-of-possession key, so a leaked token alone is unusable; ``None`` until pop is enabled.
    ``conversation_id`` is the conversation a per-turn user-assertion was minted for: the registry
    proxy and the tool pod re-check it against the inbound call's ``conversation_id`` and reject a
    mismatch, so a captured user-assertion cannot be replayed into a DIFFERENT conversation (the
    cross-conversation impersonation residual). ``None`` for the handshake identity token (which
    binds no conversation) -- only the user-assertion sets it.
    """

    sub: str  # the agent_id the token authenticates
    customer_id: str
    sid: str  # session id (binds the token to a handshake session)
    pod_id: str
    iss: str
    iat: int
    exp: int
    user_id: str | None = None
    cnf: str | None = None  # holder-key thumbprint (jkt) for proof-of-possession
    conversation_id: str | None = None  # the conversation a user-assertion is bound to


def generate_signing_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """generate an Ed25519 keypair. Convenience for tests / bootstrap; production signing keys

    come from operator custody (HSM/KMS), never from this call on a live mint path.
    """
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def build_jwks(public_keys: Mapping[str, Ed25519PublicKey]) -> dict[str, Any]:
    """build a JWKS (``{"keys": [...]}``) of OKP/Ed25519 PUBLIC keys, one entry per ``kid``.

    Multiple kids are how overlap-window rotation works: publish the new key alongside the old
    so in-flight tokens signed by either verify until the old kid is retired. Only public
    material is emitted (``x``, never the private scalar ``d``).
    """
    keys: list[dict[str, Any]] = []
    for kid, public_key in public_keys.items():
        # fail closed: never let a private (or non-Ed25519) key reach the published JWKS.
        # OKPAlgorithm.to_jwk(<private key>) emits the private scalar ``d`` — publishing the
        # Hub signing key. The type hint is the contract; this is the runtime backstop for a
        # custody/KMS key mix-up on a future minting path.
        if not isinstance(public_key, Ed25519PublicKey):
            raise IdentityTokenError(f"build_jwks requires Ed25519 public keys; kid {kid!r} is not one.")
        jwk = OKPAlgorithm.to_jwk(public_key, as_dict=True)
        jwk["kid"] = kid
        jwk["use"] = "sig"
        jwk["alg"] = _ALG
        keys.append(jwk)
    return {"keys": keys}


def sign_identity_token(claims: IdentityClaims, *, signing_key: Ed25519PrivateKey, kid: str) -> str:
    """sign ``claims`` into a compact JWS, stamping ``kid`` in the header for JWKS lookup.

    The caller sets ``iat``/``exp`` on the claims (the minter owns the TTL policy); this signs
    exactly what it is given.
    """
    payload: dict[str, object] = {
        "iss": claims.iss,
        "sub": claims.sub,
        "customer_id": claims.customer_id,
        "sid": claims.sid,
        "pod_id": claims.pod_id,
        "iat": claims.iat,
        "exp": claims.exp,
    }
    if claims.user_id is not None:
        payload["user_id"] = claims.user_id
    if claims.cnf is not None:
        payload["cnf"] = {"jkt": claims.cnf}
    if claims.conversation_id is not None:
        payload["conversation_id"] = claims.conversation_id
    return jwt.encode(payload, key=signing_key, algorithm=_ALG, headers={"kid": kid})


def verify_identity_token(token: str, *, jwks: dict[str, Any], issuer: str, leeway_seconds: int = 0) -> IdentityClaims:
    """verify ``token`` against ``jwks`` and return the trusted :class:`IdentityClaims`.

    Pins EdDSA, selects the public key by the header ``kid``, and checks the signature, the
    required claims, the issuer, and expiry (with ``leeway_seconds`` of clock skew). Any
    failure raises :class:`IdentityTokenError` — the caller MUST treat that as deny, never as a
    soft fallback to the envelope's claimed identity.

    :raises IdentityTokenError: malformed token/JWKS, unexpected alg, missing/unknown kid,
        bad signature, wrong issuer, expired, or a missing required claim.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise IdentityTokenError(f"malformed token header ({type(exc).__name__}).") from None
    # reject a non-EdDSA token BEFORE selecting a key — defence in depth with the decode pin
    # below, so an HS256/none header never reaches signature verification.
    if header.get("alg") != _ALG:
        raise IdentityTokenError("unexpected token algorithm; only EdDSA is accepted.")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise IdentityTokenError("token is missing a string kid header.")
    public_key = _select_public_key(jwks, kid)
    try:
        payload = jwt.decode(
            token,
            key=public_key,
            algorithms=["EdDSA"],  # literal pin — statically auditable; never widen
            issuer=issuer,
            leeway=leeway_seconds,
            options={"require": list(_REQUIRED_CLAIMS)},
        )
    except jwt.PyJWTError as exc:
        raise IdentityTokenError(f"token verification failed ({type(exc).__name__}).") from None
    return _payload_to_claims(payload)


def _select_public_key(jwks: dict[str, Any], kid: str) -> Ed25519PublicKey:
    """resolve the public key for ``kid`` from a JWKS, or reject.

    A "key not present" condition -- an EMPTY keyset (typically a never-warmed cache) or a keyset that
    holds no key matching ``kid`` (a Hub re-key the cache has not caught up to) -- raises the distinct
    :class:`IdentityKeyNotFoundError`, the RECOVERABLE signal a verifier reacts to with one reactive
    JWKS refresh + re-verify. A structurally MALFORMED JWKS (not a JWKS document at all) raises the
    base :class:`IdentityTokenError` -- a refresh will not turn a non-JWKS into one, so it must not
    provoke a Hub fetch.
    """
    # an empty keyset is "no key for this kid", not a malformed document: surface it as the
    # recoverable key-not-found signal (a never-warmed / stale cache self-heals on a reactive refresh)
    # BEFORE PyJWKSet.from_dict raises its generic "did not contain any keys" error.
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if isinstance(keys, list) and not keys:
        raise IdentityKeyNotFoundError("JWKS holds no keys for the token kid.")
    try:
        key_set = jwt.PyJWKSet.from_dict(jwks)
    except (
        jwt.exceptions.PyJWKError,
        jwt.exceptions.PyJWKSetError,
        AttributeError,  # a non-dict jwks (e.g. a JSON array) -> from_dict's .get fails
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise IdentityTokenError(f"malformed JWKS ({type(exc).__name__}).") from None
    for jwk in key_set.keys:
        if jwk.key_id == kid:
            key = jwk.key
            if not isinstance(key, Ed25519PublicKey):
                raise IdentityTokenError("JWKS key for kid is not an Ed25519 public key.")
            return key
    raise IdentityKeyNotFoundError("no JWKS key matches the token kid.")


def _payload_to_claims(payload: dict[str, Any]) -> IdentityClaims:
    """build :class:`IdentityClaims` from a verified payload; unknown claims are ignored.

    Presence of the required claims is already guaranteed by the decode ``require`` option, but
    ``require`` accepts a present-but-empty/null value — so each identity field is additionally
    checked to be a NON-EMPTY string here (a token whose entire purpose is to stop RBAC trusting
    unverified identity must not carry ``customer_id: null`` through to the evaluator). Unknown
    (future) claims are tolerated for a forward-compatible rollout.
    """
    user_id = payload.get("user_id")
    cnf_claim = payload.get("cnf")
    cnf = cnf_claim.get("jkt") if isinstance(cnf_claim, dict) else None
    # conversation_id is an OPTIONAL claim (only a user-assertion sets it). when present it must be
    # a non-empty string -- a present-but-empty/null value is not a usable conversation binding and
    # is normalized to ``None`` so the verify gates' "user-assertion carries no conversation_id"
    # deny fires rather than a confusing empty-string mismatch.
    conversation_id = payload.get("conversation_id")
    return IdentityClaims(
        sub=_require_nonempty_str(payload, "sub"),
        customer_id=_require_nonempty_str(payload, "customer_id"),
        sid=_require_nonempty_str(payload, "sid"),
        pod_id=_require_nonempty_str(payload, "pod_id"),
        iss=_require_nonempty_str(payload, "iss"),
        iat=int(payload["iat"]),
        exp=int(payload["exp"]),
        user_id=None if user_id is None else _require_nonempty_str(payload, "user_id"),
        cnf=cnf if isinstance(cnf, str) and cnf else None,
        conversation_id=conversation_id if isinstance(conversation_id, str) and conversation_id else None,
    )


def _require_nonempty_str(payload: dict[str, Any], claim: str) -> str:
    """return ``payload[claim]`` iff it is a non-empty string, else reject. ``claim`` is a claim

    NAME (e.g. ``"customer_id"``), never a value — safe to include in the error message.
    """
    value = payload.get(claim)
    if not isinstance(value, str) or not value:
        raise IdentityTokenError(f"identity claim {claim!r} must be a non-empty string.")
    return value


def canonical_call_hash(tool_name: str, arguments: Mapping[str, Any], correlation_id: str | None) -> str:
    """SHA-256 (base64url, unpadded) of the canonical call body.

    The value both a proof-of-possession proof and a proxy->pod assertion BIND to, so a captured
    proof cannot be spliced onto a DIFFERENT call. Canonical form = compact JSON with recursively
    sorted keys; the signer and the verifier MUST hash through this one function or their digests
    diverge (the classic canonicalization footgun).

    :param tool_name: the dotted tool name being called
    :ptype tool_name: str
    :param arguments: the tool call arguments
    :ptype arguments: Mapping[str, Any]
    :param correlation_id: the call correlation id, or ``None``
    :ptype correlation_id: str | None
    :return: base64url(SHA-256(canonical-json)), unpadded
    :rtype: str
    """
    canonical = json.dumps(
        {
            "arguments": dict(arguments),
            "correlation_id": correlation_id,
            "tool_name": tool_name,
        },
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    # str(bytes, "ascii"), NOT bytes.decode(): the alg-pinning enforcement matches every ``.decode(``
    # call in this module so a sneaky jwt decode can't dodge the EdDSA pin -- a bytes->str here is
    # not a jwt decode, so it must not use ``.decode``.
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")


def jwk_thumbprint(public_key: Ed25519PublicKey) -> str:
    """RFC 7638 JWK thumbprint (SHA-256, base64url, unpadded) of an Ed25519 public key.

    The stable identifier for a holder/signing key: the Hub puts it in a token's ``cnf`` claim and
    the proxy recomputes it from a proof-of-possession proof's inline key to confirm the caller
    holds the bound key. Both sides MUST compute it through this one function so they always agree.

    :param public_key: the Ed25519 public key to fingerprint
    :ptype public_key: Ed25519PublicKey
    :return: the base64url thumbprint
    :rtype: str
    """
    jwk = OKPAlgorithm.to_jwk(public_key, as_dict=True)
    canonical = json.dumps(
        {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"]},
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    # str(bytes, "ascii"), not .decode() -- see canonical_call_hash (alg-pinning enforcement).
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")
