"""policy-driven access-control + secret primitives.

public surface:

- :class:`Sandbox` / :class:`PathSandbox` / :class:`SandboxDecision` /
  :class:`SandboxDenied` — policy-driven access-control (see ``sandbox``).
- secret references (``secret_refs``): :func:`resolve_secret` / :func:`validate_ref` /
  :func:`parse_ref` / :func:`register_scheme` / :class:`SecretResolutionError` — a
  ``scheme://locator`` reference resolved to a ``SecretStr`` at use time; apps add
  schemes via :func:`register_scheme`.
- encryption at rest (``encryption``): :func:`seal` / :func:`open_secret` /
  :class:`DecryptionError` — AES-256-GCM under a master key, for the times a secret
  must be *stored* rather than referenced.
- identity tokens (``identity_token``): :class:`IdentityClaims` / :class:`IdentityTokenError` /
  :class:`IdentityKeyNotFoundError` (the recoverable kid-not-in-cache signal a verifier reacts to
  with a reactive JWKS refresh) /
  :func:`sign_identity_token` / :func:`verify_identity_token` / :func:`build_jwks` /
  :func:`generate_signing_keypair` / :data:`IDENTITY_REFUSED` (the one code every door answers a
  forwarded identity that does not verify) — Hub-issued EdDSA-signed JWS asserting a VERIFIED caller
  identity, verified against the Hub JWKS before RBAC (platform-auth Option B).
  :class:`~threetears.core.security.jwks_provider.CachedHubJwksProvider` fetches + caches that
  JWKS over NATS so a verifier's ``jwks_provider()`` returns it with no hot-path IO.
- proxy assertions (``proxy_assertion``): :func:`mint_proxy_assertion` /
  :func:`verify_proxy_assertion` / :data:`TOOL_PROXY_ASSERTION_UNVERIFIED` (the code a tool pod
  answers a call that could not show it came through the registry for this body and this pod) /
  :data:`TOOL_POP_LEDGER_UNAVAILABLE` (the code the registry and a tool pod both answer when the
  replay ledger their single-use check depends on cannot be reached).
- proof freshness (``freshness``): :data:`ISSUE_TIME_FUTURE_TOLERANCE` (how far ahead of a
  verifier's clock a POD-signed issue time is accepted) / :data:`CLIENT_ISSUE_TIME_FUTURE_TOLERANCE`
  (the same for a proof a user's device signs) -- the numbers a proof verifier's replay guard is
  sized for / :data:`DEFAULT_PROOF_MAX_AGE` (how old one may be) /
  :func:`issue_time_is_fresh` (the check, each direction on its own bound).
"""

from threetears.core.security.encryption import DecryptionError, open_secret, seal
from threetears.core.security.freshness import (
    CLIENT_ISSUE_TIME_FUTURE_TOLERANCE,
    DEFAULT_PROOF_MAX_AGE,
    ISSUE_TIME_FUTURE_TOLERANCE,
    issue_time_is_fresh,
)
from threetears.core.security.identity_minter import (
    DEFAULT_IDENTITY_TTL_SECONDS,
    IdentityMinter,
    static_token_provider,
)
from threetears.core.security.identity_token import (
    IDENTITY_REFUSED,
    IDENTITY_REFUSED_MESSAGE,
    PLATFORM_CUSTOMER_SENTINEL,
    IdentityClaims,
    IdentityKeyNotFoundError,
    IdentityTokenError,
    VerifiedPrincipal,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    jwk_thumbprint,
    principal_from_claims,
    sign_identity_token,
    verify_identity_token,
)
from threetears.core.security.jwks_provider import CachedHubJwksProvider
from threetears.core.security.pop import access_token_hash, make_pop_proof, verify_pop_proof
from threetears.core.security.proxy_assertion import (
    TOOL_POP_LEDGER_UNAVAILABLE,
    TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE,
    TOOL_PROXY_ASSERTION_UNVERIFIED,
    TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE,
    ProxyAssertionClaims,
    mint_proxy_assertion,
    verify_proxy_assertion,
)
from threetears.core.security.proxy_signer import ProxyAssertionSigner
from threetears.core.security.sandbox import (
    PathSandbox,
    Sandbox,
    SandboxDecision,
    SandboxDenied,
)
from threetears.core.security.secret_refs import (
    Resolver,
    SecretResolutionError,
    parse_ref,
    register_scheme,
    resolve_secret,
    validate_ref,
)

__all__ = [
    # access control
    "PathSandbox",
    "Sandbox",
    "SandboxDecision",
    "SandboxDenied",
    # secret references
    "Resolver",
    "SecretResolutionError",
    "parse_ref",
    "register_scheme",
    "resolve_secret",
    "validate_ref",
    # encryption at rest
    "DecryptionError",
    "open_secret",
    "seal",
    # proof freshness
    "CLIENT_ISSUE_TIME_FUTURE_TOLERANCE",
    "DEFAULT_PROOF_MAX_AGE",
    "ISSUE_TIME_FUTURE_TOLERANCE",
    "issue_time_is_fresh",
    # identity tokens
    "DEFAULT_IDENTITY_TTL_SECONDS",
    "IDENTITY_REFUSED",
    "IDENTITY_REFUSED_MESSAGE",
    "PLATFORM_CUSTOMER_SENTINEL",
    "TOOL_POP_LEDGER_UNAVAILABLE",
    "TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE",
    "TOOL_PROXY_ASSERTION_UNVERIFIED",
    "TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE",
    "CachedHubJwksProvider",
    "IdentityClaims",
    "IdentityKeyNotFoundError",
    "IdentityMinter",
    "IdentityTokenError",
    "ProxyAssertionClaims",
    "ProxyAssertionSigner",
    "VerifiedPrincipal",
    "access_token_hash",
    "build_jwks",
    "canonical_call_hash",
    "generate_signing_keypair",
    "jwk_thumbprint",
    "make_pop_proof",
    "mint_proxy_assertion",
    "principal_from_claims",
    "sign_identity_token",
    "static_token_provider",
    "verify_identity_token",
    "verify_pop_proof",
    "verify_proxy_assertion",
]
