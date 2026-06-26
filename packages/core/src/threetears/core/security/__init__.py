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
  :func:`sign_identity_token` / :func:`verify_identity_token` / :func:`build_jwks` /
  :func:`generate_signing_keypair` — Hub-issued EdDSA-signed JWS asserting a VERIFIED caller
  identity, verified against the Hub JWKS before RBAC (platform-auth Option B).
"""

from threetears.core.security.encryption import DecryptionError, open_secret, seal
from threetears.core.security.identity_token import (
    IdentityClaims,
    IdentityTokenError,
    build_jwks,
    generate_signing_keypair,
    sign_identity_token,
    verify_identity_token,
)
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
    # identity tokens
    "IdentityClaims",
    "IdentityTokenError",
    "build_jwks",
    "generate_signing_keypair",
    "sign_identity_token",
    "verify_identity_token",
]
