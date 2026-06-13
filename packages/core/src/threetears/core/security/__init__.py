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
"""

from threetears.core.security.encryption import DecryptionError, open_secret, seal
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
]
