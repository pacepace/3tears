"""3tears-iam: identity and access primitives.

Protocol, crypto, and policy for authenticating callers -- passwords, OAuth2/
OIDC, SAML, session tokens, DPoP, TOTP, WebAuthn, and the anti-automation
controls that guard them.

This package owns no database schema and no wire DTOs. State lives behind the
Protocols in :mod:`threetears.iam.stores`, with a NATS-KV implementation
supplied for the common case.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-iam")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

__all__ = ["__version__"]
