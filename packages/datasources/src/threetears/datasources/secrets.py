"""datasource credential resolution — back-compat re-export of the core secret registry.

The secret-reference machinery moved to :mod:`threetears.core.security.secret_refs` (the
canonical home, so any 3tears app consumes it from ``core`` without depending on this
package). This module re-exports the same public surface so existing imports —
``from threetears.datasources.secrets import resolve_secret, validate_ref`` — keep working
unchanged. The ``env://`` / ``k8s://`` backends and the
``AIBOTS_DATASOURCE_SECRETS_DIR`` mount override are identical; new schemes are added via
``threetears.core.security.register_scheme``.
"""

from __future__ import annotations

from threetears.core.security.secret_refs import (
    Resolver,
    SecretResolutionError,
    parse_ref,
    register_scheme,
    resolve_secret,
    validate_ref,
)

__all__ = [
    "Resolver",
    "SecretResolutionError",
    "parse_ref",
    "register_scheme",
    "resolve_secret",
    "validate_ref",
]
