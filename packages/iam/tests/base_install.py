"""What iam's base install is, named once for every test that holds it to its promise.

``test_imports_without_core`` runs the base modules with these roots refused,
``test_import_cost`` measures what importing each costs with them refused, and
``tests/enforcement/test_package_boundaries`` reads the source for a module-level import of
one. Three copies of these lists would let one test quietly check a different install.
"""

from __future__ import annotations

__all__ = ["BASE_MODULES", "REFUSED_ROOTS"]

#: what the base install leaves out: core, the NATS client and the agent layer, and the drivers
#: they bring
REFUSED_ROOTS = (
    "threetears.core",
    "threetears.agent",
    "threetears.nats",
    "sqlalchemy",
    "asyncpg",
    "nats",
    "nkeys",
)

#: every module the base install imports. Listed by name rather than discovered, so a new module
#: is a deliberate addition to the promise; ``dpop``, ``stores.attempt_limiter`` (the ``core``
#: extra) and ``saml`` (the ``saml`` extra) are out of it on purpose.
BASE_MODULES = (
    "threetears.iam",
    "threetears.iam.apikeys",
    "threetears.iam.breach",
    "threetears.iam.buckets",
    "threetears.iam.claim_mapping",
    "threetears.iam.clientip",
    "threetears.iam.connection_types",
    "threetears.iam.dpop_client",
    "threetears.iam.github",
    "threetears.iam.oauth_state",
    "threetears.iam.oidc",
    "threetears.iam.passwords",
    "threetears.iam.pkce",
    "threetears.iam.rotation",
    "threetears.iam.stepup",
    "threetears.iam.stores",
    "threetears.iam.stores.memory",
    "threetears.iam.stores.nats_kv",
    "threetears.iam.stores.postgres",
    "threetears.iam.tokens",
    "threetears.iam.totp",
    "threetears.iam.webauthn",
)
