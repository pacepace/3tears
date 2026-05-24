"""agent-wake v005: open the ``verification_scheme`` CHECK constraint.

shard-06's :class:`~threetears.channels.webhook.WebhookReceiver`
dispatches HMAC verification through a runtime registry
(:meth:`~threetears.channels.webhook.WebhookReceiver.register_verifier`)
so consumers can plug in vendor-specific schemes
(``'github'``, ``'stripe'``, ``'slack_signing'``, ...) without
modifying the platform. v003 declared a CHECK constraint pinning
``verification_scheme`` to the single value ``'generic_hmac_sha256'``
-- which made the registry useless because the schema rejected any
row whose scheme wasn't in the hardcoded list, forcing an ALTER
TABLE every time a new vendor scheme landed.

The database cannot validate a scheme against the in-process
registry (the registry is consumer-supplied at app-startup), so the
CHECK constraint cannot enforce "registered" semantics. What it CAN
enforce is the slug shape: scheme names must look like an
identifier (lowercase, digits, underscores, length 1-64) so a
typo'd scheme is caught at INSERT time rather than at handle time
as an opaque "unknown scheme" 400.

This migration:

1. Drops the old ``webhook_subscriptions_verification_scheme_check``
   (which pinned the value to the single string ``'generic_hmac_sha256'``).
2. Adds a new constraint of the same name that enforces the slug
   format (``^[a-z0-9_]+$``, length 1-64). The "is the scheme
   actually registered" check runs at handle time in the receiver
   layer (unknown scheme -> 400 with the scheme name in the body).

Both statements use the canonical ``DROP CONSTRAINT IF EXISTS`` +
``ADD CONSTRAINT`` pair (PostgreSQL has no ``ALTER CONSTRAINT`` for
CHECK predicates). PostgreSQL also has no ``ADD CONSTRAINT IF NOT
EXISTS``; idempotency comes from the unconditional drop-then-add.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "open_verification_scheme_check",
]

log = get_logger(__name__)


# Drop the old hardcoded-value CHECK + add the format-guard CHECK in
# its place. The constraint name is reused so a follow-up audit /
# pg_dump comparison still finds a single ``webhook_subscriptions_verification_scheme_check``
# constraint -- only the predicate changes.
_DROP_OLD_CHECK_SQL = (
    "ALTER TABLE webhook_subscriptions DROP CONSTRAINT IF EXISTS webhook_subscriptions_verification_scheme_check"
)


# The regex matches slug-shaped identifiers (lowercase letters, digits,
# underscores); the length guard (1-64) prevents both empty strings and
# pathologically long values that would bloat the table + complicate
# index lookups. Vendor schemes follow this shape by convention
# (``github``, ``stripe``, ``slack_signing``, ``generic_hmac_sha256``).
_ADD_NEW_CHECK_SQL = (
    "ALTER TABLE webhook_subscriptions "
    "ADD CONSTRAINT webhook_subscriptions_verification_scheme_check "
    "CHECK (verification_scheme ~ '^[a-z0-9_]+$' "
    "AND length(verification_scheme) BETWEEN 1 AND 64)"
)


async def open_verification_scheme_check(store: DataStore) -> None:
    """Replace the hardcoded-value CHECK with a slug-format guard.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info(
        "opening webhook_subscriptions.verification_scheme CHECK "
        "(replacing hardcoded value with slug-format guard) (v005)",
    )
    await store.execute(_DROP_OLD_CHECK_SQL)
    await store.execute(_ADD_NEW_CHECK_SQL)
