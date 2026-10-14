"""agent-wake v010: a webhook delivery that was refused is kept.

An accepted delivery is a ``wake_fires`` row. A refused one -- a bad or
missing signature, a source that is not allowed, too many in a minute, a body
that cannot be read, a subscription that is paused -- was logged and stored
nowhere, so "nothing was sent" and "everything was refused" looked the same.

``webhook_refusals`` keeps them, counted: one row per subscription, reason and
minute, with how many deliveries were refused in that minute and when the last
one was. A sender that hammers a subscription with a wrong secret costs one
row a minute, not one row a request. Rows go with their subscription
(``ON DELETE CASCADE``).

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_webhook_refusals"]

log = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS webhook_refusals (
    conversation_id UUID NOT NULL,
    subscription_id UUID NOT NULL,
    reason TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    status_code INTEGER NOT NULL,
    refusals INTEGER NOT NULL DEFAULT 1,
    last_refused_at TIMESTAMPTZ NOT NULL,
    detail TEXT,
    CONSTRAINT webhook_refusals_pkey PRIMARY KEY (subscription_id, reason, window_start),
    CONSTRAINT webhook_refusals_subscription_fk
        FOREIGN KEY (subscription_id)
        REFERENCES webhook_subscriptions (subscription_id)
        ON DELETE CASCADE,
    CONSTRAINT webhook_refusals_reason_check
        CHECK (reason IN (
            'paused',
            'missing_signature',
            'bad_signature',
            'unknown_scheme',
            'source_not_allowed',
            'rate_limited',
            'bad_payload',
            'bad_template'
        )),
    CONSTRAINT webhook_refusals_count_check CHECK (refusals >= 1)
)
"""

_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_webhook_refusals_sub_time "
    "ON webhook_refusals (subscription_id, last_refused_at DESC)"
)


async def create_webhook_refusals(store: DataStore) -> None:
    """Create ``webhook_refusals`` and its index.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent_wake v010: refused webhook deliveries are kept")
    await store.execute(_CREATE_TABLE_SQL)
    await store.execute(_TIME_INDEX_SQL)
