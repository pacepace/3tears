"""agent-wake v002: create ``wake_fires`` table + indexes.

One row per wake fire (history; success / silent / yielded / skipped
/ failed). Partition column ``conversation_id``; composite primary
key ``(conversation_id, fire_id)``.

The table supports BOTH schedule-source and webhook-source fires via
a ``(schedule_id, webhook_subscription_id)`` mutually-exclusive CHECK
constraint -- at most one is non-null per row. Both-NULL is permitted
because the subscription-side FK is ``ON DELETE SET NULL`` (audit
history outlives a subscription delete; the row then has no source
referent). A strict exclusive-OR ("exactly one non-null") would
contradict the SET NULL behaviour the spec mandates; the
"mutually-exclusive but both-null permitted" form is the resolution.
The dispatcher path always populates exactly one source at insert
time -- the CHECK protects against bugs in that path without blocking
the legitimate orphan-on-source-delete state.

FK ``schedule_id REFERENCES agent_wake_schedules(schedule_id) ON
DELETE CASCADE`` -- relies on the standalone ``UNIQUE (schedule_id)``
constraint declared in v001. Deleting a schedule cascades and
removes the fire history (analytics for a deleted schedule are noise,
not audit). The schedule's owning ``conversation_id`` is denormalised
onto the fire row so the fire history survives a schedule recreate
inside the same conversation.

FK on ``webhook_subscription_id`` is deferred to v003 because
``webhook_subscriptions`` does not exist yet at this point. v003
retro-adds the FK via an idempotent ``DO $$ ... $$`` block that
checks ``pg_constraint`` before adding.

NO FK on ``conversation_id`` (same legal reason as agent-tools'
``context_items`` and agent-skills' ``agent_skill_invocations``: the
3tears ``conversations`` table has composite PK
``(agent_id, conversation_id)`` and no standalone UNIQUE on
``conversation_id``).

The ``status`` enum after the 2026-05-19 wake-yield revision
(PLACEMENT §8.5.1) gained the ``'yielded'`` value. Original v002
enum: ``'fired'``, ``'fired_silent'``, ``'yielded'``,
``'skipped_busy'``, ``'skipped_rate_limit'``, ``'skipped_cap'``,
``'skipped_no_handler'``, ``'failed'``. CHECK-pinned. The
``'dispatching'`` placeholder was added in v004 (see
``v004_add_dispatching_status.py``).

``display_suppressed`` boolean default ``false``: ``true`` when the
agent emitted ``[SILENT]`` (status = ``'fired_silent'``); the product
reads this when rendering to apply ``messages.display='hidden'``.

Every statement is idempotent
(``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS``).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_wake_fires",
]

log = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wake_fires (
    conversation_id           UUID         NOT NULL,
    fire_id                   UUID         NOT NULL,
    schedule_id               UUID,
    webhook_subscription_id   UUID,
    scheduled_fire_at         TIMESTAMPTZ,
    actual_fired_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    status                    TEXT         NOT NULL,
    display_suppressed        BOOLEAN      NOT NULL DEFAULT false,
    output_text               TEXT,
    latency_ms                INTEGER,
    error                     TEXT,
    date_created              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, fire_id),
    UNIQUE (fire_id),
    CONSTRAINT wake_fires_schedule_fk
        FOREIGN KEY (schedule_id)
        REFERENCES agent_wake_schedules(schedule_id)
        ON DELETE CASCADE,
    CONSTRAINT wake_fires_one_source_check CHECK (
        NOT (schedule_id IS NOT NULL AND webhook_subscription_id IS NOT NULL)
    ),
    CONSTRAINT wake_fires_status_check
        CHECK (status IN (
            'fired',
            'fired_silent',
            'yielded',
            'skipped_busy',
            'skipped_rate_limit',
            'skipped_cap',
            'skipped_no_handler',
            'failed'
        ))
)
"""


# Per-schedule history index (descending time for "latest fire" hot
# path). Partial because webhook fires have a NULL schedule_id and
# should not bloat this index.
_CREATE_SCHEDULE_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_fires_schedule_time "
    "ON wake_fires (schedule_id, actual_fired_at DESC) "
    "WHERE schedule_id IS NOT NULL"
)


# Per-subscription history index (descending time). Partial for the
# same symmetric reason as above.
_CREATE_WEBHOOK_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_fires_webhook_time "
    "ON wake_fires (webhook_subscription_id, actual_fired_at DESC) "
    "WHERE webhook_subscription_id IS NOT NULL"
)


# Per-conversation history index (descending time). Powers the
# ``list_for_conversation`` query and the ``count_in_window``
# rate-limit aggregate.
_CREATE_CONV_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_fires_conv_time ON wake_fires (conversation_id, actual_fired_at DESC)"
)


async def create_wake_fires(store: DataStore) -> None:
    """Create ``wake_fires`` table + indexes.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating wake_fires table (v002)")
    await store.execute(_CREATE_TABLE_SQL)
    await store.execute(_CREATE_SCHEDULE_TIME_INDEX_SQL)
    await store.execute(_CREATE_WEBHOOK_TIME_INDEX_SQL)
    await store.execute(_CREATE_CONV_TIME_INDEX_SQL)
