"""agent-wake v003: create ``webhook_subscriptions`` + retro-add the
``wake_fires.webhook_subscription_id`` FK.

One row per inbound HTTP webhook subscription. Partition column
``conversation_id``; composite primary key ``(conversation_id,
subscription_id)``. Standalone ``UNIQUE (subscription_id)`` so the
HTTP receiver (shard 06) can look up subscriptions by bare id from a
path parameter.

Nullable ``default_skill_id UUID REFERENCES agent_skills(skill_id)
ON DELETE SET NULL`` -- single attached default skill (PLACEMENT
§1.1). ``ON DELETE SET NULL`` so deleting the skill leaves the
subscription active but unbound. Relies on the cross-package
standalone ``UNIQUE (skill_id)`` constraint declared in
agent-skills v001.

Secret storage rationale (PLACEMENT shard-01 body): HMAC-SHA256
verification needs the RAW secret to recompute ``HMAC(secret, body)``
and compare. A one-way hash (bcrypt/argon2) cannot reproduce the
HMAC. The platform stores Fernet ciphertext (``BYTEA``) -- the
consumer supplies an ``EncryptionService`` implementation; 3tears
does not own one canonical encryption service (a consumer might
ship its own ``encryption`` service; future products will have their own).
Display-once UX (raw secret returned only on create + on rotate)
lives at the agent-tool / REST layer in shard 04.

CHECK constraints:

- ``execution_mode IN ('inline', 'spawn')`` -- CHECK-pinned.
- ``verification_scheme IN ('generic_hmac_sha256')`` -- CHECK-pinned
  on creation; REPLACED by v005 with a slug-format guard once the
  receiver's pluggable verifier registry landed (the hardcoded-value
  CHECK made the registry useless because the schema rejected any
  vendor scheme at INSERT time). See v005 for the open form.
- ``status IN ('active', 'paused')`` -- CHECK-pinned. No ``expired``
  state because webhook subscriptions are long-lived (unlike one-shot
  schedules).

NO FK on ``conversation_id`` (same legal reason as v001 / v002).

The migration ALSO retro-adds the FK on
``wake_fires.webhook_subscription_id`` via an idempotent
``DO $$ ... $$`` block. The constraint is ``ON DELETE SET NULL`` so
deleting a subscription leaves its fire history visible (audit
trails outlive subscription deletions).

Every statement is idempotent (``CREATE TABLE IF NOT EXISTS`` /
``CREATE INDEX IF NOT EXISTS`` / the retro-FK guard).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_webhook_subscriptions",
]

log = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    conversation_id         UUID         NOT NULL,
    subscription_id         UUID         NOT NULL,
    user_id                 UUID         NOT NULL,
    agent_id                UUID         NOT NULL,
    default_skill_id        UUID,
    name                    TEXT,
    secret_ciphertext       BYTEA        NOT NULL,
    allowed_source_pattern  TEXT,
    execution_mode          TEXT         NOT NULL DEFAULT 'inline',
    task_prompt_template    TEXT,
    verification_scheme     TEXT         NOT NULL DEFAULT 'generic_hmac_sha256',
    status                  TEXT         NOT NULL DEFAULT 'active',
    rate_limit_per_minute   INTEGER,
    last_fired_at           TIMESTAMPTZ,
    date_created            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    date_updated            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, subscription_id),
    UNIQUE (subscription_id),
    CONSTRAINT webhook_subscriptions_default_skill_fk
        FOREIGN KEY (default_skill_id) REFERENCES agent_skills(skill_id)
        ON DELETE SET NULL,
    CONSTRAINT webhook_subscriptions_execution_mode_check
        CHECK (execution_mode IN ('inline', 'spawn')),
    CONSTRAINT webhook_subscriptions_verification_scheme_check
        CHECK (verification_scheme IN ('generic_hmac_sha256')),
    CONSTRAINT webhook_subscriptions_status_check
        CHECK (status IN ('active', 'paused'))
)
"""


_CREATE_CONV_STATUS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_webhook_subs_conv ON webhook_subscriptions (conversation_id, status)"
)


_CREATE_USER_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_webhook_subs_user ON webhook_subscriptions (user_id)"


# Retro-add the FK on wake_fires.webhook_subscription_id. Postgres has
# no ``ADD CONSTRAINT IF NOT EXISTS``, so guard via pg_constraint
# lookup. ``ON DELETE SET NULL`` so deleting a subscription leaves
# the fire history visible.
_RETRO_ADD_FK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'wake_fires_webhook_subscription_fk'
    ) THEN
        ALTER TABLE wake_fires
            ADD CONSTRAINT wake_fires_webhook_subscription_fk
            FOREIGN KEY (webhook_subscription_id)
            REFERENCES webhook_subscriptions(subscription_id)
            ON DELETE SET NULL;
        RAISE NOTICE 'v003: added wake_fires_webhook_subscription_fk';
    ELSE
        RAISE NOTICE 'v003: wake_fires_webhook_subscription_fk already present';
    END IF;
END
$$
"""


async def create_webhook_subscriptions(store: DataStore) -> None:
    """Create ``webhook_subscriptions`` + retro-add fire-side FK.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating webhook_subscriptions table (v003)")
    await store.execute(_CREATE_TABLE_SQL)
    await store.execute(_CREATE_CONV_STATUS_INDEX_SQL)
    await store.execute(_CREATE_USER_INDEX_SQL)
    log.info("v003: retro-adding wake_fires.webhook_subscription_id FK")
    await store.execute(_RETRO_ADD_FK_SQL)
