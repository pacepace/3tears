# agent-wake-01: Schema + collections + entities

> **REMOVED 2026-05-24:** the outbound delivery framework was removed as an undesigned parallel abstraction. The `delivery_target` / `delivery_config` columns on `agent_wake_schedules` and `webhook_subscriptions`, and the `delivery_target_resolved` / `delivery_status` columns on `wake_fires`, are GONE. Wake fires now always deliver into the conversation; outbound delivery, if ever needed, will be a threetears.channels adapter. Inbound webhooks are unaffected. The DDL below retains these columns for history — do NOT recreate them.

## 2026-05-19 revision deltas (apply BEFORE implementing)

The design conversation that followed the initial redesign produced these mandatory deltas. Canonical source: `<metallm>/docs/long_running/PLACEMENT.md` and `<metallm>/docs/skills/PLACEMENT.md`.

**Tables DROPPED from this shard's scope:**
- `wake_pre_check_types` — pre-checks are ordinary `TearsTool` subclasses with `tool_eligible=False, skill_eligible=True` registered in `3tears-agent-tools`. The wake's attached skill surfaces them. PLACEMENT (long_running) §1.2.
- `wake_schedule_skill_attachments` — one skill per wake max; nullable FK column on `agent_wake_schedules` instead. PLACEMENT §1.1 / §1.3.
- `webhook_subscription_skill_attachments` — same: nullable `default_skill_id` FK column on `webhook_subscriptions`.

**Columns DROPPED from `agent_wake_schedules`:**
- `no_agent` — `no_agent` mode is gone (PLACEMENT §1.6). Same effect via `prompt_mode='replace'` skill + `[SILENT]` instruction.
- `pre_check_type` — pre-checks are tools the attached skill surfaces.
- `pre_check_config` — same.

**Columns ADDED to `agent_wake_schedules`:**
- `skill_id UUID NULL REFERENCES agent_skills(skill_id) ON DELETE SET NULL` — single attached skill (PLACEMENT §1.1).
- `missed_fire_policy TEXT NOT NULL DEFAULT 'coalesce'` — values `'coalesce' | 'catch_up'` (PLACEMENT §1.7).
- `actual_fired_at TIMESTAMPTZ` on `wake_fires` if not already present — for drift recording (PLACEMENT §1.8).

**Columns ADDED to `webhook_subscriptions`:**
- `default_skill_id UUID NULL REFERENCES agent_skills(skill_id) ON DELETE SET NULL`.

**`wake_fires.status` CHECK constraint enum gains `'yielded'`** (NEW per wake-yield, see `metallm/docs/long_running/shard-10-cooperative-yield.md` YIELD-05). Full enum after the addition: `'fired'`, `'fired_silent'`, `'yielded'`, `'skipped_busy'`, `'skipped_rate_limit'`, `'skipped_cap'`, `'skipped_no_handler'`, `'failed'`.

**Requirements DROPPED:** WAKE-03 (pre-check-types table), WAKE-05 (wake-skill junction), WAKE-07 (webhook-skill junction), WAKE-14 (pre-check-type seeding), WAKE-16 (system_settings seeds for pre-check config).

**Net table count:** 3 (was 6) — `agent_wake_schedules`, `wake_fires`, `webhook_subscriptions`.

## Objective

Land the foundational data model for the agent-wake capability inside a new `3tears-agent-wake` package (`packages/agent/wake/`). **Three tables (per the 2026-05-19 revision above)**, entity classes, `SchemaBackedCollection` declarations, and an agent-scope migration registration that `depends_on=("conversations", "agent_skills")`.

This shard is platform-side schema only. Product-specific columns —
notably the `messages.source` + `messages.display` discriminators that
let consumers render wake-source messages distinctly — stay on the
product's own messages table and ship via the product's migration
(metallm side, long_running shard 01).

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| WAKE-01 | New package `3tears-agent-wake` at `packages/agent/wake/` with `pyproject.toml` declaring `name="3tears-agent-wake"`, version tracking the workspace canonical (bumped in lock-step by `scripts/bump-version.sh`), import path `threetears.agent.wake`. | P0 |
| WAKE-02 | New table `agent_wake_schedules` with columns + indexes per the schema spec below. | P0 |
| WAKE-03 | New table `wake_pre_check_types` registering the curated set of safe built-in pre-check operations (`http_get`, `loki_query`, `postgres_query` in v1; admin-extensible). | P0 |
| WAKE-04 | New table `wake_fires` recording every fired wake (success or failure), supporting BOTH schedule-source and webhook-source fires via a `(schedule_id, webhook_subscription_id)` exclusive-OR CHECK constraint. | P0 |
| WAKE-05 | New table `wake_schedule_skill_attachments` (junction) linking `agent_wake_schedules.schedule_id` → `agent_skills.skill_id` (FK target name pinned by skills redesign; declared via `depends_on`). | P0 |
| WAKE-06 | New table `webhook_subscriptions` for inbound webhook → wake adapter (full schema below; secret stored Fernet-encrypted). | P0 |
| WAKE-07 | New table `webhook_subscription_skill_attachments` (junction) linking `webhook_subscriptions.subscription_id` → `agent_skills.skill_id`. | P0 |
| WAKE-08 | FK `wake_fires.webhook_subscription_id → webhook_subscriptions(subscription_id) ON DELETE SET NULL` added in the migration that creates `webhook_subscriptions` (chronologically after `wake_fires`). | P0 |
| WAKE-09 | All migrations idempotent. `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`. `INSERT ... ON CONFLICT DO NOTHING` for seed data. Defensive `DO $$ ... $$` blocks for FK retro-add. | P0 |
| WAKE-10 | All UUID primary keys are UUIDv7 — enforce via the existing `3tears-enforcement` UUIDv7 audit pattern. | P0 |
| WAKE-11 | Entity classes follow the existing `3tears-conversations` entity pattern (BaseEntity subclass, change tracking). | P0 |
| WAKE-12 | Collections follow the `SchemaBackedCollection` pattern from `3tears-core`. CRUD is declarative; domain methods (`list_due_for_fire`, `claim_and_reschedule`, `count_in_window`) stay hand-written. | P0 |
| WAKE-13 | Migrations registered via `PackageMigrations(name="agent_wake", scope=MigrationScope.AGENT, depends_on=("conversations", "agent_skills"))`. One callable per file (`v001_create_agent_wake_schedules.py`, etc.). | P0 |
| WAKE-14 | The `WakePreCheckType` rows seeded by v002 use the exact `config_jsonschema` shapes documented in this shard. New types must go through admin endpoints (defined by consuming product), not direct INSERTs. | P0 |
| WAKE-15 | `agent_wake_schedules.context_from_schedule_id` FK is self-referential with `ON DELETE SET NULL`. App-layer cycle detection at create/update time (lives in shard 04's agent tools + the product's REST surface). | P0 |
| WAKE-16 | Per-package `system_settings` seeds for the executor framework: `wake_pre_check_http_allowed_hosts = '[]'::jsonb`, `wake_named_postgres_queries = '{}'::jsonb`, `wake_named_loki_queries = '{}'::jsonb`. **Decision point**: `system_settings` is a product-owned table in metallm today. For the platform release, expose these settings via a `WakeConfig` protocol (shard 05) the consumer satisfies; the platform does NOT seed rows into a product-owned table. metallm side seeds its own `system_settings` rows in its product-side migration. | P0 |

---

## Design Context

### Why a new `3tears-agent-wake` package (not an extension)

Considered:

- **Extend `3tears-conversations`** — rejected. The conversations
  package is intentionally narrow ("the conversation entity plus the
  collection"). Stacking the wake/tick/dispatch machinery on it
  crosses the bounded-context line.
- **Add to `3tears-agent-memory`** — rejected. Wake schedules aren't
  memory rows. Different lifecycle, different access pattern, different
  consumer surface.
- **Add to `3tears-agent-tools`** — rejected. Tools are one consumer of
  the wake feature, not the home of it.

A new `3tears-agent-wake` is the right home: own bounded context,
multi-package dependencies (`conversations`, `agent_skills`,
`agent-tools`, `nats`), clear single-package consumer surface for
products.

### Why `messages.source` / `messages.display` aren't here

Those columns live on the product's `messages` table. 3tears doesn't
own a canonical `messages` table (different products have different
shapes for messages — metallm's messages are LLM-conversation messages,
aibots' messages are channel messages, etc.). The product extends its
own messages table to carry the wake-source discriminator + display
flag and reads them when rendering. The platform doesn't impose a
schema on the product's messages.

This is the same pattern as `3tears-conversations` v005 (search_vector
on conversations) NOT extending into the product's messages table —
search_vector covers conversation titles; product messages-side FTS
stayed product-owned.

---

## Schema specification

### `agent_wake_schedules`

```sql
CREATE TABLE IF NOT EXISTS agent_wake_schedules (
    schedule_id              UUID         PRIMARY KEY,        -- uuid7
    conversation_id          UUID         NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    user_id                  UUID         NOT NULL,            -- no FK; users table is product-owned, platform does not assume its shape
    schedule_type            TEXT         NOT NULL,            -- 'daily_at' | 'every_n_hours' | 'random_within_window' | 'one_shot_at' | 'cron' | 'relative_delay'
    schedule_config          JSONB        NOT NULL,            -- shape varies by schedule_type; validated app-side in shard 04
    task_prompt              TEXT,                              -- nullable; if set, used as the wake prompt; if null, default scheduled-check-in prompt
    execution_mode           TEXT         NOT NULL DEFAULT 'inline',  -- 'inline' | 'spawn'
    status                   TEXT         NOT NULL DEFAULT 'active',  -- 'active' | 'paused' | 'expired'
    next_fire_at             TIMESTAMPTZ,                       -- null when status != 'active' or expired one-shot
    last_fired_at            TIMESTAMPTZ,
    name                     TEXT,                              -- optional human-readable name e.g. "morning check-in"
    no_agent                 BOOLEAN      NOT NULL DEFAULT false,  -- when true, pre_check output IS the message; LLM never invoked
    pre_check_type           TEXT         REFERENCES wake_pre_check_types(type_id),  -- nullable; named pre-check operation
    pre_check_config         JSONB        NOT NULL DEFAULT '{}',  -- validated against wake_pre_check_types.config_jsonschema
    context_from_schedule_id UUID         REFERENCES agent_wake_schedules(schedule_id) ON DELETE SET NULL,
    delivery_target          TEXT         NOT NULL DEFAULT 'conversation',  -- 'conversation' | 'email' | (future targets)
    delivery_config          JSONB        NOT NULL DEFAULT '{}',
    date_created             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    date_updated             TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wake_schedules_next_fire
    ON agent_wake_schedules (next_fire_at)
    WHERE status = 'active' AND next_fire_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_wake_schedules_conv_status
    ON agent_wake_schedules (conversation_id, status);

CREATE INDEX IF NOT EXISTS idx_wake_schedules_user
    ON agent_wake_schedules (user_id);

CREATE INDEX IF NOT EXISTS idx_wake_schedules_context_from
    ON agent_wake_schedules (context_from_schedule_id)
    WHERE context_from_schedule_id IS NOT NULL;
```

**`user_id` is NOT a foreign key.** The platform does not assume a
canonical `users` table (metallm has `users`, aibots has `customers`,
etc.). The collection treats `user_id` as opaque — the consumer
validates ownership at the REST / agent-tool layer.

### `wake_pre_check_types`

```sql
CREATE TABLE IF NOT EXISTS wake_pre_check_types (
    type_id            TEXT         PRIMARY KEY,
    display_name       TEXT         NOT NULL,
    description        TEXT,
    config_jsonschema  JSONB        NOT NULL,
    requires_admin     BOOLEAN      NOT NULL DEFAULT false,
    date_created       TIMESTAMPTZ  NOT NULL DEFAULT now()
);

INSERT INTO wake_pre_check_types (type_id, display_name, description, config_jsonschema, requires_admin) VALUES
    ('http_get',
     'HTTP GET',
     'Fetch a URL. Response body is the pre-check output. Output is empty (skip) if the response status code is not 2xx.',
     '{"type":"object","required":["url"],"properties":{"url":{"type":"string","format":"uri"},"headers":{"type":"object","additionalProperties":{"type":"string"}},"timeout_seconds":{"type":"integer","minimum":1,"maximum":30,"default":10},"max_response_bytes":{"type":"integer","maximum":1048576,"default":65536}}}'::jsonb,
     false),
    ('loki_query',
     'Loki LogQL Query (admin-curated)',
     'Run a LogQL query from the admin-curated named-query list (inline LogQL is rejected). Output is the matching log lines (newline-joined) or empty if zero matches.',
     '{"type":"object","required":["named_query"],"properties":{"named_query":{"type":"string"},"params":{"type":"object"},"range_minutes":{"type":"integer","minimum":1,"maximum":1440,"default":60},"limit":{"type":"integer","maximum":500,"default":50}}}'::jsonb,
     true),
    ('postgres_query',
     'Postgres SQL Query (admin-curated)',
     'Run a parameterized SQL query from the admin-curated named-query list (inline SQL is rejected). Output is the result rows formatted as TSV.',
     '{"type":"object","required":["named_query"],"properties":{"named_query":{"type":"string"},"params":{"type":"object"}}}'::jsonb,
     true)
ON CONFLICT (type_id) DO NOTHING;
```

### `wake_fires`

```sql
CREATE TABLE IF NOT EXISTS wake_fires (
    fire_id                   UUID         PRIMARY KEY,
    schedule_id               UUID         REFERENCES agent_wake_schedules(schedule_id) ON DELETE CASCADE,  -- nullable for webhook fires
    webhook_subscription_id   UUID,                              -- nullable for scheduled fires; FK added by v004 migration
    conversation_id           UUID         NOT NULL,             -- denormalized; not FK so fires outlive deleted conversations
    fired_at                  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    fire_source               TEXT         NOT NULL,
    execution_mode            TEXT         NOT NULL,
    target_conversation_id    UUID,                              -- spawn: new conv; inline: same as conversation_id
    status                    TEXT         NOT NULL,             -- 'dispatching' | 'fired' | 'skipped_busy' | 'skipped_gate' | 'failed' | 'rate_limited'
    error                     TEXT,
    duration_ms               INTEGER,
    pre_check_output          TEXT,
    pre_check_duration_ms     INTEGER,
    delivery_target_resolved  TEXT         NOT NULL DEFAULT 'conversation',
    delivery_status           TEXT,                              -- 'delivered' | 'delivered_email_failed' | 'suppressed_silent'
    display_suppressed        BOOLEAN      NOT NULL DEFAULT false,
    CONSTRAINT chk_wake_fires_one_source CHECK (
        (schedule_id IS NOT NULL AND webhook_subscription_id IS NULL)
        OR (schedule_id IS NULL AND webhook_subscription_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_wake_fires_schedule_time
    ON wake_fires (schedule_id, fired_at DESC)
    WHERE schedule_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_wake_fires_webhook_time
    ON wake_fires (webhook_subscription_id, fired_at DESC)
    WHERE webhook_subscription_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_wake_fires_conv_time
    ON wake_fires (conversation_id, fired_at DESC);

CREATE INDEX IF NOT EXISTS idx_wake_fires_conv_time_status
    ON wake_fires (conversation_id, fired_at DESC, status);
```

### `wake_schedule_skill_attachments`

```sql
CREATE TABLE IF NOT EXISTS wake_schedule_skill_attachments (
    schedule_id   UUID NOT NULL REFERENCES agent_wake_schedules(schedule_id) ON DELETE CASCADE,
    skill_id      UUID NOT NULL REFERENCES agent_skills(skill_id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    date_created  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (schedule_id, skill_id)
);

CREATE INDEX IF NOT EXISTS idx_wake_skill_attach_position
    ON wake_schedule_skill_attachments (schedule_id, position);
```

The FK target `agent_skills(skill_id)` is pinned by the parallel skills
redesign. `depends_on=("agent_skills",)` in the migration registration
ensures the FK target exists before this migration runs.

### `webhook_subscriptions`

```sql
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    subscription_id        UUID PRIMARY KEY,
    conversation_id        UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    user_id                UUID NOT NULL,
    name                   TEXT,
    secret_ciphertext      BYTEA NOT NULL,             -- Fernet-encrypted HMAC secret; raw secret returned ONCE on create + on rotate
    allowed_source_pattern TEXT,                       -- optional regex against the source IP / sender; null = any
    execution_mode         TEXT NOT NULL DEFAULT 'inline',
    task_prompt_template   TEXT,                       -- Jinja2 sandboxed template; variables: {{event}}, {{event.field}}
    delivery_target        TEXT NOT NULL DEFAULT 'conversation',
    delivery_config        JSONB NOT NULL DEFAULT '{}',
    verification_scheme    TEXT NOT NULL DEFAULT 'generic_hmac_sha256',
    status                 TEXT NOT NULL DEFAULT 'active',
    last_fired_at          TIMESTAMPTZ,
    date_created           TIMESTAMPTZ NOT NULL DEFAULT now(),
    date_updated           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_webhook_subs_conv ON webhook_subscriptions (conversation_id, status);
CREATE INDEX IF NOT EXISTS idx_webhook_subs_user ON webhook_subscriptions (user_id);

CREATE TABLE IF NOT EXISTS webhook_subscription_skill_attachments (
    subscription_id  UUID NOT NULL REFERENCES webhook_subscriptions(subscription_id) ON DELETE CASCADE,
    skill_id         UUID NOT NULL REFERENCES agent_skills(skill_id) ON DELETE CASCADE,
    position         INTEGER NOT NULL,
    date_created     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (subscription_id, skill_id)
);

CREATE INDEX IF NOT EXISTS idx_webhook_skill_attach_position
    ON webhook_subscription_skill_attachments (subscription_id, position);

-- Retro-add FK on wake_fires now that webhook_subscriptions exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_wake_fires_webhook_subscription_id'
    ) THEN
        ALTER TABLE wake_fires
            ADD CONSTRAINT fk_wake_fires_webhook_subscription_id
            FOREIGN KEY (webhook_subscription_id)
            REFERENCES webhook_subscriptions(subscription_id)
            ON DELETE SET NULL;
    END IF;
END $$;
```

**Secret storage rationale**: HMAC-SHA256 verification needs the RAW
secret to recompute `HMAC(secret, body)` and compare. A one-way hash
(bcrypt/argon2) cannot reproduce the HMAC. The platform uses Fernet
encryption — the consumer supplies an `EncryptionService` protocol
implementation (3tears does not own one canonical encryption service;
metallm has `src.services.encryption`, aibots has its own, etc.).
Display-once UX is preserved at the agent-tool / REST layer: raw secret
returned only on create + rotate; GET returns no secret field.

---

## Migration sequence

Six migration callables, one per file:

| Ver | File | DDL |
|---|---|---|
| v001 | `v001_create_agent_wake_schedules.py` | `agent_wake_schedules` + four indexes. |
| v002 | `v002_create_wake_pre_check_types.py` | `wake_pre_check_types` + the three seed rows. |
| v003 | `v003_create_wake_fires.py` | `wake_fires` + four indexes + the CHECK constraint. |
| v004 | `v004_create_webhook_subscriptions.py` | `webhook_subscriptions` + two indexes. Also retro-adds the FK on `wake_fires.webhook_subscription_id` via the `DO $$ ... $$` block. |
| v005 | `v005_create_wake_schedule_skill_attachments.py` | `wake_schedule_skill_attachments` + position index. |
| v006 | `v006_create_webhook_subscription_skill_attachments.py` | `webhook_subscription_skill_attachments` + position index. |

Each callable lives in its own file and follows the
`3tears/docs/how-to-add-a-migration.md` template:

```python
# packages/agent/wake/src/threetears/agent/wake/migrations/v001_create_agent_wake_schedules.py
from __future__ import annotations
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

log = get_logger(__name__)

_CREATE_AGENT_WAKE_SCHEDULES_SQL = """..."""  # the DDL above

async def create_agent_wake_schedules(store: DataStore) -> None:
    """create agent_wake_schedules table + indexes."""
    log.info("creating agent_wake_schedules")
    await store.execute(_CREATE_AGENT_WAKE_SCHEDULES_SQL)
```

`migrations/__init__.py` exports `register(runner)`:

```python
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)
from threetears.agent.wake.migrations.v001_create_agent_wake_schedules import create_agent_wake_schedules
# ... imports for v002..v006 ...

PACKAGE_NAME = "agent_wake"

def register(runner: MigrationRunner) -> PackageMigrations:
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
        depends_on=("conversations", "agent_skills"),
    )
    pkg.version(1)(create_agent_wake_schedules)
    pkg.version(2)(create_wake_pre_check_types)
    pkg.version(3)(create_wake_fires)
    pkg.version(4)(create_webhook_subscriptions)
    pkg.version(5)(create_wake_schedule_skill_attachments)
    pkg.version(6)(create_webhook_subscription_skill_attachments)
    runner.register(pkg)
    return pkg
```

The consumer (metallm hub broker `build_agent_runner`) wires
`register_agent_wake(runner)` alongside the existing register calls.

---

## Entity + collection structure

### Entities

`packages/agent/wake/src/threetears/agent/wake/entity.py` exports:

- `WakeSchedule(BaseEntity)` — all columns of `agent_wake_schedules`. Status transitions (`active → paused → active`, `active → expired`) are entity methods.
- `WakeFire(BaseEntity)` — all columns of `wake_fires`. Immutable post-finalize (status moves once: `dispatching → fired/skipped_busy/skipped_gate/failed/rate_limited`).
- `WakePreCheckType(BaseEntity)` — all columns of `wake_pre_check_types`. Read-mostly; admin-only writes.
- `WakeScheduleSkillAttachment(BaseEntity)` — junction row.
- `WebhookSubscription(BaseEntity)` — all columns of `webhook_subscriptions`. `secret_ciphertext` is bytes; entity exposes a `decrypt_secret(encryption_service)` method that returns the raw secret only when invoked.
- `WebhookSubscriptionSkillAttachment(BaseEntity)` — junction row.

Follow the `Conversation` entity pattern from `3tears-conversations`
(`packages/conversations/src/threetears/conversations/entity.py`).

### Collections

`packages/agent/wake/src/threetears/agent/wake/collections.py` exports
SchemaBackedCollection subclasses:

- `WakeScheduleCollection(SchemaBackedCollection[WakeSchedule])` — `schema = TableSchema(...)`; domain methods `list_for_conversation`, `list_due_for_fire`, `claim_and_reschedule`, `pause`, `resume`.
- `WakeFireCollection(SchemaBackedCollection[WakeFire])` — domain methods `list_for_schedule`, `list_for_conversation`, `list_for_subscription`, `count_in_window`.
- `WakePreCheckTypeCollection(SchemaBackedCollection[WakePreCheckType])` — domain methods `list_available_for_user(is_admin)`, `get_by_id`.
- `WakeScheduleSkillAttachmentCollection(SchemaBackedCollection[WakeScheduleSkillAttachment])` — domain methods `list_for_schedule(schedule_id)` returning ordered by `position`.
- `WebhookSubscriptionCollection(SchemaBackedCollection[WebhookSubscription])` — domain methods `list_for_conversation`, `list_for_user`, `pause`, `resume`, `rotate_secret(encryption_service)`.
- `WebhookSubscriptionSkillAttachmentCollection(SchemaBackedCollection[WebhookSubscriptionSkillAttachment])` — symmetric to `WakeScheduleSkillAttachmentCollection`.

Pattern: `3tears-conversations` `ConversationsCollection`
(`packages/conversations/src/threetears/conversations/collection.py`).

---

## Patterns to Follow

- Entity shape: `3tears-conversations` `Conversation`.
- Collection shape: `3tears-core` `SchemaBackedCollection` + `3tears-conversations` `ConversationsCollection`.
- Migration template: `3tears/docs/how-to-add-a-migration.md` (top-of-shard-set reference).
- TableSchema enrichment: `3tears` v0.8.0 (already released) — use `Column.foreign_key=("table","col")` for single FKs, `TableSchema.foreign_keys=[ForeignKey(...)]` for composite. `Column.server_default` for `DEFAULT now()`. `enum_type=...` for status enums.

---

## Files to Create

### Package skeleton

- `packages/agent/wake/pyproject.toml` — package metadata, depends on `3tears>=0.9.0`, `3tears-conversations>=0.9.0`, `3tears-observe`, `3tears-nats`, `3tears-agent-skills` (or whichever name the skills redesign settles).
- `packages/agent/wake/src/threetears/agent/wake/__init__.py` — package `__all__`.
- `packages/agent/wake/src/threetears/agent/wake/py.typed`.
- `packages/agent/wake/src/threetears/agent/wake/entity.py` — six entity classes.
- `packages/agent/wake/src/threetears/agent/wake/collections.py` — six collection classes with `SchemaBackedCollection` schema declarations.
- `packages/agent/wake/src/threetears/agent/wake/migrations/__init__.py` — `register(runner)`.
- `packages/agent/wake/src/threetears/agent/wake/migrations/v001_create_agent_wake_schedules.py` through `v006_create_webhook_subscription_skill_attachments.py`.

### Tests

- `packages/agent/wake/tests/unit/test_migration_idempotency.py` — `_CaptureStore` stub assertions on each migration callable.
- `packages/agent/wake/tests/integration/test_agent_wake_migrations.py` — apply all six migrations against a real Postgres / testcontainer; insert a row of every `schedule_type`; verify FK cascade on conversation delete; verify CHECK constraint on `wake_fires` rejects "neither source" and "both sources" inserts.
- `packages/agent/wake/tests/unit/test_collections.py` — round-trip create/get/update/delete for each collection against a fake pool.

---

## Implementation Notes

1. **`schedule_config` JSONB shape per type.** Document in the collection module's docstring; do NOT enforce at the DB layer (validation belongs in the agent tools + product REST layer):
   - `daily_at`: `{"hour": 14, "minute": 0, "tz": "America/Los_Angeles"}`
   - `every_n_hours`: `{"n": 3}`
   - `random_within_window`: `{"start_hour": 9, "end_hour": 21, "tz": "America/Los_Angeles", "fires_per_day": 1}`
   - `one_shot_at`: `{"fire_at_iso": "2026-05-25T14:00:00+00:00"}`
   - `cron`: `{"expr": "0 */3 * * *"}`
   - `relative_delay`: `{"delay": "30m"}`

2. **Status transitions.** Define on the entity:
   - `active → paused` (user/agent paused)
   - `paused → active` (resume)
   - `active → expired` (one-shot fired)
   - `expired → active` rejected.
   - All deletes via `collection.delete(id)`; no soft-delete.

3. **Why `wake_fires.conversation_id` is denormalized + not an FK.** Survives the conv FK gone when the parent schedule is being recreated. Same pattern as `memories.conversation_id` (Shard B post-v0.7.0).

4. **`next_fire_at` is set by the consumer.** The collection's `create(...)` takes `next_fire_at` as a parameter — computed by the agent tool / REST layer via `_compute_next_fire_at` from shard 02.

5. **Cycle detection for `context_from_schedule_id`.** App-layer (shard 04's agent tools + the product's REST layer). DB-layer CHECK constraints can't enforce graph acyclicity. Same-conversation check also app-layer.

6. **`pre_check_config` validation is two-layer.** `wake_pre_check_types.config_jsonschema` validates structural correctness (collection layer can run this). App layer additionally validates semantic correctness (host allow-list, named-query existence) — lives in the agent-tools shard.

7. **`webhook_subscriptions.secret_ciphertext` storage.** Bytes. The collection's `create(...)` takes the plaintext secret + an `EncryptionService` protocol instance; encrypts before insert. `rotate_secret(...)` regenerates + re-encrypts + returns the new plaintext. `get(...)` returns the entity with the ciphertext; `entity.decrypt_secret(encryption_service)` is the only way to get the plaintext.

8. **No `is_long_running` boolean on conversations.** Derived from `EXISTS (SELECT 1 FROM agent_wake_schedules WHERE conversation_id = ? AND status = 'active')`. Consumers query this if they need it.

9. **`user_id` opacity.** The collection treats `user_id` as a UUID with no FK. The consumer is responsible for validating that the user_id is real and that the requesting context is allowed to act on it.

10. **`webhook_subscriptions.task_prompt_template` is a Jinja2 sandbox template.** Variables available at render time: `{{event}}` (the entire decoded JSON payload), plus arbitrary `{{event.field.subfield}}` access. Template render lives in the webhook receiver (shard 06) using `jinja2.sandbox.SandboxedEnvironment`.

---

## Anti-patterns

- DO NOT enforce schedule_type via a DB-layer CHECK constraint. App-layer enumerated types are easier to extend.
- DO NOT add `name` as `NOT NULL`. Saoirse (or any agent consumer) will often create schedules without naming them.
- DO NOT use `gen_random_uuid()` as a column DEFAULT. UUIDs are uuid7, allocated app-side via `uuid_utils.uuid7()`.
- DO NOT make `wake_pre_check_types` writable by non-admin users at the platform layer. Admin-only — consumer products enforce this via their admin auth.
- DO NOT store the raw HMAC secret. Fernet-encrypted ciphertext only. Plaintext returned once on create + once on rotate.
- DO NOT add a CHECK constraint on `schedule_config` shape — JSONB shape varies per `schedule_type`, validation lives in the agent-tools layer.
- DO NOT FK `user_id` to a platform `users` table. There isn't one. Different products have different user tables.
- DO NOT skip the CHECK constraint on `wake_fires` `(schedule_id, webhook_subscription_id)`. The exclusive-OR invariant is what lets the per-user rate-limit query stay clean.
- DO NOT make the conversations FK ON DELETE SET NULL — it's CASCADE. A conversation being deleted should take its schedules with it.

---

## Success Criteria

- [ ] Package `3tears-agent-wake` builds + installs in the workspace.
- [ ] All six migrations apply cleanly on a fresh agent schema.
- [ ] All six migrations apply cleanly on an already-applied schema (idempotency).
- [ ] Round-trip test passes: insert a row of every `schedule_type`, FK cascade fires correctly on conversation delete.
- [ ] Collection methods pass unit tests for create/get/update/delete on each table.
- [ ] CHECK constraint on `wake_fires` rejects "neither source" and "both sources" inserts.
- [ ] Pre-check seed types are present after v002 runs.
- [ ] FK retro-add (v004) is idempotent (rerunnable safely).
- [ ] `./scripts/check-all.sh` clean across `packages/agent/wake/`.
- [ ] `depends_on=("conversations", "agent_skills")` declared correctly; runner topologically orders the migrations after both upstream packages.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run --directory packages/agent/wake pytest tests/ -v
./scripts/check-all.sh
```
