# agent-skills / shard-01: Schema + Collection layer

## Objective

Land the data model for procedural memory: an `agent_skills` table (one row per skill) and an `agent_skill_invocations` table (one row per skill load, with success/fail outcome tracking). Provide the three-tier `BaseCollection` layer + entities the rest of the package builds on. Register the migrations with the canonical `MigrationRunner` so consumers get the schema as part of their agent-scope migration pass at startup. No tools, no renderer in this shard.

## Locked design decisions (canonical source: `metallm/docs/skills/PLACEMENT.md`)

| Topic | Locked answer | PLACEMENT ref |
|---|---|---|
| Skill content shape | Three optional payload fields (`prompt_addition`, `tool_additions`, `tool_restrictions`) + `prompt_mode` enum + identity + instrumentation. At least one payload field MUST be populated (DB CHECK constraint) | §1.1 |
| When skills load | Wake-driven turns (via attached `skill_id` on the wake schedule) OR explicit `skill_invoke` mid-user-turn. NO auto-load via classifier. | §1.2 |
| Composition | One skill per turn maximum. No multi-skill blending. | §1.3 |
| Authorship | Agents author prose skills only. Code-shape capabilities arrive via tool registration. No `skill_create_from_range`. | §1.4 |
| Tool eligibility | Tools carry `tool_eligible` + `skill_eligible` flags (foundation shard `agent-tools-eligibility`). Skills' `tool_additions` references tools by `mcp_name`. | §1.5 |
| Partition column | `agent_id` (collections-task-04 convention) | |
| Composite PK shape | `(agent_id, skill_id)` and `(agent_id, invocation_id)`; also `UNIQUE (skill_id)` and `UNIQUE (invocation_id)` so cross-package FKs work | |
| FTS maintenance | Trigger-maintained search_vector for `skill_list` query-filter ranking (NOT auto-load — discovery only). Mirrors `3tears-agent-memory` v005 precedent. | |
| Entity classes | `AgentSkillEntity` + `AgentSkillInvocationEntity` — both `BaseEntity` subclasses | |
| User scope | `user_id` NOT NULL on both tables; denormalised onto invocations | |

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SK-01 | New table `agent_skills` with columns + indexes specified below. Partition column = `agent_id`; composite PK `(agent_id, skill_id)`. **`prompt_mode TEXT NOT NULL DEFAULT 'additive'` enum-by-app `'additive'\|'replace'`. `tool_additions TEXT[] NOT NULL DEFAULT '{}'`. `tool_restrictions TEXT[] NOT NULL DEFAULT '{}'`. `body TEXT NULL`. CHECK constraint requiring at least one payload field non-empty.** | P0 |
| SK-02 | New table `agent_skill_invocations` recording every skill load. Partition column = `agent_id`; composite PK `(agent_id, invocation_id)`. Composite FK `(agent_id, skill_id) REFERENCES agent_skills(agent_id, skill_id) ON DELETE CASCADE`. **`invocation_source TEXT NOT NULL` enum-by-app `'wake'\|'invoke'`.** | P0 |
| SK-03 | This shard does NOT create any junction table for wake↔skill attachment. **Per PLACEMENT §1.1 / §1.3, the wake-side schema declares a nullable `skill_id` FK column directly on `agent_wake_schedules`** (one skill per wake, no junction). The FK target is `agent_skills(skill_id)` via the standalone `UNIQUE (skill_id)` constraint added by this shard. | P0 |
| SK-04 | `agent_skills.search_vector` (tsvector) — trigger-maintained over `name || ' ' || trigger_keywords || ' ' || coalesce(body, '')`, weighted A/B/C respectively. GIN index. **Used by `skill_list` for OPTIONAL query-filter ranking when the caller passes a search string. NOT used for auto-load (auto-load is dropped).** | P0 |
| SK-05 | Migrations are idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, `CREATE OR REPLACE FUNCTION` for the FTS trigger, `DROP TRIGGER IF EXISTS ... ; CREATE TRIGGER ...`). | P0 |
| SK-06 | All UUID primary keys are UUIDv7 — enforce via package's own enforcement test pattern. | P0 |
| SK-07 | Re-applying the canonical `MigrationRunner` against a fresh DB is a no-op (verified via integration test). 3tears uses its own migration runner, not alembic — the original spec text referencing `alembic --autogenerate` was inherited from metallm's planning docs. | P0 |
| SK-08 | The `success_count` / `failure_count` columns on `agent_skills` are populated by manual classification (parsing `[SUCCESS]` / `[FAILED]` markers from the agent's response in the consumer's post-LLM hook). **No automatic classifier-tick infrastructure.** | P0 |

## Schema specification

### `agent_skills`

```sql
CREATE TABLE IF NOT EXISTS agent_skills (
    agent_id          UUID         NOT NULL,
    skill_id          UUID         NOT NULL,                      -- uuid7
    user_id           UUID         NOT NULL,                      -- denormalised
    name              TEXT         NOT NULL,
    summary           TEXT         NOT NULL,                      -- one-line catalog entry (required; visible in skill_list)
    body              TEXT         NULL,                          -- prose body (markdown); nullable for pure-tool-composition skills
    prompt_mode       TEXT         NOT NULL DEFAULT 'additive',   -- 'additive' | 'replace'
    tool_additions    TEXT[]       NOT NULL DEFAULT '{}',         -- mcp_name list — tools to surface when this skill loads
    tool_restrictions TEXT[]       NOT NULL DEFAULT '{}',         -- mcp_name list — tools to remove from default surface
    trigger_keywords  TEXT         NOT NULL DEFAULT '',           -- keywords for skill_list filter; NOT for auto-load
    tags              TEXT[]       NOT NULL DEFAULT '{}',
    source            TEXT         NOT NULL DEFAULT 'manual',     -- 'manual' (only value in v1; no distillation)
    enabled           BOOLEAN      NOT NULL DEFAULT true,
    use_count         INTEGER      NOT NULL DEFAULT 0,
    last_used_at      TIMESTAMPTZ,
    success_count     INTEGER      NOT NULL DEFAULT 0,
    failure_count     INTEGER      NOT NULL DEFAULT 0,
    last_failure_at   TIMESTAMPTZ,
    date_created      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    date_updated      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    search_vector     TSVECTOR,                                    -- trigger-maintained; weighted A/B/C
    PRIMARY KEY (agent_id, skill_id),
    UNIQUE (skill_id),                                             -- so cross-package FKs can reference skill_id alone
    CHECK (prompt_mode IN ('additive', 'replace')),
    CHECK (
        body IS NOT NULL
        OR array_length(tool_additions, 1) IS NOT NULL
        OR array_length(tool_restrictions, 1) IS NOT NULL
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_skills_agent_user_name
    ON agent_skills (agent_id, user_id, name);

CREATE INDEX IF NOT EXISTS idx_skills_agent_user_enabled
    ON agent_skills (agent_id, user_id, enabled);

CREATE INDEX IF NOT EXISTS idx_skills_search_vector
    ON agent_skills USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_skills_tags
    ON agent_skills USING GIN (tags);

-- FTS trigger
CREATE OR REPLACE FUNCTION agent_skills_search_vector_update() RETURNS TRIGGER AS $$
BEGIN
  NEW.search_vector :=
      setweight(to_tsvector('english', coalesce(NEW.name, '')), 'A') ||
      setweight(to_tsvector('english', coalesce(NEW.trigger_keywords, '')), 'B') ||
      setweight(to_tsvector('english', coalesce(NEW.body, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_agent_skills_search_vector ON agent_skills;
CREATE TRIGGER trg_agent_skills_search_vector
  BEFORE INSERT OR UPDATE OF name, trigger_keywords, body ON agent_skills
  FOR EACH ROW EXECUTE FUNCTION agent_skills_search_vector_update();
```

### `agent_skill_invocations`

```sql
CREATE TABLE IF NOT EXISTS agent_skill_invocations (
    agent_id          UUID         NOT NULL,
    invocation_id     UUID         NOT NULL,                      -- uuid7
    skill_id          UUID         NOT NULL,
    user_id           UUID         NOT NULL,
    conversation_id   UUID         NOT NULL,
    message_id        UUID,                                        -- ASSISTANT-RESPONSE message_id; consumer-populated post-LLM
    invocation_source TEXT         NOT NULL,                      -- 'wake' | 'invoke'
    invoked_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    outcome           TEXT,                                        -- 'success' | 'failure' | NULL (NULL means no marker present in response)
    outcome_source    TEXT,                                        -- 'agent_marker' (parsed [SUCCESS]/[FAILED]) | 'user_feedback'
    notes             TEXT,
    PRIMARY KEY (agent_id, invocation_id),
    UNIQUE (invocation_id),
    FOREIGN KEY (agent_id, skill_id) REFERENCES agent_skills(agent_id, skill_id) ON DELETE CASCADE,
    CHECK (invocation_source IN ('wake', 'invoke')),
    CHECK (outcome IS NULL OR outcome IN ('success', 'failure'))
);

CREATE INDEX IF NOT EXISTS idx_skill_invocations_skill_time
    ON agent_skill_invocations (agent_id, skill_id, invoked_at DESC);

CREATE INDEX IF NOT EXISTS idx_skill_invocations_conv
    ON agent_skill_invocations (agent_id, conversation_id, invoked_at DESC);
```

### Why `messages` is NOT referenced

`agent_skill_invocations.message_id` is intentionally NOT an FK. `messages` is consumer-owned (metallm has it; future consumers may differ). The consumer's loader populates `message_id` post-LLM via `set_message_id`. If a message is hard-deleted, the invocation's `message_id` is left dangling — the row still records the load event for analytics.

### Why no `outcome_classified_at` / no automatic classifier

The original redesign carried a `outcome_classified_at` column + an APScheduler classifier tick that ran heuristic rules over un-classified invocations. **Both are dropped.** Outcome is populated synchronously by the consumer's post-LLM hook when it parses `[SUCCESS]`/`[FAILED]` markers from the agent's response (or `NULL` if no marker). User-feedback-driven classification is a future enhancement.

---

## Public API

```python
# threetears.agent.skills
__all__ = [
    "AgentSkillEntity",
    "AgentSkillInvocationEntity",
    "AgentSkillCollection",
    "AgentSkillInvocationCollection",
    "SkillSource",                       # Literal['manual']
    "PromptMode",                        # Literal['additive', 'replace']
    "InvocationSource",                  # Literal['wake', 'invoke']
    "SkillOutcome",                      # Literal['success', 'failure']
    "OutcomeSource",                     # Literal['agent_marker', 'user_feedback']
    "register",                          # MigrationRunner registration entry point
]
```

### `AgentSkillEntity`

`BaseEntity` subclass; composite `_id = (agent_id, skill_id)`. `primary_key_field = "skill_id"`. Field accessors mirror the column list (including `prompt_mode`, `tool_additions`, `tool_restrictions`). Change tracking through `BaseEntity.__setattr__`.

### `AgentSkillInvocationEntity`

Same shape. Composite `_id = (agent_id, invocation_id)`. `primary_key_field = "invocation_id"`.

### `AgentSkillCollection(BaseCollection)`

`partition_column = "agent_id"`. Methods (all async):

| Method | Purpose |
|---|---|
| `get(agent_id, skill_id) -> AgentSkillEntity \| None` | `BaseCollection`-provided |
| `save_entity(entity)` | INSERT or UPDATE |
| `delete(agent_id, skill_id)` | Cascade-deletes invocations |
| `find_by_name_for_user(agent_id, user_id, name) -> AgentSkillEntity \| None` | Hit `uq_skills_agent_user_name` |
| `list_for_user(agent_id, user_id, *, enabled_only=True, tag_filter=None, query=None, limit=20, offset=0) -> list[AgentSkillEntity]` | If `query` is set, applies FTS ranking; otherwise sort by `last_used_at DESC NULLS LAST, date_created DESC` |
| `count_for_user(agent_id, user_id, *, enabled_only=True) -> int` | For list-page total_count |
| `bump_use_count(agent_id, skill_ids: Sequence[UUID])` | Single UPDATE bumping `use_count` + `last_used_at` |
| `increment_outcome_counts(agent_id, skill_id, outcome: SkillOutcome)` | UPDATE bumping `success_count` / `failure_count` + `last_failure_at` |

### `AgentSkillInvocationCollection(BaseCollection)`

`partition_column = "agent_id"`. Methods:

| Method | Purpose |
|---|---|
| `get / save_entity / delete` | `BaseCollection`-provided |
| `record(agent_id, invocation)` | Convenience wrapper around `save_entity` (partition column explicit, matching the `BaseCollection` convention) |
| `list_for_skill(agent_id, skill_id, *, limit=20, offset=0, outcome_filter=None) -> list[AgentSkillInvocationEntity]` | For `skill_history` REST endpoint |
| `list_for_conversation(agent_id, conversation_id, *, limit=20) -> list[AgentSkillInvocationEntity]` | For "what skills loaded in this conversation" |
| `set_message_id(agent_id, invocation_ids: Sequence[UUID], message_id: UUID)` | Bulk UPDATE — consumer's loader calls this after the assistant response lands |
| `set_outcome(agent_id, invocation_id, *, outcome: SkillOutcome, source: OutcomeSource)` | UPDATE setting outcome + source. Idempotent. |

---

## Files to create

```
packages/agent/skills/
├── pyproject.toml
├── README.md
├── src/threetears/agent/skills/
│   ├── __init__.py
│   ├── collections.py
│   ├── entities.py
│   ├── types.py
│   ├── py.typed
│   └── migrations/
│       ├── __init__.py
│       ├── v001_create_agent_skills.py
│       └── v002_create_agent_skill_invocations.py
└── tests/
    ├── unit/
    │   ├── test_collection_methods.py
    │   └── test_entities.py
    ├── integration/
    │   ├── test_migrations_apply.py
    │   ├── test_fts_search.py
    │   ├── test_fk_cascade.py
    │   ├── test_composite_pk_lookup.py
    │   └── test_check_constraints.py   # NEW — verify body-OR-tools constraint + prompt_mode enum
    └── enforcement/
        ├── test_uuidv7_persisted_ids.py
        └── test_partition_column_walker.py
```

---

## Implementation notes

1. **FTS for `skill_list` query-filter only, NOT for auto-load.** The trigger-maintained tsvector is retained because `skill_list(query="prod 500")` should rank by relevance. But there's no per-turn auto-classification, no top-K-retrieve-and-load-skills path on user turns. The FTS infrastructure is discovery-only.

2. **Composite PK + UNIQUE on the bare id.** Composite PK `(agent_id, skill_id)` enforces partition discipline (collections-task-04). Standalone `UNIQUE (skill_id)` lets `agent_wake_schedules.skill_id` reference the bare column without partition knowledge.

3. **`body` is nullable; CHECK enforces at-least-one-payload.** A pure tool-composition skill (no prose) is valid as long as `tool_additions` or `tool_restrictions` has at least one entry. The DB CHECK is belt-and-braces; the agent-tools layer (shard 02) also validates at creation time.

4. **`prompt_mode` is enum-by-CHECK.** Two values: `'additive'` (default) appends the body to the consumer's base system prompt; `'replace'` substitutes the base entirely. Per-user additions (NSFW, jailbreak) still apply on top in either mode — those are consumer-layer concerns, not skill-storage concerns.

5. **`tool_additions` / `tool_restrictions` are by `mcp_name`, not UUID.** Storing tool names (not registry IDs) means a skill survives tool re-registration (idempotent). The consumer's per-turn composition logic (PLACEMENT §1.10) resolves each name against the live registry, filters by ACL + eligibility, builds the turn's tool surface.

6. **`source` enum carries only `'manual'` in v1.** No `'distilled'` since `skill_create_from_range` is dropped (PLACEMENT §1.4). Reserved values not in the enum are not in the CHECK constraint — adding new values later doesn't require a migration.

7. **Skill name uniqueness scope.** Per `(agent_id, user_id)`. Two users on the same agent can both have a "deploy-helper"; same user on two agents (rare) can too.

8. **`use_count` denormalisation caveat.** Bumped optimistically; periodic reconciliation against `COUNT(*) FROM agent_skill_invocations` can detect drift. Acceptable approximation.

9. **`outcome` set synchronously post-LLM, not by a classifier tick.** The consumer's personality-node-integration hook parses `[SUCCESS]`/`[FAILED]` markers from the agent's response and calls `set_outcome(...)`. If no marker is present, `outcome` stays `NULL`. No background reclassification loop.

10. **No `source_conversation_id` column.** Was for `skill_create_from_range`'s "where did this skill come from?" UI. Dropped along with the distillation tool. (Future: a `provenance` column could re-add this if needed.)

---

## Anti-patterns

- DO NOT add a `kind: 'prose' | 'tool'` discriminator column. The four-state matrix of skills (prose-only, mixed, tool-only-composition, etc.) is fully captured by which payload fields are populated — adding a discriminator duplicates information.
- DO NOT use `gen_random_uuid()`. App-side `uuid_utils.uuid7()`.
- DO NOT add `parent_skill_id` or other "skill inheritance" mechanism. One skill is atomic.
- DO NOT cross-link `agent_skills` to `memories`. They're separate primitives.
- DO NOT make `agent_skill_invocations.message_id` an FK to `messages`. Messages can be hard-deleted; invocation history survives.
- DO NOT add `published_to_hub: bool`. No public hub in v1.
- DO NOT skip the unique-name-per-`(agent, user)` constraint.
- DO NOT use GENERATED columns for `search_vector`. Trigger-maintained (parity-check survivability).
- DO NOT reintroduce `outcome_classified_at` or any classifier-tick infrastructure. Outcome population is synchronous.

---

## Success criteria

- [ ] `pyproject.toml` declares the package.
- [ ] Both migrations apply cleanly on a fresh DB; idempotency verified.
- [ ] CHECK constraints fire correctly: skill with no payload rejected; `prompt_mode='invalid'` rejected; `outcome='partial'` rejected.
- [ ] Composite PK lookup via `Collection.get(agent_id, skill_id)` round-trips a saved entity including `prompt_mode`, `tool_additions`, `tool_restrictions`.
- [ ] FK cascade verified: delete a skill, its invocations vanish.
- [ ] FTS search: a skill with body mentioning "ruff format" matches `list_for_user(agent_id, user_id, query="ruff")`.
- [ ] `bump_use_count` updates timestamps atomically.
- [ ] `set_outcome` idempotent.
- [ ] Enforcement walkers pass: UUIDv7 + partition-column.
- [ ] Linting clean (ruff + mypy strict).
- [ ] Integration tests pass against testcontainers Postgres.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
./scripts/test.sh agent-skills
./scripts/lint.sh agent-skills
./scripts/typecheck.sh agent-skills
./scripts/check-all.sh
```
