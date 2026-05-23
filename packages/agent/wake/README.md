# 3tears-agent-wake

Long-running-agent foundation for 3tears-based agents. This shard (01)
ships the schema + Collection layer for three platform tables:

- `agent_wake_schedules` -- one row per active wake schedule for a
  conversation (cron / one-shot / random-window / etc). Carries a
  nullable `skill_id` FK referencing the cross-package
  `agent_skills.skill_id` standalone UNIQUE.
- `wake_fires` -- one row per wake fire (history; status enum spans
  `'fired'`, `'fired_silent'`, `'yielded'`, `'skipped_busy'`,
  `'skipped_rate_limit'`, `'skipped_cap'`, `'skipped_no_handler'`,
  `'failed'`).
- `webhook_subscriptions` -- one row per inbound HTTP webhook
  subscription. Carries nullable `default_skill_id` FK to
  `agent_skills.skill_id` and Fernet-encrypted `secret_ciphertext`.

Plus three `BaseEntity` subclasses, three `BaseCollection` subclasses,
and an agent-scope migration registration declaring
`depends_on=("conversations", "agent_skills")`.

No tick engine, no dispatch handler, no agent tools, no Pydantic API
models in this shard -- those land in shards 02-06.

## Partitioning

All three tables partition by `conversation_id` (wake operations are
conversation-scoped). The Collections expose `partition_column =
"conversation_id"` so the workspace partition-column enforcement walker
audits every SQL string touching these tables for the predicate.

There is intentionally NO database-level FK on `conversation_id` ->
`conversations(conversation_id)`. The 3tears `conversations` table
carries a composite PK `(agent_id, conversation_id)` and no standalone
`UNIQUE (conversation_id)` constraint, so a single-column FK is not
legal. The same precedent applies in `packages/agent/tools/`
(`context_items.conversation_id`) and `packages/agent/skills/`
(`agent_skill_invocations.conversation_id`). Conversation lifecycle is
governed by app-level cascade through `ConversationsCollection`.

## Migration registration

```python
from threetears.agent.wake import register as register_wake
from threetears.core.data.migrations import MigrationRunner

runner = MigrationRunner()
register_wake(runner)
```

Migrations are agent-scoped and declare
`depends_on=("conversations", "agent_skills")` -- the canonical
`MigrationRunner` topologically orders the agent-scope pass so the
`conversations` + `agent_skills` migrations apply before any wake
table is created.

## Design references

See `docs/agent-wake/README.md` for the package overview, the
2026-05-19 revision deltas, and the locked design decisions.
Authoritative source-of-truth precedence (per PLACEMENT §9):

1. `metallm/docs/long_running/PLACEMENT.md`
2. `docs/agent-wake/README.md`
3. The shard's revision-deltas section (top of file).
4. The shard body.
