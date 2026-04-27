# 3tears-conversations

Canonical owner of the per-agent `conversations` table plus its entity and three-tier collection.

## Purpose

Tracks one row per user-facing conversation an agent is engaged in: ownership scope (`agent_id`, `customer_id`, `user_id`), external reference (`channel_type`, `conversation_ref`), lifecycle (`status`, `summary`), timestamps (`date_created`, `date_updated`, `date_last_message`).

Previously bundled inside `3tears-agent-memory`; split out during `workspace-task-19` / `migrations-task-01` because multiple packages key off `conversation_id` (memory, agent-tools context items, workspace bindings) and no single consumer is the natural owner of the shape. Owning the table here lets each downstream package depend on `conversations` without pulling in memory's extraction pipeline.

## Public API

Exports (see `src/threetears/conversations/__init__.py`):

- `Conversation` — `BaseEntity` subclass for a single row.
- `ConversationsCollection` — three-tier (`L1` SQLite / `L2` NATS KV / `L3` postgres) collection with the usual `get` / `save` / `subscript` semantics.
- `register(runner)` — agent-scope migration entry point. Contributes the `conversations` table creation to a canonical `threetears.core.data.migrations.MigrationRunner`. No `depends_on` edges; this is the root of the agent-schema dependency graph, so memory / workspace / tools migrations that reference `conversations.id` sort after it.

## Minimal usage

```python
from threetears.conversations import Conversation, ConversationsCollection, register
from threetears.core.data.migrations import MigrationRunner

# 1. register the package's migrations on the platform's runner
runner = MigrationRunner(store)
register(runner)
await store.run_migrations(runner)

# 2. use the collection like any other 3tears collection
collection = ConversationsCollection(registry=registry)
conv = await collection.get(conversation_id)
conv.summary = "follow-up on last week's analysis"
await conv.save()
```

## Reference

- Shards: `14-eng-ai-bot/docs/workspace-task-19-access-control.md` (structural split), `3tears/docs/migrations-task-01-canonical-runner.md` (runner integration)
- Downstream consumers: `threetears.agent.memory`, `threetears.agent.tools` (context items), `threetears.agent.workspace` (bindings)
