# 3tears-conversations

Canonical owner of the per-agent `conversations` table plus its entity
and three-tier collection.

The table tracks one row per user-facing conversation an agent is
engaged in:

- ownership scope (`agent_id`, `customer_id`, `user_id`)
- external reference (`channel_type`, `conversation_ref`)
- lifecycle (`status`, `summary`)
- timestamps (`date_created`, `date_updated`, `date_last_message`)

Previously bundled inside `3tears-agent-memory`; pulled out because
multiple packages key off `conversation_id` (memory, agent-tools
context items, workspace bindings) and no single consumer is the
natural owner of the shape.

## Migration package

`threetears.conversations.migrations.register(runner)` contributes the
agent-scope `conversations` package to a canonical
`threetears.core.data.migrations.MigrationRunner`. No `depends_on`
edges — it is the root of the agent-schema dependency graph.
