# 3tears-agent-skills

Procedural memory for 3tears-based agents. Stores per-agent / per-user
labeled markdown procedures (`body`) and tool-surface modifications
(`tool_additions`, `tool_restrictions`) plus a `prompt_mode` enum
controlling how the body interacts with the consumer's base system prompt.

This shard (01) provides only the schema + Collection layer:

- `agent_skills` -- one row per skill (partition column `agent_id`).
- `agent_skill_invocations` -- one row per skill load with synchronous
  outcome tracking populated by the consumer's post-LLM hook (partition
  column `agent_id`; composite FK CASCADE on parent skill).
- Two `BaseEntity` subclasses (`AgentSkillEntity`, `AgentSkillInvocationEntity`).
- Two `BaseCollection` subclasses with the methods listed under "Public
  API" in `docs/agent-skills/shard-01-schema-and-collection.md`.
- A trigger-maintained `search_vector` (FTS for `skill_list`
  query-filter ranking; NOT auto-load).

No agent tools (shard 02) and no per-turn composition renderer
(shard 03) live here. See `docs/agent-skills/README.md` in the
3tears repo for the package overview and the canonical PLACEMENT memo
in `metallm/docs/skills/PLACEMENT.md` for the upstream design
decisions.

## Migration registration

```python
from threetears.agent.skills import register as register_skills
from threetears.core.data.migrations import MigrationRunner

runner = MigrationRunner()
register_skills(runner)
```

Migrations are agent-scoped and depend on `conversations` (the
invocation rows carry `conversation_id` -- ordering on apply, not an
FK constraint).
