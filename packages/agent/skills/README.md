# 3tears-agent-skills

Procedural memory for 3tears-based agents. Stores per-agent / per-user
labeled markdown procedures (`body`) and tool-surface modifications
(`tool_additions`, `tool_restrictions`) plus a `prompt_mode` enum
controlling how the body interacts with the consumer's base system prompt.

This package provides the schema + Collection layer:

- `agent_skills` -- one row per skill (partition column `agent_id`).
- `agent_skill_invocations` -- one row per skill load with synchronous
  outcome tracking populated by the consumer's post-LLM hook (partition
  column `agent_id`; composite FK CASCADE on parent skill).
- Two `BaseEntity` subclasses (`AgentSkillEntity`, `AgentSkillInvocationEntity`).
- Two `BaseCollection` subclasses exposing the public skill-registry API.
- A trigger-maintained `search_vector` (FTS for `skill_list`
  query-filter ranking; NOT auto-load).

Agent tools and the per-turn composition renderer do not live here.

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

## `SkillRegistryClient` Protocol -- why this package takes no ACL / tools deps

`agent-memory` (a sibling package) takes direct dependencies on
`3tears-agent-acl` and `3tears-agent-tools` because its memory tools
need first-class ACL evaluator + tool-registry types in their public
surface. `agent-skills` deliberately diverges. The tool factories take
a thin `SkillRegistryClient` Protocol (three async methods,
`acl_permits` / `list_skill_eligible_tools` / `get_tool_introspect`,
plus a `ConversationIdResolver` / `ActiveSkillProbe` /
`ActiveSkillSetter` callable triple wired by the consumer) instead of
importing the ACL evaluator or the tool registry types directly.

The trade-off: callers must implement a small adapter (~10 lines over
their existing `NamespaceCollection` + ACL cache + in-process tool
registry), but `agent-skills` ships with zero hard deps beyond core
and `langchain-core`. The consumer wires this; tests mock it. The
ACL-via-evaluator approach targets the composition renderer, not the
tools surface.

Future sibling packages should follow whichever pattern their public
surface demands -- direct deps when types are part of the contract,
Protocol when the contract is method-shaped and the deps are
incidental.
