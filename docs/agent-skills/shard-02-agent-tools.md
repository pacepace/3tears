# agent-skills / shard-02: Agent tools — skill CRUD + invoke + introspect

> **Renumbered:** was shard-03 in the prior redesign; shard-02 (retrieval + classifier) is DELETED per `metallm/docs/skills/PLACEMENT.md` §1.2.

## Objective

Land six LangChain tools as factory functions in `threetears.agent.skills.tools`: `skill_create`, `skill_list`, `skill_get`, `skill_update`, `skill_delete`, `skill_invoke`. Plus the `skill_introspect` tool returning the minimal-token shape from PLACEMENT §1.8. Factory-with-Collection-injection shape mirroring `3tears-agent-memory.tools`. Bounded payload sizes, name uniqueness, cross-user isolation, ACL-respecting validation of `tool_additions` / `tool_restrictions`. **No distillation, no `ChatModelFactory` dependency, no history tool in v1.**

## Locked design decisions (canonical source: `metallm/docs/skills/PLACEMENT.md`)

| Topic | Locked answer | PLACEMENT ref |
|---|---|---|
| Tool count | Six (CRUD + invoke + introspect). No `skill_create_from_range`. No `skill_history` (deferred to v1.1 via separate `skill_stats`). | §1.4 / §1.8 |
| `skill_create` accepts | `name`, `summary`, `body` (optional), `prompt_mode`, `tool_additions` (optional), `tool_restrictions` (optional), `trigger_keywords` (optional, for discovery via `skill_list` query), `tags`, `enabled` | §1.1 |
| `skill_invoke` semantics | First invoke per turn wins; subsequent invokes in same turn return error. Wake-driven turns cannot be `skill_invoke`d into (wake's attached skill is already active). | §1.2 |
| `skill_introspect` | Returns minimal-token shape: name + summary + kind + (body/prompt_mode/tool_additions/tool_restrictions for prose-skill) OR (args + example for tool-skill). NO operational metadata (use_count, history). | §1.8 |
| `skill_list` | Returns UNION of prose-skill rows from `agent_skills` AND skill-eligible tools from registry (where ACL permits). Discriminator field `kind: 'prose' \| 'tool'`. | §1.7 |
| Cap | 200 prose skills per `(agent_id, user_id)`. Tool-skills don't count (they live in the registry). | |

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SK-09 | **Seven tool factories** (six CRUD+invoke + one introspect): `load_skill_create_tool`, `load_skill_list_tool`, `load_skill_get_tool`, `load_skill_update_tool`, `load_skill_delete_tool`, `load_skill_invoke_tool`, `load_skill_introspect_tool`. Each takes `(agent_id, user_id, skills_collection, invocations_collection, registry_client, ...)` and returns a `BaseTool`. | P0 |
| SK-10 | `skill_create` enforces: name 1-128 chars (letters/digits/hyphens/spaces/underscores), summary 1-256 chars, body 0-32KB (nullable), trigger_keywords 0-512 chars, tags max 8 entries, tool_additions max 32 entries, tool_restrictions max 32 entries. At least one of body / tool_additions / tool_restrictions must be non-empty (matches DB CHECK from shard-01). | P0 |
| SK-11 | `skill_create` validates `tool_additions` / `tool_restrictions` entries against the registry: each entry must be a tool name (mcp_name) for which the calling `(agent_id, user_id)` has ACL grant. **Skills do NOT bypass ACL.** Reject with `[TOOL ERROR] skill_create: tool 'X' not authorized for this user` if any entry fails the check. | P0 |
| SK-12 | `skill_invoke(skill_id)` returns the skill's content for the agent to use in the current turn. **Semantics:** the calling LangGraph state carries a `_active_skill_id` field (set by the consumer's tool-loop wrapper). If `_active_skill_id` is already set when `skill_invoke` is called, the tool returns `[TOOL ERROR] skill_invoke: a skill is already active this turn ([skill:<active>]); only one invoke per turn`. Otherwise, the tool sets the field, inserts an `agent_skill_invocations` row with `invocation_source='invoke'`, and returns a formatted block (see "skill_invoke output format" below). | P0 |
| SK-13 | All tools scoped to the calling `(agent_id, user_id)` (closure-captured). Cross-user / cross-agent access rejected. | P0 |
| SK-14 | Skill-creation cap: 200 active prose skills per `(agent_id, user_id)` (configurable via factory parameter `max_prose_skills_per_user`, default 200). Rejected: "max N prose skills per user; delete or disable some first." | P0 |
| SK-15 | Tool descriptions terse (≤6 lines), action-first, imperative voice. Platform-default; consumer can override per consumer-specific voice. | P0 |
| SK-16 | `skill_list` UNIONs prose-skill rows + skill-eligible registry tools. Returns uniform entries with `kind: 'prose' \| 'tool'` discriminator. Optional `query` parameter applies FTS ranking to prose rows AND substring match against tool names/summaries. | P0 |
| SK-17 | `skill_introspect(name_or_id)` resolves either kind and returns the minimal-token shape from PLACEMENT §1.8. | P0 |

---

## Design Context

### Why drop `skill_create_from_range`

Per PLACEMENT §1.4: "Saoirse uses her existing context (memory, system prompt, the conversation she's currently in) to compose a skill directly via the regular `skill_create` tool. The tool stays dumb; her intelligence does the work." A dedicated extraction tool would require a meta-LLM call, distillation prompt template, range validation, the `ChatModelFactory` dependency — all dropped.

If Saoirse wants to author a skill from a conversation arc, she reads the relevant messages (they're already in her context) and calls `skill_create` with the body she composes. Iterative refinement via `skill_update` after testing.

### Why drop `skill_history`

Per PLACEMENT §1.8: introspection returns "what Saoirse needs to USE the skill in a job" — name, summary, body or args+example. Operational metadata (use_count, last_used_at, invocation log) is a separate `skill_stats(skill_id)` call deferred to v1.1. Most calls go through the cheap introspect path; the expensive stats path is opt-in.

### Why mirror `3tears-agent-memory.tools`

Same factory-with-Collection-injection shape. Consumer (metallm) registers loaders next to memory's; one consistent pattern.

### Why `skill_invoke` returns content as tool result (vs. mid-turn prompt rebuild)

Mid-turn system-prompt rebuild is infeasible in the LangGraph tool loop — the system prompt is fixed at turn start. Returning the skill's content as the tool result lets the agent consume it in the same turn (the LLM sees the tool result on its next iteration). The wake-driven path DOES rebuild the system prompt at turn start (per PLACEMENT §1.10) because the active skill is known before LLM invocation.

So: wake-driven turn = active skill applied to system prompt at turn start. User-driven turn with `skill_invoke` = active skill content delivered as tool result mid-turn. Both honor the active-skill-this-turn semantics; the application mechanism differs by load point.

---

## Tool surfaces

### `skill_create`

```python
class SkillCreateInput(BaseModel):
    name: str = Field(description="Short unique name (1-128 chars).")
    summary: str = Field(description="One-line catalog entry shown in skill_list.")
    body: str | None = Field(default=None, description="Markdown procedure (optional). Max 32KB.")
    prompt_mode: Literal['additive', 'replace'] = Field(default='additive', description="'additive' appends body to system prompt; 'replace' substitutes it.")
    tool_additions: list[str] = Field(default_factory=list, description="Tool mcp_names to surface when this skill loads.")
    tool_restrictions: list[str] = Field(default_factory=list, description="Tool mcp_names to remove from default surface.")
    trigger_keywords: str = Field(default="", description="Keywords for skill_list filter. Not for auto-load.")
    tags: list[str] = Field(default_factory=list, description="Classification tags (max 8).")
    enabled: bool = Field(default=True)
```

Description (platform default):
```
Save a procedure as a skill — named, reusable unit that modifies your turn.
- prose body OR tool_additions OR tool_restrictions (at least one)
- prompt_mode 'additive' (default) appends body; 'replace' substitutes
Returns [skill:<id>]. Cap of 200 prose skills. Loads on wake fires or via skill_invoke.
```

### `skill_list`

```python
class SkillListInput(BaseModel):
    query: str | None = Field(default=None, description="Optional substring/keyword filter.")
    kind_filter: Literal['all', 'prose', 'tool'] = 'all'
    tag_filter: str | None = None
    enabled_only: bool = True
    limit: int = Field(default=20, ge=1, le=200)
```

Description:
```
List skills available to you — prose skills you authored AND tools registered as skill-eligible.
Returns [skill:<id>] + name + summary + kind ('prose' | 'tool'). Use skill_introspect for details.
```

### `skill_get`

```python
class SkillGetInput(BaseModel):
    skill_id: str = Field(description="[skill:<id>] from skill_list / skill_create.")
```

Description:
```
Read a prose-skill's body, metadata, instrumentation. Use before skill_update.
```

### `skill_update`

```python
class SkillUpdateInput(BaseModel):
    skill_id: str
    name: str | None = None
    summary: str | None = None
    body: str | None = None
    prompt_mode: Literal['additive', 'replace'] | None = None
    tool_additions: list[str] | None = None
    tool_restrictions: list[str] | None = None
    trigger_keywords: str | None = None
    tags: list[str] | None = None
    enabled: bool | None = None
```

Description:
```
Edit a skill in place. Pass only fields to change. Returns the updated summary.
```

### `skill_delete`

```python
class SkillDeleteInput(BaseModel):
    skill_id: str
```

Description:
```
Delete a prose skill permanently. Invocation history cascades. Use enabled=false to disable instead.
```

### `skill_invoke`

```python
class SkillInvokeInput(BaseModel):
    skill_id: str = Field(description="[skill:<id>] to activate for the current turn.")
    rationale: str | None = Field(default=None, description="Optional one-line note recorded with the invocation.")
```

Description:
```
Activate a skill for the rest of THIS turn. First invoke per turn wins; subsequent invokes error.
Returns the skill's body + tool composition. Records the invocation.
```

### `skill_introspect`

```python
class SkillIntrospectInput(BaseModel):
    name_or_id: str = Field(description="Skill name OR [skill:<id>]. Works for prose-skills AND tool-skills.")
```

Description:
```
Examine a skill before using it — see its body, tool surface, args, examples.
Use to discover how to use a skill in a wake or skill_invoke.
```

---

## `skill_invoke` output format

When `skill_invoke` succeeds, it returns:

```
[ACTIVE SKILL: <name>]
prompt_mode: <additive|replace>
tool_additions: [<tool1>, <tool2>, ...]
tool_restrictions: [<tool1>, <tool2>, ...]

<body — if any, otherwise empty>
```

The consumer's tool-loop wrapper (metallm `personality_node`) reads this tool result and:
1. Notes that `_active_skill_id` is set for the rest of the turn.
2. Applies `tool_additions` / `tool_restrictions` to the next LLM iteration's available tools (per PLACEMENT §1.10).
3. If `prompt_mode='replace'`, log a structured event "skill_invoke replace-mode requires reset; substituting from next iteration." (Replace-mode mid-turn is awkward — see Implementation note 4.)

## `skill_introspect` output format

For a prose-skill:
```
[skill:<id>]
name: <name>
kind: prose
summary: <summary>
prompt_mode: <additive|replace>
body: |
  <body markdown — full content>
tool_additions: [<tool1>, ...]
tool_restrictions: [<tool1>, ...]
triggers: <trigger_keywords>
tags: [<tag1>, ...]
enabled: <bool>
```

For a tool-skill (registry-sourced):
```
[skill:<mcp_name>]
name: <mcp_name>
kind: tool
summary: <tool.description>
args:
  <arg_name>: <type>  # <description>
  ...
example:
  <arg_name>: <example_value>
  ...
```

The minimal-token rule: NO use_count, NO last_used_at, NO history, NO invocation log. Those go in a future `skill_stats(skill_id)` tool (deferred per PLACEMENT §1.8).

---

## Patterns to follow

- Factory shape: `packages/agent/memory/src/threetears/agent/memory/tools.py:load_memory_search_tool` is the precedent.
- ID resolution helper: `_resolve_skill(skills_collection, registry_client, agent_id, user_id, name_or_id) -> SkillCatalogEntry`. Accepts `[skill:<id>]` (UUID prefix or full) for prose-skills OR plain tool name for tool-skills. Returns a tagged dataclass discriminating prose vs. tool.
- Error format: `[TOOL ERROR] <tool>: <description>` (matches existing 3tears tool conventions).
- Cross-user isolation: every fetch goes through `Collection.find_by_id_for_user(agent_id, user_id, skill_id)` for prose; through `registry_client.list_skill_eligible_tools(actor_user_id, actor_agent_id)` for tool-skills.

---

## Files to create

```
packages/agent/skills/src/threetears/agent/skills/
├── tools.py                                          # seven loader factories + validators + skill catalog union
├── metric_names.py                                   # canonical Prometheus + Loki event-name constants
└── (existing) collections.py, entities.py, types.py, etc.

packages/agent/skills/tests/
├── unit/
│   ├── test_tools_validators.py                      # name/summary/body/keywords cap enforcement
│   ├── test_tools_skill_invoke.py                    # first-invoke-wins + invocation row insertion
│   ├── test_tools_skill_introspect.py                # both prose + tool variants return correct shape
│   ├── test_skill_create_acl_check.py                # tool_additions ACL validation
│   ├── test_skill_list_union.py                      # prose + tool UNION shape
│   └── test_metric_names.py
└── integration/
    ├── test_tools_e2e.py                             # create → invoke → list → introspect → update → delete
    ├── test_cross_user_isolation.py
    ├── test_cap_enforcement.py                       # 200-prose-skill cap
    └── test_tool_skill_via_registry.py               # registry-side tool with skill_eligible=True appears in skill_list
```

### `metric_names.py` shape

```python
"""Canonical Prometheus metric + Loki event names for skills observability."""

# Prometheus instrument names (consumer adds its own product prefix)
SKILL_LOAD_TOTAL = "agent_skill_load_total"            # labels: source ('wake'|'invoke'), outcome ('success'|'failure'|'unknown')
SKILL_CREATE_TOTAL = "agent_skill_create_total"

# Loki structured-log event_type values
EVENT_SKILL_LOADED = "skill.loaded"
EVENT_SKILL_CREATED = "skill.created"
EVENT_SKILL_INVOKED = "skill.invoked"                  # explicit skill_invoke calls
EVENT_SKILL_OUTCOME_RECORDED = "skill.outcome_recorded"
```

Consumer (metallm shard 04) imports these and registers `metallm_skill_load_total` etc.

---

## Implementation notes

1. **Name validation regex.** `^[A-Za-z0-9 _-]{1,128}$`.

2. **Body cap is hard at 32KB.** Reject-on-overflow.

3. **`tool_additions` / `tool_restrictions` ACL validation at creation time.** `skill_create` and `skill_update` call `registry_client.acl_permits(user_id, agent_id, tool_name)` for each entry. Fail-fast with explicit error if any entry rejected. Re-validation at LOAD time (PLACEMENT §1.10) is the source of truth; create-time check is a UX courtesy.

4. **`skill_invoke` + `prompt_mode='replace'` mid-turn awkwardness.** Replacing the system prompt mid-turn requires resetting the LLM context. The simplest implementation: replace-mode skills CANNOT be `skill_invoke`d mid-turn — return `[TOOL ERROR] skill_invoke: skill has prompt_mode='replace'; replace-mode skills can only be activated by attaching to a wake schedule`. Additive-mode skills work fine via `skill_invoke` (the body is delivered as tool result; consumer's wrapper applies tool composition to subsequent iterations).

5. **`skill_invoke` and `_active_skill_id` state.** The LangGraph state needs a new field — the consumer (metallm) extends its state to carry `_active_skill_id: UUID | None`. The tool factory takes a state-mutator callable from the consumer at construction time (e.g. `state_setter: Callable[[UUID], None]`). Document the contract in the factory's signature.

6. **`skill_list` UNION shape.** Each entry:
   ```
   {
     skill_id: '<UUID or mcp_name>',  # UUID for prose, mcp_name for tool
     name: '<name>',
     summary: '<summary>',
     kind: 'prose' | 'tool',
     enabled: bool  # always true for tool-skills (registry state)
   }
   ```
   Prose rows from `AgentSkillCollection.list_for_user`; tool rows from `registry_client.list_skill_eligible_tools`. Combined by name-sort; pagination via `limit`/`offset`.

7. **`skill_introspect` resolution priority.** If `name_or_id` matches a prose-skill name first, return that. Otherwise check the registry for a tool with matching `mcp_name`. If both exist (rare but possible), return the prose-skill (user-authored content wins on name collision; document this).

8. **`enabled=false` is reversible.** A disabled skill doesn't load via wake-attach OR `skill_invoke`. `skill_update(enabled=true)` reactivates.

9. **Recursive create disable.** A wake-triggered turn should NOT have `skill_create` / `skill_update` / `skill_delete` disabled. Skills are reflective; the agent should be able to refine them during wakes. This differs from the recursive-cron-create disable in long_running shard 04. Document explicitly.

10. **`registry_client` parameter.** The factory takes a `registry_client` (a `3tears-registry` client) so `skill_create` can validate ACL on `tool_additions` and `skill_list` can query skill-eligible tools. The metallm consumer wires this; tests mock it.

---

## Anti-patterns

- DO NOT silently re-enable a disabled skill on retrieval. `enabled=false` is honored everywhere except `skill_introspect` (which shows the disabled flag clearly).
- DO NOT log skill bodies at info-level. Bodies may contain user-private procedures. Debug-level only.
- DO NOT add a "duplicate skill" tool.
- DO NOT expose `agent_id` or `user_id` as input fields. Both are closure-captured.
- DO NOT call the LLM in any tool. No distillation, no extraction, no smart-anything.
- DO NOT allow `skill_invoke` to bypass the first-invoke-wins-per-turn rule. The consumer's tool-loop wrapper enforces it via `_active_skill_id` state; the tool returns an error if state says one's active.

---

## Success criteria

- [ ] All seven tool factories load + work end-to-end.
- [ ] Cross-user isolation enforced.
- [ ] 200-prose-skill cap enforced.
- [ ] `skill_create` rejects `tool_additions` entries the user lacks ACL for.
- [ ] `skill_invoke` first-invoke-wins; second invoke in same turn errors.
- [ ] `skill_invoke` with `prompt_mode='replace'` skill returns error directing user to attach via wake.
- [ ] `skill_introspect` returns minimal-token shape (no use_count, no history).
- [ ] `skill_list` UNIONs prose + tool entries correctly.
- [ ] Linting + mypy strict clean.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
./scripts/test.sh agent-skills
./scripts/lint.sh agent-skills
./scripts/typecheck.sh agent-skills
./scripts/check-all.sh
```

---

## Enforcement test suggestions

- Drift guard: `skill_create` body cap of 32KB is in one place.
- Drift guard: every tool factory captures `agent_id`+`user_id` from closure, not from input schema (AST check).
- Drift guard: NO `ChatModelFactory` or other LLM-call import in `tools.py` (AST check).
- Drift guard: `_active_skill_id` state-field contract documented in factory signature (mypy enforces the callable type).
