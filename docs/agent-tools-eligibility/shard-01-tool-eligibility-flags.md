# agent-tools-eligibility / shard-01: tool eligibility flags

**Audience:** subagent executing this shard against the `3tears-agent-tools` package.
**Status:** new shard, 2026-05-19. **Prerequisite for both `agent-skills` AND `agent-wake` planning sets.**
**Driving design decision:** see `metallm/docs/skills/PLACEMENT.md` §1.5 + §1.6.

## Objective

Extend the `3tears-agent-tools` `TearsTool` base class and the tool-registration path with two new boolean flags that decouple "is this tool in the agent's default tool surface?" from "is this tool discoverable in the skills catalog?". Adds the `tool_eligible` (default `True`) and `skill_eligible` (default `False`) class attributes; propagates them through registration; gives the tool catalog a query method for "list skill-eligible tools the actor can access"; warns at registration time on suspicious flag combinations.

This unlocks two downstream capabilities that the consuming planning sets rely on:

1. **Tool-shaped skills** — tools registered with `skill_eligible=True` appear in the unified skills catalog produced by `3tears-agent-skills.list_skills(...)`.
2. **Skill-only tools (the "code-skill without sandbox" pattern)** — tools registered with `tool_eligible=False, skill_eligible=True` are NOT in the agent's default tool surface but ARE available when a skill that lists them in `tool_additions` loads. The skill is the visibility gate.

ACL still governs authorization in both cases. Eligibility flags govern *default visibility*, not authorization. Skills can compose visibility within ACL-permitted tools; they cannot bypass ACL.

## Why this is its own shard (and not embedded in skills or wake)

- It modifies `3tears-agent-tools`, not `3tears-agent-skills` or `3tears-agent-wake`. Cross-package placement matters: each package owns its own surface.
- Both skills AND wake depend on the flag mechanism. Co-locating the change inside either would force the other to wait — better to ship the foundation first.
- The change is small and self-contained: one base-class extension, one registration-emitter update, one registry query method, one warning, one set of tests.

Ships in the same 3tears release as `agent-skills` and `agent-wake` — bundled per `metallm/docs/skills/PLACEMENT.md` §3.7. The flags are useless without consumers; releasing them separately would force a two-step migration for metallm.

## Requirements

| ID | Requirement | Priority |
|---|---|---|
| TE-01 | Add `tool_eligible: bool = True` class attribute to `threetears.agent.tools.TearsTool` | P0 |
| TE-02 | Add `skill_eligible: bool = False` class attribute to `threetears.agent.tools.TearsTool` | P0 |
| TE-03 | The `ToolNamespaceEmitter` (which writes the `platform.namespaces` row at tool registration) stamps both flags onto the namespace row | P0 |
| TE-04 | The `platform.namespaces` table gains two columns: `tool_eligible BOOLEAN NOT NULL DEFAULT TRUE`, `skill_eligible BOOLEAN NOT NULL DEFAULT FALSE`. Idempotent ALTER (`ADD COLUMN IF NOT EXISTS`). | P0 |
| TE-05 | Registry exposes a new query method `list_skill_eligible_tools(actor_user_id: UUID, actor_agent_id: UUID) -> list[ToolNamespace]` that returns tools where `skill_eligible=True` AND the existing ACL evaluator permits the actor `tool.call` on the tool's namespace | P0 |
| TE-06 | The existing registry "list tools for actor" path (which builds the agent's default tool surface) additionally filters on `tool_eligible=True`. Tools with `tool_eligible=False` are excluded from the default surface even if the actor has ACL grant. | P0 |
| TE-07 | At tool registration time (`ToolNamespaceEmitter` emit path), if a tool registers with BOTH `tool_eligible=False` AND `skill_eligible=False`, emit a structured-log WARNING: `tool '{mcp_name}' registered with both tool_eligible=False and skill_eligible=False — this tool will never be visible to any agent. Did you forget to enable a surface?` | P1 |
| TE-08 | Backwards compatibility: existing tools without an explicit `tool_eligible` class attribute default to `True` (preserved behavior). Existing tools without `skill_eligible` default to `False` (no tool appears in skills catalog by default). | P0 |
| TE-09 | Per-customer overrides via ACL grant rows are NOT in this shard. The flags are the tool's authored declaration; admin overrides happen via the ACL system at grant time, which is a separate concern. (Deferred to a possible follow-up if a real use case emerges.) | P0 |
| TE-10 | Tests: unit-level — `TearsTool` subclass with `tool_eligible=False`; namespace emission stamps flag; registry `list_for_actor` excludes it; registry `list_skill_eligible_tools` includes it if `skill_eligible=True`. Integration-level — register a tool, verify it appears/doesn't appear via the right queries. | P0 |
| TE-11 | Migration registered with the canonical `MigrationRunner` (NOT alembic — `3tears-agent-tools` uses the canonical runner). Idempotent. | P0 |
| TE-12 | **Default ACL grants for the three pre-check tools** (`http_get`, `loki_query`, `postgres_query`). The platform ships these tools registered with `tool_eligible=False, skill_eligible=True`. WITHOUT default ACL grants, no user can include them in `tool_additions` (skill_create rejects). Solution: ship the three tools with default-permitted ACL grants for the `platform-users` group (the default group every user belongs to). Specifically: seed a `PlatformBuiltinToolUser` role granting `tool.call` on `http_get`, `loki_query`, `postgres_query`. Admin can override per-customer (revoke for some customer's users). Documented as part of the canonical bootstrap-roles set. metallm consumes by inheriting the default group + role. | P0 |

## Design context

`3tears-agent-tools` already owns the `TearsTool` base class and the `ToolNamespaceEmitter` that listens on `tools.register` NATS subject and upserts a `tool`-type row in `platform.namespaces`. The RBAC evaluator at the registry / call dispatch layer reads through that row.

Adding two columns + two class attributes + two query branches is mechanical. The architectural decision (eligibility separate from authorization) is the substance; the implementation is small.

## Files to create / modify

### Create

- `3tears/packages/agent/tools/threetears/agent/tools/migrations/v0NN_tool_eligibility_columns.py` — `ALTER TABLE platform.namespaces ADD COLUMN IF NOT EXISTS tool_eligible BOOLEAN NOT NULL DEFAULT TRUE; ADD COLUMN IF NOT EXISTS skill_eligible BOOLEAN NOT NULL DEFAULT FALSE;`. Registered via `register(runner)`.

### Modify

- `3tears/packages/agent/tools/threetears/agent/tools/base.py` (or wherever `TearsTool` is defined) — add two class attributes with defaults.
- `3tears/packages/agent/tools/threetears/agent/tools/registration_manifest.py` (or wherever the registration envelope lives) — extend the `RegistrationManifest` envelope to carry `tool_eligible` + `skill_eligible`.
- `3tears/packages/agent/tools/threetears/agent/tools/tool_namespace_emitter.py` (or wherever the emitter is) — read from the envelope, write to the namespace row, emit the warning when both are False.
- `3tears/packages/registry/threetears/registry/...` — extend the registry query method `list_skill_eligible_tools`. (Cross-package — touches `3tears-registry`.)
- `3tears/packages/registry/threetears/registry/...` — extend `list_for_actor` to filter `tool_eligible=True` in the default-surface path. The pre-existing call sites that need filtering are the ones building the agent's tool list at LangGraph graph-build time; verify in the SDK.

### Tests

- `3tears/packages/agent/tools/tests/test_tool_eligibility.py` — unit tests for the class attribute defaults, the registration envelope carrying the flags, the namespace row containing them after registration.
- `3tears/packages/agent/tools/tests/test_registration_warning.py` — registration with both flags False emits the WARNING.
- `3tears/packages/registry/tests/test_skill_eligible_listing.py` — registry method returns the right set.
- `3tears/packages/registry/tests/test_default_surface_filtering.py` — default-surface query excludes `tool_eligible=False` tools.

## Implementation notes

1. **The flags are CLASS attributes, not instance.** A `TearsTool` subclass declares them; they don't change at runtime. Tool authors set them in the class body:
   ```python
   class LokiQueryTool(TearsTool):
       mcp_name = "loki_query"
       tool_eligible = False  # not in default tool surface
       skill_eligible = True   # available via skill_additions
       ...
   ```

2. **Backwards compatibility.** Existing tools have no explicit declaration. The defaults (`True`, `False`) preserve the pre-shard behavior — every existing tool appears in the default surface, no tools appear in the skills catalog. Adopting the new flags is opt-in.

3. **Migration safety.** The `ALTER TABLE` is idempotent (`ADD COLUMN IF NOT EXISTS`). Multiple migration runners running concurrently is fine; the canonical `MigrationRunner` already handles this via advisory lock + topological order. Test by running the migration twice in a row.

4. **Query method placement.** `list_skill_eligible_tools` could live on the `3tears-agent-tools.NamespaceCollection` OR on `3tears-registry.RegistrationHandler`. Pick the side that's the natural query interface for the consuming packages — likely the namespace-collection side since `3tears-agent-skills` will query it.

5. **The ACL evaluator integration is unchanged.** Eligibility filtering happens AFTER ACL evaluation, on the result set. No change to the evaluator itself.

## Anti-patterns

- DO NOT make the flags mutable at runtime. Toggling them per-conversation is a different feature (per-customer overrides via ACL grants) and explicitly deferred (TE-09).
- DO NOT special-case the flags in the call-dispatch path. Authorization is decided by ACL, period. Eligibility is purely visibility.
- DO NOT add a third flag like `default_in_skills_for_agents`. Two flags are sufficient for the four-state matrix in `PLACEMENT.md` §1.5; more would be premature.
- DO NOT auto-set `skill_eligible=True` based on heuristics ("this tool's `mcp_name` matches `pre_check_*`"). Explicit declaration only.

## Success criteria

- [ ] `TearsTool` base class has both new class attributes with documented defaults.
- [ ] Migration creates the two columns idempotently.
- [ ] `RegistrationManifest` envelope carries the flags through NATS.
- [ ] `ToolNamespaceEmitter` writes them to the namespace row.
- [ ] Registry's default-surface query filters `tool_eligible=True`.
- [ ] Registry exposes `list_skill_eligible_tools(actor_user_id, actor_agent_id)`.
- [ ] Warning emitted at registration when both flags are False.
- [ ] All tests green; mypy/ruff clean.
- [ ] No existing tests regress (backwards compat).

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
./scripts/check-all.sh
# Specifically: 3tears-agent-tools + 3tears-registry test suites green
# Specifically: enforcement tests confirm migration registered
```

## Consumers (what depends on this shard releasing first)

- `3tears-agent-skills` shard 01 (schema-and-collection) — defines the `agent_skills` table; doesn't itself require the flags but depends on the catalog query method.
- `3tears-agent-skills` shard 02 (agent-tools) — `skill_list` tool calls `list_skill_eligible_tools` to UNION with prose-skill rows.
- `3tears-agent-skills` shard 03 (skills-block-renderer) — when rendering a skill's `tool_additions`, may need to introspect the underlying tool's mcp_name; this works via existing namespace queries.
- `3tears-agent-wake` shard 03 (dispatch-handler) — pre-check tools (`http_get`, `loki_query`, `postgres_query`) are registered with `tool_eligible=False, skill_eligible=True`; their existence as platform tools is mechanically possible without this shard but their inclusion in skills' `tool_additions` requires it.
