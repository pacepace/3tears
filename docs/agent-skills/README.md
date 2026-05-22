# agent-skills — Procedural memory for 3tears agents

**Status:** draft (planning set, no implementation yet).
**Scope:** primarily `3tears/packages/agent/skills/` (new package); touches `3tears/packages/agent/tools/` via the prerequisite eligibility-flags shard (`../agent-tools-eligibility/`).
**Consumer:** metallm (see `metallm/docs/skills/PLACEMENT.md` for the canonical design + placement memo).

**Read `metallm/docs/skills/PLACEMENT.md` first** for the full design decisions log. This README is the platform-side scope summary.

---

## What this is

A new `3tears-agent-skills` package providing **procedural memory** for any 3tears-based agent: named, reusable, agent-or-user-authored units that modify how an agent behaves on a specific turn.

Skills sit alongside `3tears-agent-memory`'s declarative memory:
- **Declarative memory** (3tears-agent-memory) answers *"what do I know?"* — facts, summaries, embeddings, hybrid retrieval.
- **Procedural memory** (this package) answers *"how do I do this kind of thing?"* — labeled markdown procedures plus optional tool-surface modifications.

A skill is a **per-turn context modifier**: it can carry up to three independent payloads (any combination, at least one required):
- A **prose body** — markdown that loads as a labeled block into the system prompt.
- A **tool additions list** — tool `mcp_name` entries that become available for the turn (even tools registered with `tool_eligible=False`).
- A **tool restrictions list** — tool `mcp_name` entries removed from the agent's default surface for the turn.

Plus a `prompt_mode` field (`'additive'` or `'replace'`) controlling how the body interacts with the consumer's base system prompt.

## Why a new package, not an extension to agent-memory

Different storage shape (FTS-only text + metadata, no vector embeddings), different RBAC vocabulary (`skill.read`/`write`/`invoke` vs. memory's `read`/`write`/`extract`), different lifecycle (outcome classification per invocation), different authoring model. Sibling packages let consumers adopt either without the other.

## Prerequisite: `agent-tools-eligibility`

This package depends on `3tears-agent-tools` shipping the `tool_eligible` + `skill_eligible` flags first. See [`../agent-tools-eligibility/shard-01-tool-eligibility-flags.md`](../agent-tools-eligibility/shard-01-tool-eligibility-flags.md) for the foundation shard. Both ship in the same 3tears release as this package.

The catalog query `list_skill_eligible_tools(actor_user_id, actor_agent_id)` from the prereq is what powers `skill_list`'s UNION over prose-skill rows + tool-skill registry entries.

## Scope: what counts as a skill (revised 2026-05-19)

- **Three optional payload fields**, at least one required: `prompt_addition` (markdown body), `tool_additions` (mcp_name list), `tool_restrictions` (mcp_name list).
- **`prompt_mode` enum** — `'additive'` (default) appends body to base system prompt; `'replace'` substitutes the base entirely. Per-user additions (NSFW, jailbreak) still apply on top in either mode (applied by the consumer, not by this package).
- **Identity**: `name` (unique per `(agent_id, user_id)`), `summary` (one-line for the catalog), `trigger_keywords` (for `skill_list` query filtering only — NOT for auto-load), `tags`.
- **Instrumentation**: `use_count`, `last_used_at`, `success_count`, `failure_count`, `last_failure_at`.
- **Owned by exactly one (agent_id, user_id) pair**. No cross-user sharing in v1.
- **Created one way**: manually via `skill_create`. No `skill_create_from_range` distillation tool (dropped per PLACEMENT §1.4 — agents author skills directly from their existing context).
- **Loaded two ways**: wake-attached (a wake schedule's `skill_id` FK references this skill; loaded automatically when the wake fires) OR explicit `skill_invoke` (agent calls the tool mid-user-turn). NO auto-load via classifier.
- **One skill per turn** maximum. No multi-skill blending (PLACEMENT §1.3).
- **Edited via `skill_update`** — agents refine their own.
- **Outcome recording**: post-LLM hook in the consumer parses `[SUCCESS]`/`[FAILED]` markers from the agent's response and updates the invocation row synchronously. No background classifier tick.

## The shards (revised — 3 shards, down from 4)

| # | Shard | Concern |
|---|-------|---------|
| 01 | [Schema + Collection layer](shard-01-schema-and-collection.md) | `agent_skills` + `agent_skill_invocations` tables (with `prompt_mode`, `tool_additions`, `tool_restrictions`, CHECK constraint enforcing at-least-one-payload); entities; Collections; migration registration |
| 02 | [Agent tools](shard-02-agent-tools.md) | **Seven `TearsTool` factories** (six CRUD+invoke + one introspect): `skill_create`, `skill_list`, `skill_get`, `skill_update`, `skill_delete`, `skill_invoke`, `skill_introspect`. The introspect tool returns the minimal-token shape from PLACEMENT §1.8 |
| 03 | [Active-skill renderer + per-turn composition](shard-03-skills-block-renderer.md) | `compose_turn_context(active_skill, base_system_prompt, base_tool_names, *, acl_permits)` — the canonical per-turn composition from PLACEMENT §1.10. Pure function |

**Plus prerequisite:** [`../agent-tools-eligibility/shard-01-tool-eligibility-flags.md`](../agent-tools-eligibility/shard-01-tool-eligibility-flags.md) — modifies `3tears-agent-tools` to add `tool_eligible` + `skill_eligible` flags. Foundation for everything else.

**Dropped from prior redesign:** `shard-02-retrieval-and-classifier-framework.md` deleted entirely (auto-load is gone). `skill_create_from_range` removed from agent tools. `skill_history` deferred to v1.1 (will be a separate `skill_stats` tool).

## What this is NOT

- **NOT auto-load.** Skills do not load on every user turn. No FTS classifier, no per-turn relevance ranking, no top-K retrieval.
- **NOT multi-skill composition.** One skill per turn maximum. The `wake_schedule_skill_attachments` junction table from the prior plan is dropped; wakes get a single nullable `skill_id` FK.
- **NOT a public skill registry.** No cross-user sharing.
- **NOT a code-execution surface.** Skills are prompt content + tool-name references. Tools themselves are authored externally (TearsTool subclasses or MCP registrations) and gated by ACL.
- **NOT a replacement for memories.** Declarative facts still live in `memories`. Skills are for procedures + tool surfaces.
- **NOT a fine-tuning pipeline.** Skills modify behaviour via in-context loading, not weight updates.
- **NOT a way for agents to escalate privileges.** Skills cannot bypass ACL. `tool_additions` adds visibility within ACL-permitted tools; never grants new authorisation.
- **NOT a REST / HTTP surface.** Each product wraps the Collections + tools in its own auth-aware router. metallm provides `/api/v1/skills`.
- **NOT a UI.** Frontend is product-specific.
- **NOT a system-prompt orchestrator.** This package provides `compose_turn_context` (the composition function) and the agent tools. How the consumer's `personality_node` orchestrates wake-driven vs. user-driven skill loading + how per-user additions layer in is the consumer's responsibility.
- **NOT a distillation surface.** No `skill_create_from_range`. Agents author skills from their existing context via the regular `skill_create`.

## Locked design decisions (canonical source: `metallm/docs/skills/PLACEMENT.md`)

| Decision | Locked answer | PLACEMENT ref |
|---|---|---|
| Package name | `3tears-agent-skills` (import `threetears.agent.skills`) | |
| Package location | `packages/agent/skills/` (sibling to memory) | |
| Skill payload shape | Three optional fields (`body`, `tool_additions`, `tool_restrictions`) + `prompt_mode` enum; ≥1 payload required | §1.1 |
| When skills load | Wake-driven OR explicit `skill_invoke`. NO auto-load. | §1.2 |
| Composition | One skill per turn (no multi-skill blending) | §1.3 |
| Authorship by agent | Prose skills only (`tool_additions`/`tool_restrictions` reference existing tools) | §1.4 |
| Tool eligibility flags prereq | `agent-tools-eligibility` shard ships first | §1.5 |
| `tool_eligible=False, skill_eligible=True` | "Code-skill without sandbox" pattern | §1.5 |
| Pre-check tool collapse | Pre-checks are ordinary tools with `tool_eligible=False, skill_eligible=True` | §1.6 |
| Catalog UNION | `skill_list` returns prose-skill rows + skill-eligible tools from registry, uniformly | §1.7 |
| Introspection shape | Minimal-token (name + summary + body OR args+example). NO operational metadata. | §1.8 |
| Wake creation flow | `wake_schedule_create(..., skill_id=picked)` after `skill_list`/`skill_introspect` discovery | §1.9 |
| Per-turn composition | `compose_turn_context` applies prompt_mode + tool_additions + tool_restrictions with ACL gating | §1.10 |
| Partition column | `agent_id` (collections-task-04 convention) | |
| FTS engine | Postgres `tsvector` for `skill_list` query filter ONLY. No auto-load. | |
| RBAC v1 | Simple user-scoped (recommend integrate `3tears-agent-acl` from day one per PLACEMENT §3.2) | §3.2 |
| UUIDv7 | All UUID PKs | |
| Migration order | This package's migrations land BEFORE any consumer's that FK `agent_skills.skill_id`; enforced by `MigrationRunner` topological ordering | |

## Verification contract (every shard)

- `./scripts/test.sh agent-skills` — package-scoped tests pass.
- `./scripts/lint.sh agent-skills` — ruff + format clean.
- `./scripts/typecheck.sh agent-skills` — mypy strict clean.
- Integration tests run against `testcontainers` Postgres.
- After each shard, the existing 3tears packages still pass: `./scripts/check-all.sh`.
- After the package's release, metallm's release bumps the 3tears pin to include this package's version.

## Out of scope (deferred)

- Vector-embedding-based semantic skill retrieval (FTS-only).
- Cross-user / cross-organisation skill sharing.
- Skills Hub / public skill registry.
- Skill versioning with rollback (edit-in-place).
- Skills with multimedia content.
- `skill_create_from_range` distillation tool.
- `skill_history` / `skill_stats` operational-metadata tool (deferred to v1.1).
- Per-organisation skill libraries.
- Auto-load classifier / FTS retrieval for skills.
- Code-shaped skills authored by agents (capability expansion stays admin-gated via tool registration).
